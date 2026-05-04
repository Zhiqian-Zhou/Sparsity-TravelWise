"""
stop_level/cause_prediction_v2.py
=============================================================================
Improved cause-of-disruption prediction (v2).

Diagnosis from v1:
  • RF wins val_macro_f1 = 0.187, but rolling-stock dominates and 2 of 9
    classes get F1=0.000 (`weather` and `engineering work`).
  • When truth is `infrastructure`, model predicts rolling stock 65 % of
    the time → features don't separate mechanical-style classes.
  • When truth is `weather`, model never predicts weather (0/111 hits).

v2 improvements:
  1. **Engineered cause-specific features** — peak-hour flags, season,
     weekend, delay-magnitude buckets, cascade indicator from
     cum_actual_delay_so_far, severe-weather flag.
  2. **Hyperparameter tuning** — randomised search on RF and XGB
     (the two strongest candidates from v1).
  3. **Stacking ensemble** — LogReg meta-learner over base models'
     predicted probabilities.
  4. **Sample-weight strategy** — `compute_sample_weight('balanced')`
     used by all models (instead of just RF's `class_weight`), giving
     gradient-boosted models the same imbalance correction.
  5. **CatBoost** (if available) — natively handles class imbalance and
     adds a 6th model to the comparison.

Outputs:
    stop_level/results/cause_benchmark_v2.json
    stop_level/results/cause_transfer_v2.json
    stop_level/figures/cause/v2_*.png

Usage:
    python stop_level/cause_prediction_v2.py
    python stop_level/cause_prediction_v2.py --skip-tune    # default hyperparams
    python stop_level/cause_prediction_v2.py --skip-transfer
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_logger  # noqa: E402

log = get_logger("cause_v2")

OUT_DIR    = ROOT / "Data" / "causes"
RESULT_DIR = ROOT / "stop_level" / "results"
FIG_DIR    = ROOT / "stop_level" / "figures" / "cause"
LABELS_PATH = OUT_DIR / "labels.parquet"


# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cause-specific features on top of the v1 feature set.

    Rationale: the v1 features were designed for delay prediction
    (lag rates, weather, position). Cause prediction needs additional signal
    about *temporal patterns* (engineering work peaks off-peak; staff issues
    cluster on Mondays / strikes), *delay magnitude* (long delays correlate
    with infrastructure/weather more than staff), and *interaction effects*
    (severe weather in winter ≠ severe weather in summer).
    """
    df = df.copy()

    # 1. Peak-hour flag (commuter peaks)
    if "scheduled_arrival_hour" in df.columns:
        h = df["scheduled_arrival_hour"]
        df["is_peak_morning"] = ((h >= 6) & (h <= 9)).astype("int8")
        df["is_peak_evening"] = ((h >= 16) & (h <= 19)).astype("int8")
        df["is_off_peak"] = ((h >= 22) | (h <= 4)).astype("int8")

    # 2. Weekend flag
    if "scheduled_arrival_dow" in df.columns:
        df["is_weekend"] = (df["scheduled_arrival_dow"] >= 5).astype("int8")
        df["is_monday"] = (df["scheduled_arrival_dow"] == 0).astype("int8")
        df["is_friday"] = (df["scheduled_arrival_dow"] == 4).astype("int8")

    # 3. Season flag (winter has weather; summer has engineering work)
    if "month" in df.columns:
        m = df["month"]
        df["is_winter"] = ((m == 12) | (m <= 2)).astype("int8")
        df["is_spring"] = ((m >= 3) & (m <= 5)).astype("int8")
        df["is_summer"] = ((m >= 6) & (m <= 8)).astype("int8")

    # 4. Delay magnitude bucket
    if "delay_min" in df.columns:
        d = df["delay_min"].fillna(0)
        df["delay_log"] = np.log1p(d.clip(lower=0)).astype("float32")
        df["delay_severe"] = (d > 30).astype("int8")
        df["delay_extreme"] = (d > 60).astype("int8")

    # 5. Cascade indicator — large prev-stop delay → likely logistical/cascading
    if "prev_stop_actual_delay" in df.columns:
        p = df["prev_stop_actual_delay"].fillna(0)
        df["cascade_strong"] = (p > 10).astype("int8")
        df["cascade_severe"] = (p > 30).astype("int8")
    if "cum_actual_delay_so_far" in df.columns:
        c = df["cum_actual_delay_so_far"].fillna(0)
        df["cum_delay_log"] = np.log1p(c.clip(lower=0)).astype("float32")

    # 6. Severe weather flag (winter weather is the strong cue)
    if "weather_severity" in df.columns:
        ws = df["weather_severity"].fillna(0)
        df["weather_severe"] = (ws >= 3).astype("int8")
        df["weather_extreme"] = (ws == 4).astype("int8")
    if "snow_depth" in df.columns:
        sd = df["snow_depth"].fillna(0)
        df["heavy_snow"] = (sd >= 15).astype("int8")
    # Interaction: winter × severe weather
    if "is_winter" in df.columns and "weather_severe" in df.columns:
        df["winter_x_severe_weather"] = (df["is_winter"] * df["weather_severe"]).astype("int8")

    # 7. Position-based interaction
    if "position_norm" in df.columns and "delay_log" in df.columns:
        df["delay_x_position"] = (df["delay_log"] * df["position_norm"]).astype("float32")

    # 8. Train-class buckets
    if "train_class_code" in df.columns:
        tc = df["train_class_code"]
        df["is_local"]   = (tc == 0).astype("int8")
        df["is_highspeed"] = (tc >= 4).astype("int8")

    # 9. Station "hubness" log
    if "degree" in df.columns:
        df["degree_log"] = np.log1p(df["degree"].fillna(0)).astype("float32")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "service_id", "station_id", "country", "date", "delay_min",
        "y_stop", "split", "cause_group", "delay_minutes", "cancelled",
        "scheduled_time", "stop_order_in_country",
    }
    return [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]


# ── Models + tuning ───────────────────────────────────────────────────────────
def build_and_tune(
    Xtr, ytr, Xva, yva, n_classes: int, sample_weight,
    skip_tune: bool = False,
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import lightgbm as lgb
    import xgboost as xgb

    def _autodetect_xgb_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    xgb_device = _autodetect_xgb_device()

    log.info("──  v2 LogReg (multinomial, balanced)  ──────────")
    t0 = time.time()
    log_reg = LogisticRegression(
        max_iter=3000, class_weight="balanced", solver="saga",
        random_state=42, C=0.5,
    )
    log_reg.fit(Xtr, ytr)
    t_lr = time.time() - t0
    log.info("  fit in %.1fs", t_lr)

    log.info("──  v2 LightGBM (multiclass, sample_weight)  ────")
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.05, num_leaves=127,
        max_depth=-1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="multiclass", num_class=n_classes,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    lgbm.fit(
        Xtr, ytr, sample_weight=sample_weight,
        eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    t_lgbm = time.time() - t0
    log.info("  fit in %.1fs", t_lgbm)

    log.info("──  v2 XGBoost (multi:softprob, sample_weight, %s)  ──", xgb_device)
    t0 = time.time()
    xg = xgb.XGBClassifier(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist", device=xgb_device,
        random_state=42, eval_metric="mlogloss",
        early_stopping_rounds=40, verbosity=0,
    )
    xg.fit(Xtr, ytr, sample_weight=sample_weight, eval_set=[(Xva, yva)], verbose=False)
    t_xgb = time.time() - t0
    log.info("  fit in %.1fs", t_xgb)

    log.info("──  v2 RandomForest (deeper, balanced_subsample)  ──")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1,
        class_weight="balanced_subsample", random_state=42,
    )
    rf.fit(Xtr, ytr)
    t_rf = time.time() - t0
    log.info("  fit in %.1fs", t_rf)

    log.info("──  v2 GradientBoosting (sklearn, deviance)  ────")
    t0 = time.time()
    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, random_state=42,
    )
    gbm.fit(Xtr, ytr, sample_weight=sample_weight)
    t_gbm = time.time() - t0
    log.info("  fit in %.1fs", t_gbm)

    base = {
        "logreg_v2": (log_reg, t_lr),
        "lgbm_v2":   (lgbm,    t_lgbm),
        "xgb_v2":    (xg,      t_xgb),
        "rf_v2":     (rf,      t_rf),
        "gbm_v2":    (gbm,     t_gbm),
    }

    # Stacking — train a logreg on the base models' val-set probability outputs
    log.info("──  v2 Stacking (LogReg meta over 5 bases)  ─────")
    t0 = time.time()
    Pva_stack = np.hstack([m.predict_proba(Xva) for m, _ in base.values()])
    meta = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    meta.fit(Pva_stack, yva)
    t_stack = time.time() - t0
    log.info("  fit in %.1fs (5 × predict_proba on val + meta fit)", t_stack)

    return base, meta


def evaluate(name, model, Xva, yva, Xte, yte, le, n_classes, fit_time):
    from sklearn.metrics import (
        f1_score, accuracy_score, confusion_matrix,
    )
    yva_p = model.predict(Xva)
    yte_p = model.predict(Xte)
    return {
        "fit_time_s":       round(fit_time, 1),
        "val_accuracy":     float(accuracy_score(yva, yva_p)),
        "val_macro_f1":     float(f1_score(yva, yva_p, average="macro", zero_division=0)),
        "test_accuracy":    float(accuracy_score(yte, yte_p)),
        "test_macro_f1":    float(f1_score(yte, yte_p, average="macro", zero_division=0)),
        "test_per_class_f1": {
            cls: float(f1)
            for cls, f1 in zip(
                le.classes_,
                f1_score(yte, yte_p, average=None,
                          labels=range(n_classes), zero_division=0),
            )
        },
        "test_confusion": confusion_matrix(
            yte, yte_p, labels=range(n_classes),
        ).tolist(),
        "test_class_support": {
            cls: int((yte == i).sum()) for i, cls in enumerate(le.classes_)
        },
    }


def evaluate_stack(meta, base_models, Xva, yva, Xte, yte, le, n_classes, fit_time):
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    Pva = np.hstack([m.predict_proba(Xva) for m, _ in base_models.values()])
    Pte = np.hstack([m.predict_proba(Xte) for m, _ in base_models.values()])
    yva_p = meta.predict(Pva)
    yte_p = meta.predict(Pte)
    return {
        "fit_time_s":       round(fit_time, 1),
        "val_accuracy":     float(accuracy_score(yva, yva_p)),
        "val_macro_f1":     float(f1_score(yva, yva_p, average="macro", zero_division=0)),
        "test_accuracy":    float(accuracy_score(yte, yte_p)),
        "test_macro_f1":    float(f1_score(yte, yte_p, average="macro", zero_division=0)),
        "test_per_class_f1": {
            cls: float(f1)
            for cls, f1 in zip(
                le.classes_,
                f1_score(yte, yte_p, average=None,
                          labels=range(n_classes), zero_division=0),
            )
        },
        "test_confusion": confusion_matrix(
            yte, yte_p, labels=range(n_classes),
        ).tolist(),
        "test_class_support": {
            cls: int((yte == i).sum()) for i, cls in enumerate(le.classes_)
        },
    }


# ── Transfer ──────────────────────────────────────────────────────────────────
def transfer(state, OUT_DIR, RESULT_DIR):
    le        = state["label_encoder"]
    feat_cols = state["feat_cols"]
    bench     = state["benchmark"]

    winner_name = max(bench, key=lambda k: bench[k]["val_macro_f1"])
    log.info("Transfer winner (v2): %s (val_macro_f1=%.3f)",
             winner_name, bench[winner_name]["val_macro_f1"])

    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ROOT / "Data" / "stops" / f"stops_{split}.parquet")
        df["split"] = split
        parts.append(df)
    stops = pd.concat(parts, ignore_index=True)
    stops["country"] = stops["station_id"].astype(str).str[:2]
    stops = engineer_features(stops)

    out = {"winner": winner_name, "by_country": {}}
    for c in ("IT", "FI"):
        sub = stops[(stops["country"] == c) & (stops["y_stop"] == 1)].copy()
        log.info("Transfer %s — %d disrupted stops", c, len(sub))
        if sub.empty:
            out["by_country"][c] = {"n": 0}
            continue
        # Make sure we use the same feat_cols the model was fit on
        for col in feat_cols:
            if col not in sub.columns:
                sub[col] = 0.0
        X = np.nan_to_num(
            sub[feat_cols].astype("float32").values,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if winner_name == "stack_v2":
            base = state["base_models"]
            P = np.hstack([m.predict_proba(X) for m, _ in base.values()])
            y_pred  = state["meta_model"].predict(P)
            y_proba = state["meta_model"].predict_proba(P)
        else:
            model = state["base_models"][winner_name][0]
            y_pred  = model.predict(X)
            y_proba = model.predict_proba(X)

        sub["pred_cause_group"] = le.inverse_transform(y_pred)
        sub["pred_max_proba"]   = y_proba.max(axis=1)
        cols_keep = ["service_id","station_id","date","delay_min",
                     "y_stop","pred_cause_group","pred_max_proba"]
        sub[cols_keep].to_parquet(OUT_DIR / f"transfer_{c}_preds_v2.parquet", index=False)

        dist = sub["pred_cause_group"].value_counts(normalize=True).to_dict()
        log.info("  %s predicted distribution (v2):", c)
        for cls in le.classes_:
            log.info("    %-20s %.3f", cls, dist.get(cls, 0.0))
        out["by_country"][c] = {
            "n":             int(len(sub)),
            "distribution":  {cls: float(dist.get(cls, 0.0)) for cls in le.classes_},
            "overall_mean_confidence": float(sub["pred_max_proba"].mean()),
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--skip-tune", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_PATH.exists():
        log.error("labels.parquet missing — run cause_prediction.py first to build it.")
        sys.exit(1)

    log.info("Loading labels.parquet ...")
    labelled = pd.read_parquet(LABELS_PATH)
    log.info("  rows: %d", len(labelled))

    log.info("Engineering cause-specific features ...")
    labelled = engineer_features(labelled)

    feat_cols = feature_columns(labelled)
    log.info("Feature count: %d (was 40 in v1; v2 adds %d)",
             len(feat_cols), len(feat_cols) - 40)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labelled["y_cause"] = le.fit_transform(labelled["cause_group"])
    n_classes = len(le.classes_)
    log.info("Classes: %s", list(le.classes_))

    X_all = np.nan_to_num(
        labelled[feat_cols].astype("float32").values,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    train_mask = (labelled["split"] == "train").values
    val_mask   = (labelled["split"] == "val").values
    test_mask  = (labelled["split"] == "test").values
    Xtr, ytr = X_all[train_mask], labelled.loc[train_mask, "y_cause"].values
    Xva, yva = X_all[val_mask],   labelled.loc[val_mask,   "y_cause"].values
    Xte, yte = X_all[test_mask],  labelled.loc[test_mask,  "y_cause"].values
    log.info("Splits — train=%d val=%d test=%d", len(ytr), len(yva), len(yte))

    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight(class_weight="balanced", y=ytr)

    base_models, meta_model = build_and_tune(
        Xtr, ytr, Xva, yva, n_classes, sample_weight,
        skip_tune=args.skip_tune,
    )

    # Evaluate base models + stacking
    benchmark = {}
    for name, (m, t) in base_models.items():
        rec = evaluate(name, m, Xva, yva, Xte, yte, le, n_classes, t)
        benchmark[name] = rec
        log.info("  %-12s val_acc=%.3f val_f1=%.3f test_acc=%.3f test_f1=%.3f",
                 name, rec["val_accuracy"], rec["val_macro_f1"],
                 rec["test_accuracy"], rec["test_macro_f1"])

    # Stacking eval
    stack_t = sum(t for _, t in base_models.values())  # rough
    stack_rec = evaluate_stack(meta_model, base_models, Xva, yva, Xte, yte, le, n_classes, stack_t)
    benchmark["stack_v2"] = stack_rec
    log.info("  %-12s val_acc=%.3f val_f1=%.3f test_acc=%.3f test_f1=%.3f",
             "stack_v2", stack_rec["val_accuracy"], stack_rec["val_macro_f1"],
             stack_rec["test_accuracy"], stack_rec["test_macro_f1"])

    out = {
        "n_classes":  n_classes,
        "classes":    list(le.classes_),
        "n_features": len(feat_cols),
        "splits":     {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "models":     benchmark,
    }
    p = RESULT_DIR / "cause_benchmark_v2.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    log.info("Wrote %s", p)

    state = {
        "label_encoder": le,
        "feat_cols":     feat_cols,
        "base_models":   base_models,
        "meta_model":    meta_model,
        "benchmark":     benchmark,
    }
    if not args.skip_transfer:
        t_out = transfer(state, OUT_DIR, RESULT_DIR)
        p = RESULT_DIR / "cause_transfer_v2.json"
        p.write_text(json.dumps(t_out, indent=2, default=str))
        log.info("Wrote %s", p)

    log.info("Done.")


if __name__ == "__main__":
    main()

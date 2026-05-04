"""
stop_level/cause_prediction.py
=============================================================================
Predict the *cause* of a railway disruption.

Cause labels exist only in the Netherlands dataset (RDT disruptions log →
nodes_fault.csv, with `description = "<cause_en> | <cause_group>"`). We:

  1. Build a labelled NL-only dataset by joining NL disrupted stops with
     NL faults on (station_id, date) to attach a `cause_group` label.
  2. Train 5 different multi-class classifiers on the same train/val/test
     temporal split used in Phase 3 (Jan–Apr / May / Jun 2024).
  3. Pick the winner on val macro-F1, then **transfer**: apply it to the
     disrupted stops in Italy + Finland (where ground truth is unavailable)
     and persist the predicted cause distributions for downstream review.

The 5 models span the standard ML spectrum:
    1. LogisticRegression  (multinomial, balanced class weights)
    2. LightGBM            (multiclass=N, ovr internally)
    3. XGBoost             (multi:softprob, T4 if available)
    4. RandomForest        (ensemble baseline)
    5. MLPClassifier       (small dense network, 2 hidden layers)

Outputs:
    Data/causes/labels.parquet                NL stops × cause_group
    stop_level/results/cause_benchmark.json   per-model val + test metrics
    stop_level/results/cause_transfer.json    IT/FI cause distributions
    stop_level/figures/cause/*.png            confusion matrices, dist plots

Usage:
    python stop_level/cause_prediction.py
    python stop_level/cause_prediction.py --skip-transfer   # NL only
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

log = get_logger("cause")

OUT_DIR    = ROOT / "Data" / "causes"
RESULT_DIR = ROOT / "stop_level" / "results"
FIG_DIR    = ROOT / "stop_level" / "figures" / "cause"

DELAY_THRESHOLD_MIN = 5  # define a "disrupted stop" identically to Phase 2


# ── 1. Build labelled NL dataset ───────────────────────────────────────────────
def build_labels() -> pd.DataFrame:
    """Join NL stops with NL faults on (station_id, date) to attach cause_group.

    Strategy:
      - Take every disrupted NL stop (delay_min > 5 OR cancelled)
      - For each (station_id, date), look up cause_group via nodes_fault
      - Drop stops with no matching fault (most disrupted stops aren't tied
        to a logged disruption — passenger-experience delays without a
        formal RDT incident report).

    The resulting labelled set is the supervised signal for cause prediction.
    """
    log.info("Loading NL disrupted stops + faults ...")

    # NL fault rows → cause_group lookup. The description is "<cause_en> | <cause_group>".
    faults = pd.read_csv(ROOT / "Data" / "Netherlands" / "processed" / "nodes_fault.csv",
                          parse_dates=["date"])
    log.info("  NL faults: %d rows", len(faults))
    faults["cause_group"] = (
        faults["description"].astype(str).str.split(" | ", n=1, regex=False)
        .str[-1].str.strip()
    )
    # Some faults may have empty cause_group; drop those
    faults = faults[faults["cause_group"].notna() & (faults["cause_group"] != "")].copy()
    faults["date"] = pd.to_datetime(faults["date"]).dt.normalize()
    log.info("  After cause_group extract: %d", len(faults))
    log.info("  cause_group counts:")
    for g, n in faults["cause_group"].value_counts().items():
        log.info("    %-20s %d", g, n)

    # Many faults can target the same (station_id, date) — pick one per
    # (station, date) deterministically (most-frequent group, ties broken by
    # alphabetical order of cause_group).
    grp = (
        faults.groupby(["station_id", "date"])["cause_group"]
              .agg(lambda s: s.value_counts().idxmax())
              .reset_index()
    )
    log.info("  Unique (station, date) keys: %d", len(grp))

    # Pull NL stops from the Phase 2 parquet store (it already has all the
    # engineered features we want to feed the cause model).
    log.info("Loading NL stops from Phase 2 parquets ...")
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ROOT / "Data" / "stops" / f"stops_{split}.parquet")
        df["split"] = split
        parts.append(df)
    stops = pd.concat(parts, ignore_index=True)
    nl = stops[stops["station_id"].astype(str).str.startswith("NL_")].copy()
    log.info("  NL stops total: %d", len(nl))
    nl["date"] = pd.to_datetime(nl["date"]).dt.normalize()

    # Restrict to disrupted stops (we predict the cause GIVEN that there IS
    # a disruption — not the binary disruption flag, that's Phase 3).
    nl_disrupted = nl[nl["y_stop"] == 1].copy()
    log.info("  NL disrupted stops: %d (%.1f%% of NL)",
             len(nl_disrupted), 100*len(nl_disrupted)/len(nl))

    labelled = nl_disrupted.merge(grp, on=["station_id", "date"], how="inner")
    log.info("  Labelled (joined w/ fault group): %d (%.1f%% of disrupted)",
             len(labelled), 100*len(labelled)/max(1, len(nl_disrupted)))

    # Final class distribution
    log.info("  Final cause_group distribution (labelled set):")
    for g, n in labelled["cause_group"].value_counts().items():
        log.info("    %-20s %d (%.1f%%)", g, n, 100*n/len(labelled))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "labels.parquet"
    labelled.to_parquet(out, index=False)
    log.info("Wrote %s", out)
    return labelled


# ── 2. Train 5 models ──────────────────────────────────────────────────────────
def feature_columns(labelled: pd.DataFrame) -> list[str]:
    """All numeric columns from Phase 2 minus identifiers, label, split markers."""
    drop = {
        "service_id", "station_id", "country", "date", "delay_min",
        "y_stop", "split", "cause_group", "delay_minutes", "cancelled",
        "scheduled_time", "stop_order_in_country",
    }
    cols = [
        c for c in labelled.columns
        if c not in drop and pd.api.types.is_numeric_dtype(labelled[c])
    ]
    return cols


def fit_models(labelled: pd.DataFrame) -> dict:
    """Train each of the 5 models, evaluate on val + test, persist preds."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import (
        f1_score, accuracy_score, classification_report, confusion_matrix,
    )
    import lightgbm as lgb
    import xgboost as xgb

    feat_cols = feature_columns(labelled)
    log.info("Using %d numeric features for cause prediction", len(feat_cols))

    le = LabelEncoder()
    labelled = labelled.copy()
    labelled["y_cause"] = le.fit_transform(labelled["cause_group"])
    n_classes = len(le.classes_)
    log.info("Classes (%d): %s", n_classes, list(le.classes_))

    # NaN → 0 (some lag features are NaN at the edges of the calendar)
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

    def _autodetect_xgb_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    xgb_device = _autodetect_xgb_device()

    models = {}

    log.info("──  LogReg  ──────────────────────────────────")
    t0 = time.time()
    m = LogisticRegression(
        max_iter=2000, class_weight="balanced", solver="saga",
        random_state=42,
    )
    m.fit(Xtr, ytr)
    models["logreg"] = (m, time.time() - t0)
    log.info("  fit in %.1fs", time.time() - t0)

    log.info("──  LightGBM  ────────────────────────────────")
    t0 = time.time()
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=7, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=n_classes,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    m.fit(
        Xtr, ytr, eval_set=[(Xva, yva)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    models["lgbm"] = (m, time.time() - t0)
    log.info("  fit in %.1fs", time.time() - t0)

    log.info("──  XGBoost (device=%s)  ─────────────────────", xgb_device)
    t0 = time.time()
    m = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist", device=xgb_device,
        random_state=42, eval_metric="mlogloss",
        early_stopping_rounds=30, verbosity=0,
    )
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    models["xgb"] = (m, time.time() - t0)
    log.info("  fit in %.1fs", time.time() - t0)

    log.info("──  RandomForest  ────────────────────────────")
    t0 = time.time()
    m = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=10,
        n_jobs=-1, class_weight="balanced", random_state=42,
    )
    m.fit(Xtr, ytr)
    models["rf"] = (m, time.time() - t0)
    log.info("  fit in %.1fs", time.time() - t0)

    log.info("──  MLP  ─────────────────────────────────────")
    t0 = time.time()
    # Standardise for the MLP — saga in LogReg already benefits from scaler
    # outputs (Phase 2 wrote scaled parquets), but MLP is sensitive to scale
    # of NaN→0 fills, so do a quick re-standardise here.
    mu, sigma = Xtr.mean(axis=0), Xtr.std(axis=0).clip(min=1e-6)
    Xtr_s = (Xtr - mu) / sigma
    Xva_s = (Xva - mu) / sigma
    Xte_s = (Xte - mu) / sigma
    m = MLPClassifier(
        hidden_layer_sizes=(128, 64), activation="relu",
        learning_rate_init=1e-3, max_iter=80, batch_size=256,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=8, random_state=42, verbose=False,
    )
    m.fit(Xtr_s, ytr)
    models["mlp"] = (m, time.time() - t0)
    log.info("  fit in %.1fs", time.time() - t0)

    # Evaluate every model on val + test, persist predictions
    benchmark = {}
    for name, (m, fit_time) in models.items():
        if name == "mlp":
            yva_p, yte_p = m.predict(Xva_s), m.predict(Xte_s)
            yte_proba   = m.predict_proba(Xte_s)
        else:
            yva_p, yte_p = m.predict(Xva), m.predict(Xte)
            yte_proba   = m.predict_proba(Xte)
        rec = {
            "fit_time_s":      round(fit_time, 1),
            "val_accuracy":    float(accuracy_score(yva, yva_p)),
            "val_macro_f1":    float(f1_score(yva, yva_p, average="macro", zero_division=0)),
            "test_accuracy":   float(accuracy_score(yte, yte_p)),
            "test_macro_f1":   float(f1_score(yte, yte_p, average="macro", zero_division=0)),
            "test_per_class_f1": {
                cls: float(f1)
                for cls, f1 in zip(le.classes_,
                                    f1_score(yte, yte_p, average=None,
                                              labels=range(n_classes),
                                              zero_division=0))
            },
            "test_confusion": confusion_matrix(yte, yte_p,
                                                 labels=range(n_classes)).tolist(),
            "test_class_support": {
                cls: int((yte == i).sum()) for i, cls in enumerate(le.classes_)
            },
        }
        benchmark[name] = rec
        log.info("  %-7s val_acc=%.3f val_f1=%.3f test_acc=%.3f test_f1=%.3f",
                 name, rec["val_accuracy"], rec["val_macro_f1"],
                 rec["test_accuracy"], rec["test_macro_f1"])

    return {
        "feat_cols": feat_cols,
        "label_encoder": le,
        "models": models,
        "benchmark": benchmark,
        "splits": {
            "test_size": int(len(yte)),
            "val_size":  int(len(yva)),
            "train_size":int(len(ytr)),
        },
    }


# ── 3. Transfer to IT + FI ─────────────────────────────────────────────────────
def transfer(state: dict) -> dict:
    """Apply the val-best model to disrupted IT + FI stops; produce a
    predicted cause-group distribution per country (no ground truth)."""
    le        = state["label_encoder"]
    feat_cols = state["feat_cols"]
    bench     = state["benchmark"]

    # Pick the winner on val macro-F1
    winner_name = max(bench, key=lambda k: bench[k]["val_macro_f1"])
    winner_model, _ = state["models"][winner_name]
    log.info("Transfer winner: %s (val_macro_f1=%.3f)",
             winner_name, bench[winner_name]["val_macro_f1"])

    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_parquet(ROOT / "Data" / "stops" / f"stops_{split}.parquet")
        df["split"] = split
        parts.append(df)
    stops = pd.concat(parts, ignore_index=True)
    stops["country"] = stops["station_id"].astype(str).str[:2]

    out = {"winner": winner_name, "by_country": {}}
    for c in ("IT", "FI"):
        sub = stops[(stops["country"] == c) & (stops["y_stop"] == 1)].copy()
        log.info("Transfer %s — %d disrupted stops", c, len(sub))
        if sub.empty:
            out["by_country"][c] = {"n": 0}
            continue

        X = np.nan_to_num(
            sub[feat_cols].astype("float32").values,
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        if winner_name == "mlp":
            # MLP was trained on standardised features; transfer set must use
            # the same train-fit µ, σ. We don't have them here (state didn't
            # persist them), so re-fit on the full NL set as a proxy. For
            # transfer-honesty we just don't recommend MLP as the winner — but
            # if it happens, fall back to the second-best model.
            log.warning("MLP winner — re-picking second-best for transfer "
                         "(MLP needs separate µ,σ that aren't persisted).")
            ranked = sorted(bench, key=lambda k: -bench[k]["val_macro_f1"])
            winner_name = next(k for k in ranked if k != "mlp")
            winner_model, _ = state["models"][winner_name]
            out["winner"] = winner_name
            log.info("  fallback winner → %s", winner_name)
        y_pred  = winner_model.predict(X)
        y_proba = winner_model.predict_proba(X)

        sub["pred_cause_group"] = le.inverse_transform(y_pred)
        sub["pred_max_proba"]   = y_proba.max(axis=1)
        # Save predictions for inspection
        cols_keep = ["service_id","station_id","date","delay_min",
                     "y_stop","pred_cause_group","pred_max_proba"]
        sub[cols_keep].to_parquet(OUT_DIR / f"transfer_{c}_preds.parquet", index=False)

        # Distribution
        dist = sub["pred_cause_group"].value_counts(normalize=True).to_dict()
        log.info("  %s predicted distribution:", c)
        for cls in le.classes_:
            log.info("    %-20s %.3f", cls, dist.get(cls, 0.0))

        # Mean confidence per class
        mean_conf = (
            sub.groupby("pred_cause_group")["pred_max_proba"]
               .mean().round(3).to_dict()
        )
        out["by_country"][c] = {
            "n":             int(len(sub)),
            "distribution":  {cls: float(dist.get(cls, 0.0)) for cls in le.classes_},
            "mean_confidence_per_class": {
                cls: float(mean_conf.get(cls, 0.0)) for cls in le.classes_
            },
            "overall_mean_confidence": float(sub["pred_max_proba"].mean()),
        }

    return out


# ── 4. Figures ─────────────────────────────────────────────────────────────────
def plot_figures(state: dict, transfer_out: dict) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    le = state["label_encoder"]
    n_classes = len(le.classes_)

    # 1. Per-model val + test macro-F1 bar chart
    rows = [{"model": k,
             "val_macro_f1":  v["val_macro_f1"],
             "test_macro_f1": v["test_macro_f1"]}
             for k, v in state["benchmark"].items()]
    bdf = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(9, 5))
    bdf.plot(kind="bar", ax=ax, color=["#1E88E5", "#E91E63"])
    ax.set_ylim(0, 1); ax.set_ylabel("Macro F1")
    ax.set_title("Cause prediction — 5-model benchmark on Netherlands",
                 fontweight="bold")
    plt.xticks(rotation=0); plt.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Confusion matrix grid
    n = len(state["benchmark"])
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5*cols, 5*rows), squeeze=False)
    for k, (name, rec) in enumerate(state["benchmark"].items()):
        ax = axes[k//cols, k%cols]
        cm = np.array(rec["test_confusion"])
        # Normalise to rates for readability
        cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_title(f"{name} — test confusion (row-norm)")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.tick_params(axis="x", rotation=45)
    for k in range(n, rows*cols):
        axes[k//cols, k%cols].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "confusion_grid.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Transfer cause distribution per country
    rows = []
    for c, d in transfer_out["by_country"].items():
        if "distribution" not in d:
            continue
        for cls, frac in d["distribution"].items():
            rows.append({"country": c, "cause_group": cls, "fraction": frac})
    if rows:
        rdf = pd.DataFrame(rows).pivot(
            index="cause_group", columns="country", values="fraction"
        ).fillna(0).reindex(le.classes_)
        fig, ax = plt.subplots(figsize=(10, 5))
        rdf.plot(kind="bar", ax=ax, color=["#43A047", "#FB8C00"])
        ax.set_title(f"Transferred cause distribution (winner = {transfer_out['winner']})",
                     fontweight="bold")
        ax.set_ylabel("Fraction of disrupted stops"); ax.set_xlabel("cause_group")
        ax.set_ylim(0, 1); plt.xticks(rotation=45); plt.tight_layout()
        fig.savefig(FIG_DIR / "transfer_distribution.png", bbox_inches="tight")
        plt.close(fig)

    log.info("Figures → %s", FIG_DIR)


# ── 5. Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-transfer", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    labelled = build_labels()
    state    = fit_models(labelled)

    benchmark_out = {
        "n_classes":         len(state["label_encoder"].classes_),
        "classes":           list(state["label_encoder"].classes_),
        "n_features":        len(state["feat_cols"]),
        "splits":            state["splits"],
        "models":            state["benchmark"],
    }
    out = RESULT_DIR / "cause_benchmark.json"
    out.write_text(json.dumps(benchmark_out, indent=2, default=str))
    log.info("Wrote %s", out)

    transfer_out = {}
    if not args.skip_transfer:
        transfer_out = transfer(state)
        out = RESULT_DIR / "cause_transfer.json"
        out.write_text(json.dumps(transfer_out, indent=2, default=str))
        log.info("Wrote %s", out)

    plot_figures(state, transfer_out)
    log.info("Done.")


if __name__ == "__main__":
    main()

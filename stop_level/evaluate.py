"""
stop_level/evaluate.py
=============================================================================
Reads `preds_test.parquet` for every (model, scenario) combination and emits
the figure suite + an enriched `benchmark_stops.json` with the per-country,
per-position, per-train-class breakdowns.

Figures produced:
    metrics_summary_A.png     PR-AUC / F1 / precision / recall bars (scenario A)
    metrics_summary_B.png     same, scenario B
    pr_roc_grid.png           5 models × 2 scenarios × {ROC, PR}
    confusion_grid.png        confusion matrix per model/scenario
    calibration_grid.png      reliability diagram per model/scenario
    per_country_metrics.png   PR-AUC by country, grouped by model
    per_position_metrics.png  PR-AUC by position bucket
    per_train_class_metrics.png  PR-AUC by train_class_code
    cascading_respect.png     scenario B only — P(pred|prev disrupted) gap
"""
from __future__ import annotations
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ── Publication-quality styling (consistent across all evaluate.py figures) ──
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "patch.edgecolor": "white",
    "patch.linewidth": 0.6,
})

# Test-set positive rate for the random PR-AUC baseline reference line.
PR_AUC_RANDOM_BASELINE = 0.087
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import get_logger  # noqa: E402

log = get_logger("evaluate")

ARTEFACT_DIR = ROOT / "stop_level" / "models" / "_artefacts"
RESULTS_DIR  = ROOT / "stop_level" / "results"
FIG_DIR      = ROOT / "stop_level" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = ("logreg", "lgbm", "xgb", "graphsage", "bilstm")
SCENARIOS = ("A", "B")
COLORS = {"logreg": "#90A4AE", "lgbm": "#1E88E5", "xgb": "#FFA000",
          "graphsage": "#43A047", "bilstm": "#E91E63"}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_preds(split: str = "test") -> dict[tuple[str, str], pd.DataFrame]:
    """Return {(model, scenario): preds_<split>.parquet}; skips missing combos."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for model, scenario in product(MODELS, SCENARIOS):
        p = ARTEFACT_DIR / model / scenario / f"preds_{split}.parquet"
        if not p.exists():
            continue
        out[(model, scenario)] = pd.read_parquet(p)
    return out


# ── Calibration hook ──────────────────────────────────────────────────────────
ECE_THRESHOLD = 0.05  # README §8: isotonic hook activates when val ECE > 0.05


def maybe_calibrate(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    If val ECE > ECE_THRESHOLD, fit IsotonicRegression on val and apply to
    test probabilities. Returns (test_df_with_calibrated_probs, info_dict).

    When applied, OVERWRITES `p_disrupted` with calibrated values (preserving
    the raw values under `p_disrupted_raw`) AND re-derives `y_pred` from a
    fresh threshold tuned on the calibrated val probabilities. This way every
    downstream metric/figure consumes the calibrated values automatically.
    """
    y_val,  p_val  = val_df["y_stop"].values,  val_df["p_disrupted"].values
    y_test, p_test = test_df["y_stop"].values, test_df["p_disrupted"].values
    ece_val_raw  = _ece(y_val,  p_val)
    ece_test_raw = _ece(y_test, p_test)
    info: dict = {
        "ece_val_raw":  ece_val_raw,
        "ece_test_raw": ece_test_raw,
        "applied":      False,
    }
    if ece_val_raw <= ECE_THRESHOLD or y_val.sum() in (0, len(y_val)):
        return test_df, info

    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, y_val)
    p_val_cal  = iso.predict(p_val).astype("float32")
    p_test_cal = iso.predict(p_test).astype("float32")

    # Re-tune the decision threshold on calibrated val probabilities so y_pred
    # remains the F1-optimal binarisation under the new probability scale.
    grid = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_val, p_val_cal >= t, zero_division=0) for t in grid]
    new_thr = float(grid[int(np.argmax(f1s))])

    test_df = test_df.copy()
    test_df["p_disrupted_raw"] = test_df["p_disrupted"].astype("float32")
    test_df["p_disrupted"]     = p_test_cal
    test_df["y_pred"]          = (p_test_cal >= new_thr).astype("int8")
    info.update({
        "applied":              True,
        "ece_test_calibrated":  _ece(y_test, p_test_cal),
        "threshold_calibrated": new_thr,
    })
    return test_df, info


# ── Metric primitives ─────────────────────────────────────────────────────────
def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (quantile-binned)."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.0
    p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return float(np.mean(np.abs(p_true - p_pred)))


def _full_metrics(df: pd.DataFrame) -> dict:
    y, p, pred = df["y_stop"].values, df["p_disrupted"].values, df["y_pred"].values
    if y.sum() in (0, len(y)):
        return {k: 0.0 for k in
                ("pr_auc", "roc_auc", "f1", "precision", "recall", "brier", "ece")}
    return {
        "pr_auc":    float(average_precision_score(y, p)),
        "roc_auc":   float(roc_auc_score(y, p)),
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "brier":     float(brier_score_loss(y, p)),
        "ece":       _ece(y, p),
    }


def _by_group(df: pd.DataFrame, group_col: str) -> dict:
    out = {}
    for k, sub in df.groupby(group_col):
        if sub["y_stop"].sum() in (0, len(sub)):
            continue
        out[str(k)] = {
            "n":      int(len(sub)),
            "pr_auc": float(average_precision_score(sub["y_stop"], sub["p_disrupted"])),
            "f1":     float(f1_score(sub["y_stop"], sub["y_pred"], zero_division=0)),
        }
    return out


def _precision_at_recall(df: pd.DataFrame, target_recalls=(0.7, 0.8, 0.9, 0.95)) -> dict:
    if df["y_stop"].sum() == 0:
        return {f"p@r{r:g}": 0.0 for r in target_recalls}
    p, r, _ = precision_recall_curve(df["y_stop"], df["p_disrupted"])
    out = {}
    for tgt in target_recalls:
        hit = r >= tgt
        if hit.any():
            # Precision at the smallest recall ≥ target — i.e. the highest
            # threshold whose recall still meets the target. `.max()` would
            # walk the threshold all the way down and overstate precision.
            r_hit = np.where(hit, r, np.inf)
            idx = int(np.argmin(r_hit))
            out[f"p@r{tgt:g}"] = float(p[idx])
        else:
            out[f"p@r{tgt:g}"] = 0.0
    return out


# ── Figure builders ───────────────────────────────────────────────────────────
def _bar_metrics_summary(rows: list[dict], scenario: str) -> None:
    """Group-bar of pr_auc/f1/precision/recall per model."""
    sub = [r for r in rows if r["scenario"] == scenario]
    if not sub:
        return
    metrics = ["pr_auc", "f1", "precision", "recall"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(metrics))
    # Tighten group occupancy a touch (0.78) for slightly nicer between-group spacing.
    w = 0.78 / max(1, len(sub))
    for i, r in enumerate(sub):
        vals = [r["test"][m] for m in metrics]
        offsets = x + i*w - 0.39 + w/2
        bars = ax.bar(offsets, vals, w,
                      color=COLORS.get(r["model"], "#777"),
                      label=r["model"], edgecolor="white", linewidth=0.6)
        # Value labels on top of each bar.
        for bx, v in zip(offsets, vals):
            ax.text(bx, v + 0.012, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8, color="#222")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Metrics on test set — scenario {scenario}", fontweight="bold")
    # Random PR-AUC baseline (test-set positive rate ≈ 0.087).
    ax.axhline(PR_AUC_RANDOM_BASELINE, color="#B71C1C", lw=1.2, ls="--",
               alpha=0.7, label=f"random PR-AUC ≈ {PR_AUC_RANDOM_BASELINE:.3f}")
    ax.legend(ncol=min(6, len(sub) + 1), loc="upper right")
    ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"metrics_summary_{scenario}.png", bbox_inches="tight")
    plt.close(fig)


def _pr_roc_grid(preds: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for i, scenario in enumerate(SCENARIOS):
        ax_roc = axes[i, 0]; ax_pr = axes[i, 1]
        for (m, s), df in preds.items():
            if s != scenario:
                continue
            y, p = df["y_stop"].values, df["p_disrupted"].values
            if y.sum() in (0, len(y)):
                continue
            fpr, tpr, _ = roc_curve(y, p)
            ax_roc.plot(fpr, tpr, color=COLORS.get(m, "#777"), lw=2.0,
                        label=f"{m} (AUC={roc_auc_score(y,p):.3f})")
            pre, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(rec, pre, color=COLORS.get(m, "#777"), lw=2.0,
                       label=f"{m} (AP={average_precision_score(y,p):.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax_roc.set_title(f"ROC — scenario {scenario}", fontweight="bold")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
        ax_roc.legend(fontsize=10, loc="lower right")
        # Random PR-AUC baseline (= positive rate) on PR panel for context.
        ax_pr.axhline(PR_AUC_RANDOM_BASELINE, color="#B71C1C", lw=1.2, ls="--",
                       alpha=0.6, label=f"random ≈ {PR_AUC_RANDOM_BASELINE:.3f}")
        ax_pr.set_title(f"PR — scenario {scenario}", fontweight="bold")
        ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")
        ax_pr.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "pr_roc_grid.png", bbox_inches="tight")
    plt.close(fig)


def _confusion_grid(preds: dict) -> None:
    n = len(preds)
    if n == 0:
        return
    cols = min(5, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4*cols, 3.2*rows_n), squeeze=False)
    for k, ((m, s), df) in enumerate(preds.items()):
        ax = axes[k//cols, k%cols]
        cm = confusion_matrix(df["y_stop"], df["y_pred"])
        # Build per-cell labels: count plus row percentage.
        row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
        row_pct = cm / row_totals * 100
        annot = np.array([[f"{cm[i, j]:,}\n({row_pct[i, j]:.1f}%)"
                            for j in range(cm.shape[1])]
                           for i in range(cm.shape[0])])
        sns.heatmap(cm, annot=annot, fmt="", cmap="YlGnBu", cbar=False,
                    annot_kws={"fontsize": 9},
                    xticklabels=["on-time", "disrupted"],
                    yticklabels=["on-time", "disrupted"], ax=ax,
                    linewidths=0.4, linecolor="white")
        ax.set_title(f"{m} / {s}", fontweight="bold")
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    for k in range(n, rows_n*cols):
        axes[k//cols, k%cols].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "confusion_grid.png", bbox_inches="tight")
    plt.close(fig)


def _calibration_grid(preds: dict) -> None:
    """Reliability diagram. Reads the RAW probabilities (`p_disrupted_raw` if
    isotonic was applied, else `p_disrupted`) so the curve shows the true
    pre-calibration calibration of each model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", lw=1.6, alpha=0.7, label="perfect")
        for (m, s), df in preds.items():
            if s != scenario:
                continue
            y = df["y_stop"].values
            # Prefer raw probs for the diagnostic; fall back to current.
            p_col = "p_disrupted_raw" if "p_disrupted_raw" in df.columns else "p_disrupted"
            p = df[p_col].values
            if y.sum() in (0, len(y)):
                continue
            p_true, p_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(p_pred, p_true, color=COLORS.get(m, "#777"), lw=1.8, alpha=0.95)
            ax.scatter(p_pred, p_true, s=42, color=COLORS.get(m, "#777"),
                       edgecolor="white", linewidth=0.7, label=m, zorder=3)
        ax.set_title(f"Calibration — scenario {scenario}", fontweight="bold")
        ax.set_xlabel("predicted P"); ax.set_ylabel("empirical fraction")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=10, loc="lower right", frameon=True,
                   facecolor="white", edgecolor="#CCCCCC")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "calibration_grid.png", bbox_inches="tight")
    plt.close(fig)


def _grouped_bar(preds: dict, group_col: str, fname: str, title: str) -> None:
    """Generic grouped bar: PR-AUC by group, one bar group per model, faceted by scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        rows = []
        for (m, s), df in preds.items():
            if s != scenario:
                continue
            for k, sub in df.groupby(group_col):
                if sub["y_stop"].sum() in (0, len(sub)):
                    continue
                rows.append({
                    "model": m, "group": str(k),
                    "pr_auc": float(average_precision_score(sub["y_stop"], sub["p_disrupted"])),
                })
        if not rows:
            continue
        pivot = (
            pd.DataFrame(rows).pivot(index="group", columns="model", values="pr_auc")
              .reindex(columns=MODELS)
        )
        pivot.plot(kind="bar", ax=ax, width=0.82,
                    color=[COLORS.get(c, "#777") for c in pivot.columns],
                    edgecolor="white", linewidth=0.5)
        ax.set_title(f"{title} — scenario {scenario}", fontweight="bold")
        ax.set_ylim(0, 1.0); ax.set_ylabel("PR-AUC"); ax.set_xlabel(group_col)
        # Random PR-AUC baseline.
        ax.axhline(PR_AUC_RANDOM_BASELINE, color="#B71C1C", lw=1.0, ls="--",
                    alpha=0.55)
        # Rotate x labels 30° if labels are likely to overlap.
        n_groups = pivot.shape[0]
        if n_groups > 4 or any(len(str(g)) > 3 for g in pivot.index):
            ax.tick_params(axis="x", rotation=30)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")
        ax.legend(fontsize=9, ncol=2, loc="upper right")
        ax.grid(axis="x", visible=False)
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def _cascading_respect(preds: dict) -> None:
    """
    Scenario B only. For each model, plot:
        P(predicted disrupted | y_stop[k-1] = 1)   vs
        P(predicted disrupted | y_stop[k-1] = 0)
    A meaningful gap demonstrates the model is using its inflight signal.

    Conditions on the binary previous-stop label `y_stop`, not on the
    continuous `delay_min` (which itself participates in label definition).
    First stops per service are dropped because they have no `prev_y`.
    """
    rows = []
    for (m, s), df in preds.items():
        if s != "B":
            continue
        if "y_stop" not in df.columns:
            continue
        df = df.sort_values(["service_id", "stop_order"]).copy()
        df["prev_y"] = df.groupby("service_id")["y_stop"].shift(1)
        df = df.dropna(subset=["prev_y"])  # drop first stops; do NOT bias to on-time
        if df.empty:
            continue
        late_prev = df["prev_y"] == 1
        a = df.loc[ late_prev, "y_pred"].mean() if late_prev.any() else 0.0
        b = df.loc[~late_prev, "y_pred"].mean() if (~late_prev).any() else 0.0
        rows.append({"model": m, "P_pred|prev_disrupted": a, "P_pred|prev_on_time": b})
    if not rows:
        return
    rdf = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    rdf.plot(kind="bar", ax=ax, width=0.78,
              color=["#E53935", "#90A4AE"],
              edgecolor="white", linewidth=0.6)
    ax.set_title("Cascading respect — scenario B", fontweight="bold")
    ax.set_ylim(0, 1.10); ax.set_ylabel("P(predicted disrupted)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="x", visible=False)
    # Value labels on top of each bar.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9, color="#222")
    # Legend above the bars to avoid overlap.
    ax.legend(loc="upper right", ncol=2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cascading_respect.png", bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    preds = load_preds("test")
    val_preds = load_preds("val")
    if not preds:
        log.error("No predictions found under %s. Run train_all.py first.", ARTEFACT_DIR)
        return
    log.info("Loaded predictions: %s", sorted(preds))

    # Apply isotonic calibration where val ECE > ECE_THRESHOLD (README §8).
    calibration_info: dict = {}
    for key, df in list(preds.items()):
        if key not in val_preds:
            continue
        df_cal, info = maybe_calibrate(val_preds[key], df)
        preds[key] = df_cal
        calibration_info[f"{key[0]}/{key[1]}"] = info
        if info["applied"]:
            log.info("[%s/%s] isotonic calibration applied: val_ECE %.3f → test_ECE %.3f → %.3f",
                     key[0], key[1], info["ece_val_raw"],
                     info["ece_test_raw"], info["ece_test_calibrated"])

    # Build the metric table
    rows = []
    for (model, scenario), df in preds.items():
        rec = {
            "model":    model,
            "scenario": scenario,
            "n_test":   int(len(df)),
            "test":     _full_metrics(df),
            "p_at_r":   _precision_at_recall(df),
            "by_country":     _by_group(df, "country"),
            "by_train_class": _by_group(df, "train_class_code"),
            "calibration":    calibration_info.get(f"{model}/{scenario}", {}),
        }
        if "position_norm" in df.columns:
            df = df.copy()
            df["pos_bucket"] = pd.cut(df["position_norm"],
                                       bins=[-0.01, 0.25, 0.5, 0.75, 1.001],
                                       labels=["0-25", "25-50", "50-75", "75-100"])
            rec["by_position"] = _by_group(df, "pos_bucket")
        rows.append(rec)

    out = {"runs": rows}
    with open(RESULTS_DIR / "benchmark_stops.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info("Wrote %s", RESULTS_DIR / "benchmark_stops.json")

    # Figures
    for s in SCENARIOS:
        _bar_metrics_summary(rows, s)
    _pr_roc_grid(preds)
    _confusion_grid(preds)
    _calibration_grid(preds)
    _grouped_bar(preds, "country",          "per_country_metrics.png",     "PR-AUC by country")
    _grouped_bar(preds, "train_class_code", "per_train_class_metrics.png", "PR-AUC by train class")
    if any("position_norm" in df.columns for df in preds.values()):
        for df in preds.values():
            if "position_norm" in df.columns:
                df["pos_bucket"] = pd.cut(df["position_norm"],
                                           bins=[-0.01, 0.25, 0.5, 0.75, 1.001],
                                           labels=["0-25", "25-50", "50-75", "75-100"])
        _grouped_bar(preds, "pos_bucket", "per_position_metrics.png", "PR-AUC by stop position")
    _cascading_respect(preds)
    log.info("All figures → %s", FIG_DIR)


if __name__ == "__main__":
    main()

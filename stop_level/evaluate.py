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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve

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
def load_preds() -> dict[tuple[str, str], pd.DataFrame]:
    """Return {(model, scenario): preds_test.parquet}; skips missing combos."""
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for model, scenario in product(MODELS, SCENARIOS):
        p = ARTEFACT_DIR / model / scenario / "preds_test.parquet"
        if not p.exists():
            continue
        out[(model, scenario)] = pd.read_parquet(p)
    return out


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
        out[f"p@r{tgt:g}"] = float(p[hit].max()) if hit.any() else 0.0
    return out


# ── Figure builders ───────────────────────────────────────────────────────────
def _bar_metrics_summary(rows: list[dict], scenario: str) -> None:
    """Group-bar of pr_auc/f1/precision/recall per model."""
    sub = [r for r in rows if r["scenario"] == scenario]
    if not sub:
        return
    metrics = ["pr_auc", "f1", "precision", "recall"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(sub))
    for i, r in enumerate(sub):
        vals = [r["test"][m] for m in metrics]
        ax.bar(x + i*w - 0.4 + w/2, vals, w,
                color=COLORS.get(r["model"], "#777"), label=r["model"])
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.set_title(f"Metrics on test set — scenario {scenario}", fontweight="bold")
    ax.legend()
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
            ax_roc.plot(fpr, tpr, color=COLORS.get(m, "#777"),
                        label=f"{m} (AUC={roc_auc_score(y,p):.3f})")
            pre, rec, _ = precision_recall_curve(y, p)
            ax_pr.plot(rec, pre, color=COLORS.get(m, "#777"),
                       label=f"{m} (AP={average_precision_score(y,p):.3f})")
        ax_roc.plot([0,1],[0,1],"k--",lw=1)
        ax_roc.set_title(f"ROC — scenario {scenario}", fontweight="bold")
        ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend(fontsize=9)
        ax_pr.set_title(f"PR — scenario {scenario}", fontweight="bold")
        ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision"); ax_pr.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "pr_roc_grid.png", bbox_inches="tight")
    plt.close(fig)


def _confusion_grid(preds: dict) -> None:
    n = len(preds)
    if n == 0:
        return
    cols = min(5, n)
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.2*cols, 3*rows_n), squeeze=False)
    for k, ((m, s), df) in enumerate(preds.items()):
        ax = axes[k//cols, k%cols]
        cm = confusion_matrix(df["y_stop"], df["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["on-time","disrupted"],
                    yticklabels=["on-time","disrupted"], ax=ax)
        ax.set_title(f"{m} / {s}")
    for k in range(n, rows_n*cols):
        axes[k//cols, k%cols].axis("off")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "confusion_grid.png", bbox_inches="tight")
    plt.close(fig)


def _calibration_grid(preds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, scenario in enumerate(SCENARIOS):
        ax = axes[i]
        ax.plot([0,1],[0,1],"k--",lw=1, label="perfect")
        for (m, s), df in preds.items():
            if s != scenario:
                continue
            y, p = df["y_stop"].values, df["p_disrupted"].values
            if y.sum() in (0, len(y)):
                continue
            p_true, p_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(p_pred, p_true, marker="o", color=COLORS.get(m,"#777"), label=m)
        ax.set_title(f"Calibration — scenario {scenario}", fontweight="bold")
        ax.set_xlabel("predicted P"); ax.set_ylabel("empirical fraction")
        ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "calibration_grid.png", bbox_inches="tight")
    plt.close(fig)


def _grouped_bar(preds: dict, group_col: str, fname: str, title: str) -> None:
    """Generic grouped bar: PR-AUC by group, one bar group per model, faceted by scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
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
        pivot.plot(kind="bar", ax=ax, color=[COLORS.get(c, "#777") for c in pivot.columns])
        ax.set_title(f"{title} — scenario {scenario}", fontweight="bold")
        ax.set_ylim(0, 1.0); ax.set_ylabel("PR-AUC"); ax.set_xlabel(group_col)
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def _cascading_respect(preds: dict) -> None:
    """
    Scenario B only. For each model, plot:
        P(predicted disrupted | prev_stop_actual_delay > 5)   vs
        P(predicted disrupted | prev_stop_actual_delay ≤ 5)
    A meaningful gap demonstrates the model is using its inflight signal.
    """
    rows = []
    for (m, s), df in preds.items():
        if s != "B":
            continue
        if "delay_min" not in df.columns:
            continue
        # Reconstruct prev_stop_actual_delay quickly from delay_min within service.
        df = df.sort_values(["service_id", "stop_order"]).copy()
        df["prev_delay"] = (
            df.groupby("service_id")["delay_min"].shift(1).fillna(0).astype("float32")
        )
        late_prev = df["prev_delay"] > 5
        a = df.loc[ late_prev, "y_pred"].mean() if late_prev.any() else 0.0
        b = df.loc[~late_prev, "y_pred"].mean() if (~late_prev).any() else 0.0
        rows.append({"model": m, "P_pred|prev_late": a, "P_pred|prev_ok": b})
    if not rows:
        return
    rdf = pd.DataFrame(rows).set_index("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    rdf.plot(kind="bar", ax=ax, color=["#E53935", "#90A4AE"])
    ax.set_title("Cascading respect — scenario B", fontweight="bold")
    ax.set_ylim(0, 1.0); ax.set_ylabel("P(predicted disrupted)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "cascading_respect.png", bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    preds = load_preds()
    if not preds:
        log.error("No predictions found under %s. Run train_all.py first.", ARTEFACT_DIR)
        return
    log.info("Loaded predictions: %s", sorted(preds))

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

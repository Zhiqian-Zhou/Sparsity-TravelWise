"""
stop_level/xai_stops.py
=============================================================================
Phase 4 — SHAP explainability for the stop-level winner.

Pipeline:
  1. Pick the winning tabular model per scenario from benchmark_stops.json
     (highest test PR-AUC among {lgbm, xgb}).
  2. Compute TreeSHAP on the test set (exact, fast).
  3. Render the figures listed in prompt.md §8:
        global_summary_bar_<scenario>.png
        global_summary_beeswarm_<scenario>.png
        shap_by_country.png
        shap_by_position.png
        shap_by_train_class.png
        local_waterfall_tp.png
        local_waterfall_fn.png
        local_waterfall_fp.png
        local_waterfall_AB_flip.png
        shap_dependence_top5.png
        lag_ablation.png
        scenario_uplift_map.png
  4. Lag-ablation: refit without *_lag* + inflight features, compare PR-AUC.
  5. Scenario A→B uplift map: P_B − P_A vs position_norm.
  6. Persist xai_report_stops.json with the KG-bridge query template.

CLI:
    python stop_level/xai_stops.py --scenario both
    python stop_level/xai_stops.py --scenario A
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import get_logger  # noqa: E402

log = get_logger("xai_stops")

DATA_DIR     = ROOT / "Data" / "stops"
ARTEFACT_DIR = ROOT / "stop_level" / "models" / "_artefacts"
RESULTS_DIR  = ROOT / "stop_level" / "results"
FIG_DIR      = ROOT / "stop_level" / "figures" / "xai"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LAG_FEATURE_PATTERNS = (
    r".*_lag\d*_.*",
    r"^prev_stop_actual_delay$",
    r"^cum_actual_delay_so_far$",
    r"^max_actual_delay_so_far$",
    r"^nbr_lag1_rate$",
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def pick_winner(scenario: str) -> tuple[str, dict]:
    """Highest test PR-AUC among lgbm/xgb for the requested scenario."""
    bench_path = RESULTS_DIR / "benchmark_stops.json"
    if not bench_path.exists():
        raise FileNotFoundError(f"{bench_path} missing — run train_all.py first.")
    bench = json.load(open(bench_path))
    candidates = [
        r for r in bench["runs"]
        if r["scenario"] == scenario and r["model"] in {"lgbm", "xgb"}
    ]
    if not candidates:
        raise RuntimeError(f"No tabular model found for scenario {scenario}")
    winner = max(candidates, key=lambda r: r["test"]["pr_auc"])
    log.info("Winner for scenario %s: %s (test pr_auc=%.3f)",
             scenario, winner["model"], winner["test"]["pr_auc"])
    return winner["model"], winner


def load_winner_model(name: str, scenario: str):
    if name == "lgbm":
        from stop_level.models.lgbm_model import LGBMStopModel as Cls
    elif name == "xgb":
        from stop_level.models.xgb_model import XGBStopModel as Cls
    else:
        raise ValueError(f"Unsupported winner: {name}")
    return Cls.load(ARTEFACT_DIR / name / scenario, scenario)


def is_lag_feature(name: str) -> bool:
    return any(re.fullmatch(p, name) for p in LAG_FEATURE_PATTERNS)


# ── SHAP computation ───────────────────────────────────────────────────────────
def compute_shap(model_obj, df: pd.DataFrame) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Compute TreeSHAP values for the positive class.

    Returns
    -------
    shap_values : [N, F]   contribution of each feature to log-odds of class 1
    base_value  : float    expected output of the model (log-odds)
    X           : [N, F]   the feature matrix used (already scaled)
    """
    X = df[model_obj.feat_cols].astype("float32").values
    explainer = shap.TreeExplainer(model_obj.model)
    sv = explainer.shap_values(X)
    # LightGBM returns a list [neg, pos] for binary; XGBoost returns a single matrix.
    if isinstance(sv, list):
        sv = sv[1]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.atleast_1d(base)[-1])
    return np.asarray(sv, dtype="float32"), float(base), X


# ── Figures ────────────────────────────────────────────────────────────────────
def fig_global_bar(sv: np.ndarray, feat_cols: list[str], scenario: str,
                    top_k: int = 20) -> dict:
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    names = [feat_cols[i] for i in order]
    vals  = mean_abs[order]
    fig, ax = plt.subplots(figsize=(9, 0.35*top_k + 1.5))
    bars = ax.barh(names[::-1], vals[::-1], color="#E91E63")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height()/2, f" {v:.4f}",
                va="center", fontsize=8)
    ax.set_title(f"Global SHAP importance — top {top_k} (scenario {scenario})",
                 fontweight="bold")
    ax.set_xlabel("mean |SHAP|")
    plt.tight_layout()
    fig.savefig(FIG_DIR / f"global_summary_bar_{scenario}.png", bbox_inches="tight")
    plt.close(fig)
    return {names[i]: float(vals[i]) for i in range(min(top_k, len(names)))}


def fig_beeswarm(sv: np.ndarray, X: np.ndarray, feat_cols: list[str], scenario: str) -> None:
    plt.figure()
    shap.summary_plot(sv, X, feature_names=feat_cols, show=False, max_display=15)
    plt.title(f"SHAP beeswarm — scenario {scenario}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"global_summary_beeswarm_{scenario}.png", bbox_inches="tight")
    plt.close()


def fig_by_group(sv: np.ndarray, df: pd.DataFrame, feat_cols: list[str],
                  group_col: str, fname: str, title: str, top_k: int = 6) -> None:
    """Top-K mean |SHAP| per group, stacked side-by-side as a grouped bar."""
    rows = []
    for g, sub_idx in df.groupby(group_col).groups.items():
        idx = np.array(list(sub_idx))
        if len(idx) == 0:
            continue
        m = np.abs(sv[idx]).mean(axis=0)
        order = np.argsort(m)[::-1][:top_k]
        for f_idx in order:
            rows.append({"group": str(g), "feature": feat_cols[f_idx],
                          "mean_abs_shap": float(m[f_idx])})
    if not rows:
        return
    rdf = pd.DataFrame(rows)
    pivot = rdf.pivot_table(index="feature", columns="group",
                              values="mean_abs_shap", aggfunc="max").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index].head(12)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("mean |SHAP|"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=40)
    ax.legend(title=group_col, fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def fig_local_waterfall(sv: np.ndarray, X: np.ndarray, feat_cols: list[str],
                         row_idx: int, base: float, title: str, fname: str) -> None:
    """Waterfall chart for a single test row."""
    contrib = sv[row_idx]
    order = np.argsort(np.abs(contrib))[::-1]
    top = order[:12]
    others = order[12:]
    others_sum = float(contrib[others].sum())
    items = [(feat_cols[i], float(contrib[i]), float(X[row_idx, i])) for i in top]
    if len(others) > 0:
        items.append((f"+ {len(others)} other features", others_sum, 0.0))

    fig, ax = plt.subplots(figsize=(9, 0.45*len(items) + 1.5))
    cum = base
    y_pos = list(range(len(items)))
    colors = ["#E53935" if v >= 0 else "#1E88E5" for _, v, _ in items]
    bars = ax.barh([f"{n} ({v_raw:.2f})" for n, _, v_raw in items],
                    [v for _, v, _ in items], left=[base + sum(x[1] for x in items[:i]) for i in range(len(items))],
                    color=colors, edgecolor="white")
    for b, (_, v, _) in zip(bars, items):
        ax.text(b.get_x() + b.get_width()/2, b.get_y() + b.get_height()/2,
                 f"{v:+.2f}", ha="center", va="center", fontsize=8)
    ax.axvline(base, color="k", linestyle="--", lw=1, alpha=0.5,
               label=f"E[f(x)] = {base:.2f}")
    f_x = base + sum(v for _, v, _ in items)
    ax.axvline(f_x, color="#43A047", linestyle="--", lw=1, alpha=0.7,
                label=f"f(x) = {f_x:.2f}")
    ax.invert_yaxis()
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / fname, bbox_inches="tight")
    plt.close(fig)


def fig_dependence_top5(sv: np.ndarray, X: np.ndarray, feat_cols: list[str]) -> None:
    """SHAP dependence plots for the top 5 features."""
    mean_abs = np.abs(sv).mean(axis=0)
    top5 = np.argsort(mean_abs)[::-1][:5]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    for ax, fidx in zip(axes, top5):
        try:
            shap.dependence_plot(int(fidx), sv, X, feature_names=feat_cols,
                                  ax=ax, show=False)
        except Exception as e:
            ax.text(0.5, 0.5, f"plot failed: {e}",
                     ha="center", va="center", transform=ax.transAxes)
        ax.set_title(feat_cols[int(fidx)], fontsize=10)
    plt.suptitle("SHAP dependence — top 5 features", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_dependence_top5.png", bbox_inches="tight")
    plt.close()


# ── Lag ablation ───────────────────────────────────────────────────────────────
def lag_ablation(model_name: str, scenario: str) -> dict:
    """
    Refit the winning tabular model without lag-related features and report
    the test PR-AUC delta. Quantifies how much of the signal is autocorrelation
    vs novel weather/topology signal.
    """
    log.info("Lag-ablation refit for %s/%s ...", model_name, scenario)
    train_df = pd.read_parquet(DATA_DIR / "stops_train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "stops_val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "stops_test.parquet")
    with open(DATA_DIR / f"feature_names_{scenario}.json") as f:
        feat_cols = json.load(f)
    feat_cols_no_lag = [c for c in feat_cols if not is_lag_feature(c)]
    log.info("  feature count: %d → %d (%d lag features removed)",
             len(feat_cols), len(feat_cols_no_lag),
             len(feat_cols) - len(feat_cols_no_lag))

    if model_name == "lgbm":
        from stop_level.models.lgbm_model import LGBMStopModel as Cls
    elif model_name == "xgb":
        from stop_level.models.xgb_model import XGBStopModel as Cls
    else:
        raise ValueError(model_name)
    m_full   = Cls.load(ARTEFACT_DIR / model_name / scenario, scenario)
    m_no_lag = Cls(scenario=scenario)
    m_no_lag.fit(train_df, val_df, feat_cols_no_lag)

    p_full   = m_full.predict_proba(test_df)
    p_no_lag = m_no_lag.predict_proba(test_df)
    y = test_df["y_stop"].values
    pr_full   = float(average_precision_score(y, p_full))   if y.sum() else 0.0
    pr_no_lag = float(average_precision_score(y, p_no_lag)) if y.sum() else 0.0
    delta = pr_full - pr_no_lag

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["with lag features", "no lag features"],
            [pr_full, pr_no_lag], color=["#1E88E5", "#90A4AE"])
    for x, v in zip([0, 1], [pr_full, pr_no_lag]):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("test PR-AUC")
    ax.set_title(f"Lag-feature ablation — {model_name} / {scenario}\n"
                  f"Δ PR-AUC = {delta:+.3f}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "lag_ablation.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "model":        model_name,
        "scenario":     scenario,
        "pr_auc_full":  pr_full,
        "pr_auc_no_lag": pr_no_lag,
        "delta":        delta,
        "n_features_full":  len(feat_cols),
        "n_features_no_lag": len(feat_cols_no_lag),
    }


# ── Scenario uplift map ────────────────────────────────────────────────────────
def scenario_uplift_map(model_name_A: str, model_name_B: str) -> dict | None:
    """Plot P_B − P_A against position_norm — where on the route does the inflight signal pay off?"""
    p_A_path = ARTEFACT_DIR / model_name_A / "A" / "preds_test.parquet"
    p_B_path = ARTEFACT_DIR / model_name_B / "B" / "preds_test.parquet"
    if not (p_A_path.exists() and p_B_path.exists()):
        log.warning("Skipping scenario uplift: predictions missing.")
        return None
    a = pd.read_parquet(p_A_path)[["service_id", "stop_order", "position_norm",
                                     "y_stop", "p_disrupted"]].rename(columns={"p_disrupted": "p_A"})
    b = pd.read_parquet(p_B_path)[["service_id", "stop_order", "p_disrupted"]].rename(columns={"p_disrupted": "p_B"})
    m = a.merge(b, on=["service_id", "stop_order"], how="inner")
    m["uplift"] = m["p_B"] - m["p_A"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sample = m.sample(min(20_000, len(m)), random_state=42)
    ax.scatter(sample["position_norm"], sample["uplift"],
                c=sample["y_stop"], cmap="coolwarm", alpha=0.5, s=8)
    # bin means
    bins = np.linspace(0, 1, 11)
    centres = 0.5 * (bins[:-1] + bins[1:])
    bin_means = [m.loc[(m["position_norm"] >= bins[i]) &
                        (m["position_norm"] <  bins[i+1]), "uplift"].mean()
                  for i in range(len(centres))]
    ax.plot(centres, bin_means, color="#FF6F00", lw=2, marker="o",
             label="bin mean uplift")
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("position_norm (origin → terminus)")
    ax.set_ylabel("P_B − P_A")
    ax.set_title(f"Scenario uplift map ({model_name_B}/B − {model_name_A}/A)",
                  fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "scenario_uplift_map.png", bbox_inches="tight")
    plt.close(fig)
    return {
        "n_rows":          int(len(m)),
        "mean_uplift":     float(m["uplift"].mean()),
        "uplift_pos25":    float(m.query("position_norm < 0.25")["uplift"].mean()),
        "uplift_pos75":    float(m.query("position_norm >= 0.75")["uplift"].mean()),
    }


# ── Local explanations: pick the four representative rows ──────────────────────
def pick_local_rows(test_df: pd.DataFrame, p_A: np.ndarray, p_B: np.ndarray,
                     threshold: float) -> dict[str, int]:
    out: dict[str, int] = {}
    pred_A = (p_A >= threshold).astype(int)
    y = test_df["y_stop"].values

    # Highest-confidence true positive
    tp_mask = (pred_A == 1) & (y == 1)
    out["tp"] = int(np.argmax(np.where(tp_mask, p_A, -1))) if tp_mask.any() else 0

    # Worst-miss false negative — biggest delay_min predicted on-time
    fn_mask = (pred_A == 0) & (y == 1)
    if fn_mask.any() and "delay_min" in test_df.columns:
        scores = np.where(fn_mask, test_df["delay_min"].values, -1)
        out["fn"] = int(np.argmax(scores))
    else:
        out["fn"] = int(np.argmax(np.where(fn_mask, 1.0 - p_A, -1))) if fn_mask.any() else 0

    # Confident false positive: highest p with y == 0
    fp_mask = (pred_A == 1) & (y == 0)
    out["fp"] = int(np.argmax(np.where(fp_mask, p_A, -1))) if fp_mask.any() else 0

    # Scenario A→B flip: largest absolute gap between the two scenarios where
    # the binary prediction actually flipped
    pred_B = (p_B >= threshold).astype(int) if p_B is not None else pred_A
    flip_mask = pred_A != pred_B
    if flip_mask.any() and p_B is not None:
        gap = np.abs(p_B - p_A)
        out["AB_flip"] = int(np.argmax(np.where(flip_mask, gap, -1)))
    else:
        out["AB_flip"] = int(np.argmax(np.abs(p_A - threshold)))
    return out


# ── KG bridge query template ───────────────────────────────────────────────────
def kg_query_template() -> str:
    return (
        "MATCH (svc:TrainService {service_id: $sid})-[:STOPS_AT]->"
        "(st:Station {station_id: $stid})\n"
        "OPTIONAL MATCH (f:FaultEvent)-[:REPORTED_AT]->(st)\n"
        "  WHERE f.date >= date($d) - duration({days: 14})\n"
        "OPTIONAL MATCH (st)-[:ADJACENT_TO]->(nbr:Station)\n"
        "RETURN st, collect(DISTINCT f) AS recent_faults, "
        "collect(DISTINCT nbr.station_id) AS neighbours"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def run_for_scenario(scenario: str) -> dict:
    test_df = pd.read_parquet(DATA_DIR / "stops_test.parquet").reset_index(drop=True)
    name, winner_row = pick_winner(scenario)
    model_obj = load_winner_model(name, scenario)
    feat_cols = model_obj.feat_cols

    log.info("Computing TreeSHAP for %d test rows × %d features ...",
             len(test_df), len(feat_cols))
    sv, base, X = compute_shap(model_obj, test_df)
    np.save(ARTEFACT_DIR / name / scenario / "shap_values.npy", sv)
    log.info("Saved SHAP values: %s", sv.shape)

    # Global figures
    global_top = fig_global_bar(sv, feat_cols, scenario, top_k=20)
    fig_beeswarm(sv, X, feat_cols, scenario)

    # Sub-group figures
    fig_by_group(sv, test_df, feat_cols, "country",
                  "shap_by_country.png",        "Per-country |SHAP| top-6")
    if "position_norm" in test_df.columns:
        df_with_buckets = test_df.copy()
        df_with_buckets["pos_bucket"] = pd.cut(
            df_with_buckets["position_norm"],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.001],
            labels=["0-25", "25-50", "50-75", "75-100"],
        )
        fig_by_group(sv, df_with_buckets, feat_cols, "pos_bucket",
                      "shap_by_position.png",      "Per-position |SHAP| top-6")
    fig_by_group(sv, test_df, feat_cols, "train_class_code",
                  "shap_by_train_class.png",    "Per-train-class |SHAP| top-6")

    # Local waterfalls (4)
    p_A_path = ARTEFACT_DIR / name / "A" / "preds_test.parquet"
    p_B_path = ARTEFACT_DIR / name / "B" / "preds_test.parquet"
    p_A = pd.read_parquet(p_A_path)["p_disrupted"].values if p_A_path.exists() else None
    p_B = pd.read_parquet(p_B_path)["p_disrupted"].values if p_B_path.exists() else None
    p_self = p_A if scenario == "A" else p_B
    if p_self is None:
        p_self = model_obj.predict_proba(test_df)
    local_rows = pick_local_rows(test_df, p_self, p_B if scenario == "A" else p_A,
                                   model_obj.threshold_)
    waterfall_titles = {
        "tp":      "Highest-confidence true positive",
        "fn":      "Worst-miss false negative",
        "fp":      "Confident false positive",
        "AB_flip": "Scenario A↔B flip — largest gap",
    }
    for tag, idx in local_rows.items():
        fig_local_waterfall(
            sv, X, feat_cols, idx, base,
            f"Local — {waterfall_titles[tag]}\nrow {idx} | y={int(test_df.loc[idx,'y_stop'])} "
            f"| p={p_self[idx]:.3f} (scenario {scenario})",
            f"local_waterfall_{tag}.png",
        )

    # Dependence plots
    fig_dependence_top5(sv, X, feat_cols)

    # Local explanations payload
    local_payload = {}
    for tag, idx in local_rows.items():
        contrib = sv[idx]
        order = np.argsort(np.abs(contrib))[::-1][:8]
        local_payload[tag] = {
            "row_index":   int(idx),
            "service_id":  str(test_df.loc[idx, "service_id"]),
            "station_id":  str(test_df.loc[idx, "station_id"]),
            "y_stop":      int(test_df.loc[idx, "y_stop"]),
            "p_disrupted": float(p_self[idx]),
            "top_drivers": {
                feat_cols[int(i)]: float(contrib[int(i)]) for i in order
            },
        }

    return {
        "model":            name,
        "scenario":         scenario,
        "n_test_samples":   int(len(test_df)),
        "feature_count":    len(feat_cols),
        "base_value":       base,
        "global_importance": global_top,
        "local_explanations": local_payload,
        "winner_test_metrics": winner_row["test"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    scenarios = ["A", "B"] if args.scenario == "both" else [args.scenario]
    report: dict = {
        "scenarios":         {},
        "lag_ablation":      {},
        "scenario_uplift":   None,
        "kg_bridge_query":   kg_query_template(),
        "kg_bridge_note": (
            "After a high-risk stop prediction, plug the (service_id, "
            "station_id, date) tuple into kg_bridge_query to retrieve the "
            "station's recent faults and neighbours from the Knowledge Graph."
        ),
    }

    for scenario in scenarios:
        try:
            report["scenarios"][scenario] = run_for_scenario(scenario)
        except FileNotFoundError as e:
            log.error("Skipping scenario %s: %s", scenario, e)
            continue
        try:
            name, _ = pick_winner(scenario)
            report["lag_ablation"][scenario] = lag_ablation(name, scenario)
        except Exception as e:
            log.warning("Lag ablation failed for scenario %s: %s", scenario, e)

    if "A" in scenarios and "B" in scenarios:
        try:
            name_A, _ = pick_winner("A")
            name_B, _ = pick_winner("B")
            report["scenario_uplift"] = scenario_uplift_map(name_A, name_B)
        except Exception as e:
            log.warning("Scenario uplift failed: %s", e)

    out = RESULTS_DIR / "xai_report_stops.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Wrote %s", out)
    log.info("All XAI figures → %s", FIG_DIR)


if __name__ == "__main__":
    main()

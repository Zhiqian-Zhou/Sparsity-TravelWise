"""
stop_level/train_all.py
=============================================================================
CLI orchestrator for the model zoo.

Usage:
    python stop_level/train_all.py --model all --scenario both
    python stop_level/train_all.py --model lgbm --scenario A
    python stop_level/train_all.py --model logreg lgbm --scenario A
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import get_logger  # noqa: E402
from stop_level.leakage_guards import assert_no_leakage  # noqa: E402
from stop_level.models.base import write_predictions, quick_val_metrics  # noqa: E402

log = get_logger("train_all")

DATA_DIR = ROOT / "Data" / "stops"
MODEL_DIR = ROOT / "stop_level" / "models" / "_artefacts"
RESULTS_DIR = ROOT / "stop_level" / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts(scenario: str) -> dict:
    """Load splits + feature names + ctx (graph tensors, station_id_to_idx)."""
    train_df = pd.read_parquet(DATA_DIR / "stops_train.parquet")
    val_df   = pd.read_parquet(DATA_DIR / "stops_val.parquet")
    test_df  = pd.read_parquet(DATA_DIR / "stops_test.parquet")
    with open(DATA_DIR / f"feature_names_{scenario}.json") as f:
        feat_cols = json.load(f)
    with open(DATA_DIR / "station_id_to_idx.json") as f:
        sid_to_idx = json.load(f)
    edge_index = np.load(DATA_DIR / "edge_index.npy")
    edge_attr  = np.load(DATA_DIR / "edge_attr.npy")
    log.info("Loaded train=%d val=%d test=%d | feat_cols=%d (%s)",
             len(train_df), len(val_df), len(test_df), len(feat_cols), scenario)
    assert_no_leakage(feat_cols, scenario)
    # nodes_station for the GNN — derived from the data so we don't need the
    # original CSVs available at training time.
    return {
        "train_df":  train_df,
        "val_df":    val_df,
        "test_df":   test_df,
        "feat_cols": feat_cols,
        "ctx": {
            "edge_index":         edge_index,
            "edge_attr":          edge_attr,
            "station_id_to_idx":  sid_to_idx,
            "nodes_station":      _build_nodes_station(train_df),
        },
    }


def _build_nodes_station(train_df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a per-station static feature table from the parquet."""
    return (
        train_df.groupby("station_id")
                .agg(lat=("lat", "first"),
                     lon=("lon", "first"),
                     degree=("degree", "first"),
                     betweenness_centrality=("betweenness_centrality", "first"),
                     avg_historical_delay=("avg_historical_delay", "first"))
                .reset_index()
    )


def get_model(name: str, scenario: str, seed: int = 42):
    if name == "logreg":
        from stop_level.models.logreg import LogRegStopModel
        return LogRegStopModel(scenario, seed)
    if name == "lgbm":
        from stop_level.models.lgbm_model import LGBMStopModel
        return LGBMStopModel(scenario, seed)
    if name == "xgb":
        from stop_level.models.xgb_model import XGBStopModel
        return XGBStopModel(scenario, seed)
    if name == "graphsage":
        from stop_level.models.graphsage_model import GraphSAGEStopModel
        return GraphSAGEStopModel(scenario, seed)
    if name == "bilstm":
        from stop_level.models.bilstm_model import BiLSTMStopModel
        return BiLSTMStopModel(scenario, seed)
    raise ValueError(f"Unknown model {name!r}")


def train_one(model_name: str, scenario: str, art: dict, seed: int) -> dict:
    """Train one (model × scenario), persist preds + model, return metric snapshot."""
    log.info("── %s | scenario %s ─────────────────────────", model_name, scenario)
    t0 = time.time()
    model = get_model(model_name, scenario, seed=seed)

    # GraphSAGE wants a cached edge_index for predict_proba calls
    if model_name == "graphsage":
        model.fit_with_cache(art["train_df"], art["val_df"],
                              art["feat_cols"], ctx=art["ctx"])
    else:
        model.fit(art["train_df"], art["val_df"], art["feat_cols"], ctx=art["ctx"])

    fit_secs = time.time() - t0

    # Predictions
    y_val_prob  = model.predict_proba(art["val_df"])
    y_test_prob = model.predict_proba(art["test_df"])
    val_m  = quick_val_metrics(art["val_df"]["y_stop"].values,  y_val_prob,  model.threshold_)
    test_m = quick_val_metrics(art["test_df"]["y_stop"].values, y_test_prob, model.threshold_)

    # Persist
    out_dir = MODEL_DIR / model_name / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save(out_dir)
    except Exception as e:
        log.warning("save failed for %s/%s: %s (continuing)", model_name, scenario, e)

    write_predictions(art["val_df"],  y_val_prob,  model.threshold_, out_dir / "preds_val.parquet")
    write_predictions(art["test_df"], y_test_prob, model.threshold_, out_dir / "preds_test.parquet")

    log.info("[%s/%s] val pr_auc=%.3f f1=%.3f | test pr_auc=%.3f f1=%.3f | %.1fs",
             model_name, scenario, val_m["pr_auc"], val_m["f1"],
             test_m["pr_auc"], test_m["f1"], fit_secs)
    return {
        "model":     model_name,
        "scenario":  scenario,
        "threshold": float(model.threshold_),
        "fit_seconds": fit_secs,
        "val":       val_m,
        "test":      test_m,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["all"],
                        choices=["all", "logreg", "lgbm", "xgb", "graphsage", "bilstm"])
    parser.add_argument("--scenario", choices=["A", "B", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if "all" in args.model:
        models = ["logreg", "lgbm", "xgb", "graphsage", "bilstm"]
    else:
        models = args.model
    scenarios = ["A", "B"] if args.scenario == "both" else [args.scenario]

    # Global RNG seeding for reproducibility — covers pandas hash ops, numpy
    # samples, and (conditionally) the torch RNGs that the per-model seeds
    # don't touch. torch is imported only when a torch-using model is queued,
    # so tabular-only runs don't pay its startup cost — and don't crash on
    # broken numpy/torch ABI envs they would never have hit otherwise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    if any(m in models for m in ("graphsage", "bilstm")):
        try:
            import torch
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass

    rows: list[dict] = []
    for scenario in scenarios:
        art = load_artifacts(scenario)
        for name in models:
            try:
                rows.append(train_one(name, scenario, art, args.seed))
            except ImportError as e:
                log.error("[%s/%s] skipped: %s", name, scenario, e)
            except Exception as e:
                log.exception("[%s/%s] failed: %s", name, scenario, e)

    out = RESULTS_DIR / "benchmark_stops.json"
    with open(out, "w") as f:
        json.dump({"runs": rows}, f, indent=2, default=str)
    log.info("Wrote %s (%d runs)", out, len(rows))


if __name__ == "__main__":
    main()

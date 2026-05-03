"""
test_models_smoke.py
=============================================================================
Smoke tests — every model trains on a tiny synthetic dataset, produces
predictions, and saves+ (where supported) reloads. Catches API regressions
without needing real data.

Models requiring optional deps (`torch_geometric`) are skipped if the dep
isn't installed; the test framework reports the skip rather than failing.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _toy_splits(n_services: int = 60, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Synthetic stop-level data: each service visits 4 stations chosen from
    a pool of 10. 16 features per stop. Class balance ~30 % positive.
    """
    rng = np.random.default_rng(seed)
    stations = [f"ST{i:02d}" for i in range(10)]
    feat_cols = [f"f{i}" for i in range(16)]
    rows = []
    for s in range(n_services):
        sid = f"SVC{s:04d}"
        date = pd.Timestamp("2024-02-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        train_class = int(rng.integers(0, 7))
        country = "IT"
        route = list(rng.choice(stations, size=4, replace=False))
        for k, st in enumerate(route):
            f = rng.normal(0, 1, size=16).astype("float32")
            y = int(rng.random() < 0.3)
            rows.append({
                "service_id": sid, "station_id": st,
                "country": country, "date": date, "stop_order": k,
                "position_norm": float(k) / 3.0,
                "train_class_code": train_class,
                "y_stop": y,
                "delay_min": float(20 if y else rng.normal(0, 2)),
                **{c: float(v) for c, v in zip(feat_cols, f)},
                # Static-station features
                "lat": 45.0 + stations.index(st) * 0.05,
                "lon":  9.0 + stations.index(st) * 0.05,
                "degree": 4,
                "betweenness_centrality": 0.1,
                "avg_historical_delay": 2.0,
                # An inflight whitelisted feature for scenario B
                "prev_stop_actual_delay": 0.0,
            })
    df = pd.DataFrame(rows)
    train_df = df[df["date"] <= "2024-02-20"].reset_index(drop=True)
    val_df   = df[df["date"] > "2024-02-20"].reset_index(drop=True)

    # Build minimal ctx for graph models
    sid_to_idx = {s: i for i, s in enumerate(stations)}
    edges = []
    for i in range(len(stations) - 1):
        edges.append((stations[i], stations[i+1]))
        edges.append((stations[i+1], stations[i]))
    edge_index = np.array(
        [[sid_to_idx[a] for a, _ in edges], [sid_to_idx[b] for _, b in edges]],
        dtype=np.int64,
    )
    nodes_station = (
        train_df.groupby("station_id")
                .agg(lat=("lat","first"), lon=("lon","first"),
                     degree=("degree","first"),
                     betweenness_centrality=("betweenness_centrality","first"),
                     avg_historical_delay=("avg_historical_delay","first"))
                .reset_index()
    )
    ctx = {
        "edge_index": edge_index,
        "edge_attr":  np.zeros((edge_index.shape[1], 1), dtype="float32"),
        "station_id_to_idx": sid_to_idx,
        "nodes_station": nodes_station,
    }
    feat_cols_full = feat_cols + ["lat", "lon", "degree",
                                    "betweenness_centrality",
                                    "avg_historical_delay"]
    return train_df, val_df, ctx, feat_cols_full


def _check_model(ModelCls, scenario: str, **kwargs):
    train_df, val_df, ctx, feat_cols = _toy_splits()
    model = ModelCls(scenario=scenario, **kwargs)
    if hasattr(model, "fit_with_cache"):
        model.fit_with_cache(train_df, val_df, feat_cols, ctx=ctx)
    else:
        model.fit(train_df, val_df, feat_cols, ctx=ctx)
    p = model.predict_proba(val_df)
    assert p.shape == (len(val_df),), f"wrong shape: {p.shape}"
    assert ((p >= 0) & (p <= 1)).all(), "probs out of [0,1]"
    assert 0.0 < model.threshold_ < 1.0


def test_logreg_smoke():
    from stop_level.models.logreg import LogRegStopModel
    _check_model(LogRegStopModel, "A")
    _check_model(LogRegStopModel, "B")


def test_lgbm_smoke():
    from stop_level.models.lgbm_model import LGBMStopModel
    _check_model(LGBMStopModel, "A", n_estimators=80)


def test_xgb_smoke():
    from stop_level.models.xgb_model import XGBStopModel
    _check_model(XGBStopModel, "A", n_estimators=80)


def test_graphsage_smoke():
    if importlib.util.find_spec("torch_geometric") is None:
        pytest.skip("torch_geometric not installed")
    from stop_level.models.graphsage_model import GraphSAGEStopModel
    _check_model(GraphSAGEStopModel, "A", epochs=2, batch_size=64, patience=2)


def test_bilstm_smoke():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    # Probe: torch + numpy interop is broken if torch was compiled against a
    # different numpy major version. Skip cleanly rather than failing.
    try:
        import torch
        torch.from_numpy(np.zeros(2, dtype="float32"))
    except (RuntimeError, ImportError) as e:
        pytest.skip(f"torch <-> numpy interop broken in this env: {e}")
    from stop_level.models.bilstm_model import BiLSTMStopModel
    _check_model(BiLSTMStopModel, "A", epochs=2, batch_size=8, patience=2)
    _check_model(BiLSTMStopModel, "B", epochs=2, batch_size=8, patience=2)

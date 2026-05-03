"""
test_feature_shapes.py
=============================================================================
Sanity-check the feature builders' output shapes and dtypes on a tiny toy
DataFrame, so we catch regressions without needing the 16.6 M-row dataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stop_level import features as F  # noqa: E402


def _toy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    2 services × 3 stops, two distinct stations, with everything Phase 1
    would supply (delay, weather, cancelled, scheduled_time).
    """
    rows = [
        # service 1 — Italy
        ("IT_S1", "IT_AAA", "2024-02-15", 0.0, 1, 5.0, 0, 0,
         "2024-02-15T08:00:00Z"),
        ("IT_S1", "IT_BBB", "2024-02-15", 7.0, 2, 6.0, 0, 0,
         "2024-02-15T09:00:00Z"),
        ("IT_S1", "IT_AAA", "2024-02-15", 12.0, 1, 7.0, 0, 0,
         "2024-02-15T10:00:00Z"),
        # service 2 — Italy, next day
        ("IT_S2", "IT_AAA", "2024-02-16", 0.0, 0, 4.5, 0, 0,
         "2024-02-16T08:00:00Z"),
        ("IT_S2", "IT_BBB", "2024-02-16", 0.0, 1, 5.0, 0, 0,
         "2024-02-16T09:00:00Z"),
    ]
    cols = ["service_id", "station_id", "date", "delay_minutes",
            "weather_severity", "wind_speed", "cancelled", "_pad",
            "scheduled_time"]
    df = pd.DataFrame(rows, columns=cols).drop(columns=["_pad"])
    df["date"] = pd.to_datetime(df["date"])
    df["cancelled"] = df["cancelled"].astype(bool)

    nodes_station = pd.DataFrame([
        {"station_id": "IT_AAA", "lat": 45.0, "lon": 9.0,
         "avg_historical_delay": 2.0, "degree": 4},
        {"station_id": "IT_BBB", "lat": 45.5, "lon": 9.2,
         "avg_historical_delay": 3.0, "degree": 2},
    ])
    edges_adj = pd.DataFrame([
        {"station_from": "IT_AAA", "station_to": "IT_BBB", "distance_km": 12.0},
        {"station_from": "IT_BBB", "station_to": "IT_AAA", "distance_km": 12.0},
    ])
    nodes_service = pd.DataFrame([
        {"service_id": "IT_S1", "train_class_code": 2, "date": "2024-02-15",
         "is_disrupted": 1},
        {"service_id": "IT_S2", "train_class_code": 2, "date": "2024-02-16",
         "is_disrupted": 0},
    ])
    betweenness = {"IT_AAA": 0.4, "IT_BBB": 0.1}
    return df, nodes_station, edges_adj, {"betweenness": betweenness,
                                            "nodes_service": nodes_service}


def test_stop_position_shapes():
    df, *_ = _toy()
    out = F.add_stop_position(df)
    for col in ["stop_order", "n_total_stops", "position_norm",
                "is_origin", "is_terminus",
                "cum_scheduled_minutes", "cum_distance_km"]:
        assert col in out.columns, col
    # First stop of each service must be origin
    first = out.sort_values(["service_id", "stop_order"]).groupby("service_id").first()
    assert (first["is_origin"] == 1).all()
    # Last stop must be terminus
    last = out.sort_values(["service_id", "stop_order"]).groupby("service_id").last()
    assert (last["is_terminus"] == 1).all()


def test_stop_time_features_dtypes():
    df, *_ = _toy()
    out = F.add_stop_time(F.add_stop_position(df))
    assert out["scheduled_arrival_hour"].dtype == np.int8
    assert out["month_sin"].dtype == np.float32
    assert out["is_holiday"].dtype == np.int8


def test_station_static_join():
    df, ns, _, ctx = _toy()
    out = F.add_station_static(df, ns, ctx["betweenness"])
    assert "betweenness_centrality" in out.columns
    # IT_AAA has betweenness 0.4 in our toy
    aaa = out[out["station_id"] == "IT_AAA"]["betweenness_centrality"].iloc[0]
    assert np.isclose(aaa, 0.4)


def test_weather_diff_first_stop_zero():
    df, *_ = _toy()
    df = F.add_stop_position(df)
    out = F.add_weather_diff(df)
    first_per_service = (
        out.sort_values(["service_id", "stop_order"]).groupby("service_id").first()
    )
    assert (first_per_service["delta_severity_vs_prev_stop"] == 0).all()
    assert (first_per_service["delta_wind_vs_prev_stop"] == 0).all()


def test_inflight_first_stop_is_zero():
    df, *_ = _toy()
    df = F.add_stop_position(df)
    out = F.add_inflight_features(df)
    first = out.sort_values(["service_id", "stop_order"]).groupby("service_id").first()
    assert (first["prev_stop_actual_delay"] == 0).all()
    assert (first["cum_actual_delay_so_far"] == 0).all()


def test_full_pipeline_runs():
    """End-to-end on the toy frame — must not error."""
    df, ns, ea, ctx = _toy()
    df = F.add_stop_position(df)
    df = F.add_stop_time(df)
    df = F.add_station_static(df, ns, ctx["betweenness"])
    df = F.add_service_identity(df, ctx["nodes_service"], ea)
    df = F.add_weather_diff(df)
    df = F.compute_station_lag(df)
    df = F.compute_train_lag(df)
    df = F.compute_train_station_lag(df)
    df = F.compute_neighbour_signal(df, ea)
    df = F.add_inflight_features(df)

    expected = [
        "stop_order", "position_norm", "is_origin", "is_terminus",
        "month_sin", "month_cos", "is_holiday",
        "lat", "lon", "degree", "betweenness_centrality",
        "train_class_code", "country", "service_n_stops",
        "delta_severity_vs_prev_stop", "delta_wind_vs_prev_stop",
        "station_lag1_rate", "station_lag7_rate",
        "train_lag1_rate", "train_lag7_rate",
        "train_station_lag7_rate", "nbr_lag1_rate",
        "prev_stop_actual_delay", "cum_actual_delay_so_far",
    ]
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"missing features after pipeline: {missing}"
    assert len(df) == 5  # toy has 5 stops

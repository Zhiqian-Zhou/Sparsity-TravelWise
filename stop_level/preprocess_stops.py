"""
stop_level/preprocess_stops.py
=============================================================================
Phase 2 — unified stop-level preprocessing.

Reads the five standardized CSVs from each country (`Data/<C>/processed/`),
joins them at stop grain, computes anti-leakage features for both prediction
scenarios, fits a StandardScaler on the training rows only, builds graph
tensors for the GNN, and persists everything to `Data/stops/` as Parquet.

Two scenarios:
  A — pre-departure: features in `feature_names_A.json`. No same-service
      delay information allowed. The leakage guard `LEAKY_COLS_A` is
      enforced before persisting.
  B — inflight: same as A plus `prev_stop_actual_delay`,
      `cum_actual_delay_so_far`, `max_actual_delay_so_far`.

CLI:
    python stop_level/preprocess_stops.py
    python stop_level/preprocess_stops.py --sample 200000   # dev mode
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stop_level"))

from utils import get_logger, SCHEMA  # noqa: E402
from stop_level import features as F  # noqa: E402
from stop_level.splits import day_split, assert_no_service_overlap  # noqa: E402
from stop_level.leakage_guards import (  # noqa: E402
    assert_no_leakage, feature_columns,
    LEAKY_COLS_A, LEAKY_COLS_B,
)

log = get_logger("preprocess_stops")

COUNTRIES = ["Italy", "Finland", "Netherlands"]
PROC_DIRS = {c: ROOT / "Data" / c / "processed" for c in COUNTRIES}
OUT_DIR   = ROOT / "Data" / "stops"


# ── Load + concat ──────────────────────────────────────────────────────────────
def load_all_countries() -> dict[str, pd.DataFrame]:
    """Load and concatenate the 5 standardized CSVs from all three countries."""
    tables: dict[str, list[pd.DataFrame]] = {
        k: [] for k in ["nodes_station", "nodes_service",
                         "edges_stops_at", "edges_adjacent", "nodes_fault"]
    }
    for country in COUNTRIES:
        d = PROC_DIRS[country]
        log.info("Loading %s ...", country)
        for key in tables:
            path = d / f"{key}.csv"
            if not path.exists():
                log.warning("Missing %s", path)
                continue
            df = pd.read_csv(path, low_memory=False)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            tables[key].append(df)

    merged = {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}
    log.info("Merged: %s", {k: v.shape for k, v in merged.items()})
    required = {"nodes_station", "nodes_service", "edges_stops_at", "edges_adjacent"}
    missing  = required - set(merged)
    if missing:
        raise FileNotFoundError(
            f"Phase 1 outputs missing: {sorted(missing)}.\n"
            "Run the per-country preprocessing notebooks first:\n"
            "    bash preprocess/run_all_notebooks.sh"
        )
    if "nodes_fault" not in merged:
        merged["nodes_fault"] = pd.DataFrame(
            columns=SCHEMA["nodes_fault"]
        )
    return merged


def build_stop_grain(merged: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join stops × services → one row per (service, station) with date and
    train_class_code attached. The label is computed here from delay_minutes
    and persisted as `y_stop`.
    """
    stops = merged["edges_stops_at"].copy()
    svc = merged["nodes_service"][["service_id", "date", "train_class_code"]].copy()
    df = stops.merge(svc, on="service_id", how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # Per-row label (stop-level): late > 5 min OR cancelled
    cancelled = df.get("cancelled", pd.Series(False, index=df.index)).astype(bool)
    delay = pd.to_numeric(df["delay_minutes"], errors="coerce")
    df["y_stop"] = (
        (delay > F.DELAY_THRESHOLD_MIN) | cancelled
    ).astype("int8")
    df["delay_min"] = delay.astype("float32")  # kept for evaluation slicing only
    log.info("Stop grain: %d rows | y_stop rate=%.3f", len(df), df["y_stop"].mean())
    return df


# ── Topology features (one-time) ───────────────────────────────────────────────
def compute_betweenness(
    edges_adj: pd.DataFrame,
    station_ids: list[str],
) -> dict[str, float]:
    try:
        import networkx as nx
    except ImportError:
        log.warning("networkx not installed — betweenness defaults to 0.")
        return {sid: 0.0 for sid in station_ids}
    G = nx.DiGraph()
    G.add_nodes_from(station_ids)
    valid = edges_adj.dropna(subset=["station_from", "station_to"])
    for _, row in valid.iterrows():
        w = float(row.get("distance_km", 1.0)) or 1.0
        G.add_edge(row["station_from"], row["station_to"], weight=w)
    log.info("Computing betweenness for %d nodes ...", G.number_of_nodes())
    return nx.betweenness_centrality(G, normalized=True, weight="weight")


def build_graph_tensors(
    edges_adj: pd.DataFrame,
    sid_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    valid = edges_adj.dropna(subset=["station_from", "station_to"]).copy()
    valid = valid[
        valid["station_from"].isin(sid_to_idx) &
        valid["station_to"].isin(sid_to_idx)
    ]
    src = valid["station_from"].map(sid_to_idx).astype(int).values
    dst = valid["station_to"].map(sid_to_idx).astype(int).values
    dist = valid["distance_km"].fillna(0).astype("float32").values
    d_max = dist.max() if dist.max() > 0 else 1.0
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr  = (dist / d_max).reshape(-1, 1).astype("float32")
    log.info("edge_index %s | edge_attr %s", edge_index.shape, edge_attr.shape)
    return edge_index, edge_attr


# ── Feature assembly ───────────────────────────────────────────────────────────
def assemble_features(
    df: pd.DataFrame,
    merged: dict[str, pd.DataFrame],
    betweenness: dict[str, float],
) -> pd.DataFrame:
    log.info("→ stop position")
    df = F.add_stop_position(df)
    log.info("→ stop time")
    df = F.add_stop_time(df)
    log.info("→ station static")
    df = F.add_station_static(df, merged["nodes_station"], betweenness)
    log.info("→ service identity")
    df = F.add_service_identity(df, merged["nodes_service"], merged["edges_adjacent"])
    log.info("→ weather diff")
    df = F.add_weather_diff(df)
    log.info("→ station lag")
    df = F.compute_station_lag(df)
    log.info("→ train lag")
    df = F.compute_train_lag(df)
    log.info("→ train×station lag")
    df = F.compute_train_station_lag(df)
    log.info("→ neighbour signal")
    df = F.compute_neighbour_signal(df, merged["edges_adjacent"])
    log.info("→ fault context")
    df = F.compute_fault_context(df, merged.get("nodes_fault"))
    log.info("→ inflight features (scenario B)")
    df = F.add_inflight_features(df)

    # Drop helper columns and ensure required dtypes
    df = df.drop(columns=["_train_key"], errors="ignore")
    return df


# ── Persist ────────────────────────────────────────────────────────────────────
def persist(
    splits: dict[str, pd.DataFrame],
    feat_cols_A: list[str],
    feat_cols_B: list[str],
    scaler_A: StandardScaler,
    scaler_B: StandardScaler,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    sid_to_idx: dict[str, int],
    service_ids: list[str],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Parquet per split
    for name, df in splits.items():
        path = OUT_DIR / f"stops_{name}.parquet"
        df.to_parquet(path, index=False)
        log.info("Wrote %s (%d rows × %d cols)", path, len(df), df.shape[1])

    np.save(OUT_DIR / "edge_index.npy", edge_index)
    np.save(OUT_DIR / "edge_attr.npy",  edge_attr)
    np.save(OUT_DIR / "scaler_mean_A.npy",  scaler_A.mean_.astype("float32"))
    np.save(OUT_DIR / "scaler_scale_A.npy", scaler_A.scale_.astype("float32"))
    np.save(OUT_DIR / "scaler_mean_B.npy",  scaler_B.mean_.astype("float32"))
    np.save(OUT_DIR / "scaler_scale_B.npy", scaler_B.scale_.astype("float32"))

    with open(OUT_DIR / "feature_names_A.json", "w") as f:
        json.dump(feat_cols_A, f, indent=2)
    with open(OUT_DIR / "feature_names_B.json", "w") as f:
        json.dump(feat_cols_B, f, indent=2)
    with open(OUT_DIR / "station_id_to_idx.json", "w") as f:
        json.dump(sid_to_idx, f, indent=2)
    with open(OUT_DIR / "service_id_to_idx.json", "w") as f:
        json.dump({sid: i for i, sid in enumerate(service_ids)}, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Limit to N rows for dev runs (0 = full)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # 1. Load + join at stop grain
    merged = load_all_countries()
    df = build_stop_grain(merged)
    if args.sample > 0 and len(df) > args.sample:
        df = df.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        log.info("Sampled to %d rows for dev run", len(df))

    # 2. Topology features
    station_ids = merged["nodes_station"]["station_id"].drop_duplicates().tolist()
    sid_to_idx = {sid: i for i, sid in enumerate(station_ids)}
    betweenness = compute_betweenness(merged["edges_adjacent"], station_ids)
    edge_index, edge_attr = build_graph_tensors(merged["edges_adjacent"], sid_to_idx)

    # 3. Feature assembly
    df = assemble_features(df, merged, betweenness)

    # 4. Day-level split  (no service crosses splits — assertion below)
    splits_dict = {k: v.copy() for k, v in day_split(df).items()}
    assert_no_service_overlap(splits_dict)
    log.info("Splits — train=%d val=%d test=%d",
             len(splits_dict["train"]), len(splits_dict["val"]), len(splits_dict["test"]))

    # 5. Decide feature columns per scenario
    all_cols = list(df.columns)
    feat_A = feature_columns(all_cols, "A")
    feat_B = feature_columns(all_cols, "B")
    # Numeric only (gives a stable scaler input)
    train_df = splits_dict["train"]
    feat_A = [c for c in feat_A if pd.api.types.is_numeric_dtype(train_df[c])]
    feat_B = [c for c in feat_B if pd.api.types.is_numeric_dtype(train_df[c])]
    assert_no_leakage(feat_A, "A")
    assert_no_leakage(feat_B, "B")

    # 6. Fit scalers on TRAIN ONLY
    scaler_A = StandardScaler().fit(train_df[feat_A].astype("float32"))
    scaler_B = StandardScaler().fit(train_df[feat_B].astype("float32"))
    log.info("Scalers fit | feat_A=%d feat_B=%d", len(feat_A), len(feat_B))

    # 7. Persist
    service_ids = sorted(df["service_id"].unique().tolist())
    persist(splits_dict, feat_A, feat_B, scaler_A, scaler_B,
            edge_index, edge_attr, sid_to_idx, service_ids)

    # 8. Manifest
    manifest = {
        "version":         "stops_v1",
        "n_stops_total":   int(len(df)),
        "n_stations":      len(station_ids),
        "n_services":      len(service_ids),
        "n_edges":         int(edge_index.shape[1]),
        "feature_count_A": len(feat_A),
        "feature_count_B": len(feat_B),
        "leaky_cols_A":    sorted(LEAKY_COLS_A),
        "leaky_cols_B":    sorted(LEAKY_COLS_B),
        "splits": {
            name: {
                "rows":          int(len(d)),
                "y_stop_rate":   float(d["y_stop"].mean()) if len(d) else 0.0,
                "n_services":    int(d["service_id"].nunique()),
                "date_range":    [str(d["date"].min()), str(d["date"].max())],
            }
            for name, d in splits_dict.items()
        },
        "anti_leakage_notes": (
            "Scenario A bans every same-service delay column. Scenario B "
            "additionally allows prev_stop_actual_delay, "
            "cum_actual_delay_so_far, max_actual_delay_so_far. Lag features "
            "use strict `<` joins. Scaler fit on TRAIN rows only."
        ),
    }
    with open(OUT_DIR / "manifest_stops.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("Manifest written → %s", OUT_DIR / "manifest_stops.json")


if __name__ == "__main__":
    main()

"""
preprocess/lib_italy.py
=============================================================================
Heavy helpers for the Italy preprocessing notebook
(`italy_preprocessing.ipynb`).

The notebook orchestrates the pipeline and renders inspection plots; this
module owns the chunked streaming, sentinel cleaning, and the five CSV
builders. Every function returns a DataFrame so the notebook can plot
intermediate state.

Public surface used by the notebook:
    load_operations_chunked(...)   → (ops_df, chunk_stats_df)
    clean_delay_column(series)     → (delay_numeric, cancelled_mask)
    map_weather_severity(weather)  → ordinal Series
    build_nodes_station(...)       → DataFrame
    build_nodes_service(...)       → DataFrame
    build_edges_stops_at(...)      → DataFrame
    build_edges_adjacent(...)      → DataFrame
    build_nodes_fault(faults)      → DataFrame
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    DATE_START, DATE_END, SCHEMA,
    map_train_class, get_logger,
)

log = get_logger("italy_lib")

COUNTRY_PFX = "IT_"
DISRUPTION_THRESHOLD = 5  # minutes
CHUNK_SIZE = 250_000


# ── Sentinels & weather ────────────────────────────────────────────────────────
def clean_delay_column(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse Italian rail delay column with 'N' / 'S' sentinels."""
    s = series.astype(str).str.strip()
    cancelled_mask = s.eq("S")
    s = s.replace({"N": np.nan, "S": np.nan})
    delay_numeric = pd.to_numeric(s, errors="coerce").astype("float32")
    return delay_numeric, cancelled_mask


WEATHER_SEVERITY_MAP: dict[str, int] = {
    "sunny":            0,
    "partially cloudy": 0,
    "cloudy":           0,
    "light rain":       1,
    "moderate rain":    2,
    "heavy rain":       3,
    "light snow":       2,
    "moderate snow":    3,
    "heavy snow":       4,
}


def map_weather_severity(weather: pd.Series) -> pd.Series:
    """Map free-text Trenitalia weather strings (English labels) to an int8 0-4 ordinal scale."""
    s = weather.astype(str).str.strip().str.lower()
    mapped = s.map(WEATHER_SEVERITY_MAP)
    unknown = s[mapped.isna() & s.ne("nan")].value_counts()
    if not unknown.empty:
        log.warning("Unknown weather labels (defaulting to 0): %s",
                    unknown.head(10).to_dict())
    return mapped.fillna(0).astype("int8")


# ── Streaming load with per-chunk telemetry for the notebook ──────────────────
def load_operations_chunked(
    raw_path: Path,
    chunk_size: int = CHUNK_SIZE,
    return_stats: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stream the operations CSV in chunks and date-filter on the fly.

    Returns
    -------
    ops : DataFrame
        Concatenated chunks within [DATE_START, DATE_END].
    stats : DataFrame
        Per-chunk telemetry  (chunk_idx, raw_rows, kept_rows, cum_kept).
        Used by the notebook to plot streaming progress.
    """
    dtype_map = {
        "train_number":           "int32",
        "station_order":          "int16",
        "week":                   "int8",
        "wind_speed":             "float32",
        "temperature_min":        "float32",
        "temperature_max":        "float32",
        "scheduled_running_time": "str",
        "scheduled_stop_time":    "str",
        "initial_delay":          "str",
        "final_delay":            "str",
    }

    kept: list[pd.DataFrame] = []
    stats_rows: list[dict] = []
    cum_kept = 0
    log.info("Streaming %s ...", raw_path)

    reader = pd.read_csv(
        raw_path, dtype=dtype_map, low_memory=False,
        parse_dates=["date"], chunksize=chunk_size,
    )
    for i, chunk in enumerate(reader):
        mask = (chunk["date"] >= DATE_START) & (chunk["date"] <= DATE_END)
        kept_chunk = chunk.loc[mask].copy()
        cum_kept += len(kept_chunk)
        kept.append(kept_chunk)
        stats_rows.append({
            "chunk_idx": i + 1,
            "raw_rows":  len(chunk),
            "kept_rows": len(kept_chunk),
            "cum_kept":  cum_kept,
        })

    ops = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    stats = pd.DataFrame(stats_rows)
    log.info("Done. %d rows retained.", len(ops))
    return (ops, stats) if return_stats else ops


# ── nodes_station ─────────────────────────────────────────────────────────────
def build_nodes_station(
    ops: pd.DataFrame,
    stations_master: pd.DataFrame,
    mileage: pd.DataFrame,
) -> pd.DataFrame:
    """nodes_station.csv: station_id | lat | lon | avg_historical_delay | degree

    `avg_historical_delay` uses only the first month (Jan 2024) as a
    leakage-free historical reference. Computing it over the full Jan–Jun
    would bake val/test-month labels into a station-static feature consumed
    by every model.
    """
    _HISTORICAL_REF_END = pd.Timestamp("2024-01-31")
    hist = ops[ops["date"] <= _HISTORICAL_REF_END]
    ops_nc = hist[~hist["cancelled"]].dropna(subset=["arrival_delay"])
    delay_avg = (
        ops_nc.groupby("station_name", observed=True)["arrival_delay"]
              .mean().rename("avg_historical_delay")
    )

    mileage_s = mileage.sort_values(["train_id", "station_order"])
    edge_rows: list[tuple[str, str]] = []
    for _, grp in mileage_s.groupby("train_id"):
        g = grp.reset_index(drop=True)
        for i in range(len(g) - 1):
            edge_rows.append((g.loc[i, "station_name"], g.loc[i + 1, "station_name"]))
            edge_rows.append((g.loc[i + 1, "station_name"], g.loc[i, "station_name"]))
    edge_df = pd.DataFrame(edge_rows, columns=["from", "to"]).drop_duplicates()
    degree = edge_df.groupby("from")["to"].nunique().rename("degree")

    coords = stations_master[["name", "lat", "lon"]].copy()
    coords["name"] = coords["name"].str.strip().str.upper()

    all_names = ops["station_name"].dropna().unique()
    node_df = pd.DataFrame({"station_name": all_names})
    node_df = (
        node_df
        .merge(coords.rename(columns={"name": "station_name"}),
               on="station_name", how="left")
        .merge(delay_avg.reset_index(), on="station_name", how="left")
        .merge(degree.reset_index().rename(columns={"from": "station_name"}),
               on="station_name", how="left")
    )
    node_df["avg_historical_delay"] = node_df["avg_historical_delay"].fillna(0.0).astype("float32")
    node_df["degree"] = node_df["degree"].fillna(1).astype("int32")
    node_df["station_id"] = COUNTRY_PFX + node_df["station_name"].str.replace(" ", "_")

    log.info("nodes_station: %d stations  (coords missing: %d)",
             len(node_df), node_df["lat"].isna().sum())
    return node_df[SCHEMA["nodes_station"]]


# ── nodes_service ─────────────────────────────────────────────────────────────
def build_nodes_service(ops: pd.DataFrame) -> pd.DataFrame:
    """nodes_service.csv: service_id | train_class_code | date | is_disrupted"""
    def label_service(grp: pd.DataFrame) -> int:
        any_cancelled = grp["cancelled"].any()
        delays = grp["arrival_delay"].dropna()
        any_late = (delays > DISRUPTION_THRESHOLD).any()
        return int(any_cancelled or any_late)

    labels = (
        ops.groupby(["train_id", "date", "train_class"], observed=True)
           .apply(label_service, include_groups=False)
           .rename("is_disrupted").reset_index()
    )
    # Date-encode service_id so the same train running on different days yields
    # distinct services (matches FI/NL convention). Without this, Phase 2's
    # stops×services merge explodes ~23x because every stop matches every day.
    date_str = pd.to_datetime(labels["date"]).dt.strftime("%Y-%m-%d")
    labels["service_id"] = COUNTRY_PFX + labels["train_id"].astype(str) + "_" + date_str
    labels["train_class_code"] = map_train_class(labels["train_class"])
    return labels[SCHEMA["nodes_service"]].copy()


# ── edges_stops_at ────────────────────────────────────────────────────────────
def build_edges_stops_at(
    ops: pd.DataFrame,
    sid_map: dict[str, str],
) -> pd.DataFrame:
    """edges_stops_at.csv with weather columns; missing weather metrics → NaN (not 0)."""
    cols = ["train_id", "station_name", "arrival_delay", "weather_severity"]
    for c in ["temperature", "wind_speed", "precipitation", "snow_depth"]:
        if c not in ops.columns:
            ops[c] = np.nan
    cols.extend(["temperature", "wind_speed", "precipitation", "snow_depth"])

    tmp = ops[cols + ["date"]].copy()
    # Date-encode service_id to match nodes_service (above). Without this the
    # downstream Phase-2 merge between edges_stops_at and nodes_service
    # row-explodes because train_id alone repeats across days.
    date_str = pd.to_datetime(tmp["date"]).dt.strftime("%Y-%m-%d")
    tmp["service_id"]    = COUNTRY_PFX + tmp["train_id"].astype(str) + "_" + date_str
    tmp["station_id"]    = tmp["station_name"].map(sid_map)
    tmp["delay_minutes"] = tmp["arrival_delay"].astype("float32")
    return tmp.dropna(subset=["station_id"])[SCHEMA["edges_stops_at"]]


# ── edges_adjacent ────────────────────────────────────────────────────────────
def build_edges_adjacent(
    mileage: pd.DataFrame,
    sid_map: dict[str, str],
) -> pd.DataFrame:
    """Derive adjacent station pairs + distance_km from the mileage table."""
    mileage_s = mileage.sort_values(["train_id", "station_order"])
    rows: list[dict] = []
    for _, grp in mileage_s.groupby("train_id"):
        g = grp.reset_index(drop=True)
        for i in range(len(g) - 1):
            s_from = g.loc[i,   "station_name"].strip().upper()
            s_to   = g.loc[i+1, "station_name"].strip().upper()
            dist = abs(g.loc[i + 1, "distance"] - g.loc[i, "distance"])
            id_f, id_t = sid_map.get(s_from), sid_map.get(s_to)
            if id_f and id_t and id_f != id_t:
                rows.append({"station_from": id_f, "station_to": id_t, "distance_km": dist})
                rows.append({"station_from": id_t, "station_to": id_f, "distance_km": dist})
    if not rows:
        return pd.DataFrame(columns=SCHEMA["edges_adjacent"])
    result = (
        pd.DataFrame(rows)
          .groupby(["station_from", "station_to"])["distance_km"]
          .mean().reset_index()
    )
    result["distance_km"] = result["distance_km"].round(2)
    return result[SCHEMA["edges_adjacent"]]


# ── nodes_fault ───────────────────────────────────────────────────────────────
def build_nodes_fault(faults: pd.DataFrame) -> pd.DataFrame:
    """Empty IT nodes_fault: source `Train_fault_information.csv` is line-level
    (free-text route segments in `line` column), not station-level. There is
    no usable per-station identifier, so emitting rows with `station_id = NaN`
    would silently break `compute_fault_context` (drops on NaN station_id) and
    `build_kg.REPORTED_AT` (filters by valid station_id). Honest empty frame
    matches Finland's `build_nodes_fault_empty` convention."""
    log.info("IT nodes_fault: emitting empty frame (source has no per-station identifier)")
    return pd.DataFrame(columns=SCHEMA["nodes_fault"])

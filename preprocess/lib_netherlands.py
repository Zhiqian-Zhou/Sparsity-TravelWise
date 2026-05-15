"""
preprocess/lib_netherlands.py
=============================================================================
Heavy helpers for the Netherlands preprocessing notebook
(`netherlands_preprocessing.ipynb`).

NL specifics:
  • RDT export with `Service:Foo` / `Stop:Foo` colon-prefixed columns
  • `N` / `S` delay sentinels (same as Italy) + 3 boolean cancellation flags
  • Wide tariff-distance matrix (must be melted)
  • Disruptions log with comma-separated affected stations (must be exploded)
  • Open-Meteo weather enrichment via the existing weather_enrichment_nl module
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    DATE_START, DATE_END, SCHEMA,
    map_train_class, get_logger,
)

log = get_logger("netherlands_lib")

COUNTRY_PFX = "NL_"
DISRUPTION_THRESHOLD = 5

NL_RENAME = {
    "Service:RDT-ID":               "service_id_raw",
    "Service:Date":                 "date",
    "Service:Type":                 "train_type",
    "Service:Company":              "company",
    "Service:Train number":         "train_number",
    "Service:Completely cancelled": "fully_cancelled",
    "Service:Partly cancelled":     "partly_cancelled",
    "Service:Maximum delay":        "max_delay_min",
    "Stop:RDT-ID":                  "stop_id",
    "Stop:Station code":            "station_code",
    "Stop:Station name":            "station_name",
    "Stop:Arrival time":            "arrival_time",
    "Stop:Arrival delay":           "arrival_delay",
    "Stop:Arrival cancelled":       "arrival_cancelled",
    "Stop:Departure time":          "departure_time",
    "Stop:Departure delay":         "departure_delay",
    "Stop:Departure cancelled":     "departure_cancelled",
    "Stop:Platform change":         "platform_change",
    "Stop:Planned platform":        "planned_platform",
    "Stop:Actual platform":         "actual_platform",
}


# ── Sentinel cleaning ──────────────────────────────────────────────────────────
def clean_delay_sentinel(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.astype(str).str.strip()
    cancelled_mask = s.eq("S")
    s = s.replace({"N": np.nan, "S": np.nan})
    numeric = pd.to_numeric(s, errors="coerce").astype("float32")
    return numeric, cancelled_mask


# ── Load + filter + flag ───────────────────────────────────────────────────────
def load_services(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Glob services-2024-*.csv, rename, filter by date, clean delays, derive
    cancellation flags.

    Returns
    -------
    df       : DataFrame
    file_log : per-file row counts (for the notebook to plot)
    """
    files = sorted(glob.glob(str(raw_dir / "services-2024-*.csv")))
    if not files:
        raise FileNotFoundError(f"No services-2024-*.csv in {raw_dir}")

    chunks, file_log = [], []
    for path in files:
        chunk = pd.read_csv(path, dtype=str, low_memory=False)
        chunk.rename(columns=NL_RENAME, inplace=True)
        chunks.append(chunk)
        file_log.append({"file": Path(path).name, "rows": len(chunk)})

    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()

    df["arrival_delay"],   arr_c = clean_delay_sentinel(df["arrival_delay"])
    df["departure_delay"], dep_c = clean_delay_sentinel(df["departure_delay"])
    df["max_delay_min"] = pd.to_numeric(df["max_delay_min"], errors="coerce").astype("float32")

    def to_bool(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        return (
            df[col].astype(str).str.lower()
                   .map({"true": True, "false": False})
                   .fillna(False).astype(bool)
        )

    cancel_components = {
        "S_arrival":           arr_c.astype(bool),
        "S_departure":         dep_c.astype(bool),
        "fully_cancelled":     to_bool("fully_cancelled"),
        "arrival_cancelled":   to_bool("arrival_cancelled"),
        "departure_cancelled": to_bool("departure_cancelled"),
    }
    cancel_df = pd.DataFrame(cancel_components, index=df.index)
    df["cancelled"] = cancel_df.any(axis=1)

    df["station_code"] = df["station_code"].astype(str).str.strip().str.upper()
    df["train_type"]   = df["train_type"].astype(str).str.strip()

    return df, pd.DataFrame(file_log)


def cancellation_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-source counts of the cancellation flag for plotting."""
    sources = {
        "fully_cancelled":     df.get("fully_cancelled"),
        "arrival_cancelled":   df.get("arrival_cancelled"),
        "departure_cancelled": df.get("departure_cancelled"),
    }
    rows = []
    for name, ser in sources.items():
        if ser is None:
            continue
        b = ser.astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
        rows.append({"source": name, "n_true": int(b.sum())})
    rows.append({"source": "union (cancelled)", "n_true": int(df["cancelled"].sum())})
    return pd.DataFrame(rows)


# Reference window for "historical" station-level statistics. Computing
# avg_historical_delay over the full Jan–Jun would bake val/test labels into a
# station-static feature consumed by all models. Restricting to the first
# month gives a leakage-free historical reference that's still representative.
_HISTORICAL_REF_END = pd.Timestamp("2024-01-31")


# ── Builders ───────────────────────────────────────────────────────────────────
def build_nodes_station(
    df: pd.DataFrame,
    stations_master: pd.DataFrame,
) -> pd.DataFrame:
    # Use only the first month as the historical reference so val/test labels
    # cannot leak into the station-static avg_historical_delay feature.
    hist = df[df["date"] <= _HISTORICAL_REF_END]
    ops_nc = hist[~hist["cancelled"]].dropna(subset=["arrival_delay"])
    delay_avg = (
        ops_nc.groupby("station_code", observed=True)["arrival_delay"]
              .mean().rename("avg_historical_delay")
    )
    degree = (
        df.groupby("station_code", observed=True)["service_id_raw"]
          .nunique().rename("degree")
    )
    coords = stations_master[["code", "geo_lat", "geo_lng"]].copy()
    coords.rename(columns={"code": "station_code",
                            "geo_lat": "lat", "geo_lng": "lon"}, inplace=True)
    coords["station_code"] = coords["station_code"].str.strip().str.upper()

    node_df = (
        pd.DataFrame({"station_code": df["station_code"].dropna().unique()})
          .merge(coords,                  on="station_code", how="left")
          .merge(delay_avg.reset_index(), on="station_code", how="left")
          .merge(degree.reset_index(),    on="station_code", how="left")
    )
    node_df["avg_historical_delay"] = node_df["avg_historical_delay"].fillna(0.0).astype("float32")
    node_df["degree"] = node_df["degree"].fillna(1).astype("int32")
    node_df["station_id"] = COUNTRY_PFX + node_df["station_code"]
    return node_df[SCHEMA["nodes_station"]]


def build_nodes_service(df: pd.DataFrame) -> pd.DataFrame:
    def label_service(grp: pd.DataFrame) -> int:
        any_cancelled = grp["cancelled"].any()
        delays = grp["arrival_delay"].dropna()
        return int(any_cancelled or (delays > DISRUPTION_THRESHOLD).any())

    labels = (
        df.groupby(["service_id_raw", "date", "train_type"], observed=True)
          .apply(label_service, include_groups=False)
          .rename("is_disrupted").reset_index()
    )
    labels["service_id"] = COUNTRY_PFX + labels["service_id_raw"].astype(str)
    labels["train_class_code"] = map_train_class(labels["train_type"])
    return labels[SCHEMA["nodes_service"]]


def build_edges_stops_at(
    df: pd.DataFrame,
    sid_map: dict[str, str],
) -> pd.DataFrame:
    cols = ["service_id_raw", "station_code", "arrival_delay",
            "weather_severity", "temperature", "wind_speed",
            "precipitation", "snow_depth"]
    # Required base columns must already exist; if missing it's a real error.
    for c in ["service_id_raw", "station_code", "arrival_delay"]:
        if c not in df.columns:
            raise KeyError(f"Required column missing from NL services frame: {c}")
    # Optional weather columns: missing → NaN (not 0; 0 is a valid measurement
    # for temperature/wind/precip/snow and would silently inject fake data).
    # `weather_severity` is an ordinal where 0 means "clear" → keep 0 as the
    # honest default if missing.
    if "weather_severity" not in df.columns:
        df["weather_severity"] = 0
    for c in ["temperature", "wind_speed", "precipitation", "snow_depth"]:
        if c not in df.columns:
            df[c] = np.nan
    tmp = df[cols].copy()
    tmp["service_id"]    = COUNTRY_PFX + tmp["service_id_raw"].astype(str)
    tmp["station_id"]    = tmp["station_code"].map(sid_map)
    tmp["delay_minutes"] = tmp["arrival_delay"].astype("float32")
    return tmp.dropna(subset=["station_id"])[SCHEMA["edges_stops_at"]]


def build_edges_adjacent(
    dist_path: Path,
    sid_map: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    """
    Melt the wide tariff-distance matrix to an edge list.

    Returns
    -------
    edges_df : DataFrame
    info     : dict  with sparsity statistics for the notebook to plot.
    """
    dist_raw = pd.read_csv(dist_path, index_col=0)
    n_total = dist_raw.shape[0] * dist_raw.shape[1]
    n_xxx   = (dist_raw == "XXX").sum().sum()
    dist_raw.replace("XXX", np.nan, inplace=True)
    dist_raw = dist_raw.apply(pd.to_numeric, errors="coerce")
    n_nan = int(dist_raw.isna().sum().sum())

    dist_long = (
        dist_raw.stack().reset_index()
                .rename(columns={"Station": "from_code",
                                  "level_1": "to_code",
                                  0:         "distance_km"})
                .dropna(subset=["distance_km"])
    )
    dist_long["station_from"] = dist_long["from_code"].str.strip().str.upper().map(sid_map)
    dist_long["station_to"]   = dist_long["to_code"].str.strip().str.upper().map(sid_map)
    result = (
        dist_long.dropna(subset=["station_from", "station_to"])
                 .query("station_from != station_to")
                 [SCHEMA["edges_adjacent"]]
    )
    result["distance_km"] = result["distance_km"].astype("float32").round(2)
    info = {
        "matrix_shape": tuple(dist_raw.shape),
        "n_total":      int(n_total),
        "n_xxx_diag":   int(n_xxx),
        "n_nan":        int(n_nan),
        "n_edges":      int(len(result)),
    }
    return result, info


def build_nodes_fault(
    disruptions: pd.DataFrame,
    sid_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build NL nodes_fault by exploding `rdt_station_codes`.

    Returns
    -------
    result    : standardized nodes_fault rows
    cause_top : top-20 causes for the notebook bar chart
    """
    dis = disruptions.copy()
    dis["start_time"] = pd.to_datetime(dis["start_time"], errors="coerce")
    dis = dis[(dis["start_time"] >= DATE_START) &
              (dis["start_time"] <= DATE_END)].copy()
    dis["date"] = dis["start_time"].dt.normalize()

    dis_exp = (
        dis.assign(station_code=dis["rdt_station_codes"].str.split(","))
           .explode("station_code")
    )
    dis_exp["station_code"] = dis_exp["station_code"].astype(str).str.strip().str.upper()
    dis_exp["station_id"]   = dis_exp["station_code"].map(sid_map)
    dis_exp["fault_id"]     = COUNTRY_PFX + "FAULT_" + dis_exp["rdt_id"].astype(str)
    dis_exp["description"]  = (
        dis_exp["cause_en"].astype(str) + " | " +
        dis_exp.get("cause_group", "").astype(str)
    ).str[:300]

    result = dis_exp[SCHEMA["nodes_fault"]].copy()

    cause_top = (
        dis_exp.groupby("cause_en", dropna=True)["fault_id"]
               .nunique().rename("n_faults")
               .sort_values(ascending=False).head(20).reset_index()
    )
    return result, cause_top

"""
preprocess/lib_finland.py
=============================================================================
Heavy helpers for the Finland preprocessing notebook
(`finland_preprocessing.ipynb`).

Finland's challenge is the FI-TW format: one row per train with a serialized
`timeTableRows` cell containing all stops. This module:
  • parses and explodes that cell to one row per stop
  • flattens the nested `weather_observations` dict
  • applies the FI-TW imputation conventions
  • builds the five standardized CSVs

The notebook drives this pipeline step-by-step and renders inspection plots.
"""
from __future__ import annotations

import ast
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (
    DATE_START, DATE_END, SCHEMA,
    map_train_class, get_logger,
)

log = get_logger("finland_lib")

COUNTRY_PFX = "FI_"
DISRUPTION_THRESHOLD = 5
WEATHER_COLS = [
    "air_temp", "wind_speed", "wind_gust", "precipitation_1h",
    "snow_depth", "cloud_cover", "visibility", "dew_point", "humidity",
]

ALIAS_MAP = {
    "trainNumber":             "train_number",
    "departureDate":           "departure_date",
    "trainType":               "train_type",
    "trainCategory":           "train_category",
    "operatorShortCode":       "operator_short_code",
    "operatorUICCode":         "operator_uic_code",
    "commuterLineID":          "commuter_line_id",
    "runningCurrently":        "running_currently",
    "timetableType":           "timetable_type",
    "timetableAcceptanceDate": "timetable_acceptance_date",
    "stationShortCode":        "station_code",
    "differenceInMinutes":     "delay_minutes",
    "scheduledTime":           "scheduled_time",
    "type":                    "stop_type",
    "trainStopping":           "train_stopping",
    "commercialStop":          "commercial_stop",
    "commercialTraffic":       "commercial_traffic",
    "airTemperature":          "air_temp",
    "windSpeed":               "wind_speed",
    "windGust":                "wind_gust",
    "precipitation1h":         "precipitation_1h",
    "snowDepth":               "snow_depth",
    "cloudAmount":             "cloud_cover",
    "horizontalVisibility":    "visibility",
    "dewPoint":                "dew_point",
    "relativeHumidity":        "humidity",
}


# ── Parse + explode timeTableRows ──────────────────────────────────────────────
def parse_time_table_rows(raw_text: str) -> tuple[list[dict], str]:
    """
    Parse the serialized timeTableRows cell.

    Returns
    -------
    parsed : list[dict]
        Stops as dicts (empty list on failure).
    status : str
        "success" | "empty" | "parse_error"
    """
    if not isinstance(raw_text, str) or raw_text.strip() in ("", "nan", "None"):
        return [], "empty"

    text = re.sub(r"(?<![A-Za-z_])nan(?![A-Za-z_])", "None", raw_text)
    try:
        parsed = ast.literal_eval(text)
        parsed = parsed if isinstance(parsed, list) else [parsed]
        return parsed, "success"
    except (ValueError, SyntaxError):
        return [], "parse_error"


def explode_train_row(base: dict, stops: list[dict]) -> list[dict]:
    """Combine train-level fields with per-stop fields and rename via ALIAS_MAP."""
    if not stops:
        return [{ALIAS_MAP.get(k, k): v for k, v in base.items()}]
    result: list[dict] = []
    for stop in stops:
        stop = dict(stop)
        weather = stop.pop("weather_observations", None)
        row = {**base, **stop}
        if isinstance(weather, dict):
            row.update(weather)
        result.append({ALIAS_MAP.get(k, k): v for k, v in row.items()})
    return result


def load_and_explode(
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read every matched_data_2024_*.csv file, parse `timeTableRows`,
    explode to one row per stop, return:
        df         : flat DataFrame (one row per stop)
        parse_log  : per-file telemetry  (file, n_trains, n_stops,
                                          n_success, n_empty, n_error)
    """
    files = sorted(glob.glob(str(raw_dir / "matched_data_2024_*.csv")))
    if not files:
        raise FileNotFoundError(f"No matched_data_2024_*.csv in {raw_dir}")

    all_chunks: list[pd.DataFrame] = []
    parse_log: list[dict] = []

    for path in files:
        raw_chunk = pd.read_csv(path, dtype=str, low_memory=False)
        has_ttr = "timeTableRows" in raw_chunk.columns
        n_trains = len(raw_chunk)

        if not has_ttr:
            raw_chunk.rename(columns=ALIAS_MAP, inplace=True)
            all_chunks.append(raw_chunk)
            parse_log.append({
                "file":      Path(path).name,
                "n_trains":  n_trains,
                "n_stops":   len(raw_chunk),
                "n_success": n_trains, "n_empty": 0, "n_error": 0,
            })
            continue

        base_cols = [c for c in raw_chunk.columns if c != "timeTableRows"]
        expanded: list[dict] = []
        n_success = n_empty = n_error = 0

        for _, row in raw_chunk.iterrows():
            base = row[base_cols].to_dict()
            stops, status = parse_time_table_rows(row["timeTableRows"])
            n_success += int(status == "success")
            n_empty   += int(status == "empty")
            n_error   += int(status == "parse_error")
            expanded.extend(explode_train_row(base, stops))

        chunk_df = pd.DataFrame.from_records(expanded)
        all_chunks.append(chunk_df)
        parse_log.append({
            "file":      Path(path).name,
            "n_trains":  n_trains,
            "n_stops":   len(chunk_df),
            "n_success": n_success,
            "n_empty":   n_empty,
            "n_error":   n_error,
        })

    df = pd.concat(all_chunks, ignore_index=True)
    return df, pd.DataFrame(parse_log)


# ── Type coercion + date filter + cancellation parsing ─────────────────────────
def coerce_types(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply numeric / boolean / datetime coercions.

    Returns
    -------
    df : DataFrame   coerced
    nan_log : DataFrame   per-column NaN rate before/after numeric conversion
    """
    if "departure_date" in df.columns:
        df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

    if "scheduled_time" in df.columns:
        df["scheduled_time"] = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)

    if "delay_minutes" in df.columns:
        df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").astype("float32")
    else:
        df["delay_minutes"] = np.float32(0.0)

    bool_map = {"true": True, "false": False, "True": True, "False": False,
                "1": True, "0": False, True: True, False: False}
    if "cancelled" in df.columns:
        df["cancelled"] = df["cancelled"].map(bool_map).fillna(False).astype(bool)
    else:
        df["cancelled"] = False

    nan_log_rows: list[dict] = []
    for col in WEATHER_COLS + ["delay_minutes"]:
        if col not in df.columns:
            continue
        before = df[col].isna().mean() if df[col].dtype.kind in "fiu" else 1.0
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        after = df[col].isna().mean()
        nan_log_rows.append({"column": col, "nan_rate": float(after),
                             "nan_rate_before": float(before)})

    return df, pd.DataFrame(nan_log_rows)


def filter_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep DATE_START..DATE_END; return (df_filtered, info)."""
    before = len(df)
    df = df[(df["departure_date"] >= DATE_START) & (df["departure_date"] <= DATE_END)].copy()
    return df, {"rows_before": before, "rows_after": len(df)}


# ── Imputation ─────────────────────────────────────────────────────────────────
def impute_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """FI-TW imputation conventions; returns (df, log_df) for plotting."""
    log_rows: list[dict] = []
    for zero_col in ["precipitation_1h", "snow_depth"]:
        if zero_col in df.columns:
            n_nan = int(df[zero_col].isna().sum())
            df[zero_col] = df[zero_col].fillna(0.0)
            log_rows.append({"column": zero_col, "strategy": "zero_fill",
                             "n_imputed": n_nan})
    for med_col in ["air_temp", "visibility"]:
        if med_col in df.columns and df[med_col].isna().any():
            month_num = df["departure_date"].dt.month
            month_med = df.groupby(month_num)[med_col].transform("median")
            global_med = df[med_col].median()
            n_nan = int(df[med_col].isna().sum())
            df[med_col] = df[med_col].fillna(month_med).fillna(global_med)
            log_rows.append({"column": med_col, "strategy": "monthly_median",
                             "n_imputed": n_nan,
                             "global_median": float(global_med)})
    return df, pd.DataFrame(log_rows)


def compute_weather_severity(df: pd.DataFrame) -> pd.Series:
    """0-4 ordinal from precipitation, snow depth and visibility."""
    sev = pd.Series(0, index=df.index, dtype="int8")
    if "precipitation_1h" in df.columns:
        p = df["precipitation_1h"].fillna(0.0)
        sev = sev.where(p < 0.5, np.int8(1))
        sev = sev.where(p < 2.0, np.int8(2))
        sev = sev.where(p < 5.0, np.int8(3))
    if "snow_depth" in df.columns:
        s = df["snow_depth"].fillna(0.0)
        m_l = (s >= 5)  & (s < 15)
        m_m = (s >= 15) & (s < 40)
        m_h = (s >= 40)
        sev = sev.where(~m_l, sev.clip(lower=2).astype("int8"))
        sev = sev.where(~m_m, sev.clip(lower=3).astype("int8"))
        sev = sev.where(~m_h, np.int8(4))
    if "visibility" in df.columns:
        low_vis = df["visibility"].fillna(9999) < 200
        sev = sev.where(~low_vis, sev.clip(lower=3).astype("int8"))
    return sev.clip(0, 4).astype("int8")


def normalize_codes(df: pd.DataFrame) -> pd.DataFrame:
    if "station_code" not in df.columns:
        for alt in ["stationShortCode", "stationShortcode", "station_short_code"]:
            if alt in df.columns:
                df.rename(columns={alt: "station_code"}, inplace=True)
                break
    df["station_code"] = df["station_code"].astype(str).str.strip().str.upper()
    if "train_type" not in df.columns:
        df["train_type"] = "OTHER"
    df["train_type"] = df["train_type"].astype(str).str.strip().str.upper()
    if "train_number" in df.columns:
        df["train_number"] = pd.to_numeric(df["train_number"], errors="coerce")
    return df


# ── Builders for the 5 standardized CSVs ───────────────────────────────────────
def build_station_coords(df: pd.DataFrame) -> pd.DataFrame:
    coord_cols = [c for c in ["lat", "lon"] if c in df.columns]
    if not coord_cols:
        return pd.DataFrame({"station_code": df["station_code"].unique(),
                             "lat": np.nan, "lon": np.nan})
    for c in coord_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.groupby("station_code")[["lat", "lon"]].first().reset_index()


def build_nodes_station(df: pd.DataFrame) -> pd.DataFrame:
    ops_nc = df[~df["cancelled"]].dropna(subset=["delay_minutes"])
    delay_avg = (
        ops_nc.groupby("station_code")["delay_minutes"]
              .mean().rename("avg_historical_delay")
    )
    id_col = "train_number" if "train_number" in df.columns else "train_type"
    degree = df.groupby("station_code")[id_col].nunique().rename("degree")
    coords = build_station_coords(df)
    node_df = (
        coords
        .merge(delay_avg.reset_index(), on="station_code", how="left")
        .merge(degree.reset_index(),    on="station_code", how="left")
    )
    node_df["avg_historical_delay"] = node_df["avg_historical_delay"].fillna(0.0).astype("float32")
    node_df["degree"] = node_df["degree"].fillna(1).astype("int32")
    node_df["station_id"] = COUNTRY_PFX + node_df["station_code"]
    return node_df[SCHEMA["nodes_station"]]


def build_nodes_service(df: pd.DataFrame) -> pd.DataFrame:
    id_col = "train_number" if "train_number" in df.columns else "train_type"
    group_key = [c for c in [id_col, "departure_date", "train_type"] if c in df.columns]

    def label_service(grp: pd.DataFrame) -> int:
        return int(grp["cancelled"].any() or
                   (grp["delay_minutes"].dropna() > DISRUPTION_THRESHOLD).any())

    labels = (
        df.groupby(group_key)
          .apply(label_service, include_groups=False)
          .rename("is_disrupted").reset_index()
    )
    labels["service_id"] = (
        COUNTRY_PFX + labels[id_col].astype(str)
        + "_" + labels["departure_date"].astype(str)
    )
    labels["train_class_code"] = map_train_class(labels["train_type"])
    labels["date"] = labels["departure_date"]
    return labels[SCHEMA["nodes_service"]]


def build_edges_stops_at(
    df: pd.DataFrame,
    sid_map: dict[str, str],
) -> pd.DataFrame:
    id_col = "train_number" if "train_number" in df.columns else "train_type"
    cols = [id_col, "departure_date", "station_code", "delay_minutes",
            "weather_severity", "air_temp", "wind_speed",
            "precipitation_1h", "snow_depth"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0

    tmp = df[cols].copy()
    tmp["service_id"] = (
        COUNTRY_PFX + tmp[id_col].astype(str)
        + "_" + tmp["departure_date"].astype(str)
    )
    tmp["station_id"]    = tmp["station_code"].map(sid_map)
    tmp["delay_minutes"] = tmp["delay_minutes"].astype("float32")
    tmp.rename(columns={"air_temp": "temperature",
                         "precipitation_1h": "precipitation"}, inplace=True)
    return tmp.dropna(subset=["station_id"])[SCHEMA["edges_stops_at"]]


def build_edges_adjacent(
    df: pd.DataFrame,
    sid_map: dict[str, str],
) -> pd.DataFrame:
    id_col = "train_number" if "train_number" in df.columns else "train_type"
    order_col = next(
        (c for c in ["scheduled_time", "station_order", "stop_sequence"] if c in df.columns),
        None,
    )
    if order_col is None:
        return pd.DataFrame(columns=SCHEMA["edges_adjacent"])

    sort_keys = [c for c in [id_col, "departure_date", order_col] if c in df.columns]
    df_s = df.sort_values(sort_keys)
    rows: list[dict] = []
    for _, grp in df_s.groupby([id_col, "departure_date"], sort=False):
        g = grp.reset_index(drop=True)
        for i in range(len(g) - 1):
            s_f, s_t = g.loc[i, "station_code"], g.loc[i + 1, "station_code"]
            id_f, id_t = sid_map.get(s_f), sid_map.get(s_t)
            if id_f and id_t and id_f != id_t:
                rows.append({"station_from": id_f, "station_to": id_t, "distance_km": 0.0})
                rows.append({"station_from": id_t, "station_to": id_f, "distance_km": 0.0})
    if not rows:
        return pd.DataFrame(columns=SCHEMA["edges_adjacent"])
    return (
        pd.DataFrame(rows)
          .groupby(["station_from", "station_to"])["distance_km"]
          .mean().reset_index()
    )[SCHEMA["edges_adjacent"]]


def build_nodes_fault_empty() -> pd.DataFrame:
    """Finland has no structured fault log."""
    return pd.DataFrame(columns=SCHEMA["nodes_fault"])

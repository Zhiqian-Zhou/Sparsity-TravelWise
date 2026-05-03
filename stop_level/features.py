"""
stop_level/features.py
=============================================================================
Feature builders for the stop-level pipeline.

Each function takes (and returns) a flat per-stop DataFrame and adds one
group of columns. Functions are pure (no I/O) so they can be unit-tested.

The lag builders share a single contract:
    • build a daily aggregation table per (key, date)
    • forward-fill onto a contiguous calendar so shift(1) is meaningful
    • shift / rolling-mean STRICTLY less than current date
    • merge back onto the per-stop df

Strict `<` is the load-bearing invariant — every lag join filters with
`prior_date == current_date - timedelta(days=1)` (or rolling on the
prior-N-days window), never `<=`.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import DATE_START, DATE_END  # noqa: E402

DELAY_THRESHOLD_MIN = 5

HOLIDAYS_2024 = {
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-04-25", "2024-05-01",
    "2024-05-09", "2024-05-12", "2024-05-17", "2024-05-20", "2024-06-21",
}
_HOLIDAY_DATES = pd.to_datetime(list(HOLIDAYS_2024)).normalize()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _calendar() -> pd.DatetimeIndex:
    """Inclusive calendar over the full pipeline window."""
    return pd.date_range(DATE_START, DATE_END, freq="D")


def _stop_late(df: pd.DataFrame) -> pd.Series:
    """
    Compute the stop-level binary label without leaking into X.
    Used internally to build aggregate stats; never persisted as a feature.
    """
    delay = pd.to_numeric(df["delay_minutes"], errors="coerce")
    cancelled = df.get("cancelled", pd.Series(False, index=df.index)).astype(bool)
    return ((delay > DELAY_THRESHOLD_MIN) | cancelled).astype("int8")


# ── Stop position within the service ───────────────────────────────────────────
def add_stop_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stop_order, position_norm, is_origin, is_terminus, n_total_stops,
    cum_distance_km, cum_scheduled_minutes.

    Ordering key: scheduled_time if available, else the row order within
    each service (stable as produced by Phase 1).
    """
    df = df.copy()
    sort_keys = ["service_id"]
    if "scheduled_time" in df.columns:
        sort_keys.append("scheduled_time")

    df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    df["stop_order"] = df.groupby("service_id").cumcount().astype("int32")
    n_stops = df.groupby("service_id")["stop_order"].transform("max") + 1
    df["n_total_stops"] = n_stops.astype("int32")
    df["position_norm"] = (df["stop_order"] / (df["n_total_stops"] - 1).clip(lower=1)).astype("float32")
    df["is_origin"]   = (df["stop_order"] == 0).astype("int8")
    df["is_terminus"] = (df["stop_order"] == df["n_total_stops"] - 1).astype("int8")

    # Cumulative scheduled minutes within service (best-effort; 0 if unavailable)
    if "scheduled_time" in df.columns:
        sch = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)
        first = df.groupby("service_id")["scheduled_time"].transform("min")
        first = pd.to_datetime(first, errors="coerce", utc=True)
        cum_min = (sch - first).dt.total_seconds() / 60.0
        df["cum_scheduled_minutes"] = cum_min.fillna(0).astype("float32")
    else:
        df["cum_scheduled_minutes"] = np.float32(0.0)

    df["cum_distance_km"] = np.float32(0.0)  # filled by add_service_identity
    return df


# ── Stop time / calendar ───────────────────────────────────────────────────────
def add_stop_time(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features: scheduled_arrival_hour, dow, is_holiday, month, sin/cos."""
    df = df.copy()
    if "scheduled_time" in df.columns:
        ts = pd.to_datetime(df["scheduled_time"], errors="coerce", utc=True)
        df["scheduled_arrival_hour"] = ts.dt.hour.fillna(12).astype("int8")
    else:
        df["scheduled_arrival_hour"] = np.int8(12)
    d = pd.to_datetime(df["date"]).dt.normalize()
    df["scheduled_arrival_dow"] = d.dt.dayofweek.astype("int8")
    df["month"] = d.dt.month.astype("int8")
    df["is_holiday"] = d.isin(_HOLIDAY_DATES).astype("int8")
    angle = 2 * np.pi * df["month"] / 12
    df["month_sin"] = np.sin(angle).astype("float32")
    df["month_cos"] = np.cos(angle).astype("float32")
    return df


# ── Station-static features ────────────────────────────────────────────────────
def add_station_static(
    df: pd.DataFrame,
    nodes_station: pd.DataFrame,
    betweenness: dict[str, float],
) -> pd.DataFrame:
    """Join lat, lon, degree, betweenness_centrality, avg_historical_delay."""
    df = df.copy()
    sta = nodes_station.set_index("station_id")
    for col in ["lat", "lon", "degree", "avg_historical_delay"]:
        if col in sta.columns:
            df[col] = df["station_id"].map(sta[col]).fillna(0).astype("float32")
        else:
            df[col] = np.float32(0.0)
    df["betweenness_centrality"] = (
        df["station_id"].map(betweenness).fillna(0).astype("float32")
    )
    return df


# ── Service identity / route aggregates ────────────────────────────────────────
def add_service_identity(
    df: pd.DataFrame,
    nodes_service: pd.DataFrame,
    edges_adjacent: pd.DataFrame,
) -> pd.DataFrame:
    """train_class_code, country, service_n_stops, service_route_distance_km, service_scheduled_duration_min."""
    df = df.copy()
    svc = nodes_service.set_index("service_id")
    if "train_class_code" in svc.columns:
        df["train_class_code"] = (
            df["service_id"].map(svc["train_class_code"]).fillna(6).astype("int8")
        )
    else:
        df["train_class_code"] = np.int8(6)
    df["country"] = df["station_id"].astype(str).str[:2]

    df["service_n_stops"] = df["n_total_stops"].astype("float32")

    # Route scheduled duration = max cum_scheduled_minutes per service
    if "cum_scheduled_minutes" in df.columns:
        dur = df.groupby("service_id")["cum_scheduled_minutes"].transform("max")
        df["service_scheduled_duration_min"] = dur.fillna(0).astype("float32")
    else:
        df["service_scheduled_duration_min"] = np.float32(0.0)

    # Approximate route distance via cumulative adjacency km between consecutive
    # stops on this service. The lookup is from `edges_adjacent`.
    edges_adjacent = edges_adjacent.dropna(subset=["station_from", "station_to"]).copy()
    edges_key = (
        edges_adjacent
        .assign(_pair=edges_adjacent["station_from"] + "→" + edges_adjacent["station_to"])
        .set_index("_pair")["distance_km"]
    )
    df_sorted = df.sort_values(["service_id", "stop_order"], kind="mergesort").reset_index(drop=True)
    prev_st = df_sorted.groupby("service_id")["station_id"].shift(1)
    pair_key = (prev_st + "→" + df_sorted["station_id"]).fillna("")
    seg_km = pair_key.map(edges_key).fillna(0).astype("float32")
    df_sorted["cum_distance_km"] = (
        seg_km.groupby(df_sorted["service_id"]).cumsum().astype("float32")
    )
    route_total = df_sorted.groupby("service_id")["cum_distance_km"].transform("max")
    df_sorted["service_route_distance_km"] = route_total.astype("float32")

    # Re-align back to original ordering (preserve df row order from caller)
    df_sorted = df_sorted.set_index(["service_id", "stop_order"])
    df = df.set_index(["service_id", "stop_order"])
    df["cum_distance_km"]            = df_sorted["cum_distance_km"]
    df["service_route_distance_km"]  = df_sorted["service_route_distance_km"]
    return df.reset_index()


# ── Weather differential vs previous stop ──────────────────────────────────────
def add_weather_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Δ_severity_vs_prev_stop, Δ_wind_vs_prev_stop within each service."""
    df = df.sort_values(["service_id", "stop_order"], kind="mergesort").reset_index(drop=True)
    grp = df.groupby("service_id")
    if "weather_severity" in df.columns:
        df["delta_severity_vs_prev_stop"] = (
            (df["weather_severity"].astype("float32") - grp["weather_severity"].shift(1))
            .fillna(0).astype("float32")
        )
    else:
        df["delta_severity_vs_prev_stop"] = np.float32(0.0)
    if "wind_speed" in df.columns:
        df["delta_wind_vs_prev_stop"] = (
            (df["wind_speed"].astype("float32") - grp["wind_speed"].shift(1))
            .fillna(0).astype("float32")
        )
    else:
        df["delta_wind_vs_prev_stop"] = np.float32(0.0)
    return df


# ── Lagged station history ─────────────────────────────────────────────────────
def _daily_station_stats(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (station_id, date) with daily disruption rate and avg delay."""
    df = df.copy()
    df["_late"] = _stop_late(df)
    daily = (
        df.groupby(["station_id", pd.to_datetime(df["date"]).dt.normalize()])
          .agg(daily_rate=("_late", "mean"),
                daily_avg_delay=("delay_minutes", "mean"))
          .reset_index()
          .rename(columns={"date": "date"})
    )
    daily.columns = ["station_id", "date", "daily_rate", "daily_avg_delay"]
    return daily


def compute_station_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add station_lag1_rate, station_lag7_rate, station_lag1_avg_delay.

    Strict `<`: lag1 = daily_rate at date-1 day; lag7 = mean over [date-7, date-1].
    """
    daily = _daily_station_stats(df)
    cal = _calendar()

    # Forward-fill onto the contiguous calendar per station (NaN for missing days).
    grid = (
        pd.MultiIndex.from_product([daily["station_id"].unique(), cal],
                                    names=["station_id", "date"])
        .to_frame(index=False)
    )
    daily = grid.merge(daily, on=["station_id", "date"], how="left")
    daily = daily.sort_values(["station_id", "date"])

    grouped = daily.groupby("station_id", sort=False)
    daily["station_lag1_rate"]      = grouped["daily_rate"].shift(1)
    daily["station_lag1_avg_delay"] = grouped["daily_avg_delay"].shift(1)
    daily["station_lag7_rate"] = (
        grouped["daily_rate"]
        .rolling(window=7, min_periods=1).mean().shift(1)
        .reset_index(level=0, drop=True)
    )

    out_cols = ["station_id", "date",
                "station_lag1_rate", "station_lag7_rate", "station_lag1_avg_delay"]
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.normalize()
    df_out = df_out.merge(daily[out_cols], on=["station_id", "date"], how="left")
    for c in out_cols[2:]:
        df_out[c] = df_out[c].fillna(0).astype("float32")
    return df_out


# ── Lagged train history ───────────────────────────────────────────────────────
def compute_train_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    train_lag1_rate, train_lag7_rate keyed by (train_number, date).

    `train_number` is derived from `service_id` after stripping the country
    prefix and the trailing `_<date>` (FI/NL pattern). For Italy the
    service_id is `IT_<train_id>` directly.
    """
    df = df.copy()
    sid = df["service_id"].astype(str)

    # Heuristic: train_number = service_id without country prefix or trailing _date
    train_num = sid.str.replace(r"^[A-Z]{2}_", "", regex=True)
    train_num = train_num.str.replace(r"_\d{4}-\d{2}-\d{2}$", "", regex=True)
    df["_train_key"] = df["country"] + "_" + train_num if "country" in df.columns \
                      else sid

    df["_late"] = _stop_late(df)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    daily = (
        df.groupby(["_train_key", "date"])["_late"].mean()
          .rename("daily_rate").reset_index()
    )
    cal = _calendar()
    grid = (
        pd.MultiIndex.from_product([daily["_train_key"].unique(), cal],
                                    names=["_train_key", "date"])
        .to_frame(index=False)
    )
    daily = grid.merge(daily, on=["_train_key", "date"], how="left")
    daily = daily.sort_values(["_train_key", "date"])

    grouped = daily.groupby("_train_key", sort=False)
    daily["train_lag1_rate"] = grouped["daily_rate"].shift(1)
    daily["train_lag7_rate"] = (
        grouped["daily_rate"]
        .rolling(window=7, min_periods=1).mean().shift(1)
        .reset_index(level=0, drop=True)
    )

    df = df.merge(
        daily[["_train_key", "date", "train_lag1_rate", "train_lag7_rate"]],
        on=["_train_key", "date"], how="left",
    )
    for c in ["train_lag1_rate", "train_lag7_rate"]:
        df[c] = df[c].fillna(0).astype("float32")
    return df.drop(columns=["_late"])


# ── Lagged train × station history ─────────────────────────────────────────────
def compute_train_station_lag(df: pd.DataFrame) -> pd.DataFrame:
    """train_station_lag7_rate keyed by (train_key, station_id, date)."""
    df = df.copy()
    df["_late"] = _stop_late(df)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    if "_train_key" not in df.columns:
        sid = df["service_id"].astype(str)
        train_num = sid.str.replace(r"^[A-Z]{2}_", "", regex=True)
        train_num = train_num.str.replace(r"_\d{4}-\d{2}-\d{2}$", "", regex=True)
        df["_train_key"] = df.get("country", "") + "_" + train_num

    daily = (
        df.groupby(["_train_key", "station_id", "date"])["_late"].mean()
          .rename("daily_rate").reset_index()
    )
    daily = daily.sort_values(["_train_key", "station_id", "date"])

    grouped = daily.groupby(["_train_key", "station_id"], sort=False)
    daily["train_station_lag7_rate"] = (
        grouped["daily_rate"]
        .rolling(window=7, min_periods=1).mean().shift(1)
        .reset_index(level=[0, 1], drop=True)
    )
    df = df.merge(
        daily[["_train_key", "station_id", "date", "train_station_lag7_rate"]],
        on=["_train_key", "station_id", "date"], how="left",
    )
    df["train_station_lag7_rate"] = (
        df["train_station_lag7_rate"].fillna(0).astype("float32")
    )
    return df.drop(columns=["_late"], errors="ignore")


# ── Neighbour signal ───────────────────────────────────────────────────────────
def compute_neighbour_signal(
    df: pd.DataFrame,
    edges_adjacent: pd.DataFrame,
) -> pd.DataFrame:
    """
    nbr_lag1_rate = mean of station_lag1_rate over adjacent stations
    on the same date. Requires station_lag1_rate to have been added already.
    """
    if "station_lag1_rate" not in df.columns:
        df = df.assign(nbr_lag1_rate=np.float32(0.0))
        return df

    edges_adjacent = edges_adjacent.dropna(subset=["station_from", "station_to"])

    # daily station_lag1_rate per (station_id, date)
    daily = (
        df.groupby(["station_id", "date"])["station_lag1_rate"]
          .mean().reset_index()
    )

    # For each edge endpoint, the neighbour's lag1 rate on the same date
    edges = edges_adjacent[["station_from", "station_to"]].copy()
    nbr = edges.merge(
        daily.rename(columns={"station_id": "station_to",
                                "station_lag1_rate": "nbr_rate"}),
        on="station_to", how="left",
    )
    nbr_agg = (
        nbr.dropna(subset=["nbr_rate"])
           .groupby(["station_from", "date"])["nbr_rate"].mean()
           .rename("nbr_lag1_rate").reset_index()
           .rename(columns={"station_from": "station_id"})
    )
    df = df.merge(nbr_agg, on=["station_id", "date"], how="left")
    df["nbr_lag1_rate"] = df["nbr_lag1_rate"].fillna(0).astype("float32")
    return df


# ── Fault context ──────────────────────────────────────────────────────────────
def compute_fault_context(
    df: pd.DataFrame,
    nodes_fault: pd.DataFrame,
    window_days: int = 14,
) -> pd.DataFrame:
    """
    station_n_active_faults_14d = number of fault events at this station with
    `fault_date ∈ [current_date - window_days, current_date - 1]`.
    """
    df = df.copy()
    if nodes_fault is None or len(nodes_fault) == 0 or "station_id" not in nodes_fault.columns:
        df["station_n_active_faults_14d"] = np.float32(0.0)
        return df

    f = nodes_fault.dropna(subset=["station_id", "date"]).copy()
    f["date"] = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
    f = f.dropna(subset=["date"])
    if f.empty:
        df["station_n_active_faults_14d"] = np.float32(0.0)
        return df

    cal = _calendar()
    grid = (
        pd.MultiIndex.from_product([f["station_id"].unique(), cal],
                                    names=["station_id", "date"])
        .to_frame(index=False)
    )
    daily = (
        f.groupby(["station_id", "date"]).size()
         .rename("n_faults").reset_index()
    )
    daily = grid.merge(daily, on=["station_id", "date"], how="left").fillna({"n_faults": 0})
    daily = daily.sort_values(["station_id", "date"])
    daily["rolling"] = (
        daily.groupby("station_id", sort=False)["n_faults"]
             .rolling(window=window_days, min_periods=1).sum()
             .shift(1).reset_index(level=0, drop=True)
    )
    daily["rolling"] = daily["rolling"].fillna(0)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.merge(
        daily[["station_id", "date", "rolling"]]
        .rename(columns={"rolling": "station_n_active_faults_14d"}),
        on=["station_id", "date"], how="left",
    )
    df["station_n_active_faults_14d"] = (
        df["station_n_active_faults_14d"].fillna(0).astype("float32")
    )
    return df


# ── Inflight (scenario B only) ─────────────────────────────────────────────────
def add_inflight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    prev_stop_actual_delay, cum_actual_delay_so_far, max_actual_delay_so_far.

    These read prior stops on the same service. They are FORBIDDEN under
    scenario A. The leakage guard `LEAKY_COLS_A` lists them so they will be
    rejected if they accidentally end up in a scenario-A feature matrix.
    """
    df = df.sort_values(["service_id", "stop_order"], kind="mergesort").reset_index(drop=True)
    delay = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0).astype("float32")
    by_svc = delay.groupby(df["service_id"], sort=False)

    df["prev_stop_actual_delay"] = (
        by_svc.shift(1).fillna(0).astype("float32")
    )
    # cum / max of delays at stops 0..k-1 — shift first, then cum within group
    df["cum_actual_delay_so_far"] = (
        by_svc.apply(lambda s: s.shift(1).fillna(0).cumsum())
              .reset_index(level=0, drop=True)
              .astype("float32")
    )
    df["max_actual_delay_so_far"] = (
        by_svc.apply(lambda s: s.shift(1).fillna(0).cummax())
              .reset_index(level=0, drop=True)
              .astype("float32")
    )
    return df

"""
utils.py — Shared constants and helpers for the European Railway Pipeline
=============================================================================
Every preprocessing artefact imports from here to guarantee identical
class-code mappings, date boundaries, and schema column names across
the three countries. The unified model depends on this consistency.
"""
from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd

# ── Countries and their processed-data directories ────────────────────────────
COUNTRIES = ["Italy", "Finland", "Netherlands"]
PROC_DIRS = {
    "Italy":       Path("Data/Italy/processed"),
    "Finland":     Path("Data/Finland/processed"),
    "Netherlands": Path("Data/Netherlands/processed"),
}

# ── Date window ────────────────────────────────────────────────────────────────
DATE_START  = pd.Timestamp("2024-01-01")
DATE_END    = pd.Timestamp("2024-06-30")
MONTHS      = list(range(1, 7))                      # [1..6]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]

# ── Country prefixes ───────────────────────────────────────────────────────────
COUNTRY_PREFIX = {"italy": "IT", "finland": "FI", "netherlands": "NL"}

# ── Global train-class code map (0–6 ordinal) ─────────────────────────────────
# 0 = Local / Commuter,  1 = Express / Semi-fast,  2 = InterCity,
# 3 = EuroCity / International, 4 = High-Speed domestic,
# 5 = Ultra high-speed, 6 = Other / Unknown
GLOBAL_CLASS_MAP: dict[str, int] = {
    # Italy — Trenitalia
    "EC":  3, "FA":  4, "FB":  1, "FR":  5,
    "IC":  2, "ICN": 2, "REG": 0,
    # Netherlands — NS
    "INTERCITY":                2,
    "INTERCITY DIRECT":         4,
    "SPRINTER":                 0,
    "NIGHTJET":                 3,
    "INTERCITY DIRECT BRUSSEL": 4,
    # Finland — VR
    "S":   5,
    "P":   1,
    "R":   0,
    "H":   0,
    "T":   0,
    "MUS": 6,
    "PVV": 6,
}


def map_train_class(raw_class: pd.Series) -> pd.Series:
    """Normalise a raw train-class column to the global 0-6 integer code."""
    return (
        raw_class.astype(str)
                 .str.strip().str.upper()
                 .map(GLOBAL_CLASS_MAP)
                 .fillna(6).astype("int8")
    )


# ── Disruption label ───────────────────────────────────────────────────────────
DELAY_THRESHOLD_MIN = 5


def compute_is_disrupted(
    delay_series: pd.Series,
    cancelled_series: pd.Series,
) -> int:
    """Service-level disruption: 1 if any stop late > threshold or cancelled."""
    any_late      = (delay_series.dropna() > DELAY_THRESHOLD_MIN).any()
    any_cancelled = cancelled_series.any()
    return int(any_late or any_cancelled)


# ── Unified column schemas ─────────────────────────────────────────────────────
SCHEMA = {
    "nodes_station":  ["station_id", "lat", "lon", "avg_historical_delay", "degree"],
    "nodes_service":  ["service_id", "train_class_code", "date", "is_disrupted"],
    "edges_stops_at": ["service_id", "station_id", "delay_minutes",
                        "weather_severity", "temperature", "wind_speed",
                        "precipitation", "snow_depth"],
    "edges_adjacent": ["station_from", "station_to", "distance_km"],
    "nodes_fault":    ["fault_id", "date", "station_id", "description"],
}


# ── Logging helper ─────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def save_csv(df: pd.DataFrame, path, schema_key: str) -> None:
    """Save with schema validation: warn if expected columns are missing."""
    log = get_logger("utils")
    expected = SCHEMA.get(schema_key, [])
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        log.warning("Schema '%s' missing columns: %s", schema_key, missing)
    df.to_csv(path, index=False)
    log.info("Saved %s (%d rows) → %s", schema_key, len(df), path)

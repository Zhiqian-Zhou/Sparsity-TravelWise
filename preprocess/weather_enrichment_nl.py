"""
preprocess/weather_enrichment_nl.py
=============================================================================
Open-Meteo Historical Archive enrichment for the Netherlands pipeline.

The RDT services dump ships without weather. We fetch daily aggregates per
station (one HTTP request per station) and join them back to per-stop rows
on (station_code, date). Output is also persisted to
`Data/Netherlands/processed/weather_by_station.parquet` so subsequent runs
skip the network entirely.

API: https://archive-api.open-meteo.com/v1/archive  (free, no auth, rate
limited; cached locally via `requests_cache`).

Public surface:
    enrich_nl_with_weather(df, stations_master) -> df_enriched
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import DATE_START, DATE_END, get_logger  # noqa: E402

log = get_logger("nl_weather")

OUT_PARQUET = ROOT / "Data" / "Netherlands" / "processed" / "weather_by_station.parquet"
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

# Daily metrics requested from the Historical Archive endpoint.
DAILY_VARS = [
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "precipitation_sum",
    "snow_depth_max",
]


def _make_session(cache_dir: Path | None = None):
    """Return a cached requests session if requests-cache is installed,
    else fall back to a plain `requests.Session`. Cache makes re-runs free."""
    if cache_dir is None:
        cache_dir = ROOT / "Data" / "Netherlands" / "_om_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import requests_cache
        return requests_cache.CachedSession(
            str(cache_dir / "openmeteo"),
            expire_after=-1,
            allowable_methods=("GET",),
        )
    except Exception:
        import requests
        return requests.Session()


def _wmo_severity(precip: float, snow: float) -> int:
    """Map daily precipitation + snow depth to the same 0–4 ordinal used by
    the IT/FI lib so feature semantics stay consistent across countries."""
    sev = 0
    if precip is not None and not np.isnan(precip):
        if precip >= 0.5: sev = max(sev, 1)
        if precip >= 2.0: sev = max(sev, 2)
        if precip >= 5.0: sev = max(sev, 3)
    if snow is not None and not np.isnan(snow):
        if snow >= 5:    sev = max(sev, 2)
        if snow >= 15:   sev = max(sev, 3)
        if snow >= 40:   sev = max(sev, 4)
    return int(sev)


def _fetch_one_station(session, lat: float, lon: float,
                        start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   "Europe/Amsterdam",
    }
    r = session.get(ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    daily = payload.get("daily") or {}
    if not daily:
        return pd.DataFrame(columns=["date"] + DAILY_VARS)
    df = pd.DataFrame({
        "date":               pd.to_datetime(daily.get("time", []), errors="coerce"),
        "temperature":        daily.get("temperature_2m_mean", []),
        "wind_speed":         daily.get("wind_speed_10m_max", []),
        "precipitation":      daily.get("precipitation_sum", []),
        "snow_depth":         daily.get("snow_depth_max", []),
    })
    return df


def fetch_weather_by_station(stations_master: pd.DataFrame,
                              start: pd.Timestamp = DATE_START,
                              end: pd.Timestamp = DATE_END,
                              max_stations: int | None = None) -> pd.DataFrame:
    """Fetch daily weather for every NL station with coords; return a
    long-form `weather_by_station` DataFrame keyed on (station_code, date)."""
    coords = stations_master[["code", "geo_lat", "geo_lng"]].copy()
    coords.rename(columns={"code": "station_code",
                            "geo_lat": "lat", "geo_lng": "lon"}, inplace=True)
    coords["station_code"] = coords["station_code"].str.strip().str.upper()
    coords = coords.dropna(subset=["lat", "lon"]).drop_duplicates("station_code")
    if max_stations:
        coords = coords.head(max_stations)
    log.info("Fetching Open-Meteo daily weather for %d stations [%s..%s]",
             len(coords), start.date(), end.date())

    session = _make_session()
    rows: list[pd.DataFrame] = []
    failures = 0
    t0 = time.time()
    for i, r in enumerate(coords.itertuples(index=False), 1):
        try:
            sub = _fetch_one_station(session, r.lat, r.lon,
                                       start.strftime("%Y-%m-%d"),
                                       end.strftime("%Y-%m-%d"))
            sub.insert(0, "station_code", r.station_code)
            rows.append(sub)
        except Exception as e:
            failures += 1
            log.warning("station %s failed: %s", r.station_code, e)
            continue
        if i % 50 == 0:
            log.info("  %d/%d stations  (%.1fs elapsed, %d failures)",
                     i, len(coords), time.time() - t0, failures)

    if not rows:
        return pd.DataFrame(columns=["station_code", "date"] + DAILY_VARS +
                                     ["weather_severity"])
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for c in ["temperature", "wind_speed", "precipitation", "snow_depth"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    out["weather_severity"] = [
        _wmo_severity(p, s)
        for p, s in zip(out["precipitation"].fillna(0.0),
                        out["snow_depth"].fillna(0.0))
    ]
    out["weather_severity"] = out["weather_severity"].astype("int8")
    log.info("Built weather_by_station: %s rows, %d unique stations",
             f"{len(out):,}", out["station_code"].nunique())
    return out


def enrich_nl_with_weather(df: pd.DataFrame,
                             stations_master: pd.DataFrame,
                             cache_path: Path = OUT_PARQUET) -> pd.DataFrame:
    """Enrich NL stop-level df with daily weather columns. Caches the
    long-form per-station weather to `cache_path` so the next run skips
    network calls. Returns df + 5 new columns: temperature, wind_speed,
    precipitation, snow_depth, weather_severity."""
    if cache_path.exists():
        wbs = pd.read_parquet(cache_path)
        log.info("Loaded cached weather_by_station from %s (%d rows)",
                 cache_path, len(wbs))
    else:
        try:
            wbs = fetch_weather_by_station(stations_master)
        except Exception as e:
            log.error("Open-Meteo fetch failed (%s) — falling back to "
                      "zero-fill weather for NL.", e)
            wbs = pd.DataFrame(columns=["station_code", "date"] + DAILY_VARS +
                                       ["weather_severity"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        wbs.to_parquet(cache_path, index=False)

    if wbs.empty:
        for c in ["temperature", "wind_speed", "precipitation",
                  "snow_depth", "weather_severity"]:
            df[c] = 0.0 if c != "weather_severity" else np.int8(0)
        return df

    join_df = df.copy()
    join_df["date_norm"] = pd.to_datetime(join_df["date"]).dt.normalize()
    join_df["station_code"] = join_df["station_code"].astype(str).str.strip().str.upper()

    enriched = join_df.merge(
        wbs.rename(columns={"date": "date_norm"}),
        on=["station_code", "date_norm"], how="left",
    )

    for c, default in [("temperature", np.float32(10.0)),
                        ("wind_speed", np.float32(2.0)),
                        ("precipitation", np.float32(0.0)),
                        ("snow_depth", np.float32(0.0))]:
        if c in enriched.columns:
            enriched[c] = enriched[c].fillna(default).astype("float32")
        else:
            enriched[c] = default
    if "weather_severity" in enriched.columns:
        enriched["weather_severity"] = enriched["weather_severity"].fillna(0).astype("int8")
    else:
        enriched["weather_severity"] = np.int8(0)

    enriched.drop(columns=["date_norm"], inplace=True, errors="ignore")
    log.info("Enriched %d rows with %d non-null weather joins",
             len(enriched),
             int((enriched["temperature"] != 10.0).sum()))
    return enriched

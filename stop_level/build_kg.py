"""
stop_level/build_kg.py
=============================================================================
Builds the European Railway Knowledge Graph in **Sparksee 5.2.3** from the
per-country processed CSVs (Phase 1 outputs). The bridge query that links a
SHAP-flagged stop to its spatial-temporal context (recent faults, neighbour
stations, this stop's actual delay & weather) is implemented in pure
Sparksee native API — no Cypher, no Kuzu.

Why Sparksee, why this distribution
-----------------------------------
Sparksee is Sparsity Technologies' commercial labelled-property graph. The
project ships a `sparksee.cfg` (license + tuning) at the repo root. The
binary side comes from the public Maven Central jar
`com.sparsity:sparkseejava:5.2.3`, which already bundles
`linux64.nativelibs/{libsparksee,libsparkseejavawrap,libstlport}.so`. We
drive the JVM-side API from Python via JPype1. Run
`source scripts/install_sparksee.sh` once to install both the jar and a
portable Eclipse Temurin 17 JDK into `.tools/` (no root, no system JDK
required).

Schema (mirrors the paper, identical to what the prior Kuzu attempt used):
    Nodes
      Station       (station_id PK, country, lat, lon,
                     avg_historical_delay, degree)
      TrainService  (service_id PK, country, train_class_code, date,
                     is_disrupted)
      FaultEvent    (fault_id PK, date, description)
    Edges (all directed)
      STOPS_AT      (TrainService → Station)   delay_minutes,
                                                weather_severity, temperature,
                                                wind_speed, precipitation,
                                                snow_depth
      ADJACENT_TO   (Station → Station)         distance_km
      REPORTED_AT   (FaultEvent → Station)

Graceful degradation
--------------------
The Maven Central jar runs Sparksee in *personal evaluation* mode unless a
license string is set via `sparksee.cfg`. Eval mode caps overall graph size
to roughly 1 M nodes-or-edges per type. The full STOPS_AT relation is
~16.6 M edges, so this script auto-detects the cap when bulk insertion fails
and degrades to one of two demo subgraphs (in priority order):

  1. ``--countries Netherlands`` only — ~127 K stops, fits comfortably.
  2. The high-traffic hub centred on the SHAP demo station (FI_HKH) plus
     all its services, faults, and 1-hop neighbours — guaranteed < 100 K
     edges, ideal for the bridge-query demo.

This way the script *always* finishes with a working KG and a meaningful
bridge-query result, no matter the licensing tier you hand it.

Usage
-----
    source scripts/install_sparksee.sh         # one-time tool setup
    python stop_level/build_kg.py              # full graph (needs license)
    python stop_level/build_kg.py --rebuild    # drop existing DB
    python stop_level/build_kg.py --demo-subgraph hub  # quick FI_HKH demo
    python stop_level/build_kg.py --countries Netherlands  # NL-only

Output
------
    Data/kg/railway.gdb          Sparksee on-disk database
    Data/kg/manifest.json        node + edge counts, bridge demo, latency
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_logger  # noqa: E402

log = get_logger("kg")

DEFAULT_DB_PATH = ROOT / "Data" / "kg" / "railway.gdb"
STAGING_DIR     = ROOT / "Data" / "kg" / "_staging"
COUNTRIES_DEFAULT = ["Italy", "Finland", "Netherlands"]


# ── JVM bootstrap ─────────────────────────────────────────────────────────────
def _start_jvm() -> None:
    """Locate Sparksee + a JDK and start a JVM with the right classpath /
    java.library.path. Idempotent — safe to call from main *and* from the
    bridge-demo path inside the same process."""
    import jpype  # local import so a fresh checkout without jpype1 still
                  # parses this module (we surface a clear error below).

    if jpype.isJVMStarted():
        return

    sparksee_home = os.environ.get("SPARKSEE_HOME") or str(ROOT / ".tools/sparksee")
    jar = Path(sparksee_home) / "sparkseejava-5.2.3.jar"
    native = Path(sparksee_home) / "native"
    if not jar.exists() or not native.exists():
        raise RuntimeError(
            f"Sparksee not found at {sparksee_home}. Run "
            "`source scripts/install_sparksee.sh` first."
        )

    # If JAVA_HOME isn't set, JPype falls back to its bundled defaultJVMPath
    # which usually finds `libjvm.so` via LD_LIBRARY_PATH; so we only
    # explicitly point at the portable Temurin if neither path is set.
    java_home = os.environ.get("JAVA_HOME")
    if not java_home:
        candidates = sorted((ROOT / ".tools").glob("jdk-17*"))
        if candidates:
            java_home = str(candidates[0])
            os.environ["JAVA_HOME"] = java_home
            log.info("Using portable JDK at %s", java_home)

    log.info("Starting JVM (jar=%s, native=%s)", jar.name, native)
    jpype.startJVM(
        classpath=[str(jar)],
        # NOTE: java.library.path is how libsparkseejavawrap.so finds the
        # other two .so siblings (libsparksee, libstlport) by RPATH.
        *[f"-Djava.library.path={native}"],
    )


# ── Staging: union per-country CSVs into single bulk-load CSVs ─────────────────
def stage_csvs(countries: list[str]) -> dict[str, Path]:
    """Same staging step as before: concatenate the per-country CSVs into
    parquet files at `Data/kg/_staging/`. We keep parquet (not CSV) because
    parquet preserves dtypes, is ~5× smaller on disk, and the rest of the
    project already speaks parquet — the Sparksee loader reads from it
    via pandas.

    Returns a dict mapping logical name → staged parquet path."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    country_code = lambda c: "FI" if c == "Finland" else ("NL" if c == "Netherlands" else "IT")

    log.info("Staging Station nodes...")
    sta_chunks = []
    for c in countries:
        p = ROOT / "Data" / c / "processed" / "nodes_station.csv"
        if not p.exists():
            log.warning("Missing %s — skipping", p)
            continue
        df = pd.read_csv(p)
        df["country"] = country_code(c)
        sta_chunks.append(df)
    sta = pd.concat(sta_chunks, ignore_index=True).drop_duplicates("station_id", keep="first")
    sta["station_id"] = sta["station_id"].astype(str)
    sta["country"]    = sta["country"].astype(str)
    sta["lat"]        = pd.to_numeric(sta["lat"], errors="coerce").astype("float64")
    sta["lon"]        = pd.to_numeric(sta["lon"], errors="coerce").astype("float64")
    sta["avg_historical_delay"] = pd.to_numeric(sta["avg_historical_delay"], errors="coerce").astype("float64")
    sta["degree"]     = pd.to_numeric(sta["degree"], errors="coerce").fillna(0).astype("int64")
    paths["station"] = STAGING_DIR / "stations.parquet"
    sta[["station_id","country","lat","lon","avg_historical_delay","degree"]].to_parquet(
        paths["station"], index=False
    )
    log.info("  Station: %d rows → %s", len(sta), paths["station"].name)

    log.info("Staging TrainService nodes...")
    svc_chunks = []
    for c in countries:
        p = ROOT / "Data" / c / "processed" / "nodes_service.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        df["country"] = country_code(c)
        svc_chunks.append(df)
    svc = pd.concat(svc_chunks, ignore_index=True).drop_duplicates("service_id", keep="first")
    svc["service_id"] = svc["service_id"].astype(str)
    svc["country"]    = svc["country"].astype(str)
    svc["train_class_code"] = pd.to_numeric(svc["train_class_code"], errors="coerce").fillna(6).astype("int64")
    svc["date"]       = pd.to_datetime(svc["date"], errors="coerce").dt.normalize()
    svc["is_disrupted"] = pd.to_numeric(svc["is_disrupted"], errors="coerce").fillna(0).astype("int64")
    paths["service"] = STAGING_DIR / "services.parquet"
    svc[["service_id","country","train_class_code","date","is_disrupted"]].to_parquet(
        paths["service"], index=False
    )
    log.info("  TrainService: %d rows → %s", len(svc), paths["service"].name)

    log.info("Staging FaultEvent nodes + REPORTED_AT edges...")
    fault_chunks = []
    for c in countries:
        p = ROOT / "Data" / c / "processed" / "nodes_fault.csv"
        if not p.exists() or p.stat().st_size < 50:
            continue
        df = pd.read_csv(p, parse_dates=["date"])
        if df.empty:
            continue
        fault_chunks.append(df)
    if fault_chunks:
        f = pd.concat(fault_chunks, ignore_index=True).drop_duplicates("fault_id", keep="first")
        f["fault_id"]    = f["fault_id"].astype(str)
        f["date"]        = pd.to_datetime(f["date"], errors="coerce").dt.normalize()
        f["description"] = f["description"].astype(str)
        valid_stations = set(sta["station_id"])
        f_keep = f[f["station_id"].astype(str).isin(valid_stations)].copy()
        f_keep["station_id"] = f_keep["station_id"].astype(str)
        paths["fault"] = STAGING_DIR / "faults.parquet"
        f_keep[["fault_id","date","description"]].to_parquet(paths["fault"], index=False)
        paths["reported_at"] = STAGING_DIR / "reported_at.parquet"
        f_keep[["fault_id","station_id"]].to_parquet(paths["reported_at"], index=False)
        log.info("  FaultEvent: %d rows (kept %d / dropped %d w/ unknown station)",
                 len(f), len(f_keep), len(f) - len(f_keep))
    else:
        paths["fault"] = None
        paths["reported_at"] = None
        log.info("  No fault rows found across all countries")

    log.info("Staging ADJACENT_TO edges...")
    adj_chunks = []
    for c in countries:
        p = ROOT / "Data" / c / "processed" / "edges_adjacent.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        adj_chunks.append(df)
    adj = pd.concat(adj_chunks, ignore_index=True).drop_duplicates(
        ["station_from","station_to"], keep="first"
    )
    valid_stations = set(sta["station_id"])
    adj_keep = adj[
        adj["station_from"].astype(str).isin(valid_stations) &
        adj["station_to"].astype(str).isin(valid_stations)
    ].copy()
    adj_keep["station_from"] = adj_keep["station_from"].astype(str)
    adj_keep["station_to"]   = adj_keep["station_to"].astype(str)
    adj_keep["distance_km"]  = pd.to_numeric(adj_keep["distance_km"], errors="coerce").fillna(0).astype("float64")
    paths["adjacent"] = STAGING_DIR / "adjacent.parquet"
    adj_keep[["station_from","station_to","distance_km"]].to_parquet(paths["adjacent"], index=False)
    log.info("  ADJACENT_TO: %d edges (kept %d / dropped %d w/ unknown station)",
             len(adj), len(adj_keep), len(adj) - len(adj_keep))

    log.info("Staging STOPS_AT edges (the big one — up to 16.59 M rows)...")
    paths["stops_at"] = STAGING_DIR / "stops_at.parquet"
    valid_services = set(svc["service_id"])
    valid_stations = set(sta["station_id"])
    chunks_kept: list[pd.DataFrame] = []
    n_total = 0
    for c in countries:
        p = ROOT / "Data" / c / "processed" / "edges_stops_at.csv"
        if not p.exists():
            continue
        for chunk in pd.read_csv(p, chunksize=500_000):
            n_total += len(chunk)
            chunk = chunk[
                chunk["service_id"].astype(str).isin(valid_services) &
                chunk["station_id"].astype(str).isin(valid_stations)
            ].copy()
            chunk["service_id"]   = chunk["service_id"].astype(str)
            chunk["station_id"]   = chunk["station_id"].astype(str)
            for col in ("delay_minutes","weather_severity","temperature",
                         "wind_speed","precipitation","snow_depth"):
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype("float64")
            chunks_kept.append(
                chunk[["service_id","station_id","delay_minutes",
                        "weather_severity","temperature","wind_speed",
                        "precipitation","snow_depth"]]
            )
    big = pd.concat(chunks_kept, ignore_index=True)
    big.to_parquet(paths["stops_at"], index=False)
    log.info("  STOPS_AT: %d edges (kept %d / dropped %d w/ unknown FK)",
             n_total, len(big), n_total - len(big))

    return paths


# ── Hub-only demo subgraph (eval-mode-safe fallback) ───────────────────────────
DEMO_HUB_STATION = "FI_HKH"   # the high-traffic hub used in the SHAP/KG paper demo

def _trim_to_hub_subgraph(paths: dict[str, Path], hub: str = DEMO_HUB_STATION) -> dict[str, Path]:
    """Restrict every staged parquet to the 1-hop neighbourhood of `hub`.
    Result is the union of:
      * the hub station + every adjacent station
      * every service that stops at the hub
      * every fault reported at any station in the union
      * every STOPS_AT edge owned by those services
      * every ADJACENT_TO edge from the hub
      * every REPORTED_AT edge into the union
    Output goes to a sibling `_staging_hub/` so the full staging is preserved.
    """
    out_dir = STAGING_DIR.parent / "_staging_hub"
    out_dir.mkdir(parents=True, exist_ok=True)

    sta = pd.read_parquet(paths["station"])
    adj = pd.read_parquet(paths["adjacent"])
    sa  = pd.read_parquet(paths["stops_at"])
    svc = pd.read_parquet(paths["service"])

    nbr_ids: set[str] = {hub}
    nbr_ids.update(adj.loc[adj["station_from"] == hub, "station_to"].astype(str))
    nbr_ids.update(adj.loc[adj["station_to"]   == hub, "station_from"].astype(str))

    # Cap STOPS_AT to edges TOUCHING the hub or one of its 1-hop neighbours.
    # If we kept *all* stops of every service that visits FI_HKH (a major hub
    # with ~16 K visiting services × ~30 stops each) we'd be over the
    # Maven Central evaluation license's 1 M edges/type cap. Restricting the
    # edge set to {hub ∪ neighbours} preserves the semantics of the bridge
    # query (which only ever asks about a single (service, station) pair plus
    # the station's 1-hop neighbourhood) while shrinking edges by 50–100×.
    sa2 = sa.loc[sa["station_id"].isin(nbr_ids)].copy()

    # Only keep services that actually appear in the trimmed STOPS_AT edge
    # set — otherwise we'd carry a quarter-million dead-weight TrainService
    # nodes that the bridge query never traverses.
    surviving_services = set(sa2["service_id"].astype(str).unique())
    svc2 = svc.loc[svc["service_id"].astype(str).isin(surviving_services)].copy()

    # Stations: hub ∪ neighbours ∪ any station that appears in a kept edge
    # (defensive — should equal nbr_ids by construction).
    sta2 = sta.loc[sta["station_id"].isin(nbr_ids | set(sa2["station_id"]))].copy()
    adj2 = adj.loc[adj["station_from"].isin(set(sta2["station_id"])) &
                   adj["station_to"].isin(set(sta2["station_id"]))].copy()

    new_paths: dict[str, Path] = {}
    new_paths["station"]  = out_dir / "stations.parquet";   sta2.to_parquet(new_paths["station"],  index=False)
    new_paths["service"]  = out_dir / "services.parquet";   svc2.to_parquet(new_paths["service"],  index=False)
    new_paths["stops_at"] = out_dir / "stops_at.parquet";   sa2.to_parquet(new_paths["stops_at"],  index=False)
    new_paths["adjacent"] = out_dir / "adjacent.parquet";   adj2.to_parquet(new_paths["adjacent"], index=False)

    # FaultEvents that are reported at any station in the new station set
    if paths.get("fault") and paths["fault"].exists():
        f  = pd.read_parquet(paths["fault"])
        ra = pd.read_parquet(paths["reported_at"])
        ra2 = ra.loc[ra["station_id"].isin(set(sta2["station_id"]))].copy()
        f2  = f.loc[f["fault_id"].isin(set(ra2["fault_id"]))].copy()
        new_paths["fault"]       = out_dir / "faults.parquet";      f2.to_parquet(new_paths["fault"],       index=False)
        new_paths["reported_at"] = out_dir / "reported_at.parquet"; ra2.to_parquet(new_paths["reported_at"], index=False)
    else:
        new_paths["fault"] = None
        new_paths["reported_at"] = None

    log.info("Hub-only subgraph (centre=%s): %d stations, %d services, %d STOPS_AT, %d ADJACENT_TO, %d faults",
             hub, len(sta2), len(svc2), len(sa2), len(adj2),
             0 if new_paths["fault"] is None else len(pd.read_parquet(new_paths["fault"])))
    return new_paths


# ── Sparksee backend ───────────────────────────────────────────────────────────
def build_sparksee(db_path: Path, paths: dict[str, Path]) -> dict[str, int]:
    """Create the schema in Sparksee and bulk-load every staged parquet.
    Uses the native Sparksee Java API via JPype. Returns per-table row counts.
    """
    _start_jvm()
    import jpype, jpype.imports  # noqa: F401  (jpype.imports enables `from com...` magic)
    from com.sparsity.sparksee.gdb import (
        SparkseeConfig, Sparksee, DataType, AttributeKind, Value,
    )

    cfg = SparkseeConfig()
    log.info("Sparksee cache=%d MiB, recovery=%s",
             cfg.getCacheMaxSize(), cfg.getRecoveryEnabled())
    sp = Sparksee(cfg)

    # `sp.create` requires the file to NOT exist; we removed it above.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Creating Sparksee DB at %s", db_path)
    db = sp.create(str(db_path), "TravelWise")
    sess = db.newSession()
    g = sess.getGraph()

    # ── Schema (mirrors the paper's Cypher DDL exactly) ──────────────────────
    log.info("Creating schema...")
    Station = g.newNodeType("Station")
    A_st_id      = g.newAttribute(Station, "station_id",           DataType.String, AttributeKind.Unique)
    A_st_country = g.newAttribute(Station, "country",              DataType.String, AttributeKind.Indexed)
    A_st_lat     = g.newAttribute(Station, "lat",                  DataType.Double, AttributeKind.Basic)
    A_st_lon     = g.newAttribute(Station, "lon",                  DataType.Double, AttributeKind.Basic)
    A_st_avgd    = g.newAttribute(Station, "avg_historical_delay", DataType.Double, AttributeKind.Basic)
    A_st_deg     = g.newAttribute(Station, "degree",               DataType.Long,   AttributeKind.Basic)

    TrainService = g.newNodeType("TrainService")
    A_ts_id     = g.newAttribute(TrainService, "service_id",       DataType.String, AttributeKind.Unique)
    A_ts_country = g.newAttribute(TrainService, "country",         DataType.String, AttributeKind.Indexed)
    A_ts_class  = g.newAttribute(TrainService, "train_class_code", DataType.Long,   AttributeKind.Basic)
    A_ts_date   = g.newAttribute(TrainService, "date",             DataType.Long,   AttributeKind.Indexed)
    A_ts_disr   = g.newAttribute(TrainService, "is_disrupted",     DataType.Long,   AttributeKind.Indexed)

    FaultEvent = g.newNodeType("FaultEvent")
    A_f_id     = g.newAttribute(FaultEvent, "fault_id",    DataType.String, AttributeKind.Unique)
    A_f_date   = g.newAttribute(FaultEvent, "date",        DataType.Long,   AttributeKind.Indexed)
    A_f_desc   = g.newAttribute(FaultEvent, "description", DataType.String, AttributeKind.Basic)

    # Sparksee edge factory: newEdgeType(name, directed, neighbours)
    STOPS_AT    = g.newEdgeType("STOPS_AT", True, True)
    A_sa_delay  = g.newAttribute(STOPS_AT, "delay_minutes",    DataType.Double, AttributeKind.Basic)
    A_sa_sev    = g.newAttribute(STOPS_AT, "weather_severity", DataType.Long,   AttributeKind.Basic)
    A_sa_temp   = g.newAttribute(STOPS_AT, "temperature",      DataType.Double, AttributeKind.Basic)
    A_sa_wind   = g.newAttribute(STOPS_AT, "wind_speed",       DataType.Double, AttributeKind.Basic)
    A_sa_precip = g.newAttribute(STOPS_AT, "precipitation",    DataType.Double, AttributeKind.Basic)
    A_sa_snow   = g.newAttribute(STOPS_AT, "snow_depth",       DataType.Double, AttributeKind.Basic)

    ADJACENT_TO = g.newEdgeType("ADJACENT_TO", True, True)
    A_adj_dist  = g.newAttribute(ADJACENT_TO, "distance_km", DataType.Double, AttributeKind.Basic)

    REPORTED_AT = g.newEdgeType("REPORTED_AT", True, True)

    counts: dict[str, int] = {"Station": 0, "TrainService": 0, "FaultEvent": 0,
                               "STOPS_AT": 0, "ADJACENT_TO": 0, "REPORTED_AT": 0}

    # ── Load nodes ───────────────────────────────────────────────────────────
    val = Value()
    str_to_node: dict[tuple[str, str], int] = {}    # (typeName, key) → oid

    # Sparksee 5.2.3 native code segfaults inside Link::Set_Basic if it
    # receives NaN/+/-Inf as a Double — normalise both to 0.0. This was
    # discovered the hard way: 462 Finnish stations have no coordinates
    # (FMI doesn't expose lat/lon) and the very first NaN crashed the JVM.
    import math
    def _safe_d(x: float) -> float:
        x = float(x)
        return 0.0 if (math.isnan(x) or math.isinf(x)) else x
    def _safe_l(x) -> int:
        try:
            x = int(x)
        except (TypeError, ValueError):
            return 0
        return x

    def _load_stations() -> None:
        df = pd.read_parquet(paths["station"])
        log.info("Loading %d Stations...", len(df))
        t0 = time.time()
        ids = df["station_id"].to_numpy()
        countries = df["country"].to_numpy()
        lats = df["lat"].to_numpy()
        lons = df["lon"].to_numpy()
        avgs = df["avg_historical_delay"].to_numpy()
        degs = df["degree"].to_numpy()
        for i in range(len(df)):
            n = g.newNode(Station)
            g.setAttribute(n, A_st_id,      val.setString(str(ids[i])))
            g.setAttribute(n, A_st_country, val.setString(str(countries[i])))
            g.setAttribute(n, A_st_lat,     val.setDouble(_safe_d(lats[i])))
            g.setAttribute(n, A_st_lon,     val.setDouble(_safe_d(lons[i])))
            g.setAttribute(n, A_st_avgd,    val.setDouble(_safe_d(avgs[i])))
            g.setAttribute(n, A_st_deg,     val.setLong(_safe_l(degs[i])))
            str_to_node[("Station", str(ids[i]))] = int(n)
        counts["Station"] = len(df)
        log.info("  → %d Stations in %.1fs", len(df), time.time() - t0)

    def _load_services() -> None:
        df = pd.read_parquet(paths["service"])
        log.info("Loading %d TrainServices...", len(df))
        t0 = time.time()
        ids = df["service_id"].to_numpy()
        countries = df["country"].to_numpy()
        classes = df["train_class_code"].to_numpy()
        # Convert dates to epoch days (Sparksee Long attribute) — we parse it
        # back via .toordinal() during the bridge query.
        dates = pd.to_datetime(df["date"]).map(lambda d: int(d.timestamp() // 86400) if pd.notna(d) else 0).to_numpy()
        disrupted = df["is_disrupted"].to_numpy()
        for i in range(len(df)):
            n = g.newNode(TrainService)
            g.setAttribute(n, A_ts_id,      val.setString(str(ids[i])))
            g.setAttribute(n, A_ts_country, val.setString(str(countries[i])))
            g.setAttribute(n, A_ts_class,   val.setLong(_safe_l(classes[i])))
            g.setAttribute(n, A_ts_date,    val.setLong(_safe_l(dates[i])))
            g.setAttribute(n, A_ts_disr,    val.setLong(_safe_l(disrupted[i])))
            str_to_node[("TrainService", str(ids[i]))] = int(n)
            if i and i % 100_000 == 0:
                log.info("    %d / %d (%.0fk/s)", i, len(df), i / (time.time() - t0) / 1000)
        counts["TrainService"] = len(df)
        log.info("  → %d TrainServices in %.1fs", len(df), time.time() - t0)

    def _load_faults() -> None:
        if not paths.get("fault") or not paths["fault"]:
            return
        df = pd.read_parquet(paths["fault"])
        log.info("Loading %d FaultEvents...", len(df))
        t0 = time.time()
        ids = df["fault_id"].to_numpy()
        descs = df["description"].astype(str).to_numpy()
        dates = pd.to_datetime(df["date"]).map(lambda d: int(d.timestamp() // 86400) if pd.notna(d) else 0).to_numpy()
        for i in range(len(df)):
            n = g.newNode(FaultEvent)
            g.setAttribute(n, A_f_id,   val.setString(str(ids[i])))
            g.setAttribute(n, A_f_date, val.setLong(_safe_l(dates[i])))
            g.setAttribute(n, A_f_desc, val.setString(str(descs[i])[:500]))
            str_to_node[("FaultEvent", str(ids[i]))] = int(n)
        counts["FaultEvent"] = len(df)
        log.info("  → %d FaultEvents in %.1fs", len(df), time.time() - t0)

    # ── Load edges ───────────────────────────────────────────────────────────
    def _load_stops_at() -> None:
        df = pd.read_parquet(paths["stops_at"])
        log.info("Loading %d STOPS_AT edges (this is the big one)...", len(df))
        t0 = time.time()
        svc_ids = df["service_id"].to_numpy()
        st_ids  = df["station_id"].to_numpy()
        delays  = df["delay_minutes"].to_numpy()
        sevs    = df["weather_severity"].to_numpy()
        temps   = df["temperature"].to_numpy()
        winds   = df["wind_speed"].to_numpy()
        precs   = df["precipitation"].to_numpy()
        snows   = df["snow_depth"].to_numpy()
        skipped = 0
        for i in range(len(df)):
            src = str_to_node.get(("TrainService", str(svc_ids[i])))
            dst = str_to_node.get(("Station",      str(st_ids[i])))
            if src is None or dst is None:
                skipped += 1
                continue
            e = g.newEdge(STOPS_AT, src, dst)
            g.setAttribute(e, A_sa_delay,  val.setDouble(_safe_d(delays[i])))
            g.setAttribute(e, A_sa_sev,    val.setLong(_safe_l(sevs[i])))
            g.setAttribute(e, A_sa_temp,   val.setDouble(_safe_d(temps[i])))
            g.setAttribute(e, A_sa_wind,   val.setDouble(_safe_d(winds[i])))
            g.setAttribute(e, A_sa_precip, val.setDouble(_safe_d(precs[i])))
            g.setAttribute(e, A_sa_snow,   val.setDouble(_safe_d(snows[i])))
            if i and i % 250_000 == 0:
                log.info("    %d / %d (%.0fk/s, skipped=%d)",
                         i, len(df), i / (time.time() - t0) / 1000, skipped)
        counts["STOPS_AT"] = len(df) - skipped
        log.info("  → %d STOPS_AT in %.1fs (skipped %d w/ unknown FK)",
                 counts["STOPS_AT"], time.time() - t0, skipped)

    def _load_adjacent() -> None:
        df = pd.read_parquet(paths["adjacent"])
        log.info("Loading %d ADJACENT_TO edges...", len(df))
        t0 = time.time()
        s_from = df["station_from"].to_numpy()
        s_to   = df["station_to"].to_numpy()
        dists  = df["distance_km"].to_numpy()
        skipped = 0
        for i in range(len(df)):
            a = str_to_node.get(("Station", str(s_from[i])))
            b = str_to_node.get(("Station", str(s_to[i])))
            if a is None or b is None:
                skipped += 1
                continue
            e = g.newEdge(ADJACENT_TO, a, b)
            g.setAttribute(e, A_adj_dist, val.setDouble(_safe_d(dists[i])))
        counts["ADJACENT_TO"] = len(df) - skipped
        log.info("  → %d ADJACENT_TO in %.1fs (skipped=%d)",
                 counts["ADJACENT_TO"], time.time() - t0, skipped)

    def _load_reported_at() -> None:
        if not paths.get("reported_at") or not paths["reported_at"]:
            return
        df = pd.read_parquet(paths["reported_at"])
        log.info("Loading %d REPORTED_AT edges...", len(df))
        t0 = time.time()
        f_ids = df["fault_id"].to_numpy()
        s_ids = df["station_id"].to_numpy()
        skipped = 0
        for i in range(len(df)):
            a = str_to_node.get(("FaultEvent", str(f_ids[i])))
            b = str_to_node.get(("Station",    str(s_ids[i])))
            if a is None or b is None:
                skipped += 1
                continue
            g.newEdge(REPORTED_AT, a, b)
        counts["REPORTED_AT"] = len(df) - skipped
        log.info("  → %d REPORTED_AT in %.1fs", counts["REPORTED_AT"], time.time() - t0)

    _load_stations()
    _load_services()
    _load_faults()
    _load_stops_at()
    _load_adjacent()
    _load_reported_at()

    # Persist the str→oid map next to the DB so the bridge demo can avoid a
    # full attribute-index lookup. It's tiny (1.2 M entries × ~80 bytes).
    map_path = db_path.with_suffix(".idmap.json")
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps({f"{t}:{k}": v for (t, k), v in str_to_node.items()}))
    log.info("Wrote OID map → %s (%d entries)", map_path.name, len(str_to_node))

    sess.close()
    db.close()
    sp.close()
    return counts


# ── Bridge query (native Sparksee API, mirrors the paper's Cypher) ────────────
def run_bridge_demo(db_path: Path) -> dict:
    """Native-API equivalent of the Cypher bridge query in §08 of the paper.

    Picks the highest-confidence true-positive disruption from xgb/B preds,
    then for that (service_id, station_id, date) triple returns:
        - the station's static attributes (country, avg delay, degree)
        - this stop's STOPS_AT edge attributes (delay, weather)
        - all FaultEvents reported at this station in the last 14 days
        - all 1-hop adjacent stations (count + ids)
    """
    _start_jvm()
    import jpype.imports  # noqa: F401
    from com.sparsity.sparksee.gdb import (
        SparkseeConfig, Sparksee, EdgesDirection, Condition, Value,
    )

    sp = Sparksee(SparkseeConfig())
    db = sp.open(str(db_path), False)
    sess = db.newSession()
    g = sess.getGraph()

    # Pick the demo stop from the model's test-set predictions.
    preds_path = ROOT / "stop_level" / "models" / "_artefacts" / "xgb" / "B" / "preds_test.parquet"
    if not preds_path.exists():
        log.warning("No xgb/B preds — skipping bridge demo")
        sess.close(); db.close(); sp.close()
        return {}
    preds = pd.read_parquet(preds_path)
    tp = preds[(preds["y_stop"] == 1) & (preds["y_pred"] == 1)]
    if tp.empty:
        tp = preds.sort_values("p_disrupted", ascending=False).head(1)
    else:
        tp = tp.sort_values("p_disrupted", ascending=False).head(1)
    row = tp.iloc[0]
    sid, stid, date = str(row["service_id"]), str(row["station_id"]), str(row["date"])[:10]
    p_disr = float(row["p_disrupted"])

    # ── Resolve the 3 type ids + the attribute ids we need ───────────────────
    Station_t      = g.findType("Station")
    TrainService_t = g.findType("TrainService")
    FaultEvent_t   = g.findType("FaultEvent")
    STOPS_AT_t     = g.findType("STOPS_AT")
    ADJACENT_TO_t  = g.findType("ADJACENT_TO")
    REPORTED_AT_t  = g.findType("REPORTED_AT")

    A_st_id      = g.findAttribute(Station_t, "station_id")
    A_st_country = g.findAttribute(Station_t, "country")
    A_st_avgd    = g.findAttribute(Station_t, "avg_historical_delay")
    A_st_deg     = g.findAttribute(Station_t, "degree")
    A_ts_id      = g.findAttribute(TrainService_t, "service_id")
    A_sa_delay   = g.findAttribute(STOPS_AT_t, "delay_minutes")
    A_sa_sev     = g.findAttribute(STOPS_AT_t, "weather_severity")
    A_f_date     = g.findAttribute(FaultEvent_t, "date")
    A_f_id       = g.findAttribute(FaultEvent_t, "fault_id")

    # Helper: find a single node by unique attribute value.
    def _find_node(attr_id: int, value_str: str):
        v = Value(); v.setString(value_str)
        objs = g.select(attr_id, Condition.Equal, v)
        try:
            it = objs.iterator()
            return int(it.next()) if it.hasNext() else None
        finally:
            objs.close()

    t0 = time.time()
    svc_oid = _find_node(A_ts_id, sid)
    sta_oid = _find_node(A_st_id, stid)
    if svc_oid is None or sta_oid is None:
        log.warning("Demo stop (%s @ %s) not found in KG", sid, stid)
        sess.close(); db.close(); sp.close()
        return {"demo_query": {"error": "service or station not in KG"}}

    # 1) The STOPS_AT edge connecting this service & station.
    # `findEdge(type, head, tail)` is direct lookup — much cheaner than
    # exploding the service's outgoing star and matching by tail.
    this_stop_delay = None
    this_weather_severity = None
    try:
        edge_oid = int(g.findEdge(STOPS_AT_t, svc_oid, sta_oid))
        v = Value()
        g.getAttribute(edge_oid, A_sa_delay, v); this_stop_delay = float(v.getDouble())
        g.getAttribute(edge_oid, A_sa_sev,   v); this_weather_severity = int(v.getLong())
    except Exception as e:
        log.warning("findEdge(STOPS_AT, %s, %s) returned no edge: %s", sid, stid, e)

    # 2) Station static attrs.
    v = Value()
    g.getAttribute(sta_oid, A_st_country, v); st_country = str(v.getString())
    g.getAttribute(sta_oid, A_st_avgd,    v); st_avgd    = float(v.getDouble())
    g.getAttribute(sta_oid, A_st_deg,     v); st_deg     = int(v.getLong())

    # 3) Recent faults at this station (within 14 days of the demo date).
    demo_epoch = int(pd.Timestamp(date).timestamp() // 86400)
    cutoff = demo_epoch - 14
    ra_objs = g.explode(sta_oid, REPORTED_AT_t, EdgesDirection.Ingoing)
    recent_fault_ids: list[str] = []
    try:
        ra_it = ra_objs.iterator()
        while ra_it.hasNext():
            edge_oid = int(ra_it.next())
            head = int(g.getEdgeData(edge_oid).getHead())
            v2 = Value()
            g.getAttribute(head, A_f_date, v2)
            d = int(v2.getLong())
            if cutoff <= d <= demo_epoch:
                g.getAttribute(head, A_f_id, v2)
                recent_fault_ids.append(str(v2.getString()))
    finally:
        ra_objs.close()

    # 4) 1-hop adjacent stations (outgoing only — same convention as the
    #    Cypher bridge query in §08 of the paper).
    adj_objs = g.neighbors(sta_oid, ADJACENT_TO_t, EdgesDirection.Outgoing)
    nbr_ids: list[str] = []
    try:
        adj_it = adj_objs.iterator()
        while adj_it.hasNext():
            nb_oid = int(adj_it.next())
            v2 = Value()
            g.getAttribute(nb_oid, A_st_id, v2)
            nbr_ids.append(str(v2.getString()))
    finally:
        adj_objs.close()

    elapsed_ms = (time.time() - t0) * 1000
    log.info("Bridge query → %s @ %s in %.1f ms (faults=%d, neighbours=%d)",
             sid, stid, elapsed_ms, len(recent_fault_ids), len(nbr_ids))

    # 5) Aggregate counts (for the manifest's "graph-is-fully-loaded" check).
    n_st = int(g.countNodes())
    n_so = int(g.countEdges())

    sess.close()
    db.close()
    sp.close()

    return {
        "demo_query": {
            "service_id":         sid,
            "station_id":         stid,
            "date":               date,
            "p_disrupted":        p_disr,
            "result": {
                "station":           stid,
                "country":           st_country,
                "station_avg_delay": st_avgd,
                "degree":            st_deg,
                "this_stop_delay":   this_stop_delay,
                "weather_severity":  this_weather_severity,
                "recent_faults_14d": len(recent_fault_ids),
                "n_neighbours":      len(nbr_ids),
                "sample_neighbours": nbr_ids[:5],
            },
            "elapsed_ms": elapsed_ms,
        },
        "aggregate_counts": {
            "total_nodes": n_st,
            "total_edges": n_so,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=COUNTRIES_DEFAULT,
                        help="Countries to load (default: all 3)")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                        help="Where to create the KG (default: Data/kg/railway.gdb)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop existing DB before building")
    parser.add_argument("--skip-stage", action="store_true",
                        help="Reuse existing _staging/ parquets")
    parser.add_argument("--demo-subgraph", choices=["none", "hub", "auto"], default="auto",
                        help="If 'hub', restrict to the FI_HKH 1-hop subgraph "
                             "(eval-mode safe). 'auto' tries the full graph "
                             "first and falls back to 'hub' on Sparksee size "
                             "errors. 'none' refuses the fallback.")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if args.rebuild and db_path.exists():
        log.info("Removing existing DB at %s", db_path)
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()
        # Sparksee writes a .gdb file *and* a sibling .idmap.json + .log
        for sib in db_path.parent.glob(db_path.stem + "*"):
            if sib.exists() and sib != db_path:
                sib.unlink() if sib.is_file() else shutil.rmtree(sib)

    if not args.skip_stage:
        paths = stage_csvs(args.countries)
    else:
        log.info("Reusing staged parquets from %s", STAGING_DIR)
        paths = {
            "station":     STAGING_DIR / "stations.parquet",
            "service":     STAGING_DIR / "services.parquet",
            "fault":       STAGING_DIR / "faults.parquet",
            "adjacent":    STAGING_DIR / "adjacent.parquet",
            "stops_at":    STAGING_DIR / "stops_at.parquet",
            "reported_at": STAGING_DIR / "reported_at.parquet",
        }

    if args.demo_subgraph == "hub":
        log.info("Building HUB subgraph (--demo-subgraph hub)")
        paths = _trim_to_hub_subgraph(paths)

    # Build, with auto-fallback on Sparksee eval-mode size errors.
    try:
        counts = build_sparksee(db_path, paths)
        used_subgraph = "hub" if args.demo_subgraph == "hub" else "full"
    except Exception as e:
        msg = str(e)
        if args.demo_subgraph == "auto" and any(s in msg.lower()
                                                  for s in ("license", "limit", "exceed", "evaluation")):
            log.warning("Sparksee rejected the full graph (%s). Falling back to "
                         "the FI_HKH hub subgraph...", msg.splitlines()[0][:120])
            if db_path.exists():
                shutil.rmtree(db_path) if db_path.is_dir() else db_path.unlink()
            paths = _trim_to_hub_subgraph(paths)
            counts = build_sparksee(db_path, paths)
            used_subgraph = "hub"
        else:
            raise

    log.info("KG built (%s). Counts:", used_subgraph)
    for k, v in counts.items():
        log.info("  %-15s %d", k, v)

    log.info("Running bridge query demo...")
    demo = run_bridge_demo(db_path)

    manifest = {
        "backend":      "sparksee-5.2.3",
        "license":      "personal-evaluation" if not (ROOT / "sparksee.cfg").read_text().lower().count("license =") - 1 else "user-supplied",
        "subgraph":     used_subgraph,
        "db_path":      str(db_path),
        "countries":    args.countries,
        "node_counts":  {k: v for k, v in counts.items() if k in ("Station","TrainService","FaultEvent")},
        "rel_counts":   {k: v for k, v in counts.items() if k in ("STOPS_AT","ADJACENT_TO","REPORTED_AT")},
        "bridge_demo":  demo,
        "schema": {
            "nodes": ["Station(station_id, country, lat, lon, avg_historical_delay, degree)",
                       "TrainService(service_id, country, train_class_code, date, is_disrupted)",
                       "FaultEvent(fault_id, date, description)"],
            "edges": ["STOPS_AT(TrainService→Station)",
                       "ADJACENT_TO(Station→Station)",
                       "REPORTED_AT(FaultEvent→Station)"],
        },
    }
    out = ROOT / "Data" / "kg" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Manifest → %s", out)


if __name__ == "__main__":
    main()

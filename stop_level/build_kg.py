"""
stop_level/build_kg.py
=============================================================================
Builds the European Railway Knowledge Graph from the per-country processed
CSVs (Phase 1 outputs). Intended to be a drop-in target for the
`kg_bridge_query` Cypher template emitted by `xai_stops.py` — when the SHAP
report flags a high-risk stop, the analyst can run that query against this
KG to get the surrounding station / fault / neighbour context.

Backend selection:
  • Default: Kuzu (embedded, Python 3 native, Cypher). Pip-installable.
  • Optional: real Sparksee. Set `SPARKSEE_HOME=/path/to/sparksee` (where
    `_sparksee.so` and `libsparkseejavakernel.so` live) and pass
    `--backend sparksee`. The schema and Cypher queries are 100 %
    portable — only the driver call sites differ.

Schema:
  Nodes
    Station(station_id PK, country, lat, lon, avg_historical_delay, degree)
    TrainService(service_id PK, country, train_class_code, date, is_disrupted)
    FaultEvent(fault_id PK, date, description)
  Relationships
    STOPS_AT     (TrainService → Station)  delay_minutes, weather_severity,
                                            temperature, wind_speed,
                                            precipitation, snow_depth
    ADJACENT_TO  (Station → Station)        distance_km
    REPORTED_AT  (FaultEvent → Station)     -

Usage:
    python stop_level/build_kg.py
    python stop_level/build_kg.py --rebuild    # drop existing DB
    python stop_level/build_kg.py --countries Italy Finland   # subset

Output:
    Data/kg/railway.kuzu/    Kuzu DB directory (analogous to .dex)
    Data/kg/manifest.json    node + relationship counts, query latency
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import get_logger  # noqa: E402

log = get_logger("kg")

DEFAULT_DB_PATH = ROOT / "Data" / "kg" / "railway.kuzu"
STAGING_DIR     = ROOT / "Data" / "kg" / "_staging"
COUNTRIES_DEFAULT = ["Italy", "Finland", "Netherlands"]


# ── Staging: union per-country CSVs into single bulk-load CSVs ─────────────────
def stage_csvs(countries: list[str]) -> dict[str, Path]:
    """Concatenate the per-country CSVs into staged **parquet** files.
    Parquet is preferred over CSV here because:
      • Some IT station_ids contain commas (e.g. "IT_P.M._KM._5,420") that
        Kuzu's CSV parser doesn't quote-handle robustly.
      • Parquet preserves dtypes (float32, int64, date) without re-parsing.
      • COPY FROM parquet is ~3-5× faster than CSV in Kuzu.
    Output suffix is .parquet but we keep the dict-key names from the original
    CSV-flow so downstream code reads more naturally.
    """
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

    log.info("Staging STOPS_AT edges (the big one — 16.59 M rows)...")
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


# ── Kuzu backend ───────────────────────────────────────────────────────────────
def build_kuzu(db_path: Path, paths: dict[str, Path]) -> tuple[int, int]:
    """Create schema + bulk-COPY all staged CSVs. Returns (n_nodes, n_rels)."""
    import kuzu

    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    log.info("Creating schema...")
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Station(
            station_id            STRING,
            country               STRING,
            lat                   DOUBLE,
            lon                   DOUBLE,
            avg_historical_delay  DOUBLE,
            degree                INT64,
            PRIMARY KEY (station_id)
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS TrainService(
            service_id        STRING,
            country           STRING,
            train_class_code  INT64,
            date              DATE,
            is_disrupted      INT64,
            PRIMARY KEY (service_id)
        )
    """)
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS FaultEvent(
            fault_id     STRING,
            date         DATE,
            description  STRING,
            PRIMARY KEY (fault_id)
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS STOPS_AT(
            FROM TrainService TO Station,
            delay_minutes      DOUBLE,
            weather_severity   INT64,
            temperature        DOUBLE,
            wind_speed         DOUBLE,
            precipitation      DOUBLE,
            snow_depth         DOUBLE
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS ADJACENT_TO(
            FROM Station TO Station,
            distance_km  DOUBLE
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS REPORTED_AT(
            FROM FaultEvent TO Station
        )
    """)

    counts: dict[str, int] = {}
    for label, csv_key in [
        ("Station",     "station"),
        ("TrainService","service"),
        ("FaultEvent",  "fault"),
        ("STOPS_AT",    "stops_at"),
        ("ADJACENT_TO", "adjacent"),
        ("REPORTED_AT", "reported_at"),
    ]:
        p = paths.get(csv_key)
        if p is None or not Path(p).exists():
            log.warning("Skipping %s — no staged CSV", label)
            counts[label] = 0
            continue
        t0 = time.time()
        log.info("COPY %s FROM %s ...", label, p.name)
        # Parquet COPY in Kuzu doesn't accept HEADER=true (parquet has
        # its own schema); CSV COPY does. Detect by suffix.
        if str(p).endswith(".parquet"):
            conn.execute(f"COPY {label} FROM '{p}'")
        else:
            conn.execute(f"COPY {label} FROM '{p}' (HEADER=true)")
        # row count
        result = conn.execute(f"MATCH (n:{label}) RETURN count(n)" if label in
                              ("Station","TrainService","FaultEvent")
                              else f"MATCH ()-[r:{label}]->() RETURN count(r)")
        n = int(result.get_next()[0])
        counts[label] = n
        log.info("  → %d %s in %.1fs", n, label, time.time() - t0)

    return counts


# ── Sparksee backend (placeholder for when binary is available) ────────────────
def build_sparksee(db_path: Path, paths: dict[str, Path]) -> dict[str, int]:
    """Build the same KG using real Sparksee. Requires SPARKSEE_HOME pointing
    at a valid install. Schema and CSVs are byte-identical to the Kuzu path —
    only the driver call sites differ.

    Implementation deferred until Sparksee binary is available. The expected
    flow is documented inline.
    """
    try:
        import sparksee  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Sparksee Python module not found. Either:\n"
            "  • Set SPARKSEE_HOME=/path/to/sparksee with _sparksee.so + libsparkseejavakernel.so\n"
            "  • Or use --backend kuzu (default) for a Cypher-compatible substitute."
        ) from e

    # Documented flow (uncomment when sparksee binary is installed):
    #
    # cfg = sparksee.SparkseeConfig()
    # cfg.setLicense(open(ROOT / "sparksee.cfg").read())
    # sp = sparksee.Sparksee(cfg)
    # db = sp.create(str(db_path), "Railway")
    # session = db.newSession()
    # graph = session.getGraph()
    # # Define types via TypeLoader (mirrors Kuzu CREATE NODE/REL TABLE)
    # # Then bulk-load via graph.newTypeLoader(...) with the staged CSVs.
    # session.close(); db.close(); sp.close()
    raise NotImplementedError(
        "Sparksee backend skeleton present; binary not currently available "
        "in this environment. See inline docstring for the canonical flow."
    )


# ── Bridge query — verify the KG end-to-end ────────────────────────────────────
KG_BRIDGE_QUERY = """
MATCH (svc:TrainService {service_id: $sid})-[stop:STOPS_AT]->(st:Station {station_id: $stid})
OPTIONAL MATCH (f:FaultEvent)-[:REPORTED_AT]->(st)
  WHERE f.date >= date($d) - INTERVAL('14 days') AND f.date <= date($d)
OPTIONAL MATCH (st)-[adj:ADJACENT_TO]->(nbr:Station)
RETURN
  st.station_id          AS station,
  st.country             AS country,
  st.avg_historical_delay AS station_avg_delay,
  st.degree              AS degree,
  stop.delay_minutes     AS this_stop_delay,
  stop.weather_severity  AS weather_severity,
  count(DISTINCT f)      AS recent_faults_14d,
  count(DISTINCT nbr)    AS n_neighbours,
  collect(DISTINCT nbr.station_id) AS all_neighbours
"""


def run_bridge_demo(db_path: Path) -> dict:
    """Pick a high-risk stop from the test predictions and run the bridge
    query against the KG. Validates the schema is queryable."""
    import kuzu
    db = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    preds_path = ROOT / "stop_level" / "models" / "_artefacts" / "xgb" / "B" / "preds_test.parquet"
    if not preds_path.exists():
        log.warning("No xgb/B preds — skipping bridge demo")
        return {}

    preds = pd.read_parquet(preds_path)
    # Pick the highest-confidence true-positive disruption call
    tp = preds[(preds["y_stop"] == 1) & (preds["y_pred"] == 1)].copy()
    if tp.empty:
        log.warning("No true positives — picking highest-confidence prediction instead")
        tp = preds.sort_values("p_disrupted", ascending=False).head(1)
    else:
        tp = tp.sort_values("p_disrupted", ascending=False).head(1)
    row = tp.iloc[0]
    sid, stid, date = str(row["service_id"]), str(row["station_id"]), str(row["date"])[:10]
    log.info("Bridge demo: service=%s station=%s date=%s p=%.3f",
             sid, stid, date, row["p_disrupted"])

    t0 = time.time()
    res = conn.execute(KG_BRIDGE_QUERY, {"sid": sid, "stid": stid, "d": date})
    elapsed_ms = (time.time() - t0) * 1000
    cols = res.get_column_names()
    result_rows = []
    while res.has_next():
        rrow = dict(zip(cols, res.get_next()))
        # Cap the neighbours list at 5 — it can be large for hub stations
        if "all_neighbours" in rrow and isinstance(rrow["all_neighbours"], list):
            rrow["sample_neighbours"] = rrow["all_neighbours"][:5]
            rrow["n_neighbours"] = len(rrow["all_neighbours"])
            del rrow["all_neighbours"]
        result_rows.append(rrow)
    log.info("Bridge query → %d rows in %.1fms", len(result_rows), elapsed_ms)
    if result_rows:
        log.info("Result: %s", json.dumps(result_rows[0], default=str, indent=2))

    # Also run a global aggregate to prove the graph is fully loaded
    n_st = int(conn.execute("MATCH (n:Station) RETURN count(n)").get_next()[0])
    n_sv = int(conn.execute("MATCH (n:TrainService) RETURN count(n)").get_next()[0])
    n_so = int(conn.execute("MATCH ()-[r:STOPS_AT]->() RETURN count(r)").get_next()[0])

    return {
        "demo_query": {
            "service_id":      sid,
            "station_id":      stid,
            "date":            date,
            "p_disrupted":     float(row["p_disrupted"]),
            "result":          result_rows,
            "elapsed_ms":      elapsed_ms,
        },
        "aggregate_counts": {
            "Station": n_st, "TrainService": n_sv, "STOPS_AT": n_so,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="+", default=COUNTRIES_DEFAULT,
                        help="Countries to load (default: all 3)")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH),
                        help="Where to create the KG (default: Data/kg/railway.kuzu)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop existing DB before building")
    parser.add_argument("--backend", choices=["kuzu", "sparksee"], default="kuzu")
    parser.add_argument("--skip-stage", action="store_true",
                        help="Reuse existing _staging/ CSVs")
    args = parser.parse_args()

    db_path = Path(args.db_path)

    if args.rebuild and db_path.exists():
        log.info("Removing existing DB at %s", db_path)
        shutil.rmtree(db_path)

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

    log.info("Building KG with %s backend at %s", args.backend, db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "kuzu":
        counts = build_kuzu(db_path, paths)
    else:
        counts = build_sparksee(db_path, paths)

    log.info("KG built. Counts:")
    for k, v in counts.items():
        log.info("  %-15s %d", k, v)

    log.info("Running bridge query demo...")
    demo = run_bridge_demo(db_path)

    manifest = {
        "backend":      args.backend,
        "db_path":      str(db_path),
        "countries":    args.countries,
        "node_counts":  {k: v for k, v in counts.items() if k in ("Station","TrainService","FaultEvent")},
        "rel_counts":   {k: v for k, v in counts.items() if k in ("STOPS_AT","ADJACENT_TO","REPORTED_AT")},
        "bridge_demo":  demo,
        "schema": {
            "nodes": ["Station(station_id, country, lat, lon, avg_historical_delay, degree)",
                       "TrainService(service_id, country, train_class_code, date, is_disrupted)",
                       "FaultEvent(fault_id, date, description)"],
            "relationships": ["STOPS_AT(TrainService→Station)",
                               "ADJACENT_TO(Station→Station)",
                               "REPORTED_AT(FaultEvent→Station)"],
        },
    }
    out = ROOT / "Data" / "kg" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("Manifest → %s", out)


if __name__ == "__main__":
    main()

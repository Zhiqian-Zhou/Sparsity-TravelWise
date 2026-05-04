"""
Bake all dashboard JSON files from the existing pipeline outputs.
Run from repo root:  python docs/website/bake_data.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT  = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Headline metrics + per-country/per-position breakdown ──────────────────
print("[1/6] Baking benchmark.json ...")
b = json.load(open(ROOT / "stop_level" / "results" / "benchmark_stops.json"))
out = {
    "splits": {
        "train": 11_915_114,
        "val":   2_368_484,
        "test":  2_303_236,
        "total": 16_586_834,
    },
    "models": [],
    "by_country":     {},
    "by_position":    {},
    "by_train_class": {},
}
for r in b["runs"]:
    out["models"].append({
        "model":    r["model"],
        "scenario": r["scenario"],
        "test_pr_auc":  round(r["test"]["pr_auc"], 4),
        "test_f1":      round(r["test"]["f1"], 4),
        "test_precision": round(r["test"]["precision"], 4),
        "test_recall":  round(r["test"]["recall"], 4),
        "test_brier":   round(r["test"]["brier"], 4),
        "test_ece":     round(r["test"]["ece"], 4),
    })
    if r["scenario"] == "B" and r["model"] == "xgb":
        out["by_country"]     = r.get("by_country", {})
        out["by_position"]    = r.get("by_position", {})
        out["by_train_class"] = r.get("by_train_class", {})
(OUT / "benchmark.json").write_text(json.dumps(out, indent=2, default=str))


# ── 2. SHAP top-features (A and B) ────────────────────────────────────────────
print("[2/6] Baking xai.json ...")
x = json.load(open(ROOT / "stop_level" / "results" / "xai_report_stops.json"))
xai = {
    "scenarios": {},
    "lag_ablation": x.get("lag_ablation", {}),
    "scenario_uplift": x.get("scenario_uplift", {}),
    "kg_bridge_query": x.get("kg_bridge_query", ""),
}
for s, sc in x["scenarios"].items():
    feats = list(sc["global_importance"].items())
    feats.sort(key=lambda kv: -kv[1])
    xai["scenarios"][s] = {
        "model": sc.get("model"),
        "n_test_samples": sc.get("n_test_samples"),
        "top_features": [{"name": k, "value": round(v, 4)} for k, v in feats[:20]],
        "local_explanations": sc.get("local_explanations", {}),
        "base_value": sc.get("base_value"),
    }
(OUT / "xai.json").write_text(json.dumps(xai, indent=2, default=str))


# ── 3. Cause prediction v1 vs v2 ──────────────────────────────────────────────
print("[3/6] Baking causes.json ...")
v1_b = json.load(open(ROOT / "stop_level" / "results" / "cause_benchmark.json"))
v2_b = json.load(open(ROOT / "stop_level" / "results" / "cause_benchmark_v2.json"))
v1_t = json.load(open(ROOT / "stop_level" / "results" / "cause_transfer.json"))
v2_t = json.load(open(ROOT / "stop_level" / "results" / "cause_transfer_v2.json"))
causes = {
    "classes": v1_b["classes"],
    "v1": {
        "models": {k: {"val_macro_f1": v["val_macro_f1"],
                        "test_macro_f1": v["test_macro_f1"],
                        "val_acc": v["val_accuracy"],
                        "test_acc": v["test_accuracy"],
                        "test_per_class_f1": v["test_per_class_f1"]}
                  for k, v in v1_b["models"].items()},
        "transfer": v1_t["by_country"],
    },
    "v2": {
        "models": {k: {"val_macro_f1": v["val_macro_f1"],
                        "test_macro_f1": v["test_macro_f1"],
                        "val_acc": v["val_accuracy"],
                        "test_acc": v["test_accuracy"],
                        "test_per_class_f1": v["test_per_class_f1"]}
                  for k, v in v2_b["models"].items()},
        "transfer": v2_t["by_country"],
    },
}
(OUT / "causes.json").write_text(json.dumps(causes, indent=2, default=str))


# ── 4. Stations: lat/lon + degree + avg_delay (full set, lightly filtered) ────
print("[4/6] Baking stations.json ...")
parts = []
for c in ("Italy", "Finland", "Netherlands"):
    p = ROOT / "Data" / c / "processed" / "nodes_station.csv"
    df = pd.read_csv(p)
    df["country"] = "FI" if c == "Finland" else ("NL" if c == "Netherlands" else "IT")
    parts.append(df)
sta_all = pd.concat(parts, ignore_index=True).drop_duplicates("station_id", keep="first")
n_total = len(sta_all)
# Drop NaN coords; relax the bounding box to keep international cross-border
# stops (NL→Berlin/Brussels/Vienna; IT→Switzerland/France); only filter
# obvious outliers (lat 0/0 placeholders, etc).
sta = sta_all.dropna(subset=["lat", "lon"]).copy()
sta = sta[(sta["lat"].between(-90, 90)) & (sta["lon"].between(-180, 180))]
sta = sta[~((sta["lat"] == 0) & (sta["lon"] == 0))]                  # drop 0,0 placeholders
sta = sta[(sta["lat"].between(30, 75)) & (sta["lon"].between(-15, 40))]   # Europe envelope
n_kept = len(sta)
n_no_coords = n_total - len(sta_all.dropna(subset=["lat", "lon"]))
print(f"  total stations: {n_total}, with coords: {n_total - n_no_coords}, "
      f"after Europe filter: {n_kept}")
# Geographic bounding boxes for "domestic" classification (rough but useful).
COUNTRY_BBOX = {
    "IT": {"lat": (35.0, 47.5), "lon": (6.0, 19.0)},
    "FI": {"lat": (59.5, 70.5), "lon": (19.0, 32.0)},
    "NL": {"lat": (50.5, 53.7), "lon": (3.0, 7.5)},
}
def _is_domestic(country, lat, lon):
    bb = COUNTRY_BBOX.get(country)
    if not bb:
        return True
    return bb["lat"][0] <= lat <= bb["lat"][1] and bb["lon"][0] <= lon <= bb["lon"][1]

sta_out = []
n_dom = {"IT": 0, "FI": 0, "NL": 0}
n_intl = {"IT": 0, "FI": 0, "NL": 0}
for _, row in sta.iterrows():
    lat = round(float(row["lat"]), 4)
    lon = round(float(row["lon"]), 4)
    domestic = _is_domestic(row["country"], lat, lon)
    bucket = n_dom if domestic else n_intl
    bucket[row["country"]] += 1
    sta_out.append({
        "id":         row["station_id"],
        "country":    row["country"],
        "lat":        lat,
        "lon":        lon,
        "degree":     int(row["degree"]) if pd.notna(row["degree"]) else 0,
        "delay":      round(float(row["avg_historical_delay"]), 2)
                       if pd.notna(row["avg_historical_delay"]) else 0.0,
        "domestic":   domestic,
    })
print(f"  domestic per country: {n_dom}")
print(f"  international cross-border per country: {n_intl}")
(OUT / "stations.json").write_text(json.dumps(sta_out))

# Counts of stations missing coords entirely (so the UI can surface this honestly)
n_missing = {}
for c in ("IT", "FI", "NL"):
    cc_full = "Italy" if c == "IT" else ("Finland" if c == "FI" else "Netherlands")
    p = ROOT / "Data" / cc_full / "processed" / "nodes_station.csv"
    if p.exists():
        df_c = pd.read_csv(p)
        n_missing[c] = int(df_c[["lat", "lon"]].isna().any(axis=1).sum())
print(f"  stations missing lat/lon entirely: {n_missing}")
(OUT / "stations_meta.json").write_text(json.dumps({
    "total_with_coords": len(sta_out),
    "domestic_per_country": n_dom,
    "international_per_country": n_intl,
    "missing_coords_per_country": n_missing,
}, indent=2))


# ── 4b. KG sample: a representative subgraph for in-browser visualisation ────
print("[4b/6] Baking kg_sample.json ...")
# Pick ~12 hub stations per country (high degree) and seed the neighborhood
import collections
adj_parts = []
for c in ("Italy", "Finland", "Netherlands"):
    p = ROOT / "Data" / c / "processed" / "edges_adjacent.csv"
    if p.exists():
        adj_parts.append(pd.read_csv(p))
adj_all = pd.concat(adj_parts, ignore_index=True).drop_duplicates(["station_from", "station_to"])
# Build a station_id → neighbours dict
adj_dict = collections.defaultdict(set)
for _, r in adj_all.iterrows():
    adj_dict[r["station_from"]].add(r["station_to"])

valid_ids = set(s["id"] for s in sta_out)
# Top-k per country by degree (proxy for hub-ness)
hubs = []
for cc in ("IT", "FI", "NL"):
    pool = [s for s in sta_out if s["country"] == cc]
    pool.sort(key=lambda s: -s["degree"])
    hubs += pool[:12]                    # 12 per country = 36 hubs total
seed_ids = {h["id"] for h in hubs}

# Expand: add 1-hop neighbours of each hub (up to 4 each, to cap total nodes)
expanded = set(seed_ids)
for sid in seed_ids:
    for nbr in list(adj_dict.get(sid, set()))[:4]:
        if nbr in valid_ids:
            expanded.add(nbr)

# Build kg_sample with nodes + edges
node_lookup = {s["id"]: s for s in sta_out}
kg_nodes = [
    {**node_lookup[i], "kind": "Station", "is_hub": (i in seed_ids)}
    for i in expanded if i in node_lookup
]
kg_edges = []
for fr in expanded:
    for to in adj_dict.get(fr, set()):
        if to in expanded and fr != to:
            kg_edges.append({"source": fr, "target": to, "kind": "ADJACENT_TO"})

# Add a few sample services + fault events per hub for richness
print(f"  KG sample: {len(kg_nodes)} stations + {len(kg_edges)} ADJACENT_TO edges")
# Sample some services that visit hubs
svc_chunks = []
for c in ("Italy", "Finland", "Netherlands"):
    p = ROOT / "Data" / c / "processed" / "edges_stops_at.csv"
    if p.exists():
        for chunk in pd.read_csv(p, chunksize=200_000, usecols=["service_id", "station_id"]):
            chunk = chunk[chunk["station_id"].isin(seed_ids)]
            svc_chunks.append(chunk)
            if sum(len(c) for c in svc_chunks) > 500:
                break
        if sum(len(c) for c in svc_chunks) > 500:
            break
svc_df = pd.concat(svc_chunks, ignore_index=True) if svc_chunks else pd.DataFrame()
svc_per_hub = svc_df.groupby("station_id")["service_id"].apply(
    lambda s: list(s.unique())[:3]).to_dict() if len(svc_df) else {}
# Add up to 3 services per hub as nodes
for hub_id, svc_ids in svc_per_hub.items():
    for sid in svc_ids:
        kg_nodes.append({"id": sid, "kind": "TrainService", "country": node_lookup[hub_id]["country"]})
        kg_edges.append({"source": sid, "target": hub_id, "kind": "STOPS_AT"})

# Add NL faults (only NL has them)
faults_p = ROOT / "Data" / "Netherlands" / "processed" / "nodes_fault.csv"
if faults_p.exists():
    f = pd.read_csv(faults_p, parse_dates=["date"])
    # Pick 5 fault events that hit our hubs
    f_hub = f[f["station_id"].isin(seed_ids)].drop_duplicates("fault_id").head(5)
    for _, r in f_hub.iterrows():
        kg_nodes.append({
            "id":          r["fault_id"],
            "kind":        "FaultEvent",
            "date":        str(r["date"])[:10],
            "description": (str(r["description"])[:60] + "…")
                            if len(str(r["description"])) > 60 else str(r["description"]),
        })
        kg_edges.append({"source": r["fault_id"], "target": r["station_id"], "kind": "REPORTED_AT"})

kg_sample = {
    "schema": {
        "nodes": [
            {"label": "Station", "color": "#1E88E5", "shape": "ellipse",
             "fields": ["station_id (PK)", "country", "lat", "lon",
                        "avg_historical_delay", "degree"]},
            {"label": "TrainService", "color": "#43A047", "shape": "rectangle",
             "fields": ["service_id (PK)", "country", "train_class_code",
                        "date", "is_disrupted"]},
            {"label": "FaultEvent", "color": "#E53935", "shape": "diamond",
             "fields": ["fault_id (PK)", "date", "description"]},
        ],
        "edges": [
            {"label": "STOPS_AT",    "from": "TrainService", "to": "Station",
             "fields": ["delay_minutes", "weather_severity",
                        "temperature", "wind_speed",
                        "precipitation", "snow_depth"]},
            {"label": "ADJACENT_TO", "from": "Station", "to": "Station",
             "fields": ["distance_km"]},
            {"label": "REPORTED_AT", "from": "FaultEvent", "to": "Station",
             "fields": []},
        ],
        "totals": {
            "Station": 2397, "TrainService": 1217406, "FaultEvent": 2938,
            "STOPS_AT": 16586834, "ADJACENT_TO": 163902, "REPORTED_AT": 2938,
        },
    },
    "sample": {
        "nodes": kg_nodes,
        "edges": kg_edges,
        "hub_ids": list(seed_ids),
    },
}
print(f"  total sample: {len(kg_nodes)} nodes, {len(kg_edges)} edges")
(OUT / "kg_sample.json").write_text(json.dumps(kg_sample, default=str))


# ── 5. Sample predictions for the interactive panel (5K rows max) ─────────────
print("[5/6] Baking predictions sample ...")
preds_path = ROOT / "stop_level" / "models" / "_artefacts" / "xgb" / "B" / "preds_test.parquet"
df = pd.read_parquet(preds_path)
df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
# Stratified sample: keep all 4 prototype categories balanced
df["bucket"] = pd.cut(df["p_disrupted"], bins=[-0.01,0.25,0.5,0.75,1.01], labels=["low","mid_low","mid_high","high"])
samples = []
for b in df["bucket"].unique():
    sub = df[df["bucket"] == b]
    samples.append(sub.sample(n=min(1250, len(sub)), random_state=42))
sample = pd.concat(samples, ignore_index=True)
sample = sample[["service_id","station_id","country","date","train_class_code",
                  "stop_order","position_norm","y_stop","delay_min",
                  "p_disrupted","y_pred"]].copy()
for c in ("p_disrupted","position_norm","delay_min"):
    sample[c] = sample[c].round(3)
print(f"  → {len(sample)} sample predictions")
(OUT / "predictions_sample.json").write_text(sample.to_json(orient="records"))


# ── 6. Manifest of available figures ──────────────────────────────────────────
print("[6/6] Indexing figures ...")
figs = {
    "preprocessing": {},
    "evaluation": [],
    "xai": [],
}
for c in ("italy", "finland", "netherlands"):
    d = ROOT / "preprocess" / "figures" / c
    figs["preprocessing"][c] = sorted([p.name for p in d.glob("*.png")])
for p in sorted((ROOT / "stop_level" / "figures").glob("*.png")):
    figs["evaluation"].append(p.name)
for p in sorted((ROOT / "stop_level" / "figures" / "xai").glob("*.png")):
    figs["xai"].append(p.name)
(OUT / "figures.json").write_text(json.dumps(figs, indent=2))

print(f"\nDone. Wrote {len(list(OUT.glob('*.json')))} files to {OUT}")
for f in sorted(OUT.glob("*.json")):
    sz = f.stat().st_size
    print(f"  {sz:>10,} B  {f.name}")

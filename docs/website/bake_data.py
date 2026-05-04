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


# ── 4. Stations: lat/lon + degree + avg_delay (downsampled if needed) ─────────
print("[4/6] Baking stations.json ...")
parts = []
for c in ("Italy", "Finland", "Netherlands"):
    p = ROOT / "Data" / c / "processed" / "nodes_station.csv"
    df = pd.read_csv(p)
    df["country"] = "FI" if c == "Finland" else ("NL" if c == "Netherlands" else "IT")
    parts.append(df)
sta = pd.concat(parts, ignore_index=True).drop_duplicates("station_id", keep="first")
# Drop NaN coords + clip outliers (some IT international stops have weird coords)
sta = sta.dropna(subset=["lat", "lon"])
sta = sta[(sta["lat"].between(35, 72)) & (sta["lon"].between(-10, 35))]
sta_out = []
for _, row in sta.iterrows():
    sta_out.append({
        "id":      row["station_id"],
        "country": row["country"],
        "lat":     round(float(row["lat"]), 4),
        "lon":     round(float(row["lon"]), 4),
        "degree":  int(row["degree"]) if pd.notna(row["degree"]) else 0,
        "delay":   round(float(row["avg_historical_delay"]), 2)
                    if pd.notna(row["avg_historical_delay"]) else 0.0,
    })
print(f"  → {len(sta_out)} stations after coordinate filter")
(OUT / "stations.json").write_text(json.dumps(sta_out))


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

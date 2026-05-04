# TravelWise Dashboard

Interactive single-page dashboard for the Sparsity-TravelWise pipeline. **No backend, no build step** — vanilla HTML/CSS/JS with three CDN-loaded libraries.

## Run locally

```bash
cd docs/website
python -m http.server 8000
# open http://localhost:8000
```

Browsers block `fetch()` over `file://`, so a tiny static server is required.

## Deploy

Any static host:

| Host | Steps |
|---|---|
| **GitHub Pages** | Push to `main`, enable Pages on `/docs` folder, **set source to `/docs`**. |
| **Netlify / Vercel / Cloudflare Pages** | Point at `docs/website/`. No build command needed. |
| **S3 / nginx** | `aws s3 sync docs/website/ s3://your-bucket/` then enable static hosting. |

## Data

All dashboard JSONs are pre-baked from the existing pipeline outputs by `bake_data.py`:

```bash
python docs/website/bake_data.py
```

This produces 6 files in `docs/website/data/` (~1.3 MB total):

| File | Size | Source |
|---|---|---|
| `benchmark.json` | 3 KB | `stop_level/results/benchmark_stops.json` |
| `xai.json` | 10 KB | `stop_level/results/xai_report_stops.json` |
| `causes.json` | 9 KB | `cause_benchmark.json` + `cause_benchmark_v2.json` + transfers |
| `stations.json` | 200 KB | per-country `nodes_station.csv` (1,935 stations after coord filter) |
| `predictions_sample.json` | 1.1 MB | stratified 5K-row sample of `xgb/B/preds_test.parquet` |
| `figures.json` | 2 KB | inventory of 51 PNG figures |

## What's on the dashboard

| Section | Content |
|---|---|
| Hero | 5 headline KPIs (total stops, test set, best PR-AUC, cause v2 lift, KG relationships) |
| ① Model benchmark | 5-tab metric switcher (PR-AUC / F1 / Precision / Recall / ECE) → 4-model bar charts for scenario A and B side-by-side, plus per-country / per-position / per-train-class breakdowns |
| ② SHAP | 2-tab scenario switcher (A / B) → top-18 features bar chart; lag-ablation card with both scenario deltas; scenario-uplift card |
| ③ Cause prediction | KPI cards comparing v1 vs v2 + 9-class per-class F1 chart + IT/FI transfer distribution chart |
| ④ Map | All 1,935 valid-coordinate stations, marker size = degree, colour = country (border) and avg delay (fill), down-sampled to ≤3 markers per 0.5° grid cell to reduce occlusion |
| ⑤ Prediction panel | Filter (country, probability bucket, service ID, date) → "Random sample" / "Highest-conf TP" / "Confident FP" buttons → result card with probability bar, all features, and a synthesised KG-bridge query result |
| ⑥ KG | 6 KPI cards (node + relationship counts) + the Cypher bridge-query template |

## Tech stack

| Concern | Choice | Bundle size |
|---|---|---|
| Framework | None — vanilla ES module | 0 |
| Charts | [ECharts 5.5](https://echarts.apache.org) | ~120 KB gz from CDN |
| Map | [Leaflet 1.9](https://leafletjs.com) | ~40 KB gz from CDN |
| Fonts | Inter + JetBrains Mono via Google Fonts | ~30 KB gz |
| **Total first paint JS** | | **< 200 KB gz** |

The choice was made because:
1. The dashboard is a one-off project artifact, not a long-lived SaaS — a build pipeline (Vite, React, etc.) would just add maintenance overhead.
2. ECharts handles bars / lines / scatter / heatmap natively; Leaflet handles maps natively. No custom D3 needed.
3. Vanilla ES modules + CDN means a single `git push` to GitHub Pages publishes everything.

## Theme support

The dashboard auto-detects `prefers-color-scheme` and provides a manual toggle button (top-right). The selected theme persists in `localStorage` as `travelwise-theme`. ECharts and Leaflet basemaps both adapt.

## Accessibility

- WCAG AA contrast palette (4.5:1 text / 3:1 graphics, both light and dark)
- Keyboard navigation on tabs, form inputs, and theme toggle (`:focus-visible` rings)
- ARIA landmarks (`<header>`, `<main>`, `<nav>`, `aria-label` on map)
- All charts have `aria-label`-equivalent tooltip text via ECharts hover content
- No animations critical to comprehension; respects `prefers-reduced-motion` implicitly (CSS transitions are short)

## Caveats

- The "Prediction panel" looks up rows from a 5,000-row stratified sample of the test set, not a live model. To run the actual XGB model client-side we'd need to convert it to ONNX and load via `onnxruntime-web` (~2 MB extra) — left as future work.
- The "Knowledge-Graph bridge" result in the prediction panel is **synthesised from the available station + neighbour data**, not a live query against Sparksee. The same query runs server-side in 7.7 ms — see `stop_level/build_kg.py::run_bridge_demo`. Putting it on the static site would need a small server proxy (Sparksee is JVM-only); future work.
- The map down-samples to ≤3 stations per 0.5° grid cell to keep first-paint snappy on mobile. The full 2,397-station list is in `Data/stops/station_id_to_idx.json` if you need it.

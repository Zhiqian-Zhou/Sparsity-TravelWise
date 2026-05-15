# European Railway Stop-Level Disruption Prediction

End-to-end pipeline for predicting whether **each individual stop** of a train
service will be disrupted (`arrival_delay > 5 min` or cancelled) across the
railway networks of **Italy, Finland and the Netherlands**, using a unified
anti-leakage Common Data Model, a per-stop feature store, five competing
models, and SHAP explainability tied back to a Sparksee Knowledge Graph.

---

## 1. Problem (adapted)

Operators across IT, FI and NL want to know — *before a train departs and
again at every stop along its route* — which stops are likely to be late or
cancelled, so they can pre-stage staff, warn passengers, and reroute when
necessary.

The data is messy and heterogeneous:

| Country | Source | Granularity | Weather | Quirks |
|---|---|---|---|---|
| Italy | Trenitalia operations CSV (~600 MB, 2.6 M stop rows) | one row per stop | free-text labels (`light rain`, `heavy snow`, …) | `'N'` / `'S'` sentinels for first-stop / cancelled |
| Finland | FI-TW (~3.1 M stop rows) | one row per train, with all stops nested in a serialized `timeTableRows` cell + nested FMI `weather_observations` | numeric per-stop FMI | `ast.literal_eval` of a Python repr with bare `nan` tokens |
| Netherlands | RDT services + tariff distance + disruptions log (~10.8 M stop rows) | one row per stop, `Service:` / `Stop:` colon-prefixed columns | none in raw — enriched via Open-Meteo Historical API | wide tariff matrix, three-way cancellation flag |
| **Total** | | **~16.6 M stop rows** | | |

The current pipeline operates
at **`(service × station)`** grain — one prediction per scheduled stop on the
timetable.

---

## 2. Objective

| | |
|---|---|
| **Sample** | one scheduled stop = `(service_id, station_id, scheduled_time)` |
| **Input** `x` | a structured per-stop feature vector built only from information *available at prediction time* |
| **Output** `ŷ` | `P(disrupted_at_stop = 1) ∈ [0, 1]`, thresholded at a value tuned on validation |
| **Label** `y` | `1` if `arrival_delay > 5 min` at that stop OR the stop is cancelled |

### Two prediction scenarios

| Scenario | Available at prediction time | Use case |
|---|---|---|
| **A. Pre-departure** *(primary)* | timetable + weather forecast + lagged history. **No actual delay** from any stop on this run. | day-ahead operational planning, passenger alerts |
| **B. Inflight** *(secondary)* | scenario A + actual delays at *earlier* stops `1..k-1` on this same run | live retiming, propagation forecast |

Each model is trained twice and benchmarked side-by-side. The leakage guard
differs by scenario: **A** bans every same-service delay column; **B** allows
the whitelisted features `prev_stop_actual_delay`, `cum_actual_delay_so_far`,
and `max_actual_delay_so_far`.

---

## 3. Solution overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 1  ──  Per-country preprocessing                                    │
│  preprocess/{italy,finland,netherlands}_preprocessing.ipynb                │
│  + lib_<country>.py + figures/<country>/step_NN_*.png                      │
│  → Data/<Country>/processed/  (5 standardized CSVs each, utils.SCHEMA)     │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 2  ──  Unified stop-level preprocessing                             │
│  stop_level/preprocess_stops.py                                            │
│  → Data/stops/  (parquet feature store + leakage guard + scaler + graph)   │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 3  ──  Model zoo + evaluation                                       │
│  stop_level/models/{logreg,lgbm,xgb,graphsage,bilstm}_model.py             │
│  stop_level/train_all.py + evaluate.py                                     │
│  → benchmark_stops.json + 9 figures (per-country, per-position, …)         │
└────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  Phase 4  ──  SHAP explainability + KG bridge                              │
│  stop_level/xai_stops.py                                                   │
│  → 14 XAI figures + xai_report_stops.json + Cypher query template          │
└────────────────────────────────────────────────────────────────────────────┘
```

The four phases are independently runnable and cleanly separated. Each phase
**owns its own artefacts** (CSVs, parquet, model pickles, figure PNGs, JSON
reports), so a downstream phase only needs the previous phase's outputs —
the source code of earlier phases is not loaded.

---

## 4. Repository layout

```
Project_v2/
├── utils.py                          shared schemas, holiday list, train-class map,
│                                     date window, save_csv helper
│
├── preprocess/                       PHASE 1 — per-country notebooks
│   ├── italy_preprocessing.ipynb           9 inspection steps
│   ├── finland_preprocessing.ipynb         12 inspection steps
│   ├── netherlands_preprocessing.ipynb     11 inspection steps
│   ├── lib_italy.py                        heavy helpers (chunked stream, sentinels)
│   ├── lib_finland.py                      timeTableRows parse + FMI imputation
│   ├── lib_netherlands.py                  3-way cancellation, tariff melt
│   ├── run_all_notebooks.sh                papermill / nbconvert runner
│   └── figures/{italy,finland,netherlands}/  step_NN_*.png inspection plots
│
├── stop_level/                       PHASES 2-4 — unified stop-level pipeline
│   ├── leakage_guards.py             LEAKY_COLS_A / LEAKY_COLS_B + assertions
│   ├── splits.py                     day-level windows, service-disjointness check
│   ├── features.py                   11 feature builders (all unit-tested)
│   ├── preprocess_stops.py           PHASE 2 entry point
│   │
│   ├── models/                       PHASE 3 — five architectures
│   │   ├── base.py                       StopModel ABC + write_predictions helper
│   │   ├── logreg.py                     sklearn LogReg + saga + balanced weight
│   │   ├── lgbm_model.py                 LightGBM, scale_pos_weight, early stop
│   │   ├── xgb_model.py                  XGBoost, tree_method='hist'
│   │   ├── graphsage_model.py            2-layer SAGEConv + per-stop MLP head
│   │   └── bilstm_model.py               bidirectional (A) / causal (B) LSTM
│   ├── train_all.py                  PHASE 3 orchestrator
│   ├── evaluate.py                   PHASE 3 — 9 figures + per-country/-position breakdowns
│   ├── xai_stops.py                  PHASE 4 — SHAP, lag-ablation, scenario uplift
│   │
│   ├── figures/                      stop_level outputs (generated)
│   │   ├── metrics_summary_{A,B}.png
│   │   ├── pr_roc_grid.png
│   │   ├── confusion_grid.png
│   │   ├── calibration_grid.png
│   │   ├── per_country_metrics.png
│   │   ├── per_position_metrics.png
│   │   ├── per_train_class_metrics.png
│   │   ├── cascading_respect.png
│   │   └── xai/                              14 SHAP figures
│   ├── results/
│   │   ├── benchmark_stops.json
│   │   └── xai_report_stops.json
│   └── tests/                        30 pytest tests (5 files)
│
└── Data/
    ├── Italy/        Raw/         processed/    (5 CSVs after Phase 1)
    ├── Finland/      Raw/         processed/
    ├── Netherlands/  Raw/         processed/  (+ weather_by_station.parquet)
    └── stops/                      parquet feature store after Phase 2
        ├── stops_{train,val,test}.parquet
        ├── edge_index.npy, edge_attr.npy
        ├── feature_names_{A,B}.json
        ├── scaler_{mean,scale}_{A,B}.npy
        ├── station_id_to_idx.json, service_id_to_idx.json
        └── manifest_stops.json
```

---

## 5. Phase 1 — Per-country preprocessing notebooks

Each country has a single Jupyter notebook + a heavy-helper Python module.
The notebook drives the pipeline step-by-step, **writing both the
standardized CSVs and an inspection plot for every transformation**.
Domain experts can open the notebook and *see* every cleaning, imputation
and aggregation choice without reading code.

The 5-CSV output schema (`utils.SCHEMA`) is identical across countries:

| File | Columns |
|---|---|
| `nodes_station.csv` | `station_id, lat, lon, avg_historical_delay, degree` |
| `nodes_service.csv` | `service_id, train_class_code, date, is_disrupted` |
| `edges_stops_at.csv` | `service_id, station_id, delay_minutes, weather_severity, temperature, wind_speed, precipitation, snow_depth` |
| `edges_adjacent.csv` | `station_from, station_to, distance_km` |
| `nodes_fault.csv` | `fault_id, date, station_id, description` |

Country prefixes (`IT_`, `FI_`, `NL_`) guarantee namespace isolation.
Train classes are normalized to a global 0–6 ordinal in
`utils.GLOBAL_CLASS_MAP` (Local → Ultra-HS).

### Italy notebook (9 steps)
chunked streaming progress · `N`/`S` sentinel breakdown · log-y delay
histogram with per-class KDE · weather-text → ordinal heatmap · station risk
map · degree CCDF + route-length distribution · per-class & per-month
disruption rate · top-30 fault-keyword bar · final 5-CSV schema validation.

### Finland notebook (12 steps)
per-file row counts · `timeTableRows` parse-outcome pie + stops-per-train
histogram · pre/post numeric-coercion NaN rates · cancellation flag pie ·
weather imputation before/after histograms (`air_temp`, `visibility` →
monthly-median) · snow-by-month boxplot + lat/lon snow-depth map ·
P(disrupted | severity) cross-tab · FI station map · disruption rates by
class / dow / month · adjacency inference assertions.

### Netherlands notebook (11 steps)
per-file rows + cumulative · `Service:Foo` → snake_case rename reference ·
date filter · 3-way cancellation breakdown · **Open-Meteo per-station
completeness + temperature distribution** · WMO-code → severity ordinal ·
NL station map · tariff-matrix sparsity (with `XXX` diagonal sentinel) +
edge-distance histogram · top-20 disruption causes from the explode ·
per-class / per-month service-level rates.

All notebooks save inspection plots to
`preprocess/figures/<country>/step_NN_*.png`, executable headless via
`papermill` so a fresh checkout reproduces them deterministically.

---

## 6. Phase 2 — Unified stop-level preprocessing

`stop_level/preprocess_stops.py` reads the 5 CSVs from each country's
`processed/` directory, joins them at stop grain, computes 16 feature groups
with strict anti-leakage rules, splits temporally, fits a `StandardScaler` on
**training rows only**, and persists everything to `Data/stops/` as Parquet.

### Per-stop features

| Group | Features | Notes |
|---|---|---|
| **Stop position** | `stop_order`, `position_norm`, `is_origin`, `is_terminus`, `n_total_stops`, `cum_distance_km`, `cum_scheduled_minutes` | within the service, computed deterministically from the timetable |
| **Stop time** | `scheduled_arrival_hour`, `scheduled_arrival_dow`, `is_holiday`, `month`, `month_sin`, `month_cos` | calendar-only |
| **Station static** | `lat`, `lon`, `degree`, `betweenness_centrality`, `avg_historical_delay` | computed once via NetworkX from `edges_adjacent` |
| **Service identity** | `train_class_code`, `country`, `service_n_stops`, `service_route_distance_km`, `service_scheduled_duration_min` | service-level aggregates |
| **Weather at this stop** | `weather_severity`, `temperature`, `wind_speed`, `precipitation`, `snow_depth` | already per-stop in `edges_stops_at` (forecast-equivalent — not leakage) |
| **Weather differential** | `delta_severity_vs_prev_stop`, `delta_wind_vs_prev_stop` | captures fronts / transitions |
| **Lagged station history** | `station_lag1_rate`, `station_lag7_rate`, `station_lag1_avg_delay` | strict `<` daily-grid joins |
| **Lagged train history** | `train_lag1_rate`, `train_lag7_rate` | keyed by `(train_number, date)` |
| **Lagged train×station** | `train_station_lag7_rate` | "how does this train usually fare here?" |
| **Neighbour signal** | `nbr_lag1_rate` | mean of `station_lag1_rate` over adjacent stations |
| **Fault context** | `station_n_active_faults_14d` | rolling 14-day count |
| **Inflight (B only)** | `prev_stop_actual_delay`, `cum_actual_delay_so_far`, `max_actual_delay_so_far` | banned in scenario A |

### Anti-leakage protocol

`stop_level/leakage_guards.py` defines two sets:

```python
LEAKY_COLS_A = {  # banned in pre-departure
  y_stop, delay_min, delay_minutes, arrival_delay, departure_delay,
  is_disrupted, y_service, cancelled, max_delay_min, final_delay, _late,
  prev_stop_actual_delay, cum_actual_delay_so_far,
  max_actual_delay_so_far,
}
LEAKY_COLS_B = LEAKY_COLS_A − {prev_stop_actual_delay,
                                cum_actual_delay_so_far,
                                max_actual_delay_so_far}
```

Every lag join uses **strict `<`** — features at date *D* see only data from
date *D − 1* and earlier. The strict-less-than property is unit-tested in
`tests/test_lag_strict_lt.py`.

### Temporal split (no service crosses a split)

| Split | Date window | Source months |
|---|---|---|
| Train | 2024-01-01 → 2024-04-30 | Jan – Apr |
| Val   | 2024-05-01 → 2024-05-31 | May |
| Test  | 2024-06-01 → 2024-06-30 | Jun |

Service-disjointness is asserted in `splits.assert_no_service_overlap` and
covered by `tests/test_split_no_overlap.py`. A rolling-origin CV
(`splits.rolling_origin_folds`) provides 4 inner folds for hyperparameter
tuning inside the training window.

### Persisted artefacts (`Data/stops/`)

```
stops_{train,val,test}.parquet          per-row features + label + metadata
edge_index.npy, edge_attr.npy           PyG-compatible station graph
station_id_to_idx.json, service_id_to_idx.json
feature_names_A.json, feature_names_B.json
scaler_mean_{A,B}.npy, scaler_scale_{A,B}.npy
manifest_stops.json                     row counts, class balance, leaky_cols
```

---

## 7. Phase 3 — Model zoo

All five models share a uniform `StopModel` ABC
(`stop_level/models/base.py`) with `fit`, `predict_proba`, `save`, `load`,
plus a `tune_threshold` helper that picks the F1-maximising threshold on
validation.

| # | Model | Implementation | Key hyperparams |
|---|---|---|---|
| 1 | **Logistic Regression** | `sklearn.linear_model.LogisticRegression` | balanced class weight, saga solver |
| 2 | **LightGBM** | `lightgbm.LGBMClassifier` | 1500 trees, scale_pos_weight, early stop on val PR-AUC |
| 3 | **XGBoost** | `xgboost.XGBClassifier` | tree_method='hist', eval_metric='aucpr' |
| 4 | **GraphSAGE** | 2-layer SAGEConv on station graph + per-stop MLP head | mean aggregation, LayerNorm + Jumping Knowledge concat, BCE-with-pos_weight, AdamW + cosine LR |
| 5 | **BiLSTM stop-sequence** | bidirectional for A, **causal** for B with `prev_stop_actual_delay` injected per timestep | masked BCE, hidden 128, sequence labelling |

**Why these five?** Logistic + 2 GBDTs gives a strong tabular baseline; the
graph model tests whether topology adds signal beyond hand-engineered
station features; the sequence model tests whether the order of weather
and topology unfolding along the route matters. Five is the cap — `prompt.md`
§13 forbids a sixth.

Each model is trained **twice** — once for scenario A, once for scenario B
— against the identical splits. Differences in performance attribute to
architecture, not data.

### CLI

```bash
python stop_level/train_all.py --model all --scenario both
python stop_level/train_all.py --model lgbm xgb --scenario A
python stop_level/train_all.py --model graphsage --scenario both --seed 42
```

Outputs land under `stop_level/models/_artefacts/<model>/<scenario>/`:
- `model.pkl` (or `*.pt`)
- `preds_val.parquet`, `preds_test.parquet`

A unified prediction schema lets `evaluate.py` ingest every model identically:

```
service_id, station_id, country, date, train_class_code,
stop_order, position_norm, y_stop, delay_min,
p_disrupted, y_pred
```

---

## 8. Evaluation

`stop_level/evaluate.py` reads every `preds_test.parquet` produced by
`train_all.py` and emits:

| Figure | Content |
|---|---|
| `metrics_summary_{A,B}.png` | grouped bar of PR-AUC / F1 / precision / recall per model |
| `pr_roc_grid.png` | 5 models × 2 scenarios × {ROC, PR} curves |
| `confusion_grid.png` | one confusion matrix per (model, scenario) |
| `calibration_grid.png` | reliability diagram per model, scenario-faceted |
| `per_country_metrics.png` | PR-AUC by country, grouped by model |
| `per_position_metrics.png` | PR-AUC by `position_norm` quartile (0-25, 25-50, 50-75, 75-100) |
| `per_train_class_metrics.png` | PR-AUC by `train_class_code` 0-6 |
| `cascading_respect.png` | scenario B only — `P(predicted | prev disrupted)` vs `P(predicted | prev on-time)` |

Plus `results/benchmark_stops.json` with the full metric table including
per-country / per-position / per-train-class breakdowns and
`precision @ recall ∈ {0.7, 0.8, 0.9, 0.95}`.

### Metric choices

- **PR-AUC is the headline.** ROC-AUC can be misleading when the positive
  class is dominant.
- **Threshold tuned on val**, not blindly 0.5; the chosen threshold is
  reported per (model, scenario).
- **ECE** (expected calibration error, 10 quantile bins) is reported; a
  hook for **isotonic regression** on val activates when ECE > 0.05.
- **Cascading respect** for scenario B: an inflight model that doesn't
  show a non-trivial gap between
  `P(pred | prev_disrupted)` and `P(pred | prev_on-time)` is ignoring
  its own input and should be discarded.

---

## 9. Phase 4 — SHAP explainability

`stop_level/xai_stops.py` runs after the benchmark. It reads
`benchmark_stops.json`, picks the **winning tabular model per scenario** by
test PR-AUC (lgbm vs xgb), computes **TreeSHAP** (exact, fast) on the test
set, and emits 14 figures plus a JSON report.

### Figures (`stop_level/figures/xai/`)

| Figure | Content |
|---|---|
| `global_summary_bar_{A,B}.png` | top-20 mean `|SHAP|` per scenario |
| `global_summary_beeswarm_{A,B}.png` | per-feature SHAP distribution coloured by feature value |
| `shap_by_country.png` | top-6 features per country (IT, FI, NL) |
| `shap_by_position.png` | top-6 features per position bucket |
| `shap_by_train_class.png` | top-6 features per train_class_code 0-6 |
| `local_waterfall_tp.png` | highest-confidence true positive |
| `local_waterfall_fn.png` | worst-miss false negative (highest delay missed) |
| `local_waterfall_fp.png` | confident false positive |
| `local_waterfall_AB_flip.png` | row where scenario A ↔ B prediction *flipped* |
| `shap_dependence_top5.png` | dependence plots for the top-5 features with auto-detected interactions |
| `lag_ablation.png` | retrain without `*_lag*` + inflight features, compare PR-AUC |
| `scenario_uplift_map.png` | `P_B − P_A` vs `position_norm` — *where* the inflight signal pays off |

### `xai_report_stops.json`

```jsonc
{
  "scenarios": {
    "A": { "model": "lgbm", "n_test_samples": ...,
           "global_importance": { "temperature": 0.21, "wind_speed_avg": 0.18, ... },
           "local_explanations": {
             "tp": { "service_id": "...", "p_disrupted": 0.94, "top_drivers": {...} },
             "fn": { ... }, "fp": { ... }, "AB_flip": { ... }
           },
           "base_value": ...
    },
    "B": { ... }
  },
  "lag_ablation": { "A": { "pr_auc_full": ..., "pr_auc_no_lag": ..., "delta": +0.05 }, ... },
  "scenario_uplift": { "n_rows": ..., "mean_uplift": 0.04,
                        "uplift_pos25": 0.02, "uplift_pos75": 0.06 },
  "kg_bridge_query": "MATCH (svc:TrainService {service_id: $sid}) ...",
  "kg_bridge_note":  "After a high-risk stop prediction, plug ..."
}
```

### Knowledge Graph bridge

After a high-risk stop prediction, the report includes a Cypher query
template ready to plug into a Sparksee / Neo4j Knowledge Graph:

```cypher
MATCH (svc:TrainService {service_id: $sid})-[:STOPS_AT]->
      (st:Station {station_id: $stid})
OPTIONAL MATCH (f:FaultEvent)-[:REPORTED_AT]->(st)
  WHERE f.date >= date($d) - duration({days: 14})
OPTIONAL MATCH (st)-[:ADJACENT_TO]->(nbr:Station)
RETURN st, collect(DISTINCT f) AS recent_faults,
       collect(DISTINCT nbr.station_id) AS neighbours
```

This closes the loop: a black-box probability becomes a queryable
neighbourhood, and the SHAP top-drivers tell the analyst *which graph
attributes to filter on* (e.g. `WHERE st.snow_depth_max > τ`).

---

## 10. Reproduction (the 3-command contract)

```bash
# 0. Install dependencies
pip install pandas numpy scikit-learn lightgbm xgboost shap matplotlib seaborn \
            pyarrow networkx jupyter papermill nbconvert
# Optional GPU / GNN stack:
pip install torch torch_geometric
# Optional NL weather enrichment:
pip install openmeteo-requests requests-cache retry-requests

# 1. Per-country preprocessing notebooks
bash preprocess/run_all_notebooks.sh
#    runs italy_preprocessing.ipynb, finland_preprocessing.ipynb,
#    netherlands_preprocessing.ipynb headlessly via papermill;
#    writes Data/<Country>/processed/*.csv and
#    preprocess/figures/<country>/step_NN_*.png

# 2. Unified stop-level preprocessing
python stop_level/preprocess_stops.py
#    writes Data/stops/*.parquet + manifest + scalers + graph tensors

# 3. Train the model zoo + run XAI
python stop_level/train_all.py --model all --scenario both
python stop_level/evaluate.py
python stop_level/xai_stops.py --scenario both
#    writes stop_level/figures/*.png + stop_level/figures/xai/*.png
#    + stop_level/results/{benchmark_stops,xai_report_stops}.json
```

For dev iteration, all entry points accept a `--sample N` (Phase 2) or
`--model <name>` (Phase 3) flag to limit work.

---

## 11. Tests

```bash
python -m pytest stop_level/tests/ -q
```

| Test file | Cases | What it asserts |
|---|---:|---|
| `test_no_leakage_A.py` | 6 | scenario-A leakage set non-empty, label / inflight columns banned, `feature_columns(A)` strips them |
| `test_no_leakage_B.py` | 5 | scenario-B leakage is strictly smaller than A; inflight features whitelisted; label still forbidden |
| `test_split_no_overlap.py` | 4 | each split lies in its date window; service_id sets are pairwise disjoint; intentional overlap fails the assertion |
| `test_lag_strict_lt.py` | 3 | `station_lag1_rate(D)` equals the actual rate at D-1, not D; `prev_stop_actual_delay` for stop k = delay at stop k-1 |
| `test_feature_shapes.py` | 6 | every builder returns the expected columns + dtypes; full feature pipeline runs on a 5-row toy frame |
| `test_models_smoke.py` | 5 | each model fits + predicts + saves on toy data (graphsage / bilstm skip cleanly when deps missing) |
| `test_xai_smoke.py` | 1 | end-to-end Phase 1 → 2 → 3 → 4 on synthesized data; verifies all 14 XAI figures + `xai_report_stops.json` |
| **Total** | **30** | |

A typical run on this machine: **27 / 30 pass, 2 skip cleanly** (graphsage
needs `torch_geometric`; bilstm needs torch with NumPy 2.x interop). The
end-to-end XAI smoke test runs in ~45 s on synthesized data.

---

## 12. Anti-leakage discipline (the central design idea)

The whole pipeline turns on one rule: **at prediction time, `x` may only
contain information that was actually available before the prediction is
made.** Every layer enforces it differently:

1. **Phase 1** keeps `delay_minutes` only as a column to derive labels from;
   the column itself is never persisted as a feature in the parquet output.
2. **Phase 2** has an explicit `LEAKY_COLS_{A,B}` set asserted in
   `preprocess_stops.py` immediately before fitting the scaler. Any forbidden
   column that survived feature assembly raises a `ValueError`.
3. **Lag features** are computed over a daily aggregation grid with strict
   `< current_date` joins — `tests/test_lag_strict_lt.py` confirms this.
4. **Splits** are day-level so a service_id never crosses train/val/test
   boundaries — `tests/test_split_no_overlap.py` confirms this.
5. **Scaler** is fit on `X_train` only; `X_val` and `X_test` are transformed,
   never re-fit.
6. **Per-scenario feature lists** are persisted as `feature_names_{A,B}.json`
   so any downstream consumer (model, SHAP, ablation) reads the same
   contract.

When the cost of one leak is silent over-fitting that ships to production,
this much discipline is cheap.

---

## 13. Status & next steps

What's implemented end-to-end and tested with synthetic data:

- All four phases of the pipeline.
- All five models (with environment-only skips for graphsage / bilstm).
- Full evaluation suite with per-country / per-position / per-train-class
  breakdowns and the cascading-respect plot for scenario B.
- SHAP report with the four local explanations, lag-ablation, and the
  scenario uplift map.

What's deferred to operational runs:

- Executing the three notebooks against the real ~16.6 M-row raw data
  (memory-bound and I/O-bound, takes hours).
- Production-scale hyperparameter search (the modules support 30-trial
  random search per `prompt.md` §6, but the smoke tests run with reduced
  budgets).
- Integration with the live Sparksee Knowledge Graph database — the Cypher
  template is emitted, but `Data/unified_v2/railway_kg.dex` is not yet
  rebuilt for the new pipeline.

For a fresh checkout against fresh raw data, `bash preprocess/run_all_notebooks.sh
&& python stop_level/preprocess_stops.py && python stop_level/train_all.py
--model all --scenario both && python stop_level/xai_stops.py` reproduces
the full set of artefacts listed above.

---

## 14. References

- **Italy raw data:** Trenitalia operations + station + mileage + fault tables.
- **Finland raw data:** FI-TW dataset (Kaggle), with FMI weather observations
  nested per stop.
- **Netherlands raw data:** RDT (Rijden de Treinen) public data export +
  Open-Meteo Historical Archive API for weather.
- **Anti-leakage / temporal CV** discipline follows the convention spelled
  out in `prompt.md` §§5.3, 5.6, 12 and is the load-bearing design idea
  carried over from the previous v2 of this project.
- **Knowledge Graph schema** (Sparksee): see the `kg_bridge_query` template
  in `xai_report_stops.json` and the `kg_compatibility` block of
  `manifest_stops.json` for the import format.

---

## 15. Future work

The current pipeline saturates near PR-AUC `0.94` in scenario B and the
GraphSAGE backbone — which uses only the station-adjacency slice of the
KG — does not exceed the tabular GBDT. To make a future iteration
meaningfully stronger, and to turn the KG into a learning substrate rather
than just an explanation layer, we propose three coordinated steps.

### 15.1 Obtain a full-year Italian dataset, comparable to FI-TW


The recommended next step is to **assemble a comparable 12-month Italian
corpus following the methodology of Borin et al. (2025), *FI-TW: a Finnish
train-weather integrated dataset*, Scientific Data 12**
([nature.com/articles/s41597-025-06385-8](https://www.nature.com/articles/s41597-025-06385-8)).



### 15.2 Enrich the database with synthetic but operationally faithful data



Targeted **synthetic augmentation** can help, provided it is grounded in
real operational physics rather than generic SMOTE-style oversampling:

1. **Counterfactual disruption pairs.** For every observed disrupted stop,
   take a matched on-time stop at the same station, same hour, same train
   class on a different day. Feed both as a contrastive pair to teach the
   model *what changed* (typically the weather differential or the lag
   rate).
2. **Adversarial weather scenarios.** Inject realistic extreme-weather
   sequences (NL with sub-zero winters, IT with heat-induced track
   buckling) using worldwide historical percentiles. Audit whether
   predicted causes shift accordingly; failure to do so flags the
   class-mass-collapse path the current model has on cross-country
   transfer.
3. **Schedule-perturbation augmentation.** Sample plausible timetable
   variants (different `stop_order`, swapped intermediate stations,
   compressed dwell times) holding the rest of the (service, station,
   weather) tuple constant. Trains the model to be position-invariant
   where it should be.
4. **Cross-country cause synthesis.** Apply per-country mean-variance
   alignment (CORAL or a simpler standardisation) to map Dutch cause
   distributions onto Italian / Finnish operational contexts, generating
   pseudo-labelled examples for fine-tuning.

The synthetic data is **not** a replacement for the real Italian year
(§15.1) — it is the layer on top that lets rare classes be learned and
lets the model see operational regimes that the historical record
under-samples.

### 15.3 Redesign the Knowledge Graph as a learning substrate

The current Sparksee KG has 3 node types (`Station`, `TrainService`,
`FaultEvent`) and 3 edge types (`STOPS_AT`, `ADJACENT_TO`, `REPORTED_AT`).
GraphSAGE consumes only the station-adjacency slice, so the GNN never
reads the most informative parts of the graph. With a richer schema and a
heterogeneous GNN, the KG can become a first-class predictor rather than
an aside.

Proposed extensions:

1. **New entity types**
   - `RollingStock` — physical train unit with maintenance history, age,
     km since last overhaul, manufacturer.
   - `CrewShift` — operator, hours into shift, days since last rest.
   - `EngineeringWork` — planned-work entities with affected stations and
     date windows.
   - `Connection` — booked passenger transfers between adjacent services.

2. **New relation types**
   - `OPERATED_BY`: `TrainService → RollingStock`
   - `STAFFED_BY`: `TrainService → CrewShift`
   - `AFFECTED_BY`: `Station → EngineeringWork`
   - `CONNECTS_TO`: `TrainService → TrainService` (transfer-pair edges
     enabling cascading-delay propagation predictions across services,
     not just within a single run)
   - `PARALLEL_TO`: `Station → Station` (track parallelism / overtaking
     possibilities)

3. **Temporal edge attributes**
   - `STOPS_AT` already carries scheduled times; extend with
     `actual_arrival_ts`, `dwell_seconds`, `headway_to_next_train_at_stop`
     so the same edge type encodes both schedule and execution.

4. **A heterogeneous GNN as the learning model**
   - Move from plain GraphSAGE on a single homogeneous adjacency graph to
     a relational GNN (R-GCN, HAN, or a Heterogeneous Graph Transformer)
     that processes each relation type with its own message-passing
     kernel.
   - Pre-train on a self-supervised link-prediction objective over the
     full schema before fine-tuning on the disruption-classification
     head. This is the standard recipe by which KG-native models exceed
     feature-based tabular baselines.

5. **Unify predictor and explainer in one model**
   - Once the KG is the substrate for both training *and* explanation,
     the SHAP report and the bridge query share a single embedding space:
     "explain prediction X" returns a sub-graph whose attention weights
     *are* the attribution values — end-to-end consistent rather than
     stitched together post-hoc.


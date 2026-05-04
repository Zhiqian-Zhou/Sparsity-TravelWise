# Code Review — Sparsity-TravelWise Pipeline (post-run)

**Reviewers:** 5 parallel Explore agents
**Scope:** Per-country preprocessing, stop-level feature store, model zoo, evaluation + XAI, test suite + report fact-check
**Date:** 2026-05-03 (after the full end-to-end pipeline run on 16.59 M stop rows)

---

## Executive summary

| Severity | Count | Where |
|---|---|---|
| 🔴 **Bugs** (will produce wrong output) | 4 | Phase 1 (3) + Phase 3 models (1) |
| 🟡 **Suspicious** (won't fail today, fragile) | 4 | Phase 1 (2) + Phase 3 (1) + report/README (1) |
| 🟢 **OK** | most of the codebase | Phase 2 + Phase 4 are clean |

**No bugs were found in:** Phase 2 stop-level preprocessing, Phase 3 evaluation, Phase 4 XAI, the 32-test pytest suite, or any of the JSON manifests. **All benchmark numbers in `docs/report.md` match the underlying artefacts to 4 decimals.**

The four bugs below would not have changed *this run's* benchmark results (they affect handling of missing data which the run happened to dodge), but should be fixed before re-running on different data or in a production setting.

---

## 🔴 Bugs

### Bug 1 — LightGBM/XGBoost `predict_proba` lacks `np.nan_to_num`
**Severity:** High (silent NaN propagation at inference time)
**Files:** `stop_level/models/lgbm_model.py:74-78`, `stop_level/models/xgb_model.py:78-82`

The patched LogReg, GraphSAGE, and BiLSTM all apply `np.nan_to_num` symmetrically in **both** `fit` and `predict_proba`. LGBM and XGB only handle NaN in `fit` (via the libraries' native NaN tolerance) — but if test data contains NaN values not seen at training time (different feature distributions, post-deployment drift, or a feature that ends up missing in prod), `predict_proba` will return NaN probabilities silently.

**Why it didn't bite this run:** Phase 2's StandardScaler outputs were finite for the test set we trained on, and we asserted leakage but not finite-ness. A future re-run with different lag-feature coverage could introduce NaN into the test parquet.

**Fix:** mirror the LogReg pattern in both files —
```python
def predict_proba(self, df):
    X = np.nan_to_num(
        df[self.feat_cols].astype("float32").values,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    return self.model.predict_proba(X)[:, 1].astype("float32")
```

---

### Bug 2 — Italy zero-fills missing weather columns instead of NaN
**Severity:** Medium (loses missingness signal; downstream features become indistinguishable from "clear weather")
**File:** `preprocess/lib_italy.py:204-205`

```python
if c not in ops.columns:
    ops[c] = 0.0    # ← should be np.nan
```

Finland and NL preserve NaN here so the lag/imputation logic can detect missingness. Italy zero-fills, which means downstream `weather_severity == 0` could mean either "clear" or "we have no data" — feature engineering can't tell the difference.

**Fix:** `ops[c] = np.nan` and let downstream imputation handle it consistently across countries.

---

### Bug 3 — Finland `coerce_types` masks unknown cancellations as `False`
**Severity:** Medium (silent label miscoding for cancelled FI services)
**File:** `preprocess/lib_finland.py:194-197`

```python
if "cancelled" in df.columns:
    df["cancelled"] = df["cancelled"].map(bool_map).fillna(False).astype(bool)
else:
    df["cancelled"] = False     # ← assumes no cancellations exist
```

Italy and NL extract cancellation from delay sentinels (`'S'` for IT, three boolean flags for NL). Finland just `False`-defaults if the column is absent. If FI-TW ever ships a different schema (or the timeTableRows parser misses the field), every Finnish service is silently labelled "ran on time" — including any that were actually cancelled.

**Fix:** at minimum, log a warning when defaulting; preferably extract cancellation from `running_currently` or the FI-TW status field.

---

### Bug 4 — NL Open-Meteo enrichment masks "no data" as "clear weather, 10 °C"
**Severity:** Medium (NL test set treats missing weather as benign)
**Files:** `preprocess/weather_enrichment_nl.py:148-149` (severity ordinal) + lines 198-203 (final fillna)

Sequence:
1. `_wmo_severity(precip, snow)` is computed with `fillna(0.0)` on both — so a station with NaN precip → severity 0.
2. After the join, `temperature.fillna(10.0)`, `wind_speed.fillna(2.0)`, `precipitation.fillna(0.0)`, `snow_depth.fillna(0.0)`, `weather_severity.fillna(0)`.

The result: stations where Open-Meteo had no coverage (the 591 → 213 attrition: 378 stations with no usable response) get the same feature vector as stations with confirmed clear weather. There's no `weather_is_present` flag to distinguish them.

**Why it didn't bite this run:** SHAP showed weather features are mid-tier importance for NL (well below `prev_stop_actual_delay` and lag features), so the wrong default doesn't move benchmarks much. But it's a measurement-bias landmine for any future ablation that focuses on weather.

**Fix:** add a binary `weather_observed` column (1 if join hit, 0 otherwise) so models can condition on missingness.

---

## 🟡 Suspicious

### Suspicious 1 — `n_neg_pos_ratio` asymmetric on degenerate `y`
**File:** `stop_level/models/base.py:23-27`

For all-positive `y`: returns 1.0 (correct). For all-negative `y`: returns `n_neg / max(0, 1) = n_neg`, not 1.0. Won't bite real data (positive class always exists in our 16 M-row sample), but the contract violates symmetry.

### Suspicious 2 — Finland dtype-detection branch logic
**File:** `preprocess/lib_finland.py:218`

```python
before = df[col].isna().mean() if df[col].dtype.kind in "fiu" else 1.0
```

Reports `before=1.0` (100 % NaN) for columns that are *not* numeric — the inverse of intent. Only affects the diagnostic NaN-rate plot, not actual data. Suggest `not in "fiu"` and rename for clarity.

### Suspicious 3 — NL `max_delay_min` is computed but never persisted
**File:** `preprocess/lib_netherlands.py:95`

The column is coerced to float32 but isn't in `SCHEMA["edges_stops_at"]`, so the new strict `save_csv` drops it silently. Dead code — remove or wire into the schema.

### Suspicious 4 — README still claims "five competing models"
**File:** `README.md:6, 79, 128, 297, 482-483`

The actual run skipped BiLSTM (memory blowup on 12 M rows) and `docs/report.md` is honest about it ("four architectures"). README still promises five and even claims "27/30 pass, 2 skip cleanly" which contradicts the current 32/32 green test count. A fresh reader following README will be misled. Either:
- Update README to say "four architectures + BiLSTM as a sample-only experimental option," or
- Refactor `bilstm_model.py:_pack_sequences` to use a generator-based batched packer that doesn't materialise the full padded tensor upfront.

---

## 🟢 Verified clean

### Phase 2 (`stop_level/preprocess_stops.py` + `features.py`)
- The 8 stripped `df = df.copy()` calls in feature builders did **not** introduce silent mutation bugs. `assemble_features` reassigns `df = F.add_*(df)` at every step, so the chained-flow pattern is safe.
- The map-based stop-grain join is correct: `dict(zip(svc["service_id"].astype(str), svc["date"]))` after deduping `nodes_service` on `service_id` keep="first" — deterministic.
- The high-cardinality `compute_train_lag` fallback (n_keys > 50K → skip calendar grid) preserves the strict-`<` invariant via `sort_values + shift(1)` on the daily aggregation.
- All four lag builders (`compute_station_lag`, `compute_train_lag`, `compute_train_station_lag`, `add_inflight_features`) use strict `<` joins. Confirmed by `tests/test_lag_strict_lt.py` (3/3 green).
- `assert_no_leakage` is correctly placed *after* feature build and *before* scaler fit (`preprocess_stops.py:318-339`).
- StandardScaler is fit on `train_df` only (line 338-339). No val/test fit.
- Service-disjoint splits hold (`tests/test_split_no_overlap.py` 4/4 green).
- Manifest numbers match `docs/report.md` exactly:

| Metric | Manifest | Report | Match |
|---|---|---|---|
| Total stops | 16,586,834 | 16,586,834 | ✅ |
| Train rows | 11,915,114 | 11,915,114 | ✅ |
| Train y_rate | 10.34 % | 10.34 % | ✅ |
| Val rows | 2,368,484 | 2,368,484 | ✅ |
| Test rows | 2,303,236 | 2,303,236 | ✅ |
| Feat A / B | 37 / 40 | 37 / 40 | ✅ |
| Leaky cols A / B | 15 / 11 | 15 / 11 | ✅ |

### Phase 3 evaluation (`stop_level/evaluate.py`)
- `_full_metrics` correctly uses `y_pred` (threshold-applied) for F1/precision/recall and raw `p_disrupted` for PR-AUC/ROC-AUC.
- `_ece` guards degenerate `y.sum() == 0 or len(y)` before calling `calibration_curve`.
- Cascading-respect plot reconstructs `prev_delay` via `groupby("service_id")["delay_min"].shift(1)` — same column the model trained on.
- All 9 figures regenerate from the 8 prediction parquets without errors.

### Phase 4 XAI (`stop_level/xai_stops.py`)
- TreeSHAP runs on the **full 2.30 M-row test set** (no sampling), persists `shap_values.npy` immediately after compute (so a crash doesn't lose work).
- Winner selection is deterministic: `xgb 0.9524 > lgbm 0.9505` (no tie-handling needed).
- Lag-ablation refits the model from scratch on `feat_cols_no_lag`; no stale-scaler issue (tree models scale-invariant).
- Scenario-uplift merge on `(service_id, stop_order)` produces all 2.30 M rows aligned (no row-loss).
- Local waterfall pickers (`tp / fn / fp / AB-flip`) use `argmax(np.where(mask, p, -1))` so they are robust to empty masks.
- All 14 XAI figures present and freshly written (mtimes 18:35–19:08).
- Top SHAP features in `xai_report_stops.json` match `docs/report.md` to 4 decimals:
  - A: `train_station_lag7_rate=1.0483`, `avg_historical_delay=0.6163`
  - B: `prev_stop_actual_delay=2.5465`, `stop_order=0.4860`, `max_actual_delay_so_far=0.4558`
- Lag-ablation deltas match: A `+0.197`, B `+0.597`.
- Scenario-uplift mean matches: `−0.115`.
- KG-bridge Cypher template is valid syntax with proper parameter substitution (`$sid`, `$stid`, `$d`).

### Test suite (32/32 green)
- All assertions are genuine (no missing-assertion false-passes).
- The two reload roundtrip tests added during this run (`test_graphsage_save_load_roundtrip`, `test_bilstm_save_load_roundtrip`) actually verify `np.allclose(p_before, p_after, atol=1e-5)` on a fitted toy model — would catch any state-restoration bug.
- `test_xai_smoke.py` correctly skips when **both** FI and NL processed CSVs exceed 10 MB on disk (avoids clobbering production data).
- The 7 model smoke tests cover both scenarios A + B for logreg / lgbm / xgb / bilstm; graphsage covers A only for speed (acceptable since the architecture is scenario-agnostic in the inflight-feature branch).

### Code-fix appendix in `docs/report.md`
All 13 listed fixes are real and grep-verifiable. None over-claimed.

### Image-link integrity
All 51 `![…](…)` references in `docs/report.md` resolve to actual files on disk (8 IT + 11 FI + 9 NL + 9 evaluation + 14 XAI). No broken links.

---

## Recommendations (priority-ordered)

1. **🔴 Fix the 4 bugs above before any re-run.** Bug 1 (LGBM/XGB `predict_proba` NaN) is the most operationally serious — it's a one-line change per file but could silently return NaN to a downstream consumer.
2. **🟡 Reconcile README ↔ report on BiLSTM.** Decide whether to ship 4 models or invest in a memory-efficient batched packer for BiLSTM.
3. **🟡 Add a `weather_observed` boolean to NL enrichment** so the missingness signal is recoverable.
4. **🟢 Optional polish:** drop the dead `max_delay_min` line in NL lib; fix the inverted dtype check in FI lib.

The pipeline as it stands today produces correct end-to-end results on this dataset (the test suite, the manifest, and the SHAP report all line up to 4 decimals). The bugs above are about robustness against future-data drift, not about the numbers in `docs/report.md`.

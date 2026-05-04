# Second-Pass Code Review

**Reviewers:** 5 parallel Explore agents
**Date:** 2026-05-03 (after Phase 5 KG + Phase 6 cause prediction added)
**Coverage:** Phase 1 / Phase 2 / Phase 3 / Phase 4-6 / cross-cutting consistency

---

## Headline verdict

**Overall consistency: MEDIUM** (robust on this dataset; fragile on drift). All 32 tests pass, all 51 image links resolve, all numerical claims in `report.md` match the JSON manifests to 4 decimals.

| Phase | Verdict | Critical issues |
|---|---|---|
| Phase 1 — country preprocessing | ✅ PASS (with 4 known bugs) | All 4 bugs from `REVIEW.md` still present |
| Phase 2 — stop-level features | ✅ PASS | 24/24 pytest green; lag invariants correct |
| Phase 3 — model zoo + benchmark | ✅ PASS (with 1 known bug) | Bug #1 (LGBM/XGB predict_proba NaN) still unfixed |
| Phase 4 — XAI | ✅ PASS | 14 figures all present; numbers match JSON exactly |
| Phase 5 — KG | ✅ PASS | 8.5 GB Kuzu DB, all counts match, bridge query 52 ms |
| Phase 6 — cause prediction | ✅ PASS / ❌ Transfer USELESS | NL benchmark honest; transfer to IT/FI fails per CAUSE_REVIEW.md |

---

## Cross-validation matrix (numerical claims)

| Metric | Source of truth | Report claim | Match |
|---|---|---|---|
| Total stops | manifest_stops.json | 16,586,834 | ✅ |
| Per-country stops (IT/FI/NL) | processed CSVs | 2,678,291 / 3,112,754 / 10,795,789 | ✅ |
| Train/val/test split | manifest_stops.json | 11,915,114 / 2,368,484 / 2,303,236 | ✅ |
| Features A/B | feature_names_*.json | 37/40 | ✅ |
| Leaky cols A/B | manifest | 15/11 | ✅ |
| 8 model PR-AUC values | benchmark_stops.json | All match `report.md §6` to 3 decimals | ✅ |
| Top SHAP A/B | xai_report_stops.json | 1.0483 / 2.5465 | ✅ |
| Lag ablation A/B | xai_report_stops.json | +0.197 / +0.597 | ✅ |
| KG counts (6 tables) | Data/kg/manifest.json | 2397 / 1217406 / 2938 / 16586834 / 163902 / 2938 | ✅ |
| KG bridge query latency | manifest.json | 52 ms | ✅ |
| Cause winner | cause_benchmark.json | RF (val_macro_f1=0.187) | ✅ |
| Cause IT/FI distributions | cause_transfer.json | staff 70.1% / rolling stock 65.9% etc. | ✅ |

**No numerical drift detected anywhere.**

---

## Outstanding issues (from REVIEW.md, still present)

The 5 agents independently confirmed all 4 bugs from the prior `REVIEW.md` are still in the code, exactly as documented:

1. **🔴 LGBM/XGB predict_proba lacks `np.nan_to_num`** — `lgbm_model.py:74-78`, `xgb_model.py:78-82`. LogReg/GraphSAGE/BiLSTM are symmetric; LGBM/XGB are not. Silent NaN propagation risk on data drift.
2. **🔴 Italy zero-fills missing weather as 0.0** — `lib_italy.py:204`. Loses missingness signal; downstream the temperature column is all-zero on IT, which is the smoking gun for the cause-prediction transfer failure to Italy.
3. **🟡 Finland defaults `cancelled = False`** — `lib_finland.py:209-212`. Silent miscoding if timeTableRows ever lacks cancellation.
4. **🟡 NL weather defaults mask missingness** — `weather_enrichment_nl.py:194-203`. 378 of 591 NL stations got `temperature=10°C` as a fillna default, indistinguishable from "actually 10°C".

These were not blocking the current run but will surface on re-runs or different data. Not yet fixed because the user hasn't asked us to.

## README ↔ report mismatch (still unresolved)

- `README.md:6, 79, 128, 297, 482-483` claims "five competing models" / "27/30 pass, 2 skip cleanly"
- `docs/report.md:308` honestly says "Four architectures (BiLSTM deferred — memory blowup)"
- Tests are now **32/32 green**, not 27/3
- Recommendation: update README to align with what actually shipped.

## Net new findings from this round

The KG (Phase 5) and cause prediction (Phase 6) modules added since the last review pass were audited and found clean:

- `build_kg.py` — clean code, proper schema staging, correct dedup, clear Sparksee swap path. Bridge query syntactically valid in Kuzu and Cypher-portable to Sparksee.
- `cause_prediction.py` — code is **technically correct**; the documented failure mode (transfer to IT/FI) is a domain-shift / class-balance pathology, not a code bug.
- `weather_enrichment_nl.py` — Bug #4 confirmed (already documented). No new bugs.

## Top-3 priority fixes

1. **🔴 Fix LGBM/XGB predict_proba NaN handling** — single-line change per file. Highest production risk.
2. **🟡 Update README to reflect 4-model reality + 32/32 tests** — purely documentation hygiene.
3. **🟡 Add `weather_observed` boolean to NL enrichment** — unlocks missingness signal recovery; one-column change.

## Per-agent verdicts

| Agent | Scope | Verdict |
|---|---|---|
| Agent 1 | Phase 1 country libs | SUSPICIOUS (4 bugs persist; otherwise PASS) |
| Agent 2 | Phase 2 features + leakage | PASS (24/24 tests green) |
| Agent 3 | Phase 3 models + eval | PASS-with-bug (Bug #1 still active) |
| Agent 4 | Phase 4-6 (XAI + KG + cause) | PASS (12/12 verification items) |
| Agent 5 | Cross-cutting consistency | MEDIUM (robust here, fragile on drift) |

The pipeline as it stands is **safe for this run's dataset and reporting**. The bugs catalogued in `REVIEW.md` are robustness landmines that should be resolved before re-running on different data or putting any of these models into production serving.

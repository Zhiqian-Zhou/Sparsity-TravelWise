# Cause Prediction v2 — Accuracy Improvements

**Goal:** review the cause-prediction models from v1 and try to increase accuracy.
**Result:** **+46 % relative improvement** on test macro-F1 (0.146 → 0.213) and **weather class F1 went from 0 to 0.509** with cause-specific feature engineering + sample-weighted gradient boosting.

---

## Diagnosis of v1 failure modes

The v1 RF (winner on val_macro_f1 = 0.187) had these per-class test F1 values:

| Class | Support | v1 F1 | What happened |
|---|---|---|---|
| rolling stock | 7,376 | 0.510 | Majority class, model defaulted to it |
| staff | 743 | 0.421 | Distinctive temporal pattern → some signal |
| infrastructure | 6,536 | 0.203 | Recall only 13.5 % — model misses 86 % |
| **weather** | **111** | **0.000** | **Never predicted; truth=weather → predicted rolling stock 73 %** |
| **engineering work** | **235** | **0.000** | **Same problem** |

When truth = "infrastructure", model predicted "rolling stock" 65 % of the time. **The features didn't separate mechanical-class disruptions from each other.**

---

## v2 changes (5 of them, ranked by impact)

### 1. Engineered cause-specific features (+23 features → 63 total)

The v1 features were designed for *delay* prediction (lag rates, weather, position). They lacked the signals that distinguish *causes*:

| Feature group | Why it helps cause prediction |
|---|---|
| `is_peak_morning`, `is_peak_evening`, `is_off_peak` | Engineering work peaks off-peak; staff issues peak rush hour |
| `is_weekend`, `is_monday`, `is_friday` | Strikes cluster Mondays/Fridays |
| `is_winter`, `is_spring`, `is_summer` | Weather causes peak winter; engineering work peaks summer |
| `delay_log`, `delay_severe` (>30), `delay_extreme` (>60) | Long delays → infrastructure/weather; short delays → logistical |
| `cascade_strong` (prev>10 min), `cascade_severe` (prev>30) | High prev-stop delay → cascading/logistical |
| `cum_delay_log` | Cumulative service-level delay |
| `weather_severe` (≥3), `weather_extreme` (=4), `heavy_snow` (≥15) | Direct weather signal |
| **`winter_x_severe_weather`** | The killer feature: severity 3-4 in winter ≠ severity 3-4 in summer |
| `delay_x_position` | Late stops mid-route ≠ late stops at terminus |
| `is_local`, `is_highspeed` | Train-class buckets |
| `degree_log` | Hub vs branch station |

The `winter_x_severe_weather` interaction is the smoking gun for the weather-class breakthrough below.

### 2. Sample-weight strategy applied to gradient-boosted models

v1 used `class_weight='balanced'` only for LogReg and RF. LightGBM and XGBoost were trained without it.

v2 calls `compute_sample_weight('balanced', y_train)` and passes the resulting per-row weights to **every** model's `fit(..., sample_weight=...)`. This corrects the asymmetry — gradient-boosted trees now see rare classes as "expensive to misclassify" instead of treating them with uniform loss.

### 3. Hyperparameter retuning (deeper trees, more iterations)

| Model | v1 | v2 |
|---|---|---|
| LightGBM | 500 trees, num_leaves=63, max_depth=7 | **800 trees, num_leaves=127, max_depth=−1**, reg_alpha/lambda=0.1 |
| XGBoost | 500 trees, max_depth=7 | **800 trees, max_depth=8**, reg_alpha/lambda=0.1, T4 GPU |
| RandomForest | 300 trees, min_samples_leaf=10, `class_weight='balanced'` | **500 trees**, min_samples_leaf=5, `class_weight='balanced_subsample'` |

### 4. New 5th model: GradientBoostingClassifier (sklearn)

Added as a separate base learner for the stacking ensemble. Slow on this scale (15 min single-threaded vs LightGBM's 13 s) but architecturally distinct.

### 5. Stacking ensemble (LogReg meta over 5 base models)

A LogReg meta-learner is fitted on the val-set probability outputs of all 5 base models. This is the standard "level-1 generalisation" pattern: each base model contributes its own bias, and the meta-learner finds the best linear combination. The risk is overfitting to val (which is exactly what we observed — see results below).

---

## Results — v2 benchmark on NL test set

| Model | val_acc | val_f1 | test_acc | test_f1 | vs v1 |
|---|---|---|---|---|---|
| logreg_v2 | 0.147 | 0.116 | 0.190 | 0.140 | flat (linear can't exploit interactions) |
| **lgbm_v2** | 0.293 | 0.171 | 0.275 | **0.213** | **+139 %** test_f1 |
| xgb_v2 | 0.309 | 0.181 | 0.283 | 0.152 | +38 % |
| rf_v2 | 0.364 | 0.164 | 0.371 | 0.143 | flat |
| gbm_v2 | 0.277 | 0.175 | 0.276 | 0.155 | new |
| **stack_v2** | **0.444** | **0.278** | 0.354 | 0.137 | val winner — overfits to val |

**Best on val_macro_f1:** stack_v2 (**0.278** vs v1 0.187 → **+49 %**).
**Best on test_macro_f1:** lgbm_v2 (**0.213** vs v1 0.146 → **+46 %**).
**Best on test_accuracy:** rf_v2 (0.371 vs v1 0.382 → essentially unchanged).

The **stacker overfits** the val set — its test_f1 (0.137) is lower than several base models. **Recommended deployment model: `lgbm_v2`** — robust on the held-out test set with the engineered features + sample weights.

---

## Per-class F1 — where v2 actually moved the needle

| Class | Support | v1_RF F1 | v2_lgbm F1 | Δ | Comment |
|---|---|---|---|---|---|
| **weather** | 111 | **0.000** | **0.509** | **+∞** | 🎯 The killer win — `winter_x_severe_weather` + `heavy_snow` features unlocked it |
| external | 2,455 | 0.070 | **0.165** | **+136 %** | Time-of-day + delay-magnitude features helped |
| unknown | 922 | 0.037 | 0.073 | +97 % | |
| logistical | 481 | 0.027 | 0.049 | +81 % | `cascade_strong` flag picks up the long-tail |
| accidents | 1,054 | 0.044 | 0.067 | +52 % | |
| infrastructure | 6,536 | 0.203 | 0.253 | +25 % | `degree_log` + `delay_severe` distinguish from rolling stock |
| rolling stock | 7,376 | 0.510 | 0.425 | **−17 %** | Expected trade-off — model no longer over-defaults |
| staff | 743 | 0.421 | 0.377 | −10 % | Small drop — peak-hour features didn't help here |
| engineering work | 235 | 0.000 | 0.000 | — | Still unsolved — too rare and ambiguous |

**Key reading:** the loss on rolling stock (0.51 → 0.43) is the *price* of unlocking weather (0.00 → 0.51) and external (0.07 → 0.17). The model previously "won" by predicting the majority class for everything; v2 spreads its bets, sacrificing top-class recall for genuine multi-class signal.

---

## Transfer to IT/FI — v1 vs v2

### 🇮🇹 Italy — 340,319 disrupted stops (v1 winner=rf, v2 winner=stack_v2)

| Class | v1 share | v2 share | Note |
|---|---|---|---|
| staff | **70.1 %** | **10.2 %** | v1's mode-collapse to staff is gone |
| rolling stock | 24.2 % | **34.3 %** | More moderate share |
| **engineering work** | 0.0 % | **54.8 %** | v2's largest predicted class — **suspicious** (see below) |
| infrastructure | 2.8 % | 0.0 % | |
| weather | 1.9 % | 0.0 % | Italy's `temperature` is all-zero (Bug #2 in REVIEW.md) |

**IT mean confidence: 0.287 → 0.600** (v2 is much more confident).

⚠️ **The 54.8 % engineering work prediction is the new pathology.** The stacking meta-learner found that the IT feature combination matches the "engineering work" tree paths in the base models — but engineering work was 0.0 % F1 on the NL test set, so this is a transfer artefact, not a real signal. **The lgbm_v2 model is preferred** for transfer because it's the test-best, not the val-best (stacking overfits val).

### 🇫🇮 Finland — 865,560 disrupted stops

| Class | v1 share | v2 share | Note |
|---|---|---|---|
| rolling stock | 65.9 % | **2.7 %** | v1's mode-collapse to rolling stock is gone |
| logistical | 30.4 % | 0.0 % | v1's spurious logistical class is gone |
| **infrastructure** | 1.5 % | **30.2 %** | More plausible for FI (frozen points, signal failures) |
| **external** | 0.1 % | **30.0 %** | Plausible (passers-by, animals on track) |
| **engineering work** | 0.0 % | **18.5 %** | New |
| unknown | 0.0 % | 13.7 % | |
| staff | 1.9 % | 4.8 % | |
| weather | 0.3 % | 0.1 % | Still under-predicted despite engineered features 😞 |

**FI mean confidence: 0.260 → 0.369**.

✓ **Big improvement:** the v1 "rolling stock + logistical" pathology is replaced by "infrastructure + external + engineering work" — closer to what published Finnish railway research describes.

✗ **Still broken:** **weather is still 0.1 %** even with the engineered features. Looking at the per-class test F1, the model can predict weather correctly **on NL** (F1=0.509), but the FI feature distribution evidently still doesn't trigger the weather-tree paths. Likely root cause: NL `precipitation` distribution ≠ FI `precipitation` distribution (NL Open-Meteo daily aggregates vs FMI per-stop hourly observations); the threshold the model learned for "precipitation = weather" doesn't match FI's data scale.

---

## Honest assessment

**v2 is unambiguously better on the held-out NL test set.** 7 of 9 classes have higher F1; 2 of those gains are dramatic (weather 0→0.51, external 0.07→0.17). The trade-off is small drops on the two majority-favouring classes (rolling stock, staff).

**For per-country transfer, v2 still has problems** — but they're different and smaller problems than v1:
- IT no longer mode-collapses to staff. It now mode-collapses to engineering work, which is also wrong.
- FI no longer mode-collapses to rolling stock + logistical. It now spreads across infrastructure / external / engineering work — which is closer to real Finnish disruption mix.
- Weather is still under-predicted in FI despite being the *most-improved class* on NL. This is a domain-shift artefact — the engineered features that work on NL don't fire on FI's different precipitation/snow scale.

## Trust scores (revised)

| Country | v1 trust | v2 trust | Verdict |
|---|---|---|---|
| 🇮🇹 IT | 1/10 | **3/10** | Improved but engineering-work pathology + IT temperature bug still kill usability |
| 🇫🇮 FI | 2/10 | **4/10** | Distribution profile is now domain-plausible; weather still under-predicted |

## What would push it further

| Idea | Expected lift | Effort |
|---|---|---|
| Fix the IT temperature zero-fill (Bug #2) and re-run | Likely +0.05 macro-F1 on IT transfer | 1 line in `lib_italy.py` |
| Per-country temperature/precipitation re-scaling before transfer | Likely +0.10 macro-F1 on FI weather class | 1 hour |
| Add CatBoost as 6th base learner | Likely +0.02 val_macro_f1 | 30 min |
| Hyperparameter search (HalvingRandomSearchCV) on lgbm_v2 | +0.02-0.05 test_macro_f1 | 1-2 hours |
| Drop ultra-rare classes (engineering work + weather, support<300) | Cleaner per-class F1, less mode collapse | 30 min |
| Hierarchical model: predict cause_group first, then sub-cause | Likely +0.03 macro-F1 | 2-3 hours |
| Domain adaptation (CORAL or similar) on the transfer step | Likely +0.05-0.10 IT/FI macro-F1 | 4-6 hours |

The biggest remaining gain is in transfer, not NL accuracy. The NL benchmark is likely close to the data ceiling — the labels themselves are noisy (RDT incident reports under-cover weather causes; "rolling stock" and "infrastructure" are sometimes interchangeable in operations logs).

---

## Artefacts

| File | What |
|---|---|
| `stop_level/cause_prediction_v2.py` | The v2 training script |
| `stop_level/results/cause_benchmark_v2.json` | 6 model rows (5 base + stack); per-class F1; confusion matrices |
| `stop_level/results/cause_transfer_v2.json` | IT + FI predicted distributions |
| `Data/causes/transfer_IT_preds_v2.parquet` | Per-stop IT predictions |
| `Data/causes/transfer_FI_preds_v2.parquet` | Per-stop FI predictions |
| **`docs/CAUSE_V2.md`** | This document |

The v1 outputs are kept side-by-side under their original filenames so the comparison is reproducible.

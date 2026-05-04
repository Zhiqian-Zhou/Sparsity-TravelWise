# Cause-of-Disruption Predictions — Audit Report

**Reviewers:** 2 parallel Explore agents (one per transfer country)
**Pipeline:** `stop_level/cause_prediction.py`
**Models:** LogReg, LightGBM, XGBoost, RandomForest, MLP (5 distinct families)
**Training data:** 126,810 NL stops with cause_group from RDT disruptions log (9 classes)
**Transfer targets:** 340,319 IT disrupted stops + 865,560 FI disrupted stops
**Date:** 2026-05-03

---

## Executive verdict

| Country | Verdict | Trust score | Top predicted cause | Plausibility |
|---|---|---|---|---|
| 🇮🇹 IT | **USELESS** | **1 / 10** | `staff` 70.1 % | ❌ NL prior is 2.5 % — implausible 32× inflation |
| 🇫🇮 FI | **USELESS** | **2 / 10** | `rolling stock` 65.9 %, `logistical` 30.4 % | ❌ Weather predicted only 0.3 % despite 27 % of FI stops having severe winter weather |

**Both transfers fail catastrophically due to domain shift + Random-Forest class-balancing pathology.** The numbers in `cause_transfer.json` should not be used for any operational decision.

---

## NL benchmark (the trained-model side)

| Model | val accuracy | val macro-F1 | test accuracy | test macro-F1 |
|---|---|---|---|---|
| LogReg | 0.147 | 0.116 | 0.190 | 0.140 |
| LightGBM | **0.374** | 0.093 | 0.357 | 0.089 |
| XGBoost | **0.382** | 0.089 | 0.376 | 0.110 |
| **RandomForest (winner)** | 0.367 | **0.187** | 0.332 | 0.146 |
| MLP | 0.197 | 0.131 | 0.136 | 0.106 |

The winner was picked by **val macro-F1**. Note the asymmetry: LightGBM and XGBoost have the best **accuracy** (~38 %) but the worst **macro-F1** (~0.09) — they're collapsing onto the majority class (`rolling stock` = 36 %). Random Forest with `class_weight='balanced'` gets a higher macro-F1 by spreading predictions across rare classes — which then becomes the source of the transfer pathology described below.

For context, the NL class distribution in the labelled training set is:

| Class | Count | Share |
|---|---|---|
| rolling stock | 48,747 | 38.5 % |
| infrastructure | 32,956 | 26.0 % |
| external | 17,756 | 14.0 % |
| accidents | 13,958 | 11.0 % |
| unknown | 5,072 | 4.0 % |
| logistical | 4,434 | **3.5 %** |
| engineering work | 3,294 | 2.6 % |
| **staff** | 3,158 | **2.5 %** |
| weather | 1,381 | **1.1 %** |

The two top-predicted classes in the transfer (`staff` for IT, `logistical` for FI) are **rare classes in the training set**. That's the smoking gun.

---

## 🇮🇹 Italy — agent finding (Trust: 1/10)

**Predicted distribution:** staff **70.1 %**, rolling stock 24.2 %, infrastructure 2.8 %, weather 1.9 %, logistical 1.0 %, external 0.1 %.

### Why staff = 70 % is impossible

#### 1. Catastrophic feature distribution shift

| Feature | NL train | IT transfer | Shift |
|---|---|---|---|
| `temperature` | 9.97 °C mean | **0.0 °C always (100 % zero-filled)** | total info loss |
| `degree` (station connectivity) | 52,408 mean | 5.8 mean | **9 000× smaller** |
| `train_station_lag7_rate` | 0.532 | 0.240 | −55 % |
| `month` distribution | balanced Jan–Jun | 329 K stops in Jan, < 3 K in May-Jun | severe skew |

Italy's `temperature` column is **all zeros** in the disrupted-stops parquet — Phase 1 zero-filled missing weather columns instead of preserving NaN (this is the same bug flagged in `docs/REVIEW.md` Bug #2). The cause model trained on real NL temperatures sees Italy's zero-everywhere as "extreme winter conditions" and pushes those rows into staff (the class it learned to associate with anomalous-temperature signatures during training).

#### 2. Random-Forest class-balanced weighting pathology

`RandomForestClassifier(class_weight='balanced')` upweights rare classes at training time. On NL, it produces a higher macro-F1 by lifting per-class recall on rare classes (`staff` got F1=0.421 vs `weather` F1=0.027). On Italy, where the feature distribution is unfamiliar, the upweighted rare-class branch becomes the **default fallback** — the model doesn't have a confident signal so it falls into the path that maximizes balanced loss, which happens to be `staff`.

**Mode collapse evidence (from `transfer_IT_preds.parquet`):**

| Confidence band | N rows | Predicted class breakdown |
|---|---|---|
| Confidence > 0.5 | 96 (0.03 %) | **100 %** staff |
| Confidence 0.3 – 0.5 | 200,000 (58.8 %) | **99.95 %** staff |
| Confidence < 0.2 | 8,000 (2.4 %) | mixed (low-info regime) |

There is **no subset** of IT predictions where the model is informative. Even the highest-confidence predictions are mode-collapsed onto staff.

### Recommendation for Italian rail operator

**Do not deploy.** The predictions are noise dressed up as probability:

1. Collect ~5–10 K Italian-specific labelled disrupted stops with root-cause from operations logs.
2. Retrain locally with the IT-specific class distribution.
3. Until then, fall back to deterministic rules (`if snow_depth > 5cm → weather`, etc.) as a stop-gap.

---

## 🇫🇮 Finland — agent finding (Trust: 2/10)

**Predicted distribution:** rolling stock **65.9 %**, logistical **30.4 %**, staff 1.9 %, infrastructure 1.5 %, weather **0.3 %**, external 0.1 %, accidents ≈ 0 %.

### The smoking gun: weather is 0.3 % despite 27 % of stops being in severe weather

| FI subset | N stops | Weather predicted |
|---|---|---|
| All FI disrupted stops | 865,560 | 0.3 % (2,597) |
| Severity ≥ 3 (severe) | 235,431 (27.2 %) | 0.7 % (1,648) |
| Severity = 4 (extreme) | 276,946 | 0.7 % (1,983) |
| Snow depth ≥ 25 cm | 502,206 | 0.625 % (3,137) |

The Phase 1 figure `step_08_severity.png` shows clearly that `P(disrupted | severity=4) ≈ 0.4` for Finland — i.e., extreme weather is a **major** disruption driver. But the cause model **ignores severity**: the predicted weather rate is the same at severity 0 as at severity 4. **The model has learned that weather is rare in the Netherlands (1.1 % of NL training), and it transferred that rarity verbatim — without recalibrating against the very different Finnish weather regime.**

### Logistical = 30 % is class-balance pathology

`logistical` is the third-rarest NL class (3.5 %). On NL test, the RF achieves F1 = 0.027 on this class — essentially zero discriminative power. Yet the model predicts it for 30 % of FI disruptions.

Same root cause as Italy: when the FI feature space is unfamiliar (sparser network, longer routes, different `degree` distribution, no `train_station_lag7_rate` history), the class-balanced RF defaults to upweighted rare classes. `logistical` and `rolling_stock` end up as the two "fallback" labels with near-equal confidence (0.259 vs 0.263).

### Contradicts known Finnish railway research

Finnish railway disruption studies (Liikennevirasto reports) consistently find:
- **Weather (extreme cold + snow + icing)** is the **primary** cause of Jan–Mar disruptions
- **Infrastructure failures** (frozen points, signal icing) are a strong second
- Rolling-stock and logistical causes are **minor in winter**

The model predicts essentially the **opposite** of this pattern.

### Recommendation for Finnish rail operator

**Do not deploy.** Three options to fix:

| Fix | Effort | Risk |
|---|---|---|
| **Quick (band-aid):** SMOTE oversample weather class 10× during training | 1 hour | Inflates FP weather predictions |
| **Better:** Label a 100–500-row FI ground-truth set, validate that NL cause taxonomy is even meaningful in FI | 1–2 days | Reveals if the taxonomy itself doesn't transfer |
| **Best (deterministic):** Hand-coded rules (`if weather_severity ≥ 3 AND snow_depth ≥ 15 cm: cause=weather`) replacing the ML cause model entirely for FI | 1 day | Less rich than ML but actually correct |

The agent's strong recommendation is the deterministic-rule path: the NL cause taxonomy may not have a meaningful Finnish analogue ("logistical" may not even be a coherent FI cause class), and an interpretable rule is more useful than a black-box probability that's wrong by 180°.

---

## Cross-cutting root causes

Both agents independently identified the same three problems:

### 1. Severe covariate shift between NL ↔ IT/FI
The cause model was trained on NL features that don't transfer:
- NL stations are dense (mean degree 52,408); IT stations are sparse (mean degree 5.8); FI is in between
- NL has Open-Meteo enriched temperatures; IT has zero-filled temperature; FI has FMI observations with very different distribution
- NL service IDs encode date (one service-id per day); FI service IDs are date+train_number (different cardinality)

There's no **domain-adaptation step** in `cause_prediction.py` between training and transfer — it's a naive `predict_proba` call. With features that look completely different, a tree-based classifier will give garbage outputs.

### 2. Random-Forest `class_weight='balanced'` mode collapse under shift
Class balancing inflates rare-class recall on in-distribution data (which is why RF won on NL macro-F1). But on out-of-distribution data, the balanced class weights make rare classes the default fallback when the model is uncertain. Result: the rare class with the strongest mid-tier signal in the trained trees dominates the transfer predictions.

### 3. NL ground-truth bias from RDT incident reporting
Only 32.2 % of NL disrupted stops have a matching RDT disruption record (the join rate). Weather disruptions are the most under-reported class (operators log them less because they're "acts of God"). This makes the model systematically under-predict weather even on NL — and the bias compounds catastrophically on Finland.

---

## Recommended next steps

If the goal is operational cause prediction across IT/FI:

1. **Drop the NL→IT/FI transfer pretence.** It doesn't work and the agents agree on this.
2. **Build per-country cause models.** For IT, even ~5 K hand-labelled rows would beat the current transfer. For FI, focus on infrastructure/weather/rolling_stock — the three classes that meaningfully exist there.
3. **If transfer is required, fix the prerequisites first:**
   - Repair the IT temperature zero-fill (Bug #2 in `docs/REVIEW.md`)
   - Add a `weather_observed` boolean to NL enrichment so missingness is recoverable (Bug #4)
   - Drop `class_weight='balanced'` for the transfer model; use `compute_sample_weight('balanced')` per-batch instead, which is more domain-shift-robust.
4. **Replace the binary-classification accuracy headline metric with macro-F1 or per-class-recall.** A 38 % accuracy that's 99 % "predicted majority class" is worse than useless; the report headlines should reflect that.
5. **Consider abandoning multi-class cause prediction in favour of per-class binary models** (one-vs-rest classifiers, calibrated independently). That avoids the class-balance pathology entirely.

The 5-model NL benchmark itself is **technically correct** — it's the *transfer* that fails. The benchmark numbers are honest and well-reported; the issue is purely the unwarranted optimism that the predictions would be portable across countries.

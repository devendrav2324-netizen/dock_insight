# Evaluation Methodology

_Charter-AI — Task 2: Walk-Forward Out-of-Sample Validation_

---

## 1. Guiding Principle

> **All reported model metrics must be based on out-of-sample predictions.**

A metric is out-of-sample (OOS) if and only if the model that generated the prediction was **not trained on the data point being evaluated**. In-sample metrics (where the model has seen the evaluation data during training) are systematically optimistic and must never be reported as validation results.

---

## 2. Walk-Forward (Expanding Window) Cross-Validation

Implemented in `src/models/model_evaluation.py → walk_forward_cv()`.

### Procedure

```
Dataset (N rows, sorted chronologically by date):

Fold 0:   TRAIN [0 … T₀]             TEST [T₀+1 … T₀+H]
Fold 1:   TRAIN [0 … T₀+S]           TEST [T₀+S+1 … T₀+S+H]
Fold 2:   TRAIN [0 … T₀+2S]          TEST [T₀+2S+1 … T₀+2S+H]
...

Where:
  T₀  = initial_train_size (minimum 180 days by default)
  H   = horizon_days (e.g. 7 days)
  S   = step_size (e.g. 14 days)
```

### Guarantees

| Property | Implementation |
|----------|---------------|
| Chronological sort | `df.sort_values(date_col)` before any split |
| No random shuffle | Splits are index-based on sorted data only |
| No future leakage | Test slice starts at `current_cutoff`, train ends at `current_cutoff - 1` |
| Fresh model per fold | `model_factory()` creates a new instance each fold |
| MASE uses training scale only | `y_train` passed to `calculate_mase()` is the training window |

### Window Types

- **`expanding`** (default): Training window grows with each fold — maximum history.
- **`rolling`**: Fixed-size window slides forward — more suitable for detecting drift.

---

## 3. Ensemble Weight Derivation

Implemented in `src/models/ensemble_forecaster.py → EnsembleForecaster.fit()`.

### Problem with the Prior Approach

The original implementation derived inverse-RMSE weights from `model.evaluate(df_train.tail(30))`. Because the model was already fitted on `df_train`, this was in-sample evaluation — a model that memorized the last 30 training rows would score artificially low RMSE and receive disproportionately high weight.

### OOF Holdout Scheme (Current Implementation)

```
Full training data df_train (N rows, sorted chronologically):
│
├── fit_df:  rows[0 : N-V]    ← models fitted here (Pass 1 only)
└── oof_df:  rows[N-V : N]    ← weights computed from RMSE here

V = min(30, max(14, N // 5))

Weight formula:
  w_m = (1 / OOF_RMSE_m) / Σ_k (1 / OOF_RMSE_k)
```

After weight estimation, **all models are refit on the full `df_train`** (Pass 2) so predictions benefit from the maximum available history.

### Fallback

If `N < 2 * V` (data too short for a meaningful OOF window), all models receive equal weights.

---

## 4. Metric Definitions

All metrics are computed in `src/models/model_evaluation.py`.

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | `mean(|y - ŷ|)` | Linear penalty; interpretable in original units |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penalises large errors more heavily |
| **MAPE** | `mean(|y - ŷ| / |y|) × 100` | Zero-safe: zero actuals excluded |
| **sMAPE** | `mean(2|y - ŷ| / (|y| + |ŷ|)) × 100` | Bounded 0–200%; symmetric |
| **MASE** | `MAE(test) / mean(|diff(y_train)|)` | < 1 means better than naïve 1-step |

### MASE Calculation Detail

```
scale = mean(|y_train[t] - y_train[t-1]|)   ← 1-step naïve MAE on training window only
MASE  = MAE(y_test, ŷ_test) / scale
```

The `scale` is computed **exclusively from the training window**. It is never computed from the test window, which would introduce leakage.

---

## 5. Baseline Models

Two baselines are always computed alongside the main model:

| Baseline | Description | OOS? |
|----------|-------------|------|
| **Naïve Last Value** | `ŷ_{t+h} = y_t` (last known rate) | ✅ Uses only history up to t |
| **Moving Average (7-day SMA)** | `ŷ_{t+h} = mean(y_{t-6:t})` | ✅ Uses only history up to t |

The naïve baseline `last_value` is captured from the **training set only** during `fit()`. It is never updated with test observations.

---

## 6. CongestionPredictor Evaluation

Implemented in `src/models/congestion_predictor.py → CongestionPredictor.fit()`.

### Chronological 80/20 Split

```
All congestion rows (sorted by [port, date]):
│
├── First 80%: fitted (model_p50 pass 1)
└── Last 20%:  OOS evaluation → self.metrics{"MAE", "RMSE"}

Then: all models refit on 100% of data for deployment.
```

The stored `self.metrics` dictionary includes `"_evaluation": "OOS_chronological_holdout"` to make the methodology transparent. Previously, metrics were computed from in-sample residuals on the same `X` used for training — this has been corrected.

---

## 7. FreightForecaster On-the-Fly Evaluation

Implemented in `src/models/freight_forecaster.py → predict_freight()`.

When no pre-trained model artifact exists, a model is trained on-the-fly. The reported `metrics` in the API response now use a chronological OOS holdout:

```
featured_df (all historical data, sorted):
│
├── First N-H rows: model fitted here
└── Last H rows:    OOS evaluation → metrics returned to API caller

H = min(30, max(7, N // 5))

Then: model refit on full featured_df for actual prediction.
```

The metrics dict includes `"_evaluation": "OOS_chronological_holdout"` when this path is taken. If the series is too short (< H + 14 rows), the dict contains `"_evaluation": "insufficient_data_for_OOS"` so callers are not misled.

---

## 8. What Is Still In-Sample (Known Limitations)

| Component | Status | Reason |
|-----------|--------|--------|
| `BaselineForecaster.evaluate()` caller passing train data | Caller responsibility | The `evaluate()` method itself is agnostic — ensure callers pass held-out data |
| ARIMA in-sample residuals (internal to statsmodels) | Not surfaced | AIC/BIC are inherently in-sample; not used for model selection here |
| XGBoost tree importance scores | Not surfaced as metrics | Not used for weight calculation |
| Rolling feature warm-up rows (first ~30 days) | NaN → ffill/bfill | These rows have partially approximated features; MASE uses valid rows only |

---

## 9. Test Coverage

Tests for OOS validation are in `tests/test_oos_validation.py`:

| Test | What it verifies |
|------|----------------|
| `test_walk_forward_chronological_order` | `train_end < test_start` in every fold |
| `test_no_random_shuffle` | Shuffled input is re-sorted, not evaluated in random order |
| `test_oos_predictions_from_unseen_model` | Test rows are strictly after train_end |
| `test_leakage_resistance_future_observation` | Mutating a future row doesn't change earlier fold predictions |
| `test_metrics_from_oos_predictions` | overall_metrics = evaluate_forecast(concatenated fold actuals, preds) |
| `test_naive_baseline_uses_only_history` | `last_value` equals training data's last row, not any future row |
| `test_ensemble_weights_not_derived_from_val_targets` | Perturbing OOF window changes weights; OOF provenance flag present |
| `test_deterministic_walk_forward_split` | 50-row synthetic dataset produces exactly 3 correct folds |

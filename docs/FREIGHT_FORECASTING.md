# DockInsights Freight Forecasting Engine — Technical Specification & User Manual

## 1. Executive Summary

DockInsights Phase 3 implements an end-to-end, mathematically defensible **Dry-Bulk Freight Forecasting Engine** tailored for Indian East Coast import and export corridors.

The engine forecasts spot freight rates (in USD/tonne) across specific dry-bulk commercial dimensions:
$$\text{origin} \longrightarrow \text{destination} \longrightarrow \text{vessel\_class} \longrightarrow \text{cargo\_type}$$
across four standardized commercial horizons:
- **3 days** (Immediate prompt fixture decisions)
- **7 days** (Short-term laycan negotiation)
- **14 days** (Medium-term voyage planning)
- **30 days** (Strategic spot vs. period charter contracting)

---

## 2. Model Architecture & Families

DockInsights deploys **six model families**, balancing simple baselines, parametric statistical time-series models, gradient-boosted decision trees, and multi-model ensembles:

| Model ID | Family | Mathematical Formulation | Prediction Interval Method |
| :--- | :--- | :--- | :--- |
| `naive` | Baseline | $\hat{y}_{t+h} = y_t$ | Residual variance: $\pm 1.645 \cdot \sigma_{res} \sqrt{h}$ |
| `moving_average` | Baseline | $\hat{y}_{t+h} = \frac{1}{W}\sum_{i=0}^{W-1} y_{t-i}$ | Residual variance: $\pm 1.645 \cdot \sigma_{res} \sqrt{1 + \frac{h-1}{W}}$ |
| `seasonal` | Baseline | $\hat{y}_{t+h} = y_{t+h-S}$ ($S=7$ weekly cycle) | Seasonal residual variance |
| `arima` | Statistical | $\Phi(B)(1-B)^d y_t = \Theta(B)\epsilon_t$ | Parametric statsmodels forecast variance |
| `xgboost` | Machine Learning | Gradient Boosted Regression Trees | Quantile regression ($\alpha \in \{0.10, 0.50, 0.90\}$) |
| `ensemble` | Hybrid | $\hat{y}_{ens,t} = \sum_{m} w_m \hat{y}_{m,t}$ | Inverse-RMSE weighted quantile pooling |

> [!IMPORTANT]
> **No Automatic Superiority Assumption**:
> XGBoost is **not** assumed to be automatically superior. In empirical dry-bulk time series, prompt horizons (3–7 days) often favor ARIMA or random walk baselines due to high autocorrelation and market friction, while medium-to-longer horizons (14–30 days) favor XGBoost and Ensembles leveraging exogenous commodity, bunker, and congestion signals.

---

## 3. Time-Series Walk-Forward Cross-Validation

### Why Random Train/Test Splitting is Strictly Prohibited
Random sampling across time series violates temporal ordering, destroys autocorrelation, and causes catastrophic **forward look-ahead leakage** (training on future information to predict the past).

### Walk-Forward Cross-Validation Protocol
DockInsights enforces expanding chronological walk-forward validation:

```
Fold 0: [=== Train Window (Day 0 to 180) ===] -> [ Test (Day 181 to 187) ]
Fold 1: [==== Train Window (Day 0 to 210) ===] -> [ Test (Day 211 to 217) ]
Fold 2: [===== Train Window (Day 0 to 240) ===] -> [ Test (Day 241 to 247) ]
...
```

1. **Expanding Window**: At fold $k$, the model fits strictly on data available up to cutoff date $T_k$.
2. **Multi-Step Forecast**: The model generates an out-of-sample forecast for $T_k + 1 \dots T_k + H$.
3. **Step Roll**: The cutoff advances by a fixed step size (e.g., 30 days).
4. **Out-of-Sample Metrics**: True test observations are recorded against predictions to compute aggregate cross-validation performance.

---

## 4. Evaluation Metrics

All models are evaluated on five complementary metrics:

1. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
3. **Mean Absolute Percentage Error (MAPE, %)**:
   $$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
4. **Symmetric Mean Absolute Percentage Error (sMAPE, %)**:
   $$\text{sMAPE} = \frac{100\%}{n}\sum_{i=1}^n \frac{2 |y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}$$
   *Bounded between 0% and 200%, avoiding asymptotic distortions when rates are near zero.*
5. **Mean Absolute Scaled Error (MASE)**:
   $$\text{MASE} = \frac{\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|}{\frac{1}{N-1}\sum_{t=2}^N |y_t - y_{t-1}|}$$
   *A score $< 1.0$ proves the model outperforms a naive in-sample 1-step forecast.*

---

## 5. Multi-Domain Feature Engineering Catalog

Features are engineered by `FreightFeatureBuilder` in `src/models/forecast_features.py`:

| Domain | Feature Name | Description | Lookback / Formula |
| :--- | :--- | :--- | :--- |
| **Freight Lags** | `lag_1`, `lag_3`, `lag_7`, `lag_14`, `lag_21`, `lag_28` | Historical rate observations | $y_{t-k}$ |
| **Rolling Stats** | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` | Short & medium moving averages | $\frac{1}{W}\sum_{i=1}^W y_{t-i}$ |
| **Rolling Volatility**| `rolling_std_7`, `rolling_std_28` | Short & long-term volatility | Standard deviation of past $W$ days |
| **Momentum** | `momentum_7`, `momentum_30` | Absolute price momentum | $y_t - y_{t-p}$ |
| **Market Indices** | `bdi`, `capesize_index`, `panamax_index`, `supramax_index`, `handysize_index` | Baltic Dry & sub-indices | Baltic Exchange spot values |
| **Index Dynamics** | `bdi_lag_1`, `bdi_momentum_7` | Baltic market momentum | $BDI_t - BDI_{t-7}$ |
| **Commodity Prices**| `coal_price`, `iron_ore_price`, `grain_price` | Global dry-bulk commodities | Newcastle coal, Platts 62% Fe, US Gulf grain |
| **Energy Prices** | `bunker_price`, `mgo_price`, `crude_oil_price` | Marine fuels & Brent crude | VLSFO ($/t), MGO ($/t), Brent ($/bbl) |
| **Macroeconomics** | `usd_inr`, `economic_indicator_pmi` | Currency exchange & manufacturing | RBI USD/INR, S&P Global India PMI |
| **Port Congestion** | `port_congestion`, `vessels_waiting`, `average_waiting_days` | Discharge port operational status | Vessels queueing & berth occupancy |
| **Operational** | `vessel_availability`, `route_disruption` | Fleet supply & disruption indicators | Regional ship availability & weather flags |
| **Calendar** | `month`, `week`, `day_of_week`, `quarter` | Standard calendar time indices | ISO calendar units |
| **Cyclical Season**| `month_sin`, `month_cos`, `day_sin`, `day_cos` | Cyclical trigonometric transforms | $\sin(2\pi m / 12)$, $\cos(2\pi m / 12)$ |
| **Regional Season**| `is_monsoon`, `is_cyclone_season` | Indian Ocean maritime weather | SW Monsoon (Jun–Sep), Cyclone (Apr–May, Oct–Nov) |

---

## 6. Uncertainty Quantification & Quantile Forecasting

DockInsights strictly prohibits arbitrary heuristics (such as `np.std(history) * 0.1`). All prediction intervals are statistically defensible:

### XGBoost Quantile Regression
Trained using `objective="reg:quantileerror"`:
- **P10 Model**: $\alpha = 0.10$ (10th percentile / lower bound)
- **P50 Model**: $\alpha = 0.50$ (50th percentile / median rate)
- **P90 Model**: $\alpha = 0.90$ (90th percentile / upper bound)

Non-crossing monotonic constraints are enforced post-inference:
$$0 \le P10 \le P50 \le P90$$

### ARIMA Parametric Intervals
Statsmodels calculates standard errors of forecast errors $\sigma_h^2$:
$$\text{CI}_{1-\alpha} = \hat{y} \pm z_{\alpha/2} \sigma_h$$

### Ensemble Quantile Pooling
The ensemble synthesizes the component intervals weighted by inverse-RMSE:
$$P10_{ens} = \sum_{m} w_m P10_m, \quad P90_{ens} = \sum_{m} w_m P90_m$$

---

## 7. Model Persistence & Metadata Schema

Trained models are persisted in `models/<origin>_<destination>_<vessel_class>/<model_name>/`:
- `model.pkl`: Serialized model weights and parameters.
- `metadata.json`: Machine-readable provenance and audit metadata.

### Example `metadata.json`
```json
{
  "training_date": "2026-09-10T03:54:59.675256+00:00",
  "dataset_version": "v2.0",
  "route": "AUS_NEW->IND_GVM",
  "origin": "AUS_NEW",
  "destination": "IND_GVM",
  "vessel_class": "Capesize",
  "cargo_type": "thermal_coal",
  "model_name": "xgboost",
  "model_version": "1.0.0",
  "features": [
    "lag_1", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_std_7", "momentum_7", "bdi",
    "capesize_index", "coal_price", "bunker_price", "usd_inr",
    "port_congestion", "is_cyclone_season"
  ],
  "metrics": {
    "MAE": 0.305,
    "RMSE": 0.385,
    "MAPE": 2.11,
    "sMAPE": 2.11,
    "MASE": 1.546
  },
  "horizons": [3, 7, 14, 30]
}
```

---

## 8. CLI Commands

### Train Models
```bash
# Train all models with walk-forward cross-validation for a specific corridor
./venv/bin/python scripts/train_freight_model.py \
    --origin AUS_NEW \
    --destination IND_GVM \
    --vessel-class Capesize \
    --model all

# Train a single model (e.g. XGBoost)
./venv/bin/python scripts/train_freight_model.py \
    --route AUS_NEW_IND_GVM \
    --vessel-class Capesize \
    --model xgboost
```

### Benchmark & Evaluate Models
```bash
# Run walk-forward benchmark across 3, 7, 14, and 30-day horizons
./venv/bin/python scripts/evaluate_freight_models.py \
    --route AUS_NEW_IND_GVM \
    --vessel-class Capesize \
    --horizons 3 7 14 30
```

---

## 9. HTTP REST API Specification

### Endpoint: `GET /api/v1/forecast`

#### Parameters
| Parameter | Type | Required | Description | Example |
| :--- | :--- | :--- | :--- | :--- |
| `origin` | `str` | Yes | Origin port code | `AUS_NEW` |
| `destination`| `str` | Yes | Destination port code | `IND_GVM` |
| `vessel_class`| `str`| Yes | Vessel class | `Capesize` |
| `horizon_days`| `int`| No | Forecast horizon in days (default: 7) | `7` |
| `cargo_type` | `str` | No | Cargo category (default: thermal_coal) | `thermal_coal` |
| `model_type` | `str` | No | Model family override | `ensemble` |

#### Canonical JSON Response
```json
{
  "current_rate": 13.98,
  "forecast_rate": 14.12,
  "lower_bound": 13.68,
  "upper_bound": 14.53,
  "trend": "stable",
  "confidence": 0.97,
  "model_used": "Ensemble_Inverse_Rmse (baseline: 16%, arima: 15%, xgboost: 69%)",
  "metrics": {
    "MAE": 0.273,
    "RMSE": 0.319,
    "sMAPE": 2.00,
    "MASE": 1.294
  },
  "data_scope": "ROUTE_VESSEL",
  "fallback_level": "ROUTE_VESSEL",
  "training_observations": 2192,
  "data_quality": "MEDIUM"
}
```

---

## 10. Historical Data Fallback Hierarchy

To ensure mathematical defensibility, the forecasting engine must train on historical data that is strictly representative of the requested forecast. When the requested exact corridor combination lacks sufficient historical data (minimum 180 observations preferred, 60 minimum for fallback), the engine degrades gracefully through a strict 4-level hierarchy. 

**Under no circumstances will the engine silently pool all global historical records.**

| Fallback Level | Matching Criteria | Data Quality |
| :--- | :--- | :--- |
| **1. EXACT** | `origin` + `destination` + `vessel_class` + `cargo_type` | **HIGH** |
| **2. ROUTE_VESSEL** | `origin` + `destination` + `vessel_class` (Drops cargo constraint) | **MEDIUM** |
| **3. ROUTE** | `origin` + `destination` (Drops vessel class constraint) | **LOW** |
| **4. ROUTE_FAMILY**| Matches `origin` OR `destination` region | **LOW** |
| **INSUFFICIENT** | Less than 60 observations available at level 4 | **Error / Abort** |

The exact fallback level used is explicitly exposed in the API response under `data_scope` / `fallback_level` alongside `training_observations` and `data_quality`. Clients downstream (e.g. the Market Timing Engine) use these fields to scale confidence scores.

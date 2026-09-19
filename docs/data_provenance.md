# Data Provenance, Synthetic-Data Transparency & Data Quality Contract

## 1. Overview

CharterAI requires full transparency regarding data origin, quality, and execution mode across all system layers. This document details the data provenance architecture, dataset inventory, synthetic generator specifications, data quality contract, and machine-readable metadata propagation.

---

## 2. Dataset Inventory & Provenance Status

All demonstration datasets in `data/demo/` are programmatically generated and explicitly labeled under **`SYNTHETIC_DEMO`** data mode.

| Dataset Name | Mode | Source Type | Generator / Source | Date Coverage | Row Count | Reference Context |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `freight_rates` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 21,920 | Emulates dry bulk route freight rates ($/tonne) |
| `dry_bulk_indices` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 10,960 | Emulates Baltic Dry Bulk Index series (BDI, BCI, BPI, BSI, BHSI) |
| `commodity_prices` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 6,576 | Emulates GlobalCoal, Platts, and USDA commodity benchmarks |
| `bunker_prices` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 13,152 | Emulates Bunkerworld VLSFO and MGO fuel price series |
| `congestion` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 21,920 | Emulates port authority queues, berth occupancy, wait times |
| `weather` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 21,920 | Emulates port weather telemetry (wind, wave, rain, cyclone) |
| `economic_indicators` | `SYNTHETIC_DEMO` | `GENERATED` | `scripts/seed_historical_timeseries.py` | 2019-01-01 → 2024-12-31 | 4,384 | Emulates USD/INR FX rates and manufacturing PMI metrics |
| `events` | `SYNTHETIC_DEMO` | `GENERATED` | `data/demo/events.csv` | 2024-01-15 → 2026-01-15 | 6 | Sample disruption events for development testing |
| `ports` | `SYNTHETIC_DEMO` | `GENERATED` | `data/demo/ports.csv` | N/A | 15 | Dry bulk port specifications and bathymetry constraints |
| `routes` | `SYNTHETIC_DEMO` | `GENERATED` | `data/demo/routes.csv` | N/A | 14 | Dry bulk shipping routes and nautical mile distances |
| `vessels` | `SYNTHETIC_DEMO` | `GENERATED` | `data/demo/vessels.csv` | N/A | 10 | Candidate dry bulk fleet specifications |
| `vessel_ais` | `SYNTHETIC_DEMO` | `GENERATED` | `data/demo/vessel_ais.csv` | 2026-01-01 → 2026-01-01 | 6 | Sample AIS vessel telemetry positions |

---

## 3. Provenance Controlled Vocabulary

CharterAI uses controlled enums (`src/data/provenance.py`) to categorize datasets:

- **`DataMode`**:
  - `SYNTHETIC_DEMO`: Programmatically generated or mock data for architecture validation and SIH demonstration.
  - `VERIFIED_EXTERNAL`: Verified real-world observations retrieved from an authenticated external API/feed.
  - `USER_PROVIDED`: User-uploaded scenario or contract input.
  - `DERIVED`: Pipeline-computed metrics (e.g. out-of-sample forecast features).
  - `UNKNOWN`: Unregistered dataset without verified provenance metadata.

- **`SourceType`**:
  - `GENERATED`, `EXTERNAL_API`, `MANUAL_UPLOAD`, `PIPELINE_DERIVED`, `UNVERIFIED`.

- **`QualityStatus`**:
  - `VALID`, `WARNING`, `INVALID`.

---

## 4. Data Quality Validation Contract

Data quality validation (`src/data/validate_data.py` & `src/data/validators/business_validator.py`) enforces:

1. **Required Columns**: All canonical domain columns must be present.
2. **Date Parseability & Coverage**: Dates must be valid ISO format, chronological, and within expected boundaries.
3. **Numeric Finiteness**: No `NaN` or `+/-Inf` values permitted in numeric time-series columns.
4. **Physical Domain Limits**: Non-negative constraints on rates, distances, speeds, DWT, draft, LOA, beam, and prices.
5. **Primary Key Uniqueness**: Duplicate rows are flagged and logged.
6. **Provenance Integrity**: Deceptive claims of external verified sources on synthetic datasets raise a `QualityStatus.INVALID` status.

---

## 5. End-to-End Provenance Propagation

Provenance metadata propagates seamlessly through all layers:

```
[ Dataset (data_mode: SYNTHETIC_DEMO) ]
                 │
                 ▼
[ Feature Engineering (build_forecast_features) ]
                 │
                 ▼
[ ML Model Training (TrainingPipeline) ]
                 │
                 ▼
[ Model Persistence (metadata.json with data_mode: SYNTHETIC_DEMO) ]
                 │
                 ▼
[ Inference (FreightForecaster / predict_freight) ]
                 │
                 ▼
[ Decision Engine & FastAPI (/forecast response with data_mode: SYNTHETIC_DEMO) ]
```

---

## 6. Synthetic Data Generator Methodology

`scripts/seed_historical_timeseries.py` generates 2,192 days (2019-01-01 to 2024-12-31) of continuous, physically coherent dry bulk time-series using:
- **Random Seed**: Fixed `np.random.seed(42)` for deterministic reproducibility.
- **Seasonality & Trends**: Annual sine/cosine wave cycles for Baltic indices and commodity demand.
- **Autoregressive Processes**: Mean-reverting AR(1) random walk noise ($x_t = \phi x_{t-1} + \epsilon_t$).
- **Multi-Variable Coupling**: Route freight rates coupled to bunker prices, index multipliers, and port congestion.

---

## 7. SIH Presentation Safety Statement

> *"The current demonstration environment uses clearly labeled synthetic datasets to validate CharterAI's forecasting, optimization, and risk architecture; the production architecture is designed to accept verified external maritime and market data."*

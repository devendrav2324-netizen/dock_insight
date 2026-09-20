# DockInsights — Project Audit V2

> **Audit Date**: 2026-09-10
> **Auditor**: Automated deep-inspection of full codebase
> **Verdict**: System is operational in SIH Demo Mode with synthetic data.
> All 60 tests pass. Core decision pipeline works end-to-end.
> Several modules are scaffolds/stubs. Significant architectural duplications exist.

---

## 1. Current Architecture

```
dock-insights/
├── src/
│   ├── api/                  # FastAPI application layer
│   │   ├── main.py           # App factory, CORS, router registration
│   │   ├── deps.py           # Dependency injection (DB sessions)
│   │   ├── serializers.py    # Pydantic request/response models
│   │   └── routes/           # 9 route modules
│   ├── data/                 # Data layer
│   │   ├── db.py             # SQLAlchemy async/sync engine
│   │   ├── models.py         # 10 ORM models (PostgreSQL + PostGIS)
│   │   ├── schemas.py        # Pydantic CSV ingestion schemas
│   │   ├── repository.py     # Async data access repositories
│   │   ├── mock_db.py        # Synthetic demo data
│   │   ├── ingestion.py      # CSV → PostgreSQL pipeline
│   │   ├── load_data.py      # Generic CSV loading
│   │   ├── preprocess.py     # Type conversion & string cleaning
│   │   ├── validate_data.py  # Column/missing/duplicate checks
│   │   └── feature_engineering.py  # ML feature creation
│   ├── models/               # ML / Forecasting
│   │   ├── base_forecaster.py        # ABC interface (ForecastModel)
│   │   ├── freight_forecaster.py     # DUPLICATE ABC + XGBoost/SARIMA stubs
│   │   ├── xgboost_forecaster.py     # Working XGBoost forecaster
│   │   ├── arima_forecaster.py       # Working ARIMA forecaster
│   │   ├── baseline_forecaster.py    # Moving-average baseline
│   │   ├── features.py              # Feature pipeline (lag/rolling/calendar)
│   │   ├── model_evaluation.py       # MAE/RMSE/MAPE metrics
│   │   ├── idle_time_predictor.py    # STUB — NotImplemented
│   │   ├── market_timing.py          # STUB — NotImplemented
│   │   ├── registry.py              # STUB — NotImplemented
│   │   └── training.py              # STUB — NotImplemented
│   ├── economics/            # Voyage cost calculation
│   │   ├── voyage_cost.py           # Simple formula-based calculator
│   │   ├── voyage_calculator.py     # Full calculator (bunker+demurrage+port)
│   │   ├── bunker_estimator.py      # Fuel cost estimation
│   │   ├── demurrage_calculator.py  # Demurrage/despatch calculation
│   │   └── cost_models.py          # Port cost lookup tables
│   ├── optimization/         # Core optimization engines
│   │   ├── vessel_selector.py       # VesselSelector + VesselOptimizer
│   │   ├── port_compatibility.py    # Deep port compatibility check
│   │   ├── contract_optimizer.py    # Contract strategy rules engine
│   │   ├── charter_scorer.py        # Multi-criteria charter scorer
│   │   └── decision_engine.py       # Central orchestrator (6 engines)
│   ├── risk/                 # Risk assessment
│   │   ├── risk_engine.py           # Standalone MaritimeRiskEngine
│   │   ├── aggregator.py            # Weighted risk aggregation
│   │   ├── market_risk.py           # Statistical market risk
│   │   ├── port_risk.py             # Port congestion risk
│   │   ├── weather_risk.py          # Seasonal/cyclone risk
│   │   └── operational_risk.py      # Route/chokepoint risk
│   └── utils/                # Shared utilities
│       ├── config.py                # Pydantic Settings (.env)
│       ├── constants.py             # Enums, weights, seasons
│       ├── geo.py                   # Haversine, sailing-day estimation
│       └── logging.py              # Structured JSON logging
├── frontend/                 # React + Vite + TypeScript
│   └── src/
│       ├── App.tsx                  # Layout, routing, sidebar
│       ├── api.ts                   # Axios client (2 endpoints)
│       ├── pages/
│       │   └── VoyagePlanner.tsx    # Main demo page (282 lines)
│       ├── index.css                # Full design system
│       └── App.css                  # Component styles
├── tests/                    # 12 test files, 60 tests
├── data/raw/                 # 9 synthetic CSV files (tiny)
├── scripts/run_simulation.py # Backtest simulation
├── notebooks/                # 2 Jupyter notebooks
├── docs/                     # 7 documentation files
└── trained_models/           # Empty directory
```

### Runtime Architecture

| Layer | Technology | Status |
|-------|-----------|--------|
| Frontend | React 19 + Vite + TypeScript | **Operational** — only VoyagePlanner implemented |
| Backend | FastAPI 0.115 + Uvicorn | **Operational** with SIH Demo Mode |
| Database | PostgreSQL + PostGIS (designed) | **NOT CONNECTED** — runs entirely on mock_db.py |
| ML | XGBoost, ARIMA, statsmodels | **Partially implemented** — 3 forecasters work in isolation |
| Orchestration | DecisionEngine | **Operational** — uses hard-coded forecast values |

---

## 2. Existing Modules and Responsibilities

### Backend Modules

| Module | File | Purpose | Status |
|--------|------|---------|--------|
| **DecisionEngine** | `decision_engine.py` | Orchestrates 6 sub-engines into unified recommendation | ✅ Working — but uses hard-coded inputs |
| **VesselSelector** | `vessel_selector.py` | Constraint-based vessel filtering (draft/LOA/beam/DWT) | ✅ Fully working |
| **VesselOptimizer** | `vessel_selector.py` | Multi-criteria scoring (cargo, port, econ, ops, risk) | ✅ Working — uses heuristic economics |
| **PortCompatibility** | `port_compatibility.py` | Deep physical+operational vessel-port check | ✅ Fully working |
| **VoyageCost (simple)** | `voyage_cost.py` | Formula-based total voyage cost | ✅ Working |
| **VoyageCalculator** | `voyage_calculator.py` | Full calculator with bunker/demurrage/port detail | ✅ Working |
| **BunkerEstimator** | `bunker_estimator.py` | Fuel consumption × days × price | ✅ Working |
| **DemurrageCalculator** | `demurrage_calculator.py` | Laytime/demurrage/despatch formulas | ✅ Working |
| **CostModels** | `cost_models.py` | Port cost lookup (6 Indian ports) | ✅ Working — hard-coded values |
| **ContractOptimizer** | `contract_optimizer.py` | Rule-based strategy (Spot/Term/Hybrid) | ✅ Fully working |
| **CharterScorer** | `charter_scorer.py` | Multi-criteria option ranker | ✅ Working — **UNUSED** by DecisionEngine |
| **MaritimeRiskEngine** | `risk_engine.py` | Simplified 6-dimension risk | ✅ Working — independent from modular risk |
| **RiskAggregator** | `aggregator.py` | Weighted risk combination | ✅ Working |
| **MarketRisk** | `market_risk.py` | Volatility/trend scoring | ✅ Working (requires real data) |
| **PortRisk** | `port_risk.py` | Congestion scoring | ✅ Working |
| **WeatherRisk** | `weather_risk.py` | Cyclone/monsoon/wind scoring | ✅ Working |
| **OperationalRisk** | `operational_risk.py` | Route/chokepoint scoring | ✅ Working |
| **BaselineForecaster** | `baseline_forecaster.py` | Moving-average naive model | ✅ Working |
| **ARIMAForecaster** | `arima_forecaster.py` | ARIMA(1,1,1) univariate | ✅ Working |
| **XGBoostForecaster** | `xgboost_forecaster.py` | XGBoost with lag features | ✅ Working |
| **IdleTimePredictor** | `idle_time_predictor.py` | Idle time ML model | ❌ STUB — NotImplementedError |
| **MarketTimingDetector** | `market_timing.py` | Regime detection / trough finding | ❌ STUB — NotImplementedError |
| **ModelRegistry** | `registry.py` | Model version tracking | ❌ STUB — NotImplementedError |
| **TrainingPipeline** | `training.py` | End-to-end training orchestration | ❌ STUB — NotImplementedError |
| **EnsembleForecaster** | `freight_forecaster.py` | XGBoost+SARIMA ensemble | ❌ STUB — NotImplementedError |

---

## 3. Existing APIs

| Method | Endpoint | Status | Notes |
|--------|----------|--------|-------|
| `GET` | `/api/v1/health` | ✅ Working | Returns version, demo mode, DB status |
| `GET` | `/api/v1/ports` | ⚠️ **501 likely** | Requires DB connection (not in demo mode) |
| `GET` | `/api/v1/ports/{port_id}` | ⚠️ Same | Requires DB |
| `GET` | `/api/v1/ports/{port_id}/congestion` | ⚠️ Same | Requires DB |
| `GET` | `/api/v1/routes` | ⚠️ Same | Requires DB |
| `GET` | `/api/v1/vessels` | ⚠️ Same | Requires DB |
| `POST` | `/api/v1/vessels/compatibility` | ⚠️ Same | Requires DB |
| `GET` | `/api/v1/forecast` | ❌ **501** | Explicitly returns NotImplemented error |
| `POST` | `/api/v1/recommend` | ❌ **501** | Explicitly returns NotImplemented error |
| `POST` | `/api/v1/economics` | ✅ Working | Full voyage economics calculation |
| `GET` | `/api/v1/risk` | ✅ Working | Weather + operational risk (no market/port) |
| `POST` | `/api/v1/analyze-voyage` | ✅ Working | **Primary endpoint** — full DecisionEngine |

### Observation
Only 3 of 11 endpoints are fully functional. The `/analyze-voyage` endpoint bypasses the DB entirely via `mock_db.py`. The remaining DB-dependent endpoints will fail without PostgreSQL.

---

## 4. Existing Database Models (ORM)

| Model | Table | Status | Notes |
|-------|-------|--------|-------|
| `Port` | `ports` | Defined | Includes PostGIS geometry column |
| `VesselClassModel` | `vessel_classes` | Defined | |
| `Route` | `routes` | Defined | |
| `FreightRate` | `freight_rates` | Defined | Time-series indexed |
| `Congestion` | `congestion` | Defined | |
| `Weather` | `weather` | Defined | |
| `Commodity` | `commodities` | Defined | |
| `EconomicIndicator` | `economic_indicators` | Defined | |
| `Event` | `events` | Defined | |
| `ModelRegistryEntry` | `model_registry` | Defined | |
| `RecommendationLog` | `recommendation_logs` | Defined | |

**None of these tables are populated** — the application runs entirely on `mock_db.py` synthetic data.

---

## 5. Existing ML Models

| Model | Implementation | Trained? | Connected to API? |
|-------|---------------|----------|-------------------|
| `BaselineForecaster` | `baseline_forecaster.py` — Moving Average | Can train on synthetic data | ❌ Not connected |
| `ARIMAForecaster` | `arima_forecaster.py` — statsmodels ARIMA(1,1,1) | Can train on synthetic data | ❌ Not connected |
| `XGBoostForecaster` | `xgboost_forecaster.py` — XGBoost with lag features | Can train on synthetic data | ❌ Not connected |
| `XGBoostForecaster (stub)` | `freight_forecaster.py` — Duplicate interface | ❌ Raises NotImplementedError | ❌ Dead code |
| `SARIMAForecaster (stub)` | `freight_forecaster.py` — Stub only | ❌ Raises NotImplementedError | ❌ Dead code |
| `EnsembleForecaster (stub)` | `freight_forecaster.py` — Stub only | ❌ Raises NotImplementedError | ❌ Dead code |
| `IdleTimePredictor (stub)` | `idle_time_predictor.py` — Stub only | ❌ Raises NotImplementedError | ❌ Dead code |
| `MarketTimingDetector (stub)` | `market_timing.py` — Stub only | ❌ Raises NotImplementedError | ❌ Dead code |

**Critical finding**: The `DecisionEngine` hard-codes `forecast_rate = 22.50` and does NOT use any ML model. The 3 working forecasters (Baseline, ARIMA, XGBoost) have no path from training → API serving.

---

## 6. Existing Optimization Logic

### DecisionEngine Pipeline (6 steps):
1. **Freight Forecast** → ❌ HARD-CODED: `forecast_rate = 22.50`, `direction = RISING`, `uncertainty = MODERATE`
2. **Vessel Selection** → ✅ Real: `VesselOptimizer.optimize()` with multi-criteria scoring
3. **Port Compatibility** → ✅ Real: `check_vessel_port_compatibility()` draft/LOA/beam check
4. **Voyage Economics** → ⚠️ PARTIALLY HARD-CODED: uses real calculator but with placeholder inputs:
   - `route_distance_nm = 4500.0` (placeholder)
   - `vessel_speed_knots = 12.5` (fixed)
   - `fuel_price_usd_per_t = 600.0` (fixed)
   - `positioning_distance_nm = 500.0` (fixed)
5. **Risk Engine** → ⚠️ PARTIALLY HARD-CODED: uses `MaritimeRiskEngine` but with synthetic `risk_inputs` dict
6. **Contract Strategy** → ✅ Real: deterministic rule engine, receives actual risk/cost values

### Scoring Systems:
- **VesselOptimizer**: Weights `cargo(0.35), econ(0.35), ops(0.20), risk(0.10)` — economic score uses heuristic cost multipliers, risk score is fixed at `80.0`
- **CharterScorer**: Weights `cost(0.40), risk(0.25), reliability(0.20), flexibility(0.15)` — **UNUSED** by any API endpoint

---

## 7. Hard-Coded / Demo Values

| Location | Value | Impact |
|----------|-------|--------|
| `decision_engine.py:57` | `forecast_rate = 22.50` | All recommendations use same forecast |
| `decision_engine.py:58` | `forecast_trend = RISING` | Contract strategy always sees rising market |
| `decision_engine.py:77-78` | `PortConstraints(..., 25.0, 350.0, 60.0, 200000)` | Default port has no real restrictions |
| `decision_engine.py:143` | `route_distance_nm = 4500.0` | Every voyage uses same distance |
| `decision_engine.py:144` | `vessel_speed_knots = 12.5` | No per-class speed variation |
| `decision_engine.py:146` | `fuel_price_usd_per_t = 600.0` | Static fuel price |
| `decision_engine.py:159-165` | `risk_inputs = {...}` | All risk assessments use identical synthetic inputs |
| `decision_engine.py:174` | `current_freight_rate = 21.0` | Contract optimizer sees fixed current rate |
| `vessel_selector.py:302` | `est_voyage_days = 14` | All vessels use same voyage estimate |
| `vessel_selector.py:312-316` | `base_cost_per_tonne = 30.0` with fixed multipliers | Economic score unrelated to real voyage cost |
| `vessel_selector.py:330` | `risk_score = 80.0` | Identical risk for all vessels |
| `mock_db.py:21-29` | `get_mock_port_info()` → `max_draft_m=20.0, max_loa_m=350.0` | All ports have identical generous constraints |
| `cost_models.py:33-93` | 6 port cost lookups | Illustrative, not real tariffs |
| `run_simulation.py:109-110` | Delivery dates `{month}-01` to `{month}-15` | Only 14 days → will fail for early months |

---

## 8. TODO / NotImplemented Areas

| File | Line | TODO/Stub Description |
|------|------|-----------------------|
| `freight_forecaster.py:98` | `XGBoostForecaster.fit()` | Raises NotImplementedError (duplicate class) |
| `freight_forecaster.py:104` | `XGBoostForecaster.predict()` | Raises NotImplementedError |
| `freight_forecaster.py:129` | `SARIMAForecaster.fit()` | Raises NotImplementedError |
| `freight_forecaster.py:134` | `SARIMAForecaster.predict()` | Raises NotImplementedError |
| `freight_forecaster.py:161` | `EnsembleForecaster.fit()` | Raises NotImplementedError |
| `freight_forecaster.py:183` | `EnsembleForecaster.predict()` | Raises NotImplementedError |
| `idle_time_predictor.py:60` | `IdleTimePredictor.fit()` | Raises NotImplementedError |
| `idle_time_predictor.py:80` | `IdleTimePredictor.predict()` | Raises NotImplementedError |
| `market_timing.py:68` | `detect_regime()` | Raises NotImplementedError |
| `market_timing.py:86` | `recommend_timing()` | Raises NotImplementedError |
| `market_timing.py:106` | `find_troughs()` | Raises NotImplementedError |
| `registry.py:65` | `ModelRegistry.save_model()` | Raises NotImplementedError |
| `registry.py:85` | `ModelRegistry.load_model()` | Raises NotImplementedError |
| `registry.py:97` | `get_active_version()` | Raises NotImplementedError |
| `training.py:74` | `TrainingPipeline.run()` | Raises NotImplementedError |
| `training.py:86` | `_time_series_split()` | Raises NotImplementedError |
| `market_risk.py:76-88` | Coal price + BDI momentum scoring | Documented as TODO placeholder |
| `recommend.py:33` | Full recommendation endpoint | Returns HTTP 501 |
| `forecast.py:28` | Forecast endpoint | Returns HTTP 501 |

---

## 9. Duplicate Interfaces / Classes

### Critical Duplications

| Item | Location 1 | Location 2 | Issue |
|------|-----------|-----------|-------|
| **ForecastPoint** | `base_forecaster.py:10-16` | `freight_forecaster.py:24-30` | Two competing dataclasses with different fields |
| **ForecastResult** | `base_forecaster.py:18-26` | `freight_forecaster.py:34-57` | Completely different schemas |
| **ForecastModel ABC** | `base_forecaster.py:28-75` | `freight_forecaster.py:60-81` (as `BaseForecaster`) | Two separate ABCs with incompatible signatures |
| **XGBoostForecaster** | `xgboost_forecaster.py` (working) | `freight_forecaster.py:84-110` (stub) | Same class name, different implementations |
| **VoyageCostBreakdown** | `voyage_cost.py:29-37` | `voyage_calculator.py:22-30` | Different fields, same purpose |
| **Feature engineering** | `src/models/features.py` | `src/data/feature_engineering.py` | Two feature pipelines, different approaches |
| **Risk Engine** | `risk/risk_engine.py` (MaritimeRiskEngine) | `risk/aggregator.py` (RiskAggregator) | Parallel risk systems, DecisionEngine uses `risk_engine.py`, API `/risk` uses `aggregator.py` |

### Unused Modules
- `charter_scorer.py` — fully implemented but never called
- `freight_forecaster.py` — all classes are stubs, shadowed by working implementations
- `repository.py` — all async repos exist but no endpoint uses them (demo mode bypasses DB)

---

## 10. Current Test Status

```
======================== 60 passed, 2 warnings in 2.25s ========================
```

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_api.py` | 4 | ✅ All pass |
| `test_contract_optimizer.py` | 4 | ✅ All pass |
| `test_data_pipeline.py` | 9 | ✅ All pass |
| `test_decision_engine.py` | 2 | ✅ All pass |
| `test_forecasting.py` | 5 | ✅ All pass |
| `test_port_compatibility.py` | 5 | ✅ All pass |
| `test_risk.py` | 8 | ✅ All pass |
| `test_risk_engine.py` | 2 | ✅ All pass |
| `test_vessel_selector.py` | 7 | ✅ All pass |
| `test_voyage_calculator.py` | 7 | ✅ All pass |
| `test_voyage_cost.py` | 2 | ✅ All pass |

### Coverage Gaps:
- No tests for `forecast` or `recommend` API endpoints (they return 501)
- No integration test for DB-backed routes (ports, routes, vessels)
- No frontend tests
- No test for `CharterScorer` (unused module)
- No test for `run_simulation.py`

---

## 11. Broken Dependencies

| Issue | Impact | Severity |
|-------|--------|----------|
| `geoalchemy2` imported in `models.py` but no PostgreSQL + PostGIS available | ORM models cannot be used without PostGIS | ⚠️ Medium |
| `asyncpg` required for async DB but no PostgreSQL running | DB-dependent endpoints will fail | ⚠️ Medium |
| `psycopg2-binary` required for sync ingestion scripts | Ingestion pipeline unusable | ⚠️ Medium |
| `main.py:58-60` — lazy import comment (`analyze_voyage`) | Code smell but not broken | ℹ️ Low |
| `config.py` — `get_settings()` creates new instance on every call | No singleton caching (`@lru_cache` missing) | ⚠️ Medium |
| `shap` in requirements.txt but never imported | Unused dependency | ℹ️ Low |

---

## 12. Data Currently Available

| Dataset | File | Rows | Content |
|---------|------|------|---------|
| `freight_rates.csv` | `data/raw/` | ~10 | SYNTHETIC_DEMO_DATA — 10 fake rates |
| `vessels.csv` | `data/raw/` | ~10 | SYNTHETIC_DEMO_DATA — 10 fake vessels |
| `ports.csv` | `data/raw/` | ~11 | SYNTHETIC_DEMO_DATA — 11 fake ports |
| `routes.csv` | `data/raw/` | ~6 | SYNTHETIC_DEMO_DATA — 6 fake routes |
| `commodities.csv` | `data/raw/` | ~4 | SYNTHETIC_DEMO_DATA — 4 entries |
| `congestion.csv` | `data/raw/` | ~9 | SYNTHETIC_DEMO_DATA |
| `weather.csv` | `data/raw/` | ~9 | SYNTHETIC_DEMO_DATA |
| `economic_indicators.csv` | `data/raw/` | ~9 | SYNTHETIC_DEMO_DATA |
| `events.csv` | `data/raw/` | ~4 | SYNTHETIC_DEMO_DATA |

**CSV column schemas do NOT match the ORM model schemas** — the CSV headers (e.g. `port`, `vessel_type`, `freight_rate`) differ from the ORM column names (e.g. `port_id`, `vessel_class`, `freight_rate_usd_per_day`). The ingestion pipeline in `ingestion.py` uses the ORM schema, not the CSV schema.

---

## 13. Missing Data

| Required Data | Current State | Impact |
|--------------|---------------|--------|
| Real historical freight rates (multi-year) | ❌ Only 10 synthetic rows | Cannot train meaningful forecasting models |
| Real port specifications (Indian East Coast) | ❌ Generic constraints in mock_db | All ports appear identical |
| Real route distances | ❌ Hard-coded 4500nm | All voyages appear identical |
| Real-time congestion feeds | ❌ 9 synthetic rows | Port risk is untested |
| Real weather/cyclone data | ❌ 9 synthetic rows | Weather risk uses calendar heuristics only |
| Bunker fuel price feeds (VLSFO) | ❌ Hard-coded $600/t | No dynamic fuel pricing |
| Economic indicators (BDI, coal prices, FX) | ❌ 9 synthetic rows | Market features unavailable |
| Individual vessel data (AIS, specific ships) | ❌ Only class-level data | Cannot recommend specific vessels |
| Port tariff schedules | ⚠️ Hard-coded for 6 ports | Illustrative only |

---

## 14. High-Risk Architectural Issues

### 🔴 CRITICAL

1. **DecisionEngine uses hard-coded forecasts** — The entire recommendation is driven by `forecast_rate = 22.50` and `forecast_trend = RISING`. Every query to `/analyze-voyage` returns the same market view regardless of route, date, or conditions.

2. **Two incompatible forecasting architectures** — `base_forecaster.py` defines one ABC, `freight_forecaster.py` defines another. The working models (`xgboost_forecaster.py`, `arima_forecaster.py`) use the first. The ensemble stub uses the second. This split will cause integration failures.

3. **Risk engine duplication** — The `DecisionEngine` uses `MaritimeRiskEngine` (simplified, dict-based input), while the `/risk` API endpoint uses the modular `RiskAggregator` with `WeatherRiskAssessor` + `OperationalRiskAssessor`. These two risk systems can produce different results for the same scenario.

### 🟡 HIGH

4. **No data path from forecasters to API** — The 3 working forecasters can train on DataFrames but have no integration with the API. There is no model-serving layer, no trained model files, no model loading in the API.

5. **Voyage economics hard-coded in DecisionEngine** — Route distance, fuel price, speed, port costs, and positioning distance are all constants. The VoyageCalculator (which can compute real values) is unused by the DecisionEngine.

6. **CSV-to-ORM schema mismatch** — The raw CSV files have completely different column names than what `ingestion.py` expects. Ingestion will fail on the existing CSVs.

### 🟠 MEDIUM

7. **Settings not cached** — `get_settings()` creates a new `Settings()` instance on every call, re-reading `.env` each time.

8. **Frontend only has 1 functional page** — 6 of 7 navigation items render placeholder `<div>` tags.

9. **Port data is uniform** — `mock_db.py` returns identical port specs for every port. In reality, Indian East Coast ports have dramatically different constraints (e.g., Gopalpur max draft 12.5m vs Gangavaram 21.0m).

---

## 15. Recommended Migration Path to DockInsights V2

### Phase 1: Foundation (Consolidation)
- Eliminate duplicate forecasting ABCs — adopt `base_forecaster.py` as canonical
- Unify risk engine — make `DecisionEngine` use the modular risk assessors
- Wire VoyageCalculator into DecisionEngine (replace `voyage_cost.py` simple calculator)
- Fix Settings caching
- Fix CSV schemas to match ORM

### Phase 2: Data (Enable Real Operation)
- Generate realistic synthetic data (1000+ freight rate records across routes/vessel classes)
- Populate per-port constraints in mock_db with real Indian East Coast data
- Add real route distances (from routes table)
- Wire forecasters into DecisionEngine (replacing hard-coded values)

### Phase 3: ML Pipeline (Model Serving)
- Implement model registry `save/load`
- Create training script that produces serialized models
- Build model-serving middleware for API
- Implement IdleTimePredictor with XGBoost on congestion data
- Implement MarketTimingDetector

### Phase 4: Frontend (Full Dashboard)
- Implement Dashboard page with summary metrics
- Implement Forecast page with chart visualization
- Implement Port Intelligence with map
- Implement Risk Center with dimension breakdown
- Implement Contract Strategy comparison view

### Phase 5: Production Hardening
- Connect PostgreSQL database
- Implement Alembic migrations
- Add authentication/authorization
- Add rate limiting
- Implement recommendation logging
- Add SHAP-based feature importance to XAI

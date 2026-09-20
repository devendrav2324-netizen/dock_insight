# DockInsights — Migration Plan (V1 → V2)

> **Created**: 2026-09-10
> **Scope**: Every planned change classified as KEEP / MODIFY / REPLACE / NEW
> **Constraint**: No working functionality deleted. No large refactor yet.

---

## Classification Key

| Label | Meaning |
|-------|---------|
| **KEEP** | Module is correct as-is. No changes needed. |
| **MODIFY** | Module works but needs targeted improvements. |
| **REPLACE** | Module has a working replacement or is dead code. Will be superseded. |
| **NEW** | Module does not exist yet and must be created. |

---

## 1. Utilities (`src/utils/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `constants.py` | **KEEP** | Well-structured enums, weights, and seasonal calendars. No issues. |
| `geo.py` | **KEEP** | Correct haversine and sailing-day calculation. No changes needed. |
| `logging.py` | **KEEP** | Clean structured logging. Works correctly. |
| `config.py` | **MODIFY** | Add `@lru_cache` to `get_settings()` to avoid re-reading `.env` on every call. Add route-distance lookup config. |

---

## 2. Data Layer (`src/data/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `db.py` | **KEEP** | Async/sync engine design is correct. No changes needed until PostgreSQL is connected. |
| `models.py` | **MODIFY** | ORM models are well-designed. Need to verify PostGIS import doesn't crash without PostgreSQL extension. Consider lazy import. |
| `schemas.py` | **MODIFY** | Pydantic schemas are sound. Must update `max_dwt` alias and align with actual CSV column names. |
| `repository.py` | **KEEP** | Clean async repository pattern. Will be used when DB is connected. |
| `mock_db.py` | **MODIFY** | Expand with per-port differentiated constraints (real Indian East Coast specs). Add route distance lookup. Add per-vessel-class speed/fuel values. |
| `ingestion.py` | **MODIFY** | Fix column name mapping between CSVs and ORM. Add remaining table ingestors (congestion, weather, etc.). |
| `load_data.py` | **KEEP** | Generic CSV loader works correctly. |
| `preprocess.py` | **KEEP** | Type conversion and string cleaning utilities are correct. |
| `validate_data.py` | **KEEP** | Validation utilities work correctly. |
| `feature_engineering.py` | **KEEP** | Chronological splitting and feature creation are correct with proper data-leakage prevention. |

---

## 3. ML / Forecasting (`src/models/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `base_forecaster.py` | **KEEP** | Canonical ABC for all forecasters. Well-designed interface. |
| `freight_forecaster.py` | **REPLACE** | Contains duplicate, incompatible ABC (`BaseForecaster`) and 3 stub classes (`XGBoostForecaster`, `SARIMAForecaster`, `EnsembleForecaster`) that shadow the working implementations. Should be removed or consolidated. |
| `xgboost_forecaster.py` | **KEEP** | Working XGBoost forecaster implementing `ForecastModel` ABC. Correct recursive prediction. |
| `arima_forecaster.py` | **KEEP** | Working ARIMA forecaster with native confidence intervals. |
| `baseline_forecaster.py` | **KEEP** | Correct moving-average baseline. Needed for backtesting comparison. |
| `model_evaluation.py` | **KEEP** | Clean MAE/RMSE/MAPE calculation. `compare_models` utility is useful. |
| `features.py` | **MODIFY** | Good feature pipeline but partially overlaps with `data/feature_engineering.py`. Consolidate: keep this file as the ML-specific feature pipeline, use `data/feature_engineering.py` for generic data prep. |
| `idle_time_predictor.py` | **MODIFY** | Stub → implement with XGBoost regression on congestion + weather + seasonal features. |
| `market_timing.py` | **MODIFY** | Stub → implement regime detection using rolling z-scores and gradient analysis. |
| `registry.py` | **MODIFY** | Stub → implement joblib serialization, metadata.json, version tracking. |
| `training.py` | **MODIFY** | Stub → implement with TimeSeriesSplit cross-validation and evaluation. |

### NEW modules needed:
| Module | Purpose |
|--------|---------|
| `model_server.py` | **NEW** — Load trained models at API startup. Provide `get_forecaster()` dependency. |
| `ensemble_forecaster.py` | **NEW** — Proper ensemble combining XGBoost + ARIMA with horizon-dependent weights (replaces stub in `freight_forecaster.py`). |

---

## 4. Economics (`src/economics/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `voyage_cost.py` | **REPLACE** | Simple standalone calculator duplicates `voyage_calculator.py`. The `VoyageCalculator` is more complete (bunker, demurrage, port charges). The `DecisionEngine` should use `VoyageCalculator` instead. Keep `voyage_cost.py` only for backwards-compatible test use. |
| `voyage_calculator.py` | **MODIFY** | Wire into DecisionEngine. Accept dynamic route distance and fuel price instead of relying on hardcoded values passed from the engine. |
| `bunker_estimator.py` | **KEEP** | Correct fuel consumption model. |
| `demurrage_calculator.py` | **KEEP** | Correct laytime/demurrage/despatch calculation. |
| `cost_models.py` | **MODIFY** | Add remaining origin port costs (Indonesian, Australian ports). Mark all values as `[DEMO]` with source attribution. |

---

## 5. Optimization (`src/optimization/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `vessel_selector.py` | **MODIFY** | `VesselOptimizer` economic score and risk score are heuristic placeholders. Wire real VoyageCalculator economics and risk engine scores instead of fixed multipliers. |
| `port_compatibility.py` | **KEEP** | Clean physical + operational compatibility check. Works correctly. |
| `contract_optimizer.py` | **KEEP** | Well-designed deterministic rule engine. All branches tested. |
| `charter_scorer.py` | **MODIFY** | Currently UNUSED. Either wire into the DecisionEngine as a final ranking pass, or remove. Recommend wiring it in. |
| `decision_engine.py` | **MODIFY** | **Priority target**. Replace hard-coded forecast values with actual model calls. Replace hard-coded voyage economics inputs with real route/fuel data. Replace synthetic risk inputs with modular risk assessors. Pass real port constraints from mock_db per port_id. |

---

## 6. Risk (`src/risk/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `risk_engine.py` | **REPLACE** | Simplified standalone engine with dict-based inputs. The modular assessors (`market_risk.py`, `port_risk.py`, `weather_risk.py`, `operational_risk.py`) + `aggregator.py` provide the same functionality with better structure. DecisionEngine should switch to the modular system. |
| `aggregator.py` | **KEEP** | Correct weighted aggregation with renormalization for missing dimensions. |
| `market_risk.py` | **KEEP** | Statistical volatility + trend scoring. Ready for real data. |
| `port_risk.py` | **KEEP** | Berth occupancy + waiting + queue scoring. Ready for real data. |
| `weather_risk.py` | **KEEP** | Seasonal + wind + wave + cyclone scoring. Comprehensive. |
| `operational_risk.py` | **KEEP** | Route region + voyage length + chokepoint scoring. Well-designed. |

---

## 7. API (`src/api/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `main.py` | **MODIFY** | Clean up lazy import comment (line 58-60). Move `analyze_voyage` import to top-level with other routes. |
| `deps.py` | **KEEP** | Standard dependency injection pattern. |
| `serializers.py` | **MODIFY** | Add missing response models for individual modules (port detail, vessel detail). Align `ExplainabilityResponse` with `ExplainabilityReport`. |
| `routes/health.py` | **KEEP** | Works correctly. |
| `routes/ports.py` | **KEEP** | Will work when DB is connected. |
| `routes/routes.py` | **KEEP** | Will work when DB is connected. |
| `routes/vessels.py` | **KEEP** | Will work when DB is connected. |
| `routes/forecast.py` | **MODIFY** | Replace 501 stub with actual model-serving logic using `model_server.py` dependency. |
| `routes/recommend.py` | **MODIFY** | Replace 501 stub with DecisionEngine call (similar to `analyze_voyage.py` but using `RecommendationRequest` schema). |
| `routes/economics.py` | **KEEP** | Fully functional. |
| `routes/risk.py` | **MODIFY** | Add market risk and port risk dimensions (currently only weather + operational). |
| `routes/analyze_voyage.py` | **MODIFY** | Add input validation, default delivery date guard. Consider consolidating with `/recommend`. |

---

## 8. Frontend (`frontend/src/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `App.tsx` | **MODIFY** | Replace placeholder `<div>` elements for 5 routes with actual page components. |
| `api.ts` | **MODIFY** | Add API calls for `/forecast`, `/economics`, `/risk`, `/ports`, `/vessels`. Currently only has 2 endpoints. |
| `pages/VoyagePlanner.tsx` | **MODIFY** | Functional but could benefit from better loading states, error display, and result visualization. |
| `index.css` | **KEEP** | Complete design system with dark theme, glassmorphism. Well done. |
| `App.css` | **KEEP** | Component styles are clean. |

### NEW frontend pages needed:
| Page | Purpose |
|------|---------|
| `pages/Dashboard.tsx` | **NEW** — Summary metrics, system health, recent recommendations |
| `pages/FreightForecast.tsx` | **NEW** — Chart-based forecast visualization with model selection |
| `pages/PortIntelligence.tsx` | **NEW** — Map-based port view with congestion data |
| `pages/RiskCenter.tsx` | **NEW** — Risk dimension breakdown with radar chart |
| `pages/ContractStrategy.tsx` | **NEW** — Strategy comparison view |
| `pages/VesselOptimizer.tsx` | **NEW** — Vessel scoring breakdown table |

---

## 9. Tests (`tests/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `test_api.py` | **MODIFY** | Add tests for forecast, recommend, economics, risk endpoints. |
| `test_contract_optimizer.py` | **KEEP** | Good coverage of strategy branches. |
| `test_data_pipeline.py` | **KEEP** | Comprehensive validation tests. |
| `test_decision_engine.py` | **MODIFY** | Add test for each hard-coded override once real data is wired. |
| `test_forecasting.py` | **KEEP** | Tests all 3 working forecasters. |
| `test_port_compatibility.py` | **KEEP** | Good coverage. |
| `test_risk.py` | **KEEP** | Tests all 4 modular risk assessors + aggregator. |
| `test_risk_engine.py` | **REPLACE** | Tests the standalone `MaritimeRiskEngine` which should be replaced by modular risk system. |
| `test_vessel_selector.py` | **KEEP** | Comprehensive tests for both VesselSelector and VesselOptimizer. |
| `test_voyage_calculator.py` | **KEEP** | Tests bunker, demurrage, and full calculator. |
| `test_voyage_cost.py` | **KEEP** | Tests simple cost calculator. |

### NEW tests needed:
| Test | Purpose |
|------|---------|
| `test_charter_scorer.py` | **NEW** — Test the currently unused CharterScorer module |
| `test_feature_engineering.py` | **NEW** — Test data/feature_engineering.py split and leakage prevention |
| `test_model_server.py` | **NEW** — Test model loading/serving when model_server.py is created |

---

## 10. Infrastructure & Configuration

| File | Classification | Rationale |
|------|---------------|-----------|
| `requirements.txt` | **MODIFY** | Remove `shap` if not used yet (or implement SHAP). Add version pins for reproducibility. |
| `.env.example` | **KEEP** | Comprehensive env template. |
| `Dockerfile.backend` | **KEEP** | Standard Python Docker image. |
| `docker-compose.yml` | **KEEP** | Backend + PostgreSQL + Redis defined. |
| `README.md` | **MODIFY** | Update to reflect actual system state (not aspirational features). |
| `.gitignore` | **KEEP** | Correct exclusions. |

---

## 11. Scripts & Notebooks

| File | Classification | Rationale |
|------|---------------|-----------|
| `scripts/run_simulation.py` | **MODIFY** | Fix delivery date issue (month-01 to month-15 is only 14 days, fails for some months). Fix hard-coded artifact path. |
| `notebooks/01_freight_eda.ipynb` | **KEEP** | EDA notebook for exploration. |
| `notebooks/02_forecasting.ipynb` | **KEEP** | Forecasting demo notebook. |

---

## 12. Documentation (`docs/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| `architecture.md` | **MODIFY** | Update to reflect actual architecture (mock_db vs DB, hard-coded forecasts). |
| `data_dictionary.md` | **MODIFY** | Align with actual CSV and ORM schemas. |
| `ml_methodology.md` | **MODIFY** | Document which models are implemented vs stub. |
| `optimization_methodology.md` | **MODIFY** | Document hard-coded values and heuristics. |
| `api_documentation.md` | **MODIFY** | Document which endpoints actually work vs return 501. |
| `assumptions.md` | **MODIFY** | Add hard-coded assumptions from DecisionEngine. |
| `limitations.md` | **MODIFY** | Add CSV schema mismatch, unified port specs, etc. |
| `PROJECT_AUDIT_V2.md` | **NEW** | This audit document. |
| `MIGRATION_PLAN.md` | **NEW** | This migration plan. |

---

## 13. Data (`data/raw/`)

| File | Classification | Rationale |
|------|---------------|-----------|
| All 9 CSV files | **REPLACE** | Column schemas don't match ORM. Data is too sparse (5-10 rows). Need to generate realistic synthetic datasets with 1000+ records and correct column names. |

---

## Summary Statistics

| Classification | Count |
|---------------|-------|
| **KEEP** | 32 |
| **MODIFY** | 30 |
| **REPLACE** | 5 |
| **NEW** | 11 |

### Priority Order for V2 Implementation

1. **MODIFY** `decision_engine.py` — Remove hard-coded values (highest impact, single file)
2. **MODIFY** `mock_db.py` — Differentiate port specs, add route distances
3. **REPLACE** `freight_forecaster.py` — Eliminate duplicate ABCs
4. **REPLACE** `risk_engine.py` — Switch DecisionEngine to modular risk
5. **MODIFY** `config.py` — Cache settings
6. **NEW** `model_server.py` — Enable forecast serving
7. **MODIFY** Frontend pages — Build remaining 5 pages
8. **REPLACE** data CSVs — Generate proper synthetic data

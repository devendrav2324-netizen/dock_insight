# CharterAI V2 Architecture

## System Overview

CharterAI is built on a clean, layered architecture separating HTTP routing, high-level orchestration, modular services, and low-level ML/Optimization algorithms.

```ascii
                      +-----------------------------+
                      |       Client (React)        |
                      +-------------+---------------+
                                    |
+-----------------------------------v-----------------------------------+
|                            API Layer                                  |
|  (FastAPI /api/v1/*, Request/Response validation, Dependency Inject)  |
+-----------------------------------+-----------------------------------+
                                    |
+-----------------------------------v-----------------------------------+
|                        Decision Engine                                |
|  (Orchestrates calls to services, aggregates results, builds XAI)     |
+-----------------------------------+-----------------------------------+
                                    |
+-----------------------------------v-----------------------------------+
|                          Service Layer                                |
|   +---------------+ +-------------+ +---------------+ +-----------+   |
|   | Freight       | | Vessel      | | Voyage        | | Risk      |   |
|   | Forecast      | | Optimizer   | | Economics     | | Service   |   |
|   +---------------+ +-------------+ +---------------+ +-----------+   |
+-----------+---------------+---------------+---------------+-----------+
            |               |               |               |
+-----------v-------+ +-----v-------+ +-----v-------+ +-----v-----------+
|    ML Models      | | Optimization| | Economics   | | Risk Assessors  |
| - XGBoost         | | - Contracts | | - Bunker    | | - Weather       |
| - ARIMA           | | - Selection | | - Demurrage | | - Market        |
| - Baseline        | | - Ports     | | - Port Costs| | - Operational   |
+-----------+-------+ +-----+-------+ +-----+-------+ +-----+-----------+
            |               |               |               |
+-----------v---------------v---------------v---------------v-----------+
|                         Data Access Layer                             |
|         (Repository Pattern, SQLAlchemy, AsyncPG, PostGIS)            |
+-----------------------------------------------------------------------+
```

## Core Components

### 1. API Layer (`src/api/`)
- Purely handles HTTP, authentication, and validation.
- Routes do not contain business logic.
- Services are injected via `request.app.state` to ensure singletons are used during the application lifecycle.

### 2. Decision Engine (`src/optimization/decision_engine.py`)
- The brain of the application.
- Receives a parsed `DecisionEngineInputs` object.
- Calls the 5 core services in sequence: Forecast -> Vessel Optimization -> Voyage Economics -> Risk -> Contract Strategy.
- Constructs the final `ExplainabilityReport` (XAI).

### 3. Service Layer (`src/services/`)
- Encapsulates complex ML models and rule engines into simple interfaces.
- Handles default configurations (from `config.py`) when real data or models are unavailable (e.g., in Demo Mode).
- Standardizes output schemas before they reach the Decision Engine.

### 4. Machine Learning (`src/models/`)
- All forecasters implement the canonical `ForecastModel` ABC and return `ForecastResult`.
- Isolated from API and Services, focusing purely on data frames and math.

### 5. Risk & Economics (`src/risk/`, `src/economics/`)
- Modular calculators that take highly specific inputs and return scores or dollar values.
- Independent of database state; completely deterministic.

### 6. Data Access (`src/data/`)
- Separated into DB connection (`db.py`), ORM models (`models.py`), schemas (`schemas.py`), and repositories (`repository.py`).
- Synthetic data generator (`mock_db.py`) used when `sih_demo_mode=True`.

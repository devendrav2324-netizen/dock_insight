# DockInsights 🚢

An Advanced Bulk Vessel Chartering & Freight Intelligence Platform built for the Smart India Hackathon.

---

## 1. Problem Statement
Maritime freight chartering, especially for dry bulk cargo on the Indian East Coast, relies heavily on fragmented, manual decision-making. Logistics managers struggle to optimize vessel class selection against complex port constraints, unpredictable freight markets, and severe demurrage risks, resulting in inefficient spot-charter dependencies and inflated voyage costs.

## 2. Proposed Solution
**DockInsights** is a comprehensive, centralized decision-support orchestration engine. It ingests cargo requirements and dynamically evaluates time-series freight forecasts, port physical limitations (Draft/LOA), and maritime risks to recommend the optimal vessel class and contract strategy (Spot vs. Term).

## 3. System Architecture
The platform utilizes a decoupled micro-engine architecture. A central **Decision Engine** acts as the orchestrator, passing data sequentially through six independent modules (Forecast, Vessel, Port, Economics, Risk, Contract). For a deeper dive, read [architecture.md](docs/architecture.md).

## 4. Technology Stack
- **Backend**: Python 3.10+, FastAPI, Pydantic, scikit-learn, statsmodels, XGBoost.
- **Frontend**: React 18, Vite, TypeScript, Recharts.
- **Infrastructure**: Docker, Docker Compose, Pytest.

## 5. Data Pipeline
DockInsights utilizes a strict validation boundary using Pydantic. Synthetic/Mock data (enabled via `SIH_DEMO_MODE`) is explicitly partitioned into `src/data/mock_db.py` to prevent data leakage into the core optimization logic. See [data_dictionary.md](docs/data_dictionary.md).

## 6. ML Methodology
The Forecasting Engine leverages a hybrid approach:
- **ARIMA** for stable, linear seasonal trends.
- **XGBoost** for non-linear regression against external macroeconomic features.
See [ml_methodology.md](docs/ml_methodology.md).

## 7. Vessel Optimization
The `VesselSelector` algorithm ranks vessel classes (Handysize up to Capesize) by mathematically comparing the cargo request against optimal deadweight tonnage (DWT).

## 8. Port Compatibility
The `PortCompatibility` engine enforces strict physical limitations, instantly disqualifying vessels if their Draft, LOA, or Beam exceed the origin or destination port limits.

## 9. Voyage Economics
Calculates raw profitability. It goes beyond simple freight rates to estimate holistic voyage costs, including daily hire rates, bunker consumption, and mathematical demurrage penalties.

## 10. Risk Engine
Quantifies external threats such as market volatility and port congestion into an objective score out of 100, providing actionable warnings to the Contract Optimizer.

## 11. Contract Optimization
Rather than blindly recommending Spot contracts, the optimizer suggests Medium-Term or Hybrid strategies if the Forecast Engine predicts rising freight rates, locking in savings. See [optimization_methodology.md](docs/optimization_methodology.md).

## 12. API
The entire intelligence backend is exposed via a robust FastAPI REST interface. See [api_documentation.md](docs/api_documentation.md).

## 13. Frontend
A sleek, professional React dashboard built for logistics managers. It visualizes the JSON payloads from the API into actionable charts and hierarchical recommendations.

## 14. Installation
```bash
git clone https://github.com/your-org/dock-insights.git
cd dock-insights
cp .env.example .env
```

## 15. Running the Project
The entire system (Backend + Frontend) is reproducible via Docker.
```bash
docker-compose up --build
```
Navigate to `http://localhost:5173` to view the dashboard.

## 16. Testing
The framework boasts a 100% pass rate across 60 unit and integration tests.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```
A chronological backtesting simulation is also available via `python scripts/run_simulation.py`.

## 17. Demo Workflow (SIH Scenario)
1. Ensure the system is running with `SIH_DEMO_MODE=True` (indicated by a red banner in the UI).
2. Open the **Voyage Planner**.
3. Input the required scenario: **100,000 MT Coal from Indonesia (INA_TAB) to Dhamra (IND_DHA)** for **3 Voyages**.
4. Click **Analyze Voyage**.
5. Watch the Decision Engine dynamically upgrade the vessel recommendation to a Capesize (optimizing out the need for multiple smaller Panamax voyages) and suggest a Contract Strategy based on the forecasted rates.

## 18. Limitations
The forecasting models cannot predict geopolitical black swans, and the current underlying port database is `[DEMO/SYNTHETIC]`. See [limitations.md](docs/limitations.md).

## 19. Future Improvements
- Integration with live Baltic Exchange FFA feeds.
- Integration with live AIS weather routing data.
- Connecting to a production PostgreSQL database for live port dimensions.

# System Architecture

DockInsights is built on a modular, decoupled architecture consisting of independent micro-engines orchestrated by a central Decision Engine.

## High-Level Data Flow
1. **Frontend (React/Vite)**: The user submits cargo parameters via the professional dashboard.
2. **API Gateway (FastAPI)**: Validates incoming JSON schemas using Pydantic and routes the request.
3. **Decision Engine**: The central brain. It receives the request and sequentially triggers:
   - `Forecast Engine`: Predicts forward spot freight rates (ARIMA/XGBoost).
   - `Vessel Selector`: Filters and ranks vessel classes by draft/cargo capacity.
   - `Port Compatibility`: Validates vessel physical constraints against origin/destination ports.
   - `Voyage Economics`: Calculates raw profitability (Bunkers, Wait time, Freight cost).
   - `Risk Engine`: Quantifies external threats (Weather, Market Volatility).
   - `Contract Optimizer`: Suggests SPOT, HYBRID, or TERM contracts based on combined outputs.
4. **Response**: A unified JSON packet is returned and rendered via Recharts on the frontend.

## Technology Stack
- **Backend**: Python 3.10+, FastAPI, Pydantic, Uvicorn, scikit-learn, statsmodels, xgboost.
- **Frontend**: React 18, Vite, TypeScript, Lucide Icons, Recharts.
- **Deployment**: Docker, Docker Compose.

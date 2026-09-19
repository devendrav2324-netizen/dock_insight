# API Documentation

The CharterAI backend exposes a RESTful API powered by FastAPI.

## Endpoints

### `GET /api/v1/health`
Returns the system status and flags if the system is running in `SIH_DEMO_MODE`.
**Response:**
```json
{
  "status": "healthy",
  "version": "1.0",
  "db_connected": true,
  "sih_demo_mode": true
}
```

### `POST /api/v1/analyze-voyage`
The primary orchestration endpoint triggering the Decision Engine.
**Request:**
```json
{
  "cargo_type": "coal",
  "cargo_quantity": 100000,
  "origin": "INA_TAB",
  "destination": "IND_DHA",
  "required_delivery_date": "2026-09-15",
  "number_of_voyages": 3
}
```
**Response:**
Returns a deeply nested JSON object mapping to `market_forecast`, `recommended_vessel`, `port_analysis`, `voyage_economics`, `risk_analysis`, `contract_strategy`, and a unified `final_recommendation`.

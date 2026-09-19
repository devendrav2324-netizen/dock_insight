import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_analyze_voyage_success(client):
    payload = {
        "cargo_type": "thermal_coal",
        "cargo_quantity": 80000,
        "origin": "IDN_TAB",
        "destination": "IND_DHM",
        "required_delivery_date": "2026-10-15",
        "number_of_voyages": 3
    }

    response = client.post("/api/v1/analyze-voyage", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "SUCCESS"
    assert "market_forecast" in data
    assert "recommended_vessel" in data
    assert "voyage_economics" in data
    assert "contract_strategy" in data
    assert data["recommended_vessel"]["class"] == "Panamax"

def test_analyze_voyage_failure_validation(client):
    # Sending missing fields should trigger Pydantic validation error (422)
    payload = {
        "cargo_type": "coal"
    }
    response = client.post("/api/v1/analyze-voyage", json=payload)
    assert response.status_code == 422

def test_complete_voyage_analysis_workflow(client):
    """
    End-to-End integration test covering the complete voyage-analysis workflow.
    Validates: Coal, 100,000 MT, Indonesia -> Dhamra, 3 Voyages
    """
    payload = {
        "cargo_type": "thermal_coal",
        "cargo_quantity": 100000.0,
        "origin": "AUS_NEW",
        "destination": "IND_GVM",
        "required_delivery_date": "2026-12-01",
        "number_of_voyages": 3
    }

    response = client.post("/api/v1/analyze-voyage", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()

    # Verify strict structure matches output requirement
    assert data["status"] == "SUCCESS"
    assert "market_forecast" in data
    assert "recommended_vessel" in data
    assert "port_analysis" in data
    assert "voyage_economics" in data
    assert "risk_analysis" in data
    assert "contract_strategy" in data
    assert "final_recommendation" in data
    assert "explanation" in data

    # Verify specific Vessel Optimizer routing (100,000 MT should pick Capesize if available,
    # but the mock DB allows Capesize for >100,000 max_dwt)
    assert data["recommended_vessel"]["class"] == "Capesize"

    # Validate the data flows correctly
    assert data["voyage_economics"]["cost_per_tonne"] > 0
    assert data["risk_analysis"]["overall_score"] >= 0

    # Assert there's at least one human readable explanation
    assert len(data["explanation"]) > 0

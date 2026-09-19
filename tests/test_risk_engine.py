import pytest
from src.risk.risk_engine import MaritimeRiskEngine, RiskSeverity

@pytest.fixture
def risk_engine():
    return MaritimeRiskEngine()

def test_low_risk_scenario(risk_engine):
    inputs = {
        "market_data": {"price_volatility_pct": 5.0},      # score: 15
        "port_data": {"expected_wait_days": 1.0},          # score: 15
        "weather_data": {"wave_height_m": 1.0, "storm_warning": False}, # score: 10
        "vessel_data": {"available_vessels_in_region": 15}, # score: 25
        "geopolitical_data": {"route_conflict_level": 1.0}, # score: 10
        "operational_data": {"maintenance_due": False}     # score: 20
    }

    result = risk_engine.evaluate_total_risk(inputs)

    assert result.overall_severity == RiskSeverity.LOW
    assert result.overall_score < 40.0
    assert "Market" in result.categories
    assert result.categories["Market"].severity == RiskSeverity.LOW
    assert result.categories["market"].level == RiskSeverity.LOW
    assert "All risk dimensions are within acceptable limits." in result.summary_messages

def test_high_risk_scenario(risk_engine):
    inputs = {
        "market_data": {"price_volatility_pct": 30.0},     # score: 90
        "port_data": {"expected_wait_days": 6.0},          # score: 90
        "weather_data": {"wave_height_m": 4.5, "storm_warning": True}, # score: 85
        "vessel_data": {"available_vessels_in_region": 2}, # score: 90
        "geopolitical_data": {"route_conflict_level": 8.0}, # score: 80
        "operational_data": {"maintenance_due": True}      # score: 80
    }

    result = risk_engine.evaluate_total_risk(inputs)

    assert result.overall_severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)
    assert result.overall_score >= 80.0
    assert result.categories["Weather"].severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)
    assert result.categories["weather"].level in (RiskSeverity.HIGH, RiskSeverity.CRITICAL)
    assert len(result.summary_messages) > 1 # multiple warnings
    assert any("Market Risk is HIGH" in msg or "Market Risk is CRITICAL" in msg for msg in result.summary_messages)

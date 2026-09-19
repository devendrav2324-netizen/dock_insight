"""
Tests for the Risk Assessment modules.
"""

from datetime import date

import pytest

from src.risk.aggregator import RiskAggregator
from src.risk.operational_risk import OperationalRiskAssessor
from src.risk.port_risk import PortRiskAssessor
from src.risk.weather_risk import WeatherRiskAssessor
from src.utils.constants import RiskLevel


class TestWeatherRisk:
    """Test weather risk scoring."""

    def test_calm_season_low_risk(self):
        assessor = WeatherRiskAssessor()
        result = assessor.assess(
            port_id="IND_GVM",
            target_date=date(2026, 2, 15),  # February — calm
            wind_speed_kmh=10.0,
            wave_height_m=0.8,
            cyclone_alert_level="None",
        )
        assert result.score < 25
        assert result.level == RiskLevel.LOW

    def test_cyclone_season_high_risk(self):
        assessor = WeatherRiskAssessor()
        result = assessor.assess(
            port_id="IND_DHM",
            target_date=date(2026, 11, 10),  # November — cyclone season
            wind_speed_kmh=45.0,
            wave_height_m=3.5,
            cyclone_alert_level="Warning",
        )
        assert result.score >= 50
        assert result.is_cyclone_season is True

    def test_monsoon_season_detected(self):
        assessor = WeatherRiskAssessor()
        result = assessor.assess(
            port_id="IND_VZG",
            target_date=date(2026, 7, 15),
        )
        assert result.is_monsoon_season is True


class TestPortRisk:
    """Test port congestion risk scoring."""

    def test_low_congestion(self):
        assessor = PortRiskAssessor()
        result = assessor.assess(
            port_id="IND_GVM",
            vessels_waiting=3,
            avg_waiting_days=0.5,
            berth_occupancy_pct=45.0,
        )
        assert result.score < 25
        assert result.level == RiskLevel.LOW

    def test_high_congestion(self):
        assessor = PortRiskAssessor()
        result = assessor.assess(
            port_id="IND_GOP",
            vessels_waiting=15,
            avg_waiting_days=5.0,
            berth_occupancy_pct=92.0,
        )
        assert result.score >= 50


class TestOperationalRisk:
    """Test operational risk scoring."""

    def test_australia_low_risk(self):
        assessor = OperationalRiskAssessor()
        result = assessor.assess("AUS_NEW", 5500)
        assert result.score < 40

    def test_russia_high_risk(self):
        assessor = OperationalRiskAssessor()
        result = assessor.assess("RUS_VOS", 8000, "Via Suez Canal")
        # Russia should have higher risk than Australia
        aus_result = assessor.assess("AUS_NEW", 5500)
        assert result.score > aus_result.score


class TestRiskAggregator:
    """Test composite risk aggregation."""

    def test_aggregation_with_all_dimensions(self):
        weather = WeatherRiskAssessor().assess("IND_GVM", date(2026, 2, 15))
        port = PortRiskAssessor().assess("IND_GVM", 3, 0.5, 45.0)
        operational = OperationalRiskAssessor().assess("AUS_NEW", 5500)

        aggregator = RiskAggregator()
        composite = aggregator.aggregate(
            weather=weather,
            port=port,
            operational=operational,
        )
        assert 0 <= composite.composite_score <= 100
        assert composite.level in RiskLevel
        assert len(composite.breakdown) == 3
        assert composite.dominant_risk is not None

    def test_aggregation_with_partial_dimensions(self):
        """Should renormalize weights when some dimensions are missing."""
        weather = WeatherRiskAssessor().assess("IND_GVM", date(2026, 11, 10))
        aggregator = RiskAggregator()
        composite = aggregator.aggregate(weather=weather)
        assert composite.composite_score == weather.score  # Only dimension

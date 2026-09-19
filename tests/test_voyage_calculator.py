"""
Tests for the deterministic Voyage Calculator.
"""

import pytest

from src.economics.bunker_estimator import BunkerEstimator
from src.economics.demurrage_calculator import DemurrageCalculator
from src.economics.voyage_calculator import VoyageCalculator


class TestBunkerEstimator:
    """Test bunker fuel cost estimation."""

    def test_panamax_estimate(self):
        estimator = BunkerEstimator()
        result = estimator.estimate(
            vessel_class="Panamax",
            sailing_distance_nm=5500,
            vlsfo_price_usd_per_tonne=550.0,
            port_days=5.0,
        )
        assert result.sailing_days > 0
        assert result.total_bunker_cost_usd > 0
        assert result.total_fuel_tonnes > result.port_fuel_tonnes

    def test_larger_vessel_more_fuel(self):
        estimator = BunkerEstimator()
        panamax = estimator.estimate("Panamax", 5500, 550.0)
        capesize = estimator.estimate("Capesize", 5500, 550.0)
        assert capesize.total_bunker_cost_usd > panamax.total_bunker_cost_usd

    def test_longer_voyage_more_fuel(self):
        estimator = BunkerEstimator()
        short = estimator.estimate("Panamax", 3000, 550.0)
        long = estimator.estimate("Panamax", 8000, 550.0)
        assert long.total_bunker_cost_usd > short.total_bunker_cost_usd


class TestDemurrageCalculator:
    """Test demurrage and despatch calculations."""

    def test_no_demurrage_within_laytime(self):
        calc = DemurrageCalculator()
        result = calc.calculate(
            vessel_class="Panamax",
            predicted_idle_days=3.0,  # Under default laytime of 6.0
        )
        assert result.demurrage_days == 0.0
        assert result.total_demurrage_usd == 0.0
        assert result.despatch_days > 0.0
        assert result.potential_despatch_usd > 0.0

    def test_demurrage_over_laytime(self):
        calc = DemurrageCalculator()
        result = calc.calculate(
            vessel_class="Panamax",
            predicted_idle_days=10.0,  # 4 days over default laytime
        )
        assert result.demurrage_days == 4.0
        assert result.total_demurrage_usd == 4.0 * 20_000  # Panamax rate
        assert result.despatch_days == 0.0

    def test_custom_laytime(self):
        calc = DemurrageCalculator()
        result = calc.calculate(
            vessel_class="Panamax",
            predicted_idle_days=5.0,
            allowed_laytime_days=3.0,
        )
        assert result.demurrage_days == 2.0


class TestVoyageCalculator:
    """Test full voyage economics."""

    def test_basic_calculation(self):
        calc = VoyageCalculator()
        result = calc.calculate(
            vessel_class="Panamax",
            cargo_tonnage=75_000,
            sailing_distance_nm=5500,
            freight_rate_usd_per_day=15_000,
            vlsfo_price_usd_per_tonne=550.0,
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
        )
        assert result.total_voyage_cost_usd > 0
        assert result.cost_per_tonne_usd > 0
        assert result.sailing_days > 0
        assert result.breakdown.freight_cost_usd > 0
        assert result.breakdown.bunker_cost_usd > 0

    def test_higher_rate_increases_cost(self):
        calc = VoyageCalculator()
        low = calc.calculate("Panamax", 75_000, 5500, 10_000, 550.0, "AUS_NEW", "IND_GVM")
        high = calc.calculate("Panamax", 75_000, 5500, 25_000, 550.0, "AUS_NEW", "IND_GVM")
        assert high.total_voyage_cost_usd > low.total_voyage_cost_usd

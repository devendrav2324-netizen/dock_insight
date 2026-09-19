"""
Tests for the deterministic Vessel Selector.
"""

import pytest

from src.optimization.vessel_selector import (
    PortConstraints,
    VesselSelector,
    VesselOptimizer,
    VesselSpecs,
)
from datetime import datetime, timedelta


@pytest.fixture
def selector():
    return VesselSelector()


@pytest.fixture
def gangavaram_port():
    """Gangavaram — deepest port, accepts Capesize (beam up to 50m per port data)."""
    return PortConstraints(
        port_id="IND_GVM",
        port_name="Gangavaram",
        max_draft_m=21.0,
        max_loa_m=300.0,
        max_beam_m=50.0,
        max_dwt=200_000,
    )


@pytest.fixture
def gopalpur_port():
    """Gopalpur — shallower port, cannot accept Capesize."""
    return PortConstraints(
        port_id="IND_GOP",
        port_name="Gopalpur",
        max_draft_m=14.5,
        max_loa_m=230.0,
        max_beam_m=32.0,
        max_dwt=85_000,
    )


@pytest.fixture
def sagar_anchorage():
    """Sagar — anchorage/lightering point, no berthing."""
    return PortConstraints(
        port_id="IND_SAG",
        port_name="Sagar Anchorage",
        max_draft_m=-9999,
        max_loa_m=-9999,
        max_beam_m=-9999,
        max_dwt=-9999,
        is_anchorage_only=True,
    )


@pytest.fixture
def newcastle_port():
    """Newcastle — large origin port, no significant constraints for bulk."""
    return PortConstraints(
        port_id="AUS_NEW",
        port_name="Newcastle",
        max_draft_m=20.0,
        max_loa_m=300.0,
        max_beam_m=50.0,
        max_dwt=200_000,
    )


@pytest.fixture
def vessel_specs():
    """Standard vessel class specifications."""
    return [
        VesselSpecs("Handysize", 10_000, 40_000, 32_000, 11.0, 190.0, 30.0),
        VesselSpecs("Supramax", 40_000, 60_000, 56_000, 13.0, 200.0, 32.5),
        VesselSpecs("Panamax", 60_000, 100_000, 82_000, 14.5, 240.0, 36.0),
        VesselSpecs("Capesize", 120_000, 220_000, 180_000, 18.5, 300.0, 50.0),
    ]


class TestVesselPortCompatibility:
    """Test individual vessel-port compatibility checks."""

    def test_handysize_fits_gopalpur(self, selector, gopalpur_port, vessel_specs):
        handysize = vessel_specs[0]
        result = selector.check_compatibility(handysize, gopalpur_port)
        assert result.is_compatible is True
        assert len(result.violations) == 0

    def test_capesize_rejected_at_gopalpur(self, selector, gopalpur_port, vessel_specs):
        capesize = vessel_specs[3]
        result = selector.check_compatibility(capesize, gopalpur_port)
        assert result.is_compatible is False
        assert any("Draft" in v for v in result.violations)

    def test_capesize_fits_gangavaram(self, selector, gangavaram_port, vessel_specs):
        capesize = vessel_specs[3]
        result = selector.check_compatibility(capesize, gangavaram_port)
        assert result.is_compatible is True

    def test_anchorage_rejects_all(self, selector, sagar_anchorage, vessel_specs):
        for vessel in vessel_specs:
            result = selector.check_compatibility(vessel, sagar_anchorage)
            assert result.is_compatible is False
            assert "anchorage" in result.violations[0].lower()


class TestVesselSelection:
    """Test full origin-destination vessel selection."""

    def test_newcastle_to_gangavaram_all_fit(
        self, selector, newcastle_port, gangavaram_port, vessel_specs
    ):
        result = selector.select_vessels(
            newcastle_port, gangavaram_port, vessel_specs
        )
        # All 4 classes should fit (Gangavaram is deep)
        assert len(result.feasible) == 4
        assert len(result.excluded) == 0

    def test_newcastle_to_gopalpur_capesize_excluded(
        self, selector, newcastle_port, gopalpur_port, vessel_specs
    ):
        result = selector.select_vessels(
            newcastle_port, gopalpur_port, vessel_specs
        )
        assert "Capesize" not in result.feasible_classes
        assert any(r.vessel_class == "Capesize" for r in result.excluded)

    def test_cargo_tonnage_filter(
        self, selector, newcastle_port, gangavaram_port, vessel_specs
    ):
        # 150,000 tonnes exceeds Handysize, Supramax, and Panamax max DWT
        result = selector.select_vessels(
            newcastle_port, gangavaram_port, vessel_specs, cargo_tonnage=150_000
        )
        assert "Capesize" in result.feasible_classes
        assert "Handysize" not in result.feasible_classes

class TestVesselOptimizer:
    @pytest.fixture
    def optimizer(self):
        return VesselOptimizer()
        
    def test_exact_cargo_match(self, optimizer, newcastle_port, gangavaram_port, vessel_specs):
        # 82,000 is the typical DWT for Panamax
        now = datetime.now()
        delivery = now + timedelta(days=20)
        
        result = optimizer.optimize(
            cargo_type="Coal",
            cargo_quantity=82_000,
            origin_port=newcastle_port,
            destination_port=gangavaram_port,
            expected_loading_date=now,
            required_delivery_date=delivery,
            vessel_specs=vessel_specs
        )
        
        # Should recommend Panamax
        assert result.recommended_vessel == "Panamax"
        assert result.score > 80.0
        assert "Optimal cargo utilization" in result.reasons

    def test_tight_delivery_dates(self, optimizer, newcastle_port, gangavaram_port, vessel_specs):
        # Delivery date is before the 14 days baseline heuristic
        now = datetime.now()
        delivery = now + timedelta(days=10)
        
        result = optimizer.optimize(
            cargo_type="Coal",
            cargo_quantity=82_000,
            origin_port=newcastle_port,
            destination_port=gangavaram_port,
            expected_loading_date=now,
            required_delivery_date=delivery,
            vessel_specs=vessel_specs
        )
        
        # Because op_score will be 0.0, total score will be 0.0, recommended "None"
        assert result.recommended_vessel == "None"
        assert result.score == 0.0
        
    def test_port_restrictions(self, optimizer, newcastle_port, gopalpur_port, vessel_specs):
        # Capesize cargo size, but gopalpur port cannot accept Capesize
        now = datetime.now()
        delivery = now + timedelta(days=20)
        
        result = optimizer.optimize(
            cargo_type="Coal",
            cargo_quantity=180_000,
            origin_port=newcastle_port,
            destination_port=gopalpur_port,
            expected_loading_date=now,
            required_delivery_date=delivery,
            vessel_specs=vessel_specs
        )
        
        # Capesize fits the cargo perfectly but will fail port compatibility.
        # Other vessels fail cargo compatibility because 180,000 exceeds their max DWT.
        # As a result, no vessel should be feasible.
        assert result.recommended_vessel == "None"
        assert result.score == 0.0
        # Check alternatives
        capesize_alt = next((a for a in result.alternatives if a.vessel_class == "Capesize"), None)
        assert capesize_alt is not None
        assert capesize_alt.port_compatibility_score == 0.0

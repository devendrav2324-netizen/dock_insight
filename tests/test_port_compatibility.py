import pytest
from src.optimization.port_compatibility import (
    PortInfo,
    VesselInfo,
    check_vessel_port_compatibility
)

@pytest.fixture
def gangavaram_port():
    return PortInfo(
        port_id="IND_GVM",
        port_name="Gangavaram",
        max_draft_m=21.0,
        max_loa_m=300.0,
        max_beam_m=50.0,
        cargo_handling_rate_tpd=50000.0,
        berthing_capacity=6,
        current_congestion_factor=1.0
    )

@pytest.fixture
def haldia_port():
    return PortInfo(
        port_id="IND_HLD",
        port_name="Haldia",
        max_draft_m=8.5,
        max_loa_m=230.0,
        max_beam_m=32.2,
        cargo_handling_rate_tpd=20000.0,
        berthing_capacity=10,
        current_congestion_factor=1.3
    )

@pytest.fixture
def panamax_vessel():
    return VesselInfo(
        class_name="Panamax",
        draft_m=14.5,
        loa_m=240.0,
        beam_m=36.0,
        cargo_to_handle_t=82000.0
    )

@pytest.fixture
def capesize_vessel():
    return VesselInfo(
        class_name="Capesize",
        draft_m=18.5,
        loa_m=300.0,
        beam_m=50.0,
        cargo_to_handle_t=180000.0
    )

def test_compatible_vessel(gangavaram_port, panamax_vessel):
    result = check_vessel_port_compatibility(panamax_vessel, gangavaram_port)
    assert result["compatible"] is True
    assert result["score"] >= 90
    assert len(result["failed_constraints"]) == 0
    assert "fully compatible" in result["explanation"].lower()

def test_incompatible_draft(haldia_port, capesize_vessel):
    result = check_vessel_port_compatibility(capesize_vessel, haldia_port)
    assert result["compatible"] is False
    assert result["score"] == 0
    assert len(result["failed_constraints"]) > 0
    assert any("Draft" in fc for fc in result["failed_constraints"])

def test_incompatible_loa(haldia_port, panamax_vessel):
    result = check_vessel_port_compatibility(panamax_vessel, haldia_port)
    assert result["compatible"] is False
    assert result["score"] == 0
    assert len(result["failed_constraints"]) > 0
    assert any("LOA" in fc for fc in result["failed_constraints"])
    assert any("Draft" in fc for fc in result["failed_constraints"]) # Panamax draft is 14.5, Haldia is 8.5
    
def test_warnings_generation(gangavaram_port, capesize_vessel):
    # Artificially lower handling rate to trigger slow turnaround warning
    gangavaram_port.cargo_handling_rate_tpd = 10000.0 
    # 180,000 / 10,000 = 18 days turnaround (> 7 days)
    
    result = check_vessel_port_compatibility(capesize_vessel, gangavaram_port)
    assert result["compatible"] is True
    assert result["score"] < 100
    assert len(result["warnings"]) > 0
    assert any("Slow turnaround warning" in w for w in result["warnings"])
    assert "operational warning" in result["explanation"].lower()

def test_congestion_warning(haldia_port, panamax_vessel):
    # Make vessel artificially fit Haldia
    panamax_vessel.draft_m = 8.0
    panamax_vessel.loa_m = 220.0
    panamax_vessel.beam_m = 30.0
    
    result = check_vessel_port_compatibility(panamax_vessel, haldia_port)
    assert result["compatible"] is True
    assert len(result["warnings"]) > 0
    assert any("High port congestion" in w for w in result["warnings"])

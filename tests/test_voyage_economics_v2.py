"""
DockInsights — Phase 5 Realistic Voyage Economics Unit & Integration Tests.

Validates all 9 cost components, dynamic sailing and port handling durations,
contractual laytime and demurrage exposure modeling, delivery probability,
and multi-factor sensitivity analysis from the charterer / cargo-owner perspective.
"""

import pytest
from src.economics.voyage_cost import (
    VoyageCostInputs,
    VoyageCostBreakdown,
    calculate_voyage_cost,
    calculate_sailing_days,
    calculate_port_operational_time,
    calculate_delivery_probability,
    perform_sensitivity_analysis,
    run_full_sensitivity_matrix,
)
from src.services.voyage_economics_service import VoyageEconomicsService


# =============================================================================
# 1. Sailing & Handling Duration Unit Tests
# =============================================================================

def test_sailing_days_formula():
    """Verify sailing_days = distance_nm / (speed_knots * 24) without hardcoded 14 days."""
    # 2880 nm at 12 knots -> 2880 / (12 * 24) = 10.0 days
    assert calculate_sailing_days(2880.0, 12.0) == 10.0

    # 4320 nm at 15 knots -> 4320 / (15 * 24) = 12.0 days
    assert calculate_sailing_days(4320.0, 15.0) == 12.0

    # Vessel speed variation strictly affects duration
    days_slow = calculate_sailing_days(3600.0, 10.0)  # 15.0 days
    days_fast = calculate_sailing_days(3600.0, 15.0)  # 10.0 days
    assert days_slow > days_fast
    assert days_slow == 15.0
    assert days_fast == 10.0


def test_port_operational_handling_time():
    """Verify loading and discharge durations depend strictly on cargo quantity / handling rate."""
    # 60,000 MT at 15,000 MT/day -> 4.0 days
    assert calculate_port_operational_time(60000.0, 15000.0) == 4.0

    # 75,000 MT at 25,000 MT/day -> 3.0 days
    assert calculate_port_operational_time(75000.0, 25000.0) == 3.0

    # Larger cargo quantity strictly takes longer to handle
    time_small = calculate_port_operational_time(40000.0, 20000.0)  # 2.0 days
    time_large = calculate_port_operational_time(80000.0, 20000.0)  # 4.0 days
    assert time_large == 2 * time_small


# =============================================================================
# 2. Component Cost Unit Tests (All 9 Components)
# =============================================================================

def test_freight_cost_component():
    """Test freight cost calculation for both Voyage Charter ($/t) and Time Charter ($/day)."""
    # Voyage charter: 70,000 MT * $25/t = $1,750,000
    voyage_inputs = VoyageCostInputs(
        cargo_quantity_t=70000.0,
        freight_rate_usd=25.0,
        vessel_speed_knots=13.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=18000.0,
        route_distance_nm=3120.0,  # 10 days
        charter_type="voyage",
    )
    res_voyage = calculate_voyage_cost(voyage_inputs)
    assert res_voyage.freight_cost == 1750000.0

    # Time charter: $20,000/day * 10 sailing days = $200,000
    tc_inputs = VoyageCostInputs(
        cargo_quantity_t=70000.0,
        freight_rate_usd=20000.0,
        vessel_speed_knots=13.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=18000.0,
        route_distance_nm=3120.0,  # 10 days
        charter_type="time_charter",
    )
    res_tc = calculate_voyage_cost(tc_inputs)
    assert res_tc.freight_cost == 20000.0 * 10.0


def test_bunker_cost_and_exposure_component():
    """Test bunker cost across sea sailing, port auxiliary consumption, and exposure factor."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=60000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.5,
        vessel_daily_fuel_consumption_tpd=28.0,
        port_daily_fuel_consumption_tpd=3.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=3000.0,  # 3000 / (12.5 * 24) = 10.0 sailing days
        fuel_price_usd_per_t=600.0,
        port_handling_rate_tpd=30000.0,  # load = 2.0 days
        discharge_port_handling_rate_tpd=20000.0,  # disch = 3.0 days
        expected_waiting_days=1.0,  # total port days = 6.0
        bunker_exposure_pct=1.0,
    )
    res = calculate_voyage_cost(inputs)
    # Sea fuel: 10 days * 28 tpd = 280 MT
    # Port fuel: 6 days * 3 tpd = 18 MT
    # Total fuel = 298 MT * $600 = $178,800
    assert res.sailing_days == 10.0
    assert res.total_port_days == 6.0
    assert res.bunker_cost == 178800.0

    # Test 50% bunker adjustment factor (BAF exposure)
    inputs.bunker_exposure_pct = 0.50
    res_half = calculate_voyage_cost(inputs)
    assert res_half.bunker_cost == 178800.0 * 0.50


def test_port_charges_component():
    """Verify aggregation of origin and destination port charges."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=25.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0,
        load_port_cost_usd=45000.0,
        discharge_port_cost_usd=55000.0,
    )
    res = calculate_voyage_cost(inputs)
    assert res.port_cost == 100000.0


def test_waiting_cost_component():
    """Verify waiting cost directly scales with congestion waiting days."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=25.0,
        vessel_daily_hire_cost_usd=16000.0,
        route_distance_nm=2880.0,
        expected_waiting_days=3.5,
    )
    res = calculate_voyage_cost(inputs)
    # 3.5 days * $16,000/day = $56,000
    assert res.waiting_cost == 56000.0


def test_demurrage_exposure_model():
    """
    Verify complete contractual demurrage model:
    laytime allowed, actual operational time, waiting, excess time, demurrage rate.
    """
    inputs = VoyageCostInputs(
        cargo_quantity_t=60000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.5,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=18000.0,
        route_distance_nm=3000.0,
        port_handling_rate_tpd=20000.0,  # 3.0 days loading
        discharge_port_handling_rate_tpd=20000.0,  # 3.0 days discharge
        # Actual operational time = 6.0 days
        expected_waiting_days=4.0,  # Total port time = 10.0 days
        laytime_allowed_days=7.0,  # Contractual laytime allowed = 7.0 days
        daily_demurrage_rate_usd=25000.0,
    )
    res = calculate_voyage_cost(inputs)

    assert res.loading_days == 3.0
    assert res.discharge_days == 3.0
    assert res.waiting_days == 4.0
    assert res.total_port_days == 10.0
    assert res.laytime_allowed_days == 7.0
    # Excess time = 10.0 - 7.0 = 3.0 days
    assert res.excess_time_days == 3.0
    # Demurrage exposure = 3.0 days * $25,000/day = $75,000
    assert res.demurrage_exposure == 75000.0


def test_despatch_savings_when_within_laytime():
    """Verify despatch reward to charterer when port turnaround is faster than laytime."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=40000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=25.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0,
        port_handling_rate_tpd=20000.0,  # 2.0 days loading
        discharge_port_handling_rate_tpd=20000.0,  # 2.0 days discharge
        expected_waiting_days=1.0,  # Total port days = 5.0
        laytime_allowed_days=7.0,  # Allowed = 7.0 -> 2.0 days early
        daily_demurrage_rate_usd=20000.0,
        despatch_rate_fraction=0.50,  # $10,000/day
    )
    res = calculate_voyage_cost(inputs)
    assert res.demurrage_exposure == 0.0
    assert res.despatch_savings == 2.0 * 10000.0  # $20,000 savings credit


def test_positioning_deadheading_cost():
    """Verify deadheading ballast leg cost (hire opportunity cost)."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0,
        positioning_distance_nm=1440.0,  # 1440 / (12 * 24) = 5.0 positioning days
    )
    res = calculate_voyage_cost(inputs)
    assert res.positioning_days == 5.0
    assert res.positioning_cost == 5.0 * 15000.0  # $75,000


def test_canal_agency_contingency_miscellaneous_cost():
    """Verify canal/route tolls, port agency fees, contingency buffers, and sundry expenses."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000.0,
        freight_rate_usd=20.0,
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=25.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0,
        canal_charges_usd=35000.0,
        agency_fees_usd=8500.0,
        contingency_cost_usd=15000.0,
        other_costs_usd=4000.0,
    )
    res = calculate_voyage_cost(inputs)
    assert res.canal_charges == 35000.0
    assert res.agency_fees == 8500.0
    assert res.contingency_cost == 15000.0
    assert res.other_costs == 4000.0
    assert res.miscellaneous_cost == 35000.0 + 8500.0 + 15000.0 + 4000.0


# =============================================================================
# 3. Total Delivered Cost & Canonical Return Dictionary Tests
# =============================================================================

def test_canonical_return_schema():
    """Verify exact 11-field return format specified in user prompt."""
    inputs = VoyageCostInputs(
        cargo_quantity_t=75000.0,
        freight_rate_usd=22.0,
        vessel_speed_knots=13.0,
        vessel_daily_fuel_consumption_tpd=32.0,
        vessel_daily_hire_cost_usd=20000.0,
        route_distance_nm=3744.0,  # 12.0 sailing days
        fuel_price_usd_per_t=580.0,
        load_port_cost_usd=50000.0,
        discharge_port_cost_usd=55000.0,
        port_handling_rate_tpd=37500.0,  # 2.0 days
        discharge_port_handling_rate_tpd=25000.0,  # 3.0 days
        expected_waiting_days=2.5,
        daily_demurrage_rate_usd=22000.0,
        positioning_distance_nm=312.0,  # 1.0 day
        canal_charges_usd=12000.0,
        agency_fees_usd=6000.0,
        contingency_cost_usd=10000.0,
        delivery_deadline_days=25.0,
    )
    breakdown = calculate_voyage_cost(inputs)
    out = breakdown.to_dict()

    expected_keys = {
        "freight_cost",
        "bunker_cost",
        "port_cost",
        "waiting_cost",
        "demurrage_exposure",
        "positioning_cost",
        "miscellaneous_cost",
        "total_cost",
        "cost_per_tonne",
        "voyage_days",
        "delivery_probability",
    }
    assert set(out.keys()) == expected_keys

    # Subscript access works identically
    for k in expected_keys:
        assert breakdown[k] == out[k]

    # Total cost equals sum of individual components
    expected_sum = (
        out["freight_cost"]
        + out["bunker_cost"]
        + out["port_cost"]
        + out["waiting_cost"]
        + out["demurrage_exposure"]
        + out["positioning_cost"]
        + out["miscellaneous_cost"]
        - breakdown.despatch_savings
    )
    assert abs(out["total_cost"] - expected_sum) <= 0.05
    assert abs(out["cost_per_tonne"] - (out["total_cost"] / 75000.0)) <= 0.02


def test_delivery_probability():
    """Verify schedule risk calculation under tight vs generous deadlines."""
    # Ample deadline (30 days for 15-day voyage)
    prob_ample = calculate_delivery_probability(total_elapsed_days=15.0, delivery_deadline_days=30.0)
    assert prob_ample > 0.95

    # Tightly matched deadline (15 days for 15-day voyage)
    prob_exact = calculate_delivery_probability(total_elapsed_days=15.0, delivery_deadline_days=15.0)
    assert 0.45 <= prob_exact <= 0.55

    # Impossible deadline (10 days for 15-day voyage)
    prob_late = calculate_delivery_probability(total_elapsed_days=15.0, delivery_deadline_days=10.0)
    assert prob_late < 0.05


# =============================================================================
# 4. Sensitivity Analysis Tests (Bunker, Freight, Congestion, Cargo)
# =============================================================================

@pytest.fixture
def base_sensitivity_inputs():
    return VoyageCostInputs(
        cargo_quantity_t=70000.0,
        freight_rate_usd=22.0,
        vessel_speed_knots=13.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=18000.0,
        route_distance_nm=3744.0,
        fuel_price_usd_per_t=600.0,
        load_port_cost_usd=50000.0,
        discharge_port_cost_usd=50000.0,
        port_handling_rate_tpd=35000.0,
        discharge_port_handling_rate_tpd=25000.0,
        expected_waiting_days=2.0,
        daily_demurrage_rate_usd=20000.0,
    )


def test_bunker_price_sensitivity(base_sensitivity_inputs):
    """Verify bunker price sweep impacts bunker cost and total cost monotonically."""
    results = perform_sensitivity_analysis(base_sensitivity_inputs, "bunker_price", values=[400.0, 500.0, 600.0, 700.0])
    assert len(results) == 4
    for i in range(len(results) - 1):
        assert results[i]["bunker_cost"] < results[i + 1]["bunker_cost"]
        assert results[i]["total_cost"] < results[i + 1]["total_cost"]
        assert results[i]["cost_per_tonne"] < results[i + 1]["cost_per_tonne"]


def test_freight_rate_sensitivity(base_sensitivity_inputs):
    """Verify freight rate sweep directly alters freight cost and total delivered cost."""
    results = perform_sensitivity_analysis(base_sensitivity_inputs, "freight_rate", values=[18.0, 20.0, 22.0, 24.0])
    assert len(results) == 4
    for i in range(len(results) - 1):
        assert results[i]["freight_cost"] < results[i + 1]["freight_cost"]
        assert results[i]["cost_per_tonne"] < results[i + 1]["cost_per_tonne"]


def test_congestion_sensitivity(base_sensitivity_inputs):
    """Verify congestion waiting days sweep triggers waiting cost and demurrage liability."""
    results = perform_sensitivity_analysis(base_sensitivity_inputs, "congestion", values=[0.0, 2.0, 5.0, 8.0])
    assert len(results) == 4
    # Zero congestion should have zero waiting cost
    assert results[0]["waiting_cost"] == 0.0
    # Severe congestion (8 days) should have high demurrage exposure
    assert results[3]["demurrage_exposure"] > results[1]["demurrage_exposure"]
    assert results[3]["voyage_days"] > results[0]["voyage_days"]


def test_cargo_quantity_sensitivity(base_sensitivity_inputs):
    """Verify cargo parcel sizing affects port turnaround days and total cost."""
    results = perform_sensitivity_analysis(base_sensitivity_inputs, "cargo_quantity", values=[50000.0, 70000.0, 90000.0])
    assert len(results) == 3
    # Larger parcels have higher total freight and voyage days
    assert results[2]["freight_cost"] > results[0]["freight_cost"]
    assert results[2]["voyage_days"] > results[0]["voyage_days"]


def test_full_sensitivity_matrix(base_sensitivity_inputs):
    """Verify full multi-factor matrix execution returns all 4 key dimensions."""
    matrix = run_full_sensitivity_matrix(base_sensitivity_inputs)
    assert set(matrix.keys()) == {"bunker_price", "freight_rate", "congestion", "cargo_quantity"}
    for k, v in matrix.items():
        assert len(v) >= 4
        assert "delta_cost_usd" in v[0]
        assert "delta_cost_per_tonne_usd" in v[0]


# =============================================================================
# 5. Service & Congestion Integration Tests
# =============================================================================

def test_voyage_economics_service_delivered_cost():
    """Verify VoyageEconomicsService seamlessly resolves port data and dynamic congestion."""
    service = VoyageEconomicsService()
    res = service.calculate_delivered_cost(
        cargo_quantity_t=65000.0,
        freight_rate_usd=23.5,
        origin_port_id="INA_TAB",
        destination_port_id="IND_PAR",
        vessel_class="Panamax",
        as_dict=True,
    )
    assert isinstance(res, dict)
    assert res["freight_cost"] == 65000.0 * 23.5
    assert res["cost_per_tonne"] > 23.5  # Includes bunkers, ports, waiting
    assert res["voyage_days"] > 10.0
    assert 0.0 <= res["delivery_probability"] <= 1.0

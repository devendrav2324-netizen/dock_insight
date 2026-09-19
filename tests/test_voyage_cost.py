import pytest
from src.economics.voyage_cost import VoyageCostInputs, calculate_voyage_cost

def test_standard_voyage():
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000,
        freight_rate_usd=20.0, # 20 * 50000 = $1,000,000
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0, # 2880 / (12*24) = 10 days
        positioning_distance_nm=0.0, # 0 days
        fuel_price_usd_per_t=600.0,
        load_port_cost_usd=50000.0,
        discharge_port_cost_usd=50000.0,
        expected_waiting_days=1.0, # Waiting cost = $15,000. No demurrage (since <= 2.0)
        daily_demurrage_rate_usd=20000.0,
        other_costs_usd=10000.0
    )
    
    result = calculate_voyage_cost(inputs)
    
    assert result.freight_cost_usd == 1000000.0
    assert result.bunker_cost_usd == 10 * 30.0 * 600.0 # 180,000
    assert result.port_costs_usd == 100000.0
    assert result.waiting_cost_usd == 15000.0
    assert result.expected_demurrage_usd == 0.0
    assert result.positioning_cost_usd == 0.0
    assert result.other_costs_usd == 10000.0
    assert result.total_voyage_cost_usd == 1000000 + 180000 + 100000 + 15000 + 0 + 0 + 10000

def test_congested_voyage_with_positioning():
    inputs = VoyageCostInputs(
        cargo_quantity_t=50000,
        freight_rate_usd=20.0, # 1,000,000
        vessel_speed_knots=12.0,
        vessel_daily_fuel_consumption_tpd=30.0,
        vessel_daily_hire_cost_usd=15000.0,
        route_distance_nm=2880.0, # 10 days
        positioning_distance_nm=1440.0, # 5 days
        fuel_price_usd_per_t=600.0,
        load_port_cost_usd=50000.0,
        discharge_port_cost_usd=50000.0,
        expected_waiting_days=5.0, # Waiting = 5 * 15k = 75k. Demurrage = 3 * 20k = 60k
        daily_demurrage_rate_usd=20000.0,
        other_costs_usd=10000.0
    )
    
    result = calculate_voyage_cost(inputs)
    
    assert result.freight_cost_usd == 1000000.0
    assert result.bunker_cost_usd == 15 * 30.0 * 600.0 # 270,000
    assert result.port_costs_usd == 100000.0
    assert result.waiting_cost_usd == 75000.0
    assert result.expected_demurrage_usd == 60000.0
    assert result.positioning_cost_usd == 5 * 15000.0 # 75,000
    assert result.other_costs_usd == 10000.0
    assert result.total_voyage_cost_usd == 1000000 + 270000 + 100000 + 75000 + 60000 + 75000 + 10000

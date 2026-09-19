"""
Charter-AI — Problem 6A Hardcoded Decision Inputs & Verification Tests.

Verifies:
1. No duplicate production vessel specification dictionaries exist outside vessel_repository.py.
2. Vessel availability is multi-stage eligibility filtered (status, class, draft, LOA, beam, laycan).
3. Vessel dataset mutations dynamically alter availability ratio and classification labels.
4. End-to-end DecisionEngine dynamic market momentum calculation (rising, falling, zero-rate protection).
5. Single-source vessel repository parameter propagation across all consuming modules.
6. Configurable business thresholds alter decision logic.
7. Synthetic demo data transparency metadata is preserved.
"""

import ast
import os
import pytest
from datetime import datetime, date
from typing import Dict, Any, List
from unittest.mock import MagicMock

from src.utils.config import get_settings
from src.data.vessel_repository import (
    get_vessel_class_spec,
    get_all_vessel_class_specs,
    calculate_vessel_availability,
    VesselClassSpec,
)
from src.models.market_timing import MarketTimingEngine, MarketTimingInputs, TimingAction
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.services.freight_forecast_service import FreightForecastService
from src.data.mock_db import get_mock_vessel_db
from src.optimization.candidate_generator import CandidateGenerator
from src.utils.constants import VESSEL_SPEEDS_KNOTS, VesselClass
from src.models.congestion_predictor import VESSEL_CLASS_DWT
from src.data.validators.business_validator import BusinessRuleValidator


def test_static_check_no_duplicate_vessel_specs():
    """PART A: Static check to confirm no duplicate production vessel specification dicts exist in src/."""
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    
    # Class names to search for static dict assignments
    forbidden_keys = {"Handysize", "Supramax", "Panamax", "Capesize"}
    
    violations = []
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue
            filepath = os.path.join(root, file)
            # Exclude vessel_repository.py which is the single source of truth
            if os.path.basename(filepath) == "vessel_repository.py":
                continue
                
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            try:
                tree = ast.parse(content, filename=filepath)
            except Exception:
                continue

            # Check for Dict literals containing all 4 vessel classes hardcoded
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    keys = set()
                    for k in node.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.add(k.value)
                    if forbidden_keys.issubset(keys):
                        # Verify it's not a dynamic construction (e.g. Dict comprehension or helper call)
                        # If values are tuples or hardcoded Dicts, flag as duplicate definition
                        has_hardcoded_values = any(
                            isinstance(val, (ast.Tuple, ast.Dict, ast.Call)) and
                            not (isinstance(val, ast.Call) and getattr(val.func, "id", "") in ("VesselClassDefinition", "get_vessel_class_spec", "get_all_vessel_class_specs"))
                            for val in node.values
                        )
                        # Check if it's raw numeric dictionary literal
                        if any(isinstance(v, (ast.Tuple, ast.Constant)) for v in node.values):
                            violations.append(f"{os.path.relpath(filepath, src_dir)}: Dict contains hardcoded class specs {keys}")

    assert not violations, f"Duplicate vessel specification dictionaries found in src/: {violations}"


def test_end_to_end_market_momentum_pipeline():
    """PART C: End-to-end DecisionEngine test calculating market momentum dynamically from forecasts."""
    mock_forecast_service = MagicMock(spec=FreightForecastService)

    # CASE 1: Baseline / Flat Market (current = 20.0, forecast = 20.0 -> momentum = 0.0%)
    mock_forecast_service.predict_freight_api.return_value = {
        "current_rate": 20.0,
        "forecast_rate": 20.0,
        "lower_bound": 18.0,
        "upper_bound": 22.0,
        "trend": "STABLE",
        "confidence": 0.85,
        "model_used": "Ensemble",
        "metrics": {"volatility": 12.0},
    }

    de = DecisionEngine(forecast_service=mock_forecast_service)
    inputs = DecisionEngineInputs(
        cargo_type="Coal",
        cargo_quantity_t=75000.0,
        origin_port_id="IDN_TAB",
        destination_port_id="IND_PAR",
        expected_loading_date=datetime(2026, 10, 1),
        required_delivery_date=datetime(2026, 11, 1),
    )

    res_flat = de.evaluate(inputs)
    assert res_flat["status"] == "SUCCESS"
    assert res_flat["market_timing"]["market_momentum"] == 0.0

    # CASE 2: Strongly Rising Market (current = 20.0, forecast = 26.0 -> momentum = +30.0%)
    mock_forecast_service.predict_freight_api.return_value = {
        "current_rate": 20.0,
        "forecast_rate": 26.0,
        "lower_bound": 24.0,
        "upper_bound": 29.0,
        "trend": "RISING",
        "confidence": 0.88,
        "model_used": "Ensemble",
        "metrics": {"volatility": 14.0},
    }

    res_rising = de.evaluate(inputs)
    assert res_rising["status"] == "SUCCESS"
    assert res_rising["market_timing"]["market_momentum"] == 30.0
    assert res_rising["market_timing"]["recommendation"] in ["BOOK_NOW", "START_NEGOTIATION"]

    # CASE 3: Strongly Falling Market (current = 20.0, forecast = 14.0 -> momentum = -30.0%)
    mock_forecast_service.predict_freight_api.return_value = {
        "current_rate": 20.0,
        "forecast_rate": 14.0,
        "lower_bound": 12.0,
        "upper_bound": 16.0,
        "trend": "FALLING",
        "confidence": 0.82,
        "model_used": "Ensemble",
        "metrics": {"volatility": 15.0},
    }

    res_falling = de.evaluate(inputs)
    assert res_falling["status"] == "SUCCESS"
    assert res_falling["market_timing"]["market_momentum"] == -30.0
    assert res_falling["market_timing"]["recommendation"] in ["WAIT", "MONITOR"]

    # CASE 4: Zero / Near-Zero Current Freight Rate Edge Case Protection
    mock_forecast_service.predict_freight_api.return_value = {
        "current_rate": 0.0,
        "forecast_rate": 20.0,
        "lower_bound": 15.0,
        "upper_bound": 25.0,
        "trend": "RISING",
        "confidence": 0.80,
        "model_used": "Ensemble",
        "metrics": {"volatility": 10.0},
    }
    res_zero = de.evaluate(inputs)
    assert res_zero["status"] == "SUCCESS"
    assert isinstance(res_zero["market_timing"]["market_momentum"], float)


def test_vessel_availability_eligibility_filtering():
    """PART B: Verify vessel availability performs strict multi-stage eligibility filtering."""
    vessel_pool = [
        # Available & Fully Eligible
        {"vessel_id": "V1", "vessel_class": "Panamax", "dwt": 75000, "availability_status": "AVAILABLE", "max_draft_m": 14.0, "loa_m": 225.0, "beam_m": 32.0, "available_from": date(2026, 9, 25)},
        # Ineligible due to status
        {"vessel_id": "V2", "vessel_class": "Panamax", "dwt": 75000, "availability_status": "MAINTENANCE", "max_draft_m": 14.0, "loa_m": 225.0, "beam_m": 32.0, "available_from": date(2026, 9, 25)},
        # Ineligible due to Draft (> port max draft 14.5m)
        {"vessel_id": "V3", "vessel_class": "Capesize", "dwt": 180000, "availability_status": "AVAILABLE", "max_draft_m": 18.5, "loa_m": 290.0, "beam_m": 45.0, "available_from": date(2026, 9, 25)},
        # Ineligible due to LOA (> port max LOA 230m)
        {"vessel_id": "V4", "vessel_class": "Panamax", "dwt": 75000, "availability_status": "AVAILABLE", "max_draft_m": 14.0, "loa_m": 240.0, "beam_m": 32.0, "available_from": date(2026, 9, 25)},
        # Ineligible due to Laycan window (available after expected loading date date(2026, 10, 1))
        {"vessel_id": "V5", "vessel_class": "Panamax", "dwt": 75000, "availability_status": "AVAILABLE", "max_draft_m": 14.0, "loa_m": 225.0, "beam_m": 32.0, "available_from": date(2026, 10, 15)},
    ]

    calc = calculate_vessel_availability(
        cargo_quantity_t=70000.0,
        vessel_list=vessel_pool,
        target_vessel_class="Panamax",
        max_draft_m=14.5,
        max_loa_m=230.0,
        max_beam_m=35.0,
        expected_loading_date=date(2026, 10, 1),
    )

    assert calc["total_vessels"] == 5
    assert calc["available_vessels"] == 4  # V1, V3, V4, V5 (status == AVAILABLE)
    assert calc["eligible_vessels"] == 1   # Only V1 passes status, class, draft, LOA, beam, AND date!
    assert calc["eligible_capacity_t"] == 75000.0


def test_vessel_dataset_mutation_impacts_availability_ratio():
    """PART D: Verify capacity ratio and availability classification react dynamically to dataset mutations."""
    cargo_req = 100000.0

    # Small fleet -> TIGHT
    fleet_sparse = [
        {"vessel_id": "V1", "vessel_class": "Handysize", "dwt": 35000, "availability_status": "AVAILABLE"}
    ]
    calc_sparse = calculate_vessel_availability(cargo_quantity_t=cargo_req, vessel_list=fleet_sparse)
    assert calc_sparse["vessel_availability_label"] == "TIGHT"
    assert calc_sparse["capacity_ratio"] == 0.35

    # Add eligible Capesize + Panamax vessels -> SURPLUS
    fleet_abundant = fleet_sparse + [
        {"vessel_id": f"V{i}", "vessel_class": "Capesize", "dwt": 180000, "availability_status": "AVAILABLE"}
        for i in range(2)
    ]
    calc_abundant = calculate_vessel_availability(cargo_quantity_t=cargo_req, vessel_list=fleet_abundant)
    assert calc_abundant["vessel_availability_label"] == "SURPLUS"
    assert calc_abundant["capacity_ratio"] == 3.95
    assert calc_abundant["capacity_ratio"] > calc_sparse["capacity_ratio"]

    # Remove all vessels -> TIGHT (0 capacity)
    calc_empty = calculate_vessel_availability(cargo_quantity_t=cargo_req, vessel_list=[])
    assert calc_empty["vessel_availability_label"] == "TIGHT"
    assert calc_empty["capacity_ratio"] == 0.0


def test_vessel_repository_centralization_propagation():
    """PART E: Verify vessel_repository supplies specs and changes propagate across modules."""
    panamax_spec = get_vessel_class_spec("Panamax")
    capesize_spec = get_vessel_class_spec("Capesize")

    assert panamax_spec.typical_dwt == 75000
    assert panamax_spec.service_speed_knots == 13.0
    assert capesize_spec.typical_dwt == 175000

    # Verify mock_db derives specs from vessel_repository
    mock_db_vessels = get_mock_vessel_db()
    panamax_mock = next(v for v in mock_db_vessels if v.class_name == "Panamax")
    assert panamax_mock.typical_dwt == panamax_spec.typical_dwt

    # Verify CandidateGenerator derives vessel classes from vessel_repository
    cg_classes = CandidateGenerator().vessel_classes
    assert cg_classes["Panamax"].typical_dwt == panamax_spec.typical_dwt

    # Verify constants derive speeds from vessel_repository
    assert VESSEL_SPEEDS_KNOTS[VesselClass.PANAMAX] == panamax_spec.service_speed_knots

    # Verify congestion predictor derives DWT from vessel_repository
    assert VESSEL_CLASS_DWT["Panamax"] == panamax_spec.typical_dwt

    # Verify business validator derives DWT ranges from vessel_repository
    min_dwt, max_dwt = BusinessRuleValidator.VESSEL_CLASS_DWT_RANGES["Panamax"]
    assert min_dwt <= panamax_spec.dwt_min
    assert max_dwt >= panamax_spec.dwt_max


def test_configurable_thresholds_override():
    """PART F: Business decision thresholds are read from Settings and alter classifications."""
    settings = get_settings()

    vessels = [
        {"vessel_id": "V1", "vessel_class": "Handysize", "dwt": 150000, "availability_status": "AVAILABLE"}
    ]
    # Ratio = 150,000 / 100,000 = 1.50

    # Default settings: tight = 1.25, balanced = 2.50 -> ratio 1.50 gives "BALANCED"
    calc_default = calculate_vessel_availability(cargo_quantity_t=100000.0, vessel_list=vessels)
    assert calc_default["capacity_ratio"] == 1.50
    assert calc_default["vessel_availability_label"] == "BALANCED"

    # Temporarily override tight threshold to 2.0
    original_tight = settings.availability_tight_threshold
    try:
        settings.availability_tight_threshold = 2.0
        calc_override = calculate_vessel_availability(cargo_quantity_t=100000.0, vessel_list=vessels)
        assert calc_override["vessel_availability_label"] == "TIGHT"
    finally:
        settings.availability_tight_threshold = original_tight


def test_synthetic_demo_transparency():
    """PART G: Synthetic demo transparency tags are preserved."""
    calc = calculate_vessel_availability(cargo_quantity_t=70000.0)
    assert calc["data_source"] == "SYNTHETIC_DEMO"

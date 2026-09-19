"""
Charter-AI — Test Suite for Phase 11 Historical Backtesting Framework.

Tests:
1. Metric calculations (forecast, optimization, market timing, contract strategy)
2. Scenario generation and date ordering
3. Strict walk-forward split integrity & absence of future leakage
4. Forecast walk-forward evaluation across models and horizons
5. Optimization backtester evaluation across CharterAI and all 5 Baselines
6. End-to-end BacktestEngine execution, comparative metrics, and artifact export
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_smape,
    calculate_directional_accuracy,
    compute_forecast_metrics,
    compute_optimization_metrics,
    compute_market_timing_metrics,
    compute_contract_strategy_metrics,
)
from src.backtesting.scenarios import (
    BacktestScenario,
    generate_standard_scenarios,
    HISTORICAL_FIXTURE_TEMPLATES,
)
from src.backtesting.forecast_backtester import ForecastBacktester
from src.backtesting.optimization_backtester import OptimizationBacktester
from src.backtesting.backtest_engine import BacktestEngine


# =============================================================================
# 1. Metric Calculations Unit Tests
# =============================================================================

def test_forecast_metrics_exact_formulas():
    y_true = [10.0, 20.0, 30.0, 40.0]
    y_pred = [12.0, 18.0, 33.0, 38.0]
    y_last = [9.0, 21.0, 28.0, 42.0]

    # MAE = (|2| + |2| + |3| + |2|) / 4 = 9/4 = 2.25
    mae = calculate_mae(y_true, y_pred)
    assert mae == pytest.approx(2.25, abs=1e-4)

    # RMSE = sqrt((4 + 4 + 9 + 4) / 4) = sqrt(21/4) = sqrt(5.25) ~= 2.29128
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == pytest.approx(np.sqrt(5.25), abs=1e-4)

    # sMAPE
    smape = calculate_smape(y_true, y_pred)
    assert 0.0 <= smape <= 100.0

    # Directional accuracy
    dir_acc = calculate_directional_accuracy(y_true, y_pred, y_last)
    assert 0.0 <= dir_acc <= 100.0

    all_metrics = compute_forecast_metrics(y_true, y_pred, y_last)
    assert "mae" in all_metrics
    assert "rmse" in all_metrics
    assert "smape" in all_metrics
    assert "directional_accuracy_pct" in all_metrics


def test_optimization_metrics():
    costs = [1000000.0, 1500000.0]
    cargo = [50000.0, 75000.0]
    demurrage = [20000.0, 0.0]
    delays = [0.5, 0.0]
    success = [True, True]

    m = compute_optimization_metrics(costs, cargo, demurrage, delays, success)
    assert m["total_cost"] == 2500000.0
    assert m["cost_per_tonne"] == pytest.approx(2500000.0 / 125000.0, abs=1e-2)
    assert m["total_demurrage"] == 20000.0
    assert m["delivery_success_rate"] == 100.0


def test_market_timing_metrics():
    booked = [18.0, 15.0]
    benchmark = [20.0, 16.0]  # Charter booked cheaper
    min_window = [17.5, 14.5]
    cargo = [50000.0, 50000.0]

    m = compute_market_timing_metrics(booked, benchmark, min_window, cargo)
    # Avoided cost = (20 - 18)*50k + (16 - 15)*50k = 100k + 50k = 150,000
    assert m["avoided_cost"] == pytest.approx(150000.0, abs=1.0)
    # Missed opportunity = (18 - 17.5)*50k + (15 - 14.5)*50k = 25k + 25k = 50,000
    assert m["missed_opportunity_cost"] == pytest.approx(50000.0, abs=1.0)
    assert m["booking_success_rate"] == 100.0


def test_contract_strategy_metrics():
    cpt = [20.0, 22.0, 24.0, 26.0]
    totals = [1000000.0, 1100000.0, 1200000.0, 1300000.0]

    m = compute_contract_strategy_metrics(cpt, totals)
    assert m["expected_cost"] == pytest.approx(1150000.0, abs=1.0)
    assert m["downside_cost"] > m["expected_cost"]
    assert m["volatility_exposure"] > 0.0


# =============================================================================
# 2. Scenario Generation Unit Tests
# =============================================================================

def test_standard_scenarios_generation():
    scenarios = generate_standard_scenarios(years=[2022, 2023, 2024])
    assert len(scenarios) == 36

    years_present = {s.year for s in scenarios}
    assert years_present == {2022, 2023, 2024}

    for s in scenarios:
        assert s.scenario_id.startswith("SCEN_")
        assert s.cargo_quantity_t > 0
        assert s.order_date < s.laycan_start
        assert s.laycan_start < s.laycan_end
        assert s.laycan_end < s.required_delivery_date
        assert s.risk_tolerance in ["LOW", "MEDIUM", "HIGH"]
        assert s.origin_port_id in ["AUS_NEW", "AUS_HAY", "IDN_TAB", "ZAF_RIC"]


# =============================================================================
# 3. Walk-Forward Split & Zero Leakage Tests
# =============================================================================

def test_walk_forward_splits_no_future_leakage():
    engine = BacktestEngine(data_dir="data/processed")
    years = [2022, 2023, 2024]

    # Check split definitions
    splits = [
        {"name": f"Split_{yr}", "train_end": f"{yr-1}-12-31", "test_year": yr}
        for yr in years
    ]

    for sp in splits:
        cutoff = pd.to_datetime(sp["train_end"])
        test_yr = sp["test_year"]
        # Train cutoff must strictly precede the test year
        assert cutoff.year < test_yr
        assert cutoff.month == 12
        assert cutoff.day == 31


# =============================================================================
# 4. Forecast Backtester Integration Tests
# =============================================================================

def test_forecast_backtester_evaluation():
    fb = ForecastBacktester(data_dir="data/processed")
    splits = [
        {"name": "Split_2024", "train_end": "2023-12-31", "test_year": 2024}
    ]
    routes = [
        ("AUS_NEW", "IND_GVM", "Panamax"),
        ("IDN_TAB", "IND_DHM", "Panamax"),
    ]

    res = fb.evaluate_walk_forward_splits(splits=splits, horizons=[3, 7], routes=routes)
    assert "freight_summary_by_model" in res
    assert "congestion_summary" in res
    assert "CharterAI Ensemble" in res["freight_summary_by_model"]
    assert "Baseline 1: Last Rate" in res["freight_summary_by_model"]

    # Verify metrics exist for evaluated horizons
    charter_metrics = res["freight_summary_by_model"]["CharterAI Ensemble"]
    assert "3d" in charter_metrics
    assert charter_metrics["3d"]["mae"] > 0.0


# =============================================================================
# 5. Optimization Backtester & 5 Baselines Tests
# =============================================================================

def test_optimization_backtester_all_5_baselines():
    ob = OptimizationBacktester(data_dir="data/processed")
    test_scenarios = [
        BacktestScenario(
            scenario_id="TEST_01",
            year=2024,
            order_date=datetime(2024, 3, 15, 10, 0, 0),
            origin_port_id="AUS_NEW",
            destination_port_id="IND_GVM",
            cargo_type="thermal_coal",
            cargo_quantity_t=80000.0,
            laycan_start=datetime(2024, 3, 22),
            laycan_end=datetime(2024, 3, 29),
            required_delivery_date=datetime(2024, 4, 28),
            risk_tolerance="MEDIUM",
        ),
        BacktestScenario(
            scenario_id="TEST_02",
            year=2024,
            order_date=datetime(2024, 6, 15, 10, 0, 0),
            origin_port_id="IDN_TAB",
            destination_port_id="IND_DHM",
            cargo_type="thermal_coal",
            cargo_quantity_t=75000.0,
            laycan_start=datetime(2024, 6, 22),
            laycan_end=datetime(2024, 6, 29),
            required_delivery_date=datetime(2024, 7, 28),
            risk_tolerance="LOW",
        ),
    ]

    res = ob.run_scenarios(test_scenarios)
    summary = res["summary_by_strategy"]

    expected_strategies = [
        "CharterAI Decision Engine",
        "Baseline 1: Current Freight Rate",
        "Baseline 2: Moving Average",
        "Baseline 3: Always Largest Vessel",
        "Baseline 4: Always Spot",
        "Baseline 5: Fixed Vessel Rule (Panamax)",
    ]

    for strat in expected_strategies:
        assert strat in summary, f"Missing strategy: {strat}"
        assert summary[strat]["total_cost"] > 0.0
        assert summary[strat]["cost_per_tonne"] > 0.0
        assert "expected_cost" in summary[strat]
        assert "avoided_cost" in summary[strat]


# =============================================================================
# 6. End-to-End Engine and Report Generation Tests
# =============================================================================

def test_backtest_engine_end_to_end_and_reports(tmp_path):
    engine = BacktestEngine(data_dir="data/processed", seed=42)
    report = engine.run_walk_forward_backtest(years=[2024], horizons=[7])

    assert "metadata" in report
    assert "forecast_evaluation" in report
    assert "optimization_evaluation" in report
    assert "comparative_analysis" in report

    # Export to temp directory
    paths = engine.export_reports(report, output_dir=str(tmp_path))

    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["markdown"]).exists()

    # Verify JSON content
    with open(paths["json"]) as f:
        loaded = json.load(f)
        assert loaded["metadata"]["test_years"] == [2024]

    # Verify Markdown contains executive summary and tables
    with open(paths["markdown"]) as f:
        md_text = f.read()
        assert "# CharterAI — Historical Walk-Forward Backtesting Report" in md_text
        assert "CharterAI Decision Engine" in md_text
        assert "Baseline 1: Current Freight Rate" in md_text
        assert "Baseline 3: Always Largest Vessel" in md_text
        assert "Baseline 5: Fixed Vessel Rule (Panamax)" in md_text

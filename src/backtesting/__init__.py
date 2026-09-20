"""
DockInsights — Backtesting Framework (Phase 11).

Provides rigorous, leak-free walk-forward historical validation across:
- Time-series freight and port congestion forecasting
- Vessel fleet selection and delivered cost economics
- Market timing and contract portfolio strategy optimization
"""

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
from src.backtesting.forecast_backtester import (
    ForecastBacktester,
    ForecastEvaluationResult,
)
from src.backtesting.optimization_backtester import (
    OptimizationBacktester,
    ScenarioExecutionResult,
)
from src.backtesting.backtest_engine import BacktestEngine

__all__ = [
    "BacktestEngine",
    "ForecastBacktester",
    "ForecastEvaluationResult",
    "OptimizationBacktester",
    "ScenarioExecutionResult",
    "BacktestScenario",
    "generate_standard_scenarios",
    "HISTORICAL_FIXTURE_TEMPLATES",
    "calculate_mae",
    "calculate_rmse",
    "calculate_smape",
    "calculate_directional_accuracy",
    "compute_forecast_metrics",
    "compute_optimization_metrics",
    "compute_market_timing_metrics",
    "compute_contract_strategy_metrics",
]

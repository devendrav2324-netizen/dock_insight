"""
Charter-AI — Backtesting Evaluation Metrics (Phase 11).

Provides standardized, objective metric calculations across four operational domains:
1. Time-Series Forecasting: MAE, RMSE, sMAPE, Directional Accuracy
2. Voyage & Fleet Optimization: Total Delivered Cost, Cost per Tonne, Demurrage, Schedule Delay, Delivery Success Rate
3. Market Timing: Avoided Cost, Missed Opportunity Cost, Booking Success Rate
4. Contract Strategy: Expected Cost, Downside Cost (P90), Volatility Exposure
"""

import math
from typing import Dict, List, Optional, Sequence, Union
import numpy as np


# =============================================================================
# 1. Forecasting Metrics
# =============================================================================

def calculate_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(np.abs(y_t - y_p)))


def calculate_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_t - y_p) ** 2)))


def calculate_smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 200].
    Formula: 100 * mean(2 * |y - y_hat| / (|y| + |y_hat| + eps))
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    if len(y_t) == 0:
        return 0.0
    denominator = np.abs(y_t) + np.abs(y_p) + 1e-8
    return float(100.0 * np.mean(2.0 * np.abs(y_t - y_p) / denominator))


def calculate_directional_accuracy(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_last: Sequence[float],
) -> float:
    """
    Directional accuracy: percentage of times the predicted movement direction
    (relative to the last known point) matches the actual movement direction.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    y_l = np.asarray(y_last, dtype=float)
    if len(y_t) == 0:
        return 0.0
    true_dir = np.sign(y_t - y_l)
    pred_dir = np.sign(y_p - y_l)
    return float(100.0 * np.mean(true_dir == pred_dir))


def compute_forecast_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_last: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Computes all standard forecast accuracy metrics."""
    metrics = {
        "mae": round(calculate_mae(y_true, y_pred), 3),
        "rmse": round(calculate_rmse(y_true, y_pred), 3),
        "smape": round(calculate_smape(y_true, y_pred), 2),
    }
    if y_last is not None and len(y_last) == len(y_true):
        metrics["directional_accuracy_pct"] = round(
            calculate_directional_accuracy(y_true, y_pred, y_last), 1
        )
    return metrics


# =============================================================================
# 2. Voyage & Fleet Optimization Metrics
# =============================================================================

def compute_optimization_metrics(
    costs: Sequence[float],
    cargo_tonnages: Sequence[float],
    demurrages: Sequence[float],
    delays_days: Sequence[float],
    delivery_successes: Sequence[bool],
) -> Dict[str, float]:
    """
    Computes portfolio-level optimization metrics across a set of executed voyages.
    """
    c = np.asarray(costs, dtype=float)
    t = np.asarray(cargo_tonnages, dtype=float)
    d = np.asarray(demurrages, dtype=float)
    delays = np.asarray(delays_days, dtype=float)
    succ = np.asarray(delivery_successes, dtype=bool)

    total_cargo = float(np.sum(t)) if len(t) > 0 else 1.0
    total_cost = float(np.sum(c)) if len(c) > 0 else 0.0

    return {
        "total_cost": round(total_cost, 2),
        "cost_per_tonne": round(total_cost / max(1.0, total_cargo), 2),
        "total_demurrage": round(float(np.sum(d)), 2) if len(d) > 0 else 0.0,
        "avg_demurrage_per_voyage": round(float(np.mean(d)), 2) if len(d) > 0 else 0.0,
        "avg_schedule_delay_days": round(float(np.mean(delays)), 2) if len(delays) > 0 else 0.0,
        "max_schedule_delay_days": round(float(np.max(delays)), 2) if len(delays) > 0 else 0.0,
        "delivery_success_rate": round(float(100.0 * np.mean(succ)), 1) if len(succ) > 0 else 100.0,
    }


# =============================================================================
# 3. Market Timing Metrics
# =============================================================================

def compute_market_timing_metrics(
    booked_rates: Sequence[float],
    benchmark_rates: Sequence[float],
    min_window_rates: Sequence[float],
    cargo_tonnages: Sequence[float],
) -> Dict[str, float]:
    """
    Evaluates market timing intelligence:
    - avoided_cost: Savings achieved vs benchmark (e.g. naive booking on request day)
    - missed_opportunity_cost: Extra cost incurred vs perfect hindsight bottom
    - booking_success_rate: % of bookings where booked_rate <= benchmark_rate
    """
    b_rates = np.asarray(booked_rates, dtype=float)
    bench = np.asarray(benchmark_rates, dtype=float)
    min_r = np.asarray(min_window_rates, dtype=float)
    t = np.asarray(cargo_tonnages, dtype=float)

    if len(b_rates) == 0:
        return {
            "avoided_cost": 0.0,
            "missed_opportunity_cost": 0.0,
            "booking_success_rate": 0.0,
        }

    # Avoided cost = (benchmark - booked) * cargo
    avoided = np.sum((bench - b_rates) * t)
    # Missed opportunity = (booked - min_rate) * cargo (always >= 0)
    missed = np.sum(np.maximum(0.0, b_rates - min_r) * t)
    # Booking success = booked <= benchmark
    success = float(100.0 * np.mean(b_rates <= bench + 1e-4))

    return {
        "avoided_cost": round(float(avoided), 2),
        "missed_opportunity_cost": round(float(missed), 2),
        "booking_success_rate": round(success, 1),
    }


# =============================================================================
# 4. Contract Strategy Metrics
# =============================================================================

def compute_contract_strategy_metrics(
    costs_per_tonne: Sequence[float],
    total_costs: Sequence[float],
) -> Dict[str, float]:
    """
    Evaluates contract portfolio performance:
    - expected_cost: Mean total cost
    - downside_cost: 90th percentile (P90) tail risk cost
    - volatility_exposure: Standard deviation of $/tonne across executions
    """
    cpt = np.asarray(costs_per_tonne, dtype=float)
    tc = np.asarray(total_costs, dtype=float)

    if len(tc) == 0:
        return {
            "expected_cost": 0.0,
            "downside_cost": 0.0,
            "volatility_exposure": 0.0,
        }

    return {
        "expected_cost": round(float(np.mean(tc)), 2),
        "downside_cost": round(float(np.percentile(tc, 90)), 2),
        "volatility_exposure": round(float(np.std(cpt)), 3),
    }

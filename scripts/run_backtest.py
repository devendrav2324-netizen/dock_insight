#!/usr/bin/env python3
"""
Charter-AI — Historical Walk-Forward Backtesting CLI (Phase 11).

Executes comprehensive historical walk-forward backtesting across expanding
historical windows (2019-2021 -> 2022, 2019-2022 -> 2023, 2019-2023 -> 2024).

Rigorously benchmarks CharterAI against 5 commercial baselines:
1. Baseline 1: Current / Last Freight Rate
2. Baseline 2: Simple Moving Average
3. Baseline 3: Always Choose Largest Feasible Vessel
4. Baseline 4: Always Use Spot
5. Baseline 5: Fixed Vessel Class Rule (Panamax)

Generates:
- reports/backtest_report.json
- reports/backtest_report.csv
- reports/BACKTEST_REPORT.md

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --quick
    python scripts/run_backtest.py --years 2022 2023 2024 --output-dir reports
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.backtesting.backtest_engine import BacktestEngine
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CharterAI Historical Walk-Forward Backtesting Engine"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Test years for walk-forward splits (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed historical datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save generated reports (default: reports)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick execution mode (runs single test year 2024)",
    )
    return parser.parse_args()


def print_terminal_summary(report: dict):
    """Prints clean, formatted terminal summary tables."""
    meta = report.get("metadata", {})
    opt = report.get("optimization_evaluation", {})
    comp = report.get("comparative_analysis", {})
    f_summary = report.get("forecast_evaluation", {}).get("freight_summary_by_model", {})

    print("\n" + "=" * 96)
    print(" CharterAI — Historical Walk-Forward Backtest Results")
    print(f" Test Windows: {meta.get('test_years')} | Total Historical Fixtures: {meta.get('scenario_count')}")
    print("=" * 96)

    # 1. Forecast Accuracy
    print("\n--- 1. Multi-Horizon Forecast Accuracy (MAE in $/t) ---")
    print(f"{'Model':<32} {'3-Day':<12} {'7-Day':<12} {'14-Day':<12} {'30-Day':<12} {'14-Day sMAPE'}")
    print("-" * 96)
    for m in ["CharterAI Ensemble", "Baseline 1: Last Rate", "Baseline 2: Moving Average"]:
        h_data = f_summary.get(m, {})
        mae_3 = f"${h_data.get('3d', {}).get('mae', 0):.2f}"
        mae_7 = f"${h_data.get('7d', {}).get('mae', 0):.2f}"
        mae_14 = f"${h_data.get('14d', {}).get('mae', 0):.2f}"
        mae_30 = f"${h_data.get('30d', {}).get('mae', 0):.2f}"
        smape_14 = f"{h_data.get('14d', {}).get('smape', 0):.1f}%"
        print(f"{m:<32} {mae_3:<12} {mae_7:<12} {mae_14:<12} {mae_30:<12} {smape_14}")

    # 2. Strategy Comparison
    print("\n--- 2. Strategy & Baseline Voyage Economics ---")
    print(f"{'Strategy / Baseline':<40} {'Total Cost':<16} {'Cost/t':<10} {'Demurrage':<14} {'Delay':<10} {'Delivery %'}")
    print("-" * 96)
    for s_name, m in opt.items():
        tc = f"${m.get('total_cost', 0):,.0f}"
        cpt = f"${m.get('cost_per_tonne', 0):.2f}"
        dem = f"${m.get('total_demurrage', 0):,.0f}"
        delay = f"{m.get('avg_schedule_delay_days', 0):.1f}d"
        succ = f"{m.get('delivery_success_rate', 0):.1f}%"
        print(f"{s_name:<40} {tc:<16} {cpt:<10} {dem:<14} {delay:<10} {succ}")

    # 3. Savings vs Baselines
    print("\n--- 3. CharterAI Performance Advantage vs Baselines ---")
    print(f"{'Comparison':<40} {'Cost Savings ($)':<20} {'Cost Red. %':<14} {'Demurrage Saved':<18}")
    print("-" * 96)
    for b_name, c in comp.items():
        s_usd = f"${c.get('cost_savings_usd', 0):,.2f}"
        s_pct = f"{c.get('cost_savings_pct', 0):.2f}%"
        d_usd = f"${c.get('demurrage_savings_usd', 0):,.2f} ({c.get('demurrage_reduction_pct', 0):.0f}%)"
        print(f"{b_name:<40} {s_usd:<20} {s_pct:<14} {d_usd:<18}")

    print("\n" + "=" * 96 + "\n")


def main():
    args = parse_args()
    setup_logging(level="INFO")

    years = [2024] if args.quick else args.years

    logger.info("Initializing BacktestEngine (seed=%d, data_dir=%s)...", args.seed, args.data_dir)
    engine = BacktestEngine(data_dir=args.data_dir, seed=args.seed)

    report = engine.run_walk_forward_backtest(years=years)

    paths = engine.export_reports(report, output_dir=args.output_dir)

    print_terminal_summary(report)

    print("Reports successfully generated:")
    for k, p in paths.items():
        print(f"  [{k.upper()}] {p}")


if __name__ == "__main__":
    main()

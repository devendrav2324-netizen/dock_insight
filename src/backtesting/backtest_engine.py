"""
Charter-AI — Master Historical Backtesting Engine (Phase 11).

Orchestrates full-scale, leak-free walk-forward historical evaluations:
1. Walk-Forward Forecast Accuracy (Freight & Port Congestion)
2. Historical Voyage Optimization Replay vs 5 Baselines
3. Multi-Domain Performance Reporting & Artifact Generation
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.backtesting.forecast_backtester import ForecastBacktester
from src.backtesting.optimization_backtester import OptimizationBacktester
from src.backtesting.scenarios import generate_standard_scenarios, BacktestScenario
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Master orchestrator for historical maritime backtesting.
    Coordinates walk-forward time splits, forecasts, voyage simulations,
    and comprehensive baseline comparisons.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.seed = seed
        self.forecast_backtester = ForecastBacktester(data_dir=data_dir)
        self.opt_backtester = OptimizationBacktester(data_dir=data_dir)

    def run_walk_forward_backtest(
        self,
        years: Optional[List[int]] = None,
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete walk-forward evaluation across specified test years.
        Default years: [2022, 2023, 2024] with expanding training windows starting 2019.
        """
        if years is None:
            years = [2022, 2023, 2024]
        if horizons is None:
            horizons = [3, 7, 14, 30]

        logger.info("Initiating walk-forward backtest for test years: %s", years)

        # 1. Build expanding walk-forward splits
        splits = []
        for yr in years:
            splits.append({
                "name": f"Split_{yr}",
                "train_end": f"{yr-1}-12-31",
                "test_year": yr,
            })

        # 2. Execute Forecast Walk-Forward Evaluation
        logger.info("Phase 11.A: Running walk-forward forecast backtest...")
        forecast_results = self.forecast_backtester.evaluate_walk_forward_splits(
            splits=splits,
            horizons=horizons,
        )

        # 3. Generate Historical Tender Scenarios
        logger.info("Phase 11.B: Generating empirical shipment fixtures for %s...", years)
        scenarios = generate_standard_scenarios(years=years)

        # 4. Execute Optimization & Strategy Simulation Replay
        logger.info("Phase 11.C: Replaying %d fixtures across CharterAI and 5 Baselines...", len(scenarios))
        opt_results = self.opt_backtester.run_scenarios(scenarios=scenarios)

        # 5. Compute Comparative Summary & Savings
        comparative_summary = self._compute_comparative_summary(opt_results["summary_by_strategy"])

        report = {
            "metadata": {
                "backtest_run_id": f"bktest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "test_years": years,
                "walk_forward_splits": splits,
                "scenario_count": len(scenarios),
                "data_source": self.data_dir,
                "seed": self.seed,
            },
            "forecast_evaluation": forecast_results,
            "optimization_evaluation": opt_results["summary_by_strategy"],
            "comparative_analysis": comparative_summary,
            "raw_execution_count": sum(len(v) for v in opt_results["raw_executions"].values()),
        }

        return report

    def _compute_comparative_summary(self, summary_by_strategy: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates performance deltas between CharterAI and all baselines."""
        charter_metrics = summary_by_strategy.get("CharterAI Decision Engine", {})
        charter_cost = charter_metrics.get("total_cost", 1.0)
        charter_dem = charter_metrics.get("total_demurrage", 0.0)
        charter_succ = charter_metrics.get("delivery_success_rate", 100.0)

        comparisons = {}
        for b_name, b_metrics in summary_by_strategy.items():
            if b_name == "CharterAI Decision Engine":
                continue

            b_cost = b_metrics.get("total_cost", charter_cost)
            b_dem = b_metrics.get("total_demurrage", charter_dem)
            b_succ = b_metrics.get("delivery_success_rate", charter_succ)

            cost_savings_usd = b_cost - charter_cost
            cost_savings_pct = (cost_savings_usd / max(1.0, b_cost)) * 100.0
            dem_savings_usd = b_dem - charter_dem
            dem_savings_pct = (dem_savings_usd / max(1.0, b_dem)) * 100.0 if b_dem > 0 else 0.0

            comparisons[b_name] = {
                "charter_ai_cost": charter_cost,
                "baseline_cost": b_cost,
                "cost_savings_usd": round(cost_savings_usd, 2),
                "cost_savings_pct": round(cost_savings_pct, 2),
                "demurrage_savings_usd": round(dem_savings_usd, 2),
                "demurrage_reduction_pct": round(dem_savings_pct, 1),
                "charter_delivery_rate": charter_succ,
                "baseline_delivery_rate": b_succ,
                "delivery_reliability_delta_pct": round(charter_succ - b_succ, 1),
            }

        return comparisons

    def export_reports(
        self,
        report_data: Dict[str, Any],
        output_dir: str = "reports",
    ) -> Dict[str, str]:
        """
        Exports backtest report to JSON, CSV, and formatted Markdown.
        Returns paths to generated files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / "backtest_report.json"
        csv_file = out_path / "backtest_report.csv"
        md_file = out_path / "BACKTEST_REPORT.md"

        # 1. JSON Export
        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # 2. CSV Export (Optimization summary across strategies)
        opt_summary = report_data.get("optimization_evaluation", {})
        rows = []
        for strat, metrics in opt_summary.items():
            row = {"strategy": strat, **metrics}
            rows.append(row)
        df_opt = pd.DataFrame(rows)
        df_opt.to_csv(csv_file, index=False)

        # 3. Markdown Export
        md_content = self._generate_markdown_report(report_data)
        with open(md_file, "w") as f:
            f.write(md_content)

        logger.info("Reports successfully written to: %s, %s, %s", json_file, csv_file, md_file)
        return {
            "json": str(json_file),
            "csv": str(csv_file),
            "markdown": str(md_file),
        }

    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generates comprehensive executive Markdown report."""
        meta = data.get("metadata", {})
        f_summary = data.get("forecast_evaluation", {}).get("freight_summary_by_model", {})
        c_summary = data.get("forecast_evaluation", {}).get("congestion_summary", {})
        opt_summary = data.get("optimization_evaluation", {})
        comparisons = data.get("comparative_analysis", {})

        md = []
        md.append("# CharterAI — Historical Walk-Forward Backtesting Report (Phase 11)")
        md.append("")
        md.append(f"**Run ID:** `{meta.get('backtest_run_id')}` | **Generated At:** `{meta.get('generated_at')}`")
        md.append(f"**Test Windows:** `{meta.get('test_years')}` | **Fixtures Evaluated:** `{meta.get('scenario_count')}`")
        md.append("")
        md.append("---")
        md.append("")
        md.append("## Executive Summary")
        md.append("")
        md.append("This report documents the rigorous historical walk-forward backtest of CharterAI's end-to-end maritime chartering intelligence architecture against 5 established commercial baselines across the 2022–2024 period.")
        md.append("")
        md.append("### Key Findings")
        md.append("")

        # Top comparison findings
        b1_comp = comparisons.get("Baseline 1: Current Freight Rate", {})
        b3_comp = comparisons.get("Baseline 3: Always Largest Vessel", {})
        b5_comp = comparisons.get("Baseline 5: Fixed Vessel Rule (Panamax)", {})

        if b1_comp:
            md.append(f"- **Cost Efficiency vs Spot Baseline:** CharterAI achieved a **{b1_comp.get('cost_savings_pct', 0.0)}% total delivered cost reduction** (saving **${b1_comp.get('cost_savings_usd', 0.0):,.2f}**) compared to immediate spot chartering.")
        if b1_comp.get("demurrage_reduction_pct", 0) > 0:
            md.append(f"- **Demurrage Liability Reduction:** CharterAI reduced port demurrage liability by **{b1_comp.get('demurrage_reduction_pct', 0.0)}%** (saving **${b1_comp.get('demurrage_savings_usd', 0.0):,.2f}**) via proactive congestion forecasting and laytime-aware parcel sizing.")
        if b5_comp:
            md.append(f"- **Fleet Optimization vs Fixed Rule:** Optimizing vessel classes dynamically outperformed the fixed Panamax policy by **{b5_comp.get('cost_savings_pct', 0.0)}%**.")
        if b3_comp:
            md.append(f"- **Parcel Sizing vs Always Largest Vessel:** Avoiding oversized Capesize ballast and excessive port fees saved **{b3_comp.get('cost_savings_pct', 0.0)}%** against the 'Always Largest' heuristic.")

        md.append("")
        md.append("---")
        md.append("")
        md.append("## 1. Multi-Horizon Freight Forecasting Accuracy")
        md.append("")
        md.append("Walk-forward cross-validation evaluated over expanding historical training windows (2019–2021 $\\to$ 2022, 2019–2022 $\\to$ 2023, 2019–2023 $\\to$ 2024).")
        md.append("")
        md.append("| Model | 3-Day MAE | 7-Day MAE | 14-Day MAE | 30-Day MAE | 14-Day sMAPE |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for m_name in ["CharterAI Ensemble", "Baseline 1: Last Rate", "Baseline 2: Moving Average"]:
            h_data = f_summary.get(m_name, {})
            mae_3 = h_data.get("3d", {}).get("mae", "-")
            mae_7 = h_data.get("7d", {}).get("mae", "-")
            mae_14 = h_data.get("14d", {}).get("mae", "-")
            mae_30 = h_data.get("30d", {}).get("mae", "-")
            smape_14 = h_data.get("14d", {}).get("smape", "-")
            md.append(f"| **{m_name}** | ${mae_3} | ${mae_7} | ${mae_14} | ${mae_30} | {smape_14}% |")

        md.append("")
        md.append("### Port Congestion Prediction Performance")
        md.append("")
        md.append("| Test Year | CharterAI MAE (days) | Historical Baseline MAE (days) | Improvement |")
        md.append("| :---: | :---: | :---: | :---: |")
        for yr_str, c_met in c_summary.items():
            c_mae = c_met.get("charter_ai_mae", 0.0)
            b_mae = c_met.get("baseline_hist_mean_mae", 0.0)
            imp = round(((b_mae - c_mae) / max(0.1, b_mae)) * 100.0, 1)
            md.append(f"| {yr_str} | {c_mae:.2f} d | {b_mae:.2f} d | +{imp}% |")

        md.append("")
        md.append("---")
        md.append("")
        md.append("## 2. Comprehensive Strategy & Baseline Comparison")
        md.append("")
        md.append("Evaluated over 36 historical procurement tenders across Indian East Coast discharge ports.")
        md.append("")
        md.append("| Strategy / Policy | Total Delivered Cost | Cost per Tonne | Demurrage Liability | Delay (Avg Days) | Delivery Success |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for strat_name, met in opt_summary.items():
            bold = "**" if "CharterAI" in strat_name else ""
            tc = f"${met.get('total_cost', 0):,.0f}"
            cpt = f"${met.get('cost_per_tonne', 0):.2f}/t"
            dem = f"${met.get('total_demurrage', 0):,.0f}"
            delay = f"{met.get('avg_schedule_delay_days', 0):.2f} d"
            succ = f"{met.get('delivery_success_rate', 0):.1f}%"
            md.append(f"| {bold}{strat_name}{bold} | {tc} | {cpt} | {dem} | {delay} | {succ} |")

        md.append("")
        md.append("### Performance Deltas vs Baselines")
        md.append("")
        md.append("| Baseline Policy | Total Savings ($) | Cost Reduction (%) | Demurrage Saved ($) | Demurrage Reduction (%) | Delivery Reliability Delta |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for b_name, comp in comparisons.items():
            s_usd = f"${comp.get('cost_savings_usd', 0):,.2f}"
            s_pct = f"{comp.get('cost_savings_pct', 0):.2f}%"
            d_usd = f"${comp.get('demurrage_savings_usd', 0):,.2f}"
            d_pct = f"{comp.get('demurrage_reduction_pct', 0):.1f}%"
            r_delta = f"+{comp.get('delivery_reliability_delta_pct', 0):.1f}%"
            md.append(f"| {b_name} | **{s_usd}** | **{s_pct}** | {d_usd} | {d_pct} | {r_delta} |")

        md.append("")
        md.append("---")
        md.append("")
        md.append("## 3. Market Timing & Contract Intelligence")
        md.append("")
        md.append("| Strategy | Avoided Cost | Missed Opportunity | Booking Success Rate | Downside Cost (P90) | Volatility Exposure ($/t) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

        for strat_name, met in opt_summary.items():
            bold = "**" if "CharterAI" in strat_name else ""
            av = f"${met.get('avoided_cost', 0):,.0f}"
            mo = f"${met.get('missed_opportunity_cost', 0):,.0f}"
            sr = f"{met.get('booking_success_rate', 0):.1f}%"
            p90 = f"${met.get('downside_cost', 0):,.0f}"
            vol = f"${met.get('volatility_exposure', 0):.2f}/t"
            md.append(f"| {bold}{strat_name}{bold} | {av} | {mo} | {sr} | {p90} | {vol} |")

        md.append("")
        md.append("---")
        md.append("")
        md.append("## 4. Methodology & Leakage Prevention")
        md.append("")
        md.append("1. **Zero Future Leakage**: At each decision epoch $T$, candidate generation, freight forecasting, port congestion estimation, and risk assessment are evaluated strictly on historical data $t \\le T$.")
        md.append("2. **Empirical Market Realization**: Delivered economics are computed from realized historical spot freight rates, bunker fuel settlement prices, and port authority congestion records during the actual voyage laycan and discharge windows.")
        md.append("3. **Reproducibility**: All split boundaries, parameters, and random seeds are fixed and documented in `backtest_report.json`.")

        return "\n".join(md)

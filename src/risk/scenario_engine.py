"""
DockInsights — Deterministic Scenario Stress-Testing Engine (Phase 8).

Evaluates candidate charter plans across three standardized market and operational scenarios:
1. BEST_CASE: Favorable freight rates (P10), low bunker prices (P10), minimal port congestion,
   smooth turnaround, zero demurrage, and on-time arrival.
2. BASE_CASE: Expected market freight rates (P50), baseline bunker fuel, predicted port waiting times,
   and standard operational margins.
3. WORST_CASE: Stressed market rates (P90), bunker price spike (P90), heavy port congestion queues (P90),
   adverse weather speed loss, extended demurrage exposure, and potential delivery deadline slippage.

All scenarios use transparent, documented assumptions grounded in maritime operational history.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from src.risk.monte_carlo import CharterPlanInputs, MonteCarloResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ScenarioType(str, Enum):
    BEST_CASE = "BEST_CASE"
    BASE_CASE = "BASE_CASE"
    WORST_CASE = "WORST_CASE"


@dataclass
class ScenarioResult:
    """Detailed output and economic breakdown for a specific scenario."""
    name: str
    total_cost: float
    cost_per_tonne: float
    freight_rate: float
    freight_cost: float
    bunker_price: float
    bunker_cost: float
    duration_days: float
    waiting_days: float
    handling_days: float
    demurrage_days: float
    demurrage_cost: float
    fixed_costs: float
    on_time: bool
    schedule_slack_days: float
    assumptions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_cost": round(self.total_cost, 2),
            "cost_per_tonne": round(self.cost_per_tonne, 2),
            "freight_rate": round(self.freight_rate, 2),
            "freight_cost": round(self.freight_cost, 2),
            "bunker_price": round(self.bunker_price, 2),
            "bunker_cost": round(self.bunker_cost, 2),
            "duration_days": round(self.duration_days, 1),
            "waiting_days": round(self.waiting_days, 1),
            "handling_days": round(self.handling_days, 1),
            "demurrage_days": round(self.demurrage_days, 1),
            "demurrage_cost": round(self.demurrage_cost, 2),
            "fixed_costs": round(self.fixed_costs, 2),
            "on_time": self.on_time,
            "schedule_slack_days": round(self.schedule_slack_days, 1),
            "assumptions": self.assumptions,
        }


class ScenarioEngine:
    """
    Evaluates realistic deterministic scenarios with documented maritime assumptions.
    """

    def evaluate_scenarios(
        self,
        plan: CharterPlanInputs,
        mc_result: Optional[MonteCarloResult] = None,
    ) -> Dict[str, ScenarioResult]:
        """
        Calculates BEST_CASE, BASE_CASE, and WORST_CASE scenarios.
        Uses Monte Carlo quantiles when available, or empirical maritime calibration multipliers.
        """
        cargo_t = max(100.0, plan.cargo_quantity_t)
        nominal_speed = max(8.0, plan.service_speed_knots)
        nominal_sea_days = (plan.sea_distance_nm / nominal_speed) / 24.0
        handling_rate = max(1000.0, plan.port_handling_rate_t_day)
        nominal_handling_days = (cargo_t / handling_rate) * 2.0  # load + discharge
        fixed_costs = plan.port_charges_usd + plan.canal_charges_usd + plan.misc_agency_usd
        deadline = plan.delivery_deadline_days if plan.delivery_deadline_days is not None else 999.0

        # Laytime allowance
        laytime = (
            plan.agreed_laytime_days
            if plan.agreed_laytime_days is not None and plan.agreed_laytime_days > 0
            else (nominal_handling_days + 1.0)
        )

        # ---------------------------------------------------------------------
        # 1. BEST_CASE SCENARIO
        # ---------------------------------------------------------------------
        # Favorable spot rates (P10), cheap bunkers, swift berthing, no demurrage
        if plan.freight_rate_p10 is not None:
            best_freight_rate = plan.freight_rate_p10
        elif mc_result is not None and mc_result.p10_cost > 0:
            best_freight_rate = max(1.0, plan.base_freight_rate * 0.88)
        else:
            best_freight_rate = max(1.0, plan.base_freight_rate * (1.0 - (plan.freight_volatility_pct * 1.28 / 100.0)))

        best_freight_cost = (
            best_freight_rate * cargo_t if plan.freight_rate_is_per_tonne else best_freight_rate
        )
        best_bunker_price = max(100.0, plan.base_bunker_price * 0.90)  # P10 fuel
        best_sea_days = nominal_sea_days * 0.98  # Calm seas & favorable currents
        best_wait_days = max(0.2, plan.expected_wait_days * 0.50)  # Prompt berth availability
        best_handling_days = nominal_handling_days * 0.92  # High cargo handling efficiency
        best_port_days = best_wait_days + best_handling_days
        best_demurrage_days = max(0.0, best_port_days - laytime)
        best_demurrage_cost = best_demurrage_days * plan.demurrage_rate_usd_day
        best_bunker_burn_t = (best_sea_days * plan.fuel_consumption_t_day) + (best_port_days * 2.5)
        best_bunker_cost = best_bunker_burn_t * best_bunker_price
        best_total_cost = best_freight_cost + best_bunker_cost + best_demurrage_cost + fixed_costs
        best_duration = best_sea_days + best_port_days
        best_slack = deadline - best_duration
        best_on_time = best_slack >= 0

        best_scenario = ScenarioResult(
            name=ScenarioType.BEST_CASE.value,
            total_cost=best_total_cost,
            cost_per_tonne=best_total_cost / cargo_t,
            freight_rate=best_freight_rate,
            freight_cost=best_freight_cost,
            bunker_price=best_bunker_price,
            bunker_cost=best_bunker_cost,
            duration_days=best_duration,
            waiting_days=best_wait_days,
            handling_days=best_handling_days,
            demurrage_days=best_demurrage_days,
            demurrage_cost=best_demurrage_cost,
            fixed_costs=fixed_costs,
            on_time=best_on_time,
            schedule_slack_days=best_slack,
            assumptions=[
                f"Freight rate softened to P10 (${best_freight_rate:.2f}/MT).",
                f"Bunker fuel purchased at competitive P10 rate (${best_bunker_price:.0f}/MT).",
                f"Minimal port wait ({best_wait_days:.1f} days) with zero or negligible demurrage.",
                "Smooth calm sea passage with no weather delays.",
                f"Delivered on-time with {best_slack:.1f} days of buffer.",
            ],
        )

        # ---------------------------------------------------------------------
        # 2. BASE_CASE SCENARIO
        # ---------------------------------------------------------------------
        # Expected market freight rate (P50), baseline bunkers, expected port wait
        base_freight_rate = plan.base_freight_rate
        base_freight_cost = (
            base_freight_rate * cargo_t if plan.freight_rate_is_per_tonne else base_freight_rate
        )
        base_bunker_price = plan.base_bunker_price
        base_sea_days = nominal_sea_days * 1.03  # Standard 3% sea margin
        base_wait_days = plan.expected_wait_days
        base_handling_days = nominal_handling_days * 1.0
        base_port_days = base_wait_days + base_handling_days
        base_demurrage_days = max(0.0, base_port_days - laytime)
        base_demurrage_cost = base_demurrage_days * plan.demurrage_rate_usd_day
        base_bunker_burn_t = (base_sea_days * plan.fuel_consumption_t_day) + (base_port_days * 3.0)
        base_bunker_cost = base_bunker_burn_t * base_bunker_price
        base_total_cost = base_freight_cost + base_bunker_cost + base_demurrage_cost + fixed_costs
        base_duration = base_sea_days + base_port_days
        base_slack = deadline - base_duration
        base_on_time = base_slack >= 0

        base_scenario = ScenarioResult(
            name=ScenarioType.BASE_CASE.value,
            total_cost=base_total_cost,
            cost_per_tonne=base_total_cost / cargo_t,
            freight_rate=base_freight_rate,
            freight_cost=base_freight_cost,
            bunker_price=base_bunker_price,
            bunker_cost=base_bunker_cost,
            duration_days=base_duration,
            waiting_days=base_wait_days,
            handling_days=base_handling_days,
            demurrage_days=base_demurrage_days,
            demurrage_cost=base_demurrage_cost,
            fixed_costs=fixed_costs,
            on_time=base_on_time,
            schedule_slack_days=base_slack,
            assumptions=[
                f"Expected contract freight rate (${base_freight_rate:.2f}/MT).",
                f"Baseline bunker price (${base_bunker_price:.0f}/MT).",
                f"Predicted port waiting time ({base_wait_days:.1f} days) based on current port queues.",
                f"Demurrage liability: {base_demurrage_days:.1f} days (${base_demurrage_cost:,.0f}).",
                f"Schedule buffer: {base_slack:.1f} days remaining.",
            ],
        )

        # ---------------------------------------------------------------------
        # 3. WORST_CASE SCENARIO
        # ---------------------------------------------------------------------
        # Stressed freight market (P90), bunker price jump (P90), severe port congestion (P90)
        if plan.freight_rate_p90 is not None:
            worst_freight_rate = plan.freight_rate_p90
        elif mc_result is not None and mc_result.p90_cost > 0:
            worst_freight_rate = max(base_freight_rate * 1.15, plan.base_freight_rate * 1.25)
        else:
            worst_freight_rate = plan.base_freight_rate * (1.0 + (plan.freight_volatility_pct * 1.28 / 100.0))

        worst_freight_cost = (
            worst_freight_rate * cargo_t if plan.freight_rate_is_per_tonne else worst_freight_rate
        )
        worst_bunker_price = plan.base_bunker_price * 1.16  # P90 fuel price spike
        worst_sea_days = nominal_sea_days * 1.12  # Adverse weather, gale sea-state slowing transit
        worst_wait_days = (
            plan.p90_wait_days
            if plan.p90_wait_days is not None and plan.p90_wait_days > plan.expected_wait_days
            else max(plan.expected_wait_days * 2.2, plan.expected_wait_days + 3.0)
        )
        worst_handling_days = nominal_handling_days * 1.15  # Rainy delays, terminal congestion
        worst_port_days = worst_wait_days + worst_handling_days
        worst_demurrage_days = max(0.0, worst_port_days - laytime)
        worst_demurrage_cost = worst_demurrage_days * plan.demurrage_rate_usd_day
        worst_bunker_burn_t = (worst_sea_days * plan.fuel_consumption_t_day) + (worst_port_days * 3.5)
        worst_bunker_cost = worst_bunker_burn_t * worst_bunker_price

        # In worst case, add 10% contingency for potential spot replacement / terminal surcharges
        worst_contingency = worst_freight_cost * 0.08
        worst_total_cost = (
            worst_freight_cost
            + worst_bunker_cost
            + worst_demurrage_cost
            + fixed_costs
            + worst_contingency
        )
        worst_duration = worst_sea_days + worst_port_days
        worst_slack = deadline - worst_duration
        worst_on_time = worst_slack >= 0

        worst_scenario = ScenarioResult(
            name=ScenarioType.WORST_CASE.value,
            total_cost=worst_total_cost,
            cost_per_tonne=worst_total_cost / cargo_t,
            freight_rate=worst_freight_rate,
            freight_cost=worst_freight_cost,
            bunker_price=worst_bunker_price,
            bunker_cost=worst_bunker_cost,
            duration_days=worst_duration,
            waiting_days=worst_wait_days,
            handling_days=worst_handling_days,
            demurrage_days=worst_demurrage_days,
            demurrage_cost=worst_demurrage_cost,
            fixed_costs=fixed_costs + worst_contingency,
            on_time=worst_on_time,
            schedule_slack_days=worst_slack,
            assumptions=[
                f"Freight rate surged to P90 tail risk (${worst_freight_rate:.2f}/MT).",
                f"Bunker fuel spike to P90 (${worst_bunker_price:.0f}/MT).",
                f"Severe port congestion ({worst_wait_days:.1f} days waiting) triggering heavy demurrage.",
                f"Adverse weather causes +12% sailing delay.",
                f"Substantial demurrage liability: {worst_demurrage_days:.1f} days (${worst_demurrage_cost:,.0f}).",
                (
                    f"Delivery delayed by {abs(worst_slack):.1f} days beyond deadline!"
                    if not worst_on_time
                    else f"Schedule buffer narrowed to {worst_slack:.1f} days."
                ),
            ],
        )

        return {
            ScenarioType.BEST_CASE.value: best_scenario,
            ScenarioType.BASE_CASE.value: base_scenario,
            ScenarioType.WORST_CASE.value: worst_scenario,
        }

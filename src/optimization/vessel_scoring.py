"""
Charter-AI — Multi-Criteria Soft Objective Scoring Engine (Phase 6).

Evaluates candidate charter plans across multiple soft objectives:
1. Delivered cost per tonne ($/MT)
2. Total schedule duration & delivery deadline confidence
3. Operational and congestion risk
4. Cargo deadweight capacity utilization
5. Port waiting and demurrage liability exposure

Objective weights are user-configurable and explicitly not claimed as an industry standard.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from src.optimization.voyage_plan import CharterPlan
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationWeights:
    """
    Configurable weights for soft multi-criteria objective optimization.
    Prioritizes delivered cost efficiency from the charterer's perspective.
    """
    cost_weight: float = 0.50
    schedule_weight: float = 0.20
    risk_weight: float = 0.15
    utilization_weight: float = 0.05
    demurrage_weight: float = 0.10

    def __post_init__(self):
        # Normalize weights so they sum to 1.0
        tot = (
            self.cost_weight
            + self.schedule_weight
            + self.risk_weight
            + self.utilization_weight
            + self.demurrage_weight
        )
        if tot > 0 and abs(tot - 1.0) > 1e-4:
            self.cost_weight /= tot
            self.schedule_weight /= tot
            self.risk_weight /= tot
            self.utilization_weight /= tot
            self.demurrage_weight /= tot


class VesselScoringEngine:
    """
    Calculates multi-criteria soft scores (0 to 100) for candidate charter plans.
    """

    def __init__(self, weights: Optional[OptimizationWeights] = None):
        self.weights = weights or OptimizationWeights()

    def score_plans(self, plans: List[CharterPlan]) -> List[CharterPlan]:
        """
        Scores and ranks a collection of candidate plans using relative and absolute metrics.
        Infeasible plans automatically receive a composite score of 0.0.
        """
        feasible_plans = [p for p in plans if p.feasibility]
        if not feasible_plans:
            for p in plans:
                p.score = 0.0
            return plans

        # ---------------------------------------------------------
        # 1. Compute Min/Max Extrema Across Feasible Candidates
        # ---------------------------------------------------------
        costs_per_t = [p.cost_per_tonne for p in feasible_plans]
        durations = [p.total_duration for p in feasible_plans]

        min_cpt, max_cpt = min(costs_per_t), max(costs_per_t)
        min_dur, max_dur = min(durations), max(durations)

        cost_range = max(1.0, max_cpt - min_cpt)
        dur_range = max(1.0, max_dur - min_dur)

        # ---------------------------------------------------------
        # 2. Score Each Plan Individually
        # ---------------------------------------------------------
        for plan in plans:
            if not plan.feasibility:
                plan.score = 0.0
                plan.cost_score = 0.0
                plan.schedule_score = 0.0
                plan.risk_sub_score = 0.0
                plan.utilization_score = 0.0
                plan.demurrage_score = 0.0
                continue

            # 2.1 Cost Score (Lower delivered $/t -> higher score)
            # Relative score scaled between 50 and 100 for feasible set
            cost_efficiency = (max_cpt - plan.cost_per_tonne) / cost_range
            cost_score = 60.0 + (cost_efficiency * 40.0)

            # 2.2 Schedule Score (Shorter duration + higher on-time probability)
            dur_efficiency = (max_dur - plan.total_duration) / dur_range
            schedule_score = (dur_efficiency * 50.0) + (plan.delivery_probability * 50.0)

            # 2.3 Risk Score (Dynamic from Risk Engine: lower risk -> higher score)
            # plan.risk_score is 0-100 (where 0 is low risk, 100 is critical)
            risk_sub_score = max(0.0, min(100.0, 100.0 - plan.risk_score))

            # 2.4 Cargo Utilization Score (Capacity efficiency)
            # Penalizes deadweight waste or severe underloading
            util_val = plan.utilization
            if util_val <= 1.0:
                utilization_score = util_val * 100.0
            else:
                utilization_score = max(0.0, 100.0 - ((util_val - 1.0) * 100.0))

            # 2.5 Demurrage Score (Lower demurrage exposure -> higher score)
            dem_ratio = plan.expected_demurrage / max(1.0, plan.total_cost)
            demurrage_score = max(0.0, min(100.0, 100.0 - (dem_ratio * 300.0)))

            # ---------------------------------------------------------
            # 3. Composite Weighted Multi-Criteria Score
            # ---------------------------------------------------------
            composite = (
                (cost_score * self.weights.cost_weight)
                + (schedule_score * self.weights.schedule_weight)
                + (risk_sub_score * self.weights.risk_weight)
                + (utilization_score * self.weights.utilization_weight)
                + (demurrage_score * self.weights.demurrage_weight)
            )

            plan.cost_score = round(cost_score, 1)
            plan.schedule_score = round(schedule_score, 1)
            plan.risk_sub_score = round(risk_sub_score, 1)
            plan.utilization_score = round(utilization_score, 1)
            plan.demurrage_score = round(demurrage_score, 1)
            plan.score = round(composite, 1)

            # Add human-readable reason highlights
            if cost_score > 85:
                plan.reasons.append("Highly competitive delivered freight cost.")
            if utilization_score > 85:
                plan.reasons.append("Optimal vessel deadweight capacity utilization.")
            if plan.expected_demurrage == 0:
                plan.reasons.append("Minimal demurrage exposure.")
            if schedule_score > 85:
                plan.reasons.append("Fast port turnaround and high schedule reliability.")

        # Sort plans: feasible first by score descending, then infeasible
        plans.sort(key=lambda p: (1 if p.feasibility else 0, p.score), reverse=True)
        return plans

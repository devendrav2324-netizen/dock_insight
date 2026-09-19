"""
Charter-AI — Voyage Plan & Fleet Allocation Data Models (Phase 6).

Represents individual voyage legs and full multi-voyage charter plans
supporting homogeneous fleets, mixed vessel classes, single and multiple voyages.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VoyageLeg:
    """
    A single voyage leg within a charter plan.
    """
    leg_id: int
    vessel_class: str
    cargo_quantity_t: float
    vessel_dwt: float
    utilization: float  # cargo_quantity_t / vessel_dwt
    origin_port_id: str
    destination_port_id: str
    route_distance_nm: float
    sailing_days: float
    loading_days: float
    discharge_days: float
    waiting_days: float
    total_leg_days: float

    # Financial breakdown (from Voyage Economics Engine)
    freight_cost: float
    bunker_cost: float
    port_cost: float
    waiting_cost: float
    demurrage_exposure: float
    positioning_cost: float
    miscellaneous_cost: float
    total_cost: float
    cost_per_tonne: float


@dataclass
class CharterPlan:
    """
    A complete charter plan covering the entire cargo requirement.
    Can consist of 1 vessel doing 1 voyage, 1 vessel doing multiple sequential voyages,
    or multiple vessels operating in parallel.
    """
    plan_id: str
    vessel_classes: List[str]
    number_of_vessels: int
    number_of_voyages: int
    execution_mode: str = "sequential"  # "sequential" or "parallel"
    legs: List[VoyageLeg] = field(default_factory=list)
    cargo_per_voyage: List[float] = field(default_factory=list)
    total_cargo_t: float = 0.0
    utilization: float = 0.0  # Weighted average deadweight utilization (0.0 to 1.0)

    # Aggregated Financial Metrics
    total_freight_cost: float = 0.0
    bunker_cost: float = 0.0
    port_cost: float = 0.0
    waiting_cost: float = 0.0
    expected_demurrage: float = 0.0
    positioning_cost: float = 0.0
    miscellaneous_cost: float = 0.0
    total_cost: float = 0.0
    cost_per_tonne: float = 0.0

    # Operational & Schedule Metrics
    total_duration: float = 0.0  # Total elapsed calendar days from commencement to final discharge
    demurrage_probability: float = 0.0  # Likelihood of incurring port demurrage
    delivery_probability: float = 0.95  # Confidence of meeting delivery deadline
    risk_score: float = 0.0  # Dynamic risk score (0 to 100, lower is better)

    # Phase 8: Probabilistic & Scenario Analytics
    monte_carlo: Optional[Dict[str, Any]] = None
    scenarios: Optional[Dict[str, Any]] = None

    # Hard Constraints & Feasibility
    feasibility: bool = True
    failed_constraints: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    # Soft Multi-Criteria Scoring (0 to 100, higher is better)
    score: float = 0.0
    cost_score: float = 0.0
    schedule_score: float = 0.0
    risk_sub_score: float = 0.0
    utilization_score: float = 0.0
    demurrage_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Canonical dictionary output matching Phase 6 requirements.
        """
        return {
            "plan_id": self.plan_id,
            "vessel_classes": self.vessel_classes,
            "number_of_vessels": self.number_of_vessels,
            "number_of_voyages": self.number_of_voyages,
            "total_cost": round(self.total_cost, 2),
            "cost_per_tonne": round(self.cost_per_tonne, 2),
            "utilization": round(self.utilization, 4),
            "total_duration": round(self.total_duration, 2),
            "demurrage_probability": round(self.demurrage_probability, 4),
            "delivery_probability": round(self.delivery_probability, 4),
            "risk_score": round(self.risk_score, 2),
            "feasibility": self.feasibility,
            "reasons": self.reasons,
        }

    def to_detailed_dict(self) -> Dict[str, Any]:
        """
        Detailed output including itemized leg economics and scoring breakdowns.
        """
        base = self.to_dict()
        base.update({
            "execution_mode": self.execution_mode,
            "cargo_per_voyage": self.cargo_per_voyage,
            "total_cargo_t": self.total_cargo_t,
            "total_freight_cost": round(self.total_freight_cost, 2),
            "bunker_cost": round(self.bunker_cost, 2),
            "port_cost": round(self.port_cost, 2),
            "waiting_cost": round(self.waiting_cost, 2),
            "expected_demurrage": round(self.expected_demurrage, 2),
            "positioning_cost": round(self.positioning_cost, 2),
            "miscellaneous_cost": round(self.miscellaneous_cost, 2),
            "failed_constraints": self.failed_constraints,
            "score": round(self.score, 2),
            "sub_scores": {
                "cost_score": round(self.cost_score, 2),
                "schedule_score": round(self.schedule_score, 2),
                "risk_sub_score": round(self.risk_sub_score, 2),
                "utilization_score": round(self.utilization_score, 2),
                "demurrage_score": round(self.demurrage_score, 2),
            }
        })
        if self.monte_carlo:
            base["monte_carlo"] = self.monte_carlo
        if self.scenarios:
            base["scenarios"] = self.scenarios
        return base

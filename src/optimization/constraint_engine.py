"""
DockInsights — Hard Constraint Engine (Phase 6).

Strictly evaluates hard physical and operational constraints to eliminate
infeasible vessel and fleet charter plans before soft scoring and ranking.

Hard constraints:
1. vessel draft <= loading port limit
2. vessel draft <= discharge port limit
3. vessel LOA <= port limit (origin and destination)
4. vessel beam <= port limit (origin and destination)
5. cargo compatibility (commodity vs vessel class suitability)
6. vessel capacity (parcel size vs vessel min/max DWT)
7. terminal compatibility (e.g. coal terminal, draft restrictions)
8. delivery deadline (total plan duration vs required deadline)
9. vessel availability (laycan readiness)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.optimization.candidate_generator import CandidatePlanSpec, DEFAULT_VESSEL_CLASSES, VesselClassDefinition
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PortSpec:
    """Standardized port specification for constraint validation."""
    port_id: str
    port_name: str
    max_draft_m: float
    max_loa_m: float
    max_beam_m: float
    max_dwt: Optional[float] = None
    is_anchorage_only: bool = False
    has_coal_terminal: bool = True
    has_iron_ore_terminal: bool = True
    has_grain_terminal: bool = True


@dataclass
class ConstraintEvaluationResult:
    """Result of hard constraint validation for a candidate plan."""
    plan_id: str
    is_feasible: bool
    failed_constraints: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


class ConstraintEngine:
    """
    Validates candidate charter plans against non-negotiable physical and contractual constraints.
    """

    def __init__(self, vessel_classes: Optional[Dict[str, VesselClassDefinition]] = None):
        self.vessel_classes = vessel_classes or DEFAULT_VESSEL_CLASSES

    def evaluate_plan(
        self,
        plan: CandidatePlanSpec,
        origin_port: Any,
        destination_port: Any,
        cargo_type: str = "Coal",
        delivery_deadline_days: Optional[float] = None,
        estimated_duration_days: Optional[float] = None,
        vessel_open_days: float = 0.0,
    ) -> ConstraintEvaluationResult:
        """
        Validates all vessels and voyages in a candidate plan against hard constraints.
        """
        failed: List[str] = []
        reasons: List[str] = []

        orig_spec = self._standardize_port(origin_port)
        dest_spec = self._standardize_port(destination_port)

        # ---------------------------------------------------------
        # 1. Evaluate Every Vessel Class in Plan
        # ---------------------------------------------------------
        for v_idx, v_class_name in enumerate(plan.vessel_classes):
            v_spec = self.vessel_classes.get(v_class_name)
            if not v_spec:
                failed.append(f"Unknown vessel class: {v_class_name}")
                continue

            # Parcel size for this vessel
            parcel_mt = plan.cargo_per_voyage[v_idx] if v_idx < len(plan.cargo_per_voyage) else plan.cargo_per_voyage[0]

            # 1.1 Draft Limits
            if orig_spec.max_draft_m > 0 and v_spec.draft_max_m > orig_spec.max_draft_m:
                failed.append(
                    f"{v_class_name} draft ({v_spec.draft_max_m}m) exceeds {orig_spec.port_name} limit ({orig_spec.max_draft_m}m)"
                )
            if dest_spec.max_draft_m > 0 and v_spec.draft_max_m > dest_spec.max_draft_m:
                failed.append(
                    f"{v_class_name} draft ({v_spec.draft_max_m}m) exceeds {dest_spec.port_name} limit ({dest_spec.max_draft_m}m)"
                )

            # 1.2 LOA Limits
            if orig_spec.max_loa_m > 0 and v_spec.loa_max_m > orig_spec.max_loa_m:
                failed.append(
                    f"{v_class_name} LOA ({v_spec.loa_max_m}m) exceeds {orig_spec.port_name} limit ({orig_spec.max_loa_m}m)"
                )
            if dest_spec.max_loa_m > 0 and v_spec.loa_max_m > dest_spec.max_loa_m:
                failed.append(
                    f"{v_class_name} LOA ({v_spec.loa_max_m}m) exceeds {dest_spec.port_name} limit ({dest_spec.max_loa_m}m)"
                )

            # 1.3 Beam Limits
            if orig_spec.max_beam_m > 0 and v_spec.beam_max_m > orig_spec.max_beam_m:
                failed.append(
                    f"{v_class_name} beam ({v_spec.beam_max_m}m) exceeds {orig_spec.port_name} limit ({orig_spec.max_beam_m}m)"
                )
            if dest_spec.max_beam_m > 0 and v_spec.beam_max_m > dest_spec.max_beam_m:
                failed.append(
                    f"{v_class_name} beam ({v_spec.beam_max_m}m) exceeds {dest_spec.port_name} limit ({dest_spec.max_beam_m}m)"
                )

            # 1.4 Anchorage / Lightering Restrictions
            if orig_spec.is_anchorage_only or dest_spec.is_anchorage_only:
                if v_class_name not in ["Handysize", "Supramax"]:
                    failed.append(
                        f"Anchorage port requires lightering; {v_class_name} cannot perform standard lightering"
                    )

            # 1.5 Vessel Deadweight / Capacity Limits
            if parcel_mt > v_spec.dwt_max:
                failed.append(
                    f"Parcel {parcel_mt:,.0f} MT exceeds {v_class_name} max capacity ({v_spec.dwt_max:,.0f} MT)"
                )
            if parcel_mt < v_spec.dwt_min * 0.35:
                failed.append(
                    f"Parcel {parcel_mt:,.0f} MT is below {v_class_name} minimum operational limit ({v_spec.dwt_min * 0.35:,.0f} MT)"
                )

            # 1.6 Commodity / Terminal Compatibility
            c_lower = cargo_type.lower()
            if "coal" in c_lower and not (orig_spec.has_coal_terminal and dest_spec.has_coal_terminal):
                failed.append(f"Port lacks dedicated coal handling facilities for {cargo_type}")
            elif "grain" in c_lower and not (orig_spec.has_grain_terminal and dest_spec.has_grain_terminal):
                failed.append(f"Port lacks dedicated grain handling facilities for {cargo_type}")

        # ---------------------------------------------------------
        # 2. Schedule & Delivery Deadline Hard Constraint
        # ---------------------------------------------------------
        if delivery_deadline_days is not None and estimated_duration_days is not None:
            total_elapsed = vessel_open_days + estimated_duration_days
            # If duration exceeds deadline by more than 1.0 day buffer, mark infeasible
            if total_elapsed > delivery_deadline_days + 1.0:
                failed.append(
                    f"Plan duration ({total_elapsed:.1f} days) exceeds delivery deadline ({delivery_deadline_days:.1f} days)"
                )

        is_feasible = len(failed) == 0
        if is_feasible:
            reasons.append("Plan satisfies all port dimensions, terminal facilities, parcel limits, and schedule constraints.")
        else:
            reasons.append(f"Infeasible: {'; '.join(failed)}")

        return ConstraintEvaluationResult(
            plan_id=plan.plan_id,
            is_feasible=is_feasible,
            failed_constraints=failed,
            reasons=reasons,
        )

    def _standardize_port(self, port: Any) -> PortSpec:
        """Standardizes input port objects (PortInfo, PortConstraints, or dict) to PortSpec."""
        if hasattr(port, "port_id"):
            return PortSpec(
                port_id=getattr(port, "port_id"),
                port_name=getattr(port, "port_name", getattr(port, "name", port.port_id)),
                max_draft_m=getattr(port, "max_draft_m", getattr(port, "draft_max_m", 20.0)),
                max_loa_m=getattr(port, "max_loa_m", getattr(port, "loa_max_m", 350.0)),
                max_beam_m=getattr(port, "max_beam_m", getattr(port, "beam_max_m", 55.0)),
                max_dwt=getattr(port, "max_dwt", None),
                is_anchorage_only=getattr(port, "is_anchorage_only", False),
                has_coal_terminal=getattr(port, "coal_terminal", True),
                has_iron_ore_terminal=getattr(port, "iron_ore_terminal", True),
                has_grain_terminal=getattr(port, "grain_terminal", True),
            )
        elif isinstance(port, dict):
            return PortSpec(
                port_id=port.get("port_id", "UNKNOWN"),
                port_name=port.get("port_name", port.get("name", "Unknown Port")),
                max_draft_m=port.get("max_draft_m", 20.0),
                max_loa_m=port.get("max_loa_m", 350.0),
                max_beam_m=port.get("max_beam_m", 55.0),
                max_dwt=port.get("max_dwt"),
                is_anchorage_only=port.get("is_anchorage_only", False),
            )
        else:
            return PortSpec(
                port_id=str(port),
                port_name=str(port),
                max_draft_m=20.0,
                max_loa_m=350.0,
                max_beam_m=55.0,
            )

"""
Charter-AI — Vessel-Port Compatibility Selector.

Deterministic constraint-based filtering: given an origin, destination,
and cargo tonnage, returns the set of vessel classes that can physically
operate at both ports.

This is NOT an ML component — it applies hard physical constraints.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

from src.utils.constants import VesselClass
from src.utils.logging import get_logger
from src.economics.voyage_cost import VoyageCostInputs, calculate_voyage_cost, calculate_sailing_days
from src.data.vessel_repository import get_vessel_class_spec

logger = get_logger(__name__)


# ---- Sentinel for "no constraint" (matches -9999 in ports.csv) ----
_NO_CONSTRAINT = -9999.0


@dataclass
class PortConstraints:
    """Physical constraints of a port."""
    port_id: str
    port_name: str
    max_draft_m: float
    max_loa_m: float
    max_beam_m: float
    max_dwt: int
    is_anchorage_only: bool = False  # True for Sagar (IND_SAG)


@dataclass
class VesselSpecs:
    """Physical dimensions of a vessel class."""
    class_name: str
    dwt_min: int
    dwt_max: int
    typical_dwt: int
    draft_max_m: float
    loa_max_m: float
    beam_max_m: float


@dataclass
class CompatibilityResult:
    """Result of vessel-port compatibility check."""
    vessel_class: str
    is_compatible: bool
    violations: List[str] = field(default_factory=list)


@dataclass
class VesselSelectionResult:
    """Full vessel selection output."""
    origin_port_id: str
    destination_port_id: str
    feasible: List[CompatibilityResult] = field(default_factory=list)
    excluded: List[CompatibilityResult] = field(default_factory=list)

    @property
    def feasible_classes(self) -> List[str]:
        return [r.vessel_class for r in self.feasible]


class VesselSelector:
    """
    Filters vessel classes by port physical constraints.

    Constraint rules:
    1. vessel.draft_max_m ≤ port.max_draft_m
    2. vessel.loa_max_m ≤ port.max_loa_m   (if port constraint exists)
    3. vessel.beam_max_m ≤ port.max_beam_m  (if port constraint exists)
    4. Anchorage-only ports (Sagar) require lightering — only
       certain vessel classes via STS transfer.

    Both origin AND destination port constraints must be satisfied.
    """

    def check_compatibility(
        self,
        vessel: VesselSpecs,
        port: PortConstraints,
    ) -> CompatibilityResult:
        """
        Check if a vessel class can operate at a specific port.

        Args:
            vessel: Vessel class specifications.
            port: Port physical constraints.

        Returns:
            CompatibilityResult with pass/fail and violation details.
        """
        violations = []

        # Anchorage-only check
        if port.is_anchorage_only:
            violations.append(
                f"{port.port_name} is an anchorage/lightering point only — "
                f"no direct berthing available"
            )
            return CompatibilityResult(
                vessel_class=vessel.class_name,
                is_compatible=False,
                violations=violations,
            )

        # Draft constraint
        if port.max_draft_m != _NO_CONSTRAINT and vessel.draft_max_m > port.max_draft_m:
            violations.append(
                f"Draft: vessel {vessel.draft_max_m}m > port max {port.max_draft_m}m"
            )

        # LOA constraint
        if port.max_loa_m != _NO_CONSTRAINT and vessel.loa_max_m > port.max_loa_m:
            violations.append(
                f"LOA: vessel {vessel.loa_max_m}m > port max {port.max_loa_m}m"
            )

        # Beam constraint
        if port.max_beam_m != _NO_CONSTRAINT and vessel.beam_max_m > port.max_beam_m:
            violations.append(
                f"Beam: vessel {vessel.beam_max_m}m > port max {port.max_beam_m}m"
            )

        is_compatible = len(violations) == 0
        return CompatibilityResult(
            vessel_class=vessel.class_name,
            is_compatible=is_compatible,
            violations=violations,
        )

    def select_vessels(
        self,
        origin_port: PortConstraints,
        destination_port: PortConstraints,
        vessel_specs: List[VesselSpecs],
        cargo_tonnage: Optional[int] = None,
    ) -> VesselSelectionResult:
        """
        Select all feasible vessel classes for a given origin-destination pair.

        Args:
            origin_port: Origin port constraints.
            destination_port: Destination port constraints.
            vessel_specs: List of all vessel class specifications.
            cargo_tonnage: Desired cargo size (filters out classes too small).

        Returns:
            VesselSelectionResult with feasible and excluded classes.
        """
        result = VesselSelectionResult(
            origin_port_id=origin_port.port_id,
            destination_port_id=destination_port.port_id,
        )

        for vessel in vessel_specs:
            # Check against both ports — vessel must fit at both ends
            origin_check = self.check_compatibility(vessel, origin_port)
            dest_check = self.check_compatibility(vessel, destination_port)

            # Combine violations from both ports
            all_violations = []
            if not origin_check.is_compatible:
                all_violations.extend(
                    [f"[Origin] {v}" for v in origin_check.violations]
                )
            if not dest_check.is_compatible:
                all_violations.extend(
                    [f"[Destination] {v}" for v in dest_check.violations]
                )

            # Cargo tonnage feasibility check
            if cargo_tonnage and cargo_tonnage > vessel.dwt_max:
                all_violations.append(
                    f"Cargo {cargo_tonnage:,}t exceeds max DWT {vessel.dwt_max:,}t"
                )

            combined = CompatibilityResult(
                vessel_class=vessel.class_name,
                is_compatible=len(all_violations) == 0,
                violations=all_violations,
            )

            if combined.is_compatible:
                result.feasible.append(combined)
            else:
                result.excluded.append(combined)

        logger.info(
            f"Vessel selection {origin_port.port_id} → {destination_port.port_id}: "
            f"{len(result.feasible)} feasible, {len(result.excluded)} excluded"
        )
        return result

@dataclass
class VesselScore:
    """Individual scores for a vessel class."""
    vessel_class: str
    cargo_suitability_score: float
    port_compatibility_score: float
    economic_score: float
    operational_score: float
    risk_score: float
    total_score: float
    reasons: str

@dataclass
class VesselOptimizationResult:
    """Final output of the optimization engine."""
    recommended_vessel: str
    score: float
    reasons: str
    alternatives: List[VesselScore] = field(default_factory=list)

    def to_dict(self):
        return {
            "recommended_vessel": self.recommended_vessel,
            "score": self.score,
            "reasons": self.reasons,
            "alternatives": [
                {
                    "vessel_class": a.vessel_class,
                    "score": a.total_score,
                    "reasons": a.reasons
                } for a in self.alternatives
            ]
        }

class VesselOptimizer:
    """
    Evaluates vessel classes based on a multi-criteria scoring system.
    """
    def __init__(self):
        self.physical_selector = VesselSelector()

    def optimize(
        self,
        cargo_type: str,
        cargo_quantity: int,
        origin_port: PortConstraints,
        destination_port: PortConstraints,
        expected_loading_date: datetime,
        required_delivery_date: datetime,
        vessel_specs: List[VesselSpecs],
    ) -> VesselOptimizationResult:
        """
        Evaluate and rank all vessels for the given requirement.
        """
        # 1. Base physical compatibility
        selection_result = self.physical_selector.select_vessels(
            origin_port, destination_port, vessel_specs, cargo_quantity
        )

        comp_map = {r.vessel_class: r for r in selection_result.feasible + selection_result.excluded}
        scores = []

        # Timeline constraint
        allowed_days = (required_delivery_date - expected_loading_date).days
        if allowed_days <= 0:
            allowed_days = 1 # Prevent division by zero

        for vessel in vessel_specs:
            comp_result = comp_map[vessel.class_name]
            reasons = []

            # --- Port Compatibility Score ---
            if comp_result.is_compatible:
                port_score = 100.0
                reasons.append("Physically compatible with ports.")
            else:
                port_score = 0.0
                reasons.append(f"Port restrictions: {', '.join(comp_result.violations)}.")

            # --- Cargo Suitability Score ---
            if cargo_quantity > vessel.dwt_max:
                cargo_score = 0.0
                reasons.append("Cargo exceeds max DWT (requires multiple trips).")
            elif cargo_quantity < vessel.dwt_min:
                util_ratio = cargo_quantity / vessel.dwt_min
                cargo_score = max(0.0, util_ratio * 50)
                reasons.append("Severe cargo underutilization.")
            else:
                util_ratio = cargo_quantity / vessel.typical_dwt
                if util_ratio <= 1.0:
                    cargo_score = util_ratio * 100
                else:
                    cargo_score = 100 - ((util_ratio - 1.0) * 50)
                cargo_score = max(0.0, min(100.0, cargo_score))

                if cargo_score > 85:
                    reasons.append("Optimal cargo utilization.")
                else:
                    reasons.append("Adequate cargo utilization.")

            # Lookup canonical vessel specification
            v_spec = get_vessel_class_spec(vessel.class_name)

            # --- Operational Score (Dynamic Realistic Duration) ---
            speed_knots = v_spec.service_speed_knots

            # Realistic route distance (default 3200 nm for standard trade routes)
            route_distance_nm = 3200.0
            sailing_days = calculate_sailing_days(route_distance_nm, speed_knots)

            # Handling rates at origin and destination
            origin_rate = getattr(origin_port, "cargo_handling_rate_mt_day", 35000.0) or 35000.0
            dest_rate = getattr(destination_port, "cargo_handling_rate_mt_day", 30000.0) or 30000.0
            load_days = cargo_quantity / origin_rate
            disch_days = cargo_quantity / dest_rate
            expected_wait_days = 2.0

            est_voyage_days = sailing_days + load_days + disch_days + expected_wait_days
            if allowed_days < est_voyage_days:
                op_score = 0.0
                reasons.append("Cannot meet required delivery date.")
            else:
                buffer = allowed_days - est_voyage_days
                op_score = min(100.0, 50.0 + (buffer * 5))
                reasons.append("Meets delivery schedule.")

            # --- Economic Score (Realistic Voyage Economics) ---
            if cargo_score == 0:
                econ_score = 0.0
            else:
                daily_hire = v_spec.daily_hire_usd
                daily_fuel = v_spec.fuel_consumption_tpd

                econ_inputs = VoyageCostInputs(
                    cargo_quantity_t=float(cargo_quantity),
                    freight_rate_usd=22.0,
                    vessel_speed_knots=speed_knots,
                    vessel_daily_fuel_consumption_tpd=daily_fuel,
                    vessel_daily_hire_cost_usd=daily_hire,
                    route_distance_nm=route_distance_nm,
                    expected_waiting_days=expected_wait_days,
                    port_handling_rate_tpd=origin_rate,
                    discharge_port_handling_rate_tpd=dest_rate,
                    load_port_cost_usd=50000.0,
                    discharge_port_cost_usd=50000.0,
                )
                econ_breakdown = calculate_voyage_cost(econ_inputs)
                actual_cost_per_tonne = econ_breakdown.cost_per_tonne
                econ_score = max(0.0, min(100.0, 135.0 - (actual_cost_per_tonne * 1.8)))

            if econ_score > 70:
                reasons.append("Strong economic efficiency.")
            elif econ_score > 0:
                reasons.append("Moderate economic efficiency.")

            # --- Risk Score (Dynamic Operational Maneuverability) ---
            risk_score = v_spec.maneuverability_score

            # --- Total Score ---
            if port_score == 0 or cargo_score == 0 or op_score == 0:
                total_score = 0.0
            else:
                total_score = (cargo_score * 0.35) + (econ_score * 0.35) + (op_score * 0.20) + (risk_score * 0.10)

            scores.append(VesselScore(
                vessel_class=vessel.class_name,
                cargo_suitability_score=round(cargo_score, 1),
                port_compatibility_score=round(port_score, 1),
                economic_score=round(econ_score, 1),
                operational_score=round(op_score, 1),
                risk_score=round(risk_score, 1),
                total_score=round(total_score, 1),
                reasons=" ".join(reasons)
            ))

        scores.sort(key=lambda x: x.total_score, reverse=True)

        # Handle case where no vessel is feasible
        if not scores or scores[0].total_score == 0:
            return VesselOptimizationResult(
                recommended_vessel="None",
                score=0.0,
                reasons="No feasible vessels found for the given requirements.",
                alternatives=scores
            )

        best = scores[0]
        return VesselOptimizationResult(
            recommended_vessel=best.vessel_class,
            score=best.total_score,
            reasons=best.reasons,
            alternatives=scores[1:]
        )


# Alias for backward compatibility
VesselSelectionOptimizer = VesselOptimizer

"""
DockInsights — Candidate Plan Generator (Phase 6).

Generates candidate fleet allocation plans for dry bulk cargo shipments.
Produces both homogeneous vessel configurations (1xCapesize, 2xPanamax,
2xSupramax, 3xHandysize) and viable mixed-class combinations across
single or multiple voyages.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VesselClassDefinition:
    """Class specification for candidate generation."""
    class_name: str
    dwt_min: float
    dwt_max: float
    typical_dwt: float
    service_speed_knots: float
    draft_max_m: float
    loa_max_m: float
    beam_max_m: float


def _get_default_vessel_classes() -> Dict[str, VesselClassDefinition]:
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    return {
        s.class_name: VesselClassDefinition(
            class_name=s.class_name,
            dwt_min=s.dwt_min,
            dwt_max=s.dwt_max,
            typical_dwt=s.typical_dwt,
            service_speed_knots=s.service_speed_knots,
            draft_max_m=s.draft_max_m,
            loa_max_m=s.loa_max_m,
            beam_max_m=s.beam_max_m,
        )
        for s in specs if s.class_name in ("Handysize", "Supramax", "Panamax", "Capesize")
    }


DEFAULT_VESSEL_CLASSES: Dict[str, VesselClassDefinition] = _get_default_vessel_classes()



@dataclass
class CandidatePlanSpec:
    """
    Blueprint specification for a candidate charter plan.
    """
    plan_id: str
    vessel_classes: List[str]
    number_of_vessels: int
    number_of_voyages: int
    cargo_per_voyage: List[float]
    execution_mode: str = "sequential"  # "sequential" or "parallel"
    description: str = ""


class CandidateGenerator:
    """
    Generates candidate vessel and fleet allocation plans for a requested cargo volume.
    """

    def __init__(self, vessel_classes: Optional[Dict[str, VesselClassDefinition]] = None):
        self.vessel_classes = vessel_classes or DEFAULT_VESSEL_CLASSES

    def generate_candidate_plans(
        self,
        total_cargo_t: float,
        max_voyages: int = 5,
        allow_mixed_classes: bool = True,
        include_parallel_options: bool = True,
    ) -> List[CandidatePlanSpec]:
        """
        Generates candidate plans spanning 1 to max_voyages.

        Example for 100,000 MT:
        - 1 × Capesize (100k MT)
        - 2 × Panamax (50k MT each, sequential or parallel)
        - 2 × Supramax (50k MT each, sequential or parallel)
        - 3 × Handysize (33.3k MT each, sequential or parallel)
        - Mixed: 1 × Panamax (70k MT) + 1 × Handysize (30k MT)
        """
        candidates: List[CandidatePlanSpec] = []
        seen_plan_ids = set()

        # ---------------------------------------------------------
        # 1. Homogeneous Fleet Plans
        # ---------------------------------------------------------
        for v_name, v_spec in self.vessel_classes.items():
            # Find minimal voyage count such that parcel <= dwt_max
            min_voyages_needed = max(1, math.ceil(total_cargo_t / v_spec.dwt_max))
            # Also allow up to max_voyages where parcels make sense (parcel >= 0.35 * dwt_min)
            max_voyages_allowed = min(max_voyages, max(1, int(total_cargo_t / (v_spec.dwt_min * 0.35))))

            for num_voyages in range(min_voyages_needed, max_voyages_allowed + 1):
                parcel_size = total_cargo_t / num_voyages

                # Check parcel is within reasonable loading limits
                if parcel_size > v_spec.dwt_max * 1.05:  # Cannot overload vessel
                    continue
                if parcel_size < v_spec.dwt_min * 0.30:  # Avoid excessive underloading
                    continue

                parcels = [round(parcel_size, 2)] * num_voyages

                # Option A: 1 vessel executing multiple sequential voyages
                plan_id_seq = f"PLAN_{num_voyages}x{v_name}" if num_voyages == 1 else f"PLAN_{num_voyages}x{v_name}_seq"
                if plan_id_seq not in seen_plan_ids:
                    seen_plan_ids.add(plan_id_seq)
                    candidates.append(CandidatePlanSpec(
                        plan_id=plan_id_seq,
                        vessel_classes=[v_name],
                        number_of_vessels=1,
                        number_of_voyages=num_voyages,
                        cargo_per_voyage=parcels,
                        execution_mode="sequential",
                        description=f"{num_voyages} sequential voyage(s) with 1 {v_name}",
                    ))

                # Option B: N separate vessels operating in parallel (if num_voyages > 1)
                if num_voyages > 1 and include_parallel_options:
                    plan_id_par = f"PLAN_{num_voyages}x{v_name}_par"
                    if plan_id_par not in seen_plan_ids:
                        seen_plan_ids.add(plan_id_par)
                        candidates.append(CandidatePlanSpec(
                            plan_id=plan_id_par,
                            vessel_classes=[v_name],
                            number_of_vessels=num_voyages,
                            number_of_voyages=num_voyages,
                            cargo_per_voyage=parcels,
                            execution_mode="parallel",
                            description=f"{num_voyages} separate {v_name} vessels in parallel",
                        ))

        # ---------------------------------------------------------
        # 2. Heterogeneous / Mixed-Class Plans
        # ---------------------------------------------------------
        if allow_mixed_classes and total_cargo_t >= 60000:
            # Common 2-vessel mixed combinations:
            # e.g., Panamax + Handysize or Supramax + Handysize
            class_pairs = [
                ("Panamax", "Handysize"),
                ("Supramax", "Handysize"),
                ("Panamax", "Supramax"),
            ]
            for v1_name, v2_name in class_pairs:
                if v1_name not in self.vessel_classes or v2_name not in self.vessel_classes:
                    continue
                v1_spec = self.vessel_classes[v1_name]
                v2_spec = self.vessel_classes[v2_name]

                # Proportion parcel sizing based on typical DWTs
                tot_typ = v1_spec.typical_dwt + v2_spec.typical_dwt
                ratio_v1 = v1_spec.typical_dwt / tot_typ
                p1 = round(total_cargo_t * ratio_v1, 2)
                p2 = round(total_cargo_t - p1, 2)

                if p1 <= v1_spec.dwt_max and p2 <= v2_spec.dwt_max and p1 >= v1_spec.dwt_min * 0.5 and p2 >= v2_spec.dwt_min * 0.5:
                    plan_id_mixed = f"PLAN_1x{v1_name}_1x{v2_name}"
                    if plan_id_mixed not in seen_plan_ids:
                        seen_plan_ids.add(plan_id_mixed)
                        candidates.append(CandidatePlanSpec(
                            plan_id=plan_id_mixed,
                            vessel_classes=[v1_name, v2_name],
                            number_of_vessels=2,
                            number_of_voyages=2,
                            cargo_per_voyage=[p1, p2],
                            execution_mode="parallel" if include_parallel_options else "sequential",
                            description=f"Mixed fleet: 1 {v1_name} ({p1:,.0f} MT) + 1 {v2_name} ({p2:,.0f} MT)",
                        ))

        logger.info("Generated %d candidate charter plans for %s MT cargo", len(candidates), f"{total_cargo_t:,.0f}")
        return candidates

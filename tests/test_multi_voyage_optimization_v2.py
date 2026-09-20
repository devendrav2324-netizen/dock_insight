"""
DockInsights — Multi-Voyage Vessel & Fleet Optimization Tests (Phase 6).

Validates:
1. Complete charter plan generation (1xCapesize, 2xPanamax, 2xSupramax, 3xHandysize, mixed)
2. Hard physical and operational constraint elimination (draft, LOA, beam, deadlines)
3. Integration of Voyage Economics, Port Congestion, and Risk
4. Multi-criteria soft objective scoring with configurable weights
5. The 100,000 MT Australia -> Indian East Coast coal benchmark scenario
"""

from datetime import datetime, timedelta
import pytest

from src.optimization.candidate_generator import CandidateGenerator
from src.optimization.constraint_engine import ConstraintEngine, PortSpec
from src.optimization.voyage_plan import CharterPlan, VoyageLeg
from src.optimization.vessel_scoring import OptimizationWeights, VesselScoringEngine
from src.optimization.multi_voyage_optimizer import (
    MultiVoyageOptimizer,
    FleetOptimizationRequest,
)
from src.optimization.vessel_selector import PortConstraints, VesselSpecs, VesselSelector, VesselOptimizer


# =============================================================================
# 1. Candidate Plan Generator Tests
# =============================================================================

def test_candidate_generator_100k_tonnes():
    """Verify plan generation for 100,000 MT produces all required vessel configurations."""
    generator = CandidateGenerator()
    specs = generator.generate_candidate_plans(total_cargo_t=100_000.0, max_voyages=4)

    plan_ids = {s.plan_id for s in specs}
    vessel_classes_used = {tuple(s.vessel_classes) for s in specs}

    # 1. 1 x Capesize
    assert any("Capesize" in s.vessel_classes and s.number_of_voyages == 1 for s in specs)

    # 2. 2 x Panamax (50k each)
    assert any("Panamax" in s.vessel_classes and s.number_of_voyages == 2 for s in specs)

    # 3. 2 x Supramax (50k each)
    assert any("Supramax" in s.vessel_classes and s.number_of_voyages == 2 for s in specs)

    # 4. 3 x Handysize (33.3k each)
    assert any("Handysize" in s.vessel_classes and s.number_of_voyages == 3 for s in specs)

    # Check parcel quantities sum to 100,000 MT
    for s in specs:
        assert abs(sum(s.cargo_per_voyage) - 100_000.0) < 1.0


def test_mixed_fleet_plan_generation():
    """Verify mixed-class candidate generation for parcel-split cargoes."""
    generator = CandidateGenerator()
    specs = generator.generate_candidate_plans(total_cargo_t=100_000.0, allow_mixed_classes=True)

    mixed_specs = [s for s in specs if len(s.vessel_classes) > 1]
    assert len(mixed_specs) > 0
    # E.g. Panamax + Handysize or Supramax + Handysize
    assert any(set(s.vessel_classes) == {"Panamax", "Handysize"} for s in mixed_specs)


# =============================================================================
# 2. Hard Constraint Engine Tests
# =============================================================================

def test_hard_constraint_eliminates_capesize_at_shallow_port():
    """
    Capesize draft (18.5m) exceeds Gopalpur draft limit (14.5m).
    Hard constraint engine must eliminate Capesize with explicit reasons.
    """
    engine = ConstraintEngine()
    deep_origin = PortSpec("AUS_NEW", "Newcastle", max_draft_m=20.0, max_loa_m=320.0, max_beam_m=50.0)
    shallow_dest = PortSpec("IND_GOP", "Gopalpur", max_draft_m=14.5, max_loa_m=230.0, max_beam_m=32.0)

    # 1 x Capesize plan
    generator = CandidateGenerator()
    plans = generator.generate_candidate_plans(total_cargo_t=100_000.0)
    capesize_spec = next(p for p in plans if "Capesize" in p.vessel_classes and p.number_of_voyages == 1)

    result = engine.evaluate_plan(
        plan=capesize_spec,
        origin_port=deep_origin,
        destination_port=shallow_dest,
        cargo_type="Coal"
    )

    assert result.is_feasible is False
    assert any("draft" in r.lower() and "gopalpur" in r.lower() for r in result.failed_constraints)


def test_hard_constraint_passes_supramax_at_shallow_port():
    """Supramax draft (12.5m) and LOA (199.9m) comfortably satisfy Gopalpur limits."""
    engine = ConstraintEngine()
    deep_origin = PortSpec("AUS_NEW", "Newcastle", max_draft_m=20.0, max_loa_m=320.0, max_beam_m=50.0)
    shallow_dest = PortSpec("IND_GOP", "Gopalpur", max_draft_m=14.5, max_loa_m=230.0, max_beam_m=33.0)

    generator = CandidateGenerator()
    plans = generator.generate_candidate_plans(total_cargo_t=100_000.0)
    supramax_spec = next(p for p in plans if "Supramax" in p.vessel_classes and p.number_of_voyages == 2)

    result = engine.evaluate_plan(
        plan=supramax_spec,
        origin_port=deep_origin,
        destination_port=shallow_dest,
        cargo_type="Coal"
    )

    assert result.is_feasible is True
    assert len(result.failed_constraints) == 0


def test_delivery_deadline_hard_constraint():
    """When a plan's total duration exceeds the deadline, it must be marked infeasible."""
    engine = ConstraintEngine()
    port = PortSpec("IND_PAR", "Paradip", max_draft_m=18.5, max_loa_m=300.0, max_beam_m=48.0)

    generator = CandidateGenerator()
    plans = generator.generate_candidate_plans(total_cargo_t=100_000.0)
    handysize_spec = next(p for p in plans if "Handysize" in p.vessel_classes and p.number_of_voyages == 3)

    # If 3 sequential voyages take 65 days, but deadline is 30 days
    result = engine.evaluate_plan(
        plan=handysize_spec,
        origin_port=port,
        destination_port=port,
        delivery_deadline_days=30.0,
        estimated_duration_days=65.0,
    )
    assert result.is_feasible is False
    assert any("deadline" in r.lower() for r in result.failed_constraints)


# =============================================================================
# 3. Soft Scoring & Weight Configuration Tests
# =============================================================================

def test_configurable_optimization_weights():
    """Verify custom objective weights alter plan rankings."""
    # Create two plans: Plan 1 is cheaper but slower, Plan 2 is faster but pricier
    plan_cheap = CharterPlan(
        plan_id="CHEAP_PLAN",
        vessel_classes=["Panamax"],
        number_of_vessels=1,
        number_of_voyages=2,
        total_cargo_t=100_000.0,
        total_cost=2_000_000.0,
        cost_per_tonne=20.0,
        total_duration=45.0,
        utilization=0.90,
        feasibility=True,
    )
    plan_fast = CharterPlan(
        plan_id="FAST_PLAN",
        vessel_classes=["Panamax"],
        number_of_vessels=2,
        number_of_voyages=2,
        total_cargo_t=100_000.0,
        total_cost=2_400_000.0,
        cost_per_tonne=24.0,
        total_duration=22.0,
        utilization=0.90,
        feasibility=True,
    )

    # Cost-dominant weights (cost 70%, schedule 10%)
    cost_weights = OptimizationWeights(cost_weight=0.70, schedule_weight=0.10, risk_weight=0.10, utilization_weight=0.05, demurrage_weight=0.05)
    scorer_cost = VesselScoringEngine(weights=cost_weights)
    scored_cost = scorer_cost.score_plans([plan_cheap, plan_fast])
    assert scored_cost[0].plan_id == "CHEAP_PLAN"

    # Schedule-dominant weights (cost 10%, schedule 70%)
    plan_cheap_reset = CharterPlan(**plan_cheap.__dict__)
    plan_fast_reset = CharterPlan(**plan_fast.__dict__)
    sched_weights = OptimizationWeights(cost_weight=0.10, schedule_weight=0.70, risk_weight=0.10, utilization_weight=0.05, demurrage_weight=0.05)
    scorer_sched = VesselScoringEngine(weights=sched_weights)
    scored_sched = scorer_sched.score_plans([plan_cheap_reset, plan_fast_reset])
    assert scored_sched[0].plan_id == "FAST_PLAN"


# =============================================================================
# 4. End-to-End Australia -> Indian East Coast Benchmark Scenario
# =============================================================================

def test_australia_to_gangavaram_100k_coal_scenario():
    """
    Scenario: 100,000 MT Coal from Australia (AUS_NEW) to Gangavaram (IND_GVM).
    Gangavaram has 21.0m draft (deepest port) which accepts Capesize.
    Compare:
    - 1 x Capesize
    - 2 x Panamax
    - 2 x Supramax
    - 3 x Handysize
    """
    optimizer = MultiVoyageOptimizer()
    request = FleetOptimizationRequest(
        cargo_quantity_t=100_000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        cargo_type="Coal",
        route_distance_nm=5200.0,
        freight_rate_usd=22.0,
        delivery_deadline_days=45.0,
        max_voyages=3,
        include_parallel_options=True,
    )

    ranked_plans = optimizer.optimize(request)
    assert len(ranked_plans) >= 4

    # Verify every plan has complete itemized calculations
    for plan in ranked_plans:
        d = plan.to_dict()
        assert "plan_id" in d
        assert "vessel_classes" in d
        assert "number_of_vessels" in d
        assert "number_of_voyages" in d
        assert "total_cost" in d
        assert "cost_per_tonne" in d
        assert "utilization" in d
        assert "total_duration" in d
        assert "delivery_probability" in d
        assert "risk_score" in d
        assert "feasibility" in d
        assert "reasons" in d

        assert plan.total_cost > 0.0
        assert plan.cost_per_tonne > 0.0
        assert plan.total_duration > 0.0
        # No hardcoded 80 risk score
        assert plan.risk_score != 80.0

    # At Gangavaram, Capesize is physically feasible
    capesize_plan = next((p for p in ranked_plans if "Capesize" in p.vessel_classes and p.number_of_voyages == 1), None)
    assert capesize_plan is not None
    assert capesize_plan.feasibility is True
    # But Capesize utilization is ~100k/175k = 57.1% (underutilized)
    assert capesize_plan.utilization < 0.65

    # 2 x Supramax utilization is ~50k/55k = 90.9% (optimal utilization)
    supramax_plan = next((p for p in ranked_plans if "Supramax" in p.vessel_classes and p.number_of_voyages == 2), None)
    assert supramax_plan is not None
    assert supramax_plan.utilization > 0.85


def test_australia_to_gopalpur_100k_coal_scenario_eliminates_capesize():
    """
    Scenario: 100,000 MT Coal from Australia (AUS_NEW) to Gopalpur (IND_GOP).
    Gopalpur has 14.5m draft.
    Capesize MUST be eliminated due to 18.5m draft violation.
    Panamax or Supramax must rank at the top.
    """
    optimizer = MultiVoyageOptimizer()
    request = FleetOptimizationRequest(
        cargo_quantity_t=100_000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GOP",
        cargo_type="Coal",
        route_distance_nm=5200.0,
        freight_rate_usd=22.0,
        delivery_deadline_days=50.0,
        max_voyages=3,
        include_parallel_options=True,
    )

    ranked_plans = optimizer.optimize(request)

    # 1. Capesize must be infeasible
    capesize_plans = [p for p in ranked_plans if "Capesize" in p.vessel_classes]
    for cp in capesize_plans:
        assert cp.feasibility is False
        assert cp.score == 0.0
        assert any("draft" in r.lower() for r in cp.failed_constraints)

    # 2. Feasible plans exist (e.g. Supramax or Handysize)
    feasible = [p for p in ranked_plans if p.feasibility]
    assert len(feasible) > 0

    # 3. Top recommended plan is feasible
    top_plan = ranked_plans[0]
    assert top_plan.feasibility is True
    assert "Capesize" not in top_plan.vessel_classes
    assert top_plan.score > 0.0


def test_parallel_vs_sequential_voyage_duration():
    """Parallel multi-vessel plans must have shorter total duration than sequential multi-voyage plans."""
    optimizer = MultiVoyageOptimizer()
    request = FleetOptimizationRequest(
        cargo_quantity_t=100_000.0,
        origin_port_id="AUS_NEW",
        destination_port_id="IND_GVM",
        route_distance_nm=5200.0,
        include_parallel_options=True,
    )
    plans = optimizer.optimize(request)

    seq_plan = next((p for p in plans if "Supramax" in p.vessel_classes and p.execution_mode == "sequential" and p.number_of_voyages == 2), None)
    par_plan = next((p for p in plans if "Supramax" in p.vessel_classes and p.execution_mode == "parallel" and p.number_of_voyages == 2), None)

    assert seq_plan is not None
    assert par_plan is not None
    # Parallel completes significantly faster than sequential (which includes ballast return)
    assert par_plan.total_duration < seq_plan.total_duration

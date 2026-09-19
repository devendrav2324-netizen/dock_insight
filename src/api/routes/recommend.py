"""
Charter-AI — Unified Recommendation & Decision Endpoint (Phase 10).

The central decision-support endpoint: orchestrates vessel selection,
probabilistic forecasting, port congestion predictions, market timing,
voyage economics, 8-category risk assessment, Monte Carlo simulations,
multi-voyage fleet optimization, and contract strategy optimization.
"""

from datetime import datetime, timezone
from typing import Optional, Union
from fastapi import APIRouter, HTTPException, Request, Query

from src.api.serializers import (
    RecommendationRequest,
    RecommendationResponse,
    DecisionResponse,
    PrimaryRecommendation,
    ContractRecommendationResponse,
    ForecastResponse,
    VoyageEconomicsResponse,
    CostBreakdownResponse,
    RiskAssessmentResponse,
    RiskDimensionResponse,
    VesselSelectionResponse,
    VesselCompatibilityResponse,
)
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.data.mock_db import get_mock_vessel_db, get_mock_port_info, require_sih_demo_mode
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/recommend", tags=["Recommendation"])


def _build_engine(request: Request) -> DecisionEngine:
    """Instantiate the unified DecisionEngine using app-state singletons."""
    return DecisionEngine(
        forecast_service=getattr(request.app.state, "forecast_service", None),
        risk_service=getattr(request.app.state, "risk_service", None),
        economics_service=getattr(request.app.state, "economics_service", None),
        vessel_service=getattr(request.app.state, "vessel_service", None),
        contract_service=getattr(request.app.state, "contract_service", None),
        congestion_service=getattr(request.app.state, "congestion_service", None),
    )


def _map_to_recommendation_response(
    req: RecommendationRequest,
    result: dict,
) -> RecommendationResponse:
    """Maps the canonical DecisionEngine output dictionary to RecommendationResponse."""
    rec_plan = result["recommended_plan"]
    timing = result.get("market_timing", {})
    contract = result.get("contract_strategy", {})
    econ = result.get("economics", {})
    risk_data = result.get("risk", {})
    forecast_data = result.get("freight_forecast", {})
    mkt = result.get("market_analysis", {})
    explanation = result.get("explanation", {})

    # 1. Primary Recommendation
    daily_rate = (
        rec_plan["total_cost"] / max(1.0, rec_plan["voyage_duration"])
        if rec_plan.get("voyage_duration")
        else 25000.0
    )

    primary_rec = PrimaryRecommendation(
        vessel_class=rec_plan["vessel_class"],
        optimal_booking_window=timing.get(
            "recommended_booking_window",
            {"start": str(req.earliest_date), "end": str(req.latest_date)},
        ),
        contract=ContractRecommendationResponse(
            contract_type=contract.get("recommended_strategy", "100% SPOT"),
            duration_months=12 if "TERM" in contract.get("recommended_strategy", "") else None,
            reasoning="; ".join(contract.get("reasons", ["Balanced risk-reward profile"])),
            confidence=result.get("confidence", 0.85),
        ),
        estimated_rate_usd_per_day=round(daily_rate, 2),
        confidence=result.get("confidence", 0.85),
        reasoning=explanation.get("tradeoff_analysis", "Optimal multi-criteria charter plan."),
    )

    # 2. Alternatives
    alternatives = []
    for alt in result.get("alternative_plans", []):
        v_class = alt.get("vessel_classes", ["Unknown"])[0] if alt.get("vessel_classes") else "Alternative"
        alt_dur = alt.get("total_duration", 20.0)
        alt_rate = alt.get("total_cost", 1000000.0) / max(1.0, alt_dur)
        alternatives.append(
            PrimaryRecommendation(
                vessel_class=v_class,
                optimal_booking_window=timing.get(
                    "recommended_booking_window",
                    {"start": str(req.earliest_date), "end": str(req.latest_date)},
                ),
                contract=ContractRecommendationResponse(
                    contract_type=contract.get("recommended_strategy", "SPOT"),
                    duration_months=None,
                    reasoning=f"Plan {alt.get('plan_id', 'alt')}: Score {alt.get('score', 0):.1f}/100",
                    confidence=round(alt.get("score", 70.0) / 100.0, 2),
                ),
                estimated_rate_usd_per_day=round(alt_rate, 2),
                confidence=round(alt.get("score", 70.0) / 100.0, 2),
                reasoning=f"Alternative option {alt.get('plan_id', '')} (${alt.get('cost_per_tonne', 0.0):.2f}/t).",
            )
        )

    # 3. Forecast
    forecast_resp = ForecastResponse(
        current_rate=mkt.get("current_rate", 22.0),
        forecast_rate=mkt.get("forecast", 22.5),
        lower_bound=forecast_data.get("p10", 20.0),
        upper_bound=forecast_data.get("p90", 25.0),
        trend=mkt.get("direction", "STABLE").lower(),
        confidence=mkt.get("confidence", 0.85),
        model_used=forecast_data.get("model_used", "Ensemble"),
        metrics={"volatility": mkt.get("volatility", 16.0)},
        origin_port_id=req.origin_port_id,
        destination_port_id=req.destination_port_id,
        vessel_class=rec_plan["vessel_class"],
    )

    # 4. Economics
    sailing_est = max(1.0, rec_plan["voyage_duration"] - rec_plan["expected_waiting"] - 4.0)
    econ_resp = VoyageEconomicsResponse(
        total_voyage_cost_usd=econ.get("total_cost", rec_plan["total_cost"]),
        cost_per_tonne_usd=econ.get("cost_per_tonne", rec_plan["cost_per_tonne"]),
        cargo_tonnage=req.cargo_tonnage,
        vessel_class=rec_plan["vessel_class"],
        sailing_days=round(sailing_est, 1),
        total_voyage_days=rec_plan["voyage_duration"],
        breakdown=CostBreakdownResponse(
            freight_cost_usd=econ.get("freight_cost", 0.0),
            bunker_cost_usd=econ.get("bunker_cost", 0.0),
            load_port_charges_usd=round(econ.get("port_charges", 50000.0) * 0.5, 2),
            discharge_port_charges_usd=round(econ.get("port_charges", 50000.0) * 0.5, 2),
            insurance_usd=round(econ.get("miscellaneous_cost", 20000.0) * 0.4, 2),
            expected_demurrage_usd=econ.get("demurrage_exposure", 0.0),
            miscellaneous_usd=round(econ.get("miscellaneous_cost", 20000.0) * 0.6, 2),
        ),
    )

    # 5. Risk
    risk_breakdown = {}
    for cat_name, cat_val in risk_data.get("categories", {}).items():
        if isinstance(cat_val, dict) and "score" in cat_val:
            risk_breakdown[cat_name] = RiskDimensionResponse(
                score=float(cat_val["score"]),
                level=str(cat_val.get("level", "LOW")),
                detail=str(cat_val.get("reasoning", "")),
            )

    risk_resp = RiskAssessmentResponse(
        composite_score=float(risk_data.get("composite_score", rec_plan["risk_score"])),
        level=str(risk_data.get("level", "MEDIUM")),
        breakdown=risk_breakdown,
        dominant_risk=risk_data.get("dominant_risk"),
        recommendation=str(risk_data.get("recommendation", "Proceed with standard charter.")),
    )

    # 6. Vessel Compatibility
    vessel_compat = VesselSelectionResponse(
        origin_port_id=req.origin_port_id,
        destination_port_id=req.destination_port_id,
        feasible=[
            VesselCompatibilityResponse(vessel_class=rec_plan["vessel_class"], is_compatible=True, violations=[])
        ],
        excluded=[],
    )

    return RecommendationResponse(
        request_id=result.get("decision_id", f"dec_{datetime.now().timestamp()}"),
        generated_at=datetime.fromisoformat(result.get("timestamp", datetime.now(timezone.utc).isoformat())),
        recommendation=primary_rec,
        alternatives=alternatives,
        forecast=forecast_resp,
        economics=econ_resp,
        risk=risk_resp,
        vessel_compatibility=vessel_compat,
    )


@router.post("", response_model=RecommendationResponse)
async def get_recommendation(
    request: RecommendationRequest,
    req_context: Request,
):
    """
    Generate a full chartering recommendation.

    Orchestrates:
    1. Vessel-port compatibility & multi-voyage plan formulation
    2. Real freight rate forecasting (ML/statistical ensemble)
    3. Port congestion prediction (quantile XGBoost waiting time & delay probability)
    4. Market timing engine (expected economic benefit analysis)
    5. Voyage economics (delivered 9-component cost)
    6. 8-category maritime risk engine + Monte Carlo distribution
    7. Quantitative contract optimization (spot vs term vs hybrid)
    8. Structured tradeoff explainability
    """
    require_sih_demo_mode()

    engine = _build_engine(req_context)

    orig_info = get_mock_port_info(request.origin_port_id)
    dest_info = get_mock_port_info(request.destination_port_id)

    expected_loading = datetime.combine(request.earliest_date, datetime.min.time())
    required_delivery = datetime.combine(request.latest_date, datetime.min.time())

    risk_tol = request.risk_appetite.upper()
    if risk_tol not in ["LOW", "MEDIUM", "HIGH"]:
        risk_tol = "LOW" if request.risk_appetite.lower() == "conservative" else (
            "HIGH" if request.risk_appetite.lower() == "aggressive" else "MEDIUM"
        )

    inputs = DecisionEngineInputs(
        cargo_type=request.cargo_type,
        cargo_quantity_t=float(request.cargo_tonnage),
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        expected_loading_date=expected_loading,
        required_delivery_date=required_delivery,
        number_of_voyages=1,
        risk_tolerance=risk_tol,
        vessel_specs_db=get_mock_vessel_db(),
        origin_port_info=orig_info,
        destination_port_info=dest_info,
    )

    result = engine.evaluate(inputs)

    if result.get("status") == "ERROR":
        raise HTTPException(
            status_code=400,
            detail=result.get("error_message", "Recommendation optimization failed."),
        )

    return _map_to_recommendation_response(request, result)


@router.post("/decision", response_model=DecisionResponse)
async def get_decision(
    request: RecommendationRequest,
    req_context: Request,
):
    """
    Generate the complete canonical Phase 10 DecisionResponse.
    Provides all 13 recommended plan attributes, reproducible model & data
    versions, market timing, contract optimization, and structured explainability.
    """
    require_sih_demo_mode()

    engine = _build_engine(req_context)

    orig_info = get_mock_port_info(request.origin_port_id)
    dest_info = get_mock_port_info(request.destination_port_id)

    expected_loading = datetime.combine(request.earliest_date, datetime.min.time())
    required_delivery = datetime.combine(request.latest_date, datetime.min.time())

    risk_tol = request.risk_appetite.upper()
    if risk_tol not in ["LOW", "MEDIUM", "HIGH"]:
        risk_tol = "LOW" if request.risk_appetite.lower() == "conservative" else (
            "HIGH" if request.risk_appetite.lower() == "aggressive" else "MEDIUM"
        )

    inputs = DecisionEngineInputs(
        cargo_type=request.cargo_type,
        cargo_quantity_t=float(request.cargo_tonnage),
        origin_port_id=request.origin_port_id,
        destination_port_id=request.destination_port_id,
        expected_loading_date=expected_loading,
        required_delivery_date=required_delivery,
        number_of_voyages=1,
        risk_tolerance=risk_tol,
        vessel_specs_db=get_mock_vessel_db(),
        origin_port_info=orig_info,
        destination_port_info=dest_info,
    )

    result = engine.evaluate(inputs)

    if result.get("status") == "ERROR":
        raise HTTPException(
            status_code=400,
            detail=result.get("error_message", "Decision optimization failed."),
        )

    return result

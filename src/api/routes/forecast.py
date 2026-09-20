"""
DockInsights — Freight Forecast Endpoints.

Returns real statistical and machine-learning freight rate forecasts
for a given route, vessel class, cargo type, and horizon.

Returns:
{
    "current_rate": float,
    "forecast_rate": float,
    "lower_bound": float,
    "upper_bound": float,
    "trend": "rising" | "falling" | "stable",
    "confidence": float,
    "model_used": str,
    "metrics": dict
}
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import APIRouter, Request, Query, HTTPException

from src.api.serializers import ForecastResponse, FreightForecastApiResponse
from src.services.freight_forecast_service import FreightForecastService
from src.models.market_timing import MarketTimingEngine, MarketTimingInputs

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("", response_model=ForecastResponse)
@router.get("/predict", response_model=ForecastResponse)
async def get_forecast(
    request: Request,
    origin: str = Query(..., description="Origin port code, e.g. AUS_NEW"),
    destination: str = Query(..., description="Destination port code, e.g. IND_GVM"),
    vessel_class: str = Query(..., description="Vessel class, e.g. Capesize, Panamax"),
    horizon_days: int = Query(7, description="Forecast horizon: 3, 7, 14, 30 days"),
    cargo_type: str = Query("thermal_coal", description="Cargo type, e.g. thermal_coal"),
    model_type: Optional[str] = Query(None, description="Optional model family: naive, moving_average, seasonal, arima, xgboost, ensemble")
):
    """
    Generate dry-bulk freight rate forecast for specified corridor and horizon.
    """
    service: FreightForecastService = getattr(
        request.app.state, "forecast_service", None
    )
    if service is None:
        service = FreightForecastService()

    try:
        pred = service.predict_freight_api(
            origin=origin,
            destination=destination,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            horizon_days=horizon_days,
            model_type=model_type
        )

        # Also populate route metadata for backward compatibility
        pred["origin_port_id"] = origin
        pred["destination_port_id"] = destination
        pred["vessel_class"] = vessel_class
        pred["horizon_days"] = horizon_days
        pred["model_version"] = "v2.0"

        return pred
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Freight forecasting engine error: {str(e)}"
        )


class MarketTimingApiRequest(BaseModel):
    """Request body for market timing recommendation."""
    current_freight_rate: float = Field(..., gt=0, description="Current spot freight rate USD/MT")
    forecast_rate: float = Field(..., gt=0, description="Forecasted freight rate USD/MT")
    p10_forecast: Optional[float] = Field(None, gt=0, description="10th percentile freight forecast")
    p50_forecast: Optional[float] = Field(None, gt=0, description="50th percentile (median) freight forecast")
    p90_forecast: Optional[float] = Field(None, gt=0, description="90th percentile freight forecast")
    forecast_confidence: float = Field(0.85, ge=0.0, le=1.0)
    market_momentum: float = Field(0.0, description="Market momentum indicator ($/t or %)")
    freight_volatility: float = Field(1.5, ge=0.0, description="Freight rate volatility ($/t)")
    vessel_availability: str = Field("BALANCED", description="TIGHT, BALANCED, or SURPLUS")
    congestion_forecast: float = Field(2.5, ge=0.0, description="Predicted port waiting days")
    cargo_deadline_days: Optional[float] = Field(None, gt=0, description="Cargo deadline in days from now")
    cargo_quantity_t: float = Field(75000.0, gt=0, description="Shipment cargo volume")
    route_voyage_days: float = Field(18.0, gt=0, description="Estimated voyage duration in days")


_timing_engine = MarketTimingEngine()


@router.post("/market-timing")
async def evaluate_market_timing(request: MarketTimingApiRequest) -> Dict[str, Any]:
    """
    Phase 7: Evaluate expected economic waiting benefits, deadline pressure, and tonnage availability
    to recommend whether to BOOK_NOW, WAIT, MONITOR, START_NEGOTIATION, or use HYBRID_BOOKING.
    """
    p50 = request.p50_forecast if request.p50_forecast is not None else request.forecast_rate
    p10 = request.p10_forecast if request.p10_forecast is not None else p50 * 0.92
    p90 = request.p90_forecast if request.p90_forecast is not None else p50 * 1.08

    inputs = MarketTimingInputs(
        current_freight_rate=request.current_freight_rate,
        forecast_rate=request.forecast_rate,
        p10_forecast=p10,
        p50_forecast=p50,
        p90_forecast=p90,
        forecast_confidence=request.forecast_confidence,
        market_momentum=request.market_momentum,
        freight_volatility=request.freight_volatility,
        vessel_availability=request.vessel_availability,
        congestion_forecast=request.congestion_forecast,
        cargo_deadline=request.cargo_deadline_days,
        cargo_quantity_t=request.cargo_quantity_t,
        route_voyage_days=request.route_voyage_days,
    )

    result = _timing_engine.evaluate_timing(inputs)
    return result.to_dict()

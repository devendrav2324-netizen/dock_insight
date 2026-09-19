"""
Charter-AI — Port Congestion & Idle Time Prediction Endpoints.

Returns expected vessel waiting time in days, P10/P50/P90 quantile intervals,
delay probabilities, and congestion level for loading and discharge ports.
"""

from typing import Optional
from datetime import date
from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel

from src.services.congestion_service import CongestionService

router = APIRouter(prefix="/congestion", tags=["Congestion"])


class CongestionResponse(BaseModel):
    expected_wait_days: float
    p10_wait_days: float
    p50_wait_days: float
    p90_wait_days: float
    delay_probability: float
    congestion_level: str  # "LOW", "MODERATE", "HIGH", "SEVERE"
    confidence: float
    data_source: Optional[str] = None
    model_used: Optional[str] = None


@router.get("/predict", response_model=CongestionResponse)
async def predict_congestion(
    request: Request,
    port_id: str = Query(..., description="UN/LOCODE or port code, e.g. IND_PAR, IND_GVM, AUS_NEW"),
    target_date: Optional[str] = Query(None, description="Expected arrival date (YYYY-MM-DD)"),
    vessel_class: Optional[str] = Query("Panamax", description="Vessel class, e.g. Capesize, Panamax"),
    cargo_type: Optional[str] = Query("thermal_coal", description="Cargo type, e.g. thermal_coal"),
    cargo_quantity: Optional[float] = Query(75000.0, description="Cargo quantity in metric tonnes"),
):
    """
    Predict vessel waiting time, uncertainty quantiles, and delay probability at a port.
    """
    service: CongestionService = getattr(request.app.state, "congestion_service", None)
    if service is None:
        service = CongestionService()

    try:
        result = service.predict_congestion(
            port_id=port_id,
            target_date=target_date,
            vessel_class=vessel_class,
            cargo_type=cargo_type,
            cargo_quantity=cargo_quantity
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Congestion prediction error: {str(e)}"
        )

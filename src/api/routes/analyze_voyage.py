from fastapi import APIRouter, HTTPException, Request
from datetime import datetime

from src.api.serializers import AnalyzeVoyageRequest, AnalyzeVoyageResponse
from src.optimization.decision_engine import DecisionEngine, DecisionEngineInputs
from src.data.mock_db import get_mock_vessel_db, get_mock_port_info, require_sih_demo_mode

router = APIRouter(prefix="/analyze-voyage", tags=["Analysis"])

@router.post("", response_model=AnalyzeVoyageResponse)
async def analyze_voyage(request_data: AnalyzeVoyageRequest, request: Request):
    require_sih_demo_mode()

    # Fetch initialized services from app state
    engine = DecisionEngine(
        forecast_service=request.app.state.forecast_service,
        risk_service=request.app.state.risk_service,
        economics_service=request.app.state.economics_service,
        vessel_service=request.app.state.vessel_service,
        contract_service=request.app.state.contract_service,
        congestion_service=getattr(request.app.state, "congestion_service", None)
    )

    origin_info = get_mock_port_info(request_data.origin)
    dest_info = get_mock_port_info(request_data.destination)

    expected_loading = datetime.now()
    required_delivery = datetime.combine(request_data.required_delivery_date, datetime.min.time())

    inputs = DecisionEngineInputs(
        cargo_type=request_data.cargo_type,
        cargo_quantity_t=request_data.cargo_quantity,
        origin_port_id=request_data.origin,
        destination_port_id=request_data.destination,
        expected_loading_date=expected_loading,
        required_delivery_date=required_delivery,
        number_of_voyages=request_data.number_of_voyages,
        contract_preference=request_data.contract_preference,
        vessel_specs_db=get_mock_vessel_db(),
        origin_port_info=origin_info,
        destination_port_info=dest_info
    )

    result = engine.evaluate(inputs)

    if result["status"] == "ERROR":
        raise HTTPException(status_code=400, detail=result)

    return result

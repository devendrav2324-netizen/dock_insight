"""
DockInsights — Routes (Shipping Routes) Endpoint.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import RouteResponse
from src.data.repository import RouteRepository

router = APIRouter(prefix="/routes", tags=["Routes"])


@router.get("", response_model=List[RouteResponse])
async def list_routes(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    """
    List shipping routes, optionally filtered by origin/destination port ID.

    Examples:
        GET /api/routes?origin=AUS_NEW
        GET /api/routes?destination=IND_VZG
        GET /api/routes?origin=AUS_NEW&destination=IND_VZG
    """
    repo = RouteRepository(db)
    routes = await repo.find(origin_port_id=origin, destination_port_id=destination)
    return [
        RouteResponse(
            origin_port_id=r.origin_port_id,
            origin_port_name=r.origin_port_name,
            origin_country=r.origin_country,
            destination_port_id=r.destination_port_id,
            destination_port_name=r.destination_port_name,
            great_circle_nm=r.great_circle_nm,
            est_sailing_distance_nm=r.est_sailing_distance_nm,
            typical_cargo=r.typical_cargo,
            routing_note=r.routing_note,
        )
        for r in routes
    ]

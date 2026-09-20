"""
DockInsights — Ports Routes.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_session
from src.api.serializers import PortCongestionResponse, PortResponse
from src.data.repository import CongestionRepository, PortRepository

router = APIRouter(prefix="/ports", tags=["Ports"])


@router.get("", response_model=List[PortResponse])
async def list_ports(
    country: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
):
    """List all ports, optionally filtered by country."""
    repo = PortRepository(db)
    ports = await repo.get_all(country=country)
    return [
        PortResponse(
            port_id=p.port_id,
            port_name=p.port_name,
            state=p.state,
            country=p.country,
            latitude=p.latitude,
            longitude=p.longitude,
            port_type=p.port_type,
            operator=p.operator,
            berths_total=p.berths_total,
            max_draft_m=p.max_draft_m,
            max_loa_m=p.max_loa_m,
            max_beam_m=p.max_beam_m,
            max_dwt=p.max_dwt,
            annual_capacity_mtpa=p.annual_capacity_mtpa,
            primary_cargo=p.primary_cargo,
        )
        for p in ports
    ]


@router.get("/{port_id}", response_model=PortResponse)
async def get_port(
    port_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Get a specific port by ID."""
    repo = PortRepository(db)
    port = await repo.get_by_id(port_id)
    if not port:
        raise HTTPException(status_code=404, detail=f"Port {port_id} not found")
    return PortResponse(
        port_id=port.port_id,
        port_name=port.port_name,
        state=port.state,
        country=port.country,
        latitude=port.latitude,
        longitude=port.longitude,
        port_type=port.port_type,
        operator=port.operator,
        berths_total=port.berths_total,
        max_draft_m=port.max_draft_m,
        max_loa_m=port.max_loa_m,
        max_beam_m=port.max_beam_m,
        max_dwt=port.max_dwt,
        annual_capacity_mtpa=port.annual_capacity_mtpa,
        primary_cargo=port.primary_cargo,
    )


@router.get("/{port_id}/congestion", response_model=Optional[PortCongestionResponse])
async def get_port_congestion(
    port_id: str,
    db: AsyncSession = Depends(get_session),
):
    """Get the latest congestion data for a port."""
    repo = CongestionRepository(db)
    congestion = await repo.get_latest(port_id)
    if not congestion:
        raise HTTPException(
            status_code=404, detail=f"No congestion data for port {port_id}"
        )
    return PortCongestionResponse(
        port_id=congestion.port_id,
        date=congestion.date,
        vessels_waiting=congestion.vessels_waiting,
        avg_waiting_time_days=congestion.avg_waiting_time_days,
        berth_occupancy_pct=congestion.berth_occupancy_pct,
    )

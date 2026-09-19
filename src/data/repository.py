"""
Charter-AI — Normalized Data Access Repository (V2).

All database queries are centralized here behind an async repository interface.
Supports all 12 normalized domain entities.
"""

from datetime import date, datetime
from typing import List, Optional
import pandas as pd
from sqlalchemy import select, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models import (
    Port,
    Vessel,
    VesselClassModel,
    Route,
    FreightRate,
    BunkerPrice,
    DryBulkIndex,
    CommodityPrice,
    Commodity,
    PortCongestion,
    Weather,
    EconomicIndicator,
    GeopoliticalEvent,
    VesselAISPosition,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# 1. Port Repository
# =============================================================================

class PortRepository:
    """Read and query operations for ports."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, country: Optional[str] = None) -> List[Port]:
        stmt = select(Port)
        if country:
            stmt = stmt.where(Port.country == country)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, port_id: str) -> Optional[Port]:
        result = await self.session.execute(
            select(Port).where(Port.port_id == port_id)
        )
        return result.scalar_one_or_none()


# =============================================================================
# 2. Vessel Repository
# =============================================================================

class VesselRepository:
    """Read operations for individual fleet assets."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self, vessel_class: Optional[str] = None) -> List[Vessel]:
        stmt = select(Vessel)
        if vessel_class:
            stmt = stmt.where(Vessel.vessel_class == vessel_class)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_id(self, vessel_id: str) -> Optional[Vessel]:
        result = await self.session.execute(
            select(Vessel).where(Vessel.vessel_id == vessel_id)
        )
        return result.scalar_one_or_none()

    async def get_by_imo(self, imo_number: str) -> Optional[Vessel]:
        result = await self.session.execute(
            select(Vessel).where(Vessel.imo_number == imo_number)
        )
        return result.scalar_one_or_none()

    async def get_available(self, target_date: Optional[date] = None) -> List[Vessel]:
        stmt = select(Vessel).where(Vessel.availability_status == "AVAILABLE")
        if target_date:
            stmt = stmt.where(
                or_(Vessel.available_from.is_(None), Vessel.available_from <= target_date)
            )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# =============================================================================
# 3. Vessel Class Repository
# =============================================================================

class VesselClassRepository:
    """Read operations for vessel classes."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> List[VesselClassModel]:
        result = await self.session.execute(select(VesselClassModel))
        return list(result.scalars().all())

    async def get_by_id(self, class_id: str) -> Optional[VesselClassModel]:
        result = await self.session.execute(
            select(VesselClassModel).where(VesselClassModel.class_id == class_id)
        )
        return result.scalar_one_or_none()


# =============================================================================
# 4. Route Repository
# =============================================================================

class RouteRepository:
    """Read operations for routes."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> List[Route]:
        result = await self.session.execute(select(Route))
        return list(result.scalars().all())

    async def find(
        self,
        origin_port_id: Optional[str] = None,
        destination_port_id: Optional[str] = None,
    ) -> List[Route]:
        stmt = select(Route)
        if origin_port_id:
            stmt = stmt.where(
                or_(Route.origin_port == origin_port_id, Route.origin_port_id == origin_port_id)
            )
        if destination_port_id:
            stmt = stmt.where(
                or_(Route.destination_port == destination_port_id, Route.destination_port_id == destination_port_id)
            )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# =============================================================================
# 5. Freight Rate Repository
# =============================================================================

class FreightRateRepository:
    """Read operations for freight rates."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_history(
        self,
        origin_port_id: str,
        destination_port_id: str,
        vessel_class: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[FreightRate]:
        """Fetch freight rate time series for a specific route and vessel class."""
        stmt = (
            select(FreightRate)
            .where(
                or_(FreightRate.origin == origin_port_id, FreightRate.origin_port_id == origin_port_id)
            )
            .where(
                or_(FreightRate.destination == destination_port_id, FreightRate.destination_port_id == destination_port_id)
            )
            .where(FreightRate.vessel_class == vessel_class)
            .order_by(FreightRate.date)
        )
        if start_date:
            stmt = stmt.where(FreightRate.date >= start_date)
        if end_date:
            stmt = stmt.where(FreightRate.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest(
        self,
        origin_port_id: str,
        destination_port_id: str,
        vessel_class: str,
    ) -> Optional[FreightRate]:
        """Fetch the most recent freight rate observation."""
        stmt = (
            select(FreightRate)
            .where(
                or_(FreightRate.origin == origin_port_id, FreightRate.origin_port_id == origin_port_id)
            )
            .where(
                or_(FreightRate.destination == destination_port_id, FreightRate.destination_port_id == destination_port_id)
            )
            .where(FreightRate.vessel_class == vessel_class)
            .order_by(FreightRate.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


# =============================================================================
# 6. Bunker Price Repository
# =============================================================================

class BunkerPriceRepository:
    """Read operations for bunker fuel prices."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest(self, location: str, fuel_type: str = "VLSFO") -> Optional[BunkerPrice]:
        stmt = (
            select(BunkerPrice)
            .where(BunkerPrice.location == location)
            .where(BunkerPrice.fuel_type == fuel_type)
            .order_by(BunkerPrice.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_history(
        self,
        location: str,
        fuel_type: str = "VLSFO",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[BunkerPrice]:
        stmt = (
            select(BunkerPrice)
            .where(BunkerPrice.location == location)
            .where(BunkerPrice.fuel_type == fuel_type)
            .order_by(BunkerPrice.date)
        )
        if start_date:
            stmt = stmt.where(BunkerPrice.date >= start_date)
        if end_date:
            stmt = stmt.where(BunkerPrice.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# =============================================================================
# 7. Dry Bulk Index Repository
# =============================================================================

class DryBulkIndexRepository:
    """Read operations for Baltic dry bulk indices."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest(self, index_name: str = "BDI") -> Optional[DryBulkIndex]:
        stmt = (
            select(DryBulkIndex)
            .where(DryBulkIndex.index_name == index_name)
            .order_by(DryBulkIndex.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_history(
        self,
        index_name: str = "BDI",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[DryBulkIndex]:
        stmt = (
            select(DryBulkIndex)
            .where(DryBulkIndex.index_name == index_name)
            .order_by(DryBulkIndex.date)
        )
        if start_date:
            stmt = stmt.where(DryBulkIndex.date >= start_date)
        if end_date:
            stmt = stmt.where(DryBulkIndex.date <= end_date)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# =============================================================================
# 8. Congestion, Weather & Events Repositories
# =============================================================================

class CongestionRepository:
    """Read operations for port congestion."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest(self, port_id: str) -> Optional[PortCongestion]:
        stmt = (
            select(PortCongestion)
            .where(or_(PortCongestion.port == port_id, PortCongestion.port_id == port_id))
            .order_by(PortCongestion.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class WeatherRepository:
    """Read operations for weather observations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest(self, port_id: str) -> Optional[Weather]:
        stmt = (
            select(Weather)
            .where(or_(Weather.port == port_id, Weather.port_id == port_id))
            .order_by(Weather.date.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()


class EventRepository:
    """Read operations for geopolitical and operational events."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_recent(self, days: int = 30) -> List[GeopoliticalEvent]:
        cutoff = date.fromordinal(date.today().toordinal() - days)
        stmt = (
            select(GeopoliticalEvent)
            .where(GeopoliticalEvent.date >= cutoff)
            .order_by(GeopoliticalEvent.date.desc())
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())


# =============================================================================
# 9. Vessel AIS Repository
# =============================================================================

class VesselAISRepository:
    """Read operations for vessel AIS positions."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_position(self, imo_number: str) -> Optional[VesselAISPosition]:
        stmt = (
            select(VesselAISPosition)
            .where(VesselAISPosition.imo_number == imo_number)
            .order_by(VesselAISPosition.timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

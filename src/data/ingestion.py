"""
Charter-AI — Data Ingestion Pipeline (V2).

Loads CSV datasets into SQL database models. Designed to be idempotent.
Uses normalized domain models and robust CSV parsing.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.data.db import get_sync_engine, get_sync_session_factory
from src.data.loaders.csv_loader import CSVLoader
from src.data.loaders.db_loader import DatabaseLoader
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
)
from src.utils.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def ingest_ports(session: Session, data_dir: Path) -> int:
    """Load ports.csv into the ports table."""
    loader = CSVLoader()
    csv_path = data_dir / "ports.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} does not exist.")
        return 0

    df = loader.load(csv_path)
    count = 0
    for _, row in df.iterrows():
        port = Port(
            port_id=str(row["port_id"]).strip(),
            port_name=str(row["port_name"]).strip(),
            country=str(row.get("country", "IND")).strip(),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            max_draft_m=float(row.get("max_draft_m", row.get("max_draft", 16.0))),
            max_loa_m=float(row.get("max_loa_m", row.get("max_loa", 260.0))),
            max_beam_m=float(row.get("max_beam_m", row.get("max_beam", 40.0))),
            cargo_handling_rate_mt_day=float(row.get("cargo_handling_rate_mt_day", row.get("cargo_handling_rate", 25000.0))),
            berth_count=int(row.get("berth_count", row.get("berths_total", row.get("berthing_capacity", 4)))),
            coal_terminal=bool(row.get("coal_terminal", True)),
            iron_ore_terminal=bool(row.get("iron_ore_terminal", False)),
            grain_terminal=bool(row.get("grain_terminal", False)),
            tidal_restriction=bool(row.get("tidal_restriction", False)),
            night_navigation_restriction=bool(row.get("night_navigation_restriction", False)),
            source=str(row.get("source", "SYNTHETIC_DEMO")),
            confidence_level=str(row.get("confidence_level", "LOW")),
            state=str(row.get("state", "")) if pd.notna(row.get("state")) else None,
            max_dwt=int(row["max_dwt"]) if pd.notna(row.get("max_dwt")) else None,
        )
        session.merge(port)
        count += 1

    session.commit()
    logger.info(f"Ingested {count} ports")
    return count


def ingest_vessels(session: Session, data_dir: Path) -> int:
    """Load vessels.csv into the vessels table."""
    loader = CSVLoader()
    csv_path = data_dir / "vessels.csv"
    if not csv_path.exists():
        logger.warning(f"{csv_path} does not exist.")
        return 0

    df = loader.load(csv_path)
    count = 0
    for _, row in df.iterrows():
        vessel = Vessel(
            vessel_id=str(row["vessel_id"]).strip(),
            imo_number=str(row.get("imo_number", f"9{count:06d}")).strip(),
            vessel_name=str(row.get("vessel_name", f"Bulk Carrier {row['vessel_id']}")).strip(),
            vessel_class=str(row.get("vessel_class", row.get("vessel_type", "Panamax"))).strip(),
            dwt=int(row["dwt"]),
            loa_m=float(row.get("loa_m", row.get("loa", 225.0))),
            beam_m=float(row.get("beam_m", row.get("beam", 32.2))),
            max_draft_m=float(row.get("max_draft_m", row.get("draft", 14.0))),
            service_speed_knots=float(row.get("service_speed_knots", row.get("speed", 13.0))),
            ballast_speed_knots=float(row.get("ballast_speed_knots", 13.5)),
            laden_speed_knots=float(row.get("laden_speed_knots", 12.5)),
            fuel_consumption_mt_day=float(row.get("fuel_consumption_mt_day", row.get("fuel_consumption", 30.0))),
            age_years=int(row.get("age_years", row.get("age", 5))),
            current_latitude=float(row["current_latitude"]) if pd.notna(row.get("current_latitude")) else None,
            current_longitude=float(row["current_longitude"]) if pd.notna(row.get("current_longitude")) else None,
            availability_status=str(row.get("availability_status", "AVAILABLE")),
            data_source=str(row.get("data_source", row.get("source", "SYNTHETIC_DEMO"))),
        )
        session.merge(vessel)
        count += 1

    session.commit()
    logger.info(f"Ingested {count} vessels")
    return count


def ingest_routes(session: Session, data_dir: Path) -> int:
    """Load routes.csv into routes table."""
    loader = CSVLoader()
    csv_path = data_dir / "routes.csv"
    if not csv_path.exists():
        return 0

    df = loader.load(csv_path)
    count = 0
    for _, row in df.iterrows():
        route = Route(
            route_id=str(row["route_id"]).strip(),
            origin_port=str(row.get("origin_port", row.get("origin"))).strip(),
            destination_port=str(row.get("destination_port", row.get("destination"))).strip(),
            distance_nm=float(row["distance_nm"]),
            typical_duration_days=float(row.get("typical_duration_days", float(row["distance_nm"]) / (13.0 * 24))),
            route_type=str(row.get("route_type", "direct")),
            seasonal_factor=float(row.get("seasonal_factor", 1.0)),
            source=str(row.get("source", "SYNTHETIC_DEMO")),
        )
        session.merge(route)
        count += 1

    session.commit()
    logger.info(f"Ingested {count} routes")
    return count


def ingest_all(data_dir: Optional[str] = None) -> Dict[str, int]:
    """
    Run the full data ingestion pipeline.
    """
    settings = get_settings()
    target_dir = Path(data_dir) if data_dir else Path(settings.raw_data_dir)

    session_factory = get_sync_session_factory()
    counts = {}

    with session_factory() as session:
        counts["ports"] = ingest_ports(session, target_dir)
        counts["vessels"] = ingest_vessels(session, target_dir)
        counts["routes"] = ingest_routes(session, target_dir)

    return counts

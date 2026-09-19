"""
Charter-AI — Central Data Provenance & Dataset Registry (V2).

Explicit, machine-readable provenance tracking for all maritime datasets, ML models,
and decision pipeline outputs. Defines controlled vocabulary and metadata registries.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DataMode(str, Enum):
    """Controlled vocabulary for dataset and model execution modes."""
    SYNTHETIC_DEMO = "SYNTHETIC_DEMO"
    VERIFIED_EXTERNAL = "VERIFIED_EXTERNAL"
    USER_PROVIDED = "USER_PROVIDED"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"


class SourceType(str, Enum):
    """Origin mechanism of the dataset."""
    GENERATED = "GENERATED"
    EXTERNAL_API = "EXTERNAL_API"
    MANUAL_UPLOAD = "MANUAL_UPLOAD"
    PIPELINE_DERIVED = "PIPELINE_DERIVED"
    UNVERIFIED = "UNVERIFIED"


class QualityStatus(str, Enum):
    """Validation cleanliness level of a dataset."""
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


class DataProvenance(BaseModel):
    """
    Metadata representation of a dataset's origin, mode, coverage, and quality.
    """
    dataset_name: str = Field(..., description="Canonical dataset name, e.g. freight_rates")
    dataset_version: str = Field(default="v2.0", description="Dataset version identifier")
    data_mode: DataMode = Field(default=DataMode.SYNTHETIC_DEMO, description="Execution mode")
    source_type: SourceType = Field(default=SourceType.GENERATED, description="Origin mechanism")
    source_name: str = Field(..., description="Human-readable provider or generator name")
    generator: Optional[str] = Field(default=None, description="Generating script or algorithm path")
    date_start: Optional[str] = Field(default=None, description="Earliest observation date (YYYY-MM-DD)")
    date_end: Optional[str] = Field(default=None, description="Latest observation date (YYYY-MM-DD)")
    row_count: int = Field(default=0, ge=0, description="Total row count")
    quality_status: QualityStatus = Field(default=QualityStatus.VALID, description="Cleanliness status")
    intended_use: str = Field(
        default="DEVELOPMENT_TESTING_DEMO",
        description="DEVELOPMENT_TESTING_DEMO or PRODUCTION_TRAINING"
    )
    reference_context: Optional[str] = Field(
        default=None,
        description="Optional emulated market or benchmark reference description"
    )
    is_verified_external: bool = Field(
        default=False,
        description="Strict boolean flag confirming whether data originates from a verified external source"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "data_mode": self.data_mode.value,
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "generator": self.generator,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "row_count": self.row_count,
            "quality_status": self.quality_status.value,
            "intended_use": self.intended_use,
            "reference_context": self.reference_context,
            "is_verified_external": self.is_verified_external,
        }


# =============================================================================
# Canonical Dataset Provenance Registry
# =============================================================================

DATASET_REGISTRY: Dict[str, DataProvenance] = {
    "freight_rates": DataProvenance(
        dataset_name="freight_rates",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=21920,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate dry bulk route freight rates ($/tonne)",
        is_verified_external=False,
    ),
    "dry_bulk_indices": DataProvenance(
        dataset_name="dry_bulk_indices",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=10960,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate Baltic Dry Bulk Index series (BDI, BCI, BPI, BSI, BHSI)",
        is_verified_external=False,
    ),
    "commodity_prices": DataProvenance(
        dataset_name="commodity_prices",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=6576,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate GlobalCoal, Platts, and USDA commodity benchmarks",
        is_verified_external=False,
    ),
    "bunker_prices": DataProvenance(
        dataset_name="bunker_prices",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=13152,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate Bunkerworld VLSFO and MGO fuel price series",
        is_verified_external=False,
    ),
    "congestion": DataProvenance(
        dataset_name="congestion",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=21920,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate port authority queueing, berth occupancy, and wait times",
        is_verified_external=False,
    ),
    "weather": DataProvenance(
        dataset_name="weather",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=21920,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate port meteorological telemetry (wind, wave, rain, cyclone alerts)",
        is_verified_external=False,
    ),
    "economic_indicators": DataProvenance(
        dataset_name="economic_indicators",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Deterministic Time-Series Generator",
        generator="scripts/seed_historical_timeseries.py",
        date_start="2019-01-01",
        date_end="2024-12-31",
        row_count=4384,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Synthetically generated to emulate USD/INR FX rates and manufacturing PMI metrics",
        is_verified_external=False,
    ),
    "events": DataProvenance(
        dataset_name="events",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Static Scenario Seed",
        generator="data/demo/events.csv",
        date_start="2024-01-15",
        date_end="2026-01-15",
        row_count=6,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Sample operational and geopolitical disruption events for development testing",
        is_verified_external=False,
    ),
    "ports": DataProvenance(
        dataset_name="ports",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Dry-Bulk Port Reference Database",
        generator="data/demo/ports.csv",
        date_start=None,
        date_end=None,
        row_count=15,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Dry bulk port specifications and bathymetry constraints for Indian and overseas ports",
        is_verified_external=False,
    ),
    "routes": DataProvenance(
        dataset_name="routes",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Maritime Corridor Database",
        generator="data/demo/routes.csv",
        date_start=None,
        date_end=None,
        row_count=14,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Dry bulk shipping routes and nautical mile distance specifications",
        is_verified_external=False,
    ),
    "vessels": DataProvenance(
        dataset_name="vessels",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local Candidate Fleet Specification Database",
        generator="data/demo/vessels.csv",
        date_start=None,
        date_end=None,
        row_count=10,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Candidate dry bulk fleet specifications across Handysize, Supramax, Panamax, and Capesize",
        is_verified_external=False,
    ),
    "vessel_ais": DataProvenance(
        dataset_name="vessel_ais",
        dataset_version="v2.0",
        data_mode=DataMode.SYNTHETIC_DEMO,
        source_type=SourceType.GENERATED,
        source_name="Local AIS Position Sample Database",
        generator="data/demo/vessel_ais.csv",
        date_start="2026-01-01",
        date_end="2026-01-01",
        row_count=6,
        quality_status=QualityStatus.VALID,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Sample AIS vessel telemetry positions for testing live tracking pipeline",
        is_verified_external=False,
    ),
}


def get_dataset_provenance(dataset_name: str) -> DataProvenance:
    """
    Retrieve provenance metadata for a registered dataset.
    Returns default UNKNOWN synthetic metadata if the dataset name is unregistered.
    """
    clean_name = dataset_name.lower().replace(".csv", "").strip()
    if clean_name in DATASET_REGISTRY:
        return DATASET_REGISTRY[clean_name]
    
    return DataProvenance(
        dataset_name=clean_name,
        dataset_version="v0.0-unregistered",
        data_mode=DataMode.UNKNOWN,
        source_type=SourceType.UNVERIFIED,
        source_name="Unregistered Dataset",
        quality_status=QualityStatus.WARNING,
        intended_use="DEVELOPMENT_TESTING_DEMO",
        reference_context="Unregistered dataset without verified provenance",
        is_verified_external=False,
    )


def list_registered_datasets() -> List[Dict[str, Any]]:
    """Return a list of all registered dataset provenance summaries."""
    return [prov.to_dict() for prov in DATASET_REGISTRY.values()]

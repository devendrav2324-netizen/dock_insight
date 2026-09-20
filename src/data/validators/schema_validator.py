"""
DockInsights — Schema Validator.

Validates tabular datasets against Pydantic model schemas.
Produces detailed field-level error diagnostics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.data.schemas import (
    PortSchema,
    VesselSchema,
    VesselClassSchema,
    RouteSchema,
    FreightRateSchema,
    BunkerPriceSchema,
    DryBulkIndexSchema,
    CommodityPriceSchema,
    CommoditySchema,
    CongestionSchema,
    WeatherSchema,
    EconomicIndicatorSchema,
    EventSchema,
    VesselAISSchema,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaValidationResult:
    dataset_name: str
    is_valid: bool
    total_rows: int
    valid_rows: int
    error_count: int
    errors: List[Dict[str, str]] = field(default_factory=list)


class SchemaValidator:
    """
    Validates DataFrames against defined Pydantic data models.
    """

    SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
        "ports": PortSchema,
        "vessels": VesselSchema,
        "vessel_classes": VesselClassSchema,
        "routes": RouteSchema,
        "freight_rates": FreightRateSchema,
        "bunker_prices": BunkerPriceSchema,
        "dry_bulk_indices": DryBulkIndexSchema,
        "commodity_prices": CommodityPriceSchema,
        "commodities": CommoditySchema,
        "congestion": CongestionSchema,
        "weather": WeatherSchema,
        "economic_indicators": EconomicIndicatorSchema,
        "events": EventSchema,
        "vessel_ais": VesselAISSchema,
    }

    def __init__(self):
        pass

    def get_schema_for_dataset(self, dataset_name: str) -> Optional[Type[BaseModel]]:
        """Find the matching schema class for a dataset identifier."""
        clean_name = dataset_name.lower().replace(".csv", "").strip()
        return self.SCHEMA_REGISTRY.get(clean_name)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        max_errors: int = 100,
    ) -> SchemaValidationResult:
        """
        Validate all rows in a DataFrame against the registered schema.
        """
        schema_cls = self.get_schema_for_dataset(dataset_name)
        if not schema_cls:
            raise ValueError(f"No registered schema found for dataset: {dataset_name}")

        errors = []
        valid_count = 0
        total_rows = len(df)

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            # Clean NaN to None
            cleaned_row = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
            try:
                schema_cls(**cleaned_row)
                valid_count += 1
            except ValidationError as ve:
                for err in ve.errors():
                    field_name = ".".join(str(loc) for loc in err["loc"])
                    msg = err["msg"]
                    if len(errors) < max_errors:
                        errors.append({
                            "row": idx,
                            "field": field_name,
                            "error": msg,
                            "value": str(cleaned_row.get(field_name, ""))
                        })

        is_valid = len(errors) == 0
        return SchemaValidationResult(
            dataset_name=dataset_name,
            is_valid=is_valid,
            total_rows=total_rows,
            valid_rows=valid_count,
            error_count=len(errors),
            errors=errors,
        )

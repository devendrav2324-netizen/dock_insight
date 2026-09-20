"""
DockInsights — Business Rule Validator.

Domain-specific integrity validation:
1. Missing required fields
2. Invalid units
3. Negative values
4. Impossible vessel dimensions
5. Invalid coordinates
6. Duplicate records
7. Future historical dates
8. Inconsistent vessel classes
"""

from datetime import date, datetime
from typing import Dict, List, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_vessel_class_dwt_ranges() -> dict[str, tuple[int, int]]:
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    res = {s.class_name: (int(s.dwt_min * 0.8), int(s.dwt_max * 1.25)) for s in specs}
    fallbacks = {
        "Ultramax": (60000, 72000),
        "Kamsarmax": (80000, 90000),
        "Post-Panamax": (85000, 120000),
    }
    for k, v in fallbacks.items():
        if k not in res:
            res[k] = v
    return res


class BusinessRuleValidator:
    """
    Maritime domain rule validator for data cleanliness and physical consistency.
    """

    VESSEL_CLASS_DWT_RANGES = _get_vessel_class_dwt_ranges()


    VALID_UNITS = {
        "freight_rate": ["per_tonne", "per_day", "usd_per_tonne", "usd_per_day", "USD/t", "USD/day"],
        "currency": ["USD", "INR"],
        "bunker": ["mt", "tonne", "USD/mt", "USD/tonne"],
        "speed": ["knots", "kts"],
        "distance": ["nm", "nautical_miles"],
        "economic": ["USD/bbl", "INR", "pct", "index_points", "value", "bbl", "tonnes", "mt", "USD/t", "USD/day"],
    }

    REQUIRED_FIELDS_BY_DATASET = {
        "ports": ["port_id", "port_name", "country", "latitude", "longitude", "max_draft_m", "max_loa_m", "max_beam_m"],
        "vessels": ["vessel_id", "imo_number", "vessel_name", "vessel_class", "dwt", "loa_m", "beam_m", "max_draft_m"],
        "routes": ["route_id", "origin_port", "destination_port", "distance_nm", "typical_duration_days"],
        "freight_rates": ["date", "origin", "destination", "vessel_class", "cargo_type", "freight_rate", "currency", "unit"],
        "bunker_prices": ["date", "location", "fuel_type", "price_usd_mt"],
        "dry_bulk_indices": ["date", "index_name", "value"],
        "congestion": ["date", "port", "vessels_waiting", "average_waiting_days", "berth_occupancy_pct"],
        "weather": ["date", "port"],
        "economic_indicators": ["date", "indicator_name", "value"],
        "events": ["date", "event_type", "severity"],
        "vessel_ais": ["imo_number", "timestamp", "latitude", "longitude", "speed_knots"],
    }

    PRIMARY_KEYS_BY_DATASET = {
        "ports": ["port_id"],
        "vessels": ["vessel_id"],
        "routes": ["route_id"],
        "freight_rates": ["date", "origin", "destination", "vessel_class"],
        "bunker_prices": ["date", "location", "fuel_type"],
        "dry_bulk_indices": ["date", "index_name"],
        "congestion": ["date", "port"],
        "weather": ["date", "port"],
        "economic_indicators": ["date", "indicator_name"],
        "events": ["date", "event_type", "affected_region"],
        "vessel_ais": ["imo_number", "timestamp"],
    }

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Run all 8 business rule checks on a dataset and return a structured report.
        """
        clean_name = dataset_name.lower().replace(".csv", "").strip()
        issues = []

        # 1. Missing required fields
        missing = self.check_missing_fields(df, clean_name)
        if missing:
            issues.extend(missing)

        # 2. Invalid units
        unit_issues = self.check_invalid_units(df, clean_name)
        if unit_issues:
            issues.extend(unit_issues)

        # 3. Negative values
        neg_issues = self.check_negative_values(df, clean_name)
        if neg_issues:
            issues.extend(neg_issues)

        # 4. Impossible vessel dimensions
        if clean_name in ("vessels", "vessel_classes"):
            dim_issues = self.check_impossible_vessel_dimensions(df)
            if dim_issues:
                issues.extend(dim_issues)

        # 5. Invalid coordinates
        if any(c in df.columns for c in ["latitude", "longitude", "current_latitude"]):
            coord_issues = self.check_invalid_coordinates(df)
            if coord_issues:
                issues.extend(coord_issues)

        # 6. Duplicate records
        dup_issues = self.check_duplicate_records(df, clean_name)
        if dup_issues:
            issues.extend(dup_issues)

        # 7. Future historical dates
        date_issues = self.check_future_historical_dates(df)
        if date_issues:
            issues.extend(date_issues)

        # 8. Inconsistent vessel classes
        if clean_name in ("vessels", "freight_rates"):
            class_issues = self.check_inconsistent_vessel_classes(df)
            if class_issues:
                issues.extend(class_issues)

        return {
            "dataset": clean_name,
            "total_rows": len(df),
            "is_valid": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues,
        }

    def check_missing_fields(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        required = self.REQUIRED_FIELDS_BY_DATASET.get(dataset_name, [])
        issues = []
        for col in required:
            # Check if column exists
            # Also allow common aliases
            col_found = False
            aliases = [col]
            if col == "max_draft_m": aliases.append("max_draft")
            elif col == "max_loa_m": aliases.append("max_loa")
            elif col == "max_beam_m": aliases.append("max_beam")
            elif col == "origin_port": aliases.append("origin")
            elif col == "destination_port": aliases.append("destination")
            elif col == "vessel_class": aliases.append("vessel_type")
            elif col == "indicator_name": aliases.append("indicator")
            elif col == "average_waiting_days": aliases.append("average_waiting_time")
            elif col == "berth_occupancy_pct": aliases.append("berth_occupancy")

            matched_col = next((a for a in aliases if a in df.columns), None)
            if not matched_col:
                issues.append(f"Missing required column: '{col}'")
            else:
                null_count = df[matched_col].isna().sum()
                if null_count > 0:
                    issues.append(f"Column '{matched_col}' has {null_count} null values")
        return issues

    def check_invalid_units(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        issues = []
        all_units = set()
        for ulist in self.VALID_UNITS.values():
            all_units.update(ulist)

        if "unit" in df.columns:
            invalid = df[df["unit"].notna() & ~df["unit"].isin(all_units)]
            if len(invalid) > 0:
                issues.append(f"Found {len(invalid)} records with unrecognized unit: {invalid['unit'].unique().tolist()}")
        if "currency" in df.columns:
            invalid_curr = df[df["currency"].notna() & ~df["currency"].isin(self.VALID_UNITS["currency"])]
            if len(invalid_curr) > 0:
                issues.append(f"Found {len(invalid_curr)} records with invalid currency: {invalid_curr['currency'].unique().tolist()}")
        return issues

    def check_negative_values(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        issues = []
        numeric_cols = df.select_dtypes(include=["number"]).columns
        # Exclude coordinates and seasonal factor from strict non-negative check
        cols_to_check = [c for c in numeric_cols if c not in ("latitude", "longitude", "current_latitude", "current_longitude", "heading")]
        for col in cols_to_check:
            negatives = df[df[col] < 0]
            if len(negatives) > 0:
                issues.append(f"Negative values detected in '{col}': {len(negatives)} occurrences (min: {df[col].min()})")
        return issues

    def check_impossible_vessel_dimensions(self, df: pd.DataFrame) -> List[str]:
        issues = []
        # Check draft
        draft_col = next((c for c in ["max_draft_m", "draft", "draft_max_m"] if c in df.columns), None)
        if draft_col:
            bad_draft = df[(df[draft_col] < 3.0) | (df[draft_col] > 30.0)]
            if len(bad_draft) > 0:
                issues.append(f"Impossible vessel draft in '{draft_col}': {len(bad_draft)} rows outside [3.0m, 30.0m]")

        # Check LOA
        loa_col = next((c for c in ["max_loa_m", "loa", "loa_max_m"] if c in df.columns), None)
        if loa_col:
            bad_loa = df[(df[loa_col] < 40.0) | (df[loa_col] > 450.0)]
            if len(bad_loa) > 0:
                issues.append(f"Impossible vessel LOA in '{loa_col}': {len(bad_loa)} rows outside [40m, 450m]")

        # Check Beam
        beam_col = next((c for c in ["max_beam_m", "beam", "beam_max_m"] if c in df.columns), None)
        if beam_col:
            bad_beam = df[(df[beam_col] < 8.0) | (df[beam_col] > 75.0)]
            if len(bad_beam) > 0:
                issues.append(f"Impossible vessel beam in '{beam_col}': {len(bad_beam)} rows outside [8m, 75m]")

        # Check DWT
        dwt_col = next((c for c in ["dwt", "dwt_max", "max_dwt"] if c in df.columns), None)
        if dwt_col:
            bad_dwt = df[(df[dwt_col] < 3000) | (df[dwt_col] > 500000)]
            if len(bad_dwt) > 0:
                issues.append(f"Impossible vessel DWT in '{dwt_col}': {len(bad_dwt)} rows outside [3,000, 500,000]")

        return issues

    def check_invalid_coordinates(self, df: pd.DataFrame) -> List[str]:
        issues = []
        lat_cols = [c for c in ["latitude", "current_latitude"] if c in df.columns]
        lon_cols = [c for c in ["longitude", "current_longitude"] if c in df.columns]

        for c in lat_cols:
            invalid = df[df[c].notna() & ((df[c] < -90.0) | (df[c] > 90.0))]
            if len(invalid) > 0:
                issues.append(f"Invalid latitude in '{c}': {len(invalid)} rows outside [-90.0, 90.0]")

        for c in lon_cols:
            invalid = df[df[c].notna() & ((df[c] < -180.0) | (df[c] > 180.0))]
            if len(invalid) > 0:
                issues.append(f"Invalid longitude in '{c}': {len(invalid)} rows outside [-180.0, 180.0]")

        return issues

    def check_duplicate_records(self, df: pd.DataFrame, dataset_name: str) -> List[str]:
        issues = []
        pk_cols = self.PRIMARY_KEYS_BY_DATASET.get(dataset_name)
        if pk_cols:
            # Map aliases if present
            mapped_cols = []
            for col in pk_cols:
                if col in df.columns:
                    mapped_cols.append(col)
                elif col == "vessel_class" and "vessel_type" in df.columns:
                    mapped_cols.append("vessel_type")
                elif col == "origin_port" and "origin" in df.columns:
                    mapped_cols.append("origin")
                elif col == "destination_port" and "destination" in df.columns:
                    mapped_cols.append("destination")
                elif col == "indicator_name" and "indicator" in df.columns:
                    mapped_cols.append("indicator")

            if len(mapped_cols) == len(pk_cols):
                dup_count = df.duplicated(subset=mapped_cols).sum()
                if dup_count > 0:
                    issues.append(f"Duplicate records found: {dup_count} duplicate rows on keys {mapped_cols}")
        return issues

    def check_future_historical_dates(self, df: pd.DataFrame) -> List[str]:
        issues = []
        date_cols = [c for c in ["date", "event_date", "timestamp"] if c in df.columns]
        # Maximum allowed historical year for valid historical maritime datasets
        max_allowed_year = 2030
        for c in date_cols:
            try:
                dt_series = pd.to_datetime(df[c], errors="coerce")
                future_dates = dt_series[dt_series.dt.year > max_allowed_year]
                if len(future_dates) > 0:
                    issues.append(f"Future dates detected in '{c}': {len(future_dates)} records with year > {max_allowed_year}")
            except Exception:
                pass
        return issues

    def check_inconsistent_vessel_classes(self, df: pd.DataFrame) -> List[str]:
        issues = []
        class_col = next((c for c in ["vessel_class", "vessel_type"] if c in df.columns), None)
        dwt_col = next((c for c in ["dwt", "dwt_max"] if c in df.columns), None)

        if class_col and dwt_col:
            for idx, row in df.iterrows():
                v_class = str(row[class_col]).capitalize()
                dwt_val = row[dwt_col]
                if pd.notna(dwt_val) and v_class in self.VESSEL_CLASS_DWT_RANGES:
                    min_dwt, max_dwt = self.VESSEL_CLASS_DWT_RANGES[v_class]
                    if dwt_val < (min_dwt * 0.8) or dwt_val > (max_dwt * 1.25):
                        issues.append(
                            f"Inconsistent vessel class '{v_class}' for DWT {dwt_val} at row {idx} (expected ~{min_dwt:,}-{max_dwt:,})"
                        )
                        if len(issues) >= 10:
                            issues.append("... (additional vessel class inconsistencies truncated)")
                            break
        return issues

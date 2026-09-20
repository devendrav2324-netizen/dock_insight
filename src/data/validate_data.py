"""
Data validation and provenance quality contract module for DockInsights.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src.data.provenance import get_dataset_provenance, DataMode, QualityStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)

def check_required_columns(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """Check if all expected columns are present in the DataFrame."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logger.error(f"Validation failed: Missing required columns: {missing}")
        return False
    return True

def check_missing_values(df: pd.DataFrame, threshold_pct: float = 0.1) -> Dict[str, float]:
    """
    Check for missing values in the DataFrame.
    Logs a warning if missing values for any column exceed the threshold.
    """
    missing_pct = df.isnull().mean().to_dict()
    warnings = {}
    for col, pct in missing_pct.items():
        if pct > threshold_pct:
            logger.warning(f"Column '{col}' has {pct:.1%} missing values (threshold: {threshold_pct:.1%})")
            warnings[col] = pct
    return warnings

def check_duplicates(df: pd.DataFrame, subset: List[str] = None) -> int:
    """
    Check for duplicate rows.
    Returns the number of duplicate rows found.
    """
    duplicates = df.duplicated(subset=subset).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows based on subset {subset}")
    else:
        logger.info(f"No duplicate rows found based on subset {subset}")
    return duplicates

def validate_dates(df: pd.DataFrame, date_columns: List[str]) -> bool:
    """Check if specified columns contain valid dates."""
    valid = True
    for col in date_columns:
        if col not in df.columns:
            continue
        try:
            # Check if parsing succeeds without errors (coerce puts NaT for errors)
            parsed = pd.to_datetime(df[col], errors='coerce')
            invalid_count = parsed.isna().sum() - df[col].isna().sum()
            if invalid_count > 0:
                logger.error(f"Column '{col}' contains {invalid_count} invalid date formats.")
                valid = False
        except Exception as e:
            logger.error(f"Failed to validate dates in '{col}': {e}")
            valid = False
    return valid

def check_numeric_finiteness(df: pd.DataFrame) -> List[str]:
    """Check for non-finite numeric values (Inf or NaN) in numeric columns."""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            issues.append(f"Column '{col}' contains {inf_count} infinite values (+/-Inf)")
    return issues

def check_date_coverage(df: pd.DataFrame, date_col: str = "date") -> Dict[str, Any]:
    """Measure actual date range coverage and detect date ordering issues."""
    if date_col not in df.columns:
        return {"date_col": date_col, "has_dates": False, "date_start": None, "date_end": None, "is_ordered": True}
    
    parsed = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if len(parsed) == 0:
        return {"date_col": date_col, "has_dates": False, "date_start": None, "date_end": None, "is_ordered": True}
    
    min_date = parsed.min().strftime("%Y-%m-%d")
    max_date = parsed.max().strftime("%Y-%m-%d")
    is_ordered = parsed.is_monotonic_increasing
    
    return {
        "date_col": date_col,
        "has_dates": True,
        "date_start": min_date,
        "date_end": max_date,
        "is_ordered": bool(is_ordered),
        "total_days_spanned": int((parsed.max() - parsed.min()).days) + 1,
    }

def validate_dataset_provenance(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Validates provenance metadata for a dataset against the central provenance registry.
    """
    prov = get_dataset_provenance(dataset_name)
    source_col = next((c for c in df.columns if c in ("source", "data_source")), None)
    
    df_sources = df[source_col].dropna().unique().tolist() if source_col else []
    
    # Check if df explicitly contains SYNTHETIC_DEMO or unverified strings
    has_synthetic_label = any(s == "SYNTHETIC_DEMO" for s in df_sources)
    has_external_claim = any("HISTORICAL" in str(s) or "REAL" in str(s) for s in df_sources if s != "SYNTHETIC_DEMO")
    
    return {
        "dataset_name": dataset_name,
        "registered_mode": prov.data_mode.value,
        "registered_source_name": prov.source_name,
        "has_source_column": source_col is not None,
        "df_source_values": df_sources,
        "has_synthetic_label": has_synthetic_label,
        "misleading_external_claims": has_external_claim,
        "is_verified_external": prov.is_verified_external,
        "provenance_valid": not has_external_claim,
    }

def generate_quality_report(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Generate a comprehensive data quality and provenance report."""
    coverage = check_date_coverage(df)
    prov_res = validate_dataset_provenance(df, dataset_name)
    finiteness_issues = check_numeric_finiteness(df)
    
    null_sum = int(df.isnull().sum().sum())
    dup_sum = int(df.duplicated().sum())
    
    status = QualityStatus.VALID
    if null_sum > 0 or dup_sum > 0 or len(finiteness_issues) > 0 or prov_res["misleading_external_claims"]:
        status = QualityStatus.WARNING
    if not prov_res["provenance_valid"]:
        status = QualityStatus.INVALID
        
    return {
        "dataset": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values_count": null_sum,
        "duplicate_rows": dup_sum,
        "finiteness_issues": finiteness_issues,
        "date_coverage": coverage,
        "provenance": prov_res,
        "quality_status": status.value,
        "column_types": df.dtypes.astype(str).to_dict()
    }


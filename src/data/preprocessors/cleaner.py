"""
DockInsights — Data Cleaner.

Handles missing values, whitespace stripping, and column type casting.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def clean_strings(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Trim leading/trailing whitespace from string columns."""
    df_clean = df.copy()
    cols = columns or df_clean.select_dtypes(include=["object"]).columns
    for col in cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    return df_clean


def convert_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """Cast DataFrame columns to specified types."""
    df_converted = df.copy()
    for col, dtype in type_mapping.items():
        if col not in df_converted.columns:
            continue
        try:
            if dtype == "datetime":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
            else:
                df_converted[col] = df_converted[col].astype(dtype)
        except Exception as e:
            logger.error(f"Error converting column '{col}' to {dtype}: {e}")
    return df_converted


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_values: Optional[Dict[str, Any]] = None,
    subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values by dropping or imputing.
    Strategies: 'drop', 'fill'.
    """
    df_out = df.copy()
    if strategy == "drop":
        df_out = df_out.dropna(subset=subset)
    elif strategy == "fill":
        if fill_values:
            df_out = df_out.fillna(value=fill_values)
        else:
            df_out = df_out.ffill().bfill()
    else:
        raise ValueError(f"Unknown missing value strategy: {strategy}")
    return df_out


class DataCleaner:
    """
    Configurable data cleaning pipeline.
    """

    def clean(
        self,
        df: pd.DataFrame,
        type_mapping: Optional[Dict[str, str]] = None,
        missing_strategy: str = "fill",
        fill_values: Optional[Dict[str, Any]] = None,
        string_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Runs complete cleaning workflow on a DataFrame.
        """
        df_clean = clean_strings(df, columns=string_cols)
        if type_mapping:
            df_clean = convert_types(df_clean, type_mapping)
        if missing_strategy:
            df_clean = handle_missing_values(df_clean, strategy=missing_strategy, fill_values=fill_values)
        return df_clean

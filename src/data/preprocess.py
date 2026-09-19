"""
Data preprocessing module for CharterAI.
"""
import pandas as pd
from typing import Dict, List
from src.utils.logging import get_logger

logger = get_logger(__name__)

def convert_types(df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Convert column types in a DataFrame.
    """
    df_cleaned = df.copy()
    for col, dtype in type_mapping.items():
        if col in df_cleaned.columns:
            try:
                if dtype == 'datetime':
                    df_cleaned[col] = pd.to_datetime(df_cleaned[col])
                else:
                    df_cleaned[col] = df_cleaned[col].astype(dtype)
            except Exception as e:
                logger.error(f"Failed to convert column '{col}' to {dtype}: {e}")
    return df_cleaned

def clean_strings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from string columns.
    """
    df_cleaned = df.copy()
    for col in columns:
        if col in df_cleaned.columns:
            # Only apply str.strip if it has a .str accessor (or cast to string first if we expect them to be strings)
            try:
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            except Exception as e:
                logger.error(f"Failed to clean string column '{col}': {e}")
    return df_cleaned

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_values: Dict[str, any] = None) -> pd.DataFrame:
    """
    Handle missing values by dropping or filling.
    """
    df_cleaned = df.copy()
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna()
        logger.info(f"Dropped rows with missing values. Rows remaining: {len(df_cleaned)}")
    elif strategy == 'fill' and fill_values:
        df_cleaned = df_cleaned.fillna(fill_values)
        logger.info(f"Filled missing values using provided fill_values.")
    return df_cleaned

def preprocess_freight_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Specific preprocessing for freight rates dataset."""
    type_map = {
        'date': 'datetime',
        'freight_rate': 'float64'
    }
    df_clean = convert_types(df, type_map)
    df_clean = clean_strings(df_clean, ['origin', 'destination', 'vessel_type', 'currency', 'unit', 'route'])
    return df_clean

def preprocess_vessels(df: pd.DataFrame) -> pd.DataFrame:
    """Specific preprocessing for vessels dataset."""
    type_map = {
        'dwt': 'int64',
        'loa': 'float64',
        'beam': 'float64',
        'draft': 'float64',
        'speed': 'float64',
        'fuel_consumption': 'float64',
        'age': 'int64'
    }
    df_clean = convert_types(df, type_map)
    df_clean = clean_strings(df_clean, ['vessel_id', 'vessel_type'])
    return df_clean

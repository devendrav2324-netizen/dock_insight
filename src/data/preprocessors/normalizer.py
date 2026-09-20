"""
DockInsights — Data Normalizer.

Standardizes units, timestamps, and maritime entity codes.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataNormalizer:
    """
    Standardizes timestamps, units, and identifiers.
    """

    PORT_CODE_ALIASES: Dict[str, str] = {
        "VISAKHAPATNAM": "IND_VZG",
        "VIZAG": "IND_VZG",
        "VZG": "IND_VZG",
        "PARADIP": "IND_PAR",
        "PARADEEP": "IND_PAR",
        "PAR": "IND_PAR",
        "DHAMRA": "IND_DHM",
        "DHM": "IND_DHM",
        "IND_DHA": "IND_DHM",
        "GANGAVARAM": "IND_GVM",
        "GVM": "IND_GVM",
        "HALDIA": "IND_HLD",
        "HLD": "IND_HLD",
        "GOPALPUR": "IND_GOP",
        "GOP": "IND_GOP",
        "NEWCASTLE": "AUS_NEW",
        "HAY POINT": "AUS_HAY",
        "GLADSTONE": "AUS_GLD",
        "TABANG": "IDN_TAB",
        "INA_TAB": "IDN_TAB",
        "BALIKPAPAN": "IDN_BAL",
        "RICHARDS BAY": "ZAF_RIC",
        "HAMPTON ROADS": "USA_HAM",
        "VOSTOCHNY": "RUS_VOS",
    }

    UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
        # Currency to USD
        "currency": {
            "USD": 1.0,
            "INR": 1.0 / 83.5,
        },
        # Distance to Nautical Miles
        "distance": {
            "nm": 1.0,
            "km": 0.539957,
            "miles": 0.868976,
        },
        # Weight to Metric Tonnes
        "weight": {
            "t": 1.0,
            "mt": 1.0,
            "tonne": 1.0,
            "kg": 0.001,
            "lbs": 0.000453592,
        },
    }

    def normalize_timestamps(
        self,
        df: pd.DataFrame,
        date_cols: Optional[List[str]] = None,
        as_date_only: bool = True,
    ) -> pd.DataFrame:
        """
        Convert date/timestamp columns to standardized ISO 8601 string or date object.
        """
        df_out = df.copy()
        cols = date_cols or [c for c in ["date", "event_date", "timestamp", "source_date", "available_from"] if c in df_out.columns]

        for col in cols:
            try:
                dt_series = pd.to_datetime(df_out[col], errors="coerce")
                if as_date_only and col != "timestamp":
                    df_out[col] = dt_series.dt.date
                else:
                    df_out[col] = dt_series
            except Exception as e:
                logger.error(f"Failed normalizing timestamps in {col}: {e}")

        return df_out

    def normalize_port_codes(
        self,
        df: pd.DataFrame,
        port_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize port names and informal codes to standardized UN/LOCODE-style identifiers.
        """
        df_out = df.copy()
        cols = port_cols or [c for c in ["port", "port_id", "origin", "destination", "origin_port", "destination_port"] if c in df_out.columns]

        for col in cols:
            def _map_code(val):
                if pd.isna(val):
                    return val
                s = str(val).strip().upper()
                return self.PORT_CODE_ALIASES.get(s, s)

            df_out[col] = df_out[col].apply(_map_code)

        return df_out

    def normalize_units(
        self,
        df: pd.DataFrame,
        freight_rate_col: str = "freight_rate",
        currency_col: str = "currency",
        unit_col: str = "unit",
    ) -> pd.DataFrame:
        """
        Normalize freight rate currencies and units into USD per metric tonne.
        """
        df_out = df.copy()
        if currency_col in df_out.columns and freight_rate_col in df_out.columns:
            # If INR, convert to USD
            mask_inr = df_out[currency_col].str.upper() == "INR"
            if mask_inr.any():
                df_out.loc[mask_inr, freight_rate_col] = df_out.loc[mask_inr, freight_rate_col] * self.UNIT_CONVERSIONS["currency"]["INR"]
                df_out.loc[mask_inr, currency_col] = "USD"

        return df_out

"""
Charter-AI — Feature Engineering.

Transforms raw time-series data into ML-ready features.
All feature logic is centralized here so that training, inference,
and backtesting share identical feature pipelines.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def add_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: List[int] = None,
) -> pd.DataFrame:
    """
    Add lagged versions of a column.

    Args:
        df: DataFrame with a DatetimeIndex or sorted date column.
        column: Column name to lag.
        lags: List of lag periods (in rows). Defaults to [1, 3, 7, 14, 30].

    Returns:
        DataFrame with new lag columns added.
    """
    if lags is None:
        lags = [1, 3, 7, 14, 30]

    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int] = None,
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation features.

    Args:
        df: DataFrame sorted by date.
        column: Column to compute rolling statistics on.
        windows: Rolling window sizes. Defaults to [7, 14, 30, 60].

    Returns:
        DataFrame with rolling mean and std columns.
    """
    if windows is None:
        windows = [7, 14, 30, 60]

    for w in windows:
        df[f"{column}_rmean_{w}"] = df[column].rolling(window=w, min_periods=1).mean()
        df[f"{column}_rstd_{w}"] = df[column].rolling(window=w, min_periods=1).std()

    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar-based features from a date column.

    Features: day_of_week, day_of_month, month, quarter, week_of_year,
    is_month_start, is_month_end, is_monsoon, is_cyclone_season.
    """
    dates = pd.to_datetime(df[date_col])

    df["day_of_week"] = dates.dt.dayofweek
    df["day_of_month"] = dates.dt.day
    df["month"] = dates.dt.month
    df["quarter"] = dates.dt.quarter
    df["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    df["is_month_start"] = dates.dt.is_month_start.astype(int)
    df["is_month_end"] = dates.dt.is_month_end.astype(int)

    # Bay of Bengal cyclone season (Apr–Jun, Oct–Dec)
    df["is_cyclone_season"] = dates.dt.month.isin({4, 5, 6, 10, 11, 12}).astype(int)

    # Southwest monsoon season (Jun–Sep)
    df["is_monsoon"] = dates.dt.month.isin({6, 7, 8, 9}).astype(int)

    return df


def add_rate_of_change(
    df: pd.DataFrame,
    column: str,
    periods: List[int] = None,
) -> pd.DataFrame:
    """
    Add percentage rate-of-change features.

    Args:
        column: Column to compute returns on.
        periods: Lookback periods. Defaults to [1, 7, 30].
    """
    if periods is None:
        periods = [1, 7, 30]

    for p in periods:
        df[f"{column}_pct_change_{p}"] = df[column].pct_change(periods=p)

    return df


def build_freight_features(
    freight_df: pd.DataFrame,
    economic_df: Optional[pd.DataFrame] = None,
    congestion_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full feature matrix for freight rate forecasting.

    This is the master feature pipeline. It chains:
    1. Calendar features
    2. Lag features on freight rates
    3. Rolling statistics on freight rates
    4. Rate-of-change on freight rates
    5. Merge economic indicators (if provided)
    6. Merge congestion data (if provided)

    Args:
        freight_df: Historical freight rates (must have 'date' and
                    'freight_rate_usd_per_day' columns).
        economic_df: Economic indicators with 'date' column.
        congestion_df: Port congestion with 'date' and 'port_id' columns.

    Returns:
        Feature matrix DataFrame ready for model training.
    """
    df = freight_df.copy().sort_values("date").reset_index(drop=True)

    # Calendar
    df = add_calendar_features(df, date_col="date")

    # Freight rate features
    rate_col = "freight_rate_usd_per_day"
    df = add_lag_features(df, rate_col)
    df = add_rolling_features(df, rate_col)
    df = add_rate_of_change(df, rate_col)

    # BDI proxy features
    if "bdi_proxy_index" in df.columns:
        df = add_lag_features(df, "bdi_proxy_index", lags=[1, 7, 14])
        df = add_rolling_features(df, "bdi_proxy_index", windows=[7, 30])

    # Merge economic indicators
    if economic_df is not None and not economic_df.empty:
        econ = economic_df.copy()
        econ["date"] = pd.to_datetime(econ["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = pd.merge_asof(
            df.sort_values("date"),
            econ.sort_values("date"),
            on="date",
            direction="backward",
            suffixes=("", "_econ"),
        )

    # Merge congestion
    if congestion_df is not None and not congestion_df.empty:
        cong = congestion_df.copy()
        cong["date"] = pd.to_datetime(cong["date"]).dt.date
        # Merge on date + port_id (destination)
        if "destination_port_id" in df.columns and "port_id" in cong.columns:
            df = pd.merge_asof(
                df.sort_values("date"),
                cong.rename(columns={"port_id": "destination_port_id"}).sort_values("date"),
                on="date",
                by="destination_port_id",
                direction="backward",
                suffixes=("", "_cong"),
            )

    logger.info(
        f"Feature matrix shape: {df.shape} "
        f"({df.shape[1] - 1} features)"
    )
    return df

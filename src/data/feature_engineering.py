"""
Feature engineering module for CharterAI freight forecasting.
Provides reusable functions to create time-series features while avoiding data leakage.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple

def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for a specific column.
    
    Args:
        df: Input DataFrame (must be sorted chronologically).
        column: The column to create lags for.
        lags: List of integer lag periods.
        
    Returns:
        DataFrame with new lag columns added.
    """
    df_feat = df.copy()
    for lag in lags:
        df_feat[f'{column}_lag_{lag}'] = df_feat[column].shift(lag)
    return df_feat

def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """
    Create rolling mean and standard deviation features.
    To prevent data leakage, rolling features are shifted by 1 so the current step 
    only sees past data.
    
    Args:
        df: Input DataFrame (must be sorted chronologically).
        column: The column to calculate rolling features for.
        windows: List of integer window sizes.
        
    Returns:
        DataFrame with new rolling features added.
    """
    df_feat = df.copy()
    for window in windows:
        # Shift by 1 to avoid including the current row's value (prevent leakage)
        shifted_col = df_feat[column].shift(1)
        df_feat[f'{column}_rolling_mean_{window}'] = shifted_col.rolling(window=window, min_periods=1).mean()
        df_feat[f'{column}_rolling_std_{window}'] = shifted_col.rolling(window=window, min_periods=1).std().fillna(0)
    return df_feat

def create_percentage_changes(df: pd.DataFrame, column: str, periods: List[int]) -> pd.DataFrame:
    """
    Create percentage change features over specified periods.
    
    Args:
        df: Input DataFrame (must be sorted chronologically).
        column: The column to calculate percentage changes for.
        periods: List of integer periods for calculating changes.
        
    Returns:
        DataFrame with new percentage change features added.
    """
    df_feat = df.copy()
    for period in periods:
        df_feat[f'{column}_pct_change_{period}'] = df_feat[column].pct_change(periods=period)
    return df_feat

def create_time_components(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Extract day, week, month, and year components from a datetime column.
    
    Args:
        df: Input DataFrame.
        date_column: Name of the datetime column.
        
    Returns:
        DataFrame with new time component features added.
    """
    df_feat = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_feat[date_column]):
        df_feat[date_column] = pd.to_datetime(df_feat[date_column])
        
    df_feat['day_of_week'] = df_feat[date_column].dt.dayofweek
    df_feat['day_of_month'] = df_feat[date_column].dt.day
    df_feat['day_of_year'] = df_feat[date_column].dt.dayofyear
    df_feat['week_of_year'] = df_feat[date_column].dt.isocalendar().week.astype(int)
    df_feat['month'] = df_feat[date_column].dt.month
    df_feat['year'] = df_feat[date_column].dt.year
    df_feat['quarter'] = df_feat[date_column].dt.quarter
    
    return df_feat

def create_seasonal_features(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Create cyclical seasonal features (sine/cosine transformations) for months and days 
    to capture cyclic nature of time.
    
    Args:
        df: Input DataFrame.
        date_column: Name of the datetime column.
        
    Returns:
        DataFrame with new seasonal sine/cosine features.
    """
    df_feat = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_feat[date_column]):
        df_feat[date_column] = pd.to_datetime(df_feat[date_column])
        
    month = df_feat[date_column].dt.month
    day_of_year = df_feat[date_column].dt.dayofyear
    
    # Months in a year: 12
    df_feat['month_sin'] = np.sin(2 * np.pi * month / 12.0)
    df_feat['month_cos'] = np.cos(2 * np.pi * month / 12.0)
    
    # Days in a year: 365.25
    df_feat['day_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_feat['day_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    
    return df_feat

def chronological_train_val_test_split(
    df: pd.DataFrame, 
    date_column: str, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically to prevent data leakage.
    
    Args:
        df: Input DataFrame.
        date_column: Column to sort by.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    df_sorted = df.sort_values(by=date_column).reset_index(drop=True)
    
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

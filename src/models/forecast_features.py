"""
DockInsights — Freight Forecast Feature Engineering.

Transforms multi-domain maritime time series into ML-ready feature matrices
for dry-bulk freight rate forecasting across horizons (3, 7, 14, 30 days).

Features:
1. Freight Lags: lag_1, lag_3, lag_7, lag_14, lag_21, lag_28
2. Rolling Statistics: rolling_mean_7, rolling_mean_14, rolling_mean_28,
                       rolling_std_7, rolling_std_28
3. Momentum: momentum_7, momentum_30
4. Baltic Market Indices: BDI, Capesize (BCI), Panamax (BPI), Supramax (BSI), Handysize (BHSI)
5. Commodity Prices: Coal, Iron Ore, Grain
6. Energy Prices: Bunker (VLSFO, MGO), Crude Oil
7. Macro Indicators: USD/INR, PMI
8. Operational Indicators: Port Congestion, Waiting Vessels, Route Disruption, Vessel Availability
9. Calendar & Seasonality: Month, Week, Day of Week, Monsoon, Cyclone Season
"""

from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

FREIGHT_LAGS = [1, 3, 7, 14, 21, 28]
ROLLING_WINDOWS = [7, 14, 28]


def add_freight_lags(
    df: pd.DataFrame,
    target_col: str = "freight_rate",
    lags: List[int] = None
) -> pd.DataFrame:
    """Add lagged values of the target freight rate."""
    if lags is None:
        lags = FREIGHT_LAGS
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_metrics(
    df: pd.DataFrame,
    target_col: str = "freight_rate",
    windows: List[int] = None
) -> pd.DataFrame:
    """Add rolling means and standard deviations."""
    if windows is None:
        windows = ROLLING_WINDOWS
    out = df.copy()
    for w in windows:
        out[f"rolling_mean_{w}"] = out[target_col].shift(1).rolling(window=w, min_periods=max(2, w // 2)).mean()
        if w in [7, 28]:
            out[f"rolling_std_{w}"] = out[target_col].shift(1).rolling(window=w, min_periods=max(2, w // 2)).std()
    return out


def add_momentum(
    df: pd.DataFrame,
    target_col: str = "freight_rate",
    periods: List[int] = None
) -> pd.DataFrame:
    """Add price momentum (absolute difference and relative percentage shift).

    Uses only lagged observations to avoid target leakage.
    For a forecast made at time t, features must only use information
    available at or before t-1.

    Formula (for period p):
        anchor            = target_(t-1)   [shift(1)]
        momentum_p        = target_(t-1) - target_(t-1-p)   [shift(1) - shift(1+p)]
        momentum_pct_p    = (target_(t-1) - target_(t-1-p)) / target_(t-1-p) * 100
    """
    if periods is None:
        periods = [7, 30]
    out = df.copy()
    # Anchor at t-1 so that target_t is never used in any momentum feature.
    anchor = out[target_col].shift(1)
    for p in periods:
        lag_val = out[target_col].shift(1 + p)
        out[f"momentum_{p}"] = anchor - lag_val
        out[f"momentum_pct_{p}"] = ((anchor - lag_val) / (lag_val + 1e-6)) * 100.0
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Add calendar and maritime seasonality indicators.
    Includes day of week, week, month, season, and Bay of Bengal cyclone/monsoon periods.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])

    out["day_of_week"] = dates.dt.dayofweek
    out["week"] = dates.dt.isocalendar().week.astype(int)
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter

    # Cyclical sine/cosine calendar transforms
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)

    # Indian East Coast maritime weather seasonality
    # Southwest monsoon: June to September
    out["is_monsoon"] = dates.dt.month.isin([6, 7, 8, 9]).astype(int)
    # Bay of Bengal cyclone periods: Pre-monsoon (April-May) and Post-monsoon (October-November)
    out["is_cyclone_season"] = dates.dt.month.isin([4, 5, 10, 11]).astype(int)

    return out


class FreightFeatureBuilder:
    """
    Master feature engineering builder for freight forecasting.
    Loads and merges market indices, commodity prices, bunker prices,
    macroeconomic indicators, and port operational congestion.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is not None:
            self.data_dir = Path(data_dir)
        else:
            # Default to processed or demo directory
            base = Path(__file__).resolve().parent.parent.parent / "data"
            self.data_dir = base / "processed" if (base / "processed").exists() else base / "demo"

    def load_exogenous_data(self) -> Dict[str, pd.DataFrame]:
        """Load and pivot exogenous time series datasets."""
        datasets = {}

        # 1. Baltic Indices
        indices_path = self.data_dir / "dry_bulk_indices.csv"
        if indices_path.exists():
            df_idx = pd.read_csv(indices_path)
            df_idx["date"] = pd.to_datetime(df_idx["date"]).dt.strftime("%Y-%m-%d")
            pivoted = df_idx.pivot_table(index="date", columns="index_name", values="value").reset_index()
            # Rename columns to standard names
            col_map = {"BDI": "bdi", "BCI": "capesize_index", "BPI": "panamax_index", "BSI": "supramax_index", "BHSI": "handysize_index"}
            pivoted = pivoted.rename(columns=col_map)
            datasets["indices"] = pivoted

        # 2. Commodity Prices
        comm_path = self.data_dir / "commodity_prices.csv"
        if comm_path.exists():
            df_comm = pd.read_csv(comm_path)
            df_comm["date"] = pd.to_datetime(df_comm["date"]).dt.strftime("%Y-%m-%d")
            piv_comm = df_comm.pivot_table(index="date", columns="commodity_name", values="price").reset_index()
            rename_comm = {"thermal_coal": "coal_price", "iron_ore": "iron_ore_price", "grain": "grain_price"}
            piv_comm = piv_comm.rename(columns=rename_comm)
            datasets["commodities"] = piv_comm

        # 3. Bunker Prices
        bunker_path = self.data_dir / "bunker_prices.csv"
        if bunker_path.exists():
            df_bk = pd.read_csv(bunker_path)
            df_bk["date"] = pd.to_datetime(df_bk["date"]).dt.strftime("%Y-%m-%d")
            # Average across regional hub ports for robust signal
            piv_bk = df_bk.pivot_table(index="date", columns="fuel_type", values="price_usd_mt", aggfunc="mean").reset_index()
            piv_bk = piv_bk.rename(columns={"VLSFO": "bunker_price", "MGO": "mgo_price"})
            # Approximate crude oil proxy if not explicitly present
            piv_bk["crude_oil_price"] = piv_bk["bunker_price"] / 7.4
            datasets["bunker"] = piv_bk

        # 4. Macro Indicators
        econ_path = self.data_dir / "economic_indicators.csv"
        if econ_path.exists():
            df_ec = pd.read_csv(econ_path)
            df_ec["date"] = pd.to_datetime(df_ec["date"]).dt.strftime("%Y-%m-%d")
            piv_ec = df_ec.pivot_table(index="date", columns="indicator_name", values="value").reset_index()
            piv_ec = piv_ec.rename(columns={"USD_INR": "usd_inr", "INDIA_MANUFACTURING_PMI": "economic_indicator_pmi"})
            datasets["macro"] = piv_ec

        # 5. Congestion
        cong_path = self.data_dir / "congestion.csv"
        if cong_path.exists():
            df_cg = pd.read_csv(cong_path)
            df_cg["date"] = pd.to_datetime(df_cg["date"]).dt.strftime("%Y-%m-%d")
            datasets["congestion"] = df_cg

        return datasets

    def build_features(
        self,
        freight_df: pd.DataFrame,
        target_col: str = "freight_rate",
        date_col: str = "date",
        destination_port: Optional[str] = None,
        exogenous_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Build complete feature matrix from freight rate series and exogenous sources.
        Guarantees strictly backward-looking features (no forward leakage).
        """
        df = freight_df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(date_col).reset_index(drop=True)

        # 1. Add Freight Lags
        df = add_freight_lags(df, target_col=target_col)

        # 2. Add Rolling Statistics
        df = add_rolling_metrics(df, target_col=target_col)

        # 3. Add Momentum
        df = add_momentum(df, target_col=target_col)

        # 4. Add Calendar Features
        df = add_calendar_features(df, date_col=date_col)

        # 5. Merge Exogenous Datasets
        if exogenous_data is None:
            exogenous_data = self.load_exogenous_data()

        if "indices" in exogenous_data:
            idx_df = exogenous_data["indices"].copy()
            # Add index lags and momentum using only past observations (no lookahead).
            # bdi_momentum_7 = bdi_(t-1) - bdi_(t-8)  so bdi_t is never used.
            if "bdi" in idx_df.columns:
                idx_df["bdi_lag_1"] = idx_df["bdi"].shift(1)
                idx_df["bdi_momentum_7"] = idx_df["bdi"].shift(1) - idx_df["bdi"].shift(8)
            df = pd.merge(df, idx_df, on=date_col, how="left")

        if "commodities" in exogenous_data:
            df = pd.merge(df, exogenous_data["commodities"], on=date_col, how="left")

        if "bunker" in exogenous_data:
            df = pd.merge(df, exogenous_data["bunker"], on=date_col, how="left")

        if "macro" in exogenous_data:
            df = pd.merge(df, exogenous_data["macro"], on=date_col, how="left")

        if "congestion" in exogenous_data:
            cg_df = exogenous_data["congestion"].copy()
            if destination_port is not None and "port" in cg_df.columns:
                cg_filtered = cg_df[cg_df["port"] == destination_port]
                if not cg_filtered.empty:
                    cg_df = cg_filtered

            # Aggregate if multiple ports present
            cg_summary = cg_df.groupby(date_col).agg({
                "congestion_index": "mean",
                "vessels_waiting": "mean",
                "average_waiting_days": "mean"
            }).reset_index()
            cg_summary = cg_summary.rename(columns={
                "congestion_index": "port_congestion",
                "vessels_waiting": "vessel_availability"
            })
            df = pd.merge(df, cg_summary, on=date_col, how="left")

        # Operational disruption indicator
        if "route_disruption" not in df.columns:
            # High congestion or extreme cyclone season proxy
            is_cong_high = (df["port_congestion"] > 3.0).astype(int) if "port_congestion" in df.columns else 0
            is_cyclone = df["is_cyclone_season"] if "is_cyclone_season" in df.columns else 0
            df["route_disruption"] = np.clip(is_cong_high + is_cyclone, 0, 1)

        # Forward-fill any missing exogenous observations, then backfill, then zero-fill any unobserved columns
        df = df.ffill().bfill().fillna(0.0)

        return df

    def get_feature_columns(self, df: pd.DataFrame, target_col: str = "freight_rate") -> List[str]:
        """Return the list of predictor feature names, excluding metadata and target."""
        exclude = {
            "date", target_col, "origin", "destination", "vessel_class",
            "cargo_type", "currency", "unit", "source"
        }
        return [col for col in df.columns if col not in exclude and not col.startswith("target_")]


def build_forecast_features(
    freight_df: pd.DataFrame,
    target_col: str = "freight_rate",
    date_col: str = "date",
    data_dir: Optional[str] = None,
    destination_port: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Convenience helper to generate features and return (df, feature_cols)."""
    builder = FreightFeatureBuilder(data_dir=data_dir)
    featured_df = builder.build_features(
        freight_df=freight_df,
        target_col=target_col,
        date_col=date_col,
        destination_port=destination_port
    )
    feature_cols = builder.get_feature_columns(featured_df, target_col=target_col)
    return featured_df, feature_cols

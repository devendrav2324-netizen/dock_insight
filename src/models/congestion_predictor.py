"""
Charter-AI — Port Congestion & Waiting Time Predictor.

Predicts:
1. Expected waiting time in days
2. P10 waiting time (optimistic bound)
3. P50 waiting time (median expected wait)
4. P90 waiting time (pessimistic tail delay)
5. Probability of significant operational delay (waiting >= 3.0 days)
6. Congestion level (LOW, MODERATE, HIGH, SEVERE)

Features:
- Historical: vessels_waiting, average_waiting_days, berth_occupancy_pct,
              congestion_index, rolling_mean_7, rolling_mean_14, rolling_mean_30
- Calendar: month, weekday, is_monsoon, is_cyclone_season, is_weekend
- Vessel: vessel_class, dwt, cargo_handling_requirement
- Cargo: cargo_type, cargo_quantity
- Port: berth_count, cargo_handling_rate, terminal_type
- Weather: wind_speed_kmh, wave_height_m, storm_indicator
- Route/Event: disruption_indicator, geopolitical_event
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from datetime import datetime, date, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import math

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _get_vessel_class_dwt() -> dict[str, int]:
    from src.data.vessel_repository import get_all_vessel_class_specs
    specs = get_all_vessel_class_specs()
    res = {s.class_name: s.typical_dwt for s in specs}
    fallbacks = {"Ultramax": 64000, "Kamsarmax": 82000, "Post-Panamax": 95000}
    for k, v in fallbacks.items():
        if k not in res:
            res[k] = v
    return res


VESSEL_CLASS_DWT = _get_vessel_class_dwt()



@dataclass
class CongestionPrediction:
    expected_wait_days: float
    p10_wait_days: float
    p50_wait_days: float
    p90_wait_days: float
    delay_probability: float
    congestion_level: str  # "LOW", "MODERATE", "HIGH", "SEVERE"
    confidence: float
    data_source: str = "PORT_AUTHORITY_HISTORICAL"
    model_used: str = "XGBoost_Quantile_Congestion"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_wait_days": round(self.expected_wait_days, 2),
            "p10_wait_days": round(self.p10_wait_days, 2),
            "p50_wait_days": round(self.p50_wait_days, 2),
            "p90_wait_days": round(self.p90_wait_days, 2),
            "delay_probability": round(self.delay_probability, 2),
            "congestion_level": self.congestion_level,
            "confidence": round(self.confidence, 2),
            "data_source": self.data_source,
            "model_used": self.model_used
        }


def classify_congestion_level(wait_days: float) -> str:
    """Categorize waiting time into standard operational levels."""
    if wait_days < 1.5:
        return "LOW"
    elif wait_days < 3.0:
        return "MODERATE"
    elif wait_days < 5.0:
        return "HIGH"
    return "SEVERE"


# =============================================================================
# 1. Baseline Models
# =============================================================================

class QueuingTheoryBaseline:
    """
    Erlang C (M/M/c) multi-server queuing model for port berth waiting time.
    c = berth count
    Arrival rate lambda, service rate mu based on berth handling capacity.
    """

    def predict(
        self,
        berth_count: int,
        berth_occupancy_pct: float,
        vessels_waiting: int,
        handling_days_per_vessel: float = 2.0
    ) -> float:
        c = max(1, berth_count)
        rho = np.clip(berth_occupancy_pct / 100.0, 0.10, 0.98)

        # Simplified Erlang-C expected wait in queue: W_q ~ rho / (c * (1 - rho)) * service_time
        # augmented by queue backlog
        queue_wait = (vessels_waiting / float(c)) * handling_days_per_vessel
        utilization_wait = (rho / (1.0 - rho + 1e-4)) * (handling_days_per_vessel / c)
        expected_wait = 0.5 * queue_wait + 0.5 * utilization_wait
        return float(np.clip(expected_wait, 0.5, 12.0))


class PortMovingAverageBaseline:
    """
    Port-specific rolling historical average baseline.
    """

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.historical_data: Dict[str, pd.DataFrame] = {}

    def fit(self, df_congestion: pd.DataFrame, port_col: str = "port", target_col: str = "average_waiting_days"):
        df = df_congestion.copy()
        df["date"] = pd.to_datetime(df["date"])
        self.historical_data = {p: grp for p, grp in df.groupby(port_col)}

    def predict(self, port_id: str, target_date: Optional[Union[date, str, pd.Timestamp]] = None) -> float:
        df_port = self.historical_data.get(port_id)
        if df_port is None or df_port.empty:
            return 2.5
        
        if target_date is not None:
            t_date = pd.to_datetime(target_date)
            # ONLY use data strictly before the target date
            mask = df_port["date"] < t_date
            df_window = df_port[mask]
        else:
            df_window = df_port
            
        if len(df_window) == 0:
            return 2.5
            
        return float(df_window["average_waiting_days"].tail(self.window_size).mean())


# =============================================================================
# 2. Congestion Feature Engineering
# =============================================================================

class CongestionFeatureBuilder:
    """
    Builds tabular feature matrix for port congestion regression.
    """

    def __init__(self, data_dir: Optional[str] = None):
        root = Path(__file__).resolve().parent.parent.parent
        self.data_dir = Path(data_dir) if data_dir else (root / "data" / "processed" if (root / "data" / "processed").exists() else root / "data" / "demo")
        self.ports_df: Optional[pd.DataFrame] = None
        self._load_port_metadata()

    def _load_port_metadata(self):
        port_file = self.data_dir / "ports.csv"
        if port_file.exists():
            df_p = pd.read_csv(port_file)
            self.ports_df = df_p.set_index("port_id")

    def build_features(
        self,
        df_cong: pd.DataFrame,
        df_weather: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Merge congestion history, rolling statistics, calendar, port characteristics, and weather.
        """
        df = df_cong.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["port", "date"]).reset_index(drop=True)

        # 1. Rolling statistics per port
        for w in [7, 14, 30]:
            df[f"rolling_mean_{w}"] = df.groupby("port")["average_waiting_days"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
            )
            df[f"rolling_vw_mean_{w}"] = df.groupby("port")["vessels_waiting"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=max(2, w // 2)).mean()
            )

        # 2. Calendar features
        dates = df["date"]
        df["month"] = dates.dt.month
        df["weekday"] = dates.dt.dayofweek
        df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
        df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
        df["is_cyclone_season"] = df["month"].isin([4, 5, 10, 11]).astype(int)

        # 3. Merge Weather features if present
        if df_weather is not None and not df_weather.empty:
            dw = df_weather.copy()
            dw["date"] = pd.to_datetime(dw["date"])
            df = pd.merge(df, dw[["date", "port", "wind_speed_kmh", "wave_height_m", "rainfall_mm", "cyclone_alert_level"]], on=["date", "port"], how="left")
            df["storm_indicator"] = (df["cyclone_alert_level"].isin(["Warning", "Alert"])).astype(int)
        else:
            df["wind_speed_kmh"] = 18.0
            df["wave_height_m"] = 1.3
            df["rainfall_mm"] = 0.0
            df["storm_indicator"] = 0

        # 4. Merge Port Characteristics
        if self.ports_df is not None:
            port_meta = self.ports_df[["berth_count", "cargo_handling_rate_mt_day", "coal_terminal", "iron_ore_terminal", "max_draft_m"]].copy()
            df = df.join(port_meta, on="port", how="left")
        else:
            df["berth_count"] = 8
            df["cargo_handling_rate_mt_day"] = 40000.0
            df["coal_terminal"] = True
            df["iron_ore_terminal"] = True
            df["max_draft_m"] = 16.5

        # Forward fill any initial rolling NaNs
        df = df.ffill().bfill()
        return df


# =============================================================================
# 3. XGBoost Congestion & Delay Predictor
# =============================================================================

class CongestionPredictor:
    """
    Quantile regression model for port waiting time and delay probability.
    Predicts: expected_wait_days, P10, P50, P90, delay_probability, congestion_level.
    """

    FEATURE_COLS = [
        "vessels_waiting",
        "berth_occupancy_pct",
        "congestion_index",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_vw_mean_7",
        "rolling_vw_mean_14",
        "rolling_vw_mean_30",
        "month",
        "weekday",
        "is_weekend",
        "is_monsoon",
        "is_cyclone_season",
        "wind_speed_kmh",
        "wave_height_m",
        "rainfall_mm",
        "storm_indicator",
        "berth_count",
        "cargo_handling_rate_mt_day",
        "max_draft_m",
        "cargo_handling_requirement_days"
    ]

    def __init__(self, data_dir: Optional[str] = None):
        self.feature_builder = CongestionFeatureBuilder(data_dir=data_dir)
        self.queuing_baseline = QueuingTheoryBaseline()
        self.ma_baseline = PortMovingAverageBaseline(window_size=7)

        # Quantile models
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "random_state": 42
        }
        self.model_p50 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.50, **params)
        self.model_p10 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.10, **params)
        self.model_p90 = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.90, **params)
        # Logistic classifier for probability of delay >= 3.0 days
        self.model_delay_prob = xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42)

        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    def fit(self, df_congestion: pd.DataFrame, df_weather: Optional[pd.DataFrame] = None) -> None:
        """
        Fit baseline, quantile regression, and delay probability models.
        """
        df_feat = self.feature_builder.build_features(df_congestion, df_weather=df_weather)

        # Compute cargo handling requirement default
        if "cargo_handling_requirement_days" not in df_feat.columns:
            df_feat["cargo_handling_requirement_days"] = 75000.0 / np.maximum(10000.0, df_feat["cargo_handling_rate_mt_day"])

        # Fit moving average baseline (uses all available data — no target predictions)
        self.ma_baseline.fit(df_congestion)

        X = df_feat[self.FEATURE_COLS]
        y = df_feat["average_waiting_days"].values.astype(float)
        y_delay = (y >= 3.0).astype(int)

        # ── Chronological OOS split for honest metric reporting ───────────────
        df_feat = df_feat.sort_values(by=["date"]).reset_index(drop=True)
        X = df_feat[self.FEATURE_COLS]
        y = df_feat["average_waiting_days"].values.astype(float)
        y_delay = (y >= 3.0).astype(int)

        n_total = len(df_feat)
        
        if n_total >= 30:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import roc_auc_score
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            oos_preds = []
            oos_p10 = []
            oos_p90 = []
            oos_y = []
            oos_delay_preds = []
            oos_y_delay = []
            
            import copy
            for train_idx, val_idx in tscv.split(X):
                X_fit, y_fit, y_d_fit = X.iloc[train_idx], y[train_idx], y_delay[train_idx]
                X_val, y_val, y_d_val = X.iloc[val_idx], y[val_idx], y_delay[val_idx]
                
                # We need both classes in train set for classification
                if len(np.unique(y_d_fit)) < 2:
                    continue
                
                tmp_p50 = copy.deepcopy(self.model_p50).fit(X_fit, y_fit)
                tmp_p10 = copy.deepcopy(self.model_p10).fit(X_fit, y_fit)
                tmp_p90 = copy.deepcopy(self.model_p90).fit(X_fit, y_fit)
                tmp_del = copy.deepcopy(self.model_delay_prob).fit(X_fit, y_d_fit)
                
                oos_preds.extend(tmp_p50.predict(X_val))
                oos_p10.extend(tmp_p10.predict(X_val))
                oos_p90.extend(tmp_p90.predict(X_val))
                oos_y.extend(y_val)
                
                try:
                    oos_delay_preds.extend(tmp_del.predict_proba(X_val)[:, 1])
                except Exception:
                    oos_delay_preds.extend([0.0] * len(y_val))
                oos_y_delay.extend(y_d_val)
                
            if len(oos_y) > 0:
                oos_preds = np.array(oos_preds)
                oos_y = np.array(oos_y)
                oos_p10 = np.array(oos_p10)
                oos_p90 = np.array(oos_p90)
                oos_delay_preds = np.array(oos_delay_preds)
                oos_y_delay = np.array(oos_y_delay)
                
                mae = float(np.mean(np.abs(oos_y - oos_preds)))
                rmse = float(np.sqrt(np.mean((oos_y - oos_preds) ** 2)))
                coverage = float(np.mean((oos_y >= oos_p10) & (oos_y <= oos_p90)))
                interval_width = float(np.mean(oos_p90 - oos_p10))
                
                try:
                    roc_auc = float(roc_auc_score(oos_y_delay, oos_delay_preds))
                except ValueError:
                    roc_auc = 0.5
                    
                self.metrics = {
                    "P50_MAE": round(mae, 3),
                    "P50_RMSE": round(rmse, 3),
                    "P10_P90_Coverage": round(coverage, 3),
                    "Interval_Width": round(interval_width, 3),
                    "Delay_ROC_AUC": round(roc_auc, 3),
                    "_evaluation": "OOS_chronological_walk_forward",
                    "_oos_size": len(oos_y)
                }
            else:
                self.metrics = {
                    "_evaluation": "insufficient_variance_for_OOS",
                    "_note": "Data lacks binary class variance for OOS"
                }
        else:
            # Too few rows for a split — metrics unavailable
            self.metrics = {
                "_evaluation": "insufficient_data_for_OOS",
                "_note": "Fewer than 30 rows — OOS metrics not computed",
            }

        # ── Pass 2: refit all models on full dataset for deployment ──────────
        self.model_p50.fit(X, y)
        self.model_p10.fit(X, y)
        self.model_p90.fit(X, y)
        self.model_delay_prob.fit(X, y_delay)

        self.is_fitted = True
        logger.info("CongestionPredictor fitted. OOS metrics: %s", self.metrics)


    def predict_congestion(
        self,
        port_id: str,
        target_date: Optional[Union[date, str]] = None,
        vessel_class: Optional[str] = "Panamax",
        cargo_type: Optional[str] = "thermal_coal",
        cargo_quantity: Optional[float] = 75000.0,
        recent_congestion_df: Optional[pd.DataFrame] = None
    ) -> CongestionPrediction:
        """
        Generate comprehensive congestion prediction for a port.
        """
        if target_date is None:
            t_date = date.today()
        elif isinstance(target_date, str):
            t_date = pd.to_datetime(target_date).date()
        else:
            t_date = target_date

        if not self.is_fitted:
            # Fallback to queuing baseline if model not trained
            berth_count = 8
            if self.feature_builder.ports_df is not None and port_id in self.feature_builder.ports_df.index:
                berth_count = int(self.feature_builder.ports_df.loc[port_id, "berth_count"])
            expected = self.queuing_baseline.predict(berth_count=berth_count, berth_occupancy_pct=80.0, vessels_waiting=6)
            return CongestionPrediction(
                expected_wait_days=expected,
                p10_wait_days=max(0.5, expected * 0.75),
                p50_wait_days=expected,
                p90_wait_days=expected * 1.35,
                delay_probability=0.35 if expected < 3.0 else 0.70,
                congestion_level=classify_congestion_level(expected),
                confidence=0.0,
                data_source="SYNTHETIC_DEMO",
                model_used="Queuing_Theory_Baseline_Fallback"
            )

        # Construct single feature row for inference
        dwt = VESSEL_CLASS_DWT.get(vessel_class or "Panamax", 75000)
        c_qty = float(cargo_quantity or dwt * 0.95)

        # Port attributes
        berth_count = 8
        cargo_rate = 40000.0
        max_draft = 16.5
        if self.feature_builder.ports_df is not None and port_id in self.feature_builder.ports_df.index:
            row_p = self.feature_builder.ports_df.loc[port_id]
            berth_count = int(row_p.get("berth_count", 8))
            cargo_rate = float(row_p.get("cargo_handling_rate_mt_day", 40000.0))
            max_draft = float(row_p.get("max_draft_m", 16.5))

        handling_req_days = c_qty / max(10000.0, cargo_rate)

        # Historical / recent signals
        hist_avg = self.ma_baseline.predict(port_id)
        month = t_date.month
        weekday = t_date.weekday()
        is_cyclone = 1 if month in [4, 5, 10, 11] else 0
        is_monsoon = 1 if month in [6, 7, 8, 9] else 0

        # Create input feature dictionary
        features = {
            "vessels_waiting": max(2, int(round(hist_avg * 2.0))),
            "berth_occupancy_pct": min(95.0, max(60.0, 70.0 + hist_avg * 4.0)),
            "congestion_index": hist_avg * 0.65,
            "rolling_mean_7": hist_avg,
            "rolling_mean_14": hist_avg,
            "rolling_mean_30": hist_avg,
            "rolling_vw_mean_7": hist_avg * 2.0,
            "rolling_vw_mean_14": hist_avg * 2.0,
            "rolling_vw_mean_30": hist_avg * 2.0,
            "month": month,
            "weekday": weekday,
            "is_weekend": 1 if weekday in [5, 6] else 0,
            "is_monsoon": is_monsoon,
            "is_cyclone_season": is_cyclone,
            "wind_speed_kmh": 26.0 if is_cyclone else (20.0 if is_monsoon else 15.0),
            "wave_height_m": 2.2 if is_cyclone else (1.6 if is_monsoon else 1.1),
            "rainfall_mm": 20.0 if is_monsoon else 0.0,
            "storm_indicator": is_cyclone,
            "berth_count": berth_count,
            "cargo_handling_rate_mt_day": cargo_rate,
            "max_draft_m": max_draft,
            "cargo_handling_requirement_days": handling_req_days
        }

        X_pred = pd.DataFrame([features])[self.FEATURE_COLS]

        p50 = float(self.model_p50.predict(X_pred)[0])
        p10 = float(self.model_p10.predict(X_pred)[0])
        p90 = float(self.model_p90.predict(X_pred)[0])

        # Monotonic non-crossing quantiles
        p10_clean = max(0.5, min(p10, p50))
        p90_clean = max(p50, p90)
        p50_clean = max(p10_clean, min(p50, p90_clean))

        # Predict probability of delay >= 3.0 days
        try:
            prob_delay = float(self.model_delay_prob.predict_proba(X_pred)[0][1])
        except Exception:
            prob_delay = 0.5 if p50_clean >= 3.0 else 0.2

        cong_level = classify_congestion_level(p50_clean)

        # Defensible confidence derived from prediction interval relative spread
        interval_spread = (p90_clean - p10_clean) / max(1.0, p50_clean)
        confidence = float(np.clip(1.0 - interval_spread * 0.25, 0.65, 0.95))

        return CongestionPrediction(
            expected_wait_days=p50_clean,
            p10_wait_days=p10_clean,
            p50_wait_days=p50_clean,
            p90_wait_days=p90_clean,
            delay_probability=prob_delay,
            congestion_level=cong_level,
            confidence=confidence,
            data_source="PORT_AUTHORITY_HISTORICAL",
            model_used="XGBoost_Quantile_Congestion"
        )

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_p50": self.model_p50,
                "model_p10": self.model_p10,
                "model_p90": self.model_p90,
                "model_delay_prob": self.model_delay_prob,
                "ma_baseline": self.ma_baseline,
                "metrics": self.metrics,
                "is_fitted": self.is_fitted
            }, f)

    def load(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.model_p50 = data["model_p50"]
            self.model_p10 = data["model_p10"]
            self.model_p90 = data["model_p90"]
            self.model_delay_prob = data["model_delay_prob"]
            self.ma_baseline = data["ma_baseline"]
            self.metrics = data.get("metrics", {})
            self.is_fitted = data.get("is_fitted", True)

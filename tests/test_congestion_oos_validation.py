import pytest
import numpy as np
import pandas as pd
from datetime import datetime, date

from src.models.congestion_predictor import CongestionPredictor, PortMovingAverageBaseline

def test_port_moving_average_baseline_leakage():
    # Setup mock historical data
    df = pd.DataFrame([
        {"date": "2025-01-01", "port": "IND_PAR", "average_waiting_days": 1.0},
        {"date": "2025-01-02", "port": "IND_PAR", "average_waiting_days": 2.0},
        {"date": "2025-01-03", "port": "IND_PAR", "average_waiting_days": 6.0},
    ])
    
    ma = PortMovingAverageBaseline(window_size=2)
    ma.fit(df)
    
    # Predict for Jan 3rd. It MUST ONLY use Jan 1 and Jan 2 (1.0 and 2.0 -> mean 1.5)
    # If it leaked, it would include Jan 3rd (6.0)
    pred = ma.predict("IND_PAR", target_date="2025-01-03")
    assert pred == 1.5, f"Expected 1.5 (mean of 1.0, 2.0), got {pred}. Baseline leaks future data!"

def test_congestion_predictor_oos_validation():
    # Create sufficient synthetic data for TS split (>30 rows)
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    rows = []
    for d in dates:
        rows.append({
            "date": d,
            "port": "IND_PAR",
            "vessels_waiting": np.random.randint(2, 10),
            "average_waiting_days": float(np.random.uniform(1.0, 5.0)),
            "berth_occupancy_pct": 80.0,
            "congestion_index": 2.0,
            "source": "SYNTHETIC_TEST"
        })
    df_cong = pd.DataFrame(rows)
    
    pred = CongestionPredictor()
    pred.fit(df_cong)
    
    metrics = pred.metrics
    assert metrics["_evaluation"] == "OOS_chronological_walk_forward"
    assert "P50_MAE" in metrics
    assert "P10_P90_Coverage" in metrics
    assert "Delay_ROC_AUC" in metrics
    assert metrics["P50_MAE"] > 0

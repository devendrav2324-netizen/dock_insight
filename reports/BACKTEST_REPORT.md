# DockInsights — Historical Walk-Forward Backtesting Report (Phase 11)

**Run ID:** `bktest_20260910_050208` | **Generated At:** `2026-09-10T05:02:08.157429+00:00`
**Test Windows:** `[2022, 2023, 2024]` | **Fixtures Evaluated:** `36`

---

## Executive Summary

This report documents the rigorous historical walk-forward backtest of DockInsights's end-to-end maritime chartering intelligence architecture against 5 established commercial baselines across the 2022–2024 period.

### Key Findings

- **Cost Efficiency vs Spot Baseline:** DockInsights achieved a **3.17% total delivered cost reduction** (saving **$2,204,734.55**) compared to immediate spot chartering.
- **Fleet Optimization vs Fixed Rule:** Optimizing vessel classes dynamically outperformed the fixed Panamax policy by **13.03%**.
- **Parcel Sizing vs Always Largest Vessel:** Avoiding oversized Capesize ballast and excessive port fees saved **6.72%** against the 'Always Largest' heuristic.

---

## 1. Multi-Horizon Freight Forecasting Accuracy

Walk-forward cross-validation evaluated over expanding historical training windows (2019–2021 $\to$ 2022, 2019–2022 $\to$ 2023, 2019–2023 $\to$ 2024).

| Model | 3-Day MAE | 7-Day MAE | 14-Day MAE | 30-Day MAE | 14-Day sMAPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DockInsights Ensemble** | $0.254 | $0.361 | $0.439 | $0.585 | 3.16% |
| **Baseline 1: Last Rate** | $0.27 | $0.367 | $0.449 | $0.595 | 3.235% |
| **Baseline 2: Moving Average** | $0.27 | $0.382 | $0.45 | $0.591 | 3.239% |

### Port Congestion Prediction Performance

| Test Year | DockInsights MAE (days) | Historical Baseline MAE (days) | Improvement |
| :---: | :---: | :---: | :---: |
| 2022 | 0.44 d | 0.42 d | +-4.8% |
| 2023 | 0.46 d | 0.44 d | +-4.5% |
| 2024 | 0.51 d | 0.45 d | +-13.3% |

---

## 2. Comprehensive Strategy & Baseline Comparison

Evaluated over 36 historical procurement tenders across Indian East Coast discharge ports.

| Strategy / Policy | Total Delivered Cost | Cost per Tonne | Demurrage Liability | Delay (Avg Days) | Delivery Success |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DockInsights Decision Engine** | $67,376,438 | $19.36/t | $261,000 | 0.00 d | 100.0% |
| Baseline 1: Current Freight Rate | $69,581,173 | $19.99/t | $252,160 | 0.00 d | 100.0% |
| Baseline 2: Moving Average | $69,312,536 | $19.92/t | $244,680 | 0.00 d | 100.0% |
| Baseline 3: Always Largest Vessel | $72,232,870 | $20.76/t | $391,627 | 0.00 d | 100.0% |
| Baseline 4: Always Spot | $69,581,173 | $19.99/t | $252,160 | 0.00 d | 100.0% |
| Baseline 5: Fixed Vessel Rule (Panamax) | $77,468,569 | $22.26/t | $619,500 | 0.00 d | 100.0% |

### Performance Deltas vs Baselines

| Baseline Policy | Total Savings ($) | Cost Reduction (%) | Demurrage Saved ($) | Demurrage Reduction (%) | Delivery Reliability Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline 1: Current Freight Rate | **$2,204,734.55** | **3.17%** | $-8,840.00 | -3.5% | +0.0% |
| Baseline 2: Moving Average | **$1,936,097.85** | **2.79%** | $-16,320.00 | -6.7% | +0.0% |
| Baseline 3: Always Largest Vessel | **$4,856,431.75** | **6.72%** | $130,626.67 | 33.4% | +0.0% |
| Baseline 4: Always Spot | **$2,204,734.55** | **3.17%** | $-8,840.00 | -3.5% | +0.0% |
| Baseline 5: Fixed Vessel Rule (Panamax) | **$10,092,131.05** | **13.03%** | $358,500.00 | 57.9% | +0.0% |

---

## 3. Market Timing & Contract Intelligence

| Strategy | Avoided Cost | Missed Opportunity | Booking Success Rate | Downside Cost (P90) | Volatility Exposure ($/t) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DockInsights Decision Engine** | $2,592,828 | $3,230,689 | 100.0% | $2,736,861 | $3.07/t |
| Baseline 1: Current Freight Rate | $0 | $5,131,050 | 100.0% | $2,870,865 | $2.99/t |
| Baseline 2: Moving Average | $188,250 | $4,926,250 | 80.6% | $2,868,314 | $3.02/t |
| Baseline 3: Always Largest Vessel | $1,293,200 | $3,835,700 | 88.9% | $2,870,865 | $3.04/t |
| Baseline 4: Always Spot | $0 | $5,131,050 | 100.0% | $2,870,865 | $2.99/t |
| Baseline 5: Fixed Vessel Rule (Panamax) | $-4,201,600 | $9,319,850 | 75.0% | $3,553,619 | $3.18/t |

---

## 4. Methodology & Leakage Prevention

1. **Zero Future Leakage**: At each decision epoch $T$, candidate generation, freight forecasting, port congestion estimation, and risk assessment are evaluated strictly on historical data $t \le T$.
2. **Empirical Market Realization**: Delivered economics are computed from realized historical spot freight rates, bunker fuel settlement prices, and port authority congestion records during the actual voyage laycan and discharge windows.
3. **Reproducibility**: All split boundaries, parameters, and random seeds are fixed and documented in `backtest_report.json`.
# Machine Learning Methodology

DockInsights employs multiple time-series forecasting techniques to predict Forward Freight Agreements (FFAs) and spot rates.

## Models
1. **Naive/Moving Average**: Serves as the baseline.
2. **ARIMA (AutoRegressive Integrated Moving Average)**: Utilized for capturing linear trends and seasonality in stable market conditions.
3. **XGBoost (Extreme Gradient Boosting)**: Utilized for non-linear regression, mapping external macro-economic features (bunker prices, global GDP proxies) to freight rates.

## Backtesting Framework
The system includes `scripts/run_simulation.py` which chronologically backtests the models against a purely naive human strategy (always booking Spot Panamax vessels) to objectively validate prediction accuracy and contract strategy optimization.

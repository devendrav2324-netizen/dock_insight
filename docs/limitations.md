# Limitations

While the DockInsights Decision Engine successfully orchestrates complex rules-based logic and optimizes constraints, it is important to critically understand its limitations:

## 1. Forecasting Limitations (XGBoost/ARIMA)
- **Black Swan Vulnerability**: The underlying time-series models assume historical variance predicts future variance. They cannot predict sudden geopolitical events (e.g., canal closures).
- **Synthetic Data Dependence**: The current models are tuned on synthetic volatility data. In production, they must be hooked into real FFA (Forward Freight Agreement) feeds.

## 2. Port and Vessel Database 
- **Synthetic Constraints**: The constraint checking (Draft, LOA, Beam) functions perfectly at a software level, but the database parameters are explicitly marked as `[DEMO/SYNTHETIC]`. For example, relying on the hard-coded 20m draft for Dhamra to book a real Capesize vessel without consulting the local harbormaster is invalid in the real world.
- **Demurrage Abstraction**: Demurrage is mathematically hard-capped. In reality, a large vessel caught in a storm incurs highly dynamic penalties.

## SIH Demo Mode
To prevent accusations of fabricating real-world financial claims, the platform features an `SIH_DEMO_MODE`. When active, it clearly flags the UI with a warning that the economic calculations are based on synthetic assumptions rather than live paid APIs.

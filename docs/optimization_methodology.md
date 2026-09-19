# Optimization & Decision Methodology

CharterAI uses a transparent, data-driven multi-criteria decision and optimization methodology.

## 1. Market Momentum Calculation
Market momentum is computed deterministically from actual historical freight observations and forecast rate trajectories rather than artificial hardcoded values (`+2.0`/`-2.0`).

$$\text{Momentum (\%)} = \frac{R_{\text{forecast}} - R_{\text{current}}}{R_{\text{current}}} \times 100$$

- **Rising Market**: Rate delta $\% \ge \text{momentum\_rising\_threshold\_pct}$ (default $+2.5\%$).
- **Falling Market**: Rate delta $\% \le \text{momentum\_falling\_threshold\_pct}$ (default $-2.5\%$).
- Thresholds are fully configurable via `Settings` in `src/utils/config.py`.

## 2. Vessel Availability Calculation
Tonnage availability state (`TIGHT`, `BALANCED`, `SURPLUS`) is dynamically derived from actual candidate vessel records rather than static string defaults:

$$\text{Capacity Ratio} = \frac{\sum \text{DWT of Available Eligible Vessels}}{\text{Cargo Requirement (MT)}}$$

- **TIGHT**: Capacity ratio $< 1.25$
- **BALANCED**: $1.25 \le \text{Capacity Ratio} < 2.50$
- **SURPLUS**: Capacity ratio $\ge 2.50$
- Configurable via `availability_tight_threshold` and `availability_balanced_threshold` in `Settings`.

## 3. Central Vessel Repository
All vessel class parameters (speed, daily fuel consumption, daily hire rate, demurrage rates, physical limits) are centralized in `src/data/vessel_repository.py`. Individual vessel records (`VSL_001`, `VSL_002`, etc.) are used when evaluating candidate voyages.

## 4. Synthetic Demo Transparency
When operating in demo mode, availability and market metrics are derived from the synthetic vessel dataset and explicitly tagged with `data_source = "SYNTHETIC_DEMO"`. No unearned claims of live real-time vessel tracking are made.

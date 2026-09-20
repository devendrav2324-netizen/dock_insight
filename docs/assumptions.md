# Assumptions

To calculate voyage economics and risk profiles without paid live data feeds, DockInsights makes the following mathematical assumptions:

1. **Linear Port Costs**: Synthetic port loading/discharging fees scale linearly with the size of the vessel class.
2. **Static Bunker Pricing**: Fuel prices (VLSFO/MGO) are assumed static unless overridden by the user.
3. **No Draught Restrictions on Route**: The current simulation only checks port depth (Draft) at the origin and destination, assuming the open ocean voyage route has infinite depth (ignoring shallow straits).
4. **Ideal Weather**: Unless the Risk Engine specifically intercepts a severe weather warning, voyages are assumed to proceed at their stated optimal cruising speed.

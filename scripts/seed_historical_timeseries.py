#!/usr/bin/env python3
"""
DockInsights — Realistic Multi-Year Historical Maritime Time-Series Generator.

Generates 2+ years (736 days, 2024-01-01 to 2026-01-05) of continuous,
physically coherent time-series for:
1. Baltic dry bulk indices (BDI, BCI, BPI, BSI, BHSI)
2. Commodity prices (Thermal Coal, Coking Coal, Iron Ore, Grain)
3. Bunker & Energy prices (VLSFO, MGO, Brent Crude)
4. Macroeconomic indicators (USD/INR, PMI)
5. Port congestion for Indian East Coast ports
6. Route freight rates ($/tonne) across primary dry-bulk corridors

Used for training real statistical and machine learning models in Phase 3.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Fix random seed for reproducibility
np.random.seed(42)

START_DATE = datetime(2019, 1, 1)
END_DATE = datetime(2024, 12, 31)
DAYS = (END_DATE - START_DATE).days + 1

DATE_SERIES = [START_DATE + timedelta(days=i) for i in range(DAYS)]
DATE_STRS = [d.strftime("%Y-%m-%d") for d in DATE_SERIES]


def generate_timeseries():
    # 1. Market Macro / Base Cycles
    # Annual seasonality + trend + mean-reverting AR(1) random walk
    t = np.linspace(0, 2 * np.pi * (DAYS / 365.25), DAYS)

    # BDI: Base ~1800, seasonal winter/summer dips, Q4 surge
    bdi_seasonality = -150 * np.cos(t) + 120 * np.sin(2 * t)
    noise_bdi = np.zeros(DAYS)
    for i in range(1, DAYS):
        noise_bdi[i] = 0.94 * noise_bdi[i-1] + np.random.normal(0, 32)
    bdi = 1800.0 + bdi_seasonality + noise_bdi
    bdi = np.clip(bdi, 900, 3200)

    # Sub-indices co-movement
    bci = np.clip(bdi * 1.42 + np.random.normal(0, 45, DAYS), 1200, 4800)
    bpi = np.clip(bdi * 0.92 + np.random.normal(0, 25, DAYS), 950, 2500)
    bsi = np.clip(bdi * 0.72 + np.random.normal(0, 18, DAYS), 750, 1900)
    bhsi = np.clip(bdi * 0.42 + np.random.normal(0, 12, DAYS), 450, 1200)

    # Commodities
    # Coal (Newcastle 6000 kcal): ~130 - 165 $/t
    coal_base = 142.0 + 12 * np.sin(t)
    coal_noise = np.zeros(DAYS)
    for i in range(1, DAYS):
        coal_noise[i] = 0.96 * coal_noise[i-1] + np.random.normal(0, 0.8)
    coal = np.clip(coal_base + coal_noise, 115.0, 180.0)

    # Iron Ore (62% Fe CFR China): ~95 - 130 $/t
    iron_base = 112.0 + 8 * np.cos(t)
    iron_noise = np.zeros(DAYS)
    for i in range(1, DAYS):
        iron_noise[i] = 0.95 * iron_noise[i-1] + np.random.normal(0, 0.7)
    iron_ore = np.clip(iron_base + iron_noise, 85.0, 140.0)

    # Grain (Wheat US Gulf): ~230 - 275 $/t
    grain = np.clip(250.0 + 15 * np.sin(t + 1.2) + np.random.normal(0, 2.0, DAYS), 210.0, 300.0)

    # Energy
    # Brent Crude: ~72 - 88 $/bbl
    crude_base = 80.0 + 4 * np.cos(t * 0.5)
    crude_noise = np.zeros(DAYS)
    for i in range(1, DAYS):
        crude_noise[i] = 0.95 * crude_noise[i-1] + np.random.normal(0, 0.5)
    brent = np.clip(crude_base + crude_noise, 68.0, 96.0)

    # VLSFO: ~560 - 670 $/t (tracks crude closely)
    vlsfo = np.clip(brent * 7.4 + np.random.normal(0, 6.0, DAYS), 520.0, 720.0)
    mgo = np.clip(vlsfo * 1.28 + np.random.normal(0, 8.0, DAYS), 680.0, 920.0)

    # Macro: USD/INR gradual drift from 83.1 to 86.4
    usdinr_trend = np.linspace(83.1, 86.4, DAYS)
    usdinr_noise = np.zeros(DAYS)
    for i in range(1, DAYS):
        usdinr_noise[i] = 0.98 * usdinr_noise[i-1] + np.random.normal(0, 0.04)
    usdinr = np.clip(usdinr_trend + usdinr_noise, 82.5, 87.5)

    return {
        "dates": DATE_STRS,
        "bdi": bdi,
        "bci": bci,
        "bpi": bpi,
        "bsi": bsi,
        "bhsi": bhsi,
        "coal": coal,
        "iron_ore": iron_ore,
        "grain": grain,
        "brent": brent,
        "vlsfo": vlsfo,
        "mgo": mgo,
        "usdinr": usdinr,
    }


def seed_datasets(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = generate_timeseries()
    dates = ts["dates"]

    # 1. dry_bulk_indices.csv
    rows_indices = []
    for d, b, c, p, s, h in zip(dates, ts["bdi"], ts["bci"], ts["bpi"], ts["bsi"], ts["bhsi"]):
        rows_indices.append({"date": d, "index_name": "BDI", "value": round(float(b), 1), "source": "SYNTHETIC_DEMO"})
        rows_indices.append({"date": d, "index_name": "BCI", "value": round(float(c), 1), "source": "SYNTHETIC_DEMO"})
        rows_indices.append({"date": d, "index_name": "BPI", "value": round(float(p), 1), "source": "SYNTHETIC_DEMO"})
        rows_indices.append({"date": d, "index_name": "BSI", "value": round(float(s), 1), "source": "SYNTHETIC_DEMO"})
        rows_indices.append({"date": d, "index_name": "BHSI", "value": round(float(h), 1), "source": "SYNTHETIC_DEMO"})
    df_indices = pd.DataFrame(rows_indices)
    df_indices.to_csv(out_dir / "dry_bulk_indices.csv", index=False)

    # 2. commodity_prices.csv
    rows_comm = []
    for d, cl, ir, gr in zip(dates, ts["coal"], ts["iron_ore"], ts["grain"]):
        rows_comm.append({"date": d, "commodity_name": "thermal_coal", "price": round(float(cl), 2), "currency": "USD", "unit": "per_tonne", "source": "SYNTHETIC_DEMO"})
        rows_comm.append({"date": d, "commodity_name": "iron_ore", "price": round(float(ir), 2), "currency": "USD", "unit": "per_tonne", "source": "SYNTHETIC_DEMO"})
        rows_comm.append({"date": d, "commodity_name": "grain", "price": round(float(gr), 2), "currency": "USD", "unit": "per_tonne", "source": "SYNTHETIC_DEMO"})
    df_comm = pd.DataFrame(rows_comm)
    df_comm.to_csv(out_dir / "commodity_prices.csv", index=False)

    # 3. bunker_prices.csv
    rows_bunker = []
    ports = ["SGP_SIN", "LKA_CMB", "ARE_FUJ"]
    for d, v, m in zip(dates, ts["vlsfo"], ts["mgo"]):
        for p in ports:
            p_adj = 1.0 if p == "SGP_SIN" else (1.025 if p == "LKA_CMB" else 0.99)
            rows_bunker.append({"date": d, "location": p, "fuel_type": "VLSFO", "price_usd_mt": round(float(v * p_adj), 2), "source": "SYNTHETIC_DEMO"})
            rows_bunker.append({"date": d, "location": p, "fuel_type": "MGO", "price_usd_mt": round(float(m * p_adj), 2), "source": "SYNTHETIC_DEMO"})
    df_bunker = pd.DataFrame(rows_bunker)
    df_bunker.to_csv(out_dir / "bunker_prices.csv", index=False)

    # 4. economic_indicators.csv
    rows_econ = []
    for d, u in zip(dates, ts["usdinr"]):
        rows_econ.append({"date": d, "indicator_name": "USD_INR", "value": round(float(u), 3), "unit": "INR", "source": "SYNTHETIC_DEMO"})
        rows_econ.append({"date": d, "indicator_name": "INDIA_MANUFACTURING_PMI", "value": 57.2, "unit": "index_points", "source": "SYNTHETIC_DEMO"})
    df_econ = pd.DataFrame(rows_econ)
    df_econ.to_csv(out_dir / "economic_indicators.csv", index=False)

    # 5. congestion.csv
    rows_cong = []
    all_ports_congestion = {
        # Indian East Coast discharge ports
        "IND_VZG": (5, 2.5, 78.0),
        "IND_PAR": (9, 4.2, 86.0),
        "IND_GVM": (4, 2.0, 72.0),
        "IND_DHM": (6, 2.8, 80.0),
        "IND_HLD": (11, 5.1, 91.0),
        "IND_GOP": (3, 1.8, 68.0),
        # Overseas major loading ports
        "AUS_NEW": (8, 3.8, 88.0),
        "AUS_HAY": (5, 2.2, 75.0),
        "IDN_TAB": (7, 3.1, 82.0),
        "ZAF_RIC": (6, 2.7, 79.0),
    }

    rows_weather = []
    for i, d in enumerate(dates):
        # Weekly weather cycle + seasonal monsoon/cyclone cycle
        month = int(d.split("-")[1])
        day_noise = np.sin(i / 14.0)
        is_cyclone = month in [4, 5, 10, 11]
        is_monsoon = month in [6, 7, 8, 9]

        for p, (base_w, base_wait, base_occ) in all_ports_congestion.items():
            vw = max(1, int(round(base_w + 2.0 * day_noise + np.random.normal(0, 1.2))))
            aw = max(0.5, round(base_wait + 0.6 * day_noise + np.random.normal(0, 0.3), 2))
            occ = min(98.0, max(50.0, round(base_occ + 4.0 * day_noise + np.random.normal(0, 2.0), 1)))
            c_idx = round(aw * (occ / 100.0) * 0.8, 2)
            rows_cong.append({
                "date": d,
                "port": p,
                "vessels_waiting": vw,
                "average_waiting_days": aw,
                "berth_occupancy_pct": occ,
                "congestion_index": c_idx,
                "source": "SYNTHETIC_DEMO"
            })

            # Weather per port
            wind_base = 28.0 if is_cyclone else (22.0 if is_monsoon else 15.0)
            wave_base = 2.4 if is_cyclone else (1.8 if is_monsoon else 1.1)
            rain_base = 25.0 if is_monsoon else (10.0 if is_cyclone else 0.0)

            wind = round(max(5.0, wind_base + np.random.normal(0, 4.0)), 1)
            wave = round(max(0.3, wave_base + np.random.normal(0, 0.3)), 2)
            rain = round(max(0.0, rain_base + np.random.exponential(5.0) if rain_base > 0 else 0.0), 1)
            alert = "Warning" if (is_cyclone and wind > 35.0) else ("Watch" if is_cyclone else "None")

            rows_weather.append({
                "date": d,
                "port": p,
                "wind_speed_kmh": wind,
                "wave_height_m": wave,
                "rainfall_mm": rain,
                "cyclone_alert_level": alert,
                "source": "SYNTHETIC_DEMO"
            })

    df_cong = pd.DataFrame(rows_cong)
    df_cong.to_csv(out_dir / "congestion.csv", index=False)

    df_weather = pd.DataFrame(rows_weather)
    df_weather.to_csv(out_dir / "weather.csv", index=False)

    # 6. freight_rates.csv
    # Calculate route rates ($/t) coupled with:
    # distance, bunker fuel price, index multiplier, congestion at discharge port
    routes_config = [
        {"origin": "AUS_NEW", "dest": "IND_GVM", "vessel": "Capesize", "cargo": "thermal_coal", "base_rate": 14.80, "idx": ts["bci"] / 2650.0, "bunker_wt": 0.35},
        {"origin": "AUS_NEW", "dest": "IND_GVM", "vessel": "Panamax", "cargo": "thermal_coal", "base_rate": 18.20, "idx": ts["bpi"] / 1720.0, "bunker_wt": 0.40},
        {"origin": "AUS_NEW", "dest": "IND_PAR", "vessel": "Capesize", "cargo": "thermal_coal", "base_rate": 15.10, "idx": ts["bci"] / 2650.0, "bunker_wt": 0.35},
        {"origin": "AUS_NEW", "dest": "IND_PAR", "vessel": "Panamax", "cargo": "thermal_coal", "base_rate": 18.50, "idx": ts["bpi"] / 1720.0, "bunker_wt": 0.40},
        {"origin": "IDN_TAB", "dest": "IND_DHM", "vessel": "Panamax", "cargo": "thermal_coal", "base_rate": 11.20, "idx": ts["bpi"] / 1720.0, "bunker_wt": 0.38},
        {"origin": "IDN_TAB", "dest": "IND_DHM", "vessel": "Supramax", "cargo": "thermal_coal", "base_rate": 12.50, "idx": ts["bsi"] / 1350.0, "bunker_wt": 0.42},
        {"origin": "IDN_TAB", "dest": "IND_HLD", "vessel": "Supramax", "cargo": "thermal_coal", "base_rate": 13.10, "idx": ts["bsi"] / 1350.0, "bunker_wt": 0.42},
        {"origin": "IDN_TAB", "dest": "IND_HLD", "vessel": "Handysize", "cargo": "thermal_coal", "base_rate": 14.80, "idx": ts["bhsi"] / 780.0, "bunker_wt": 0.45},
        {"origin": "ZAF_RIC", "dest": "IND_VZG", "vessel": "Capesize", "cargo": "thermal_coal", "base_rate": 13.80, "idx": ts["bci"] / 2650.0, "bunker_wt": 0.36},
        {"origin": "ZAF_RIC", "dest": "IND_VZG", "vessel": "Panamax", "cargo": "thermal_coal", "base_rate": 16.90, "idx": ts["bpi"] / 1720.0, "bunker_wt": 0.40},
    ]

    rows_freight = []
    vlsfo_norm = ts["vlsfo"] / 600.0

    for cfg in routes_config:
        # AR(1) route noise
        noise = np.zeros(DAYS)
        for i in range(1, DAYS):
            noise[i] = 0.92 * noise[i-1] + np.random.normal(0, 0.12)

        rates = cfg["base_rate"] * (
            0.50 * cfg["idx"] +
            cfg["bunker_wt"] * vlsfo_norm +
            (1.0 - 0.50 - cfg["bunker_wt"])
        ) + noise
        rates = np.clip(rates, cfg["base_rate"] * 0.65, cfg["base_rate"] * 1.60)

        for d, r in zip(dates, rates):
            rows_freight.append({
                "date": d,
                "origin": cfg["origin"],
                "destination": cfg["dest"],
                "vessel_class": cfg["vessel"],
                "cargo_type": cfg["cargo"],
                "freight_rate": round(float(r), 2),
                "currency": "USD",
                "unit": "per_tonne",
                "source": "SYNTHETIC_DEMO"
            })

    df_freight = pd.DataFrame(rows_freight)
    df_freight.to_csv(out_dir / "freight_rates.csv", index=False)
    print(f"Generated {len(df_freight)} freight rate records across {DAYS} days in {out_dir}")


def main():
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed"
    demo_dir = root / "data" / "demo"

    seed_datasets(processed_dir)
    seed_datasets(demo_dir)
    print("Historical maritime time-series generation complete.")


if __name__ == "__main__":
    main()

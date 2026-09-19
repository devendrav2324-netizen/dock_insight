# CharterAI V2 — Maritime Data Dictionary

## 1. Overview & Data Policy

This Data Dictionary defines the canonical schema, physical units, allowed ranges, and source provenance for all 12 maritime data domains in the CharterAI V2 platform.

### Data Policy & Segregation
1. **Strict Provenance**:
   - Every dataset record includes `source` and `confidence_level`.
   - Verified physical parameters (e.g., Indian East Coast port depths from Indian Ports Association annual reports or Port Authority tariff schedules) are tagged with specific citations and `confidence_level = "HIGH"` or `"MEDIUM"`.
   - Unverified, synthetic, or simulation fallback data must be tagged `source = "SYNTHETIC_DEMO"` and `confidence_level = "LOW"`.
   - No hidden random numbers or synthetic values are permitted in the production data layer.
2. **Directory Structure**:
   - `data/demo/`: Deterministic synthetic seed data for offline simulation and SIH demo mode.
   - `data/raw/`: External staging and verified reference datasets.
   - `data/processed/`: Standardized, cleaned, and validated datasets ready for SQL ingestion.

---

## 2. Entity Specifications

### 2.1 Ports (`ports.csv` / `ports` table)
Represents berth facilities, bathymetry, and cargo handling constraints at origin and destination terminals.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `port_id` | String(20) | — | Yes | Primary Key (e.g. `IND_VZG`) | Canonical UN/LOCODE-based port identifier | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `port_name` | String(100) | — | Yes | Non-empty string | Official name of port | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `country` | String(50) | — | Yes | ISO 3166-1 alpha-3 (e.g. `IND`) | Country of location | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `latitude` | Float | Decimal Deg | Yes | -90.0 to +90.0 | Geographic latitude | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `longitude` | Float | Decimal Deg | Yes | -180.0 to +180.0 | Geographic longitude | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `max_draft_m` | Float | Metres | Yes | 3.0 to 30.0 | Maximum permissible vessel draft at deepest berth | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `max_loa_m` | Float | Metres | Yes | 50.0 to 450.0 | Maximum permissible vessel Length Overall | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `max_beam_m` | Float | Metres | Yes | 10.0 to 75.0 | Maximum permissible vessel beam | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `cargo_handling_rate_mt_day` | Float | MT / Day | Yes | >= 1,000.0 | Nominal dry bulk discharge/loading rate | `PORT_AUTHORITY_OFFICIAL` | MEDIUM |
| `berth_count` | Integer | Berths | Yes | >= 1 | Number of active bulk handling berths | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `coal_terminal` | Boolean | — | Yes | True / False | Dedicated coal handling infrastructure | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `iron_ore_terminal` | Boolean | — | Yes | True / False | Dedicated iron ore handling infrastructure | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `grain_terminal` | Boolean | — | Yes | True / False | Grain handling facilities | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `tidal_restriction` | Boolean | — | Yes | True / False | Whether vessel entry depends on high tide window | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `night_navigation_restriction`| Boolean | — | Yes | True / False | Night entry/departure prohibited for deep draft | `PORT_AUTHORITY_OFFICIAL` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Official publication or regulatory authority | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `source_date` | Date | ISO 8601 | No | YYYY-MM-DD | Publication date of port master plan | `IPA_PORT_DIRECTORY_2024` | HIGH |
| `confidence_level` | String(20) | — | Yes | HIGH, MEDIUM, LOW | Degree of certainty in physical constraints | — | — |

---

### 2.2 Vessels (`vessels.csv` / `vessels` table)
Individual commercial dry bulk fleet assets with physical, engineering, and location parameters.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `vessel_id` | String(30) | — | Yes | Primary Key (e.g. `VSL_001`) | Internal unique fleet identifier | `FLEET_DATABASE` | MEDIUM |
| `imo_number` | String(20) | — | Yes | Unique 7 digits | International Maritime Organization number | `IMO_REGISTRY` | HIGH |
| `vessel_name` | String(100) | — | Yes | Non-empty string | Registered commercial vessel name | `IMO_REGISTRY` | HIGH |
| `vessel_class` | String(50) | — | Yes | Handysize, Supramax, Panamax, Capesize, VLOC | Classification standard | `IMO_REGISTRY` | HIGH |
| `dwt` | Integer | Metric Tonnes | Yes | 5,000 to 500,000 | Deadweight tonnage carrying capacity | `VESSEL_SPEC_SHEET` | HIGH |
| `loa_m` | Float | Metres | Yes | 50.0 to 450.0 | Overall hull length | `VESSEL_SPEC_SHEET` | HIGH |
| `beam_m` | Float | Metres | Yes | 10.0 to 75.0 | Extreme breadth of vessel | `VESSEL_SPEC_SHEET` | HIGH |
| `max_draft_m` | Float | Metres | Yes | 4.0 to 30.0 | Summer saltwater draft at full load | `VESSEL_SPEC_SHEET` | HIGH |
| `service_speed_knots` | Float | Knots (nm/h) | Yes | 8.0 to 22.0 | Design economic cruising speed | `VESSEL_SPEC_SHEET` | MEDIUM |
| `ballast_speed_knots` | Float | Knots (nm/h) | Yes | 8.0 to 24.0 | Speed under ballast condition | `VESSEL_SPEC_SHEET` | MEDIUM |
| `laden_speed_knots` | Float | Knots (nm/h) | Yes | 7.0 to 20.0 | Speed under full cargo laden condition | `VESSEL_SPEC_SHEET` | MEDIUM |
| `fuel_consumption_mt_day` | Float | MT / Day | Yes | 5.0 to 120.0 | Daily VLSFO fuel consumption at sea | `VESSEL_SPEC_SHEET` | MEDIUM |
| `age_years` | Integer | Years | Yes | 0 to 40 | Vessel age since delivery from yard | `VESSEL_SPEC_SHEET` | HIGH |
| `current_latitude` | Float | Decimal Deg | No | -90.0 to +90.0 | Last reported AIS latitude | `AIS_STREAM` | MEDIUM |
| `current_longitude` | Float | Decimal Deg | No | -180.0 to +180.0 | Last reported AIS longitude | `AIS_STREAM` | MEDIUM |
| `availability_status` | String(30) | — | Yes | AVAILABLE, ON_HIRE, IN_TRANSIT, DRY_DOCK | Current charter availability state | `FLEET_OPERATIONS` | HIGH |
| `available_from` | Date | ISO 8601 | No | YYYY-MM-DD | Date vessel becomes available for laycan | `FLEET_OPERATIONS` | HIGH |
| `data_source` | String(100) | — | Yes | Source identifier | Commercial broker or AIS provider | `FLEET_DATABASE` | MEDIUM |
| `source_date` | Date | ISO 8601 | No | YYYY-MM-DD | Observation date | — | — |

---

### 2.3 Routes (`routes.csv` / `routes` table)
Pre-computed shipping corridors connecting export origins to Indian East Coast discharge terminals.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `route_id` | String(50) | — | Yes | Primary Key (e.g. `RT_NEW_VZG`) | Unique corridor ID | `DISTANCE_TABLE` | HIGH |
| `origin_port` | String(20) | — | Yes | Valid port ID | Origin port code | `DISTANCE_TABLE` | HIGH |
| `destination_port` | String(20) | — | Yes | Valid port ID | Discharge port code | `DISTANCE_TABLE` | HIGH |
| `distance_nm` | Float | Nautical Miles | Yes | 100.0 to 25,000.0 | Nautical distance via standard shipping lanes | `ADMIRALTY_TABLES` | HIGH |
| `typical_duration_days` | Float | Days | Yes | 1.0 to 90.0 | Expected sailing duration at 13.0 knots | `ADMIRALTY_TABLES` | HIGH |
| `route_type` | String(50) | — | Yes | direct, via_suez, via_cape, coastal | Routing characteristic | `ADMIRALTY_TABLES` | HIGH |
| `seasonal_factor` | Float | Ratio | Yes | 0.8 to 1.5 | Weather slowdown multiplier (monsoon/cyclone) | `CLIMATOLOGY` | MEDIUM |
| `source` | String(100) | — | Yes | Source identifier | Navigation authority | `OFFICIAL_DISTANCES`| HIGH |

---

### 2.4 Freight Rates (`freight_rates.csv` / `freight_rates` table)
Historical and spot freight rate observations for bulk trade lanes.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Trade date | `BALTIC_EXCHANGE` | HIGH |
| `origin` | String(20) | — | Yes | Port or Region code | Loading terminal | `BALTIC_EXCHANGE` | HIGH |
| `destination` | String(20) | — | Yes | Port or Region code | Discharge terminal | `BALTIC_EXCHANGE` | HIGH |
| `vessel_class` | String(50) | — | Yes | Handysize to Capesize | Fixture vessel class | `BALTIC_EXCHANGE` | HIGH |
| `cargo_type` | String(50) | — | Yes | thermal_coal, coking_coal, iron_ore | Commodity fixtures | `BALTIC_EXCHANGE` | HIGH |
| `freight_rate` | Float | Currency/Unit | Yes | > 0.0 | Freight rate value | `BALTIC_EXCHANGE` | HIGH |
| `currency` | String(10) | — | Yes | USD, INR | Currency code | `BALTIC_EXCHANGE` | HIGH |
| `unit` | String(30) | — | Yes | per_tonne, per_day | Rate metric | `BALTIC_EXCHANGE` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Broker reports, Baltic Exchange, Platts | `BALTIC_EXCHANGE` | HIGH |

---

### 2.5 Bunker Prices (`bunker_prices.csv` / `bunker_prices` table)
Marine fuel price observations at major global and regional bunkering ports.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Price quotation date | `SHIP_AND_BUNKER` | HIGH |
| `location` | String(50) | — | Yes | Singapore, Fujairah, Visakhapatnam | Bunkering port / anchorage | `SHIP_AND_BUNKER` | HIGH |
| `fuel_type` | String(30) | — | Yes | VLSFO, MGO, IFO380 | Fuel grade specification | `SHIP_AND_BUNKER` | HIGH |
| `price_usd_mt` | Float | USD / Metric Tonne | Yes | 100.0 to 2,000.0 | Fuel price per metric tonne | `SHIP_AND_BUNKER` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Platts, Bunker Index, Ship & Bunker | `SHIP_AND_BUNKER` | HIGH |

---

### 2.6 Dry Bulk Indices (`dry_bulk_indices.csv` / `dry_bulk_indices` table)
Baltic Exchange market sentiment benchmarks.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Assessment date | `BALTIC_EXCHANGE` | HIGH |
| `index_name` | String(50) | — | Yes | BDI, BCI, BPI, BSI, BHSI | Benchmark index code | `BALTIC_EXCHANGE` | HIGH |
| `value` | Float | Index Points | Yes | >= 0.0 | Reported index value | `BALTIC_EXCHANGE` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Baltic Exchange | `BALTIC_EXCHANGE` | HIGH |

---

### 2.7 Commodity Prices (`commodity_prices.csv` / `commodity_prices` table)
Thermal coal, coking coal, and iron ore spot and index prices.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Quotation date | `COMMODITY_EXCHANGE` | HIGH |
| `commodity_name` | String(100) | — | Yes | Non-empty string | Benchmark commodity grade | `COMMODITY_EXCHANGE` | HIGH |
| `price` | Float | Currency/Unit | Yes | > 0.0 | Commodity price | `COMMODITY_EXCHANGE` | HIGH |
| `currency` | String(10) | — | Yes | USD | Quoted currency | `COMMODITY_EXCHANGE` | HIGH |
| `unit` | String(30) | — | Yes | tonne, mt | Weight unit | `COMMODITY_EXCHANGE` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Platts, Argus, ICE | `COMMODITY_EXCHANGE` | HIGH |

---

### 2.8 Port Congestion (`congestion.csv` / `port_congestion` table)
Turnaround and queue depths at Indian ports.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Observation date | `PORT_AUTHORITY` | HIGH |
| `port` | String(20) | — | Yes | Valid port ID | Port identifier | `PORT_AUTHORITY` | HIGH |
| `vessels_waiting` | Integer | Vessels | Yes | >= 0 | Number of ships waiting at anchorage | `PORT_AUTHORITY` | HIGH |
| `average_waiting_days`| Float | Days | Yes | >= 0.0 | Mean waiting time before berthing | `PORT_AUTHORITY` | HIGH |
| `berth_occupancy_pct` | Float | Percentage | Yes | 0.0 to 100.0 | Berth utilization rate | `PORT_AUTHORITY` | HIGH |
| `congestion_index` | Float | Score (0-100) | Yes | 0.0 to 100.0 | Composite congestion severity | `PORT_AUTHORITY` | MEDIUM |
| `source` | String(100) | — | Yes | Source identifier | Daily port logistics reports | `PORT_AUTHORITY` | HIGH |

---

### 2.9 Weather (`weather.csv` / `weather` table)
Sea-state and meteorological records for ports and navigation channels.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Observation date | `MET_OFFICE` | HIGH |
| `port` | String(20) | — | Yes | Valid port ID | Port location | `MET_OFFICE` | HIGH |
| `wind_speed_kmh` | Float | km/h | No | 0.0 to 350.0 | Wind velocity | `IMD_OFFICIAL` | HIGH |
| `wave_height_m` | Float | Metres | No | 0.0 to 25.0 | Significant wave height | `INCOIS_OFFICIAL`| HIGH |
| `rainfall_mm` | Float | Millimetres | No | >= 0.0 | 24-hour precipitation | `IMD_OFFICIAL` | HIGH |
| `cyclone_alert_level`| String(30) | — | Yes | None, Watch, Warning, Severe | IMD coastal warning status | `IMD_OFFICIAL` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | IMD, INCOIS, NOAA | `IMD_OFFICIAL` | HIGH |

---

### 2.10 Economic Indicators (`economic_indicators.csv` / `economic_indicators` table)
Macroeconomic drivers affecting international trade and bunker fuel markets.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Indicator date | `MACRO_FEED` | HIGH |
| `indicator_name` | String(100) | — | Yes | Brent_Crude, USD_INR, China_PMI | Variable name | `MACRO_FEED` | HIGH |
| `value` | Float | As specified by unit | Yes | Unconstrained | Numeric indicator observation | `MACRO_FEED` | HIGH |
| `unit` | String(30) | — | Yes | USD/bbl, INR, pct, index_points | Unit of measure | `MACRO_FEED` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | RBI, EIA, NBS China | `MACRO_FEED` | HIGH |

---

### 2.11 Geopolitical / Operational Events (`events.csv` / `events` table)
Disruptions, strikes, chokepoint closures, and extreme weather phenomena.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `date` | Date | ISO 8601 | Yes | YYYY-MM-DD | Event onset date | `MARITIME_NEWS` | MEDIUM |
| `event_type` | String(50) | — | Yes | Weather, Geopolitical, Port_Operation, Chokepoint | Category of disruption | `MARITIME_NEWS` | MEDIUM |
| `affected_region` | String(100) | — | No | Geographic descriptor | Route or territory affected | `MARITIME_NEWS` | MEDIUM |
| `severity` | String(30) | — | Yes | low, moderate, high, critical | Disruption impact severity | `MARITIME_NEWS` | MEDIUM |
| `description` | Text | — | No | Free text | Incident narrative | `MARITIME_NEWS` | MEDIUM |
| `source` | String(100) | — | Yes | Source identifier | Lloyd's List, TradeWinds, Reuters | `MARITIME_NEWS` | MEDIUM |

---

### 2.12 Vessel AIS Positions (`vessel_ais.csv` / `vessel_ais_positions` table)
High-frequency automated identification telemetry.

| Field | Type | Unit | Required | Range / Constraints | Description | Default Source | Confidence |
|---|---|---|:---:|---|---|---|:---:|
| `imo_number` | String(20) | — | Yes | Unique 7 digits | Vessel identifier | `AIS_RECEIVER` | HIGH |
| `timestamp` | DateTime | ISO 8601 UTC | Yes | YYYY-MM-DDTHH:MM:SSZ | Position fix timestamp | `AIS_RECEIVER` | HIGH |
| `latitude` | Float | Decimal Deg | Yes | -90.0 to +90.0 | Vessel latitude fix | `AIS_RECEIVER` | HIGH |
| `longitude` | Float | Decimal Deg | Yes | -180.0 to +180.0 | Vessel longitude fix | `AIS_RECEIVER` | HIGH |
| `speed_knots` | Float | Knots | Yes | 0.0 to 35.0 | Speed Over Ground (SOG) | `AIS_RECEIVER` | HIGH |
| `heading` | Float | Degrees | No | 0.0 to 360.0 | True heading | `AIS_RECEIVER` | HIGH |
| `destination` | String(50) | — | No | Port name or code | Reported voyage destination | `AIS_RECEIVER` | MEDIUM |
| `eta` | DateTime | ISO 8601 UTC | No | Estimated time of arrival | Master's reported ETA | `AIS_RECEIVER` | MEDIUM |
| `nav_status` | String(50) | — | Yes | under_way, at_anchor, moored | Navigational status | `AIS_RECEIVER` | HIGH |
| `source` | String(100) | — | Yes | Source identifier | Spire, MarineTraffic, AISHub | `AIS_RECEIVER` | HIGH |

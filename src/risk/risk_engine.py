"""
DockInsights — Maritime Risk Engine (Phase 8 Upgrade)

Comprehensive service-driven risk engine evaluating transparent risk scores
across 8 core maritime risk categories:
1. Market Risk
2. Port Congestion Risk
3. Weather Risk
4. Vessel Availability Risk
5. Operational Risk
6. Geopolitical / Route Risk
7. Schedule Risk
8. Demurrage Risk

Consumes live service outputs from Freight Forecasting, Port Congestion,
Voyage Economics, Fleet Availability, Weather, and Route intelligence.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.utils.constants import RiskLevel


class RiskSeverity(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def get_severity(score: float) -> RiskSeverity:
    if score < 40.0:
        return RiskSeverity.LOW
    elif score < 70.0:
        return RiskSeverity.MODERATE
    elif score < 85.0:
        return RiskSeverity.HIGH
    else:
        return RiskSeverity.CRITICAL


def risk_level_to_severity(level: RiskLevel) -> RiskSeverity:
    mapping = {
        RiskLevel.LOW: RiskSeverity.LOW,
        RiskLevel.MEDIUM: RiskSeverity.MODERATE,
        RiskLevel.HIGH: RiskSeverity.HIGH,
        RiskLevel.CRITICAL: RiskSeverity.CRITICAL,
    }
    return mapping.get(level, RiskSeverity.MODERATE)


@dataclass
class RiskCategoryResult:
    category: str
    score: float  # 0-100
    severity: RiskSeverity
    contributing_factors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def level(self) -> RiskSeverity:
        return self.severity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "score": round(self.score, 1),
            "severity": self.severity.value,
            "level": self.severity.value,
            "contributing_factors": self.contributing_factors,
            "warnings": self.warnings,
        }


@dataclass
class ComprehensiveRiskAssessment:
    overall_score: float
    overall_severity: RiskSeverity
    categories: Dict[str, RiskCategoryResult]
    summary_messages: List[str] = field(default_factory=list)

    # V2 & Aggregator Compatibility properties
    @property
    def composite_score(self) -> float:
        return self.overall_score

    @property
    def level(self) -> RiskSeverity:
        return self.overall_severity

    @property
    def breakdown(self) -> Dict[str, RiskCategoryResult]:
        return self.categories

    @property
    def dominant_risk(self) -> Optional[str]:
        if not self.categories:
            return None
        # Exclude duplicate lowercase keys when finding dominant
        primary_cats = {k: v for k, v in self.categories.items() if not k.islower()}
        target_dict = primary_cats if primary_cats else self.categories
        return max(target_dict, key=lambda k: target_dict[k].score)

    @property
    def recommendation(self) -> str:
        return self.summary_messages[0] if self.summary_messages else "Risk profile within normal operational limits."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(self.overall_score, 1),
            "composite_score": round(self.overall_score, 1),
            "overall_severity": self.overall_severity.value,
            "level": self.overall_severity.value,
            "dominant_risk": self.dominant_risk,
            "recommendation": self.recommendation,
            "summary_messages": self.summary_messages,
            "categories": {k: v.to_dict() for k, v in self.categories.items() if not k.islower()},
        }


class MaritimeRiskEngine:
    """
    Core engine evaluating transparent risk scores across 8 maritime dimensions.
    Consumes structured service outputs or raw dictionary feeds.
    """

    DEFAULT_WEIGHTS = {
        "Market": 0.15,
        "Port Congestion": 0.15,
        "Weather": 0.10,
        "Vessel Availability": 0.12,
        "Operational": 0.10,
        "Geopolitical": 0.13,
        "Schedule": 0.13,
        "Demurrage": 0.12,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def evaluate_total_risk(self, inputs: Dict[str, Any]) -> ComprehensiveRiskAssessment:
        """
        Calculates all 8 risk categories and aggregates them dynamically.

        inputs: Dictionary of raw data or service result models.
        """
        # Extract sub-dictionaries / service objects
        market_data = inputs.get("market_data") or inputs.get("freight_forecast") or {}
        port_data = inputs.get("port_data") or inputs.get("congestion_prediction") or {}
        weather_data = inputs.get("weather_data") or {}
        vessel_data = inputs.get("vessel_data") or inputs.get("vessel_availability") or {}
        geo_data = inputs.get("geopolitical_data") or inputs.get("route_data") or inputs.get("route_events") or {}
        ops_data = inputs.get("operational_data") or inputs.get("vessel_info") or {}
        sched_data = inputs.get("schedule_data") or inputs
        demurrage_data = inputs.get("demurrage_data") or inputs.get("voyage_economics") or inputs

        # 1. Market Risk
        cat_market = self.calculate_market_risk(market_data)

        # 2. Port Congestion Risk
        cat_port = self.calculate_port_congestion_risk(port_data)

        # 3. Weather Risk
        cat_weather = self.calculate_weather_risk(weather_data)

        # 4. Vessel Availability Risk
        cat_vessel = self.calculate_vessel_availability_risk(vessel_data)

        # 5. Operational Risk
        cat_ops = self.calculate_operational_risk(ops_data)

        # 6. Geopolitical / Route Risk
        cat_geo = self.calculate_geopolitical_risk(geo_data)

        # 7. Schedule Risk
        cat_sched = self.calculate_schedule_risk(sched_data, port_data)

        # 8. Demurrage Risk
        cat_dem = self.calculate_demurrage_risk(demurrage_data, port_data)

        # Canonical category mapping
        primary_categories: Dict[str, RiskCategoryResult] = {
            "Market": cat_market,
            "Port Congestion": cat_port,
            "Weather": cat_weather,
            "Vessel Availability": cat_vessel,
            "Operational": cat_ops,
            "Geopolitical": cat_geo,
            "Schedule": cat_sched,
            "Demurrage": cat_dem,
        }

        # Backward compatibility aliases (including legacy "Port" key)
        categories_dict = dict(primary_categories)
        categories_dict.update({
            "Port": cat_port,
            "market": cat_market,
            "port": cat_port,
            "port_congestion": cat_port,
            "weather": cat_weather,
            "vessel_availability": cat_vessel,
            "vessel": cat_vessel,
            "operational": cat_ops,
            "geopolitical": cat_geo,
            "route": cat_geo,
            "schedule": cat_sched,
            "demurrage": cat_dem,
        })

        # Calculate weighted average
        # Map configured weights to primary category keys
        weight_sum = 0.0
        weighted_score_sum = 0.0
        for name, cat in primary_categories.items():
            # Check weight in user config (support exact name or aliases)
            w = self.weights.get(name)
            if w is None and name == "Port Congestion":
                w = self.weights.get("Port", 0.15)
            elif w is None:
                w = self.weights.get(name.lower(), 0.12)

            weight_sum += w
            weighted_score_sum += cat.score * w

        overall_score = weighted_score_sum / weight_sum if weight_sum > 0 else 50.0
        overall_score = max(0.0, min(100.0, overall_score))
        overall_severity = get_severity(overall_score)

        # Generate summary messages and warnings
        summary_messages = []
        for name, cat in primary_categories.items():
            if cat.severity in (RiskSeverity.HIGH, RiskSeverity.CRITICAL):
                warning_text = cat.warnings[0] if cat.warnings else f"Elevated {name} risk detected."
                summary_messages.append(f"{name} Risk is {cat.severity.value}: {warning_text}")

        if not summary_messages:
            summary_messages.append("All risk dimensions are within acceptable limits.")

        return ComprehensiveRiskAssessment(
            overall_score=round(overall_score, 1),
            overall_severity=overall_severity,
            categories=categories_dict,
            summary_messages=summary_messages,
        )

    # -------------------------------------------------------------------------
    # 1. Market Risk
    # -------------------------------------------------------------------------
    def calculate_market_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes freight forecast volatility, confidence, and forecast spread.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        volatility = float(data.get("price_volatility_pct", data.get("volatility_pct", 10.0)))
        confidence = float(data.get("confidence_score", data.get("confidence", 0.85)))
        p10 = data.get("p10")
        p90 = data.get("p90")
        p50 = data.get("p50", data.get("predicted_rate"))

        factors = [f"Market Volatility: {volatility:.1f}%"]
        warnings = []

        # Baseline score proportional to volatility (10% -> 30, 30% -> 90)
        score = volatility * 3.0

        # Adjust for forecast uncertainty spread if available
        if p10 is not None and p90 is not None and p50 and float(p50) > 0:
            spread_ratio = (float(p90) - float(p10)) / float(p50)
            factors.append(f"Forecast Spread: {spread_ratio * 100:.1f}%")
            if spread_ratio > 0.40:
                score += 10.0
                warnings.append("Wide forecast quantile dispersion indicates high market rate uncertainty.")

        # Confidence penalty
        if confidence < 0.70:
            conf_penalty = (0.70 - confidence) * 30.0
            score += conf_penalty
            factors.append(f"Forecast Confidence: {confidence * 100:.0f}%")
            warnings.append("Low forecasting confidence on volatile freight route.")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("High freight market volatility detected.")

        return RiskCategoryResult(
            category="Market Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 2. Port Congestion Risk
    # -------------------------------------------------------------------------
    def calculate_port_congestion_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes port congestion predictions, expected wait days, delay probability, queue size.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        wait_days = float(data.get("expected_wait_days", data.get("waiting_time_days", 2.0)))
        vessels_waiting = data.get("vessels_waiting")
        delay_prob = data.get("delay_probability", data.get("probability_of_delay"))
        p90_wait = data.get("p90_waiting_days", data.get("p90_wait_days"))

        # Base wait days score (1 day -> 15, 6 days -> 90)
        score = wait_days * 15.0
        factors = [f"Expected Wait: {wait_days:.1f} days"]
        warnings = []

        if vessels_waiting is not None:
            v_wait = int(vessels_waiting)
            factors.append(f"Vessels in Queue: {v_wait}")
            if v_wait >= 10:
                score = max(score, min(100.0, score + 10.0))

        if delay_prob is not None:
            d_prob = float(delay_prob)
            factors.append(f"Delay Probability: {d_prob * 100:.1f}%")
            if d_prob > 0.50:
                score = max(score, d_prob * 100.0)

        if p90_wait is not None and float(p90_wait) > wait_days * 1.5:
            factors.append(f"P90 Tail Wait: {float(p90_wait):.1f} days")
            if float(p90_wait) > 5.0:
                warnings.append(f"Severe tail-risk port congestion (P90: {float(p90_wait):.1f} days).")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append(f"Severe port congestion ({wait_days:.1f} days wait).")

        return RiskCategoryResult(
            category="Port Congestion Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    def calculate_port_risk(self, data: Any) -> RiskCategoryResult:
        """Backward compatibility alias for calculate_port_congestion_risk."""
        return self.calculate_port_congestion_risk(data)

    # -------------------------------------------------------------------------
    # 3. Weather Risk
    # -------------------------------------------------------------------------
    def calculate_weather_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes wave height, wind speed, storm warnings, and ocean seasonality.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        wave_height = float(data.get("wave_height_m", 1.5))
        wind_speed = float(data.get("wind_speed_kmh", 25.0))
        storm = bool(data.get("storm_warning", False) or data.get("storm_alert", False))
        cyclone = bool(data.get("is_cyclone_season", False) or data.get("cyclone_warning", False))
        monsoon = bool(data.get("is_monsoon_season", False))

        score = wave_height * 10.0
        factors = [f"Wave Height: {wave_height:.1f}m"]
        warnings = []

        if wind_speed > 50.0:
            score += (wind_speed - 50.0) * 0.5
            factors.append(f"Wind Speed: {wind_speed:.0f} km/h")

        if storm:
            score = max(score, 85.0)
            factors.append("Active Storm Warning")
            warnings.append("Active storm warning on transit corridor.")

        if cyclone:
            score = max(score, 75.0)
            factors.append("Cyclone Season Active")
            warnings.append("Voyage traverses active cyclone basin.")

        if monsoon and score < 50.0:
            score = max(score, 45.0)
            factors.append("Monsoon Season Active")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("Adverse sea and weather conditions expected.")

        return RiskCategoryResult(
            category="Weather Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 4. Vessel Availability Risk
    # -------------------------------------------------------------------------
    def calculate_vessel_availability_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes regional available vessel count, position tightness, and lead time slack.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        available = int(data.get("available_vessels_in_region", data.get("available_vessels", 10)))
        lead_time_days = data.get("lead_time_days", data.get("open_date_slack_days"))

        # 15 vessels -> 25.0; 2 vessels -> 90.0
        score = max(0.0, 100.0 - (available * 5.0))
        factors = [f"Available Vessels: {available}"]
        warnings = []

        if lead_time_days is not None:
            slack = float(lead_time_days)
            factors.append(f"Lead Time Slack: {slack:.1f} days")
            if slack < 2.0:
                score = min(100.0, score + 20.0)
                warnings.append("Extremely tight laycan window with minimal vessel positioning slack.")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("Severe vessel shortage in region.")

        return RiskCategoryResult(
            category="Vessel Availability Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 5. Operational Risk
    # -------------------------------------------------------------------------
    def calculate_operational_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes vessel age, maintenance status, vetting inspections, and cargo handling hazards.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        maintenance_due = bool(data.get("maintenance_due", False))
        vessel_age = float(data.get("vessel_age_years", data.get("age_y", 8.0)))
        cargo_type = str(data.get("cargo_type", "")).lower()
        vetting_issues = bool(data.get("vetting_flags", False))

        score = 80.0 if maintenance_due else 20.0
        factors = ["Maintenance Due" if maintenance_due else "Normal Maintenance Status"]
        warnings = []

        if vessel_age > 18.0:
            age_penalty = min(20.0, (vessel_age - 18.0) * 4.0)
            score += age_penalty
            factors.append(f"Vessel Age: {vessel_age:.0f}y (Elevated mechanical wear)")
            if vessel_age >= 20.0:
                warnings.append(f"Over-age vessel ({vessel_age:.0f} years) carries heightened breakdown risk.")

        if "nickel" in cargo_type or "fines" in cargo_type:
            score += 15.0
            factors.append("Cargo Liquefaction / Moisture Hazard")
            warnings.append("Bulk cargo prone to moisture liquefaction requires certified IMSBC monitoring.")

        if vetting_issues:
            score = max(score, 85.0)
            factors.append("Vetting Deficiencies Reported")
            warnings.append("Vessel has open PSC or RightShip vetting deficiencies.")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("Vessel requires maintenance soon.")

        return RiskCategoryResult(
            category="Operational Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 6. Geopolitical / Route Risk
    # -------------------------------------------------------------------------
    def calculate_geopolitical_risk(self, data: Any) -> RiskCategoryResult:
        """
        Consumes route conflict index, High Risk Area (HRA) transit, and chokepoint disruptions.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        conflict_level = float(data.get("route_conflict_level", data.get("conflict_level", 0.0)))
        chokepoints = data.get("chokepoints", [])
        routing_note = str(data.get("routing_note", "")).lower()

        score = conflict_level * 10.0
        factors = [f"Conflict Index: {conflict_level:.1f}/10"]
        warnings = []

        chokepoint_str = " ".join([str(c).lower() for c in chokepoints]) + " " + routing_note
        if "suez" in chokepoint_str or "red sea" in chokepoint_str or "bab el mandeb" in chokepoint_str:
            score += 25.0
            factors.append("Red Sea / Suez Canal Transit")
            warnings.append("Route transits active Red Sea / Gulf of Aden High Risk Area.")

        if "panama" in chokepoint_str:
            score += 15.0
            factors.append("Panama Canal Transit (Draft/Booking Restrictions)")

        if "strait of hormuz" in chokepoint_str or "hormuz" in chokepoint_str:
            score += 20.0
            factors.append("Strait of Hormuz Security Transit")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("Route transits high-risk geopolitical zones.")

        return RiskCategoryResult(
            category="Geopolitical Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 7. Schedule Risk
    # -------------------------------------------------------------------------
    def calculate_schedule_risk(self, data: Any, port_data: Optional[Dict[str, Any]] = None) -> RiskCategoryResult:
        """
        Consumes voyage duration, delivery deadline, schedule buffer, and late probability.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        port_data = port_data or {}
        duration_days = float(data.get("total_duration", data.get("total_duration_days", data.get("duration_days", 14.0))))
        deadline_days = data.get("delivery_deadline_days", data.get("deadline_days"))
        late_prob = data.get("late_delivery_probability", data.get("delay_probability"))

        factors = [f"Estimated Voyage Duration: {duration_days:.1f} days"]
        warnings = []

        if deadline_days is not None:
            deadline = float(deadline_days)
            buffer_days = deadline - duration_days
            factors.append(f"Schedule Buffer: {buffer_days:.1f} days (Deadline: {deadline:.1f}d)")

            if buffer_days < 0:
                score = 95.0
                warnings.append(f"Estimated duration ({duration_days:.1f}d) exceeds deadline ({deadline:.1f}d)!")
            elif buffer_days < 2.0:
                score = 80.0
                warnings.append(f"Dangerously low schedule buffer ({buffer_days:.1f} days) before delivery deadline.")
            elif buffer_days < 4.0:
                score = 55.0
            elif buffer_days < 7.0:
                score = 30.0
            else:
                score = 15.0
        else:
            # If no strict deadline, use duration & wait risk
            wait_days = float(port_data.get("expected_wait_days", 1.5))
            score = min(60.0, 15.0 + (wait_days * 5.0))

        if late_prob is not None:
            p = float(late_prob)
            score = max(score, p * 100.0)
            factors.append(f"Late Delivery Probability: {p * 100:.1f}%")

        score = max(0.0, min(100.0, score))
        if score >= 70.0 and not warnings:
            warnings.append("High risk of schedule slippage and missing delivery window.")

        return RiskCategoryResult(
            category="Schedule Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

    # -------------------------------------------------------------------------
    # 8. Demurrage Risk
    # -------------------------------------------------------------------------
    def calculate_demurrage_risk(self, data: Any, port_data: Optional[Dict[str, Any]] = None) -> RiskCategoryResult:
        """
        Consumes expected demurrage USD, laytime allowed vs port time, and demurrage probability.
        """
        if hasattr(data, "model_dump"):
            data = data.model_dump()
        elif hasattr(data, "__dict__") and not isinstance(data, dict):
            data = data.__dict__

        port_data = port_data or {}
        expected_demurrage = float(data.get("expected_demurrage", data.get("demurrage_cost", 0.0)))
        freight_cost = float(data.get("freight_cost", data.get("total_freight", 300000.0)))
        demurrage_prob = data.get("demurrage_probability")
        wait_days = float(port_data.get("expected_wait_days", 1.5))

        factors = [f"Expected Demurrage: ${expected_demurrage:,.0f}"]
        warnings = []

        if demurrage_prob is not None:
            d_prob = float(demurrage_prob)
            score = d_prob * 100.0
            factors.append(f"Demurrage Probability: {d_prob * 100:.1f}%")
        elif expected_demurrage > 0 and freight_cost > 0:
            ratio = expected_demurrage / freight_cost
            score = min(100.0, 30.0 + (ratio * 300.0))
            factors.append(f"Demurrage/Freight Exposure: {ratio * 100:.1f}%")
        else:
            # Heuristic from port wait days vs standard 2-day laytime allowance
            if wait_days > 4.0:
                score = 80.0
            elif wait_days > 2.0:
                score = 50.0
            elif wait_days > 1.0:
                score = 25.0
            else:
                score = 10.0

        score = max(0.0, min(100.0, score))
        if score >= 70.0:
            warnings.append(f"Substantial demurrage liability exposure (${expected_demurrage:,.0f}).")

        return RiskCategoryResult(
            category="Demurrage Risk",
            score=round(score, 1),
            severity=get_severity(score),
            contributing_factors=factors,
            warnings=warnings,
        )

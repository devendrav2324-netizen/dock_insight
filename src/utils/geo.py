"""
Charter-AI — Geospatial Utilities.

Great-circle distance, sailing distance estimation, and bearing calculations
for maritime routing.
"""

import math
from typing import Tuple

# Earth radius in nautical miles
_EARTH_RADIUS_NM = 3440.065


def haversine_nm(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the great-circle distance between two points in nautical miles.

    Args:
        lat1, lon1: Origin coordinates in decimal degrees.
        lat2, lon2: Destination coordinates in decimal degrees.

    Returns:
        Distance in nautical miles.
    """
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_NM * c


def estimate_sailing_distance_nm(great_circle_nm: float) -> float:
    """
    Estimate actual sailing distance from great-circle distance.

    Applies a standard maritime routing factor of ~1.12 to account for
    coastline avoidance, straits, and weather routing.

    Args:
        great_circle_nm: Great-circle distance in nautical miles.

    Returns:
        Estimated sailing distance in nautical miles.
    """
    ROUTING_FACTOR = 1.12
    return great_circle_nm * ROUTING_FACTOR


def estimate_sailing_days(
    distance_nm: float, speed_knots: float = 13.0
) -> float:
    """
    Estimate sailing time in days.

    Args:
        distance_nm: Sailing distance in nautical miles.
        speed_knots: Vessel speed in knots (default 13.0 for bulk carriers).

    Returns:
        Estimated sailing days (decimal).
    """
    if speed_knots <= 0:
        raise ValueError("Speed must be positive.")
    hours = distance_nm / speed_knots
    return hours / 24.0


def initial_bearing(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Calculate the initial bearing from point 1 to point 2.

    Args:
        lat1, lon1: Origin coordinates in decimal degrees.
        lat2, lon2: Destination coordinates in decimal degrees.

    Returns:
        Bearing in degrees (0-360).
    """
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlon = lon2_r - lon1_r
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(
        lat2_r
    ) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360

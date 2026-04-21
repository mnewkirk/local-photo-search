"""Temporal-neighbor GPS inference (M19).

Walks photos sorted by date_taken and copies coordinates from GPS-bearing
neighbors within a time window. Supports cascading — inferred photos
become anchors for further inference — with multiplicative confidence
decay so chains self-limit.

Called by the `infer-locations` CLI and the /api/geocode/infer-preview +
/api/geocode/infer-apply endpoints. Pure-functional core: no DB writes
happen inside infer_locations(); the caller decides.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points.

    Uses the standard haversine formula; correctly handles the
    International Date Line because sin^2 is symmetric around ±π.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))

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


def _parse_date(s: str) -> datetime:
    """Parse a date_taken string. Python 3.11+ fromisoformat accepts both
    'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS'."""
    return datetime.fromisoformat(s)


def _scan_photos(db) -> tuple[list[dict], int]:
    """Return (time_sorted_photos, no_date_count).

    Each photo dict contains: id, filepath, date_taken (original str),
    date_taken_dt (parsed datetime), gps_lat, gps_lon.
    Rows whose date_taken fails to parse are counted as no_date.
    """
    rows = db.conn.execute(
        "SELECT id, filepath, date_taken, gps_lat, gps_lon "
        "FROM photos WHERE date_taken IS NOT NULL"
    ).fetchall()
    photos: list[dict] = []
    parse_failures = 0
    for r in rows:
        try:
            dt = _parse_date(r["date_taken"])
        except (ValueError, TypeError):
            parse_failures += 1
            continue
        photos.append({
            "id": r["id"],
            "filepath": r["filepath"],
            "date_taken": r["date_taken"],
            "date_taken_dt": dt,
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
        })
    photos.sort(key=lambda p: p["date_taken_dt"])

    no_date_row = db.conn.execute(
        "SELECT COUNT(*) AS cnt FROM photos WHERE date_taken IS NULL"
    ).fetchone()
    return photos, int(no_date_row["cnt"]) + parse_failures

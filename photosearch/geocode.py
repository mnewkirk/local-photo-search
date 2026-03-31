"""Reverse geocoding for GPS coordinates and location search.

Uses the offline reverse_geocoder library (GeoNames data) so no API calls
are needed — fully local.
"""

import re
from typing import Optional

# Lazy-loaded to avoid import cost when not needed
_rg = None


def _get_rg():
    global _rg
    if _rg is None:
        import reverse_geocoder as rg
        _rg = rg
    return _rg


def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """Convert GPS coordinates to a place name string.

    Returns a string like "San Francisco, California, United States"
    or None on failure.
    """
    try:
        rg = _get_rg()
        results = rg.search([(lat, lon)])
        if results:
            r = results[0]
            parts = []
            if r.get("name"):
                parts.append(r["name"])
            if r.get("admin1"):
                parts.append(r["admin1"])
            if r.get("cc"):
                parts.append(r["cc"])
            return ", ".join(parts) if parts else None
    except Exception:
        return None
    return None


def reverse_geocode_batch(coords: list[tuple[float, float]]) -> list[Optional[str]]:
    """Reverse geocode a batch of (lat, lon) pairs.

    Returns a list of place name strings (same length as input).
    """
    if not coords:
        return []
    try:
        rg = _get_rg()
        results = rg.search(coords)
        places = []
        for r in results:
            parts = []
            if r.get("name"):
                parts.append(r["name"])
            if r.get("admin1"):
                parts.append(r["admin1"])
            if r.get("cc"):
                parts.append(r["cc"])
            places.append(", ".join(parts) if parts else None)
        return places
    except Exception:
        return [None] * len(coords)


def extract_location_from_query(query: str) -> tuple[Optional[str], str]:
    """Try to extract location-related terms from a query.

    Looks for patterns like:
      "photos from Hawaii"
      "birds in California"
      "sunset at Yosemite"

    Returns (location_term, cleaned_query).
    """
    # Patterns: "in <place>", "from <place>", "at <place>", "near <place>"
    m = re.search(
        r'\b(?:in|from|at|near)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:on|in|during|from|to|this|last|today|yesterday|\d{4})|\s*$)',
        query
    )
    if m:
        location = m.group(1).strip()
        cleaned = query[:m.start()].strip() + " " + query[m.end():].strip()
        return (location, cleaned.strip())

    # Simpler: "from <place>" at end of query
    m = re.search(r'\b(?:in|from|at|near)\s+([A-Z][a-zA-Z\s]+?)\s*$', query)
    if m:
        location = m.group(1).strip()
        cleaned = query[:m.start()].strip()
        return (location, cleaned.strip())

    return (None, query)

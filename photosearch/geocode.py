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


# Common country-name-to-ISO-3166-alpha-2 mapping. The offline reverse
# geocoder produces place_names shaped "Locality, Admin1, CC" with a
# 2-letter country code, so a query like "Calvin in France" otherwise
# only matches photos whose place_name literally contains the string
# "France" (e.g. "Île-de-France" admin1) — missing every French photo
# in other regions. For uncommon countries not in this map, users can
# still search by the 2-letter ISO code directly.
_COUNTRY_NAME_TO_CODE = {
    # Americas
    "united states": "US", "usa": "US", "us": "US",
    "united states of america": "US", "america": "US",
    "canada": "CA", "mexico": "MX", "cuba": "CU", "jamaica": "JM",
    "bahamas": "BS", "costa rica": "CR", "panama": "PA",
    "brazil": "BR", "argentina": "AR", "chile": "CL", "peru": "PE",
    "colombia": "CO", "ecuador": "EC", "uruguay": "UY", "bolivia": "BO",
    "venezuela": "VE",
    # Europe
    "united kingdom": "GB", "uk": "GB", "britain": "GB",
    "great britain": "GB", "england": "GB", "scotland": "GB", "wales": "GB",
    "ireland": "IE", "france": "FR", "germany": "DE", "italy": "IT",
    "spain": "ES", "portugal": "PT", "netherlands": "NL", "holland": "NL",
    "belgium": "BE", "luxembourg": "LU", "switzerland": "CH",
    "austria": "AT", "greece": "GR", "norway": "NO", "sweden": "SE",
    "denmark": "DK", "finland": "FI", "iceland": "IS",
    "poland": "PL", "hungary": "HU", "croatia": "HR",
    "czech republic": "CZ", "czechia": "CZ",
    "slovakia": "SK", "slovenia": "SI",
    "romania": "RO", "bulgaria": "BG", "serbia": "RS", "turkey": "TR",
    "russia": "RU", "ukraine": "UA",
    # Asia / Oceania
    "japan": "JP", "china": "CN", "south korea": "KR", "korea": "KR",
    "north korea": "KP",
    "thailand": "TH", "vietnam": "VN", "indonesia": "ID",
    "malaysia": "MY", "singapore": "SG", "philippines": "PH",
    "taiwan": "TW", "hong kong": "HK", "india": "IN", "nepal": "NP",
    "sri lanka": "LK", "pakistan": "PK", "bangladesh": "BD",
    "australia": "AU", "new zealand": "NZ", "fiji": "FJ",
    # Africa / Middle East
    "morocco": "MA", "egypt": "EG", "tunisia": "TN", "algeria": "DZ",
    "south africa": "ZA", "kenya": "KE", "tanzania": "TZ",
    "ethiopia": "ET", "nigeria": "NG", "ghana": "GH",
    "israel": "IL", "jordan": "JO", "lebanon": "LB",
    "united arab emirates": "AE", "uae": "AE", "saudi arabia": "SA",
    "qatar": "QA", "oman": "OM",
}


def country_name_to_code(name: str) -> Optional[str]:
    """Resolve a country name (English, common variants) to its ISO 3166-1
    alpha-2 code. Returns None if the name isn't in the mapping.

    Accepts case-insensitively; strips surrounding whitespace. Also
    accepts bare 2-letter alphabetic strings and treats them as codes.
    """
    if not name:
        return None
    s = name.strip().lower()
    if not s:
        return None
    if s in _COUNTRY_NAME_TO_CODE:
        return _COUNTRY_NAME_TO_CODE[s]
    # Bare 2-letter uppercase-able string → treat as pre-normalized code.
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return None


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

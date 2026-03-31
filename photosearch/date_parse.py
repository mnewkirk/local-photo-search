"""Date parsing for natural-language and explicit date queries.

Supports:
  - Exact dates: "2026-03-13", "march 13 2026", "13 march 2026"
  - Month+year: "2026-03", "march 2026"
  - Year only: "2026"
  - Ranges: "2026-01 to 2026-03", "january to march 2026", "jan 2026 - mar 2026"
  - Relative: "today", "yesterday", "this week", "this month", "last month"

Returns a (date_from, date_to, cleaned_query) tuple where dates are ISO strings
and cleaned_query has the date tokens removed so CLIP/description matching
doesn't try to interpret them.
"""

import calendar
import re
from datetime import date, datetime, timedelta
from typing import Optional

_MONTH_NAMES = {
    name.lower(): num
    for num, name in enumerate(calendar.month_name) if num
}
_MONTH_ABBR = {
    name.lower(): num
    for num, name in enumerate(calendar.month_abbr) if num
}
_MONTH_MAP = {**_MONTH_NAMES, **_MONTH_ABBR}

# Pattern: "march 2026", "mar 2026"
_MONTH_YEAR_RE = re.compile(
    r'\b(' + '|'.join(sorted(_MONTH_MAP.keys(), key=len, reverse=True)) +
    r')\s+(\d{4})\b', re.IGNORECASE
)

# Pattern: "march 13 2026" or "march 13, 2026"
_MONTH_DAY_YEAR_RE = re.compile(
    r'\b(' + '|'.join(sorted(_MONTH_MAP.keys(), key=len, reverse=True)) +
    r')\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE
)

# Pattern: "13 march 2026"
_DAY_MONTH_YEAR_RE = re.compile(
    r'\b(\d{1,2})\s+(' + '|'.join(sorted(_MONTH_MAP.keys(), key=len, reverse=True)) +
    r')\s+(\d{4})\b', re.IGNORECASE
)

# ISO-like: "2026-03-13"
_ISO_DATE_RE = re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b')

# ISO-like month: "2026-03"
_ISO_MONTH_RE = re.compile(r'\b(\d{4})-(\d{2})\b')

# Year only: "2026" (standalone 4-digit year)
_YEAR_RE = re.compile(r'\b(20\d{2})\b')

# Range connectors
_RANGE_RE = re.compile(r'\s+(?:to|through|thru|-)\s+', re.IGNORECASE)

# Relative terms
_RELATIVE_TERMS = {
    "today", "yesterday", "this week", "last week",
    "this month", "last month", "this year", "last year",
}


def _month_end(year: int, month: int) -> int:
    """Return last day of month."""
    return calendar.monthrange(year, month)[1]


def _parse_single_date(text: str) -> Optional[tuple[str, str, str]]:
    """Try to parse a single date/month/year from text.

    Returns (date_from, date_to, matched_text) or None.
    """
    text = text.strip()

    # Relative dates
    today = date.today()
    text_lower = text.lower().strip()

    if text_lower == "today":
        d = today.isoformat()
        return (d, d, text)
    if text_lower == "yesterday":
        d = (today - timedelta(days=1)).isoformat()
        return (d, d, text)
    if text_lower == "this week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return (start.isoformat(), end.isoformat(), text)
    if text_lower == "last week":
        start = today - timedelta(days=today.weekday() + 7)
        end = start + timedelta(days=6)
        return (start.isoformat(), end.isoformat(), text)
    if text_lower == "this month":
        start = today.replace(day=1)
        end = today.replace(day=_month_end(today.year, today.month))
        return (start.isoformat(), end.isoformat(), text)
    if text_lower == "last month":
        first = today.replace(day=1)
        prev_last = first - timedelta(days=1)
        prev_first = prev_last.replace(day=1)
        return (prev_first.isoformat(), prev_last.isoformat(), text)
    if text_lower == "this year":
        return (f"{today.year}-01-01", f"{today.year}-12-31", text)
    if text_lower == "last year":
        y = today.year - 1
        return (f"{y}-01-01", f"{y}-12-31", text)

    # "march 13 2026" or "march 13, 2026"
    m = _MONTH_DAY_YEAR_RE.search(text)
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        day = int(m.group(2))
        year = int(m.group(3))
        d = f"{year}-{month:02d}-{day:02d}"
        return (d, d, m.group(0))

    # "13 march 2026"
    m = _DAY_MONTH_YEAR_RE.search(text)
    if m:
        day = int(m.group(1))
        month = _MONTH_MAP[m.group(2).lower()]
        year = int(m.group(3))
        d = f"{year}-{month:02d}-{day:02d}"
        return (d, d, m.group(0))

    # "march 2026" or "mar 2026"
    m = _MONTH_YEAR_RE.search(text)
    if m:
        month = _MONTH_MAP[m.group(1).lower()]
        year = int(m.group(2))
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{_month_end(year, month):02d}"
        return (start, end, m.group(0))

    # "2026-03-13"
    m = _ISO_DATE_RE.search(text)
    if m:
        d = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return (d, d, m.group(0))

    # "2026-03"
    m = _ISO_MONTH_RE.search(text)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        start = f"{year}-{month:02d}-01"
        end = f"{year}-{month:02d}-{_month_end(year, month):02d}"
        return (start, end, m.group(0))

    # "2026"
    m = _YEAR_RE.search(text)
    if m:
        year = int(m.group(1))
        return (f"{year}-01-01", f"{year}-12-31", m.group(0))

    return None


def parse_date_from_query(query: str) -> tuple[Optional[str], Optional[str], str]:
    """Extract date filter from a search query.

    Returns (date_from, date_to, cleaned_query) where cleaned_query has the
    date portion removed.
    """
    if not query:
        return (None, None, query)

    # Check for range pattern: "X to Y"
    range_match = _RANGE_RE.search(query)
    if range_match:
        left = query[:range_match.start()]
        right = query[range_match.end():]

        # Try parsing both sides as dates
        left_date = _parse_single_date(left)
        right_date = _parse_single_date(right)

        if left_date and right_date:
            # Both sides are dates — it's a pure date range
            return (left_date[0], right_date[1], "")

        # Maybe the range is embedded: "birds january to march 2026"
        # Try extracting range from the full query with month names
        range_in_query = _try_extract_embedded_range(query)
        if range_in_query:
            return range_in_query

    # Check for embedded range anyway (handles "january to march 2026" etc.)
    range_in_query = _try_extract_embedded_range(query)
    if range_in_query:
        return range_in_query

    # Try single date
    parsed = _parse_single_date(query)
    if parsed:
        date_from, date_to, matched = parsed
        cleaned = query.replace(matched, "").strip()
        # Clean up leftover connectors
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return (date_from, date_to, cleaned)

    return (None, None, query)


def _try_extract_embedded_range(query: str) -> Optional[tuple[str, str, str]]:
    """Try to find a date range embedded in a larger query.

    Handles patterns like:
      "birds january to march 2026"
      "sunset 2026-01 to 2026-03"
      "beach march 2026 to june 2026"
    """
    # Pattern: "month to month year" (e.g., "january to march 2026")
    month_names = '|'.join(sorted(_MONTH_MAP.keys(), key=len, reverse=True))
    m = re.search(
        r'\b(' + month_names + r')\s+(?:to|through|thru|-)\s+(' + month_names + r')\s+(\d{4})\b',
        query, re.IGNORECASE
    )
    if m:
        m1 = _MONTH_MAP[m.group(1).lower()]
        m2 = _MONTH_MAP[m.group(2).lower()]
        year = int(m.group(3))
        date_from = f"{year}-{m1:02d}-01"
        date_to = f"{year}-{m2:02d}-{_month_end(year, m2):02d}"
        cleaned = query[:m.start()].strip() + " " + query[m.end():].strip()
        return (date_from, date_to, cleaned.strip())

    # Pattern: "month year to month year" (e.g., "march 2026 to june 2026")
    m = re.search(
        r'\b(' + month_names + r')\s+(\d{4})\s+(?:to|through|thru|-)\s+(' + month_names + r')\s+(\d{4})\b',
        query, re.IGNORECASE
    )
    if m:
        m1 = _MONTH_MAP[m.group(1).lower()]
        y1 = int(m.group(2))
        m2 = _MONTH_MAP[m.group(3).lower()]
        y2 = int(m.group(4))
        date_from = f"{y1}-{m1:02d}-01"
        date_to = f"{y2}-{m2:02d}-{_month_end(y2, m2):02d}"
        cleaned = query[:m.start()].strip() + " " + query[m.end():].strip()
        return (date_from, date_to, cleaned.strip())

    # Pattern: "YYYY-MM to YYYY-MM" or "YYYY-MM-DD to YYYY-MM-DD"
    m = re.search(
        r'\b(\d{4}-\d{2}(?:-\d{2})?)\s+(?:to|through|thru|-)\s+(\d{4}-\d{2}(?:-\d{2})?)\b',
        query
    )
    if m:
        left = _parse_single_date(m.group(1))
        right = _parse_single_date(m.group(2))
        if left and right:
            cleaned = query[:m.start()].strip() + " " + query[m.end():].strip()
            return (left[0], right[1], cleaned.strip())

    return None

"""Tests for the date parsing module.

Covers: ISO dates, natural language, relative dates, ranges, edge cases.
"""

import pytest
from datetime import date, timedelta

from photosearch.date_parse import parse_date_from_query


class TestISODates:
    def test_full_date(self):
        d_from, d_to, cleaned = parse_date_from_query("2026-03-13")
        assert d_from == "2026-03-13"
        assert d_to == "2026-03-13"

    def test_month_only(self):
        d_from, d_to, cleaned = parse_date_from_query("2026-03")
        assert d_from == "2026-03-01"
        assert d_to == "2026-03-31"

    def test_year_only(self):
        d_from, d_to, cleaned = parse_date_from_query("2026")
        assert d_from == "2026-01-01"
        assert d_to == "2026-12-31"


class TestNaturalLanguageDates:
    def test_month_year(self):
        d_from, d_to, cleaned = parse_date_from_query("march 2026")
        assert d_from == "2026-03-01"
        assert d_to == "2026-03-31"

    def test_month_abbrev_year(self):
        d_from, d_to, cleaned = parse_date_from_query("mar 2026")
        assert d_from == "2026-03-01"
        assert d_to == "2026-03-31"

    def test_month_day_year(self):
        d_from, d_to, cleaned = parse_date_from_query("march 13, 2026")
        assert d_from == "2026-03-13"
        assert d_to == "2026-03-13"

    def test_day_month_year(self):
        d_from, d_to, cleaned = parse_date_from_query("13 march 2026")
        assert d_from == "2026-03-13"
        assert d_to == "2026-03-13"


class TestRelativeDates:
    def test_today(self):
        d_from, d_to, cleaned = parse_date_from_query("today")
        expected = date.today().isoformat()
        assert d_from == expected
        assert d_to == expected

    def test_yesterday(self):
        d_from, d_to, cleaned = parse_date_from_query("yesterday")
        expected = (date.today() - timedelta(days=1)).isoformat()
        assert d_from == expected
        assert d_to == expected


class TestDateRanges:
    def test_iso_range(self):
        d_from, d_to, cleaned = parse_date_from_query("2026-01 to 2026-03")
        assert d_from == "2026-01-01"
        assert d_to == "2026-03-31"


class TestCleanedQuery:
    def test_date_removed_from_query(self):
        """Date tokens should be stripped from the cleaned query."""
        d_from, d_to, cleaned = parse_date_from_query("sunset march 2026")
        assert "march" not in cleaned.lower()
        assert "2026" not in cleaned
        assert "sunset" in cleaned.lower()

    def test_no_date_returns_none(self):
        d_from, d_to, cleaned = parse_date_from_query("beautiful sunset")
        assert d_from is None
        assert d_to is None
        assert "beautiful sunset" in cleaned


class TestEdgeCases:
    def test_empty_query(self):
        d_from, d_to, cleaned = parse_date_from_query("")
        assert d_from is None
        assert d_to is None

    def test_none_query(self):
        d_from, d_to, cleaned = parse_date_from_query(None)
        assert d_from is None
        assert d_to is None

    def test_february_leap_year(self):
        d_from, d_to, cleaned = parse_date_from_query("2024-02")
        assert d_from == "2024-02-01"
        assert d_to == "2024-02-29"  # 2024 is a leap year

    def test_february_non_leap_year(self):
        d_from, d_to, cleaned = parse_date_from_query("2026-02")
        assert d_from == "2026-02-01"
        assert d_to == "2026-02-28"

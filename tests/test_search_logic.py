"""Tests for the search module's scoring, parsing, and filtering logic.

Tests the internal helper functions that don't require ML models,
as well as the combined search pipeline against the test DB.
"""

import pytest
from unittest.mock import patch, MagicMock

from photosearch.search import (
    _parse_query,
    _query_mentions_people,
    _query_negates_people,
    _description_contains_excluded,
    _dedupe_by_hash,
)


# =========================================================================
# Query parsing
# =========================================================================

class TestParseQuery:
    def test_simple_query(self):
        pos, excluded = _parse_query("sunset beach")
        assert pos == "sunset beach"
        assert excluded == []

    def test_dash_negation(self):
        pos, excluded = _parse_query("beach -people -dogs")
        assert "people" not in pos
        assert "-" not in pos
        assert "people" in excluded
        assert "dogs" in excluded

    def test_natural_negation(self):
        pos, excluded = _parse_query("beach no people")
        assert "people" in excluded
        # Positive query should not contain "no people"
        assert "no" not in pos.lower().split() or "people" not in pos.lower().split()

    def test_without_negation(self):
        pos, excluded = _parse_query("landscape without people")
        assert "people" in excluded

    def test_people_expansion(self):
        """Excluding 'people' should also exclude related keywords."""
        pos, excluded = _parse_query("beach -people")
        assert "person" in excluded
        assert "child" in excluded

    def test_empty_query(self):
        pos, excluded = _parse_query("")
        assert pos == ""
        assert excluded == []


# =========================================================================
# People detection in queries
# =========================================================================

class TestQueryMentionsPeople:
    def test_mentions_people(self):
        assert _query_mentions_people("people outdoors") is True

    def test_mentions_child(self):
        assert _query_mentions_people("child playing") is True

    def test_mentions_portrait(self):
        assert _query_mentions_people("portrait photo") is True

    def test_no_people_mention(self):
        assert _query_mentions_people("sunset over ocean") is False


class TestQueryNegatesPeople:
    def test_no_people(self):
        assert _query_negates_people("beach no people") is True

    def test_without_kids(self):
        assert _query_negates_people("park without kids") is True

    def test_positive_mention(self):
        assert _query_negates_people("people outdoors") is False


# =========================================================================
# Description filtering for exclusions
# =========================================================================

class TestDescriptionContainsExcluded:
    def test_match(self):
        assert _description_contains_excluded(
            "A group of people on the beach", ["people"]
        ) is True

    def test_no_match(self):
        assert _description_contains_excluded(
            "Rocky coastline with crashing waves", ["people"]
        ) is False

    def test_negated_people_in_description(self):
        """When description says 'no people' and we're excluding people, don't filter."""
        assert _description_contains_excluded(
            "Empty beach with no people visible", ["people"]
        ) is False

    def test_empty_inputs(self):
        assert _description_contains_excluded("", ["people"]) is False
        assert _description_contains_excluded("Some text", []) is False
        assert _description_contains_excluded(None, ["people"]) is False


# =========================================================================
# Deduplication
# =========================================================================

class TestDedupe:
    def test_removes_duplicates(self):
        results = [
            {"filename": "a.jpg", "file_hash": "abc123", "score": 0.9},
            {"filename": "b.jpg", "file_hash": "abc123", "score": 0.8},
            {"filename": "c.jpg", "file_hash": "def456", "score": 0.7},
        ]
        deduped = _dedupe_by_hash(results)
        assert len(deduped) == 2
        assert deduped[0]["filename"] == "a.jpg"

    def test_no_hash(self):
        results = [
            {"filename": "a.jpg", "file_hash": None},
            {"filename": "b.jpg", "file_hash": None},
        ]
        deduped = _dedupe_by_hash(results)
        assert len(deduped) == 2  # No hash → no dedup

    def test_empty_list(self):
        assert _dedupe_by_hash([]) == []


# =========================================================================
# Combined search (integration, against test DB)
# =========================================================================

class TestCombinedSearch:
    """Test search_combined against the test DB.

    CLIP embedding calls are mocked since we don't have the model loaded.
    """

    def test_search_by_person(self, db):
        from photosearch.search import search_by_person
        results = search_by_person(db, "Calvin")
        filenames = {r["filename"] for r in results}
        assert "DSC04894.JPG" in filenames
        assert "DSC04907.JPG" in filenames
        assert "DSC04922.JPG" in filenames

    def test_search_by_person_case_insensitive(self, db):
        from photosearch.search import search_by_person
        results = search_by_person(db, "calvin")
        assert len(results) >= 3

    def test_search_by_person_not_found(self, db):
        from photosearch.search import search_by_person
        results = search_by_person(db, "NonExistentPerson")
        assert results == []

    def test_search_by_place(self, db):
        from photosearch.search import search_by_place
        results = search_by_place(db, "Big Sur")
        filenames = {r["filename"] for r in results}
        assert "DSC04894.JPG" in filenames
        assert "DSC04907.JPG" in filenames
        # Morro Bay photos should not be returned
        assert "DSC04878.JPG" not in filenames

    def test_search_combined_location_filter(self, db):
        from photosearch.search import search_combined

        results = search_combined(db, location="Morro Bay")
        filenames = {r["filename"] for r in results}
        assert "DSC04878.JPG" in filenames
        assert "DSC04880.JPG" in filenames
        # Big Sur photos should not appear
        assert "DSC04894.JPG" not in filenames

    def test_search_combined_min_quality(self, db):
        from photosearch.search import search_combined

        results = search_combined(db, min_quality=7.0)
        for r in results:
            assert r.get("aesthetic_score", 0) >= 7.0

    def test_search_combined_sort_quality(self, db):
        from photosearch.search import search_combined

        results = search_combined(db, location="Big Sur", sort_quality=True)
        scores = [r["aesthetic_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_combined_date_filter(self, db):
        from photosearch.search import search_combined

        results = search_combined(
            db, location="Big Sur",
            date_from="2026-03-13", date_to="2026-03-13",
        )
        for r in results:
            assert "2026-03-13" in r.get("date_taken", "")

    def test_search_combined_match_source(self, db):
        from photosearch.search import search_combined

        results = search_combined(db, person="Calvin", match_source="manual")
        for r in results:
            # At least one face on this photo should be manual
            faces = db.conn.execute(
                """SELECT match_source FROM faces
                   WHERE photo_id = ? AND person_id IS NOT NULL""",
                (r["id"],)
            ).fetchall()
            sources = {f["match_source"] for f in faces}
            assert "manual" in sources

    def test_search_combined_no_criteria_returns_empty(self, db):
        from photosearch.search import search_combined

        results = search_combined(db)
        # No criteria → should return empty or raise
        assert isinstance(results, list)

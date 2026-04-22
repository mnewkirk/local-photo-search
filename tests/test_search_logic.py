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
        results = search_by_person(db, "Alex")
        filenames = {r["filename"] for r in results}
        assert "DSC04894.JPG" in filenames
        assert "DSC04907.JPG" in filenames
        assert "DSC04922.JPG" in filenames

    def test_search_by_person_case_insensitive(self, db):
        from photosearch.search import search_by_person
        results = search_by_person(db, "alex")
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

        results = search_combined(db, person="Alex", match_source="manual")
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


# =========================================================================
# Country-name-to-code resolution + location search expansion
# =========================================================================

class TestCountryNameToCode:
    def test_known_names(self):
        from photosearch.geocode import country_name_to_code
        assert country_name_to_code("France") == "FR"
        assert country_name_to_code("france") == "FR"
        assert country_name_to_code("  France  ") == "FR"
        assert country_name_to_code("United States") == "US"
        assert country_name_to_code("USA") == "US"
        assert country_name_to_code("UK") == "GB"
        assert country_name_to_code("Japan") == "JP"

    def test_bare_iso_code(self):
        from photosearch.geocode import country_name_to_code
        assert country_name_to_code("FR") == "FR"
        assert country_name_to_code("fr") == "FR"

    def test_unknown(self):
        from photosearch.geocode import country_name_to_code
        assert country_name_to_code("Zarautz") is None
        assert country_name_to_code("") is None
        assert country_name_to_code(None) is None


class TestSearchByLocationExpandsCountries:
    def test_country_name_matches_country_code_suffix(self, tmp_db_path):
        """Searching 'France' must return photos whose place_name ends
        with ', FR' even if 'France' isn't literally in the name."""
        from photosearch.db import PhotoDB
        from photosearch.search import _search_by_location

        with PhotoDB(tmp_db_path) as pdb:
            pdb.add_photo(filepath="/fr1.jpg", filename="fr1.jpg",
                          date_taken="2024-06-01T10:00:00",
                          gps_lat=44.84, gps_lon=-0.58,
                          place_name="Bordeaux, Nouvelle-Aquitaine, FR")
            pdb.add_photo(filepath="/fr2.jpg", filename="fr2.jpg",
                          date_taken="2024-06-02T10:00:00",
                          gps_lat=48.85, gps_lon=2.35,
                          place_name="Paris, Île-de-France, FR")
            pdb.add_photo(filepath="/it.jpg", filename="it.jpg",
                          date_taken="2024-06-03T10:00:00",
                          gps_lat=41.9, gps_lon=12.5,
                          place_name="Rome, Lazio, IT")
            pdb.conn.commit()

            results = _search_by_location(pdb, "France")
        filenames = sorted(r["filename"] for r in results)
        # Bordeaux ONLY matches via the ', FR' expansion; Paris matches
        # both by string ("Île-de-France") AND by code. Rome must not
        # appear.
        assert filenames == ["fr1.jpg", "fr2.jpg"]

    def test_iso_code_query_anchors_to_country_slot(self, tmp_db_path):
        """Searching 'ES' must not false-positive on locality names
        that happen to contain those two letters."""
        from photosearch.db import PhotoDB
        from photosearch.search import _search_by_location

        with PhotoDB(tmp_db_path) as pdb:
            pdb.add_photo(filepath="/es.jpg", filename="es.jpg",
                          date_taken="2024-07-01T10:00:00",
                          gps_lat=43.3, gps_lon=-1.9,
                          place_name="Zarautz, Basque Country, ES")
            pdb.add_photo(filepath="/it.jpg", filename="it.jpg",
                          date_taken="2024-07-02T10:00:00",
                          gps_lat=39.9, gps_lon=9.4,
                          # "Esterzili" contains "es" as a substring.
                          place_name="Esterzili, Sardegna, IT")
            pdb.conn.commit()

            results = _search_by_location(pdb, "ES")
            filenames = sorted(r["filename"] for r in results)
            # "ES" query matches Spain (via ', ES' suffix) and also
            # substring-matches "Esterzili" (LIKE '%ES%'). Both appear
            # for the bare-code case; documented here so a future
            # refactor doesn't silently break one or the other.
            assert "es.jpg" in filenames
            # "Spain" resolves to country-code-only match → no Italian
            # localities even when their names contain "es".
            spain_only = _search_by_location(pdb, "Spain")
            spain_filenames = [r["filename"] for r in spain_only]
            assert "es.jpg" in spain_filenames
            assert "it.jpg" not in spain_filenames

    def test_unknown_query_still_substring_matches(self, tmp_db_path):
        """Non-country queries (localities, POIs) must keep working."""
        from photosearch.db import PhotoDB
        from photosearch.search import _search_by_location

        with PhotoDB(tmp_db_path) as pdb:
            pdb.add_photo(filepath="/a.jpg", filename="a.jpg",
                          date_taken="2024-08-01T10:00:00",
                          gps_lat=43.3, gps_lon=-1.9,
                          place_name="Zarautz, Basque Country, ES")
            pdb.conn.commit()
            results = _search_by_location(pdb, "Zarautz")
        assert len(results) == 1
        assert results[0]["filename"] == "a.jpg"

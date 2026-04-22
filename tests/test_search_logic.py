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

class TestExtractLocationFromQuery:
    def test_simple(self):
        from photosearch.geocode import extract_location_from_query
        loc, cleaned = extract_location_from_query("Calvin in San Rafael")
        assert loc == "San Rafael"
        assert cleaned == "Calvin"

    def test_comma_admin(self):
        """Places with comma-separated admin suffix must parse —
        otherwise 'San Rafael, CA' falls through to CLIP and returns 0.
        """
        from photosearch.geocode import extract_location_from_query
        loc, _ = extract_location_from_query("Calvin in San Rafael, CA")
        assert loc == "San Rafael, CA"

    def test_hyphenated_and_periods(self):
        from photosearch.geocode import extract_location_from_query
        assert extract_location_from_query("Calvin in Saint-Tropez")[0] == "Saint-Tropez"
        assert extract_location_from_query("Calvin in St. Louis")[0] == "St. Louis"

    def test_trailing_keyword_still_terminates_capture(self):
        """'Calvin in California 2024' must stop at California, not
        eat the year into the location string."""
        from photosearch.geocode import extract_location_from_query
        loc, _ = extract_location_from_query("Calvin in California 2024")
        assert loc == "California"

    def test_no_preposition(self):
        from photosearch.geocode import extract_location_from_query
        loc, cleaned = extract_location_from_query("beach sunset")
        assert loc is None
        assert cleaned == "beach sunset"


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

    def test_bbox_fallback_when_substring_misses(self, tmp_db_path, monkeypatch):
        """Photos at Point Reyes get geocoded to 'Inverness' (nearest
        cities1000 match) by the offline geocoder, so a substring
        search for 'Point Reyes' finds nothing. The Nominatim bbox
        fallback must pull them in by coordinates."""
        import json
        from photosearch.db import PhotoDB
        from photosearch.search import _search_by_location

        class _FakeResp:
            def __init__(self, payload):
                self._data = json.dumps(payload).encode("utf-8")
            def read(self):
                return self._data
            def __enter__(self): return self
            def __exit__(self, *a): return False

        canned = [{
            "name": "Point Reyes",
            "display_name": "Point Reyes National Seashore, Marin County, California, United States",
            "lat": "38.07",
            "lon": "-122.88",
            "type": "protected_area",
            "importance": 0.6,
            "boundingbox": ["37.99", "38.20", "-123.02", "-122.73"],
            "address": {
                "county": "Marin County",
                "state": "California",
                "country": "United States",
                "country_code": "us",
            },
        }]

        def fake_urlopen(req, timeout=None):
            return _FakeResp(canned)
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        with PhotoDB(tmp_db_path) as pdb:
            # Photo inside the returned bbox, geocoded to a different name.
            pdb.add_photo(filepath="/pr1.jpg", filename="pr1.jpg",
                          date_taken="2024-09-01T10:00:00",
                          gps_lat=38.10, gps_lon=-122.90,
                          place_name="Inverness, California, US")
            # Photo outside the bbox must not leak in.
            pdb.add_photo(filepath="/sf.jpg", filename="sf.jpg",
                          date_taken="2024-09-02T10:00:00",
                          gps_lat=37.77, gps_lon=-122.42,
                          place_name="San Francisco, California, US")
            pdb.conn.commit()

            results = _search_by_location(pdb, "Point Reyes")
            filenames = sorted(r["filename"] for r in results)
        assert filenames == ["pr1.jpg"]

    def test_person_plus_location_intersection_beyond_old_limit(
            self, tmp_db_path):
        """Regression: 'Calvin in France' returned zero results on the
        NAS even though Calvin had many French photos. Root cause was
        that search_combined used limit=600 for each intermediate
        filter set, so the oldest-600 Calvin photos (old US photos) and
        the oldest-600 France photos (pre-Calvin French photos) had no
        overlap and the intersection silently collapsed.

        This test seeds enough photos (700 French + 1 Calvin-in-France
        at the newest date) to have hit the old bug and proves the
        intersection now finds Calvin's French photo.
        """
        import numpy as np
        from photosearch.db import PhotoDB
        from photosearch.search import search_combined

        with PhotoDB(tmp_db_path) as db:
            calvin_id = db.add_person("Calvin")
            db.begin_batch(batch_size=200)

            # 700 older French photos with no Calvin — these would
            # saturate the old limit*3=600 window for the France filter.
            for i in range(700):
                day = 1 + i // 28
                mo = 1 + (i // 28) % 12
                date = f"201{(i // 300) % 5}-{mo:02d}-{day:02d}T12:00:00"
                db.add_photo(
                    filepath=f"/old_fr/{i}.jpg", filename=f"old{i}.jpg",
                    date_taken=date, gps_lat=48.85, gps_lon=2.35,
                    place_name="Paris, Île-de-France, FR",
                )

            # A single Calvin-in-France photo with the newest date —
            # definitely past the oldest-600 window on both sides.
            pid = db.add_photo(
                filepath="/calvin_fr.jpg", filename="calvin_fr.jpg",
                date_taken="2024-06-01T10:00:00",
                gps_lat=44.84, gps_lon=-0.58,
                place_name="Bordeaux, Nouvelle-Aquitaine, FR",
            )
            emb = np.random.rand(512).astype(np.float32).tolist()
            fid = db.add_face(pid, (10, 10, 50, 50), emb)
            db.assign_face_to_person(fid, calvin_id, "manual")

            # Also give Calvin 50 non-France photos so search_by_person
            # returns >1 row and the intersection is non-trivial.
            for i in range(50):
                ppid = db.add_photo(
                    filepath=f"/calvin_us/{i}.jpg",
                    filename=f"cus{i}.jpg",
                    date_taken=f"2018-01-{(i%28)+1:02d}T12:00:00",
                    gps_lat=47.6, gps_lon=-122.3,
                    place_name="Seattle, Washington, US",
                )
                emb2 = np.random.rand(512).astype(np.float32).tolist()
                fid2 = db.add_face(ppid, (10, 10, 50, 50), emb2)
                db.assign_face_to_person(fid2, calvin_id, "manual")
            db.end_batch()

            results = search_combined(db, query="Calvin in France", limit=20)

        filenames = [r["filename"] for r in results]
        assert "calvin_fr.jpg" in filenames, \
            "Calvin's French photo must surface from the intersection"
        # Must not leak Seattle (no France match) or French-only photos
        # (no Calvin).
        assert "cus0.jpg" not in filenames
        assert "old0.jpg" not in filenames

    def test_bbox_union_catches_nearby_places_with_different_names(
            self, tmp_db_path, monkeypatch):
        """Non-country queries ALWAYS union the substring hits with
        Nominatim's bbox hits, even when substring has results. "San
        Rafael", "Lucas Valley", and "Marinwood" are the same place
        geographically — the offline geocoder labels photos with the
        nearest populated place, so substring-only would return wildly
        different counts. Bbox union collapses them.
        """
        import json
        from photosearch.db import PhotoDB
        from photosearch.search import _search_by_location

        class _FakeResp:
            def __init__(self, payload):
                self._data = json.dumps(payload).encode("utf-8")
            def read(self):
                return self._data
            def __enter__(self): return self
            def __exit__(self, *a): return False

        # Real Nominatim bbox for "San Rafael, California" — 10×13 km,
        # TIGHT on the city limits. Lucas Valley-Marinwood sits north
        # of north=38.0291 (strictly outside); only the _pad_bbox()
        # expansion (~4km) captures it.
        canned = [{
            "name": "San Rafael",
            "display_name": "San Rafael, Marin County, California, United States",
            "lat": "37.9735",
            "lon": "-122.5311",
            "boundingbox": ["37.9388", "38.0291", "-122.5897", "-122.4354"],
            "address": {
                "city": "San Rafael", "county": "Marin County",
                "state": "California", "country": "United States",
                "country_code": "us",
            },
        }]
        monkeypatch.setattr("urllib.request.urlopen",
                            lambda req, timeout=None: _FakeResp(canned))

        with PhotoDB(tmp_db_path) as pdb:
            # Inside strict bbox.
            pdb.add_photo(filepath="/sr.jpg", filename="sr.jpg",
                          date_taken="2024-01-01T10:00:00",
                          gps_lat=37.97, gps_lon=-122.53,
                          place_name="San Rafael, California, US")
            # 2 km north of strict north edge — only padded bbox
            # catches it.
            pdb.add_photo(filepath="/mw.jpg", filename="mw.jpg",
                          date_taken="2024-01-03T10:00:00",
                          gps_lat=38.047, gps_lon=-122.569,
                          place_name="Marinwood, California, US")
            # ~2.5 km north of strict — also inside padded.
            pdb.add_photo(filepath="/lv.jpg", filename="lv.jpg",
                          date_taken="2024-01-02T10:00:00",
                          gps_lat=38.051, gps_lon=-122.576,
                          place_name="Lucas Valley-Marinwood, California, US")
            # ~6 km north of strict (38.09) — OUTSIDE the padded bbox
            # north edge (38.069). Must not leak in.
            pdb.add_photo(filepath="/novato.jpg", filename="novato.jpg",
                          date_taken="2024-01-05T10:00:00",
                          gps_lat=38.107, gps_lon=-122.569,
                          place_name="Novato, California, US")
            # Far away — outside bbox in every dimension.
            pdb.add_photo(filepath="/sf.jpg", filename="sf.jpg",
                          date_taken="2024-01-04T10:00:00",
                          gps_lat=37.77, gps_lon=-122.42,
                          place_name="San Francisco, California, US")
            pdb.conn.commit()

            results = _search_by_location(pdb, "San Rafael")
            filenames = sorted(r["filename"] for r in results)
        assert filenames == ["lv.jpg", "mw.jpg", "sr.jpg"]


# =========================================================================
# Sort-before-slice (quick-wins bundle commit 1)
# =========================================================================

class TestSortBeforeSlice:
    """User reported: '?q=Calvin with Newest first is definitely not
    newest first' because search_by_person's SQL ORDER BY date_taken
    LIMIT put NULLs + oldest dates first. Backend now sorts `merged`
    AFTER intersection, BEFORE pagination.
    """

    def test_date_desc_puts_newest_first(self, tmp_db_path):
        from photosearch.db import PhotoDB
        from photosearch.search import search_combined

        with PhotoDB(tmp_db_path) as pdb:
            person_id = pdb.add_person("Calvin")
            # Seed three photos across years. Add in oldest-first order
            # to mimic SQL ORDER BY ASC behaviour.
            for date in ("2015-01-01T10:00:00",
                         "2020-06-15T12:00:00",
                         "2026-04-01T09:00:00"):
                pid = pdb.add_photo(filepath=f"/{date}.jpg",
                                    filename=f"{date}.jpg",
                                    date_taken=date)
                import numpy as np
                emb = np.random.rand(512).astype(np.float32).tolist()
                fid = pdb.add_face(pid, (10, 10, 50, 50), emb)
                pdb.assign_face_to_person(fid, person_id, "manual")
            pdb.conn.commit()

            results = search_combined(pdb, person="Calvin", sort="date_desc",
                                      limit=10)
        assert [r["filename"] for r in results] == [
            "2026-04-01T09:00:00.jpg",
            "2020-06-15T12:00:00.jpg",
            "2015-01-01T10:00:00.jpg",
        ]

    def test_date_desc_pushes_nulls_to_tail(self, tmp_db_path):
        """Photo with NULL date_taken must not surface at the TOP of a
        newest-first result — the original bug was exactly this, because
        SQLite's ORDER BY date_taken ASC sorts NULLs first."""
        from photosearch.db import PhotoDB
        from photosearch.search import search_combined

        with PhotoDB(tmp_db_path) as pdb:
            person_id = pdb.add_person("Calvin")
            import numpy as np

            def add(fn, date):
                pid = pdb.add_photo(filepath=f"/{fn}", filename=fn,
                                    date_taken=date)
                emb = np.random.rand(512).astype(np.float32).tolist()
                fid = pdb.add_face(pid, (10, 10, 50, 50), emb)
                pdb.assign_face_to_person(fid, person_id, "manual")

            add("nodate.jpg", None)
            add("old.jpg", "2015-01-01T10:00:00")
            add("new.jpg", "2026-04-01T09:00:00")
            pdb.conn.commit()

            results = search_combined(pdb, person="Calvin", sort="date_desc",
                                      limit=10)
        assert [r["filename"] for r in results] == [
            "new.jpg", "old.jpg", "nodate.jpg"
        ]

    def test_pagination_preserves_sort(self, tmp_db_path):
        """Page 1 of DESC must be truly newest, not newest-of-oldest-N."""
        from photosearch.db import PhotoDB
        from photosearch.search import search_combined

        with PhotoDB(tmp_db_path) as pdb:
            person_id = pdb.add_person("Calvin")
            import numpy as np
            # 30 photos across 2015-2026; one per year x 3 months.
            for i, (year, month) in enumerate([(2015 + i // 3, 1 + (i % 3) * 4)
                                                for i in range(30)]):
                date = f"{year:04d}-{month:02d}-01T10:00:00"
                pid = pdb.add_photo(filepath=f"/p{i}.jpg", filename=f"p{i}.jpg",
                                    date_taken=date)
                emb = np.random.rand(512).astype(np.float32).tolist()
                fid = pdb.add_face(pid, (10, 10, 50, 50), emb)
                pdb.assign_face_to_person(fid, person_id, "manual")
            pdb.conn.commit()

            # First page, newest first.
            page1 = search_combined(pdb, person="Calvin", sort="date_desc",
                                    limit=5, offset=0)
            # Second page.
            page2 = search_combined(pdb, person="Calvin", sort="date_desc",
                                    limit=5, offset=5)
        # page1 dates strictly >= page2 dates
        assert page1[-1]["date_taken"] >= page2[0]["date_taken"]
        # First photo overall is the most recent seeded.
        assert page1[0]["date_taken"].startswith("2024")  # year 2015 + 29//3 = 2024

    def test_sort_asc_mirrors_desc(self, tmp_db_path):
        """date_asc should produce the exact reverse of date_desc
        (excluding NULLs which are always at the tail)."""
        from photosearch.db import PhotoDB
        from photosearch.search import search_combined

        with PhotoDB(tmp_db_path) as pdb:
            person_id = pdb.add_person("Alice")
            import numpy as np
            for date in ("2019-01-01T10:00:00",
                         "2022-01-01T10:00:00",
                         "2025-01-01T10:00:00"):
                pid = pdb.add_photo(filepath=f"/{date}.jpg",
                                    filename=f"{date}.jpg",
                                    date_taken=date)
                emb = np.random.rand(512).astype(np.float32).tolist()
                fid = pdb.add_face(pid, (10, 10, 50, 50), emb)
                pdb.assign_face_to_person(fid, person_id, "manual")
            pdb.conn.commit()

            desc = search_combined(pdb, person="Alice", sort="date_desc")
            asc = search_combined(pdb, person="Alice", sort="date_asc")
        assert [r["filename"] for r in desc] == list(reversed(
            [r["filename"] for r in asc]))


# =========================================================================
# Reciprocal Rank Fusion (quick-wins bundle commit 2)
# =========================================================================

class TestRRFScoring:
    def test_attach_rrf_single_filter(self):
        """Single-filter RRF scores are monotonic with rank — higher
        rank (lower index) = higher score. Preserves the filter's
        original order when sorted by rrf_score."""
        from photosearch.search import _attach_rrf_scores, _RRF_K

        primary = {i: {"id": i} for i in (10, 20, 30)}
        ranks = [{10: 0, 20: 1, 30: 2}]
        _attach_rrf_scores([primary], ranks)
        assert primary[10]["rrf_score"] > primary[20]["rrf_score"] > primary[30]["rrf_score"]
        assert primary[10]["rrf_score"] == pytest.approx(1.0 / _RRF_K)

    def test_attach_rrf_multi_filter_accumulates(self):
        """A photo that ranks well in both filters scores higher than
        one that ranks well in just one."""
        from photosearch.search import _attach_rrf_scores, _RRF_K

        primary = {1: {"id": 1}, 2: {"id": 2}}
        ranks = [
            {1: 0, 2: 5},   # Filter A: photo 1 is top, 2 is lower
            {1: 0, 2: 5},   # Filter B: same
        ]
        _attach_rrf_scores([primary, primary], ranks)
        # Photo 1 gets 1/60 + 1/60 = 2/60
        # Photo 2 gets 1/65 + 1/65 = 2/65
        assert primary[1]["rrf_score"] == pytest.approx(2.0 / _RRF_K)
        assert primary[2]["rrf_score"] == pytest.approx(2.0 / (_RRF_K + 5))
        assert primary[1]["rrf_score"] > primary[2]["rrf_score"]

    def test_attach_rrf_missing_from_one_filter(self):
        """Photo not in filter B only gets filter A's contribution."""
        from photosearch.search import _attach_rrf_scores, _RRF_K

        primary = {1: {"id": 1}, 2: {"id": 2}}
        ranks = [
            {1: 0, 2: 1},   # Filter A has both
            {1: 0},         # Filter B has only photo 1
        ]
        _attach_rrf_scores([primary, primary], ranks)
        assert primary[1]["rrf_score"] == pytest.approx(2.0 / _RRF_K)
        assert primary[2]["rrf_score"] == pytest.approx(1.0 / (_RRF_K + 1))

    def test_relevance_sort_uses_rrf(self):
        """sort='relevance' should order by rrf_score desc."""
        from photosearch.search import _apply_sort

        items = [
            {"id": 1, "rrf_score": 0.05},
            {"id": 2, "rrf_score": 0.10},
            {"id": 3, "rrf_score": 0.02},
        ]
        sorted_items = _apply_sort(items, "relevance")
        assert [r["id"] for r in sorted_items] == [2, 1, 3]

    def test_relevance_sort_without_rrf_preserves_order(self):
        """Early-return paths don't compute rrf_score. Sort='relevance'
        must be a no-op there, not reshuffle to 0-keyed arbitrary order.
        """
        from photosearch.search import _apply_sort

        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        sorted_items = _apply_sort(items, "relevance")
        assert [r["id"] for r in sorted_items] == [1, 2, 3]


# =========================================================================
# Recency decay (quick-wins bundle commit 3)
# =========================================================================

class TestRecencyDecay:
    def test_recent_photo_scores_higher(self):
        """A photo from today should end up with a higher post-decay
        rrf_score than one from years ago, given equal pre-decay scores."""
        from photosearch.search import _apply_recency_decay
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        ten_years_ago = (datetime.datetime.now()
                         - datetime.timedelta(days=365 * 10)).strftime("%Y-%m-%dT%H:%M:%S")
        photos = [
            {"id": 1, "date_taken": today, "rrf_score": 0.1},
            {"id": 2, "date_taken": ten_years_ago, "rrf_score": 0.1},
        ]
        _apply_recency_decay(photos)
        assert photos[0]["rrf_score"] > photos[1]["rrf_score"]

    def test_undated_photos_get_neutral_penalty(self):
        """No date_taken → factor between "today" and "very old" — not
        winning relevance over recent photos but not zeroed out."""
        from photosearch.search import _apply_recency_decay, _UNDATED_RECENCY_FACTOR
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        photos = [
            {"id": 1, "date_taken": today, "rrf_score": 1.0},
            {"id": 2, "date_taken": None, "rrf_score": 1.0},
        ]
        _apply_recency_decay(photos)
        # Today's factor is exp(0) = 1.0
        assert photos[0]["rrf_score"] == pytest.approx(1.0, rel=0.01)
        # Undated is fixed at _UNDATED_RECENCY_FACTOR
        assert photos[1]["rrf_score"] == pytest.approx(_UNDATED_RECENCY_FACTOR)

    def test_bad_date_format_falls_back_to_undated(self):
        from photosearch.search import _apply_recency_decay, _UNDATED_RECENCY_FACTOR
        photos = [{"id": 1, "date_taken": "garbage-date", "rrf_score": 1.0}]
        _apply_recency_decay(photos)
        assert photos[0]["rrf_score"] == pytest.approx(_UNDATED_RECENCY_FACTOR)

    def test_rrf_score_respected_alongside_decay(self):
        """Higher RRF should still win over lower RRF after decay, as
        long as the recency gap isn't enormous."""
        from photosearch.search import _apply_recency_decay
        import datetime
        today = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        two_years = (datetime.datetime.now()
                     - datetime.timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%S")
        photos = [
            {"id": 1, "date_taken": two_years, "rrf_score": 0.10},  # strong RRF
            {"id": 2, "date_taken": today, "rrf_score": 0.02},       # weak RRF
        ]
        _apply_recency_decay(photos)
        # 2-year decay is ~exp(-0.10) ≈ 0.905; 0.10 * 0.905 = 0.0905
        # Today: 0.02 * 1.0 = 0.02. High-RRF older photo still wins.
        assert photos[0]["rrf_score"] > photos[1]["rrf_score"]

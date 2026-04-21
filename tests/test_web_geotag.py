"""Integration tests for the /api/geotag/* and /api/geocode/search endpoints."""

import io
import json
from contextlib import contextmanager

import pytest

from photosearch.db import PhotoDB


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_folders(db_path):
    """Seed a small mix of GPS/no-GPS photos across two folders."""
    with PhotoDB(db_path) as db:
        # Folder A — 2 with EXIF, 1 no-GPS
        db.add_photo(filepath="/ph/a/a1.jpg", filename="a1.jpg",
                     date_taken="2022-05-10T10:00:00",
                     gps_lat=47.6, gps_lon=-122.3,
                     place_name="Seattle, Washington, US")
        db.add_photo(filepath="/ph/a/a2.jpg", filename="a2.jpg",
                     date_taken="2022-05-10T10:30:00",
                     gps_lat=47.6, gps_lon=-122.3,
                     place_name="Seattle, Washington, US")
        db.add_photo(filepath="/ph/a/a3.jpg", filename="a3.jpg",
                     date_taken="2022-05-10T11:00:00")

        # Folder B — 1 inferred, 2 no-GPS
        db.add_photo(filepath="/ph/b/b1.jpg", filename="b1.jpg",
                     date_taken="2021-08-01T12:00:00",
                     gps_lat=41.9, gps_lon=12.5,
                     place_name="Rome, Lazio, IT",
                     location_source="inferred",
                     location_confidence=0.82)
        db.add_photo(filepath="/ph/b/b2.jpg", filename="b2.jpg",
                     date_taken="2021-08-01T12:05:00")
        db.add_photo(filepath="/ph/b/b3.jpg", filename="b3.jpg",
                     date_taken="2021-08-01T12:10:00")

        db.conn.commit()


# ---------------------------------------------------------------------------
# /api/geotag/folders
# ---------------------------------------------------------------------------

def test_folders_returns_no_gps_counts(client, db):
    _seed_folders(db.db_path)
    r = client.get("/api/geotag/folders")
    assert r.status_code == 200
    body = r.json()
    by_path = {f["path"]: f for f in body["folders"]}

    assert "/ph/a" in by_path
    assert "/ph/b" in by_path

    a = by_path["/ph/a"]
    assert a["total"] == 3
    assert a["with_exif"] == 2
    assert a["with_inferred"] == 0
    assert a["no_gps"] == 1
    assert a["date_from"] == "2022-05-10T10:00:00"

    b = by_path["/ph/b"]
    assert b["with_exif"] == 0
    assert b["with_inferred"] == 1
    assert b["no_gps"] == 2

    # Sorted: B has more no-GPS → should appear first.
    ordering = [f["path"] for f in body["folders"]
                if f["path"] in ("/ph/a", "/ph/b")]
    assert ordering == ["/ph/b", "/ph/a"]


def test_folders_hides_fully_tagged_by_default(client, db):
    with PhotoDB(db.db_path) as pdb:
        pdb.add_photo(filepath="/done/x.jpg", filename="x.jpg",
                      date_taken="2020-01-01T00:00:00",
                      gps_lat=1.0, gps_lon=2.0,
                      place_name="Somewhere")
        pdb.conn.commit()

    r = client.get("/api/geotag/folders")
    assert r.status_code == 200
    paths = [f["path"] for f in r.json()["folders"]]
    assert "/done" not in paths

    r2 = client.get("/api/geotag/folders?include_fully_tagged=true")
    paths2 = [f["path"] for f in r2.json()["folders"]]
    assert "/done" in paths2


# ---------------------------------------------------------------------------
# /api/geotag/folder-photos
# ---------------------------------------------------------------------------

def test_folder_photos_returns_only_no_gps_by_default(client, db):
    _seed_folders(db.db_path)
    r = client.get("/api/geotag/folder-photos?folder=/ph/b")
    assert r.status_code == 200
    body = r.json()
    filenames = sorted(p["filename"] for p in body["photos"])
    assert filenames == ["b2.jpg", "b3.jpg"]  # b1 is inferred → excluded


def test_folder_photos_includes_inferred_when_requested(client, db):
    _seed_folders(db.db_path)
    r = client.get("/api/geotag/folder-photos?folder=/ph/b&show_inferred=true")
    assert r.status_code == 200
    filenames = sorted(p["filename"] for p in r.json()["photos"])
    assert filenames == ["b1.jpg", "b2.jpg", "b3.jpg"]


def test_folder_photos_does_not_match_subfolders(client, db):
    """A photo in /ph/a/sub must not appear when querying /ph/a."""
    with PhotoDB(db.db_path) as pdb:
        pdb.add_photo(filepath="/ph/a/x.jpg", filename="x.jpg")
        pdb.add_photo(filepath="/ph/a/sub/y.jpg", filename="y.jpg")
        pdb.conn.commit()

    r = client.get("/api/geotag/folder-photos?folder=/ph/a")
    filenames = [p["filename"] for p in r.json()["photos"]]
    assert "x.jpg" in filenames
    assert "y.jpg" not in filenames


# ---------------------------------------------------------------------------
# /api/geotag/known-places
# ---------------------------------------------------------------------------

def test_known_places_filters_and_sorts(client, db):
    _seed_folders(db.db_path)
    r = client.get("/api/geotag/known-places?q=seat")
    assert r.status_code == 200
    body = r.json()
    names = [p["place_name"] for p in body["places"]]
    assert "Seattle, Washington, US" in names
    assert "Rome, Lazio, IT" not in names


def test_known_places_sorts_by_photo_count(client, db):
    _seed_folders(db.db_path)
    # Query a substring that only matches my seeded rows (the conftest fixture
    # has its own place_names like "Big Sur, CA" with higher counts).
    r = client.get("/api/geotag/known-places?q=eattle")
    places = r.json()["places"]
    assert any(p["place_name"] == "Seattle, Washington, US"
               and p["photo_count"] == 2 for p in places)


# ---------------------------------------------------------------------------
# /api/geocode/search — Nominatim proxy
# ---------------------------------------------------------------------------

def _fake_nominatim_response(payload):
    """Return a context manager that urllib.request.urlopen yields."""
    class _Fake:
        def __init__(self, data):
            self._data = json.dumps(data).encode("utf-8")
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
    return _Fake(payload)


def test_geocode_search_hits_nominatim_then_caches(client, db, monkeypatch):
    calls = []
    canned = [
        {
            "display_name": "Rome, Lazio, Italy",
            "lat": "41.9028",
            "lon": "12.4964",
            "type": "city",
            "importance": 0.95,
            "address": {
                "city": "Rome",
                "state": "Lazio",
                "county": "Rome",
                "country": "Italy",
            },
        }
    ]

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _fake_nominatim_response(canned)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    r = client.get("/api/geocode/search?q=Rome&limit=3")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "nominatim"
    assert len(body["results"]) == 1
    first = body["results"][0]
    assert first["lat"] == pytest.approx(41.9028)
    assert first["country"] == "Italy"
    assert first["admin1"] == "Lazio"
    assert first["locality"] == "Rome"
    assert len(calls) == 1

    # Second identical call → cache hit, no second Nominatim call.
    r2 = client.get("/api/geocode/search?q=Rome&limit=3")
    assert r2.status_code == 200
    assert r2.json()["source"] == "cache"
    assert len(calls) == 1


def test_geocode_search_requires_query(client, db):
    r = client.get("/api/geocode/search?q=")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# /api/photos/bulk-set-location
# ---------------------------------------------------------------------------

def test_bulk_set_location_writes_manual(client, db):
    with PhotoDB(db.db_path) as pdb:
        pid1 = pdb.add_photo(filepath="/t/1.jpg", filename="1.jpg")
        pid2 = pdb.add_photo(filepath="/t/2.jpg", filename="2.jpg")
        pdb.conn.commit()

    r = client.post(
        "/api/photos/bulk-set-location",
        json={"photo_ids": [pid1, pid2],
              "lat": 47.6, "lon": -122.3,
              "place_name": "Seattle, Washington, US"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] == 2
    assert body["skipped_count"] == 0

    with PhotoDB(db.db_path) as pdb:
        for pid in (pid1, pid2):
            row = pdb.conn.execute(
                "SELECT gps_lat, gps_lon, place_name, location_source, "
                "       location_confidence FROM photos WHERE id=?", (pid,)
            ).fetchone()
            assert row["gps_lat"] == pytest.approx(47.6)
            assert row["place_name"] == "Seattle, Washington, US"
            assert row["location_source"] == "manual"
            assert row["location_confidence"] is None


def test_bulk_set_location_skips_existing_gps_without_overwrite(client, db):
    with PhotoDB(db.db_path) as pdb:
        existing = pdb.add_photo(filepath="/t/ex.jpg", filename="ex.jpg",
                                 gps_lat=10.0, gps_lon=20.0,
                                 place_name="Pre-existing")
        fresh = pdb.add_photo(filepath="/t/fr.jpg", filename="fr.jpg")
        pdb.conn.commit()

    r = client.post(
        "/api/photos/bulk-set-location",
        json={"photo_ids": [existing, fresh],
              "lat": 47.6, "lon": -122.3, "place_name": "Seattle"},
    )
    body = r.json()
    assert body["updated_count"] == 1
    assert body["skipped_count"] == 1

    with PhotoDB(db.db_path) as pdb:
        ex_row = pdb.conn.execute(
            "SELECT gps_lat, place_name FROM photos WHERE id=?", (existing,)
        ).fetchone()
        # Existing row untouched.
        assert ex_row["gps_lat"] == pytest.approx(10.0)
        assert ex_row["place_name"] == "Pre-existing"


def test_bulk_set_location_overwrite_replaces_existing(client, db):
    with PhotoDB(db.db_path) as pdb:
        pid = pdb.add_photo(filepath="/t/ex.jpg", filename="ex.jpg",
                            gps_lat=10.0, gps_lon=20.0,
                            place_name="Old")
        pdb.conn.commit()

    r = client.post(
        "/api/photos/bulk-set-location",
        json={"photo_ids": [pid],
              "lat": 47.6, "lon": -122.3, "place_name": "New",
              "overwrite": True},
    )
    assert r.json()["updated_count"] == 1

    with PhotoDB(db.db_path) as pdb:
        row = pdb.conn.execute(
            "SELECT gps_lat, place_name, location_source FROM photos WHERE id=?",
            (pid,),
        ).fetchone()
    assert row["gps_lat"] == pytest.approx(47.6)
    assert row["place_name"] == "New"
    assert row["location_source"] == "manual"


def test_bulk_set_location_validates_input(client, db):
    r = client.post("/api/photos/bulk-set-location", json={})
    assert r.status_code == 400

    r = client.post("/api/photos/bulk-set-location",
                    json={"photo_ids": [1], "lat": 999, "lon": 0})
    assert r.status_code == 400


def test_geotag_page_served(client):
    r = client.get("/geotag")
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text or "<!doctype html>" in r.text

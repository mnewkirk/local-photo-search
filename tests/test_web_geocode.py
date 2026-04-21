"""Integration tests for /api/geocode/* endpoints."""

import pytest
from photosearch.db import PhotoDB


def _seed_inferrable(db_path):
    """Seed the DB so there's exactly one inferrable no-GPS photo."""
    with PhotoDB(db_path) as db:
        db.add_photo(filepath="/anchor.jpg", filename="anchor.jpg",
                     date_taken="2020-06-15T10:00:00",
                     gps_lat=47.6, gps_lon=-122.3)
        db.add_photo(filepath="/target.jpg", filename="target.jpg",
                     date_taken="2020-06-15T10:10:00")
        db.conn.commit()


def test_preview_returns_counts_and_samples(client, db):
    _seed_inferrable(db.db_path)
    r = client.post(
        "/api/geocode/infer-preview",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_count"] >= 1
    assert "confidence_buckets" in body
    assert "hop_distribution" in body
    assert len(body["samples"]) >= 1
    sample = body["samples"][0]
    for key in ("photo_id", "filepath", "inferred_lat", "inferred_lon",
                "confidence", "hop_count", "time_gap_min", "drift_km",
                "sides", "source_photo_id", "thumbnail_url"):
        assert key in sample


def test_apply_requires_confirm(client, db):
    _seed_inferrable(db.db_path)
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10},
    )
    assert r.status_code == 400
    assert "confirm" in r.text.lower()


def test_apply_writes_inferred_source(client, db, monkeypatch):
    _seed_inferrable(db.db_path)
    monkeypatch.setattr(
        "photosearch.geocode.reverse_geocode_batch",
        lambda coords: ["Seattle, Washington, US"] * len(coords),
    )
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10, "confirm": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] >= 1

    with PhotoDB(db.db_path) as pdb:
        row = pdb.conn.execute(
            "SELECT gps_lat, location_source, location_confidence "
            "FROM photos WHERE filename='target.jpg'"
        ).fetchone()
    assert row["location_source"] == "inferred"
    assert 0 < row["location_confidence"] <= 1.0
    assert row["gps_lat"] == pytest.approx(47.6)


def test_apply_untouched_when_no_candidates(client, db):
    """With min_confidence=0.99 no inference will pass the threshold, so
    the apply path should be a clean no-op even though the db fixture
    has no-GPS photos present."""
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.99, "cascade": True,
              "max_cascade_rounds": 10, "confirm": True},
    )
    assert r.status_code == 200
    assert r.json()["updated_count"] == 0

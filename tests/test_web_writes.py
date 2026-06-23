"""M26b write path — db.set_photo_location / db.set_photo_tags and the
bulk-set-location / bulk-set-tags endpoints (NAS-authoritative writes that
return canonical applied values for the replica mirror)."""

import json

import pytest

from photosearch.db import PhotoDB


# --- db methods ------------------------------------------------------------

def test_set_photo_location_respects_overwrite(db):
    a = db.add_photo(filepath="w/a.jpg", filename="a.jpg")
    b = db.add_photo(filepath="w/b.jpg", filename="b.jpg", gps_lat=1.0, gps_lon=2.0)
    db.conn.commit()
    assert db.set_photo_location(a, 47.6, -122.3, "Seattle") is True
    # b already has GPS → skipped without overwrite, written with it.
    assert db.set_photo_location(b, 47.6, -122.3, "Seattle") is False
    assert db.set_photo_location(b, 47.6, -122.3, "Seattle", overwrite=True) is True
    row = db.conn.execute("SELECT gps_lat, place_name, location_source FROM photos WHERE id=?", (a,)).fetchone()
    assert row["place_name"] == "Seattle"
    assert row["location_source"] == "manual"


def test_set_photo_tags_add_unions_and_dedupes(db):
    p = db.add_photo(filepath="w/t.jpg", filename="t.jpg")
    db.conn.execute("UPDATE photos SET categories=? WHERE id=?", (json.dumps(["beach"]), p))
    db.conn.commit()
    final = db.set_photo_tags(p, categories=["beach", "sunset"], mode="add")
    assert final["categories"] == ["beach", "sunset"]      # existing first, deduped
    assert final["keywords"] == []                          # untouched column preserved


def test_set_photo_tags_replace_overwrites(db):
    p = db.add_photo(filepath="w/r.jpg", filename="r.jpg")
    db.conn.execute("UPDATE photos SET keywords=? WHERE id=?", (json.dumps(["old"]), p))
    db.conn.commit()
    final = db.set_photo_tags(p, keywords=["new1", "new2"], mode="replace")
    assert final["keywords"] == ["new1", "new2"]


def test_set_photo_tags_logs_generation(db):
    p = db.add_photo(filepath="w/g.jpg", filename="g.jpg")
    db.conn.commit()
    db.set_photo_tags(p, categories=["x"], mode="add", log_model="manual")
    rows = db.conn.execute(
        "SELECT text_type, generated_text, model_used FROM generations WHERE photo_id=?", (p,)
    ).fetchall()
    assert any(r["text_type"] == "category-content" and json.loads(r["generated_text"]) == ["x"]
               and r["model_used"] == "manual" for r in rows)


def test_set_photo_tags_missing_photo_returns_none(db):
    assert db.set_photo_tags(999999, categories=["x"]) is None


# --- endpoints -------------------------------------------------------------

def test_bulk_set_location_returns_applied(client, db):
    a = db.add_photo(filepath="w/e1.jpg", filename="e1.jpg")
    db.conn.commit()
    r = client.post("/api/photos/bulk-set-location",
                    json={"photo_ids": [a], "lat": 47.6, "lon": -122.3,
                          "place_name": "Seattle, WA, US"})
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] == 1
    assert body["updated_ids"] == [a]
    assert body["applied"]["place_name"] == "Seattle, WA, US"
    assert body["applied"]["location_source"] == "manual"


def test_bulk_set_tags_endpoint(client, db):
    a = db.add_photo(filepath="w/e2.jpg", filename="e2.jpg")
    db.conn.commit()
    r = client.post("/api/photos/bulk-set-tags",
                    json={"photo_ids": [a], "keywords": ["hawaii", "beach"], "mode": "add"})
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] == 1
    assert body["results"][0]["keywords"] == ["hawaii", "beach"]
    # persisted
    row = db.conn.execute("SELECT keywords FROM photos WHERE id=?", (a,)).fetchone()
    assert json.loads(row["keywords"]) == ["hawaii", "beach"]


def test_bulk_set_tags_requires_a_column(client, db):
    a = db.add_photo(filepath="w/e3.jpg", filename="e3.jpg")
    db.conn.commit()
    r = client.post("/api/photos/bulk-set-tags", json={"photo_ids": [a]})
    assert r.status_code == 400


def test_collection_add_photos_resolves_existing(client, db):
    a = db.add_photo(filepath="w/c1.jpg", filename="c1.jpg")
    db.conn.commit()
    r = client.post("/api/collections/add-photos",
                    json={"collection": "Best of March", "photo_ids": [a]})
    assert r.status_code == 200
    body = r.json()
    assert body["created"] is False
    assert body["added"] == 1
    assert body["collection"]["name"] == "Best of March"


def test_collection_add_photos_missing_needs_create(client, db):
    a = db.add_photo(filepath="w/c2.jpg", filename="c2.jpg")
    db.conn.commit()
    r = client.post("/api/collections/add-photos",
                    json={"collection": "Nope", "photo_ids": [a]})
    assert r.status_code == 404
    r = client.post("/api/collections/add-photos",
                    json={"collection": "Fresh Album", "photo_ids": [a], "create": True})
    assert r.status_code == 200
    body = r.json()
    assert body["created"] is True
    assert body["collection"]["id"] and body["collection"]["name"] == "Fresh Album"

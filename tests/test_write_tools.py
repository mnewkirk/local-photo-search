"""M26b write tools (photosearch/tools.py) — set_photo_location / set_photo_tags.

Covers the guardrails (explicit id-set, dry-run-by-default, affected-count cap),
the local-authoritative path (no PHOTOSEARCH_NAS_URL), and the dual-write +
mirror path (NAS configured → POST mocked → values mirrored into the local DB).
"""

import json

import pytest

from photosearch import tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ids(db, *names):
    return [db._test_photo_ids[n] for n in names]


@pytest.fixture
def local_writer(monkeypatch):
    """No NAS configured → the local DB is the authoritative writer."""
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_WRITE_MAX_ROWS", raising=False)


# ---------------------------------------------------------------------------
# Guardrails — scoping + dry-run
# ---------------------------------------------------------------------------

def test_location_requires_photo_ids(db, local_writer):
    out = tools.call_tool(db, "set_photo_location", {"lat": 1.0, "lon": 2.0})
    assert "error" in out and "photo_ids" in out["error"]


def test_location_dry_run_writes_nothing(db, local_writer):
    pid = _ids(db, "DSC04880.JPG")[0]  # no GPS in the fixture
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [pid], "lat": 47.6, "lon": -122.3,
                           "place_name": "Seattle"})
    assert out["dry_run"] is True
    assert out["affected_count"] == 1
    assert out["sample"][0]["after"]["place_name"] == "Seattle"
    # nothing persisted
    row = db.conn.execute("SELECT gps_lat FROM photos WHERE id=?", (pid,)).fetchone()
    assert row["gps_lat"] is None


def test_location_skips_existing_gps_in_count(db, local_writer):
    has_gps = _ids(db, "DSC04878.JPG")[0]   # fixture gives this one GPS
    no_gps = _ids(db, "DSC04880.JPG")[0]
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [has_gps, no_gps], "lat": 47.6, "lon": -122.3})
    assert out["affected_count"] == 1          # only the no-GPS one
    assert out["skipped_existing_gps"] == 1


def test_location_confirm_writes_local(db, local_writer):
    pid = _ids(db, "DSC04880.JPG")[0]
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [pid], "lat": 47.6, "lon": -122.3,
                           "place_name": "Seattle", "confirm": True})
    assert out["authority"] == "local"
    assert out["updated_count"] == 1
    row = db.conn.execute(
        "SELECT gps_lat, place_name, location_source FROM photos WHERE id=?",
        (pid,)).fetchone()
    assert row["gps_lat"] == 47.6
    assert row["place_name"] == "Seattle"
    assert row["location_source"] == "manual"


def test_location_overwrite_guard(db, local_writer):
    has_gps = _ids(db, "DSC04878.JPG")[0]
    # Without overwrite → skipped.
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [has_gps], "lat": 1.0, "lon": 2.0,
                           "confirm": True})
    assert out["updated_count"] == 0
    # With overwrite → applied.
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [has_gps], "lat": 1.0, "lon": 2.0,
                           "overwrite": True, "confirm": True})
    assert out["updated_count"] == 1


def test_location_cap_blocks_large_confirm(db, local_writer, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_WRITE_MAX_ROWS", "1")
    ids = _ids(db, "DSC04880.JPG", "DSC04894.JPG")  # both lack GPS... 894 has none
    # Ensure both are GPS-less so both count.
    db.conn.execute("UPDATE photos SET gps_lat=NULL, gps_lon=NULL WHERE id IN (?,?)", ids)
    db.conn.commit()
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": ids, "lat": 1.0, "lon": 2.0, "confirm": True})
    assert "error" in out and out["needs"] == "confirm_large=true"
    # second acknowledgement lets it through
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": ids, "lat": 1.0, "lon": 2.0,
                           "confirm": True, "confirm_large": True})
    assert out["updated_count"] == 2


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def test_tags_requires_a_column(db, local_writer):
    pid = _ids(db, "DSC04880.JPG")[0]
    out = tools.call_tool(db, "set_photo_tags", {"photo_ids": [pid]})
    assert "error" in out


def test_tags_dry_run_shows_merge(db, local_writer):
    pid = _ids(db, "DSC04922.JPG")[0]  # categories=["landscape","people"]
    out = tools.call_tool(db, "set_photo_tags",
                          {"photo_ids": [pid], "categories": ["people", "sunset"]})
    assert out["dry_run"] is True
    assert out["affected_count"] == 1
    after = out["sample"][0]["after"]["categories"]
    assert after == ["landscape", "people", "sunset"]   # union, existing-first
    # not persisted
    row = db.conn.execute("SELECT categories FROM photos WHERE id=?", (pid,)).fetchone()
    assert json.loads(row["categories"]) == ["landscape", "people"]


def test_tags_unchanged_not_counted(db, local_writer):
    pid = _ids(db, "DSC04922.JPG")[0]
    # Adding a tag it already has → no change.
    out = tools.call_tool(db, "set_photo_tags",
                          {"photo_ids": [pid], "categories": ["landscape"]})
    assert out["affected_count"] == 0


def test_tags_confirm_writes_and_logs(db, local_writer):
    pid = _ids(db, "DSC04880.JPG")[0]
    out = tools.call_tool(db, "set_photo_tags",
                          {"photo_ids": [pid], "keywords": ["hawaii"],
                           "confirm": True})
    assert out["authority"] == "local"
    row = db.conn.execute("SELECT keywords FROM photos WHERE id=?", (pid,)).fetchone()
    assert json.loads(row["keywords"]) == ["hawaii"]
    # provenance logged (model defaults to 'manual')
    gen = db.conn.execute(
        "SELECT model_used FROM generations WHERE photo_id=? AND text_type='keywords'",
        (pid,)).fetchone()
    assert gen["model_used"] == "manual"


def test_tags_replace_mode(db, local_writer):
    pid = _ids(db, "DSC04922.JPG")[0]
    out = tools.call_tool(db, "set_photo_tags",
                          {"photo_ids": [pid], "categories": ["new"],
                           "mode": "replace", "confirm": True})
    row = db.conn.execute("SELECT categories FROM photos WHERE id=?", (pid,)).fetchone()
    assert json.loads(row["categories"]) == ["new"]


# ---------------------------------------------------------------------------
# Dual-write + mirror (NAS configured)
# ---------------------------------------------------------------------------

def test_location_dual_write_mirrors_nas_values(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.local:8000")
    pid = _ids(db, "DSC04880.JPG")[0]

    captured = {}

    def fake_post(path, body, timeout=60.0):
        captured["path"] = path
        captured["body"] = body
        # NAS canonicalizes the place label and reports what it wrote.
        return {"updated_count": 1, "skipped_count": 0, "updated_ids": [pid],
                "applied": {"gps_lat": 47.6, "gps_lon": -122.3,
                            "place_name": "Seattle, WA, US",
                            "location_source": "manual"}}

    monkeypatch.setattr(tools, "_nas_post", fake_post)
    out = tools.call_tool(db, "set_photo_location",
                          {"photo_ids": [pid], "lat": 47.6, "lon": -122.3,
                           "place_name": "Seattle", "confirm": True})
    assert captured["path"] == "/api/photos/bulk-set-location"
    assert out["authority"] == "nas"
    assert out["mirrored"] == 1
    # The LOCAL replica now holds the NAS's canonical place label byte-for-byte.
    row = db.conn.execute("SELECT place_name, location_source FROM photos WHERE id=?",
                          (pid,)).fetchone()
    assert row["place_name"] == "Seattle, WA, US"
    assert row["location_source"] == "manual"


# ---------------------------------------------------------------------------
# add_to_collection
# ---------------------------------------------------------------------------

def test_collection_dry_run_existing(db, local_writer):
    # Fixture has "Best of March" containing 04907 + 04922.
    coll = db.get_collection_by_name("Best of March")
    ids = _ids(db, "DSC04907.JPG", "DSC04878.JPG")  # 04907 already in, 04878 new
    out = tools.call_tool(db, "add_to_collection",
                          {"photo_ids": ids, "collection": "Best of March"})
    assert out["dry_run"] is True
    assert out["collection_exists"] is True
    assert out["affected_count"] == 1            # only 04878 is new
    assert out["already_in_collection"] == 1
    # nothing written
    assert set(db.get_collection_photo_ids(coll["id"])) == \
        {db._test_photo_ids["DSC04907.JPG"], db._test_photo_ids["DSC04922.JPG"]}


def test_collection_dry_run_missing_needs_create(db, local_writer):
    ids = _ids(db, "DSC04878.JPG")
    out = tools.call_tool(db, "add_to_collection",
                          {"photo_ids": ids, "collection": "Brand New"})
    assert out["dry_run"] is True
    assert out["collection_exists"] is False
    assert out["would_create"] is True
    assert "create=true" in out["note"]


def test_collection_confirm_without_create_errors(db, local_writer):
    ids = _ids(db, "DSC04878.JPG")
    out = tools.call_tool(db, "add_to_collection",
                          {"photo_ids": ids, "collection": "Brand New", "confirm": True})
    assert "error" in out and "create=true" in out["error"]


def test_collection_create_and_add_local(db, local_writer):
    ids = _ids(db, "DSC04878.JPG", "DSC04880.JPG")
    out = tools.call_tool(db, "add_to_collection",
                          {"photo_ids": ids, "collection": "Trip 2026",
                           "create": True, "confirm": True})
    assert out["authority"] == "local"
    assert out["created"] is True
    assert out["added"] == 2
    coll = db.get_collection_by_name("Trip 2026")
    assert set(db.get_collection_photo_ids(coll["id"])) == set(ids)


def test_collection_dual_write_mirrors_same_id(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.local:8000")
    ids = _ids(db, "DSC04878.JPG", "DSC04880.JPG")

    captured = {}

    def fake_post(path, body, timeout=60.0):
        captured["path"] = path
        captured["body"] = body
        # NAS created a collection and assigned id 777.
        return {"collection": {"id": 777, "name": "Hawaii", "description": None},
                "added": 2, "created": True}

    monkeypatch.setattr(tools, "_nas_post", fake_post)
    out = tools.call_tool(db, "add_to_collection",
                          {"photo_ids": ids, "collection": "Hawaii",
                           "create": True, "confirm": True})
    assert captured["path"] == "/api/collections/add-photos"
    assert captured["body"]["create"] is True
    assert out["authority"] == "nas"
    # Mirror re-created the collection under the SAME id 777 with the same photos.
    local = db.get_collection(777)
    assert local is not None and local["name"] == "Hawaii"
    assert set(db.get_collection_photo_ids(777)) == set(ids)


def test_tags_dual_write_mirrors_without_double_logging(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.local:8000")
    pid = _ids(db, "DSC04880.JPG")[0]

    def fake_post(path, body, timeout=60.0):
        # NAS applies the merge and returns the full canonical tag state.
        return {"updated_count": 1, "mode": "add",
                "results": [{"id": pid, "categories": [], "visual_tags": [],
                             "keywords": ["hawaii", "beach"]}]}

    monkeypatch.setattr(tools, "_nas_post", fake_post)
    out = tools.call_tool(db, "set_photo_tags",
                          {"photo_ids": [pid], "keywords": ["hawaii", "beach"],
                           "confirm": True})
    assert out["authority"] == "nas"
    row = db.conn.execute("SELECT keywords FROM photos WHERE id=?", (pid,)).fetchone()
    assert json.loads(row["keywords"]) == ["hawaii", "beach"]
    # Mirror must NOT add a local provenance row (the NAS already logged it).
    gen = db.conn.execute(
        "SELECT COUNT(*) c FROM generations WHERE photo_id=?", (pid,)).fetchone()
    assert gen["c"] == 0

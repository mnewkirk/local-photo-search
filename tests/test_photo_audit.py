"""The per-photo audit log (photosearch/audit.py, schema v40).

Origin: a photo was removed from a collection by hand, the collection was
rebuilt from a stale list, and nothing could say which photo had been removed.
"""

import sqlite3

import pytest

from photosearch import audit
from photosearch.faces import REJECTED_MATCH_SOURCE


def events(db, **kw):
    return audit.query(db.conn, **kw)


def pid(db, name):
    return db._test_photo_ids[name]


@pytest.fixture
def clean(db):
    """The fixture's own setup writes rows; start each test from none."""
    db.conn.execute("DELETE FROM photo_audit")
    db.conn.commit()
    return db


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def test_v39_db_gains_the_audit_table_and_trigger(tmp_path):
    from photosearch.db import PhotoDB
    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_info (key, value) VALUES ('version', '39')")
    conn.commit()
    conn.close()
    with PhotoDB(path) as db:
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(photo_audit)")}
        assert {"ts", "photo_id", "action", "field", "old_value", "new_value",
                "collection_id", "collection_name", "face_id", "detail",
                "source", "intent"} <= cols
        names = {r["name"] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE '%photo_audit%'")}
        assert {"idx_photo_audit_photo", "idx_photo_audit_collection",
                "trg_photo_audit_delete"} <= names


# --------------------------------------------------------------------------
# Collections — the incident this exists for
# --------------------------------------------------------------------------

def test_add_logs_only_photos_that_were_actually_added(clean):
    db = clean
    c = db.create_collection("Soccer")
    a, b = pid(db, "DSC04894.JPG"), pid(db, "DSC04907.JPG")
    db.add_photos_to_collection(c, [a])
    db.add_photos_to_collection(c, [a, b])        # a is already in
    rows = events(db, collection_id=c)
    assert sorted((r["photo_id"], r["action"]) for r in rows) == [
        (a, "collection_added"), (b, "collection_added")]
    assert {r["collection_name"] for r in rows} == {"Soccer"}


def test_removal_is_logged_and_only_for_actual_members(clean):
    db = clean
    c = db.create_collection("Soccer")
    a, b = pid(db, "DSC04894.JPG"), pid(db, "DSC04907.JPG")
    db.add_photos_to_collection(c, [a])
    db.remove_photos_from_collection(c, [a, b, 424242])
    removed = events(db, collection_id=c, action=audit.COLLECTION_REMOVED)
    assert [r["photo_id"] for r in removed] == [a]
    assert events(db, photo_id=b) == []


def test_which_photo_is_no_longer_in_the_collection(clean):
    """Remove one by hand, then 'rebuild' with add: the removed photo is still
    named once it has been put back and taken out again; one put back is not."""
    db = clean
    c = db.create_collection("Soccer")
    a, b, d = (pid(db, n) for n in ("DSC04894.JPG", "DSC04907.JPG", "DSC04922.JPG"))
    db.add_photos_to_collection(c, [a, b, d])
    db.remove_photos_from_collection(c, [b])
    db.remove_photos_from_collection(c, [d])
    db.add_photos_to_collection(c, [d])
    gone = audit.removed_from_collection(db.conn, c)
    assert [r["photo_id"] for r in gone] == [b]


def test_deleting_a_collection_logs_a_removal_for_every_member(clean):
    db = clean
    c = db.create_collection("Soccer")
    ids = [pid(db, n) for n in ("DSC04894.JPG", "DSC04907.JPG", "DSC04922.JPG")]
    db.add_photos_to_collection(c, ids)
    db.delete_collection(c)
    rows = events(db, collection_id=c, action=audit.COLLECTION_REMOVED)
    assert sorted(r["photo_id"] for r in rows) == sorted(ids)
    assert {r["field"] for r in rows} == {"collection_deleted"}
    assert {r["collection_name"] for r in rows} == {"Soccer"}
    # ...and the rebuilt-under-a-new-id case is answerable by name.
    c2 = db.create_collection("Soccer")
    db.add_photos_to_collection(c2, ids[:1])
    by_name = events(db, collection_name="soccer")
    assert {r["collection_id"] for r in by_name} == {c, c2}


def test_a_deleted_photo_leaves_its_collections_with_a_reason(clean):
    """ON DELETE CASCADE removes collection_photos without any method call."""
    db = clean
    c = db.create_collection("Soccer")
    a = pid(db, "DSC04894.JPG")
    db.add_photos_to_collection(c, [a])
    db.conn.execute("DELETE FROM photos WHERE id = ?", (a,))
    db.conn.commit()
    rows = events(db, photo_id=a)
    acts = {(r["action"], r["field"], r["source"]) for r in rows}
    assert ("collection_removed", "photo_deleted", "db-trigger") in acts
    assert ("photo_deleted", "filepath", "db-trigger") in acts
    assert [r["photo_id"] for r in audit.removed_from_collection(db.conn, c)] == [a]


def test_the_audit_row_shares_the_changes_transaction(clean):
    db = clean
    c = db.create_collection("Soccer")
    db.conn.commit()
    a = pid(db, "DSC04894.JPG")
    db.begin_batch(50)               # defer commits like add_photo's batch mode
    db.add_photos_to_collection(c, [a])   # this one commits itself...
    db.conn.execute("BEGIN")
    db.conn.execute("DELETE FROM collection_photos WHERE collection_id = ?", (c,))
    audit.record(db.conn, a, audit.COLLECTION_REMOVED, collection_id=c)
    db.conn.rollback()
    db.end_batch()
    assert [r["action"] for r in events(db, photo_id=a)] == ["collection_added"]


# --------------------------------------------------------------------------
# Photo fields
# --------------------------------------------------------------------------

def test_new_photo_is_logged_once(clean):
    db = clean
    p = db.add_photo(filepath="/photos/2026/new.jpg", filename="new.jpg")
    db.add_photo(filepath="/photos/2026/new.jpg", filename="new.jpg")   # ignored
    rows = events(db, photo_id=p)
    assert [(r["action"], r["new_value"]) for r in rows] == [
        ("photo_added", "/photos/2026/new.jpg")]


def test_filepath_move_logs_old_and_new_through_both_writers(clean):
    from photosearch.db import set_photo_filepath
    db = clean
    p = db.add_photo(filepath="/photos/2026/a.jpg", filename="a.jpg")
    db.conn.execute("DELETE FROM photo_audit")
    set_photo_filepath(db.conn, p, "/photos/2026/b.jpg")
    set_photo_filepath(db.conn, p, "/photos/2026/b.jpg")        # no change, no row
    db.update_photo(p, filepath="/photos/2026/c.jpg", description="irrelevant")
    db.conn.commit()
    rows = list(reversed(events(db, photo_id=p)))
    assert [(r["action"], r["old_value"], r["new_value"]) for r in rows] == [
        ("filepath_changed", "/photos/2026/a.jpg", "/photos/2026/b.jpg"),
        ("filepath_changed", "/photos/2026/b.jpg", "/photos/2026/c.jpg")]


def test_update_photo_ignores_unwatched_columns_and_logs_watched_ones(clean):
    db = clean
    p = pid(db, "DSC04894.JPG")
    db.update_photo(p, description="a new description", aesthetic_score=7.0)
    assert events(db, photo_id=p) == []
    old = db.get_photo(p)["date_taken"]
    db.update_photo(p, date_taken="2020-01-01 10:00:00")
    (row,) = events(db, photo_id=p)
    assert (row["action"], row["field"], row["old_value"], row["new_value"]) == (
        "field_changed", "date_taken", old, "2020-01-01 10:00:00")


def test_location_and_tags_log_before_and_after(clean):
    import json
    db = clean
    p = db.add_photo(filepath="/photos/2026/loc.jpg", filename="loc.jpg")
    db.conn.execute("DELETE FROM photo_audit")
    assert db.set_photo_location(p, 1.5, 2.5, "Somewhere")
    assert not db.set_photo_location(p, 9.0, 9.0, "Elsewhere")      # located: refused
    db.set_photo_tags(p, keywords=["goal"])
    db.set_photo_tags(p, keywords=["goal"])                          # no-op: no row
    db.set_photo_tags(p, keywords=["save"], mode="replace")
    rows = list(reversed(events(db, photo_id=p)))
    assert [r["action"] for r in rows] == ["location_set", "tags_set", "tags_set"]
    loc = rows[0]
    assert json.loads(loc["old_value"])["gps_lat"] is None
    assert json.loads(loc["new_value"]) == {
        "gps_lat": 1.5, "gps_lon": 2.5, "place_name": "Somewhere",
        "location_source": "manual"}
    assert (json.loads(rows[1]["old_value"]), json.loads(rows[1]["new_value"])) == ([], ["goal"])
    assert (json.loads(rows[2]["old_value"]), json.loads(rows[2]["new_value"])) == (["goal"], ["save"])


# --------------------------------------------------------------------------
# Face labels
# --------------------------------------------------------------------------

def test_assign_clear_and_dedupe_unmatch_are_distinct_actions(clean):
    from photosearch.db import unmatch_faces_as_duplicates
    db = clean
    face = db._test_face_ids["alex_894"]          # Alex, strict
    photo = pid(db, "DSC04894.JPG")
    sam = db._test_person_ids["Sam"]
    db.assign_face_to_person(face, sam, "manual")
    db.set_face_labels([face], None, REJECTED_MATCH_SOURCE)
    db.assign_face_to_person(face, sam, "temporal")
    unmatch_faces_as_duplicates(db.conn, [face])
    db.conn.commit()
    rows = list(reversed(events(db, photo_id=photo)))
    assert [(r["action"], r["old_value"], r["new_value"], r["face_id"]) for r in rows] == [
        ("face_assigned", "Alex", "Sam", face),
        ("face_cleared", "Sam", None, face),
        ("face_assigned", None, "Sam", face),
        ("face_unmatched", "Sam", None, face)]
    assert "rejected" in rows[1]["detail"]


def test_relabelling_the_same_person_same_source_is_not_a_change(clean):
    db = clean
    face = db._test_face_ids["alex_922"]          # Alex, manual
    db.assign_face_to_person(face, db._test_person_ids["Alex"], "manual")
    assert events(db, photo_id=pid(db, "DSC04922.JPG")) == []


# --------------------------------------------------------------------------
# Who asked
# --------------------------------------------------------------------------

def test_source_and_intent_come_from_the_context(clean):
    db = clean
    c = db.create_collection("Soccer")
    a = pid(db, "DSC04894.JPG")
    db.add_photos_to_collection(c, [a])
    with audit.context(source="claude", intent="restore the photo I removed"):
        db.remove_photos_from_collection(c, [a])
    with audit.tool_context("add_to_collection", "put it back", source="claude-mcp"):
        db.add_photos_to_collection(c, [a])
    rows = list(reversed(events(db, photo_id=a)))
    assert [(r["source"], r["intent"]) for r in rows] == [
        ("internal", None),
        ("claude", "restore the photo I removed"),
        ("claude-mcp", "tool add_to_collection: put it back")]


def test_replica_mirror_writes_are_not_logged(clean, monkeypatch):
    db = clean
    c = db.create_collection("Soccer")
    a = pid(db, "DSC04894.JPG")
    with audit.suppressed():
        db.add_photos_to_collection(c, [a])
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.example")
    db.remove_photos_from_collection(c, [a])
    assert events(db, photo_id=a) == []


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def test_api_records_the_callers_source_and_intent(client, db):
    db.conn.execute("DELETE FROM photo_audit")
    db.conn.commit()
    c = db._test_collection_id
    a = db._test_photo_ids["DSC04907.JPG"]
    r = client.post(f"/api/collections/{c}/photos/remove", json={"photo_ids": [a]},
                    headers={"X-Photosearch-Source": "claude",
                             "X-Photosearch-Intent": "Matt removed this by hand"})
    assert r.status_code == 200
    body = client.get(f"/api/photos/{a}/audit").json()
    (ev,) = body["events"]
    assert (ev["action"], ev["collection_id"], ev["source"], ev["intent"]) == (
        "collection_removed", c, "claude", "Matt removed this by hand")

    # A browser (Referer) is the UI, with an inferred intent.
    client.post(f"/api/collections/{c}/photos", json={"photo_ids": [a]},
                headers={"Referer": "http://x/collections"})
    top = client.get(f"/api/photos/{a}/audit").json()["events"][0]
    assert (top["action"], top["source"]) == ("collection_added", "ui")
    assert top["intent"]


def test_collection_audit_endpoint_lists_removed_photos(client, db):
    c = db._test_collection_id
    a = db._test_photo_ids["DSC04907.JPG"]
    client.post(f"/api/collections/{c}/photos/remove", json={"photo_ids": [a]})
    data = client.get(f"/api/collections/{c}/audit").json()
    assert data["collection"]["name"] == "Best of March"
    assert [r["photo_id"] for r in data["removed"]] == [a]
    # After the collection is deleted the answer is still there.
    client.delete(f"/api/collections/{c}")
    data = client.get(f"/api/collections/{c}/audit").json()
    assert data["collection"] is None
    assert sorted(r["photo_id"] for r in data["removed"]) == sorted(
        [a, db._test_photo_ids["DSC04922.JPG"]])


def test_general_audit_endpoint_filters(client, db):
    c = db._test_collection_id
    a = db._test_photo_ids["DSC04907.JPG"]
    client.post(f"/api/collections/{c}/photos/remove", json={"photo_ids": [a]})
    ev = client.get("/api/audit", params={
        "collection_name": "best of march", "action": "collection_removed"}).json()["events"]
    assert [e["photo_id"] for e in ev] == [a]
    assert client.get("/api/audit", params={"since": "2999-01-01"}).json()["events"] == []


def test_face_clear_through_the_api_is_a_rejection(client, db):
    face = db._test_face_ids["alex_894"]
    client.post(f"/api/faces/{face}/clear")
    ev = client.get(f"/api/photos/{db._test_photo_ids['DSC04894.JPG']}/audit",
                    params={"action": "face_cleared"}).json()["events"]
    assert [(e["old_value"], e["face_id"]) for e in ev] == [("Alex", face)]


def test_bulk_set_location_and_tags_are_logged_with_the_request(client, db):
    a = db._test_photo_ids["DSC04880.JPG"]
    client.post("/api/photos/bulk-set-location",
                json={"photo_ids": [a], "lat": 10.0, "lon": 20.0, "place_name": "Pitch",
                      "overwrite": True},
                headers={"X-Photosearch-Intent": "label the pitch"})
    client.post("/api/photos/bulk-set-tags", json={"photo_ids": [a], "keywords": ["pitch"]})
    ev = client.get(f"/api/photos/{a}/audit").json()["events"]
    assert {e["action"] for e in ev} >= {"location_set", "tags_set"}
    loc = next(e for e in ev if e["action"] == "location_set")
    assert loc["intent"] == "label the pitch"


def test_replica_reads_the_nas_log_and_says_so_when_it_cannot(client, db, monkeypatch):
    from photosearch import web
    a = db._test_photo_ids["DSC04880.JPG"]
    monkeypatch.setattr(web, "_nas_url", "http://nas.example")
    seen = []

    def fake(method, path, body=None, timeout=120.0):
        seen.append(path)
        return {"photo_id": a, "events": [{"id": 1, "action": "from_nas"}]}
    monkeypatch.setattr(web, "_nas_json", fake)
    assert client.get(f"/api/photos/{a}/audit").json()["events"][0]["action"] == "from_nas"
    assert seen == [f"/api/photos/{a}/audit?limit=200"]

    def down(*a, **k):
        raise OSError("no route")
    monkeypatch.setattr(web, "_nas_json", down)
    out = client.get(f"/api/photos/{a}/audit").json()
    assert out["stale"] is True and "events" in out


def test_replica_requests_do_not_write_audit_rows(client, db, monkeypatch):
    from photosearch import web
    monkeypatch.setattr(web, "_nas_url", "http://nas.example")
    monkeypatch.setattr(web, "_nas_json", lambda *a, **k: {})
    c = db._test_collection_id
    before = db.conn.execute("SELECT COUNT(*) FROM photo_audit").fetchone()[0]
    client.post(f"/api/collections/{c}/photos",
                json={"photo_ids": [db._test_photo_ids["DSC04880.JPG"]]})
    assert db.conn.execute("SELECT COUNT(*) FROM photo_audit").fetchone()[0] == before


# --------------------------------------------------------------------------
# Tool + rules
# --------------------------------------------------------------------------

def test_write_tools_are_attributed_to_the_tool(clean):
    from photosearch import tools
    db = clean
    a = pid(db, "DSC04894.JPG")
    with audit.context(source="claude-mcp", intent="soccer album"):
        tools.call_tool(db, "add_to_collection", {
            "collection": "Soccer", "create": True, "confirm": True,
            "photo_ids": [a]})
    (ev,) = events(db, photo_id=a, action="collection_added")
    assert ev["source"] == "claude-mcp"
    assert ev["intent"] == "tool add_to_collection: soccer album"


def test_every_audit_endpoint_has_an_intent_rule():
    from photosearch.request_intent import infer_intent
    for path in ("/api/photos/12/audit", "/api/collections/3/audit", "/api/audit"):
        text = infer_intent("GET", path, "")
        assert "{id}" not in text and not text.startswith("GET "), (path, text)


def test_the_cli_group_sets_its_context_and_does_not_leak_it():
    from click.testing import CliRunner
    import cli as cli_mod
    seen = {}

    @cli_mod.cli.command("audit-probe")
    def _probe():
        seen.update(audit.current())

    try:
        res = CliRunner().invoke(cli_mod.cli, ["audit-probe"])
        assert res.exit_code == 0, res.output
    finally:
        cli_mod.cli.commands.pop("audit-probe", None)
    assert (seen["source"], seen["intent"]) == ("cli", "photosearch audit-probe")
    assert audit.current()["source"] == "internal"

"""Person attributes (schema v42, docs/plans/description-names.md Revision 2 C):
gender / birth_year / birth_year_approx on persons, the API that sets them, and
the replica's NAS-authoritative write."""

import datetime
import sqlite3

import pytest
from fastapi import HTTPException

from photosearch.db import PhotoDB, SCHEMA_VERSION, validate_person_attributes


def _person(db, name):
    pid = db.add_person(name)
    db.conn.commit()
    return pid


def _attrs(db, pid):
    r = db.conn.execute("SELECT gender, birth_year, birth_year_approx FROM persons "
                        "WHERE id = ?", (pid,)).fetchone()
    return tuple(r)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_v41_db_migrates_to_v42_person_attributes(tmp_path):
    path = str(tmp_path / "v41.db")
    with PhotoDB(path) as db:
        pid = _person(db, "Asa")
    conn = sqlite3.connect(path)
    for col in ("birth_year_approx", "birth_year", "gender"):
        conn.execute(f"ALTER TABLE persons DROP COLUMN {col}")
    conn.execute("UPDATE schema_info SET value = '41' WHERE key = 'version'")
    conn.commit()
    conn.close()
    with PhotoDB(path) as db:
        assert _attrs(db, pid) == (None, None, 0)          # nothing backfilled
        version = db.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'").fetchone()["value"]
        assert int(version) == SCHEMA_VERSION == 43
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute("UPDATE persons SET gender = 'boy' WHERE id = ?", (pid,))
        db.conn.execute("UPDATE persons SET gender = 'male' WHERE id = ?", (pid,))
        assert _attrs(db, pid)[0] == "male"


def test_a_fresh_db_has_the_columns(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(persons)")}
    assert {"gender", "birth_year", "birth_year_approx"} <= cols


# ---------------------------------------------------------------------------
# Validation + DB write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fields,err", [
    ({"gender": "boy"}, "gender"),
    ({"birth_year": 1850}, "between"),
    ({"birth_year": datetime.date.today().year + 1}, "between"),
    ({"birth_year": "2014"}, "integer"),
    ({"birth_year": True}, "integer"),
    ({"birth_year_approx": 2}, "approx"),
    ({"nickname": "x"}, "unknown"),
])
def test_bad_values_are_refused(fields, err):
    with pytest.raises(ValueError, match=err):
        validate_person_attributes(fields)


def test_absent_is_unchanged_and_null_clears(db):
    pid = _person(db, "Asa")
    db.set_person_attributes([pid], {"gender": "male", "birth_year": 2015,
                                     "birth_year_approx": True})
    assert _attrs(db, pid) == ("male", 2015, 1)
    db.set_person_attributes([pid], {"gender": None})           # birth year untouched
    assert _attrs(db, pid) == (None, 2015, 1)
    db.set_person_attributes([pid], {"birth_year": None})       # approx resets with it
    assert _attrs(db, pid) == (None, None, 0)


def test_unknown_ids_are_skipped_and_reported(db):
    pid = _person(db, "Asa")
    out = db.set_person_attributes([pid, 987654], {"gender": "male"})
    assert out["updated"] == 1 and out["unknown"] == [987654]
    assert out["persons"][0]["gender"] == "male"


# ---------------------------------------------------------------------------
# API (NAS / standalone)
# ---------------------------------------------------------------------------

def test_persons_list_carries_the_attributes(client, db):
    pid = _person(db, "Asa")
    db.set_person_attributes([pid], {"gender": "male", "birth_year": 2015})
    rows = {p["id"]: p for p in client.get("/api/persons").json()["persons"]}
    assert rows[pid]["gender"] == "male" and rows[pid]["birth_year"] == 2015
    assert rows[pid]["birth_year_approx"] is False
    assert all({"gender", "birth_year", "birth_year_approx"} <= set(p) for p in rows.values())


def test_patch_one_person(client, db):
    pid = _person(db, "Asa")
    r = client.patch(f"/api/persons/{pid}", json={"gender": "male", "birth_year": 2015,
                                                   "birth_year_approx": True})
    assert r.status_code == 200, r.text
    assert r.json()["person"] == {"id": pid, "name": "Asa", "gender": "male",
                                  "birth_year": 2015, "birth_year_approx": True}
    r = client.patch(f"/api/persons/{pid}", json={"birth_year": None})
    assert r.json()["person"]["gender"] == "male"
    assert _attrs(db, pid) == ("male", None, 0)


@pytest.mark.parametrize("body,code", [
    ({"gender": "boy"}, 400), ({"birth_year": 1492}, 400), ({}, 400),
    ({"name": "Zed"}, 400)])
def test_patch_refuses_bad_bodies(client, db, body, code):
    pid = _person(db, "Asa")
    assert client.patch(f"/api/persons/{pid}", json=body).status_code == code
    assert _attrs(db, pid) == (None, None, 0)


def test_patch_unknown_person_is_404(client):
    assert client.patch("/api/persons/987654", json={"gender": "male"}).status_code == 404


def test_bulk_attributes_counts(client, db):
    a, b = _person(db, "Asa"), _person(db, "Zack Test")
    r = client.post("/api/persons/bulk-attributes",
                    json={"person_ids": [a, b, 987654], "gender": "male"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["updated"] == 2 and d["unknown"] == 1 and d["unknown_ids"] == [987654]
    assert _attrs(db, a)[0] == _attrs(db, b)[0] == "male"
    # only birth year this time: gender stays
    client.post("/api/persons/bulk-attributes",
                json={"person_ids": [a], "birth_year": 2012, "birth_year_approx": True})
    assert _attrs(db, a) == ("male", 2012, 1)


@pytest.mark.parametrize("body,code", [
    ({"person_ids": [], "gender": "male"}, 400),
    ({"person_ids": ["x"], "gender": "male"}, 400),
    ({"person_ids": [987654], "gender": "male"}, 404),
    ({"person_ids": [1]}, 400),
    ({"person_ids": [1], "gender": "girl"}, 400)])
def test_bulk_refuses(client, body, code):
    assert client.post("/api/persons/bulk-attributes", json=body).status_code == code


def test_request_intent_rules_cover_the_endpoints():
    from photosearch.request_intent import infer_intent
    assert "person 7" in infer_intent("PATCH", "/api/persons/7", "")
    assert "several people" in infer_intent("POST", "/api/persons/bulk-attributes", "")


# ---------------------------------------------------------------------------
# Replica mode: NAS-authoritative, then mirrored
# ---------------------------------------------------------------------------

def test_replica_forwards_then_mirrors(client, db, monkeypatch):
    from photosearch import web
    pid = _person(db, "Asa")
    calls = []

    def fake_nas(method, path, body=None, timeout=120.0):
        calls.append((method, path, body))
        return {"ok": True, "person": {"id": pid, "name": "Asa", "gender": "male",
                                       "birth_year": 2015, "birth_year_approx": False}}
    monkeypatch.setattr(web, "_nas_url", "http://nas.invalid:8000")
    monkeypatch.setattr(web, "_nas_json", fake_nas)
    r = client.patch(f"/api/persons/{pid}", json={"gender": "male", "birth_year": 2015})
    assert r.status_code == 200, r.text
    assert calls == [("PATCH", f"/api/persons/{pid}", {"gender": "male", "birth_year": 2015})]
    assert r.json()["mirror"] == {"mirrored": 1}
    assert _attrs(db, pid) == ("male", 2015, 0)


def test_replica_bulk_forwards_then_mirrors(client, db, monkeypatch):
    from photosearch import web
    a, b = _person(db, "Asa"), _person(db, "Zack Test")

    def fake_nas(method, path, body=None, timeout=120.0):
        assert (method, path) == ("POST", "/api/persons/bulk-attributes")
        assert body == {"person_ids": sorted([a, b]), "gender": "female"}
        return {"ok": True, "updated": 2, "unknown": 0, "unknown_ids": [],
                "persons": [{"id": i, "name": n, "gender": "female", "birth_year": None,
                             "birth_year_approx": False}
                            for i, n in ((a, "Asa"), (b, "Zack Test"))]}
    monkeypatch.setattr(web, "_nas_url", "http://nas.invalid:8000")
    monkeypatch.setattr(web, "_nas_json", fake_nas)
    r = client.post("/api/persons/bulk-attributes",
                    json={"person_ids": [b, a], "gender": "female"})
    assert r.status_code == 200 and r.json()["mirror"] == {"mirrored": 2}
    assert _attrs(db, a)[0] == _attrs(db, b)[0] == "female"


@pytest.mark.parametrize("code,detail", [(405, '{"detail":"Method Not Allowed"}'),
                                         (404, '{"detail":"Not Found"}')])
def test_replica_with_an_old_nas_says_so_and_writes_nothing(client, db, monkeypatch,
                                                            code, detail):
    from photosearch import web
    pid = _person(db, "Asa")

    def old_nas(method, path, body=None, timeout=120.0):
        raise HTTPException(status_code=code, detail=f"NAS: {detail}")
    monkeypatch.setattr(web, "_nas_url", "http://nas.invalid:8000")
    monkeypatch.setattr(web, "_nas_json", old_nas)
    for r in (client.patch(f"/api/persons/{pid}", json={"gender": "male"}),
              client.post("/api/persons/bulk-attributes",
                          json={"person_ids": [pid], "gender": "male"})):
        assert r.status_code == 409 and "v42" in r.json()["detail"]
    assert _attrs(db, pid) == (None, None, 0)


def test_replica_passes_a_real_nas_404_through(client, db, monkeypatch):
    from photosearch import web

    def nas(method, path, body=None, timeout=120.0):
        raise HTTPException(status_code=404, detail='NAS: {"detail":"Person not found"}')
    monkeypatch.setattr(web, "_nas_url", "http://nas.invalid:8000")
    monkeypatch.setattr(web, "_nas_json", nas)
    r = client.patch("/api/persons/987654", json={"gender": "male"})
    assert r.status_code == 404 and "Person not found" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Never off the machine
# ---------------------------------------------------------------------------

def test_attributes_stay_out_of_mcp_and_outbound_captions():
    """The MCP/Ask tools and the Google/Shutterfly upload code never read the
    attribute columns."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "photosearch"
    for name in ("tools.py", "agent.py", "mcp_server.py", "google_photos.py",
                 "shutterfly_export.py"):
        src = (root / name).read_text()
        assert "birth_year" not in src and "gender" not in src, name

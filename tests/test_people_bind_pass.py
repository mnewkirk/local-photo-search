"""The people-bind pass: schema v40, the claim predicate, the claim payload,
the submit re-validation and invalidation (docs/plans/description-names.md)."""
import json
import sqlite3

import pytest

from photosearch import people_bind as PB
from photosearch.db import (
    DISPLAY_NAME_SOURCES, MAX_PROCESS_ATTEMPTS, NON_BATCH_PASSES, PASS_MEDIA,
    PEOPLE_BIND_MAX_FACES, PEOPLE_BIND_MIN_BBOX_EDGE, PEOPLE_BIND_MIN_DET_SCORE,
    PhotoDB, SCHEMA_VERSION,
)
from tests.test_worker_submit_resilience import _open, client  # noqa: F401  (fixture)

DESC = "A young boy with reddish hair walks across a grassy field."
SPAN = "A young boy with reddish hair"
BOX = (100, 400, 400, 100)       # top, right, bottom, left: a 300 px face


@pytest.fixture
def db(tmp_path):
    d = PhotoDB(str(tmp_path / "t.db"))
    yield d
    d.close()


def _photo(db, name="a", desc=DESC, verified=True):
    pid = db.add_photo(filepath=f"2026/x/{name}.JPG", filename=f"{name}.JPG",
                       description=desc, image_width=6000, image_height=4000)
    if verified:
        db.conn.execute("UPDATE photos SET verified_at='2026-10-04T00:00:00' WHERE id=?", (pid,))
    return pid


def _face(db, pid, source="manual", box=BOX, det=0.9, person=True):
    pers = db.add_person("Zack Azerad") if person and not db.get_person_by_name("Zack Azerad") \
        else (db.get_person_by_name("Zack Azerad") or {}).get("id")
    fid = db.add_face(pid, box, [], person_id=pers if person else None, det_score=det)
    db.conn.execute("UPDATE faces SET match_source=? WHERE id=?", (source, fid))
    db.conn.commit()
    return fid


def _claimable(db):
    return [r["id"] for r in db.get_unprocessed_photos("people-bind", limit=100)]


def test_constants_match_the_modules_they_mirror():
    from photosearch import faces, face_verify
    assert DISPLAY_NAME_SOURCES == face_verify.TRUSTED_LABEL_SOURCES
    assert PEOPLE_BIND_MIN_DET_SCORE == PB.MIN_DET_SCORE == faces.CLUSTER_MIN_DET_SCORE
    assert PEOPLE_BIND_MIN_BBOX_EDGE == PB.MIN_BBOX_EDGE == faces.CLUSTER_MIN_BBOX_EDGE
    assert PEOPLE_BIND_MAX_FACES == PB.MAX_FACES
    from photosearch import description_render as R
    assert R.REJECTED == faces.REJECTED_MATCH_SOURCE


def test_pass_is_registered_as_a_still_non_batch_pass():
    assert PASS_MEDIA["people-bind"] == "still"
    assert NON_BATCH_PASSES == ("people-bind",)
    from photosearch import batch_state, describe, rerun, worker_api
    assert "people-bind" not in batch_state.WORKER_PASSES      # never gates a batch
    assert "people-bind" in rerun.ALL_PASSES and "people-bind" in worker_api._ALL_PASSES
    assert describe.PASS_ROLES["people-bind"] == rerun._PASS_LLM["people-bind"][0] == "bind"
    assert "bind" in rerun.FLEET_ROLE_MODELS


def test_bind_role_falls_back_to_the_visual_model(monkeypatch):
    from photosearch import describe
    monkeypatch.delenv("PHOTOSEARCH_LLM_BIND_MODEL", raising=False)
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "some-vlm")
    assert describe._resolve_openai_model("qwen2.5-vl", "bind") == "some-vlm"


def test_v39_db_migrates_to_v40(tmp_path):
    path = str(tmp_path / "v39.db")
    with PhotoDB(path) as d:
        pid = d.add_photo(filepath="a.jpg", filename="a.jpg", description="x")
    conn = sqlite3.connect(path)
    conn.executescript("DROP TABLE description_subjects; DROP INDEX idx_faces_trusted_photo; "
                       "UPDATE schema_info SET value='39' WHERE key='version';")
    conn.commit(); conn.close()
    with PhotoDB(path) as d:
        assert d.conn.execute("SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 0
        names = {r[0] for r in d.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_faces_trusted_photo" in names
        assert d.conn.execute("SELECT description FROM photos WHERE id=?", (pid,)).fetchone()[0] == "x"
        assert int(d.conn.execute("SELECT value FROM schema_info WHERE key='version'"
                                  ).fetchone()[0]) == SCHEMA_VERSION == 40


# --- the claim predicate ------------------------------------------------------------

def test_claim_needs_description_verify_done_and_a_trusted_face(db):
    ok = _photo(db, "ok"); _face(db, ok)
    nodesc = _photo(db, "nd", desc=None); _face(db, nodesc)
    unverified = _photo(db, "uv", verified=False); _face(db, unverified)
    nolabel = _photo(db, "nl"); _face(db, nolabel, person=False, source=None)
    temporal = _photo(db, "tp"); _face(db, temporal, source="temporal")
    rejected = _photo(db, "rj"); _face(db, rejected, source="rejected")
    dedupe = _photo(db, "dd"); _face(db, dedupe, source="dedupe_unmatched")
    strict = _photo(db, "st"); _face(db, strict, source="strict")
    merge = _photo(db, "mr"); _face(db, merge, source="merge_review")
    assert sorted(_claimable(db)) == sorted([ok, strict, merge])
    assert db.count_unprocessed_photos("people-bind") == 3


def test_a_verify_that_gave_up_unblocks_the_photo(db):
    pid = _photo(db, verified=False); _face(db, pid)
    assert _claimable(db) == []
    db.conn.execute("INSERT INTO worker_processed(photo_id, pass_type, attempts) "
                    "VALUES (?, 'verify', ?)", (pid, MAX_PROCESS_ATTEMPTS))
    assert _claimable(db) == [pid]


def test_crowd_cap_counts_only_the_faces_that_would_be_drawn(db):
    crowd = _photo(db, "crowd")
    for i in range(PEOPLE_BIND_MAX_FACES + 1):
        _face(db, crowd, box=(100, 400 + i * 400, 400, 100 + i * 400))
    assert _claimable(db) == []
    # tiny / low-confidence faces are not drawn, so they do not make a crowd
    busy = _photo(db, "busy"); _face(db, busy)
    for i in range(10):
        _face(db, busy, source="temporal", box=(0, 20, 20, 0), det=0.3, person=False)
    assert _claimable(db) == [busy]


def test_row_and_exhausted_attempts_remove_the_photo(db):
    a = _photo(db, "a"); _face(db, a)
    b = _photo(db, "b"); _face(db, b)
    db.conn.execute("INSERT INTO description_subjects(photo_id, description_sha, slots_json, "
                    "method, bind_version) VALUES (?,?,?,?,1)", (a, PB.sha_of(DESC), "[]", "vlm"))
    db.conn.execute("INSERT INTO worker_processed(photo_id, pass_type, attempts) "
                    "VALUES (?, 'people-bind', ?)", (b, MAX_PROCESS_ATTEMPTS))
    assert _claimable(db) == []
    assert db.count_unprocessed_photos("people-bind") == 0


def test_a_video_is_never_offered(db):
    v = db.add_photo(filepath="v.mp4", filename="v.mp4", description=DESC, media_type="video")
    db.conn.execute("UPDATE photos SET verified_at='x' WHERE id=?", (v,))
    _face(db, v)
    assert _claimable(db) == []


def test_scoped_claim_and_count_agree(db):
    a = _photo(db, "a"); _face(db, a)
    b = _photo(db, "b"); _face(db, b)
    assert [r["id"] for r in db.get_unprocessed_photos("people-bind", photo_ids=[a])] == [a]
    assert db.count_unprocessed_photos("people-bind", photo_ids=[a, b]) == 2


def test_claim_sql_enumerates_labelled_faces_and_never_scans_photos(db):
    """EXISTS-correlated form scanned the whole photos table (250k rows with
    three correlated subqueries each on the N100). The IN form drives from the
    faces index and probes photos by rowid."""
    from photosearch import db as dbmod
    a = _photo(db); _face(db, a)
    for q in (f"SELECT p.id FROM photos p WHERE {dbmod.PEOPLE_BIND_MISSING_SQL} "
              f"AND p.media_type IS NULL LIMIT 10",
              f"SELECT COUNT(*) FROM photos p WHERE {dbmod.PEOPLE_BIND_MISSING_SQL} "
              f"AND p.media_type IS NULL"):
        plan = " | ".join(r[3] for r in db.conn.execute("EXPLAIN QUERY PLAN " + q))
        assert "SCAN p" not in plan and "SEARCH p USING INTEGER PRIMARY KEY" in plan, plan
        assert "LIST SUBQUERY" in plan and "SEARCH f USING" in plan, plan


# --- invalidation ------------------------------------------------------------------------

def _row(db, pid):
    db.conn.execute("INSERT OR REPLACE INTO description_subjects(photo_id, description_sha, "
                    "slots_json, method, bind_version) VALUES (?,?,?,?,1)",
                    (pid, PB.sha_of(DESC), "[]", "vlm"))
    db.conn.execute("INSERT OR REPLACE INTO worker_processed(photo_id, pass_type, attempts) "
                    "VALUES (?, 'people-bind', 1)", (pid,))
    db.conn.commit()


@pytest.mark.parametrize("include_verify", [True, False])
def test_a_new_description_drops_the_row_and_the_ledger_even_for_verifys_rewrite(db, include_verify):
    pid = _photo(db); _row(db, pid)
    db.invalidate_description_dependents(pid, include_verify=include_verify)
    assert db.conn.execute("SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed WHERE "
                           "pass_type='people-bind'").fetchone()[0] == 0


def test_invalidation_tolerates_a_pre_v40_db(db):
    pid = _photo(db)
    db.conn.execute("DROP TABLE description_subjects")
    db.invalidate_description_dependents(pid)      # must not raise


# --- the worker API --------------------------------------------------------------------------

def _seed_api(client_):
    with _open() as d:
        d.conn.execute("UPDATE photos SET verified_at='2090-01-01' WHERE id=1")
        d.conn.execute("UPDATE photos SET image_width=6000, image_height=4000 WHERE id=1")
        pid = d.add_person("Zack Azerad")
        fid = d.add_face(1, BOX, [], person_id=pid, det_score=0.9)
        d.conn.execute("UPDATE faces SET match_source='manual' WHERE id=?", (fid,))
        d.conn.execute("UPDATE photos SET description=? WHERE id=1", (DESC,))
        d.conn.commit()
    return fid


def _slot(fid, start=0, text=SPAN):
    return {"slot": "s1", "kind": "person",
            "spans": [{"start": start, "end": start + len(text), "text": text}],
            "faces": [{"face_id": fid, "bbox": list(BOX)}]}


def _submit(client, results, failures=None):
    body = {"batch_id": "no-claim", "pass_type": "people-bind", "people_bind_results": results}
    r = client.post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _attempts():
    with _open() as d:
        return {r[0]: r[1] for r in d.conn.execute(
            "SELECT photo_id, attempts FROM worker_processed WHERE pass_type='people-bind'")}


def _stored(pid=1):
    with _open() as d:
        r = d.conn.execute("SELECT * FROM description_subjects WHERE photo_id=?", (pid,)).fetchone()
        gens = d.conn.execute("SELECT model_used FROM generations WHERE photo_id=? AND "
                              "text_type='people-bind'", (pid,)).fetchall()
        return (dict(r) if r else None), [g[0] for g in gens]


def test_claim_payload_is_label_free_and_carries_the_drawn_faces(client):
    fid = _seed_api(client)
    r = client.post("/api/worker/claim-batch",
                    json={"worker_id": "w", "pass_type": "people-bind", "limit": 4})
    assert r.status_code == 200, r.text
    photos = r.json()["photos"]
    assert [p["id"] for p in photos] == [1]
    inp = photos[0]["people_bind_input"]
    assert inp["description"] == DESC and inp["description_sha"] == PB.sha_of(DESC)
    assert [f["face_id"] for f in inp["faces"]] == [fid]
    assert (inp["image_width"], inp["image_height"]) == (6000, 4000)
    blob = json.dumps(inp)
    assert "Zack" not in blob and "person" not in blob.replace("subject_boxes", "")


def test_a_valid_vlm_binding_is_stored_with_provenance_and_needs_no_ledger_row(client):
    fid = _seed_api(client)
    resp = _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                             "slots": [_slot(fid)], "method": "vlm", "model": "qwen-vl",
                             "model_version": "lmstudio"}])
    assert resp["written"] == 1 and resp["status"] == "ok"
    row, gens = _stored()
    assert json.loads(row["slots_json"])[0]["faces"][0]["face_id"] == fid
    assert row["method"] == "vlm" and row["bind_version"] == PB.BIND_VERSION
    assert gens == ["qwen-vl"] and _attempts() == {}
    with _open() as d:
        assert d.get_unprocessed_photos("people-bind", limit=5) == []     # the row is the marker


def test_rule_tier_rows_log_no_generation(client):
    fid = _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                      "slots": [_slot(fid)], "method": "rule", "model": "rule"}])
    row, gens = _stored()
    assert row["model"] == "rule" and gens == []


def test_empty_slots_is_a_result_done_in_one_pass(client):
    _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC), "slots": [],
                      "method": "vlm", "model": "m"}])
    row, _ = _stored()
    assert row["slots_json"] == "[]"
    with _open() as d:
        assert d.count_unprocessed_photos("people-bind") == 0


def test_stale_sha_defers_without_spending_an_attempt(client):
    fid = _seed_api(client)
    resp = _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of("an older text"),
                             "slots": [_slot(fid)], "method": "vlm", "model": "m"}])
    assert resp["deferred"] == 1 and 1 in resp["deferred_photo_ids"]
    assert _stored()[0] is None and _attempts() == {}


@pytest.mark.parametrize("mutate", [
    lambda s, fid: dict(s, spans=[{"start": 0, "end": 5, "text": "wrong"}]),
    lambda s, fid: dict(s, faces=[{"face_id": 999999, "bbox": list(BOX)}]),
    lambda s, fid: dict(s, kind="group"),
])
def test_an_invalid_binding_spends_one_attempt_and_writes_nothing(client, mutate):
    fid = _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                      "slots": [mutate(_slot(fid), fid)], "method": "vlm", "model": "m"}])
    assert _stored()[0] is None and _attempts() == {1: 1}
    with _open() as d:
        assert d.conn.execute("SELECT COUNT(*) FROM index_errors WHERE pass_type='people-bind'"
                              ).fetchone()[0] == 1


def test_a_worker_reported_failure_spends_one_attempt(client):
    _seed_api(client)
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-claim", "pass_type": "people-bind", "people_bind_results": [],
        "failures": [{"photo_id": 1, "error": "TruncatedOutput: cut off"}]})
    assert r.status_code == 200
    assert _attempts() == {1: 1}


def test_clear_pass_requeues(client):
    fid = _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                      "slots": [_slot(fid)], "method": "vlm", "model": "m"}])
    r = client.post("/api/worker/clear-pass", json={"pass_type": "people-bind", "photo_ids": [1]})
    assert r.status_code == 200, r.text
    assert _stored()[0] is None
    with _open() as d:
        assert d.count_unprocessed_photos("people-bind") == 1


def test_redescribe_through_the_api_drops_the_binding(client):
    fid = _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                      "slots": [_slot(fid)], "method": "vlm", "model": "m"}])
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-claim", "pass_type": "describe",
        "describe_results": [{"photo_id": 1, "description": "a different description"}],
        "model": "m", "model_version": "v"})
    assert r.status_code == 200
    assert _stored()[0] is None


# --- the audit seam ----------------------------------------------------------------------

def test_audit_seam_never_raises_and_is_called_on_bind_and_unbind(client, monkeypatch):
    from photosearch import description_audit as A
    seen = []
    monkeypatch.setattr(A, "record", lambda conn, pid, kind, details=None: seen.append((pid, kind)))
    fid = _seed_api(client)
    _submit(client, [{"photo_id": 1, "description_sha": PB.sha_of(DESC),
                      "slots": [_slot(fid)], "method": "vlm", "model": "m"}])
    client.post("/api/worker/submit-results", json={
        "batch_id": "no-claim", "pass_type": "describe",
        "describe_results": [{"photo_id": 1, "description": "different"}]})
    assert seen == [(1, A.BOUND), (1, A.UNBOUND)]


def test_audit_record_swallows_everything():
    from photosearch import description_audit as A
    A.record(None, 1, "bound", {"x": object()})        # unserialisable details
    A.record(None, 1, "nonsense")

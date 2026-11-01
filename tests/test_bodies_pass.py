"""The `bodies` worker pass (schema v43, docs/plans/people-context-describe.md):
person boxes + keypoints, a non-batch still pass that people-bind waits for."""
import json
import sqlite3

import pytest

from photosearch import worker as W
from photosearch.db import MAX_PROCESS_ATTEMPTS, NON_BATCH_PASSES, PASS_MEDIA, PhotoDB, SCHEMA_VERSION
from tests.test_worker_submit_resilience import _attempts, _open, client  # noqa: F401  (fixture)

KPS = [[10.0, 20.0, 5.0]] * 17
BODY = {"bbox": [50, 600, 3000, 0], "score": 0.97, "keypoints": KPS, "facing": "front"}


@pytest.fixture
def db(tmp_path):
    d = PhotoDB(str(tmp_path / "t.db"))
    yield d
    d.close()


def _claimable(db, **kw):
    return [r["id"] for r in db.get_unprocessed_photos("bodies", limit=100, **kw)]


def test_registered_as_a_still_non_batch_pass_before_people_bind():
    from photosearch import batch_state, rerun, worker_api
    assert PASS_MEDIA["bodies"] == "still"
    assert NON_BATCH_PASSES.index("bodies") < NON_BATCH_PASSES.index("people-bind")
    assert "bodies" not in batch_state.WORKER_PASSES
    assert rerun.ALL_PASSES.index("bodies") < rerun.ALL_PASSES.index("people-bind")
    assert worker_api._ALL_PASSES.index("bodies") < worker_api._ALL_PASSES.index("people-bind")


def test_claim_is_stills_without_rows_under_the_cap(db):
    a = db.add_photo(filepath="2026/x/a.JPG", filename="a.JPG")
    b = db.add_photo(filepath="2026/x/b.JPG", filename="b.JPG")
    c = db.add_photo(filepath="2026/x/c.JPG", filename="c.JPG")
    v = db.add_photo(filepath="2026/x/v.mp4", filename="v.mp4", media_type="video")
    assert sorted(_claimable(db)) == [a, b, c]
    from photosearch.people_bind_store import store_bodies
    store_bodies(db.conn, a, [BODY], 1)
    db.conn.execute("INSERT INTO worker_processed(photo_id, pass_type, attempts) "
                    "VALUES (?, 'bodies', ?)", (b, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    assert _claimable(db) == [c] and v not in _claimable(db)
    assert db.count_unprocessed_photos("bodies") == 1
    assert _claimable(db, photo_ids=[a, c]) == [c]
    assert db.count_unprocessed_photos("bodies", photo_ids=[a, b]) == 0


def test_v42_db_migrates_to_v43(tmp_path):
    path = str(tmp_path / "v42.db")
    with PhotoDB(path) as d:
        d.add_photo(filepath="a.jpg", filename="a.jpg")
    conn = sqlite3.connect(path)
    conn.executescript("DROP TABLE person_boxes; "
                       "UPDATE schema_info SET value='42' WHERE key='version';")
    conn.commit(); conn.close()
    with PhotoDB(path) as d:
        cols = {r[1] for r in d.conn.execute("PRAGMA table_info(person_boxes)")}
        assert {"bbox_top", "bbox_right", "bbox_bottom", "bbox_left", "score",
                "detector_version", "keypoints_json", "facing"} <= cols
        idx = {r[0] for r in d.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_person_boxes_photo" in idx
        assert int(d.conn.execute("SELECT value FROM schema_info WHERE key='version'"
                                  ).fetchone()[0]) == SCHEMA_VERSION == 43


def _submit(client, results, failures=None):
    body = {"batch_id": "no-claim", "pass_type": "bodies", "bodies_results": results}
    if failures:
        body["failures"] = failures
    r = client.post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _rows(pid):
    with _open() as d:
        return [dict(r) for r in d.conn.execute(
            "SELECT * FROM person_boxes WHERE photo_id=? ORDER BY id", (pid,))]


def test_submit_stores_rows_and_needs_no_requeue(client):
    out = _submit(client, [{"photo_id": 1, "bodies": [BODY, dict(BODY, keypoints=None, facing=None,
                                                                    score=0.4)]}])
    assert out["written"] == 2
    rows = _rows(1)
    assert [(r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"]) for r in rows] == \
        [(50, 600, 3000, 0)] * 2
    assert json.loads(rows[0]["keypoints_json"]) == KPS and rows[0]["facing"] == "front"
    assert rows[1]["keypoints_json"] is None and rows[0]["detector_version"] == 1
    with _open() as d:
        assert 1 not in [r["id"] for r in d.get_unprocessed_photos("bodies", limit=100)]
    # a re-detection replaces, never appends
    _submit(client, [{"photo_id": 1, "bodies": [BODY]}])
    assert len(_rows(1)) == 1


def test_an_empty_result_is_terminal_and_a_failure_spends_one_attempt(client):
    _submit(client, [{"photo_id": 1, "bodies": []}], failures=[{"photo_id": 2, "error": "decode"}])
    att = _attempts("bodies")
    assert att[1] == MAX_PROCESS_ATTEMPTS and att[2] == 1
    assert _rows(1) == [] and _rows(2) == []


def test_clear_pass_requeues(client):
    _submit(client, [{"photo_id": 1, "bodies": [BODY]}, {"photo_id": 2, "bodies": []}])
    r = client.post("/api/worker/clear-pass", json={"pass_type": "bodies", "photo_ids": [1, 2]})
    assert r.status_code == 200, r.text
    assert _rows(1) == [] and _attempts("bodies") == {}


def test_worker_reports_a_detection_error_as_a_failure_never_an_empty_list(monkeypatch):
    import photosearch.person_boxes as PBX
    def detect(path):
        if path == "bad":
            raise RuntimeError("CUDA error")
        return [] if path == "empty" else [BODY]
    monkeypatch.setattr(PBX, "detect_people", detect)
    rows = W._process_bodies([({"id": 1, "filename": "a"}, "ok"), ({"id": 2, "filename": "b"}, "bad"),
                              ({"id": 3, "filename": "c"}, "empty")])
    by = {r["photo_id"]: r for r in rows}
    assert by[1]["bodies"] == [BODY] and by[1]["detector_version"] == PBX.DETECTOR_VERSION
    assert "error" in by[2] and "bodies" not in by[2]
    assert by[3]["bodies"] == []

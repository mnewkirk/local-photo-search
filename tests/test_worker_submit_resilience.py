"""A failed SERVER-SIDE write must not burn a worker retry attempt.

Evidence (live NAS, 2026-09-20): two perfectly good photos ended with
`visual_tags IS NULL` and `worker_processed.attempts = 5` for the
'category-visual' pass — permanently abandoned, past MAX_PROCESS_ATTEMPTS.
At the second of their last attempt the errors table logged `database is
locked` for four sibling photos of the same submitted batch: the nightly
maintenance sweep held the write lock for longer than the 60 s busy timeout
while the fleet was submitting.

`submit_results` counted a photo as "processed" BEFORE attempting its write,
so a lock on the server consumed an attempt exactly as if the PHOTO were
bad, and threw the GPU work away with it. An attempt may only be counted
when the result was actually persisted, or when the WORKER reported a
genuine per-photo failure (an empty description / empty tag list).
"""

import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Same shape as tests/test_worker_categories.py's fixture: a tmp DB with
    web._db_path / worker_api._db_path patched, plus _shutting_down reset so
    an earlier TestClient teardown can't leak a 503 into these tests.

    Photo years are 2090/2091 to stay clear of anything a shared fixture
    seeds under 2026.
    """
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web, worker_api
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_shutting_down", False, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        db.conn.executemany(
            "INSERT INTO photos (id, filepath, filename, description, date_taken) "
            "VALUES (?, ?, ?, ?, ?)",
            [(1, "2090/a.jpg", "a.jpg", "a dog at the beach", "2090-04-01 10:00:00"),
             (2, "2091/b.jpg", "b.jpg", "a misty mountain", "2091-04-01 10:00:00"),
             (3, "2091/c.jpg", "c.jpg", "a red bicycle", "2091-04-02 10:00:00")],
        )
        db.conn.commit()
    return TestClient(web.app)


def _open(path=None):
    from photosearch.db import PhotoDB
    return PhotoDB(path or os.environ["PHOTOSEARCH_DB"])


def _attempts(pass_type: str) -> dict[int, int]:
    with _open() as db:
        return {r[0]: r[1] for r in db.conn.execute(
            "SELECT photo_id, attempts FROM worker_processed WHERE pass_type=?",
            (pass_type,)).fetchall()}


def _lock_photo(monkeypatch, bad_id: int, method: str = "update_photo"):
    """Make one photo's DB write raise 'database is locked', as the real
    contention did — every other photo in the batch writes normally."""
    from photosearch.db import PhotoDB
    real = getattr(PhotoDB, method)

    def flaky(self, photo_id, *a, **kw):
        if photo_id == bad_id:
            raise sqlite3.OperationalError("database is locked")
        return real(self, photo_id, *a, **kw)

    monkeypatch.setattr(PhotoDB, method, flaky)


def _claim(client, pass_type):
    return client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": pass_type, "limit": 10,
    }).json()["batch_id"]


# ---------------------------------------------------------------------------
# The headline case: category-visual
# ---------------------------------------------------------------------------

def test_locked_write_does_not_burn_an_attempt(client, monkeypatch):
    """Photo 2's write raises; photo 1's succeeds. Only photo 1 is marked."""
    _lock_photo(monkeypatch, 2)
    batch_id = _claim(client, "category-visual")
    r = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "pass_type": "category-visual",
        "category_visual_results": [
            {"photo_id": 1, "visual_tags": ["dramatic"], "model": "llava"},
            {"photo_id": 2, "visual_tags": ["foggy"], "model": "llava"},
        ],
    })
    assert r.status_code == 200, r.text
    assert _attempts("category-visual") == {1: 1}, "the locked photo must not be marked"
    with _open() as db:
        rows = {r[0]: r[1] for r in db.conn.execute(
            "SELECT id, visual_tags FROM photos WHERE id IN (1,2)").fetchall()}
    assert json.loads(rows[1]) == ["dramatic"]
    assert rows[2] is None


def test_the_deferred_photo_is_still_claimable(client, monkeypatch):
    """The real defect: five collisions pushed attempts past
    MAX_PROCESS_ATTEMPTS and the photo stopped being claimed forever. Seed
    it one short of the cap, so a burned attempt would retire it."""
    from photosearch.db import MAX_PROCESS_ATTEMPTS
    with _open() as db:
        for _ in range(MAX_PROCESS_ATTEMPTS - 1):
            db.mark_processed([2], "category-visual")

    _lock_photo(monkeypatch, 2)
    batch_id = _claim(client, "category-visual")
    client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "pass_type": "category-visual",
        "category_visual_results": [
            {"photo_id": 2, "visual_tags": ["foggy"], "model": "llava"},
        ],
    })
    assert _attempts("category-visual")[2] == MAX_PROCESS_ATTEMPTS - 1

    with _open() as db:
        claimable = {p["id"] for p in db.get_unprocessed_photos("category-visual", limit=50)}
        assert 2 in claimable, "a server-side lock must leave the photo claimable"
        assert db.count_unprocessed_photos("category-visual") >= 1


def test_response_reports_deferred_without_breaking_old_workers(client, monkeypatch):
    """Workers running older code read only `written` / `processed`; the new
    `deferred` count is additive."""
    _lock_photo(monkeypatch, 2)
    batch_id = _claim(client, "category-visual")
    body = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "pass_type": "category-visual",
        "category_visual_results": [
            {"photo_id": 1, "visual_tags": ["dramatic"]},
            {"photo_id": 2, "visual_tags": ["foggy"]},
        ],
    }).json()
    assert body["written"] == 1
    assert body["processed"] == 1
    assert body["batch_id"] == batch_id
    assert body["deferred"] == 1
    assert body["deferred_photo_ids"] == [2]
    assert body["status"] == "partial"


def test_empty_category_visual_still_marks_processed(client):
    """PRESERVED behaviour: a successful-but-empty result writes '[]' and is
    done in ONE pass. Only the DB-write-raised case changes."""
    batch_id = _claim(client, "category-visual")
    body = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id, "pass_type": "category-visual",
        "category_visual_results": [{"photo_id": 1, "visual_tags": []}],
    }).json()
    assert _attempts("category-visual") == {1: 1}
    assert body["status"] == "ok"
    assert body["deferred"] == 0
    with _open() as db:
        row = db.conn.execute("SELECT visual_tags FROM photos WHERE id=1").fetchone()
    assert json.loads(row[0]) == []


# ---------------------------------------------------------------------------
# Every pass branch had the same shape
# ---------------------------------------------------------------------------

_PAYLOADS = {
    "describe": ("describe_results",
                 [{"photo_id": 1, "description": "d1"}, {"photo_id": 2, "description": "d2"}]),
    "tags": ("tags_results",
             [{"photo_id": 1, "tags": ["a"]}, {"photo_id": 2, "tags": ["b"]}]),
    "category-content": ("category_content_results",
                         [{"photo_id": 1, "categories": ["a"]},
                          {"photo_id": 2, "categories": ["b"]}]),
    "category-visual": ("category_visual_results",
                        [{"photo_id": 1, "visual_tags": ["a"]},
                         {"photo_id": 2, "visual_tags": ["b"]}]),
    "keywords": ("keywords_results",
                 [{"photo_id": 1, "keywords": ["a"]}, {"photo_id": 2, "keywords": ["b"]}]),
    "aesthetics": ("aesthetics_results",
                   [{"photo_id": 1, "scores": {"aes_overall": 7.0}},
                    {"photo_id": 2, "scores": {"aes_overall": 6.0}}]),
}


@pytest.mark.parametrize("pass_type", sorted(_PAYLOADS))
def test_every_marking_pass_defers_a_failed_write(client, monkeypatch, pass_type):
    key, results = _PAYLOADS[pass_type]
    _lock_photo(monkeypatch, 2)
    body = client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": pass_type, key: results,
    }).json()
    assert body["deferred_photo_ids"] == [2], pass_type
    assert _attempts(pass_type).get(2) is None, f"{pass_type} burned an attempt"


def test_faces_defers_when_a_face_row_fails_but_keeps_no_faces_done(client, monkeypatch):
    """PRESERVED: a photo with NO faces is done, not failed. CHANGED: a photo
    whose face row failed to write is deferred, not marked."""
    from photosearch.db import PhotoDB
    real_add = PhotoDB.add_face

    def flaky(self, photo_id, *a, **kw):
        if photo_id == 2:
            raise sqlite3.OperationalError("database is locked")
        return real_add(self, photo_id, *a, **kw)

    monkeypatch.setattr(PhotoDB, "add_face", flaky)
    enc = [0.01] * 512
    body = client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": "faces",
        "face_results": [
            {"photo_id": 1, "faces": [{"bbox": [1, 2, 3, 4], "encoding": enc}]},
            {"photo_id": 2, "faces": [{"bbox": [1, 2, 3, 4], "encoding": enc}]},
            {"photo_id": 3, "faces": []},
        ],
    }).json()
    assert body["deferred_photo_ids"] == [2]
    marked = _attempts("faces")
    assert marked.get(3) == 1, "no faces found is a completed pass"
    assert marked.get(1) == 1
    assert marked.get(2) is None


def test_describe_with_no_text_is_still_marked(client):
    """PRESERVED: the model returning nothing is a completed attempt (the
    worker omits true deferrals from the payload entirely)."""
    client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": "describe",
        "describe_results": [{"photo_id": 1, "description": None}],
    })
    assert _attempts("describe") == {1: 1}


# ---------------------------------------------------------------------------
# A failed batch COMMIT loses the whole batch
# ---------------------------------------------------------------------------

def test_failed_batch_commit_marks_nothing(client, monkeypatch):
    """begin_batch/end_batch defer the commit, so a lock usually surfaces at
    COMMIT rather than at the UPDATE. When that happens none of the batch was
    persisted, so none of it may be marked processed — and the worker must
    hear about it instead of getting a 500."""
    from photosearch.db import PhotoDB

    def boom(self):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(PhotoDB, "end_batch", boom)
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": "category-visual",
        "category_visual_results": [
            {"photo_id": 1, "visual_tags": ["a"]},
            {"photo_id": 2, "visual_tags": ["b"]},
        ],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "deferred"
    assert body["written"] == 0
    assert body["processed"] == 0
    assert sorted(body["deferred_photo_ids"]) == [1, 2]
    assert _attempts("category-visual") == {}


def test_failed_batch_commit_logs_no_activity(client, monkeypatch):
    from photosearch.db import PhotoDB
    monkeypatch.setattr(PhotoDB, "end_batch",
                        lambda self: (_ for _ in ()).throw(
                            sqlite3.OperationalError("database is locked")))
    client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": "category-visual",
        "category_visual_results": [{"photo_id": 1, "visual_tags": ["a"]}],
    })
    with _open() as db:
        rows = db.conn.execute(
            "SELECT COUNT(*) FROM index_activity WHERE pass_type='category-visual'"
        ).fetchone()[0]
    assert rows == 0


# ---------------------------------------------------------------------------
# Never swallow the reason
# ---------------------------------------------------------------------------

def test_a_failing_log_error_still_reaches_the_logger(client, monkeypatch, caplog):
    """The original `except Exception: pass` around db.log_error meant that
    under exactly the conditions that caused the failure — a locked DB — the
    reason could vanish entirely."""
    from photosearch.db import PhotoDB
    _lock_photo(monkeypatch, 2)
    monkeypatch.setattr(PhotoDB, "log_error",
                        lambda *a, **k: (_ for _ in ()).throw(
                            sqlite3.OperationalError("database is locked")))
    with caplog.at_level("WARNING", logger="photosearch.worker_api"):
        client.post("/api/worker/submit-results", json={
            "batch_id": "no-such-claim", "pass_type": "category-visual",
            "category_visual_results": [{"photo_id": 2, "visual_tags": ["b"]}],
        })
    text = caplog.text
    assert "category-visual" in text
    assert "log" in text.lower(), "the failure to log the reason must itself be logged"
    assert _attempts("category-visual") == {}

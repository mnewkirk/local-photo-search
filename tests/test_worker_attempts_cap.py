"""Every ledgered worker pass must be bounded by MAX_PROCESS_ATTEMPTS, and a
clean empty `faces` result must be retired in ONE submit.

Two defects, one ledger:

A. `quality` and `verify` had no attempts clause in their claim predicates and
   submit_results never marked them processed, while the worker silently
   DROPPED a photo it could not process (quality: an unloadable image; verify:
   any exception). Such a photo left no trace and was re-claimed every TTL
   forever — the same shape as the clip infinite re-claim. A quality photo
   whose concept analysis failed rewrote its score and stayed claimable too.

B. `faces` re-detected every photo with nobody in frame MAX_PROCESS_ATTEMPTS
   times (67,853 photos, ~136k wasted InsightFace runs), because an empty
   result only incremented the counter. It could not simply be made terminal
   while the worker sent `faces: []` for a detection EXCEPTION as well — so the
   worker now reports errors as failure rows instead.

Photo years are 2090/2091 to stay clear of anything a shared fixture seeds.
"""

import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from photosearch.db import MAX_PROCESS_ATTEMPTS


@pytest.fixture
def client(tmp_path, monkeypatch):
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


def _open():
    from photosearch.db import PhotoDB
    return PhotoDB(os.environ["PHOTOSEARCH_DB"])


def _attempts(pass_type: str) -> dict[int, int]:
    with _open() as db:
        return {r[0]: r[1] for r in db.conn.execute(
            "SELECT photo_id, attempts FROM worker_processed WHERE pass_type=?",
            (pass_type,)).fetchall()}


def _claimable(pass_type: str) -> set[int]:
    with _open() as db:
        return {p["id"] for p in db.get_unprocessed_photos(pass_type, limit=50)}


def _count(pass_type: str, ids=None) -> int:
    with _open() as db:
        return db.count_unprocessed_photos(pass_type, photo_ids=ids)


def _submit(client, pass_type, **payload):
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": pass_type, **payload})
    assert r.status_code == 200, r.text
    return r.json()


def _exhaust(pass_type, ids):
    with _open() as db:
        for pid in ids:
            db.conn.execute(
                "INSERT OR REPLACE INTO worker_processed (photo_id, pass_type, attempts) "
                "VALUES (?, ?, ?)", (pid, pass_type, MAX_PROCESS_ATTEMPTS))
        db.conn.commit()


def _lock_update(monkeypatch, bad_id: int):
    from photosearch.db import PhotoDB
    real = PhotoDB.update_photo

    def flaky(self, photo_id, *a, **kw):
        if photo_id == bad_id:
            raise sqlite3.OperationalError("database is locked")
        return real(self, photo_id, *a, **kw)

    monkeypatch.setattr(PhotoDB, "update_photo", flaky)


# ---------------------------------------------------------------------------
# A — the predicates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pass_type", ["quality", "verify"])
def test_exhausted_photo_is_neither_claimed_nor_counted(client, pass_type):
    assert 1 in _claimable(pass_type)
    before = _count(pass_type)
    before_scoped = _count(pass_type, [1, 2])
    _exhaust(pass_type, [1])
    assert 1 not in _claimable(pass_type)
    assert _count(pass_type) == before - 1
    assert _count(pass_type, [1, 2]) == before_scoped - 1


@pytest.mark.parametrize("pass_type", ["quality", "verify"])
def test_below_the_cap_is_still_claimed(client, pass_type):
    with _open() as db:
        db.mark_processed([1], pass_type)
        db.mark_processed([1], pass_type)
    assert 1 in _claimable(pass_type)


def test_quality_self_heal_is_kept(client):
    """Either column missing still claims — the cap is an extra clause, not a
    replacement for the heal."""
    with _open() as db:
        db.update_photo(1, aesthetic_score=5.0)          # concepts still NULL
        db.update_photo(2, aesthetic_score=5.0, aesthetic_concepts="{}")
    assert 1 in _claimable("quality")
    assert 2 not in _claimable("quality")


# ---------------------------------------------------------------------------
# A — submit bookkeeping
# ---------------------------------------------------------------------------

def test_quality_success_spends_an_attempt(client):
    _submit(client, "quality", quality_results=[
        {"photo_id": 1, "aesthetic_score": 6.1, "aesthetic_concepts": "{}"}])
    assert _attempts("quality") == {1: 1}
    assert 1 not in _claimable("quality")


def test_quality_concepts_only_failure_is_capped(client):
    """Score written, concepts missing: used to be re-claimed forever."""
    for _ in range(MAX_PROCESS_ATTEMPTS):
        assert 1 in _claimable("quality")
        _submit(client, "quality", quality_results=[
            {"photo_id": 1, "aesthetic_score": 6.1, "aesthetic_concepts": None}])
    assert _attempts("quality") == {1: MAX_PROCESS_ATTEMPTS}
    assert 1 not in _claimable("quality")


@pytest.mark.parametrize("pass_type", ["quality", "verify", "faces"])
def test_a_failure_row_spends_one_attempt_and_writes_nothing(client, pass_type):
    body = _submit(client, pass_type,
                   failures=[{"photo_id": 2, "error": "cannot identify image file"}])
    assert _attempts(pass_type) == {2: 1}
    assert body["deferred"] == 0
    assert body["written"] == 0
    with _open() as db:
        row = db.conn.execute(
            "SELECT aesthetic_score, verified_at FROM photos WHERE id=2").fetchone()
        n_faces = db.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE photo_id=2").fetchone()[0]
        errs = db.conn.execute(
            "SELECT COUNT(*) FROM index_errors WHERE pass_type=?", (pass_type,)
        ).fetchone()[0]
    assert tuple(row) == (None, None)
    assert n_faces == 0
    assert errs == 1, "the worker's reason must be kept"


def test_failures_ride_alongside_results(client):
    _submit(client, "quality",
            quality_results=[{"photo_id": 1, "aesthetic_score": 6.0,
                              "aesthetic_concepts": "{}"}],
            failures=[{"photo_id": 2, "error": "boom"}])
    assert _attempts("quality") == {1: 1, 2: 1}


def test_verify_success_spends_an_attempt(client):
    _submit(client, "verify", verify_results=[
        {"photo_id": 1, "status": "pass", "verified_at": "2090-04-01T12:00:00"}])
    assert _attempts("verify") == {1: 1}
    assert 1 not in _claimable("verify")


@pytest.mark.parametrize("pass_type,key,row", [
    ("quality", "quality_results",
     {"photo_id": 2, "aesthetic_score": 6.0, "aesthetic_concepts": "{}"}),
    ("verify", "verify_results",
     {"photo_id": 2, "status": "pass", "verified_at": "2091-04-01T12:00:00"}),
])
def test_a_transient_lock_defers_without_spending(client, monkeypatch, pass_type, key, row):
    with _open() as db:
        for _ in range(MAX_PROCESS_ATTEMPTS - 1):
            db.mark_processed([2], pass_type)
    _lock_update(monkeypatch, 2)
    body = _submit(client, pass_type, **{key: [row]})
    assert body["deferred_photo_ids"] == [2]
    assert _attempts(pass_type)[2] == MAX_PROCESS_ATTEMPTS - 1
    assert 2 in _claimable(pass_type), "a server lock must not retire the photo"


def test_a_transient_commit_failure_defers_worker_failures_too(client, monkeypatch):
    """A worker-reported failure is a real outcome, but if the batch COMMIT
    hits a lock nothing is marked — the photo simply comes back."""
    from photosearch.db import PhotoDB
    monkeypatch.setattr(PhotoDB, "end_batch", lambda self: (_ for _ in ()).throw(
        sqlite3.OperationalError("database is locked")))
    body = _submit(client, "quality", failures=[{"photo_id": 2, "error": "x"}])
    assert body["deferred_photo_ids"] == [2]
    assert _attempts("quality") == {}


@pytest.mark.parametrize("pass_type", ["quality", "verify"])
def test_poison_photo_stops_being_claimed_after_the_cap(client, pass_type):
    """End to end through claim-batch: a photo the worker fails on every time
    is offered exactly MAX_PROCESS_ATTEMPTS times, then never again."""
    offered = 0
    for _ in range(MAX_PROCESS_ATTEMPTS + 2):
        claim = client.post("/api/worker/claim-batch", json={
            "worker_id": "w1", "pass_type": pass_type, "limit": 10,
            "directory": "2091"}).json()
        ids = [p["id"] for p in claim["photos"]]
        if 2 not in ids:
            break
        offered += 1
        client.post("/api/worker/submit-results", json={
            "batch_id": claim["batch_id"], "pass_type": pass_type,
            "failures": [{"photo_id": 2, "error": "poison"}]})
    assert offered == MAX_PROCESS_ATTEMPTS
    assert 2 not in _claimable(pass_type)


@pytest.mark.parametrize("pass_type", ["quality", "verify"])
def test_clear_pass_resets_the_ledger(client, pass_type):
    _exhaust(pass_type, [1, 2])
    r = client.post("/api/worker/clear-pass",
                    json={"pass_type": pass_type, "photo_ids": [1]})
    assert r.status_code == 200, r.text
    assert _attempts(pass_type) == {2: MAX_PROCESS_ATTEMPTS}
    assert 1 in _claimable(pass_type)


def test_old_worker_payload_without_failures_still_accepted(client):
    """Additive only: older workers never send `failures`."""
    body = _submit(client, "quality", quality_results=[
        {"photo_id": 1, "aesthetic_score": 6.0}])
    assert body["status"] == "ok"
    assert body["written"] == 1


# ---------------------------------------------------------------------------
# B — faces: empty is terminal, an error is not empty
# ---------------------------------------------------------------------------

def test_empty_faces_result_is_terminal_after_one_submit(client):
    body = _submit(client, "faces", face_results=[{"photo_id": 1, "faces": []}])
    assert body["processed"] == 1
    assert _attempts("faces") == {1: MAX_PROCESS_ATTEMPTS}
    assert 1 not in _claimable("faces")


def test_terminal_mark_never_lowers_a_higher_count(client):
    with _open() as db:
        db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                        "VALUES (1, 'faces', 7)")
        db.conn.commit()
        db.mark_processed([1], "faces", terminal=True)
    assert _attempts("faces") == {1: 7}


def test_non_empty_faces_result_is_unchanged(client):
    enc = [0.01] * 512
    _submit(client, "faces", face_results=[
        {"photo_id": 1, "faces": [{"bbox": [1, 2, 3, 4], "encoding": enc}]}])
    assert _attempts("faces") == {1: 1}
    with _open() as db:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE photo_id=1").fetchone()[0] == 1


def test_faces_detection_error_spends_one_attempt_not_the_cap(client):
    _submit(client, "faces", failures=[{"photo_id": 1, "error": "HIP failure"}])
    assert _attempts("faces") == {1: 1}
    assert 1 in _claimable("faces"), "a transient detection error must be retried"


def test_worker_detection_error_is_a_failure_row_not_an_empty_list(monkeypatch):
    from photosearch import faces as F
    from photosearch import worker as W

    def detect(path, use_cnn=False):
        if path.endswith("bad.jpg"):
            raise RuntimeError("cannot identify image file")
        return []

    monkeypatch.setattr(F, "check_available", lambda: None)
    monkeypatch.setattr(F, "detect_faces", detect)
    results = W._process_faces([
        ({"id": 1, "filename": "ok.jpg"}, "/x/ok.jpg"),
        ({"id": 2, "filename": "bad.jpg"}, "/x/bad.jpg"),
    ])
    kwargs = W._submit_kwargs("face_results", results)
    assert kwargs["face_results"] == [{"photo_id": 1, "faces": []}]
    assert [f["photo_id"] for f in kwargs["failures"]] == [2]
    assert "cannot identify" in kwargs["failures"][0]["error"]
    assert all("faces" not in f for f in kwargs["failures"])


def test_worker_quality_reports_an_unloadable_image(monkeypatch):
    # A stub module, not the real one: photosearch.quality imports torch.nn at
    # module level, which the CI venv's torch mock cannot provide.
    import sys
    import types
    from photosearch import worker as W
    Q = types.ModuleType("photosearch.quality")
    monkeypatch.setitem(sys.modules, "photosearch.quality", Q)

    def score(paths, batch_size=8):
        for i, p in enumerate(paths):
            if not p.endswith("bad.jpg"):
                yield i, 5.5

    def analyze(paths, batch_size=8):
        for i, p in enumerate(paths):
            if not p.endswith("bad.jpg"):
                yield i, {"sharp": 0.3}

    Q.score_photos_stream = score
    Q.analyze_photos_stream = analyze
    results = W._process_quality([
        ({"id": 1, "filename": "ok.jpg"}, "/x/ok.jpg"),
        ({"id": 2, "filename": "bad.jpg"}, "/x/bad.jpg"),
    ])
    kwargs = W._submit_kwargs("quality_results", results)
    assert [r["photo_id"] for r in kwargs["quality_results"]] == [1]
    assert [f["photo_id"] for f in kwargs["failures"]] == [2]


def test_submit_kwargs_omits_failures_when_there_are_none():
    from photosearch import worker as W
    assert W._submit_kwargs("quality_results", [{"photo_id": 1, "aesthetic_score": 5}]) \
        == {"quality_results": [{"photo_id": 1, "aesthetic_score": 5}]}


def test_worker_verify_timeout_defers_but_an_error_is_a_failure(monkeypatch):
    import requests
    from photosearch import describe as D
    from photosearch import worker as W

    monkeypatch.setattr(D, "check_available", lambda model: None)

    class FakeClient:
        def get_photo_detail(self, pid):
            if pid == 1:
                raise requests.exceptions.ReadTimeout("slow")
            raise ValueError("broken detail")

    results = W._process_verify([
        ({"id": 1, "filename": "a.jpg"}, "/x/a.jpg"),
        ({"id": 2, "filename": "b.jpg"}, "/x/b.jpg"),
    ], client=FakeClient())
    kwargs = W._submit_kwargs("verify_results", results)
    assert kwargs["verify_results"] == []
    assert [f["photo_id"] for f in kwargs["failures"]] == [2], \
        "the timeout is omitted (deferred); only the real error is reported"


# ---------------------------------------------------------------------------
# B — in-process index.py writers share the contract
# ---------------------------------------------------------------------------

def test_in_process_faces_marks_match_the_server(client):
    from photosearch.index import (_clear_faces_ledger, _faces_exhausted,
                                   _mark_faces_outcomes)
    with _open() as db:
        _mark_faces_outcomes(db, no_face_ids=[1], failed_ids=[2])
    assert _attempts("faces") == {1: MAX_PROCESS_ATTEMPTS, 2: 1}
    with _open() as db:
        assert _faces_exhausted(db, [1, 2, 3]) == {1}
        _clear_faces_ledger(db, [1])
        db.conn.commit()
    assert _attempts("faces") == {2: 1}


# ---------------------------------------------------------------------------
# clip joins the ledger (it used to be the one uncapped pass)
# ---------------------------------------------------------------------------

def _emb(seed=0):
    v = [0.0] * 512
    v[seed] = 1.0
    return v


def test_clip_exhausted_photo_is_neither_claimed_nor_counted(client):
    assert 1 in _claimable("clip")
    before = _count("clip")
    _exhaust("clip", [1])
    assert 1 not in _claimable("clip")
    assert _count("clip") == before - 1
    assert _count("clip", [1, 2]) == 1


def test_clip_success_is_its_own_marker(client):
    """A stored embedding needs no ledger row — the highest-volume pass should
    not write one per photo on the N100."""
    body = _submit(client, "clip", clip_results=[{"photo_id": 1, "embedding": _emb()}])
    assert body["written"] == 1
    assert body["processed"] == 1
    assert _attempts("clip") == {}
    assert 1 not in _claimable("clip")


def test_clip_failure_row_spends_one_attempt(client):
    body = _submit(client, "clip", failures=[{"photo_id": 2, "error": "cannot identify"}])
    assert _attempts("clip") == {2: 1}
    assert body["deferred"] == 0
    assert 2 in _claimable("clip"), "one failure must still be retried"


def test_clip_transient_lock_defers_without_spending(client, monkeypatch):
    from photosearch.db import PhotoDB
    real = PhotoDB.add_clip_embedding

    def flaky(self, photo_id, *a, **kw):
        if photo_id == 2:
            raise sqlite3.OperationalError("database is locked")
        return real(self, photo_id, *a, **kw)

    monkeypatch.setattr(PhotoDB, "add_clip_embedding", flaky)
    with _open() as db:
        for _ in range(MAX_PROCESS_ATTEMPTS - 1):
            db.mark_processed([2], "clip")
    body = _submit(client, "clip", clip_results=[
        {"photo_id": 1, "embedding": _emb(1)}, {"photo_id": 2, "embedding": _emb(2)}])
    assert body["deferred_photo_ids"] == [2]
    assert _attempts("clip") == {2: MAX_PROCESS_ATTEMPTS - 1}
    assert 2 in _claimable("clip")


def test_clip_non_lock_write_error_spends_an_attempt(client):
    """A malformed vector is a repeatable failure — capped, not deferred."""
    _submit(client, "clip", clip_results=[{"photo_id": 2, "embedding": [0.1] * 7}])
    assert _attempts("clip") == {2: 1}


def test_poison_clip_photo_stops_being_claimed_and_batch_reads_blocked(client):
    from photosearch.batch_state import batch_state
    from photosearch.ingest_batches import register_batch

    offered = 0
    for _ in range(MAX_PROCESS_ATTEMPTS + 2):
        claim = client.post("/api/worker/claim-batch", json={
            "worker_id": "w1", "pass_type": "clip", "limit": 10,
            "directory": "2091"}).json()
        ids = [p["id"] for p in claim["photos"]]
        if not ids:
            break
        if 2 in ids:
            offered += 1
        client.post("/api/worker/submit-results", json={
            "batch_id": claim["batch_id"], "pass_type": "clip",
            "clip_results": [{"photo_id": i, "embedding": _emb(i)}
                             for i in ids if i != 2],
            "failures": [{"photo_id": 2, "error": "PK zip, not a JPEG"}]})
    assert offered == MAX_PROCESS_ATTEMPTS
    assert 2 not in _claimable("clip")

    with _open() as db:
        db.conn.execute("UPDATE photos SET folder='2091' WHERE id IN (2,3)")
        db.conn.commit()
        batch_id = register_batch(db, "2091")
        step = next(s for s in batch_state(db, batch_id)["steps"] if s["step"] == "clip")
    assert step["failed"] == 1
    assert step["done"] == 1
    assert step["state"] == "blocked"


def test_clear_pass_resets_the_clip_ledger(client):
    _exhaust("clip", [1, 2])
    r = client.post("/api/worker/clear-pass", json={"pass_type": "clip", "photo_ids": [1]})
    assert r.status_code == 200, r.text
    assert _attempts("clip") == {2: MAX_PROCESS_ATTEMPTS}
    assert 1 in _claimable("clip")


def test_worker_clip_reports_an_unloadable_image(monkeypatch):
    from photosearch import clip_embed as C
    from photosearch import worker as W

    def stream(paths, batch_size=8):
        for i, p in enumerate(paths):
            if not p.endswith("bad.jpg"):
                yield i, _emb(i)

    monkeypatch.setattr(C, "embed_images_stream", stream)
    results = W._process_clip([
        ({"id": 1, "filename": "ok.jpg"}, "/x/ok.jpg"),
        ({"id": 2, "filename": "bad.jpg"}, "/x/bad.jpg"),
    ])
    kwargs = W._submit_kwargs("clip_results", results)
    assert [r["photo_id"] for r in kwargs["clip_results"]] == [1]
    assert [f["photo_id"] for f in kwargs["failures"]] == [2]

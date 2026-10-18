"""Tests for photosearch/batch_api.py (M "ingest batch" Task 4).

The cache is the point of this module — see its docstring. Two shapes get
exercised directly:

  1. two GETs inside the TTL must call ``batch_state`` exactly once
     (memoized read);
  2. a GET while the module lock is held by someone else must return
     promptly with ``stale: True`` and must NOT call ``batch_state`` — this
     is the direct regression test for the 2026-09-19 incident where
     stacked status polls filled all 40 request threads.

Per globals.md the shared ``db`` fixture is pre-seeded under ``2026/``, so
every fixture here builds photos under ``2090/``.
"""

import pytest

from photosearch import batch_api, ingest_batches

_DIR = "2090/2090-01-01_TEST"


@pytest.fixture(autouse=True)
def _reset_batch_cache():
    """The module-level memo/lock are process-global state — reset around
    every test so two different DB paths (or two tests) can't serve each
    other's data, and so a test that holds the lock never leaks it."""
    batch_api._reset_cache()
    yield
    if batch_api._lock.locked():
        batch_api._lock.release()
    batch_api._reset_cache()


def _make_batch(db, directory: str = _DIR, count: int = 3, **kw):
    """Add `count` photos under `directory` and register the batch."""
    ids = []
    for i in range(count):
        ids.append(db.add_photo(
            filepath=f"{directory}/img{i:03d}.jpg",
            filename=f"img{i:03d}.jpg",
            date_taken=f"2090-01-01T10:0{i}:00",
        ))
    batch_id = ingest_batches.register_batch(db, directory, **kw)
    return batch_id, ids


# =========================================================================
# GET /api/batches — pure SQL, no derivation
# =========================================================================

class TestListBatches:
    def test_no_sweep_and_lists_registered_batch(self, client, db):
        batch_id, _ = _make_batch(db)
        resp = client.get("/api/batches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sweep"] is None
        assert any(b["id"] == batch_id for b in data["batches"])

    def test_active_sweep_surfaced(self, client, db):
        run_id = ingest_batches.start_sweep(db)
        resp = client.get("/api/batches")
        data = resp.json()
        assert data["sweep"] is not None
        assert data["sweep"]["run_id"] == run_id
        assert data["sweep"]["status"] == "moving"

    def test_dismissed_hidden_by_default_and_included_on_request(self, client, db):
        batch_id, _ = _make_batch(db)
        ingest_batches.dismiss_batch(db, batch_id)

        resp = client.get("/api/batches")
        assert all(b["id"] != batch_id for b in resp.json()["batches"])

        resp2 = client.get("/api/batches", params={"include_dismissed": 1})
        assert any(b["id"] == batch_id for b in resp2.json()["batches"])


# =========================================================================
# GET /api/batches/{id} — cached derived state
# =========================================================================

class TestGetBatchStatus:
    def test_404_unknown_batch(self, client):
        resp = client.get("/api/batches/999999")
        assert resp.status_code == 404

    def test_shape(self, client, db):
        batch_id, _ = _make_batch(db)
        resp = client.get(f"/api/batches/{batch_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch"]["id"] == batch_id
        assert data["stale"] is False
        assert "computed_at" in data
        assert isinstance(data["steps"], list) and data["steps"]
        assert "ready" in data
        assert "next_action" in data

    def test_two_gets_within_ttl_call_batch_state_once(self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)

        calls = {"n": 0}
        real = batch_api._batch_state

        def counting(db_, bid):
            calls["n"] += 1
            return real(db_, bid)

        monkeypatch.setattr(batch_api, "_batch_state", counting)

        resp1 = client.get(f"/api/batches/{batch_id}")
        resp2 = client.get(f"/api/batches/{batch_id}")

        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert calls["n"] == 1
        assert resp1.json() == resp2.json()

    def test_lock_busy_with_existing_memo_returns_stale_without_computing(
        self, client, db, monkeypatch
    ):
        batch_id, _ = _make_batch(db)
        first = client.get(f"/api/batches/{batch_id}").json()
        assert first["stale"] is False

        # Force the memo entry to look expired so the next GET attempts a
        # recompute (and hits the busy lock) instead of a cheap TTL hit.
        key = batch_api._key(batch_id)
        computed_at, cached_value = batch_api._memo[key]
        batch_api._memo[key] = (
            computed_at - batch_api._TTL_SECONDS - 1, cached_value
        )

        calls = {"n": 0}
        real = batch_api._batch_state

        def counting(db_, bid):
            calls["n"] += 1
            return real(db_, bid)

        monkeypatch.setattr(batch_api, "_batch_state", counting)

        # Hold the lock from the TEST thread — the endpoint runs its sync
        # `def` in FastAPI's threadpool, so this reproduces the incident's
        # "another request is already deriving" shape without needing a
        # second Thread object.
        batch_api._lock.acquire()
        try:
            resp = client.get(f"/api/batches/{batch_id}")
        finally:
            batch_api._lock.release()

        assert resp.status_code == 200
        data = resp.json()
        assert data["stale"] is True
        assert data["batch"]["id"] == batch_id
        assert data["computed_at"] == first["computed_at"]  # unchanged, still old value
        assert calls["n"] == 0  # never recomputed

    def test_lock_busy_with_no_memo_returns_minimal_placeholder(self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)

        calls = {"n": 0}
        real = batch_api._batch_state

        def counting(db_, bid):
            calls["n"] += 1
            return real(db_, bid)

        monkeypatch.setattr(batch_api, "_batch_state", counting)

        batch_api._lock.acquire()
        try:
            resp = client.get(f"/api/batches/{batch_id}")
        finally:
            batch_api._lock.release()

        assert resp.status_code == 200
        data = resp.json()
        assert data["stale"] is True
        assert data["computing"] is True
        assert data["steps"] == []
        assert data["batch"]["id"] == batch_id
        assert calls["n"] == 0

    def test_lock_busy_no_memo_unknown_batch_still_404s(self, client):
        batch_api._lock.acquire()
        try:
            resp = client.get("/api/batches/999999")
        finally:
            batch_api._lock.release()
        assert resp.status_code == 404


# =========================================================================
# Write endpoints
# =========================================================================

class TestWriteEndpoints:
    def test_dismiss_invalidates_cache_and_hides_from_list(self, client, db):
        batch_id, _ = _make_batch(db)
        client.get(f"/api/batches/{batch_id}")
        assert batch_api._key(batch_id) in batch_api._memo

        resp = client.post(f"/api/batches/{batch_id}/dismiss")
        assert resp.status_code == 200
        assert batch_api._key(batch_id) not in batch_api._memo

        listing = client.get("/api/batches").json()
        assert all(b["id"] != batch_id for b in listing["batches"])

    def test_dismiss_unknown_batch_404(self, client):
        resp = client.post("/api/batches/999999/dismiss")
        assert resp.status_code == 404

    def test_ready_invalidates_cache(self, client, db):
        batch_id, _ = _make_batch(db)
        client.get(f"/api/batches/{batch_id}")
        assert batch_api._key(batch_id) in batch_api._memo

        resp = client.post(f"/api/batches/{batch_id}/ready")
        assert resp.status_code == 200
        assert batch_api._key(batch_id) not in batch_api._memo
        assert resp.json()["batch"]["ready_at"] is not None

    def test_ready_unknown_batch_404(self, client):
        resp = client.post("/api/batches/999999/ready")
        assert resp.status_code == 404

    def test_register_400_on_directory_with_no_photos(self, client, db):
        resp = client.post("/api/batches/register", json={"directory": "2090/nope"})
        assert resp.status_code == 400

    def test_register_adopts_existing_folder(self, client, db):
        directory = "2090/2090-02-02_adopt"
        for i in range(2):
            db.add_photo(
                filepath=f"{directory}/img{i:03d}.jpg",
                filename=f"img{i:03d}.jpg",
                date_taken=f"2090-02-02T10:0{i}:00",
            )
        resp = client.post(
            "/api/batches/register",
            json={"directory": directory, "source": "ingest-incoming"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["batch"]["directory"] == directory
        assert data["batch"]["photo_count"] == 2

    def test_register_invalidates_memo_when_widening_existing_batch(self, client, db):
        batch_id, _ = _make_batch(db, count=1)
        client.get(f"/api/batches/{batch_id}")
        assert batch_api._key(batch_id) in batch_api._memo

        db.add_photo(
            filepath=f"{_DIR}/img999.jpg", filename="img999.jpg",
            date_taken="2090-01-01T10:09:00",
        )
        resp = client.post("/api/batches/register", json={"directory": _DIR})
        assert resp.status_code == 200
        assert resp.json()["batch_id"] == batch_id
        assert batch_api._key(batch_id) not in batch_api._memo


# =========================================================================
# Fix round 1 — review findings
# =========================================================================

class TestGenerationGuardsAgainstLostInvalidation:
    """FINDING 1: a write's invalidation must never be clobbered by an
    in-flight recompute that started before it. Simulated deterministically
    (no threads/sleeps) by making the FIRST call into the real
    ``batch_state`` also perform the invalidation itself mid-derivation —
    exactly the interleaving the finding describes: thread A is already
    inside ``_batch_state`` when thread B's write commits and invalidates.
    """

    def test_invalidation_during_compute_is_not_cached(self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)
        key = batch_api._key(batch_id)

        original = batch_api._batch_state
        calls = {"n": 0}

        def racing(db_, bid):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate a concurrent write's invalidation landing while
                # THIS call is still mid-derivation — before the caller has
                # had a chance to store anything in the memo.
                batch_api._invalidate(key)
            return original(db_, bid)

        monkeypatch.setattr(batch_api, "_batch_state", racing)

        first = client.get(f"/api/batches/{batch_id}")
        assert first.status_code == 200
        assert first.json()["stale"] is False  # still a valid, freshly-computed response

        # The critical assertion: despite computing successfully, the
        # response must NOT have been cached, because the generation
        # changed underneath it mid-compute.
        assert key not in batch_api._memo
        assert calls["n"] == 1

        # So the very next GET must recompute rather than serve a
        # resurrected pre-write snapshot for another _TTL_SECONDS.
        second = client.get(f"/api/batches/{batch_id}")
        assert second.status_code == 200
        assert calls["n"] == 2
        # This time nothing raced it, so it's cached normally.
        assert key in batch_api._memo


class TestMemoKeyedByDbPath:
    """FINDING 2: the memo must be keyed on (db_path, batch_id), not
    batch_id alone — batch_id is only unique within one DB file. Two
    independent DB files, each minting batch_id 1 for their first batch,
    must not serve each other's cached state. Deliberately does NOT call
    `_reset_cache()` between the two DBs — that's the scenario the finding
    describes (the next task's tests, in another file, would otherwise
    inherit this file's memo).
    """

    def test_two_db_paths_same_batch_id_do_not_share_memo(self, client, db, tmp_path):
        from photosearch import web
        from photosearch.db import PhotoDB

        batch_id, _ = _make_batch(db)
        first = client.get(f"/api/batches/{batch_id}")
        assert first.status_code == 200
        first_data = first.json()
        assert first_data["batch"]["directory"] == _DIR

        other_db_path = str(tmp_path / "other.db")
        other_db = PhotoDB(other_db_path)
        other_db.set_photo_root("/photos")
        other_dir = "2091/2091-01-01_OTHER"
        for i in range(2):
            other_db.add_photo(
                filepath=f"{other_dir}/img{i:03d}.jpg",
                filename=f"img{i:03d}.jpg",
                date_taken=f"2091-01-01T10:0{i}:00",
            )
        other_batch_id = ingest_batches.register_batch(other_db, other_dir)
        # The whole point: same numeric id as the first DB's batch (both are
        # the first row ever inserted into a fresh `ingest_batches` table).
        assert other_batch_id == batch_id

        original_db_path = web._db_path
        original_photo_root = web._photo_root
        try:
            web._db_path = other_db_path
            web._photo_root = "/photos"
            resp = client.get(f"/api/batches/{other_batch_id}")
        finally:
            web._db_path = original_db_path
            web._photo_root = original_photo_root
            other_db.close()

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch"]["directory"] == other_dir
        assert data["batch"]["directory"] != first_data["batch"]["directory"]


# =========================================================================
# GET /{id}/blocked and POST /{id}/invalidate (docs/plans/batch-blocked-rerun.md)
# =========================================================================

def _exhaust(db, ids, pass_type):
    from photosearch.db import MAX_PROCESS_ATTEMPTS
    for pid in ids:
        db.conn.execute(
            "INSERT OR REPLACE INTO worker_processed (photo_id, pass_type, attempts) "
            "VALUES (?, ?, ?)", (pid, pass_type, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()


class TestBlockedEndpoint:
    def test_lists_the_blocked_photos(self, client, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids[:1], "describe")
        db.log_error("describe", str(ids[0]), "worker: description ends mid-sentence")
        resp = client.get(f"/api/batches/{batch_id}/blocked",
                          params={"step": "describe"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1 and data["truncated"] is False
        assert data["photos"] == [{
            "photo_id": ids[0], "filename": "img000.jpg", "attempts": 3,
            "last_error": "worker: description ends mid-sentence"}]

    @pytest.mark.parametrize("step", ["stacking", "ingest", "bogus"])
    def test_400_on_a_non_worker_pass(self, client, db, step):
        batch_id, _ = _make_batch(db)
        resp = client.get(f"/api/batches/{batch_id}/blocked", params={"step": step})
        assert resp.status_code == 400

    def test_404_unknown_batch(self, client):
        resp = client.get("/api/batches/999999/blocked", params={"step": "describe"})
        assert resp.status_code == 404

    def test_faces_is_an_empty_list_not_an_error(self, client, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "faces")   # exhausted with no face rows == done
        resp = client.get(f"/api/batches/{batch_id}/blocked", params={"step": "faces"})
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "truncated": False, "photos": []}


class TestInvalidateEndpoint:
    def test_invalidate_clears_the_memo_and_the_next_get_recomputes(
            self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)
        calls = {"n": 0}
        real = batch_api._batch_state

        def counting(db_, bid):
            calls["n"] += 1
            return real(db_, bid)

        monkeypatch.setattr(batch_api, "_batch_state", counting)

        client.get(f"/api/batches/{batch_id}")
        client.get(f"/api/batches/{batch_id}")
        assert calls["n"] == 1                       # memoized

        resp = client.post(f"/api/batches/{batch_id}/invalidate")
        assert resp.status_code == 200
        assert resp.json() == {"batch_id": batch_id, "invalidated": True}
        assert batch_api._key(batch_id) not in batch_api._memo

        client.get(f"/api/batches/{batch_id}")
        assert calls["n"] == 2                       # recomputed after invalidate

    def test_invalidate_bumps_the_generation(self, client, db):
        """So a recompute already in flight cannot cache its pre-re-run
        snapshot (TestGenerationGuardsAgainstLostInvalidation)."""
        batch_id, _ = _make_batch(db)
        key = batch_api._key(batch_id)
        before = batch_api._generation.get(key, 0)
        client.post(f"/api/batches/{batch_id}/invalidate")
        assert batch_api._generation.get(key, 0) == before + 1

    def test_404_unknown_batch(self, client):
        resp = client.post("/api/batches/999999/invalidate")
        assert resp.status_code == 404


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture
def nas(monkeypatch):
    """Replica mode with a recording stand-in for `requests.request`."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.example:8000/")
    calls = []
    box = {"response": _FakeResponse({"ok": True})}

    def fake_request(method, url, params=None, json=None, timeout=None,
                     headers=None):
        calls.append({"method": method, "url": url, "params": params})
        return box["response"]

    import requests
    monkeypatch.setattr(requests, "request", fake_request)
    return {"calls": calls, "box": box}


class TestBlockedAndInvalidateProxy:
    def test_blocked_comes_from_the_nas_not_the_local_copy(self, client, db, nas):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "describe")   # local copy says 3 — must NOT be used
        nas["box"]["response"] = _FakeResponse(
            {"total": 1, "truncated": False, "photos": [
                {"photo_id": 77, "filename": "x.jpg", "attempts": 3,
                 "last_error": "worker: boom"}]})
        data = client.get(f"/api/batches/{batch_id}/blocked",
                          params={"step": "describe"}).json()
        assert data["total"] == 1 and data["photos"][0]["photo_id"] == 77
        call = nas["calls"][0]
        assert call["method"] == "GET"
        assert call["url"] == f"http://nas.example:8000/api/batches/{batch_id}/blocked"
        assert call["params"] == {"step": "describe"}

    def test_blocked_validation_is_the_nas_s_job(self, client, db, nas):
        """The NAS's own 400 survives the hop; the replica does not answer
        for it locally."""
        nas["box"]["response"] = _FakeResponse(
            {"detail": "not a worker pass: stacking"}, status_code=400)
        r = client.get("/api/batches/1/blocked", params={"step": "stacking"})
        assert r.status_code == 400
        assert r.json()["detail"] == "not a worker pass: stacking"
        assert nas["calls"], "did not proxy"

    def test_invalidate_goes_to_the_nas(self, client, db, nas):
        """The memo that matters is the NAS's — the replica's /batches
        reads are proxied and never touch its own memo."""
        batch_id, _ = _make_batch(db)
        nas["box"]["response"] = _FakeResponse(
            {"batch_id": batch_id, "invalidated": True})
        resp = client.post(f"/api/batches/{batch_id}/invalidate")
        assert resp.status_code == 200
        assert resp.json() == {"batch_id": batch_id, "invalidated": True}
        assert nas["calls"][0]["method"] == "POST"
        assert nas["calls"][0]["url"].endswith(f"/api/batches/{batch_id}/invalidate")

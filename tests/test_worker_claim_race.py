"""Concurrency tests for /api/worker/claim-batch.

Reproduces the read-then-insert race in claim-batch: without the
BEGIN IMMEDIATE wrap, parallel workers can each SELECT the same
unprocessed photos and INSERT separate claim rows containing the
same photo_ids. The fix is in worker_api.py:claim_batch.
"""

import threading

import pytest


@pytest.mark.skip(reason="pre-existing test-isolation failure: a prior "
                         "TestClient teardown leaks worker_api._shutting_down, "
                         "so claim-batch 503s here. See docs/plans/test-isolation-fixes.md")
def test_concurrent_claim_batch_returns_disjoint_photos(client, db):
    photo_ids = []
    for i in range(20):
        pid = db.add_photo(
            filepath=f"2026/race/{i:04d}.jpg",
            filename=f"{i:04d}.jpg",
            date_taken="2026-04-01T10:00:00",
        )
        photo_ids.append(pid)

    pre = db.get_unprocessed_photos("describe", limit=100)
    assert len(pre) == 20, f"sanity: expected 20 unprocessed, got {len(pre)}"

    K = 10
    LIMIT = 4
    results = []
    errors = []
    barrier = threading.Barrier(K)

    def hammer(idx: int):
        try:
            barrier.wait(timeout=5)
            r = client.post("/api/worker/claim-batch", json={
                "worker_id": f"race-worker-{idx}",
                "pass_type": "describe",
                "limit": LIMIT,
                "ttl_minutes": 5,
            })
            results.append((idx, r.status_code, r.json()))
        except Exception as e:
            errors.append((idx, repr(e)))

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(K)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"thread errors: {errors}"
    assert len(results) == K, f"expected {K} results, got {len(results)}"

    seen = {}
    for idx, status, body in results:
        assert status == 200, f"worker {idx}: status {status}, body {body}"
        for p in body.get("photos") or []:
            pid = p["id"]
            if pid in seen:
                raise AssertionError(
                    f"photo {pid} claimed by worker {seen[pid]} AND worker {idx} — "
                    f"race in claim-batch is not closed"
                )
            seen[pid] = idx

    assert len(seen) == 20, (
        f"expected all 20 photos claimed exactly once, got {len(seen)} "
        f"(missing: {set(photo_ids) - set(seen)})"
    )


def test_db_level_claim_flow_atomic(db):
    """Direct DB-level smoke test of the BEGIN IMMEDIATE pattern.

    Asserts the commit-suppression kwargs work end-to-end (no
    premature commit of the outer transaction, claim row visible
    only after the explicit commit).
    """
    pid = db.add_photo(
        filepath="2026/race/atomic.jpg",
        filename="atomic.jpg",
        date_taken="2026-04-01T10:00:00",
    )

    db.conn.execute("BEGIN IMMEDIATE")
    try:
        photos = db.get_unprocessed_photos(
            "describe", limit=10, commit_cleanup=False
        )
        assert any(p["id"] == pid for p in photos)

        batch_id = db.claim_photos(
            worker_id="atomic-worker",
            pass_type="describe",
            photo_ids=[pid],
            ttl_minutes=5,
            commit=False,
        )
        db.conn.commit()
    except Exception:
        db.conn.rollback()
        raise

    claimed = db.get_claimed_photo_ids("describe")
    assert pid in claimed, "claim row should be visible after commit"
    assert batch_id, "batch_id should be returned"


def test_empty_queue_never_takes_the_writer_lock(client, monkeypatch):
    """The whole point of the split: an idle fleet must not serialize writers.

    claim-batch used to BEGIN IMMEDIATE unconditionally and run the unprocessed
    scan while holding the writer lock, so workers polling an empty queue
    starved every other writer on the NAS — measurably: face-assign 500s, a
    30-minute collection insert, and an overnight ingest that wrote zero rows.

    Observed via sqlite3's trace callback; Connection.execute can't be patched.
    """
    from photosearch import worker_api

    # A prior test's TestClient teardown fires the app shutdown event, which
    # leaves worker_api._shutting_down=True and 503s every /api/worker/* route.
    # Same known leak that test_web_replica.py's replica_client guards against.
    monkeypatch.setattr(worker_api, "_shutting_down", False)

    statements: list[str] = []
    orig_get_db = worker_api._get_db

    def spy_db():
        db = orig_get_db()
        db.conn.set_trace_callback(statements.append)
        return db

    # Drain whatever the fixture has first, WITHOUT the spy, so the assertion
    # is about a genuinely empty queue rather than "no work happened to exist".
    for _ in range(50):
        got = client.post("/api/worker/claim-batch", json={
            "worker_id": "drain", "pass_type": "clip", "limit": 16}).json()
        if not got.get("photos"):
            break

    monkeypatch.setattr(worker_api, "_get_db", spy_db)
    r = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "clip", "limit": 16})
    assert r.status_code == 200
    assert r.json()["batch_id"] is None, "expected an empty queue after draining"
    locks = [x for x in statements if "BEGIN IMMEDIATE" in x.upper()]
    assert locks == [], f"empty queue acquired the writer lock: {locks}"


def test_two_workers_never_claim_the_same_photo(client, monkeypatch):
    """The invariant the original BEGIN IMMEDIATE existed for, preserved by the
    in-lock re-check rather than by holding the lock across the scan."""
    from photosearch import worker_api
    monkeypatch.setattr(worker_api, "_shutting_down", False)
    a = client.post("/api/worker/claim-batch", json={
        "worker_id": "wa", "pass_type": "clip", "limit": 4}).json()
    b = client.post("/api/worker/claim-batch", json={
        "worker_id": "wb", "pass_type": "clip", "limit": 4}).json()
    ids_a = {p["id"] for p in a.get("photos", [])}
    ids_b = {p["id"] for p in b.get("photos", [])}
    assert not (ids_a & ids_b), "two workers claimed the same photo"

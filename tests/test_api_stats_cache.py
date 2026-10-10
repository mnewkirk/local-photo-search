"""/api/stats: cheap queries, a stale-while-revalidate memo, and /api/health.

/api/stats used to read the whole `photos` table four times per call, and the
worker fleet called it at startup from every worker at once (3 x 117 s on a
cold NAS, 2026-10-07). These pin the replacement:

- the counters still equal the original per-counter queries,
- `photos` is read by at most one table scan,
- a memoized value is served without recomputing, and a stale one is served
  immediately while one background thread refreshes it,
- the worker probes /api/health (no DB), and tolerates an older server's 404.
"""

import json
import threading
import time

import pytest


def _old_stats(conn):
    """The pre-2026-10-09 per-counter SQL, kept as the oracle."""
    one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
    out = {
        "described": one("SELECT COUNT(*) FROM photos WHERE description IS NOT NULL"),
        "concepts_analyzed": one(
            "SELECT COUNT(*) FROM photos WHERE aesthetic_concepts IS NOT NULL"),
        "aesthetics_scored": one(
            "SELECT COUNT(*) FROM photos WHERE aes_overall IS NOT NULL"),
    }
    row = conn.execute(
        "SELECT SUM(CASE WHEN aes_overall_pct IS NOT NULL THEN 1 ELSE 0 END) "
        "FROM photos WHERE aes_overall IS NOT NULL").fetchone()
    out["normalized"] = row[0]
    counts = {"pass": 0, "fail": 0, "regenerated": 0}
    for status, c in conn.execute(
            "SELECT verification_status, COUNT(*) FROM photos "
            "WHERE verified_at IS NOT NULL GROUP BY verification_status"):
        st = status or "pass"
        if st in counts:
            counts[st] += c
    out.update(verify_passed=counts["pass"], verify_failed=counts["fail"],
               verify_regenerated=counts["regenerated"])
    return out


def _vary(db):
    """Spread values across the columns the merged scan counts."""
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id")]
    db.conn.execute("UPDATE photos SET description = NULL WHERE id = ?", (ids[0],))
    db.conn.execute("UPDATE photos SET aesthetic_concepts = '{}' WHERE id IN (?, ?)",
                    (ids[1], ids[2]))
    db.conn.execute("UPDATE photos SET aes_overall = 6.5 WHERE id IN (?, ?, ?)",
                    tuple(ids[:3]))
    db.conn.execute("UPDATE photos SET aes_overall_pct = 0.5 WHERE id = ?", (ids[1],))
    for pid, status in zip(ids, [None, "pass", "fail", "regenerated", "weird"]):
        db.conn.execute(
            "UPDATE photos SET verified_at = '2026-10-09', verification_status = ? "
            "WHERE id = ?", (status, pid))
    db.conn.commit()


def test_counters_match_the_original_queries(client, db):
    _vary(db)
    body = client.get("/api/stats").json()
    expected = _old_stats(db.conn)
    for key in ("described", "concepts_analyzed", "aesthetics_scored",
                "verify_passed", "verify_failed", "verify_regenerated"):
        assert body[key] == expected[key], key
    assert body["aesthetics_stats"]["normalized"] == expected["normalized"]
    assert body["verify_passed"] == 2  # NULL status counts as a pass
    assert "computed_at" in body and "ask_available" in body


def test_photos_is_scanned_at_most_once(db):
    from photosearch.web import _compute_stats

    _vary(db)
    statements = []
    db.conn.set_trace_callback(statements.append)
    try:
        _compute_stats(db)
    finally:
        db.conn.set_trace_callback(None)

    scans = []
    for sql in statements:
        if "FROM photos" not in sql:
            continue
        for row in db.conn.execute("EXPLAIN QUERY PLAN " + sql):
            detail = row[3]
            if detail.startswith("SCAN photos") and "INDEX" not in detail:
                scans.append(sql)
    assert len(scans) == 1, scans


def test_memo_serves_without_recomputing(client, db, monkeypatch):
    from photosearch import web

    first = client.get("/api/stats").json()
    calls = []
    monkeypatch.setattr(web, "_compute_stats",
                        lambda d: calls.append(1) or {"photos": -1})
    second = client.get("/api/stats").json()
    assert calls == []
    assert second["photos"] == first["photos"]
    assert second["computed_at"] == first["computed_at"]


def test_stale_value_is_served_while_one_thread_refreshes(client, db, monkeypatch):
    from photosearch import web

    first = client.get("/api/stats").json()
    # Age the memo past the TTL.
    ts, payload = web._stats_memo[web._db_path]
    web._stats_memo[web._db_path] = (ts - web._STATS_TTL_SECONDS - 1, payload)

    gate = threading.Event()
    calls = []

    def slow_compute(d):
        calls.append(1)
        gate.wait(5)
        return {**payload, "photos": 999}

    monkeypatch.setattr(web, "_compute_stats", slow_compute)
    # Several stale hits: all answer immediately with the old value, and only
    # one refresh starts.
    for _ in range(3):
        body = client.get("/api/stats").json()
        assert body["photos"] == first["photos"]
    deadline = time.time() + 2
    while not calls and time.time() < deadline:
        time.sleep(0.01)
    assert len(calls) == 1

    gate.set()
    deadline = time.time() + 5
    while web._stats_lock.locked() and time.time() < deadline:
        time.sleep(0.01)
    assert client.get("/api/stats").json()["photos"] == 999


def test_concurrent_first_requests_compute_once(client, db, monkeypatch):
    from photosearch import web

    web._stats_memo.clear()
    real = web._compute_stats
    calls = []

    def counted(d):
        calls.append(1)
        time.sleep(0.2)
        return real(d)

    monkeypatch.setattr(web, "_compute_stats", counted)
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        client.get("/api/stats").json()["photos"])) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == 4 and len(set(results)) == 1
    assert len(calls) == 1


def test_health_touches_no_db(client, monkeypatch):
    from photosearch import web

    def boom(*a, **k):
        raise AssertionError("health must not open the DB")
    monkeypatch.setattr(web, "_get_db", boom)
    monkeypatch.setattr(web, "PhotoDB", boom)
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json() == {"ok": True}


class _Resp:
    def __init__(self, code):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")


@pytest.mark.parametrize("code", [200, 404])
def test_worker_probe_uses_health_and_tolerates_old_server(monkeypatch, code):
    import requests
    from photosearch.worker import WorkerClient

    urls = []

    def fake_get(self, url, **k):
        urls.append(url)
        return _Resp(code)

    monkeypatch.setattr(requests.Session, "get", fake_get)
    WorkerClient("http://nas:8000")
    assert urls == ["http://nas:8000/api/health"]


def test_worker_probe_fails_on_server_error(monkeypatch):
    import requests
    from photosearch.worker import WorkerClient

    monkeypatch.setattr(requests.Session, "get", lambda self, url, **k: _Resp(500))
    with pytest.raises(ConnectionError):
        WorkerClient("http://nas:8000")


def test_replica_status_reads_the_nas_fingerprint(client, monkeypatch, tmp_path):
    import io
    import urllib.request

    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://fake-nas:8000")
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req.full_url)
        return io.BytesIO(json.dumps({"photo_count": 42}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    body = client.get("/api/admin/replica-status").json()
    assert seen == ["http://fake-nas:8000/api/admin/maintenance-fingerprint"]
    assert body["nas_photos"] == 42

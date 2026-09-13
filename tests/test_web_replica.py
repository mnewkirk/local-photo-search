"""Tests for replica-mode image serving (M26a).

When this instance runs off a synced read-replica (PHOTOSEARCH_NAS_URL set) on
a machine without the original photo files, the thumbnail/preview/full routes
fall back to fetching the rendered asset from the source NAS (and cache the
thumbnail/preview locally). The conftest fixture photos have no files on disk,
so the "original missing" path is always exercised.
"""

import os

import pytest

from photosearch import web, worker_api


@pytest.fixture
def replica_client(client, monkeypatch):
    """The standard TestClient, but with replica mode pointed at a fake NAS."""
    # A prior test's TestClient teardown fires the app shutdown event, which
    # sets worker_api._shutting_down=True and makes the middleware 503 the
    # /full route. Clear it so these tests aren't a victim of that leak.
    monkeypatch.setattr(worker_api, "_shutting_down", False)
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")
    fetched = {"calls": []}

    def fake_fetch(photo_id, kind, timeout=30.0):
        fetched["calls"].append((photo_id, kind))
        return b"\xff\xd8\xff" + f"{kind}-{photo_id}".encode()  # fake JPEG-ish bytes

    monkeypatch.setattr(web, "_fetch_from_nas", fake_fetch)
    client._fetched = fetched
    return client


def _first_photo_id(client):
    r = client.get("/api/search?q=&date_from=2026-03-13")
    # fall back to a known fixture id via stats if search empty
    data = client.get("/api/stats").json()
    assert data["photos"] >= 1
    # the search endpoint returns ids; pick the first
    res = client.get("/api/search?location=Big Sur").json()
    return res["results"][0]["id"]


def test_thumbnail_proxies_from_nas_and_caches(replica_client, tmp_path):
    pid = _first_photo_id(replica_client)
    r = replica_client.get(f"/api/photos/{pid}/thumbnail")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert (pid, "thumbnail") in replica_client._fetched["calls"]
    # Cached locally → a second request should NOT hit the NAS again.
    replica_client._fetched["calls"].clear()
    r2 = replica_client.get(f"/api/photos/{pid}/thumbnail")
    assert r2.status_code == 200
    assert replica_client._fetched["calls"] == []


def test_preview_proxies_from_nas(replica_client):
    pid = _first_photo_id(replica_client)
    r = replica_client.get(f"/api/photos/{pid}/preview")
    assert r.status_code == 200
    assert (pid, "preview") in replica_client._fetched["calls"]


def test_full_proxies_from_nas_not_cached(replica_client):
    pid = _first_photo_id(replica_client)
    r = replica_client.get(f"/api/photos/{pid}/full")
    assert r.status_code == 200
    # full is streamed, not cached → every request hits the NAS
    replica_client._fetched["calls"].clear()
    replica_client.get(f"/api/photos/{pid}/full")
    assert (pid, "full") in replica_client._fetched["calls"]


def test_nas_fetch_failure_maps_to_502(client, monkeypatch):
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")

    def boom(photo_id, kind, timeout=30.0):
        raise OSError("connection refused")
    monkeypatch.setattr(web, "_fetch_from_nas", boom)

    res = client.get("/api/search?location=Big Sur").json()
    pid = res["results"][0]["id"]
    r = client.get(f"/api/photos/{pid}/thumbnail")
    assert r.status_code == 502


def test_no_nas_url_still_404s(client, monkeypatch):
    # Without replica mode, a missing original is a plain 404 (unchanged).
    monkeypatch.setattr(web, "_nas_url", None)
    res = client.get("/api/search?location=Big Sur").json()
    pid = res["results"][0]["id"]
    r = client.get(f"/api/photos/{pid}/thumbnail")
    assert r.status_code == 404


# ── M26a UI: replica freshness card + Sync-from-NAS button on /status ────────

def test_status_page_ships_replica_card(client):
    """/status carries the ReplicaCard markup. It renders only when
    /api/admin/replica-status reports replica_mode, so the NAS's own status
    page stays visually unchanged — but the component must ship in the HTML."""
    body = client.get("/status").text
    assert "Sync from NAS" in body
    assert "/api/admin/replica-status" in body
    assert "/api/admin/replica-sync" in body


def test_replica_status_endpoint_reflects_env(client, monkeypatch, tmp_path):
    # Not a replica: no PHOTOSEARCH_NAS_URL → replica_mode False.
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setenv("PHOTOSEARCH_DB", str(tmp_path / "replica-status-probe.db"))
    r = client.get("/api/admin/replica-status")
    assert r.status_code == 200
    body = r.json()
    assert body["replica_mode"] is False
    assert body["nas_url"] is None

    # Replica mode: URL is normalized (trailing slash stripped) and reported;
    # the NAS count probe is stubbed out so the test never touches a network.
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://fake-nas:8000/")
    import urllib.request

    def no_network(*a, **k):
        raise OSError("no network in tests")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)

    body = client.get("/api/admin/replica-status").json()
    assert body["replica_mode"] is True
    assert body["nas_url"] == "http://fake-nas:8000"
    assert body["nas_photos"] is None          # probe failed → None, not an error
    assert body["drift"] is None
    assert "last_sync" in body and "sync_script" in body


# ---------------------------------------------------------------------------
# Face WRITES must go to the NAS, never only to the replica.
#
# sync-replica.sh REPLACES the replica DB wholesale, so a face merge/assign
# written only locally is silently destroyed on the next sync. These pin the
# proxy-and-mirror behaviour for every face-mutating route.
# ---------------------------------------------------------------------------

@pytest.fixture
def nas_spy(replica_client, monkeypatch):
    """Record NAS calls instead of making them, and stub the mirror-back."""
    calls = []

    def fake_nas_json(method, path, body=None, timeout=120.0):
        calls.append((method, path, body))
        return {"ok": True, "proxied": True}

    def fake_mirror(photo_ids):
        calls.append(("MIRROR", sorted(photo_ids), None))
        return {"mirrored": len(photo_ids), "errors": 0, "missing": 0}

    monkeypatch.setattr(web, "_nas_json", fake_nas_json)
    monkeypatch.setattr(web, "_mirror_face_photos", fake_mirror)
    replica_client._nas_calls = calls
    return replica_client


def _face_row(client, where):
    from photosearch.db import PhotoDB
    with PhotoDB(web._db_path) as db:
        return db.conn.execute(f"SELECT * FROM faces WHERE {where}").fetchone()


def test_assign_proxies_to_nas_and_does_not_write_locally(nas_spy):
    face = _face_row(nas_spy, "person_id IS NULL LIMIT 1")
    fid, pid = face["id"], face["photo_id"]

    r = nas_spy.post(f"/api/faces/{fid}/assign?name=Calvin")
    assert r.status_code == 200
    methods = [(m, p) for m, p, _ in nas_spy._nas_calls]
    assert ("POST", f"/api/faces/{fid}/assign?name=Calvin") in methods
    assert ("MIRROR", [pid]) in [(m, p) for m, p, _ in nas_spy._nas_calls]
    # the replica must NOT have applied the write itself
    assert _face_row(nas_spy, f"id = {fid}")["person_id"] is None


def test_clear_proxies_to_nas(nas_spy):
    face = _face_row(nas_spy, "person_id IS NOT NULL LIMIT 1")
    fid, pid, person = face["id"], face["photo_id"], face["person_id"]

    r = nas_spy.post(f"/api/faces/{fid}/clear")
    assert r.status_code == 200
    assert ("POST", f"/api/faces/{fid}/clear", None) in nas_spy._nas_calls
    assert _face_row(nas_spy, f"id = {fid}")["person_id"] == person  # untouched locally


def test_bulk_assign_proxies_and_mirrors_every_touched_photo(nas_spy):
    from photosearch.db import PhotoDB
    with PhotoDB(web._db_path) as db:
        rows = db.conn.execute("SELECT id, photo_id FROM faces LIMIT 3").fetchall()
    fids = [r["id"] for r in rows]
    want_photos = sorted({r["photo_id"] for r in rows})

    r = nas_spy.post("/api/faces/bulk-assign",
                     json={"face_ids": fids, "person_name": "Calvin"})
    assert r.status_code == 200
    posted = [b for m, p, b in nas_spy._nas_calls if p == "/api/faces/bulk-assign"]
    assert posted and posted[0]["face_ids"] == fids
    assert ("MIRROR", want_photos) in [(m, p) for m, p, _ in nas_spy._nas_calls]


def test_merge_proxies_and_resolves_photos_before_the_merge(nas_spy):
    """The photo set must be read BEFORE proxying — the merge clears cluster_id,
    so resolving afterwards would mirror nothing."""
    from photosearch.db import PhotoDB
    with PhotoDB(web._db_path) as db:
        row = db.conn.execute(
            "SELECT photo_id FROM faces WHERE cluster_id = 99").fetchone()
        person = db.conn.execute("SELECT id FROM persons LIMIT 1").fetchone()["id"]

    r = nas_spy.post("/api/faces/merges", json={
        "source": {"type": "cluster", "id": 99},
        "target": {"type": "person", "id": person}})
    assert r.status_code == 200
    assert any(p == "/api/faces/merges" for _, p, _ in nas_spy._nas_calls)
    assert ("MIRROR", [row["photo_id"]]) in [(m, p) for m, p, _ in nas_spy._nas_calls]
    # local cluster membership untouched — the NAS owns the change
    with PhotoDB(web._db_path) as db:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM faces WHERE cluster_id = 99").fetchone()[0] == 1


def test_ignore_writes_nas_first_then_locally(nas_spy):
    from photosearch.db import PhotoDB
    r = nas_spy.post("/api/faces/ignore", json={"cluster_ids": [99]})
    assert r.status_code == 200
    assert ("POST", "/api/faces/ignore", {"cluster_ids": [99]}) in nas_spy._nas_calls
    # cluster ids have no photo dimension, so this one DOES apply locally too
    with PhotoDB(web._db_path) as db:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ignored_clusters WHERE cluster_id = 99").fetchone()[0] == 1

    nas_spy.post("/api/faces/unignore", json={"cluster_ids": [99]})
    assert ("POST", "/api/faces/unignore", {"cluster_ids": [99]}) in nas_spy._nas_calls
    with PhotoDB(web._db_path) as db:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ignored_clusters WHERE cluster_id = 99").fetchone()[0] == 0


def test_non_replica_mode_still_writes_locally(client, monkeypatch):
    """Guard the other direction: with no NAS configured the NAS is us."""
    monkeypatch.setattr(web, "_nas_url", None)
    face = _face_row(client, "person_id IS NULL LIMIT 1")
    r = client.post(f"/api/faces/{face['id']}/assign?name=Calvin")
    assert r.status_code == 200
    assert _face_row(client, f"id = {face['id']}")["person_id"] is not None

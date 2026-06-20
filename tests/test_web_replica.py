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

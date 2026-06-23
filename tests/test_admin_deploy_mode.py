"""Deploy-control awareness of the native local-replica run (M26).

The /admin/deploy page must work off the desktop replica, not just the NAS
container: `_deploy_mode()` distinguishes the two, /version reports it, the
docker-only actions (build / restart-mcp) refuse in native mode, and /restart
re-execs the process instead of swapping a container.

The test suite itself runs from the checkout, so `_deploy_mode()` is 'native'
here — which is exactly the case we want to cover.
"""
import pytest

from photosearch import admin_api


def test_deploy_mode_is_native_from_checkout():
    assert admin_api._deploy_mode() == "native"
    assert admin_api._repo_available() is True
    assert admin_api._active_repo_dir() == admin_api._native_repo_dir()


def test_native_deployed_sha_is_startup_head():
    """In native mode the 'deployed' sha is the HEAD captured at process start,
    not the docker BUILD_SHA file."""
    assert admin_api._deployed_sha() == admin_api._NATIVE_STARTUP_SHA
    # And it's a real 40-char sha (the checkout has commits).
    assert admin_api._NATIVE_STARTUP_SHA
    assert len(admin_api._NATIVE_STARTUP_SHA) == 40


def test_version_endpoint_reports_native_mode(client):
    r = client.get("/api/admin/version")
    assert r.status_code == 200
    v = r.json()
    assert v["available"] is True
    assert v["mode"] == "native"
    assert v["head"] and v["head"]["sha"]
    assert v["deployed_sha"] == admin_api._NATIVE_STARTUP_SHA


def test_docker_build_refused_in_native_mode(client):
    r = client.post("/api/admin/docker-build")
    assert r.status_code == 400
    assert "native" in r.json()["detail"].lower()


def test_restart_mcp_refused_in_native_mode(client):
    r = client.post("/api/admin/restart-mcp")
    assert r.status_code == 400
    assert "native" in r.json()["detail"].lower()


def test_native_restart_reexecs_without_killing_runner(client, monkeypatch):
    """/restart in native mode must schedule a process re-exec — but we stub the
    Thread so the test runner is never actually replaced."""
    started = {}

    class _FakeThread:
        def __init__(self, target=None, daemon=None):
            started["target"] = target
            started["daemon"] = daemon

        def start(self):
            started["start_called"] = True  # deliberately do NOT run target

    monkeypatch.setattr(admin_api.threading, "Thread", _FakeThread)

    r = client.post("/api/admin/restart")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "re-exec" in body["note"].lower()
    assert started.get("start_called") is True
    assert started.get("daemon") is True
    assert callable(started.get("target"))

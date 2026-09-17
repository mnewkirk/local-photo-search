"""POST /api/admin/warm-face-crops — command construction and the job lock.

Face crops are generated on first view (a full image decode, ~2s on the N100),
so a cold review grid dribbles in over minutes. This endpoint pre-generates
them. The thing worth testing is the ARGV it builds: an unscoped run is hours
on that box, so the scope flags have to actually reach the CLI.
"""
from unittest.mock import patch

import pytest

from photosearch import admin_api


def _argv(client, qs=""):
    """Capture the command the endpoint would run, without running it."""
    seen = {}

    async def fake_stream(cmd, cwd=None, env=None):
        seen["cmd"] = cmd
        yield 'event: line\ndata: {"returncode": 0}\n\n'

    with patch.object(admin_api, "_stream_subprocess", fake_stream):
        r = client.post("/api/admin/warm-face-crops" + qs)
        assert r.status_code == 200, r.text
        r.read()
    return seen.get("cmd", [])


def test_date_scope_reaches_the_cli(client):
    cmd = _argv(client, "?date_from=2026-09-12&date_to=2026-09-12")
    assert "warm-face-crops" in cmd
    assert cmd[cmd.index("--date-from") + 1] == "2026-09-12"
    assert cmd[cmd.index("--date-to") + 1] == "2026-09-12"


def test_defaults_to_every_face_not_just_matched(client):
    """The panels that need warming browse unknown clusters too, so
    --matched-only would leave exactly the grids that crawl still cold."""
    cmd = _argv(client, "?date_from=2026-09-12")
    assert "--all" in cmd


def test_person_scope_replaces_all(client):
    """--person and --all are mutually exclusive in the CLI; sending both
    would make the person argument silently meaningless."""
    cmd = _argv(client, "?person=Calvin")
    assert cmd[cmd.index("--person") + 1] == "Calvin"
    assert "--all" not in cmd


def test_worker_count_is_clamped(client):
    """Left open, a typo'd worker count would oversubscribe the N100 into
    swapping — the decode loop is the whole cost of this job."""
    assert _argv(client, "?workers=99")[-1] != "99"
    cmd = _argv(client, "?workers=99")
    assert cmd[cmd.index("--workers") + 1] == "8"
    cmd = _argv(client, "?workers=0")
    assert cmd[cmd.index("--workers") + 1] == "1"


def test_second_run_is_refused_while_one_is_active(client):
    """Shares the long-job lock: two concurrent decode sweeps would fight for
    the same cores and the same cache files."""
    admin_api._ingest_lock.acquire()
    try:
        r = client.post("/api/admin/warm-face-crops?date_from=2026-09-12")
        assert r.status_code == 409
    finally:
        admin_api._ingest_lock.release()

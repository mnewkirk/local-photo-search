"""Tests for the maintenance sync layer (replica -> NAS push).

Covers the fingerprint, the stage push-mode taxonomy, payload collection,
the timestamp comparison rules, and the NAS-side apply path.
"""

import pytest
import requests

from photosearch.maintenance_sync import (
    EXCLUDED_STAGES,
    TRANSFER_STAGES,
    TRIGGER_STAGES,
    fingerprints_match,
    photo_fingerprint,
    push_mode,
)


def test_photo_fingerprint_counts_and_max_id(db):
    fp = photo_fingerprint(db)
    expected_n = db.conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
    expected_max = db.conn.execute("SELECT MAX(id) AS m FROM photos").fetchone()["m"]
    assert fp == {"photo_count": expected_n, "photo_max_id": expected_max}


def test_photo_fingerprint_on_empty_db(tmp_db_path):
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as empty:
        assert photo_fingerprint(empty) == {"photo_count": 0, "photo_max_id": None}


def test_fingerprints_match_is_exact():
    a = {"photo_count": 10, "photo_max_id": 99}
    assert fingerprints_match(a, {"photo_count": 10, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 11, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 10, "photo_max_id": 100})


@pytest.mark.parametrize("stage", sorted(TRIGGER_STAGES))
def test_trigger_stages(stage):
    assert push_mode(stage) == "trigger"


def test_stacking_is_the_only_transfer_stage():
    assert TRANSFER_STAGES == frozenset({"stacking"})
    assert push_mode("stacking") == "transfer"


@pytest.mark.parametrize("stage", ["colors", "dedup_photos", "match_faces",
                                   "recluster", "requeue"])
def test_excluded_stages(stage):
    assert push_mode(stage) == "excluded"


def test_taxonomy_covers_every_sweep_stage_exactly_once():
    """Every stage the sweep can emit must have exactly one push mode.

    Guards against a new stage being added to maintenance.py without a
    reconciliation decision — which would silently mean 'lost on next sync'.
    """
    from photosearch.maintenance import SWEEP_STAGE_ORDER
    known = TRIGGER_STAGES | TRANSFER_STAGES | EXCLUDED_STAGES
    assert set(SWEEP_STAGE_ORDER) - known == set(), "sweep stage with no push mode"
    assert not (TRIGGER_STAGES & TRANSFER_STAGES)
    assert not (TRIGGER_STAGES & EXCLUDED_STAGES)
    assert not (TRANSFER_STAGES & EXCLUDED_STAGES)


def test_push_mode_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown maintenance stage"):
        push_mode("not_a_stage")


# ---------------------------------------------------------------------------
# The sweep stamps watermarks
# ---------------------------------------------------------------------------

def test_sweep_stamps_watermark_only_for_done_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(db, apply=True, do_colors=False,
                                   do_stacking=False, do_match=False,
                                   source="replica")
    runs = db.get_maintenance_runs()
    done = {s["stage"] for s in result["stages"] if s["status"] == "done"}
    skipped = {s["stage"] for s in result["stages"] if s["status"] != "done"}

    assert done, "expected at least one stage to run against the fixture"
    assert done <= set(runs), "every done stage must be stamped"
    assert not (skipped & set(runs)), "skipped stages must not be stamped"
    for stage in done:
        assert runs[stage]["source"] == "replica"
        assert runs[stage]["last_run_at"]


def test_dry_run_stamps_nothing(db):
    from photosearch.maintenance import run_maintenance_sweep

    run_maintenance_sweep(db, apply=False, do_colors=False,
                          do_stacking=False, do_match=False)
    assert db.get_maintenance_runs() == {}


def test_stages_subset_runs_only_those_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(
        db, apply=True, stages=["normalize_aesthetics"],
    )
    assert [s["stage"] for s in result["stages"]] == ["normalize_aesthetics"]


def test_unknown_stage_subset_is_rejected(db):
    from photosearch.maintenance import run_maintenance_sweep

    with pytest.raises(ValueError, match="unknown maintenance stage"):
        run_maintenance_sweep(db, apply=True, stages=["bogus_stage"])


# ---------------------------------------------------------------------------
# Payload collection
# ---------------------------------------------------------------------------

from photosearch.maintenance_sync import (  # noqa: E402
    collect_payload,
    collect_stacking_rows,
    eligible_stages,
)


def test_eligible_stages_only_includes_done_and_pushable():
    results = [
        {"stage": "normalize_aesthetics", "status": "done"},
        {"stage": "geocode", "status": "skipped"},
        {"stage": "stacking", "status": "done"},
        {"stage": "colors", "status": "done"},      # excluded mode
        {"stage": "infer", "status": "preview"},
    ]
    assert eligible_stages(results) == ["normalize_aesthetics", "stacking"]


def test_cancelled_stage_is_not_eligible():
    """A half-finished stacking run must never ship."""
    results = [{"stage": "stacking", "status": "cancelled"}]
    assert eligible_stages(results) == []


def _make_stack(db, photo_ids, top_index=0):
    cur = db.conn.execute("INSERT INTO photo_stacks DEFAULT VALUES")
    stack_id = cur.lastrowid
    for i, pid in enumerate(photo_ids):
        db.conn.execute(
            "INSERT INTO stack_members (stack_id, photo_id, is_top) VALUES (?, ?, ?)",
            (stack_id, pid, 1 if i == top_index else 0),
        )
    db.conn.commit()
    return stack_id


def test_collect_stacking_rows_groups_members_by_stack(db):
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 3")]
    _make_stack(db, pids[:2], top_index=1)

    rows = collect_stacking_rows(db)
    assert len(rows) == 1
    members = rows[0]["members"]
    assert {m["photo_id"] for m in members} == set(pids[:2])
    assert sum(m["is_top"] for m in members) == 1


def test_collect_stacking_rows_keeps_separate_stacks_separate(db):
    """Two stacks must stay two stacks.

    The single-stack case can't catch a grouping bug that merges every member
    into one bucket, so this asserts the members of distinct stacks are not
    flattened together.
    """
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 4")]
    assert len(pids) >= 4, "fixture needs at least 4 photos"
    _make_stack(db, pids[0:2], top_index=0)
    _make_stack(db, pids[2:4], top_index=1)

    rows = collect_stacking_rows(db)
    assert len(rows) == 2, "two stacks must not be merged into one"

    grouped = sorted(
        [sorted(m["photo_id"] for m in row["members"]) for row in rows]
    )
    assert grouped == [sorted(pids[0:2]), sorted(pids[2:4])]
    # Exactly one top per stack.
    for row in rows:
        assert sum(m["is_top"] for m in row["members"]) == 1


def test_collect_payload_includes_stacking_only_when_stacking_ran(db):
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")
    db.record_maintenance_run(
        stage="normalize_aesthetics", last_run_at="2026-07-17T09:00:01+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    with_stacking = collect_payload(db, [
        {"stage": "stacking", "status": "done"},
        {"stage": "normalize_aesthetics", "status": "done"},
    ])
    assert with_stacking["stacking"] is not None
    assert with_stacking["stages"]["stacking"]["mode"] == "transfer"
    assert with_stacking["stages"]["normalize_aesthetics"]["mode"] == "trigger"
    assert with_stacking["fingerprint"] == photo_fingerprint(db)

    without = collect_payload(db, [{"stage": "normalize_aesthetics", "status": "done"}])
    assert without["stacking"] is None
    assert "stacking" not in without["stages"]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def test_fingerprint_endpoint_reports_index_and_stages(client, db, monkeypatch):
    # admin_api reads PHOTOSEARCH_DB fresh per-request (not web._db_path, which
    # the `client` fixture patches) — mirrors test_web_replica.py's
    # test_replica_status_endpoint_reflects_env for the sibling endpoint.
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=3, photo_max_id=42, applied=2, source="replica")

    r = client.get("/api/admin/maintenance-fingerprint")
    assert r.status_code == 200
    body = r.json()
    expected = photo_fingerprint(db)
    assert body["photo_count"] == expected["photo_count"]
    assert body["photo_max_id"] == expected["photo_max_id"]
    assert body["stages"]["stacking"]["source"] == "replica"
    assert body["stages"]["stacking"]["last_run_at"] == "2026-07-17T09:00:00+00:00"
    assert body["replica_mode"] is False


def test_nas_fingerprint_proxy_returns_error_body_outside_replica_mode(client, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    r = client.get("/api/admin/maintenance-nas-fingerprint")
    assert r.status_code == 200
    assert r.json() == {"error": "not in replica mode", "stages": {}}


def test_nas_fingerprint_proxy_forwards_the_nas_response(client, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    nas_body = {
        "photo_count": 10, "photo_max_id": 99,
        "stages": {"stacking": {"last_run_at": "2026-07-17T09:00:00+00:00",
                                 "source": "nas", "applied": 3}},
        "replica_mode": False,
    }

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return nas_body

    def fake_get(url, timeout=None):
        assert url == "http://nas:8000/api/admin/maintenance-fingerprint"
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    r = client.get("/api/admin/maintenance-nas-fingerprint")
    assert r.status_code == 200
    assert r.json() == nas_body


def test_nas_fingerprint_proxy_returns_error_body_when_nas_unreachable(client, monkeypatch):
    # Mirrors the admin_workers_queue_status proxy: a degraded NAS comes back
    # as a normal 200 with an `error` field, not an HTTP error status, so the
    # frontend's plain fetch().then(r => r.json()) chain doesn't need to
    # special-case a non-2xx response.
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://unreachable:8000")

    def boom(url, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", boom)

    r = client.get("/api/admin/maintenance-nas-fingerprint")
    assert r.status_code == 200
    body = r.json()
    assert body["stages"] == {}
    assert "connection refused" in body["error"]


def _payload_for(db, *, last_run_at, stacking=None):
    fp = photo_fingerprint(db)
    stages = {"stacking": {"mode": "transfer", "last_run_at": last_run_at}}
    return {"fingerprint": fp, "stages": stages, "stacking": stacking or []}


def test_apply_rejects_fingerprint_mismatch(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 2")]
    stack_id = _make_stack(db, pids[:2])

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",
        stacking=[{"members": [{"photo_id": pids[0], "is_top": 1}]}],
    )
    body["fingerprint"]["photo_count"] += 1

    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "fingerprint_mismatch"

    # Nothing was written: the seeded stack is unchanged.
    rows = db.conn.execute(
        "SELECT photo_id FROM stack_members WHERE stack_id = ?",
        (stack_id,)).fetchall()
    assert {r["photo_id"] for r in rows} == set(pids[:2])
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 1


def test_apply_rejects_missing_stacking_payload(client, db, monkeypatch):
    """A missing 'stacking' key must NOT be treated as 'zero stacks'.

    [] means "the replica ran stacking and found nothing" (a legitimate
    full-replace-to-empty). A missing/None key means no stacking payload
    was sent at all, and must be rejected rather than silently wiping
    every stack on the NAS.
    """
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 2")]
    stack_id = _make_stack(db, pids[:2])

    fp = photo_fingerprint(db)
    body = {
        "fingerprint": fp,
        "stages": {"stacking": {"mode": "transfer",
                                 "last_run_at": "2026-07-17T09:00:00+00:00"}},
        # Deliberately no "stacking" key at all.
    }
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "stacking_payload_missing"

    # The pre-existing stack must still be present and unchanged.
    rows = db.conn.execute(
        "SELECT photo_id FROM stack_members WHERE stack_id = ?",
        (stack_id,)).fetchall()
    assert {r["photo_id"] for r in rows} == set(pids[:2])
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 1


def test_apply_with_empty_stacking_list_wipes_stacks(client, db, monkeypatch):
    """Pin the None-vs-[] distinction the sibling test above guards the other
    side of: 'stacking' missing/None means no payload was sent (400, refuse
    to wipe), but 'stacking': [] means the replica ran stacking and genuinely
    found zero stacks — that must still legitimately WIPE every existing
    stack (full-replace semantics). Without this test, "hardening" [] into a
    400 alongside None reads like a reasonable fix and would ship with every
    other test green, silently breaking legitimate empty-stacking pushes.
    """
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 2")]
    _make_stack(db, pids[:2])

    body = _payload_for(db, last_run_at="2026-07-17T09:00:00+00:00", stacking=[])
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["stacking"]["status"] == "applied"

    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 0
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM stack_members").fetchone()["n"] == 0


def test_apply_rolls_back_on_foreign_key_violation(client, db, monkeypatch):
    """One transaction: a failure must leave prior state intact.

    A stacking row referencing a photo_id that doesn't exist violates the
    stack_members -> photos foreign key. The whole apply must roll back,
    not leave a half-replaced stack table.

    Uses a raise_server_exceptions=False TestClient (rather than the shared
    `client` fixture) so the unhandled sqlite3.IntegrityError surfaces as
    the 500 response a real deployment would return, instead of propagating
    out of the test itself.
    """
    from fastapi.testclient import TestClient
    from photosearch import web

    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 2")]
    stack_id = _make_stack(db, pids[:2])
    bogus_photo_id = db.conn.execute(
        "SELECT COALESCE(MAX(id), 0) + 1000000 AS m FROM photos").fetchone()["m"]

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",
        stacking=[{"members": [{"photo_id": bogus_photo_id, "is_top": 1}]}],
    )
    with TestClient(web.app, raise_server_exceptions=False) as lenient_client:
        r = lenient_client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code >= 400 and r.status_code < 600 and r.status_code != 200

    # The original stack survived the rollback, unchanged.
    rows = db.conn.execute(
        "SELECT photo_id FROM stack_members WHERE stack_id = ?",
        (stack_id,)).fetchall()
    assert {r["photo_id"] for r in rows} == set(pids[:2])
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 1
    # No watermark was stamped for the failed apply.
    runs = db.get_maintenance_runs()
    assert "stacking" not in runs


def test_apply_reports_not_transfer_mode_for_trigger_stage(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    fp = photo_fingerprint(db)
    body = {
        "fingerprint": fp,
        "stages": {
            "normalize_aesthetics": {
                "mode": "trigger",
                "last_run_at": "2026-07-17T09:00:00+00:00",
            },
        },
    }
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["normalize_aesthetics"] == {
        "status": "skipped",
        "reason": "not_transfer_mode",
    }


def test_apply_replaces_stacks_and_stamps_watermark(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    pids = [r["id"] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 3")]
    # Pre-existing stack that must be REPLACED, not merged.
    _make_stack(db, pids[:2])

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",
        stacking=[{"members": [{"photo_id": pids[2], "is_top": 1}]}],
    )
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["stacking"]["status"] == "applied"

    rows = db.conn.execute(
        "SELECT photo_id, is_top FROM stack_members").fetchall()
    assert [r["photo_id"] for r in rows] == [pids[2]]
    assert db.conn.execute(
        "SELECT COUNT(*) AS n FROM photo_stacks").fetchone()["n"] == 1

    runs = db.get_maintenance_runs()
    assert runs["stacking"]["source"] == "replica"
    assert runs["stacking"]["last_run_at"] == "2026-07-17T09:00:00+00:00"


def test_apply_skips_stage_when_nas_is_fresher(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T12:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="nas")
    pids = [r["id"] for r in db.conn.execute("SELECT id FROM photos LIMIT 1")]

    body = _payload_for(
        db, last_run_at="2026-07-17T09:00:00+00:00",  # older than the NAS
        stacking=[{"members": [{"photo_id": pids[0], "is_top": 1}]}],
    )
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 200
    assert r.json()["applied"]["stacking"]["status"] == "skipped"
    assert r.json()["applied"]["stacking"]["reason"] == "nas_fresher"
    # Watermark untouched.
    assert db.get_maintenance_runs()["stacking"]["source"] == "nas"


def test_apply_skips_stage_when_timestamps_are_equal(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    same = "2026-07-17T09:00:00+00:00"
    db.record_maintenance_run(
        stage="stacking", last_run_at=same,
        photo_count=1, photo_max_id=1, applied=1, source="nas")

    body = _payload_for(db, last_run_at=same, stacking=[])
    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.json()["applied"]["stacking"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# push_to_nas orchestration (NAS calls mocked)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or 'event: message\ndata: {"type": "done"}\n\n'

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# _sse_has_fatal
# ---------------------------------------------------------------------------

def test_sse_has_fatal_detects_default_separator():
    from photosearch.maintenance_sync import _sse_has_fatal
    body = 'event: message\ndata: {"type": "fatal"}\n\n'
    assert _sse_has_fatal(body) is True


def test_sse_has_fatal_detects_compact_separator():
    """The whole point: a compact (no-space) json.dumps must still be caught.

    The old check was a substring match on '"type": "fatal"' (with a space),
    which silently stops matching if the server ever serializes compactly.
    """
    from photosearch.maintenance_sync import _sse_has_fatal
    body = 'event: message\ndata: {"type":"fatal"}\n\n'
    assert _sse_has_fatal(body) is True


def test_sse_has_fatal_false_on_done():
    from photosearch.maintenance_sync import _sse_has_fatal
    body = 'event: message\ndata: {"type": "done"}\n\n'
    assert _sse_has_fatal(body) is False


def test_sse_has_fatal_ignores_word_in_message_text():
    """A message that merely mentions 'fatal' must not false-positive."""
    from photosearch.maintenance_sync import _sse_has_fatal
    body = 'data: {"type": "progress", "message": "no fatal errors"}\n\n'
    assert _sse_has_fatal(body) is False


def test_push_sends_transfer_before_trigger(db, monkeypatch):
    from photosearch import maintenance_sync

    calls = []

    def fake_post(url, json=None, timeout=None, **kw):
        calls.append(url)
        if url.endswith("/maintenance-apply"):
            return _FakeResponse(200, {"applied": {"stacking": {"status": "applied"}}})
        return _FakeResponse(200, {})

    monkeypatch.setattr(maintenance_sync.requests, "post", fake_post)

    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")
    db.record_maintenance_run(
        stage="normalize_aesthetics", last_run_at="2026-07-17T09:00:01+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    result = maintenance_sync.push_to_nas(db, "http://nas:8000", [
        {"stage": "stacking", "status": "done"},
        {"stage": "normalize_aesthetics", "status": "done"},
    ])

    assert result["ok"] is True
    assert calls[0].endswith("/maintenance-apply"), "transfer must go first"
    assert calls[1].endswith("/maintenance-sweep"), "triggers go second"
    assert result["stages"]["stacking"]["status"] == "applied"
    assert result["stages"]["normalize_aesthetics"]["status"] == "triggered"


def test_push_reports_fingerprint_mismatch(db, monkeypatch):
    from photosearch import maintenance_sync

    def fake_post(url, json=None, timeout=None, **kw):
        return _FakeResponse(409, {"detail": {"error": "fingerprint_mismatch"}})

    monkeypatch.setattr(maintenance_sync.requests, "post", fake_post)
    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    result = maintenance_sync.push_to_nas(db, "http://nas:8000",
                                          [{"stage": "stacking", "status": "done"}])
    assert result["ok"] is False
    assert result["error"] == "fingerprint_mismatch"


def test_push_with_nothing_eligible_is_a_noop(db, monkeypatch):
    from photosearch import maintenance_sync

    def boom(*a, **kw):
        raise AssertionError("must not call the NAS with nothing to push")

    monkeypatch.setattr(maintenance_sync.requests, "post", boom)
    result = maintenance_sync.push_to_nas(db, "http://nas:8000",
                                          [{"stage": "geocode", "status": "skipped"}])
    assert result["ok"] is True
    assert result["stages"] == {}


def test_push_partial_failure_keeps_successful_transfer_stage(db, monkeypatch):
    """Transfer succeeds, trigger fails: the successful half must not be
    collapsed by the failed half — each stage reports its own outcome."""
    from photosearch import maintenance_sync

    def fake_post(url, json=None, timeout=None, **kw):
        if url.endswith("/maintenance-apply"):
            return _FakeResponse(200, {"applied": {"stacking": {"status": "applied"}}})
        raise requests.exceptions.ConnectionError("NAS unreachable")

    monkeypatch.setattr(maintenance_sync.requests, "post", fake_post)

    db.record_maintenance_run(
        stage="stacking", last_run_at="2026-07-17T09:00:00+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")
    db.record_maintenance_run(
        stage="normalize_aesthetics", last_run_at="2026-07-17T09:00:01+00:00",
        photo_count=1, photo_max_id=1, applied=1, source="replica")

    result = maintenance_sync.push_to_nas(db, "http://nas:8000", [
        {"stage": "stacking", "status": "done"},
        {"stage": "normalize_aesthetics", "status": "done"},
    ])

    assert result["ok"] is False
    assert result["error"] == "unreachable"
    assert result["stages"]["stacking"]["status"] == "applied"
    assert result["stages"]["normalize_aesthetics"]["status"] == "failed"
    assert result["stages"]["normalize_aesthetics"]["reason"] == "unreachable"


# ---------------------------------------------------------------------------
# Replica-mode gating
# ---------------------------------------------------------------------------

def test_apply_blocked_in_replica_mode_when_nas_unreachable(client, monkeypatch):
    """THE gate: never write results that the next sync is guaranteed to wipe."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://unreachable:8000")

    from photosearch import maintenance_sync

    def boom(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(maintenance_sync.requests, "get", boom)

    r = client.post("/api/admin/maintenance-sweep", json={"apply": True})
    assert r.status_code == 503
    assert r.json()["detail"]["error"] == "nas_unreachable"


def test_dry_run_allowed_in_replica_mode_without_nas(client, monkeypatch):
    """Dry-runs write nothing, so there's nothing to lose."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://unreachable:8000")
    r = client.post("/api/admin/maintenance-sweep", json={"apply": False})
    assert r.status_code == 200


def test_excluded_stage_rejected_in_replica_mode(client, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "do_colors": True})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "excluded_stage_in_replica_mode"
    assert "colors" in r.json()["detail"]["stages"]


def test_excluded_stage_allowed_on_the_nas(client, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "do_colors": True})
    assert r.status_code == 200


def test_push_status_starts_idle(client):
    r = client.get("/api/admin/maintenance-push-status")
    assert r.status_code == 200
    assert r.json()["state"] == "idle"


def test_push_status_reflects_last_push(client):
    from photosearch import web

    with web._push_lock:
        web._push_status.update({
            "state": "failed", "error": "fingerprint_mismatch",
            "stages": {"stacking": {"status": "failed"}},
            "finished_at": "2026-07-17T09:00:00+00:00",
        })
    try:
        body = client.get("/api/admin/maintenance-push-status").json()
        assert body["state"] == "failed"
        assert body["error"] == "fingerprint_mismatch"
    finally:
        with web._push_lock:
            web._push_status.update({"state": "idle", "error": None,
                                     "stages": {}, "finished_at": None})


def test_start_push_sets_running_before_thread_completes(client, monkeypatch):
    """_start_push must flip state to "running" SYNCHRONOUSLY, before the
    background thread starts — the sweep's SSE stream has already closed by
    the time _start_push runs, so a client polling maintenance-push-status
    right after the "done" event must never observe a stale PREVIOUS push's
    terminal state (e.g. "ok" with an old finished_at) as if the new push had
    already succeeded."""
    import threading
    from photosearch import maintenance_sync, web

    release = threading.Event()

    def blocking_push(db, nas_url, stage_results):
        release.wait(timeout=5)
        return {"ok": True, "stages": {}}

    monkeypatch.setattr(maintenance_sync, "push_to_nas", blocking_push)
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")

    # Seed a stale terminal state, as if left over from a previous push.
    with web._push_lock:
        web._push_status.update({
            "state": "ok", "stages": {"stacking": {"status": "applied"}},
            "error": None, "finished_at": "2026-01-01T00:00:00+00:00",
        })

    try:
        web._start_push([{"stage": "stacking", "status": "done"}])

        # The push thread is blocked on `release`, so this observes exactly
        # what _start_push left behind before returning — not anything the
        # thread itself could have written.
        assert web._push_status["state"] == "running"
        assert web._push_status["finished_at"] is None
        assert web._push_status["error"] is None
    finally:
        release.set()
        for t in threading.enumerate():
            if t.name == "maintenance-push":
                t.join(timeout=5)
        with web._push_lock:
            web._push_status.update({"state": "idle", "error": None,
                                     "stages": {}, "finished_at": None})


def test_preflight_sync_runs_before_sweep_compute(client, db, monkeypatch):
    """THE central guarantee: the fingerprint check + auto-sync must complete
    BEFORE the sweep computes. sync-replica.sh REPLACES the whole local DB
    file, so if the sweep ran first, the sync it triggers would destroy the
    very results the sweep just computed."""
    from photosearch import admin_api, maintenance, maintenance_sync

    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")

    local_fp = maintenance_sync.photo_fingerprint(db)
    # Start deliberately mismatched so the pre-flight takes the sync path.
    state = {"remote": {"photo_count": local_fp["photo_count"] + 1,
                        "photo_max_id": local_fp["photo_max_id"]}}
    calls = []

    def fake_fetch_fingerprint(nas_url):
        return dict(state["remote"])

    def fake_sync(timeout=1800):
        calls.append("sync")
        # A real sync reconciles the local DB to the NAS; flip the fake
        # fingerprint to match so the post-sync re-check passes and the
        # sweep is reached.
        state["remote"] = dict(local_fp)

    def fake_sweep(db, **kwargs):
        calls.append("sweep")
        return {"stages": []}

    def fake_push(*a, **kw):
        return {"ok": True, "stages": {}}

    monkeypatch.setattr(maintenance_sync, "fetch_nas_fingerprint",
                        fake_fetch_fingerprint)
    monkeypatch.setattr(admin_api, "run_replica_sync_blocking", fake_sync)
    monkeypatch.setattr(maintenance, "run_maintenance_sweep", fake_sweep)
    monkeypatch.setattr(maintenance_sync, "push_to_nas", fake_push)

    r = client.post("/api/admin/maintenance-sweep", json={"apply": True})
    assert r.status_code == 200

    assert calls == ["sync", "sweep"], calls


def test_maintenance_sweep_endpoint_threads_stages_kwarg(client, db, monkeypatch):
    """THE fix: the endpoint parses body['stages'] but must also PASS it
    through to run_maintenance_sweep — a function-level test of
    run_maintenance_sweep(stages=...) alone can't catch the endpoint dropping
    it on the floor between parsing and the call. Push_to_nas's "trigger"
    mode relies on exactly this to ask the NAS for a stage subset instead of
    its whole default plan."""
    from photosearch import maintenance

    captured = {}

    def fake_sweep(db, **kwargs):
        captured.update(kwargs)
        return {"stages": [{"stage": "normalize_aesthetics", "status": "done",
                             "would": 0, "applied": 0}]}

    monkeypatch.setattr(maintenance, "run_maintenance_sweep", fake_sweep)

    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "stages": ["normalize_aesthetics"]})
    assert r.status_code == 200
    assert captured.get("stages") == ["normalize_aesthetics"], captured.get("stages")


def test_maintenance_sweep_endpoint_rejects_malformed_stages(client, monkeypatch):
    """'stages' must be a list of strings; a malformed value 400s immediately
    rather than starting the background sweep thread and exploding deep
    inside run_maintenance_sweep."""
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "stages": "normalize_aesthetics"})
    assert r.status_code == 400


def test_maintenance_sweep_endpoint_reports_unknown_stage_as_fatal_not_500(client, monkeypatch):
    """An unknown-but-well-shaped stage name is validated deep inside
    run_maintenance_sweep (raises ValueError). That must surface as a clean
    SSE 'fatal' event — the sweep runs on a background thread, so an
    unhandled exception there would otherwise never reach the client as a
    normal HTTP error."""
    r = client.post("/api/admin/maintenance-sweep",
                    json={"apply": True, "stages": ["bogus_stage"]})
    assert r.status_code == 200
    assert '"type": "fatal"' in r.text or '"type":"fatal"' in r.text

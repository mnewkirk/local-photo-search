"""Tests for the maintenance sync layer (replica -> NAS push).

Covers the fingerprint, the stage push-mode taxonomy, payload collection,
the timestamp comparison rules, and the NAS-side apply path.
"""

import pytest

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


def _payload_for(db, *, last_run_at, stacking=None):
    fp = photo_fingerprint(db)
    stages = {"stacking": {"mode": "transfer", "last_run_at": last_run_at}}
    return {"fingerprint": fp, "stages": stages, "stacking": stacking or []}


def test_apply_rejects_fingerprint_mismatch(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    body = _payload_for(db, last_run_at="2026-07-17T09:00:00+00:00")
    body["fingerprint"]["photo_count"] += 1

    r = client.post("/api/admin/maintenance-apply", json=body)
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "fingerprint_mismatch"


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

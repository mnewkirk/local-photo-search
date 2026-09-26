"""Tests for photosearch/batch_state.py (M "ingest batch" Task 3).

The two tests this module exists for are marked MANDATORY below. Both are
cases where `count_unprocessed_photos(...) == 0` and a naive
`remaining == 0 -> completed` reading is WRONG:

  (a) before any description exists, `count_unprocessed_photos` for
      `category-content` / `keywords` / `verify` returns 0 because those
      passes gate on `description IS NOT NULL` — the pass has not become
      *eligible*, not finished. Must read `waiting` on "describe".
  (b) a pass whose every photo has `worker_processed.attempts >=
      MAX_PROCESS_ATTEMPTS` also counts 0, because the claim predicate
      excludes exhausted rows. Must read `blocked`.

Per globals.md the shared `db` fixture is pre-seeded under `2026/`, so
every fixture here builds photos under `2091/`.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from photosearch.batch_state import (
    WORKER_PASSES,
    NAS_STEPS,
    OPTIONAL_STEPS,
    STEP_ORDER,
    DEPENDS_ON,
    STATES,
    batch_state,
    fleet_launch_passes,
)
from photosearch.db import MAX_PROCESS_ATTEMPTS
from photosearch.ingest_batches import (
    STALL_SECONDS,
    register_batch,
    get_batch,
    open_job,
    close_job,
    start_sweep,
    set_sweep_status,
    heartbeat_sweep,
)


_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
_DIR = "2091/2091-09-19_ILCE-7RM6"


def _fmt(dt: datetime) -> str:
    return dt.strftime(_TS_FORMAT)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_batch(db, directory: str = _DIR, count: int = 3, **kw):
    """Add `count` photos under `directory` and register the batch."""
    ids = []
    for i in range(count):
        ids.append(db.add_photo(
            filepath=f"{directory}/img{i:03d}.jpg",
            filename=f"img{i:03d}.jpg",
            date_taken=f"2091-09-19T10:0{i}:00",
        ))
    batch_id = register_batch(db, directory, **kw)
    return batch_id, ids


def _step(state: dict, name: str) -> dict:
    return next(s for s in state["steps"] if s["step"] == name)


def _embedding(seed: int = 0) -> list[float]:
    vec = [0.0] * 512
    vec[seed % 512] = 1.0
    return vec


# --- per-pass "make the output exist" helpers -----------------------------

def _set_col(db, ids, col, value):
    ph = ",".join("?" * len(ids))
    db.conn.execute(f"UPDATE photos SET {col} = ? WHERE id IN ({ph})",
                    [value] + list(ids))
    db.conn.commit()


def _do_clip(db, ids):
    for i, pid in enumerate(ids):
        db.add_clip_embedding(pid, _embedding(i))


def _do_faces(db, ids):
    for i, pid in enumerate(ids):
        db.add_face(pid, (10, 100, 90, 20), _embedding(200 + i))


def _do_quality(db, ids):
    _set_col(db, ids, "aesthetic_score", 6.0)
    _set_col(db, ids, "aesthetic_concepts", json.dumps(["sharp"]))


def _do_aesthetics(db, ids):
    _set_col(db, ids, "aes_overall", 7.5)


def _do_describe(db, ids):
    _set_col(db, ids, "description", "A description.")


_PASS_DOERS = {
    "clip": _do_clip,
    "faces": _do_faces,
    "quality": _do_quality,
    "aesthetics": _do_aesthetics,
    "describe": _do_describe,
    "category-visual": lambda db, ids: _set_col(db, ids, "visual_tags", json.dumps(["x"])),
    "category-content": lambda db, ids: _set_col(db, ids, "categories", json.dumps(["x"])),
    "keywords": lambda db, ids: _set_col(db, ids, "keywords", json.dumps(["x"])),
    "verify": lambda db, ids: _set_col(db, ids, "verified_at", "2091-09-19 12:00:00"),
}


def _complete_pass(db, ids, pass_type):
    _PASS_DOERS[pass_type](db, ids)


def _complete_all_worker_passes(db, ids):
    # describe first so the text passes become eligible.
    _do_describe(db, ids)
    for p in WORKER_PASSES:
        if p != "describe":
            _complete_pass(db, ids, p)


def _complete_all_nas_steps(db, batch_id, ids, include_optional=False):
    """Close every REQUIRED NAS step's job. The optional ones are left alone
    by default — `ready` must not depend on them, and a helper that quietly
    closed them would make that impossible to test."""
    _set_col(db, ids, "aes_overall_pct", 50.0)
    for step in NAS_STEPS:
        if step in OPTIONAL_STEPS and not include_optional:
            continue
        open_job(db, batch_id, step, "nas")
        close_job(db, batch_id, step)


def _exhaust(db, ids, pass_type, attempts=None):
    """Write worker_processed rows at/over the attempts cap."""
    attempts = MAX_PROCESS_ATTEMPTS if attempts is None else attempts
    for pid in ids:
        db.conn.execute(
            "INSERT OR REPLACE INTO worker_processed (photo_id, pass_type, attempts) "
            "VALUES (?, ?, ?)", (pid, pass_type, attempts))
    db.conn.commit()


def _claim(db, pass_type, photo_ids, *, expires_in_minutes=30, batch="claim-1"):
    expires = _fmt(_utcnow() + timedelta(minutes=expires_in_minutes))
    db.conn.execute(
        "INSERT OR REPLACE INTO worker_claims "
        "  (batch_id, worker_id, pass_type, photo_ids, claimed_at, expires_at) "
        "VALUES (?, 'w1', ?, ?, ?, ?)",
        (batch, pass_type, json.dumps(list(photo_ids)), _fmt(_utcnow()), expires))
    db.conn.commit()


# =========================================================================
# Shape
# =========================================================================

class TestShape:
    def test_steps_are_every_step_in_step_order(self, db):
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        assert [s["step"] for s in state["steps"]] == list(STEP_ORDER)

    def test_step_order_composition(self):
        assert STEP_ORDER == ("ingest",) + WORKER_PASSES + NAS_STEPS

    def test_rank_measure_is_a_nas_step_and_runs_last(self):
        """It reads the ORIGINAL files at full resolution and only the NAS has
        them (the replica holds no originals), so it is not a desktop step —
        the plan's "desktop-only" label was simply wrong. Last, because it is
        the optional one."""
        assert NAS_STEPS[-1] == "rank_measure"
        assert OPTIONAL_STEPS == ("rank_measure",)
        assert set(OPTIONAL_STEPS) <= set(NAS_STEPS)
        assert DEPENDS_ON["rank_measure"] == "faces"

    def test_kinds_match_the_step_family(self, db):
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        kinds = {s["step"]: s["kind"] for s in state["steps"]}
        assert kinds["ingest"] == "ingest"
        assert all(kinds[p] == "worker" for p in WORKER_PASSES)
        assert all(kinds[p] == "nas" for p in NAS_STEPS)
        assert "desktop" not in set(kinds.values())

    def test_every_step_carries_the_full_field_set(self, db):
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        for s in state["steps"]:
            assert set(s) == {"step", "kind", "state", "total", "eligible",
                              "done", "remaining", "failed", "waiting_on", "detail"}
            assert s["state"] in STATES
            assert s["total"] == len(ids)

    def test_top_level_keys(self, db):
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        assert set(state) == {"batch", "sweep", "ready", "next_action", "steps"}
        assert state["batch"]["id"] == batch_id
        assert state["sweep"] is None

    def test_unknown_batch_raises(self, db):
        with pytest.raises(ValueError):
            batch_state(db, 99999)


# =========================================================================
# Worker-pass states — one test per state
# =========================================================================

class TestWorkerPassStates:
    def test_needs_queue_when_nothing_done_and_no_job(self, db):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), "describe")
        assert step["state"] == "needs_queue"
        assert step["remaining"] == len(ids)
        assert step["done"] == 0
        assert step["failed"] == 0
        assert step["eligible"] == len(ids)

    def test_queued_when_an_open_job_exists(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "worker")
        assert _step(batch_state(db, batch_id), "describe")["state"] == "queued"

    def test_running_when_a_claim_intersects_the_batch(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "worker")  # running beats queued
        _claim(db, "describe", ids[:1])
        assert _step(batch_state(db, batch_id), "describe")["state"] == "running"

    def test_claim_that_does_not_intersect_is_not_running(self, db):
        batch_id, ids = _make_batch(db)
        other = db.add_photo(filepath="2091/other/x.jpg", filename="x.jpg")
        _claim(db, "describe", [other])
        assert _step(batch_state(db, batch_id), "describe")["state"] == "needs_queue"

    def test_claim_for_a_different_pass_is_not_running(self, db):
        batch_id, ids = _make_batch(db)
        _claim(db, "faces", ids)
        assert _step(batch_state(db, batch_id), "describe")["state"] == "needs_queue"

    def test_expired_claim_is_ignored(self, db):
        batch_id, ids = _make_batch(db)
        _claim(db, "describe", ids, expires_in_minutes=-5)
        assert _step(batch_state(db, batch_id), "describe")["state"] == "needs_queue"

    def test_completed_when_every_photo_has_output(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "describe")
        step = _step(batch_state(db, batch_id), "describe")
        assert step["state"] == "completed"
        assert step["remaining"] == 0
        assert step["failed"] == 0
        assert step["done"] == len(ids)

    def test_waiting_names_the_dependency(self, db):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), "keywords")
        assert step["state"] == "waiting"
        assert step["waiting_on"] == "describe"

    def test_blocked_when_remaining_zero_but_failures_exist(self, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "describe")
        step = _step(batch_state(db, batch_id), "describe")
        assert step["state"] == "blocked"

    def test_completed_pass_reports_no_waiting_on(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "describe")
        assert _step(batch_state(db, batch_id), "describe")["waiting_on"] is None


# =========================================================================
# (a) MANDATORY — the text passes must not read `completed` before describe
# =========================================================================

class TestTextPassesWaitForDescribe:
    @pytest.mark.parametrize("pass_type", ["category-content", "keywords", "verify"])
    def test_waiting_on_describe_never_completed(self, db, pass_type):
        batch_id, ids = _make_batch(db)
        # The trap: count_unprocessed_photos gates these on description, so it
        # returns 0 here — a naive `remaining == 0 -> completed` says done.
        from photosearch import worker_api
        assert worker_api._count_scoped(db, pass_type, ids) == 0

        step = _step(batch_state(db, batch_id), pass_type)
        assert step["state"] == "waiting", f"{pass_type} must wait, not {step['state']}"
        assert step["state"] != "completed"
        assert step["waiting_on"] == "describe"
        assert step["eligible"] == 0
        assert step["done"] == 0

    @pytest.mark.parametrize("pass_type", ["category-content", "keywords", "verify"])
    def test_becomes_actionable_once_descriptions_land(self, db, pass_type):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "describe")
        step = _step(batch_state(db, batch_id), pass_type)
        assert step["state"] == "needs_queue"
        assert step["eligible"] == len(ids)
        assert step["remaining"] == len(ids)

    def test_partial_descriptions_still_wait(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _set_col(db, ids[:2], "description", "Half of them.")
        step = _step(batch_state(db, batch_id), "keywords")
        assert step["state"] == "waiting"
        assert step["eligible"] == 2
        assert step["total"] == 4

    def test_empty_string_description_is_not_eligible(self, db):
        batch_id, ids = _make_batch(db)
        _set_col(db, ids, "description", "")
        assert _step(batch_state(db, batch_id), "keywords")["eligible"] == 0


# =========================================================================
# (b) MANDATORY — exhausted attempts must read `blocked`, never `completed`
# =========================================================================

class TestExhaustedAttemptsAreBlocked:
    @pytest.mark.parametrize(
        "pass_type", ["describe", "category-visual", "aesthetics"])
    def test_blocked_not_completed(self, db, pass_type):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, pass_type)

        # The trap: the claim predicate skips exhausted rows, so remaining is 0.
        from photosearch import worker_api
        assert worker_api._count_scoped(db, pass_type, ids) == 0

        step = _step(batch_state(db, batch_id), pass_type)
        assert step["state"] == "blocked", f"{pass_type} read {step['state']}"
        assert step["state"] != "completed"
        assert step["remaining"] == 0
        assert step["failed"] == len(ids)
        assert step["done"] == 0

    def test_attempts_below_the_cap_are_not_failures(self, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "describe", attempts=MAX_PROCESS_ATTEMPTS - 1)
        step = _step(batch_state(db, batch_id), "describe")
        assert step["failed"] == 0
        assert step["remaining"] == len(ids)
        assert step["state"] == "needs_queue"

    def test_exhausted_but_output_present_is_done_not_failed(self, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "describe")
        _complete_pass(db, ids, "describe")
        step = _step(batch_state(db, batch_id), "describe")
        assert step["failed"] == 0
        assert step["state"] == "completed"
        assert step["done"] == len(ids)

    def test_partial_failure_leaves_the_pass_actionable(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _exhaust(db, ids[:1], "describe")
        step = _step(batch_state(db, batch_id), "describe")
        assert step["failed"] == 1
        assert step["remaining"] == 3
        assert step["done"] == 0
        assert step["state"] == "needs_queue"

    # --- quality / verify joined the attempts ledger ------------------------
    #
    # They used to be the two passes (besides clip) whose claim predicate had
    # no attempts clause, so an exhausted photo stayed inside `remaining` and
    # needed the subset arithmetic (`done = eligible - remaining`,
    # `blocked = remaining == failed`). They now filter exhausted photos like
    # every other ledgered pass, so `failed` and `remaining` are DISJOINT and
    # the ordinary `done = eligible - remaining - failed` applies.

    def test_quality_failed_is_disjoint_from_remaining(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _do_quality(db, ids[:2])          # 2 genuinely scored
        _exhaust(db, ids[2:], "quality")  # 2 given up on
        step = _step(batch_state(db, batch_id), "quality")
        assert step["remaining"] == 0, "exhausted photos are no longer claimable"
        assert step["failed"] == 2
        assert step["done"] == 2, "the 2 scored photos are done"

    def test_quality_can_read_blocked(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _do_quality(db, ids[:2])
        _exhaust(db, ids[2:], "quality")
        step = _step(batch_state(db, batch_id), "quality")
        assert step["state"] == "blocked"

    def test_quality_stays_actionable_while_a_fresh_photo_remains(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _do_quality(db, ids[:2])
        _exhaust(db, ids[2:3], "quality")   # 1 exhausted, 1 untried
        step = _step(batch_state(db, batch_id), "quality")
        assert step["remaining"] == 1
        assert step["failed"] == 1
        assert step["done"] == 2
        assert step["state"] == "needs_queue"

    def test_quality_concepts_only_failure_counts_as_failed(self, db):
        """Score written, concepts NULL, attempts exhausted: that is the
        infinite re-claim the cap exists to stop — it reads failed, not
        remaining."""
        batch_id, ids = _make_batch(db, count=2)
        _do_quality(db, ids[:1])
        _set_col(db, ids[1:], "aesthetic_score", 5.0)
        _exhaust(db, ids[1:], "quality")
        step = _step(batch_state(db, batch_id), "quality")
        assert step["remaining"] == 0
        assert step["failed"] == 1
        assert step["done"] == 1
        assert step["state"] == "blocked"

    def test_verify_failed_is_disjoint_from_remaining(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _complete_pass(db, ids, "describe")          # all eligible
        _set_col(db, ids[:2], "verified_at", "2091-09-19 12:00:00")
        _exhaust(db, ids[2:], "verify")
        step = _step(batch_state(db, batch_id), "verify")
        assert step["eligible"] == 4
        assert step["remaining"] == 0
        assert step["failed"] == 2
        assert step["done"] == 2

    def test_verify_can_read_blocked(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _complete_pass(db, ids, "describe")
        _set_col(db, ids[:2], "verified_at", "2091-09-19 12:00:00")
        _exhaust(db, ids[2:], "verify")
        assert _step(batch_state(db, batch_id), "verify")["state"] == "blocked"

    def test_verify_below_the_cap_is_still_remaining(self, db):
        batch_id, ids = _make_batch(db, count=2)
        _complete_pass(db, ids, "describe")
        _exhaust(db, ids, "verify", attempts=MAX_PROCESS_ATTEMPTS - 1)
        step = _step(batch_state(db, batch_id), "verify")
        assert step["remaining"] == 2
        assert step["failed"] == 0
        assert step["state"] == "needs_queue"

    def test_quality_and_verify_still_complete_normally(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _complete_pass(db, ids, "describe")
        _do_quality(db, ids)
        _complete_pass(db, ids, "verify")
        state = batch_state(db, batch_id)
        for name in ("quality", "verify"):
            step = _step(state, name)
            assert step["state"] == "completed"
            assert step["done"] == 4

    def test_clip_has_no_attempts_ledger(self, db):
        batch_id, ids = _make_batch(db)
        _exhaust(db, ids, "clip")
        step = _step(batch_state(db, batch_id), "clip")
        assert step["failed"] == 0
        assert step["remaining"] == len(ids)
        assert step["state"] == "needs_queue"


# =========================================================================
# A batch whose photos are gone
# =========================================================================

class TestFacesWithNothingToFind:
    """`faces` is the one pass where "no output" is a legitimate result.

    A photo with nobody facing the camera has no `faces` rows after a perfectly
    successful run. The claim path cannot tell that from a failure, so the fleet
    re-tries it up to MAX_PROCESS_ATTEMPTS and then stops — and reading those as
    `failed` is wrong. Measured on the first real batch (2026-09-19, 1,373
    photos): 113 photos sat at attempts=3 with no face rows, every one a player
    facing away or a distant shot, while 3,696 faces were found in the other
    1,260. That read `blocked`, and held match_faces / warm_crops / rank_measure
    in `waiting` behind a pass that had in fact finished.
    """

    def test_exhausted_no_face_photos_count_as_done(self, db):
        batch_id, ids = _make_batch(db, count=4)
        _do_faces(db, ids[:3])
        _exhaust(db, ids[3:], "faces")          # detector ran 3x, found nobody
        step = _step(batch_state(db, batch_id), "faces")
        assert step["state"] == "completed", step
        assert step["failed"] == 0
        assert step["done"] == 4
        assert "1" in (step["detail"] or "") and "no detectable face" in step["detail"]

    def test_it_does_not_hold_the_face_steps_in_waiting(self, db):
        batch_id, ids = _make_batch(db, count=2)
        _do_faces(db, ids[:1])
        _exhaust(db, ids[1:], "faces")
        state = batch_state(db, batch_id)
        assert _step(state, "match_faces")["state"] != "waiting"
        assert _step(state, "warm_crops")["state"] != "waiting"

    def test_a_photo_still_being_retried_is_not_done_yet(self, db):
        batch_id, ids = _make_batch(db, count=2)
        _do_faces(db, ids[:1])
        _exhaust(db, ids[1:], "faces", attempts=MAX_PROCESS_ATTEMPTS - 1)
        step = _step(batch_state(db, batch_id), "faces")
        assert step["state"] == "needs_queue"    # the fleet will still claim it
        assert step["remaining"] == 1
        assert step["done"] == 1

    def test_no_detail_when_every_photo_has_a_face(self, db):
        batch_id, ids = _make_batch(db, count=2)
        _do_faces(db, ids)
        assert _step(batch_state(db, batch_id), "faces")["detail"] is None


class TestEmptiedBatch:
    """Membership is derived live from photos.folder, so a registered batch
    can empty out later — a dedup prune, purge-nonimage-photos, or a clock
    retime that rewrites `folder`. Every branch of
    count_unprocessed_photos tests `if photo_ids:`, so an empty id list
    falls through to the WHOLE-LIBRARY query: without a guard the batch
    reports the entire backlog as its own and asks for a fleet run."""

    def _emptied(self, db):
        batch_id, ids = _make_batch(db)
        ph = ",".join("?" * len(ids))
        db.conn.execute(f"DELETE FROM photos WHERE id IN ({ph})", ids)
        db.conn.commit()
        return batch_id

    def test_the_library_has_a_backlog_to_leak(self, db):
        # Guards the test below against passing vacuously: at least one pass
        # must have unprocessed photos outside the batch, or "remaining == 0"
        # would prove nothing about the empty-list fall-through.
        from photosearch import worker_api
        leaky = [p for p in WORKER_PASSES
                 if worker_api._count_scoped(db, p, []) > 0]
        assert leaky, "fixture library has no backlog; the leak test is vacuous"

    def test_no_step_reports_library_wide_numbers(self, db):
        batch_id = self._emptied(db)
        state = batch_state(db, batch_id)
        for step in state["steps"]:
            assert step["total"] == 0
            assert step["remaining"] == 0, f"{step['step']} leaked the library"
            assert step["eligible"] == 0
            assert step["done"] == 0
            assert step["failed"] == 0

    def test_next_action_is_not_launch_fleet(self, db):
        batch_id = self._emptied(db)
        state = batch_state(db, batch_id)
        # Documented choice: ready False (the frozen `completed` rule is
        # `total > 0 and done == total`, so an empty batch is never
        # completed) and next_action None (there is nothing to launch).
        assert state["ready"] is False
        assert state["next_action"] is None


# =========================================================================
# The ingest step
# =========================================================================

class TestIngestStep:
    def test_completed_when_no_sweep_references_the_batch(self, db):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), "ingest")
        assert step["state"] == "completed"
        assert step["done"] == len(ids)

    def test_completed_when_the_sweep_finished(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        set_sweep_status(db, run_id, "registered")
        assert _step(batch_state(db, batch_id), "ingest")["state"] == "completed"

    def test_running_while_the_sweep_is_moving(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        state = batch_state(db, batch_id)
        step = _step(state, "ingest")
        assert step["state"] == "running"
        assert step["detail"] is None
        assert state["sweep"]["run_id"] == run_id

    def test_running_while_the_sweep_is_indexing(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        set_sweep_status(db, run_id, "indexing")
        assert _step(batch_state(db, batch_id), "ingest")["state"] == "running"

    def test_detail_is_stalled_when_the_heartbeat_is_old(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        old = _fmt(_utcnow() - timedelta(seconds=STALL_SECONDS + 60))
        db.conn.execute("UPDATE ingest_sweeps SET heartbeat_at = ? WHERE run_id = ?",
                        (old, run_id))
        db.conn.commit()
        step = _step(batch_state(db, batch_id), "ingest")
        assert step["state"] == "running"
        assert step["detail"] == "stalled"

    def test_blocked_when_the_sweep_failed(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        set_sweep_status(db, run_id, "failed", error="disk full")
        step = _step(batch_state(db, batch_id), "ingest")
        assert step["state"] == "blocked"

    def test_another_batchs_running_sweep_is_ignored(self, db):
        batch_id, ids = _make_batch(db)              # no run_id
        start_sweep(db)                              # someone else's sweep
        assert _step(batch_state(db, batch_id), "ingest")["state"] == "completed"


# =========================================================================
# NAS steps
# =========================================================================

class TestStacking:
    def test_needs_queue_when_dated_photos_have_no_stack(self, db):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), "stacking")
        assert step["state"] == "needs_queue"
        assert step["remaining"] == len(ids)

    def test_completed_when_every_dated_photo_is_stacked(self, db):
        batch_id, ids = _make_batch(db)
        stack_id = db.conn.execute(
            "INSERT INTO photo_stacks (created_at) VALUES (datetime('now'))").lastrowid
        for pid in ids:
            db.conn.execute(
                "INSERT INTO stack_members (stack_id, photo_id, is_top) VALUES (?, ?, 0)",
                (stack_id, pid))
        db.conn.commit()
        assert _step(batch_state(db, batch_id), "stacking")["state"] == "completed"

    def test_a_closed_job_counts_as_done_for_a_shoot_with_no_bursts(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "stacking", "nas")
        close_job(db, batch_id, "stacking")
        # Nothing was stacked — legitimately, there were no bursts.
        assert _step(batch_state(db, batch_id), "stacking")["state"] == "completed"

    def test_undated_photos_are_not_counted(self, db):
        batch_id, ids = _make_batch(db)
        _set_col(db, ids, "date_taken", None)
        step = _step(batch_state(db, batch_id), "stacking")
        assert step["eligible"] == 0
        assert step["state"] == "completed"

    def test_queued_when_a_job_is_open(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "stacking", "nas")
        assert _step(batch_state(db, batch_id), "stacking")["state"] == "queued"


class TestNormalizeAesthetics:
    def test_waits_on_aesthetics_even_though_remaining_is_zero(self, db):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), "normalize_aesthetics")
        assert step["remaining"] == 0     # no aes_overall yet -> nothing to normalize
        assert step["state"] == "waiting"
        assert step["waiting_on"] == "aesthetics"

    def test_needs_queue_once_aesthetics_is_complete(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "aesthetics")
        step = _step(batch_state(db, batch_id), "normalize_aesthetics")
        assert step["state"] == "needs_queue"
        assert step["remaining"] == len(ids)

    def test_completed_when_percentiles_are_filled(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "aesthetics")
        _set_col(db, ids, "aes_overall_pct", 42.0)
        step = _step(batch_state(db, batch_id), "normalize_aesthetics")
        assert step["state"] == "completed"
        assert step["done"] == len(ids)


class TestJobOnlyNasSteps:
    @pytest.mark.parametrize("step_name", ["match_faces", "resolve_dups",
                                           "warm_crops", "rank_measure"])
    def test_waiting_until_the_dependency_completes(self, db, step_name):
        batch_id, ids = _make_batch(db)
        step = _step(batch_state(db, batch_id), step_name)
        assert step["state"] == "waiting"
        assert step["waiting_on"] == DEPENDS_ON[step_name]

    def test_needs_queue_once_faces_is_complete(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        assert _step(batch_state(db, batch_id), "match_faces")["state"] == "needs_queue"

    def test_queued_with_an_open_job(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        open_job(db, batch_id, "match_faces", "nas")
        assert _step(batch_state(db, batch_id), "match_faces")["state"] == "queued"

    def test_completed_with_a_closed_job(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        open_job(db, batch_id, "match_faces", "nas")
        close_job(db, batch_id, "match_faces")
        step = _step(batch_state(db, batch_id), "match_faces")
        assert step["state"] == "completed"
        assert step["done"] == len(ids)

    def test_resolve_dups_waits_on_match_faces(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        step = _step(batch_state(db, batch_id), "resolve_dups")
        assert step["state"] == "waiting"
        assert step["waiting_on"] == "match_faces"
        open_job(db, batch_id, "match_faces", "nas")
        close_job(db, batch_id, "match_faces")
        assert _step(batch_state(db, batch_id), "resolve_dups")["state"] == "needs_queue"


# =========================================================================
# ready + next_action
# =========================================================================

class TestReadyAndNextAction:
    def test_fresh_batch_wants_the_fleet(self, db):
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        assert state["ready"] is False
        assert state["next_action"] == "launch_fleet"

    def test_running_sweep_wins(self, db):
        run_id = start_sweep(db)
        batch_id, ids = _make_batch(db, run_id=run_id)
        assert batch_state(db, batch_id)["next_action"] == "wait_ingest"

    def test_advance_nas_once_the_worker_passes_are_done(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        state = batch_state(db, batch_id)
        assert all(_step(state, p)["state"] == "completed" for p in WORKER_PASSES)
        assert state["ready"] is False
        assert state["next_action"] == "advance_nas"

    def test_ready_when_every_worker_and_nas_step_is_complete(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids, include_optional=True)
        state = batch_state(db, batch_id)
        assert state["ready"] is True
        assert state["next_action"] is None

    def test_rank_measure_does_not_gate_ready(self, db):
        """Moving `rank_measure` into NAS_STEPS would silently make it gate
        `ready` — `ready` is "every step in WORKER_PASSES + NAS_STEPS". It is
        optional work (a ranking nicety), so OPTIONAL_STEPS is subtracted."""
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids)
        state = batch_state(db, batch_id)
        assert _step(state, "rank_measure")["state"] == "needs_queue"
        assert state["ready"] is True

    def test_a_ready_batch_still_offers_to_run_the_optional_step(self, db):
        """…and does NOT sit on `next_action: null`, which is what made
        `rank_measure` a dead end: the box read "Needs to be queued 0 / 1,373"
        forever with no button that would ever run it."""
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids)
        state = batch_state(db, batch_id)
        assert state["ready"] is True
        assert state["next_action"] == "advance_nas"

    def test_next_action_is_none_once_the_optional_step_has_run(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids, include_optional=True)
        state = batch_state(db, batch_id)
        assert _step(state, "rank_measure")["state"] == "completed"
        assert state["ready"] is True
        assert state["next_action"] is None

    def test_a_queued_optional_step_is_not_offered_again(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids)
        open_job(db, batch_id, "rank_measure", "nas")
        state = batch_state(db, batch_id)
        assert _step(state, "rank_measure")["state"] == "queued"
        assert state["ready"] is True
        assert state["next_action"] is None

    def test_review_blocked_when_nothing_is_actionable(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids)
        # Now break one pass: clear the output and exhaust its attempts.
        _set_col(db, ids, "visual_tags", None)
        _exhaust(db, ids, "category-visual")
        state = batch_state(db, batch_id)
        assert _step(state, "category-visual")["state"] == "blocked"
        assert state["ready"] is False
        assert state["next_action"] == "review_blocked"

    def test_wait_when_everything_left_is_in_flight(self, db):
        batch_id, ids = _make_batch(db)
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids)
        _set_col(db, ids, "visual_tags", None)
        _claim(db, "category-visual", ids)
        state = batch_state(db, batch_id)
        assert _step(state, "category-visual")["state"] == "running"
        assert state["next_action"] == "wait"

    def test_launch_fleet_outranks_advance_nas(self, db):
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        state = batch_state(db, batch_id)
        assert _step(state, "match_faces")["state"] == "needs_queue"
        assert state["next_action"] == "launch_fleet"


# =========================================================================
# Worker-pass precedence: running > completed > blocked > queued > waiting
# =========================================================================
#
# A worker pass's job row is opened by a fleet launch and NOTHING ever closes
# one — completion is provable from the data instead. So an open row must
# never outrank what the data says. When it did, a pass the fleet had already
# finished read `queued` for the row's full six-hour TTL: the batch could
# never reach `ready`, and the launch button's "a fleet is already running"
# refusal stayed armed long after the fleet had exited.

class TestWorkerPrecedenceOverAnOpenJob:
    def test_a_finished_pass_reads_completed_despite_an_open_fleet_job(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "fleet")
        _complete_pass(db, ids, "describe")
        step = _step(batch_state(db, batch_id), "describe")
        assert step["state"] == "completed"
        assert step["done"] == len(ids)

    def test_ready_is_reachable_with_open_fleet_rows_still_present(self, db):
        """The consequence that mattered: nothing closes a fleet row, so if
        `queued` won the batch could never be reviewed."""
        batch_id, ids = _make_batch(db)
        for p in WORKER_PASSES:
            open_job(db, batch_id, p, "fleet")
        _complete_all_worker_passes(db, ids)
        _complete_all_nas_steps(db, batch_id, ids, include_optional=True)
        state = batch_state(db, batch_id)
        assert state["ready"] is True
        assert state["next_action"] is None

    def test_an_exhausted_pass_reads_blocked_despite_an_open_job(self, db):
        """An open row must not hide a pass whose every remaining photo has
        exhausted its attempts — there is nothing queued about work that can
        never be claimed again."""
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "fleet")
        _exhaust(db, ids, "describe")
        assert _step(batch_state(db, batch_id), "describe")["state"] == "blocked"

    def test_an_unfinished_pass_with_an_open_job_still_reads_queued(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "fleet")
        assert _step(batch_state(db, batch_id), "describe")["state"] == "queued"

    def test_a_live_claim_still_outranks_everything(self, db):
        batch_id, ids = _make_batch(db)
        open_job(db, batch_id, "describe", "fleet")
        _exhaust(db, ids, "describe")
        _claim(db, "describe", ids[:1])
        assert _step(batch_state(db, batch_id), "describe")["state"] == "running"

    def test_job_only_steps_keep_the_old_ordering(self, db):
        """For match_faces/resolve_dups/warm_crops a CLOSED row is the only
        evidence of success, so an open one still means queued — the change
        above is about worker passes only."""
        batch_id, ids = _make_batch(db)
        _complete_pass(db, ids, "faces")
        open_job(db, batch_id, "match_faces", "nas")
        assert _step(batch_state(db, batch_id), "match_faces")["state"] == "queued"


# =========================================================================
# fleet_launch_passes — what ONE launch covers
# =========================================================================
#
# MIRRORED in JS as PS.BatchFlow.fleetLaunchPasses (frontend/dist/
# batch-flow.js). The button says "N passes" and this decides which N, so the
# two must not drift: the five scenarios below are duplicated case-for-case
# in frontend/__tests__/batch-flow.test.js.

def _fake_state(states: dict) -> dict:
    """A batch_state-shaped response with the given worker-pass states."""
    return {"steps": [
        {"step": p, "kind": "worker", "state": states.get(p, "needs_queue"),
         "total": 3, "eligible": 3, "done": 0, "remaining": 3, "failed": 0,
         "waiting_on": DEPENDS_ON.get(p) if states.get(p) == "waiting" else None,
         "detail": None}
        for p in WORKER_PASSES]}


class TestFleetLaunchPasses:
    def test_fresh_batch_launches_the_whole_pipeline_in_order(self, db):
        """Case 1. The gated passes are `waiting` at click time, and a second
        launch mid-run is refused, so leaving them out meant they were never
        queued by the button at all."""
        batch_id, ids = _make_batch(db)
        state = batch_state(db, batch_id)
        assert fleet_launch_passes(state) == list(WORKER_PASSES)

    def test_a_completed_dependency_admits_its_dependents(self, db):
        """Case 2."""
        got = fleet_launch_passes(_fake_state({
            "describe": "completed", "category-content": "waiting",
            "keywords": "waiting", "verify": "waiting"}))
        assert got == ["clip", "faces", "quality", "aesthetics",
                       "category-visual", "category-content", "keywords", "verify"]

    def test_a_blocked_dependency_does_not_admit_its_dependents(self, db):
        """Case 3. `describe` having given up on every photo means there will
        be no descriptions for the text passes to read."""
        got = fleet_launch_passes(_fake_state({
            "describe": "blocked", "category-content": "waiting",
            "keywords": "waiting", "verify": "waiting"}))
        assert "category-content" not in got
        assert "keywords" not in got
        assert "verify" not in got
        assert "describe" not in got

    def test_a_running_or_queued_dependency_admits_its_dependents(self, db):
        """Case 4. Another fleet is mid-describe; its dependents still need
        queueing and will have input by the time they are claimed."""
        for dep_state in ("running", "queued"):
            got = fleet_launch_passes(_fake_state({
                "describe": dep_state, "category-content": "waiting",
                "keywords": "waiting", "verify": "waiting"}))
            assert "category-content" in got, dep_state
            assert "describe" not in got, dep_state

    def test_nothing_to_launch_is_an_empty_list(self, db):
        """Case 5."""
        assert fleet_launch_passes(_fake_state(
            {p: "completed" for p in WORKER_PASSES})) == []
        assert fleet_launch_passes({"steps": []}) == []
        assert fleet_launch_passes({}) == []

    def test_only_worker_passes_are_ever_returned(self, db):
        batch_id, ids = _make_batch(db)
        got = fleet_launch_passes(batch_state(db, batch_id))
        assert set(got) <= set(WORKER_PASSES)
        assert not set(got) & set(NAS_STEPS)

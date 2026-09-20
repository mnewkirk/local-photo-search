"""Tests for photosearch/batch_advance.py + the two advance endpoints
(M "ingest batch" Task 6).

`batch-advance` is the ONE action that takes a batch toward "ready to
review", so the things worth pinning are the ones that would quietly do the
wrong work:

- **`clear` is never truthy on a scoped stacking run.** Scoped stacking with
  a clear has twice wiped the whole library's stacks. `run_stacking` has no
  `clear` parameter at all today, so the test pins both halves: no truthy
  `clear` in the call, and the scope that decides what gets cleared
  (`photo_ids`) is the batch's own ids.
- **Only the STRICT face matcher runs.** `match_faces_temporal` is ~4%
  accurate on these shoots (CLAUDE.md) and a batch advance must never
  silently pour that into a fresh folder.
- **Dry-run writes nothing** — no job rows, no runner calls. A job row that
  leaked out of a preview would read as `queued` for its whole TTL and stop
  the real run from ever starting.
- **A failure DELETES the job row** — not closed (that reads `completed`) and
  not left open (that reads `queued`, making the step unretryable for its
  whole six-hour TTL). The step goes back to `needs_queue` and a retry
  re-runs it.
- **In replica mode every `/api/batches` route proxies to the NAS**, and an
  unreachable NAS is an error rather than a silent fall back to the synced
  copy. A stale pipeline view is worse than a visible failure.

Per globals.md the shared `db` fixture is pre-seeded under `2026/`, so every
fixture here builds photos under `2090/` and `2091/`.
"""

import json
from unittest.mock import patch

import pytest

from photosearch import batch_advance, batch_api, batch_state, ingest_batches

_DIR = "2090/2090-03-01_TEST"


# =========================================================================
# fixtures / helpers
# =========================================================================

@pytest.fixture(autouse=True)
def _reset_batch_cache():
    batch_api._reset_cache()
    yield
    if batch_api._lock.locked():
        batch_api._lock.release()
    batch_api._reset_cache()


@pytest.fixture(autouse=True)
def _no_nas_url(monkeypatch):
    """Most tests are the single-machine case. A stray PHOTOSEARCH_NAS_URL in
    the developer's environment would silently flip every endpoint into the
    replica-proxy branch."""
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)


def _make_batch(db, directory: str = _DIR, count: int = 3):
    ids = []
    for i in range(count):
        ids.append(db.add_photo(
            filepath=f"{directory}/img{i:03d}.jpg",
            filename=f"img{i:03d}.jpg",
            date_taken=f"2090-03-01T10:0{i}:00",
        ))
    return ingest_batches.register_batch(db, directory), ids


def _finish_aesthetics(db, ids):
    """Make the `aesthetics` worker pass read `completed` while leaving
    `normalize_aesthetics` with work to do (percentile still NULL)."""
    db.conn.executemany(
        "UPDATE photos SET aes_overall = 7.0, aes_overall_pct = NULL WHERE id = ?",
        [(i,) for i in ids])
    db.conn.commit()


def _finish_faces(db, ids):
    """Give every photo a face row so the `faces` worker pass reads
    `completed` — which is what `match_faces` and `warm_crops` wait on."""
    for pid in ids:
        db.add_face(pid, (10, 60, 60, 10), [0.0] * 512)
    db.conn.commit()


def _fake_runners(calls, fail_on=None):
    def make(step):
        def run(db, ctx):
            calls.append(step)
            if fail_on == step:
                raise RuntimeError(f"{step} exploded")
            return {"step": step}
        return run
    return {s: make(s) for s in batch_state.NAS_STEPS}


def _by_step(result):
    return {r["step"]: r for r in result["steps"]}


def _step_state(db, batch_id, step):
    return next(s["state"] for s in batch_state.batch_state(db, batch_id)["steps"]
                if s["step"] == step)


# =========================================================================
# advance_nas_steps — orchestration
# =========================================================================

class TestOrchestration:
    def test_runs_needs_queue_steps_in_nas_step_order(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        calls = []

        result = batch_advance.advance_nas_steps(
            db, batch_id, apply=True, runners=_fake_runners(calls))

        assert calls == list(batch_state.NAS_STEPS)
        assert result["ran"] == list(batch_state.NAS_STEPS)
        assert result["stopped_at"] is None

    def test_stops_at_the_first_waiting_step(self, db):
        """`faces` has not run, so `match_faces` is waiting on it — and
        everything after it must be left alone, not skipped past."""
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        calls = []

        result = batch_advance.advance_nas_steps(
            db, batch_id, apply=True, runners=_fake_runners(calls))

        assert calls == ["stacking", "normalize_aesthetics"]
        assert result["stopped_at"] == "match_faces"
        assert _by_step(result)["match_faces"]["waiting_on"] == "faces"
        assert "warm_crops" not in _by_step(result)

    def test_completed_steps_are_skipped_not_rerun(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        # A closed job is what makes a job-only step read `completed`.
        ingest_batches.open_job(db, batch_id, "match_faces", "nas")
        ingest_batches.close_job(db, batch_id, "match_faces")
        calls = []

        batch_advance.advance_nas_steps(
            db, batch_id, apply=True, runners=_fake_runners(calls))

        assert "match_faces" not in calls
        assert calls == ["stacking", "normalize_aesthetics",
                         "resolve_dups", "warm_crops"]

    def test_job_rows_are_opened_then_closed(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)

        batch_advance.advance_nas_steps(
            db, batch_id, apply=True, runners=_fake_runners([]))

        closed = ingest_batches.closed_jobs(db, batch_id)
        assert set(batch_state.NAS_STEPS) <= closed
        assert ingest_batches.open_jobs(db, batch_id) == {}

    def test_failure_deletes_the_job_row_and_reports_the_error(self, db):
        """The row must go, not linger open and not be closed.

        Closed would read `completed` (a job-only step's only evidence of
        success). Open would read `queued`, so the retry skips the step and
        stops at the next thing depending on it — for the full six-hour TTL,
        with no recovery short of editing the table by hand.
        """
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        calls = []

        result = batch_advance.advance_nas_steps(
            db, batch_id, apply=True,
            runners=_fake_runners(calls, fail_on="normalize_aesthetics"))

        assert "normalize_aesthetics" not in ingest_batches.open_jobs(db, batch_id)
        assert "normalize_aesthetics" not in ingest_batches.closed_jobs(db, batch_id)
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs WHERE batch_id = ? AND step = ?",
            (batch_id, "normalize_aesthetics")).fetchone()[0] == 0
        row = _by_step(result)["normalize_aesthetics"]
        assert row["status"] == "failed"
        assert "exploded" in row["error"]
        assert "exploded" in result["error"]
        # And it stops — a later step could depend on the one that failed.
        assert result["stopped_at"] == "normalize_aesthetics"
        assert calls == ["stacking", "normalize_aesthetics"]

    def test_a_retry_after_a_failure_re_runs_the_failed_step(self, db):
        """The point of deleting the row: the operator's fix is to press the
        button again, not to open sqlite3."""
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        first, second = [], []

        batch_advance.advance_nas_steps(
            db, batch_id, apply=True,
            runners=_fake_runners(first, fail_on="normalize_aesthetics"))
        result = batch_advance.advance_nas_steps(
            db, batch_id, apply=True, runners=_fake_runners(second))

        assert "normalize_aesthetics" in second
        assert result["stopped_at"] is None
        assert set(batch_state.NAS_STEPS) <= ingest_batches.closed_jobs(db, batch_id)

    def test_an_abort_mid_step_deletes_that_step_s_job_row(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)

        def boom(db_, ctx):
            raise InterruptedError("cancelled")

        runners = dict(_fake_runners([]))
        runners["stacking"] = boom
        with pytest.raises(InterruptedError):
            batch_advance.advance_nas_steps(db, batch_id, apply=True, runners=runners)

        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs WHERE batch_id = ?",
            (batch_id,)).fetchone()[0] == 0

    def test_dry_run_writes_nothing_and_calls_no_runner(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        calls = []

        result = batch_advance.advance_nas_steps(
            db, batch_id, runners=_fake_runners(calls))

        assert calls == []
        assert ingest_batches.open_jobs(db, batch_id) == {}
        assert ingest_batches.closed_jobs(db, batch_id) == set()
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs WHERE batch_id = ?",
            (batch_id,)).fetchone()[0] == 0
        assert [r["step"] for r in result["steps"] if r["status"] == "would_run"] \
            == list(batch_state.NAS_STEPS)
        assert result["apply"] is False

    def test_dry_run_plans_past_a_dependency_it_would_itself_satisfy(self, db):
        """`resolve_dups` waits on `match_faces`, which is a step in this very
        run. A preview that stopped there would under-report the plan by two
        steps every single time."""
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)

        result = batch_advance.advance_nas_steps(db, batch_id)

        assert _by_step(result)["resolve_dups"]["status"] == "would_run"

    def test_should_abort_between_steps_raises_interrupted(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        calls = []
        seen = {"n": 0}

        def should_abort():
            seen["n"] += 1
            return seen["n"] > 1

        with pytest.raises(InterruptedError):
            batch_advance.advance_nas_steps(
                db, batch_id, apply=True, should_abort=should_abort,
                runners=_fake_runners(calls))

        assert calls == ["stacking"]

    def test_progress_events_name_the_step(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        events = []

        batch_advance.advance_nas_steps(
            db, batch_id, apply=True, on_progress=events.append,
            runners=_fake_runners([]))

        assert {e["step"] for e in events} >= {"stacking", "normalize_aesthetics"}
        assert all(e.get("phase") == "batch-advance" for e in events)

    def test_a_runner_result_cannot_overwrite_the_step_outcome(self, db):
        """`maintenance._stage_resolve_dups` returns `{"status": "skipped"}`
        when it finds nothing to change. Splatted over the event it would
        report a step that ran (and whose job is closed) as skipped."""
        batch_id, ids = _make_batch(db)
        events = []
        runners = {s: (lambda d, c: {"status": "skipped", "phase": "sweep"})
                   for s in batch_state.NAS_STEPS}

        batch_advance.advance_nas_steps(
            db, batch_id, apply=True, on_progress=events.append, runners=runners)

        done = [e for e in events if e["step"] == "stacking" and e["status"] == "done"]
        assert len(done) == 1
        assert done[0]["phase"] == "batch-advance"
        assert "stacking" in ingest_batches.closed_jobs(db, batch_id)

    def test_unknown_batch_raises(self, db):
        with pytest.raises(ValueError):
            batch_advance.advance_nas_steps(db, 99999, apply=True)


# =========================================================================
# step-name validation — `ingest_batch_jobs.step` is unvalidated storage
# =========================================================================

class TestStepValidation:
    def test_open_step_job_rejects_an_unknown_step(self, db):
        """A writer typo would sit in the table reading `needs_queue`
        forever, because nothing downstream validates the string."""
        batch_id, _ = _make_batch(db)
        with pytest.raises(ValueError):
            batch_advance.open_step_job(db, batch_id, "stackign", "nas")
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs").fetchone()[0] == 0

    def test_open_step_job_accepts_every_known_step(self, db):
        batch_id, _ = _make_batch(db)
        for step in batch_state.STEP_ORDER:
            batch_advance.open_step_job(db, batch_id, step, "nas")
        assert set(ingest_batches.open_jobs(db, batch_id)) == set(batch_state.STEP_ORDER)

    def test_runners_with_an_unknown_step_are_rejected(self, db):
        batch_id, _ = _make_batch(db)
        with pytest.raises(ValueError):
            batch_advance.advance_nas_steps(
                db, batch_id, apply=True, runners={"stackign": lambda *a: None})


# =========================================================================
# default runners — what they actually call
# =========================================================================

class TestDefaultRunners:
    def _ctx(self, db, batch_id):
        batch = ingest_batches.get_batch(db, batch_id)
        return {
            "batch": batch, "batch_id": batch_id,
            "photo_ids": ingest_batches.batch_photo_ids(db, batch),
            "day": "2090-03-01", "emit": lambda ev: None,
            "check_abort": lambda: None, "apply": True,
        }

    def test_stacking_never_passes_a_truthy_clear_and_is_scoped(self, db):
        """THE safety test. Scoped stacking with a clear flag has twice wiped
        the whole library's stacks; `run_stacking` decides its clear scope
        from `photo_ids`, so passing the batch's ids is what keeps a batch
        advance from touching any other folder's stacks."""
        batch_id, ids = _make_batch(db)
        seen = {}

        def fake_run_stacking(db_, **kw):
            seen.update(kw)
            return [[ids[0], ids[1]]]

        from photosearch import stacking
        with patch.object(stacking, "run_stacking", fake_run_stacking):
            out = batch_advance.default_runners()["stacking"](
                db, self._ctx(db, batch_id))

        assert not seen.get("clear")
        assert seen["photo_ids"] == ids
        assert seen["dry_run"] is False
        # `directory` would make run_stacking re-resolve its own clear scope.
        assert seen.get("directory") is None
        assert out["stacks"] == 1

    def test_stacking_refuses_an_empty_batch_rather_than_going_library_wide(self, db):
        """`detect_stacks` reads `photo_ids=[]` as "no scope given" and falls
        through to the whole library — and `save_stacks` would then clear
        every stack in it. A batch CAN empty out: membership is derived live
        from photos.folder, so a prune, a purge or a clock retime does it."""
        batch_id, _ = _make_batch(db)
        ctx = self._ctx(db, batch_id)
        ctx["photo_ids"] = []

        from photosearch import stacking

        def must_not_run(*a, **k):  # pragma: no cover - the whole point
            raise AssertionError("run_stacking must not be called for an empty batch")

        with patch.object(stacking, "run_stacking", must_not_run):
            out = batch_advance.default_runners()["stacking"](db, ctx)
        assert out["stacks"] == 0

    def test_match_faces_uses_the_strict_matcher_only(self, db):
        """`temporal` is ~4% accurate on these shoots — pouring it into a
        fresh batch would make the folder's labels worse, not better."""
        batch_id, ids = _make_batch(db)
        seen = {}

        def fake_strict(db_, **kw):
            seen.update(kw)
            return 7

        def fake_temporal(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("temporal matcher must never run in a batch advance")

        from photosearch import faces
        with patch.object(faces, "match_faces_to_persons", fake_strict), \
                patch.object(faces, "match_faces_temporal", fake_temporal):
            out = batch_advance.default_runners()["match_faces"](
                db, self._ctx(db, batch_id))

        assert seen["photo_ids"] == ids
        assert out["matched"] == 7

    def test_normalize_aesthetics_calls_the_maintenance_stage(self, db):
        batch_id, _ = _make_batch(db)
        seen = {}

        def fake_stage(db_, apply, emit, check_abort, **kw):
            seen["apply"] = apply
            return {"stage": "normalize_aesthetics", "applied": 3}

        from photosearch import maintenance
        with patch.object(maintenance, "_stage_normalize_aesthetics", fake_stage):
            out = batch_advance.default_runners()["normalize_aesthetics"](
                db, self._ctx(db, batch_id))
        assert seen["apply"] is True
        assert out["applied"] == 3

    def test_resolve_dups_calls_the_maintenance_stage(self, db):
        batch_id, _ = _make_batch(db)
        seen = {}

        def fake_stage(db_, apply, emit, check_abort, **kw):
            seen["apply"] = apply
            return {"stage": "resolve_dups", "applied": 0}

        from photosearch import maintenance
        with patch.object(maintenance, "_stage_resolve_dups", fake_stage):
            batch_advance.default_runners()["resolve_dups"](
                db, self._ctx(db, batch_id))
        assert seen["apply"] is True

    def test_warm_crops_is_scoped_to_the_batch(self, db):
        batch_id, ids = _make_batch(db)
        seen = {}

        def fake_warm(db_, **kw):
            seen.update(kw)
            return {"ok": 2, "missing": 0, "errors": 0, "total": 2}

        from photosearch import face_crop
        with patch.object(face_crop, "warm_crops", fake_warm):
            batch_advance.default_runners()["warm_crops"](
                db, self._ctx(db, batch_id))

        assert seen["photo_ids"] == ids


# =========================================================================
# fleet_directory — the `-d` argument
# =========================================================================

class TestFleetDirectory:
    def test_builds_the_absolute_photos_form(self):
        assert batch_advance.fleet_directory(
            "/photos", "2091/2091-09-19_ILCE-7RM6") == "/photos/2091/2091-09-19_ILCE-7RM6"

    def test_refuses_the_photo_root_itself(self):
        """`-d /photos` always 404s: get_directory_photo_ids strips the root
        prefix, leaving an empty prefix that matches no relative path."""
        for bad in ("", "   ", "/", "."):
            with pytest.raises(ValueError):
                batch_advance.fleet_directory("/photos", bad)

    def test_defaults_the_root_when_the_db_has_none(self):
        assert batch_advance.fleet_directory(None, "2091/x") == "/photos/2091/x"


# =========================================================================
# POST /api/admin/batch-advance  (SSE)
# =========================================================================

def _advance_cmd(client, body):
    seen = {}

    async def fake_stream(cmd, cwd=None, env=None):
        seen["cmd"] = cmd
        seen["env"] = env
        yield 'event: done\ndata: {"returncode": 0}\n\n'

    with patch.object(batch_advance_admin(), "_stream_subprocess", fake_stream):
        r = client.post("/api/admin/batch-advance", json=body)
        assert r.status_code == 200, r.text
        r.read()
    return seen


def batch_advance_admin():
    from photosearch import admin_api
    return admin_api


class TestAdvanceEndpoint:
    def test_runs_the_cli_with_the_batch_and_apply(self, client, db):
        batch_id, _ = _make_batch(db)
        seen = _advance_cmd(client, {"batch_id": batch_id, "apply": True})
        cmd = seen["cmd"]
        assert "batch-advance" in cmd
        assert cmd[cmd.index("--batch") + 1] == str(batch_id)
        assert "--apply" in cmd

    def test_dry_run_is_the_default_and_omits_apply(self, client, db):
        batch_id, _ = _make_batch(db)
        cmd = _advance_cmd(client, {"batch_id": batch_id})["cmd"]
        assert "--apply" not in cmd

    def test_unknown_batch_is_404(self, client, db):
        r = client.post("/api/admin/batch-advance", json={"batch_id": 4242})
        assert r.status_code == 404

    def test_second_run_is_refused_while_one_is_active(self, client, db):
        batch_id, _ = _make_batch(db)
        admin = batch_advance_admin()
        admin._ingest_lock.acquire()
        try:
            r = client.post("/api/admin/batch-advance", json={"batch_id": batch_id})
            assert r.status_code == 409
        finally:
            admin._ingest_lock.release()

    def test_replica_mode_proxies_the_stream_to_the_nas(self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)
        monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.example/")
        seen = {}

        class FakeResp:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            def iter_content(self, chunk_size=1):
                yield b'event: done\ndata: {"returncode": 0}\n\n'

            def close(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_post(url, **kw):
            seen["url"] = url
            seen["json"] = kw.get("json")
            return FakeResp()

        import requests
        monkeypatch.setattr(requests, "post", fake_post)
        r = client.post("/api/admin/batch-advance",
                        json={"batch_id": batch_id, "apply": True})
        assert r.status_code == 200
        body = r.read().decode()
        assert seen["url"].endswith("/api/admin/batch-advance")
        assert seen["json"] == {"batch_id": batch_id, "apply": True}
        assert "done" in body


# =========================================================================
# POST /api/admin/batch-launch-fleet
# =========================================================================

def _fleet_cmd(client, body):
    seen = {}

    class R:
        returncode = 0
        stdout = "launched"
        stderr = ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return R()

    admin = batch_advance_admin()
    with patch.object(admin.subprocess, "run", fake_run):
        r = client.post("/api/admin/batch-launch-fleet", json=body)
    seen["status"] = r.status_code
    seen["body"] = r.json() if r.headers.get("content-type", "").startswith(
        "application/json") else {}
    return seen


class TestLaunchFleetEndpoint:
    def test_one_click_launches_the_whole_pipeline_in_worker_pass_order(self, client, db):
        """The three description-gated passes are `waiting`, not
        `needs_queue`, before `describe` has run — but the fleet runs
        `--sequential` through the dependency order, and a second launch
        mid-run is refused (it would kill the running fleet). So they ride
        along on this launch or they are never queued by the button at all."""
        batch_id, ids = _make_batch(db)
        seen = _fleet_cmd(client, {"batch_id": batch_id, "count": 2})
        assert seen["status"] == 200, seen
        cmd = seen["cmd"]
        passes = cmd[cmd.index("-p") + 1].split(",")
        assert passes == list(batch_state.WORKER_PASSES)
        assert cmd[cmd.index("-n") + 1] == "2"
        assert "--sequential" in cmd

    def test_every_launched_pass_gets_a_fleet_job_row(self, client, db):
        """Including the ones that were `waiting`: with the new precedence a
        waiting pass with an open row reads `queued`, which is the truthful
        display — the fleet really is going to run it."""
        batch_id, _ = _make_batch(db)
        _fleet_cmd(client, {"batch_id": batch_id})
        assert set(ingest_batches.open_jobs(db, batch_id)) == set(
            batch_state.WORKER_PASSES)
        state = batch_state.batch_state(db, batch_id)
        by = {s["step"]: s["state"] for s in state["steps"]}
        assert by["keywords"] == "queued"
        assert state["next_action"] != "launch_fleet"

    def test_scopes_the_fleet_to_the_batch_directory(self, client, db):
        batch_id, _ = _make_batch(db)
        cmd = _fleet_cmd(client, {"batch_id": batch_id})["cmd"]
        assert cmd[cmd.index("-d") + 1] == "/photos/" + _DIR

    def test_opens_a_fleet_job_per_pass(self, client, db):
        batch_id, _ = _make_batch(db)
        seen = _fleet_cmd(client, {"batch_id": batch_id})
        opened = ingest_batches.open_jobs(db, batch_id)
        assert set(opened) == set(seen["body"]["passes"])
        assert all(r["job_kind"] == "fleet" for r in opened.values())

    def test_400_when_run_workers_script_is_missing(self, client, db, monkeypatch):
        batch_id, _ = _make_batch(db)
        admin = batch_advance_admin()
        monkeypatch.setattr(admin, "_run_workers_script",
                            lambda: "/nonexistent/run-workers.sh")
        r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": batch_id})
        assert r.status_code == 400
        assert "run-workers.sh" in r.json()["detail"]

    def test_400_when_no_pass_needs_queueing(self, client, db):
        """Every worker pass done means the answer is `advance-batch`, not a
        fleet that would claim nothing."""
        batch_id, ids = _make_batch(db)
        from photosearch import batch_state as bs
        # Built without touching `db` — this runs on the TestClient's thread,
        # and the fixture's sqlite connection belongs to the test's.
        with patch.object(bs, "batch_state", lambda d, b: {
            "batch": {"id": b, "directory": _DIR}, "ready": True,
            "next_action": None,
            "steps": [{"step": p, "kind": "worker", "state": "completed",
                       "total": 3, "eligible": 3, "done": 3, "remaining": 0,
                       "failed": 0, "waiting_on": None, "detail": None}
                      for p in bs.WORKER_PASSES],
        }):
            r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": batch_id})
        assert r.status_code == 400

    def test_unknown_batch_is_404(self, client, db):
        r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": 4242})
        assert r.status_code == 404


class TestWorkersStartRequestDirectory:
    def test_rejects_directory_together_with_filters(self):
        from pydantic import ValidationError
        from photosearch.admin_api import WorkersStartRequest
        with pytest.raises(ValidationError):
            WorkersStartRequest(passes=["clip"], directory="/photos/2091/x",
                                filters={"camera": "ILCE-7RM6"})

    def test_rejects_directory_together_with_collection(self):
        from pydantic import ValidationError
        from photosearch.admin_api import WorkersStartRequest
        with pytest.raises(ValidationError):
            WorkersStartRequest(passes=["clip"], directory="/photos/2091/x",
                                collection=4)

    def test_directory_alone_is_fine(self):
        from photosearch.admin_api import WorkersStartRequest
        req = WorkersStartRequest(passes=["clip"], directory="/photos/2091/x")
        assert req.directory == "/photos/2091/x"


# =========================================================================
# POST /api/batches/{id}/jobs — the replica's way to write on the NAS
# =========================================================================

class TestLaunchFleetAlreadyRunning:
    def test_409_when_a_worker_pass_already_has_an_open_fleet_job(self, client, db):
        """A second `run-workers.sh --name ui` KILLS the running fleet and
        starts over, so a double-click or a second tab must be refused. An
        open job row derives as `queued`, which is also the only way to see
        the NAS's job rows from a replica."""
        batch_id, _ = _make_batch(db)
        ingest_batches.open_job(db, batch_id, "clip", "fleet")
        r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": batch_id})
        assert r.status_code == 409
        assert "already" in r.json()["detail"]

    def test_a_launch_makes_the_next_one_409(self, client, db):
        batch_id, _ = _make_batch(db)
        assert _fleet_cmd(client, {"batch_id": batch_id})["status"] == 200
        assert _fleet_cmd(client, {"batch_id": batch_id})["status"] == 409

    def test_409_on_a_live_claim_even_with_no_job_row(self, client, db):
        """A fleet launched from the CLI leaves no job row, only claims."""
        batch_id, ids = _make_batch(db)
        db.conn.execute(
            "INSERT INTO worker_claims (batch_id, worker_id, pass_type, "
            "  photo_ids, claimed_at, expires_at) "
            "VALUES ('c1', 'w1', 'clip', ?, datetime('now'), "
            "        datetime('now', '+30 minutes'))",
            (json.dumps(ids),))
        db.conn.commit()
        r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": batch_id})
        assert r.status_code == 409

    def test_a_finished_fleet_no_longer_blocks_a_relaunch(self, client, db):
        """The 409's whole failure mode before: nothing closes a fleet job
        row, so with `queued` outranking `completed` the refusal stayed armed
        for six hours after the fleet had exited."""
        batch_id, ids = _make_batch(db)
        assert _fleet_cmd(client, {"batch_id": batch_id})["status"] == 200
        # clip finishes; its row is still open and nothing will ever close it.
        for pid in ids:
            db.add_clip_embedding(pid, [0.0] * 511 + [1.0])
        db.conn.commit()
        assert _step_state(db, batch_id, "clip") == "completed"
        # Everything else is still queued, so this one is still refused —
        # but for the right reason, naming only the unfinished passes.
        r = client.post("/api/admin/batch-launch-fleet", json={"batch_id": batch_id})
        assert r.status_code == 409
        assert "clip" not in r.json()["detail"]


# =========================================================================
# Replica mode — every /api/batches route proxies to the NAS
# =========================================================================

class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def nas(monkeypatch):
    """Replica mode with a recording stand-in for `requests.request`."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas.example:8000/")
    calls = []
    box = {"response": FakeResponse({"ok": True}), "raise": None}

    def fake_request(method, url, params=None, json=None, timeout=None):
        calls.append({"method": method, "url": url, "params": params,
                      "json": json, "timeout": timeout})
        if box["raise"] is not None:
            raise box["raise"]
        resp = box["response"]
        return resp(url) if callable(resp) else resp

    import requests
    monkeypatch.setattr(requests, "request", fake_request)
    return {"calls": calls, "box": box}


class TestReplicaProxy:
    def test_list_comes_from_the_nas_not_the_local_copy(self, client, db, nas):
        """The replica's DB is a periodically synced copy — a batch
        registered since the last sync is simply missing from it."""
        _make_batch(db)   # exists locally; must NOT be what we answer with
        nas["box"]["response"] = FakeResponse(
            {"sweep": None, "batches": [{"id": 77, "directory": "2091/x"}]})
        data = client.get("/api/batches").json()
        assert [b["id"] for b in data["batches"]] == [77]
        assert nas["calls"][0]["url"] == "http://nas.example:8000/api/batches"

    def test_detail_comes_from_the_nas(self, client, db, nas):
        batch_id, _ = _make_batch(db)
        nas["box"]["response"] = FakeResponse(
            {"batch": {"id": batch_id}, "steps": [], "ready": True})
        assert client.get(f"/api/batches/{batch_id}").json()["ready"] is True
        assert nas["calls"][0]["url"].endswith(f"/api/batches/{batch_id}")

    @pytest.mark.parametrize("method,path,body", [
        ("post", "/api/batches/1/dismiss", None),
        ("post", "/api/batches/1/ready", None),
        ("post", "/api/batches/1/jobs", {"steps": ["clip"]}),
        ("post", "/api/batches/register", {"directory": "2091/x"}),
    ])
    def test_writes_go_to_the_nas_and_never_touch_the_local_db(
            self, client, db, nas, method, path, body):
        """The NAS is the sole writer; a local write is destroyed by the next
        sync-replica.sh, which swaps PHOTOSEARCH_DB wholesale."""
        _make_batch(db)
        getattr(client, method)(path, json=body)
        assert nas["calls"], f"{path} did not proxy"
        assert nas["calls"][0]["url"].startswith("http://nas.example:8000")
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs").fetchone()[0] == 0

    def test_an_unreachable_nas_is_an_error_not_stale_local_data(
            self, client, db, nas):
        """A wrong batch view is worse than an error: the operator who sees
        'could not reach' knows what to do, the one shown a stale pipeline
        does not."""
        import requests
        _make_batch(db)
        nas["box"]["raise"] = requests.ConnectionError("refused")
        for path in ("/api/batches", "/api/batches/1"):
            r = client.get(path)
            assert r.status_code == 502, path
            assert "could not reach" in r.json()["detail"]

    def test_the_nas_s_own_status_codes_survive_the_hop(self, client, db, nas):
        _make_batch(db)
        nas["box"]["response"] = FakeResponse(
            {"detail": "no such batch: 4242"}, status_code=404)
        r = client.get("/api/batches/4242")
        assert r.status_code == 404
        assert r.json()["detail"] == "no such batch: 4242"


class TestLaunchFleetInReplicaMode:
    def _nas_state(self, batch_id, states):
        return FakeResponse({
            "batch": {"id": batch_id, "directory": _DIR},
            "ready": False, "next_action": "launch_fleet",
            "steps": [{"step": p, "kind": "worker", "state": st, "total": 3,
                       "eligible": 3, "done": 0, "remaining": 3, "failed": 0,
                       "waiting_on": None, "detail": None}
                      for p, st in states.items()],
        })

    def test_passes_are_read_from_the_nas_not_the_replica(self, client, db, nas):
        """The replica's worker-pass counts lag the NAS by up to a full sync,
        so a local read keeps offering a launch for passes already draining."""
        batch_id, _ = _make_batch(db)
        nas["box"]["response"] = self._nas_state(batch_id, {
            "clip": "completed", "faces": "needs_queue", "quality": "needs_queue"})
        seen = _fleet_cmd(client, {"batch_id": batch_id})
        assert seen["status"] == 200, seen
        assert seen["cmd"][seen["cmd"].index("-p") + 1] == "faces,quality"
        # And the job intent was POSTed to the NAS, not written locally.
        assert any(c["url"].endswith(f"/api/batches/{batch_id}/jobs")
                   for c in nas["calls"])
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs").fetchone()[0] == 0

    def test_409_from_the_nas_s_queued_passes(self, client, db, nas):
        batch_id, _ = _make_batch(db)
        nas["box"]["response"] = self._nas_state(batch_id, {
            "clip": "queued", "faces": "needs_queue"})
        assert _fleet_cmd(client, {"batch_id": batch_id})["status"] == 409

    def test_503_while_the_nas_is_still_computing(self, client, db, nas):
        """An empty `steps` list would read as 'no pass needs queueing' — a
        wrong answer, not a slow one."""
        batch_id, _ = _make_batch(db)
        nas["box"]["response"] = FakeResponse(
            {"batch": {"id": batch_id}, "steps": [], "computing": True})
        assert _fleet_cmd(client, {"batch_id": batch_id})["status"] == 503

    def test_unreachable_nas_is_502_not_a_local_launch(self, client, db, nas):
        import requests
        batch_id, _ = _make_batch(db)
        nas["box"]["raise"] = requests.ConnectionError("refused")
        seen = _fleet_cmd(client, {"batch_id": batch_id})
        assert seen["status"] == 502
        assert "cmd" not in seen


class TestJobsEndpoint:
    def test_opens_the_named_steps(self, client, db):
        batch_id, _ = _make_batch(db)
        r = client.post(f"/api/batches/{batch_id}/jobs",
                        json={"steps": ["clip", "faces"], "job_kind": "fleet"})
        assert r.status_code == 200, r.text
        opened = ingest_batches.open_jobs(db, batch_id)
        assert set(opened) == {"clip", "faces"}
        assert opened["clip"]["job_kind"] == "fleet"

    def test_rejects_an_unknown_step(self, client, db):
        batch_id, _ = _make_batch(db)
        r = client.post(f"/api/batches/{batch_id}/jobs", json={"steps": ["clipp"]})
        assert r.status_code == 400
        assert db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batch_jobs").fetchone()[0] == 0

    def test_unknown_batch_is_404(self, client, db):
        r = client.post("/api/batches/4242/jobs", json={"steps": ["clip"]})
        assert r.status_code == 404

    def test_invalidates_the_memo(self, client, db):
        batch_id, _ = _make_batch(db)
        first = client.get(f"/api/batches/{batch_id}").json()
        assert first["steps"][1]["state"] == "needs_queue"   # clip
        client.post(f"/api/batches/{batch_id}/jobs", json={"steps": ["clip"]})
        second = client.get(f"/api/batches/{batch_id}").json()
        assert second["steps"][1]["state"] == "queued"

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
- **A failure leaves the job open**, so the step reads `queued` until the TTL
  expires rather than `completed`.

Per globals.md the shared `db` fixture is pre-seeded under `2026/`, so every
fixture here builds photos under `2090/` and `2091/`.
"""

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

    def test_failure_leaves_the_job_open_and_reports_the_error(self, db):
        batch_id, ids = _make_batch(db)
        _finish_aesthetics(db, ids)
        _finish_faces(db, ids)
        calls = []

        result = batch_advance.advance_nas_steps(
            db, batch_id, apply=True,
            runners=_fake_runners(calls, fail_on="normalize_aesthetics"))

        assert "normalize_aesthetics" in ingest_batches.open_jobs(db, batch_id)
        assert "normalize_aesthetics" not in ingest_batches.closed_jobs(db, batch_id)
        row = _by_step(result)["normalize_aesthetics"]
        assert row["status"] == "failed"
        assert "exploded" in row["error"]
        assert "exploded" in result["error"]
        # And it stops — a later step could depend on the one that failed.
        assert result["stopped_at"] == "normalize_aesthetics"
        assert calls == ["stacking", "normalize_aesthetics"]

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
    def test_picks_only_needs_queue_passes_in_worker_pass_order(self, client, db):
        """The three description-gated passes are `waiting`, not
        `needs_queue`, before `describe` has run — queueing them would have
        the fleet claim zero photos and retire."""
        batch_id, ids = _make_batch(db)
        seen = _fleet_cmd(client, {"batch_id": batch_id, "count": 2})
        assert seen["status"] == 200, seen
        cmd = seen["cmd"]
        passes = cmd[cmd.index("-p") + 1].split(",")
        assert passes == ["clip", "faces", "quality", "aesthetics", "describe",
                          "category-visual"]
        assert cmd[cmd.index("-n") + 1] == "2"
        assert "--sequential" in cmd

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

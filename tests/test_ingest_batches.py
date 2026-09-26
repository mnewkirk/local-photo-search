"""Tests for photosearch/ingest_batches.py (schema v30, M "ingest batch" Task 1).

Covers: sweep lifecycle (start/heartbeat/status/active-lookup + the computed
`stalled`/`files_per_min` fields), batch register/re-open/list/get/dismiss/
ready, directory normalization parity, and per-step job open/close/expiry.

A batch's photo membership is never materialized — `batch_photo_ids` always
derives it live from `photos.folder` — so these tests add real photo rows
under 2090/2091 folders (per globals.md: the shared `db` fixture is
pre-seeded under 2026/, so new fixtures use different years) rather than
faking membership some other way.
"""

from datetime import datetime, timedelta, timezone

import pytest

from photosearch.ingest_batches import (
    STALL_SECONDS,
    start_sweep,
    heartbeat_sweep,
    set_sweep_status,
    get_active_sweep,
    register_batch,
    list_batches,
    get_batch,
    batch_photo_ids,
    mark_ready,
    dismiss_batch,
    open_job,
    close_job,
    open_jobs,
    closed_jobs,
)


_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _fmt(dt: datetime) -> str:
    return dt.strftime(_TS_FORMAT)


def _add_photos(db, directory: str, count: int, start: int = 0) -> list[int]:
    ids = []
    for i in range(start, start + count):
        ids.append(db.add_photo(
            filepath=f"{directory}/img{i:03d}.jpg",
            filename=f"img{i:03d}.jpg",
            date_taken="2091-09-19T10:00:00",
        ))
    return ids


# =========================================================================
# Sweep lifecycle
# =========================================================================

class TestSweepLifecycle:
    def test_start_sweep_returns_uuid4_hex_run_id(self, db):
        run_id = start_sweep(db)
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # uuid4().hex has no dashes
        int(run_id, 16)  # is valid hex

    def test_start_sweep_row_defaults(self, db):
        run_id = start_sweep(db)
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "moving"
        assert row["files_seen"] == 0
        assert row["files_moved"] == 0
        assert row["finished_at"] is None
        assert row["error"] is None

    def test_heartbeat_sweep_updates_counts(self, db):
        run_id = start_sweep(db)
        heartbeat_sweep(db, run_id, files_seen=10, files_moved=4)
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["files_seen"] == 10
        assert row["files_moved"] == 4

    def test_set_sweep_status_transitions_and_terminal_finished_at(self, db):
        run_id = start_sweep(db)
        set_sweep_status(db, run_id, "indexing")
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "indexing"
        assert row["finished_at"] is None  # not terminal yet

        set_sweep_status(db, run_id, "registered")
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "registered"
        assert row["finished_at"] is not None
        assert row["error"] is None

    def test_set_sweep_status_failed_records_error(self, db):
        run_id = start_sweep(db)
        set_sweep_status(db, run_id, "failed", error="disk full")
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error"] == "disk full"
        assert row["finished_at"] is not None


class TestGetActiveSweep:
    def test_none_when_no_sweeps(self, db):
        assert get_active_sweep(db) is None

    def test_none_when_only_terminal_sweeps(self, db):
        run_id = start_sweep(db)
        set_sweep_status(db, run_id, "registered")
        assert get_active_sweep(db) is None

    def test_returns_active_moving_sweep(self, db):
        run_id = start_sweep(db)
        active = get_active_sweep(db)
        assert active is not None
        assert active["run_id"] == run_id
        assert active["status"] == "moving"

    def test_returns_active_indexing_sweep(self, db):
        run_id = start_sweep(db)
        set_sweep_status(db, run_id, "indexing")
        active = get_active_sweep(db)
        assert active["run_id"] == run_id
        assert active["status"] == "indexing"

    def test_returns_newest_active_sweep(self, db):
        older = start_sweep(db)
        db.conn.execute(
            "UPDATE ingest_sweeps SET started_at = ?, heartbeat_at = ? WHERE run_id = ?",
            (_fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)),
             _fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=10)), older),
        )
        db.conn.commit()
        newer = start_sweep(db)

        active = get_active_sweep(db)
        assert active["run_id"] == newer

    def test_files_per_min_zero_when_span_under_one_second(self, db):
        run_id = start_sweep(db)
        # started_at == heartbeat_at (both stamped at INSERT time).
        active = get_active_sweep(db)
        assert active["files_per_min"] == 0.0

    def test_files_per_min_computed_over_span(self, db):
        run_id = start_sweep(db)
        started = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=2)
        heartbeat = datetime.now(timezone.utc).replace(tzinfo=None)
        db.conn.execute(
            "UPDATE ingest_sweeps SET started_at = ?, heartbeat_at = ?, files_moved = ? "
            "WHERE run_id = ?",
            (_fmt(started), _fmt(heartbeat), 20, run_id),
        )
        db.conn.commit()
        active = get_active_sweep(db)
        # 20 files over ~2 minutes -> ~10/min
        assert active["files_per_min"] == pytest.approx(10.0, rel=0.1)

    def test_stalled_true_when_moving_and_heartbeat_old(self, db):
        run_id = start_sweep(db)
        old_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=STALL_SECONDS + 60)
        db.conn.execute(
            "UPDATE ingest_sweeps SET heartbeat_at = ? WHERE run_id = ?",
            (_fmt(old_heartbeat), run_id),
        )
        db.conn.commit()
        active = get_active_sweep(db)
        assert active["stalled"] is True

    def test_not_stalled_when_heartbeat_recent(self, db):
        run_id = start_sweep(db)
        active = get_active_sweep(db)
        assert active["stalled"] is False

    def test_not_stalled_when_indexing_even_if_heartbeat_old(self, db):
        run_id = start_sweep(db)
        set_sweep_status(db, run_id, "indexing")
        old_heartbeat = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=STALL_SECONDS + 60)
        db.conn.execute(
            "UPDATE ingest_sweeps SET heartbeat_at = ? WHERE run_id = ?",
            (_fmt(old_heartbeat), run_id),
        )
        db.conn.commit()
        active = get_active_sweep(db)
        assert active["stalled"] is False


# =========================================================================
# register_batch — identity, normalization, re-open rule
# =========================================================================

class TestRegisterBatch:
    def test_raises_on_empty_directory(self, db):
        with pytest.raises(ValueError):
            register_batch(db, "2090/2090-01-01_nonexistent")

    def test_creates_batch_with_photo_count(self, db):
        _add_photos(db, "2090/2090-01-01_ILCE-7RM6", 3)
        batch_id = register_batch(db, "2090/2090-01-01_ILCE-7RM6", source="cron", run_id="r1")
        batch = get_batch(db, batch_id)
        assert batch["directory"] == "2090/2090-01-01_ILCE-7RM6"
        assert batch["photo_count"] == 3
        assert batch["source"] == "cron"
        assert batch["run_id"] == "r1"
        assert batch["ready_at"] is None
        assert batch["dismissed_at"] is None

    @pytest.mark.parametrize("spelling", [
        "2090/2090-01-02_ILCE-7RM6",
        "./2090/2090-01-02_ILCE-7RM6",
        "./2090/2090-01-02_ILCE-7RM6/",
        "/photos/2090/2090-01-02_ILCE-7RM6",
        "  2090/2090-01-02_ILCE-7RM6  ",
    ])
    def test_directory_spellings_normalize_to_one_row(self, db, spelling):
        _add_photos(db, "2090/2090-01-02_ILCE-7RM6", 2)
        first_id = register_batch(db, "2090/2090-01-02_ILCE-7RM6")
        second_id = register_batch(db, spelling)
        assert first_id == second_id
        rows = db.conn.execute(
            "SELECT COUNT(*) FROM ingest_batches WHERE directory = ?",
            ("2090/2090-01-02_ILCE-7RM6",),
        ).fetchone()[0]
        assert rows == 1

    def test_reopen_without_growth_is_idempotent_upsert(self, db):
        _add_photos(db, "2090/2090-01-03_ILCE-7RM6", 2)
        first_id = register_batch(db, "2090/2090-01-03_ILCE-7RM6")
        second_id = register_batch(db, "2090/2090-01-03_ILCE-7RM6")
        assert first_id == second_id
        count = db.conn.execute("SELECT COUNT(*) FROM ingest_batches").fetchone()[0]
        assert count == 1

    def test_reopen_with_growth_clears_ready_and_dismissed(self, db):
        _add_photos(db, "2090/2090-01-04_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-04_ILCE-7RM6")
        mark_ready(db, batch_id)
        dismiss_batch(db, batch_id)
        batch = get_batch(db, batch_id)
        assert batch["ready_at"] is not None
        assert batch["dismissed_at"] is not None

        _add_photos(db, "2090/2090-01-04_ILCE-7RM6", 1, start=2)  # grows 2 -> 3
        register_batch(db, "2090/2090-01-04_ILCE-7RM6")
        batch = get_batch(db, batch_id)
        assert batch["photo_count"] == 3
        assert batch["ready_at"] is None
        assert batch["dismissed_at"] is None

    def test_reopen_with_growth_invalidates_job_only_completion(self, db):
        """A closed job row is the ONLY completion evidence the job-only steps
        have (match_faces / resolve_dups / warm_crops / rank_measure write no
        per-photo column). When late photos re-open a batch, that row would
        still read `completed` although the new photos have no matched faces,
        no warmed crops and no measurement — so the page would say the batch
        was finished when a third of it had never been touched. Growth
        therefore drops those rows; the derived NAS steps (stacking,
        normalize_aesthetics) need no help — they count photos.
        """
        from photosearch.batch_state import JOB_ONLY_STEPS

        _add_photos(db, "2090/2090-01-09_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-09_ILCE-7RM6")
        for step in JOB_ONLY_STEPS:
            open_job(db, batch_id, step, "nas")
            close_job(db, batch_id, step)
        open_job(db, batch_id, "stacking", "nas")   # still open: not touched
        assert closed_jobs(db, batch_id) == set(JOB_ONLY_STEPS)

        _add_photos(db, "2090/2090-01-09_ILCE-7RM6", 1, start=2)
        register_batch(db, "2090/2090-01-09_ILCE-7RM6")

        assert closed_jobs(db, batch_id) == set()
        assert "stacking" in open_jobs(db, batch_id)

    def test_reopen_without_growth_keeps_job_only_completion(self, db):
        """A plain re-scan must not undo finished work — nothing changed."""
        from photosearch.batch_state import JOB_ONLY_STEPS

        _add_photos(db, "2090/2090-01-10_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-10_ILCE-7RM6")
        open_job(db, batch_id, JOB_ONLY_STEPS[0], "nas")
        close_job(db, batch_id, JOB_ONLY_STEPS[0])

        register_batch(db, "2090/2090-01-10_ILCE-7RM6")

        assert closed_jobs(db, batch_id) == {JOB_ONLY_STEPS[0]}

    def test_reopen_without_growth_preserves_ready_and_dismissed(self, db):
        _add_photos(db, "2090/2090-01-05_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-05_ILCE-7RM6")
        mark_ready(db, batch_id)
        dismiss_batch(db, batch_id)
        batch_before = get_batch(db, batch_id)

        register_batch(db, "2090/2090-01-05_ILCE-7RM6")  # same photo_count
        batch_after = get_batch(db, batch_id)
        assert batch_after["photo_count"] == 2
        assert batch_after["ready_at"] == batch_before["ready_at"]
        assert batch_after["dismissed_at"] == batch_before["dismissed_at"]

    def test_reopen_overwrites_source_and_run_id_only_when_not_none(self, db):
        _add_photos(db, "2090/2090-01-06_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-06_ILCE-7RM6", source="cron", run_id="r1")

        register_batch(db, "2090/2090-01-06_ILCE-7RM6")  # both None -> preserved
        batch = get_batch(db, batch_id)
        assert batch["source"] == "cron"
        assert batch["run_id"] == "r1"

        register_batch(db, "2090/2090-01-06_ILCE-7RM6", source="manual")  # run_id None -> preserved
        batch = get_batch(db, batch_id)
        assert batch["source"] == "manual"
        assert batch["run_id"] == "r1"

    def test_reopen_updates_updated_at(self, db):
        _add_photos(db, "2090/2090-01-07_ILCE-7RM6", 2)
        batch_id = register_batch(db, "2090/2090-01-07_ILCE-7RM6")
        db.conn.execute(
            "UPDATE ingest_batches SET updated_at = ? WHERE id = ?",
            (_fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1)), batch_id),
        )
        db.conn.commit()
        before = get_batch(db, batch_id)["updated_at"]

        register_batch(db, "2090/2090-01-07_ILCE-7RM6")
        after = get_batch(db, batch_id)["updated_at"]
        assert after != before


# =========================================================================
# list_batches / get_batch / batch_photo_ids / mark_ready / dismiss_batch
# =========================================================================

class TestBatchQueries:
    def test_get_batch_returns_none_for_missing(self, db):
        assert get_batch(db, 999999) is None

    def test_batch_photo_ids_matches_folder(self, db):
        ids = _add_photos(db, "2090/2090-02-01_ILCE-7RM6", 3)
        batch_id = register_batch(db, "2090/2090-02-01_ILCE-7RM6")
        batch = get_batch(db, batch_id)
        assert sorted(batch_photo_ids(db, batch)) == sorted(ids)

    def test_batch_photo_ids_derived_live_not_snapshotted(self, db):
        _add_photos(db, "2090/2090-02-02_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-02-02_ILCE-7RM6")
        batch = get_batch(db, batch_id)
        assert len(batch_photo_ids(db, batch)) == 1

        _add_photos(db, "2090/2090-02-02_ILCE-7RM6", 1, start=1)  # new photo, no re-register
        assert len(batch_photo_ids(db, batch)) == 2

    def test_list_batches_excludes_dismissed_by_default(self, db):
        _add_photos(db, "2090/2090-02-03_a", 1)
        _add_photos(db, "2090/2090-02-03_b", 1)
        id_a = register_batch(db, "2090/2090-02-03_a")
        id_b = register_batch(db, "2090/2090-02-03_b")
        dismiss_batch(db, id_a)

        visible_ids = {b["id"] for b in list_batches(db)}
        assert id_a not in visible_ids
        assert id_b in visible_ids

        all_ids = {b["id"] for b in list_batches(db, include_dismissed=True)}
        assert id_a in all_ids
        assert id_b in all_ids

    def test_mark_ready_sets_timestamp(self, db):
        _add_photos(db, "2090/2090-02-04_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-02-04_ILCE-7RM6")
        assert get_batch(db, batch_id)["ready_at"] is None
        mark_ready(db, batch_id)
        assert get_batch(db, batch_id)["ready_at"] is not None

    def test_dismiss_batch_sets_timestamp(self, db):
        _add_photos(db, "2090/2090-02-05_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-02-05_ILCE-7RM6")
        assert get_batch(db, batch_id)["dismissed_at"] is None
        dismiss_batch(db, batch_id)
        assert get_batch(db, batch_id)["dismissed_at"] is not None


# =========================================================================
# Job open/close/expiry
# =========================================================================

class TestJobs:
    def test_open_job_appears_in_open_jobs(self, db):
        _add_photos(db, "2090/2090-03-01_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-01_ILCE-7RM6")
        open_job(db, batch_id, "clip", "worker")
        jobs = open_jobs(db, batch_id)
        assert "clip" in jobs
        assert jobs["clip"]["job_kind"] == "worker"
        assert jobs["clip"]["closed_at"] is None

    def test_close_job_removes_from_open_jobs_and_adds_to_closed(self, db):
        _add_photos(db, "2090/2090-03-02_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-02_ILCE-7RM6")
        open_job(db, batch_id, "faces", "worker")
        close_job(db, batch_id, "faces")

        assert "faces" not in open_jobs(db, batch_id)
        assert "faces" in closed_jobs(db, batch_id)

    def test_reopen_job_clears_closed_at(self, db):
        _add_photos(db, "2090/2090-03-03_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-03_ILCE-7RM6")
        open_job(db, batch_id, "quality", "worker")
        close_job(db, batch_id, "quality")
        assert "quality" in closed_jobs(db, batch_id)

        open_job(db, batch_id, "quality", "worker")
        assert "quality" in open_jobs(db, batch_id)
        assert "quality" not in closed_jobs(db, batch_id)

        row = db.conn.execute(
            "SELECT closed_at FROM ingest_batch_jobs WHERE batch_id = ? AND step = ?",
            (batch_id, "quality"),
        ).fetchone()
        assert row["closed_at"] is None

    def test_reopen_job_resets_opened_at_and_expires_at(self, db):
        _add_photos(db, "2090/2090-03-04_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-04_ILCE-7RM6")
        open_job(db, batch_id, "describe", "worker", ttl_seconds=60)
        old_expires = db.conn.execute(
            "SELECT expires_at FROM ingest_batch_jobs WHERE batch_id = ? AND step = ?",
            (batch_id, "describe"),
        ).fetchone()["expires_at"]

        # Force the row's opened_at/expires_at far into the past so a re-open
        # is verifiably a reset, not a no-op that happened to already qualify.
        db.conn.execute(
            "UPDATE ingest_batch_jobs SET opened_at = ?, expires_at = ? "
            "WHERE batch_id = ? AND step = ?",
            (_fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)),
             _fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)),
             batch_id, "describe"),
        )
        db.conn.commit()
        assert "describe" not in open_jobs(db, batch_id)  # expired

        open_job(db, batch_id, "describe", "worker", ttl_seconds=3600)
        assert "describe" in open_jobs(db, batch_id)

    def test_expired_job_excluded_from_open_jobs(self, db):
        _add_photos(db, "2090/2090-03-05_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-05_ILCE-7RM6")
        open_job(db, batch_id, "verify", "worker")
        db.conn.execute(
            "UPDATE ingest_batch_jobs SET expires_at = ? WHERE batch_id = ? AND step = ?",
            (_fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)), batch_id, "verify"),
        )
        db.conn.commit()
        assert "verify" not in open_jobs(db, batch_id)

    def test_expired_job_not_in_closed_jobs_either(self, db):
        """Expired-but-never-closed is a distinct state from closed."""
        _add_photos(db, "2090/2090-03-06_ILCE-7RM6", 1)
        batch_id = register_batch(db, "2090/2090-03-06_ILCE-7RM6")
        open_job(db, batch_id, "keywords", "worker")
        db.conn.execute(
            "UPDATE ingest_batch_jobs SET expires_at = ? WHERE batch_id = ? AND step = ?",
            (_fmt(datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1)), batch_id, "keywords"),
        )
        db.conn.commit()
        assert "keywords" not in closed_jobs(db, batch_id)

    def test_open_jobs_scoped_to_batch(self, db):
        _add_photos(db, "2090/2090-03-07_a", 1)
        _add_photos(db, "2090/2090-03-07_b", 1)
        batch_a = register_batch(db, "2090/2090-03-07_a")
        batch_b = register_batch(db, "2090/2090-03-07_b")
        open_job(db, batch_a, "clip", "worker")
        open_job(db, batch_b, "clip", "worker")

        assert set(open_jobs(db, batch_a).keys()) == {"clip"}
        assert set(open_jobs(db, batch_b).keys()) == {"clip"}

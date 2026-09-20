"""Ingest batches (schema v30) — identity, lifecycle, sweep progress, and
per-step job intent for one dated ingest folder.

**A batch is one dated folder**: ``ingest_batches.directory`` is exactly
``photos.folder`` (relative to ``photo_root``, e.g.
``2091/2091-09-19_ILCE-7RM6``). Membership is derived **live** —
``batch_photo_ids`` always runs ``SELECT id FROM photos WHERE folder = ?`` —
never materialized into a table or a collection, so a batch row only ever
holds identity, lifecycle timestamps, sweep progress, and job intent.

Three tables, one module function group per table:

- ``ingest_sweeps`` — one row per move+index sweep run
  (``start_sweep`` / ``heartbeat_sweep`` / ``set_sweep_status`` /
  ``get_active_sweep``).
- ``ingest_batches`` — one row per directory, upserted by ``register_batch``
  (re-opening an existing directory is the norm, not an error — a sweep
  that adds more files to today's folder just widens the same batch).
- ``ingest_batch_jobs`` — one row per (batch, step) recording that some
  downstream job (worker-fleet pass, NAS stage, desktop step — Task 3
  assigns the ``job_kind``/``step`` vocabulary) was told to run and whether
  it has finished. TTL-gated the same way ``worker_claims`` is, so a crashed
  job doesn't wedge a batch forever.

This module only implements the schema-level primitives; the batch-readiness
state machine (STEP_ORDER, DEPENDS_ON, batch_state()) is Task 3's
``photosearch/batch_state.py``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

# A "moving" sweep whose heartbeat hasn't moved in this long is presumed
# dead (crashed mid-move) rather than merely slow. Only status=='moving' is
# checked — 'indexing' hands off to index_directory()/the worker fleet,
# which have their own liveness signals.
STALL_SECONDS = 300

# SQLite's datetime('now') (and datetime('now', modifier)) format: UTC,
# space-separated, no fractional seconds. Every timestamp column in the
# three tables below is stamped with one of those two forms, so this is the
# one format Python-side parsing ever needs to handle.
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, _TS_FORMAT)


# ---------------------------------------------------------------------------
# ingest_sweeps
# ---------------------------------------------------------------------------

def start_sweep(db) -> str:
    """Start a new sweep run. Returns its run_id (a uuid4 hex string).

    A sweep always starts in the 'moving' phase (files are being relocated
    into the dated library folders); ``set_sweep_status`` advances it to
    'indexing', then a terminal 'registered' or 'failed'.
    """
    run_id = uuid.uuid4().hex
    db.conn.execute(
        "INSERT INTO ingest_sweeps "
        "  (run_id, status, started_at, heartbeat_at, files_seen, files_moved) "
        "VALUES (?, 'moving', datetime('now'), datetime('now'), 0, 0)",
        (run_id,),
    )
    db.conn.commit()
    return run_id


def heartbeat_sweep(db, run_id: str, *, files_seen: int, files_moved: int) -> None:
    """Record sweep progress and bump the liveness heartbeat."""
    db.conn.execute(
        "UPDATE ingest_sweeps SET heartbeat_at = datetime('now'), "
        "  files_seen = ?, files_moved = ? "
        "WHERE run_id = ?",
        (files_seen, files_moved, run_id),
    )
    db.conn.commit()


def set_sweep_status(db, run_id: str, status: str, error: Optional[str] = None) -> None:
    """Advance a sweep's status: moving -> indexing -> registered | failed.

    ``finished_at`` is stamped exactly when the new status is terminal
    (anything other than 'moving'/'indexing') — `get_active_sweep` uses that
    same moving/indexing pair to decide what's still running.
    """
    terminal = status not in ("moving", "indexing")
    db.conn.execute(
        "UPDATE ingest_sweeps SET status = ?, error = ?, "
        "  finished_at = CASE WHEN ? THEN datetime('now') ELSE finished_at END "
        "WHERE run_id = ?",
        (status, error, terminal, run_id),
    )
    db.conn.commit()


def get_sweep(db, run_id: str) -> Optional[dict]:
    """One sweep by run_id, whatever its status.

    ``get_active_sweep`` only ever returns moving/indexing rows, so a caller
    that needs to know a *specific* sweep reached 'failed' (batch_state's
    ingest step does) has to look it up directly. The computed ``stalled`` /
    ``files_per_min`` fields are added here too, so the two lookups return
    the same shape.
    """
    row = db.conn.execute(
        "SELECT * FROM ingest_sweeps WHERE run_id = ?", (run_id,)
    ).fetchone()
    return _enrich_sweep(row) if row else None


def _enrich_sweep(row) -> dict:
    """Add the two computed fields the status page needs and the table
    doesn't store: ``stalled`` and ``files_per_min``."""
    sweep = dict(row)
    started = _parse_ts(sweep["started_at"])
    heartbeat = _parse_ts(sweep["heartbeat_at"])
    span_seconds = (heartbeat - started).total_seconds()
    sweep["files_per_min"] = (
        sweep["files_moved"] / (span_seconds / 60.0) if span_seconds >= 1 else 0.0
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    since_heartbeat = (now - heartbeat).total_seconds()
    sweep["stalled"] = sweep["status"] == "moving" and since_heartbeat > STALL_SECONDS
    return sweep


def get_active_sweep(db) -> Optional[dict]:
    """The newest sweep still in progress (status 'moving' or 'indexing'),
    with two computed fields the status page needs and the table doesn't
    store: ``stalled`` and ``files_per_min``. None if nothing is running.
    """
    row = db.conn.execute(
        "SELECT * FROM ingest_sweeps WHERE status IN ('moving', 'indexing') "
        "ORDER BY started_at DESC, rowid DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return _enrich_sweep(row)


# ---------------------------------------------------------------------------
# ingest_batches
# ---------------------------------------------------------------------------

def register_batch(db, directory: str, *, source: Optional[str] = None,
                    run_id: Optional[str] = None) -> int:
    """Upsert the batch for ``directory``. Re-opens an existing row.

    Normalizes ``directory`` exactly as ``db.normalize_directory`` (the same
    rule ``db.get_directory_photo_ids`` uses for worker-fleet scoping), so
    ``./2091/x/``, ``/photos/2091/x``, and ``2091/x`` all resolve to one row.

    Raises ``ValueError`` if the directory has no photos — guards hand-made
    subfolders and typos rather than silently registering an empty batch.

    Re-open rule: ``photo_count``/``updated_at`` are always refreshed. If the
    count grew since the last call, ``ready_at``/``dismissed_at`` are cleared
    (new files mean the batch isn't the one that was marked ready/dismissed
    anymore). If the count didn't grow, they're left alone. ``source``/
    ``run_id`` are overwritten only when the new value is not None, so a
    plain re-scan (called with neither) never clobbers who/what registered
    the batch originally.
    """
    directory = db.normalize_directory(directory)
    photo_count = db.conn.execute(
        "SELECT COUNT(*) FROM photos WHERE folder = ?", (directory,)
    ).fetchone()[0]
    if photo_count == 0:
        raise ValueError(f"no photos found in directory {directory!r}")

    existing = db.conn.execute(
        "SELECT id, photo_count FROM ingest_batches WHERE directory = ?", (directory,)
    ).fetchone()

    if existing is None:
        cur = db.conn.execute(
            "INSERT INTO ingest_batches (directory, source, run_id, photo_count) "
            "VALUES (?, ?, ?, ?)",
            (directory, source, run_id, photo_count),
        )
        db.conn.commit()
        return cur.lastrowid

    batch_id = existing["id"]
    grew = photo_count > existing["photo_count"]
    if grew:
        db.conn.execute(
            "UPDATE ingest_batches SET photo_count = ?, updated_at = datetime('now'), "
            "  ready_at = NULL, dismissed_at = NULL, "
            "  source = COALESCE(?, source), run_id = COALESCE(?, run_id) "
            "WHERE id = ?",
            (photo_count, source, run_id, batch_id),
        )
    else:
        db.conn.execute(
            "UPDATE ingest_batches SET photo_count = ?, updated_at = datetime('now'), "
            "  source = COALESCE(?, source), run_id = COALESCE(?, run_id) "
            "WHERE id = ?",
            (photo_count, source, run_id, batch_id),
        )
    db.conn.commit()
    return batch_id


def list_batches(db, include_dismissed: bool = False, limit: int = 50) -> list[dict]:
    """Batches newest-first. Dismissed batches are hidden unless asked for."""
    sql = "SELECT * FROM ingest_batches"
    if not include_dismissed:
        sql += " WHERE dismissed_at IS NULL"
    sql += " ORDER BY created_at DESC LIMIT ?"
    rows = db.conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_batch(db, batch_id: int) -> Optional[dict]:
    row = db.conn.execute(
        "SELECT * FROM ingest_batches WHERE id = ?", (batch_id,)
    ).fetchone()
    return dict(row) if row else None


def batch_photo_ids(db, batch: dict) -> list[int]:
    """Live membership: every photo whose folder is this batch's directory.

    Never materialized — a batch stores only ``directory``, so this is
    recomputed on every call. Takes the batch dict (from ``get_batch`` /
    ``list_batches``) rather than a bare id, since every caller has one on
    hand already and this avoids an extra round-trip.
    """
    rows = db.conn.execute(
        "SELECT id FROM photos WHERE folder = ?", (batch["directory"],)
    ).fetchall()
    return [r[0] for r in rows]


def mark_ready(db, batch_id: int) -> None:
    db.conn.execute(
        "UPDATE ingest_batches SET ready_at = datetime('now'), "
        "  updated_at = datetime('now') "
        "WHERE id = ?",
        (batch_id,),
    )
    db.conn.commit()


def dismiss_batch(db, batch_id: int) -> None:
    db.conn.execute(
        "UPDATE ingest_batches SET dismissed_at = datetime('now'), "
        "  updated_at = datetime('now') "
        "WHERE id = ?",
        (batch_id,),
    )
    db.conn.commit()


# ---------------------------------------------------------------------------
# ingest_batch_jobs
# ---------------------------------------------------------------------------

def open_job(db, batch_id: int, step: str, job_kind: str, ttl_seconds: int = 21600) -> None:
    """Record that ``step`` was told to run for this batch. Upserts —
    re-opening an already-open-or-closed step clears ``closed_at`` and
    resets ``opened_at``/``expires_at`` from now, exactly like re-claiming a
    worker batch resets its claim TTL.
    """
    db.conn.execute(
        "INSERT INTO ingest_batch_jobs "
        "  (batch_id, step, job_kind, opened_at, expires_at, closed_at) "
        "VALUES (?, ?, ?, datetime('now'), datetime('now', ?), NULL) "
        "ON CONFLICT(batch_id, step) DO UPDATE SET "
        "  job_kind = excluded.job_kind, "
        "  opened_at = excluded.opened_at, "
        "  expires_at = excluded.expires_at, "
        "  closed_at = NULL",
        (batch_id, step, job_kind, f"+{int(ttl_seconds)} seconds"),
    )
    db.conn.commit()


def delete_job(db, batch_id: int, step: str) -> None:
    """Remove a step's job row entirely — the failure/cancel path.

    Deliberately NOT ``close_job``: a closed row is how a job-only step
    (match_faces, resolve_dups, warm_crops) proves it *succeeded*, so closing
    a failed one would mark it complete. And leaving it open is no better —
    an open row reads `queued`, so a retry skips the step and stops at the
    next thing depending on it, for the full six-hour TTL, with no recovery
    short of editing the table by hand. Deleting puts the step back to
    `needs_queue`, which is the truth: it did not run.
    """
    db.conn.execute(
        "DELETE FROM ingest_batch_jobs WHERE batch_id = ? AND step = ?",
        (batch_id, step),
    )
    db.conn.commit()


def close_job(db, batch_id: int, step: str) -> None:
    db.conn.execute(
        "UPDATE ingest_batch_jobs SET closed_at = datetime('now') "
        "WHERE batch_id = ? AND step = ?",
        (batch_id, step),
    )
    db.conn.commit()


def open_jobs(db, batch_id: int) -> dict[str, dict]:
    """Steps with a live, unclosed job: keyed by step. Excludes both closed
    rows and expired-but-never-closed rows (a crashed job leaves the latter;
    see Task 3 for how that state gets reconciled)."""
    rows = db.conn.execute(
        "SELECT * FROM ingest_batch_jobs "
        "WHERE batch_id = ? AND closed_at IS NULL AND expires_at > datetime('now')",
        (batch_id,),
    ).fetchall()
    return {r["step"]: dict(r) for r in rows}


def closed_jobs(db, batch_id: int) -> set[str]:
    """Steps whose job row has ``closed_at`` set (finished, successfully or
    not — Task 3's state machine distinguishes success from failure)."""
    rows = db.conn.execute(
        "SELECT step FROM ingest_batch_jobs WHERE batch_id = ? AND closed_at IS NOT NULL",
        (batch_id,),
    ).fetchall()
    return {r["step"] for r in rows}

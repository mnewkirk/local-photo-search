"""Backfill ``photos.sharpness*`` from the ORIGINAL pixels (schema v33).

Step 5 of ``docs/plans/sharpness-measurement.md``. ``sharpness.measure_photo``
does the measuring; this module decides WHICH photos, WHEN, and how gently.
It is the only writer of the four ``sharpness*`` columns.

WHERE IT RUNS -- only where the originals are. On the NAS that is every photo.
The desktop replica holds none (its ``photo_root`` is unset, so a stored path
stays relative), so every photo there is counted ``not_local`` and nothing is
written: the replica never measures, and never records a "failure" for a file
it simply does not have.

MISSING-ONLY: a photo is a candidate when ``sharpness_version IS NULL OR
sharpness_version < SHARPNESS_VERSION`` -- so bumping the version re-measures
the library, and nothing else ever does. The predicate is served by
``idx_photos_sharpness_version`` (see db.py v33); the candidate query orders by
``+id`` so the planner uses that index instead of scanning the wide ``photos``
table by rowid.

A photo is also only a candidate once its FACES pass is done (it has a face
row, or its ``worker_processed`` faces attempts are exhausted -- the terminal
"nobody here" state). Face boxes are the headline region; measuring a fresh
shoot before detection has run would stamp a frame-only answer with the
current version, and missing-only would never revisit it.

A DECODE ERROR IS STORED, not skipped: ``sharpness`` NULL, ``sharpness_json``
the ``{"error": ...}`` detail, ``sharpness_version`` set. Otherwise an
unreadable file (a ZIP-wrapped ``.JPG``) would be re-decoded on every run
forever -- the CLIP re-claim trap. A file that is merely ABSENT is different:
nothing is recorded (``not_local``), because on the replica that is normal
and on the NAS it is a moved file that a re-run should pick up.

WRITES: each UPDATE is guarded on the version it was selected at
(``WHERE id = ? AND sharpness_version IS ?``), like the chunked percentile
refresh, so a concurrent writer is never clobbered (counted ``raced``). One
commit every ``COMMIT_EVERY`` photos keeps the write lock short: the fleet's
``submit-results`` must never wait on this.

THROTTLES -- the 2026-09-19 incident (an unrelated SMB bulk read starved the
NAS's disks; ingest crawled at ~4 min/file) is the design constraint:

- ``pause_s`` sleep between photos (default 0.2 s);
- ``os.nice(10)`` and the Linux ``ioprio_set`` IDLE I/O class (via a raw
  syscall -- no psutil in the image; skipped silently where unavailable).
  Both apply to the CALLING THREAD on Linux: the CLI is its own process, and
  inside the web server it is the sweep's / advance's own worker thread, not
  the request threads. Neither can be undone without privilege, which is why
  it is opt-out (``lower_priority=False``) for tests and never applied on a
  dry run;
- refuses to START while an ``ingest-incoming`` sweep holds its lock
  (``ingest.sweep_lock_held`` -- a probe, never a hold) or a batch-advance
  NAS job is open, and re-checks between commit chunks, stopping cleanly
  (``status: "yielded"``) if one begins mid-run;
- a per-run ``limit`` (the maintenance stage caps at ~5000/night).

SCOPE: ``photo_ids=None`` is "no id scope" (the whole library, or ``folder``).
``photo_ids=[]`` is NO PHOTOS -- never the whole library. That distinction is
the one that has twice wiped the library's stacks (``detect_stacks`` reads
``[]`` as "no scope"); here an empty list returns before any query.
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import platform
import time
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from .sharpness import SHARPNESS_VERSION, measure_photo

logger = logging.getLogger("photosearch.sharpness_backfill")

#: Sleep between photos. ~0.5 s of N100 decode + this keeps the disk far from
#: saturated even when something else is reading.
DEFAULT_PAUSE_S = 0.2
#: Photos per commit (and per busy re-check / progress event).
COMMIT_EVERY = 25
#: The maintenance stage's per-run cap: ~5000 photos is ~45-60 min on the
#: N100 at the default pause, i.e. one quiet night.
DEFAULT_STAGE_LIMIT = 5000

_SQLITE_CHUNK = 900  # stays under SQLITE_MAX_VARIABLE_NUMBER on old builds

# Photo is past the faces pass: rows exist, or detection ran to its terminal
# "nothing here" (attempts exhausted). Mirrors the faces claim predicate.
_FACES_DONE_SQL = (
    "(EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id) "
    " OR EXISTS (SELECT 1 FROM worker_processed wp WHERE wp.photo_id = p.id "
    "            AND wp.pass_type = 'faces' AND wp.attempts >= {max_attempts}))"
)


def missing_sql(alias: str = "p") -> str:
    """The missing-only predicate (one ``?``: the current version)."""
    return (f"({alias}.sharpness_version IS NULL "
            f"OR {alias}.sharpness_version < ?)")


class _Busy(Exception):
    pass


# ---------------------------------------------------------------------------
# busy / priority helpers
# ---------------------------------------------------------------------------

def busy_reason(db, *, ignore_batch_steps: tuple = ()) -> Optional[str]:
    """Why a sharpness run must not start now, or None.

    Reuses the existing signals rather than inventing a lock: ingest's own
    ``flock`` (probed, never held) and ``ingest_batch_jobs`` rows for NAS
    steps that are open and unexpired. ``ignore_batch_steps`` lets the batch
    runner skip its OWN open ``sharpness`` row.
    """
    from .ingest import sweep_lock_held
    db_path = getattr(db, "db_path", None)
    if db_path and sweep_lock_held(db_path):
        return "an ingest-incoming sweep is running"
    try:
        sql = ("SELECT batch_id, step FROM ingest_batch_jobs "
               "WHERE job_kind = 'nas' AND closed_at IS NULL "
               "AND expires_at > datetime('now')")
        rows = db.conn.execute(sql).fetchall()
    except Exception:  # table absent on a very old DB -> nothing to yield to
        rows = []
    for r in rows:
        if r[1] in ignore_batch_steps:
            continue
        return f"batch-advance step {r[1]!r} is running for batch {r[0]}"
    return None


def lower_priority() -> dict:
    """``nice(10)`` + IDLE I/O class for the calling thread; best effort.

    Returns what was applied (for the run summary). Linux-only for ionice:
    ``ioprio_set(IOPRIO_WHO_PROCESS, 0, IOPRIO_CLASS_IDLE << 13)`` via a raw
    syscall, since the image has no psutil. BFQ honours it; mq-deadline
    ignores it, which is why the pause and the busy check exist too.
    """
    applied = {"nice": False, "ionice": False}
    try:
        os.nice(10)
        applied["nice"] = True
    except (AttributeError, OSError):
        pass
    try:
        nr = {"x86_64": 251, "aarch64": 30, "i386": 289, "i686": 289,
              "armv7l": 314}.get(platform.machine())
        if nr is not None and platform.system() == "Linux":
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.syscall(nr, 1, 0, 3 << 13) == 0:
                applied["ionice"] = True
    except Exception:
        pass
    return applied


# ---------------------------------------------------------------------------
# candidate selection
# ---------------------------------------------------------------------------

def _chunks(ids):
    for i in range(0, len(ids), _SQLITE_CHUNK):
        yield ids[i:i + _SQLITE_CHUNK]


def _scope_ids(db, photo_ids, folder) -> Optional[list[int]]:
    """None = unscoped (whole library); a list = exactly those ids (may be
    empty). A folder that resolves to nothing is EMPTY, never unscoped."""
    if folder:
        sql, params = db._directory_scope_sql(folder)
        ids = [r[0] for r in db.conn.execute(sql, params).fetchall()]
        if photo_ids is not None:
            keep = set(photo_ids)
            ids = [i for i in ids if i in keep]
        return ids
    if photo_ids is None:
        return None
    return [int(i) for i in photo_ids]


def candidate_ids(db, *, photo_ids=None, folder=None, limit=None,
                  require_faces: bool = True, version: int = SHARPNESS_VERSION
                  ) -> list[int]:
    """Photo ids still needing a measurement, newest first, up to ``limit``."""
    from .db import MAX_PROCESS_ATTEMPTS
    scope = _scope_ids(db, photo_ids, folder)
    if scope is not None and not scope:
        return []
    where = missing_sql("p")
    if require_faces:
        where += " AND " + _FACES_DONE_SQL.format(max_attempts=int(MAX_PROCESS_ATTEMPTS))
    # `+p.id`: keep the planner on idx_photos_sharpness_version (MULTI-INDEX
    # OR + a temp sort of the hits) instead of a rowid scan of the table.
    if scope is None:
        sql = f"SELECT p.id FROM photos p WHERE {where} ORDER BY +p.id DESC"
        params: list = [version]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return [r[0] for r in db.conn.execute(sql, params).fetchall()]
    out: list[int] = []
    for chunk in _chunks(sorted(set(scope), reverse=True)):
        ph = ",".join("?" * len(chunk))
        rows = db.conn.execute(
            f"SELECT p.id FROM photos p WHERE p.id IN ({ph}) AND {where}",
            list(chunk) + [version]).fetchall()
        out.extend(r[0] for r in rows)
    out.sort(reverse=True)
    return out[:int(limit)] if limit is not None else out


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def _clean(o):
    """JSON-safe copy: numpy scalars -> Python, non-finite floats -> None
    (a stored ``NaN`` would break ``JSON.parse`` in the browser)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _score(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _is_local(path) -> bool:
    return bool(path) and os.path.isabs(path) and os.path.isfile(path)


def iter_sharpness_backfill(
    db, *, photo_ids=None, folder: Optional[str] = None,
    limit: Optional[int] = None, apply: bool = False,
    pause_s: float = DEFAULT_PAUSE_S,
    should_abort: Optional[Callable[[], bool]] = None,
    measure: Callable = measure_photo,
    lower_prio: bool = True, check_busy: bool = True,
    ignore_batch_steps: tuple = (), require_faces: bool = True,
    commit_every: int = COMMIT_EVERY, sleep: Callable = time.sleep,
) -> Iterator[dict]:
    """Streaming generator: yields ``progress`` events, then ONE final
    ``{"event": "done", ...summary}``. Raises ``InterruptedError`` on abort
    (after committing what was measured -- the pass is resumable)."""
    counts = {"measured": 0, "errors": 0, "not_local": 0, "raced": 0}
    summary = {"apply": apply, "version": SHARPNESS_VERSION, **counts}

    busy = busy_reason(db, ignore_batch_steps=ignore_batch_steps) if check_busy else None
    todo = candidate_ids(db, photo_ids=photo_ids, folder=folder, limit=limit,
                         require_faces=require_faces)
    summary["would"] = len(todo)
    if busy:
        summary["busy"] = busy
    if not apply:
        summary["status"] = "preview" if todo else "skipped"
        yield {"event": "done", **summary}
        return
    if not todo:
        summary["status"] = "skipped"
        yield {"event": "done", **summary}
        return
    if busy:
        summary["status"] = "refused"
        summary["message"] = f"not started: {busy}"
        yield {"event": "done", **summary}
        return

    if lower_prio:
        summary["priority"] = lower_priority()

    pending = 0

    def _commit():
        nonlocal pending
        if pending:
            db.conn.commit()
            pending = 0

    total = len(todo)
    for n, pid in enumerate(todo, 1):
        if should_abort is not None and should_abort():
            _commit()
            raise InterruptedError("sharpness backfill cancelled")
        row = db.conn.execute(
            "SELECT id, filepath, subject_boxes, sharpness_version FROM photos "
            "WHERE id = ?", (pid,)).fetchone()
        if row is None:          # deleted since selection
            continue
        path = db.resolve_filepath(row["filepath"])
        if not _is_local(path):
            counts["not_local"] += 1
        else:
            faces = [dict(f) for f in db.conn.execute(
                "SELECT bbox_left, bbox_top, bbox_right, bbox_bottom "
                "FROM faces WHERE photo_id = ?", (pid,)).fetchall()]
            subjects = None
            if row["subject_boxes"]:
                try:
                    subjects = json.loads(row["subject_boxes"])
                except (TypeError, ValueError):
                    subjects = None
            try:
                res = measure(path, faces, subjects)
            except Exception as exc:  # measure_photo never raises; a stub might
                res = {"score": None, "version": SHARPNESS_VERSION,
                       "detail": {"error": f"{type(exc).__name__}: {exc}"}}
            detail = _clean(res.get("detail") or {})
            if "error" in detail:
                counts["errors"] += 1
            cur = db.conn.execute(
                "UPDATE photos SET sharpness = ?, sharpness_json = ?, "
                "sharpness_version = ?, sharpness_scored_at = ? "
                "WHERE id = ? AND sharpness_version IS ?",
                (_score(res.get("score")), json.dumps(detail),
                 int(res.get("version") or SHARPNESS_VERSION),
                 datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 pid, row["sharpness_version"]))
            if cur.rowcount:
                counts["measured"] += 1
                pending += 1
            else:
                counts["raced"] += 1
        if n % commit_every == 0 or n == total:
            _commit()
            summary.update(counts)
            yield {"event": "progress", "done": n, "total": total, **counts}
            if check_busy and n < total:
                busy = busy_reason(db, ignore_batch_steps=ignore_batch_steps)
                if busy:
                    summary["status"] = "yielded"
                    summary["message"] = f"stopped after {n}/{total}: {busy}"
                    yield {"event": "done", **summary}
                    return
        if pause_s and n < total:
            sleep(pause_s)

    _commit()
    summary.update(counts)
    summary["status"] = "done"
    yield {"event": "done", **summary}


def run_sharpness_backfill(db, *, on_progress: Optional[Callable[[dict], None]] = None,
                           **kwargs) -> dict:
    """Drain ``iter_sharpness_backfill``; return the final summary.

    Keyword arguments are those of the generator (``photo_ids``, ``folder``,
    ``limit``, ``apply``, ``pause_s``, ``should_abort``, ``measure``, ...).
    """
    final: dict = {}
    for ev in iter_sharpness_backfill(db, **kwargs):
        if ev.get("event") == "done":
            final = {k: v for k, v in ev.items() if k != "event"}
        elif on_progress is not None:
            try:
                on_progress(ev)
            except Exception:  # a progress sink must never kill the job
                logger.debug("on_progress sink raised", exc_info=True)
    return final


# ---------------------------------------------------------------------------
# read-only DB for dry runs
# ---------------------------------------------------------------------------

def open_readonly(db_path: str, photo_root: Optional[str] = None):
    """A ``mode=ro`` DB for a dry run: ``PhotoDB`` migrates on open and would
    write (and would CREATE a stub on a mistyped path). Reuses refile's
    read-only stand-in so path resolution cannot drift from PhotoDB's."""
    from .db import PhotoDB
    from .refile import _ReadOnlyDB

    class _RO(_ReadOnlyDB):
        normalize_directory = PhotoDB.normalize_directory
        _directory_scope_sql = PhotoDB._directory_scope_sql

    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    ro = _RO(db_path, photo_root)
    ro.db_path = db_path
    return ro

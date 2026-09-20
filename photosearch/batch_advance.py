"""`batch-advance` — the one action that takes an ingest batch toward "ready
to review" (M "ingest batch" Task 6).

``batch_state`` says what each step of the pipeline *is*; this module is the
half that makes it *become* something else. It only ever runs the **NAS
steps** — stacking, normalize_aesthetics, match_faces, resolve_dups,
warm_crops, rank_measure — because those are the ones that run where the DB
and the photo files live. (`rank_measure` was once labelled desktop-only and
had no runner at all, so its box read "Needs to be queued" forever; it
decodes the ORIGINALS at native resolution, which only the NAS has.) The
worker passes are launched separately
(``POST /api/admin/batch-launch-fleet``): the NAS has no GPU and cannot host
a fleet, so that half has to run on the desktop.

**Three rules this module exists to enforce.**

1. **Never a truthy `clear` on a scoped stacking run.** `run_stacking` has no
   `clear` parameter — the clear scope is decided by what restricted the run,
   and a run with `photo_ids=<batch>` clears only stacks overlapping that
   batch. Passing `directory=` instead would make it re-resolve its own scope,
   and passing neither clears the *whole library*. Scoped stacking has twice
   wiped every stack in the library; `tests/test_batch_advance.py` pins it.
2. **The STRICT face matcher only.** `match_faces_temporal` is ~4% accurate on
   these shoots (CLAUDE.md, "Bulk-undoing one person's bad labels"), so a
   batch advance must never pour it into a fresh folder. The maintenance
   sweep's `_stage_match_faces` runs both, which is exactly why this module
   calls `faces.match_faces_to_persons` directly instead of reusing it.
3. **Every step written to `ingest_batch_jobs` is validated against
   `STEP_ORDER`.** That column is unvalidated storage: a writer typo would
   never match a derived step name, so the batch would read `needs_queue`
   forever and the typo'd row would sit there until its TTL. This module is
   the writer, so it validates (``open_step_job``).

**Dependency handling — the `satisfied` set.** `batch_state` is derived once,
up front. But `resolve_dups` depends on `match_faces`, whose only completion
evidence is a closed job row, so on a fresh batch it *always* derives as
`waiting` even when `match_faces` is about to run in this very pass. Walking
the plan naively would therefore stop two steps early every single time. So a
step whose `waiting_on` is a step this run has already run (or, in a dry-run,
would run) is promoted to runnable. A step waiting on anything else — a
worker pass, typically `faces` — still stops the run, which is correct: this
module cannot make that happen.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from . import ingest_batches
from .batch_state import NAS_STEPS, STEP_ORDER, batch_state

# job_kind values written to ingest_batch_jobs. "nas" is a stage this module
# runs itself; "fleet" is a worker pass handed to run-workers.sh.
JOB_KIND_NAS = "nas"
JOB_KIND_FLEET = "fleet"

# The photo root to assume when the DB has none recorded. Matches the
# container mount and worker_api's own default.
DEFAULT_PHOTO_ROOT = "/photos"


# ---------------------------------------------------------------------------
# validated job writes
# ---------------------------------------------------------------------------

def open_step_job(db, batch_id: int, step: str, job_kind: str,
                  ttl_seconds: int = 21600) -> None:
    """``ingest_batches.open_job`` with the step name validated.

    ``ingest_batch_jobs.step`` is a bare TEXT column that nothing downstream
    checks, so a typo'd write is invisible: it never matches a name in
    ``STEP_ORDER``, so the real step keeps reading `needs_queue` and the bogus
    row lingers for its whole TTL. Raising here is the only place that can
    catch it, because this module (and the launch-fleet endpoint) are the only
    writers.
    """
    if step not in STEP_ORDER:
        raise ValueError(
            f"unknown step {step!r} — must be one of {', '.join(STEP_ORDER)}")
    ingest_batches.open_job(db, batch_id, step, job_kind, ttl_seconds=ttl_seconds)


def fleet_directory(photo_root: Optional[str], directory: str) -> str:
    """The absolute ``/photos/...`` form of a batch's directory, for `-d`.

    ``-d`` must never be the photo root itself: `get_directory_photo_ids`
    strips the root prefix, leaving an empty prefix that matches no relative
    DB path, so `-d /photos` 404s with "No photos found" (CLAUDE.md,
    "`-d` gotcha"). A batch whose directory is empty would produce exactly
    that, so it raises instead.
    """
    rel = (directory or "").strip().strip("/")
    while rel.startswith("./"):
        rel = rel[2:].strip("/")
    if not rel or rel == ".":
        raise ValueError(
            "batch directory is empty — a fleet scoped to the photo root "
            "claims nothing (get_directory_photo_ids strips the root prefix)")
    root = (photo_root or os.environ.get("PHOTO_ROOT") or DEFAULT_PHOTO_ROOT).rstrip("/")
    return f"{root}/{rel}"


# ---------------------------------------------------------------------------
# default runners
# ---------------------------------------------------------------------------
#
# A runner is ``run(db, ctx) -> dict``. ``ctx`` carries everything a step
# could need without re-querying: ``batch``, ``batch_id``, ``photo_ids`` (the
# batch's live membership), ``day`` (the YYYY-MM-DD its folder is named for,
# or None), ``emit`` and ``check_abort``.

def _inner(event: dict, step: str) -> dict:
    """Re-stamp a sub-job's progress event as this step's.

    `run_stacking` and `warm_crops` emit their own `phase`/`status`
    vocabulary; forwarded raw, a sub-job's `"status": "done"` would read as
    the STEP finishing and a `"status": "skipped"` would claim a step was
    skipped when it actually ran. Everything from inside a step is `running`.
    """
    out = {k: v for k, v in (event or {}).items() if k != "phase"}
    out["step"] = step
    out["status"] = "running"
    return out


def _run_stacking(db, ctx) -> dict:
    """Burst/bracket detection over the batch's photos.

    Scoped by ``photo_ids`` — which is ALSO what decides the clear scope
    inside `save_stacks`. `directory=` is deliberately not passed: it would
    make run_stacking re-resolve the scope itself, and getting that wrong is
    how the whole library's stacks have been wiped before.
    """
    from . import stacking
    if not ctx["photo_ids"]:
        # THE guard. `detect_stacks` treats `photo_ids=[]` as "no scope given"
        # and falls through to the whole library — and `save_stacks` would
        # then clear every stack in it. A batch really can empty out
        # (membership is derived live from photos.folder: a dedup prune, a
        # purge, a clock retime that rewrites the folder), so this is a state
        # that happens, not a defensive nicety.
        return {"stacks": 0, "photos_stacked": 0, "skipped": "empty batch"}
    stacks = stacking.run_stacking(
        db,
        photo_ids=ctx["photo_ids"],
        directory=None,
        dry_run=False,
        on_progress=lambda ev: ctx["emit"](_inner(ev, "stacking")),
        should_abort=lambda: _abort_flag(ctx["check_abort"]),
    )
    return {"stacks": len(stacks),
            "photos_stacked": sum(len(s) for s in stacks)}


def _run_normalize_aesthetics(db, ctx) -> dict:
    """Percentile refresh for the VLM aesthetic scores.

    Library-wide on purpose, and not scopeable: `aes_overall_pct` is a
    library-relative percentile, so recomputing it for one folder in isolation
    would produce numbers that do not compare with any other folder's.
    """
    from . import maintenance
    return maintenance._stage_normalize_aesthetics(
        db, True, ctx["emit"], ctx["check_abort"])


def _run_match_faces(db, ctx) -> dict:
    """STRICT matcher only, scoped to the batch's photos.

    Not `maintenance._stage_match_faces` — that one also runs
    `match_faces_temporal`, which is ~4% accurate on these shoots and would
    make a fresh batch's labels worse than leaving them unmatched.
    """
    from . import faces
    matched = faces.match_faces_to_persons(db, photo_ids=ctx["photo_ids"])
    return {"matched": matched}


def _run_resolve_dups(db, ctx) -> dict:
    """One-person-per-photo. Library-wide (the maintenance stage groups by
    (photo_id, person_id) over the whole faces table), but it only ever
    changes photos that actually carry a duplicate, so running it after a
    batch's match pass is cheap and idempotent."""
    from . import maintenance
    return maintenance._stage_resolve_dups(
        db, True, ctx["emit"], ctx["check_abort"])


def _run_warm_crops(db, ctx) -> dict:
    """Pre-render the batch's face crops so the review grids open warm.

    Scoped by ``photo_ids``, not by the folder's date: a batch is one dated
    FOLDER, and two folders can share a day (a phone sync and a card dump),
    while a photo with no readable `date_taken` would fall out of a date
    scope entirely even though it is in the batch.
    """
    from . import face_crop
    return face_crop.warm_crops(
        db,
        photo_ids=ctx["photo_ids"],
        nas_url=os.environ.get("PHOTOSEARCH_NAS_URL") or None,
        on_progress=lambda ev: ctx["emit"](_inner(ev, "warm_crops")),
    )


def _run_rank_measure(db, ctx) -> dict:
    """Native-resolution face-sharpness measurement for the batch's photos.

    Runs HERE, on the NAS, because it decodes the originals at full
    resolution and the desktop replica holds none of them. ~10 min for 1,260
    photos on the N100, and resumable — a re-run after a late-arriving photo
    only measures what is new.

    Scoped by ``photo_ids`` (one dated FOLDER — two folders can share a day),
    but it writes the cache file keyed by the folder's DATE, because the
    selection phase (`scripts/rank_shoot.py --date D`, no `--measure`) reads
    it by date. That is the whole point of the step: after an advance, the
    owner's next command needs no extra flags.

    A batch whose folder is not dated (``_undated/...``) is SKIPPED with a
    message rather than guessed at: the cache is per-date, so there is no file
    a selection run would ever read.
    """
    from . import rank_measure
    day = ctx["day"]
    if day is None:
        return {"skipped": "undated batch — rank_shoot.py selects by date, so "
                           "there is no dated cache to write"}
    if not ctx["photo_ids"]:
        return {"skipped": "empty batch"}
    cache_path = rank_measure.default_cache_path(db.db_path, day)
    return rank_measure.measure(
        db, day, cache_path,
        photo_ids=ctx["photo_ids"],
        log=lambda msg: ctx["emit"](_inner({"line": msg}, "rank_measure")),
        on_progress=lambda ev: ctx["emit"](_inner(ev, "rank_measure")))


def default_runners() -> dict[str, Callable]:
    """step -> runner. Every NAS step has one; tests substitute fakes."""
    return {
        "stacking": _run_stacking,
        "normalize_aesthetics": _run_normalize_aesthetics,
        "match_faces": _run_match_faces,
        "resolve_dups": _run_resolve_dups,
        "warm_crops": _run_warm_crops,
        "rank_measure": _run_rank_measure,
    }


def _abort_flag(check_abort) -> bool:
    """Adapt a raising check into the boolean `should_abort` shape that
    stacking.py consumes (mirrors maintenance._abort_flag)."""
    try:
        check_abort()
        return False
    except InterruptedError:
        return True


# ---------------------------------------------------------------------------
# the advance itself
# ---------------------------------------------------------------------------

def _day_of(directory: str) -> Optional[str]:
    """`2091/2091-09-19_ILCE-7RM6` -> `2091-09-19`; None for `_undated/...`."""
    import re
    base = (directory or "").rstrip("/").split("/")[-1]
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", base)
    return m.group(1) if m else None


def advance_nas_steps(db, batch_id: int, *, apply: bool = False,
                      on_progress: Optional[Callable[[dict], None]] = None,
                      should_abort: Optional[Callable[[], bool]] = None,
                      runners: Optional[dict[str, Callable]] = None) -> dict:
    """Run (or preview) every NAS step this batch still needs, in order.

    Walks ``NAS_STEPS``: `completed` steps are skipped, `queued`/`running`
    ones are left to whoever owns them, and the run **stops** at the first
    `waiting` or `blocked` step — nothing after it can be trusted to be
    meaningful. See the module docstring for why a step waiting on another
    step of this same run is promoted instead of stopping it.

    ``apply=False`` (the default) is a true dry run: no runner is called and
    **no job row is written**. A job row that leaked out of a preview would
    read as `queued` for its whole TTL and keep the real run from starting.

    A step that fails or is cancelled has its job row **deleted**, not closed
    and not left open, so the step reads `needs_queue` again and the next
    ``batch-advance`` retries it. The error is still reported in the result
    and on the progress stream.

    ``should_abort`` is checked before each step; returning True raises
    ``InterruptedError``.
    """
    runners = dict(default_runners() if runners is None else runners)
    unknown = [s for s in runners if s not in STEP_ORDER]
    if unknown:
        raise ValueError(f"unknown runner step(s): {', '.join(sorted(unknown))}")

    state = batch_state(db, batch_id)     # raises ValueError on a bad id
    batch = state["batch"]
    rows = {s["step"]: s for s in state["steps"]}
    photo_ids = ingest_batches.batch_photo_ids(db, batch)

    def emit(event: dict) -> None:
        if on_progress:
            try:
                # `phase` last: a forwarded sub-job event carries its own and
                # would otherwise shadow ours.
                on_progress({**event, "phase": "batch-advance"})
            except Exception:  # a progress sink must never kill the job
                pass

    def check_abort() -> None:
        if should_abort and should_abort():
            raise InterruptedError("batch advance cancelled")

    ctx = {"batch": batch, "batch_id": batch_id, "photo_ids": photo_ids,
           "day": _day_of(batch["directory"]), "emit": emit,
           "check_abort": check_abort, "apply": apply}

    # Steps whose dependency is satisfied as far as THIS run is concerned:
    # already completed, or run/planned here. See the module docstring.
    satisfied = {name for name, row in rows.items() if row["state"] == "completed"}

    result = {"batch_id": batch_id, "directory": batch["directory"],
              "apply": apply, "steps": [], "ran": [],
              "stopped_at": None, "stopped_reason": None, "error": None}

    for step in NAS_STEPS:
        row = rows[step]
        state_name = row["state"]
        waiting_on = row.get("waiting_on")

        if state_name == "completed":
            result["steps"].append({"step": step, "status": "skipped",
                                    "state": state_name, "reason": "already complete",
                                    "waiting_on": None, "result": None, "error": None})
            continue

        if state_name == "waiting" and waiting_on in satisfied:
            state_name = "needs_queue"

        if state_name in ("waiting", "blocked"):
            reason = (f"waiting on {waiting_on}" if state_name == "waiting" and waiting_on
                      else state_name)
            result["steps"].append({"step": step, "status": "stopped",
                                    "state": row["state"], "reason": reason,
                                    "waiting_on": waiting_on, "result": None,
                                    "error": None})
            result["stopped_at"] = step
            result["stopped_reason"] = reason
            emit({"step": step, "status": "stopped", "reason": reason})
            break

        if state_name != "needs_queue":
            # queued / running — someone else owns it. Anything that depends
            # on it derives as `waiting`, so the run stops there next.
            result["steps"].append({"step": step, "status": "skipped",
                                    "state": state_name, "reason": state_name,
                                    "waiting_on": None, "result": None, "error": None})
            continue

        check_abort()

        if not apply:
            result["steps"].append({"step": step, "status": "would_run",
                                    "state": row["state"], "reason": None,
                                    "waiting_on": waiting_on, "result": None,
                                    "error": None})
            emit({"step": step, "status": "would_run"})
            satisfied.add(step)
            continue

        emit({"step": step, "status": "running"})
        open_step_job(db, batch_id, step, JOB_KIND_NAS)
        try:
            out = runners[step](db, ctx)
        except InterruptedError:
            # Cancelled mid-step: the step did not finish, so its job row is
            # deleted. Leaving it open would read as `queued` and make the
            # step unretryable for its whole TTL (see delete_job).
            ingest_batches.delete_job(db, batch_id, step)
            emit({"step": step, "status": "cancelled"})
            raise
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            # Same rule as cancel — and NOT close_job, which is how a
            # job-only step proves it succeeded. The error is still reported
            # in the result and on the SSE stream; only the row goes.
            ingest_batches.delete_job(db, batch_id, step)
            result["steps"].append({"step": step, "status": "failed",
                                    "state": row["state"], "reason": None,
                                    "waiting_on": waiting_on, "result": None,
                                    "error": str(exc)})
            result["stopped_at"] = step
            result["stopped_reason"] = f"{step} failed"
            result["error"] = str(exc)
            emit({"step": step, "status": "failed", "error": str(exc)})
            break

        ingest_batches.close_job(db, batch_id, step)
        result["steps"].append({"step": step, "status": "ran",
                                "state": row["state"], "reason": None,
                                "waiting_on": waiting_on,
                                "result": out if isinstance(out, dict) else None,
                                "error": None})
        result["ran"].append(step)
        satisfied.add(step)
        # The runner's own dict is splatted FIRST: a stage that returns
        # `{"status": "skipped"}` (the maintenance stages do, when they find
        # nothing to change) must not overwrite the step's own outcome — the
        # step ran and its job is closed.
        emit({**(out if isinstance(out, dict) else {}),
              "step": step, "status": "done"})

    return result



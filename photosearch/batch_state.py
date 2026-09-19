"""Derived readiness state for one ingest batch (M "ingest batch" Task 3).

Given a batch — one dated folder, membership derived live from
``photos.folder`` — this reports exactly one of six states for every step of
the pipeline. Nothing here is persisted: the whole module is a read-only
projection over ``photos``, ``worker_claims``, ``worker_processed`` and
``ingest_batch_jobs``, recomputed on every call.

**The trap this module exists for.**
``db.count_unprocessed_photos(pass, photo_ids=...)`` is the claim predicate,
not a progress bar. It answers "what would a worker pick up next?", and it
returns **0** in three quite different situations:

1. the pass is genuinely finished;
2. every remaining photo has ``worker_processed.attempts >=
   MAX_PROCESS_ATTEMPTS``, so the claim path skips it forever — *blocked*;
3. the pass is gated on an input that doesn't exist yet — ``category-content``,
   ``keywords`` and ``verify`` all require ``description IS NOT NULL``, so
   before ``describe`` runs they count 0 photos — *waiting*.

Reading `remaining == 0` as "completed" therefore reports a batch ready when
it has not started. So each pass carries four numbers instead of one:

  ``total``      every photo in the batch
  ``eligible``   photos this pass can act on at all (== total, except the
                 three description-gated passes)
  ``remaining``  ``worker_api._count_scoped`` — what a worker would claim
  ``failed``     eligible photos whose attempts are exhausted *and* whose
                 output column is still missing
  ``done``       ``eligible - remaining - failed``, floored at 0

and `completed` is ``done == total``, never ``remaining == 0``.

The per-pass "output missing" predicates in ``_OUTPUT_MISSING`` below are
transcribed from ``db.count_unprocessed_photos`` (photosearch/db.py ~2199)
so that ``done``/``failed``/``remaining`` stay mutually consistent — each
pass has its own column test and ``verify``'s is different again. Change one
and you must change the other.
"""

from __future__ import annotations

import json

from . import ingest_batches
from .db import MAX_PROCESS_ATTEMPTS

WORKER_PASSES = ("clip", "faces", "quality", "aesthetics", "describe",
                 "category-visual", "category-content", "keywords", "verify")
NAS_STEPS = ("stacking", "normalize_aesthetics", "match_faces",
             "resolve_dups", "warm_crops")
DESKTOP_STEPS = ("rank_measure",)
STEP_ORDER = ("ingest",) + WORKER_PASSES + NAS_STEPS + DESKTOP_STEPS
DEPENDS_ON = {"category-content": "describe", "keywords": "describe",
              "verify": "describe", "normalize_aesthetics": "aesthetics",
              "match_faces": "faces", "resolve_dups": "match_faces",
              "warm_crops": "faces", "rank_measure": "faces"}
STATES = ("completed", "running", "queued", "needs_queue", "waiting", "blocked")

# Steps whose only completion evidence is a closed ingest_batch_jobs row —
# they write no per-photo column this module could count.
_JOB_ONLY_STEPS = ("match_faces", "resolve_dups", "warm_crops", "rank_measure")

# Passes gated on an existing description. `eligible` is the count of batch
# photos that have one; everything else is `total`.
_DESCRIPTION_GATED = ("category-content", "keywords", "verify")

# SQLite's variable cap. Mirrors worker_api._SCOPE_CHUNK — a batch is
# typically 1-3k photos but a big card dump can be far larger.
_ID_CHUNK = 20000

# Per-pass "the output is still missing" column test, transcribed from
# db.count_unprocessed_photos. `p` is the photos alias. `None` means the pass
# keeps no attempts ledger, so it can never contribute a `failed` count.
#
#   clip             photos.id NOT IN clip_embeddings      (no attempts filter)
#   faces            no faces row for the photo
#   quality          aesthetic_score OR aesthetic_concepts NULL (no attempts filter)
#   aesthetics       aes_overall IS NULL
#   describe         description IS NULL
#   category-visual  visual_tags IS NULL
#   category-content categories IS NULL  AND description IS NOT NULL
#   keywords         keywords IS NULL    AND description IS NOT NULL
#   verify           verified_at IS NULL AND description IS NOT NULL (no attempts filter)
#
# The description clause is kept on the three gated passes so that
# `failed` is always a subset of `eligible`: a photo with no description has
# not failed that pass, it has never been offered it.
_OUTPUT_MISSING = {
    "clip": None,
    "faces": "NOT EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id)",
    "quality": "(p.aesthetic_score IS NULL OR p.aesthetic_concepts IS NULL)",
    "aesthetics": "p.aes_overall IS NULL",
    "describe": "p.description IS NULL",
    "category-visual": "p.visual_tags IS NULL",
    "category-content": "p.categories IS NULL AND p.description IS NOT NULL",
    "keywords": "p.keywords IS NULL AND p.description IS NOT NULL",
    "verify": "p.verified_at IS NULL AND p.description IS NOT NULL",
}


# ---------------------------------------------------------------------------
# id-list helpers (chunked — never one query per photo)
# ---------------------------------------------------------------------------

def _chunks(ids: list[int]):
    for i in range(0, len(ids), _ID_CHUNK):
        yield ids[i:i + _ID_CHUNK]


def _count_where(db, ids: list[int], where: str, params: tuple = ()) -> int:
    """COUNT of batch photos matching `where` (a fragment over alias `p`)."""
    if not ids:
        return 0
    total = 0
    for chunk in _chunks(ids):
        placeholders = ",".join("?" * len(chunk))
        row = db.conn.execute(
            f"SELECT COUNT(*) FROM photos p "
            f"WHERE p.id IN ({placeholders}) AND ({where})",
            list(chunk) + list(params),
        ).fetchone()
        total += row[0]
    return total


def _running_passes(db, id_set: set[int]) -> set[str]:
    """Pass types with a live claim covering at least one batch photo.

    One query for the whole table — worker_claims holds one row per in-flight
    worker batch, so it is tiny (tens of rows), and the JSON photo_ids column
    can't be joined against in SQL anyway.
    """
    rows = db.conn.execute(
        "SELECT pass_type, photo_ids FROM worker_claims "
        "WHERE expires_at > datetime('now')"
    ).fetchall()
    running = set()
    for row in rows:
        if row["pass_type"] in running:
            continue
        try:
            claimed = json.loads(row["photo_ids"])
        except (TypeError, ValueError):
            continue
        if id_set.intersection(claimed):
            running.add(row["pass_type"])
    return running


def _step_row(step: str, kind: str, state: str, *, total: int, eligible: int,
              done: int, remaining: int, failed: int = 0,
              waiting_on: str | None = None, detail: str | None = None) -> dict:
    return {"step": step, "kind": kind, "state": state, "total": total,
            "eligible": eligible, "done": done, "remaining": remaining,
            "failed": failed, "waiting_on": waiting_on, "detail": detail}


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

def _ingest_step(batch: dict, sweep: dict | None, total: int) -> dict:
    """The batch exists, so the move+index sweep either finished, is still
    running, or failed. There is nothing per-photo to count here — the state
    carries the meaning and the numbers just mirror it."""
    state, detail = "completed", None
    if sweep is not None:
        if sweep["status"] in ("moving", "indexing"):
            state = "running"
            detail = "stalled" if sweep.get("stalled") else None
        elif sweep["status"] == "failed":
            state = "blocked"
            detail = sweep.get("error")
    done = total if state == "completed" else 0
    return _step_row("ingest", "ingest", state, total=total, eligible=total,
                     done=done, remaining=total - done, detail=detail)


# ---------------------------------------------------------------------------
# worker passes
# ---------------------------------------------------------------------------

def _worker_step(db, pass_type: str, ids: list[int], total: int,
                 described: int, running: set[str], open_steps: set[str],
                 completed: set[str]) -> dict:
    from . import worker_api  # deferred: pulls in FastAPI

    remaining = worker_api._count_scoped(db, pass_type, ids)

    missing = _OUTPUT_MISSING[pass_type]
    if missing is None:
        # clip keeps no attempts ledger — its claim predicate is purely
        # "no embedding row", so a photo can never be permanently skipped
        # (which is the non-image re-claim loop documented in CLAUDE.md).
        failed = 0
    else:
        failed = _count_where(
            db, ids,
            f"({missing}) AND EXISTS (SELECT 1 FROM worker_processed wp "
            f"  WHERE wp.photo_id = p.id AND wp.pass_type = ? "
            f"    AND wp.attempts >= {MAX_PROCESS_ATTEMPTS})",
            (pass_type,),
        )

    eligible = described if pass_type in _DESCRIPTION_GATED else total
    done = max(0, eligible - remaining - failed)

    depends_on = DEPENDS_ON.get(pass_type)
    if pass_type in running:
        state, waiting_on = "running", None
    elif pass_type in open_steps:
        state, waiting_on = "queued", None
    elif total > 0 and done == total:
        state, waiting_on = "completed", None
    elif eligible < total and depends_on is not None and depends_on not in completed:
        # The count-zero trap: this pass hasn't become eligible yet.
        state, waiting_on = "waiting", depends_on
    elif remaining == 0 and failed > 0:
        # The other count-zero trap: every leftover photo is out of attempts.
        state, waiting_on = "blocked", None
    else:
        state, waiting_on = "needs_queue", None

    return _step_row(pass_type, "worker", state, total=total, eligible=eligible,
                     done=done, remaining=remaining, failed=failed,
                     waiting_on=waiting_on)


# ---------------------------------------------------------------------------
# NAS / desktop steps
# ---------------------------------------------------------------------------

def _stacking_step(db, ids: list[int], total: int, open_steps: set[str],
                   closed: set[str]) -> dict:
    """Every dated photo should land in a stack — but a shoot with no bursts
    legitimately produces none, so a *closed* job row also counts as done."""
    eligible = _count_where(db, ids, "p.date_taken IS NOT NULL")
    remaining = _count_where(
        db, ids,
        "p.date_taken IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM stack_members sm WHERE sm.photo_id = p.id)")
    done = max(0, eligible - remaining)

    if remaining == 0 or "stacking" in closed:
        state = "completed"
    elif "stacking" in open_steps:
        state = "queued"
    else:
        state = "needs_queue"
    return _step_row("stacking", "nas", state, total=total, eligible=eligible,
                     done=done, remaining=remaining)


def _normalize_aesthetics_step(db, ids: list[int], total: int,
                               open_steps: set[str], completed: set[str]) -> dict:
    """Percentile refresh for photos the VLM has scored.

    `waiting` is checked BEFORE `completed` here: with no `aes_overall`
    anywhere, there is nothing to normalize and `remaining` is 0 — which
    would read as completed before the aesthetics pass has run at all.
    """
    eligible = _count_where(db, ids, "p.aes_overall IS NOT NULL")
    remaining = _count_where(
        db, ids, "p.aes_overall IS NOT NULL AND p.aes_overall_pct IS NULL")
    done = max(0, eligible - remaining)

    depends_on = DEPENDS_ON["normalize_aesthetics"]
    waiting_on = None
    if depends_on not in completed:
        state, waiting_on = "waiting", depends_on
    elif remaining == 0:
        state = "completed"
    elif "normalize_aesthetics" in open_steps:
        state = "queued"
    else:
        state = "needs_queue"
    return _step_row("normalize_aesthetics", "nas", state, total=total,
                     eligible=eligible, done=done, remaining=remaining,
                     waiting_on=waiting_on)


def _job_only_step(step: str, kind: str, total: int, open_steps: set[str],
                   closed: set[str], completed: set[str]) -> dict:
    """match_faces / resolve_dups / warm_crops / rank_measure — these write no
    per-photo column this module can count, so a closed job row is the only
    completion evidence there is."""
    depends_on = DEPENDS_ON.get(step)
    waiting_on = None
    if step in closed:
        state = "completed"
    elif depends_on is not None and depends_on not in completed:
        state, waiting_on = "waiting", depends_on
    elif step in open_steps:
        state = "queued"
    else:
        state = "needs_queue"
    done = total if state == "completed" else 0
    return _step_row(step, kind, state, total=total, eligible=total, done=done,
                     remaining=total - done, waiting_on=waiting_on)


# ---------------------------------------------------------------------------
# next_action
# ---------------------------------------------------------------------------

def _next_action(steps: dict[str, dict], ready: bool) -> str | None:
    if ready:
        return None
    if steps["ingest"]["state"] == "running":
        return "wait_ingest"
    if any(steps[p]["state"] == "needs_queue" for p in WORKER_PASSES):
        return "launch_fleet"
    if any(steps[s]["state"] == "needs_queue" for s in NAS_STEPS):
        return "advance_nas"
    if any(s["state"] == "blocked" for s in steps.values()):
        return "review_blocked"
    return "wait"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def batch_state(db, batch_id: int) -> dict:
    """Full derived state for one batch, one dict per step in STEP_ORDER.

    Costs a fixed handful of indexed COUNT queries (two or three per worker
    pass, one claims scan, a few for the NAS steps) — no per-photo queries and
    no filesystem access, so it is safe on a polled status endpoint.
    """
    batch = ingest_batches.get_batch(db, batch_id)
    if batch is None:
        raise ValueError(f"no such batch: {batch_id!r}")

    ids = ingest_batches.batch_photo_ids(db, batch)
    total = len(ids)
    id_set = set(ids)

    sweep = ingest_batches.get_sweep(db, batch["run_id"]) if batch["run_id"] else None
    running = _running_passes(db, id_set)
    open_steps = set(ingest_batches.open_jobs(db, batch_id))
    closed = ingest_batches.closed_jobs(db, batch_id)
    described = _count_where(
        db, ids, "p.description IS NOT NULL AND p.description != ''")

    # STEP_ORDER is a valid topological order for DEPENDS_ON (every
    # dependency appears before its dependent), so one forward pass suffices
    # and `completed` is always populated before it is consulted.
    steps: dict[str, dict] = {}
    completed: set[str] = set()
    for step in STEP_ORDER:
        if step == "ingest":
            row = _ingest_step(batch, sweep, total)
        elif step in WORKER_PASSES:
            row = _worker_step(db, step, ids, total, described, running,
                               open_steps, completed)
        elif step == "stacking":
            row = _stacking_step(db, ids, total, open_steps, closed)
        elif step == "normalize_aesthetics":
            row = _normalize_aesthetics_step(db, ids, total, open_steps, completed)
        else:
            kind = "nas" if step in NAS_STEPS else "desktop"
            row = _job_only_step(step, kind, total, open_steps, closed, completed)
        steps[step] = row
        if row["state"] == "completed":
            completed.add(step)

    # rank_measure is optional — a desktop nicety, not a readiness gate.
    ready = all(steps[s]["state"] == "completed"
                for s in WORKER_PASSES + NAS_STEPS)

    return {
        "batch": batch,
        "sweep": sweep,
        "ready": ready,
        "next_action": _next_action(steps, ready),
        "steps": [steps[s] for s in STEP_ORDER],
    }

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
  ``done``       ``eligible - remaining - failed``, floored at 0 — but only
                 for the seven passes whose ``remaining`` excludes exhausted
                 photos. For ``quality`` / ``verify`` / ``clip`` it doesn't,
                 so ``failed`` is already inside ``remaining`` and ``done``
                 is ``eligible - remaining``. See
                 ``_REMAINING_FILTERS_ATTEMPTS``.

and `completed` is ``done == total``, never ``remaining == 0``.

**Worker-pass state precedence: running > completed > blocked > queued >
waiting > needs_queue.** `completed` deliberately outranks an open job row,
because a worker pass's job row is opened by a fleet launch and nothing ever
closes one — completion is provable from the data instead. With `queued`
first, a pass the fleet had finished kept reading `queued` for the row's full
six-hour TTL, so the batch could never read `ready` and the launch button's
"already running" refusal stayed armed after the fleet had exited. `blocked`
outranks `queued` for the mirror-image reason: an open row must not hide a
pass whose every remaining photo has exhausted its attempts. Job-only steps
(``_job_only_step``) and the two computed NAS steps keep their own ordering —
for them a *closed* row is the only evidence of success, so an open one still
means queued.

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
# `rank_measure` is a NAS step, not a desktop one. It decodes every photo at
# full native resolution to measure face sharpness, and only the NAS holds the
# originals — the desktop replica has the DB and the thumbnails and no files at
# all. The original plan labelled it "desktop-only" on the theory that it is
# heavy; heavy it is (~10 min for 1,260 photos on the N100), but heavy where
# the pixels are. As a desktop step it had no runner anywhere and read
# "Needs to be queued" forever.
NAS_STEPS = ("stacking", "normalize_aesthetics", "match_faces",
             "resolve_dups", "warm_crops", "rank_measure")
# NAS steps that do NOT gate `ready` — optional work the owner can run from
# the same button, but a batch is reviewable without it. Subtracted from
# `ready` explicitly: the rule is "every step in WORKER_PASSES + NAS_STEPS",
# so moving a step into NAS_STEPS would otherwise silently make it a gate.
OPTIONAL_STEPS = ("rank_measure",)
STEP_ORDER = ("ingest",) + WORKER_PASSES + NAS_STEPS
DEPENDS_ON = {"category-content": "describe", "keywords": "describe",
              "verify": "describe", "normalize_aesthetics": "aesthetics",
              "match_faces": "faces", "resolve_dups": "match_faces",
              "warm_crops": "faces", "rank_measure": "faces"}
STATES = ("completed", "running", "queued", "needs_queue", "waiting", "blocked")

# Steps whose only completion evidence is a closed ingest_batch_jobs row —
# they write no per-photo column this module could count. Public because
# `ingest_batches.register_batch` has to drop exactly these rows when late
# photos re-open a batch: their closed row would otherwise keep reading
# `completed` for photos that have never been touched.
JOB_ONLY_STEPS = ("match_faces", "resolve_dups", "warm_crops", "rank_measure")

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
# The description clause is kept on the three gated passes because a photo
# with no description has not *failed* that pass, it has never been offered
# it. (That makes `failed` a subset of `eligible` in every case but one: a
# photo whose description is the empty string is excluded from `eligible` by
# the `!= ''` rule below while db.py's `IS NOT NULL` still counts it here.
# `done` floors at 0, so the only effect is a conservative under-count.)
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

# Passes for which "attempts exhausted and still no output" means the work is
# DONE, not failed — because an empty result is a legitimate one.
#
# `faces` is the only one. A photo with nobody facing the camera has no `faces`
# rows after a perfectly successful run; the claim path cannot tell that from a
# failure, so the fleet re-tries it MAX_PROCESS_ATTEMPTS times and stops. On the
# first real batch (2026-09-19, 1,373 photos) 113 photos sat at attempts=3 with
# no face rows — every one a player facing away or a distant shot — while 3,696
# faces were found in the other 1,260. Counting those as `failed` read the pass
# as `blocked` and held match_faces / warm_crops / rank_measure in `waiting`
# behind a pass that had finished. A genuinely corrupt file ends in the same
# place and is indistinguishable here; it is rare, and the step's `detail`
# reports the count so it is not hidden.
_EMPTY_OUTPUT_IS_DONE = {"faces": "no detectable face"}

# Does this pass's *claim* predicate (what `remaining` counts) exclude photos
# whose attempts are exhausted? Seven do; `quality` (db.py:2235-2247),
# `verify` (db.py:2304-2319) and `clip` (db.py:2202-2214) carry no attempts
# filter at all. That single fact decides two things:
#
#   True  -> `failed` and `remaining` are DISJOINT sets.
#            done    = eligible - remaining - failed
#            blocked = remaining == 0 and failed > 0
#   False -> `failed` is a SUBSET of `remaining` (an exhausted photo is still
#            counted as claimable). Subtracting it as well would under-count
#            `done` by exactly `failed`.
#            done    = eligible - remaining
#            blocked = remaining == failed and failed > 0   (every photo the
#                      fleet would still claim is one it can never finish)
#
# `clip` is False but keeps no ledger, so `failed` is 0 and both rows agree.
_REMAINING_FILTERS_ATTEMPTS = {
    "clip": False,
    "faces": True,
    "quality": False,
    "aesthetics": True,
    "describe": True,
    "category-visual": True,
    "category-content": True,
    "keywords": True,
    "verify": False,
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

    # The `if ids else 0` guard is load-bearing, not defensive tidiness:
    # every branch of db.count_unprocessed_photos tests `if photo_ids:`, so an
    # EMPTY list falls through to the whole-library query and the batch would
    # report the entire backlog as its own. Membership is derived live from
    # photos.folder, so a batch really can empty out later — a dedup prune,
    # purge-nonimage-photos, or a clock retime that rewrites `folder`.
    remaining = worker_api._count_scoped(db, pass_type, ids) if ids else 0

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

    # See _EMPTY_OUTPUT_IS_DONE: for `faces`, exhausted-with-no-rows is a
    # finished photo that had nothing to find, so it counts toward `done`.
    detail = None
    if pass_type in _EMPTY_OUTPUT_IS_DONE and failed:
        detail = f"{failed:,} with {_EMPTY_OUTPUT_IS_DONE[pass_type]}"
        failed = 0

    eligible = described if pass_type in _DESCRIPTION_GATED else total

    # See _REMAINING_FILTERS_ATTEMPTS: for the passes whose claim predicate
    # has no attempts filter, `failed` is already counted inside `remaining`,
    # so subtracting it again would under-count `done` by exactly `failed`.
    disjoint = _REMAINING_FILTERS_ATTEMPTS[pass_type]
    done = max(0, eligible - remaining - failed if disjoint else eligible - remaining)
    stuck = (remaining == 0) if disjoint else (remaining == failed)

    # Precedence: running > completed > blocked > queued > waiting > needs_queue.
    #
    # **`completed` outranks `queued`, and that is the whole point.** A worker
    # pass's job row is opened by a fleet launch and nothing ever closes it —
    # nothing should have to, because completion is provable from the data
    # (`done == total`). When `queued` won, a pass the fleet had finished kept
    # reading `queued` for the row's full six-hour TTL: the batch could never
    # read `ready`, and `batch-launch-fleet`'s "a fleet is already running"
    # refusal stayed armed long after the fleet had exited.
    #
    # `blocked` outranks `queued` for the same reason in the other direction:
    # an open row must not hide a pass whose every remaining photo has
    # exhausted its attempts. There is nothing queued about work that can
    # never be claimed again.
    #
    # NOTE this also puts `blocked` above `waiting`, so a gated pass that has
    # both un-described photos and exhausted described ones reports the
    # exhaustion. That is deliberate: `waiting` reads as "nothing to do yet",
    # which would bury a real failure behind a dependency that may itself
    # never finish.
    depends_on = DEPENDS_ON.get(pass_type)
    if pass_type in running:
        state, waiting_on = "running", None
    elif total > 0 and done == total:
        state, waiting_on = "completed", None
    elif stuck and failed > 0:
        # Every photo the fleet would still claim for this pass is one it has
        # already given up on.
        state, waiting_on = "blocked", None
    elif pass_type in open_steps:
        state, waiting_on = "queued", None
    elif eligible < total and depends_on is not None and depends_on not in completed:
        # The count-zero trap: this pass hasn't become eligible yet.
        state, waiting_on = "waiting", depends_on
    else:
        state, waiting_on = "needs_queue", None

    return _step_row(pass_type, "worker", state, total=total, eligible=eligible,
                     done=done, remaining=remaining, failed=failed,
                     waiting_on=waiting_on, detail=detail)


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

def _optional_runnable(steps: dict[str, dict]) -> bool:
    """Is an OPTIONAL step sitting there waiting to be run?

    This is what keeps `rank_measure` from being a dead end. It does not gate
    `ready`, so a naive `if ready: return None` left the box reading "Needs to
    be queued" with no action that would ever run it — which is exactly the
    state the live batch was stuck in.
    """
    return any(steps[s]["state"] == "needs_queue"
               for s in OPTIONAL_STEPS if s in steps)


def _next_action(steps: dict[str, dict], ready: bool, total: int) -> str | None:
    if ready and not _optional_runnable(steps):
        return None
    if total == 0:
        # A batch whose photos are gone (deleted, re-foldered, pruned). The
        # frozen `completed` rule is `total > 0 and done == total`, so an
        # empty batch is deliberately never completed and never `ready` — but
        # there is no action either: scoping a fleet run or a NAS stage to an
        # empty directory does nothing (the worker API 404s on it). The steps
        # still read `needs_queue`; this is what stops the UI acting on that.
        return None
    if steps["ingest"]["state"] == "running":
        return "wait_ingest"
    if any(steps[p]["state"] == "needs_queue" for p in WORKER_PASSES):
        return "launch_fleet"
    if any(steps[s]["state"] == "needs_queue"
           for s in NAS_STEPS if s not in OPTIONAL_STEPS):
        return "advance_nas"
    if ready:
        # Every required step is done and an OPTIONAL one is still runnable —
        # so the button offers it while the page already says "Ready to
        # review". The optional step is deliberately offered ONLY from here:
        # a batch that is blocked, or still has a pass in flight, has a more
        # important thing to say than "you could also measure sharpness".
        return "advance_nas"
    if any(s["state"] == "blocked" for s in steps.values()):
        return "review_blocked"
    return "wait"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def fleet_launch_passes(state: dict) -> list[str]:
    """The worker passes one fleet launch should cover, in WORKER_PASSES order.

    **Not just the `needs_queue` ones.** The fleet is launched with
    `sequential=True` and WORKER_PASSES is already a valid dependency order,
    so one launch can drain the whole pipeline — and it has to, because the
    three description-gated passes are `waiting` at click time and a second
    launch mid-run is refused (it would kill the running fleet). Leaving them
    out meant `category-content` / `keywords` / `verify` were never queued by
    the button at all.

    So the set is every worker pass that is `needs_queue`, plus every
    `waiting` pass whose dependency is either already underway
    (`completed`/`running`/`queued`) or is itself in this set. One forward
    walk suffices because WORKER_PASSES is a topological order.

    A `blocked` dependency does NOT admit its dependents: `describe` giving up
    on every photo means there will be no descriptions for the text passes to
    read, so queueing them would claim nothing.

    Mirrored in JS as `PS.BatchFlow.fleetLaunchPasses` (frontend/dist/
    batch-flow.js) so the button's "N passes" counts the same set this
    launches. The two are pinned to the same cases — see the note there.
    """
    rows = {s["step"]: s for s in state.get("steps", []) if s.get("step")}
    # States that mean "the dependency is or will be satisfied without another
    # launch". `blocked` and `waiting` are deliberately absent.
    underway = ("completed", "running", "queued")

    chosen: list[str] = []
    chosen_set: set[str] = set()
    for pass_type in WORKER_PASSES:
        row = rows.get(pass_type)
        if row is None:
            continue
        pass_state = row.get("state")
        if pass_state == "needs_queue":
            chosen.append(pass_type)
            chosen_set.add(pass_type)
        elif pass_state == "waiting":
            depends_on = row.get("waiting_on") or DEPENDS_ON.get(pass_type)
            if depends_on is None:
                continue
            if depends_on in chosen_set or rows.get(depends_on, {}).get("state") in underway:
                chosen.append(pass_type)
                chosen_set.add(pass_type)
    return chosen


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
            row = _job_only_step(step, "nas", total, open_steps, closed, completed)
        steps[step] = row
        if row["state"] == "completed":
            completed.add(step)

    # OPTIONAL_STEPS (rank_measure) are deliberately NOT part of `ready`: the
    # batch is reviewable without the sharpness measurement. `next_action`
    # still offers to run it — see `_optional_runnable`.
    ready = all(steps[s]["state"] == "completed"
                for s in WORKER_PASSES + NAS_STEPS if s not in OPTIONAL_STEPS)

    return {
        "batch": batch,
        "sweep": sweep,
        "ready": ready,
        "next_action": _next_action(steps, ready, total),
        "steps": [steps[s] for s in STEP_ORDER],
    }

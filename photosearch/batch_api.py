"""Cheap, cached status API for ingest batches (M "ingest batch" Task 4).

Backs the ``/batches`` page's poll. Two GETs do the work:

  ``GET /api/batches``       — pure SQL, no derivation: the active sweep (if
                                any) plus the batch list.
  ``GET /api/batches/{id}``  — the derived readiness state from
                                ``batch_state.batch_state``, memoized.

**The cache is the point of this module.** On 2026-09-19 stacked status
polls filled all 40 of the app's sync request threads and wedged the server
for 30+ minutes. ``batch_state`` costs a fixed handful of indexed COUNT
queries per call — cheap once, but not when every open browser tab polls it
every few seconds and each poll blocks on the same in-flight derivation.

So a ``GET`` never waits on another request's derivation:

- Within ``_TTL_SECONDS`` of the last computation for that batch, return the
  memoized value as-is (``stale: False``).
- Past the TTL, try to acquire the single module-level lock **without
  blocking**. On success: recompute, refresh the memo, release in
  ``finally``. On failure — someone else is already recomputing (for *any*
  batch; this is one lock for the whole module, not one per batch, on
  purpose: it mirrors the incident's "all 40 threads" shape, where the fix
  is that no request ever waits, not that different batches get their own
  queue) — return the existing memo immediately with ``stale: True``, or,
  if there is no memo yet for this batch, a minimal placeholder
  (``computing: True``).

Every write endpoint below (dismiss / ready / register) invalidates that
batch's memo entry, so the next poll recomputes instead of serving a stale
``ready``/``next_action``.

**Two correctness details, both added in review (fix round 1):**

- **The memo is keyed on ``(db_path, batch_id)``, not ``batch_id`` alone.**
  ``batch_id`` is only unique within one DB file — every fresh test DB mints
  batch_id 1, and in production ``sync-replica.sh`` swaps the replica's DB
  file atomically, so a bare ``batch_id`` key could serve one DB's memoized
  state against another DB entirely for up to ``_TTL_SECONDS``. The key uses
  ``web._db_path`` — the exact path ``_get_db()`` opens — so it always
  matches the DB the handler is actually about to read.
- **A generation counter guards against a write's invalidation being
  clobbered by an in-flight recompute that started before it.** Without
  this: thread A holds the lock recomputing batch 5's state; thread B runs
  ``POST /5/ready``, commits, and calls ``_invalidate`` — which finds
  nothing to pop, because A hasn't stored anything yet. A then finishes and
  stores its *pre-write* snapshot under a *fresh* timestamp, so the stale
  ``ready``/``next_action`` gets served for a full TTL window and nothing
  re-invalidates it (the write's invalidation already happened, in the
  past, before there was anything to invalidate). The fix: ``_invalidate``
  also bumps a per-key generation counter; the compute path snapshots the
  generation before calling ``batch_state`` and only stores its result if
  the generation is still unchanged afterward. If it changed, the freshly
  computed response is still returned to the caller that triggered it (it
  is not wrong, just possibly no longer the latest), but it is not cached —
  so the very next GET (lock now free) recomputes instead of serving it for
  another ``_TTL_SECONDS``.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from . import ingest_batches
from .batch_state import batch_state as _batch_state

router = APIRouter(prefix="/api/batches", tags=["batches"])

_TTL_SECONDS = 30

# (db_path, batch_id) -> (computed_at_monotonic, response_dict).
#
# One lock for the WHOLE module, not one per batch — see module docstring.
_memo: dict[tuple[str, int], tuple[float, dict]] = {}
_lock = threading.Lock()

# (db_path, batch_id) -> generation counter, bumped by every _invalidate().
# Lets an in-flight recompute detect that its result was invalidated out
# from under it before it had a chance to store anything (see module
# docstring, "A generation counter guards against...").
_generation: dict[tuple[str, int], int] = {}


def _reset_cache() -> None:
    """Test hook: drop all memoized state.

    Two different DB paths in one test process must never serve each
    other's cached batch state, so tests call this between runs.
    """
    _memo.clear()
    _generation.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db():
    # Deferred import: web.py imports this module's router, so importing
    # web at module load time would be circular. Mirrors admin_api.py's
    # `from . import web; web._get_db()` pattern.
    from . import web
    return web._get_db()


def _current_db_path() -> str:
    """The exact path `_get_db()` is about to open — used as half the memo
    key so the cache can never serve one DB's state for another's."""
    from . import web
    return web._db_path


def _key(batch_id: int) -> tuple[str, int]:
    return (_current_db_path(), batch_id)


def _invalidate(key: tuple[str, int]) -> None:
    _memo.pop(key, None)
    _generation[key] = _generation.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Replica mode — EVERY route here proxies to the NAS
# ---------------------------------------------------------------------------
#
# Batches, sweeps and job rows are written on the NAS, which is the sole
# writer; the desktop replica's DB is a periodically-synced COPY. Serving
# these routes from that copy gets four things wrong at once: batches
# registered since the last sync are missing entirely, sweep progress is
# frozen, the job rows a fleet launch just POSTed to the NAS are invisible —
# so the page keeps saying `next_action == "launch_fleet"` and a second click
# makes run-workers.sh kill and restart a running fleet.
#
# So: when PHOTOSEARCH_NAS_URL is set, every route below forwards to the NAS
# and **never falls back to local data**. A wrong batch view is worse than an
# error — the page renders errors, and an operator who sees "could not reach
# the NAS" knows what to do, where one shown a stale pipeline does not. Same
# proxy shape as admin_api's `_incoming_status` / `_workers_queue_status`.

_PROXY_TIMEOUT = 15


def _nas_url() -> str:
    import os
    return (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/")


def _proxy(method: str, path: str, *, params=None, json_body=None):
    """Forward one request to the NAS and return its JSON body.

    Status codes the NAS meant (404 on an unknown batch, 400 on a bad step)
    are re-raised as themselves so the caller sees the real answer; anything
    that stops us reaching it at all becomes a 502 naming the host we tried.
    """
    import requests
    nas = _nas_url()
    url = f"{nas}{path}"
    try:
        resp = requests.request(method, url, params=params, json=json_body,
                                timeout=_PROXY_TIMEOUT)
    except requests.RequestException as exc:
        raise HTTPException(
            502, f"could not reach the authoritative server at {nas}: {exc}")
    if resp.status_code >= 400:
        detail = resp.text[:300]
        try:
            detail = resp.json().get("detail", detail)
        except ValueError:
            pass
        raise HTTPException(resp.status_code, detail)
    try:
        return resp.json()
    except ValueError:
        raise HTTPException(
            502, f"authoritative server returned a non-JSON body from {path}")


class RegisterRequest(BaseModel):
    directory: str
    source: Optional[str] = None


class OpenJobsRequest(BaseModel):
    """Steps to record as launched for a batch.

    Exists for the replica: the fleet launches on the desktop but the NAS is
    the sole writer, so the desktop has to record the job intent *there* or
    the NAS's own /batches page would keep showing those passes as
    `needs_queue` while a fleet is already draining them.
    """
    steps: list[str]
    job_kind: str = "fleet"


# ---------------------------------------------------------------------------
# GET /api/batches — pure SQL, no derivation
# ---------------------------------------------------------------------------

@router.get("")
def list_batches_endpoint(include_dismissed: bool = Query(False)):
    if _nas_url():
        return _proxy("GET", "/api/batches",
                      params={"include_dismissed": int(bool(include_dismissed))})
    with _get_db() as db:
        return {
            "sweep": ingest_batches.get_active_sweep(db),
            "batches": ingest_batches.list_batches(
                db, include_dismissed=include_dismissed
            ),
        }


# ---------------------------------------------------------------------------
# GET /api/batches/{id} — cached derived state
# ---------------------------------------------------------------------------

@router.get("/{batch_id}")
def get_batch_status(batch_id: int):
    if _nas_url():
        # No local memo on this path: the NAS runs the same non-blocking
        # cache one hop away, so memoizing its answer here would only add a
        # second TTL to every change.
        return _proxy("GET", f"/api/batches/{batch_id}")
    key = _key(batch_id)
    now = time.monotonic()
    cached = _memo.get(key)
    if cached is not None and (now - cached[0]) < _TTL_SECONDS:
        return cached[1]

    acquired = _lock.acquire(blocking=False)
    if not acquired:
        # Someone else is mid-recompute (for this batch or any other) —
        # never wait. Serve what we have.
        if cached is not None:
            stale = dict(cached[1])
            stale["stale"] = True
            return stale
        with _get_db() as db:
            batch = ingest_batches.get_batch(db, batch_id)
        if batch is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        return {"batch": batch, "steps": [], "stale": True, "computing": True}

    try:
        generation_before = _generation.get(key, 0)
        with _get_db() as db:
            try:
                state = _batch_state(db, batch_id)
            except ValueError:
                raise HTTPException(404, f"no such batch: {batch_id}")
        response = dict(state)
        response["computed_at"] = _now_iso()
        response["stale"] = False
        # Only cache if nothing invalidated this key while we were
        # computing — otherwise we'd resurrect a pre-write snapshot under a
        # fresh timestamp and the invalidation that already fired would
        # have nothing left to do (see module docstring).
        if _generation.get(key, 0) == generation_before:
            _memo[key] = (time.monotonic(), response)
        return response
    finally:
        _lock.release()


# ---------------------------------------------------------------------------
# Writes — each invalidates the batch's memo entry
# ---------------------------------------------------------------------------

@router.post("/{batch_id}/dismiss")
def dismiss_batch_endpoint(batch_id: int):
    if _nas_url():
        return _proxy("POST", f"/api/batches/{batch_id}/dismiss")
    with _get_db() as db:
        if ingest_batches.get_batch(db, batch_id) is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        ingest_batches.dismiss_batch(db, batch_id)
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(_key(batch_id))
    return {"batch": batch}


@router.post("/{batch_id}/ready")
def mark_ready_endpoint(batch_id: int):
    if _nas_url():
        return _proxy("POST", f"/api/batches/{batch_id}/ready")
    with _get_db() as db:
        if ingest_batches.get_batch(db, batch_id) is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        ingest_batches.mark_ready(db, batch_id)
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(_key(batch_id))
    return {"batch": batch}


@router.post("/{batch_id}/jobs")
def open_jobs_endpoint(batch_id: int, body: OpenJobsRequest):
    """Open a job row per step — the authoritative write behind a remote
    fleet launch.

    Steps are validated against ``STEP_ORDER`` **before anything is written**
    (``batch_advance.open_step_job``): ``ingest_batch_jobs.step`` is bare TEXT
    that nothing downstream checks, so a typo'd name would never match a
    derived step and the real one would read `needs_queue` forever.
    """
    from . import batch_advance
    if _nas_url():
        return _proxy("POST", f"/api/batches/{batch_id}/jobs",
                      json_body={"steps": body.steps, "job_kind": body.job_kind})
    with _get_db() as db:
        if ingest_batches.get_batch(db, batch_id) is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        # Validate the whole list up front so a bad name in position 3 can't
        # leave the first two half-applied.
        from .batch_state import STEP_ORDER
        bad = [s for s in body.steps if s not in STEP_ORDER]
        if bad:
            raise HTTPException(400, f"unknown step(s): {', '.join(bad)}")
        for step in body.steps:
            batch_advance.open_step_job(db, batch_id, step, body.job_kind)
    _invalidate(_key(batch_id))
    return {"batch_id": batch_id, "steps": body.steps, "job_kind": body.job_kind}


@router.post("/register")
def register_batch_endpoint(body: RegisterRequest):
    if _nas_url():
        return _proxy("POST", "/api/batches/register",
                      json_body={"directory": body.directory, "source": body.source})
    with _get_db() as db:
        try:
            batch_id = ingest_batches.register_batch(
                db, body.directory, source=body.source
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(_key(batch_id))
    return {"batch_id": batch_id, "batch": batch}

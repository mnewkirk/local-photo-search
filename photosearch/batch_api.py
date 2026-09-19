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

# batch_id -> (computed_at_monotonic, response_dict).
#
# One lock for the WHOLE module, not one per batch — see module docstring.
_memo: dict[int, tuple[float, dict]] = {}
_lock = threading.Lock()


def _reset_cache() -> None:
    """Test hook: drop all memoized state.

    Two different DB paths in one test process must never serve each
    other's cached batch state, so tests call this between runs.
    """
    _memo.clear()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_db():
    # Deferred import: web.py imports this module's router, so importing
    # web at module load time would be circular. Mirrors admin_api.py's
    # `from . import web; web._get_db()` pattern.
    from . import web
    return web._get_db()


def _invalidate(batch_id: int) -> None:
    _memo.pop(batch_id, None)


class RegisterRequest(BaseModel):
    directory: str
    source: Optional[str] = None


# ---------------------------------------------------------------------------
# GET /api/batches — pure SQL, no derivation
# ---------------------------------------------------------------------------

@router.get("")
def list_batches_endpoint(include_dismissed: bool = Query(False)):
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
    now = time.monotonic()
    cached = _memo.get(batch_id)
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
        with _get_db() as db:
            try:
                state = _batch_state(db, batch_id)
            except ValueError:
                raise HTTPException(404, f"no such batch: {batch_id}")
        response = dict(state)
        response["computed_at"] = _now_iso()
        response["stale"] = False
        _memo[batch_id] = (time.monotonic(), response)
        return response
    finally:
        _lock.release()


# ---------------------------------------------------------------------------
# Writes — each invalidates the batch's memo entry
# ---------------------------------------------------------------------------

@router.post("/{batch_id}/dismiss")
def dismiss_batch_endpoint(batch_id: int):
    with _get_db() as db:
        if ingest_batches.get_batch(db, batch_id) is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        ingest_batches.dismiss_batch(db, batch_id)
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(batch_id)
    return {"batch": batch}


@router.post("/{batch_id}/ready")
def mark_ready_endpoint(batch_id: int):
    with _get_db() as db:
        if ingest_batches.get_batch(db, batch_id) is None:
            raise HTTPException(404, f"no such batch: {batch_id}")
        ingest_batches.mark_ready(db, batch_id)
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(batch_id)
    return {"batch": batch}


@router.post("/register")
def register_batch_endpoint(body: RegisterRequest):
    with _get_db() as db:
        try:
            batch_id = ingest_batches.register_batch(
                db, body.directory, source=body.source
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        batch = ingest_batches.get_batch(db, batch_id)
    _invalidate(batch_id)
    return {"batch_id": batch_id, "batch": batch}

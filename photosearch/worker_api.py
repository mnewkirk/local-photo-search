"""Worker API — endpoints for distributed indexing.

Remote workers (e.g. a fast laptop) call these endpoints on the NAS to:
  1. Claim a batch of unprocessed photos
  2. Download photo bytes for local processing
  3. Submit results back per-batch

The NAS remains the single source of truth (SQLite DB).
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .db import PhotoDB, _serialize_float_list, _deserialize_float_list, CLIP_DIMENSIONS

logger = logging.getLogger("photosearch.worker_api")

router = APIRouter(prefix="/api/worker", tags=["worker"])

# These are set by web.py at startup
_db_path: str = ""
_photo_root: Optional[str] = None

# Flipped True by web.py's shutdown handler when uvicorn starts draining.
# Worker traffic (claim/submit/renew/photo-bytes) returns 503 + Retry-After
# while this is set so workers back off instead of piling new requests onto
# the draining process. Without this, the drain window can exceed
# stop_grace_period and the container gets SIGKILL'd mid-transfer (exit 137).
_shutting_down: bool = False


def begin_shutdown():
    """Called from web.py's FastAPI shutdown event."""
    global _shutting_down
    _shutting_down = True
    logger.info("worker_api: shutdown signaled — returning 503 to worker traffic")


def is_shutting_down() -> bool:
    return _shutting_down


def configure(db_path: str, photo_root: Optional[str] = None):
    """Called by web.py to pass DB config to the worker router."""
    global _db_path, _photo_root
    _db_path = db_path
    _photo_root = photo_root
    _extend_claims_on_startup()


def _get_db() -> PhotoDB:
    return PhotoDB(_db_path, photo_root=_photo_root)


_STARTUP_GRACE_MINUTES = 10


def _extend_claims_on_startup():
    """Extend all existing claims on service restart.

    While the server was down, workers couldn't renew their claims via heartbeat.
    Rather than letting those claims expire (and wasting the worker's in-progress
    compute), give them a grace period so the next heartbeat can reach us.
    """
    if not _db_path:
        return
    try:
        with PhotoDB(_db_path) as db:
            cur = db.conn.execute(
                "UPDATE worker_claims SET expires_at = datetime('now', ?)",
                (f"+{_STARTUP_GRACE_MINUTES} minutes",),
            )
            db.conn.commit()
            if cur.rowcount:
                logger.info(
                    f"Service restart: extended {cur.rowcount} active claim(s) "
                    f"by {_STARTUP_GRACE_MINUTES} minutes"
                )
    except Exception as e:
        logger.warning(f"Could not extend claims on startup: {e}")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ClaimRequest(BaseModel):
    worker_id: str
    pass_type: str  # 'clip', 'faces', 'quality', 'describe', 'tags', 'verify'
    limit: int = 16
    collection_id: Optional[int] = None
    directory: Optional[str] = None  # e.g. "/photos/2026/2026-04-09"
    # Structured filter set (the same vocabulary _build_filter_sql / search_photos
    # accept: date_from/date_to, people, location, min_quality, min_aesthetic,
    # camera, category, visual_tag, keyword, style_tag). Resolved server-side to a
    # photo-id scope — a third way to produce scope_ids alongside collection/dir.
    filters: Optional[dict] = None
    ttl_minutes: int = 30


class ClaimResponse(BaseModel):
    batch_id: str
    pass_type: str
    photos: list[dict]  # [{id, filepath, filename, description}]
    remaining: int = 0  # unclaimed photos left in queue after this claim


class ClipResult(BaseModel):
    photo_id: int
    embedding: list[float]


class FaceResult(BaseModel):
    photo_id: int
    faces: list[dict]  # [{bbox: [t,r,b,l], encoding: [...]}]


class QualityResult(BaseModel):
    photo_id: int
    aesthetic_score: float
    aesthetic_concepts: Optional[str] = None  # JSON string


class DescribeResult(BaseModel):
    photo_id: int
    description: Optional[str] = None


class TagsResult(BaseModel):
    photo_id: int
    tags: list[str]


class VerifyResult(BaseModel):
    photo_id: int
    status: str  # 'pass', 'regenerated'
    verified_at: str
    hallucination_flags: Optional[str] = None  # JSON string
    description: Optional[str] = None  # regenerated description
    tags: Optional[list[str]] = None  # regenerated tags


class CategoryContentResult(BaseModel):
    photo_id: int
    categories: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class CategoryVisualResult(BaseModel):
    photo_id: int
    visual_tags: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class KeywordsResult(BaseModel):
    photo_id: int
    keywords: list[str]
    model: Optional[str] = None
    model_version: Optional[str] = None


class AestheticsResult(BaseModel):
    photo_id: int
    # Flat aes_* scalar columns (aes_overall, aes_technical, aes_sharpness, ...).
    # Passed through by name to update_photo, so the worker and server stay
    # decoupled from the exact sub-attribute set.
    scores: dict[str, float] = {}
    aes_style: Optional[str] = None       # JSON: {facets, critiques}
    aes_style_tags: Optional[str] = None  # JSON array of style tags
    # Subject-aware quality (schema v27). subject_boxes: JSON array of the main
    # subject box(es); aes_subject_overall + aes_subject (JSON breakdown): the
    # primary subject's crop aesthetic score. All optional so older workers /
    # non-subject runs submit cleanly.
    subject_boxes: Optional[str] = None
    aes_subject_overall: Optional[float] = None
    aes_subject: Optional[str] = None
    model: Optional[str] = None
    model_version: Optional[str] = None


class SubmitRequest(BaseModel):
    batch_id: str
    pass_type: str
    clip_results: Optional[list[ClipResult]] = None
    face_results: Optional[list[FaceResult]] = None
    quality_results: Optional[list[QualityResult]] = None
    describe_results: Optional[list[DescribeResult]] = None
    tags_results: Optional[list[TagsResult]] = None
    verify_results: Optional[list[VerifyResult]] = None
    category_content_results: Optional[list[CategoryContentResult]] = None
    category_visual_results: Optional[list[CategoryVisualResult]] = None
    keywords_results: Optional[list[KeywordsResult]] = None
    aesthetics_results: Optional[list[AestheticsResult]] = None
    # Model provenance for describe/tags/verify — logged to the generations
    # table. Optional so older workers still submit cleanly.
    model: Optional[str] = None
    model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------

# A scope resolved from a broad filter (a whole camera model, a frequent
# person) can be tens of thousands of ids. get_unprocessed_photos binds each
# scope id as a SQL parameter, so a large scope would blow past
# SQLITE_MAX_VARIABLE_NUMBER (32766 on a default-compiled libsqlite3). Chunk the
# scope below that so queue reads stay valid regardless of scope size. Normal
# directory/collection/date-range scopes are far smaller and never chunk.
_SCOPE_CHUNK = 20000


def _unprocessed_scoped(db, pass_type, scope_ids, limit, commit_cleanup):
    """get_unprocessed_photos with large-scope chunking. Chunks are disjoint id
    ranges, each excluding already-claimed photos, so we accumulate up to `limit`
    fresh rows across chunks."""
    if not scope_ids or len(scope_ids) <= _SCOPE_CHUNK:
        return db.get_unprocessed_photos(
            pass_type=pass_type, photo_ids=scope_ids, limit=limit,
            commit_cleanup=commit_cleanup)
    out: list = []
    for i in range(0, len(scope_ids), _SCOPE_CHUNK):
        chunk = scope_ids[i:i + _SCOPE_CHUNK]
        out.extend(db.get_unprocessed_photos(
            pass_type=pass_type, photo_ids=chunk, limit=limit - len(out),
            commit_cleanup=commit_cleanup))
        if len(out) >= limit:
            break
    return out[:limit]


def _count_scoped(db, pass_type, scope_ids):
    """count_unprocessed_photos with the same large-scope chunking."""
    if not scope_ids or len(scope_ids) <= _SCOPE_CHUNK:
        return db.count_unprocessed_photos(pass_type, photo_ids=scope_ids)
    return sum(
        db.count_unprocessed_photos(pass_type, photo_ids=scope_ids[i:i + _SCOPE_CHUNK])
        for i in range(0, len(scope_ids), _SCOPE_CHUNK))


def _resolve_scope_ids(db, collection_id, directory, filters):
    """Collapse the mutually-exclusive scoping inputs into a flat photo-id list
    (or None = whole library). Precedence: collection > directory > filters.

    `filters` is the structured filter set (_build_filter_sql vocabulary). An
    empty dict, or one whose keys don't build any clause, is treated as
    "no scope" (whole library) rather than an all-rows IN list. Raises 404 when
    a scope is requested but matches zero photos.
    """
    if collection_id is not None:
        ids = db.get_collection_photo_ids(collection_id)
        if not ids:
            raise HTTPException(404, f"Collection {collection_id} has no photos")
        return ids
    if directory is not None:
        ids = db.get_directory_photo_ids(directory)
        if not ids:
            raise HTTPException(404, f"No photos found in directory {directory}")
        return ids
    if filters:
        from .tools import _build_filter_sql  # local import: keeps CLIP/torch out of import path
        where, params = _build_filter_sql(db, filters)
        if where == "1":
            return None  # no recognized filter key → whole library
        ids = [r[0] for r in db.conn.execute(
            f"SELECT id FROM photos WHERE {where}", params).fetchall()]
        if not ids:
            raise HTTPException(404, "No photos match the given filters")
        return ids
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

# How many times claim-batch re-looks when another worker took the candidates
# between the unlocked scan and the locked claim. Bounded so a pathological
# fleet can't spin here; exceeding it reports contended=True, never "empty".
_CLAIM_RACE_RETRIES = 3


@router.post("/claim-batch")
def claim_batch(req: ClaimRequest):
    """Claim a batch of unprocessed photos for a given pass type.

    Returns photo metadata + download URLs. The claim expires after ttl_minutes.
    """
    with _get_db() as db:
        # Scope to collection, directory, or a structured filter set if requested
        scope_ids = _resolve_scope_ids(
            db, req.collection_id, req.directory, req.filters)

        # Two phases, so the SQLite WRITER LOCK is never held across the scan.
        #
        # This used to be one BEGIN IMMEDIATE wrapped around the whole thing:
        # take the writer lock, run the (slow, NOT-IN) unprocessed scan, claim,
        # commit. That serialized every other writer on the box behind a scan
        # — and a fleet polling an EMPTY queue paid the full price for nothing.
        # Measured consequences on 2026-09-13: POST /api/faces/{id}/assign
        # returning 500 "database is locked", 250 collection inserts taking
        # ~30 minutes, and an overnight ingest moving 2,040 files while writing
        # ZERO rows.
        #
        # Phase 1 (NO LOCK) finds candidates. The common case — nothing to do —
        # now returns without ever acquiring the writer lock.
        # Phase 2 takes the lock only when there is work, and re-checks just
        # those few ids (an indexed lookup on <= limit rows, not a scan) before
        # inserting the claim. That re-check is what preserves the invariant the
        # original BEGIN IMMEDIATE existed for: two workers must never claim the
        # same photo and both download it.
        photos = None
        batch_id = None
        for _attempt in range(_CLAIM_RACE_RETRIES):
            candidates = _unprocessed_scoped(
                db, req.pass_type, scope_ids, req.limit, commit_cleanup=True)
            if not candidates:
                # Genuinely empty. This is the ONLY path that reports no work,
                # which matters: the worker retires a pass on an empty queue,
                # so a lost race must never look like one.
                return JSONResponse({"batch_id": None, "pass_type": req.pass_type,
                                     "photos": []})

            cand_ids = [p["id"] for p in candidates]
            db.conn.execute("BEGIN IMMEDIATE")
            try:
                # Re-verify inside the lock, scoped to the candidates only.
                confirmed = db.get_unprocessed_photos(
                    pass_type=req.pass_type, photo_ids=cand_ids,
                    limit=req.limit, commit_cleanup=False)
                if not confirmed:
                    # Another worker took them between the phases. NOT an empty
                    # queue — loop and look again rather than reporting no work.
                    db.conn.commit()
                    continue
                photos = confirmed
                batch_id = db.claim_photos(
                    worker_id=req.worker_id,
                    pass_type=req.pass_type,
                    photo_ids=[p["id"] for p in confirmed],
                    ttl_minutes=req.ttl_minutes,
                    commit=False,
                )
                db.conn.commit()
                break
            except Exception:
                db.conn.rollback()
                raise
        if photos is None:
            # Lost the race every attempt — heavy contention, not an empty
            # queue. Report "no batch" without the empty-queue semantics by
            # telling the worker explicitly that it was contended.
            return JSONResponse({"batch_id": None, "pass_type": req.pass_type,
                                 "photos": [], "contended": True})

        # Count how many remain unclaimed after this batch. Uses a COUNT(*)
        # rather than materializing every unprocessed row — with N concurrent
        # workers a full-queue scan per claim turned into ReadTimeout cascades
        # on large libraries. The count doesn't subtract active claims, so the
        # value is slightly inflated, but it's purely informational (worker
        # logs it; no decision is made on it).
        remaining = _count_scoped(db, req.pass_type, scope_ids)

        # Fetch description for every claimed photo in one query so text-only
        # passes (category-content, keywords) can run without downloading image bytes.
        photo_id_list = [p["id"] for p in photos]
        placeholders = ",".join("?" * len(photo_id_list))
        desc_rows = db.conn.execute(
            f"SELECT id, description FROM photos WHERE id IN ({placeholders})",
            photo_id_list,
        ).fetchall()
        desc_by_id = {row["id"]: row["description"] for row in desc_rows}

        result_photos = []
        for p in photos:
            result_photos.append({
                "id": p["id"],
                "filepath": p["filepath"],
                "filename": Path(p["filepath"]).name,
                "description": desc_by_id.get(p["id"]),
            })

        return ClaimResponse(
            batch_id=batch_id,
            pass_type=req.pass_type,
            photos=result_photos,
            remaining=remaining,
        )


# ---------------------------------------------------------------------------
# Submit bookkeeping — an attempt is only spent when the result was PERSISTED
# ---------------------------------------------------------------------------

class _SubmitOutcome:
    """Separates "this photo is done" from "the SERVER could not write it".

    `mark_processed` UPSERT-increments `worker_processed.attempts`, and the
    claim path stops claiming a photo at MAX_PROCESS_ATTEMPTS. So counting a
    server-side write failure as an attempt punishes the PHOTO for the
    server's problem: a handful of unlucky lock collisions permanently
    abandons a perfectly good photo and throws away the GPU work each time.
    Observed 2026-09-20 — two sharp, ordinary photos reached attempts=5 for
    'category-visual' with `visual_tags` still NULL, while the errors table
    logged `database is locked` for their batch-mates at the same second
    (the nightly maintenance sweep held the write lock past the 60 s
    busy_timeout).

    An attempt is spent only when the result was actually persisted, or when
    the WORKER reported a genuine per-photo outcome (no description, no
    faces, an empty tag list). A DB write that raised defers instead: the
    photo keeps its attempt count and is reclaimed on a later pass.
    """

    def __init__(self):
        self.written = 0
        self.processed: list[int] = []
        self.deferred: list[int] = []

    def persisted(self, photo_id: int):
        """This photo's outcome is final — spend an attempt on it."""
        if photo_id not in self.deferred and photo_id not in self.processed:
            self.processed.append(photo_id)

    def defer(self, photo_id: int):
        """A transient lock — do NOT spend an attempt; it comes back around."""
        if photo_id in self.processed:
            self.processed.remove(photo_id)
        if photo_id not in self.deferred:
            self.deferred.append(photo_id)

    def failed(self, photo_id: int):
        """A repeatable failure — spend the attempt so the cap bounds it.

        Beats a deferral for the same photo (a photo whose payload is broken
        will break again, and only the attempt cap can stop the loop).
        """
        if photo_id in self.deferred:
            self.deferred.remove(photo_id)
        if photo_id not in self.processed:
            self.processed.append(photo_id)

    def commit_failed(self, transient: bool):
        """The batch COMMIT raised, so the final flush did not land.

        A transient lock defers everything — the work is intact and comes
        back. Anything else (disk full, read-only FS, corruption) is not
        going to fix itself batch-over-batch, so the attempt is still spent:
        the 3-attempt cap is the only thing that stops the fleet re-claiming
        the same photos forever.
        """
        self.written = 0
        if transient:
            for pid in self.processed:
                if pid not in self.deferred:
                    self.deferred.append(pid)
            self.processed = []

    def status(self) -> str:
        if not self.deferred:
            return "ok"
        return "partial" if self.processed or self.written else "deferred"


# Only a TRANSIENT lock earns a free retry. Deferring anything else would
# uncap the pass: a deterministically-failing photo would never reach
# MAX_PROCESS_ATTEMPTS, be re-claimed every TTL forever, and pay for a model
# run each cycle — exactly the pathology CLAUDE.md documents for the un-capped
# `clip` pass ("workers churn at ~290% CPU and queue_depth.clip never reaches
# 0"). `faces` is the worst case: its claim predicate is NOT EXISTS(faces) AND
# attempts < MAX with no column to heal it, and a malformed payload (a short
# bbox, a missing 'encoding', a non-512 vector) raises BEFORE any INSERT, so
# nothing is written AND nothing is marked. DO NOT widen this back to a bare
# `except Exception` defer.
_TRANSIENT_DB_MARKERS = ("locked", "busy")


def _is_transient_db_error(exc: BaseException) -> bool:
    """True only for SQLite contention — a lock we should simply wait out.

    Shared by the per-row, batch-commit and mark_processed paths so their
    classifications cannot drift apart.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_DB_MARKERS)


def _merge_visual_tags(db, photo_id: int, perceived) -> list:
    """Apply the derived capture-fact merge to one category-visual answer.

    Reads the photo's EXIF here, on the authoritative writer, because that is
    the only place it exists — the worker has the pixels and nothing else. A
    row that has vanished (or a read that fails) degrades to strip-only: the
    bogus capture facts still go, nothing wrong is invented.
    """
    from .visual_tags_derive import DERIVE_COLUMNS, merge_for_row, merge_tags

    try:
        row = db.conn.execute(
            f"SELECT {', '.join(DERIVE_COLUMNS)} FROM photos WHERE id=?",
            (photo_id,),
        ).fetchone()
    except sqlite3.Error:
        row = None
    if row is None:
        return merge_tags(perceived, [])
    return merge_for_row(perceived, row)


def _has_perceived(tags) -> bool:
    """True when at least one tag in the merged array came from the model."""
    from .visual_tags_derive import PERCEIVED_VOCABULARY

    return bool(set(tags or []) & set(PERCEIVED_VOCABULARY))


def _record_write_failure(db, outcome: _SubmitOutcome, pass_type: str,
                          photo_id: int, exc: Exception):
    """Handle one photo whose DB write raised, and never lose the reason.

    A transient lock defers (no attempt spent). Any other error counts the
    attempt, exactly as before this file learned to defer at all, so the
    3-attempt cap still bounds a repeatable failure.
    """
    if _is_transient_db_error(exc):
        logger.warning("Failed to store %s for photo %s — deferring for retry: %s",
                       pass_type, photo_id, exc)
        outcome.defer(photo_id)
    else:
        logger.warning("Failed to store %s for photo %s — counting the attempt "
                       "(not a lock, so retrying would not help): %s",
                       pass_type, photo_id, exc)
        outcome.failed(photo_id)
    try:
        db.log_error(pass_type, str(photo_id), str(exc))
    except Exception as log_exc:
        # The conditions that break the write (a locked DB) are exactly the
        # ones that break logging it, so a bare `except: pass` here made the
        # reason vanish precisely when it mattered.
        logger.warning("Could not log the %s error for photo %s (%s); "
                       "original error was: %s", pass_type, photo_id, log_exc, exc)


def _commit_batch(db, outcome: _SubmitOutcome, pass_type: str) -> bool:
    """Commit the pending batch; return True when the caller may mark its
    processed set.

    begin_batch defers the commit, so under write contention SQLITE_BUSY
    usually surfaces at COMMIT rather than at the UPDATE. What is knowable
    then is only that the FINAL flush did not land: rows written before an
    intermediate flush — or before a `db.log_error` call, which commits
    unconditionally — may well be on disk. That is harmless either way, since
    a landed-but-unmarked row is simply not re-claimed by a column-guarded
    predicate.

    A transient lock defers the whole batch (nothing marked — the work is
    intact and comes back). Any OTHER commit failure (disk full, read-only
    FS, corruption) is not going to resolve batch-over-batch, so the attempt
    is still counted: re-claiming is unavoidable for a column-guarded pass,
    but the 3-attempt cap is the only thing that bounds the loop.
    """
    try:
        db.end_batch()
        return True
    except Exception as e:
        # Leaves the connection usable rather than stuck in batch mode with an
        # open transaction (and stops close() re-raising the same error).
        db.abort_batch()
        transient = _is_transient_db_error(e)
        n = len(outcome.processed) + len(outcome.deferred)
        if transient:
            logger.warning("Batch commit failed for pass %s under contention — "
                           "nothing marked processed, %d result(s) deferred: %s",
                           pass_type, n, e)
        else:
            logger.error("Batch commit failed for pass %s and it is NOT a lock — "
                         "the final flush did not land for %d result(s); counting "
                         "the attempt so the fleet cannot loop on it: %s",
                         pass_type, n, e)
        try:
            db.log_error(pass_type, "batch", f"batch commit failed: {e}")
        except Exception as log_exc:
            logger.warning("Could not log the %s batch-commit failure (%s); "
                           "original error was: %s", pass_type, log_exc, e)
        outcome.commit_failed(transient)
        return not transient


def _mark_processed(db, outcome: _SubmitOutcome, pass_type: str):
    """Spend one attempt per finished photo.

    If this write itself fails the attempt cannot be recorded at all — the
    one case that really can loop, since the photo comes back unmarked every
    TTL. Nothing here can fix an unwritable DB, so say so loudly.
    """
    if not outcome.processed:
        return
    try:
        db.mark_processed(outcome.processed, pass_type)
    except Exception as e:
        log = logger.warning if _is_transient_db_error(e) else logger.error
        log("Could not mark %d photo(s) processed for pass %s — their attempt "
            "was NOT recorded and they will be reclaimed; if this is not a "
            "lock the DB is unwritable and the fleet will keep retrying: %s",
            len(outcome.processed), pass_type, e)
        for pid in list(outcome.processed):
            outcome.defer(pid)


@router.post("/submit-results")
def submit_results(req: SubmitRequest):
    """Submit processing results for a claimed batch.

    Results are written directly to the main DB.
    The claim is released after successful write.

    A photo is only marked processed (which spends one of its
    MAX_PROCESS_ATTEMPTS) when its result was persisted or the worker
    reported a genuine per-photo outcome. Server-side write failures come
    back in `deferred` / `deferred_photo_ids` and the photo stays claimable.
    """
    with _get_db() as db:
        # Check if the claim exists — accept results even if expired, since the
        # worker already did the work and discarding it wastes compute.
        active_row = db.conn.execute(
            "SELECT * FROM worker_claims WHERE batch_id = ? AND expires_at > datetime('now')",
            (req.batch_id,),
        ).fetchone()
        if not active_row:
            # Check if the row exists but expired (vs. already cleaned up)
            any_row = db.conn.execute(
                "SELECT * FROM worker_claims WHERE batch_id = ?",
                (req.batch_id,),
            ).fetchone()
            if any_row:
                logger.warning(f"Claim {req.batch_id} expired but accepting results (work already done)")
            else:
                logger.warning(f"Claim {req.batch_id} expired and was cleaned up, but accepting results anyway")

        outcome = _SubmitOutcome()

        if req.pass_type == "clip" and req.clip_results:
            db.begin_batch(batch_size=100)
            for r in req.clip_results:
                try:
                    db.add_clip_embedding(r.photo_id, r.embedding)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "clip", r.photo_id, e)
            _commit_batch(db, outcome, "clip")
            # NOTE: clip deliberately never calls mark_processed — the
            # embedding row itself is the marker (see CLAUDE.md, "Non-image
            # rows & the clip-claim infinite re-claim").

        elif req.pass_type == "faces":
            face_results = req.face_results or []
            db.begin_batch(batch_size=50)
            for r in face_results:
                failed = False
                for face in r.faces:
                    try:
                        db.add_face(
                            photo_id=r.photo_id,
                            bbox=tuple(face["bbox"]),
                            encoding=face["encoding"],
                            det_score=face.get("det_score"),
                        )
                        outcome.written += 1
                    except Exception as e:
                        failed = True
                        _record_write_failure(db, outcome, "faces", r.photo_id, e)
                if not failed:
                    # A photo with NO faces found is done, not failed — its
                    # attempt is spent here because nothing else records it.
                    outcome.persisted(r.photo_id)
            committed = _commit_batch(db, outcome, "faces")

            # New faces land with cluster_id=NULL. Global clustering is a
            # separate, on-demand step via `photosearch recluster-faces` —
            # per-batch clustering would collide IDs across batches and
            # fragment the same person across many pseudo-clusters.

            # Mark every photo whose faces landed (including those with none).
            if committed:
                _mark_processed(db, outcome, "faces")

        elif req.pass_type == "quality" and req.quality_results:
            db.begin_batch(batch_size=100)
            for r in req.quality_results:
                try:
                    updates = {"aesthetic_score": r.aesthetic_score}
                    if r.aesthetic_concepts:
                        updates["aesthetic_concepts"] = r.aesthetic_concepts
                    db.update_photo(r.photo_id, **updates)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "quality", r.photo_id, e)
            _commit_batch(db, outcome, "quality")

        elif req.pass_type == "describe":
            describe_results = req.describe_results or []
            db.begin_batch(batch_size=100)
            for r in describe_results:
                if not r.description:
                    # The model returned nothing — a completed attempt. The
                    # worker omits true deferrals (timeouts) from the payload.
                    outcome.persisted(r.photo_id)
                    continue
                try:
                    db.update_photo(r.photo_id, description=r.description)
                    db.log_generation(r.photo_id, "describe", r.description,
                                      req.model, req.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "describe", r.photo_id, e)
            committed = _commit_batch(db, outcome, "describe")
            # Mark all submitted photos as processed (including those with no description)
            if committed:
                _mark_processed(db, outcome, "describe")

        elif req.pass_type == "tags":
            tags_results = req.tags_results or []
            db.begin_batch(batch_size=100)
            for r in tags_results:
                if not r.tags:
                    outcome.persisted(r.photo_id)
                    continue
                try:
                    tags_json = json.dumps(r.tags)
                    db.update_photo(r.photo_id, tags=tags_json)
                    db.log_generation(r.photo_id, "tags", tags_json,
                                      req.model, req.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "tags", r.photo_id, e)
            committed = _commit_batch(db, outcome, "tags")
            # Mark all submitted photos as processed (including those with no tags)
            if committed:
                _mark_processed(db, outcome, "tags")

        elif req.pass_type == "verify" and req.verify_results:
            db.begin_batch(batch_size=100)
            for r in req.verify_results:
                try:
                    updates = {
                        "verified_at": r.verified_at,
                        "verification_status": r.status,
                        "hallucination_flags": r.hallucination_flags,
                    }
                    # If hallucinations were found and descriptions regenerated
                    if r.description:
                        updates["description"] = r.description
                    if r.tags:
                        updates["tags"] = json.dumps(r.tags)
                    db.update_photo(r.photo_id, **updates)
                    # Log the regenerated description as a 'verify' generation —
                    # marks it as produced by the verify/regen pass, distinct
                    # from a first-pass describe.
                    if r.description:
                        db.log_generation(r.photo_id, "verify", r.description,
                                          req.model, req.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "verify", r.photo_id, e)
            _commit_batch(db, outcome, "verify")

        elif req.pass_type == "category-content":
            category_content_results = req.category_content_results or []
            db.begin_batch(batch_size=100)
            for r in category_content_results:
                try:
                    # Always persist the column. A successful-but-empty result
                    # writes '[]' so the photo is marked done in ONE pass (column
                    # is NOT NULL). Only a timeout/error defers, and the worker
                    # omits those from results. Provenance logged for non-empty only.
                    cats_json = json.dumps(r.categories or [])
                    db.update_photo(r.photo_id, categories=cats_json)
                    if r.categories:
                        db.log_generation(r.photo_id, "category-content", cats_json,
                                          r.model, r.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "category-content", r.photo_id, e)
            committed = _commit_batch(db, outcome, "category-content")
            # Mark all submitted photos as processed (including those with no categories)
            if committed:
                _mark_processed(db, outcome, "category-content")

        elif req.pass_type == "category-visual":
            category_visual_results = req.category_visual_results or []
            db.begin_batch(batch_size=100)
            for r in category_visual_results:
                try:
                    # The worker only ever saw the pixels, so its answer is the
                    # PERCEIVED half. The capture facts (long-exposure /
                    # low-light / panoramic / sharp / blurry) are decided HERE,
                    # from the photo's EXIF, and override whatever the model
                    # said — see photosearch/visual_tags_derive.py. The merge
                    # lives server-side because this is where the EXIF is.
                    merged = _merge_visual_tags(db, r.photo_id, r.visual_tags)
                    # Always persist the column ('[]' for empty) so a successful
                    # empty result marks done in one pass; only timeouts defer.
                    # An empty merge IS a legitimate result.
                    vtags_json = json.dumps(merged)
                    db.update_photo(r.photo_id, visual_tags=vtags_json)
                    # Provenance covers LLM artifacts only. A row whose only
                    # surviving tags are derived was not produced by the model,
                    # so it gets no `generations` entry.
                    if _has_perceived(merged):
                        db.log_generation(r.photo_id, "category-visual", vtags_json,
                                          r.model, r.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "category-visual", r.photo_id, e)
            committed = _commit_batch(db, outcome, "category-visual")
            # Mark all submitted photos as processed (including those with no visual tags)
            if committed:
                _mark_processed(db, outcome, "category-visual")

        elif req.pass_type == "keywords":
            keywords_results = req.keywords_results or []
            db.begin_batch(batch_size=100)
            for r in keywords_results:
                try:
                    # Always persist the column ('[]' for empty) so a successful
                    # empty result marks done in one pass; only timeouts defer.
                    kw_json = json.dumps(r.keywords or [])
                    db.update_photo(r.photo_id, keywords=kw_json)
                    if r.keywords:
                        db.log_generation(r.photo_id, "keywords", kw_json,
                                          r.model, r.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "keywords", r.photo_id, e)
            committed = _commit_batch(db, outcome, "keywords")
            # Mark all submitted photos as processed (including those with no keywords)
            if committed:
                _mark_processed(db, outcome, "keywords")

        elif req.pass_type == "aesthetics":
            from .aesthetics import ALL_SUBATTRS, DIMENSIONS
            # Allowlist the aes_* scalar columns a client may set — update_photo
            # interpolates keys into SQL, so never trust arbitrary column names.
            allowed_score_cols = (
                {"aes_overall", "aes_technical_iqa", "aes_overall_iqa"}
                | {f"aes_{s}" for s in ALL_SUBATTRS}
                | {f"aes_{d}" for d in DIMENSIONS}
            )
            aesthetics_results = req.aesthetics_results or []
            now = db.conn.execute("SELECT datetime('now')").fetchone()[0]
            db.begin_batch(batch_size=100)
            for r in aesthetics_results:
                try:
                    fields = {k: v for k, v in (r.scores or {}).items()
                              if k in allowed_score_cols and v is not None}
                    if not fields or "aes_overall" not in fields:
                        # Nothing usable — a completed attempt, not a write
                        # failure (worker already omits deferrals, so this is a
                        # defensive guard, not the normal path).
                        outcome.persisted(r.photo_id)
                        continue
                    fields["aes_style"] = r.aes_style
                    fields["aes_style_tags"] = r.aes_style_tags
                    fields["aes_model"] = r.model
                    fields["aes_scored_at"] = now
                    # Subject-aware quality (v27) — additive; subject_boxes may be
                    # [] (no subject) with a NULL subject score.
                    if r.subject_boxes is not None:
                        fields["subject_boxes"] = r.subject_boxes
                        if r.aes_subject_overall is not None:
                            fields["aes_subject_overall"] = r.aes_subject_overall
                            fields["aes_subject"] = r.aes_subject
                            fields["aes_subject_model"] = r.model
                            fields["aes_subject_scored_at"] = now
                    db.update_photo(r.photo_id, **fields)
                    db.log_generation(
                        r.photo_id, "aesthetics",
                        json.dumps({"scores": fields.get("aes_overall"),
                                    "style": r.aes_style,
                                    "tags": r.aes_style_tags}),
                        r.model, r.model_version)
                    outcome.written += 1
                    outcome.persisted(r.photo_id)
                except Exception as e:
                    _record_write_failure(db, outcome, "aesthetics", r.photo_id, e)
            committed = _commit_batch(db, outcome, "aesthetics")
            if committed:
                _mark_processed(db, outcome, "aesthetics")

        # Log activity for the chart
        if outcome.written > 0:
            try:
                db.log_activity(req.pass_type, "index", outcome.written)
            except Exception as e:
                # Telemetry must never fail a submit that already landed.
                logger.warning("Could not log %s activity: %s", req.pass_type, e)

        # Release the claim — deferred photos must become claimable again.
        db.release_claim(req.batch_id)

        return {
            # `status` / `written` / `processed` / `batch_id` are what older
            # workers read; `deferred*` is additive.
            "status": outcome.status(),
            "written": outcome.written,
            "processed": len(outcome.processed),
            "batch_id": req.batch_id,
            "deferred": len(outcome.deferred),
            "deferred_photo_ids": outcome.deferred,
        }


class RenewClaimRequest(BaseModel):
    batch_id: str
    ttl_minutes: int = 30


@router.post("/renew-claim")
def renew_claim(req: RenewClaimRequest):
    """Extend the TTL of an active claim (heartbeat)."""
    with _get_db() as db:
        if not db.renew_claim(req.batch_id, req.ttl_minutes):
            raise HTTPException(410, f"Claim {req.batch_id} expired or does not exist")
        return {"status": "ok", "batch_id": req.batch_id}


@router.post("/clear-claims")
def clear_claims():
    """Release every active worker claim immediately.

    Used when a worker fleet has crashed or hung and the photos it claimed
    should be reclaimable now rather than after the TTL expires. Live
    workers that submit afterward still have their results accepted —
    submit-results already tolerates expired/missing claims.
    """
    with _get_db() as db:
        cur = db.conn.execute("DELETE FROM worker_claims")
        cleared = cur.rowcount
        db.conn.commit()
        return {"status": "ok", "cleared": cleared}


class ClearPassRequest(BaseModel):
    pass_type: str
    collection_id: Optional[int] = None
    directory: Optional[str] = None
    filters: Optional[dict] = None  # structured filter set (_build_filter_sql)
    photo_ids: Optional[list[int]] = None


@router.post("/clear-pass")
def clear_pass(req: ClearPassRequest):
    """Clear processing state for a pass type, allowing re-processing.

    For faces: deletes face rows + worker_processed entries.
    For clip: deletes clip_embeddings rows.
    For quality/describe/tags: NULLs the relevant column.

    If collection_id is set, only affects photos in that collection.
    """
    with _get_db() as db:
        photo_ids = _resolve_scope_ids(
            db, req.collection_id, req.directory, req.filters)
        if photo_ids is None:
            if req.photo_ids is not None:
                photo_ids = req.photo_ids
                if not photo_ids:
                    raise HTTPException(400, "photo_ids list is empty")
            else:
                raise HTTPException(400, "collection_id, directory, filters, or photo_ids is required for clear-pass (safety)")

        placeholders = ",".join("?" * len(photo_ids))
        cleared = 0

        if req.pass_type == "faces":
            # Delete face encodings first (vec table)
            face_ids = [r[0] for r in db.conn.execute(
                f"SELECT id FROM faces WHERE photo_id IN ({placeholders})", photo_ids
            ).fetchall()]
            if face_ids:
                fp = ",".join("?" * len(face_ids))
                db.conn.execute(f"DELETE FROM face_encodings WHERE face_id IN ({fp})", face_ids)
            cur = db.conn.execute(
                f"DELETE FROM faces WHERE photo_id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            # Also clear worker_processed entries
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'faces' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "clip":
            cur = db.conn.execute(
                f"DELETE FROM clip_embeddings WHERE photo_id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
        elif req.pass_type == "quality":
            cur = db.conn.execute(
                f"UPDATE photos SET aesthetic_score = NULL, aesthetic_concepts = NULL, aesthetic_critique = NULL "
                f"WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
        elif req.pass_type == "describe":
            cur = db.conn.execute(
                f"UPDATE photos SET description = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            # Also clear worker_processed entries
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'describe' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "tags":
            cur = db.conn.execute(
                f"UPDATE photos SET tags = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            # Also clear worker_processed entries
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'tags' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "verify":
            cur = db.conn.execute(
                f"UPDATE photos SET verified_at = NULL, verification_status = NULL, "
                f"hallucination_flags = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
        elif req.pass_type == "category-content":
            cur = db.conn.execute(
                f"UPDATE photos SET categories = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'category-content' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "category-visual":
            cur = db.conn.execute(
                f"UPDATE photos SET visual_tags = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'category-visual' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "keywords":
            cur = db.conn.execute(
                f"UPDATE photos SET keywords = NULL WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'keywords' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        elif req.pass_type == "aesthetics":
            from .aesthetics import ALL_SUBATTRS, DIMENSIONS
            aes_cols = (
                ["aes_overall", "aes_overall_pct", "aes_technical_iqa",
                 "aes_overall_iqa", "aes_style", "aes_style_tags",
                 "aes_model", "aes_scored_at",
                 # Subject-aware quality (v27) — reset so a re-score doesn't
                 # leave stale subject data on a now-subjectless photo.
                 "subject_boxes", "aes_subject_overall", "aes_subject_overall_pct",
                 "aes_subject", "aes_subject_model", "aes_subject_scored_at"]
                + [f"aes_{s}" for s in ALL_SUBATTRS]
                + [f"aes_{d}" for d in DIMENSIONS]
            )
            set_clause = ", ".join(f"{c} = NULL" for c in aes_cols)
            cur = db.conn.execute(
                f"UPDATE photos SET {set_clause} WHERE id IN ({placeholders})", photo_ids
            )
            cleared = cur.rowcount
            db.conn.execute(
                f"DELETE FROM worker_processed WHERE pass_type = 'aesthetics' AND photo_id IN ({placeholders})",
                photo_ids,
            )
        else:
            raise HTTPException(400, f"Unknown pass type: {req.pass_type}")

        db.conn.commit()

        # Log the clear operation for the activity chart
        if cleared > 0:
            db.log_activity(req.pass_type, "clear", cleared)

        return {"status": "ok", "pass_type": req.pass_type, "cleared": cleared, "photo_count": len(photo_ids)}


@router.get("/photo-detail/{photo_id}")
def photo_detail(photo_id: int):
    """Get photo metadata + CLIP embedding for verify pass.

    Returns description, tags, and CLIP embedding so the worker can
    run hallucination verification without needing the full DB.
    """
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, f"Photo {photo_id} not found")

        # Fetch CLIP embedding if available
        clip_embedding = None
        try:
            row = db.conn.execute(
                "SELECT embedding FROM clip_embeddings WHERE photo_id = ?",
                (photo_id,),
            ).fetchone()
            if row:
                clip_embedding = list(_deserialize_float_list(row["embedding"], CLIP_DIMENSIONS))
        except Exception:
            pass  # sqlite-vec not loaded or no embedding

        return {
            "id": photo["id"],
            "description": photo.get("description"),
            "tags": photo.get("tags"),
            "clip_embedding": clip_embedding,
            "verified_at": photo.get("verified_at"),
            "verification_status": photo.get("verification_status"),
        }


# Note: the legacy "tags" pass (pre-v23 78-word vocab) is intentionally excluded.
# It was replaced by category-content (categories) / category-visual (visual_tags)
# / keywords, and no worker processes it anymore — its column was nulled at the
# v23 migration, so count_unprocessed("tags") is pinned at the full library size
# forever. Including it in queue_depth just showed a confusing dead counter.
_ALL_PASSES = ("clip", "faces", "quality", "describe",
               "category-content", "category-visual", "keywords", "verify",
               "aesthetics")


@router.get("/status")
def worker_status(
    collection_id: Optional[int] = None,
    directory: Optional[str] = None,
    passes: Optional[str] = None,
    filters: Optional[str] = None,  # JSON-encoded structured filter set
):
    """Show queue depth and active claims for the worker system.

    `passes` is an optional comma-separated list limiting which pass-type
    counts to compute — each count is a library-wide scan, so restricting
    to the passes the caller cares about is a significant speedup when
    many workers poll this endpoint concurrently.

    Expired claims are NOT swept here; claim-batch calls `get_claimed_photo_ids`
    which already sweeps. Keeping status read-only avoids a write lock that
    serializes concurrent status polls.
    """
    if passes:
        requested = tuple(p.strip() for p in passes.split(",") if p.strip() in _ALL_PASSES)
        if not requested:
            requested = _ALL_PASSES
    else:
        requested = _ALL_PASSES

    filters_obj = None
    if filters:
        try:
            filters_obj = json.loads(filters)
        except (ValueError, TypeError):
            raise HTTPException(400, "filters must be a JSON object")

    with _get_db() as db:
        # Active claims (read-only; no expire sweep)
        claims = db.conn.execute(
            "SELECT pass_type, worker_id, batch_id, photo_ids, claimed_at, expires_at "
            "FROM worker_claims WHERE expires_at > datetime('now')"
        ).fetchall()

        active = []
        for c in claims:
            ids = json.loads(c["photo_ids"])
            active.append({
                "batch_id": c["batch_id"],
                "worker_id": c["worker_id"],
                "pass_type": c["pass_type"],
                "photo_count": len(ids),
                "claimed_at": c["claimed_at"],
                "expires_at": c["expires_at"],
            })

        # Queue depth per requested pass type — count of photos missing each pass
        scope_ids = _resolve_scope_ids(db, collection_id, directory, filters_obj)

        queue = {}
        for pass_type in requested:
            queue[pass_type] = _count_scoped(db, pass_type, scope_ids)

        return {
            "active_claims": active,
            "queue_depth": queue,
            "collection_id": collection_id,
            "directory": directory,
            "filters": filters_obj,
        }

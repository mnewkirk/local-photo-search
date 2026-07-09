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
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/claim-batch")
def claim_batch(req: ClaimRequest):
    """Claim a batch of unprocessed photos for a given pass type.

    Returns photo metadata + download URLs. The claim expires after ttl_minutes.
    """
    with _get_db() as db:
        # Scope to collection or directory if requested
        scope_ids = None
        if req.collection_id is not None:
            scope_ids = db.get_collection_photo_ids(req.collection_id)
            if not scope_ids:
                raise HTTPException(404, f"Collection {req.collection_id} has no photos")
        elif req.directory is not None:
            scope_ids = db.get_directory_photo_ids(req.directory)
            if not scope_ids:
                raise HTTPException(404, f"No photos found in directory {req.directory}")

        # BEGIN IMMEDIATE acquires the SQLite writer lock up front so concurrent
        # claim-batch calls serialize here instead of racing through the
        # read-then-insert window. Without this, N parallel workers could each
        # SELECT the same unprocessed photos, then each INSERT a claim row
        # containing the same photo_ids — every worker would download and
        # re-describe the same images. Cleanup commits inside the read are
        # suppressed via commit_cleanup/commit kwargs so they don't end our
        # outer transaction early.
        db.conn.execute("BEGIN IMMEDIATE")
        try:
            photos = db.get_unprocessed_photos(
                pass_type=req.pass_type,
                photo_ids=scope_ids,
                limit=req.limit,
                commit_cleanup=False,
            )

            if not photos:
                db.conn.commit()
                return JSONResponse({"batch_id": None, "pass_type": req.pass_type, "photos": []})

            photo_ids = [p["id"] for p in photos]
            batch_id = db.claim_photos(
                worker_id=req.worker_id,
                pass_type=req.pass_type,
                photo_ids=photo_ids,
                ttl_minutes=req.ttl_minutes,
                commit=False,
            )
            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise

        # Count how many remain unclaimed after this batch. Uses a COUNT(*)
        # rather than materializing every unprocessed row — with N concurrent
        # workers a full-queue scan per claim turned into ReadTimeout cascades
        # on large libraries. The count doesn't subtract active claims, so the
        # value is slightly inflated, but it's purely informational (worker
        # logs it; no decision is made on it).
        remaining = db.count_unprocessed_photos(
            pass_type=req.pass_type,
            photo_ids=scope_ids,
        )

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


@router.post("/submit-results")
def submit_results(req: SubmitRequest):
    """Submit processing results for a claimed batch.

    Results are written directly to the main DB.
    The claim is released after successful write.
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

        written = 0
        processed_photo_ids = []

        if req.pass_type == "clip" and req.clip_results:
            db.begin_batch(batch_size=100)
            for r in req.clip_results:
                try:
                    db.add_clip_embedding(r.photo_id, r.embedding)
                    written += 1
                    processed_photo_ids.append(r.photo_id)
                except Exception as e:
                    logger.warning(f"Failed to store CLIP for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("clip", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()

        elif req.pass_type == "faces":
            face_results = req.face_results or []
            db.begin_batch(batch_size=50)
            for r in face_results:
                processed_photo_ids.append(r.photo_id)
                for face in r.faces:
                    try:
                        db.add_face(
                            photo_id=r.photo_id,
                            bbox=tuple(face["bbox"]),
                            encoding=face["encoding"],
                            det_score=face.get("det_score"),
                        )
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store face for photo {r.photo_id}: {e}")
                        try:
                            db.log_error("faces", str(r.photo_id), str(e))
                        except Exception:
                            pass
            db.end_batch()

            # New faces land with cluster_id=NULL. Global clustering is a
            # separate, on-demand step via `photosearch recluster-faces` —
            # per-batch clustering would collide IDs across batches and
            # fragment the same person across many pseudo-clusters.

            # Mark all submitted photos as processed (including those with no faces)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "faces")

        elif req.pass_type == "quality" and req.quality_results:
            db.begin_batch(batch_size=100)
            for r in req.quality_results:
                try:
                    updates = {"aesthetic_score": r.aesthetic_score}
                    if r.aesthetic_concepts:
                        updates["aesthetic_concepts"] = r.aesthetic_concepts
                    db.update_photo(r.photo_id, **updates)
                    written += 1
                    processed_photo_ids.append(r.photo_id)
                except Exception as e:
                    logger.warning(f"Failed to store quality for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("quality", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()

        elif req.pass_type == "describe":
            describe_results = req.describe_results or []
            db.begin_batch(batch_size=100)
            for r in describe_results:
                processed_photo_ids.append(r.photo_id)
                if r.description:
                    try:
                        db.update_photo(r.photo_id, description=r.description)
                        db.log_generation(r.photo_id, "describe", r.description,
                                          req.model, req.model_version)
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store description for photo {r.photo_id}: {e}")
                        try:
                            db.log_error("describe", str(r.photo_id), str(e))
                        except Exception:
                            pass
            db.end_batch()
            # Mark all submitted photos as processed (including those with no description)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "describe")

        elif req.pass_type == "tags":
            tags_results = req.tags_results or []
            db.begin_batch(batch_size=100)
            for r in tags_results:
                processed_photo_ids.append(r.photo_id)
                if r.tags:
                    try:
                        tags_json = json.dumps(r.tags)
                        db.update_photo(r.photo_id, tags=tags_json)
                        db.log_generation(r.photo_id, "tags", tags_json,
                                          req.model, req.model_version)
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store tags for photo {r.photo_id}: {e}")
                        try:
                            db.log_error("tags", str(r.photo_id), str(e))
                        except Exception:
                            pass
            db.end_batch()
            # Mark all submitted photos as processed (including those with no tags)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "tags")

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
                    written += 1
                    processed_photo_ids.append(r.photo_id)
                except Exception as e:
                    logger.warning(f"Failed to store verify for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("verify", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()

        elif req.pass_type == "category-content":
            category_content_results = req.category_content_results or []
            db.begin_batch(batch_size=100)
            for r in category_content_results:
                processed_photo_ids.append(r.photo_id)
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
                    written += 1
                except Exception as e:
                    logger.warning(f"Failed to store category-content for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("category-content", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()
            # Mark all submitted photos as processed (including those with no categories)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "category-content")

        elif req.pass_type == "category-visual":
            category_visual_results = req.category_visual_results or []
            db.begin_batch(batch_size=100)
            for r in category_visual_results:
                processed_photo_ids.append(r.photo_id)
                try:
                    # Always persist the column ('[]' for empty) so a successful
                    # empty result marks done in one pass; only timeouts defer.
                    vtags_json = json.dumps(r.visual_tags or [])
                    db.update_photo(r.photo_id, visual_tags=vtags_json)
                    if r.visual_tags:
                        db.log_generation(r.photo_id, "category-visual", vtags_json,
                                          r.model, r.model_version)
                    written += 1
                except Exception as e:
                    logger.warning(f"Failed to store category-visual for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("category-visual", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()
            # Mark all submitted photos as processed (including those with no visual tags)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "category-visual")

        elif req.pass_type == "keywords":
            keywords_results = req.keywords_results or []
            db.begin_batch(batch_size=100)
            for r in keywords_results:
                processed_photo_ids.append(r.photo_id)
                try:
                    # Always persist the column ('[]' for empty) so a successful
                    # empty result marks done in one pass; only timeouts defer.
                    kw_json = json.dumps(r.keywords or [])
                    db.update_photo(r.photo_id, keywords=kw_json)
                    if r.keywords:
                        db.log_generation(r.photo_id, "keywords", kw_json,
                                          r.model, r.model_version)
                    written += 1
                except Exception as e:
                    logger.warning(f"Failed to store keywords for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("keywords", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()
            # Mark all submitted photos as processed (including those with no keywords)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "keywords")

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
                processed_photo_ids.append(r.photo_id)
                try:
                    fields = {k: v for k, v in (r.scores or {}).items()
                              if k in allowed_score_cols and v is not None}
                    if not fields or "aes_overall" not in fields:
                        # Nothing usable — skip (worker already omits deferrals,
                        # so this is a defensive guard, not the normal path).
                        continue
                    fields["aes_style"] = r.aes_style
                    fields["aes_style_tags"] = r.aes_style_tags
                    fields["aes_model"] = r.model
                    fields["aes_scored_at"] = now
                    db.update_photo(r.photo_id, **fields)
                    db.log_generation(
                        r.photo_id, "aesthetics",
                        json.dumps({"scores": fields.get("aes_overall"),
                                    "style": r.aes_style,
                                    "tags": r.aes_style_tags}),
                        r.model, r.model_version)
                    written += 1
                except Exception as e:
                    logger.warning(f"Failed to store aesthetics for photo {r.photo_id}: {e}")
                    try:
                        db.log_error("aesthetics", str(r.photo_id), str(e))
                    except Exception:
                        pass
            db.end_batch()
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "aesthetics")

        # Log activity for the chart
        if written > 0:
            db.log_activity(req.pass_type, "index", written)

        # Release the claim
        db.release_claim(req.batch_id)

        return {
            "status": "ok",
            "written": written,
            "processed": len(processed_photo_ids),
            "batch_id": req.batch_id,
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
        if req.collection_id is not None:
            photo_ids = db.get_collection_photo_ids(req.collection_id)
            if not photo_ids:
                raise HTTPException(404, f"Collection {req.collection_id} has no photos")
        elif req.directory is not None:
            photo_ids = db.get_directory_photo_ids(req.directory)
            if not photo_ids:
                raise HTTPException(404, f"No photos found in directory {req.directory}")
        elif req.photo_ids is not None:
            photo_ids = req.photo_ids
            if not photo_ids:
                raise HTTPException(400, "photo_ids list is empty")
        else:
            raise HTTPException(400, "collection_id, directory, or photo_ids is required for clear-pass (safety)")

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
                 "aes_model", "aes_scored_at"]
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
        scope_ids = None
        if collection_id is not None:
            scope_ids = db.get_collection_photo_ids(collection_id)
        elif directory is not None:
            scope_ids = db.get_directory_photo_ids(directory)

        queue = {}
        for pass_type in requested:
            queue[pass_type] = db.count_unprocessed_photos(pass_type, photo_ids=scope_ids)

        return {
            "active_claims": active,
            "queue_depth": queue,
            "collection_id": collection_id,
            "directory": directory,
        }

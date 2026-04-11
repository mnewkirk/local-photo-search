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


def configure(db_path: str, photo_root: Optional[str] = None):
    """Called by web.py to pass DB config to the worker router."""
    global _db_path, _photo_root
    _db_path = db_path
    _photo_root = photo_root


def _get_db() -> PhotoDB:
    return PhotoDB(_db_path, photo_root=_photo_root)


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
    photos: list[dict]  # [{id, filepath, filename}]
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


class SubmitRequest(BaseModel):
    batch_id: str
    pass_type: str
    clip_results: Optional[list[ClipResult]] = None
    face_results: Optional[list[FaceResult]] = None
    quality_results: Optional[list[QualityResult]] = None
    describe_results: Optional[list[DescribeResult]] = None
    tags_results: Optional[list[TagsResult]] = None
    verify_results: Optional[list[VerifyResult]] = None


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

        photos = db.get_unprocessed_photos(
            pass_type=req.pass_type,
            photo_ids=scope_ids,
            limit=req.limit,
        )

        if not photos:
            return JSONResponse({"batch_id": None, "pass_type": req.pass_type, "photos": []})

        photo_ids = [p["id"] for p in photos]
        batch_id = db.claim_photos(
            worker_id=req.worker_id,
            pass_type=req.pass_type,
            photo_ids=photo_ids,
            ttl_minutes=req.ttl_minutes,
        )

        # Count how many remain unclaimed after this batch
        remaining_photos = db.get_unprocessed_photos(
            pass_type=req.pass_type,
            photo_ids=scope_ids,
            limit=999999,
        )
        remaining = len(remaining_photos)

        result_photos = []
        for p in photos:
            result_photos.append({
                "id": p["id"],
                "filepath": p["filepath"],
                "filename": Path(p["filepath"]).name,
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
        # Verify the claim exists and hasn't expired
        row = db.conn.execute(
            "SELECT * FROM worker_claims WHERE batch_id = ? AND expires_at > datetime('now')",
            (req.batch_id,),
        ).fetchone()
        if not row:
            raise HTTPException(410, f"Claim {req.batch_id} expired or does not exist")

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
            db.end_batch()

        elif req.pass_type == "faces":
            face_results = req.face_results or []
            all_encodings = []
            all_face_ids = []
            db.begin_batch(batch_size=50)
            for r in face_results:
                processed_photo_ids.append(r.photo_id)
                for face in r.faces:
                    try:
                        face_id = db.add_face(
                            photo_id=r.photo_id,
                            bbox=tuple(face["bbox"]),
                            encoding=face["encoding"],
                        )
                        all_encodings.append(face["encoding"])
                        all_face_ids.append(face_id)
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store face for photo {r.photo_id}: {e}")
            db.end_batch()

            # Cluster the new faces
            if all_encodings:
                from .faces import cluster_encodings
                cluster_ids = cluster_encodings(all_encodings)
                db.begin_batch(batch_size=200)
                for face_id, cluster_id in zip(all_face_ids, cluster_ids):
                    db.conn.execute(
                        "UPDATE faces SET cluster_id = ? WHERE id = ?",
                        (cluster_id, face_id),
                    )
                db.end_batch()

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
            db.end_batch()

        elif req.pass_type == "describe":
            describe_results = req.describe_results or []
            for r in describe_results:
                processed_photo_ids.append(r.photo_id)
                if r.description:
                    try:
                        db.update_photo(r.photo_id, description=r.description)
                        db.conn.commit()
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store description for photo {r.photo_id}: {e}")
            # Mark all submitted photos as processed (including those with no description)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "describe")

        elif req.pass_type == "tags":
            tags_results = req.tags_results or []
            for r in tags_results:
                processed_photo_ids.append(r.photo_id)
                if r.tags:
                    try:
                        db.update_photo(r.photo_id, tags=json.dumps(r.tags))
                        db.conn.commit()
                        written += 1
                    except Exception as e:
                        logger.warning(f"Failed to store tags for photo {r.photo_id}: {e}")
            # Mark all submitted photos as processed (including those with no tags)
            if processed_photo_ids:
                db.mark_processed(processed_photo_ids, "tags")

        elif req.pass_type == "verify" and req.verify_results:
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
                    db.conn.commit()
                    written += 1
                    processed_photo_ids.append(r.photo_id)
                except Exception as e:
                    logger.warning(f"Failed to store verify for photo {r.photo_id}: {e}")

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


class ClearPassRequest(BaseModel):
    pass_type: str
    collection_id: Optional[int] = None
    directory: Optional[str] = None


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
        else:
            raise HTTPException(400, "collection_id or directory is required for clear-pass (safety)")

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
        else:
            raise HTTPException(400, f"Unknown pass type: {req.pass_type}")

        db.conn.commit()
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


@router.get("/status")
def worker_status(collection_id: Optional[int] = None, directory: Optional[str] = None):
    """Show queue depth and active claims for the worker system."""
    with _get_db() as db:
        db.expire_worker_claims()

        # Active claims
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

        # Queue depth per pass type — count of photos missing each pass
        scope_ids = None
        if collection_id is not None:
            scope_ids = db.get_collection_photo_ids(collection_id)
        elif directory is not None:
            scope_ids = db.get_directory_photo_ids(directory)

        queue = {}
        for pass_type in ("clip", "faces", "quality", "describe", "tags", "verify"):
            unprocessed = db.get_unprocessed_photos(pass_type, photo_ids=scope_ids, limit=100000)
            queue[pass_type] = len(unprocessed)

        return {
            "active_claims": active,
            "queue_depth": queue,
            "collection_id": collection_id,
            "directory": directory,
        }

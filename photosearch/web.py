"""Web UI for local-photo-search.

FastAPI application serving a photo search interface with:
  - Search API endpoints (semantic, person, color, combined)
  - Thumbnail generation and serving
  - Full-resolution photo serving
  - Person and stats endpoints

Launch with:
  python cli.py serve
  # or directly:
  uvicorn photosearch.web:app --reload
"""

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

# Request logging — helps debug discrepancies between CLI and web results
logger = logging.getLogger("photosearch.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from fastapi import FastAPI, Query, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .db import PhotoDB
from .worker_api import router as worker_router, configure as configure_worker

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="local-photo-search", version="0.1.0")
app.include_router(worker_router)

# Allow the React dev server (port 5173) during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database path — set by the CLI launcher, defaults to cwd
_db_path: str = os.environ.get("PHOTOSEARCH_DB", "photo_index.db")
_photo_root: Optional[str] = os.environ.get("PHOTO_ROOT")

# Configure the worker API with DB settings
configure_worker(_db_path, _photo_root)

# Thumbnail cache directory
_thumb_dir: Optional[str] = None
_THUMB_SIZE = 600  # px, long edge

# Preview cache directory (mid-quality: larger than thumbnail, smaller than full)
_preview_dir: Optional[str] = None
_PREVIEW_SIZE = 1920  # px, long edge


def _get_db() -> PhotoDB:
    """Open a fresh DB connection per request."""
    return PhotoDB(_db_path, photo_root=_photo_root)


def _ensure_thumb_dir():
    """Create thumbnail cache directory if needed."""
    global _thumb_dir
    if _thumb_dir is None:
        db_parent = Path(_db_path).resolve().parent
        _thumb_dir = str(db_parent / "thumbnails")
    Path(_thumb_dir).mkdir(parents=True, exist_ok=True)
    return _thumb_dir


def _ensure_preview_dir():
    """Create preview cache directory if needed."""
    global _preview_dir
    if _preview_dir is None:
        db_parent = Path(_db_path).resolve().parent
        _preview_dir = str(db_parent / "previews")
    Path(_preview_dir).mkdir(parents=True, exist_ok=True)
    return _preview_dir


_face_crop_dir: Optional[str] = None


def _ensure_face_crop_dir():
    """Create face-crop cache directory if needed."""
    global _face_crop_dir
    if _face_crop_dir is None:
        db_parent = Path(_db_path).resolve().parent
        _face_crop_dir = str(db_parent / "thumbnails" / "face_crops")
    Path(_face_crop_dir).mkdir(parents=True, exist_ok=True)
    return _face_crop_dir


def _get_or_create_preview(photo: dict) -> str:
    """Return path to a cached mid-quality preview, generating it if needed.

    Previews sit between thumbnails (600px) and full-res.  They are sized
    to ~1920px on the long edge — large enough for crisp detail-view display
    on any current screen, but typically 5–15× smaller than the original.
    """
    from PIL import Image, ImageOps

    preview_dir = _ensure_preview_dir()
    photo_id = photo["id"]
    preview_path = os.path.join(preview_dir, f"{photo_id}_preview.jpg")

    if os.path.exists(preview_path):
        return preview_path

    filepath = photo.get("_resolved_filepath") or photo.get("filepath", "")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Original not found: {filepath}")

    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((_PREVIEW_SIZE, _PREVIEW_SIZE), Image.LANCZOS)
    img.save(preview_path, "JPEG", quality=82, optimize=True)
    return preview_path


def _get_or_create_thumbnail(photo: dict) -> str:
    """Return path to a cached thumbnail, generating it if needed.

    Thumbnails are keyed by photo ID (not filename) to avoid collisions
    when the same filename exists in different directories.
    """
    from PIL import Image

    thumb_dir = _ensure_thumb_dir()
    photo_id = photo["id"]
    thumb_path = os.path.join(thumb_dir, f"{photo_id}_thumb.jpg")

    if os.path.exists(thumb_path):
        return thumb_path

    filepath = photo.get("_resolved_filepath") or photo.get("filepath", "")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Original not found: {filepath}")

    from PIL import ImageOps
    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((_THUMB_SIZE, _THUMB_SIZE), Image.LANCZOS)
    img.save(thumb_path, "JPEG", quality=85)
    return thumb_path


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/search")
def api_search(
    q: Optional[str] = Query(None, description="Semantic search query"),
    person: Optional[str] = Query(None, description="Person name"),
    color: Optional[str] = Query(None, description="Color name or hex"),
    place: Optional[str] = Query(None, description="Place name"),
    limit: int = Query(1000, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    min_score: float = Query(-0.25, description="Minimum CLIP score"),
    min_quality: Optional[float] = Query(None, description="Minimum aesthetic quality (1-10)"),
    sort: str = Query("date_desc", description="Sort order: date_desc, date_asc, quality_desc, relevance"),
    sort_quality: bool = Query(False, description="Legacy: equivalent to sort=quality_desc"),
    tag_match: str = Query("both", description="Tag matching mode: dict, tags, or both"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    location: Optional[str] = Query(None, description="Filter by location name"),
    match_source: Optional[str] = Query(None, description="Face match type: strict, temporal, or manual"),
):
    """Search photos using any combination of criteria."""
    logger.info(
        "SEARCH REQUEST  q=%r  person=%r  color=%r  place=%r  limit=%d  "
        "min_score=%.3f  min_quality=%s  sort_quality=%s  date_from=%s  date_to=%s  location=%r",
        q, person, color, place, limit, min_score, min_quality, sort_quality,
        date_from, date_to, location,
    )

    if not any([q, person, color, place, min_quality is not None,
                date_from, date_to, location]):
        logger.info("SEARCH REJECTED — no criteria provided")
        return {"results": [], "count": 0, "error": "Provide at least one search criterion"}

    from .search import search_combined

    # Validate / default the sort param. Unknown values fall back to
    # date_desc so broken clients still get sensible pages.
    from .search import SORT_MODES
    if sort not in SORT_MODES:
        sort = "date_desc"

    with _get_db() as db:
        results, total = search_combined(
            db=db,
            query=q,
            person=person,
            color=color,
            place=place,
            limit=limit,
            offset=offset,
            with_total=True,
            sort=sort,
            min_score=min_score,
            min_quality=min_quality,
            sort_quality=sort_quality,
            tag_match=tag_match,
            date_from=date_from,
            date_to=date_to,
            location=location,
            match_source=match_source,
        )

        logger.info(
            "SEARCH RESULTS  count=%d  total=%d  offset=%d  sort=%s  "
            "top_scores=%s",
            len(results), total, offset, sort,
            [(r.get("filename"), round(r.get("rrf_score") or r.get("score", 0), 4))
             for r in results[:5]],
        )

        # Serialize results
        items = []
        for r in results:
            item = {
                "id": r.get("id"),
                "filename": r.get("filename"),
                "date_taken": r.get("date_taken"),
                "score": r.get("score"),
                "clip_score": r.get("clip_score"),
                "aesthetic_score": r.get("aesthetic_score"),
                "description": r.get("description"),
                "camera_model": r.get("camera_model"),
                "focal_length": r.get("focal_length"),
                "exposure_time": r.get("exposure_time"),
                "f_number": r.get("f_number"),
                "iso": r.get("iso"),
                "image_width": r.get("image_width"),
                "image_height": r.get("image_height"),
                "tags": json.loads(r["tags"]) if r.get("tags") else [],
                "place_name": r.get("place_name"),
            }
            if r.get("dominant_colors"):
                try:
                    item["colors"] = json.loads(r["dominant_colors"])
                except (json.JSONDecodeError, TypeError):
                    item["colors"] = []
            else:
                item["colors"] = []

            # Stack info (if photo belongs to a stack)
            pid = r.get("id") or r.get("photo_id")
            stack_info = db.get_photo_stack(pid) if pid else None
            if stack_info:
                item["stack_id"] = stack_info["stack_id"]
                item["stack_is_top"] = stack_info["is_top"]
                item["stack_count"] = stack_info["member_count"]

            items.append(item)

    return {
        "results": items,
        "count": len(items),
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(items)) < total,
    }


@app.get("/api/photos/{photo_id}/thumbnail")
def api_thumbnail(photo_id: int):
    """Serve a cached thumbnail for a photo."""
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        photo = dict(photo)
        photo["_resolved_filepath"] = db.resolve_filepath(photo.get("filepath", ""))

    try:
        thumb_path = _get_or_create_thumbnail(photo)
        return FileResponse(
            thumb_path, media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )
    except FileNotFoundError:
        raise HTTPException(404, "Original photo file not found")
    except Exception as e:
        raise HTTPException(500, f"Thumbnail error: {e}")


@app.get("/api/photos/{photo_id}/full")
def api_full_photo(photo_id: int):
    """Serve the full-resolution original photo."""
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")

        filepath = db.resolve_filepath(photo.get("filepath", ""))

    if not filepath or not os.path.exists(filepath):
        raise HTTPException(404, "Photo file not found on disk")

    return FileResponse(
        filepath, media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/api/photos/{photo_id}/preview")
def api_preview_photo(photo_id: int):
    """Serve a mid-quality preview (~1920px long edge, JPEG 82).

    Faster than /full for detail-view display — generated once and cached
    alongside the thumbnail cache.  Falls back to /full behaviour if the
    photo cannot be resized (e.g. non-JPEG original that Pillow can't open).
    """
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        photo = dict(photo)
        photo["_resolved_filepath"] = db.resolve_filepath(photo.get("filepath", ""))

    try:
        preview_path = _get_or_create_preview(photo)
        return FileResponse(
            preview_path, media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )
    except FileNotFoundError:
        raise HTTPException(404, "Original photo file not found")
    except Exception:
        # If preview generation fails (e.g. unsupported format), fall back to full
        filepath = photo.get("_resolved_filepath", "")
        if filepath and os.path.exists(filepath):
            return FileResponse(
                filepath, media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, must-revalidate"},
            )
        raise HTTPException(500, "Could not serve preview")


@app.get("/api/faces/crop/{face_id}")
def api_face_crop(face_id: int, size: int = Query(200, ge=50, le=800)):
    """Serve a square crop of the face from the original photo.

    Cached to disk at thumbnails/face_crops/{face_id}_{size}.jpg — face bbox
    and source photo are immutable per face_id, so the cache never needs to
    be invalidated. Generating from the original photo on every request was
    the dominant cost of /faces (RAW/JPEG decode + exif_transpose + resize).
    """
    cache_dir = _ensure_face_crop_dir()
    cache_path = os.path.join(cache_dir, f"{face_id}_{size}.jpg")
    if os.path.exists(cache_path):
        return FileResponse(
            cache_path, media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    from PIL import Image, ImageOps

    with _get_db() as db:
        row = db.conn.execute(
            """SELECT f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                      ph.filepath
               FROM faces f
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.id = ?""",
            (face_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Face not found")
        filepath = db.resolve_filepath(row["filepath"])

    if not filepath or not os.path.exists(filepath):
        raise HTTPException(404, "Photo file not found")

    if row["bbox_top"] is None:
        raise HTTPException(400, "Face has no bounding box")

    top, right, bottom, left = row["bbox_top"], row["bbox_right"], row["bbox_bottom"], row["bbox_left"]

    img = Image.open(filepath)
    img = ImageOps.exif_transpose(img)
    img_w, img_h = img.size

    # Add 20% padding around the face for tight framing
    face_w = right - left
    face_h = bottom - top
    pad_x = int(face_w * 0.2)
    pad_y = int(face_h * 0.2)
    crop_left = max(0, left - pad_x)
    crop_top = max(0, top - pad_y)
    crop_right = min(img_w, right + pad_x)
    crop_bottom = min(img_h, bottom + pad_y)

    # Make it square (expand the shorter dimension, centered)
    cw = crop_right - crop_left
    ch = crop_bottom - crop_top
    if cw > ch:
        diff = cw - ch
        crop_top = max(0, crop_top - diff // 2)
        crop_bottom = crop_top + cw
        if crop_bottom > img_h:
            crop_bottom = img_h
            crop_top = max(0, crop_bottom - cw)
    elif ch > cw:
        diff = ch - cw
        crop_left = max(0, crop_left - diff // 2)
        crop_right = crop_left + ch
        if crop_right > img_w:
            crop_right = img_w
            crop_left = max(0, crop_right - ch)

    face_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    face_img = face_img.resize((size, size), Image.LANCZOS)

    # Write atomically — save to a temp path in the same dir, then rename.
    tmp_path = cache_path + f".tmp.{os.getpid()}"
    face_img.save(tmp_path, format="JPEG", quality=85)
    os.replace(tmp_path, cache_path)

    return FileResponse(
        cache_path, media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _similarity_sort(groups: list[dict], encodings: dict[int, list[float]]) -> list[dict]:
    """Sort groups so similar faces are adjacent.

    Strategy: named persons first (alpha), then unknowns sorted by photo_count
    desc but with similar faces pulled next to each other. Uses numpy for fast
    distance matrix computation and greedy nearest-neighbor chaining.
    """
    named = [g for g in groups if g["type"] == "person"]
    named.sort(key=lambda g: g["label"].lower())

    unknowns = [g for g in groups if g["type"] != "person"]
    # Sort unknowns by photo_count desc as baseline
    unknowns.sort(key=lambda g: -g["photo_count"])

    # Build encoding lookup for unknowns that have rep faces with encodings
    has_enc = []  # indices into unknowns that have encodings
    enc_vecs = []  # corresponding encoding vectors
    for i, g in enumerate(unknowns):
        fid = g.get("rep_face_id")
        if fid and fid in encodings:
            has_enc.append(i)
            enc_vecs.append(encodings[fid])

    if len(has_enc) < 2:
        return named + unknowns

    # Compute pairwise distance matrix with numpy (fast)
    mat = np.array(enc_vecs, dtype=np.float32)  # shape (N, D)
    # Euclidean distance matrix: ||a-b||² = ||a||² + ||b||² - 2*a·b
    sq_norms = np.sum(mat ** 2, axis=1)
    dist_matrix = np.sqrt(
        np.maximum(sq_norms[:, None] + sq_norms[None, :] - 2 * mat @ mat.T, 0.0)
    )

    # Map from unknown-index to enc-index
    unk_to_enc = {unk_i: enc_i for enc_i, unk_i in enumerate(has_enc)}

    # Greedy nearest-neighbor chain starting from the largest group
    n_enc = len(has_enc)
    enc_visited = np.zeros(n_enc, dtype=bool)  # tracks visited enc indices
    visited = set()
    ordered = []
    current = 0  # Start with the first unknown (largest by photo count)
    visited.add(current)
    ordered.append(unknowns[current])
    if current in unk_to_enc:
        enc_visited[unk_to_enc[current]] = True

    while len(visited) < len(unknowns):
        best_idx = None

        if current in unk_to_enc:
            cur_enc_i = unk_to_enc[current]
            dists = dist_matrix[cur_enc_i].copy()
            dists[enc_visited] = np.inf  # mask all visited in one op
            min_enc_i = int(np.argmin(dists))
            if dists[min_enc_i] < np.inf:
                best_idx = has_enc[min_enc_i]

        if best_idx is None:
            for j in range(len(unknowns)):
                if j not in visited:
                    best_idx = j
                    break

        if best_idx is None:
            break

        visited.add(best_idx)
        ordered.append(unknowns[best_idx])
        if best_idx in unk_to_enc:
            enc_visited[unk_to_enc[best_idx]] = True
        current = best_idx

    return named + ordered


# Above this group count, similarity sort is O(N²) and dominates response time.
# Count-sort is used instead unless the caller explicitly asks for similarity.
_SIMILARITY_SORT_GROUP_LIMIT = 500


@app.get("/api/faces/groups")
def api_face_groups(
    sort: str = Query("similarity"),
    include_singletons: bool = Query(False),
    filter: str = Query("all"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
):
    """List face identities grouped by person or cluster, with filtering and pagination.

    sort: "similarity" clusters visually-similar faces together (O(N²);
          auto-downgraded to "count" when the filtered group count exceeds
          _SIMILARITY_SORT_GROUP_LIMIT). "count" sorts unknowns by face_count desc.
    include_singletons: when false (default), unknown clusters with only one
          face are hidden.
    filter: "all" (named + not-ignored clusters), "named" (persons only),
          "unknown" (not-ignored clusters only), or "ignored" (ignored clusters only).
    limit/offset: pagination over the sorted list. Counts and `total` always
          reflect the pre-pagination filtered set.
    """
    import time
    t0 = time.time()

    if filter not in ("all", "named", "unknown", "ignored"):
        filter = "all"

    with _get_db() as db:
        min_face_count = 1 if include_singletons else 2

        # Fetch ignored cluster IDs once — used for both filtering and annotation.
        ignored_set = set(
            r["cluster_id"]
            for r in db.conn.execute("SELECT cluster_id FROM ignored_clusters").fetchall()
        )

        named_groups: list[dict] = []
        if filter in ("all", "named"):
            named_rows = db.conn.execute(
                """SELECT p.id as person_id, p.name,
                          COUNT(DISTINCT f.photo_id) as photo_count,
                          COUNT(f.id) as face_count,
                          (SELECT f2.id FROM faces f2
                           WHERE f2.person_id = p.id AND f2.bbox_top IS NOT NULL
                           ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                           LIMIT 1) as rep_face_id
                   FROM persons p
                   JOIN faces f ON f.person_id = p.id
                   GROUP BY p.id
                   ORDER BY p.name"""
            ).fetchall()
            for r in named_rows:
                named_groups.append({
                    "type": "person",
                    "person_id": r["person_id"],
                    "label": r["name"],
                    "photo_count": r["photo_count"],
                    "face_count": r["face_count"],
                    "rep_face_id": r["rep_face_id"],
                })

        cluster_groups: list[dict] = []
        if filter in ("all", "unknown", "ignored"):
            cluster_rows = db.conn.execute(
                """SELECT f.cluster_id,
                          COUNT(DISTINCT f.photo_id) as photo_count,
                          COUNT(f.id) as face_count,
                          (SELECT f2.id FROM faces f2
                           WHERE f2.cluster_id = f.cluster_id AND f2.person_id IS NULL
                                 AND f2.bbox_top IS NOT NULL
                           ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                           LIMIT 1) as rep_face_id
                   FROM faces f
                   WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL
                   GROUP BY f.cluster_id
                   HAVING face_count >= ?
                   ORDER BY face_count DESC""",
                (min_face_count,),
            ).fetchall()
            for r in cluster_rows:
                is_ignored = r["cluster_id"] in ignored_set
                if filter == "unknown" and is_ignored:
                    continue
                if filter == "ignored" and not is_ignored:
                    continue
                if filter == "all" and is_ignored:
                    # "all" matches the frontend's pre-pagination default of hiding ignored.
                    continue
                cluster_groups.append({
                    "type": "cluster",
                    "cluster_id": r["cluster_id"],
                    "label": "Unknown #" + str(r["cluster_id"]),
                    "photo_count": r["photo_count"],
                    "face_count": r["face_count"],
                    "rep_face_id": r["rep_face_id"],
                    "ignored": is_ignored,
                })

        groups = named_groups + cluster_groups
        total = len(groups)
        t1 = time.time()
        logger.info("faces/groups: query took %.3fs (filter=%s, %d groups)",
                    t1 - t0, filter, total)

        # Counts over the full (pre-pagination) state so filter chips stay accurate.
        named_count = db.conn.execute(
            """SELECT COUNT(*) AS n FROM persons p
               WHERE EXISTS (SELECT 1 FROM faces f WHERE f.person_id = p.id)"""
        ).fetchone()["n"]

        cluster_size_rows = db.conn.execute(
            """SELECT f.cluster_id
               FROM faces f
               WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL
               GROUP BY f.cluster_id
               HAVING COUNT(f.id) >= ?""",
            (min_face_count,),
        ).fetchall()
        qualifying_cluster_ids = {r["cluster_id"] for r in cluster_size_rows}
        ignored_qualifying = qualifying_cluster_ids & ignored_set
        counts = {
            "named": named_count,
            "unknown": len(qualifying_cluster_ids) - len(ignored_qualifying),
            "ignored": len(ignored_qualifying),
        }

        effective_sort = sort
        if sort == "similarity" and total > _SIMILARITY_SORT_GROUP_LIMIT:
            logger.info(
                "faces/groups: downgrading similarity sort to count "
                "(%d groups > %d limit)",
                total, _SIMILARITY_SORT_GROUP_LIMIT,
            )
            effective_sort = "count"

        if effective_sort == "similarity":
            rep_face_ids = [g["rep_face_id"] for g in groups if g["rep_face_id"]]
            encodings = db.get_face_encodings_bulk(rep_face_ids)
            t2 = time.time()
            logger.info("faces/groups: encoding fetch took %.3fs (%d encodings)", t2 - t1, len(encodings))
            groups = _similarity_sort(groups, encodings)
            t3 = time.time()
            logger.info("faces/groups: similarity sort took %.3fs", t3 - t2)

        page = groups[offset : offset + limit]

    return {
        "groups": page,
        "total": total,
        "counts": counts,
        "sort": effective_sort,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/faces/face-detail/{face_id}")
def api_face_detail(face_id: int):
    """Return the photo id, filename, dimensions, date, and bbox for one face.

    Used by the /merges preview: when you click a face crop, we overlay the
    stored bbox onto the full photo so you can see context — useful when the
    bbox is tight or off-center.
    """
    with _get_db() as db:
        row = db.conn.execute(
            """SELECT f.id AS face_id, f.photo_id,
                      f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                      ph.filename, ph.date_taken, ph.image_width, ph.image_height
               FROM faces f
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.id = ?""",
            (face_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Face not found")
        return {
            "face_id": row["face_id"],
            "photo_id": row["photo_id"],
            "filename": row["filename"],
            "date_taken": row["date_taken"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
            "bbox": None if row["bbox_top"] is None else {
                "top": row["bbox_top"], "right": row["bbox_right"],
                "bottom": row["bbox_bottom"], "left": row["bbox_left"],
            },
        }


@app.get("/api/faces/group-info")
def api_face_group_info(
    cluster_id: int | None = Query(None),
    person_id: int | None = Query(None),
):
    """Look up a single group's metadata by cluster_id or person_id.

    Used by /faces to auto-open a group from a URL like
    ``/faces?cluster_id=1407`` even when the cluster isn't on the first
    paginated page (or is a hidden singleton). Returns the same shape the
    ``groups`` list emits, so the frontend can pass the result straight
    into its detail view.
    """
    if (cluster_id is None) == (person_id is None):
        raise HTTPException(400, "Pass exactly one of cluster_id or person_id")

    with _get_db() as db:
        if person_id is not None:
            row = db.conn.execute(
                """SELECT p.id as person_id, p.name,
                          COUNT(DISTINCT f.photo_id) as photo_count,
                          COUNT(f.id) as face_count,
                          (SELECT f2.id FROM faces f2
                           WHERE f2.person_id = p.id AND f2.bbox_top IS NOT NULL
                           ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                           LIMIT 1) as rep_face_id
                   FROM persons p
                   LEFT JOIN faces f ON f.person_id = p.id
                   WHERE p.id = ?
                   GROUP BY p.id""",
                (person_id,),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Person not found")
            return {
                "type": "person",
                "person_id": row["person_id"],
                "label": row["name"],
                "photo_count": row["photo_count"],
                "face_count": row["face_count"],
                "rep_face_id": row["rep_face_id"],
                "ignored": False,
            }

        # cluster_id branch
        row = db.conn.execute(
            """SELECT f.cluster_id,
                      COUNT(DISTINCT f.photo_id) as photo_count,
                      COUNT(f.id) as face_count,
                      (SELECT f2.id FROM faces f2
                       WHERE f2.cluster_id = f.cluster_id AND f2.person_id IS NULL
                             AND f2.bbox_top IS NOT NULL
                       ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                       LIMIT 1) as rep_face_id
               FROM faces f
               WHERE f.cluster_id = ? AND f.person_id IS NULL
               GROUP BY f.cluster_id""",
            (cluster_id,),
        ).fetchone()
        if not row or not row["face_count"]:
            raise HTTPException(404, "Cluster not found")
        is_ignored = bool(db.conn.execute(
            "SELECT 1 FROM ignored_clusters WHERE cluster_id = ?", (cluster_id,)
        ).fetchone())
        return {
            "type": "cluster",
            "cluster_id": row["cluster_id"],
            "label": f"Unknown #{row['cluster_id']}",
            "photo_count": row["photo_count"],
            "face_count": row["face_count"],
            "rep_face_id": row["rep_face_id"],
            "ignored": is_ignored,
        }


@app.get("/api/faces/group/{group_type}/{group_id}/photos")
def api_face_group_photos(group_type: str, group_id: int, limit: int = Query(10000)):
    """Get all photos for a specific face group (person or cluster).

    Returns up to `limit` photos ordered by date_taken DESC with NULL dates
    last. The frontend applies the user's chosen sort (date_desc / date_asc /
    quality_desc) client-side via `PS.applySortOrder`, so this endpoint's job
    is just to deliver the complete set — not a date-truncated slice that
    would silently hide old or undated photos from the client sort.

    Default limit is intentionally generous (10k) because a single person
    with >10k indexed photos is rare; if that ceiling is ever hit, the real
    fix is offset-based pagination with a matching UI, not just bumping the
    cap. Callers can still pass `?limit=N` to override.

    Each photo includes an `effective_date` field — `date_taken` when
    present, otherwise the date parsed from the parent folder name via
    `folder_date(filepath)`. The frontend's `PS.applySortOrder` uses
    `effective_date` so "oldest first" / "newest first" behave correctly
    for older imports whose EXIF never had a capture timestamp.
    """
    # Same expression is used for selection + sort so the chronology the
    # client sees matches the chronology we sorted on. Fallback chain:
    # EXIF capture date → file mtime (schema v16) → parent-folder YYYY-MM-DD.
    effective_expr = (
        "COALESCE(p.date_taken, p.date_created, folder_date(p.filepath))"
    )
    order_clause = (
        f"ORDER BY {effective_expr} IS NULL, {effective_expr} DESC"
    )
    with _get_db() as db:
        if group_type == "person":
            rows = db.conn.execute(
                f"""SELECT DISTINCT p.*, f.id as face_id,
                          f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                          f.match_source,
                          {effective_expr} AS effective_date
                   FROM photos p
                   JOIN faces f ON f.photo_id = p.id
                   WHERE f.person_id = ?
                   {order_clause}
                   LIMIT ?""",
                (group_id, limit),
            ).fetchall()
        elif group_type == "cluster":
            rows = db.conn.execute(
                f"""SELECT DISTINCT p.*, f.id as face_id,
                          f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                          f.match_source,
                          {effective_expr} AS effective_date
                   FROM photos p
                   JOIN faces f ON f.photo_id = p.id
                   WHERE f.cluster_id = ? AND f.person_id IS NULL
                   {order_clause}
                   LIMIT ?""",
                (group_id, limit),
            ).fetchall()
        else:
            raise HTTPException(400, "group_type must be 'person' or 'cluster'")

        return {"photos": [dict(r) for r in rows]}


@app.get("/api/photos/geojson")
def api_photos_geojson():
    """Compact point dump of every GPS-bearing photo for the map view.

    Returns `{count, points: [[id, lat, lon, source, year, place_name], ...]}`.
    `source` is `'exif' | 'inferred' | None`; `year` is int or None
    (parsed from the first 4 chars of date_taken); `place_name` is the
    reverse-geocoded string (`"Locality, Admin1, CC"`) or None.

    `place_name` is included so the map's preview pane can compute a
    common-suffix search term across a whole cluster without an N+1
    round-trip (see computeBestSuffix in map.html).

    Must be declared before `/api/photos/{photo_id}` so FastAPI's route
    matcher doesn't interpret `geojson` as a photo id (→ 422).
    """
    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT id, gps_lat, gps_lon, location_source, date_taken, place_name "
            "FROM photos WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
        ).fetchall()
    points = []
    for r in rows:
        dt = r["date_taken"]
        year = None
        if dt and len(dt) >= 4:
            try:
                year = int(dt[:4])
            except ValueError:
                year = None
        points.append([r["id"], r["gps_lat"], r["gps_lon"],
                       r["location_source"], year, r["place_name"]])
    return {"count": len(points), "points": points}


@app.get("/api/photos/{photo_id}")
def api_photo_detail(photo_id: int):
    """Get full metadata for a single photo, including face matches."""
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")

        # Get face data
        faces = db.conn.execute(
            """SELECT f.id, f.person_id, f.bbox_top, f.bbox_right,
                      f.bbox_bottom, f.bbox_left, f.cluster_id,
                      f.match_source, p.name as person_name
               FROM faces f
               LEFT JOIN persons p ON p.id = f.person_id
               WHERE f.photo_id = ?""",
            (photo_id,),
        ).fetchall()

        face_list = []
        for f in faces:
            face_list.append({
                "id": f["id"],
                "person_name": f["person_name"],
                "bbox": {
                    "top": f["bbox_top"],
                    "right": f["bbox_right"],
                    "bottom": f["bbox_bottom"],
                    "left": f["bbox_left"],
                } if f["bbox_top"] is not None else None,
                "cluster_id": f["cluster_id"],
                "match_source": f["match_source"],
            })

        colors = []
        if photo.get("dominant_colors"):
            try:
                colors = json.loads(photo["dominant_colors"])
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "id": photo["id"],
            "filename": photo["filename"],
            "filepath": photo["filepath"],
            "date_taken": photo["date_taken"],
            "place_name": photo.get("place_name"),
            "description": photo["description"],
            "aesthetic_score": photo.get("aesthetic_score"),
            "aesthetic_concepts": json.loads(photo["aesthetic_concepts"]) if photo.get("aesthetic_concepts") else None,
            "aesthetic_critique": photo.get("aesthetic_critique"),
            "camera_make": photo.get("camera_make"),
            "camera_model": photo.get("camera_model"),
            "focal_length": photo.get("focal_length"),
            "exposure_time": photo.get("exposure_time"),
            "f_number": photo.get("f_number"),
            "iso": photo.get("iso"),
            "image_width": photo.get("image_width"),
            "image_height": photo.get("image_height"),
            "tags": json.loads(photo["tags"]) if photo.get("tags") else [],
            "colors": colors,
            "faces": face_list,
            "stack": db.get_photo_stack(photo_id),
        }


@app.get("/api/persons")
def api_persons():
    """List all registered persons with photo counts."""
    with _get_db() as db:
        rows = db.conn.execute(
            """SELECT p.id, p.name,
                      COUNT(DISTINCT f.photo_id) as photo_count
               FROM persons p
               LEFT JOIN faces f ON f.person_id = p.id
               GROUP BY p.id
               ORDER BY p.name"""
        ).fetchall()

    return {"persons": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Face assignment endpoints
# ---------------------------------------------------------------------------

@app.post("/api/faces/{face_id}/assign")
def api_assign_face(face_id: int, name: str = Query(..., description="Person name")):
    """Assign a face to a named person (creates the person if needed)."""
    with _get_db() as db:
        # Verify face exists
        face = db.conn.execute("SELECT id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not face:
            raise HTTPException(404, "Face not found")

        # Find or create person
        person = db.get_person_by_name(name)
        if not person:
            pid = db.add_person(name)
        else:
            pid = person["id"]

        db.assign_face_to_person(face_id, pid, match_source="manual")
        db.conn.commit()

        logger.info("FACE ASSIGN  face_id=%d  person=%r  person_id=%d", face_id, name, pid)
        return {"ok": True, "person_id": pid, "person_name": name}


@app.post("/api/faces/{face_id}/clear")
def api_clear_face(face_id: int):
    """Remove the person assignment from a face."""
    with _get_db() as db:
        face = db.conn.execute("SELECT id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not face:
            raise HTTPException(404, "Face not found")
        db.conn.execute(
            "UPDATE faces SET person_id = NULL, match_source = NULL WHERE id = ?",
            (face_id,),
        )
        db.conn.commit()
        return {"ok": True}


@app.post("/api/faces/bulk-collect")
def api_bulk_collect_faces(data: dict):
    """Collect all face IDs for a list of groups (persons and/or clusters).

    Accepts: {"groups": [{"type": "person", "id": 1}, {"type": "cluster", "id": 42}, ...]}
    Returns: {"face_ids": [1, 2, 3, ...]}
    """
    group_specs = data.get("groups", [])
    if not group_specs:
        return {"face_ids": []}

    face_ids = []
    with _get_db() as db:
        for spec in group_specs:
            gtype = spec.get("type")
            gid = spec.get("id")
            if gtype == "person":
                rows = db.conn.execute(
                    "SELECT id FROM faces WHERE person_id = ?", (gid,)
                ).fetchall()
            elif gtype == "cluster":
                rows = db.conn.execute(
                    "SELECT id FROM faces WHERE cluster_id = ? AND person_id IS NULL", (gid,)
                ).fetchall()
            else:
                continue
            face_ids.extend(r["id"] for r in rows)

    return {"face_ids": face_ids}


@app.post("/api/faces/ignore")
def api_ignore_clusters(data: dict):
    """Mark clusters as ignored so they're hidden from the faces page.

    Accepts: {"cluster_ids": [42, 82, ...]}
    """
    cluster_ids = data.get("cluster_ids", [])
    if not cluster_ids:
        return {"ok": True, "ignored": 0}
    with _get_db() as db:
        for cid in cluster_ids:
            db.conn.execute(
                "INSERT OR IGNORE INTO ignored_clusters (cluster_id) VALUES (?)",
                (cid,),
            )
        db.conn.commit()
    return {"ok": True, "ignored": len(cluster_ids)}


@app.post("/api/faces/unignore")
def api_unignore_clusters(data: dict):
    """Remove clusters from the ignored list.

    Accepts: {"cluster_ids": [42, 82, ...]}
    """
    cluster_ids = data.get("cluster_ids", [])
    if not cluster_ids:
        return {"ok": True, "unignored": 0}
    with _get_db() as db:
        placeholders = ",".join("?" * len(cluster_ids))
        db.conn.execute(
            f"DELETE FROM ignored_clusters WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        db.conn.commit()
    return {"ok": True, "unignored": len(cluster_ids)}


# ---------------------------------------------------------------------------
# Merge-suggestions review (/merges page)
# ---------------------------------------------------------------------------

# Path to the JSON file produced by `suggest-face-merges --json-out`.
# Default matches the NAS Docker volume layout; override with
# PHOTOSEARCH_SUGGESTIONS_JSON for other deployments.
_suggestions_path: str = os.environ.get(
    "PHOTOSEARCH_SUGGESTIONS_JSON",
    str(Path(_db_path).resolve().parent / "suggestions.json"),
)


@app.get("/api/faces/suggestions")
def api_face_suggestions():
    """Return merge suggestions previously computed via `suggest-face-merges`.

    The CLI writes the JSON; this endpoint just serves it. This keeps the
    review page snappy (loading 2000+ groups takes ~20s on an N100) and
    makes the review-then-regenerate workflow explicit. If no file exists
    yet, returns 404 with a message pointing at the CLI.
    """
    path = _suggestions_path
    if not os.path.exists(path):
        raise HTTPException(
            404,
            "No suggestions file found. Run "
            "`photosearch suggest-face-merges --json-out <path>` first "
            f"(expected at {path}; override with PHOTOSEARCH_SUGGESTIONS_JSON).",
        )
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(500, f"Could not read suggestions: {e}")

    # Trim live counts — if accepted merges have collapsed a source cluster
    # since the JSON was written, skip those rows so the user doesn't act on
    # a ghost. Also augment each surviving suggestion with up to 4 sample
    # face_ids per side (biggest-bbox first), so the review page can show a
    # face strip instead of relying on one possibly-poorly-bboxed rep face.
    suggestions = payload.get("suggestions", [])
    if suggestions:
        with _get_db() as db:
            cluster_counts = {
                r["cluster_id"]: r["n"]
                for r in db.conn.execute(
                    """SELECT cluster_id, COUNT(*) AS n FROM faces
                       WHERE cluster_id IS NOT NULL AND person_id IS NULL
                       GROUP BY cluster_id"""
                ).fetchall()
            }

            def _still_live(side: dict) -> bool:
                if side.get("type") == "cluster":
                    return cluster_counts.get(side.get("id"), 0) > 0
                return True

            suggestions = [
                s for s in suggestions
                if _still_live(s.get("left", {})) and _still_live(s.get("right", {}))
            ]

            # Collect every unique (type, id) so we can fetch sample face_ids
            # in one pass instead of 2×N SQL queries.
            wanted: set[tuple[str, int]] = set()
            for s in suggestions:
                for side in (s.get("left"), s.get("right")):
                    if side and side.get("type") and side.get("id") is not None:
                        wanted.add((side["type"], int(side["id"])))

            sample_faces: dict[tuple[str, int], list[int]] = {}
            cluster_ids = [gid for (t, gid) in wanted if t == "cluster"]
            person_ids = [gid for (t, gid) in wanted if t == "person"]

            # Window function: top 4 biggest-bbox faces per unknown cluster.
            if cluster_ids:
                placeholders = ",".join("?" * len(cluster_ids))
                rows = db.conn.execute(
                    f"""WITH ranked AS (
                          SELECT f.id AS face_id, f.cluster_id,
                                 ROW_NUMBER() OVER (
                                   PARTITION BY f.cluster_id
                                   ORDER BY
                                     CASE WHEN f.bbox_top IS NOT NULL
                                          THEN (f.bbox_bottom - f.bbox_top)
                                               * (f.bbox_right - f.bbox_left)
                                          ELSE 0 END DESC,
                                     f.id
                                 ) AS rn
                          FROM faces f
                          WHERE f.person_id IS NULL
                                AND f.cluster_id IN ({placeholders})
                        )
                        SELECT face_id, cluster_id FROM ranked WHERE rn <= 4""",
                    cluster_ids,
                ).fetchall()
                for r in rows:
                    sample_faces.setdefault(("cluster", int(r["cluster_id"])), []).append(int(r["face_id"]))

            if person_ids:
                placeholders = ",".join("?" * len(person_ids))
                rows = db.conn.execute(
                    f"""WITH ranked AS (
                          SELECT f.id AS face_id, f.person_id,
                                 ROW_NUMBER() OVER (
                                   PARTITION BY f.person_id
                                   ORDER BY
                                     CASE WHEN f.bbox_top IS NOT NULL
                                          THEN (f.bbox_bottom - f.bbox_top)
                                               * (f.bbox_right - f.bbox_left)
                                          ELSE 0 END DESC,
                                     f.id
                                 ) AS rn
                          FROM faces f
                          WHERE f.person_id IN ({placeholders})
                        )
                        SELECT face_id, person_id FROM ranked WHERE rn <= 4""",
                    person_ids,
                ).fetchall()
                for r in rows:
                    sample_faces.setdefault(("person", int(r["person_id"])), []).append(int(r["face_id"]))

            for s in suggestions:
                for side in ("left", "right"):
                    g = s.get(side)
                    if not g:
                        continue
                    key = (g.get("type"), int(g["id"])) if g.get("id") is not None else None
                    g["sample_face_ids"] = sample_faces.get(key, [])

        payload["live_count"] = len(suggestions)
        payload["suggestions"] = suggestions

    payload["source_path"] = path
    try:
        payload["generated_at"] = os.path.getmtime(path)
    except OSError:
        pass
    return payload


@app.post("/api/faces/suggestions/regenerate")
def api_regenerate_suggestions(data: dict = None):
    """Re-run the merge-suggestion engine and overwrite the cached JSON.

    Body (all optional):
      {"centroid_cutoff": 0.95, "min_pair_cutoff": 0.60,
       "max_members": 60, "min_group_size": 1,
       "include_ignored": false}

    Blocking — the engine loads every group's encodings (~20s on an N100 for
    ~2000 groups) then computes pairwise scores. Progress streaming would be
    nicer UX; we'll add it if the wait becomes noticeable in practice.
    """
    from .face_merge import (
        CENTROID_CUTOFF, MIN_PAIR_CUTOFF, MAX_MEMBERS_PER_GROUP,
        compute_suggestions, load_groups,
    )

    data = data or {}
    centroid_cutoff = float(data.get("centroid_cutoff", CENTROID_CUTOFF))
    min_pair_cutoff = float(data.get("min_pair_cutoff", MIN_PAIR_CUTOFF))
    max_members = int(data.get("max_members", MAX_MEMBERS_PER_GROUP))
    min_group_size = int(data.get("min_group_size", 1))
    include_ignored = bool(data.get("include_ignored", False))

    if not (0 < centroid_cutoff <= 2.0):
        raise HTTPException(400, "centroid_cutoff must be in (0, 2.0]")
    if not (0 < min_pair_cutoff <= 2.0):
        raise HTTPException(400, "min_pair_cutoff must be in (0, 2.0]")
    if max_members < 2:
        raise HTTPException(400, "max_members must be ≥ 2")
    if min_group_size < 1:
        raise HTTPException(400, "min_group_size must be ≥ 1")

    # Sanity cap on N². With min_group_size=1 on a freshly reclustered library
    # there can be 30k+ unknown clusters; pair scoring is O(N²) and uvicorn
    # gets killed before it returns. Estimate group count cheaply via SQL
    # before paying the encoding-load cost, and refuse if it'd blow up.
    _MAX_GROUP_COUNT = 5000  # ~12.5M pairs; runs in a few minutes on N100.
    with _get_db() as db:
        ignored_filter = "" if include_ignored else """
            AND (f.cluster_id IS NULL OR f.cluster_id NOT IN (SELECT cluster_id FROM ignored_clusters))
        """
        cluster_count = db.conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT f.cluster_id
                FROM faces f
                WHERE f.cluster_id IS NOT NULL AND f.person_id IS NULL
                {ignored_filter}
                GROUP BY f.cluster_id HAVING COUNT(*) >= ?
            )
        """, (min_group_size,)).fetchone()[0]
        person_count = db.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        est_groups = cluster_count + person_count
        if est_groups > _MAX_GROUP_COUNT:
            raise HTTPException(
                413,
                f"Too many groups: {est_groups} (cap {_MAX_GROUP_COUNT}). "
                f"Raise min_group_size — singletons rarely produce useful "
                f"suggestions and dominate pair count. Try min_group_size=3 "
                f"or 5.",
            )

        groups = load_groups(
            db,
            include_ignored_clusters=include_ignored,
            max_members=max_members,
            min_group_size=min_group_size,
        )
        suggestions = compute_suggestions(
            groups,
            centroid_cutoff=centroid_cutoff,
            min_pair_cutoff=min_pair_cutoff,
        )

    payload = {
        "centroid_cutoff": centroid_cutoff,
        "min_pair_cutoff": min_pair_cutoff,
        "max_members": max_members,
        "min_group_size": min_group_size,
        "group_count": len(groups),
        "suggestions": [s.as_dict() for s in suggestions],
    }

    os.makedirs(os.path.dirname(_suggestions_path) or ".", exist_ok=True)
    tmp_path = _suggestions_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, _suggestions_path)

    return {
        "ok": True,
        "group_count": len(groups),
        "suggestion_count": len(suggestions),
        "written_to": _suggestions_path,
    }


@app.post("/api/faces/clusters/{cluster_id}/split")
def api_split_cluster(cluster_id: int, data: dict = None):
    """Split an unknown cluster into tighter sub-clusters via DBSCAN.

    Body (all optional): {"eps": 0.45, "min_samples": 2, "dry_run": false}.
    Dry-run returns the assignments + histogram without writing.

    Rejects named-person clusters and any cluster containing person-assigned
    faces — split is for unknown-cluster cleanup only.
    """
    from .faces import (
        split_cluster, SPLIT_DEFAULT_EPS, SPLIT_DEFAULT_MIN_SAMPLES,
        CLUSTER_MIN_DET_SCORE, CLUSTER_MIN_BBOX_EDGE,
    )

    data = data or {}
    eps = float(data.get("eps", SPLIT_DEFAULT_EPS))
    min_samples = int(data.get("min_samples", SPLIT_DEFAULT_MIN_SAMPLES))
    min_det_score = float(data.get("min_det_score", CLUSTER_MIN_DET_SCORE))
    min_bbox_edge = int(data.get("min_bbox_edge", CLUSTER_MIN_BBOX_EDGE))
    dry_run = bool(data.get("dry_run", False))

    if eps <= 0 or eps >= 2.0:
        raise HTTPException(400, "eps must be in (0, 2.0)")
    if min_samples < 1:
        raise HTTPException(400, "min_samples must be ≥ 1")

    with _get_db() as db:
        try:
            summary = split_cluster(
                db, cluster_id=cluster_id,
                eps=eps, min_samples=min_samples, dry_run=dry_run,
                min_det_score=min_det_score, min_bbox_edge=min_bbox_edge,
            )
        except ValueError as err:
            raise HTTPException(400, str(err))

        if summary["face_count"] == 0:
            raise HTTPException(404, f"Cluster #{cluster_id} has no unknown faces")

        return {
            "ok": True,
            "cluster_id": cluster_id,
            "eps": eps,
            "min_samples": min_samples,
            "min_det_score": min_det_score,
            "min_bbox_edge": min_bbox_edge,
            "dry_run": dry_run,
            "face_count": summary["face_count"],
            "sub_cluster_count": summary["sub_cluster_count"],
            "noise_count": summary["noise_count"],
            "filtered_out_count": summary.get("filtered_out_count", 0),
            "histogram": summary["histogram"],
            "new_cluster_ids": summary["new_cluster_ids"],
            "elapsed_sec": summary.get("elapsed_sec"),
        }


@app.post("/api/faces/merges")
def api_apply_face_merge(data: dict):
    """Apply a cluster→person or cluster→cluster merge.

    Body shape: ``{"source": {"type": "cluster", "id": <int>},
                   "target": {"type": "cluster"|"person", "id": <int>}}``

    - cluster → person: every source-cluster face (with person_id IS NULL)
      has its person_id set to the target and cluster_id cleared. match_source
      is set to 'merge_review' for audit.
    - cluster → cluster: every source-cluster face (with person_id IS NULL)
      has its cluster_id updated to the target. Person-id-assigned faces
      are left alone (defensive — shouldn't happen, but cheap).
    """
    source = data.get("source") or {}
    target = data.get("target") or {}

    if source.get("type") != "cluster":
        raise HTTPException(400, "source.type must be 'cluster'")
    if target.get("type") not in ("cluster", "person"):
        raise HTTPException(400, "target.type must be 'cluster' or 'person'")

    try:
        source_id = int(source["id"])
        target_id = int(target["id"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "source.id and target.id must be integers")

    if source.get("type") == target.get("type") and source_id == target_id:
        raise HTTPException(400, "source and target must differ")

    with _get_db() as db:
        # Verify the target exists — prevents accidental orphaning.
        if target["type"] == "person":
            exists = db.conn.execute(
                "SELECT 1 FROM persons WHERE id = ?", (target_id,)
            ).fetchone()
            if not exists:
                raise HTTPException(404, f"Person #{target_id} not found")
        else:
            exists = db.conn.execute(
                "SELECT 1 FROM faces WHERE cluster_id = ? AND person_id IS NULL LIMIT 1",
                (target_id,),
            ).fetchone()
            if not exists:
                raise HTTPException(404, f"Cluster #{target_id} not found")

        cur = db.conn.cursor()
        try:
            if target["type"] == "person":
                cur.execute(
                    """UPDATE faces
                       SET person_id = ?, cluster_id = NULL, match_source = 'merge_review'
                       WHERE cluster_id = ? AND person_id IS NULL""",
                    (target_id, source_id),
                )
            else:
                cur.execute(
                    """UPDATE faces
                       SET cluster_id = ?
                       WHERE cluster_id = ? AND person_id IS NULL""",
                    (target_id, source_id),
                )
            moved = cur.rowcount
            db.conn.commit()
        except Exception:
            db.conn.rollback()
            raise

    return {
        "ok": True,
        "moved_face_count": moved,
        "source": {"type": source["type"], "id": source_id},
        "target": {"type": target["type"], "id": target_id},
    }


@app.get("/api/faces/manual-assignments")
def api_export_manual_assignments():
    """Export all manual face-to-person assignments as JSON.

    Each entry contains the photo filepath and face bounding box so
    assignments can be re-imported after clearing matches or rebuilding
    the face index.
    """
    with _get_db() as db:
        rows = db.conn.execute(
            """SELECT f.id as face_id, f.bbox_top, f.bbox_right,
                      f.bbox_bottom, f.bbox_left,
                      p.name as person_name,
                      ph.filepath
               FROM faces f
               JOIN persons p ON p.id = f.person_id
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.match_source = 'manual'
               ORDER BY ph.filepath, f.id"""
        ).fetchall()

        assignments = []
        for r in rows:
            assignments.append({
                "filepath": r["filepath"],
                "person_name": r["person_name"],
                "bbox": {
                    "top": r["bbox_top"],
                    "right": r["bbox_right"],
                    "bottom": r["bbox_bottom"],
                    "left": r["bbox_left"],
                },
            })

        return {"assignments": assignments, "count": len(assignments)}


@app.post("/api/faces/import-assignments")
def api_import_manual_assignments(data: dict):
    """Re-apply manual face assignments from a previously exported JSON.

    Matches by filepath + bounding box overlap (>80% IoU) so that
    assignments survive face re-detection even if IDs change.
    """
    assignments = data.get("assignments", [])
    if not assignments:
        return {"ok": True, "matched": 0, "skipped": 0}

    matched = 0
    skipped = 0

    with _get_db() as db:
        for a in assignments:
            filepath = a.get("filepath")
            person_name = a.get("person_name")
            bbox = a.get("bbox")
            if not filepath or not person_name or not bbox:
                skipped += 1
                continue

            # Find the photo
            photo = db.conn.execute(
                "SELECT id FROM photos WHERE filepath = ?", (filepath,)
            ).fetchone()
            if not photo:
                skipped += 1
                continue

            # Find or create person
            person = db.get_person_by_name(person_name)
            if not person:
                pid = db.add_person(person_name)
            else:
                pid = person["id"]

            # Find best-matching face by bounding box overlap
            faces = db.conn.execute(
                """SELECT id, bbox_top, bbox_right, bbox_bottom, bbox_left
                   FROM faces WHERE photo_id = ?""",
                (photo["id"],),
            ).fetchall()

            best_face_id = None
            best_iou = 0.0
            for f in faces:
                if f["bbox_top"] is None:
                    continue
                # Compute IoU
                t1, r1, b1, l1 = bbox["top"], bbox["right"], bbox["bottom"], bbox["left"]
                t2, r2, b2, l2 = f["bbox_top"], f["bbox_right"], f["bbox_bottom"], f["bbox_left"]
                inter_t = max(t1, t2)
                inter_l = max(l1, l2)
                inter_b = min(b1, b2)
                inter_r = min(r1, r2)
                inter_area = max(0, inter_b - inter_t) * max(0, inter_r - inter_l)
                area1 = (b1 - t1) * (r1 - l1)
                area2 = (b2 - t2) * (r2 - l2)
                union_area = area1 + area2 - inter_area
                iou = inter_area / union_area if union_area > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_face_id = f["id"]

            if best_face_id and best_iou > 0.5:
                db.assign_face_to_person(best_face_id, pid, match_source="manual")
                matched += 1
            else:
                skipped += 1

        db.conn.commit()

    logger.info("IMPORT ASSIGNMENTS  matched=%d  skipped=%d", matched, skipped)
    return {"ok": True, "matched": matched, "skipped": skipped}


# ---------------------------------------------------------------------------
# Shoot Review endpoints
# ---------------------------------------------------------------------------

@app.get("/api/review/folders")
def api_review_folders():
    """List directories that contain indexed photos, for the folder picker."""
    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT filepath, date_taken FROM photos WHERE filepath IS NOT NULL"
        ).fetchall()

    # Extract unique parent directories with most recent date_taken per folder
    dir_dates: dict[str, str] = {}
    for row in rows:
        fp = row["filepath"]
        parent = str(Path(fp).parent)
        if parent and parent != ".":
            dt = row["date_taken"] or ""
            if parent not in dir_dates or dt > dir_dates[parent]:
                dir_dates[parent] = dt

    # Return folders sorted by most recent photo first (default), with date info
    folders = [
        {"path": d, "max_date": dir_dates[d]}
        for d in sorted(dir_dates, key=lambda d: dir_dates[d], reverse=True)
    ]
    return {"folders": folders}


@app.get("/api/review/run")
def api_review_run(
    directory: str = Query(..., description="Directory path to review"),
    target_pct: float = Query(0.10, description="Target selection percentage"),
    distance_threshold: float = Query(0.0, description="Clustering distance threshold (0 = adaptive)"),
):
    """Run the culling algorithm on a directory and return selections."""
    from .cull import select_best_photos, save_selections

    with _get_db() as db:
        # Resolve the directory — could be relative to photo_root
        resolved_dir = directory
        if db.photo_root and not Path(directory).is_absolute():
            resolved_dir = str(Path(db.photo_root) / directory)

        selections = select_best_photos(
            db, resolved_dir,
            target_pct=target_pct,
            distance_threshold=distance_threshold,
        )

        if not selections:
            return {"photos": [], "stats": {"total": 0, "selected": 0}}

        # Save selections to DB for persistence
        save_selections(db, resolved_dir, selections)

        # Build response
        items = []
        for p in selections:
            item = {
                "id": p["id"],
                "filename": p.get("filename", ""),
                "filepath": p.get("filepath", ""),
                "aesthetic_score": p.get("aesthetic_score"),
                "date_taken": p.get("date_taken"),
                "selected": p.get("selected", False),
                "cluster_id": p.get("cluster_id"),
                "rank_in_cluster": p.get("rank_in_cluster"),
                "has_raw": bool(p.get("raw_filepath")),
            }
            stack_info = db.get_photo_stack(p["id"])
            if stack_info:
                item["stack_id"] = stack_info["stack_id"]
                item["stack_is_top"] = stack_info["is_top"]
                item["stack_count"] = stack_info["member_count"]
            items.append(item)

        n_selected = sum(1 for i in items if i["selected"])
        n_clusters = len(set(i["cluster_id"] for i in items if i["cluster_id"] is not None))

    return {
        "photos": items,
        "stats": {
            "total": len(items),
            "selected": n_selected,
            "clusters": n_clusters,
            "pct": round(n_selected / len(items) * 100, 1) if items else 0,
        },
    }


@app.get("/api/review/load")
def api_review_load(
    directory: str = Query(..., description="Directory path"),
):
    """Load previously saved selections for a directory."""
    from .cull import load_selections

    with _get_db() as db:
        resolved_dir = directory
        if db.photo_root and not Path(directory).is_absolute():
            resolved_dir = str(Path(db.photo_root) / directory)

        selections = load_selections(db, resolved_dir)
        if not selections:
            return {"photos": [], "stats": {"total": 0, "selected": 0}}

        items = []
        for p in selections:
            item = {
                "id": p["photo_id"],
                "filename": p.get("filename", ""),
                "filepath": p.get("filepath", ""),
                "aesthetic_score": p.get("aesthetic_score"),
                "date_taken": p.get("date_taken"),
                "selected": bool(p["selected"]),
                "cluster_id": p.get("cluster_id"),
                "rank_in_cluster": p.get("rank_in_cluster"),
                "has_raw": bool(p.get("raw_filepath")),
            }
            stack_info = db.get_photo_stack(p["photo_id"])
            if stack_info:
                item["stack_id"] = stack_info["stack_id"]
                item["stack_is_top"] = stack_info["is_top"]
                item["stack_count"] = stack_info["member_count"]
            items.append(item)

        n_selected = sum(1 for i in items if i["selected"])
        n_clusters = len(set(i["cluster_id"] for i in items if i["cluster_id"] is not None))

    return {
        "photos": items,
        "stats": {
            "total": len(items),
            "selected": n_selected,
            "clusters": n_clusters,
            "pct": round(n_selected / len(items) * 100, 1) if items else 0,
        },
    }


@app.post("/api/review/toggle/{photo_id}")
def api_review_toggle(photo_id: int, selected: bool = Query(...)):
    """Toggle a photo's selection state."""
    from .cull import toggle_selection

    with _get_db() as db:
        toggle_selection(db, photo_id, selected)

    return {"ok": True, "photo_id": photo_id, "selected": selected}


@app.get("/api/review/export")
def api_review_export(
    directory: str = Query(..., description="Directory path"),
    include_raw: bool = Query(False, description="Include ARW raw file paths"),
):
    """Get the list of selected photo paths for export/copying."""
    from .cull import load_selections

    with _get_db() as db:
        resolved_dir = directory
        if db.photo_root and not Path(directory).is_absolute():
            resolved_dir = str(Path(db.photo_root) / directory)

        selections = load_selections(db, resolved_dir)
        if not selections:
            return {"files": []}

        files = []
        for p in selections:
            if not p["selected"]:
                continue
            abs_path = db.resolve_filepath(p["filepath"])
            files.append({"type": "jpg", "path": abs_path, "filename": p["filename"]})
            if include_raw and p.get("raw_filepath"):
                raw_abs = db.resolve_filepath(p["raw_filepath"])
                raw_name = Path(raw_abs).name if raw_abs else ""
                files.append({"type": "arw", "path": raw_abs, "filename": raw_name})

    return {"files": files, "count": len(files)}


@app.get("/api/stats")
def api_stats():
    """Database statistics."""
    with _get_db() as db:
        photo_count = db.photo_count()
        clip_count = db.conn.execute("SELECT COUNT(*) as c FROM clip_embeddings").fetchone()["c"]
        face_count = db.conn.execute("SELECT COUNT(*) as c FROM faces").fetchone()["c"]
        person_count = db.conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()["c"]
        described = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE description IS NOT NULL"
        ).fetchone()["c"]
        scored = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE aesthetic_score IS NOT NULL"
        ).fetchone()["c"]

        quality_stats = None
        if scored > 0:
            row = db.conn.execute(
                """SELECT MIN(aesthetic_score) as min_s, MAX(aesthetic_score) as max_s,
                          AVG(aesthetic_score) as avg_s
                   FROM photos WHERE aesthetic_score IS NOT NULL"""
            ).fetchone()
            quality_stats = {
                "min": round(row["min_s"], 2),
                "max": round(row["max_s"], 2),
                "mean": round(row["avg_s"], 2),
            }

        stack_row = db.conn.execute(
            """SELECT COUNT(DISTINCT stack_id) as stack_count,
                      COUNT(*) as stacked_photos
               FROM stack_members"""
        ).fetchone()
        stack_count = stack_row["stack_count"]
        stacked_photos = stack_row["stacked_photos"]

        concepts_analyzed = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE aesthetic_concepts IS NOT NULL"
        ).fetchone()["c"]

        tagged = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE tags IS NOT NULL AND tags != '[]'"
        ).fetchone()["c"]

        verify_rows = db.conn.execute(
            """SELECT verification_status AS status, COUNT(*) AS c
               FROM photos
               WHERE verified_at IS NOT NULL
               GROUP BY verification_status"""
        ).fetchall()
        verify_counts = {"pass": 0, "fail": 0, "regenerated": 0}
        for r in verify_rows:
            st = r["status"] or "pass"
            if st in verify_counts:
                verify_counts[st] += r["c"]

    return {
        "photos": photo_count,
        "clip_embedded": clip_count,
        "faces": face_count,
        "persons": person_count,
        "described": described,
        "quality_scored": scored,
        "quality_stats": quality_stats,
        "concepts_analyzed": concepts_analyzed,
        "tagged": tagged,
        "stacks": stack_count,
        "stacked_photos": stacked_photos,
        "verify_passed": verify_counts["pass"],
        "verify_failed": verify_counts["fail"],
        "verify_regenerated": verify_counts["regenerated"],
    }


@app.get("/api/stats/activity")
def api_stats_activity():
    """Hourly index activity for the last 3 days."""
    with _get_db() as db:
        rows = db.conn.execute("""
            SELECT strftime('%Y-%m-%dT%H:00:00', created_at) AS hour,
                   pass_type, action, SUM(count) AS total
            FROM index_activity
            WHERE created_at >= datetime('now', '-3 days')
            GROUP BY hour, pass_type, action
            ORDER BY hour
        """).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/stats/errors")
def api_stats_errors():
    """Recent indexing errors (last 3 days, max 200)."""
    with _get_db() as db:
        rows = db.conn.execute("""
            SELECT pass_type, filepath, message, created_at
            FROM index_errors
            WHERE created_at >= datetime('now', '-3 days')
            ORDER BY created_at DESC
            LIMIT 200
        """).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Inferred geotagging (M19)
# ---------------------------------------------------------------------------

from collections import Counter
import random


_INFER_DEFAULTS = {
    "window_minutes": 30,
    "max_drift_km": 25.0,
    "min_confidence": 0.0,
    "cascade": True,
    "max_cascade_rounds": 10,
}


def _parse_infer_params(data: dict) -> dict:
    """Pull infer params from a POST body, falling back to defaults.
    Coerces numeric types defensively (JSON-over-HTTP sometimes sends
    ints where we expect floats and vice versa)."""
    out = dict(_INFER_DEFAULTS)
    if "window_minutes" in data: out["window_minutes"] = int(data["window_minutes"])
    if "max_drift_km" in data:   out["max_drift_km"] = float(data["max_drift_km"])
    if "min_confidence" in data: out["min_confidence"] = float(data["min_confidence"])
    if "cascade" in data:        out["cascade"] = bool(data["cascade"])
    if "max_cascade_rounds" in data:
        out["max_cascade_rounds"] = int(data["max_cascade_rounds"])
    return out


def _bucket_confidence(c: float) -> str:
    if c >= 0.90:   return ">=0.90"
    if c >= 0.75:   return "0.75-0.90"
    if c >= 0.50:   return "0.50-0.75"
    if c >= 0.25:   return "0.25-0.50"
    return "<0.25"


@app.post("/api/geocode/infer-preview")
def api_infer_preview(data: dict):
    from .infer_location import infer_locations

    params = _parse_infer_params(data)
    with _get_db() as db:
        result = infer_locations(db, **params)
        candidates = result["candidates"]
        summary = result["summary"]

        buckets = Counter(_bucket_confidence(c["confidence"]) for c in candidates)
        hops = Counter(c["hop_count"] for c in candidates)

        sample_rows = random.sample(candidates, min(10, len(candidates)))
        # Pre-geocode just the sampled rows so the UI can show place_name.
        from .geocode import reverse_geocode_batch
        sample_places = reverse_geocode_batch(
            [(c["lat"], c["lon"]) for c in sample_rows]
        ) if sample_rows else []

        samples = []
        for c, place in zip(sample_rows, sample_places):
            samples.append({
                "photo_id": c["photo_id"],
                "filepath": c["filepath"],
                "thumbnail_url": f"/api/photos/{c['photo_id']}/thumbnail",
                "inferred_lat": c["lat"],
                "inferred_lon": c["lon"],
                "place_name": place,
                "confidence": c["confidence"],
                "hop_count": c["hop_count"],
                "time_gap_min": c["time_gap_min"],
                "drift_km": c["drift_km"],
                "sides": c["sides"],
                "source_photo_id": c["source_photo_id"],
            })

        return {
            "total_photos": summary["total_photos"],
            "no_gps_count": summary["no_gps_count"],
            "gps_count": summary["gps_count"],
            "candidate_count": summary["candidate_count"],
            "cascade_rounds_used": summary["cascade_rounds_used"],
            "skipped": summary["skipped"],
            "confidence_buckets": [
                {"bucket": b, "count": buckets.get(b, 0)}
                for b in (">=0.90", "0.75-0.90", "0.50-0.75",
                          "0.25-0.50", "<0.25")
            ],
            "hop_distribution": [
                {"hops": h, "count": hops[h]} for h in sorted(hops)
            ],
            "samples": samples,
        }


@app.post("/api/geocode/infer-apply")
def api_infer_apply(data: dict):
    import time
    from .infer_location import infer_locations
    from .geocode import reverse_geocode_batch

    if not data.get("confirm"):
        raise HTTPException(
            status_code=400,
            detail="confirm=true required to apply inferences",
        )

    params = _parse_infer_params(data)
    start = time.perf_counter()
    with _get_db() as db:
        result = infer_locations(db, **params)
        candidates = result["candidates"]
        if not candidates:
            return {"updated_count": 0, "rounds_used": 0, "duration_seconds": 0.0}

        coords = [(c["lat"], c["lon"]) for c in candidates]
        places = reverse_geocode_batch(coords)

        cur = db.conn.cursor()
        updated = 0
        for c, place in zip(candidates, places):
            cur.execute(
                "UPDATE photos "
                "SET gps_lat=?, gps_lon=?, place_name=COALESCE(place_name, ?), "
                "    location_source='inferred', location_confidence=? "
                "WHERE id=? AND gps_lat IS NULL",
                (c["lat"], c["lon"], place, c["confidence"], c["photo_id"]),
            )
            updated += cur.rowcount
        db.conn.commit()

    return {
        "updated_count": updated,
        "rounds_used": result["summary"]["cascade_rounds_used"],
        "duration_seconds": time.perf_counter() - start,
    }


# ---------------------------------------------------------------------------
# Manual bulk geotagging (UI-driven) — companion to M19 inferred geotagging
# ---------------------------------------------------------------------------
# Feature surface:
#   GET  /api/geotag/folders         — folder summary for the /geotag picker
#   GET  /api/geotag/folder-photos   — photos in one folder, optional inferred
#   GET  /api/geotag/known-places    — distinct place_names from the library
#   GET  /api/geocode/search         — Nominatim forward-geocode proxy + cache
#   POST /api/photos/bulk-set-location — the write (location_source='manual')


def _folder_of(filepath: str, filename: str) -> str:
    """Return the folder portion of a photo's filepath.

    Cheaper than Path().parent on a tight loop — photos store forward
    slashes even on Windows (SQLite stores text, and the indexer
    normalizes), so rsplit is safe.
    """
    if not filepath:
        return ""
    if filename and filepath.endswith("/" + filename):
        return filepath[: -(len(filename) + 1)]
    idx = filepath.rfind("/")
    return filepath[:idx] if idx > 0 else filepath


@app.get("/api/geotag/folders")
def api_geotag_folders(include_fully_tagged: bool = False):
    """Folder summary keyed for the /geotag left panel.

    Returns folders sorted by no_gps count descending. A folder's entry
    includes photo counts split by provenance (exif / inferred / none)
    plus date range, so the UI can prioritize which folders to tackle.

    With `include_fully_tagged=true` the response also covers folders
    where every photo already has GPS; default is to hide them.
    """
    from collections import defaultdict

    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT filepath, filename, location_source, gps_lat, date_taken FROM photos"
        ).fetchall()

    folders = defaultdict(lambda: {
        "total": 0, "with_exif": 0, "with_inferred": 0, "no_gps": 0,
        "date_from": None, "date_to": None,
    })
    for r in rows:
        folder = _folder_of(r["filepath"], r["filename"])
        f = folders[folder]
        f["total"] += 1
        src = r["location_source"]
        if src == "exif":
            f["with_exif"] += 1
        elif src == "inferred":
            f["with_inferred"] += 1
        if r["gps_lat"] is None:
            f["no_gps"] += 1
        dt = r["date_taken"]
        if dt:
            if f["date_from"] is None or dt < f["date_from"]:
                f["date_from"] = dt
            if f["date_to"] is None or dt > f["date_to"]:
                f["date_to"] = dt

    out = []
    for path, f in folders.items():
        if not include_fully_tagged and f["no_gps"] == 0:
            continue
        out.append({"path": path, **f})
    out.sort(key=lambda f: (-f["no_gps"], f["path"]))
    return {"folders": out, "total_folders": len(out)}


@app.get("/api/geotag/folder-photos")
def api_geotag_folder_photos(folder: str, show_inferred: bool = False,
                              limit: int = 1000):
    """Photos in one folder for the /geotag thumbnails panel.

    By default returns only photos where gps_lat IS NULL (the ones that
    need tagging). With `show_inferred=true`, also includes photos where
    `location_source='inferred'` so the user can manually correct any M19
    misfires. `location_source='exif'` photos are always excluded — those
    came from the camera and are authoritative.
    """
    with _get_db() as db:
        if show_inferred:
            where = ("WHERE (gps_lat IS NULL "
                     "   OR location_source='inferred')")
        else:
            where = "WHERE gps_lat IS NULL"
        # Photos in this folder have filepath like "<folder>/<filename>".
        pattern = folder.rstrip("/") + "/%"
        # Guard against matching sub-folders: require no additional "/"
        # between folder and filename. Done by filtering in Python below.
        rows = db.conn.execute(
            f"""SELECT id, filepath, filename, date_taken, gps_lat, gps_lon,
                       place_name, location_source, location_confidence
                FROM photos
                {where} AND filepath LIKE ?
                ORDER BY COALESCE(date_taken, filepath)
                LIMIT ?""",
            (pattern, limit),
        ).fetchall()

    photos = []
    for r in rows:
        if _folder_of(r["filepath"], r["filename"]) != folder.rstrip("/"):
            continue
        photos.append(dict(r))
    return {"folder": folder, "photos": photos, "count": len(photos)}


@app.get("/api/geotag/known-places")
def api_geotag_known_places(q: str = "", limit: int = 20):
    """Distinct place_names in the library matching q (case-insensitive
    substring). Sorted by photo count descending so frequently-used
    places float up. Used alongside Nominatim to seed the typeahead.
    """
    q_norm = (q or "").strip()
    with _get_db() as db:
        if q_norm:
            rows = db.conn.execute(
                """SELECT place_name, COUNT(*) AS photo_count
                   FROM photos
                   WHERE place_name IS NOT NULL
                     AND place_name LIKE ?
                   GROUP BY place_name
                   ORDER BY photo_count DESC
                   LIMIT ?""",
                (f"%{q_norm}%", limit),
            ).fetchall()
        else:
            rows = db.conn.execute(
                """SELECT place_name, COUNT(*) AS photo_count
                   FROM photos
                   WHERE place_name IS NOT NULL
                   GROUP BY place_name
                   ORDER BY photo_count DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
    return {"places": [{"place_name": r["place_name"],
                        "photo_count": r["photo_count"]} for r in rows]}


@app.get("/api/geocode/search")
def api_geocode_search(q: str, limit: int = 5):
    """Forward-geocode a free-text query via Nominatim, with a persistent
    cache in the geocode_cache table. Thin wrapper around
    `photosearch.geocode.forward_geocode` so search.py and other
    callers can share the same cache + normalization logic.
    """
    from .geocode import forward_geocode

    if not (q or "").strip():
        raise HTTPException(status_code=400, detail="q required")
    with _get_db() as db:
        try:
            results, source = forward_geocode(db, q, limit)
        except Exception as err:
            raise HTTPException(status_code=502, detail=f"Nominatim error: {err}")
    return {"results": results, "source": source}


@app.post("/api/photos/bulk-set-location")
def api_bulk_set_location(data: dict):
    """Manually assign GPS + place_name to a batch of photos.

    Body: `{photo_ids: [...], lat: float, lon: float, place_name: str,
            overwrite: bool = false}`.

    Writes `gps_lat`, `gps_lon`, `place_name`, `location_source='manual'`,
    `location_confidence=NULL`. With `overwrite=false` (default) rows with
    pre-existing `gps_lat` are skipped; `overwrite=true` replaces any
    existing value including 'exif' and 'inferred'.
    """
    photo_ids = data.get("photo_ids") or []
    if not isinstance(photo_ids, list) or not photo_ids:
        raise HTTPException(status_code=400, detail="photo_ids required")
    try:
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="lat/lon required (numeric)")
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(status_code=400, detail="lat/lon out of range")
    place_name = data.get("place_name") or None
    overwrite = bool(data.get("overwrite", False))

    updated = 0
    skipped = 0
    with _get_db() as db:
        cur = db.conn.cursor()
        for pid in photo_ids:
            if overwrite:
                cur.execute(
                    "UPDATE photos "
                    "SET gps_lat=?, gps_lon=?, place_name=?, "
                    "    location_source='manual', location_confidence=NULL "
                    "WHERE id=?",
                    (lat, lon, place_name, pid),
                )
            else:
                cur.execute(
                    "UPDATE photos "
                    "SET gps_lat=?, gps_lon=?, place_name=?, "
                    "    location_source='manual', location_confidence=NULL "
                    "WHERE id=? AND gps_lat IS NULL",
                    (lat, lon, place_name, pid),
                )
            if cur.rowcount > 0:
                updated += cur.rowcount
            else:
                skipped += 1
        db.conn.commit()

    return {"updated_count": updated, "skipped_count": skipped}


# ---------------------------------------------------------------------------
# Verify failures

_VERIFY_STATUSES = ("fail", "regenerated")


def _parse_statuses(raw: Optional[str]) -> list[str]:
    """Parse a comma-separated status list into a filtered, ordered list."""
    if not raw:
        return ["fail", "regenerated"]
    wanted = [s.strip() for s in raw.split(",") if s.strip()]
    return [s for s in wanted if s in _VERIFY_STATUSES]


@app.get("/api/verify/failures")
def api_verify_failures(
    statuses: Optional[str] = Query(
        None,
        description="Comma-separated statuses to include: fail, regenerated. Default: both.",
    ),
):
    """Photos flagged by verify.

    Returns rows where verification_status is in the requested set AND
    hallucination_flags is not null (i.e. the verifier actually flagged
    something). By default returns both 'fail' (not regenerated — DB still
    has bad description/tags) and 'regenerated' (description was rewritten
    but you may still want to review).
    """
    allowed = _parse_statuses(statuses)
    if not allowed:
        return {"photos": [], "count": 0, "statuses": []}

    placeholders = ",".join("?" * len(allowed))
    with _get_db() as db:
        rows = db.conn.execute(
            f"""SELECT id, filepath, filename, verified_at, verification_status,
                       hallucination_flags, description, tags
                FROM photos
                WHERE verification_status IN ({placeholders})
                  AND hallucination_flags IS NOT NULL
                ORDER BY
                  CASE verification_status WHEN 'fail' THEN 0 ELSE 1 END,
                  verified_at DESC""",
            allowed,
        ).fetchall()
        photos = []
        for r in rows:
            flags = None
            if r["hallucination_flags"]:
                try:
                    flags = json.loads(r["hallucination_flags"])
                except (ValueError, TypeError):
                    flags = None
            photos.append({
                "id": r["id"],
                "filepath": r["filepath"],
                "filename": r["filename"],
                "verified_at": r["verified_at"],
                "status": r["verification_status"],
                "description": r["description"],
                "hallucination_flags": flags,
                "thumbnail": f"/api/photos/{r['id']}/thumbnail",
            })
    return {"photos": photos, "count": len(photos), "statuses": allowed}


@app.post("/api/verify/failures/collect")
def api_verify_failures_collect(body: dict):
    """Gather verify failures and/or regenerated-with-flags photos into a collection.

    Body:
      - name (str, optional): create a new collection with this name
      - collection_id (int, optional): append to an existing collection
      - statuses (list[str], optional): any of 'fail', 'regenerated'. Defaults to ['fail'].
    """
    name = (body.get("name") or "").strip()
    collection_id = body.get("collection_id")
    if not name and not collection_id:
        return JSONResponse({"error": "name or collection_id is required"}, status_code=400)

    raw_statuses = body.get("statuses") or ["fail"]
    if isinstance(raw_statuses, str):
        raw_statuses = [s.strip() for s in raw_statuses.split(",") if s.strip()]
    allowed = [s for s in raw_statuses if s in _VERIFY_STATUSES]
    if not allowed:
        return JSONResponse(
            {"error": f"statuses must be one or more of {list(_VERIFY_STATUSES)}"},
            status_code=400,
        )

    placeholders = ",".join("?" * len(allowed))
    with _get_db() as db:
        rows = db.conn.execute(
            f"""SELECT id FROM photos
                WHERE verification_status IN ({placeholders})
                  AND hallucination_flags IS NOT NULL""",
            allowed,
        ).fetchall()
        photo_ids = [r["id"] for r in rows]
        if not photo_ids:
            return JSONResponse(
                {"error": f"No photos found for statuses {allowed}"}, status_code=404
            )

        if collection_id:
            collection = db.get_collection(collection_id)
            if not collection:
                return JSONResponse({"error": "Collection not found"}, status_code=404)
        else:
            existing = db.get_collection_by_name(name)
            if existing:
                return JSONResponse(
                    {"error": f"Collection '{name}' already exists"}, status_code=409
                )
            collection_id = db.create_collection(
                name,
                description="Auto-generated from verify ("
                + ", ".join(allowed) + ")",
            )
            collection = db.get_collection(collection_id)

        added = db.add_photos_to_collection(collection_id, photo_ids)
    return {"collection": collection, "added": added, "total_failures": len(photo_ids)}


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

@app.get("/api/collections")
def api_list_collections():
    """List all collections with photo counts."""
    with _get_db() as db:
        collections = db.list_collections()
        # Add cover thumbnail path
        for c in collections:
            cover_id = c.get("effective_cover_photo_id")
            if cover_id:
                c["cover_thumbnail"] = f"/api/photos/{cover_id}/thumbnail"
            else:
                c["cover_thumbnail"] = None
    return {"collections": collections}


@app.post("/api/collections")
def api_create_collection(body: dict):
    """Create a new collection."""
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    description = body.get("description")
    photo_ids = body.get("photo_ids", [])

    with _get_db() as db:
        existing = db.get_collection_by_name(name)
        if existing:
            return JSONResponse({"error": f"Collection '{name}' already exists"}, status_code=409)

        coll_id = db.create_collection(name, description)
        added = 0
        if photo_ids:
            added = db.add_photos_to_collection(coll_id, photo_ids)

        collection = db.get_collection(coll_id)

    return {"collection": collection, "photos_added": added}


@app.get("/api/collections/{collection_id}")
def api_get_collection(collection_id: int):
    """Get a collection with its photos."""
    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            return JSONResponse({"error": "Collection not found"}, status_code=404)

        photos = db.get_collection_photos(collection_id)
        # Add thumbnail paths and stack info
        for p in photos:
            p["thumbnail"] = f"/api/photos/{p['id']}/thumbnail"
            stack_info = db.get_photo_stack(p["id"])
            if stack_info:
                p["stack_id"] = stack_info["stack_id"]
                p["stack_is_top"] = stack_info["is_top"]
                p["stack_count"] = stack_info["member_count"]
        collection["photo_count"] = len(photos)

    return {"collection": collection, "photos": photos}


@app.put("/api/collections/{collection_id}")
def api_update_collection(collection_id: int, body: dict):
    """Update collection name and/or description."""
    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            return JSONResponse({"error": "Collection not found"}, status_code=404)

        if "name" in body:
            name = body["name"].strip()
            if name and name != collection["name"]:
                existing = db.get_collection_by_name(name)
                if existing:
                    return JSONResponse({"error": f"Collection '{name}' already exists"}, status_code=409)
                db.rename_collection(collection_id, name)
        if "description" in body:
            db.update_collection_description(collection_id, body["description"])
        if "cover_photo_id" in body:
            db.set_collection_cover(collection_id, body["cover_photo_id"])

        collection = db.get_collection(collection_id)
    return {"collection": collection}


@app.delete("/api/collections/{collection_id}")
def api_delete_collection(collection_id: int):
    """Delete a collection."""
    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            return JSONResponse({"error": "Collection not found"}, status_code=404)
        db.delete_collection(collection_id)
    return {"ok": True}


@app.post("/api/collections/{collection_id}/photos")
def api_add_to_collection(collection_id: int, body: dict):
    """Add photos to a collection."""
    photo_ids = body.get("photo_ids", [])
    if not photo_ids:
        return JSONResponse({"error": "photo_ids is required"}, status_code=400)
    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            return JSONResponse({"error": "Collection not found"}, status_code=404)
        added = db.add_photos_to_collection(collection_id, photo_ids)
    return {"added": added}


@app.post("/api/collections/{collection_id}/photos/remove")
def api_remove_from_collection(collection_id: int, body: dict):
    """Remove photos from a collection."""
    photo_ids = body.get("photo_ids", [])
    if not photo_ids:
        return JSONResponse({"error": "photo_ids is required"}, status_code=400)
    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            return JSONResponse({"error": "Collection not found"}, status_code=404)
        removed = db.remove_photos_from_collection(collection_id, photo_ids)
    return {"removed": removed}


@app.get("/api/photos/{photo_id}/collections")
def api_photo_collections(photo_id: int):
    """Get all collections a photo belongs to."""
    with _get_db() as db:
        collections = db.get_photo_collections(photo_id)
    return {"collections": collections}


# ---------------------------------------------------------------------------
# Photo stacks (burst/bracket groups)
# ---------------------------------------------------------------------------


@app.get("/api/stacks")
def api_list_stacks():
    """List all stacks with member counts."""
    with _get_db() as db:
        stacks = db.get_all_stacks()
    return {"stacks": stacks}


@app.post("/api/stacks/detect")
def api_detect_stacks(data: dict = None):
    """Run burst/bracket stacking across the library.

    Body (all optional, defaults match the CLI):
      time_window_sec     (float, default 5.0)
      clip_threshold      (float, default 0.05)
      max_stack_span_sec  (float, default 10.0)
      clear               (bool,  default false) — wipe existing stacks first
      dry_run             (bool,  default false) — detect but don't save

    Blocks for the duration of detection (no progress streaming). Returns
    summary counts so the status page can report what happened.
    """
    import time
    from .stacking import run_stacking

    data = data or {}
    time_window_sec = float(data.get("time_window_sec", 5.0))
    clip_threshold = float(data.get("clip_threshold", 0.05))
    max_stack_span_sec = float(data.get("max_stack_span_sec", 10.0))
    clear = bool(data.get("clear", False))
    dry_run = bool(data.get("dry_run", False))

    if time_window_sec <= 0 or time_window_sec > 3600:
        raise HTTPException(400, "time_window_sec must be in (0, 3600]")
    if clip_threshold <= 0 or clip_threshold >= 2.0:
        raise HTTPException(400, "clip_threshold must be in (0, 2.0)")
    if max_stack_span_sec < time_window_sec:
        raise HTTPException(400, "max_stack_span_sec must be ≥ time_window_sec")

    started = time.monotonic()
    with _get_db() as db:
        if clear and not dry_run:
            db.clear_stacks()
        stacks = run_stacking(
            db,
            time_window_sec=time_window_sec,
            clip_threshold=clip_threshold,
            max_stack_span_sec=max_stack_span_sec,
            dry_run=dry_run,
        )

    duration_seconds = round(time.monotonic() - started, 2)
    return {
        "ok": True,
        "stacks_created": len(stacks),
        "photos_stacked": sum(len(s) for s in stacks),
        "cleared": clear and not dry_run,
        "dry_run": dry_run,
        "duration_seconds": duration_seconds,
        "params": {
            "time_window_sec": time_window_sec,
            "clip_threshold": clip_threshold,
            "max_stack_span_sec": max_stack_span_sec,
        },
    }


@app.post("/api/stacks/detect/stream")
async def api_detect_stacks_stream(request: Request):
    """SSE variant of /api/stacks/detect — streams phase-by-phase progress.

    Body matches the blocking endpoint. Emits events of the form:
      {"type": "progress", "phase": "scan"|"load_embeddings"|"pairs"|"group"|"save", ...}
      {"type": "done",      "stacks_created": N, "photos_stacked": M, ...}
      {"type": "cancelled", "message": "..."}
      {"type": "fatal",     "message": "..."}

    Client disconnect (AbortController, tab close) flips a threading.Event
    that stacking.py checks inside its hot loops — work stops within ~1000
    photo iterations.
    """
    import asyncio
    import threading
    import time
    from .stacking import run_stacking

    try:
        data = await request.json()
    except Exception:
        data = {}
    data = data or {}

    time_window_sec = float(data.get("time_window_sec", 5.0))
    clip_threshold = float(data.get("clip_threshold", 0.05))
    max_stack_span_sec = float(data.get("max_stack_span_sec", 10.0))
    clear = bool(data.get("clear", False))
    dry_run = bool(data.get("dry_run", False))

    if time_window_sec <= 0 or time_window_sec > 3600:
        raise HTTPException(400, "time_window_sec must be in (0, 3600]")
    if clip_threshold <= 0 or clip_threshold >= 2.0:
        raise HTTPException(400, "clip_threshold must be in (0, 2.0)")
    if max_stack_span_sec < time_window_sec:
        raise HTTPException(400, "max_stack_span_sec must be ≥ time_window_sec")

    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()

    def _emit(event: dict):
        asyncio.run_coroutine_threadsafe(aqueue.put(event), loop)

    def _on_progress(event: dict):
        event = dict(event)
        event["type"] = "progress"
        _emit(event)

    def _should_abort() -> bool:
        return cancel_event.is_set()

    def run():
        started = time.monotonic()
        try:
            _emit({
                "type": "start",
                "params": {
                    "time_window_sec": time_window_sec,
                    "clip_threshold": clip_threshold,
                    "max_stack_span_sec": max_stack_span_sec,
                },
                "clear": clear,
                "dry_run": dry_run,
            })
            with _get_db() as db:
                if clear and not dry_run:
                    db.clear_stacks()
                    _emit({"type": "progress", "phase": "cleared"})
                stacks = run_stacking(
                    db,
                    time_window_sec=time_window_sec,
                    clip_threshold=clip_threshold,
                    max_stack_span_sec=max_stack_span_sec,
                    dry_run=dry_run,
                    on_progress=_on_progress,
                    should_abort=_should_abort,
                )
            _emit({
                "type": "done",
                "stacks_created": len(stacks),
                "photos_stacked": sum(len(s) for s in stacks),
                "cleared": clear and not dry_run,
                "dry_run": dry_run,
                "duration_seconds": round(time.monotonic() - started, 2),
            })
        except InterruptedError:
            _emit({"type": "cancelled", "message": "Stacking cancelled"})
        except Exception as exc:
            logger.exception("STACKING stream failed")
            _emit({"type": "fatal", "message": str(exc)})

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("STACKING  client disconnected — cancelling")
                    cancel_event.set()
                    return
                try:
                    event = await asyncio.wait_for(aqueue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "fatal", "cancelled"):
                    return
        finally:
            cancel_event.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/stacks/{stack_id}")
def api_get_stack(stack_id: int):
    """Get a stack with all member photos."""
    with _get_db() as db:
        stack = db.get_stack(stack_id)
    if not stack:
        raise HTTPException(404, "Stack not found")
    return stack


@app.put("/api/stacks/{stack_id}/top")
def api_set_stack_top(stack_id: int, body: dict):
    """Promote a photo to the top of a stack."""
    photo_id = body.get("photo_id")
    if not photo_id:
        raise HTTPException(400, "photo_id required")
    with _get_db() as db:
        stack = db.get_stack(stack_id)
        if not stack:
            raise HTTPException(404, "Stack not found")
        member_ids = {m["id"] for m in stack["members"]}
        if photo_id not in member_ids:
            raise HTTPException(400, "Photo is not a member of this stack")
        db.set_stack_top(stack_id, photo_id)
    return {"ok": True}


@app.delete("/api/stacks/{stack_id}")
def api_delete_stack(stack_id: int):
    """Dissolve a stack (keep the photos)."""
    with _get_db() as db:
        stack = db.get_stack(stack_id)
        if not stack:
            raise HTTPException(404, "Stack not found")
        db.delete_stack(stack_id)
    return {"ok": True}


@app.post("/api/photos/{photo_id}/unstack")
def api_unstack_photo(photo_id: int):
    """Remove a photo from its stack."""
    with _get_db() as db:
        info = db.get_photo_stack(photo_id)
        if not info:
            raise HTTPException(400, "Photo is not in a stack")
        db.unstack_photo(photo_id)
    return {"ok": True}


@app.post("/api/stacks/{stack_id}/add")
def api_add_to_stack(stack_id: int, body: dict):
    """Add a photo to an existing stack."""
    photo_id = body.get("photo_id")
    if not photo_id:
        raise HTTPException(400, "photo_id required")
    with _get_db() as db:
        stack = db.get_stack(stack_id)
        if not stack:
            raise HTTPException(404, "Stack not found")
        # Verify photo exists
        photo = db.conn.execute("SELECT id FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not photo:
            raise HTTPException(404, "Photo not found")
        db.add_to_stack(stack_id, photo_id)
    return {"ok": True}


@app.get("/api/photos/{photo_id}/nearby-stacks")
def api_nearby_stacks(photo_id: int):
    """Find stacks with photos taken near the same time as this photo.

    Returns up to 5 stacks whose members were taken within 60 seconds
    of this photo's date_taken, ordered by time proximity.
    """
    with _get_db() as db:
        photo = db.conn.execute(
            "SELECT id, date_taken FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
        if not photo:
            raise HTTPException(404, "Photo not found")
        if not photo["date_taken"]:
            return {"stacks": []}
        # Find stacks whose members have date_taken within 60s of this photo
        rows = db.conn.execute("""
            SELECT DISTINCT ps.id AS stack_id,
                   COUNT(sm2.photo_id) AS member_count,
                   MIN(ABS(julianday(p2.date_taken) - julianday(?))) * 86400 AS min_distance_sec,
                   MAX(CASE WHEN sm2.is_top = 1 THEN p2.filename END) AS top_filename,
                   MAX(CASE WHEN sm2.is_top = 1 THEN sm2.photo_id END) AS top_photo_id
            FROM stack_members sm
            JOIN photos p ON p.id = sm.photo_id
            JOIN photo_stacks ps ON ps.id = sm.stack_id
            JOIN stack_members sm2 ON sm2.stack_id = ps.id
            JOIN photos p2 ON p2.id = sm2.photo_id
            WHERE ABS(julianday(p.date_taken) - julianday(?)) * 86400 < 60
              AND sm.photo_id != ?
            GROUP BY ps.id
            ORDER BY min_distance_sec ASC
            LIMIT 5
        """, (photo["date_taken"], photo["date_taken"], photo_id)).fetchall()
        return {
            "stacks": [
                {
                    "stack_id": r["stack_id"],
                    "member_count": r["member_count"],
                    "top_filename": r["top_filename"],
                    "top_photo_id": r["top_photo_id"],
                    "distance_sec": round(r["min_distance_sec"], 1),
                }
                for r in rows
            ]
        }


# ---------------------------------------------------------------------------
# Google Photos integration
# ---------------------------------------------------------------------------

@app.get("/api/google/status")
def api_google_status():
    """Return the current Google Photos connection status.

    Returns:
      configured: bool — client_secret.json is present
      authenticated: bool — valid tokens are stored
    """
    from .google_photos import is_configured, is_authenticated
    return {
        "configured": is_configured(_db_path),
        "authenticated": is_authenticated(_db_path),
    }


@app.get("/api/google/authorize")
def api_google_authorize(port: Optional[str] = Query(None, description="Browser port, used to build localhost redirect URI")):
    """Return the Google OAuth2 authorization URL.

    The redirect_uri is always localhost-based so it works with Desktop app
    credentials (which allow http://localhost without pre-registration).
    The port parameter lets the frontend pass its own port so the redirect
    lands on the right server instance.

    For NAS users where localhost doesn't route back to the NAS, the frontend
    also shows a manual code-paste field as a fallback.
    """
    from .google_photos import get_authorization_url

    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI")
    if not redirect_uri:
        # Use localhost + caller's port so the redirect URI matches what's
        # allowed for Desktop app credentials (any http://localhost port).
        p = port or "8000"
        redirect_uri = f"http://localhost:{p}/api/google/callback"

    try:
        url = get_authorization_url(_db_path, redirect_uri=redirect_uri)
        return {"auth_url": url, "redirect_uri": redirect_uri}
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/google/exchange-code")
def api_google_exchange_code(body: dict):
    """Manually exchange an authorization code for tokens.

    Used when the automatic OAuth redirect doesn't land on the server
    (e.g. accessing a NAS from a different machine where localhost:PORT
    routes to the user's own computer rather than the NAS).

    Body: {"code": "4/0AX...", "redirect_uri": "http://localhost:8000/api/google/callback"}
    The redirect_uri must exactly match the one used in the authorization URL.
    """
    from .google_photos import exchange_code

    code = body.get("code", "").strip()
    redirect_uri = body.get("redirect_uri", "").strip()

    if not code:
        raise HTTPException(400, "code is required")
    if not redirect_uri:
        raise HTTPException(400, "redirect_uri is required")

    try:
        exchange_code(_db_path, code, redirect_uri=redirect_uri)
    except Exception as exc:
        raise HTTPException(400, f"Code exchange failed: {exc}")

    return {"ok": True}


@app.get("/api/google/callback")
def api_google_callback(
    code: Optional[str] = Query(None),
    error: Optional[str] = Query(None),
):
    """OAuth2 callback — Google redirects here after user grants access.

    Exchanges the authorization code for tokens, saves them, then serves
    a small HTML page telling the user they're connected.
    """
    from .google_photos import exchange_code

    if error:
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Google Photos — Error</title></head>
<body style="font-family:sans-serif;padding:40px;background:#0f0f0f;color:#e8e8e8">
<h2 style="color:#f87171">Authorization failed</h2>
<p>Google returned an error: <code>{error}</code></p>
<p><a href="/collections" style="color:#4a9eff">← Back to Collections</a></p>
</body></html>""")

    if not code:
        raise HTTPException(400, "Missing authorization code")

    # Reconstruct redirect_uri from the request's own host so it matches
    # what was sent in the authorization URL.
    redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI")

    try:
        exchange_code(_db_path, code, redirect_uri=redirect_uri)
    except Exception as exc:
        return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>Google Photos — Error</title></head>
<body style="font-family:sans-serif;padding:40px;background:#0f0f0f;color:#e8e8e8">
<h2 style="color:#f87171">Authorization failed</h2>
<p>{exc}</p>
<p><a href="/collections" style="color:#4a9eff">← Back to Collections</a></p>
</body></html>""")

    return HTMLResponse("""<!DOCTYPE html>
<html><head><title>Google Photos — Connected</title></head>
<body style="font-family:sans-serif;padding:40px;background:#0f0f0f;color:#e8e8e8">
<h2 style="color:#4ade80">✓ Connected to Google Photos</h2>
<p>Your account is now linked. You can close this tab and return to Photo Search.</p>
<p><a href="/collections" style="color:#4a9eff">← Back to Collections</a></p>
<script>
  // Auto-close after 3 seconds if this was opened as a popup
  if (window.opener) {
    window.opener.postMessage('google_photos_connected', '*');
    setTimeout(function() { window.close(); }, 2000);
  }
</script>
</body></html>""")


@app.delete("/api/google/disconnect")
def api_google_disconnect():
    """Revoke Google Photos access and delete stored tokens."""
    from .google_photos import revoke_token
    revoke_token(_db_path)
    return {"ok": True}


@app.post("/api/google/albums")
def api_google_create_album(body: dict):
    """Create a Google Photos album.

    Body: {"title": "My Album"}
    Returns: {"album_id": "...", "title": "..."}
    """
    from .google_photos import create_album

    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(400, "title is required")

    try:
        album_id = create_album(_db_path, title)
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Album creation failed: {exc}")

    return {"album_id": album_id, "title": title}


@app.post("/api/google/upload-status")
async def api_google_upload_status(body: dict):
    """Check which photos have already been uploaded to a given album.

    Body:
      photo_ids: list[int]
      album_id: str

    Returns:
      { uploaded: {photo_id: filename, ...}, not_uploaded: {photo_id: filename, ...} }
    """
    photo_ids = body.get("photo_ids", [])
    album_id = body.get("album_id", "")
    if not photo_ids or not album_id:
        raise HTTPException(400, "photo_ids and album_id are required")

    with _get_db() as db:
        ledger_rows = db.conn.execute(
            "SELECT filepath, media_item_id FROM google_photos_uploads WHERE album_id = ?",
            (album_id,),
        ).fetchall()
        ledger = {row[0] for row in ledger_rows if row[1]}

        uploaded = {}
        not_uploaded = {}
        for pid in photo_ids:
            photo = db.get_photo(pid)
            if not photo:
                continue
            fp = db.resolve_filepath(photo.get("filepath", ""))
            fname = os.path.basename(fp)
            if fp in ledger:
                uploaded[pid] = fname
            else:
                not_uploaded[pid] = fname

    return {"uploaded": uploaded, "not_uploaded": not_uploaded}


@app.post("/api/google/upload")
async def api_google_upload(body: dict, request: Request):
    """Upload photos to Google Photos, streaming per-file progress as SSE.

    Body:
      photo_ids: list[int] — IDs of photos to upload
      album_title: str (optional) — creates a new album with this title
      album_id: str (optional) — adds to an existing album (skips creation)
      collection_id: int (optional) — if provided, the album ID is saved back
                      to the collection for future "add to existing" uploads
      include_description: bool (default true) — use description as caption
      force_reupload: bool (default false) — ignore ledger, re-upload all
      force_reupload_ids: list[int] (optional) — specific photo IDs to force
                          re-upload even if they exist in the ledger

    Streams SSE events:
      {"type": "progress", "filename": str, "status": "uploaded"|"readded"|"error",
       "done": int, "total": int, "error": str|None}
      {"type": "done", "uploaded": int, "readded": int, "errors": int,
       "total": int, "album_id": str, "album_title": str}
      {"type": "fatal", "message": str}   — on unrecoverable error
    """
    import asyncio
    import threading
    from .google_photos import create_album, upload_photos, batch_add_to_album, refresh_access_token

    photo_ids = body.get("photo_ids", [])
    if not photo_ids:
        raise HTTPException(400, "photo_ids is required")

    album_title = body.get("album_title", "").strip()
    album_id = body.get("album_id", "").strip() or None
    collection_id = body.get("collection_id")
    include_description = body.get("include_description", True)
    force_reupload = body.get("force_reupload", False)
    force_reupload_ids = set(body.get("force_reupload_ids", []))

    # Verify authenticated before opening the stream
    try:
        token = refresh_access_token(_db_path)
        if not token:
            raise HTTPException(401, "Not authenticated with Google Photos — please authorize first.")
    except RuntimeError as exc:
        raise HTTPException(401, str(exc))

    # Create album if title was provided and no existing album_id
    if album_title and not album_id:
        try:
            album_id = create_album(_db_path, album_title)
            logger.info("GOOGLE ALBUM CREATED  id=%s  title=%s", album_id, album_title)
        except Exception as exc:
            raise HTTPException(500, f"Album creation failed: {exc}")

    # Partition photos into two groups:
    #   readd_items — already uploaded to this album (have a media_item_id in ledger)
    #   records     — never uploaded, or force_reupload requested; need full upload
    with _get_db() as db:
        if album_id and not force_reupload:
            ledger_rows = db.conn.execute(
                "SELECT filepath, media_item_id FROM google_photos_uploads WHERE album_id = ?",
                (album_id,)
            ).fetchall()
            ledger = {row[0]: row[1] for row in ledger_rows}
        else:
            ledger = {}

        records = []
        readd_items = []

        for pid in photo_ids:
            photo = db.get_photo(pid)
            if not photo:
                continue
            photo = dict(photo)
            photo["_resolved_filepath"] = db.resolve_filepath(photo.get("filepath", ""))
            fp = photo["_resolved_filepath"]
            if fp in ledger and ledger[fp] and pid not in force_reupload_ids:
                readd_items.append((photo, ledger[fp]))
            else:
                records.append(photo)

    if not records and not readd_items:
        raise HTTPException(404, "No valid photos found for the given IDs")

    grand_total = len(readd_items) + len(records)
    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()

    def _emit(event: dict):
        """Thread-safe event push onto the async queue."""
        asyncio.run_coroutine_threadsafe(aqueue.put(event), loop)

    # Build an ordered filename list for all files (readd + new) for the "start" event
    all_filenames = (
        [p.get("filename", "") for p, _ in readd_items] +
        [p.get("filename", "") for p in records]
    )
    # record_map keyed by filename for quick lookup during per-file ledger writes
    record_map = {p["filename"]: p for p in records}

    def run_upload():
        """Runs in a background thread; pushes SSE events via _emit."""
        results = []
        done_count = [0]  # mutable so nested callbacks can increment

        try:
            # Announce all filenames upfront so the UI can show the full pending list
            _emit({"type": "start", "filenames": all_filenames, "total": grand_total})

            # --- Step A: re-add already-uploaded photos (no bytes) ---
            if readd_items and album_id:
                media_ids = [mid for _, mid in readd_items]
                try:
                    readd_result = batch_add_to_album(_db_path, album_id, media_ids)
                    logger.info("GOOGLE READD  attempted=%d  ok=%d", len(media_ids), readd_result["added"])
                    for photo, mid in readd_items:
                        if cancel_event.is_set():
                            break
                        done_count[0] += 1
                        fname = photo.get("filename", "")
                        results.append({"filename": fname, "status": "readded",
                                        "error": None, "media_item_id": mid})
                        logger.info("GOOGLE UPLOAD  %d/%d  ♻  %s  (re-synced, no upload needed)",
                                    done_count[0], grand_total, fname)
                        _emit({"type": "progress", "filename": fname,
                               "status": "readded", "done": done_count[0], "total": grand_total, "error": None})
                except Exception as exc:
                    logger.warning("GOOGLE READD  failed: %s", exc)
                    for photo, mid in readd_items:
                        if cancel_event.is_set():
                            break
                        done_count[0] += 1
                        fname = photo.get("filename", "")
                        results.append({"filename": fname, "status": "error",
                                        "error": str(exc), "media_item_id": mid})
                        _emit({"type": "progress", "filename": fname,
                               "status": "error", "done": done_count[0], "total": grand_total, "error": str(exc)})

            # --- Step B: full upload for new photos ---
            if cancel_event.is_set():
                _emit({"type": "cancelled", "message": "Upload cancelled"})
                return

            if records:
                def _on_begin(queued: int, total: int, filename: str):
                    """Called just before raw bytes are POSTed for a file."""
                    if cancel_event.is_set():
                        raise InterruptedError("Upload cancelled by client")
                    overall_queued = len(readd_items) + queued + 1
                    logger.info("GOOGLE UPLOAD  %d/%d  ↑  %s  (sending bytes…)",
                                overall_queued, grand_total, filename)
                    _emit({"type": "begin", "filename": filename,
                           "queued": overall_queued, "total": grand_total})

                def _on_bytes_done(queued: int, total: int, filename: str):
                    """Called right after raw bytes finish uploading for a file."""
                    if cancel_event.is_set():
                        raise InterruptedError("Upload cancelled by client")
                    overall_queued = len(readd_items) + queued
                    logger.info("GOOGLE UPLOAD  %d/%d  ☁  %s  (bytes received, awaiting confirmation)",
                                overall_queued, grand_total, filename)
                    _emit({"type": "bytes_sent", "filename": filename,
                           "queued": overall_queued, "total": grand_total})

                def _on_progress(done: int, total: int, filename: str,
                                  status: str = "uploaded", error: str = None,
                                  media_item_id: str = None):
                    if cancel_event.is_set():
                        raise InterruptedError("Upload cancelled by client")
                    overall = len(readd_items) + done
                    if status == "uploaded":
                        logger.info("GOOGLE UPLOAD  %d/%d  ✓  %s", overall, grand_total, filename)
                        # Write to ledger immediately so a cancel doesn't lose this file
                        if album_id and media_item_id:
                            try:
                                photo = record_map.get(filename)
                                if photo:
                                    with _get_db() as db:
                                        db.record_upload(
                                            album_id, photo.get("id"),
                                            photo.get("_resolved_filepath", photo.get("filepath", "")),
                                            media_item_id,
                                        )
                            except Exception as exc:
                                logger.warning("GOOGLE UPLOAD  ledger write failed for %s: %s", filename, exc)
                    else:
                        logger.warning("GOOGLE UPLOAD  %d/%d  ✗  %s  error=%s",
                                       overall, grand_total, filename, error)
                    _emit({"type": "progress", "filename": filename, "status": status,
                           "done": overall, "total": grand_total, "error": error})

                try:
                    upload_results = upload_photos(
                        _db_path, records,
                        album_id=album_id or None,
                        include_description=include_description,
                        progress_callback=_on_progress,
                        begin_callback=_on_begin,
                        bytes_done_callback=_on_bytes_done,
                    )
                    results.extend(upload_results)
                except InterruptedError:
                    _emit({"type": "cancelled", "message": "Upload cancelled"})
                    return
                except RuntimeError as exc:
                    _emit({"type": "fatal", "message": f"Auth error: {exc}"})
                    return
                except Exception as exc:
                    _emit({"type": "fatal", "message": f"Upload failed: {exc}"})
                    return

            n_uploaded = sum(1 for r in results if r["status"] == "uploaded")
            n_readded  = sum(1 for r in results if r["status"] == "readded")
            n_errors   = sum(1 for r in results if r["status"] == "error")

            logger.info("GOOGLE UPLOAD SUMMARY  total=%d  uploaded=%d  readded=%d  errors=%d  album=%s",
                        len(results), n_uploaded, n_readded, n_errors, album_id)

            # Save album link to collection (ledger entries already written per-file above)
            if album_id:
                try:
                    with _get_db() as db:
                        if collection_id and (n_uploaded > 0 or n_readded > 0):
                            db.set_collection_google_album(collection_id, album_id, album_title)
                except Exception as exc:
                    logger.warning("GOOGLE UPLOAD  failed to update collection album link: %s", exc)

            _emit({"type": "done", "uploaded": n_uploaded, "readded": n_readded,
                   "errors": n_errors, "total": len(results),
                   "album_id": album_id, "album_title": album_title})

        except Exception as exc:
            logger.exception("GOOGLE UPLOAD  unexpected error")
            _emit({"type": "fatal", "message": str(exc)})

    # Launch upload in background thread
    thread = threading.Thread(target=run_upload, daemon=True)
    thread.start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("GOOGLE UPLOAD  client disconnected — cancelling")
                    cancel_event.set()
                    return

                try:
                    event = await asyncio.wait_for(aqueue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "fatal", "cancelled"):
                    return
        finally:
            cancel_event.set()  # ensure thread stops even if generator is abandoned

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind a proxy
        },
    )


# ---------------------------------------------------------------------------
# Serve the frontend (production mode — built files)
# ---------------------------------------------------------------------------

_frontend_dir = Path(__file__).parent.parent / "frontend" / "dist"

if _frontend_dir.exists():
    # Serve static assets if the directory exists (for npm-built frontends)
    _assets_dir = _frontend_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    @app.get("/")
    def serve_index():
        """Serve the frontend index.html."""
        index = _frontend_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Frontend not found</h1>")

    @app.get("/review")
    def serve_review():
        """Serve the shoot review page."""
        review = _frontend_dir / "review.html"
        if review.exists():
            return HTMLResponse(review.read_text())
        return HTMLResponse("<h1>Review page not found</h1>")

    @app.get("/faces")
    def serve_faces():
        """Serve the faces gallery page."""
        faces_page = _frontend_dir / "faces.html"
        if faces_page.exists():
            return HTMLResponse(faces_page.read_text())
        return HTMLResponse("<h1>Faces page not found</h1>")

    @app.get("/merges")
    def serve_merges():
        """Serve the merge-review page (M18 Phase B.0)."""
        page = _frontend_dir / "merges.html"
        if page.exists():
            return HTMLResponse(page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Merges page not found</h1>")

    @app.get("/collections")
    def serve_collections():
        """Serve the collections page."""
        coll_page = _frontend_dir / "collections.html"
        if coll_page.exists():
            return HTMLResponse(coll_page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Collections page not found</h1>")

    @app.get("/status")
    def serve_status():
        """Serve the indexing status page."""
        status_page = _frontend_dir / "status.html"
        if status_page.exists():
            return HTMLResponse(status_page.read_text())
        return HTMLResponse("<h1>Status page not found</h1>")

    @app.get("/map")
    def serve_map():
        """Serve the map view page."""
        page = _frontend_dir / "map.html"
        if page.exists():
            return HTMLResponse(page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Map page not found</h1>")

    @app.get("/geotag")
    def serve_geotag():
        """Serve the manual bulk-geotag page."""
        page = _frontend_dir / "geotag.html"
        if page.exists():
            return HTMLResponse(page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Geotag page not found</h1>")

    @app.get("/collections/{collection_id}")
    def serve_collection_detail(collection_id: int):
        """Serve collection detail page (same HTML, JS reads id from URL)."""
        coll_page = _frontend_dir / "collections.html"
        if coll_page.exists():
            return HTMLResponse(coll_page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Collections page not found</h1>")

    @app.get("/{path:path}")
    def serve_frontend(path: str = ""):
        """Serve static files or fall back to index.html for SPA routing."""
        # Try serving as a static file first
        file_path = _frontend_dir / path
        if file_path.is_file():
            # Prevent stale JS/HTML on deploy — browsers must revalidate
            headers = {}
            if file_path.suffix in (".js", ".html", ".css"):
                headers["Cache-Control"] = "no-cache"
            return FileResponse(str(file_path), headers=headers)
        # Fall back to index.html for SPA routing
        index = _frontend_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Frontend not found</h1>")

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

from fastapi import FastAPI, Query, HTTPException, Response, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .db import PhotoDB
from .aesthetics import aesthetics_from_row as _aesthetics_from_row
from .worker_api import (
    router as worker_router,
    configure as configure_worker,
    begin_shutdown as worker_begin_shutdown,
    is_shutting_down as worker_is_shutting_down,
)
from .admin_api import router as admin_router

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="local-photo-search", version="0.1.0")
app.include_router(worker_router)
app.include_router(admin_router)
from .vocab_admin import router as vocab_admin_router  # noqa: E402
app.include_router(vocab_admin_router)

# Allow the React dev server (port 5173) during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
def _on_shutdown():
    """Tell worker_api to start 503'ing worker traffic so the drain finishes."""
    worker_begin_shutdown()


@app.middleware("http")
async def _reject_worker_traffic_during_shutdown(request: Request, call_next):
    """Return 503 to worker traffic once shutdown begins, so workers back off
    and uvicorn can finish its in-flight drain instead of accumulating new
    requests. Browser UI paths are unaffected (they 'd see a normal connection
    drop a few seconds later anyway when the container swaps).

    Matched paths:
      - /api/worker/*           (claim-batch / submit-results / renew-claim / etc.)
      - /api/photos/<n>/full    (only workers fetch this; UI uses /thumbnail and /preview)
    """
    if worker_is_shutting_down():
        path = request.url.path
        is_worker_full_fetch = (
            path.startswith("/api/photos/")
            and path.endswith("/full")
        )
        if path.startswith("/api/worker/") or is_worker_full_fetch:
            return JSONResponse(
                {"detail": "server shutting down, retry shortly"},
                status_code=503,
                headers={"Retry-After": "30", "Connection": "close"},
            )
    return await call_next(request)

# Database path — set by the CLI launcher, defaults to cwd
_db_path: str = os.environ.get("PHOTOSEARCH_DB", "photo_index.db")
_photo_root: Optional[str] = os.environ.get("PHOTO_ROOT")

# Replica mode (M26a): when this instance runs off a synced read-replica on a
# machine that has the DB but NOT the original photo files, image routes can't
# generate thumbnails/previews locally. If PHOTOSEARCH_NAS_URL is set, those
# routes fall back to the source-of-truth NAS web API (and cache the rendered
# thumbnail/preview locally so the next view is local). Unset on the NAS itself.
_nas_url: Optional[str] = (os.environ.get("PHOTOSEARCH_NAS_URL") or "").rstrip("/") or None

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


# Photobook builder — curation state lives in a SEPARATE sidecar sqlite file so a
# full replica re-sync (sync-replica.sh atomically swaps _db_path) can't wipe it.
_books_db_path: str = (
    os.environ.get("PHOTOSEARCH_BOOKS_DB")
    or str(Path(_db_path).resolve().parent / "photobooks.db.local")
)


def _get_books():
    """Open the sidecar photobook store (fresh connection per request)."""
    from .book import BookStore
    return BookStore(_books_db_path)


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


def _fetch_from_nas(photo_id: int, kind: str, timeout: float = 30.0) -> bytes:
    """Fetch a rendered asset (thumbnail|preview|full) from the source NAS.

    Used in replica mode (PHOTOSEARCH_NAS_URL set) when this machine has the
    DB but not the original photo file. Raises on any failure so callers can
    map it to a 404/502.
    """
    import urllib.request
    url = f"{_nas_url}/api/photos/{photo_id}/{kind}"
    req = urllib.request.Request(url, headers={"User-Agent": "photosearch-replica"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _nas_json(method: str, path: str, body: Optional[dict] = None,
              timeout: float = 120.0) -> dict:
    """Call a NAS write/compute endpoint (replica mode) and return its JSON.

    Used by the face-edit endpoints so features 4/5/6 work from the replica UI:
    the compute (InsightFace + originals) only exists on the NAS, so the replica
    proxies the mutation there, then mirrors the touched photo's face rows back
    into the local DB. Raises on any non-2xx so callers can surface it."""
    import urllib.request
    import urllib.error
    url = f"{_nas_url}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"User-Agent": "photosearch-replica",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise HTTPException(status_code=e.code, detail=f"NAS: {detail}")


def _cache_bytes_atomic(path: str, data: bytes) -> None:
    """Write bytes to a cache path atomically (tmp + rename)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


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
    min_aesthetic: Optional[float] = Query(None, description="Minimum VLM aesthetic percentile (0-100)"),
    min_technical: Optional[float] = Query(None, description="Minimum Technical Excellence score (1-10)"),
    min_composition: Optional[float] = Query(None, description="Minimum Composition score (1-10)"),
    min_impact: Optional[float] = Query(None, description="Minimum Impact & Storytelling score (1-10)"),
    style_tag: Optional[str] = Query(None, description="Filter by aesthetic style tag (e.g. golden-hour)"),
    min_subject_aesthetic: Optional[float] = Query(None, description="Minimum SUBJECT-crop aesthetic percentile (0-100)"),
    min_day_aesthetic: Optional[float] = Query(None, description="Minimum PER-DAY aesthetic percentile (0-100) — rank among photos taken the same day"),
    sort: str = Query("date_desc", description="Sort order: date_desc, date_asc, quality_desc, aesthetic_desc, subject_aesthetic_desc, relevance"),
    sort_quality: bool = Query(False, description="Legacy: equivalent to sort=quality_desc"),
    text_match: str = Query("all", description="Text matching mode: all, dict, categories, visual, keywords, off"),
    tag_match: Optional[str] = Query(None, deprecated=True, description="Deprecated; use text_match"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    location: Optional[str] = Query(None, description="Filter by location name"),
    match_source: Optional[str] = Query(None, description="Face match type: strict, temporal, or manual"),
    category: Optional[str] = Query(None, description="Filter by exact category match"),
    visual_tag: Optional[str] = Query(None, description="Filter by visual tag"),
    keyword: Optional[str] = Query(None, description="Filter by keyword (substring match)"),
    camera: Optional[str] = Query(None, description="Filter by exact camera_model"),
    tag: Optional[str] = Query(None, deprecated=True, description="Deprecated; use category"),
):
    """Search photos using any combination of criteria."""
    logger.info(
        "SEARCH REQUEST  q=%r  person=%r  color=%r  place=%r  limit=%d  "
        "min_score=%.3f  min_quality=%s  sort_quality=%s  date_from=%s  date_to=%s  location=%r",
        q, person, color, place, limit, min_score, min_quality, sort_quality,
        date_from, date_to, location,
    )

    # Legacy aliases: tag_match → text_match, tag → category.
    if tag_match and (text_match is None or text_match == "all"):
        text_match = {"both": "all", "tags": "categories"}.get(tag_match, tag_match)
    if tag and not category:
        category = tag

    if not any([q, person, color, place, category, visual_tag, keyword, camera,
                min_quality is not None, date_from, date_to, location,
                min_aesthetic is not None, min_technical is not None,
                min_composition is not None, min_impact is not None, style_tag,
                min_subject_aesthetic is not None, min_day_aesthetic is not None,
                sort in ("aesthetic_desc", "subject_aesthetic_desc")]):
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
            text_match=text_match,
            date_from=date_from,
            date_to=date_to,
            location=location,
            match_source=match_source,
            category=category,
            visual_tag=visual_tag,
            keyword=keyword,
            min_aesthetic=min_aesthetic,
            min_technical=min_technical,
            min_composition=min_composition,
            min_impact=min_impact,
            style_tag=style_tag,
            min_subject_aesthetic=min_subject_aesthetic,
            min_day_aesthetic=min_day_aesthetic,
            camera=camera,
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
                "aes_overall": r.get("aes_overall"),
                "aes_overall_pct": r.get("aes_overall_pct"),
                "aes_subject_overall_pct": r.get("aes_subject_overall_pct"),
                "aes_overall_day_pct": r.get("aes_overall_day_pct"),
                "aes_subject_overall_day_pct": r.get("aes_subject_overall_day_pct"),
                "aes_technical": r.get("aes_technical"),
                "aes_composition": r.get("aes_composition"),
                "aes_impact": r.get("aes_impact"),
                "description": r.get("description"),
                "camera_make": r.get("camera_make"),
                "camera_model": r.get("camera_model"),
                "focal_length": r.get("focal_length"),
                "exposure_time": r.get("exposure_time"),
                "f_number": r.get("f_number"),
                "iso": r.get("iso"),
                "image_width": r.get("image_width"),
                "image_height": r.get("image_height"),
                "categories": json.loads(r["categories"]) if r.get("categories") else [],
                "visual_tags": json.loads(r["visual_tags"]) if r.get("visual_tags") else [],
                "keywords": json.loads(r["keywords"]) if r.get("keywords") else [],
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
        # Replica mode: no local original — pull the rendered thumbnail from
        # the NAS and cache it locally so the next view is served from disk.
        if _nas_url:
            try:
                cache_path = os.path.join(_ensure_thumb_dir(), f"{photo_id}_thumb.jpg")
                _cache_bytes_atomic(cache_path, _fetch_from_nas(photo_id, "thumbnail"))
                return FileResponse(
                    cache_path, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, must-revalidate"},
                )
            except Exception as e:
                raise HTTPException(502, f"NAS thumbnail fetch failed: {e}")
        raise HTTPException(404, "Original photo file not found")
    except Exception as e:
        raise HTTPException(500, f"Thumbnail error: {e}")


def _attachment_header(filename: str) -> str:
    """Build a Content-Disposition: attachment header that survives non-ASCII
    filenames (RFC 5987 ``filename*`` plus an ASCII fallback)."""
    from urllib.parse import quote
    ascii_name = (filename or "").encode("ascii", "ignore").decode() or "download"
    ascii_name = ascii_name.replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename or 'download')}"


@app.get("/api/photos/{photo_id}/full")
def api_full_photo(
    photo_id: int,
    download: bool = Query(False, description="Force a file download (attachment) "
                           "with the original filename, instead of inline display."),
):
    """Serve the full-resolution original photo. ``?download=1`` attaches it as a
    downloadable file rather than displaying inline."""
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")

        filename = photo.get("filename") or f"photo_{photo_id}.jpg"
        filepath = db.resolve_filepath(photo.get("filepath", ""))

    headers = {"Cache-Control": "no-cache, must-revalidate"}
    if download:
        headers["Content-Disposition"] = _attachment_header(filename)

    if not filepath or not os.path.exists(filepath):
        # Replica mode: stream the original through from the NAS (not cached —
        # full-res is large and rarely re-viewed).
        if _nas_url:
            try:
                data = _fetch_from_nas(photo_id, "full", timeout=120.0)
                return Response(content=data, media_type="image/jpeg", headers=headers)
            except Exception as e:
                raise HTTPException(502, f"NAS full-photo fetch failed: {e}")
        raise HTTPException(404, "Photo file not found on disk")

    return FileResponse(filepath, media_type="image/jpeg", headers=headers)


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
        # Replica mode: pull + cache the rendered preview from the NAS.
        if _nas_url:
            try:
                cache_path = os.path.join(_ensure_preview_dir(), f"{photo_id}_preview.jpg")
                _cache_bytes_atomic(cache_path, _fetch_from_nas(photo_id, "preview"))
                return FileResponse(
                    cache_path, media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache, must-revalidate"},
                )
            except Exception as e:
                raise HTTPException(502, f"NAS preview fetch failed: {e}")
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

    with _get_db() as db:
        row = db.conn.execute(
            """SELECT f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                      ph.filepath, ph.image_width, ph.image_height
               FROM faces f
               JOIN photos ph ON ph.id = f.photo_id
               WHERE f.id = ?""",
            (face_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Face not found")
        filepath = db.resolve_filepath(row["filepath"])

    if not filepath or not os.path.exists(filepath):
        # Replica mode: no local originals — proxy the rendered crop from the NAS
        # and cache it locally (same pattern as the preview/full endpoints).
        if _nas_url:
            try:
                import urllib.request
                url = f"{_nas_url.rstrip('/')}/api/faces/crop/{face_id}?size={size}"
                with urllib.request.urlopen(url, timeout=30) as r:
                    _cache_bytes_atomic(cache_path, r.read())
                return FileResponse(
                    cache_path, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400"},
                )
            except Exception as e:
                raise HTTPException(502, f"NAS face-crop fetch failed: {e}")
        raise HTTPException(404, "Photo file not found")

    if row["bbox_top"] is None:
        raise HTTPException(400, "Face has no bounding box")

    from photosearch.face_crop import render_face_crops, write_crop_atomic

    bbox = (row["bbox_top"], row["bbox_right"], row["bbox_bottom"], row["bbox_left"])
    crops = render_face_crops(filepath, bbox, row["image_width"], row["image_height"], [size])
    write_crop_atomic(cache_path, crops[size])

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


def _face_filter_photo_ids(db, date_from, date_to, location, q, person, camera=None):
    """Photo-id set matching the given content filters, or None when none are set.

    Reuses search_combined for semantic/person queries (CLIP-ranked, capped);
    falls back to a direct, unbounded date/location/camera SQL scan otherwise.
    Used to restrict /api/faces/groups to clusters/persons appearing in matching
    photos.
    """
    if not any([date_from, date_to, location, q, person, camera]):
        return None
    if q or person:
        from .search import search_combined
        res = search_combined(db, query=q or None, location=location, person=person,
                              date_from=date_from, date_to=date_to, camera=camera,
                              limit=5000, sort="date_desc")
        rows = res[0] if isinstance(res, tuple) else res
        return {r["id"] for r in rows}
    sql, params = "SELECT id FROM photos WHERE 1=1", []
    if date_from:
        sql += " AND substr(date_taken,1,10) >= ?"; params.append(date_from[:10])
    if date_to:
        sql += " AND substr(date_taken,1,10) <= ?"; params.append(date_to[:10])
    if location:
        sql += " AND place_name LIKE ?"; params.append(f"%{location}%")
    if camera:
        sql += " AND camera_model = ?"; params.append(camera)
    return {r["id"] for r in db.conn.execute(sql, params).fetchall()}


@app.get("/api/faces/groups")
def api_face_groups(
    sort: str = Query("similarity"),
    include_singletons: bool = Query(False),
    filter: str = Query("all"),
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    location: str | None = Query(None),
    q: str | None = Query(None),
    person: str | None = Query(None),
    camera: str | None = Query(None),
):
    """List face identities grouped by person or cluster, with filtering and pagination.

    sort: "similarity" clusters visually-similar faces together (O(N²);
          auto-downgraded to "count" when the filtered group count exceeds
          _SIMILARITY_SORT_GROUP_LIMIT). "count" sorts unknowns by face_count desc.
    include_singletons: when false (default), unknown clusters with only one
          face are hidden.
    filter: "all" (named + not-ignored clusters), "named" (persons only),
          "unknown" (not-ignored clusters only), or "ignored" (ignored clusters only).
    date_from/date_to/location/q/person: CONTENT filters — restrict results to
          groups that have a face in photos matching the date range / place /
          semantic (CLIP) query / person. When any are set, groups are ranked by
          how many matching photos they contain, and singletons are shown.
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

        # Optional content filter: restrict to groups whose faces appear in the
        # matching photos. Materialize the matching photo-ids into a temp table
        # and inject an EXISTS-style predicate into the group queries.
        filter_pids = _face_filter_photo_ids(db, date_from, date_to, location, q, person, camera)
        filtering = filter_pids is not None
        if filtering:
            db.conn.execute("CREATE TEMP TABLE IF NOT EXISTS _facefilter (pid INTEGER PRIMARY KEY)")
            db.conn.execute("DELETE FROM _facefilter")
            db.conn.executemany("INSERT OR IGNORE INTO _facefilter(pid) VALUES (?)",
                                [(p,) for p in filter_pids])
        ff_f = " AND f.photo_id IN (SELECT pid FROM _facefilter) " if filtering else ""
        ff_f2 = " AND f2.photo_id IN (SELECT pid FROM _facefilter) " if filtering else ""
        # When filtering, a single matching face makes a group relevant.
        having_n = 1 if filtering else min_face_count

        named_groups: list[dict] = []
        if filter in ("all", "named"):
            named_rows = db.conn.execute(
                f"""SELECT p.id as person_id, p.name,
                          COUNT(DISTINCT f.photo_id) as photo_count,
                          COUNT(f.id) as face_count,
                          (SELECT f2.id FROM faces f2
                           WHERE f2.person_id = p.id AND f2.bbox_top IS NOT NULL {ff_f2}
                           ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                           LIMIT 1) as rep_face_id
                   FROM persons p
                   JOIN faces f ON f.person_id = p.id
                   WHERE 1=1 {ff_f}
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
                f"""SELECT f.cluster_id,
                          COUNT(DISTINCT f.photo_id) as photo_count,
                          COUNT(f.id) as face_count,
                          (SELECT f2.id FROM faces f2
                           WHERE f2.cluster_id = f.cluster_id AND f2.person_id IS NULL
                                 AND f2.bbox_top IS NOT NULL {ff_f2}
                           ORDER BY (f2.bbox_bottom - f2.bbox_top) * (f2.bbox_right - f2.bbox_left) DESC
                           LIMIT 1) as rep_face_id
                   FROM faces f
                   WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL {ff_f}
                   GROUP BY f.cluster_id
                   HAVING face_count >= ?
                   ORDER BY photo_count DESC, face_count DESC""",
                (having_n,),
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

        # Counts over the full (pre-pagination) filtered state so filter chips stay
        # accurate — they also respect the content filter when one is active.
        named_count = db.conn.execute(
            f"""SELECT COUNT(DISTINCT f.person_id) AS n FROM faces f
               WHERE f.person_id IS NOT NULL {ff_f}"""
        ).fetchone()["n"]

        cluster_size_rows = db.conn.execute(
            f"""SELECT f.cluster_id
               FROM faces f
               WHERE f.person_id IS NULL AND f.cluster_id IS NOT NULL {ff_f}
               GROUP BY f.cluster_id
               HAVING COUNT(f.id) >= ?""",
            (having_n,),
        ).fetchall()
        qualifying_cluster_ids = {r["cluster_id"] for r in cluster_size_rows}
        ignored_qualifying = qualifying_cluster_ids & ignored_set
        counts = {
            "named": named_count,
            "unknown": len(qualifying_cluster_ids) - len(ignored_qualifying),
            "ignored": len(ignored_qualifying),
        }

        # When content-filtering, relevance (most matching photos first) beats
        # visual-similarity ordering — surface the groups the query actually hit.
        if filtering:
            groups.sort(key=lambda g: -g.get("photo_count", 0))
            page = groups[offset : offset + limit]
            return {"groups": page, "total": total, "counts": counts,
                    "sort": "relevance", "limit": limit, "offset": offset}

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


@app.get("/api/photos/{photo_id}/mirror-fields")
def api_photo_mirror_fields(photo_id: int):
    """Authoritative per-photo derived fields for the M28 re-run mirror.

    A replica calls this after a pass is re-run on the NAS to apply the exact
    canonical values locally — text/scalar columns verbatim, plus the CLIP
    embedding and face rows — so the local search index reflects the re-run
    without waiting for the nightly sync-replica.sh full pull. See
    photosearch/rerun.py:mirror_photos.
    """
    from .db import _deserialize_float_list, CLIP_DIMENSIONS, FACE_DIMENSIONS
    from .aesthetics import ALL_SUBATTRS, DIMENSIONS
    _aes_cols = [
        "aes_overall", "aes_overall_pct", "aes_technical_iqa", "aes_overall_iqa",
        "aes_style", "aes_style_tags", "aes_model", "aes_scored_at",
        *(f"aes_{s}" for s in ALL_SUBATTRS),
        *(f"aes_{d}" for d in DIMENSIONS),
    ]
    with _get_db() as db:
        row = db.conn.execute(
            f"""SELECT description, categories, visual_tags, keywords, tags,
                      verified_at, verification_status, hallucination_flags,
                      aesthetic_score, aesthetic_concepts, aesthetic_critique,
                      {', '.join(_aes_cols)}
                 FROM photos WHERE id = ?""",
            (photo_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "Photo not found")
        out = {k: row[k] for k in row.keys()}

        # CLIP embedding (None if not yet embedded — the mirror DELETEs the
        # local row so a cleared-but-not-reprocessed pass shows as un-embedded).
        emb = None
        try:
            er = db.conn.execute(
                "SELECT embedding FROM clip_embeddings WHERE photo_id = ?",
                (photo_id,)).fetchone()
            if er:
                emb = list(_deserialize_float_list(er["embedding"], CLIP_DIMENSIONS))
        except Exception:
            pass
        out["clip_embedding"] = emb

        # Faces with encodings (bbox order matches db.add_face: top,right,bottom,left).
        faces = []
        try:
            for fr in db.conn.execute(
                """SELECT f.id, f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left,
                          f.det_score, f.person_id, f.cluster_id, f.match_source,
                          e.encoding
                     FROM faces f LEFT JOIN face_encodings e ON e.face_id = f.id
                    WHERE f.photo_id = ?""", (photo_id,)).fetchall():
                enc = (list(_deserialize_float_list(fr["encoding"], FACE_DIMENSIONS))
                       if fr["encoding"] is not None else [])
                # person_id/cluster_id mirror verbatim — the replica is a full DB
                # copy so ids match — preserving assignments across a face mirror.
                faces.append({
                    "bbox": [fr["bbox_top"], fr["bbox_right"],
                             fr["bbox_bottom"], fr["bbox_left"]],
                    "encoding": enc,
                    "det_score": fr["det_score"],
                    "person_id": fr["person_id"],
                    "cluster_id": fr["cluster_id"],
                    "match_source": fr["match_source"],
                })
        except Exception:
            pass
        out["faces"] = faces
        return out


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
                      f.match_source, f.det_score, p.name as person_name
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
                "det_score": f["det_score"],
            })

        colors = []
        if photo.get("dominant_colors"):
            try:
                colors = json.loads(photo["dominant_colors"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Latest generation row per text_type — surfaces the model that
        # produced the currently-promoted description/tags/verify text. The UI
        # displays this inline so the operator can tell which model generated
        # what they're looking at. Full history is at the on-demand
        # /api/photos/<id>/generations endpoint.
        gen_rows = db.conn.execute("""
            WITH latest AS (
                SELECT text_type, model_used, model_version, created_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY text_type
                           ORDER BY created_at DESC, id DESC
                       ) AS rn
                FROM generations WHERE photo_id = ?
            )
            SELECT text_type, model_used, model_version, created_at
            FROM latest WHERE rn = 1
        """, (photo_id,)).fetchall()
        current_models = {
            r["text_type"]: {
                "model": r["model_used"],
                "version": r["model_version"],
                "created_at": r["created_at"],
            } for r in gen_rows
        }

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
            "aesthetics": _aesthetics_from_row(photo),
            # Flat map of every raw aes_* column (overall/pct, the 3 dimensions,
            # the 11 sub-attributes, model, scored_at) for quick inspection —
            # the structured `aesthetics` block nests the same values.
            "aes_raw": {k: v for k, v in photo.items() if k.startswith("aes_")},
            # Subject-aware quality (v27): primary-subject box(es), normalized
            # 0-1; the subject-crop scores live in aes_raw (aes_subject_*).
            "subject_boxes": json.loads(photo["subject_boxes"]) if photo.get("subject_boxes") else None,
            "camera_make": photo.get("camera_make"),
            "camera_model": photo.get("camera_model"),
            "focal_length": photo.get("focal_length"),
            "exposure_time": photo.get("exposure_time"),
            "f_number": photo.get("f_number"),
            "iso": photo.get("iso"),
            "image_width": photo.get("image_width"),
            "image_height": photo.get("image_height"),
            "categories": json.loads(photo["categories"]) if photo.get("categories") else [],
            "visual_tags": json.loads(photo["visual_tags"]) if photo.get("visual_tags") else [],
            "keywords": json.loads(photo["keywords"]) if photo.get("keywords") else [],
            "colors": colors,
            "faces": face_list,
            "stack": db.get_photo_stack(photo_id),
            "current_models": current_models,
        }


@app.get("/api/photos/{photo_id}/generations")
def api_photo_generations(photo_id: int):
    """Full describe/tags/verify generation history for one photo.

    Lazy-fetched by the photo modal's "Prior generations" expand — not pre-
    loaded with /api/photos/<id> because most modals never expand it.
    """
    with _get_db() as db:
        if not db.get_photo(photo_id):
            raise HTTPException(404, "Photo not found")
        rows = db.conn.execute("""
            SELECT id, text_type, generated_text, model_used, model_version, created_at
            FROM generations
            WHERE photo_id = ?
            ORDER BY created_at DESC, id DESC
        """, (photo_id,)).fetchall()
    return {"photo_id": photo_id, "generations": [dict(r) for r in rows]}


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


@app.get("/api/cameras")
def api_cameras():
    """Distinct camera models with photo counts — feeds the camera filter
    dropdown on search/review/faces/collections/geotag. Ordered by most-recently-
    used (latest photo timestamp) so the body you're shooting now leads; cameras
    with no dated photos fall to the end, tie-broken by count."""
    # Only well-formed YYYY-MM-DD dates count toward recency — corrupt values
    # (control bytes, DD/MM/YYYY imports) sort lexically ABOVE real 2026 dates
    # and would float dead cameras to the top. Malformed-only cameras get a NULL
    # last_taken and fall to the end.
    valid_date = ("MAX(CASE WHEN date_taken GLOB "
                  "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*' "
                  "THEN date_taken END)")
    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT camera_model AS model, "
            "       MAX(camera_make) AS make, "
            "       COUNT(*) AS count, "
            f"      {valid_date} AS last_taken "
            "FROM photos "
            "WHERE camera_model IS NOT NULL AND camera_model != '' "
            "GROUP BY camera_model "
            "ORDER BY last_taken IS NULL, last_taken DESC, count DESC"
        ).fetchall()
    return {"cameras": [{"model": r["model"], "make": r["make"],
                         "count": r["count"], "last_taken": r["last_taken"]}
                        for r in rows]}


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


def _abs_photo_path(db, photo: dict) -> str:
    """Absolute path to a photo's original file (photo_root + relative path)."""
    fp = photo["filepath"]
    if db.photo_root and not os.path.isabs(fp):
        return str(Path(db.photo_root) / fp)
    return fp


def _mirror_face_photo(photo_id: int) -> dict:
    """Replica-mode helper: re-sync one photo's authoritative face rows from the
    NAS into the local DB after a NAS-side face mutation. Returns mirror stats."""
    from . import rerun
    with _get_db() as db:
        return rerun.mirror_photos(db, [photo_id])


@app.delete("/api/faces/{face_id}")
def api_delete_face(face_id: int):
    """Delete a face detection box entirely (row + vec0 encoding). Distinct from
    /clear, which only removes the person assignment. Used to drop spurious
    detections (e.g. a mis-fired box on a non-face).

    Replica mode: proxy the delete to the NAS (authoritative), then mirror the
    photo's faces back so the local grid updates without a full sync."""
    if _nas_url:
        with _get_db() as db:
            row = db.conn.execute(
                "SELECT photo_id FROM faces WHERE id = ?", (face_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Face not found")
        photo_id = row["photo_id"]
        resp = _nas_json("DELETE", f"/api/faces/{face_id}")
        resp["mirror"] = _mirror_face_photo(photo_id)
        return resp
    with _get_db() as db:
        if not db.delete_face(face_id):
            raise HTTPException(404, "Face not found")
        db.conn.commit()
        logger.info("FACE DELETE  face_id=%d", face_id)
        return {"ok": True, "face_id": face_id}


@app.post("/api/photos/{photo_id}/detect-faces")
def api_redetect_faces(photo_id: int, data: dict = Body(default=None)):
    """Manually re-run face detection on one photo. Body `{replace?: bool}`.

    Default (augment) keeps existing faces and adds only non-overlapping new
    detections. `replace=true` wipes existing faces first. The compute runs
    where the originals + InsightFace live (the NAS); in replica mode this
    proxies there and mirrors the result locally."""
    replace = bool((data or {}).get("replace"))
    if _nas_url:
        resp = _nas_json("POST", f"/api/photos/{photo_id}/detect-faces",
                         {"replace": replace})
        resp["mirror"] = _mirror_face_photo(photo_id)
        return resp
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        abs_path = _abs_photo_path(db, photo)
        if not os.path.exists(abs_path):
            raise HTTPException(400, "original image not available on this host")
        try:
            from .face_edit import redetect_photo_faces
            result = redetect_photo_faces(db, photo_id, abs_path, replace=replace)
        except Exception as e:
            logger.exception("re-detect faces failed for photo %d", photo_id)
            raise HTTPException(500, f"detection failed: {e}")
        logger.info("FACE REDETECT photo_id=%d %s", photo_id, result)
        return result


@app.post("/api/photos/{photo_id}/add-face-box")
def api_add_face_box(photo_id: int, data: dict = Body(...)):
    """Add a face at a user-drawn box. Body `{bbox: {top,right,bottom,left} in
    0-1 EXIF-oriented coords, person?: name}`. Crops the box and runs InsightFace
    to recover an encoding; stores the box with no encoding if none is found.
    Replica mode proxies to the NAS (which has the original) and mirrors back."""
    bbox = data.get("bbox")
    if not bbox:
        raise HTTPException(400, "bbox required (normalized 0-1)")
    person = (data.get("person") or "").strip() or None
    if _nas_url:
        resp = _nas_json("POST", f"/api/photos/{photo_id}/add-face-box",
                         {"bbox": bbox, "person": person})
        resp["mirror"] = _mirror_face_photo(photo_id)
        return resp
    with _get_db() as db:
        photo = db.get_photo(photo_id)
        if not photo:
            raise HTTPException(404, "Photo not found")
        abs_path = _abs_photo_path(db, photo)
        if not os.path.exists(abs_path):
            raise HTTPException(400, "original image not available on this host")
        from .face_edit import add_manual_face
        result = add_manual_face(db, photo_id, abs_path, photo, bbox, person_name=person)
        if not result.get("ok"):
            raise HTTPException(500, result.get("error", "add face failed"))
        logger.info("FACE ADD-BOX photo_id=%d %s", photo_id, result)
        return result


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


@app.post("/api/faces/bulk-assign")
def api_bulk_assign_faces(data: dict):
    """Bulk reassign or unassign a list of faces.

    Body: {"face_ids": [...], "person_name": "Calvin" | null}
    A null/empty person_name CLEARS the assignment (person_id=NULL). Otherwise
    the faces are assigned to that person (created if needed) with
    match_source='manual'. Used by the per-person inspector to re-map / unset
    the wrong faces a person picked up via over-matching or duplicate imports.
    """
    face_ids = [int(x) for x in (data.get("face_ids") or [])]
    name = (data.get("person_name") or "").strip()
    if not face_ids:
        return {"ok": True, "updated": 0}
    with _get_db() as db:
        if name:
            person = db.get_person_by_name(name)
            pid = person["id"] if person else db.add_person(name)
            for fid in face_ids:
                db.assign_face_to_person(fid, pid, match_source="manual")
            db.conn.commit()
            logger.info("FACE BULK-ASSIGN  %d faces -> %r (id=%d)", len(face_ids), name, pid)
            return {"ok": True, "updated": len(face_ids), "person_id": pid, "person_name": name}
        # Clear
        for i in range(0, len(face_ids), 500):
            batch = face_ids[i:i + 500]
            ph = ",".join("?" * len(batch))
            db.conn.execute(
                f"UPDATE faces SET person_id = NULL, match_source = NULL "
                f"WHERE id IN ({ph})", batch)
        db.conn.commit()
        logger.info("FACE BULK-CLEAR  %d faces", len(face_ids))
        return {"ok": True, "updated": len(face_ids), "person_id": None}


@app.get("/api/faces/person/{person_id}/inspect")
def api_person_inspect(
    person_id: int,
    eps: float = Query(0.50, ge=0.2, le=1.2),
    min_samples: int = Query(3, ge=1, le=20),
):
    """Sub-structure of one person's faces, for spotting/re-mapping wrong faces.

    Runs DBSCAN over the person's ArcFace encodings to surface visual sub-groups,
    and scores every face by L2 distance to the person's "core" reference set
    (their strict/manual faces if enough exist, else the largest sub-cluster).
    Faces far from the core — and DBSCAN-noise faces — are the likely-wrong ones
    (over-matches, duplicate-import contamination, a different kid).

    Returns one flat ``faces`` list (all of them — only ids/scores, no crops, so
    even a 20k-face person is ~1MB) plus a ``sub_clusters`` summary. The frontend
    derives both views from the flat list (group by sub_id, or sort by distance)
    and paginates the crop rendering so the browser never draws 20k crops at once.
    """
    import hashlib
    import numpy as np
    from sklearn.cluster import DBSCAN

    with _get_db() as db:
        prow = db.conn.execute(
            "SELECT id, name FROM persons WHERE id = ?", (person_id,)).fetchone()
        if not prow:
            raise HTTPException(404, "Person not found")

        rows = db.conn.execute(
            """SELECT f.id AS face_id, f.photo_id, f.match_source, f.det_score,
                      fe.encoding, ph.date_taken,
                      CASE WHEN f.bbox_top IS NOT NULL
                           THEN (f.bbox_bottom - f.bbox_top) * (f.bbox_right - f.bbox_left)
                           ELSE 0 END AS area
               FROM faces f
               JOIN face_encodings fe ON fe.face_id = f.id
               LEFT JOIN photos ph ON ph.id = f.photo_id
               WHERE f.person_id = ?
               ORDER BY f.id""",
            (person_id,),
        ).fetchall()

        if not rows:
            return {"person": {"id": person_id, "name": prow["name"], "face_count": 0},
                    "faces": [], "sub_clusters": [], "params": {"eps": eps, "min_samples": min_samples}}

        face_ids = [int(r["face_id"]) for r in rows]
        X = np.stack([np.frombuffer(r["encoding"], dtype=np.float32) for r in rows]).astype(np.float32)
        # Normalize defensively (stored encodings are unit-norm, but be safe).
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        # DBSCAN sub-clusters within this person.
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                        algorithm="ball_tree", n_jobs=-1).fit_predict(X)

        # Reference set for "distance to core": trusted faces if we have enough,
        # else the largest sub-cluster. min-dist-to-refs tolerates aging / angles.
        trusted_idx = [i for i, r in enumerate(rows)
                       if r["match_source"] in ("strict", "manual")]
        ref_idx = trusted_idx if len(trusted_idx) >= 3 else None
        if ref_idx is None:
            from collections import Counter
            cnt = Counter(int(l) for l in labels if l != -1)
            if cnt:
                core_label = cnt.most_common(1)[0][0]
                ref_idx = [i for i, l in enumerate(labels) if int(l) == core_label]
            else:
                ref_idx = list(range(len(rows)))  # no structure — everything is "core"
        R = X[ref_idx]
        if len(R) > 500:
            R = R[np.linspace(0, len(R) - 1, 500).astype(int)]
        maxsim = (X @ R.T).max(axis=1)
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * maxsim))

        # Mark duplicate faces (byte-identical encoding) within this person.
        dup_group: dict[int, int] = {}
        h2idx: dict[bytes, list] = {}
        for i, r in enumerate(rows):
            h2idx.setdefault(hashlib.md5(bytes(r["encoding"])).digest(), []).append(i)
        gid = 0
        for idxs in h2idx.values():
            if len(idxs) > 1:
                for i in idxs:
                    dup_group[i] = gid
                gid += 1

        # One flat record per face — ids/scores only (no crops). The frontend
        # groups by sub_id or sorts by dist, and paginates crop rendering.
        from collections import defaultdict
        faces = [
            {
                "face_id": face_ids[i], "photo_id": int(rows[i]["photo_id"]),
                "sub_id": int(labels[i]), "dist": round(float(dists[i]), 4),
                "date_taken": rows[i]["date_taken"], "match_source": rows[i]["match_source"],
                "dup_group": dup_group.get(i),
            }
            for i in range(len(rows))
        ]

        # Sub-cluster summaries: size, rep (biggest-area), and median/max distance
        # to the core. Suspect sub-clusters (high median_dist) are whole groups
        # that are probably a different person — re-map them in one action.
        by_sub: dict = defaultdict(list)
        for i in range(len(rows)):
            by_sub[int(labels[i])].append(i)
        sub_clusters = []
        for sub_id, idxs in by_sub.items():
            rep_i = max(idxs, key=lambda i: int(rows[i]["area"] or 0))
            dvals = [float(dists[i]) for i in idxs]
            sub_clusters.append({
                "sub_id": sub_id, "size": len(idxs),
                "rep_face_id": face_ids[rep_i],
                "median_dist": round(float(np.median(dvals)), 4),
                "max_dist": round(float(np.max(dvals)), 4),
            })
        # Real clusters first (biggest first), outliers (-1) always last.
        sub_clusters.sort(key=lambda s: (s["sub_id"] == -1, -s["size"]))

        return {
            "person": {"id": person_id, "name": prow["name"], "face_count": len(rows)},
            "faces": faces,
            "sub_clusters": sub_clusters,
            "n_outliers": sum(1 for l in labels if int(l) == -1),
            "n_dup_faces": len(dup_group),
            "params": {"eps": eps, "min_samples": min_samples,
                       "ref_source": "trusted" if len(trusted_idx) >= 3 else "largest_subcluster"},
        }


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
    # Aggregate over the indexed `folder` column (schema v25, M27) instead of
    # loading every photo row and grouping in Python. COALESCE(...,'') matches
    # the old "date_taken or ''" semantics so all-undated folders still sort
    # consistently last.
    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT folder AS path, COALESCE(MAX(date_taken), '') AS max_date "
            "FROM photos "
            "WHERE filepath IS NOT NULL AND folder IS NOT NULL AND folder != '' "
            "GROUP BY folder "
            "ORDER BY max_date DESC, path ASC"
        ).fetchall()

    return {"folders": [{"path": r["path"], "max_date": r["max_date"]} for r in rows]}


def _resolve_review_scope(db, directory: str) -> str:
    """Resolve a review scope key into the persistence key used everywhere.

    Folder reviews get the photo_root prefix (so a relative folder name
    resolves to an absolute path); date-range reviews use a synthetic
    ``range:<from>..<to>`` key that is stored/looked-up verbatim.
    """
    if directory.startswith("range:"):
        return directory
    if db.photo_root and not Path(directory).is_absolute():
        return str(Path(db.photo_root) / directory)
    return directory


@app.get("/api/review/run")
def api_review_run(
    directory: str = Query("", description="Directory path to review (folder mode)"),
    target_pct: float = Query(0.10, description="Target selection percentage"),
    distance_threshold: float = Query(0.0, description="Clustering distance threshold (0 = adaptive)"),
    date_from: str = Query("", description="Inclusive start date YYYY-MM-DD (date-range mode)"),
    date_to: str = Query("", description="Inclusive end date YYYY-MM-DD (date-range mode)"),
):
    """Run the culling algorithm on a directory OR a date range and return
    selections. Date-range mode spans every folder (per-camera subfolders from
    the same shoot) so one review can cover multiple sources."""
    from .cull import select_best_photos, save_selections

    use_range = bool(date_from or date_to)
    if not use_range and not directory:
        raise HTTPException(400, "Provide either a directory or a date range")

    with _get_db() as db:
        if use_range:
            # Synthetic scope key so save/load/download all key off one string.
            scope_key = f"range:{date_from or ''}..{date_to or ''}"
            selections = select_best_photos(
                db,
                target_pct=target_pct,
                distance_threshold=distance_threshold,
                date_from=date_from or None,
                date_to=date_to or None,
            )
        else:
            scope_key = _resolve_review_scope(db, directory)
            selections = select_best_photos(
                db, scope_key,
                target_pct=target_pct,
                distance_threshold=distance_threshold,
            )
        resolved_dir = scope_key

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
                "aes_overall_pct": p.get("aes_overall_pct"),
                "aes_subject_overall_pct": p.get("aes_subject_overall_pct"),
                "aes_overall_day_pct": p.get("aes_overall_day_pct"),
                "aes_subject_overall_day_pct": p.get("aes_subject_overall_day_pct"),
                "camera_make": p.get("camera_make"),
                "camera_model": p.get("camera_model"),
                "date_taken": p.get("date_taken"),
                "selected": p.get("selected", False),
                "cluster_id": p.get("cluster_id"),
                "rank_in_cluster": p.get("rank_in_cluster"),
                "has_raw": bool(p.get("raw_filepath")),
                "place_name": p.get("place_name"),
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
        resolved_dir = _resolve_review_scope(db, directory)

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
                "aes_overall_pct": p.get("aes_overall_pct"),
                "aes_subject_overall_pct": p.get("aes_subject_overall_pct"),
                "aes_overall_day_pct": p.get("aes_overall_day_pct"),
                "aes_subject_overall_day_pct": p.get("aes_subject_overall_day_pct"),
                "camera_make": p.get("camera_make"),
                "camera_model": p.get("camera_model"),
                "date_taken": p.get("date_taken"),
                "selected": bool(p["selected"]),
                "cluster_id": p.get("cluster_id"),
                "rank_in_cluster": p.get("rank_in_cluster"),
                "has_raw": bool(p.get("raw_filepath")),
                "place_name": p.get("place_name"),
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
        resolved_dir = _resolve_review_scope(db, directory)

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


def _build_photo_zip_response(entries: list, zipname: str):
    """Stream a ZIP from ``(kind, arcname, abs_path_or_None, photo_id)`` entries.

    Built to a temp file (deleted after the response) so a large selection
    never buffers in RAM on the N100. Missing local files fall back to the NAS
    full image in replica mode. Duplicate basenames across subfolders are
    de-collided. Raises HTTPException(404) if nothing could be read. Shared by
    the review-folder and collection download endpoints."""
    import tempfile
    import zipfile
    from starlette.background import BackgroundTask

    tmp = tempfile.NamedTemporaryFile(prefix="photo_dl_", suffix=".zip", delete=False)
    used: dict = {}   # de-collide duplicate basenames across subfolders
    added = 0
    try:
        # ZIP_STORED: JPEG/ARW are already compressed — deflate burns N100 CPU
        # for ~no size win.
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            for kind, name, path, pid in entries:
                name = name or f"photo_{pid}.{kind}"
                if name in used:
                    used[name] += 1
                    stem, dot, ext = name.rpartition(".")
                    name = f"{stem}_{used[name]}{dot}{ext}" if dot else f"{name}_{used[name]}"
                else:
                    used[name] = 0
                try:
                    if path and os.path.exists(path):
                        zf.write(path, arcname=name)
                        added += 1
                    elif kind == "jpg" and _nas_url:
                        zf.writestr(name, _fetch_from_nas(pid, "full", timeout=120.0))
                        added += 1
                except Exception:
                    pass  # unreadable file → skip it, keep the rest of the zip
        tmp.close()
    except Exception as e:
        tmp.close()
        os.unlink(tmp.name)
        raise HTTPException(500, f"Zip build failed: {e}")

    if added == 0:
        os.unlink(tmp.name)
        raise HTTPException(404, "None of the selected files could be read")

    return FileResponse(
        tmp.name, media_type="application/zip",
        headers={"Content-Disposition": _attachment_header(zipname)},
        background=BackgroundTask(os.unlink, tmp.name),
    )


@app.get("/api/review/download")
def api_review_download(
    directory: str = Query(..., description="Directory path or range: scope key"),
    include_raw: bool = Query(False, description="Include ARW raw files in the zip"),
):
    """Stream a ZIP of the selected photos in a review folder (or date range) —
    a real download, unlike /export which only lists paths to copy."""
    from .cull import load_selections

    with _get_db() as db:
        resolved_dir = _resolve_review_scope(db, directory)

        selections = load_selections(db, resolved_dir)
        chosen = [p for p in (selections or []) if p["selected"]]
        if not chosen:
            raise HTTPException(404, "No selected photos in that folder")

        # Resolve every source while the DB is open (needed for NAS fallback).
        entries = []  # (kind, arcname, abs_path_or_None, photo_id)
        for p in chosen:
            entries.append(("jpg", p["filename"], db.resolve_filepath(p["filepath"]), p["photo_id"]))
            if include_raw and p.get("raw_filepath"):
                raw_abs = db.resolve_filepath(p["raw_filepath"])
                entries.append(("arw", Path(raw_abs).name if raw_abs else "",
                                raw_abs, p["photo_id"]))

    if directory.startswith("range:"):
        stem = directory[len("range:"):].replace("..", "_to_") or "range"
    else:
        stem = Path(directory).name or "selected"
    return _build_photo_zip_response(entries, f"{stem}_selected.zip")


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

        category_tagged = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE categories IS NOT NULL AND categories != '[]'"
        ).fetchone()["c"]
        visual_tagged = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE visual_tags IS NOT NULL AND visual_tags != '[]'"
        ).fetchone()["c"]
        keyword_tagged = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE keywords IS NOT NULL AND keywords != '[]'"
        ).fetchone()["c"]

        aes_scored = db.conn.execute(
            "SELECT COUNT(*) as c FROM photos WHERE aes_overall IS NOT NULL"
        ).fetchone()["c"]
        aesthetics_stats = None
        if aes_scored > 0:
            row = db.conn.execute(
                """SELECT MIN(aes_overall) as min_s, MAX(aes_overall) as max_s,
                          AVG(aes_overall) as avg_s,
                          SUM(CASE WHEN aes_overall_pct IS NOT NULL THEN 1 ELSE 0 END) as pct_done
                   FROM photos WHERE aes_overall IS NOT NULL"""
            ).fetchone()
            aesthetics_stats = {
                "min": round(row["min_s"], 2),
                "max": round(row["max_s"], 2),
                "mean": round(row["avg_s"], 2),
                "normalized": row["pct_done"],
            }

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
        "aesthetics_scored": aes_scored,
        "aesthetics_stats": aesthetics_stats,
        "concepts_analyzed": concepts_analyzed,
        "category_tagged": category_tagged,
        "visual_tagged": visual_tagged,
        "keyword_tagged": keyword_tagged,
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


@app.get("/api/stats/generations")
def api_stats_generations():
    """Per-text-type breakdown of which model produced each photo's CURRENT
    promoted artifact. Uses the latest generations row per (photo_id, text_type)
    pair — model_used = NULL means a backfilled-historical row where the
    producing model wasn't recorded.
    """
    with _get_db() as db:
        rows = db.conn.execute("""
            WITH latest AS (
                SELECT photo_id, text_type, model_used, model_version,
                       ROW_NUMBER() OVER (
                           PARTITION BY photo_id, text_type
                           ORDER BY created_at DESC, id DESC
                       ) AS rn
                FROM generations
            )
            SELECT text_type, model_used, model_version, COUNT(*) AS n
            FROM latest
            WHERE rn = 1
            GROUP BY text_type, model_used, model_version
            ORDER BY text_type, n DESC
        """).fetchall()
    out: dict[str, list[dict]] = {}
    for r in rows:
        d = dict(r)
        out.setdefault(d["text_type"], []).append({
            "model": d["model_used"],
            "version": d["model_version"],
            "count": d["n"],
        })
    return out


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


@app.get("/api/geotag/folders")
def api_geotag_folders(include_fully_tagged: bool = False,
                        camera: Optional[str] = None,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None):
    """Folder summary keyed for the /geotag left panel.

    Returns folders sorted by no_gps count descending. A folder's entry
    includes photo counts split by provenance (exif / inferred / none)
    plus date range, so the UI can prioritize which folders to tackle.

    With `include_fully_tagged=true` the response also covers folders
    where every photo already has GPS; default is to hide them. `camera` /
    `date_from` / `date_to` narrow which photos count toward each folder.
    """
    # Aggregate over the indexed `folder` column (schema v25, M27) in one SQL
    # pass instead of loading every photo row and grouping in Python.
    # NULLIF(date_taken,'') mirrors the old `if dt:` guard so blank dates don't
    # become a folder's date_from. The HAVING clause applies the
    # hide-fully-tagged filter server-side.
    having = "" if include_fully_tagged else "HAVING no_gps > 0"
    where, params = ["folder IS NOT NULL"], []
    if camera:
        where.append("camera_model = ?"); params.append(camera)
    if date_from:
        where.append("substr(date_taken,1,10) >= ?"); params.append(date_from[:10])
    if date_to:
        where.append("substr(date_taken,1,10) <= ?"); params.append(date_to[:10])
    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT folder AS path, "
            "       COUNT(*) AS total, "
            "       SUM(CASE WHEN location_source='exif' THEN 1 ELSE 0 END) AS with_exif, "
            "       SUM(CASE WHEN location_source='inferred' THEN 1 ELSE 0 END) AS with_inferred, "
            "       SUM(CASE WHEN gps_lat IS NULL THEN 1 ELSE 0 END) AS no_gps, "
            "       MIN(NULLIF(date_taken, '')) AS date_from, "
            "       MAX(NULLIF(date_taken, '')) AS date_to "
            "FROM photos "
            f"WHERE {' AND '.join(where)} "
            "GROUP BY folder "
            f"{having} "
            "ORDER BY no_gps DESC, path ASC",
            params,
        ).fetchall()

    out = [{"path": r["path"], "total": r["total"], "with_exif": r["with_exif"],
            "with_inferred": r["with_inferred"], "no_gps": r["no_gps"],
            "date_from": r["date_from"], "date_to": r["date_to"]} for r in rows]
    return {"folders": out, "total_folders": len(out)}


@app.get("/api/geotag/folder-photos")
def api_geotag_folder_photos(folder: str, show_inferred: bool = False,
                              camera: Optional[str] = None,
                              date_from: Optional[str] = None,
                              date_to: Optional[str] = None,
                              limit: int = 1000):
    """Photos in one folder for the /geotag thumbnails panel.

    By default returns only photos where gps_lat IS NULL (the ones that
    need tagging). With `show_inferred=true`, also includes photos where
    `location_source='inferred'` so the user can manually correct any M19
    misfires. `location_source='exif'` photos are always excluded — those
    came from the camera and are authoritative. `camera`/`date_from`/`date_to`
    narrow the set to match the folder-picker filters.
    """
    # Match the exact folder via the indexed `folder` column (schema v25) — no
    # LIKE + Python subfolder guard. This also fixes a latent bug in the old
    # path where LIMIT was applied to a sub-folder-inclusive LIKE before the
    # Python filter, so a folder with many sub-folder photos could return fewer
    # than `limit` of its own.
    with _get_db() as db:
        if show_inferred:
            gps_where = "(gps_lat IS NULL OR location_source='inferred')"
        else:
            gps_where = "gps_lat IS NULL"
        extra, params = "", [folder.rstrip("/")]
        if camera:
            extra += " AND camera_model = ?"; params.append(camera)
        if date_from:
            extra += " AND substr(date_taken,1,10) >= ?"; params.append(date_from[:10])
        if date_to:
            extra += " AND substr(date_taken,1,10) <= ?"; params.append(date_to[:10])
        params.append(limit)
        rows = db.conn.execute(
            f"""SELECT id, filepath, filename, date_taken, gps_lat, gps_lon,
                       place_name, location_source, location_confidence,
                       camera_make, camera_model
                FROM photos
                WHERE folder = ? AND {gps_where}{extra}
                ORDER BY COALESCE(date_taken, filepath)
                LIMIT ?""",
            params,
        ).fetchall()

    photos = [dict(r) for r in rows]
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

    updated_ids = []
    skipped = 0
    with _get_db() as db:
        for pid in photo_ids:
            if db.set_photo_location(pid, lat, lon, place_name, overwrite=overwrite):
                updated_ids.append(pid)
            else:
                skipped += 1
        db.conn.commit()

    # `applied` carries the canonical written values so a replica mirror can
    # copy them byte-for-byte (the M26b dual-write path) instead of re-deriving.
    return {
        "updated_count": len(updated_ids),
        "skipped_count": skipped,
        "updated_ids": updated_ids,
        "applied": {"gps_lat": lat, "gps_lon": lon, "place_name": place_name,
                    "location_source": "manual"},
    }


@app.post("/api/photos/bulk-set-tags")
def api_bulk_set_tags(data: dict):
    """Manually/agent-set tag columns on a batch of photos (M26b).

    Body: `{photo_ids: [...], categories?: [...], visual_tags?: [...],
            keywords?: [...], mode: 'add'|'replace' = 'add', model?: str}`.
    Only the tag columns provided are touched. `mode='add'` unions with the
    existing values; `mode='replace'` overwrites. Each touched column is logged
    to the `generations` provenance table (model defaults to 'manual') so the
    edit is auditable. Returns the canonical post-write tag lists per photo so a
    replica mirror can copy them.
    """
    photo_ids = data.get("photo_ids") or []
    if not isinstance(photo_ids, list) or not photo_ids:
        raise HTTPException(status_code=400, detail="photo_ids required")
    mode = data.get("mode") or "add"
    if mode not in ("add", "replace"):
        raise HTTPException(status_code=400, detail="mode must be 'add' or 'replace'")

    def _aslist(v):
        if v is None:
            return None
        if not isinstance(v, list):
            raise HTTPException(status_code=400,
                                detail="categories/visual_tags/keywords must be arrays")
        return [str(x) for x in v]

    categories = _aslist(data.get("categories"))
    visual_tags = _aslist(data.get("visual_tags"))
    keywords = _aslist(data.get("keywords"))
    if categories is None and visual_tags is None and keywords is None:
        raise HTTPException(status_code=400,
                            detail="at least one of categories/visual_tags/keywords required")
    model = data.get("model") or "manual"

    results = []
    with _get_db() as db:
        for pid in photo_ids:
            final = db.set_photo_tags(pid, categories=categories,
                                      visual_tags=visual_tags, keywords=keywords,
                                      mode=mode, log_model=model)
            if final is not None:
                results.append({"id": pid, **final})
        db.conn.commit()

    return {"updated_count": len(results), "mode": mode, "results": results}


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


@app.get("/api/collections/{collection_id}/download")
def api_collection_download(
    collection_id: int,
    photo_ids: str = Query("", description="Comma-separated photo ids to download; empty = whole collection"),
    include_raw: bool = Query(False, description="Include ARW raw files in the zip"),
):
    """Stream a ZIP of a collection's photos. With no ``photo_ids`` the whole
    collection is bundled; otherwise only the given ids (must belong to the
    collection) — the multi-select subset from the collection UI."""
    wanted = None
    if photo_ids.strip():
        try:
            wanted = {int(x) for x in photo_ids.split(",") if x.strip()}
        except ValueError:
            raise HTTPException(400, "photo_ids must be a comma-separated list of integers")

    with _get_db() as db:
        collection = db.get_collection(collection_id)
        if not collection:
            raise HTTPException(404, "Collection not found")

        photos = db.get_collection_photos(collection_id)
        if wanted is not None:
            photos = [p for p in photos if p["id"] in wanted]
        if not photos:
            raise HTTPException(404, "No photos to download")

        entries = []  # (kind, arcname, abs_path_or_None, photo_id)
        for p in photos:
            entries.append(("jpg", p.get("filename") or "", db.resolve_filepath(p.get("filepath", "")), p["id"]))
            if include_raw and p.get("raw_filepath"):
                raw_abs = db.resolve_filepath(p["raw_filepath"])
                entries.append(("arw", Path(raw_abs).name if raw_abs else "", raw_abs, p["id"]))

    import re
    safe_name = re.sub(r"[^\w.-]+", "_", collection.get("name") or f"collection_{collection_id}").strip("_")
    suffix = "_selected" if wanted is not None else ""
    return _build_photo_zip_response(entries, f"{safe_name or 'collection'}{suffix}.zip")


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


@app.post("/api/collections/add-photos")
def api_collection_add_photos(data: dict):
    """Resolve-or-create a collection by name (or id) and add photos in one shot
    (M26b agent write path).

    Body: `{collection: str-name | collection_id: int, photo_ids: [...],
            create: bool = false, description?: str}`.

    Resolves the target collection: by `collection_id` if given, else by
    `collection` name. A missing name is created only when `create=true` (guards
    against a typo silently spawning a new album). Returns the canonical
    `{collection: {id, name, description}, added, created}` so a replica mirror
    can re-create the collection under the SAME id and add the same photos.
    """
    photo_ids = data.get("photo_ids") or []
    if not isinstance(photo_ids, list) or not photo_ids:
        raise HTTPException(status_code=400, detail="photo_ids required")
    create = bool(data.get("create", False))
    description = data.get("description")
    cid = data.get("collection_id")
    name = (data.get("collection") or "").strip()

    with _get_db() as db:
        created = False
        if cid is not None:
            coll = db.get_collection(int(cid))
            if not coll:
                raise HTTPException(status_code=404,
                                    detail=f"no collection with id {cid}")
        elif name:
            coll = db.get_collection_by_name(name)
            if not coll:
                if not create:
                    raise HTTPException(
                        status_code=404,
                        detail=f"collection '{name}' does not exist "
                               f"(pass create=true to create it)")
                coll = {"id": db.create_collection(name, description), "name": name,
                        "description": description}
                created = True
        else:
            raise HTTPException(status_code=400,
                                detail="collection name or collection_id required")
        added = db.add_photos_to_collection(coll["id"], photo_ids)

    return {"collection": {"id": coll["id"], "name": coll["name"],
                           "description": coll.get("description")},
            "added": added, "created": created}


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
# Photobook builder (M29) — interactive proof suggestion/creation.
# State lives in the sidecar store (see _get_books); reads photo metadata from
# the replica DB. Candidate pool queries reuse the existing /api/search endpoint
# client-side (frontend posts the resulting ids here), or import a collection.
# ---------------------------------------------------------------------------

def _book_candidate_hits(bs, pdb, book_id: int) -> list[dict]:
    """Compact hits for a book's candidate pool, annotated with decision + used."""
    from .tools import _compact_hit
    dm = bs.decision_map(book_id)
    used = bs.used_photo_ids(book_id)
    ids = list(dm.keys())
    hits = []
    rmap = {}
    if ids:
        ph = ",".join("?" * len(ids))
        rows = pdb.conn.execute(
            f"SELECT id, filename, date_taken, place_name, camera_model, description, "
            f"categories, aesthetic_score, image_width, image_height "
            f"FROM photos WHERE id IN ({ph})", ids).fetchall()
        rmap = {r["id"]: dict(r) for r in rows}
    for pid in ids:
        r = rmap.get(pid)
        hit = _compact_hit(r) if r else {"id": pid, "filename": None}
        if r:
            hit["image_width"] = r.get("image_width")
            hit["image_height"] = r.get("image_height")
        hit["decision"] = dm[pid]
        hit["used"] = pid in used
        hits.append(hit)
    hits.sort(key=lambda h: (h.get("date_taken") or "", h.get("id") or 0))
    return hits


@app.get("/api/books")
def api_list_books():
    with _get_books() as bs:
        return {"books": bs.list_books()}


@app.post("/api/books")
def api_create_book(body: dict):
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    with _get_books() as bs:
        bid = bs.create_book(name, body.get("subtitle"),
                             float(body.get("trim_w") or 14),
                             float(body.get("trim_h") or 11))
        return {"book": bs.get_book_row(bid)}


@app.get("/api/books/{book_id}")
def api_get_book(book_id: int):
    with _get_books() as bs, _get_db() as pdb:
        doc = bs.get_book(book_id)
        if not doc:
            return JSONResponse({"error": "Book not found"}, status_code=404)
        doc["candidates"] = _book_candidate_hits(bs, pdb, book_id)
        return doc


@app.put("/api/books/{book_id}")
def api_update_book(book_id: int, body: dict):
    with _get_books() as bs:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        bs.update_book(book_id, body)
        return {"book": bs.get_book_row(book_id)}


@app.delete("/api/books/{book_id}")
def api_delete_book(book_id: int):
    with _get_books() as bs:
        bs.delete_book(book_id)
    return {"ok": True}


@app.post("/api/books/{book_id}/candidates")
def api_book_add_candidates(book_id: int, body: dict):
    """Add photos to the candidate pool. Body: {photo_ids?: [...],
    collection_id?: int} — a collection imports all its photos."""
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        ids = list(body.get("photo_ids") or [])
        cid = body.get("collection_id")
        if cid is not None:
            ids += [p["id"] for p in pdb.get_collection_photos(int(cid))]
        added = bs.add_candidates(book_id, [int(i) for i in ids]) if ids else 0
        return {"added": added, "candidates": _book_candidate_hits(bs, pdb, book_id)}


@app.get("/api/books/{book_id}/candidates")
def api_book_candidates(book_id: int):
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        return {"candidates": _book_candidate_hits(bs, pdb, book_id)}


@app.put("/api/books/{book_id}/decisions/{photo_id}")
def api_book_decision(book_id: int, photo_id: int, body: dict):
    """Lasting 'use this / don't use this' verdict on a photo."""
    decision = body.get("decision")
    if decision not in ("include", "exclude"):
        return JSONResponse({"error": "decision must be include|exclude"}, status_code=400)
    with _get_books() as bs:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        bs.set_decision(book_id, photo_id, decision, body.get("note"))
    return {"ok": True}


@app.post("/api/books/{book_id}/spreads")
def api_book_add_spread(book_id: int, body: dict):
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        sid = bs.add_spread(pdb, book_id,
                            archetype=body.get("archetype") or "matched 2-up",
                            photo_ids=[int(i) for i in (body.get("photo_ids") or [])],
                            label=body.get("label"), bg=body.get("bg") or "#ffffff")
        return {"spread_id": sid, "book": bs.get_book(book_id)}


@app.put("/api/books/{book_id}/spreads/{spread_id}")
def api_book_update_spread(book_id: int, spread_id: int, body: dict):
    with _get_books() as bs, _get_db() as pdb:
        try:
            bs.update_spread(pdb, spread_id, body)
        except KeyError:
            return JSONResponse({"error": "Spread not found"}, status_code=404)
        return {"book": bs.get_book(book_id)}


@app.delete("/api/books/{book_id}/spreads/{spread_id}")
def api_book_delete_spread(book_id: int, spread_id: int):
    with _get_books() as bs:
        bs.delete_spread(spread_id)
    return {"ok": True}


@app.post("/api/books/{book_id}/spreads/reorder")
def api_book_reorder_spreads(book_id: int, body: dict):
    order = body.get("order") or []
    with _get_books() as bs:
        bs.reorder_spreads(book_id, [int(x) for x in order])
        return {"book": bs.get_book(book_id)}


@app.put("/api/books/{book_id}/cells/{cell_id}")
def api_book_update_cell(book_id: int, cell_id: int, body: dict):
    with _get_books() as bs, _get_db() as pdb:
        try:
            bs.set_cell(pdb, cell_id, body)
        except KeyError:
            return JSONResponse({"error": "Cell not found"}, status_code=404)
        return {"ok": True}


@app.post("/api/books/{book_id}/auto-arrange")
def api_book_auto_arrange(book_id: int, body: dict):
    """Materialize spreads from the included candidate pool via the deterministic
    suggest_layout partition (skips excluded photos). ``per_day`` keeps only the
    best N photos of each day first — the best-of-day cull that turns a big pool
    into a Varenna-scale book."""
    from . import book_ai
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        pids = body.get("photo_ids")
        ids = [int(i) for i in pids] if pids else None
        per_day = body.get("per_day")
        if per_day:
            base = ids if ids is not None else bs.included_ids(book_id)
            excluded = {p for p, d in bs.decision_map(book_id).items() if d == "exclude"}
            ids = book_ai._thin_per_day(pdb, [i for i in base if i not in excluded], int(per_day))
        n = bs.auto_arrange(pdb, book_id, ids, body.get("spread_count"),
                            replace=body.get("replace", True))
        return {"spreads_created": n, "book": bs.get_book(book_id)}


# -- LLM drafting (whole-book draft + captions) --------------------------------

@app.post("/api/books/{book_id}/ai-draft")
def api_book_ai_draft(book_id: int, body: dict):
    """Whole-book first draft: thin the included pool (optional per_day best-of),
    auto-arrange into spreads, and draft a house-voice caption per spread."""
    from . import book_ai
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        res = book_ai.ai_draft_book(pdb, bs, book_id,
                                    per_day=body.get("per_day"),
                                    spread_count=body.get("spread_count"),
                                    caption=body.get("caption", True))
        if res.get("error"):
            return JSONResponse(res, status_code=400)
        res["book"] = bs.get_book(book_id)
        return res


@app.post("/api/books/{book_id}/spreads/{spread_id}/caption")
def api_book_draft_caption(book_id: int, spread_id: int, body: dict):
    """Draft (or redraft) one spread's caption from its photos."""
    from . import book_ai
    with _get_books() as bs, _get_db() as pdb:
        book = bs.get_book(book_id)
        if not book:
            return JSONResponse({"error": "Book not found"}, status_code=404)
        sp = next((s for s in book["spreads"] if s["id"] == spread_id), None)
        if not sp:
            return JSONResponse({"error": "Spread not found"}, status_code=404)
        pids = [c["photo_id"] for c in sp["cells"] if c["photo_id"]]
        cap = book_ai.draft_caption(pdb, pids, book["book"].get("name"))
        dark = (sp.get("bg") or "#ffffff") != "#ffffff"
        bs.update_spread(pdb, spread_id, {"caption": {"text": cap, "dark": dark} if cap else None})
        return {"caption": cap, "book": bs.get_book(book_id)}


@app.post("/api/books/{book_id}/caption-all")
def api_book_caption_all(book_id: int, body: dict):
    """Draft captions for every spread lacking one (overwrite=true redrafts all)."""
    from . import book_ai
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        n = book_ai.caption_all(pdb, bs, book_id, overwrite=body.get("overwrite", False))
        return {"captioned": n, "book": bs.get_book(book_id)}


# -- Print-grade export (Pillow, WYSIWYG with the proof) -----------------------

def _full_image_fetcher(pdb, kind: str = "full"):
    """Build a fetch(photo_id)->PIL.Image for export: the local original if
    present, else the NAS proxy (``/full`` or ``/preview``). Always EXIF-oriented
    to match the crop coordinates."""
    import io as _io
    from PIL import Image as _Image, ImageOps as _ImageOps

    def fetch(pid: int):
        img = None
        row = pdb.conn.execute("SELECT filepath FROM photos WHERE id = ?", (pid,)).fetchone()
        fp = pdb.resolve_filepath(row["filepath"]) if row and row["filepath"] else None
        try:
            if kind == "full" and fp and os.path.exists(fp):
                img = _Image.open(fp)
            elif _nas_url:
                img = _Image.open(_io.BytesIO(_fetch_from_nas(pid, kind, timeout=60.0)))
            elif fp and os.path.exists(fp):
                img = _Image.open(fp)
        except Exception:
            img = None
        if img is None:
            # last resort: a locally-cached thumbnail (keeps export working offline
            # / when the NAS is unreachable, at reduced resolution)
            for cand in (os.path.join(_thumb_dir or "thumbnails", f"{pid}_thumb.jpg"),
                         os.path.join(_preview_dir or "previews", f"{pid}_preview.jpg")):
                if os.path.exists(cand):
                    img = _Image.open(cand)
                    break
        if img is None:
            return None
        return _ImageOps.exif_transpose(img).convert("RGB")
    return fetch


@app.get("/api/books/{book_id}/render/{spread_id}")
def api_book_render_spread(book_id: int, spread_id: int,
                           dpi: int = Query(96, ge=48, le=300),
                           scope: str = Query("preview")):
    """Render a single spread to JPEG (in-server preview / validation)."""
    from . import book_export
    with _get_books() as bs, _get_db() as pdb:
        doc = bs.get_book(book_id)
        if not doc:
            raise HTTPException(404, "Book not found")
        sp = next((s for s in doc["spreads"] if s["id"] == spread_id), None)
        if not sp:
            raise HTTPException(404, "Spread not found")
        img = book_export.render_spread(sp, doc["stage_w"], doc["stage_h"],
                                        _full_image_fetcher(pdb, scope), dpi)
    import io as _io
    buf = _io.BytesIO(); img.save(buf, "JPEG", quality=88); buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")


@app.get("/api/books/{book_id}/export")
def api_book_export(book_id: int,
                    format: str = Query("pdf", pattern="^(pdf|zip)$"),
                    dpi: int = Query(150, ge=72, le=300),
                    scope: str = Query("full", pattern="^(full|preview)$")):
    """Export the whole book as a print-grade PDF (one spread per page) or a ZIP
    of per-spread JPEGs + the PDF. Full-res originals stream from the NAS proxy."""
    from . import book_export
    import io as _io, re as _re, zipfile as _zip
    with _get_books() as bs, _get_db() as pdb:
        doc = bs.get_book(book_id)
        if not doc:
            raise HTTPException(404, "Book not found")
        if not doc["spreads"]:
            raise HTTPException(400, "Book has no spreads to export")
        jpegs, pdf = book_export.export_book(doc, _full_image_fetcher(pdb, scope), dpi)
        name = _re.sub(r"[^\w.-]+", "_", doc["book"].get("name") or f"book_{book_id}").strip("_") or "book"
    if format == "pdf":
        return StreamingResponse(_io.BytesIO(pdf), media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})
    zbuf = _io.BytesIO()
    with _zip.ZipFile(zbuf, "w", _zip.ZIP_STORED) as zf:
        for i, jb in enumerate(jpegs, 1):
            zf.writestr(f"{name}/spread_{i:03d}.jpg", jb)
        zf.writestr(f"{name}/{name}.pdf", pdf)
    zbuf.seek(0)
    return StreamingResponse(zbuf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


# ---------------------------------------------------------------------------
# Authoring pipeline (M30) — auto-draft a book from a raw window, then refine.
# ---------------------------------------------------------------------------

@app.post("/api/books/{book_id}/authoring/draft")
async def api_book_authoring_draft(book_id: int, request: Request):
    """SSE one-shot draft: segment → outline → VLM heroes → assemble → captions.
    Body: {filters:{date_from,date_to,camera,place,...}, notes?, target_spreads?,
    gap_minutes?, per_scene?, captions?}. Streams {type:'progress'|'done'|'fatal',
    phase, ...}."""
    import asyncio, threading
    from . import book_authoring
    try:
        data = await request.json()
    except Exception:
        data = {}
    filters = data.get("filters") or {}
    notes = data.get("notes")
    target_spreads = int(data.get("target_spreads") or 19)
    gap_minutes = float(data.get("gap_minutes") or 20)
    per_scene = int(data.get("per_scene") or 8)
    captions = bool(data.get("captions", True))

    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()

    def _emit(ev):
        asyncio.run_coroutine_threadsafe(aqueue.put(ev), loop)

    def run():
        try:
            with _get_db() as pdb, _get_books() as bs:
                if not bs.get_book_row(book_id):
                    _emit({"type": "fatal", "message": "book not found"}); return
                # Persist the notes/target on the book for reproducibility.
                bs.update_book(book_id, {"notes": notes, "target_spreads": target_spreads})
                for ev in book_authoring.draft_book(
                        pdb, bs, book_id, filters, notes, target_spreads,
                        gap_minutes, per_scene, captions):
                    if cancel_event.is_set():
                        _emit({"type": "cancelled", "message": "cancelled"}); return
                    out = dict(ev)
                    out["type"] = "done" if ev.get("phase") == "done" else "progress"
                    _emit(out)
        except Exception as exc:
            logger.exception("AUTHORING draft failed")
            _emit({"type": "fatal", "message": str(exc)})

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set(); return
                try:
                    ev = await asyncio.wait_for(aqueue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"; continue
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") in ("done", "fatal", "cancelled"):
                    return
        finally:
            cancel_event.set()

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/books/{book_id}/authoring/outline")
def api_book_outline(book_id: int):
    with _get_books() as bs:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        return {"outline": bs.get_outline(book_id), "book": bs.get_book_row(book_id)}


@app.put("/api/books/{book_id}/authoring/beats/{beat_id}")
def api_book_update_beat(book_id: int, beat_id: int, body: dict):
    with _get_books() as bs:
        try:
            bs.update_beat(beat_id, body)
        except KeyError:
            return JSONResponse({"error": "Beat not found"}, status_code=404)
        return {"outline": bs.get_outline(book_id)}


@app.put("/api/books/{book_id}/authoring/beats/{beat_id}/candidates/{photo_id}")
def api_book_set_beat_candidate(book_id: int, beat_id: int, photo_id: int, body: dict):
    with _get_books() as bs:
        bs.set_beat_candidate(beat_id, photo_id, body)
        return {"ok": True}


@app.post("/api/books/{book_id}/authoring/beats/{beat_id}/pick-heroes")
def api_book_pick_heroes(book_id: int, beat_id: int, body: dict):
    """Re-run the VLM hero pick for one beat over its current candidates."""
    from . import book_authoring
    with _get_books() as bs, _get_db() as pdb:
        beats = {b["id"]: b for b in bs.get_outline(book_id)}
        beat = beats.get(beat_id)
        if not beat:
            return JSONResponse({"error": "Beat not found"}, status_code=404)
        cand_ids = [c["photo_id"] for c in beat["candidates"] if c["role"] != "rejected"]
        n = int(body.get("n_heroes") or beat.get("spread_budget") or 1)
        picked = book_authoring.pick_heroes(pdb, cand_ids, beat.get("title") or "", n)
        heroes = set(picked["heroes"])
        for pos, r in enumerate(picked["ranked"]):
            pid = r["id"]
            is_hero = pid in heroes
            bs.set_beat_candidate(beat_id, pid, {
                "role": "hero" if is_hero else "candidate",
                "position": pos, "vlm_score": r.get("score"), "vlm_reason": r.get("reason"),
                "crop_mode": r.get("crop_mode") or "crop"})
        return {"outline": bs.get_outline(book_id)}


@app.get("/api/books/{book_id}/authoring/preview")
def api_book_preview(book_id: int):
    """The computed spread layout for the current outline (not persisted) — powers
    the review's live spread previews so you see how beats come together."""
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        return bs.preview_spreads(pdb, book_id)


@app.post("/api/books/{book_id}/authoring/assemble")
def api_book_assemble(book_id: int, body: dict):
    with _get_books() as bs, _get_db() as pdb:
        if not bs.get_book_row(book_id):
            return JSONResponse({"error": "Book not found"}, status_code=404)
        n = bs.assemble_from_outline(pdb, book_id)
        return {"spreads_created": n, "book": bs.get_book(book_id)}


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


@app.post("/api/admin/maintenance-sweep")
async def api_maintenance_sweep(request: Request):
    """M25 — SSE stream of the dependency-ordered backfill sweep.

    Body (all optional; defaults match the CLI, ``apply`` defaults False):
      {"apply"?, "do_colors"?, "do_stacking"?, "do_recluster"?, "do_dedup"?,
       "do_requeue"?, "requeue_passes"?,
       "window_minutes"?, "max_drift_km"?, "min_confidence"?}

    Wraps ``maintenance.run_maintenance_sweep`` in a worker thread and bridges
    its ``on_progress`` events to the client over the same thread→asyncio.Queue
    pattern as /api/stacks/detect/stream. Each sweep event already carries
    {"phase":"sweep","stage","status","would","applied","message"}; we tag it
    with type:"progress". Client disconnect (AbortController/tab close) flips a
    threading.Event that the sweep's ``should_abort`` checks between + inside
    stages. Terminal events: done / cancelled / fatal.
    """
    import asyncio
    import threading
    import time
    from .maintenance import run_maintenance_sweep

    try:
        data = await request.json()
    except Exception:
        data = {}
    data = data or {}

    apply = bool(data.get("apply", False))
    # Interactive default is LIGHT: the heavy stages (colors/stacking/match) peg
    # the N100 and starve this server, so they're opt-in here (the CLI/cron keep
    # them on by default for off-hours runs).
    do_colors = bool(data.get("do_colors", False))
    do_stacking = bool(data.get("do_stacking", False))
    do_match = bool(data.get("do_match", False))
    do_recluster = bool(data.get("do_recluster", False))
    do_dedup = bool(data.get("do_dedup", False))
    do_requeue = bool(data.get("do_requeue", False))
    # Opt-in full re-rank of the aesthetic percentiles — needed after a scoring
    # batch shifts the distribution (missing-only fills new rows but doesn't
    # re-rank existing ones) or after retuning the dimension weights.
    force_normalize_aesthetics = bool(data.get("force_normalize_aesthetics", False))
    force_normalize_subject_aesthetics = bool(
        data.get("force_normalize_subject_aesthetics", False))
    requeue_passes = data.get("requeue_passes") or None
    if requeue_passes is not None and not isinstance(requeue_passes, (list, tuple)):
        raise HTTPException(400, "requeue_passes must be a list of pass names")
    window_minutes = int(data.get("window_minutes", 30))
    max_drift_km = float(data.get("max_drift_km", 25.0))
    min_confidence = float(data.get("min_confidence", 0.0))

    if window_minutes <= 0 or window_minutes > 1440:
        raise HTTPException(400, "window_minutes must be in (0, 1440]")
    if max_drift_km <= 0 or max_drift_km > 10000:
        raise HTTPException(400, "max_drift_km must be in (0, 10000]")
    if min_confidence < 0 or min_confidence > 1:
        raise HTTPException(400, "min_confidence must be in [0, 1]")

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
                "apply": apply,
                "do_colors": do_colors,
                "do_stacking": do_stacking,
                "do_match": do_match,
                "do_recluster": do_recluster,
                "do_dedup": do_dedup,
                "do_requeue": do_requeue,
            })
            with _get_db() as db:
                result = run_maintenance_sweep(
                    db,
                    apply=apply,
                    do_colors=do_colors,
                    do_stacking=do_stacking,
                    do_match=do_match,
                    do_recluster=do_recluster,
                    do_dedup=do_dedup,
                    do_requeue=do_requeue,
                    force_normalize_aesthetics=force_normalize_aesthetics,
                    force_normalize_subject_aesthetics=force_normalize_subject_aesthetics,
                    requeue_passes=tuple(requeue_passes) if requeue_passes else None,
                    window_minutes=window_minutes,
                    max_drift_km=max_drift_km,
                    min_confidence=min_confidence,
                    on_progress=_on_progress,
                    should_abort=_should_abort,
                )
            stages = result.get("stages", [])
            _emit({
                "type": "done",
                "apply": apply,
                "stages": stages,
                "total_would": sum(s.get("would", 0) for s in stages),
                "total_applied": sum(s.get("applied", 0) for s in stages),
                "duration_seconds": round(time.monotonic() - started, 2),
            })
        except InterruptedError:
            _emit({"type": "cancelled", "message": "Maintenance sweep cancelled"})
        except Exception as exc:
            logger.exception("MAINTENANCE sweep stream failed")
            _emit({"type": "fatal", "message": str(exc)})

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("MAINTENANCE client disconnected — cancelling")
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


@app.get("/api/admin/validate-data")
def api_validate_data(sample: int = 5):
    """M25 — read-only data-integrity report (corrupt date_taken, bad GPS,
    malformed JSON columns, orphaned-vec / garbage-tag hints). Backs the
    quick "Validate" button on the Maintenance card."""
    from .maintenance import validate_data
    with _get_db() as db:
        return validate_data(db, sample=sample)


@app.post("/api/ask")
async def api_ask(request: Request):
    """M24b — natural-language search via the local-LLM agent loop (SSE).

    Body: {"message": str, "history"?: [{"role","content"}, ...],
           "filters"?: {people/location/date_from/date_to/color/category/
                        visual_tag/keyword/min_quality/min_aesthetic/style_tag/
                        match_source/camera/sort},
           "thinking"?: bool, "reasoning_effort"?: str}.
    `thinking` toggles a reasoning model's traces for this request (false =>
    reasoning_effort="none"). Omit it to use PHOTOSEARCH_LLM_REASONING_EFFORT.
    `filters` is the structured Search filter bar pinned in the UI — fed into
    every search the agent runs as a HARD constraint (enforced server-side, not
    a post-filter on results), so e.g. a pinned camera can't be dropped.

    Runs photosearch.agent.run_agent over the shared tool layer on the LOCAL
    LLM backend (LM Studio / Ollama). Nothing leaves the NAS. Streams the same
    event dicts the agent yields, plus a terminal `done`:
      {"type": "tool_call" | "tool_result" | "photos" | "answer" | "error", ...}
      {"type": "done"}

    Client disconnect (AbortController) flips a threading.Event the agent loop
    checks between steps.
    """
    import asyncio
    import threading
    from .agent import run_agent

    try:
        data = await request.json()
    except Exception:
        data = {}
    data = data or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") if isinstance(data.get("history"), list) else None
    # Structured Search filters pinned in the UI, fed in as HARD constraints on
    # every search the agent runs (not a post-filter on its results). The agent
    # normalizes + enforces them; see agent._normalize_locked / _merge_locked.
    locked_filters = data.get("filters") if isinstance(data.get("filters"), dict) else None
    # Thinking toggle. `thinking: false` suppresses a reasoning model's traces —
    # ~71% of its generated tokens, and roughly half the wall-clock, with no
    # measurable accuracy change (evals/mcp_bakeoff.py). Omit the key to fall back
    # to PHOTOSEARCH_LLM_REASONING_EFFORT. `reasoning_effort` is the escape hatch
    # for a backend that understands the other levels.
    reasoning_effort = data.get("reasoning_effort")
    if not isinstance(reasoning_effort, str):
        reasoning_effort = None
    if reasoning_effort is None and "thinking" in data:
        thinking = data.get("thinking")
        if isinstance(thinking, str):
            thinking = thinking.strip().lower() in ("1", "true", "yes", "on")
        reasoning_effort = "" if thinking else "none"
    # Consolidated-search toggle. When true, the agent is offered ONE
    # search(mode=...) tool instead of the 5 separate search-family tools — a
    # ~44% smaller prompt (fits small contexts; ~11s less cold prefill) that
    # routes on par at adequate context. Omit to use PHOTOSEARCH_CONSOLIDATED_SEARCH.
    consolidated = data.get("consolidated")
    if isinstance(consolidated, str):
        consolidated = consolidated.strip().lower() in ("1", "true", "yes", "on")
    elif not isinstance(consolidated, bool):
        consolidated = None
    if not message:
        raise HTTPException(400, "message is required")

    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    cancel_event = threading.Event()

    def _emit(event: dict):
        asyncio.run_coroutine_threadsafe(aqueue.put(event), loop)

    def run():
        try:
            with _get_db() as db:
                for event in run_agent(
                    db, message, history=history,
                    should_abort=cancel_event.is_set,
                    locked_filters=locked_filters,
                    reasoning_effort=reasoning_effort,
                    consolidated=consolidated,
                ):
                    _emit(event)
                    if cancel_event.is_set():
                        break
            _emit({"type": "done"})
        except Exception as exc:
            logger.exception("ASK agent failed")
            _emit({"type": "error", "message": str(exc)})
            _emit({"type": "done"})

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("ASK  client disconnected — cancelling")
                    cancel_event.set()
                    return
                try:
                    event = await asyncio.wait_for(aqueue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    return
        finally:
            cancel_event.set()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
      authenticated: bool — a token is stored AND still refreshable
      needs_reauth: bool — a token existed but its refresh failed (expired/
                    revoked); the dead token has been cleared
    """
    from .google_photos import is_configured, is_authenticated, refresh_access_token

    configured = is_configured(_db_path)
    had_token = is_authenticated(_db_path)
    needs_reauth = False
    authenticated = had_token

    # A bare token-file check reports "connected" even when the refresh token
    # has expired, so uploads then 500. Validate by attempting a refresh; this
    # only hits the network when the access token is near expiry, and
    # refresh_access_token clears the token on invalid_grant.
    if had_token:
        try:
            authenticated = bool(refresh_access_token(_db_path))
        except RuntimeError:
            authenticated = False
            needs_reauth = True
        except Exception:
            # Transient network/Google error — don't claim the user must
            # reconnect; leave them as authenticated and let upload surface it.
            authenticated = True

    return {
        "configured": configured,
        "authenticated": authenticated,
        "needs_reauth": needs_reauth,
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


@app.get("/api/logs")
def api_logs():
    """Return the Ask query logs for today + yesterday (for the /logs page).

    Reads the agent's per-query markdown logs (ask-YYYY-MM-DD.md plus any rotated
    ask-YYYY-MM-DD.N.md) from PHOTOSEARCH_ASK_LOG_DIR (default ./ask-logs).
    """
    import glob
    import time
    log_dir = os.environ.get("PHOTOSEARCH_ASK_LOG_DIR") or "ask-logs"
    today = time.strftime("%Y-%m-%d")
    yesterday = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
    days = []
    for day in (today, yesterday):
        files = sorted(glob.glob(os.path.join(log_dir, f"ask-{day}.md"))
                       + glob.glob(os.path.join(log_dir, f"ask-{day}.*.md")))
        content = ""
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as fh:
                    content += fh.read()
            except Exception:
                pass
        days.append({"date": day, "content": content,
                     "bytes": len(content), "files": [os.path.basename(f) for f in files]})
    return {"days": days, "dir": log_dir, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}


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

    @app.get("/book")
    @app.get("/book/{book_id}")
    def serve_book(book_id: int = 0):
        """Serve the photobook builder (JS reads the id from the URL)."""
        page = _frontend_dir / "book.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Book page not found</h1>")

    @app.get("/book/{book_id}/proof")
    def serve_book_proof(book_id: int):
        """Serve the print-friendly proof view (same page; JS renders proof mode)."""
        page = _frontend_dir / "book.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Book page not found</h1>")

    @app.get("/book/{book_id}/author")
    def serve_book_author(book_id: int):
        """Serve the authoring wizard (same page; JS renders the wizard)."""
        page = _frontend_dir / "book.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Book page not found</h1>")

    @app.get("/status")
    def serve_status():
        """Serve the indexing status page."""
        status_page = _frontend_dir / "status.html"
        if status_page.exists():
            return HTMLResponse(status_page.read_text())
        return HTMLResponse("<h1>Status page not found</h1>")

    @app.get("/logs")
    def serve_logs():
        """Serve the Ask query-logs viewer page."""
        page = _frontend_dir / "logs.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Logs page not found</h1>")

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

    @app.get("/admin/vocab")
    def serve_admin_vocab():
        """Serve the vocab curator page."""
        page = _frontend_dir / "admin_vocab.html"
        if page.exists():
            return HTMLResponse(page.read_text(),
                                headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Admin vocab page not found</h1>")

    @app.get("/admin/deploy")
    def serve_admin_deploy():
        """Serve the deployment admin page (git/docker/restart controls)."""
        page = _frontend_dir / "admin_deploy.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Admin deploy page not found</h1>")

    @app.get("/admin/maintenance")
    def serve_admin_maintenance():
        """Serve the maintenance admin page (sweep / stacking / infer-locations / ingest)."""
        page = _frontend_dir / "admin_maintenance.html"
        if page.exists():
            return HTMLResponse(page.read_text(), headers={"Cache-Control": "no-cache"})
        return HTMLResponse("<h1>Admin maintenance page not found</h1>")

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

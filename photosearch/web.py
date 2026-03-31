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

# Request logging — helps debug discrepancies between CLI and web results
logger = logging.getLogger("photosearch.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .db import PhotoDB

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="local-photo-search", version="0.1.0")

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

# Thumbnail cache directory
_thumb_dir: Optional[str] = None
_THUMB_SIZE = 600  # px, long edge


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
    limit: int = Query(50, ge=1, le=500),
    min_score: float = Query(-0.25, description="Minimum CLIP score"),
    min_quality: Optional[float] = Query(None, description="Minimum aesthetic quality (1-10)"),
    sort_quality: bool = Query(False, description="Sort by quality instead of relevance"),
    tag_match: str = Query("both", description="Tag matching mode: dict, tags, or both"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)"),
    location: Optional[str] = Query(None, description="Filter by location name"),
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

    with _get_db() as db:
        results = search_combined(
            db=db,
            query=q,
            person=person,
            color=color,
            place=place,
            limit=limit,
            min_score=min_score,
            min_quality=min_quality,
            sort_quality=sort_quality,
            tag_match=tag_match,
            date_from=date_from,
            date_to=date_to,
            location=location,
        )

        logger.info(
            "SEARCH RESULTS  count=%d  top_scores=%s",
            len(results),
            [(r.get("filename"), round(r.get("score", 0), 4)) for r in results[:5]],
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
            items.append(item)

    return {"results": items, "count": len(items)}


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
                      p.name as person_name
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


@app.get("/api/stats")
def api_stats():
    """Database statistics."""
    with _get_db() as db:
        photo_count = db.photo_count()
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

    return {
        "photos": photo_count,
        "faces": face_count,
        "persons": person_count,
        "described": described,
        "quality_scored": scored,
        "quality_stats": quality_stats,
    }


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

    @app.get("/{path:path}")
    def serve_frontend(path: str = ""):
        """Serve static files or fall back to index.html for SPA routing."""
        # Try serving as a static file first
        file_path = _frontend_dir / path
        if file_path.is_file():
            return FileResponse(str(file_path))
        # Fall back to index.html for SPA routing
        index = _frontend_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Frontend not found</h1>")

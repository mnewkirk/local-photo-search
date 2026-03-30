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
import os
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
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

# Thumbnail cache directory
_thumb_dir: Optional[str] = None
_THUMB_SIZE = 600  # px, long edge


def _get_db() -> PhotoDB:
    """Open a fresh DB connection per request."""
    return PhotoDB(_db_path)


def _ensure_thumb_dir():
    """Create thumbnail cache directory if needed."""
    global _thumb_dir
    if _thumb_dir is None:
        db_parent = Path(_db_path).resolve().parent
        _thumb_dir = str(db_parent / "thumbnails")
    Path(_thumb_dir).mkdir(parents=True, exist_ok=True)
    return _thumb_dir


def _get_or_create_thumbnail(photo: dict) -> str:
    """Return path to a cached thumbnail, generating it if needed."""
    from PIL import Image

    thumb_dir = _ensure_thumb_dir()
    filename = photo["filename"]
    stem = Path(filename).stem
    thumb_path = os.path.join(thumb_dir, f"{stem}_thumb.jpg")

    if os.path.exists(thumb_path):
        return thumb_path

    filepath = photo.get("filepath", "")
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Original not found: {filepath}")

    img = Image.open(filepath)
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
    min_score: float = Query(-0.20, description="Minimum CLIP score"),
):
    """Search photos using any combination of criteria."""
    if not any([q, person, color, place]):
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
                "description": r.get("description"),
                "camera_model": r.get("camera_model"),
                "focal_length": r.get("focal_length"),
                "exposure_time": r.get("exposure_time"),
                "f_number": r.get("f_number"),
                "iso": r.get("iso"),
                "image_width": r.get("image_width"),
                "image_height": r.get("image_height"),
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

    try:
        thumb_path = _get_or_create_thumbnail(photo)
        return FileResponse(thumb_path, media_type="image/jpeg")
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

    filepath = photo.get("filepath", "")
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(404, "Photo file not found on disk")

    return FileResponse(filepath, media_type="image/jpeg")


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
            "description": photo["description"],
            "camera_make": photo.get("camera_make"),
            "camera_model": photo.get("camera_model"),
            "focal_length": photo.get("focal_length"),
            "exposure_time": photo.get("exposure_time"),
            "f_number": photo.get("f_number"),
            "iso": photo.get("iso"),
            "image_width": photo.get("image_width"),
            "image_height": photo.get("image_height"),
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

    return {
        "photos": photo_count,
        "faces": face_count,
        "persons": person_count,
        "described": described,
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

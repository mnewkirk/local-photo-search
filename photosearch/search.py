"""Search logic for local-photo-search.

Combines CLIP semantic search, text search, color search, and face/person search.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from .clip_embed import embed_text
from .db import PhotoDB


# Minimum CLIP similarity score (1 - L2_distance) required to return a result.
# For L2-normalized 512-dim CLIP vectors, L2 distance = sqrt(2*(1-cos_sim)), so:
#   score -0.25 ≈ cosine similarity 0.22  (loose relevance)
#   score  0.00 ≈ cosine similarity 0.50  (clearly related)
# Queries like "ships" against unrelated family photos score ≈ -0.30, well below
# this threshold, so they return nothing rather than the whole collection.
CLIP_MIN_SCORE = -0.25

# Face-aware reranking: when a query mentions people, photos with detected faces
# get a score boost. This helps CLIP distinguish "people outdoors" from "outdoors"
# — something it can't do from embeddings alone when all photos are visually similar.
_PEOPLE_KEYWORDS = {
    "people", "person", "child", "children", "kid", "kids", "family",
    "man", "woman", "boy", "girl", "baby", "toddler", "adult",
    "group", "crowd", "portrait", "face", "faces",
}
FACE_BOOST = 0.02  # Score bonus per detected face (enough to lift a photo above
                    # neighbors in a tight cluster, but not so much that a random
                    # face-having photo leapfrogs a genuinely relevant result).


def _query_mentions_people(query: str) -> bool:
    """Return True if the query contains people-related keywords."""
    words = set(query.lower().split())
    return bool(words & _PEOPLE_KEYWORDS)


def search_semantic(
    db: PhotoDB,
    query: str,
    limit: int = 10,
    min_score: float = CLIP_MIN_SCORE,
) -> list[dict]:
    """Semantic search using CLIP — find photos matching a natural language query.

    Only returns photos whose CLIP similarity score exceeds min_score.
    Raise min_score (toward 0) for stricter matching; lower it (toward -1) to
    be more permissive. Default (-0.25) filters out clearly unrelated results.

    When the query mentions people-related terms, photos with detected faces
    receive a small score boost to help separate them from similar-looking
    scenes without people. This is a lightweight hybrid that leverages
    existing face detection data.
    """
    query_embedding = embed_text(query)
    if query_embedding is None:
        print("Error: could not generate embedding for query.")
        return []

    # Fetch more candidates than needed so the boost + re-sort can promote
    # face-having photos that would otherwise be cut off at the limit.
    fetch_limit = max(limit * 3, 30)
    matches = db.search_clip(query_embedding, limit=fetch_limit)

    boost_people = _query_mentions_people(query)

    # Pre-load face counts if we need them
    face_counts: dict[int, int] = {}
    if boost_people:
        rows = db.conn.execute(
            "SELECT photo_id, COUNT(*) as cnt FROM faces GROUP BY photo_id"
        ).fetchall()
        face_counts = {row["photo_id"]: row["cnt"] for row in rows}

    results = []
    for match in matches:
        raw_score = 1.0 - match["distance"]
        score = raw_score
        photo_id = match["photo_id"]

        # Apply face boost: each detected face adds FACE_BOOST to the score.
        if boost_people and photo_id in face_counts:
            score += face_counts[photo_id] * FACE_BOOST

        if score < min_score:
            continue  # Too dissimilar even after boost — skip

        photo = db.get_photo(photo_id)
        if photo:
            photo["score"] = score
            photo["clip_score"] = raw_score  # preserve raw CLIP score for diagnostics
            results.append(photo)

    # Re-sort by boosted score (CLIP + face bonus), descending
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


def search_by_color(db: PhotoDB, color: str, tolerance: int = 60, limit: int = 10) -> list[dict]:
    """Find photos with dominant colors near the given color.

    Accepts hex colors (#ff0000) or common color names.
    """
    color_hex = _resolve_color_name(color)
    return db.search_by_color(color_hex, tolerance=tolerance, limit=limit)


def search_by_place(db: PhotoDB, place: str, limit: int = 10) -> list[dict]:
    """Search by place name (text match)."""
    return db.search_text(place, limit=limit)


def search_by_person(db: PhotoDB, name: str, limit: int = 10) -> list[dict]:
    """Find all photos containing a named person.

    Looks up the person by name, then finds all faces linked to that person,
    then returns the distinct photos those faces appear in.
    """
    person = db.get_person_by_name(name)
    if not person:
        print(f"  Person '{name}' not found. Use 'add-person' to register them.")
        return []

    rows = db.conn.execute(
        """SELECT DISTINCT p.*
           FROM photos p
           JOIN faces f ON f.photo_id = p.id
           WHERE f.person_id = ?
           ORDER BY p.date_taken
           LIMIT ?""",
        (person["id"], limit),
    ).fetchall()
    return [dict(r) for r in rows]


def search_by_face_reference(db: PhotoDB, image_path: str, limit: int = 10) -> list[dict]:
    """Find photos containing a face similar to the one in the given reference image.

    Encodes the face in the reference image, then searches face_encodings for matches.
    """
    from .faces import encode_reference_photo, match_face
    import struct

    encoding = encode_reference_photo(image_path)
    if encoding is None:
        print(f"  No face found in reference image: {image_path}")
        return []

    matches = db.search_faces(encoding, limit=limit * 3)
    if not matches:
        return []

    # Get distinct photos for matched face IDs
    seen_photo_ids = set()
    results = []
    for match in matches:
        face_id = match["face_id"]
        face_row = db.conn.execute(
            "SELECT photo_id FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if face_row and face_row["photo_id"] not in seen_photo_ids:
            photo = db.get_photo(face_row["photo_id"])
            if photo:
                photo["face_distance"] = match["distance"]
                results.append(photo)
                seen_photo_ids.add(face_row["photo_id"])
        if len(results) >= limit:
            break

    return results


def search_combined(
    db: PhotoDB,
    query: Optional[str] = None,
    color: Optional[str] = None,
    place: Optional[str] = None,
    person: Optional[str] = None,
    face_image: Optional[str] = None,
    limit: int = 10,
    min_score: float = CLIP_MIN_SCORE,
) -> list[dict]:
    """Run multiple search types and merge results.

    When multiple criteria are given, returns the intersection
    ranked by the primary search type (person > semantic > color > place).
    """
    result_sets = []

    if person:
        results = search_by_person(db, person, limit=limit * 3)
        result_sets.append({r["id"]: r for r in results})

    if face_image:
        results = search_by_face_reference(db, face_image, limit=limit * 3)
        result_sets.append({r["id"]: r for r in results})

    if query:
        results = search_semantic(db, query, limit=limit * 3, min_score=min_score)
        result_sets.append({r["id"]: r for r in results})

    if color:
        results = search_by_color(db, color, limit=limit * 3)
        result_sets.append({r["photo_id"]: r for r in results})

    if place:
        results = search_by_place(db, place, limit=limit * 3)
        result_sets.append({r["id"]: r for r in results})

    if not result_sets:
        return []

    if len(result_sets) == 1:
        return list(result_sets[0].values())[:limit]

    # Intersect: only keep photos present in all result sets
    common_ids = set(result_sets[0].keys())
    for rs in result_sets[1:]:
        common_ids &= set(rs.keys())

    # Use first result set for ranking/data
    merged = [result_sets[0][pid] for pid in common_ids if pid in result_sets[0]]
    return merged[:limit]


def make_results_subdir(base_dir: str, query_parts: dict) -> str:
    """Generate a timestamped subfolder name from search criteria.

    Example: results/2026-03-29_14-32-05_q-beach_color-blue
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp]
    if query_parts.get("query"):
        slug = query_parts["query"].replace(" ", "-")[:30]
        parts.append(f"q-{slug}")
    if query_parts.get("color"):
        parts.append(f"color-{query_parts['color'].lstrip('#')}")
    if query_parts.get("place"):
        slug = query_parts["place"].replace(" ", "-")[:20]
        parts.append(f"place-{slug}")
    if query_parts.get("person"):
        slug = query_parts["person"].replace(" ", "-")[:20]
        parts.append(f"person-{slug}")
    return str(Path(base_dir) / "_".join(parts))


def symlink_results(results: list[dict], output_dir: str = "results", clear: bool = False,
                    thumbnail_size: int = 1200):
    """Write results to an output directory with both a symlink and a JPEG thumbnail per photo.

    For each result, two files are created:
      001_DSC04878.JPG          — relative symlink to the original (full resolution)
      001_DSC04878_thumbnail.JPG — resized JPEG for Finder preview

    The original photos are never modified.
    """
    from PIL import Image as PilImage

    output_path = Path(output_dir)

    if clear and output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results, 1):
        filepath = result.get("filepath")
        if not filepath or not os.path.exists(filepath):
            continue

        filename = os.path.basename(filepath)
        stem = Path(filename).stem
        ext = Path(filename).suffix  # preserve original extension (e.g. .JPG)

        base_name = f"{i:03d}_{stem}"
        link_path = output_path / f"{base_name}{ext}"
        thumb_path = output_path / f"{base_name}_thumbnail.jpg"

        # Relative symlink to original (full resolution)
        try:
            rel_target = os.path.relpath(filepath, str(output_path))
            os.symlink(rel_target, link_path)
        except OSError as e:
            print(f"  Warning: could not symlink {filename}: {e}")

        # JPEG thumbnail for Finder preview
        try:
            with PilImage.open(filepath) as img:
                img = img.convert("RGB")
                img.thumbnail((thumbnail_size, thumbnail_size), PilImage.LANCZOS)
                img.save(thumb_path, "JPEG", quality=85)
        except Exception as e:
            print(f"  Warning: could not create thumbnail for {filename}: {e}")

    return str(output_path.resolve())


# ------------------------------------------------------------------
# Color name resolution
# ------------------------------------------------------------------

_COLOR_NAMES = {
    "red": "#ff0000", "green": "#00aa00", "blue": "#0000ff",
    "yellow": "#ffff00", "orange": "#ff8800", "purple": "#8800aa",
    "pink": "#ff69b4", "brown": "#8b4513", "black": "#000000",
    "white": "#ffffff", "gray": "#808080", "grey": "#808080",
    "cyan": "#00ffff", "teal": "#008080", "navy": "#000080",
    "gold": "#ffd700", "silver": "#c0c0c0", "beige": "#f5f5dc",
    "tan": "#d2b48c", "olive": "#808000", "maroon": "#800000",
    "coral": "#ff7f50", "salmon": "#fa8072", "turquoise": "#40e0d0",
    "violet": "#ee82ee", "indigo": "#4b0082", "magenta": "#ff00ff",
    "lime": "#00ff00", "aqua": "#00ffff", "sky blue": "#87ceeb",
}


def _resolve_color_name(color: str) -> str:
    """Convert a color name to hex, or pass through if already hex."""
    if color.startswith("#"):
        return color
    return _COLOR_NAMES.get(color.lower().strip(), f"#{color}")

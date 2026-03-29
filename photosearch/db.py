"""Database schema and access layer for local-photo-search.

Uses SQLite with sqlite-vec for vector similarity search.
All photo annotations live here — original photos are never modified.
"""

import json
import sqlite3
import struct
from pathlib import Path
from typing import Optional

# sqlite-vec will be imported at init time so we can fail gracefully
try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

# Dimensions for our embedding vectors
CLIP_DIMENSIONS = 512
FACE_DIMENSIONS = 128

SCHEMA_VERSION = 1


def _serialize_float_list(vec: list[float]) -> bytes:
    """Serialize a list of floats to a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_float_list(data: bytes, dim: int) -> list[float]:
    """Deserialize binary float vector back to a list."""
    return list(struct.unpack(f"{dim}f", data))


class PhotoDB:
    """Manages the SQLite database for photo metadata and embeddings."""

    def __init__(self, db_path: str = "photo_index.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

        if HAS_SQLITE_VEC:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)

        self._init_schema()

    def _init_schema(self):
        """Create tables if they don't exist."""
        cur = self.conn.cursor()

        # Schema version tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Core photos table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT,
                date_taken TEXT,
                gps_lat REAL,
                gps_lon REAL,
                place_name TEXT,
                camera_make TEXT,
                camera_model TEXT,
                focal_length TEXT,
                exposure_time TEXT,
                f_number TEXT,
                iso INTEGER,
                image_width INTEGER,
                image_height INTEGER,
                description TEXT,
                dominant_colors TEXT,
                indexed_at TEXT DEFAULT (datetime('now')),
                raw_filepath TEXT
            )
        """)

        # Persons table — named people (from clustering or pre-seeded)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Faces detected in photos
        cur.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                person_id INTEGER REFERENCES persons(id),
                bbox_top INTEGER,
                bbox_right INTEGER,
                bbox_bottom INTEGER,
                bbox_left INTEGER,
                cluster_id INTEGER,
                FOREIGN KEY (photo_id) REFERENCES photos(id),
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)

        # Reference face encodings for pre-seeded known people
        cur.execute("""
            CREATE TABLE IF NOT EXISTS face_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                source_path TEXT,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)

        # Indexes for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_place ON photos(place_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)")

        # sqlite-vec virtual tables for vector search
        if HAS_SQLITE_VEC:
            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS clip_embeddings
                USING vec0(
                    photo_id INTEGER PRIMARY KEY,
                    embedding float[{CLIP_DIMENSIONS}]
                )
            """)

            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS face_encodings
                USING vec0(
                    face_id INTEGER PRIMARY KEY,
                    encoding float[{FACE_DIMENSIONS}]
                )
            """)

            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS face_ref_encodings
                USING vec0(
                    ref_id INTEGER PRIMARY KEY,
                    encoding float[{FACE_DIMENSIONS}]
                )
            """)

        # Set schema version
        cur.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("version", str(SCHEMA_VERSION)),
        )

        self.conn.commit()

    # ------------------------------------------------------------------
    # Photo CRUD
    # ------------------------------------------------------------------

    def add_photo(self, **kwargs) -> int:
        """Insert a photo record. Returns the photo id."""
        columns = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cur = self.conn.execute(
            f"INSERT OR IGNORE INTO photos ({columns}) VALUES ({placeholders})",
            list(kwargs.values()),
        )
        self.conn.commit()
        if cur.lastrowid == 0:
            # Photo already existed — fetch its id
            row = self.conn.execute(
                "SELECT id FROM photos WHERE filepath = ?", (kwargs["filepath"],)
            ).fetchone()
            return row["id"]
        return cur.lastrowid

    def get_photo(self, photo_id: int) -> Optional[dict]:
        """Fetch a photo by id."""
        row = self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,)).fetchone()
        return dict(row) if row else None

    def get_photo_by_path(self, filepath: str) -> Optional[dict]:
        """Fetch a photo by filepath."""
        row = self.conn.execute(
            "SELECT * FROM photos WHERE filepath = ?", (filepath,)
        ).fetchone()
        return dict(row) if row else None

    def update_photo(self, photo_id: int, **kwargs):
        """Update fields on a photo record."""
        set_clause = ", ".join(f"{k} = ?" for k in kwargs.keys())
        self.conn.execute(
            f"UPDATE photos SET {set_clause} WHERE id = ?",
            list(kwargs.values()) + [photo_id],
        )
        self.conn.commit()

    def photo_count(self) -> int:
        """Return total number of indexed photos."""
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM photos").fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # CLIP embeddings
    # ------------------------------------------------------------------

    def add_clip_embedding(self, photo_id: int, embedding: list[float]):
        """Store a CLIP embedding for a photo."""
        if not HAS_SQLITE_VEC:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO clip_embeddings (photo_id, embedding) VALUES (?, ?)",
            (photo_id, _serialize_float_list(embedding)),
        )
        self.conn.commit()

    def search_clip(self, query_embedding: list[float], limit: int = 10) -> list[dict]:
        """Find photos most similar to a query embedding. Returns list of {photo_id, distance}."""
        if not HAS_SQLITE_VEC:
            return []
        rows = self.conn.execute(
            """
            SELECT photo_id, distance
            FROM clip_embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_serialize_float_list(query_embedding), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Face encodings
    # ------------------------------------------------------------------

    def add_face(self, photo_id: int, bbox: tuple, encoding: list[float],
                 person_id: Optional[int] = None, cluster_id: Optional[int] = None) -> int:
        """Store a detected face with its encoding."""
        cur = self.conn.execute(
            """INSERT INTO faces (photo_id, person_id, bbox_top, bbox_right, bbox_bottom, bbox_left, cluster_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (photo_id, person_id, bbox[0], bbox[1], bbox[2], bbox[3], cluster_id),
        )
        face_id = cur.lastrowid

        if HAS_SQLITE_VEC and encoding:
            self.conn.execute(
                "INSERT INTO face_encodings (face_id, encoding) VALUES (?, ?)",
                (face_id, _serialize_float_list(encoding)),
            )
        self.conn.commit()
        return face_id

    def search_faces(self, query_encoding: list[float], limit: int = 10) -> list[dict]:
        """Find faces most similar to a query encoding."""
        if not HAS_SQLITE_VEC:
            return []
        rows = self.conn.execute(
            """
            SELECT face_id, distance
            FROM face_encodings
            WHERE encoding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (_serialize_float_list(query_encoding), limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Persons
    # ------------------------------------------------------------------

    def add_person(self, name: str) -> int:
        """Create a named person. Returns person id."""
        cur = self.conn.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        self.conn.commit()
        return cur.lastrowid

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """Look up a person by name (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM persons WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        return dict(row) if row else None

    def assign_face_to_person(self, face_id: int, person_id: int):
        """Link a face to a named person."""
        self.conn.execute(
            "UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Color search
    # ------------------------------------------------------------------

    def search_by_color(self, color_hex: str, tolerance: int = 60, limit: int = 10) -> list[dict]:
        """Find photos whose dominant colors are near the given hex color.

        Colors are stored as a JSON list of hex strings in photos.dominant_colors.
        This does an in-Python distance check — fast enough for <10k photos.
        """
        target_rgb = _hex_to_rgb(color_hex)
        results = []
        rows = self.conn.execute(
            "SELECT id, filepath, filename, dominant_colors FROM photos WHERE dominant_colors IS NOT NULL"
        ).fetchall()
        for row in rows:
            colors = json.loads(row["dominant_colors"])
            min_dist = min(_color_distance(target_rgb, _hex_to_rgb(c)) for c in colors)
            if min_dist <= tolerance:
                results.append({"photo_id": row["id"], "filepath": row["filepath"],
                                "filename": row["filename"], "distance": min_dist})
        results.sort(key=lambda x: x["distance"])
        return results[:limit]

    # ------------------------------------------------------------------
    # Full-text search on descriptions and places
    # ------------------------------------------------------------------

    def search_text(self, query: str, limit: int = 10) -> list[dict]:
        """Simple LIKE search across description and place_name."""
        pattern = f"%{query}%"
        rows = self.conn.execute(
            """SELECT id, filepath, filename, description, place_name
               FROM photos
               WHERE description LIKE ? OR place_name LIKE ?
               ORDER BY date_taken
               LIMIT ?""",
            (pattern, pattern, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ------------------------------------------------------------------
# Color helpers
# ------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' or 'RRGGBB' to (R, G, B)."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Euclidean distance between two RGB colors."""
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

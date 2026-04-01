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
FACE_DIMENSIONS = 512  # InsightFace ArcFace produces 512-dim L2-normalized vectors

SCHEMA_VERSION = 9


def _serialize_float_list(vec: list[float]) -> bytes:
    """Serialize a list of floats to a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_float_list(data: bytes, dim: int) -> list[float]:
    """Deserialize binary float vector back to a list."""
    return list(struct.unpack(f"{dim}f", data))


class PhotoDB:
    """Manages the SQLite database for photo metadata and embeddings.

    Photo file paths are stored relative to a configurable ``photo_root``.
    At runtime the root is resolved via (in priority order):
      1. The ``photo_root`` constructor argument.
      2. The ``PHOTO_ROOT`` environment variable.
      3. The ``photo_root`` value stored in the ``schema_info`` table.

    Use :meth:`resolve_filepath` to turn a stored relative path into an
    absolute path that can be opened, and :meth:`relative_filepath` to
    convert an absolute path for storage.  The helpers are no-ops when
    the stored path is already absolute and no root is configured (i.e.
    backwards-compatible).
    """

    def __init__(self, db_path: str = "photo_index.db", photo_root: Optional[str] = None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes with WAL
        self.conn.execute("PRAGMA cache_size=-64000")    # 64 MB page cache

        if HAS_SQLITE_VEC:
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)

        self._batch_mode = False
        self._batch_count = 0
        self._batch_size = 50  # Commit every N operations in batch mode

        self._init_schema()

        # Resolve photo_root: constructor arg > env var > DB setting > None
        import os as _os
        if photo_root:
            self.photo_root = str(Path(photo_root).resolve())
        elif _os.environ.get("PHOTO_ROOT"):
            self.photo_root = str(Path(_os.environ["PHOTO_ROOT"]).resolve())
        else:
            row = self.conn.execute(
                "SELECT value FROM schema_info WHERE key = 'photo_root'"
            ).fetchone()
            self.photo_root = row["value"] if row else None

    def begin_batch(self, batch_size: int = 50):
        """Enter batch mode: defer commits until flush_batch() or every batch_size ops."""
        self._batch_mode = True
        self._batch_size = batch_size
        self._batch_count = 0

    def flush_batch(self):
        """Force a commit of any pending batch operations."""
        if self._batch_count > 0:
            self.conn.commit()
            self._batch_count = 0

    def end_batch(self):
        """Exit batch mode and commit any remaining operations."""
        self.flush_batch()
        self._batch_mode = False

    def _maybe_commit(self):
        """Commit immediately, or defer if in batch mode."""
        if self._batch_mode:
            self._batch_count += 1
            if self._batch_count >= self._batch_size:
                self.conn.commit()
                self._batch_count = 0
        else:
            self.conn.commit()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def set_photo_root(self, root: str):
        """Persist the photo root directory in the database."""
        self.photo_root = str(Path(root).resolve())
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
            ("photo_root", self.photo_root),
        )
        self.conn.commit()

    def relative_filepath(self, absolute_path: str) -> str:
        """Convert an absolute path to a path relative to photo_root.

        If no photo_root is set, returns the absolute path unchanged.
        """
        if not self.photo_root:
            return absolute_path
        try:
            return str(Path(absolute_path).relative_to(self.photo_root))
        except ValueError:
            # Path is not under photo_root — store as-is
            return absolute_path

    def resolve_filepath(self, stored_path: str) -> str:
        """Convert a stored (possibly relative) path to an absolute path.

        If photo_root is set and the path is relative, prepend the root.
        Otherwise returns the path unchanged.
        """
        if not stored_path:
            return stored_path
        p = Path(stored_path)
        if p.is_absolute():
            return stored_path
        if self.photo_root:
            return str(Path(self.photo_root) / p)
        return stored_path

    def remap_paths(self, old_prefix: str, new_prefix: str) -> int:
        """Bulk-replace a path prefix in all photo filepaths.

        Useful when moving photos between machines / mount points.
        Returns the number of rows updated.

        Example:
            db.remap_paths("/Users/matt/Photos", "/Photos")
        """
        rows = self.conn.execute(
            "SELECT id, filepath, raw_filepath FROM photos WHERE filepath LIKE ?",
            (old_prefix + "%",),
        ).fetchall()
        count = 0
        for row in rows:
            new_path = new_prefix + row["filepath"][len(old_prefix):]
            updates = {"filepath": new_path}
            if row["raw_filepath"] and row["raw_filepath"].startswith(old_prefix):
                updates["raw_filepath"] = new_prefix + row["raw_filepath"][len(old_prefix):]
            set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
            self.conn.execute(
                f"UPDATE photos SET {set_clause} WHERE id = ?",
                list(updates.values()) + [row["id"]],
            )
            count += 1
        self.conn.commit()
        return count

    def _init_schema(self):
        """Create tables if they don't exist.

        If the schema is already at the current version, this is a no-op
        (no writes attempted), which allows read-only access even when
        another process holds a write lock on the database.
        """
        cur = self.conn.cursor()

        # Schema version tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Fast path: if schema is already at the current version, skip all DDL/writes.
        try:
            row = cur.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            ).fetchone()
            if row and int(row[0]) >= SCHEMA_VERSION:
                return
        except Exception:
            pass  # table might be empty or missing — proceed with full init

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
                aesthetic_score REAL,
                aesthetic_concepts TEXT,
                aesthetic_critique TEXT,
                tags TEXT,
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
                match_source TEXT,
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

        # Migration: add aesthetic_score column if upgrading from schema v2
        try:
            cur.execute("SELECT aesthetic_score FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN aesthetic_score REAL")

        # Migration: add aesthetic explanation columns if upgrading from schema v3
        try:
            cur.execute("SELECT aesthetic_concepts FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN aesthetic_concepts TEXT")
        try:
            cur.execute("SELECT aesthetic_critique FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN aesthetic_critique TEXT")

        # Migration: add tags column if upgrading from schema v4
        try:
            cur.execute("SELECT tags FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN tags TEXT")

        # Migration: add match_source column to faces if upgrading from schema v6
        try:
            cur.execute("SELECT match_source FROM faces LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE faces ADD COLUMN match_source TEXT")

        # Migration: unique constraint on persons.name (schema v7)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_persons_name_unique ON persons(LOWER(name))")

        # Migration: hallucination verification columns (schema v9)
        try:
            cur.execute("SELECT verified_at FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN verified_at TEXT")
            cur.execute("ALTER TABLE photos ADD COLUMN verification_status TEXT")  # 'pass', 'fail', 'regenerated'
            cur.execute("ALTER TABLE photos ADD COLUMN hallucination_flags TEXT")  # JSON: what was flagged

        # Ignored clusters — hide unknown faces the user doesn't care about
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ignored_clusters (
                cluster_id INTEGER PRIMARY KEY,
                ignored_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Review selections — persists shoot review picks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS review_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                directory TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 0,
                cluster_id INTEGER,
                rank_in_cluster INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(photo_id, directory)
            )
        """)

        # Indexes for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_place ON photos(place_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_aesthetic ON photos(aesthetic_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_review_dir ON review_selections(directory)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_review_photo ON review_selections(photo_id)")

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

        self.conn.commit()  # Schema init always commits immediately

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
        self._maybe_commit()
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
        self._maybe_commit()

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
        self._maybe_commit()

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
        self._maybe_commit()
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

    def get_face_encoding(self, face_id: int) -> list[float] | None:
        """Retrieve the raw encoding vector for a face."""
        if not HAS_SQLITE_VEC:
            return None
        row = self.conn.execute(
            "SELECT encoding FROM face_encodings WHERE face_id = ?", (face_id,)
        ).fetchone()
        if row is None:
            return None
        return _deserialize_float_list(row["encoding"], FACE_DIMENSIONS)

    def get_face_encodings_bulk(self, face_ids: list[int]) -> dict[int, list[float]]:
        """Retrieve encodings for multiple faces at once. Returns {face_id: encoding}.

        Batches queries in chunks of 500 to stay within SQLite parameter limits
        and work well with sqlite-vec virtual tables.
        """
        if not HAS_SQLITE_VEC or not face_ids:
            return {}
        result = {}
        batch_size = 500
        for i in range(0, len(face_ids), batch_size):
            batch = face_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            rows = self.conn.execute(
                f"SELECT face_id, encoding FROM face_encodings WHERE face_id IN ({placeholders})",
                batch,
            ).fetchall()
            for r in rows:
                result[r["face_id"]] = _deserialize_float_list(r["encoding"], FACE_DIMENSIONS)
        return result

    # ------------------------------------------------------------------
    # Persons
    # ------------------------------------------------------------------

    def add_person(self, name: str) -> int:
        """Create a named person, or return the existing id if the name exists (case-insensitive)."""
        existing = self.get_person_by_name(name)
        if existing:
            return existing["id"]
        cur = self.conn.execute("INSERT INTO persons (name) VALUES (?)", (name,))
        self._maybe_commit()
        return cur.lastrowid

    def get_person_by_name(self, name: str) -> Optional[dict]:
        """Look up a person by name (case-insensitive)."""
        row = self.conn.execute(
            "SELECT * FROM persons WHERE LOWER(name) = LOWER(?)", (name,)
        ).fetchone()
        return dict(row) if row else None

    def assign_face_to_person(self, face_id: int, person_id: int, match_source: str | None = None):
        """Link a face to a named person.

        match_source: 'strict', 'temporal', or 'manual'.  Stored for filtering.
        """
        self.conn.execute(
            "UPDATE faces SET person_id = ?, match_source = ? WHERE id = ?",
            (person_id, match_source, face_id),
        )
        self._maybe_commit()

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
        # Flush any pending batch writes before closing
        if self._batch_mode:
            self.end_batch()
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

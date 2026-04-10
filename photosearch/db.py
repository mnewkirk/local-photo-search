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

SCHEMA_VERSION = 14


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
        self.conn = sqlite3.connect(db_path, timeout=60.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")   # 60s retry at SQLite level
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes with WAL
        self.conn.execute("PRAGMA cache_size=-64000")    # 64 MB page cache
        self.conn.execute("PRAGMA wal_autocheckpoint=1000")  # Checkpoint every 1000 pages

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

        # Upload ledger — tracks which photos have already been uploaded to which album.
        # Keyed by (album_id, filepath) so re-uploads are skipped without any API calls.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS google_photos_uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                album_id TEXT NOT NULL,
                photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
                filepath TEXT NOT NULL,
                media_item_id TEXT,
                uploaded_at TEXT DEFAULT (datetime('now')),
                UNIQUE(album_id, filepath)
            )
        """)

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

        # Photo stacks — burst/bracket groups of near-identical shots
        cur.execute("""
            CREATE TABLE IF NOT EXISTS photo_stacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS stack_members (
                stack_id INTEGER NOT NULL REFERENCES photo_stacks(id) ON DELETE CASCADE,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                is_top INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (stack_id, photo_id),
                UNIQUE(photo_id)
            )
        """)

        # Collections — named groups of photos (albums)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                cover_photo_id INTEGER REFERENCES photos(id) ON DELETE SET NULL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS collection_photos (
                collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
                photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
                added_at TEXT DEFAULT (datetime('now')),
                sort_order INTEGER DEFAULT 0,
                PRIMARY KEY (collection_id, photo_id)
            )
        """)

        # Migration: Google Photos album link on collections (must run after collections table exists)
        try:
            cur.execute("SELECT google_photos_album_id FROM collections LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE collections ADD COLUMN google_photos_album_id TEXT")
            cur.execute("ALTER TABLE collections ADD COLUMN google_photos_album_title TEXT")

        # Worker claims — distributed indexing coordination
        cur.execute("""
            CREATE TABLE IF NOT EXISTS worker_claims (
                batch_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                pass_type TEXT NOT NULL,
                photo_ids TEXT NOT NULL,
                claimed_at TEXT DEFAULT (datetime('now')),
                expires_at TEXT NOT NULL
            )
        """)

        # Worker processed — tracks which (photo, pass) combos are done.
        # Needed because some passes (e.g. faces) produce no rows for photos
        # with no results, so we can't use "has rows in X" as a processed check.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS worker_processed (
                photo_id INTEGER NOT NULL,
                pass_type TEXT NOT NULL,
                processed_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (photo_id, pass_type)
            )
        """)

        # Indexes for common queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_worker_claims_expire ON worker_claims(expires_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date_taken)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_place ON photos(place_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_aesthetic ON photos(aesthetic_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_photo ON faces(photo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_faces_cluster ON faces(cluster_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_review_dir ON review_selections(directory)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_review_photo ON review_selections(photo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stack_members_stack ON stack_members(stack_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_stack_members_photo ON stack_members(photo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_collection_photos_coll ON collection_photos(collection_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_collection_photos_photo ON collection_photos(photo_id)")

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
        blob = _serialize_float_list(query_embedding)
        try:
            rows = self.conn.execute(
                """
                SELECT photo_id, distance
                FROM clip_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (blob, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Older sqlite-vec versions require 'k = ?' instead of LIMIT
            rows = self.conn.execute(
                """
                SELECT photo_id, distance
                FROM clip_embeddings
                WHERE embedding MATCH ?
                AND k = ?
                ORDER BY distance
                """,
                (blob, limit),
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
        blob = _serialize_float_list(query_encoding)
        try:
            rows = self.conn.execute(
                """
                SELECT face_id, distance
                FROM face_encodings
                WHERE encoding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (blob, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Older sqlite-vec versions require 'k = ?' instead of LIMIT
            rows = self.conn.execute(
                """
                SELECT face_id, distance
                FROM face_encodings
                WHERE encoding MATCH ?
                AND k = ?
                ORDER BY distance
                """,
                (blob, limit),
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

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def create_collection(self, name: str, description: str = None) -> int:
        """Create a new collection. Returns the collection id."""
        cur = self.conn.execute(
            "INSERT INTO collections (name, description) VALUES (?, ?)",
            (name, description),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_collection(self, collection_id: int) -> dict | None:
        """Get a single collection by id."""
        row = self.conn.execute(
            "SELECT * FROM collections WHERE id = ?", (collection_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_collection_by_name(self, name: str) -> dict | None:
        """Get a collection by name."""
        row = self.conn.execute(
            "SELECT * FROM collections WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def list_collections(self) -> list[dict]:
        """List all collections with photo counts."""
        rows = self.conn.execute("""
            SELECT c.*,
                   COUNT(cp.photo_id) AS photo_count,
                   COALESCE(c.cover_photo_id, (
                       SELECT cp2.photo_id FROM collection_photos cp2
                       WHERE cp2.collection_id = c.id
                       ORDER BY cp2.sort_order, cp2.added_at
                       LIMIT 1
                   )) AS effective_cover_photo_id
            FROM collections c
            LEFT JOIN collection_photos cp ON cp.collection_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """).fetchall()
        return [dict(r) for r in rows]

    def rename_collection(self, collection_id: int, name: str) -> None:
        """Rename a collection."""
        self.conn.execute(
            "UPDATE collections SET name = ?, updated_at = datetime('now') WHERE id = ?",
            (name, collection_id),
        )
        self.conn.commit()

    def update_collection_description(self, collection_id: int, description: str) -> None:
        """Update a collection's description."""
        self.conn.execute(
            "UPDATE collections SET description = ?, updated_at = datetime('now') WHERE id = ?",
            (description, collection_id),
        )
        self.conn.commit()

    def set_collection_google_album(self, collection_id: int, album_id: str, album_title: str) -> None:
        """Store the linked Google Photos album ID and title on a collection."""
        self.conn.execute(
            "UPDATE collections SET google_photos_album_id = ?, google_photos_album_title = ?, updated_at = datetime('now') WHERE id = ?",
            (album_id, album_title, collection_id),
        )
        self.conn.commit()

    def get_uploaded_filepaths(self, album_id: str) -> set:
        """Return the set of filepaths already uploaded to the given album."""
        rows = self.conn.execute(
            "SELECT filepath FROM google_photos_uploads WHERE album_id = ?", (album_id,)
        ).fetchall()
        return {row[0] for row in rows}

    def record_upload(self, album_id: str, photo_id: int, filepath: str, media_item_id: str) -> None:
        """Record a successful upload in the ledger."""
        self.conn.execute(
            """INSERT OR REPLACE INTO google_photos_uploads
               (album_id, photo_id, filepath, media_item_id, uploaded_at)
               VALUES (?, ?, ?, ?, datetime('now'))""",
            (album_id, photo_id, filepath, media_item_id),
        )
        self.conn.commit()

    def clear_upload_ledger(self, album_id: str) -> int:
        """Delete all ledger entries for an album (forces full re-upload). Returns count deleted."""
        cur = self.conn.execute(
            "DELETE FROM google_photos_uploads WHERE album_id = ?", (album_id,)
        )
        self.conn.commit()
        return cur.rowcount

    def set_collection_cover(self, collection_id: int, photo_id: int) -> None:
        """Set the cover photo for a collection."""
        self.conn.execute(
            "UPDATE collections SET cover_photo_id = ?, updated_at = datetime('now') WHERE id = ?",
            (photo_id, collection_id),
        )
        self.conn.commit()

    def delete_collection(self, collection_id: int) -> None:
        """Delete a collection (cascade removes collection_photos rows)."""
        self.conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        self.conn.commit()

    def add_photos_to_collection(self, collection_id: int, photo_ids: list[int]) -> int:
        """Add photos to a collection. Returns count of newly added photos."""
        added = 0
        # Get current max sort_order
        row = self.conn.execute(
            "SELECT COALESCE(MAX(sort_order), -1) FROM collection_photos WHERE collection_id = ?",
            (collection_id,),
        ).fetchone()
        next_order = row[0] + 1

        for pid in photo_ids:
            try:
                self.conn.execute(
                    "INSERT INTO collection_photos (collection_id, photo_id, sort_order) VALUES (?, ?, ?)",
                    (collection_id, pid, next_order),
                )
                added += 1
                next_order += 1
            except Exception:
                # Already in collection (PRIMARY KEY conflict) — skip
                pass
        self.conn.execute(
            "UPDATE collections SET updated_at = datetime('now') WHERE id = ?",
            (collection_id,),
        )
        self.conn.commit()
        return added

    def remove_photos_from_collection(self, collection_id: int, photo_ids: list[int]) -> int:
        """Remove photos from a collection. Returns count removed."""
        placeholders = ",".join("?" * len(photo_ids))
        cur = self.conn.execute(
            f"DELETE FROM collection_photos WHERE collection_id = ? AND photo_id IN ({placeholders})",
            [collection_id] + photo_ids,
        )
        self.conn.execute(
            "UPDATE collections SET updated_at = datetime('now') WHERE id = ?",
            (collection_id,),
        )
        self.conn.commit()
        return cur.rowcount

    def get_collection_photos(self, collection_id: int) -> list[dict]:
        """Get all photos in a collection with full photo metadata."""
        rows = self.conn.execute("""
            SELECT p.*, cp.sort_order, cp.added_at AS added_to_collection
            FROM collection_photos cp
            JOIN photos p ON p.id = cp.photo_id
            WHERE cp.collection_id = ?
            ORDER BY cp.sort_order, cp.added_at
        """, (collection_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_collection_photo_ids(self, collection_id: int) -> list[int]:
        """Get just the photo IDs in a collection (lightweight)."""
        rows = self.conn.execute(
            "SELECT photo_id FROM collection_photos WHERE collection_id = ? ORDER BY sort_order",
            (collection_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def get_directory_photo_ids(self, directory: str) -> list[int]:
        """Get photo IDs whose filepath starts with the given directory prefix.

        Normalizes the input: strips leading './' and ensures trailing '/'
        so '2026' doesn't match '2026-extra'.
        """
        prefix = directory.strip()
        # Strip leading ./ since DB paths are relative without it
        while prefix.startswith("./"):
            prefix = prefix[2:]
        prefix = prefix.strip("/") + "/"
        rows = self.conn.execute(
            "SELECT id FROM photos WHERE filepath LIKE ?",
            (prefix + "%",),
        ).fetchall()
        return [row[0] for row in rows]

    def get_collection_photo_pairs(self, collection_id: int) -> list[tuple[int, str]]:
        """Get (photo_id, absolute_path) pairs for photos in a collection.

        Used by indexing commands to scope work to a specific collection.
        Returns resolved absolute paths using photo_root.
        """
        rows = self.conn.execute("""
            SELECT p.id, p.filepath
            FROM collection_photos cp
            JOIN photos p ON p.id = cp.photo_id
            WHERE cp.collection_id = ?
            ORDER BY cp.sort_order
        """, (collection_id,)).fetchall()
        return [(row["id"], self.resolve_filepath(row["filepath"])) for row in rows]

    def expand_to_stacks(self, photo_ids: list[int]) -> list[int]:
        """Given a list of photo IDs, expand to include all stack members.

        Returns a deduplicated list of photo IDs that includes:
        - All original photo_ids
        - Any other photos that share a stack with any of the originals
        """
        if not photo_ids:
            return []
        result = set(photo_ids)
        # Find stacks containing any of the given photos
        for i in range(0, len(photo_ids), 500):
            chunk = photo_ids[i:i + 500]
            placeholders = ",".join("?" * len(chunk))
            stack_rows = self.conn.execute(
                f"SELECT DISTINCT stack_id FROM stack_members WHERE photo_id IN ({placeholders})",
                chunk,
            ).fetchall()
            stack_ids = [r[0] for r in stack_rows]
            if stack_ids:
                sp = ",".join("?" * len(stack_ids))
                member_rows = self.conn.execute(
                    f"SELECT photo_id FROM stack_members WHERE stack_id IN ({sp})",
                    stack_ids,
                ).fetchall()
                result.update(r[0] for r in member_rows)
        return list(result)

    def get_photo_collections(self, photo_id: int) -> list[dict]:
        """Get all collections a photo belongs to."""
        rows = self.conn.execute("""
            SELECT c.id, c.name
            FROM collections c
            JOIN collection_photos cp ON cp.collection_id = c.id
            WHERE cp.photo_id = ?
            ORDER BY c.name
        """, (photo_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Photo stacks (burst/bracket groups)
    # ------------------------------------------------------------------

    def create_stack(self, photo_ids: list[int], top_photo_id: int | None = None) -> int:
        """Create a new stack from a list of photo IDs.

        Args:
            photo_ids: At least 2 photo IDs to group.
            top_photo_id: Which photo is the stack's "best" pick.
                If None, the first ID in the list is used.

        Returns:
            The new stack ID.
        """
        if len(photo_ids) < 2:
            raise ValueError("A stack requires at least 2 photos")
        if top_photo_id is None:
            top_photo_id = photo_ids[0]

        # Remove these photos from any existing stacks first
        placeholders = ",".join("?" * len(photo_ids))
        self.conn.execute(
            f"DELETE FROM stack_members WHERE photo_id IN ({placeholders})",
            photo_ids,
        )
        # Clean up orphaned stacks (stacks with no remaining members)
        self.conn.execute("""
            DELETE FROM photo_stacks
            WHERE id NOT IN (SELECT DISTINCT stack_id FROM stack_members)
        """)

        cur = self.conn.execute("INSERT INTO photo_stacks DEFAULT VALUES")
        stack_id = cur.lastrowid

        for pid in photo_ids:
            self.conn.execute(
                "INSERT INTO stack_members (stack_id, photo_id, is_top) VALUES (?, ?, ?)",
                (stack_id, pid, 1 if pid == top_photo_id else 0),
            )
        self._maybe_commit()
        return stack_id

    def get_stack(self, stack_id: int) -> dict | None:
        """Fetch a stack with its member photos."""
        row = self.conn.execute(
            "SELECT * FROM photo_stacks WHERE id = ?", (stack_id,)
        ).fetchone()
        if not row:
            return None
        members = self.conn.execute("""
            SELECT p.*, sm.is_top
            FROM stack_members sm
            JOIN photos p ON p.id = sm.photo_id
            WHERE sm.stack_id = ?
            ORDER BY sm.is_top DESC, p.aesthetic_score DESC
        """, (stack_id,)).fetchall()
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "members": [dict(m) for m in members],
        }

    def get_photo_stack(self, photo_id: int) -> dict | None:
        """Get stack info for a photo, if it belongs to one.

        Returns {stack_id, is_top, member_count} or None.
        """
        row = self.conn.execute(
            "SELECT stack_id, is_top FROM stack_members WHERE photo_id = ?",
            (photo_id,),
        ).fetchone()
        if not row:
            return None
        count = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM stack_members WHERE stack_id = ?",
            (row["stack_id"],),
        ).fetchone()["cnt"]
        return {
            "stack_id": row["stack_id"],
            "is_top": bool(row["is_top"]),
            "member_count": count,
        }

    def set_stack_top(self, stack_id: int, photo_id: int):
        """Promote a photo to be the top of its stack."""
        self.conn.execute(
            "UPDATE stack_members SET is_top = 0 WHERE stack_id = ?",
            (stack_id,),
        )
        self.conn.execute(
            "UPDATE stack_members SET is_top = 1 WHERE stack_id = ? AND photo_id = ?",
            (stack_id, photo_id),
        )
        self.conn.commit()

    def add_to_stack(self, stack_id: int, photo_id: int):
        """Add a photo to an existing stack.

        If the photo is already in another stack, it is removed from that
        stack first (and that stack is dissolved if it drops to 1 member).
        """
        # Remove from any existing stack first
        self.unstack_photo(photo_id)
        # Verify the target stack exists
        row = self.conn.execute(
            "SELECT id FROM photo_stacks WHERE id = ?", (stack_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Stack {stack_id} does not exist")
        self.conn.execute(
            "INSERT OR IGNORE INTO stack_members (stack_id, photo_id, is_top) VALUES (?, ?, 0)",
            (stack_id, photo_id),
        )
        self.conn.commit()

    def delete_stack(self, stack_id: int):
        """Dissolve a stack — members become unstacked individual photos."""
        self.conn.execute("DELETE FROM photo_stacks WHERE id = ?", (stack_id,))
        self.conn.commit()

    def unstack_photo(self, photo_id: int):
        """Remove a single photo from its stack.

        If the stack drops to 1 member, the stack is dissolved.
        If the removed photo was the top, the highest-scored remaining
        member becomes the new top.
        """
        row = self.conn.execute(
            "SELECT stack_id, is_top FROM stack_members WHERE photo_id = ?",
            (photo_id,),
        ).fetchone()
        if not row:
            return
        stack_id = row["stack_id"]
        was_top = row["is_top"]

        self.conn.execute(
            "DELETE FROM stack_members WHERE photo_id = ?", (photo_id,)
        )

        remaining = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM stack_members WHERE stack_id = ?",
            (stack_id,),
        ).fetchone()["cnt"]

        if remaining <= 1:
            # Dissolve — a stack of 1 is not a stack
            self.conn.execute("DELETE FROM photo_stacks WHERE id = ?", (stack_id,))
        elif was_top:
            # Promote the highest-scored remaining member
            best = self.conn.execute("""
                SELECT sm.photo_id FROM stack_members sm
                JOIN photos p ON p.id = sm.photo_id
                WHERE sm.stack_id = ?
                ORDER BY p.aesthetic_score DESC
                LIMIT 1
            """, (stack_id,)).fetchone()
            if best:
                self.conn.execute(
                    "UPDATE stack_members SET is_top = 1 WHERE stack_id = ? AND photo_id = ?",
                    (stack_id, best["photo_id"]),
                )
        self.conn.commit()

    def get_all_stacks(self) -> list[dict]:
        """List all stacks with member count and top photo info."""
        rows = self.conn.execute("""
            SELECT ps.id AS stack_id,
                   COUNT(sm.photo_id) AS member_count,
                   MAX(CASE WHEN sm.is_top = 1 THEN sm.photo_id END) AS top_photo_id
            FROM photo_stacks ps
            JOIN stack_members sm ON sm.stack_id = ps.id
            GROUP BY ps.id
            ORDER BY ps.id
        """).fetchall()
        return [dict(r) for r in rows]

    def clear_stacks(self):
        """Remove all stacks (useful before re-running detection)."""
        self.conn.execute("DELETE FROM stack_members")
        self.conn.execute("DELETE FROM photo_stacks")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Worker claims (distributed indexing)
    # ------------------------------------------------------------------

    def expire_worker_claims(self) -> int:
        """Delete expired claims, returning them to the queue. Returns count expired."""
        cur = self.conn.execute(
            "DELETE FROM worker_claims WHERE expires_at < datetime('now')"
        )
        self.conn.commit()
        return cur.rowcount

    def claim_photos(self, worker_id: str, pass_type: str, photo_ids: list[int],
                     ttl_minutes: int = 30) -> str:
        """Claim a batch of photos for processing. Returns batch_id."""
        import uuid
        batch_id = str(uuid.uuid4())
        self.conn.execute(
            """INSERT INTO worker_claims (batch_id, worker_id, pass_type, photo_ids, expires_at)
               VALUES (?, ?, ?, ?, datetime('now', ?))""",
            (batch_id, worker_id, pass_type, json.dumps(photo_ids),
             f"+{ttl_minutes} minutes"),
        )
        self.conn.commit()
        return batch_id

    def release_claim(self, batch_id: str):
        """Release a claim (after results submitted or on failure)."""
        self.conn.execute("DELETE FROM worker_claims WHERE batch_id = ?", (batch_id,))
        self.conn.commit()

    def mark_processed(self, photo_ids: list[int], pass_type: str):
        """Record that these photos have been processed for the given pass type.

        Used for passes like 'faces' where a no-result outcome produces no DB rows,
        so we need an explicit record to avoid re-processing.
        """
        for pid in photo_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO worker_processed (photo_id, pass_type) VALUES (?, ?)",
                (pid, pass_type),
            )
        self.conn.commit()

    def get_claimed_photo_ids(self, pass_type: str) -> set[int]:
        """Return the set of photo IDs currently claimed for a pass type."""
        self.expire_worker_claims()
        rows = self.conn.execute(
            "SELECT photo_ids FROM worker_claims WHERE pass_type = ? AND expires_at > datetime('now')",
            (pass_type,),
        ).fetchall()
        result = set()
        for row in rows:
            result.update(json.loads(row["photo_ids"]))
        return result

    def get_unprocessed_photos(self, pass_type: str, photo_ids: list[int] | None = None,
                               limit: int = 16) -> list[dict]:
        """Find photos missing data for a given pass type, excluding claimed ones.

        pass_type: 'clip', 'faces', 'quality', 'describe', 'tags'
        photo_ids: optional scope (e.g. collection photos)
        Returns list of {id, filepath} dicts.
        """
        claimed = self.get_claimed_photo_ids(pass_type)

        # Build the "missing" condition per pass type.
        # For faces, we check worker_processed because photos with no faces
        # produce no rows in the faces table (can't distinguish unprocessed
        # from processed-with-no-results). Other passes store NULL → value
        # in photos columns, so we can check IS NULL directly.
        if pass_type == "clip":
            # Photos with no CLIP embedding
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                rows = self.conn.execute(
                    f"""SELECT p.id, p.filepath FROM photos p
                        WHERE p.id IN ({placeholders})
                        AND p.id NOT IN (SELECT photo_id FROM clip_embeddings)
                        LIMIT ?""",
                    list(photo_ids) + [limit + len(claimed)],
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """SELECT p.id, p.filepath FROM photos p
                       WHERE p.id NOT IN (SELECT photo_id FROM clip_embeddings)
                       LIMIT ?""",
                    (limit + len(claimed),),
                ).fetchall()
        elif pass_type == "faces":
            # Use worker_processed to track which photos have been attempted
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                rows = self.conn.execute(
                    f"""SELECT p.id, p.filepath FROM photos p
                        WHERE p.id IN ({placeholders})
                        AND NOT EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id)
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = p.id AND wp.pass_type = 'faces')
                        LIMIT ?""",
                    list(photo_ids) + [limit + len(claimed)],
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """SELECT p.id, p.filepath FROM photos p
                       WHERE NOT EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id)
                       AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                       WHERE wp.photo_id = p.id AND wp.pass_type = 'faces')
                       LIMIT ?""",
                    (limit + len(claimed),),
                ).fetchall()
        elif pass_type == "quality":
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                rows = self.conn.execute(
                    f"SELECT id, filepath FROM photos WHERE id IN ({placeholders}) AND aesthetic_score IS NULL LIMIT ?",
                    list(photo_ids) + [limit + len(claimed)],
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT id, filepath FROM photos WHERE aesthetic_score IS NULL LIMIT ?",
                    (limit + len(claimed),),
                ).fetchall()
        elif pass_type in ("describe", "tags"):
            # Like faces, use worker_processed to track attempts — photos that
            # produce no description/tags would otherwise stay NULL forever.
            col = {"describe": "description", "tags": "tags"}[pass_type]
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                rows = self.conn.execute(
                    f"""SELECT id, filepath FROM photos
                        WHERE id IN ({placeholders})
                        AND {col} IS NULL
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = ?)
                        LIMIT ?""",
                    list(photo_ids) + [pass_type, limit + len(claimed)],
                ).fetchall()
            else:
                rows = self.conn.execute(
                    f"""SELECT id, filepath FROM photos
                        WHERE {col} IS NULL
                        AND NOT EXISTS (SELECT 1 FROM worker_processed wp
                                        WHERE wp.photo_id = photos.id AND wp.pass_type = ?)
                        LIMIT ?""",
                    (pass_type, limit + len(claimed)),
                ).fetchall()
        elif pass_type == "verify":
            # Photos that have a description but haven't been verified yet
            if photo_ids:
                placeholders = ",".join("?" * len(photo_ids))
                rows = self.conn.execute(
                    f"""SELECT id, filepath FROM photos
                        WHERE id IN ({placeholders})
                        AND description IS NOT NULL
                        AND verified_at IS NULL
                        LIMIT ?""",
                    list(photo_ids) + [limit + len(claimed)],
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """SELECT id, filepath FROM photos
                       WHERE description IS NOT NULL
                       AND verified_at IS NULL
                       LIMIT ?""",
                    (limit + len(claimed),),
                ).fetchall()
        else:
            raise ValueError(f"Unknown pass type: {pass_type}")

        # Filter out claimed photos and apply limit
        result = [dict(r) for r in rows if r["id"] not in claimed]
        return result[:limit]

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

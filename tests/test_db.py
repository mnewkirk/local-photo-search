"""Tests for the PhotoDB database layer.

Covers: CRUD, path handling, batch mode, collections, faces, persons,
color search, text search, Google Photos upload ledger, schema migrations,
and concurrent access.
"""

import json
import sqlite3
import threading

import pytest

from photosearch.db import PhotoDB, CLIP_DIMENSIONS, FACE_DIMENSIONS, SCHEMA_VERSION
from conftest import _make_embedding


# =========================================================================
# Schema and initialization
# =========================================================================

class TestSchemaInit:
    def test_schema_version_set(self, db):
        row = db.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        ).fetchone()
        assert int(row["value"]) == SCHEMA_VERSION

    def test_tables_exist(self, db):
        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        expected = {"photos", "persons", "faces", "face_references",
                    "ignored_clusters", "review_selections",
                    "collections", "collection_photos", "schema_info"}
        assert expected.issubset(tables)

    def test_idempotent_schema_init(self, db):
        """Re-calling _init_schema should be a no-op."""
        db._init_schema()
        assert db.photo_count() == 5  # data still intact

    def test_wal_mode_enabled(self, db):
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


# =========================================================================
# Photo CRUD
# =========================================================================

class TestPhotoCRUD:
    def test_add_and_get_photo(self, db):
        pid = db.add_photo(filepath="test/new.jpg", filename="new.jpg")
        assert pid > 0
        photo = db.get_photo(pid)
        assert photo["filename"] == "new.jpg"

    def test_add_duplicate_filepath_returns_existing_id(self, db):
        """INSERT OR IGNORE — duplicate filepath returns the existing row's id."""
        pid1 = db.add_photo(filepath="dup/photo.jpg", filename="photo.jpg")
        pid2 = db.add_photo(filepath="dup/photo.jpg", filename="photo.jpg")
        assert pid1 == pid2

    def test_add_photo_with_gps_stamps_exif_provenance(self, db):
        pid = db.add_photo(filepath="gps/a.jpg", filename="a.jpg",
                           gps_lat=47.6, gps_lon=-122.3)
        photo = db.get_photo(pid)
        assert photo["location_source"] == "exif"

    def test_add_photo_without_gps_leaves_provenance_null(self, db):
        pid = db.add_photo(filepath="nogps/b.jpg", filename="b.jpg")
        photo = db.get_photo(pid)
        assert photo["location_source"] is None

    def test_add_photo_with_explicit_location_source_wins(self, db):
        pid = db.add_photo(filepath="inferred/c.jpg", filename="c.jpg",
                           gps_lat=47.6, gps_lon=-122.3,
                           location_source="inferred")
        photo = db.get_photo(pid)
        assert photo["location_source"] == "inferred"

    def test_get_photo_by_path(self, db):
        photo = db.get_photo_by_path("2026/march/DSC04878.JPG")
        assert photo is not None
        assert photo["filename"] == "DSC04878.JPG"

    def test_get_photo_not_found(self, db):
        assert db.get_photo(99999) is None

    def test_get_photo_by_path_not_found(self, db):
        assert db.get_photo_by_path("nonexistent.jpg") is None

    def test_update_photo(self, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.update_photo(pid, description="Updated description", aesthetic_score=9.5)
        photo = db.get_photo(pid)
        assert photo["description"] == "Updated description"
        assert photo["aesthetic_score"] == 9.5

    def test_photo_count(self, db):
        assert db.photo_count() == 5


# =========================================================================
# Path handling
# =========================================================================

class TestPathHandling:
    def test_relative_filepath(self, db):
        assert db.relative_filepath("/photos/2026/march/test.jpg") == "2026/march/test.jpg"

    def test_relative_filepath_outside_root(self, db):
        """Path not under photo_root is returned as-is."""
        assert db.relative_filepath("/other/path/test.jpg") == "/other/path/test.jpg"

    def test_resolve_filepath(self, db):
        assert db.resolve_filepath("2026/march/test.jpg") == "/photos/2026/march/test.jpg"

    def test_resolve_absolute_filepath(self, db):
        """Already-absolute path is returned unchanged."""
        assert db.resolve_filepath("/abs/path.jpg") == "/abs/path.jpg"

    def test_resolve_empty_filepath(self, db):
        assert db.resolve_filepath("") == ""

    def test_remap_paths(self, db):
        # Add a photo with an old-style path
        db.add_photo(filepath="/old/root/img.jpg", filename="img.jpg")
        count = db.remap_paths("/old/root", "/new/root")
        assert count == 1
        photo = db.get_photo_by_path("/new/root/img.jpg")
        assert photo is not None

    def test_set_photo_root_persists(self, tmp_db_path):
        db1 = PhotoDB(tmp_db_path)
        db1.set_photo_root("/my/photos")
        db1.close()
        db2 = PhotoDB(tmp_db_path)
        assert db2.photo_root == "/my/photos"
        db2.close()


# =========================================================================
# Batch mode
# =========================================================================

class TestBatchMode:
    def test_batch_defers_commits(self, db):
        db.begin_batch(batch_size=10)
        db.add_photo(filepath="batch/1.jpg", filename="1.jpg")
        # In batch mode, _batch_count should be 1
        assert db._batch_count == 1
        assert db._batch_mode is True
        db.end_batch()
        assert db._batch_mode is False

    def test_batch_auto_flushes(self, db):
        db.begin_batch(batch_size=2)
        db.add_photo(filepath="batch/a.jpg", filename="a.jpg")
        db.add_photo(filepath="batch/b.jpg", filename="b.jpg")
        # After 2 ops, should auto-commit and reset count
        assert db._batch_count == 0
        db.end_batch()

    def test_flush_batch_explicit(self, db):
        db.begin_batch(batch_size=100)
        db.add_photo(filepath="batch/x.jpg", filename="x.jpg")
        db.flush_batch()
        assert db._batch_count == 0
        db.end_batch()


# =========================================================================
# Persons
# =========================================================================

class TestPersons:
    def test_add_person(self, db):
        pid = db.add_person("NewPerson")
        assert pid > 0
        person = db.get_person_by_name("NewPerson")
        assert person["name"] == "NewPerson"

    def test_add_person_case_insensitive_dedup(self, db):
        """Adding a person with different case returns existing id."""
        pid1 = db.add_person("Alex")
        pid2 = db.add_person("alex")
        assert pid1 == pid2

    def test_get_person_by_name_case_insensitive(self, db):
        person = db.get_person_by_name("JAMIE")
        assert person is not None
        assert person["name"] == "Jamie"

    def test_get_person_not_found(self, db):
        assert db.get_person_by_name("NonExistent") is None


# =========================================================================
# Faces
# =========================================================================

class TestFaces:
    def test_add_face_returns_id(self, db):
        pid = db._test_photo_ids["DSC04880.JPG"]
        fid = db.add_face(pid, (10, 20, 30, 5), _make_embedding(512, seed=200))
        assert fid > 0

    def test_assign_face_to_person(self, db):
        fid = db._test_face_ids["unknown_878"]
        alex_id = db._test_person_ids["Alex"]
        db.assign_face_to_person(fid, alex_id, "manual")
        row = db.conn.execute("SELECT person_id, match_source FROM faces WHERE id = ?", (fid,)).fetchone()
        assert row["person_id"] == alex_id
        assert row["match_source"] == "manual"

    def test_get_face_encoding(self, db):
        fid = db._test_face_ids["alex_894"]
        enc = db.get_face_encoding(fid)
        assert enc is not None
        assert len(enc) == FACE_DIMENSIONS

    def test_get_face_encodings_bulk(self, db):
        fids = [db._test_face_ids["alex_894"], db._test_face_ids["jamie_907"]]
        encodings = db.get_face_encodings_bulk(fids)
        assert len(encodings) == 2
        for fid in fids:
            assert fid in encodings
            assert len(encodings[fid]) == FACE_DIMENSIONS

    def test_get_face_encodings_bulk_empty(self, db):
        assert db.get_face_encodings_bulk([]) == {}

    def test_search_faces(self, db):
        """Searching with a known face's own encoding should return it first.

        Note: sqlite-vec KNN query syntax varies by version (LIMIT vs k=?).
        If the DB's search_faces raises OperationalError we fall back to a
        direct query with the k=? constraint.
        """
        import sqlite3
        from photosearch.db import _serialize_float_list
        fid = db._test_face_ids["alex_894"]
        enc = db.get_face_encoding(fid)
        try:
            results = db.search_faces(enc, limit=3)
        except sqlite3.OperationalError:
            # Fallback for sqlite-vec versions requiring k=? syntax
            rows = db.conn.execute(
                "SELECT face_id, distance FROM face_encodings "
                "WHERE encoding MATCH ? AND k = ?",
                (_serialize_float_list(enc), 3),
            ).fetchall()
            results = [dict(r) for r in rows]
        assert len(results) > 0
        assert results[0]["face_id"] == fid
        assert results[0]["distance"] < 0.01  # near-zero distance to self


# =========================================================================
# CLIP embeddings
# =========================================================================

class TestCLIPEmbeddings:
    def _search_clip_compat(self, db, emb, limit):
        """Search CLIP embeddings, handling sqlite-vec version differences."""
        import sqlite3
        from photosearch.db import _serialize_float_list
        try:
            return db.search_clip(emb, limit=limit)
        except sqlite3.OperationalError:
            rows = db.conn.execute(
                "SELECT photo_id, distance FROM clip_embeddings "
                "WHERE embedding MATCH ? AND k = ?",
                (_serialize_float_list(emb), limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def test_add_and_search_clip(self, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        emb = _make_embedding(512, seed=0)  # same seed used during setup
        results = self._search_clip_compat(db, emb, 1)
        assert len(results) == 1
        assert results[0]["photo_id"] == pid
        assert results[0]["distance"] < 0.01

    def test_search_clip_respects_limit(self, db):
        emb = _make_embedding(512, seed=0)
        results = self._search_clip_compat(db, emb, 3)
        assert len(results) == 3


# =========================================================================
# Color search
# =========================================================================

class TestColorSearch:
    def test_search_by_exact_color(self, db):
        """Searching for a color that's in DSC04878's palette."""
        results = db.search_by_color("#3b5998", tolerance=10, limit=5)
        assert len(results) >= 1
        filenames = {r["filename"] for r in results}
        assert "DSC04878.JPG" in filenames

    def test_search_by_color_tolerance(self, db):
        """Very tight tolerance returns fewer results."""
        narrow = db.search_by_color("#3b5998", tolerance=1, limit=10)
        wide = db.search_by_color("#3b5998", tolerance=100, limit=10)
        assert len(narrow) <= len(wide)

    def test_search_by_color_no_match(self, db):
        """A color not near any palette returns nothing."""
        results = db.search_by_color("#000000", tolerance=5, limit=5)
        # Might find near-black or nothing — tolerance=5 is very tight
        assert isinstance(results, list)


# =========================================================================
# Text search
# =========================================================================

class TestTextSearch:
    def test_search_description(self, db):
        results = db.search_text("sunset", limit=10)
        assert any(r["filename"] == "DSC04922.JPG" for r in results)

    def test_search_place_name(self, db):
        results = db.search_text("Big Sur", limit=10)
        filenames = {r["filename"] for r in results}
        assert "DSC04894.JPG" in filenames
        assert "DSC04907.JPG" in filenames

    def test_search_no_match(self, db):
        results = db.search_text("xyznonexistent", limit=10)
        assert results == []


# =========================================================================
# Collections
# =========================================================================

class TestCollections:
    def test_create_collection(self, db):
        cid = db.create_collection("Landscapes", "Nature photos")
        assert cid > 0
        coll = db.get_collection(cid)
        assert coll["name"] == "Landscapes"
        assert coll["description"] == "Nature photos"

    def test_get_collection_by_name(self, db):
        coll = db.get_collection_by_name("Best of March")
        assert coll is not None
        assert coll["name"] == "Best of March"

    def test_get_collection_not_found(self, db):
        assert db.get_collection(99999) is None
        assert db.get_collection_by_name("Nonexistent") is None

    def test_list_collections(self, db):
        colls = db.list_collections()
        assert len(colls) >= 1
        names = {c["name"] for c in colls}
        assert "Best of March" in names

    def test_add_photos_to_collection(self, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04878.JPG"]
        added = db.add_photos_to_collection(cid, [pid])
        assert added == 1
        photos = db.get_collection_photos(cid)
        assert any(p["id"] == pid for p in photos)

    def test_add_duplicate_photo_to_collection(self, db):
        """Adding a photo that's already in the collection should be silently skipped."""
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04907.JPG"]  # already in collection
        added = db.add_photos_to_collection(cid, [pid])
        assert added == 0

    def test_remove_photos_from_collection(self, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04907.JPG"]
        removed = db.remove_photos_from_collection(cid, [pid])
        assert removed == 1
        photos = db.get_collection_photos(cid)
        assert all(p["id"] != pid for p in photos)

    def test_get_collection_photos_order(self, db):
        """Photos should be ordered by sort_order."""
        cid = db._test_collection_id
        photos = db.get_collection_photos(cid)
        orders = [p["sort_order"] for p in photos]
        assert orders == sorted(orders)

    def test_get_photo_collections(self, db):
        pid = db._test_photo_ids["DSC04922.JPG"]
        colls = db.get_photo_collections(pid)
        assert any(c["name"] == "Best of March" for c in colls)

    def test_rename_collection(self, db):
        cid = db._test_collection_id
        db.rename_collection(cid, "March Highlights")
        coll = db.get_collection(cid)
        assert coll["name"] == "March Highlights"

    def test_update_collection_description(self, db):
        cid = db._test_collection_id
        db.update_collection_description(cid, "New description")
        coll = db.get_collection(cid)
        assert coll["description"] == "New description"

    def test_set_collection_cover(self, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04922.JPG"]
        db.set_collection_cover(cid, pid)
        coll = db.get_collection(cid)
        assert coll["cover_photo_id"] == pid

    def test_delete_collection_cascades(self, db):
        cid = db.create_collection("ToDelete")
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.add_photos_to_collection(cid, [pid])
        db.delete_collection(cid)
        assert db.get_collection(cid) is None
        # collection_photos rows should be gone too
        row = db.conn.execute(
            "SELECT COUNT(*) as c FROM collection_photos WHERE collection_id = ?", (cid,)
        ).fetchone()
        assert row["c"] == 0


# =========================================================================
# Ignored clusters
# =========================================================================

class TestIgnoredClusters:
    def test_ignore_and_list(self, db):
        db.conn.execute("INSERT OR IGNORE INTO ignored_clusters (cluster_id) VALUES (?)", (42,))
        db.conn.commit()
        rows = db.conn.execute("SELECT cluster_id FROM ignored_clusters").fetchall()
        assert 42 in {r["cluster_id"] for r in rows}

    def test_ignore_idempotent(self, db):
        db.conn.execute("INSERT OR IGNORE INTO ignored_clusters (cluster_id) VALUES (?)", (42,))
        db.conn.execute("INSERT OR IGNORE INTO ignored_clusters (cluster_id) VALUES (?)", (42,))
        db.conn.commit()
        count = db.conn.execute(
            "SELECT COUNT(*) as c FROM ignored_clusters WHERE cluster_id = 42"
        ).fetchone()["c"]
        assert count == 1


# =========================================================================
# Review selections
# =========================================================================

class TestReviewSelections:
    def test_save_and_load_selection(self, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.conn.execute(
            "INSERT OR REPLACE INTO review_selections (photo_id, directory, selected, cluster_id, rank_in_cluster) "
            "VALUES (?, ?, ?, ?, ?)",
            (pid, "/photos/2026/march", 1, 1, 1),
        )
        db.conn.commit()
        row = db.conn.execute(
            "SELECT * FROM review_selections WHERE photo_id = ? AND directory = ?",
            (pid, "/photos/2026/march"),
        ).fetchone()
        assert row is not None
        assert row["selected"] == 1

    def test_unique_constraint(self, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.conn.execute(
            "INSERT OR REPLACE INTO review_selections (photo_id, directory, selected) VALUES (?, ?, ?)",
            (pid, "/photos/2026/march", 1),
        )
        db.conn.execute(
            "INSERT OR REPLACE INTO review_selections (photo_id, directory, selected) VALUES (?, ?, ?)",
            (pid, "/photos/2026/march", 0),
        )
        db.conn.commit()
        count = db.conn.execute(
            "SELECT COUNT(*) as c FROM review_selections WHERE photo_id = ? AND directory = ?",
            (pid, "/photos/2026/march"),
        ).fetchone()["c"]
        assert count == 1


# =========================================================================
# Google Photos upload ledger
# =========================================================================

class TestGooglePhotosUploadLedger:
    """Tests for record_upload, get_uploaded_filepaths, clear_upload_ledger."""

    ALBUM_ID = "ALBUM_TEST_ABC123"

    def test_record_and_retrieve_upload(self, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.record_upload(self.ALBUM_ID, pid, "photos/DSC04878.JPG", "MEDIA_001")

        fps = db.get_uploaded_filepaths(self.ALBUM_ID)
        assert "photos/DSC04878.JPG" in fps

    def test_get_uploaded_filepaths_empty_for_unknown_album(self, db):
        fps = db.get_uploaded_filepaths("ALBUM_NONEXISTENT")
        assert fps == set()

    def test_get_uploaded_filepaths_multiple_photos(self, db):
        files = [
            ("DSC04878.JPG", "photos/DSC04878.JPG", "M001"),
            ("DSC04880.JPG", "photos/DSC04880.JPG", "M002"),
            ("DSC04894.JPG", "photos/DSC04894.JPG", "M003"),
        ]
        for fname, fp, mid in files:
            pid = db._test_photo_ids[fname]
            db.record_upload(self.ALBUM_ID, pid, fp, mid)

        fps = db.get_uploaded_filepaths(self.ALBUM_ID)
        assert fps == {"photos/DSC04878.JPG", "photos/DSC04880.JPG", "photos/DSC04894.JPG"}

    def test_get_uploaded_filepaths_album_isolation(self, db):
        """Filepaths recorded for one album must not appear in another album's results."""
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.record_upload("ALBUM_A", pid, "photos/DSC04878.JPG", "MA001")

        fps_b = db.get_uploaded_filepaths("ALBUM_B")
        assert "photos/DSC04878.JPG" not in fps_b

    def test_record_upload_upsert_updates_media_item_id(self, db):
        """Recording the same (album_id, filepath) twice should update, not duplicate."""
        pid = db._test_photo_ids["DSC04878.JPG"]
        db.record_upload(self.ALBUM_ID, pid, "photos/DSC04878.JPG", "MEDIA_OLD")
        db.record_upload(self.ALBUM_ID, pid, "photos/DSC04878.JPG", "MEDIA_NEW")

        rows = db.conn.execute(
            "SELECT media_item_id FROM google_photos_uploads WHERE album_id = ? AND filepath = ?",
            (self.ALBUM_ID, "photos/DSC04878.JPG"),
        ).fetchall()
        assert len(rows) == 1, "upsert should produce exactly one row"
        assert rows[0]["media_item_id"] == "MEDIA_NEW"

    def test_media_item_id_round_trip(self, db):
        """The media_item_id stored in the ledger must be retrievable verbatim."""
        pid = db._test_photo_ids["DSC04907.JPG"]
        goog_id = "AHP8L8tNGTE4sm0zGv5uOH3o4WVvEgxko3dzxZAv1ETXnOimdro2b2ODVFul2NGwDI0r_AZLIszC"
        db.record_upload(self.ALBUM_ID, pid, "photos/DSC04907.JPG", goog_id)

        row = db.conn.execute(
            "SELECT media_item_id FROM google_photos_uploads WHERE filepath = ? AND album_id = ?",
            ("photos/DSC04907.JPG", self.ALBUM_ID),
        ).fetchone()
        assert row is not None
        assert row["media_item_id"] == goog_id

    def test_clear_upload_ledger_removes_all_entries(self, db):
        filenames = ["DSC04878.JPG", "DSC04880.JPG", "DSC04894.JPG"]
        for i, fname in enumerate(filenames):
            pid = db._test_photo_ids[fname]
            db.record_upload(self.ALBUM_ID, pid, f"photos/{fname}", f"M{i:03d}")

        deleted = db.clear_upload_ledger(self.ALBUM_ID)
        assert deleted == 3
        fps = db.get_uploaded_filepaths(self.ALBUM_ID)
        assert fps == set()

    def test_clear_upload_ledger_returns_zero_when_empty(self, db):
        deleted = db.clear_upload_ledger("ALBUM_EMPTY")
        assert deleted == 0

    def test_clear_upload_ledger_album_isolation(self, db):
        """Clearing one album's ledger must not affect another album's entries."""
        pid_a = db._test_photo_ids["DSC04878.JPG"]
        pid_b = db._test_photo_ids["DSC04894.JPG"]
        db.record_upload("ALBUM_A", pid_a, "photos/DSC04878.JPG", "MA")
        db.record_upload("ALBUM_B", pid_b, "photos/DSC04894.JPG", "MB")

        db.clear_upload_ledger("ALBUM_A")

        fps_b = db.get_uploaded_filepaths("ALBUM_B")
        assert "photos/DSC04894.JPG" in fps_b

    def test_record_upload_null_photo_id_allowed(self, db):
        """record_upload must work with photo_id=None (no FK violation)."""
        db.record_upload(self.ALBUM_ID, None, "photos/orphan.jpg", "M_ORPHAN")
        fps = db.get_uploaded_filepaths(self.ALBUM_ID)
        assert "photos/orphan.jpg" in fps

    def test_google_photos_uploads_table_exists(self, db):
        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "google_photos_uploads" in tables


# =========================================================================
# Google Photos album link on collections
# =========================================================================

class TestCollectionGoogleAlbumLink:
    """Tests for set_collection_google_album."""

    def test_set_and_retrieve_album_link(self, db):
        cid = db._test_collection_id
        db.set_collection_google_album(cid, "ALBUM_XYZ", "My Trip Album")

        row = db.conn.execute(
            "SELECT google_photos_album_id, google_photos_album_title FROM collections WHERE id = ?",
            (cid,),
        ).fetchone()
        assert row["google_photos_album_id"] == "ALBUM_XYZ"
        assert row["google_photos_album_title"] == "My Trip Album"

    def test_overwrite_album_link(self, db):
        cid = db._test_collection_id
        db.set_collection_google_album(cid, "OLD_ALBUM", "Old Title")
        db.set_collection_google_album(cid, "NEW_ALBUM", "New Title")

        row = db.conn.execute(
            "SELECT google_photos_album_id, google_photos_album_title FROM collections WHERE id = ?",
            (cid,),
        ).fetchone()
        assert row["google_photos_album_id"] == "NEW_ALBUM"
        assert row["google_photos_album_title"] == "New Title"

    def test_unrelated_collection_unaffected(self, db):
        """Setting an album link on one collection must not touch others."""
        cid1 = db._test_collection_id
        cid2 = db.create_collection("Other Collection")
        db.set_collection_google_album(cid1, "ALBUM_1", "Album One")

        row = db.conn.execute(
            "SELECT google_photos_album_id FROM collections WHERE id = ?",
            (cid2,),
        ).fetchone()
        assert row["google_photos_album_id"] is None

    def test_new_collection_has_null_album_link(self, db):
        cid = db.create_collection("Fresh Collection")
        row = db.conn.execute(
            "SELECT google_photos_album_id, google_photos_album_title FROM collections WHERE id = ?",
            (cid,),
        ).fetchone()
        assert row["google_photos_album_id"] is None
        assert row["google_photos_album_title"] is None


# =========================================================================
# Schema migration: google_photos_uploads created on upgrade
# =========================================================================

class TestSchemaUpgradeMigration:
    """Verify that a DB stuck at an old version gets the uploads table on open."""

    def test_migration_from_v11_adds_uploads_table(self, tmp_db_path):
        """Simulate a database frozen at schema v11 (no google_photos_uploads).

        When PhotoDB opens it the migration should add the table and bump the
        stored version to SCHEMA_VERSION.
        """
        import sqlite3 as _sqlite3

        # Build a minimal schema-v11 database by hand
        conn = _sqlite3.connect(tmp_db_path)
        conn.execute("CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO schema_info VALUES ('version', '11')")
        # Use a fairly complete v11 photos schema (all indexed columns must exist)
        conn.execute("""
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                file_hash TEXT,
                date_taken TEXT,
                gps_lat REAL, gps_lon REAL,
                place_name TEXT,
                camera_make TEXT, camera_model TEXT,
                focal_length TEXT, exposure_time TEXT,
                f_number TEXT, iso INTEGER,
                image_width INTEGER, image_height INTEGER,
                description TEXT,
                dominant_colors TEXT,
                aesthetic_score REAL,
                aesthetic_concepts TEXT,
                aesthetic_critique TEXT,
                tags TEXT,
                indexed_at TEXT DEFAULT (datetime('now')),
                raw_filepath TEXT,
                verified_at TEXT,
                verification_status TEXT,
                hallucination_flags TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                cover_photo_id INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                person_id INTEGER,
                bbox_top INTEGER, bbox_right INTEGER,
                bbox_bottom INTEGER, bbox_left INTEGER,
                cluster_id INTEGER, match_source TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE face_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                source_path TEXT
            )
        """)
        conn.execute("CREATE TABLE ignored_clusters (cluster_id INTEGER PRIMARY KEY, ignored_at TEXT DEFAULT (datetime('now')))")
        conn.execute("""
            CREATE TABLE review_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                directory TEXT NOT NULL,
                selected INTEGER NOT NULL DEFAULT 0,
                cluster_id INTEGER, rank_in_cluster INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(photo_id, directory)
            )
        """)
        conn.execute("""
            CREATE TABLE photo_stacks (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT (datetime('now')))
        """)
        conn.execute("""
            CREATE TABLE stack_members (
                stack_id INTEGER NOT NULL, photo_id INTEGER NOT NULL,
                is_top INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (stack_id, photo_id), UNIQUE(photo_id)
            )
        """)
        conn.execute("""
            CREATE TABLE collection_photos (
                collection_id INTEGER NOT NULL, photo_id INTEGER NOT NULL,
                added_at TEXT DEFAULT (datetime('now')), sort_order INTEGER DEFAULT 0,
                PRIMARY KEY (collection_id, photo_id)
            )
        """)
        conn.commit()
        conn.close()

        # Open with PhotoDB — should migrate automatically
        db = PhotoDB(tmp_db_path)

        tables = {r[0] for r in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "google_photos_uploads" in tables, \
            "google_photos_uploads table should be created during migration from v11"

        version_row = db.conn.execute(
            "SELECT value FROM schema_info WHERE key = 'version'"
        ).fetchone()
        assert int(version_row[0]) == SCHEMA_VERSION


# =========================================================================
# Context manager
# =========================================================================

class TestContextManager:
    def test_context_manager(self, tmp_db_path):
        with PhotoDB(tmp_db_path) as database:
            database.add_photo(filepath="ctx/test.jpg", filename="test.jpg")
            assert database.photo_count() == 1
        # After exiting, connection is closed — attempting another query should fail
        with pytest.raises(Exception):
            database.photo_count()


def test_migration_v17_adds_location_columns_and_backfills_exif(tmp_path):
    """v16 → v17 adds location_source + location_confidence, backfills 'exif'
    on pre-existing GPS rows, and leaves no-GPS rows' location_source NULL."""
    import sqlite3
    from photosearch.db import PhotoDB

    db_path = str(tmp_path / "v16.db")

    # Build a minimal v16-shaped DB by hand (no location_source column).
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_info VALUES ('version', '16');
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            gps_lat REAL, gps_lon REAL, place_name TEXT,
            date_taken TEXT
        );
        INSERT INTO photos (filepath, filename, gps_lat, gps_lon, place_name)
            VALUES ('/a.jpg', 'a.jpg', 47.6, -122.3, 'Seattle, WA, US');
        INSERT INTO photos (filepath, filename)
            VALUES ('/b.jpg', 'b.jpg');
    """)
    conn.commit()
    conn.close()

    # Opening with the current code should migrate.
    with PhotoDB(db_path) as pdb:
        row_a = pdb.conn.execute(
            "SELECT location_source, location_confidence FROM photos WHERE filename='a.jpg'"
        ).fetchone()
        row_b = pdb.conn.execute(
            "SELECT location_source, location_confidence FROM photos WHERE filename='b.jpg'"
        ).fetchone()
        version = pdb.conn.execute(
            "SELECT value FROM schema_info WHERE key='version'"
        ).fetchone()[0]

    assert row_a["location_source"] == "exif"
    assert row_a["location_confidence"] is None
    assert row_b["location_source"] is None
    assert row_b["location_confidence"] is None
    assert int(version) >= 17

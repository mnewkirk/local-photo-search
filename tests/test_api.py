"""Tests for the FastAPI web endpoints.

Covers: search, photo detail, faces, persons, collections, review,
stats, error handling, and edge cases.
"""

import json
from unittest.mock import patch, MagicMock

import pytest


# =========================================================================
# Stats
# =========================================================================

class TestStats:
    def test_stats_endpoint(self, client, db):
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["photos"] == 5
        assert data["faces"] == 7  # 6 assigned + 1 unknown
        assert data["persons"] == 3
        assert data["described"] == 5
        assert data["quality_scored"] == 5
        assert data["quality_stats"]["min"] == 5.4
        assert data["quality_stats"]["max"] == 9.1


# =========================================================================
# Search
# =========================================================================

class TestSearch:
    def test_search_requires_criteria(self, client):
        resp = client.get("/api/search")
        data = resp.json()
        assert data["count"] == 0
        assert "error" in data

    def test_search_by_person(self, client):
        resp = client.get("/api/search", params={"person": "Alex"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        filenames = {r["filename"] for r in data["results"]}
        assert "DSC04894.JPG" in filenames

    def test_search_by_location(self, client):
        resp = client.get("/api/search", params={"location": "Big Sur"})
        assert resp.status_code == 200
        data = resp.json()
        filenames = {r["filename"] for r in data["results"]}
        assert "DSC04894.JPG" in filenames
        assert "DSC04907.JPG" in filenames

    def test_search_by_date_range(self, client):
        resp = client.get("/api/search", params={
            "date_from": "2026-03-13",
            "date_to": "2026-03-13",
            "location": "Morro Bay",
        })
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            assert "2026-03-13" in r["date_taken"]

    def test_search_by_min_quality(self, client):
        resp = client.get("/api/search", params={"min_quality": 8.0})
        assert resp.status_code == 200
        data = resp.json()
        for r in data["results"]:
            assert r["aesthetic_score"] >= 8.0

    def test_search_result_shape(self, client):
        resp = client.get("/api/search", params={"location": "Morro Bay"})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "count" in data
        if data["count"] > 0:
            r = data["results"][0]
            assert "id" in r
            assert "filename" in r
            assert "tags" in r
            assert isinstance(r["tags"], list)
            assert "colors" in r
            assert isinstance(r["colors"], list)

    def test_search_limit(self, client):
        resp = client.get("/api/search", params={"location": "Big Sur", "limit": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] <= 1

    def test_search_by_color(self, client):
        resp = client.get("/api/search", params={"color": "#3b5998"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["results"], list)

    def test_search_sort_quality(self, client):
        resp = client.get("/api/search", params={"location": "Big Sur", "sort_quality": True})
        assert resp.status_code == 200
        data = resp.json()
        if len(data["results"]) >= 2:
            scores = [r["aesthetic_score"] for r in data["results"] if r["aesthetic_score"]]
            assert scores == sorted(scores, reverse=True)


# =========================================================================
# Photo detail
# =========================================================================

class TestPhotoDetail:
    def test_get_photo_detail(self, client, db):
        pid = db._test_photo_ids["DSC04894.JPG"]
        resp = client.get(f"/api/photos/{pid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "DSC04894.JPG"
        assert data["description"] is not None
        assert isinstance(data["faces"], list)
        assert len(data["faces"]) == 2  # Alex + Sam

    def test_photo_detail_face_data(self, client, db):
        pid = db._test_photo_ids["DSC04894.JPG"]
        resp = client.get(f"/api/photos/{pid}")
        data = resp.json()
        face_names = {f["person_name"] for f in data["faces"]}
        assert "Alex" in face_names
        assert "Sam" in face_names
        for f in data["faces"]:
            assert f["bbox"] is not None
            assert "top" in f["bbox"]

    def test_photo_detail_not_found(self, client):
        resp = client.get("/api/photos/99999")
        assert resp.status_code == 404

    def test_photo_detail_tags_and_colors(self, client, db):
        pid = db._test_photo_ids["DSC04922.JPG"]
        resp = client.get(f"/api/photos/{pid}")
        data = resp.json()
        assert isinstance(data["tags"], list)
        assert len(data["tags"]) > 0
        assert isinstance(data["colors"], list)
        assert len(data["colors"]) > 0

    def test_photo_detail_metadata(self, client, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        resp = client.get(f"/api/photos/{pid}")
        data = resp.json()
        assert data["camera_model"] == "ILCE-7M4"
        assert data["focal_length"] == "70/1"
        assert data["iso"] == 200
        assert data["image_width"] == 7008


# =========================================================================
# Thumbnails & full photos
# =========================================================================

class TestPhotoServing:
    def test_thumbnail_not_found(self, client):
        resp = client.get("/api/photos/99999/thumbnail")
        assert resp.status_code == 404

    def test_full_photo_not_found(self, client):
        resp = client.get("/api/photos/99999/full")
        assert resp.status_code == 404

    def test_full_photo_file_missing(self, client, db):
        """Photo exists in DB but file is missing on disk."""
        pid = db._test_photo_ids["DSC04878.JPG"]
        resp = client.get(f"/api/photos/{pid}/full")
        assert resp.status_code == 404


# =========================================================================
# Persons
# =========================================================================

class TestPersonsAPI:
    def test_list_persons(self, client):
        resp = client.get("/api/persons")
        assert resp.status_code == 200
        data = resp.json()
        names = {p["name"] for p in data["persons"]}
        assert "Alex" in names
        assert "Jamie" in names
        assert "Sam" in names
        for p in data["persons"]:
            assert "photo_count" in p

    def test_persons_photo_counts(self, client):
        resp = client.get("/api/persons")
        data = resp.json()
        alex = next(p for p in data["persons"] if p["name"] == "Alex")
        assert alex["photo_count"] == 3  # 894, 907, 922


# =========================================================================
# Face management
# =========================================================================

class TestFacesAPI:
    def test_face_groups(self, client):
        resp = client.get("/api/faces/groups")
        assert resp.status_code == 200
        data = resp.json()
        groups = data["groups"]
        person_groups = [g for g in groups if g["type"] == "person"]
        cluster_groups = [g for g in groups if g["type"] == "cluster"]
        assert len(person_groups) == 3  # Alex, Jamie, Sam
        assert len(cluster_groups) >= 1  # at least the unknown cluster

    def test_face_groups_sort_count(self, client):
        resp = client.get("/api/faces/groups", params={"sort": "count"})
        assert resp.status_code == 200

    def test_face_groups_sort_similarity(self, client):
        resp = client.get("/api/faces/groups", params={"sort": "similarity"})
        assert resp.status_code == 200

    def test_face_group_photos_person(self, client, db):
        alex_id = db._test_person_ids["Alex"]
        resp = client.get(f"/api/faces/group/person/{alex_id}/photos")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["photos"]) == 3  # Alex in 3 photos

    def test_face_group_photos_cluster(self, client):
        resp = client.get("/api/faces/group/cluster/99/photos")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["photos"]) >= 1

    def test_face_group_photos_invalid_type(self, client):
        resp = client.get("/api/faces/group/invalid/1/photos")
        assert resp.status_code == 400

    def test_assign_face(self, client, db):
        fid = db._test_face_ids["unknown_878"]
        resp = client.post(f"/api/faces/{fid}/assign", params={"name": "TestPerson"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["person_name"] == "TestPerson"
        assert data["person_id"] > 0

    def test_assign_face_to_existing_person(self, client, db):
        fid = db._test_face_ids["unknown_878"]
        resp = client.post(f"/api/faces/{fid}/assign", params={"name": "Alex"})
        assert resp.status_code == 200
        assert resp.json()["person_id"] == db._test_person_ids["Alex"]

    def test_assign_face_not_found(self, client):
        resp = client.post("/api/faces/99999/assign", params={"name": "Nobody"})
        assert resp.status_code == 404

    def test_clear_face(self, client, db):
        fid = db._test_face_ids["alex_894"]
        resp = client.post(f"/api/faces/{fid}/clear")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Verify it's cleared
        row = db.conn.execute("SELECT person_id FROM faces WHERE id = ?", (fid,)).fetchone()
        assert row["person_id"] is None

    def test_clear_face_not_found(self, client):
        resp = client.post("/api/faces/99999/clear")
        assert resp.status_code == 404

    def test_bulk_collect_faces(self, client, db):
        alex_id = db._test_person_ids["Alex"]
        resp = client.post("/api/faces/bulk-collect", json={
            "groups": [{"type": "person", "id": alex_id}]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["face_ids"]) == 3  # Alex has 3 faces

    def test_bulk_collect_empty(self, client):
        resp = client.post("/api/faces/bulk-collect", json={"groups": []})
        assert resp.status_code == 200
        assert resp.json()["face_ids"] == []

    def test_ignore_clusters(self, client):
        resp = client.post("/api/faces/ignore", json={"cluster_ids": [99]})
        assert resp.status_code == 200
        assert resp.json()["ignored"] == 1
        # Verify in face groups
        groups_resp = client.get("/api/faces/groups")
        cluster99 = next(
            (g for g in groups_resp.json()["groups"] if g.get("cluster_id") == 99), None
        )
        assert cluster99 is not None
        assert cluster99["ignored"] is True

    def test_unignore_clusters(self, client):
        client.post("/api/faces/ignore", json={"cluster_ids": [99]})
        resp = client.post("/api/faces/unignore", json={"cluster_ids": [99]})
        assert resp.status_code == 200
        assert resp.json()["unignored"] == 1

    def test_ignore_empty(self, client):
        resp = client.post("/api/faces/ignore", json={"cluster_ids": []})
        assert resp.json()["ignored"] == 0

    def test_export_manual_assignments(self, client, db):
        resp = client.get("/api/faces/manual-assignments")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["assignments"], list)
        # alex_922 was manual
        manual = [a for a in data["assignments"] if a["person_name"] == "Alex"]
        assert len(manual) >= 1

    def test_import_assignments(self, client, db):
        # First export, then re-import
        export_resp = client.get("/api/faces/manual-assignments")
        assignments = export_resp.json()["assignments"]
        if assignments:
            resp = client.post("/api/faces/import-assignments", json={"assignments": assignments})
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            assert data["matched"] + data["skipped"] == len(assignments)

    def test_import_assignments_empty(self, client):
        resp = client.post("/api/faces/import-assignments", json={"assignments": []})
        assert resp.json()["matched"] == 0
        assert resp.json()["skipped"] == 0

    def test_import_assignments_bad_data(self, client):
        resp = client.post("/api/faces/import-assignments", json={
            "assignments": [{"filepath": None, "person_name": None, "bbox": None}]
        })
        assert resp.json()["skipped"] == 1


# =========================================================================
# Collections
# =========================================================================

class TestCollectionsAPI:
    def test_list_collections(self, client):
        resp = client.get("/api/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["collections"]) >= 1
        coll = data["collections"][0]
        assert "name" in coll
        assert "photo_count" in coll

    def test_create_collection(self, client):
        resp = client.post("/api/collections", json={
            "name": "Favorites",
            "description": "My favorite shots",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection"]["name"] == "Favorites"

    def test_create_collection_with_photos(self, client, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        resp = client.post("/api/collections", json={
            "name": "Beach",
            "photo_ids": [pid],
        })
        assert resp.status_code == 200
        assert resp.json()["photos_added"] == 1

    def test_create_collection_no_name(self, client):
        resp = client.post("/api/collections", json={"name": ""})
        assert resp.status_code == 400

    def test_create_collection_duplicate(self, client):
        client.post("/api/collections", json={"name": "DupTest"})
        resp = client.post("/api/collections", json={"name": "DupTest"})
        assert resp.status_code == 409

    def test_get_collection(self, client, db):
        cid = db._test_collection_id
        resp = client.get(f"/api/collections/{cid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["collection"]["name"] == "Best of March"
        assert len(data["photos"]) == 2
        for p in data["photos"]:
            assert "thumbnail" in p

    def test_get_collection_not_found(self, client):
        resp = client.get("/api/collections/99999")
        assert resp.status_code == 404

    def test_update_collection(self, client, db):
        cid = db._test_collection_id
        resp = client.put(f"/api/collections/{cid}", json={
            "name": "Updated March",
            "description": "Updated",
        })
        assert resp.status_code == 200
        assert resp.json()["collection"]["name"] == "Updated March"

    def test_update_collection_duplicate_name(self, client, db):
        client.post("/api/collections", json={"name": "Other"})
        cid = db._test_collection_id
        resp = client.put(f"/api/collections/{cid}", json={"name": "Other"})
        assert resp.status_code == 409

    def test_update_collection_not_found(self, client):
        resp = client.put("/api/collections/99999", json={"name": "X"})
        assert resp.status_code == 404

    def test_delete_collection(self, client, db):
        # Create then delete
        create_resp = client.post("/api/collections", json={"name": "ToDelete"})
        cid = create_resp.json()["collection"]["id"]
        resp = client.delete(f"/api/collections/{cid}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Verify gone
        get_resp = client.get(f"/api/collections/{cid}")
        assert get_resp.status_code == 404

    def test_delete_collection_not_found(self, client):
        resp = client.delete("/api/collections/99999")
        assert resp.status_code == 404

    def test_add_photos_to_collection(self, client, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04880.JPG"]
        resp = client.post(f"/api/collections/{cid}/photos", json={"photo_ids": [pid]})
        assert resp.status_code == 200
        assert resp.json()["added"] == 1

    def test_add_photos_no_ids(self, client, db):
        cid = db._test_collection_id
        resp = client.post(f"/api/collections/{cid}/photos", json={"photo_ids": []})
        assert resp.status_code == 400

    def test_add_photos_collection_not_found(self, client):
        resp = client.post("/api/collections/99999/photos", json={"photo_ids": [1]})
        assert resp.status_code == 404

    def test_remove_photos_from_collection(self, client, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04907.JPG"]
        resp = client.post(f"/api/collections/{cid}/photos/remove", json={"photo_ids": [pid]})
        assert resp.status_code == 200
        assert resp.json()["removed"] == 1

    def test_remove_photos_no_ids(self, client, db):
        cid = db._test_collection_id
        resp = client.post(f"/api/collections/{cid}/photos/remove", json={"photo_ids": []})
        assert resp.status_code == 400

    def test_remove_photos_collection_not_found(self, client):
        resp = client.post("/api/collections/99999/photos/remove", json={"photo_ids": [1]})
        assert resp.status_code == 404

    def test_photo_collections(self, client, db):
        pid = db._test_photo_ids["DSC04922.JPG"]
        resp = client.get(f"/api/photos/{pid}/collections")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["collections"], list)

    def test_set_cover_photo(self, client, db):
        cid = db._test_collection_id
        pid = db._test_photo_ids["DSC04922.JPG"]
        resp = client.put(f"/api/collections/{cid}", json={"cover_photo_id": pid})
        assert resp.status_code == 200


# =========================================================================
# Review
# =========================================================================

class TestReviewAPI:
    def test_review_folders(self, client):
        resp = client.get("/api/review/folders")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["folders"], list)
        assert len(data["folders"]) >= 1
        folder_paths = [f["path"] for f in data["folders"]]
        assert "2026/march" in folder_paths

    def test_review_load_empty(self, client):
        resp = client.get("/api/review/load", params={"directory": "/nonexistent"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["photos"] == []

    def test_review_toggle(self, client, db):
        pid = db._test_photo_ids["DSC04878.JPG"]
        # First create a review selection so there's something to toggle
        db.conn.execute(
            "INSERT OR REPLACE INTO review_selections (photo_id, directory, selected) VALUES (?, ?, ?)",
            (pid, "/photos/2026/march", 0),
        )
        db.conn.commit()
        resp = client.post(f"/api/review/toggle/{pid}", params={"selected": True})
        assert resp.status_code == 200
        assert resp.json()["selected"] is True

    def test_review_export_empty(self, client):
        resp = client.get("/api/review/export", params={"directory": "/nonexistent"})
        assert resp.status_code == 200
        assert resp.json()["files"] == []


# =========================================================================
# Static file serving
# =========================================================================

# =========================================================================
# Google Photos upload SSE stream
# =========================================================================

class TestGoogleUploadSSE:
    """Test the SSE streaming upload endpoint end-to-end.

    All Google Photos API calls are mocked — what we're testing is that the
    async SSE wiring (asyncio queue, background thread, event-loop handoff)
    actually delivers events to the client.
    """

    def _parse_sse(self, raw: str) -> list[dict]:
        """Parse SSE text into a list of JSON event dicts."""
        events = []
        for line in raw.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def test_upload_streams_start_and_done_events(self, client, db):
        """Verify we get start → progress → done events for a basic upload."""
        import tempfile, os

        # Create a real temp file so _upload_raw_bytes' open() doesn't fail
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0fake-jpeg-bytes")
            tmp_path = f.name

        try:
            pid = db.add_photo(filepath=tmp_path, filename=os.path.basename(tmp_path))
            cid = db._test_collection_id
            db.add_photos_to_collection(cid, [pid])

            # Mock the Google API functions
            with patch("photosearch.google_photos.refresh_access_token", return_value="fake-token"), \
                 patch("photosearch.google_photos.create_album", return_value="ALBUM_FAKE_123"), \
                 patch("photosearch.google_photos.upload_photos") as mock_upload:

                # Make upload_photos call its progress_callback for each photo
                def fake_upload(db_path, records, album_id=None,
                                include_description=True, progress_callback=None,
                                begin_callback=None, bytes_done_callback=None):
                    results = []
                    for i, rec in enumerate(records):
                        fn = rec.get("filename", "")
                        if begin_callback:
                            begin_callback(i, len(records), fn)
                        if bytes_done_callback:
                            bytes_done_callback(i + 1, len(records), fn)
                        if progress_callback:
                            progress_callback(i + 1, len(records), fn,
                                              "uploaded", None, f"MEDIA_{i}")
                        results.append({
                            "filename": fn, "status": "uploaded",
                            "error": None, "media_item_id": f"MEDIA_{i}",
                        })
                    return results

                mock_upload.side_effect = fake_upload

                # Also need to mock batch_add_to_album (imported in the endpoint)
                with patch("photosearch.google_photos.batch_add_to_album", return_value={"added": 0, "errors": []}):
                    resp = client.post("/api/google/upload", json={
                        "photo_ids": [pid],
                        "album_title": "Test Album",
                        "collection_id": cid,
                    })

            assert resp.status_code == 200
            events = self._parse_sse(resp.text)

            # Must have a "start" event first
            types = [e["type"] for e in events]
            assert "start" in types, f"Expected 'start' event, got: {types}"
            assert "done" in types, f"Expected 'done' event, got: {types}"

            start_evt = next(e for e in events if e["type"] == "start")
            assert os.path.basename(tmp_path) in start_evt["filenames"]

            done_evt = next(e for e in events if e["type"] == "done")
            assert done_evt["uploaded"] == 1

        finally:
            os.unlink(tmp_path)

    def test_upload_no_auth_returns_401(self, client, db):
        """Without auth, the endpoint should return 401 (not an SSE stream)."""
        pid = db._test_photo_ids["DSC04878.JPG"]
        with patch("photosearch.google_photos.refresh_access_token", return_value=None):
            resp = client.post("/api/google/upload", json={
                "photo_ids": [pid],
                "album_title": "No Auth Album",
            })
        assert resp.status_code == 401


# =========================================================================
# Static serving
# =========================================================================

class TestStaticServing:
    def test_root_serves_html(self, client):
        resp = client.get("/")
        # May serve index.html or 404 depending on frontend dir existence
        assert resp.status_code in (200, 404)

    def test_shared_js_served(self, client):
        resp = client.get("/shared.js")
        # Depends on frontend dir; verify it doesn't crash
        assert resp.status_code in (200, 404)

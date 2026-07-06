"""Tests for real photo download — single-file attachment on /full?download=1
and the /api/review/download ZIP of selected photos.

Fixture photos have no files on disk, so the success paths run in replica mode
(originals fetched from a fake NAS), exactly like test_web_replica.py.
"""

import io
import zipfile

import pytest

from photosearch import web, worker_api


@pytest.fixture
def replica_client(client, monkeypatch):
    monkeypatch.setattr(worker_api, "_shutting_down", False)
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")

    def fake_fetch(photo_id, kind, timeout=30.0):
        return b"\xff\xd8\xff" + f"{kind}-{photo_id}".encode()  # fake JPEG bytes

    monkeypatch.setattr(web, "_fetch_from_nas", fake_fetch)
    return client


def test_full_inline_has_no_attachment_header(replica_client, db):
    pid = next(iter(db._test_photo_ids.values()))
    r = replica_client.get(f"/api/photos/{pid}/full")
    assert r.status_code == 200
    assert "attachment" not in r.headers.get("content-disposition", "")


def test_full_download_sets_attachment_with_filename(replica_client, db):
    fname, pid = next(iter(db._test_photo_ids.items()))
    r = replica_client.get(f"/api/photos/{pid}/full?download=1")
    assert r.status_code == 200
    cd = r.headers.get("content-disposition", "")
    assert cd.startswith("attachment")
    assert fname in cd  # original filename preserved


def _select(db, directory, photo_ids):
    for pid in photo_ids:
        db.conn.execute(
            "INSERT OR REPLACE INTO review_selections (photo_id, directory, selected) "
            "VALUES (?, ?, 1)", (pid, directory))
    db.conn.commit()


def test_review_download_zips_selected(replica_client, db):
    ids = db._test_photo_ids
    a, b = ids["DSC04894.JPG"], ids["DSC04907.JPG"]
    # endpoint resolves a relative dir under photo_root (/photos); selections are
    # stored under the resolved absolute path.
    _select(db, "/photos/2026/march", [a, b])

    r = replica_client.get("/api/review/download?directory=2026/march")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers.get("content-disposition", "")

    zf = zipfile.ZipFile(io.BytesIO(r.content))
    names = set(zf.namelist())
    assert "DSC04894.JPG" in names
    assert "DSC04907.JPG" in names


def test_review_download_no_selection_is_404(replica_client, db):
    r = replica_client.get("/api/review/download?directory=2026/march")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Date-range review — a synthetic "range:" scope key spans every folder.
# ---------------------------------------------------------------------------

def test_review_download_date_range_scope(replica_client, db):
    ids = db._test_photo_ids
    a, b = ids["DSC04894.JPG"], ids["DSC04907.JPG"]
    key = "range:2026-03-13..2026-03-13"
    _select(db, key, [a, b])

    r = replica_client.get(f"/api/review/download?directory={key}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    cd = r.headers.get("content-disposition", "")
    assert "attachment" in cd
    # Synthetic key doesn't leak ".." into the filename.
    assert ".." not in cd

    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert "DSC04894.JPG" in names
    assert "DSC04907.JPG" in names


def test_review_run_date_range_persists_under_range_key(replica_client, db):
    # Fixture photos are dated 2026-03-13; run a date-range review over them.
    # scipy may be mocked here, but the fallback (no embeddings → top by
    # quality) still produces selections and persists them.
    r = replica_client.get("/api/review/run?date_from=2026-03-13&date_to=2026-03-13")
    assert r.status_code == 200
    data = r.json()
    assert data["stats"]["total"] > 0

    # Persisted under the synthetic scope key, loadable back.
    row = db.conn.execute(
        "SELECT COUNT(*) AS c FROM review_selections WHERE directory = ?",
        ("range:2026-03-13..2026-03-13",),
    ).fetchone()
    assert row["c"] == data["stats"]["total"]

    r2 = replica_client.get("/api/review/load?directory=range:2026-03-13..2026-03-13")
    assert r2.status_code == 200
    assert r2.json()["stats"]["total"] == data["stats"]["total"]


def test_review_run_requires_directory_or_range(replica_client):
    r = replica_client.get("/api/review/run")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Collection download — whole collection and multi-selected subset.
# ---------------------------------------------------------------------------

def test_collection_download_all(replica_client, db):
    cid = db._test_collection_id
    r = replica_client.get(f"/api/collections/{cid}/download")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert "attachment" in r.headers.get("content-disposition", "")

    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    # Both collection members present.
    assert "DSC04907.JPG" in names
    assert "DSC04922.JPG" in names


def test_collection_download_selected_subset(replica_client, db):
    cid = db._test_collection_id
    only = db._test_photo_ids["DSC04907.JPG"]
    r = replica_client.get(f"/api/collections/{cid}/download?photo_ids={only}")
    assert r.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert "DSC04907.JPG" in names
    assert "DSC04922.JPG" not in names  # not in the requested subset


def test_collection_download_unknown_collection_404(replica_client):
    r = replica_client.get("/api/collections/999999/download")
    assert r.status_code == 404


def test_collection_download_bad_photo_ids_400(replica_client, db):
    cid = db._test_collection_id
    r = replica_client.get(f"/api/collections/{cid}/download?photo_ids=abc")
    assert r.status_code == 400

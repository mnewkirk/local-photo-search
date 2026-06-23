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

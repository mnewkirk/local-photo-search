"""photosearch/video_registry.py — video rows on the NAS (schema v35, step 1).

Registration is a tree walk + inserts: no ffprobe, no decode, no hashing.
"""

import os
import sqlite3

import pytest
from click.testing import CliRunner

from photosearch.db import PhotoDB
from photosearch.video_registry import (iter_video_files, register_video,
                                        register_videos)


def _touch(p, data=b"\x00\x00\x00\x18ftypisom", mtime=None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mtime is not None:
        os.utime(p, (mtime, mtime))


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "photos"
    _touch(root / "2016" / "2016-05-01_gphotos" / "IMG_0001.MOV", mtime=1_462_100_000)
    _touch(root / "2025" / "2025-07-04_phone-matt" / "PXL_1.mp4")
    _touch(root / "2025" / "2025-07-04_phone-matt" / "PXL_2.MP4")
    _touch(root / "2025" / "2025-07-04_phone-matt" / "IMG_9.JPG", b"\xff\xd8\xff\xe0")
    # never registered: AppleDouble, ingest staging + its archive, hidden dir,
    # thumbnails, a RAW
    _touch(root / "2025" / "2025-07-04_phone-matt" / "._PXL_1.mp4")
    _touch(root / "_incoming" / "matt" / "NEW.mp4")
    _touch(root / "_incoming" / "matt" / ".processed" / "OLD.mp4")
    _touch(root / "2025" / ".staging-abc" / "X.mp4")
    _touch(root / "thumbnails" / "T.mp4")
    _touch(root / "2025" / "2025-07-04_phone-matt" / "DSC1.ARW")
    dbp = str(tmp_path / "photo_index.db")
    with PhotoDB(dbp) as db:
        db.set_photo_root(str(root))
        db.add_photo(filepath="2025/2025-07-04_phone-matt/IMG_9.JPG",
                     filename="IMG_9.JPG")
    return dbp, root


def _video_rows(dbp):
    conn = sqlite3.connect(dbp)
    conn.row_factory = sqlite3.Row
    try:
        return {r["filepath"]: dict(r) for r in conn.execute(
            "SELECT p.*, m.photo_id AS meta_id, m.probe_version "
            "FROM photos p LEFT JOIN video_meta m ON m.photo_id = p.id "
            "WHERE p.media_type = 'video'")}
    finally:
        conn.close()


def test_walk_skips_excluded_hidden_and_appledouble(lib):
    _dbp, root = lib
    got = sorted(os.path.relpath(p, root) for p in iter_video_files(str(root)))
    assert got == ["2016/2016-05-01_gphotos/IMG_0001.MOV",
                   "2025/2025-07-04_phone-matt/PXL_1.mp4",
                   "2025/2025-07-04_phone-matt/PXL_2.MP4"]


def test_dry_run_writes_nothing_and_reports_per_year(lib):
    dbp, _root = lib
    before = os.path.getmtime(dbp), os.path.getsize(dbp)
    res = register_videos(dbp)
    assert res["dry_run"] and res["files"] == 3 and res["would_register"] == 3
    assert res["registered"] == 0
    assert res["per_year"] == {"2016": 1, "2025": 2}
    assert _video_rows(dbp) == {}
    assert (os.path.getmtime(dbp), os.path.getsize(dbp)) == before


def test_dry_run_on_a_missing_db_is_an_error(tmp_path):
    missing = str(tmp_path / "nope.db")
    with pytest.raises(FileNotFoundError):
        register_videos(missing, photo_root=str(tmp_path))
    assert not os.path.exists(missing)


def test_apply_inserts_rows_and_empty_video_meta(lib):
    dbp, _root = lib
    res = register_videos(dbp, apply=True, lower_priority=False)
    assert res["registered"] == 3
    rows = _video_rows(dbp)
    assert set(rows) == {"2016/2016-05-01_gphotos/IMG_0001.MOV",
                         "2025/2025-07-04_phone-matt/PXL_1.mp4",
                         "2025/2025-07-04_phone-matt/PXL_2.MP4"}
    old = rows["2016/2016-05-01_gphotos/IMG_0001.MOV"]
    assert old["filename"] == "IMG_0001.MOV"
    assert old["folder"] == "2016/2016-05-01_gphotos"
    assert old["date_taken"].startswith("2016-05-01")  # file mtime, provisional
    assert old["file_hash"] is None                      # never hashed on the NAS
    for r in rows.values():
        assert r["meta_id"] == r["id"] and r["probe_version"] is None


def test_second_run_is_idempotent(lib):
    dbp, _root = lib
    register_videos(dbp, apply=True, lower_priority=False)
    again = register_videos(dbp, apply=True, lower_priority=False)
    assert again["registered"] == 0 and again["already"] == 3
    assert len(_video_rows(dbp)) == 3


def test_folder_scope_and_limit(lib):
    dbp, _root = lib
    res = register_videos(dbp, folder="2025", apply=True, lower_priority=False)
    assert res["registered"] == 2 and set(res["per_year"]) == {"2025"}
    res = register_videos(dbp, limit=1)
    assert res["would_register"] == 1  # only the 2016 file is left
    assert register_videos(dbp, folder="2030")["files"] == 0


def test_register_video_is_one_row_and_keeps_the_given_hash(lib):
    dbp, root = lib
    with PhotoDB(dbp) as db:
        p = root / "2025" / "2025-07-04_phone-matt" / "PXL_1.mp4"
        a = register_video(db, str(p), "abc123", date_taken="2025-07-04 09:00:00")
        b = register_video(db, str(p), "abc123")
        assert a == b
        row = db.get_photo(a)
        assert row["file_hash"] == "abc123"
        assert row["date_taken"] == "2025-07-04 09:00:00"
        assert row["media_type"] == "video"
        assert db.conn.execute("SELECT COUNT(*) FROM video_meta").fetchone()[0] == 1


def test_cli_register_videos(lib):
    from cli import cli
    dbp, _root = lib
    r = CliRunner().invoke(cli, ["register-videos", "--db", dbp])
    assert r.exit_code == 0, r.output
    assert "3 video(s) would be registered" in r.output
    assert "2016 1" in r.output
    r = CliRunner().invoke(cli, ["register-videos", "--db", dbp, "--apply"])
    assert r.exit_code == 0, r.output
    assert "Registered 3 video(s)" in r.output
    r = CliRunner().invoke(cli, ["register-videos", "--db", dbp + ".missing"])
    assert r.exit_code != 0

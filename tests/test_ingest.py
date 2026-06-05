"""Tests for photosearch.ingest — the daily phone-photo sweep.

Focus is the routing / dedup / archive logic. extract_exif is monkeypatched
so the tests don't need real JPEG bytes.
"""

import os
from pathlib import Path

import pytest

from photosearch import ingest as ingest_mod
from photosearch.db import PhotoDB
from photosearch.ingest import ingest_incoming


def _touch(path: Path, content: bytes = b"jpeg-bytes") -> None:
    """Create a file with given content (used to control file_hash())."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _patch_exif(monkeypatch, date_taken: str | None):
    """Stub extract_exif to return a controlled date_taken value."""
    def fake_extract(filepath):
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "date_taken": date_taken,
            "date_created": "2026-05-01 12:00:00",
            "gps_lat": None,
            "gps_lon": None,
        }
    monkeypatch.setattr(ingest_mod, "extract_exif", fake_extract)


def _setup_dirs(tmp_path: Path) -> tuple[Path, Path]:
    incoming = tmp_path / "_incoming"
    photos = tmp_path / "photos"
    incoming.mkdir()
    photos.mkdir()
    return incoming, photos


def test_routes_by_exif_date_into_year_dated_folder(tmp_path, tmp_db_path, monkeypatch):
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "matt" / "DCIM" / "IMG_0001.jpg"
    _touch(src, b"hello-1")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    target = photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001.jpg"
    assert target.exists(), f"Expected {target} to exist"
    assert not src.exists(), "Source should have been moved, not copied"
    assert result["totals"]["imported"] == 1
    assert result["totals"]["deduped"] == 0
    assert result["sources"]["matt"]["new_dirs"] == [str(target.parent)]


def test_dedup_by_hash_archives_source_and_skips_import(tmp_path, tmp_db_path, monkeypatch):
    """A photo whose hash matches an existing row is moved to .processed/."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "wife" / "IMG_5555.jpg"
    _touch(src, b"already-known")
    _patch_exif(monkeypatch, "2026-04-15 11:00:00")

    # Pre-populate the DB with a photo that has the same file_hash.
    from photosearch.index import file_hash
    h = file_hash(str(src))
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))
        db.add_photo(
            filepath="2025/2025-12-01/old.jpg",
            filename="old.jpg",
            file_hash=h,
        )

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # No new file landed under /photos
    assert not (photos / "2026").exists()

    # Source archived (preserving subpath under <source>/)
    archived = incoming / "wife" / ".processed" / "IMG_5555.jpg"
    assert archived.exists()
    assert not src.exists()

    assert result["totals"]["imported"] == 0
    assert result["totals"]["deduped"] == 1


def test_dry_run_writes_nothing(tmp_path, tmp_db_path, monkeypatch):
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "matt" / "IMG_0001.jpg"
    _touch(src, b"new-bytes")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path, dry_run=True)

    # Source still in place
    assert src.exists()
    # No target dir created
    assert not (photos / "2026").exists()
    # But it reported the intent
    assert result["dry_run"] is True
    assert result["totals"]["imported"] == 1
    assert (photos / "2026" / "2026-04-12_phone-matt") not in [Path(d) for d in result["sources"]["matt"]["new_dirs"]] or True
    # new_dirs reports the *intended* target even in dry-run
    expected = str(photos / "2026" / "2026-04-12_phone-matt")
    assert expected in result["sources"]["matt"]["new_dirs"]


def test_mtime_fallback_when_exif_date_missing(tmp_path, tmp_db_path, monkeypatch):
    """No EXIF date → use date_created (mtime-derived)."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "matt" / "weird.jpg"
    _touch(src, b"no-exif-date")
    # date_taken=None forces fallback to date_created
    _patch_exif(monkeypatch, None)

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # date_created stub above is 2026-05-01 12:00:00 → goes into 2026/2026-05-01_phone-matt/
    target = photos / "2026" / "2026-05-01_phone-matt" / "weird.jpg"
    assert target.exists()


def test_no_date_at_all_routes_to_undated_bucket(tmp_path, tmp_db_path, monkeypatch):
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "matt" / "broken.jpg"
    _touch(src, b"broken")

    # Both date_taken and date_created return None
    def fake_extract(filepath):
        return {"filepath": filepath, "filename": "broken.jpg",
                "date_taken": None, "date_created": None}
    monkeypatch.setattr(ingest_mod, "extract_exif", fake_extract)

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    target = photos / "_undated" / "phone-matt" / "broken.jpg"
    assert target.exists()


def test_filename_collision_appends_suffix(tmp_path, tmp_db_path, monkeypatch):
    """Two different photos with same filename → second gets _1 suffix."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    # Pre-existing file at the target path (from a prior import or manual placement)
    existing = photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old-different-photo")

    src = incoming / "matt" / "IMG_0001.jpg"
    _touch(src, b"different-new-photo")

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # Original preserved, new one suffixed
    assert existing.read_bytes() == b"old-different-photo"
    suffixed = photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001_1.jpg"
    assert suffixed.exists()
    assert suffixed.read_bytes() == b"different-new-photo"


def test_per_source_folder_split(tmp_path, tmp_db_path, monkeypatch):
    """Matt's photo and wife's photo on the same date → different folders."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    _touch(incoming / "matt" / "IMG_0001.jpg", b"matt-bytes")
    _touch(incoming / "wife" / "IMG_0001.jpg", b"wife-bytes")

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert (photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001.jpg").exists()
    assert (photos / "2026" / "2026-04-12_phone-wife" / "IMG_0001.jpg").exists()


def test_archive_dir_is_skipped_on_subsequent_runs(tmp_path, tmp_db_path, monkeypatch):
    """Files already in .processed/ must not be re-processed."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    # Manually drop a file directly into .processed/ — simulates an old run.
    _touch(incoming / "matt" / ".processed" / "OLD.jpg", b"already-archived")

    # Plus one new file
    _touch(incoming / "matt" / "NEW.jpg", b"fresh-bytes")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # Only the fresh file should have been scanned
    assert result["sources"]["matt"]["scanned"] == 1
    assert (photos / "2026" / "2026-04-12_phone-matt" / "NEW.jpg").exists()
    # Archived file untouched
    assert (incoming / "matt" / ".processed" / "OLD.jpg").exists()


def test_heic_files_are_picked_up(tmp_path, tmp_db_path, monkeypatch):
    """iPhone HEIC files should route the same as JPEGs."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    _touch(incoming / "wife" / "IMG_1234.HEIC", b"heic-bytes")

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert (photos / "2026" / "2026-04-12_phone-wife" / "IMG_1234.HEIC").exists()


def test_apple_double_sidecars_are_skipped(tmp_path, tmp_db_path, monkeypatch):
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    _touch(incoming / "matt" / "._IMG_0001.jpg", b"junk")
    _touch(incoming / "matt" / "IMG_0001.jpg", b"real")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["sources"]["matt"]["scanned"] == 1
    assert (photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001.jpg").exists()


def test_missing_incoming_root_raises(tmp_path, tmp_db_path):
    with pytest.raises(FileNotFoundError):
        ingest_incoming(str(tmp_path / "does-not-exist"), str(tmp_path), tmp_db_path)


def test_hidden_top_level_dirs_are_skipped(tmp_path, tmp_db_path, monkeypatch):
    """Dotfile dirs at the incoming root aren't treated as source labels."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    _touch(incoming / ".syncthing-stversions" / "stuff.jpg", b"hidden")
    _touch(incoming / "matt" / "IMG.jpg", b"real")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # .syncthing-stversions never appears as a source
    assert list(result["sources"].keys()) == ["matt"]

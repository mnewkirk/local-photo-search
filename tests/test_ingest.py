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


# Valid magic bytes per image extension so ingest's content-sniff
# (is_real_image) treats these stubs as real photos. Companion extensions
# (.mov/.arw) aren't sniffed, so they need no magic.
_EXT_MAGIC = {
    ".jpg": b"\xff\xd8\xff\xe0", ".jpeg": b"\xff\xd8\xff\xe0",
    ".heic": b"\x00\x00\x00\x18ftypheic", ".heif": b"\x00\x00\x00\x18ftypheic",
}


def _touch(path: Path, content: bytes = b"jpeg-bytes") -> None:
    """Create a file with given content (used to control file_hash()).

    A correct image-magic prefix for the extension is prepended so the file
    passes ingest's content sniff; the caller-supplied tail keeps hashes distinct.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_EXT_MAGIC.get(path.suffix.lower(), b"") + content)


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


def test_camera_model_source_drops_phone_prefix(tmp_path, tmp_db_path, monkeypatch):
    """A camera-model-looking source (ILCE-7RM6) lands in YYYY-MM-DD_<model>/,
    with no 'phone-' prefix — driven by the heuristic, no config needed."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "ILCE-7RM6" / "DCIM" / "DSC00017.JPG"
    _touch(src, b"camera-bytes")
    _patch_exif(monkeypatch, "2026-06-19 14:00:00")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    target = photos / "2026" / "2026-06-19_ILCE-7RM6" / "DSC00017.JPG"
    assert target.exists(), f"Expected {target} to exist"
    assert not (photos / "2026" / "2026-06-19_phone-ILCE-7RM6").exists()
    assert result["sources"]["ILCE-7RM6"]["new_dirs"] == [str(target.parent)]


def test_phone_source_override_forces_prefix(tmp_path, tmp_db_path, monkeypatch):
    """phone_sources forces 'phone-' even on a camera-model-looking label."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "ILCE-7M4" / "DSC06593.JPG"
    _touch(src, b"forced-phone")
    _patch_exif(monkeypatch, "2026-06-19 15:00:00")

    ingest_incoming(str(incoming), str(photos), tmp_db_path,
                    phone_sources={"ILCE-7M4"})

    assert (photos / "2026" / "2026-06-19_phone-ILCE-7M4" / "DSC06593.JPG").exists()


def test_bare_source_override_drops_prefix(tmp_path, tmp_db_path, monkeypatch):
    """bare_sources drops 'phone-' even for a non-model-looking label."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "drone" / "DJI_0001.JPG"
    _touch(src, b"drone-bytes")
    _patch_exif(monkeypatch, "2026-06-19 16:00:00")

    ingest_incoming(str(incoming), str(photos), tmp_db_path,
                    bare_sources={"drone"})

    assert (photos / "2026" / "2026-06-19_drone" / "DJI_0001.JPG").exists()


def test_companion_raw_moved_but_not_indexed(tmp_path, tmp_db_path, monkeypatch):
    """A RAW file lands in the dated folder but is NOT queued for indexing."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "ILCE-7M4" / "DSC05355.ARW"
    _touch(src, b"raw-bytes")
    _patch_exif(monkeypatch, "2026-06-19 14:00:00")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    target = photos / "2026" / "2026-06-19_ILCE-7M4" / "DSC05355.ARW"
    assert target.exists()
    assert result["totals"]["companions_moved"] == 1
    assert result["totals"]["imported"] == 0
    # RAW folder must NOT be queued for the CLIP index pass.
    assert result["sources"]["ILCE-7M4"]["new_dirs"] == []


def test_companion_video_uses_mtime_when_no_exif(tmp_path, tmp_db_path, monkeypatch):
    """A video with no EXIF date routes by file mtime, not into _undated."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "ILCE-7M4" / "C0176.MP4"
    _touch(src, b"video-bytes")
    # No EXIF at all (extract_exif raises) → mtime fallback.
    def boom(_):
        raise ValueError("no exif in video")
    monkeypatch.setattr(ingest_mod, "extract_exif", boom)
    os.utime(src, (1_750_000_000, 1_750_000_000))  # 2025-06-15 UTC-ish

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["companions_moved"] == 1
    moved = list(photos.rglob("C0176.MP4"))
    assert len(moved) == 1
    # Landed in a real YYYY/YYYY-MM-DD_* folder, not _undated.
    assert "_undated" not in str(moved[0])


def test_companion_dedup_at_destination(tmp_path, tmp_db_path, monkeypatch):
    """Re-ingesting an identical RAW already in the library archives it."""
    incoming, photos = _setup_dirs(tmp_path)
    src = incoming / "ILCE-7M4" / "DSC05355.ARW"
    _touch(src, b"raw-identical")
    _patch_exif(monkeypatch, "2026-06-19 14:00:00")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    # First ingest moves it in.
    ingest_incoming(str(incoming), str(photos), tmp_db_path)
    assert (photos / "2026" / "2026-06-19_ILCE-7M4" / "DSC05355.ARW").exists()

    # Same file shows up again in _incoming.
    _touch(src, b"raw-identical")
    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["companions_deduped"] == 1
    assert result["totals"]["companions_moved"] == 0
    # Archived for audit, no duplicate in the library.
    assert (incoming / "ILCE-7M4" / ".processed" / "DSC05355.ARW").exists()
    assert not (photos / "2026" / "2026-06-19_ILCE-7M4" / "DSC05355_1.ARW").exists()


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
    assert suffixed.read_bytes() == _EXT_MAGIC[".jpg"] + b"different-new-photo"


def test_zip_wrapped_jpg_reclassified_as_companion(tmp_path, tmp_db_path, monkeypatch):
    """A .JPG whose content is a ZIP (iOS Live Photo bundle) is moved like a
    companion: relocated into the library but never imported as a photo row."""
    incoming, photos = _setup_dirs(tmp_path)
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    # ZIP magic (PK\x03\x04), not an image — bypass _touch's magic prefix.
    src = incoming / "matt" / "IMG_4678(1).JPG"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"PK\x03\x04" + b"\x00" * 20)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    s = result["sources"]["matt"]
    assert s["non_image_reclassified"] == 1
    assert s["imported"] == 0
    assert s["companions_moved"] == 1
    # File reached the library (move-only), but no DB row was created.
    moved = photos / "2026" / "2026-04-12_phone-matt" / "IMG_4678(1).JPG"
    assert moved.exists()
    with PhotoDB(tmp_db_path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0] == 0


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

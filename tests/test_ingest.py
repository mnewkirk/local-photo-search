"""Tests for photosearch.ingest — the daily phone-photo sweep.

Focus is the routing / dedup / archive logic. extract_exif is monkeypatched
so the tests don't need real JPEG bytes.
"""

import os
import sqlite3
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


def _patch_exif_model(monkeypatch, date_taken, model):
    """Like _patch_exif, but the file's own EXIF also names a camera model."""
    def fake_extract(filepath):
        return {"filepath": filepath, "filename": os.path.basename(filepath),
                "date_taken": date_taken, "date_created": "2026-05-01 12:00:00",
                "gps_lat": None, "gps_lon": None, "camera_model": model}
    monkeypatch.setattr(ingest_mod, "extract_exif", fake_extract)


def test_unknown_camera_raw_is_refiled_under_its_own_exif_model(tmp_path, tmp_db_path, monkeypatch):
    """The importer's 'unknown-camera' label is a FALLBACK, not a fact.

    The Windows importer reads the model through the shell property store, which
    has no codec for a new body's RAWs — on 2026-09-19 every ILCE-7RM6 .ARW
    arrived as 'unknown-camera' and was filed away from its own JPEG. The file
    knows what took it; ingest reads EXIF anyway, so believe the file.
    """
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "unknown-camera" / "102MSDCF" / "DSC00078.ARW", b"raw-bytes")
    _patch_exif_model(monkeypatch, "2026-09-19 10:00:00", "ILCE-7RM6")
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC00078.ARW").exists()
    assert not (photos / "2026" / "2026-09-19_unknown-camera").exists()


def test_unknown_camera_stays_when_the_file_has_no_usable_model(tmp_path, tmp_db_path, monkeypatch):
    """Video has no model; a junk/odd Model string is not a folder name."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "unknown-camera" / "C0001.MP4", b"video-a")
    _patch_exif_model(monkeypatch, "2026-09-19 10:00:00", None)
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))
    ingest_incoming(str(incoming), str(photos), tmp_db_path)
    assert (photos / "2026" / "2026-09-19_unknown-camera" / "C0001.MP4").exists()

    _touch(incoming / "unknown-camera" / "C0002.MP4", b"video-b")
    _patch_exif_model(monkeypatch, "2026-09-19 10:00:00", "../../etc")
    ingest_incoming(str(incoming), str(photos), tmp_db_path)
    assert (photos / "2026" / "2026-09-19_unknown-camera" / "C0002.MP4").exists()


def test_a_real_source_label_is_never_overridden_by_exif(tmp_path, tmp_db_path, monkeypatch):
    """Only the FALLBACK label defers to EXIF. 'nicole' is a person's camera
    roll whatever body took the shot, and a named model dir was chosen on purpose."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "nicole" / "IMG_1.JPG", b"a")
    _touch(incoming / "ILCE-7M4" / "DSC1.ARW", b"b")
    _patch_exif_model(monkeypatch, "2026-09-19 10:00:00", "ILCE-7RM6")
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))
    ingest_incoming(str(incoming), str(photos), tmp_db_path)
    assert (photos / "2026" / "2026-09-19_phone-nicole" / "IMG_1.JPG").exists()
    assert (photos / "2026" / "2026-09-19_ILCE-7M4" / "DSC1.ARW").exists()


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


def test_scan_incoming_classifies_by_disposition(tmp_path):
    """scan_incoming buckets files without hashing/EXIF: staged / blocked /
    companion / other, per source, skipping hidden files and the archive dir."""
    from photosearch.ingest import scan_incoming

    incoming = tmp_path / "_incoming"
    # matt: 1 staged jpg, 1 staged heic, 1 blocked (PK zip as .JPG), 1 other sidecar
    _touch(incoming / "matt" / "good.jpg", b"a")
    _touch(incoming / "matt" / "pic.heic", b"b")
    (incoming / "matt" / "live(1).JPG").write_bytes(b"PK\x03\x04" + b"\x00" * 20)
    (incoming / "matt" / "edit.aae").write_bytes(b"<plist/>")
    # hidden clutter — must be ignored
    (incoming / "matt" / ".DS_Store").write_bytes(b"x")
    (incoming / "matt" / "._good.jpg").write_bytes(b"x")
    # archive dir contents — must be ignored
    (incoming / "matt" / ".processed").mkdir()
    _touch(incoming / "matt" / ".processed" / "old.jpg", b"old")
    # camera source: RAW + video companions (move-only, never indexed)
    (incoming / "ILCE-7RM6" / "shot.ARW").parent.mkdir(parents=True)
    (incoming / "ILCE-7RM6" / "shot.ARW").write_bytes(b"II*\x00" + b"\x00" * 20)
    (incoming / "ILCE-7RM6" / "clip.mp4").write_bytes(b"\x00" * 20)

    result = scan_incoming(str(incoming))

    assert result["exists"] is True
    assert result["sources"]["matt"] == {
        "staged": 2, "blocked": 1, "companion": 0, "other": 1,
        "total": 4, "bytes": result["sources"]["matt"]["bytes"],
    }
    assert result["sources"]["ILCE-7RM6"]["companion"] == 2
    assert result["sources"]["ILCE-7RM6"]["staged"] == 0
    assert result["totals"]["staged"] == 2
    assert result["totals"]["blocked"] == 1
    assert result["totals"]["companion"] == 2
    assert result["totals"]["other"] == 1


def test_scan_incoming_missing_dir_returns_empty(tmp_path):
    """A non-existent _incoming reports exists=False with zeroed totals rather
    than raising (the endpoint polls it on a schedule)."""
    from photosearch.ingest import scan_incoming

    result = scan_incoming(str(tmp_path / "nope"))
    assert result["exists"] is False
    assert result["sources"] == {}
    assert result["totals"]["total"] == 0


# --- concurrency: two sweeps must not race on the same files ----------------

def test_file_vanishing_mid_sweep_is_skipped_not_fatal(tmp_path, tmp_db_path, monkeypatch):
    """A file removed between the directory walk and the move (concurrent sweep
    or an SD-card import still writing) is counted and skipped — the rest of
    the sweep still completes."""
    incoming, photos = _setup_dirs(tmp_path)
    doomed = incoming / "matt" / "DCIM" / "IMG_0001.jpg"
    survivor = incoming / "matt" / "DCIM" / "IMG_0002.jpg"
    _touch(doomed, b"doomed")
    _touch(survivor, b"survivor")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    # Delete the first file after the listing snapshot is taken.
    real_iter = ingest_mod._iter_source_files

    def racing_iter(source_root):
        files = real_iter(source_root)
        doomed.unlink()
        return files

    monkeypatch.setattr(ingest_mod, "_iter_source_files", racing_iter)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["vanished"] == 1
    assert result["totals"]["errors"] == 0
    assert result["totals"]["imported"] == 1
    assert (photos / "2026" / "2026-04-12_phone-matt" / "IMG_0002.jpg").exists()


def test_archive_failure_does_not_abort_sweep(tmp_path, tmp_db_path, monkeypatch):
    """The dedup path archives the source file; if that move fails (the file
    was already archived by a concurrent sweep) the run keeps going."""
    incoming, photos = _setup_dirs(tmp_path)
    dup = incoming / "matt" / "DCIM" / "IMG_0001.jpg"
    fresh = incoming / "matt" / "DCIM" / "IMG_0002.jpg"
    _touch(dup, b"dup")
    _touch(fresh, b"fresh")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    from photosearch.index import file_hash
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))
        db.add_photo(
            filepath="2026/already.jpg",
            filename="already.jpg",
            file_hash=file_hash(str(dup)),
        )

    def exploding_move(source_root, src_file):
        raise FileNotFoundError(str(src_file))

    monkeypatch.setattr(ingest_mod, "_move_to_archive", exploding_move)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["deduped"] == 1
    assert result["totals"]["vanished"] == 1
    assert result["totals"]["imported"] == 1


def test_second_concurrent_sweep_is_refused(tmp_path, tmp_db_path, monkeypatch):
    """The cross-process flock makes a second sweep raise instead of racing the
    first one for the same files (the crash this guard was added for)."""
    from photosearch.ingest import IngestAlreadyRunning, _sweep_lock

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "DCIM" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    with _sweep_lock(tmp_db_path):  # stand in for the already-running sweep
        with pytest.raises(IngestAlreadyRunning):
            ingest_incoming(str(incoming), str(photos), tmp_db_path)
        # dry runs write nothing, so they're allowed through
        preview = ingest_incoming(str(incoming), str(photos), tmp_db_path,
                                  dry_run=True)
        assert preview["totals"]["imported"] == 1

    # Lock released — a normal sweep runs again.
    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)
    assert result["totals"]["imported"] == 1


def test_same_photo_under_two_source_labels_lands_once(tmp_path, tmp_db_path, monkeypatch):
    """Ingest inserts no photo rows itself (index_directory does, afterwards),
    so the DB-hash dedup can't see a copy moved earlier in the SAME sweep. The
    intra-run hash set covers it — this is how a truncated importer label
    ('LCE-7RM6' next to 'ILCE-7RM6') produced duplicate library files."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "ILCE-7RM6" / "DSC06192.jpg", b"same-bytes")
    _touch(incoming / "LCE-7RM6" / "DSC06192.jpg", b"same-bytes")
    _patch_exif(monkeypatch, "2026-08-05 10:00:00")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    # First source (sorted: ILCE < LCE) wins; the second is archived, not landed.
    assert result["totals"]["imported"] == 1
    assert result["totals"]["deduped"] == 1
    assert (photos / "2026" / "2026-08-05_ILCE-7RM6" / "DSC06192.jpg").exists()
    assert not (photos / "2026" / "2026-08-05_LCE-7RM6").exists()
    assert (incoming / "LCE-7RM6" / ".processed" / "DSC06192.jpg").exists()


# --- sweep heartbeat / ingest_sweeps row (Task 2) --------------------------

def test_sweep_row_created_with_counters_after_real_move(tmp_path, tmp_db_path, monkeypatch):
    """A sweep that actually moves files creates an ingest_sweeps row, left
    in status 'moving' (the caller advances it), with files_seen/files_moved
    matching what was processed."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _touch(incoming / "matt" / "IMG_0002.jpg", b"two")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["run_id"] is not None
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute(
            "SELECT * FROM ingest_sweeps WHERE run_id = ?", (result["run_id"],)
        ).fetchone()
    assert row is not None
    assert row["status"] == "moving"
    assert row["files_seen"] == 2
    assert row["files_moved"] == 2


def test_no_sweep_row_when_incoming_is_empty(tmp_path, tmp_db_path):
    """A sweep that finds nothing to move creates no ingest_sweeps row."""
    incoming, photos = _setup_dirs(tmp_path)
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["run_id"] is None
    with PhotoDB(tmp_db_path) as db:
        count = db.conn.execute("SELECT COUNT(*) FROM ingest_sweeps").fetchone()[0]
    assert count == 0


def test_no_sweep_row_on_dry_run(tmp_path, tmp_db_path, monkeypatch):
    """A dry run previews without ever creating a sweep row (it writes nothing)."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path, dry_run=True)

    assert result["run_id"] is None
    with PhotoDB(tmp_db_path) as db:
        count = db.conn.execute("SELECT COUNT(*) FROM ingest_sweeps").fetchone()[0]
    assert count == 0


def test_exception_mid_sweep_marks_sweep_failed(tmp_path, tmp_db_path, monkeypatch):
    """A crash partway through the move loop must leave the sweep row
    'failed', never a phantom 'moving' row.

    Injected via `_unique_target_path` (an unguarded call right before the
    move) rather than `shutil.move`/`file_hash` directly: those two are
    deliberately wrapped in a local try/except per file (a bad file must not
    abort the whole sweep — see the HASH FAIL / MOVE FAIL comments), so
    patching them to raise would just be swallowed and counted as a normal
    per-file error, not a crash. This exercises the same "wrap the sweep loop"
    safety net for a genuinely unhandled exception.
    """
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _touch(incoming / "matt" / "IMG_0002.jpg", b"two")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    calls = {"n": 0}
    real_unique = ingest_mod._unique_target_path

    def boom(target_dir, filename):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_unique(target_dir, filename)

    monkeypatch.setattr(ingest_mod, "_unique_target_path", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_incoming(str(incoming), str(photos), tmp_db_path)

    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT status, error FROM ingest_sweeps").fetchone()
    assert row is not None
    assert row["status"] == "failed"
    assert "simulated crash" in row["error"]


def test_heartbeat_throttled_and_always_fires_once_at_the_end(tmp_path, tmp_db_path, monkeypatch):
    """heartbeat_sweep is throttled to at most once per second of wall time,
    but the loop always emits one final (forced) call when it ends."""
    incoming, photos = _setup_dirs(tmp_path)
    for i in range(3):
        _touch(incoming / "matt" / f"IMG_000{i}.jpg", f"file-{i}".encode())
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    calls: list[tuple[int, int]] = []
    real_heartbeat = ingest_mod.heartbeat_sweep

    def spy(db, run_id, *, files_seen, files_moved):
        calls.append((files_seen, files_moved))
        return real_heartbeat(db, run_id, files_seen=files_seen, files_moved=files_moved)

    monkeypatch.setattr(ingest_mod, "heartbeat_sweep", spy)
    # Freeze wall-clock time so the 1s throttle window never elapses on its
    # own — only the very first (last_heartbeat starts at 0) and the forced
    # final call should get through.
    monkeypatch.setattr(ingest_mod.time, "monotonic", lambda: 1000.0)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["imported"] == 3
    assert len(calls) == 2
    assert calls[-1] == (3, 3)


def test_heartbeat_failure_never_aborts_the_move_loop(tmp_path, tmp_db_path, monkeypatch):
    """Progress telemetry must degrade, never take the sweep down: if
    heartbeat_sweep raises on every call (busy timeout, disk error), the
    files still move and the returned stats are still correct — no
    exception escapes ingest_incoming."""
    incoming, photos = _setup_dirs(tmp_path)
    for i in range(3):
        _touch(incoming / "matt" / f"IMG_000{i}.jpg", f"file-{i}".encode())
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    def raising_heartbeat(db, run_id, *, files_seen, files_moved):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ingest_mod, "heartbeat_sweep", raising_heartbeat)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["imported"] == 3
    assert result["totals"]["errors"] == 0
    for i in range(3):
        assert (photos / "2026" / "2026-04-12_phone-matt" / f"IMG_000{i}.jpg").exists()

    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT status FROM ingest_sweeps").fetchone()
    # The sweep row was created (start_sweep itself wasn't touched by this
    # test) and still reaches a normal terminal-ish state -- it's the
    # heartbeat writes, not the row's existence, that failed.
    assert row["status"] == "moving"


def test_start_sweep_failure_leaves_run_id_none_and_sweep_completes(
    tmp_path, tmp_db_path, monkeypatch
):
    """If start_sweep itself raises, the sweep must not abort: run_id stays
    None (heartbeats become no-ops) and the move loop runs to completion."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    def raising_start_sweep(db):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ingest_mod, "start_sweep", raising_start_sweep)

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["imported"] == 1
    assert result["run_id"] is None
    assert (photos / "2026" / "2026-04-12_phone-matt" / "IMG_0001.jpg").exists()

    with PhotoDB(tmp_db_path) as db:
        count = db.conn.execute("SELECT COUNT(*) FROM ingest_sweeps").fetchone()[0]
    # No sweep row was ever created -- start_sweep never succeeded.
    assert count == 0


def test_failing_set_sweep_status_does_not_mask_the_original_exception(
    tmp_path, tmp_db_path, monkeypatch
):
    """A real crash mid-loop plus a set_sweep_status('failed') that ALSO
    raises must still surface the ORIGINAL exception -- the secondary
    failure (e.g. the same DB error that caused the crash) must be
    swallowed-and-logged, not let to mask the real root cause."""
    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _touch(incoming / "matt" / "IMG_0002.jpg", b"two")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    calls = {"n": 0}
    real_unique = ingest_mod._unique_target_path

    def boom(target_dir, filename):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real_unique(target_dir, filename)

    monkeypatch.setattr(ingest_mod, "_unique_target_path", boom)

    def raising_set_sweep_status(db, run_id, status, error=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ingest_mod, "set_sweep_status", raising_set_sweep_status)

    with pytest.raises(RuntimeError, match="simulated crash"):
        ingest_incoming(str(incoming), str(photos), tmp_db_path)


def test_intra_run_dedup_also_covers_companions(tmp_path, tmp_db_path, monkeypatch):
    """RAW/video have no DB row at all, so the destination-path check is their
    only guard — and it misses a duplicate routed to a DIFFERENT dated folder."""
    incoming, photos = _setup_dirs(tmp_path)
    for label in ("ILCE-7RM6", "LCE-7RM6"):
        p = incoming / label / "DSC06192.ARW"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"II*\x00raw-bytes")
    _patch_exif(monkeypatch, "2026-08-05 10:00:00")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = ingest_incoming(str(incoming), str(photos), tmp_db_path)

    assert result["totals"]["companions_moved"] == 1
    assert result["totals"]["companions_deduped"] == 1
    assert (photos / "2026" / "2026-08-05_ILCE-7RM6" / "DSC06192.ARW").exists()
    assert not (photos / "2026" / "2026-08-05_LCE-7RM6").exists()


# --- CLI: batch registration + sweep status transitions (Task 2) -----------

def test_cli_ingest_incoming_registers_batch_and_marks_registered(tmp_path, tmp_db_path, monkeypatch):
    """The default (--index) path: after the post-move index pass runs,
    ingest-incoming registers a batch for the new folder and drives the
    sweep row through indexing -> registered."""
    from click.testing import CliRunner
    from cli import cli
    from photosearch.ingest_batches import list_batches

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
    ])
    assert result.exit_code == 0, result.output

    with PhotoDB(tmp_db_path) as db:
        batches = list_batches(db)
        sweep = db.conn.execute("SELECT status, run_id FROM ingest_sweeps").fetchone()

    assert len(batches) == 1
    assert batches[0]["directory"] == "2026/2026-04-12_phone-matt"
    assert batches[0]["source"] == "matt"
    assert batches[0]["run_id"] == sweep["run_id"]
    assert sweep["status"] == "registered"


def test_cli_ingest_incoming_no_index_skips_registration_for_unindexed_dir(tmp_path, tmp_db_path, monkeypatch):
    """--no-index leaves the brand-new folder with no photo rows, so
    register_batch's ValueError is swallowed (no batch row created) — but
    the sweep still finishes in 'registered', not stuck 'indexing' forever."""
    from click.testing import CliRunner
    from cli import cli
    from photosearch.ingest_batches import list_batches

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
        "--no-index",
    ])
    assert result.exit_code == 0, result.output

    with PhotoDB(tmp_db_path) as db:
        batches = list_batches(db)
        sweep = db.conn.execute("SELECT status FROM ingest_sweeps").fetchone()

    assert batches == []
    assert sweep["status"] == "registered"


def test_cli_ingest_incoming_no_index_registers_already_indexed_folder(tmp_path, tmp_db_path, monkeypatch):
    """A later --no-index sweep into a folder an earlier, indexed sweep
    already populated with photo rows still registers (widens) that batch —
    the ValueError guard only bites folders with zero rows."""
    from click.testing import CliRunner
    from cli import cli
    from photosearch.ingest_batches import list_batches

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    runner = CliRunner()
    first = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
    ])
    assert first.exit_code == 0, first.output

    _touch(incoming / "matt" / "IMG_0002.jpg", b"two")
    second = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
        "--no-index",
    ])
    assert second.exit_code == 0, second.output

    with PhotoDB(tmp_db_path) as db:
        batches = list_batches(db)

    assert len(batches) == 1
    assert batches[0]["directory"] == "2026/2026-04-12_phone-matt"
    # photo_count reflects only the first (indexed) file — the --no-index
    # second file moved into the same folder but got no DB row.
    assert batches[0]["photo_count"] == 1


def test_cli_ingest_incoming_dry_run_touches_no_batch_or_sweep_tables(tmp_path, tmp_db_path, monkeypatch):
    """--dry-run must not create a sweep row or a batch row."""
    from click.testing import CliRunner
    from cli import cli

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
        "--dry-run",
    ])
    assert result.exit_code == 0, result.output

    with PhotoDB(tmp_db_path) as db:
        sweeps = db.conn.execute("SELECT COUNT(*) FROM ingest_sweeps").fetchone()[0]
        batches = db.conn.execute("SELECT COUNT(*) FROM ingest_batches").fetchone()[0]
    assert sweeps == 0
    assert batches == 0


def test_cli_ingest_incoming_register_batch_exception_does_not_abort_sweep(
    tmp_path, tmp_db_path, monkeypatch
):
    """register_batch raising something other than ValueError (e.g. a
    sqlite3.OperationalError "database is locked" from a real lock contest)
    must not abort the rest of the sweep — bookkeeping failures are logged
    as a WARNING and folders after the failing one still get indexed and
    registered. The sweep must finish 'registered', not 'failed', and the
    command must exit 0."""
    import sqlite3

    from click.testing import CliRunner

    from cli import cli
    from photosearch import ingest_batches as ingest_batches_mod
    from photosearch.ingest_batches import list_batches

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _touch(incoming / "wife" / "IMG_0002.jpg", b"two")

    dates = {
        str(incoming / "matt" / "IMG_0001.jpg"): "2026-04-12 09:30:15",
        str(incoming / "wife" / "IMG_0002.jpg"): "2026-05-01 09:30:15",
    }

    def fake_extract(filepath):
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "date_taken": dates.get(filepath),
            "date_created": "2026-05-01 12:00:00",
            "gps_lat": None,
            "gps_lon": None,
        }

    monkeypatch.setattr(ingest_mod, "extract_exif", fake_extract)

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    real_register_batch = ingest_batches_mod.register_batch
    calls = []

    def flaky_register_batch(pdb, directory, *, source=None, run_id=None):
        calls.append(directory)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_register_batch(pdb, directory, source=source, run_id=run_id)

    monkeypatch.setattr(ingest_batches_mod, "register_batch", flaky_register_batch)

    runner = CliRunner()
    result = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
    ])
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.output
    # Both dirs were attempted -- the first dir's failure did not stop the loop.
    assert len(calls) == 2

    with PhotoDB(tmp_db_path) as db:
        batches = list_batches(db)
        sweep = db.conn.execute("SELECT status FROM ingest_sweeps").fetchone()

    # Only the second (wife) dir registered -- the first (matt) dir's
    # register_batch call raised and was skipped.
    assert len(batches) == 1
    assert batches[0]["directory"] == "2026/2026-05-01_phone-wife"
    assert sweep["status"] == "registered"


def test_cli_ingest_incoming_no_index_register_batch_exception_warns_and_continues(
    tmp_path, tmp_db_path, monkeypatch
):
    """Same guard on the --no-index registration branch: a non-ValueError
    from register_batch must not abort the sweep, and (unlike the previous
    silent-swallow behaviour) it must print a WARNING just like the --index
    branch does."""
    import sqlite3

    from click.testing import CliRunner

    from cli import cli
    from photosearch import ingest_batches as ingest_batches_mod
    from photosearch.ingest_batches import list_batches

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "matt" / "IMG_0001.jpg", b"one")
    _patch_exif(monkeypatch, "2026-04-12 09:30:15")

    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    # First (indexed) sweep so the folder has photo rows and is eligible to
    # be registered by the --no-index branch below.
    runner = CliRunner()
    first = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
    ])
    assert first.exit_code == 0, first.output

    _touch(incoming / "matt" / "IMG_0002.jpg", b"two")

    def raising_register_batch(pdb, directory, *, source=None, run_id=None):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(ingest_batches_mod, "register_batch", raising_register_batch)

    second = runner.invoke(cli, [
        "ingest-incoming",
        "--incoming-root", str(incoming),
        "--photo-root", str(photos),
        "--db", tmp_db_path,
        "--no-index",
    ])
    assert second.exit_code == 0, second.output
    assert "WARNING" in second.output

    with PhotoDB(tmp_db_path) as db:
        sweep = db.conn.execute(
            "SELECT status FROM ingest_sweeps ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    assert sweep["status"] == "registered"


# --- CLI: --no-clip registers rows without CLIP -------------------------------

def _ingest_cli(tmp_path, tmp_db_path, monkeypatch, *flags):
    """Run `ingest-incoming` on one photo; returns (result, photos_root)."""
    from click.testing import CliRunner
    from cli import cli

    incoming, photos = _setup_dirs(tmp_path)
    _touch(incoming / "ILCE-7RM6" / "DSC00001.jpg", b"cli-no-clip")
    _patch_exif(monkeypatch, "2026-09-20 10:00:00")
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))

    result = CliRunner().invoke(cli, [
        "ingest-incoming", "--incoming-root", str(incoming),
        "--photo-root", str(photos), "--db", tmp_db_path, *flags,
    ])
    assert result.exit_code == 0, result.output
    return result, photos


def test_no_clip_passes_clip_and_colors_off(tmp_path, tmp_db_path, monkeypatch):
    import cli as cli_mod
    calls = []
    monkeypatch.setattr(cli_mod, "index_directory", lambda **kw: calls.append(kw))

    _ingest_cli(tmp_path, tmp_db_path, monkeypatch, "--no-clip")

    assert len(calls) == 1
    assert calls[0]["enable_clip"] is False
    assert calls[0]["enable_colors"] is False


def test_default_still_clips(tmp_path, tmp_db_path, monkeypatch):
    import cli as cli_mod
    calls = []
    monkeypatch.setattr(cli_mod, "index_directory", lambda **kw: calls.append(kw))

    _ingest_cli(tmp_path, tmp_db_path, monkeypatch, "--no-colors")

    assert calls[0]["enable_clip"] is True
    assert calls[0]["enable_colors"] is False


def test_no_clip_still_registers_the_photo(tmp_path, tmp_db_path, monkeypatch):
    """The point of --no-clip over --no-index: the moved photo gets a DB row
    (with its hash), so the fleet can claim it and the next sweep dedups
    against it — but no embedding is computed here."""
    _, photos = _ingest_cli(tmp_path, tmp_db_path, monkeypatch, "--no-clip")

    with PhotoDB(tmp_db_path) as db:
        rows = db.conn.execute("SELECT id, file_hash FROM photos").fetchall()
        assert len(rows) == 1
        assert rows[0][1]  # hash populated -> future dedup works
        n_emb = db.conn.execute("SELECT COUNT(*) FROM clip_embeddings").fetchone()[0]
        assert n_emb == 0

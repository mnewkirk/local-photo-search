"""Tests for photosearch.refile — re-filing historical `_unknown-camera` folders.

Hermetic: every test builds a throwaway photo root under tmp_path and
monkeypatches `extract_exif` (the same shape tests/test_ingest.py uses) so no
real RAW bytes are needed. Nothing here touches a real library.
"""

import csv
import os
from pathlib import Path

import pytest

from photosearch import refile as refile_mod
from photosearch.db import PhotoDB
from photosearch.refile import (
    DestinationExists,
    refile_unknown_camera,
    undo_refile,
    _move_file,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_EXT_MAGIC = {
    ".jpg": b"\xff\xd8\xff\xe0", ".jpeg": b"\xff\xd8\xff\xe0",
    ".heic": b"\x00\x00\x00\x18ftypheic",
}


def _touch(path: Path, content: bytes = b"bytes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_EXT_MAGIC.get(path.suffix.lower(), b"") + content)


def _patch_exif_by_name(monkeypatch, models: dict):
    """Stub extract_exif: `models` maps a filename to its EXIF camera model.

    A filename absent from the map (or mapped to None) yields no model — the
    'video / unreadable RAW' case the tool must leave alone.
    """
    def fake_extract(filepath):
        return {
            "filepath": filepath,
            "filename": os.path.basename(filepath),
            "date_taken": None,
            "date_created": None,
            "camera_make": "SONY",
            "camera_model": models.get(os.path.basename(filepath)),
        }
    monkeypatch.setattr(refile_mod, "extract_exif", fake_extract)


def _root(tmp_path: Path) -> Path:
    photos = tmp_path / "photos"
    photos.mkdir()
    return photos


def _db(tmp_db_path: str, photos: Path) -> str:
    with PhotoDB(tmp_db_path) as db:
        db.set_photo_root(str(photos))
    return tmp_db_path


def _listing(root: Path) -> set:
    return {str(p.relative_to(root)) for p in root.rglob("*")}


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------

def test_single_model_day_moves_every_file(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    for n in ("DSC01.ARW", "DSC02.ARW"):
        _touch(src / n, n.encode())
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6", "DSC02.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    dest = photos / "2026" / "2026-09-19_ILCE-7RM6"
    assert (dest / "DSC01.ARW").exists()
    assert (dest / "DSC02.ARW").exists()
    assert stats["totals"]["moved"] == 2
    assert stats["folders"][0]["by_model"] == {"ILCE-7RM6": 2}


def test_two_model_day_routes_per_file(tmp_path, tmp_db_path, monkeypatch):
    """The destination comes from each file's OWN EXIF, never from the date."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-13_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _touch(src / "B.ARW", b"b")
    # Both bodies were in use that day — the sibling folders already exist.
    (photos / "2026" / "2026-09-13_ILCE-7M4").mkdir(parents=True)
    (photos / "2026" / "2026-09-13_ILCE-7RM6").mkdir(parents=True)
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7M4", "B.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (photos / "2026" / "2026-09-13_ILCE-7M4" / "A.ARW").exists()
    assert (photos / "2026" / "2026-09-13_ILCE-7RM6" / "B.ARW").exists()
    assert stats["totals"]["moved"] == 2


def test_file_with_no_model_is_left_in_place(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "CLIP.MP4", b"video")
    _touch(src / "DSC01.ARW", b"raw")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "CLIP.MP4").exists(), "no-model file must never be guessed at"
    assert stats["totals"]["no_model"] == 1
    assert stats["totals"]["moved"] == 1
    # Folder still holds the no-model file, so it is not removed.
    assert src.is_dir()


def test_date_comes_from_the_source_folder_not_exif(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")

    def fake_extract(filepath):
        # EXIF says the shot is a day earlier (midnight/timezone edge).
        return {"camera_model": "ILCE-7RM6", "date_taken": "2026-09-18 23:59:00"}
    monkeypatch.setattr(refile_mod, "extract_exif", fake_extract)
    _db(tmp_db_path, photos)

    refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                          audit_path=str(tmp_path / "audit.csv"))

    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW").exists()


# ---------------------------------------------------------------------------
# collisions
# ---------------------------------------------------------------------------

def test_identical_collision_leaves_source_and_counts_duplicate(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"same-bytes")
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW", b"same-bytes")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.ARW").exists(), "redundant copies are never deleted"
    assert stats["totals"]["duplicate_left"] == 1
    assert stats["totals"]["moved"] == 0


def test_different_content_collision_never_overwrites(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"source-bytes")
    dest = photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW"
    _touch(dest, b"destination-bytes")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert dest.read_bytes().endswith(b"destination-bytes")
    assert (src / "DSC01.ARW").exists()
    assert stats["totals"]["conflict"] == 1
    rows = list(csv.DictReader(open(tmp_path / "audit.csv")))
    conflict = [r for r in rows if r["action"] == "conflict"][0]
    assert conflict["source"].endswith("2026-09-19_unknown-camera/DSC01.ARW")
    assert conflict["destination"].endswith("2026-09-19_ILCE-7RM6/DSC01.ARW")


# ---------------------------------------------------------------------------
# dry run / safety gates
# ---------------------------------------------------------------------------

def test_dry_run_moves_nothing_and_writes_no_audit(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    before = _listing(photos)
    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert _listing(photos) == before
    assert stats["dry_run"] is True
    assert stats["totals"]["would_move"] == 1
    assert stats["totals"]["moved"] == 0
    assert not (tmp_path / "audit.csv").exists()


def test_apply_without_audit_refuses(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    with pytest.raises(ValueError, match="audit"):
        refile_unknown_camera(str(photos), tmp_db_path, apply=True)
    assert (src / "DSC01.ARW").exists()


def test_only_restricts_to_named_folders(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    a = photos / "2026" / "2026-09-18_unknown-camera"
    b = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(a / "A.ARW", b"a")
    _touch(b / "B.ARW", b"b")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6", "B.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  only=["2026-09-19_unknown-camera"])

    assert (a / "A.ARW").exists()
    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "B.ARW").exists()
    assert stats["totals"]["moved"] == 1
    assert len(stats["folders"]) == 1


def test_limit_caps_the_number_of_moves(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    for i in range(5):
        _touch(src / f"D{i}.ARW", f"raw{i}".encode())
    _patch_exif_by_name(monkeypatch, {f"D{i}.ARW": "ILCE-7RM6" for i in range(5)})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"), limit=2)

    assert stats["totals"]["moved"] == 2
    assert len(list((photos / "2026" / "2026-09-19_ILCE-7RM6").iterdir())) == 2


# ---------------------------------------------------------------------------
# DB consistency
# ---------------------------------------------------------------------------

def test_indexed_photo_is_skipped_by_default(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                     filename="DSC01.JPG", file_hash="h1")

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.JPG").exists()
    assert stats["totals"]["skipped_indexed"] == 1
    rows = list(csv.DictReader(open(tmp_path / "audit.csv")))
    assert any(r["action"] == "skipped_indexed" for r in rows)


def test_include_indexed_moves_and_updates_filepath_and_folder(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                           filename="DSC01.JPG", file_hash="h1")

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  include_indexed=True)

    assert stats["totals"]["moved"] == 1
    assert stats["totals"]["db_updated"] == 1
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute(
            "SELECT filepath, folder FROM photos WHERE id = ?", (pid,)).fetchone()
        assert row["filepath"] == "2026/2026-09-19_ILCE-7RM6/DSC01.JPG"
        assert row["folder"] == "2026/2026-09-19_ILCE-7RM6"
        assert os.path.exists(db.resolve_filepath(row["filepath"]))


def test_raw_filepath_reference_counts_as_indexed(tmp_path, tmp_db_path, monkeypatch):
    """A RAW named by some photo's raw_filepath is DB-linked even with no row of its own."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6", "DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                           filename="DSC01.JPG", file_hash="h1",
                           raw_filepath="2026/2026-09-19_unknown-camera/DSC01.ARW")

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))
    assert (src / "DSC01.ARW").exists()
    assert stats["totals"]["skipped_indexed"] == 2

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit2.csv"),
                                  include_indexed=True)
    assert stats["totals"]["moved"] == 2
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute(
            "SELECT filepath, raw_filepath FROM photos WHERE id = ?", (pid,)).fetchone()
        assert row["filepath"] == "2026/2026-09-19_ILCE-7RM6/DSC01.JPG"
        assert row["raw_filepath"] == "2026/2026-09-19_ILCE-7RM6/DSC01.ARW"


def test_crash_between_move_and_db_write_heals_on_rerun(tmp_path, tmp_db_path, monkeypatch):
    """Move-then-write: a crash leaves the row pointing at a gone file; a re-run repairs it."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    src.mkdir(parents=True)
    dest = photos / "2026" / "2026-09-19_ILCE-7RM6"
    # Simulate the crashed state: the file is already at its destination,
    # the DB still names the old path.
    _touch(dest / "DSC01.JPG", b"jpeg")
    from photosearch.index import file_hash
    h = file_hash(str(dest / "DSC01.JPG"))
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                           filename="DSC01.JPG", file_hash=h)
    _patch_exif_by_name(monkeypatch, {})

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  include_indexed=True)

    assert stats["totals"]["healed"] == 1
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT filepath, folder FROM photos WHERE id = ?",
                              (pid,)).fetchone()
        assert row["filepath"] == "2026/2026-09-19_ILCE-7RM6/DSC01.JPG"
        assert row["folder"] == "2026/2026-09-19_ILCE-7RM6"


# ---------------------------------------------------------------------------
# folder handling
# ---------------------------------------------------------------------------

def test_empty_source_dir_removed_non_empty_kept(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    clean = photos / "2026" / "2026-09-19_unknown-camera"
    messy = photos / "2026" / "2026-09-18_unknown-camera"
    _touch(clean / "A.ARW", b"a")
    _touch(messy / "B.ARW", b"b")
    _touch(messy / ".DS_Store", b"ds")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6", "B.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert not clean.exists()
    assert messy.is_dir(), "a folder with leftovers is kept, never rmtree'd"
    messy_stats = [f for f in stats["folders"] if f["name"].startswith("2026-09-18")][0]
    assert ".DS_Store" in messy_stats["remaining"]


def test_dry_run_predicts_what_would_be_left_behind(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")        # moves
    _touch(src / "CLIP.MP4", b"vid")   # no model — stays
    _touch(src / ".DS_Store", b"ds")   # dropping — stays
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    f = stats["folders"][0]
    assert f["remaining"] == [".DS_Store", "CLIP.MP4"]
    assert f["removed"] is False


def test_nested_subfolder_is_left_alone_and_reported(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _touch(src / "sub" / "B.ARW", b"b")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6", "B.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "sub" / "B.ARW").exists(), "nested files are never flattened"
    assert stats["totals"]["moved"] == 1
    assert stats["folders"][0]["nested_dirs"] == ["sub"]
    assert src.is_dir()


def test_non_matching_folder_names_are_ignored(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    for name in ("2026-09-19_ILCE-7RM6", "unknown-camera", "notadate_unknown-camera"):
        _touch(photos / "2026" / name / "X.ARW", name.encode())
    _patch_exif_by_name(monkeypatch, {"X.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert stats["folders"] == []
    assert stats["totals"]["would_move"] == 0


def test_undated_unknown_camera_is_skipped_with_a_message(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "_undated" / "unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert (src / "A.ARW").exists()
    assert stats["folders"] == []
    assert stats["skipped_undated"] == 1


def test_second_run_after_apply_is_a_no_op(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                          audit_path=str(tmp_path / "audit.csv"))
    after_first = _listing(photos)
    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit2.csv"))

    assert _listing(photos) == after_first
    assert stats["totals"]["moved"] == 0
    assert stats["folders"] == []


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

def test_undo_restores_every_moved_file(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _touch(src / "B.ARW", b"b")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7M4", "B.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    before = _listing(photos)
    audit = str(tmp_path / "audit.csv")

    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=audit)
    assert _listing(photos) != before

    stats = undo_refile(audit, tmp_db_path, apply=True)

    assert stats["restored"] == 2
    assert (src / "A.ARW").exists() and (src / "B.ARW").exists()


def test_undo_dry_run_changes_nothing(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = str(tmp_path / "audit.csv")
    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=audit)
    after_move = _listing(photos)

    stats = undo_refile(audit, tmp_db_path)

    assert _listing(photos) == after_move
    assert stats["would_restore"] == 1
    assert stats["restored"] == 0


def test_undo_refuses_when_the_destination_changed(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = str(tmp_path / "audit.csv")
    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=audit)

    moved = photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW"
    moved.write_bytes(b"edited-since-the-move")

    stats = undo_refile(audit, tmp_db_path, apply=True)

    assert stats["restored"] == 0
    assert stats["refused"] == 1
    assert moved.exists() and not (src / "A.ARW").exists()


def test_undo_refuses_when_the_source_path_is_occupied(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = str(tmp_path / "audit.csv")
    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=audit)

    _touch(src / "A.ARW", b"something-else-arrived")

    stats = undo_refile(audit, tmp_db_path, apply=True)

    assert stats["restored"] == 0
    assert stats["refused"] == 1


def test_undo_restores_the_db_row_too(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                           filename="DSC01.JPG", file_hash="h1")
    audit = str(tmp_path / "audit.csv")
    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=audit,
                          include_indexed=True)

    undo_refile(audit, tmp_db_path, apply=True)

    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT filepath, folder FROM photos WHERE id = ?",
                              (pid,)).fetchone()
        assert row["filepath"] == "2026/2026-09-19_unknown-camera/DSC01.JPG"
        assert row["folder"] == "2026/2026-09-19_unknown-camera"


# ---------------------------------------------------------------------------
# the suffix rule is ingest's, not a reimplementation
# ---------------------------------------------------------------------------

def test_model_that_fails_the_camera_model_shape_is_not_used(tmp_path, tmp_db_path, monkeypatch):
    """`_file_suffix` rejects a lowercase/odd model; such a file must stay put."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "iPhone 15 Pro"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert stats["totals"]["no_model"] == 1
    assert stats["totals"]["would_move"] == 0


def test_sibling_inference_is_off_by_default(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {})  # no model on the RAW
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert stats["totals"]["no_model"] == 1
    assert stats["totals"]["would_move"] == 0


def test_sibling_inference_routes_when_exactly_one_model_folder(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  infer_from_sibling=True)

    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW").exists()
    assert stats["totals"]["inferred"] == 1
    assert stats["totals"]["moved"] == 1


def test_sibling_inference_declines_on_a_two_body_day(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"raw")
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG", b"jpeg")
    _touch(photos / "2026" / "2026-09-19_ILCE-7M4" / "OTHER.JPG", b"other")
    _patch_exif_by_name(monkeypatch, {})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  infer_from_sibling=True)

    assert (src / "DSC01.ARW").exists()
    assert stats["totals"]["no_model"] == 1
    assert stats["totals"]["inferred"] == 0


def test_audit_records_absolute_paths(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                          audit_path=str(tmp_path / "audit.csv"))

    rows = list(csv.DictReader(open(tmp_path / "audit.csv")))
    row = [r for r in rows if r["action"] == "moved"][0]
    assert os.path.isabs(row["source"]) and os.path.isabs(row["destination"])
    assert row["model"] == "ILCE-7RM6"
    assert int(row["size"]) > 0


def _force_exdev(monkeypatch):
    """Make the hardlink path unavailable so the copy fallback is exercised."""
    import errno as _errno

    def fake_link(a, b):
        raise OSError(_errno.EXDEV, "Invalid cross-device link")
    monkeypatch.setattr(refile_mod.os, "link", fake_link)


def test_cross_device_move_copies_verifies_then_removes(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    _force_exdev(monkeypatch)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert stats["totals"]["moved"] == 1
    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW").exists()
    assert not (src / "A.ARW").exists()


# ---------------------------------------------------------------------------
# C1 — the move primitive must never overwrite
# ---------------------------------------------------------------------------

def test_move_primitive_refuses_an_existing_destination(tmp_path):
    src = tmp_path / "src.ARW"
    dst = tmp_path / "dst.ARW"
    src.write_bytes(b"source-bytes")
    dst.write_bytes(b"destination-bytes")

    with pytest.raises(DestinationExists):
        _move_file(src, dst)

    assert src.read_bytes() == b"source-bytes"
    assert dst.read_bytes() == b"destination-bytes"


def test_move_primitive_refuses_an_existing_destination_cross_device(tmp_path, monkeypatch):
    _force_exdev(monkeypatch)
    src = tmp_path / "src.ARW"
    dst = tmp_path / "dst.ARW"
    src.write_bytes(b"source-bytes")
    dst.write_bytes(b"destination-bytes")

    with pytest.raises(DestinationExists):
        _move_file(src, dst)

    assert src.read_bytes() == b"source-bytes"
    assert dst.read_bytes() == b"destination-bytes"


def test_move_primitive_refuses_a_dangling_symlink_destination(tmp_path):
    src = tmp_path / "src.ARW"
    src.write_bytes(b"source-bytes")
    dst = tmp_path / "dst.ARW"
    dst.symlink_to(tmp_path / "nowhere.ARW")

    with pytest.raises(DestinationExists):
        _move_file(src, dst)
    assert src.exists()


def test_a_file_appearing_at_the_destination_mid_run_is_not_overwritten(
        tmp_path, tmp_db_path, monkeypatch):
    """The nightly ingest can create the destination between our check and our move."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"ours")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    dest = photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW"

    real_move = refile_mod._move_file

    def racing_move(s, d):
        # Simulate ingest landing a DIFFERENT file at d after our lexists check.
        if not os.path.lexists(str(d)):
            _touch(d, b"theirs-from-the-nightly-ingest")
        return real_move(s, d)
    monkeypatch.setattr(refile_mod, "_move_file", racing_move)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert dest.read_bytes().endswith(b"theirs-from-the-nightly-ingest")
    assert (src / "DSC01.ARW").exists(), "our original must survive"
    assert stats["totals"]["conflict"] == 1
    assert stats["totals"]["moved"] == 0


def test_a_race_with_identical_bytes_becomes_duplicate_left(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"same")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    real_move = refile_mod._move_file

    def racing_move(s, d):
        if not os.path.lexists(str(d)):
            _touch(d, b"same")
        return real_move(s, d)
    monkeypatch.setattr(refile_mod, "_move_file", racing_move)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert stats["totals"]["duplicate_left"] == 1
    assert stats["totals"]["moved"] == 0
    assert (src / "DSC01.ARW").exists()


def test_source_symlinks_are_never_moved(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "REAL.ARW", b"real")
    (src / "LINK.ARW").symlink_to(src / "REAL.ARW")
    _patch_exif_by_name(monkeypatch, {"REAL.ARW": "ILCE-7RM6", "LINK.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "LINK.ARW").is_symlink()
    assert stats["totals"]["skipped_symlink"] == 1
    assert stats["totals"]["moved"] == 1


# ---------------------------------------------------------------------------
# I3 — a failed cross-device copy must leave no partial destination
# ---------------------------------------------------------------------------

def test_failed_copy_leaves_no_partial_destination(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a" * 1000)
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    _force_exdev(monkeypatch)

    def boom(fsrc, fdst, length=0):
        fdst.write(b"partial")
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(refile_mod.shutil, "copyfileobj", boom)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    dest = photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW"
    assert not os.path.lexists(str(dest)), "no truncated file may be left behind"
    assert (src / "A.ARW").read_bytes().endswith(b"a" * 1000)
    assert stats["totals"]["errors"] == 1
    assert stats["totals"]["moved"] == 0


# ---------------------------------------------------------------------------
# C2 — the DB gate must not depend on the `folder` column
# ---------------------------------------------------------------------------

def test_indexed_photo_with_a_stale_folder_column_is_still_skipped(
        tmp_path, tmp_db_path, monkeypatch):
    """`relocate-into-year-dirs` / `db.remap_paths` rewrite filepath, not folder."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                     filename="DSC01.JPG", file_hash="h1",
                     folder="1999/some-old-path")

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.JPG").exists()
    assert stats["totals"]["skipped_indexed"] == 1
    assert stats["totals"]["moved"] == 0


def test_indexed_photo_with_a_null_folder_column_is_still_skipped(
        tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                     filename="DSC01.JPG", file_hash="h1", folder=None)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.JPG").exists()
    assert stats["totals"]["skipped_indexed"] == 1


# ---------------------------------------------------------------------------
# I5 — concurrency lock
# ---------------------------------------------------------------------------

def test_apply_refuses_while_an_ingest_sweep_holds_the_lock(tmp_path, tmp_db_path, monkeypatch):
    from photosearch.ingest import IngestAlreadyRunning, _sweep_lock

    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    with _sweep_lock(tmp_db_path):
        with pytest.raises(IngestAlreadyRunning):
            refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))
    assert (src / "A.ARW").exists()


def test_dry_run_does_not_need_the_lock(tmp_path, tmp_db_path, monkeypatch):
    from photosearch.ingest import _sweep_lock

    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    with _sweep_lock(tmp_db_path):
        stats = refile_unknown_camera(str(photos), tmp_db_path)
    assert stats["totals"]["would_move"] == 1


# ---------------------------------------------------------------------------
# I6 — intent-then-confirm audit
# ---------------------------------------------------------------------------

def test_audit_records_intent_before_the_move(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = tmp_path / "audit.csv"

    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=str(audit))

    actions = [r["action"] for r in csv.DictReader(open(audit))]
    assert actions == ["intent", "moved"]


def test_undo_honours_an_intent_whose_move_completed_before_the_crash(
        tmp_path, tmp_db_path, monkeypatch):
    """Killed between the move and the confirm row: the file DID move, so undo it."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = tmp_path / "audit.csv"

    refile_unknown_camera(str(photos), tmp_db_path, apply=True, audit_path=str(audit))
    # Rewrite the audit as the crash would have left it: intent, no confirm.
    rows = [r for r in csv.DictReader(open(audit)) if r["action"] == "intent"]
    with open(audit, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=refile_mod.AUDIT_FIELDS)
        w.writeheader()
        w.writerows(rows)

    stats = undo_refile(str(audit), tmp_db_path, apply=True)

    assert stats["restored"] == 1
    assert (src / "A.ARW").exists()


def test_undo_treats_an_intent_whose_move_never_started_as_a_no_op(
        tmp_path, tmp_db_path, monkeypatch):
    """Killed before the move: the source is still there, so there is nothing to undo."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera" / "A.ARW"
    _touch(src, b"a")
    _db(tmp_db_path, photos)
    audit = tmp_path / "audit.csv"
    dest = photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW"
    with open(audit, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(refile_mod.AUDIT_FIELDS)
        w.writerow(["intent", str(src), str(dest), "ILCE-7RM6",
                    str(src.stat().st_size), "", ""])

    stats = undo_refile(str(audit), tmp_db_path, apply=True)

    assert stats["restored"] == 0
    assert stats["no_op"] == 1
    assert src.exists() and not os.path.lexists(str(dest))


# ---------------------------------------------------------------------------
# M7 / M8 — healing
# ---------------------------------------------------------------------------

def test_dry_run_heal_reports_without_hashing(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    (photos / "2026" / "2026-09-19_unknown-camera").mkdir(parents=True)
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG", b"jpeg")
    from photosearch.index import file_hash as real_hash
    h = real_hash(str(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG"))
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                     filename="DSC01.JPG", file_hash=h)
    _patch_exif_by_name(monkeypatch, {})

    def no_hashing(_p):
        raise AssertionError("a dry run must not hash whole files")
    monkeypatch.setattr(refile_mod, "file_hash", no_hashing)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert stats["totals"]["would_heal"] == 1
    assert stats["totals"]["healed"] == 0


def test_heal_refuses_a_row_with_no_stored_hash(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    (photos / "2026" / "2026-09-19_unknown-camera").mkdir(parents=True)
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.JPG", b"jpeg")
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                           filename="DSC01.JPG")
    _patch_exif_by_name(monkeypatch, {})

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  include_indexed=True)

    assert stats["totals"]["healed"] == 0
    assert stats["totals"]["heal_unverifiable"] == 1
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT filepath FROM photos WHERE id = ?", (pid,)).fetchone()
        assert row["filepath"] == "2026/2026-09-19_unknown-camera/DSC01.JPG"


def test_stale_raw_filepath_is_reported_not_repointed(tmp_path, tmp_db_path, monkeypatch):
    """A raw_filepath carries no hash, so an automatic repoint could pick another body's RAW."""
    photos = _root(tmp_path)
    (photos / "2026" / "2026-09-19_unknown-camera").mkdir(parents=True)
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW", b"someone-elses-raw")
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        pid = db.add_photo(filepath="2026/2026-09-19_ILCE-7RM6/DSC01.JPG",
                           filename="DSC01.JPG", file_hash="h1",
                           raw_filepath="2026/2026-09-19_unknown-camera/DSC01.ARW")
    _patch_exif_by_name(monkeypatch, {})

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"),
                                  include_indexed=True)

    assert stats["totals"]["healed"] == 0
    assert stats["totals"]["heal_unverifiable"] == 1
    with PhotoDB(tmp_db_path) as db:
        row = db.conn.execute("SELECT raw_filepath FROM photos WHERE id = ?",
                              (pid,)).fetchone()
        assert row["raw_filepath"] == "2026/2026-09-19_unknown-camera/DSC01.ARW"


# ---------------------------------------------------------------------------
# M9 / M10 / M11 — misc hardening
# ---------------------------------------------------------------------------

def test_dangling_symlink_at_the_destination_is_a_conflict_not_free_space(
        tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"ours")
    dest_dir = photos / "2026" / "2026-09-19_ILCE-7RM6"
    dest_dir.mkdir(parents=True)
    (dest_dir / "DSC01.ARW").symlink_to(dest_dir / "gone.ARW")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.ARW").exists()
    assert (dest_dir / "DSC01.ARW").is_symlink()
    assert stats["totals"]["moved"] == 0
    assert stats["totals"]["conflict"] == 1


def test_dry_run_never_creates_a_database(tmp_path, monkeypatch):
    photos = _root(tmp_path)
    _touch(photos / "2026" / "2026-09-19_unknown-camera" / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    missing = str(tmp_path / "definitely-not-here.db")

    with pytest.raises(FileNotFoundError, match="database"):
        refile_unknown_camera(str(photos), missing)
    assert not os.path.exists(missing)


def test_undo_dry_run_never_creates_a_database(tmp_path):
    audit = tmp_path / "audit.csv"
    with open(audit, "w", newline="") as fh:
        csv.writer(fh).writerow(refile_mod.AUDIT_FIELDS)
    missing = str(tmp_path / "definitely-not-here.db")

    with pytest.raises(FileNotFoundError, match="database"):
        undo_refile(str(audit), missing)
    assert not os.path.exists(missing)


# ---------------------------------------------------------------------------
# C2 round 2 — the link lookup must not depend on how a path is SPELLED
# ---------------------------------------------------------------------------

def _refile(photos, db, **kw):
    return refile_unknown_camera(str(photos), db, **kw)


@pytest.mark.parametrize("bad", [
    "/abs/2026/2026-09-19_unknown-camera/DSC01.JPG",
    "./2026/2026-09-19_unknown-camera/DSC01.JPG",
    "2026\\2026-09-19_unknown-camera\\DSC01.JPG",
    "2026//2026-09-19_unknown-camera/DSC01.JPG",
])
def test_preflight_refuses_a_non_canonical_filepath(tmp_path, tmp_db_path, monkeypatch, bad):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath=bad, filename="DSC01.JPG", file_hash="h1")

    with pytest.raises(ValueError, match="non-canonical"):
        _refile(photos, tmp_db_path)
    with pytest.raises(ValueError, match="non-canonical"):
        _refile(photos, tmp_db_path, apply=True, audit_path=str(tmp_path / "a.csv"))
    assert (src / "DSC01.JPG").exists()


def test_preflight_refuses_a_non_canonical_raw_filepath(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    _touch(photos / "2026" / "2026-09-19_unknown-camera" / "DSC01.ARW", b"raw")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_ILCE-7RM6/DSC01.JPG",
                     filename="DSC01.JPG",
                     raw_filepath="/abs/2026/2026-09-19_unknown-camera/DSC01.ARW")

    with pytest.raises(ValueError, match="non-canonical"):
        _refile(photos, tmp_db_path)


def test_preflight_names_the_offending_rows(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    _touch(photos / "2026" / "2026-09-19_unknown-camera" / "DSC01.JPG", b"jpeg")
    _patch_exif_by_name(monkeypatch, {"DSC01.JPG": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        for i in range(3):
            db.add_photo(filepath=f"./2026/2026-09-19_unknown-camera/X{i}.JPG",
                         filename=f"X{i}.JPG")

    with pytest.raises(ValueError) as exc:
        refile_unknown_camera(str(photos), tmp_db_path)
    msg = str(exc.value)
    assert "3 photos row(s)" in msg
    assert "X0.JPG" in msg and "normalise" in msg.lower()


def test_preflight_passes_on_a_canonical_library(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    _touch(photos / "2026" / "2026-09-19_unknown-camera" / "DSC01.ARW", b"raw")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="2026/2026-09-19_ILCE-7RM6/DSC01.JPG",
                     filename="DSC01.JPG", file_hash="h1",
                     raw_filepath="2026/2026-09-19_ILCE-7RM6/DSC01.ARW")

    stats = refile_unknown_camera(str(photos), tmp_db_path)
    assert stats["totals"]["would_move"] == 1


def test_a_row_appearing_after_the_scan_is_still_not_moved(tmp_path, tmp_db_path, monkeypatch):
    """Belt-and-braces: the gate re-queries by exact filepath just before the move."""
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.JPG", b"jpeg")
    _db(tmp_db_path, photos)

    inserted = []

    def exif_that_indexes(filepath):
        # Runs per file, after the up-front link scan and before the gate.
        if not inserted:
            inserted.append(True)
            with PhotoDB(tmp_db_path) as d:
                d.add_photo(filepath="2026/2026-09-19_unknown-camera/DSC01.JPG",
                            filename="DSC01.JPG", file_hash="h1")
        return {"camera_model": "ILCE-7RM6"}
    monkeypatch.setattr(refile_mod, "extract_exif", exif_that_indexes)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert (src / "DSC01.JPG").exists()
    assert stats["totals"]["skipped_indexed"] == 1
    assert stats["totals"]["moved"] == 0


# ---------------------------------------------------------------------------
# M7 round 2 — a dry run must not hash a collision either
# ---------------------------------------------------------------------------

def test_dry_run_reports_a_collision_without_hashing(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "DSC01.ARW", b"ours")
    _touch(photos / "2026" / "2026-09-19_ILCE-7RM6" / "DSC01.ARW", b"theirs-and-longer")
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    def no_hashing(_p):
        raise AssertionError("a dry run must not hash whole files")
    monkeypatch.setattr(refile_mod, "file_hash", no_hashing)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert stats["totals"]["would_collide"] == 1
    assert stats["totals"]["would_move"] == 0
    assert stats["totals"]["conflict"] == 0
    assert any("DSC01.ARW" in n for n in stats["folders"][0]["notes"])


# ---------------------------------------------------------------------------
# C1(b) — an interrupted hardlink leaves one inode under two names
# ---------------------------------------------------------------------------

def test_interrupted_hardlink_is_completed_without_hashing(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera" / "DSC01.ARW"
    _touch(src, b"raw")
    dest_dir = photos / "2026" / "2026-09-19_ILCE-7RM6"
    dest_dir.mkdir(parents=True)
    os.link(str(src), str(dest_dir / "DSC01.ARW"))  # killed between link and unlink
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    def no_hashing(_p):
        raise AssertionError("an inode identity check needs no hashing")
    monkeypatch.setattr(refile_mod, "file_hash", no_hashing)

    stats = refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                                  audit_path=str(tmp_path / "audit.csv"))

    assert not os.path.lexists(str(src))
    assert (dest_dir / "DSC01.ARW").read_bytes().endswith(b"raw")
    assert stats["totals"]["interrupted_link"] == 1
    assert stats["totals"]["moved"] == 1
    assert stats["totals"]["duplicate_left"] == 0
    assert not (photos / "2026" / "2026-09-19_unknown-camera").exists()
    rows = list(csv.DictReader(open(tmp_path / "audit.csv")))
    assert any("completed_interrupted_link" in r["reason"] for r in rows
               if r["action"] == "moved")


def test_interrupted_hardlink_is_only_reported_in_a_dry_run(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera" / "DSC01.ARW"
    _touch(src, b"raw")
    dest_dir = photos / "2026" / "2026-09-19_ILCE-7RM6"
    dest_dir.mkdir(parents=True)
    os.link(str(src), str(dest_dir / "DSC01.ARW"))
    _patch_exif_by_name(monkeypatch, {"DSC01.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    stats = refile_unknown_camera(str(photos), tmp_db_path)

    assert src.exists()
    assert stats["totals"]["interrupted_link"] == 1
    assert stats["totals"]["would_collide"] == 0


def test_audit_inside_the_photo_root_is_refused(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    _touch(photos / "2026" / "2026-09-19_unknown-camera" / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    with pytest.raises(ValueError, match="photo root"):
        refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                              audit_path=str(photos / "audit.csv"))


def test_move_preserves_mtime(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera" / "A.ARW"
    _touch(src, b"a")
    os.utime(src, (1_600_000_000, 1_600_000_000))
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    refile_unknown_camera(str(photos), tmp_db_path, apply=True,
                          audit_path=str(tmp_path / "audit.csv"))

    dest = photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW"
    assert int(dest.stat().st_mtime) == 1_600_000_000


def test_progress_callback_receives_folder_and_file_events(tmp_path, tmp_db_path, monkeypatch):
    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)

    seen = []
    refile_unknown_camera(str(photos), tmp_db_path, on_progress=seen.append)

    assert any(e["event"] == "folder" for e in seen)
    assert any(e["event"] == "file" for e in seen)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_dry_run_then_apply_then_undo(tmp_path, tmp_db_path, monkeypatch):
    from click.testing import CliRunner
    from cli import cli

    photos = _root(tmp_path)
    src = photos / "2026" / "2026-09-19_unknown-camera"
    _touch(src / "A.ARW", b"a")
    _patch_exif_by_name(monkeypatch, {"A.ARW": "ILCE-7RM6"})
    _db(tmp_db_path, photos)
    audit = str(tmp_path / "audit.csv")
    runner = CliRunner()

    dry = runner.invoke(cli, ["refile-unknown-camera", "--db", tmp_db_path,
                              "--photo-root", str(photos)])
    assert dry.exit_code == 0, dry.output
    assert "DRY RUN" in dry.output
    assert "ILCE-7RM6 1" in dry.output
    assert (src / "A.ARW").exists()

    refuse = runner.invoke(cli, ["refile-unknown-camera", "--db", tmp_db_path,
                                 "--photo-root", str(photos), "--apply"])
    assert refuse.exit_code != 0
    assert "audit" in refuse.output

    done = runner.invoke(cli, ["refile-unknown-camera", "--db", tmp_db_path,
                               "--photo-root", str(photos), "--apply",
                               "--audit", audit])
    assert done.exit_code == 0, done.output
    assert (photos / "2026" / "2026-09-19_ILCE-7RM6" / "A.ARW").exists()

    back = runner.invoke(cli, ["refile-unknown-camera", "--db", tmp_db_path,
                               "--undo", audit, "--apply"])
    assert back.exit_code == 0, back.output
    assert "restored=1" in back.output
    assert (src / "A.ARW").exists()

"""Pixel readers skip video rows (schema v35, docs/plans/video-support.md step 1).

A video is a `photos` row with media_type='video'. Everything that opens a
row's file expecting an image must leave it alone — most importantly
`purge-nonimage-photos`, which would otherwise DELETE every video (no CLIP
embedding, and an old MOV fails is_real_image) and cascade its faces and
collection links with it.
"""

import os

import pytest
from click.testing import CliRunner

from photosearch.db import PhotoDB

_DAY = "2091-10-17"
_DIR = f"2091/{_DAY}_phone-matt"
# An old QuickTime MOV: no leading `ftyp`, so is_real_image says no.
_OLD_MOV = b"\x00\x00\x00\x08wide\x00\x00\x10\x00mdat" + b"\x00" * 64
_ZIP_JPG = b"PK\x03\x04" + b"\x00" * 64
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "photos"
    (root / _DIR).mkdir(parents=True)
    dbp = str(tmp_path / "photo_index.db")
    db = PhotoDB(dbp)
    db.set_photo_root(str(root))

    def add(name, data, **kw):
        (root / _DIR / name).write_bytes(data)
        return db.add_photo(filepath=f"{_DIR}/{name}", filename=name, **kw)

    yield db, dbp, root, add
    db.close()


def _emb(i=0):
    v = [0.0] * 512
    v[i] = 1.0
    return v


# --- purge-nonimage-photos: the data-loss risk ---------------------------------

@pytest.mark.parametrize("scan_all", [False, True])
def test_purge_nonimage_never_deletes_a_video_row(lib, tmp_path, scan_all):
    from cli import cli
    db, dbp, root, add = lib
    vid = add("OLD.MOV", _OLD_MOV, media_type="video")
    bogus = add("IMG_1(1).JPG", _ZIP_JPG)
    db.conn.execute("INSERT INTO video_meta (photo_id) VALUES (?)", (vid,))
    db.conn.commit()
    db.close()

    args = ["purge-nonimage-photos", "--db", dbp, "--apply",
            "--audit", str(tmp_path / "audit.csv")]
    if scan_all:
        args.append("--all")
    res = CliRunner().invoke(cli, args, env={"PHOTO_ROOT": str(root)})
    assert res.exit_code == 0, res.output
    assert "1 video rows never considered" in res.output

    with PhotoDB(dbp) as again:
        ids = {r[0] for r in again.conn.execute("SELECT id FROM photos")}
        assert vid in ids, "purge-nonimage-photos deleted a video row"
        assert bogus not in ids, "the real non-image row should still be purged"
        assert again.conn.execute(
            "SELECT 1 FROM video_meta WHERE photo_id = ?", (vid,)).fetchone()


# --- stacking ------------------------------------------------------------------

def test_detect_stacks_ignores_videos(lib):
    from photosearch.stacking import detect_stacks
    db, _dbp, _root, add = lib
    ids = []
    for i, (name, data, mt) in enumerate([("A.JPG", _JPEG, None), ("B.MP4", _OLD_MOV, "video"),
                                          ("C.JPG", _JPEG, None), ("D.MP4", _OLD_MOV, "video")]):
        pid = add(name, data, media_type=mt, date_taken=f"{_DAY} 10:00:0{i}")
        db.add_clip_embedding(pid, _emb(0))
        ids.append(pid)
    stacks = detect_stacks(db, photo_ids=ids)
    members = {m for s in stacks for m in s}
    assert members == {ids[0], ids[2]}


def test_incremental_stacking_never_considers_videos(lib):
    from photosearch.stacking import run_incremental_stacking
    db, _dbp, _root, add = lib
    a = add("A.JPG", _JPEG, date_taken=f"{_DAY} 10:00:00")
    v = add("V.MP4", _OLD_MOV, media_type="video", date_taken=f"{_DAY} 10:00:01")
    for pid in (a, v):
        db.add_clip_embedding(pid, _emb(0))
    run_incremental_stacking(db)
    seen = {r[0] for r in db.conn.execute("SELECT photo_id FROM stacking_seen")}
    assert v not in seen


# --- sharpness backfill ----------------------------------------------------------

def test_sharpness_candidates_skip_videos(lib):
    from photosearch.sharpness_backfill import candidate_ids
    db, _dbp, _root, add = lib
    s = add("A.JPG", _JPEG)
    v = add("V.MP4", _OLD_MOV, media_type="video")
    assert candidate_ids(db, require_faces=False) == [s]
    assert candidate_ids(db, photo_ids=[s, v], require_faces=False) == [s]


# --- rank_measure / team review: face rows on a video ---------------------------

def _face(db, pid):
    return db.conn.execute(
        "INSERT INTO faces (photo_id, bbox_top, bbox_right, bbox_bottom, bbox_left, "
        "det_score) VALUES (?, 0, 200, 200, 0, 0.9)", (pid,)).lastrowid


def test_rank_measure_face_rows_skip_videos(lib):
    from photosearch.rank_measure import _face_rows
    db, _dbp, _root, add = lib
    s = add("A.JPG", _JPEG, date_taken=f"{_DAY} 10:00:00")
    v = add("V.MP4", _OLD_MOV, media_type="video", date_taken=f"{_DAY} 10:00:01")
    fs, _fv = _face(db, s), _face(db, v)
    assert {r["face_id"] for r in _face_rows(db, _DAY, None)} == {fs}
    assert {r["face_id"] for r in _face_rows(db, _DAY, [s, v])} == {fs}


def test_team_review_faces_skip_videos(lib):
    from photosearch.face_review import TEAM_FACES_SQL
    db, _dbp, _root, add = lib
    s = add("A.JPG", _JPEG, date_taken=f"{_DAY} 10:00:00")
    v = add("V.MP4", _OLD_MOV, media_type="video", date_taken=f"{_DAY} 10:00:01")
    fs, _fv = _face(db, s), _face(db, v)
    rows = db.conn.execute(TEAM_FACES_SQL, (_DAY, 0.5, 60)).fetchall()
    assert {r["id"] for r in rows} == {fs}


# --- one-photo exports -------------------------------------------------------------

def test_upscale_refuses_a_video(lib):
    from photosearch.upscale import upscale_photo
    db, _dbp, _root, add = lib
    v = add("V.MP4", _OLD_MOV, media_type="video")
    with pytest.raises(ValueError, match="video"):
        upscale_photo(db, v)


def test_split_export_refuses_a_video(lib):
    from photosearch.split_export import SplitError, export_panels
    db, _dbp, _root, add = lib
    v = add("V.MP4", _OLD_MOV, media_type="video")
    with pytest.raises(SplitError, match="video"):
        export_panels(db, v, pw=13, ph=19, cols=2, rows=1)


def test_book_export_fetcher_returns_nothing_for_a_video(lib):
    from photosearch import web
    db, _dbp, _root, add = lib
    v = add("V.MP4", _OLD_MOV, media_type="video")
    assert web._full_image_fetcher(db)(v) is None


def test_gphotos_records_skip_videos():
    from photosearch.shutterfly_export import build_gphotos_records
    rows = {1: {"filepath": "a.jpg", "filename": "a.jpg", "description": None},
            2: {"filepath": "v.mp4", "filename": "v.mp4", "description": None,
                "media_type": "video"}}
    recs = build_gphotos_records([1, 2], rows, lambda p: "/x/" + p)
    assert [r["filename"] for r in recs] == ["a.jpg"]


def test_gphotos_upload_refuses_video_bytes(tmp_path):
    from photosearch.google_photos import _upload_raw_bytes
    f = tmp_path / "clip.MOV"
    f.write_bytes(_OLD_MOV)
    with pytest.raises(ValueError, match="video"):
        _upload_raw_bytes("token", str(f))


# --- maintenance / backfills ---------------------------------------------------------

def test_colors_stage_skips_videos(lib):
    from photosearch.maintenance import _stage_colors
    db, _dbp, _root, add = lib
    add("A.JPG", _JPEG)
    add("V.MP4", _OLD_MOV, media_type="video")
    out = _stage_colors(db, False, lambda e: None, lambda: None)
    assert out["would"] == 1


def test_dedup_plan_does_not_group_a_video_with_unrelated_rows(lib):
    """find-duplicate-photos only joins rows on a shared file_hash or a
    byte-identical face encoding; a hashless video is never a candidate."""
    from photosearch.maintenance import find_duplicate_photo_plan
    db, _dbp, _root, add = lib
    add("A.JPG", _JPEG, file_hash="h1")
    add("V.MP4", _OLD_MOV, media_type="video")
    add("W.MP4", _OLD_MOV, media_type="video")
    assert find_duplicate_photo_plan(db)["n_redundant"] == 0


def test_cleanup_orphans_leaves_video_rows(lib):
    db, _dbp, _root, add = lib
    v = add("V.MP4", _OLD_MOV, media_type="video")
    db.cleanup_vec_orphans(dry_run=False)
    assert db.get_photo(v) is not None

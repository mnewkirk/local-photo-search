"""photosearch/unwrap_zip_videos.py — extract ZIP-wrapped clips in place (NAS).

The zip is never deleted (it survives as <name>.zip), filepath never changes,
the audit is the undo, and a dry run writes nothing anywhere.
"""

import csv
import hashlib
import os
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from photosearch import unwrap_zip_videos as UZ
from photosearch import zip_video as ZV
from photosearch.db import PhotoDB

MTIME = 1_463_300_000  # 2016-05-15


def _zip(path: Path, members, mtime=MTIME, compress=zipfile.ZIP_STORED):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=compress) as z:
        for name, data in members:
            z.writestr(name, data)
    os.chmod(path, 0o644)
    os.utime(path, (MTIME, mtime))
    return path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "photos"
    (root / "2016").mkdir(parents=True)
    dbp = str(tmp_path / "photo_index.db")
    with PhotoDB(dbp) as db:
        db.set_photo_root(str(root))
    return dbp, root, tmp_path


def _run(dbp, root, tmp_path, **kw):
    kw.setdefault("lower_priority", False)
    kw.setdefault("mbps", 0)
    if kw.get("apply"):
        kw.setdefault("audit_path", str(tmp_path / "audit.csv"))
    return UZ.unwrap_zip_videos(dbp, photo_root=str(root), **kw)


def _video_row(dbp, rel, *, file_hash="zipsha", probe_error="zip_wrapped",
               scan_version=None, scan_error=None):
    with PhotoDB(dbp) as db:
        pid = db.add_photo(filepath=rel, filename=os.path.basename(rel),
                           media_type="video", file_hash=file_hash)
        db.conn.execute(
            "INSERT INTO video_meta (photo_id, probe_version, probe_error, probe_json, "
            "scan_version, scan_error) VALUES (?, 1, ?, '{}', ?, ?)",
            (pid, probe_error, scan_version, scan_error))
        db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                        "VALUES (?, 'video-probe', 1)", (pid,))
        db.conn.commit()
    return pid


# --- shared helper keeps video_scan's behaviour -----------------------------------

def test_video_scan_uses_the_shared_rules():
    from photosearch import video_scan as VS
    assert VS.ZipRefused is ZV.ZipRefused
    assert VS._is_junk_member is ZV.is_junk_member


# --- forward ------------------------------------------------------------------------

def test_single_member_unwrapped_in_place(lib):
    dbp, root, tmp = lib
    payload = os.urandom(50_000)
    p = _zip(root / "2016" / "2016-05-15" / "IMG_1585.MOV", [("IMG_1585.MOV", payload)])
    st = os.stat(p)
    res = _run(dbp, root, tmp, apply=True)
    assert res["zip_found"] == 1 and res["unwrapped"] == 1
    assert p.read_bytes() == payload
    z = Path(str(p) + ".zip")
    assert ZV.has_zip_magic(str(z))                        # zip kept
    assert os.stat(p).st_mtime_ns == st.st_mtime_ns         # mtime preserved
    assert os.stat(p).st_mode & 0o777 == 0o644              # mode preserved
    assert res["per_year"]["2016"]["unwrapped"] == 1
    assert not [f for f in os.listdir(p.parent) if f.endswith(".tmp")]
    rows = list(csv.DictReader(open(tmp / "audit.csv")))
    assert [r["event"] for r in rows] == ["intent", "done"]
    assert rows[1]["inner_sha256"] == _sha(payload)
    assert int(rows[1]["inner_size"]) == len(payload)


def test_bundle_with_jpg_and_mac_litter_takes_the_video(lib):
    dbp, root, tmp = lib
    payload = os.urandom(20_000)
    p = _zip(root / "2016" / "d" / "IMG_1.MOV", [
        ("__MACOSX/._IMG_1.MOV", b"fork"), ("._IMG_1.MOV", b"ad"),
        ("IMG_1.JPG", os.urandom(500)), ("IMG_1.MOV", payload)])
    _run(dbp, root, tmp, apply=True)
    assert p.read_bytes() == payload


@pytest.mark.parametrize("members,key", [
    ([("a.mov", b"1"), ("b.mp4", b"2")], "video_count"),
    ([("../../evil.mov", os.urandom(100))], "unsafe_name"),
    ([("only.jpg", b"1")], "video_count"),
])
def test_refusals_leave_the_file_untouched(lib, members, key):
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "W.MOV", members)
    before = p.read_bytes()
    res = _run(dbp, root, tmp, apply=True)
    assert res["refused"] == {key: 1} and res["unwrapped"] == 0
    assert p.read_bytes() == before
    assert not os.path.exists(str(p) + ".zip")
    assert res["refusal_samples"][0]["path"] == "2016/d/W.MOV"


def test_oversize_member_is_refused(lib):
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "B.MOV", [("v.mov", b"\x00" * 300_000)],
             compress=zipfile.ZIP_DEFLATED)
    res = _run(dbp, root, tmp, apply=True)
    assert res["refused"] in ({"oversize": 1}, {"ratio": 1})
    assert ZV.has_zip_magic(str(p)) and not os.path.exists(str(p) + ".zip")


def test_eexist_on_zip_name_refuses_without_overwrite(lib):
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "IMG_2.MOV", [("IMG_2.MOV", os.urandom(1000))])
    other = Path(str(p) + ".zip")
    other.write_bytes(b"someone else's file")
    before = p.read_bytes()
    res = _run(dbp, root, tmp, apply=True)
    assert res["refused"] == {"zip_name_taken": 1}
    assert other.read_bytes() == b"someone else's file"
    assert p.read_bytes() == before


def test_rerun_is_idempotent_already_unwrapped(lib):
    dbp, root, tmp = lib
    payload = os.urandom(1000)
    p = _zip(root / "2016" / "d" / "IMG_3.MOV", [("IMG_3.MOV", payload)])
    _run(dbp, root, tmp, apply=True)
    res = _run(dbp, root, tmp, apply=True)
    assert res["zip_found"] == 0 and res["already_unwrapped"] == 1
    assert p.read_bytes() == payload


def test_interrupted_link_resumes(lib):
    """Killed between os.link and os.replace: the .zip is the same inode."""
    dbp, root, tmp = lib
    payload = os.urandom(1000)
    p = _zip(root / "2016" / "d" / "IMG_4.MOV", [("IMG_4.MOV", payload)])
    os.link(p, str(p) + ".zip")
    res = _run(dbp, root, tmp, apply=True)
    assert res["unwrapped"] == 1 and p.read_bytes() == payload
    assert ZV.has_zip_magic(str(p) + ".zip")


def test_db_row_reset_for_reprobe(lib):
    dbp, root, tmp = lib
    payload = os.urandom(3000)
    _zip(root / "2016" / "d" / "IMG_5.MOV", [("IMG_5.MOV", payload)])
    _zip(root / "2016" / "d" / "IMG_6.MOV", [("IMG_6.MOV", os.urandom(3000))])
    pid = _video_row(dbp, "2016/d/IMG_5.MOV", scan_version=3)
    pid6 = _video_row(dbp, "2016/d/IMG_6.MOV", scan_version=3, scan_error="zip: refused")
    res = _run(dbp, root, tmp, apply=True)
    assert res["rows_reset"] == 2 and res["scan_cleared"] == 1
    conn = sqlite3.connect(dbp)
    fh, fp = conn.execute("SELECT file_hash, filepath FROM photos WHERE id=?", (pid,)).fetchone()
    assert fh == _sha(payload) and fp == "2016/d/IMG_5.MOV"
    vm = conn.execute("SELECT probe_version, probe_error, probe_json, scan_version "
                      "FROM video_meta WHERE photo_id=?", (pid,)).fetchone()
    assert vm == (None, None, None, 3)                       # scan kept
    assert conn.execute("SELECT scan_version, scan_error FROM video_meta WHERE photo_id=?",
                        (pid6,)).fetchone() == (None, None)   # wrapper-refusal cleared
    assert conn.execute("SELECT COUNT(*) FROM worker_processed WHERE pass_type='video-probe'"
                        ).fetchone()[0] == 0
    done = [r for r in csv.DictReader(open(tmp / "audit.csv")) if r["event"] == "done"]
    assert {r["old_file_hash"] for r in done} == {"zipsha"}


def test_dry_run_writes_nothing_and_opens_ro(lib, monkeypatch):
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "IMG_7.MOV", [("IMG_7.MOV", os.urandom(2000))])
    _video_row(dbp, "2016/d/IMG_7.MOV")
    before_db = open(dbp, "rb").read()
    before = p.read_bytes()

    def boom(*a, **k):
        raise AssertionError("PhotoDB opened read-write in a dry run")
    monkeypatch.setattr("photosearch.db.PhotoDB.__init__", boom)
    res = _run(dbp, root, tmp)
    assert res["would_unwrap"] == 1 and res["unwrapped"] == 0
    assert p.read_bytes() == before and not os.path.exists(str(p) + ".zip")
    assert open(dbp, "rb").read() == before_db


def test_apply_requires_audit_outside_root(lib):
    dbp, root, tmp = lib
    with pytest.raises(UZ.UnwrapRefused):
        UZ.unwrap_zip_videos(dbp, photo_root=str(root), apply=True, lower_priority=False)
    with pytest.raises(UZ.UnwrapRefused, match="inside the photo root"):
        UZ.unwrap_zip_videos(dbp, photo_root=str(root), apply=True,
                             audit_path=str(root / "a.csv"), lower_priority=False)


def test_refuses_while_ingest_sweep_runs(lib):
    from photosearch.ingest import IngestAlreadyRunning, _sweep_lock
    dbp, root, tmp = lib
    _zip(root / "2016" / "d" / "IMG_8.MOV", [("IMG_8.MOV", b"x" * 100)])
    with _sweep_lock(dbp):
        with pytest.raises(IngestAlreadyRunning):
            _run(dbp, root, tmp)
        with pytest.raises(IngestAlreadyRunning):
            _run(dbp, root, tmp, apply=True)


def test_scoping_limit_and_year(lib):
    dbp, root, tmp = lib
    for i in range(3):
        _zip(root / "2016" / "d" / f"A{i}.MOV", [(f"A{i}.MOV", os.urandom(100))])
    _zip(root / "2017" / "d" / "B.MOV", [("B.MOV", os.urandom(100))])
    res = _run(dbp, root, tmp, year="2017")
    assert res["zip_found"] == 1
    res = _run(dbp, root, tmp, limit=2)
    assert res["would_unwrap"] == 2 and res["beyond_limit"] == 2


def test_rate_limiter_paces():
    t = {"now": 0.0}
    slept = []
    rl = UZ.RateLimiter(1.0, clock=lambda: t["now"], sleep=lambda s: slept.append(s))
    rl.consume(2_000_000)
    assert slept == [pytest.approx(2.0)]


# --- undo -----------------------------------------------------------------------------

def test_undo_round_trip(lib):
    dbp, root, tmp = lib
    payload = os.urandom(4000)
    p = _zip(root / "2016" / "d" / "IMG_9.MOV", [("IMG_9.MOV", payload)])
    original = p.read_bytes()
    pid = _video_row(dbp, "2016/d/IMG_9.MOV")
    _run(dbp, root, tmp, apply=True)
    audit = str(tmp / "audit.csv")
    dry = UZ.undo_unwrap(audit, dbp, apply=False, mbps=0, lower_priority=False)
    assert dry["would_restore"] == 1 and p.read_bytes() == payload
    res = UZ.undo_unwrap(audit, dbp, apply=True, mbps=0, lower_priority=False)
    assert res["restored"] == 1 and res["rows_reverted"] == 1
    assert p.read_bytes() == original and not os.path.exists(str(p) + ".zip")
    conn = sqlite3.connect(dbp)
    assert conn.execute("SELECT file_hash FROM photos WHERE id=?", (pid,)).fetchone()[0] == "zipsha"
    again = UZ.undo_unwrap(audit, dbp, apply=True, mbps=0, lower_priority=False)
    assert again["already_restored"] == 1 and again["restored"] == 0


def test_undo_refuses_a_changed_file(lib):
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "IMG_10.MOV", [("IMG_10.MOV", os.urandom(4000))])
    _run(dbp, root, tmp, apply=True)
    p.write_bytes(os.urandom(4000))       # same size, different bytes
    res = UZ.undo_unwrap(str(tmp / "audit.csv"), dbp, apply=True, mbps=0,
                         lower_priority=False)
    assert res["refused"] == {"sha_mismatch": 1} and res["restored"] == 0
    assert os.path.exists(str(p) + ".zip")


# --- .zip is invisible to every walker ------------------------------------------------

def test_dot_zip_is_not_picked_up_by_any_walker(lib):
    from photosearch.index import SUPPORTED_EXTENSIONS, find_photos
    from photosearch.ingest import ALL_MEDIA_EXTENSIONS
    from photosearch.video_registry import iter_video_files, register_videos
    dbp, root, tmp = lib
    p = _zip(root / "2016" / "d" / "IMG_11.MOV", [("IMG_11.MOV", os.urandom(500))])
    _run(dbp, root, tmp, apply=True)
    assert ".zip" not in SUPPORTED_EXTENSIONS and ".zip" not in ALL_MEDIA_EXTENSIONS
    assert list(iter_video_files(str(root))) == [str(p)]
    assert not [f for f in find_photos(str(root)) if f.endswith(".zip")]
    res = register_videos(dbp, photo_root=str(root), apply=True, lower_priority=False)
    assert res["registered"] == 1
    conn = sqlite3.connect(dbp)
    assert [r[0] for r in conn.execute("SELECT filepath FROM photos")] == ["2016/d/IMG_11.MOV"]


# --- the real fixtures ----------------------------------------------------------------

FIXTURES = Path("/home/mattn/video-fixtures")


@pytest.mark.skipif(not (FIXTURES / "moov_missing_1.mov").exists(),
                    reason="local video fixtures not present")
def test_real_fixtures(lib):
    dbp, root, tmp = lib
    dst = root / "2016" / "2016-05-15"
    dst.mkdir(parents=True)
    for n in ("moov_missing_1.mov", "moov_missing_2.mov"):
        shutil.copy2(FIXTURES / n, dst / n)
    res = _run(dbp, root, tmp, apply=True)
    assert res["unwrapped"] == 2, res
    for n in ("moov_missing_1.mov", "moov_missing_2.mov"):
        head = (dst / n).read_bytes()[:12]
        assert head[4:8] in (b"ftyp", b"wide", b"mdat", b"moov", b"free")
        assert ZV.has_zip_magic(str(dst / (n + ".zip")))
    out = UZ.undo_unwrap(str(tmp / "audit.csv"), dbp, apply=True, mbps=0,
                         lower_priority=False)
    assert out["restored"] == 2
    assert (dst / "moov_missing_1.mov").read_bytes() == (FIXTURES / "moov_missing_1.mov").read_bytes()


# --- CLI ------------------------------------------------------------------------------

def test_cli_dry_run(lib, monkeypatch):
    from cli import cli
    monkeypatch.setattr("photosearch.sharpness_backfill.lower_priority", lambda: {})
    dbp, root, tmp = lib
    _zip(root / "2016" / "d" / "IMG_12.MOV", [("IMG_12.MOV", os.urandom(500))])
    r = CliRunner().invoke(cli, ["unwrap-zip-videos", "--db", dbp,
                                 "--photo-root", str(root)])
    assert r.exit_code == 0, r.output
    assert "would unwrap:       1" in r.output
    r = CliRunner().invoke(cli, ["unwrap-zip-videos", "--db", dbp,
                                 "--photo-root", str(root), "--apply"])
    assert r.exit_code != 0 and "--audit" in r.output

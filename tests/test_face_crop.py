"""Unit tests for photosearch.face_crop — the shared face-crop renderer used by
the /api/faces/crop endpoint and the `warm-face-crops` / `export-face-crops`
replica-warming CLIs (see docs/plans/local-replica-and-writes.md)."""

import io
import os
import subprocess
import sys
import tarfile
from unittest.mock import MagicMock

import pytest
from PIL import Image

# conftest leaves either the real torch or a MagicMock in sys.modules. The
# mock is enough for in-process tests, but not for one that shells out.
# (importlib.util.find_spec can't be used here — it raises on a mocked module.)
_TORCH_REAL = not isinstance(sys.modules.get("torch"), MagicMock)

from photosearch.face_crop import (
    crop_cache_path,
    face_crop_cache_dir,
    render_face_crops,
    write_crop_atomic,
)


def _make_jpg(path, w=1200, h=800):
    img = Image.new("RGB", (w, h), (80, 120, 200))
    # a red square where the "face" bbox will be, so the crop is non-uniform
    for x in range(400, 600):
        for y in range(200, 400):
            img.putpixel((x, y), (255, 0, 0))
    img.save(path, "JPEG")
    return path


def test_render_multiple_sizes_from_one_decode(tmp_path):
    p = _make_jpg(tmp_path / "src.jpg")
    bbox = (200, 600, 400, 400)  # top, right, bottom, left
    out = render_face_crops(str(p), bbox, 1200, 800, [120, 200])
    assert set(out) == {120, 200}
    for size, data in out.items():
        im = Image.open(io.BytesIO(data))
        assert im.size == (size, size)
        assert im.format == "JPEG"


def test_render_dedups_and_sorts_sizes(tmp_path):
    p = _make_jpg(tmp_path / "src.jpg")
    out = render_face_crops(str(p), (200, 600, 400, 400), 1200, 800, [200, 120, 120])
    assert sorted(out) == [120, 200]


def test_render_raises_without_bbox(tmp_path):
    p = _make_jpg(tmp_path / "src.jpg")
    try:
        render_face_crops(str(p), (None, None, None, None), 1200, 800, [120])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_render_handles_edge_bbox(tmp_path):
    """A bbox flush against the image border must still yield a square crop
    (the squaring math clamps to image bounds)."""
    p = _make_jpg(tmp_path / "src.jpg")
    out = render_face_crops(str(p), (0, 200, 200, 0), 1200, 800, [120])
    assert Image.open(io.BytesIO(out[120])).size == (120, 120)


def test_cache_path_helpers(tmp_path):
    db = tmp_path / "sub" / "photo_index.db"
    cdir = face_crop_cache_dir(str(db))
    assert cdir.endswith(os.path.join("sub", "thumbnails", "face_crops"))
    assert crop_cache_path(cdir, 42, 120).endswith("42_120.jpg")


def test_write_crop_atomic(tmp_path):
    target = tmp_path / "x_120.jpg"
    write_crop_atomic(str(target), b"hello")
    assert target.read_bytes() == b"hello"
    # no leftover temp files
    assert [p.name for p in tmp_path.iterdir()] == ["x_120.jpg"]


@pytest.mark.skipif(
    not _TORCH_REAL,
    reason="shells out to cli.py in a subprocess, which imports the real ML "
           "stack — conftest's mocks don't reach a child process",
)
def test_export_face_crops_tar_and_since(tmp_path):
    """export-face-crops streams a tar of the cache dir, honoring --since."""
    db = tmp_path / "photo_index.db"
    db.write_bytes(b"")  # only used to locate the cache dir
    cdir = face_crop_cache_dir(str(db))
    os.makedirs(cdir, exist_ok=True)
    for fid in (1, 2):
        write_crop_atomic(crop_cache_path(cdir, fid, 120), b"jpegbytes")

    env = {**os.environ, "PHOTOSEARCH_DB": str(db)}
    full = tmp_path / "full.tar"
    subprocess.run(
        [sys.executable, "cli.py", "export-face-crops", "--to", str(full)],
        check=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    with tarfile.open(full) as tf:
        assert sorted(tf.getnames()) == ["1_120.jpg", "2_120.jpg"]

    # --since in the future → empty tar
    import time
    inc = tmp_path / "inc.tar"
    subprocess.run(
        [sys.executable, "cli.py", "export-face-crops",
         "--since", str(time.time() + 1000), "--to", str(inc)],
        check=True, env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    with tarfile.open(inc) as tf:
        assert tf.getnames() == []


# =========================================================================
# warm_crops — the SCOPE predicate
# =========================================================================
#
# `warm_crops` was extracted out of the `warm-face-crops` click command so the
# CLI and the batch-advance `warm_crops` step share one copy. The extraction
# was NOT a pure move — the scope precedence was restructured and photo_ids
# chunking is new — and scope is the whole safety story here: getting it wrong
# warms every crop in the library (hours on the N100) or, worse, silently
# warms nothing. The rendering itself is covered above; these tests stub it
# out so no image is decoded.

import photosearch.face_crop as _fc


@pytest.fixture
def warm_db(tmp_path):
    """A DB whose photos live in two folders on two different days, with one
    face each, plus one face belonging to a registered person."""
    from photosearch.db import PhotoDB
    db = PhotoDB(str(tmp_path / "warm.db"))
    db.set_photo_root("/photos")
    ctx = {"a": [], "b": [], "faces": {}}
    for i in range(3):
        pid = db.add_photo(filepath=f"2090/2090-05-01_A/a{i}.jpg", filename=f"a{i}.jpg",
                           date_taken=f"2090-05-01T09:0{i}:00")
        ctx["a"].append(pid)
        ctx["faces"][pid] = db.add_face(pid, (10, 60, 60, 10), [0.0] * 512)
    for i in range(2):
        pid = db.add_photo(filepath=f"2090/2090-05-02_B/b{i}.jpg", filename=f"b{i}.jpg",
                           date_taken=f"2090-05-02T09:0{i}:00")
        ctx["b"].append(pid)
        ctx["faces"][pid] = db.add_face(pid, (10, 60, 60, 10), [0.0] * 512)
    ctx["person"] = db.add_person("Robin")
    db.assign_face_to_person(ctx["faces"][ctx["a"][0]], ctx["person"], "manual")
    db.conn.commit()
    ctx["db"] = db
    yield ctx
    db.close()


def _warmed(ctx, monkeypatch, **kw):
    """Run warm_crops with the renderer stubbed; return the face ids it acted
    on. Every source file is absent, so each task lands in `missing` — which
    is exactly the signal we want: it proves the row was IN SCOPE without
    decoding anything."""
    seen = []

    def fake_render(filepath, bbox, w, h, sizes):  # pragma: no cover - not reached
        raise AssertionError("no crop should be rendered in a scope test")

    monkeypatch.setattr(_fc, "render_face_crops", fake_render)
    rows = _fc._scoped_face_rows(
        ctx["db"],
        photo_ids=kw.get("photo_ids"), person_ids=kw.get("person_ids"),
        matched_only=kw.get("matched_only", False),
        date_from=kw.get("date_from"), date_to=kw.get("date_to"))
    seen[:] = [r["id"] for r in rows]
    summary = _fc.warm_crops(ctx["db"], **kw)
    # The summary's `found` must agree with the predicate under test.
    assert summary["found"] == len(seen)
    return set(seen)


def test_warm_crops_unscoped_is_every_face(warm_db, monkeypatch):
    assert _warmed(warm_db, monkeypatch) == set(warm_db["faces"].values())


def test_warm_crops_matched_only_excludes_unassigned_faces(warm_db, monkeypatch):
    got = _warmed(warm_db, monkeypatch, matched_only=True)
    assert got == {warm_db["faces"][warm_db["a"][0]]}


def test_warm_crops_person_ids_wins_over_matched_only(warm_db, monkeypatch):
    got = _warmed(warm_db, monkeypatch,
                  person_ids=[warm_db["person"]], matched_only=True)
    assert got == {warm_db["faces"][warm_db["a"][0]]}


def test_warm_crops_empty_person_ids_is_an_empty_scope(warm_db, monkeypatch):
    """`IN ()` is a SQLite syntax error, and falling through to "no person
    filter" would warm the whole library from a caller that asked for
    nobody."""
    assert _warmed(warm_db, monkeypatch, person_ids=[]) == set()


def test_warm_crops_date_range(warm_db, monkeypatch):
    day_one = {warm_db["faces"][p] for p in warm_db["a"]}
    day_two = {warm_db["faces"][p] for p in warm_db["b"]}
    assert _warmed(warm_db, monkeypatch, date_from="2090-05-01",
                   date_to="2090-05-01") == day_one
    assert _warmed(warm_db, monkeypatch, date_from="2090-05-02") == day_two
    assert _warmed(warm_db, monkeypatch, date_to="2090-05-01") == day_one


def test_warm_crops_photo_ids_scope(warm_db, monkeypatch):
    ids = warm_db["b"]
    assert _warmed(warm_db, monkeypatch, photo_ids=ids) == {
        warm_db["faces"][p] for p in ids}


def test_warm_crops_empty_photo_ids_is_an_empty_scope(warm_db, monkeypatch):
    """The batch case: membership is derived live from photos.folder, so a
    batch really can empty out. Warming the library instead would be hours of
    work nobody asked for."""
    assert _warmed(warm_db, monkeypatch, photo_ids=[]) == set()


def test_warm_crops_photo_ids_longer_than_the_chunk_size(warm_db, monkeypatch):
    """The chunking is new in the extraction. Padding with ids that do not
    exist also proves a chunk boundary can't drop or duplicate a row."""
    real = warm_db["a"] + warm_db["b"]
    padded = real + list(range(900_000, 900_000 + 120))
    rows = _fc._scoped_face_rows(warm_db["db"], photo_ids=padded, id_chunk=7)
    assert {r["id"] for r in rows} == set(warm_db["faces"].values())
    assert len(rows) == len(set(r["id"] for r in rows))   # no duplicates


def test_warm_crops_scopes_combine_with_and(warm_db, monkeypatch):
    """Day one AND the registered person = one face; day two AND that person
    = none. A scope that ORed would quietly widen every batch run."""
    assert _warmed(warm_db, monkeypatch, person_ids=[warm_db["person"]],
                   date_from="2090-05-01", date_to="2090-05-01") == {
        warm_db["faces"][warm_db["a"][0]]}
    assert _warmed(warm_db, monkeypatch, person_ids=[warm_db["person"]],
                   date_from="2090-05-02") == set()
    assert _warmed(warm_db, monkeypatch, photo_ids=warm_db["a"],
                   matched_only=True) == {warm_db["faces"][warm_db["a"][0]]}


def test_warm_crops_summary_counts_a_missing_original(warm_db, monkeypatch):
    """Replica mode with no --nas-url: nothing to decode, nothing to fetch.
    It must report that, not silently claim success."""
    monkeypatch.setattr(_fc, "render_face_crops", lambda *a, **k: {})
    summary = _fc.warm_crops(warm_db["db"], photo_ids=warm_db["b"])
    assert summary["total"] == 2
    assert summary["missing"] == 2
    assert summary["ok"] == 0 and summary["errors"] == 0


def test_warm_crops_skips_faces_already_cached(warm_db, monkeypatch):
    cache = _fc.face_crop_cache_dir(warm_db["db"].db_path)
    os.makedirs(cache, exist_ok=True)
    fid = warm_db["faces"][warm_db["b"][0]]
    for size in (120, 200):
        write_crop_atomic(crop_cache_path(cache, fid, size), b"x")
    summary = _fc.warm_crops(warm_db["db"], photo_ids=warm_db["b"])
    assert summary["found"] == 2
    assert summary["cached"] == 1
    assert summary["total"] == 1

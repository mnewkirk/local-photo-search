"""Unit tests for photosearch.face_crop — the shared face-crop renderer used by
the /api/faces/crop endpoint and the `warm-face-crops` / `export-face-crops`
replica-warming CLIs (see docs/plans/local-replica-and-writes.md)."""

import io
import os
import subprocess
import sys
import tarfile

from PIL import Image

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

"""Synthetic-image tests for photosearch/sharpness.py (no files, no cv2)."""

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from photosearch import sharpness as S


def _texture(w, h, seed=0, contrast=70.0):
    """Natural-ish texture: Gaussian-smoothed noise (grain ~2 px), mid-grey."""
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    im = Image.fromarray(np.clip(128 + 40 * n, 0, 255).astype(np.uint8))
    a = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return np.clip(128 + contrast * a, 0, 255).astype(np.uint8)


def _opener(im, fmt="PNG", exif=None):
    """An `open_image` that decodes `im` from in-memory bytes, like a file."""
    buf = io.BytesIO()
    kw = {"exif": exif} if exif is not None else {}
    if fmt == "JPEG":
        kw["quality"] = 95
    im.save(buf, fmt, **kw)
    data = buf.getvalue()
    return lambda _path: Image.open(io.BytesIO(data))


def _measure(arr_or_im, long_edge=None, **kw):
    im = arr_or_im if isinstance(arr_or_im, Image.Image) else Image.fromarray(arr_or_im)
    fmt = kw.pop("fmt", "PNG")
    exif = kw.pop("exif", None)
    return S.measure_photo("mem", kw.pop("faces", None), kw.pop("subjects", None),
                           open_image=_opener(im, fmt, exif),
                           long_edge=long_edge or max(im.size))


def test_blurred_copy_scores_lower():
    tex = _texture(800, 600)
    sharp = _measure(tex)
    blurred = _measure(np.asarray(Image.fromarray(tex).filter(ImageFilter.GaussianBlur(3))))
    assert sharp["version"] == S.SHARPNESS_VERSION
    assert sharp["detail"]["source"] == "frame"
    assert sharp["score"] > 0
    assert blurred["score"] is None or blurred["score"] < 0.3 * sharp["score"]
    f_s = sharp["detail"]["regions"]["frame"]
    f_b = blurred["detail"]["regions"]["frame"]
    assert (f_b["lap_max"] or 0) < f_s["lap_max"]
    assert (f_b["ten_max"] or 0) < f_s["ten_max"]


def test_added_noise_barely_moves_the_noise_corrected_metric():
    tex = _texture(800, 600, seed=3).astype(np.float32)
    rng = np.random.default_rng(7)
    noisy = np.clip(tex + rng.normal(0, 8.0, tex.shape), 0, 255).astype(np.uint8)
    clean = _measure(tex.astype(np.uint8))["detail"]
    dirty = _measure(noisy)["detail"]
    assert dirty["noise_sigma"] > clean["noise_sigma"] + 4
    fc, fd = clean["regions"]["frame"], dirty["regions"]["frame"]
    d_raw = fd["lap_median"] - fc["lap_median"]
    d_nc = fd["lap_nc_median"] - fc["lap_nc_median"]
    assert d_raw > 500                    # noise inflates the raw measure a lot
    assert abs(d_nc) < 0.25 * d_raw       # ...and the corrected one far less
    d_raw_max = fd["lap_max"] - fc["lap_max"]
    d_nc_max = fd["lap_nc_max"] - fc["lap_nc_max"]
    assert abs(d_nc_max) < 0.35 * d_raw_max


def test_flat_image_is_all_skipped_and_scores_none():
    out = _measure(np.full((600, 800), 120, np.uint8))
    assert out["score"] is None
    d = out["detail"]
    assert "error" not in d
    assert d["source"] is None
    fr = d["regions"]["frame"]
    assert fr["tiles"] > 0 and fr["flat_skipped"] == fr["tiles"]
    assert fr["lap_max"] is None
    feats = S.flatten_features(out)
    assert feats["score"] is None and feats["frame.lap_max"] is None


def test_exif_orientation_6_crops_the_displayed_region():
    # Stored (raw) 600x400: sharp texture in the TOP-LEFT quadrant, flat
    # elsewhere. Orientation 6 = rotate 90 deg clockwise for display, so the
    # displayed frame is 400x600 and that quadrant lands TOP-RIGHT.
    raw = np.full((400, 600), 120, np.uint8)
    raw[:200, :300] = _texture(300, 200, seed=5)
    im = Image.fromarray(raw)
    exif = im.getexif()
    exif[0x0112] = 6
    exif_bytes = exif.tobytes()
    # Face boxes are in the ORIENTED original (400 wide x 600 tall).
    top_right = {"bbox_left": 210, "bbox_top": 10, "bbox_right": 390, "bbox_bottom": 290}
    top_left = {"bbox_left": 10, "bbox_top": 10, "bbox_right": 190, "bbox_bottom": 290}
    # long_edge=300 halves the image, exercising the JPEG draft path too.
    hit = _measure(im, long_edge=300, fmt="JPEG", exif=exif_bytes, faces=[top_right])
    miss = _measure(im, long_edge=300, fmt="JPEG", exif=exif_bytes, faces=[top_left])
    d = hit["detail"]
    assert d["orientation"] == 6
    assert d["orig_size"] == [400, 600] and d["norm_size"] == [200, 300]
    assert d["decode"] == "draft"
    assert d["source"] == "faces"
    assert d["regions"]["faces"]["lap_max"] > 50
    # The same box on the other side of the displayed frame is flat.
    assert miss["detail"]["regions"]["faces"]["lap_max"] is None
    assert miss["detail"]["source"] == "frame"


def test_sharp_subject_on_blurred_background():
    bg = Image.fromarray(_texture(900, 600, seed=11)).filter(ImageFilter.GaussianBlur(4))
    arr = np.asarray(bg).copy()
    arr[200:400, 350:550] = _texture(200, 200, seed=12)
    subj = [{"label": "flower", "bbox": [350 / 900, 200 / 600, 550 / 900, 400 / 600]}]
    out = _measure(arr, subjects=subj)
    d = out["detail"]
    assert d["source"] == "subjects"
    sub, fr = d["regions"]["subjects"], d["regions"]["frame"]
    assert sub["lap_max"] > 5 * (fr["lap_median"] or 0)
    # The frame's tile grid does not align with the subject, so its best tile
    # only partly covers it -- still far above the frame's own median.
    assert fr["lap_max"] > 3 * fr["lap_median"]
    assert fr["lap_max"] <= sub["lap_max"]
    assert fr["best_median_ratio"] > 5
    assert out["score"] == pytest.approx(max(0.0, sub["lap_nc_max"]))


def test_faces_take_priority_and_tiny_faces_are_skipped():
    arr = _texture(800, 600, seed=2)
    subj = [[0.1, 0.1, 0.5, 0.5]]
    faces = [(100, 100, 300, 300), (500, 500, 520, 520)]     # second is tiny
    out = _measure(arr, faces=faces, subjects=subj)
    d = out["detail"]
    assert d["source"] == "faces"
    assert d["regions"]["faces"]["boxes"] == 1
    assert d["regions"]["faces"]["boxes_skipped"] == 1
    assert "subjects" in d["regions"] and "frame" in d["regions"]


def test_bad_file_returns_error_dict():
    def garbage(_p):
        return Image.open(io.BytesIO(b"not an image at all"))
    out = S.measure_photo("x.jpg", [], [], open_image=garbage)
    assert out["score"] is None
    assert out["version"] == S.SHARPNESS_VERSION
    assert "error" in out["detail"]

    def missing(_p):
        raise FileNotFoundError("gone")
    out = S.measure_photo("x.jpg", open_image=missing)
    assert out["score"] is None and "FileNotFoundError" in out["detail"]["error"]


def test_real_path_default_opener(tmp_path):
    p = tmp_path / "t.jpg"
    Image.fromarray(_texture(640, 480)).save(p, "JPEG", quality=95)
    out = S.measure_photo(str(p), long_edge=640)
    assert out["score"] and out["score"] > 0
    assert out["detail"]["orig_size"] == [640, 480]


def test_numpy_laplacian_matches_the_classic_kernel():
    g = np.arange(25, dtype=np.float32).reshape(5, 5) ** 2
    lap = S.laplacian(g)
    # direct 4-neighbour convolution at (2,2)
    want = g[1, 2] + g[3, 2] + g[2, 1] + g[2, 3] - 4 * g[2, 2]
    assert lap[1, 1] == pytest.approx(want)
    assert S.laplacian_variance(g) == pytest.approx(float(lap.var()))


def test_immerkaer_recovers_white_noise_sigma():
    rng = np.random.default_rng(0)
    g = 128 + rng.normal(0, 5.0, (400, 400)).astype(np.float32)
    masked, raw = S.noise_sigma(g, exclude=0.0)
    assert raw == pytest.approx(5.0, rel=0.05)


def test_flatten_features_names_every_candidate():
    out = _measure(_texture(600, 400), subjects=[[0.2, 0.2, 0.8, 0.8]])
    f = S.flatten_features(out)
    assert f["score"] == out["score"]
    assert f["best.lap_nc_max"] == out["detail"]["regions"]["subjects"]["lap_nc_max"]
    assert f["faces.lap_max"] is None
    for r in S.REGION_ORDER:
        for feat in S.REGION_FEATURES:
            assert f"{r}.{feat}" in f
    err = S.flatten_features({"score": None, "detail": {"error": "x"}})
    assert err == {"score": None}

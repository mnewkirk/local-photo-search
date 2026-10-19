"""Worker side of video-scan (docs/plans/video-support.md step 3):
detect_faces_bgr parity + raise-on-error, ZIP unwrap safety, the pure
measures, build_scan_result over synthetic ScanData (CI mocks cv2 and
insightface), and _process_video_scan's outcome classification."""

import hashlib
import io
import os
import zipfile
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from photosearch import faces as F
from photosearch import video_scan as VS
from photosearch import worker as W
from photosearch.video_analysis import segment_schedule
from photosearch.video_scan_store import SCAN_VERSION


def _real_cv2():
    try:
        import cv2
    except ImportError:
        return None
    return cv2 if not isinstance(cv2, mock.Mock) and hasattr(cv2, "VideoWriter") else None


# --- detect_faces_bgr ----------------------------------------------------------

class _FakeFace:
    def __init__(self, bbox, score, emb):
        self.bbox = np.array(bbox, dtype=float)
        self.det_score = score
        self.normed_embedding = None
        self.embedding = np.array(emb, dtype=float)


class _FakeApp:
    def __init__(self, faces=None, boom=False):
        self.faces, self.boom, self.seen = faces or [], boom, []

    def get(self, img):
        self.seen.append(img.shape)
        if self.boom:
            raise RuntimeError("onnx exploded")
        return self.faces


@pytest.fixture
def fake_app(monkeypatch):
    emb = np.zeros(512)
    emb[3] = 2.0
    app = _FakeApp([_FakeFace([10, 20, 110, 140], 0.93, emb)])
    monkeypatch.setattr(F, "check_available", lambda: None)
    monkeypatch.setattr(F, "_get_face_app", lambda det_size=None: app)
    monkeypatch.setattr(F, "_downsample", lambda img: (img, 1.0))
    return app


def test_detect_faces_bgr_matches_detect_faces(fake_app, monkeypatch, tmp_path):
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    monkeypatch.setattr(F, "_read_image_bgr", lambda p: img)
    by_path = F.detect_faces(str(tmp_path / "x.jpg"))
    by_array = F.detect_faces_bgr(img)
    assert by_path == by_array
    assert by_array[0]["bbox"] == (20, 110, 140, 10)          # top, right, bottom, left
    assert by_array[0]["det_score"] == pytest.approx(0.93)
    assert np.linalg.norm(by_array[0]["encoding"]) == pytest.approx(1.0)


def test_detect_faces_bgr_raises_where_detect_faces_swallows(fake_app, monkeypatch, tmp_path):
    fake_app.boom = True
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    monkeypatch.setattr(F, "_read_image_bgr", lambda p: img)
    with pytest.raises(RuntimeError):
        F.detect_faces_bgr(img)
    assert F.detect_faces(str(tmp_path / "x.jpg")) == []
    with pytest.raises(ValueError):
        F.detect_faces_bgr(np.zeros((5, 5)))


# --- ZIP unwrap ------------------------------------------------------------------

def _zip(path, members, compress=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "w", compression=compress) as z:
        for name, data in members:
            z.writestr(name, data)
    return str(path)


def test_unwrap_single_member(tmp_path):
    payload = os.urandom(4096)
    p = _zip(tmp_path / "IMG_1585.MOV", [("IMG_1585.MOV", payload)])
    assert VS.is_zip_file(p)
    out = VS.unwrap_zip(p, str(tmp_path))
    assert open(out, "rb").read() == payload
    assert os.path.dirname(out) == str(tmp_path)


@pytest.mark.parametrize("members", [
    [("a.mov", b"1"), ("b.mov", b"2")],          # more than one file
    [],                                           # nothing at all
    [("../../etc/evil.mov", b"x")],               # traversal
    [("/abs/evil.mov", b"x")],                    # absolute
])
def test_unwrap_refuses_unsafe_zips(tmp_path, members):
    p = _zip(tmp_path / "w.mov", members)
    with pytest.raises(VS.ZipRefused):
        VS.unwrap_zip(p, str(tmp_path / "out"))


def test_unwrap_refuses_oversized_and_bombs(tmp_path):
    p = _zip(tmp_path / "big.mov", [("v.mov", b"\x00" * 200_000)])
    with pytest.raises(VS.ZipRefused):
        VS.unwrap_zip(p, str(tmp_path))
    p = _zip(tmp_path / "big2.mov", [("v.mov", os.urandom(5000))])
    with pytest.raises(VS.ZipRefused):
        VS.unwrap_zip(p, str(tmp_path), max_bytes=1000)
    assert not [f for f in os.listdir(tmp_path) if f.startswith("inner_")]


def test_unwrap_caps_the_member_relative_to_the_zip_even_without_a_ratio(tmp_path):
    """With the ratio check out of the way (as when a header says
    compress_size 0), the inner file may still be at most 2 x the ZIP + 16 MB."""
    p = _zip(tmp_path / "z.mov", [("v.mov", b"\x00" * 40_000_000)])
    with pytest.raises(VS.ZipRefused, match="bytes >"):
        VS.unwrap_zip(p, str(tmp_path), max_ratio=1e12)
    # incompressible payload of the same size passes the relative cap
    p = _zip(tmp_path / "ok.mov", [("v.mov", os.urandom(300_000))])
    assert os.path.getsize(VS.unwrap_zip(p, str(tmp_path), max_ratio=1e12)) == 300_000


def test_unwrap_accepts_a_mildly_compressible_clip(tmp_path):
    """HTC Zoe clips (2018) deflate ~1.3x; the old 1.2x cap refused four of
    them on the NAS. A modest ratio must pass."""
    payload = os.urandom(200_000) + b"\x00" * 100_000
    p = _zip(tmp_path / "zoe.mp4", [("IMAG0046_ZOEVIDEO.mp4", payload)])
    assert open(VS.unwrap_zip(p, str(tmp_path)), "rb").read() == payload


def test_unwrap_ignores_mac_litter_and_takes_the_video_of_a_bundle(tmp_path):
    payload = os.urandom(4096)
    p = _zip(tmp_path / "IMG_1.MOV", [
        ("__MACOSX/._IMG_1.MOV", b"resource fork"),
        ("._IMG_1.MOV", b"appledouble"),
        ("IMG_1.JPG", os.urandom(512)),
        ("IMG_1.MOV", payload)])
    out = VS.unwrap_zip(p, str(tmp_path))
    assert open(out, "rb").read() == payload and out.endswith("IMG_1.MOV")


def test_unwrap_refuses_two_videos_even_beside_litter(tmp_path):
    p = _zip(tmp_path / "w.mov", [("a.mov", b"1"), ("b.mp4", b"2"), ("._a.mov", b"x")])
    with pytest.raises(VS.ZipRefused, match="2 video"):
        VS.unwrap_zip(p, str(tmp_path))
    p = _zip(tmp_path / "j.mov", [("only.jpg", b"1")])
    with pytest.raises(VS.ZipRefused, match="0 video"):
        VS.unwrap_zip(p, str(tmp_path))


def test_a_4k_identity_bbox_is_in_native_display_pixels(tmp_path):
    """The rep frame JPEG is stored at <= 1920 px, but the faces bbox is in
    the video's own display pixels — photos.image_width's space."""
    scan = _synthetic_scan(tmp_path, with_frames=False, two=False)
    scan.width, scan.height, scan.store_scale = 3840, 2160, 0.5
    built = VS.build_scan_result(scan, with_clip=False)
    # _det(810): x 810..1110, y 300..600 in decoded (native) px
    assert built["result"]["identities"][0]["bbox"] == [300, 1110, 600, 810]


def test_a_plain_video_is_not_a_zip(tmp_path):
    p = tmp_path / "v.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)
    assert not VS.is_zip_file(str(p))
    with pytest.raises(VS.ZipRefused):
        VS.unwrap_zip(str(p), str(tmp_path))


# --- pure measures ------------------------------------------------------------------

def test_phase_shift_recovers_a_translation():
    rng = np.random.default_rng(0)
    base = rng.random((96, 128)) * 255
    moved = np.roll(np.roll(base, 3, axis=1), -2, axis=0)
    dx, dy = VS.phase_shift(base, moved)
    assert round(dx) == 3 and round(dy) == -2


def test_mean_unit_and_bbox_conversions():
    v = VS.mean_unit([[1, 0], [0, 2], None])
    assert np.linalg.norm(v) == pytest.approx(1.0) and v[0] == pytest.approx(v[1])
    assert VS.mean_unit([]) is None
    assert VS.trbl_from_xyxy((10, 20, 30, 40)) == [20, 30, 40, 10]
    assert VS.trbl_from_xyxy((-5, 20, 30, 50), 1.0, 25, 45) == [20, 25, 45, 0]
    assert VS.trbl_from_xyxy((10, 20, 30, 40), 0.5) == [10, 15, 20, 5]
    assert VS.laplacian_var(np.zeros((10, 10))) == 0.0
    assert VS.tenengrad(np.tile(np.arange(10.0), (10, 1))) > 0


# --- build_scan_result over a synthetic ScanData -------------------------------------

def _det(x, emb, w=300):
    return {"bbox": (x, 300.0, x + w, 300.0 + w), "det_score": 0.9, "embedding": emb,
            "sharpness": 50.0}


def _synthetic_scan(tmp_path, n=36, with_frames=True, two=True):
    e1 = np.zeros(512)
    e1[0] = 1.0
    e2 = np.zeros(512)
    e2[1] = 1.0
    frames = []
    for i in range(n):
        faces = [_det(810.0, e1.tolist())]
        if two and i < n // 2:
            faces.append(_det(100.0, e2.tolist(), w=120))
        frames.append({"t": (i + 0.5) / 4, "faces": faces})
    measures = [{"t": f["t"], "frame_no": i, "shift_x": float((-1) ** i), "shift_y": 0.0,
                 "sigma": 2.0, "clip_hi": 0.01, "clip_lo": 0.0, "sharpness": float(i),
                 "tenengrad": 100.0 + i, "luma_mean": 120.0}
                for i, f in enumerate(frames)]
    duration = n / 4.0
    segs = segment_schedule(duration)
    stored = {}
    if with_frames:
        d = tmp_path / "frames"
        d.mkdir()
        for i in range(n):
            p = d / f"f{i:05d}.jpg"
            Image.new("RGB", (1920, 1080), (i * 5 % 255, 90, 160)).save(p, "JPEG")
            stored[i] = str(p)
    return VS.ScanData(
        path="x.mov", width=1920, height=1080, duration=duration, fps=30.0,
        sample_fps=4.0, n_decoded=n * 7, frames=frames, frame_measures=measures,
        keyframe_ix=[4, 18, 30], segments=segs,
        segment_ix=[min(n - 1, int((a + b) / 2 * 4)) for a, b in segs],
        store_dir=str(tmp_path / "frames") if with_frames else None, _stored=stored)


def test_build_scan_result_payload_and_assets(tmp_path):
    scan = _synthetic_scan(tmp_path)
    calls = []

    def embed(paths):
        calls.append(paths)
        return [[1.0] + [0.0] * 511, [0.0, 1.0] + [0.0] * 510, None][:len(paths)]
    built = VS.build_scan_result(scan, embed=embed)
    r = built["result"]
    assert r["scan_version"] == SCAN_VERSION and r["scan_error"] is None
    # two people -> two identities, the dominant one first
    assert len(r["identities"]) == 2
    first = r["identities"][0]
    assert first["focus_s"] >= r["identities"][1]["focus_s"]
    assert len(first["encoding"]) == 512 and len(first["bbox"]) == 4
    # CLIP: mean of the unit keyframe embeddings, renormalised
    assert len(calls) == 1 and len(calls[0]) == 3
    assert np.linalg.norm(r["clip_embedding"]) == pytest.approx(1.0)
    assert r["clip_embedding"][0] == pytest.approx(r["clip_embedding"][1])
    names = [(a["kind"], a["name"]) for a in built["assets"]]
    assert ("keyframe", "k00.jpg") in names and ("keyframe", "k02.jpg") in names
    assert sum(1 for k, n in names if n.startswith("s")) == len(scan.segments)
    assert ("crop", "i000_120.jpg") in names and ("crop", "i001_200.jpg") in names
    assert sum(1 for k, n in names if n.startswith("rep_f")) >= 1
    assert all(a["data"][:3] == b"\xff\xd8\xff" for a in built["assets"])
    assert set(built["poster"]) == {"thumb", "preview", "full"} and r["poster_uploaded"]
    q = r["quality_json"]
    assert len(q["frames"]["segments"]) == len(scan.segments)
    assert q["frames"]["segments"][0]["name"] == "keyframe_s000.jpg"
    assert r["shake"] is not None and r["noise"] == pytest.approx(2.0)
    assert q["tenengrad_max"] == pytest.approx(135.0)
    # every segment keyframe is <= 768 px, K keyframes <= 1024
    seg = next(a for a in built["assets"] if a["name"] == "s000.jpg")
    assert max(Image.open(io.BytesIO(seg["data"])).size) == 768


def test_build_without_frames_or_clip_still_yields_identities(tmp_path):
    scan = _synthetic_scan(tmp_path, with_frames=False, two=False)
    built = VS.build_scan_result(scan, with_clip=False)
    assert built["assets"] == [] and built["poster"] is None
    assert len(built["result"]["identities"]) == 1
    assert built["result"]["clip_embedding"] is None
    # the eval harness's contract: aliases + a summary dict
    assert scan.W == 1920 and scan.H == 1080
    m = scan.measures
    assert {"shake", "noise", "focus_consistency", "tenengrad_max", "exposure"} <= set(m)
    assert scan.keyframe_times == [scan.frames[i]["t"] for i in scan.keyframe_ix]


def test_assets_bodies_split_under_the_caps_and_poster_goes_last():
    assets = [{"kind": "keyframe", "name": f"s{i:03d}.jpg", "data": b"\xff\xd8\xff" + b"x" * 10}
              for i in range(250)]
    bodies = VS.assets_bodies(assets, {"thumb": b"\xff\xd8\xffp"}, max_assets=100)
    assert [len(b["assets"]) for b in bodies] == [100, 100, 51]
    assert bodies[-1]["assets"][-1]["kind"] == "poster"
    assert all(b["version"] == SCAN_VERSION for b in bodies)


# --- scan_video with real cv2 (skipped on CI, where cv2 is mocked) --------------------

@pytest.mark.skipif(_real_cv2() is None, reason="needs real cv2")
def test_scan_video_on_a_synthetic_clip(tmp_path):
    cv2 = _real_cv2()
    path = str(tmp_path / "clip.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (320, 240))
    rng = np.random.default_rng(1)
    bg = (rng.random((240, 400, 3)) * 255).astype(np.uint8)
    for i in range(60):
        vw.write(np.ascontiguousarray(bg[:, i:i + 320]))
    vw.release()
    seen = []

    def detect(bgr):
        seen.append(bgr.shape)
        return [{"bbox": (60, 200, 160, 100), "det_score": 0.9, "encoding": [1.0] + [0.0] * 511}]
    scan = VS.scan_video(path, detect=detect)
    try:
        assert scan.width == 320 and scan.height == 240
        assert scan.duration == pytest.approx(3.0, abs=0.1)
        assert len(scan.frames) == 12 and len(seen) == 12        # 4 fps x 3 s
        assert scan.frames[0]["faces"][0]["bbox"] == (100.0, 60.0, 200.0, 160.0)
        assert scan.frame_measures[0]["shift_x"] == 0.0
        # the background pans 5 px/sample-frame-step; phase correlation sees it
        assert abs(scan.frame_measures[3]["shift_x"]) > 1.0
        assert scan.frame_path(scan.keyframe_ix[0]) is not None
    finally:
        scan.cleanup()
    assert scan.store_dir is None


@pytest.mark.skipif(_real_cv2() is None, reason="needs real cv2")
def test_scan_video_raises_open_error_on_garbage(tmp_path):
    p = tmp_path / "junk.mp4"
    p.write_bytes(os.urandom(2048))
    with pytest.raises(VS.ScanOpenError):
        VS.scan_video(str(p), detect=lambda b: [])


# --- worker._process_video_scan --------------------------------------------------------

class _Resp:
    def __init__(self, status=200, body=b"", headers=None, ok=True):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.ok, self.text = ok, ""

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), 7):
            yield self._body[i:i + 7]

    def close(self):
        pass


class _Client:
    server_url = "http://nas"

    def __init__(self, body, length=None, upload_ok=True):
        self.body, self.length, self.upload_ok, self.uploads = body, length, upload_ok, []

    def _request(self, method, url, **kw):
        if method == "GET":
            n = len(self.body) if self.length is None else self.length
            return _Resp(200, self.body, {"Content-Length": str(n)})
        self.uploads.append(kw.get("json"))
        return _Resp(200 if self.upload_ok else 400, ok=self.upload_ok)


def test_download_is_hashed_and_truncation_is_transient(tmp_path):
    c = _Client(b"hello video")
    got = W._download_hashed(c, 1, str(tmp_path / "v"))
    assert got["sha256"] == hashlib.sha256(b"hello video").hexdigest()
    with pytest.raises(W.TruncatedDownload):
        W._download_hashed(_Client(b"short", length=99), 1, str(tmp_path / "w"))
    assert not (tmp_path / "w").exists()
    assert issubclass(W.TruncatedDownload, W._TRANSIENT)


def _fake_scan(tmp_path):
    return _synthetic_scan(tmp_path, with_frames=False, two=False)


def test_outcomes_result_error_failure_omit(tmp_path, monkeypatch):
    photo = {"id": 7, "filename": "A.MOV", "duration_s": 9.0}
    # success -> a result row carrying the hash, assets uploaded
    monkeypatch.setattr(VS, "scan_video", lambda p, duration=None: _fake_scan(tmp_path))
    monkeypatch.setattr(VS, "build_scan_result", lambda s: {
        "result": {"scan_version": SCAN_VERSION, "scan_error": None, "identities": [],
                   "quality_json": {"timings": {}}, "poster_uploaded": False},
        "assets": [{"kind": "keyframe", "name": "k00.jpg", "data": b"\xff\xd8\xff"}],
        "poster": None})
    c = _Client(b"movie bytes")
    [r] = W._process_video_scan(c, [photo], str(tmp_path))
    assert r["photo_id"] == 7 and r["file_hash"] == hashlib.sha256(b"movie bytes").hexdigest()
    assert len(c.uploads) == 1

    # cv2 cannot open the COMPLETE file -> a stored verdict, not a failure
    def boom(p, duration=None):
        raise VS.ScanOpenError("cv2 could not open the file")
    monkeypatch.setattr(VS, "scan_video", boom)
    [r] = W._process_video_scan(_Client(b"x"), [photo], str(tmp_path))
    assert r["scan_error"].startswith("unreadable") and "error" not in r

    # an exception mid-analysis -> a failure row (one attempt)
    def crash(p, duration=None):
        raise RuntimeError("detection raised")
    monkeypatch.setattr(VS, "scan_video", crash)
    [r] = W._process_video_scan(_Client(b"x"), [photo], str(tmp_path))
    assert "error" in r and "detection raised" in r["error"]

    # an asset upload the NAS rejects -> a failure row
    monkeypatch.setattr(VS, "scan_video", lambda p, duration=None: _fake_scan(tmp_path))
    [r] = W._process_video_scan(_Client(b"x", upload_ok=False), [photo], str(tmp_path))
    assert "error" in r and "asset upload" in r["error"]

    # a truncated download -> omitted entirely (free retry)
    assert W._process_video_scan(_Client(b"x", length=50), [photo], str(tmp_path)) == []
    assert os.listdir(tmp_path) == [] or all(not n.startswith("vscan-")
                                             for n in os.listdir(tmp_path))


def test_zip_wrapped_bad_zip_is_a_stored_verdict(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.mov", b"1")
        z.writestr("b.mov", b"2")
    [r] = W._process_video_scan(_Client(buf.getvalue()), [{"id": 3, "filename": "Z.MOV"}],
                                str(tmp_path))
    assert r["scan_error"].startswith("zip:")


def test_zip_inner_probe_rides_on_the_result(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("IMG_1.MOV", os.urandom(512))
    monkeypatch.setattr(VS, "probe_local", lambda p, tz_name=None: {"ok": {
        "probe_version": 1, "duration_s": 5.0, "codec": "h264", "width": 1080,
        "height": 1920}})
    seen = {}

    def scan(p, duration=None):
        seen["path"], seen["duration"] = p, duration
        return _fake_scan(tmp_path)
    monkeypatch.setattr(VS, "scan_video", scan)
    monkeypatch.setattr(VS, "build_scan_result", lambda s: {
        "result": {"scan_version": SCAN_VERSION, "identities": [],
                   "quality_json": {"timings": {}}}, "assets": [], "poster": None})
    [r] = W._process_video_scan(_Client(buf.getvalue()), [{"id": 4, "filename": "Z.MOV"}],
                                str(tmp_path))
    assert r["probe"]["codec"] == "h264" and seen["duration"] == 5.0
    assert os.path.basename(seen["path"]).startswith("inner_")


def test_video_scan_claims_one_video_and_reads_the_file_itself():
    assert W.PASS_CLAIM_LIMITS["video-scan"] == 1
    assert "video-scan" in W.SELF_READ_PASSES

"""video-probe parsing (docs/plans/video-support.md step 2b).

ffprobe and cv2 are never on CI: every case here is fed canned ffprobe JSON
captured from the real fixtures in /home/mattn/video-fixtures/ (and one Pixel
clip probed on the NAS), trimmed and committed under
tests/fixtures/video_probe/. The Pixel fixture's GPS was replaced with a
public landmark.
"""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from photosearch import video_probe as VP

FIX = Path(__file__).parent / "fixtures" / "video_probe"


def _fx(name):
    return json.loads((FIX / f"{name}.json").read_text())


# ---------------------------------------------------------------------------
# The ordinary files
# ---------------------------------------------------------------------------

def test_ok_generic_2016_mp4():
    r = VP.parse_probe(_fx("generic_2016"))
    assert r["probe_error"] is None
    assert (r["codec"], r["container"], r["width"], r["height"]) == ("h264", "mp4", 1280, 720)
    assert r["rotation"] == 0 and r["has_audio"] is True
    assert r["duration_s"] == pytest.approx(26.169, abs=1e-3)
    assert r["fps"] == pytest.approx(29.902, abs=1e-3)
    assert r["bitrate"] > 0
    # no family the date policy trusts -> keep the provisional mtime
    assert r["date_taken"] is None and r["date_source"] == "mtime"
    assert "probe_json" in r and r["probe_json"]["format"]["format_name"].startswith("mov")


@pytest.mark.parametrize("name,container", [("canon_avi", "avi"), ("avchd_mts", "mpegts")])
def test_avi_and_mts_parse(name, container):
    r = VP.parse_probe(_fx(name))
    assert r["probe_error"] is None and r["container"] == container
    assert (r["width"], r["height"]) == (1920, 1080)


def test_probe_json_drops_the_display_matrix_and_url():
    r = VP.parse_probe(_fx("apple_live_hevc_rot"))
    blob = json.dumps(r["probe_json"])
    assert "displaymatrix" not in blob and "http://" not in blob
    assert r["probe_json"]["derived"] == {"camera_family": "apple", "date_source": "apple_cd"}


# ---------------------------------------------------------------------------
# Rotation: display dims are what is stored
# ---------------------------------------------------------------------------

def test_iphone_portrait_rotation_swaps_to_display_dims():
    r = VP.parse_probe(_fx("apple_2017_rot"))   # coded 1440x1080, side-data -90
    assert r["rotation"] == 90
    assert (r["width"], r["height"]) == (1080, 1440)


@pytest.mark.parametrize("side_rot,expect,dims", [
    (-90, 90, (1080, 1920)), (90, 270, (1080, 1920)),
    (180, 180, (1920, 1080)), (-180, 180, (1920, 1080)), (0, 0, (1920, 1080)),
])
def test_rotation_side_data_variants(side_rot, expect, dims):
    data = _fx("apple_live_hevc_rot")
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    v["side_data_list"] = [{"side_data_type": "Display Matrix", "rotation": side_rot}]
    r = VP.parse_probe(data)
    assert r["rotation"] == expect
    assert (r["width"], r["height"]) == dims


@pytest.mark.parametrize("tag,expect", [("90", 90), ("180", 180), ("270", 270), ("-90", 270)])
def test_legacy_rotate_tag(tag, expect):
    data = _fx("generic_2016")
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    v.pop("side_data_list", None)
    v.setdefault("tags", {})["rotate"] = tag
    r = VP.parse_probe(data)
    assert r["rotation"] == expect
    if expect in (90, 270):
        assert (r["width"], r["height"]) == (720, 1280)


# ---------------------------------------------------------------------------
# The per-camera-family date policy (Step 0 / D14)
# ---------------------------------------------------------------------------

def test_apple_creationdate_is_local_wall_time_offset_dropped():
    r = VP.parse_probe(_fx("apple_live_hevc_rot"))
    # creationdate 2025-06-24T14:16:05+0200; creation_time 12:16:05Z (UTC)
    assert r["date_taken"] == "2025-06-24 14:16:05"
    assert r["date_source"] == "apple_cd"
    assert (r["camera_make"], r["camera_model"]) == ("Apple", "iPhone 15 Pro")
    r = VP.parse_probe(_fx("apple_2017_rot"))
    assert r["date_taken"] == "2017-10-04 06:36:33"


def test_apple_without_creationdate_converts_utc_to_home_zone():
    data = _fx("apple_2017_rot")
    del data["format"]["tags"]["com.apple.quicktime.creationdate"]
    r = VP.parse_probe(data, tz_name="America/Los_Angeles")
    assert r["date_taken"] == "2017-10-04 06:36:33"  # 13:36:33Z in PDT
    assert r["date_source"] == "utc"


def test_pixel_creation_time_is_recording_end_utc():
    """PXL_20260904_165917737 started 16:59:17.7Z; the muxer stamped
    creation_time at the END (16:59:24Z) of a 4.03 s clip."""
    r = VP.parse_probe(_fx("pixel_end_time"), tz_name="America/Los_Angeles")
    assert r["date_source"] == "utc_end"
    assert r["date_taken"] == "2026-09-04 09:59:19"   # (24 - 4.03) s, in PDT
    assert (r["camera_make"], r["camera_model"]) == ("Google", "Pixel 10 Pro XL")
    assert r["rotation"] == 90 and (r["width"], r["height"]) == (2160, 3840)


def test_pixel_respects_the_configured_zone(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TZ", "Europe/Rome")
    r = VP.parse_probe(_fx("pixel_end_time"))
    assert r["date_taken"] == "2026-09-04 18:59:19"


def test_sony_z_is_local_wall_time_not_utc():
    r = VP.parse_probe(_fx("sony_ilce"))
    assert r["date_taken"] == "2026-07-08 07:33:10"  # the 'Z' value verbatim
    assert r["date_source"] == "local_z"
    assert r["camera_make"] == "SONY" and r["camera_model"] is None


def test_kodak_z_is_local_and_model_matches_the_stills():
    r = VP.parse_probe(_fx("kodak_pixpro"))
    assert r["date_taken"] == "2026-08-03 16:07:17" and r["date_source"] == "local_z"
    # the stills store make "JK Imaging, Ltd." / model "KODAK PIXPRO WPZ2"
    assert r["camera_make"] is None and r["camera_model"] == "KODAK PIXPRO WPZ2"


@pytest.mark.parametrize("tags,family", [
    ({"com.apple.quicktime.make": "Apple"}, "apple"),
    ({"com.apple.quicktime.creationdate": "2020-01-01T00:00:00+0000"}, "apple"),
    ({"com.android.version": "12"}, "android"),
    ({"major_brand": "XAVC"}, "sony"),
    ({"make": "Sony"}, "sony"),
    ({"original_format": "KODAK"}, "kodak"),
    ({"comment": "PIXPRO WPZ2"}, "kodak"),
    ({"major_brand": "MSNV"}, "unknown"),
    ({}, "unknown"),
    (None, "unknown"),
])
def test_camera_family_dispatch(tags, family):
    assert VP.camera_family(tags) == family
    assert family in VP.DATE_POLICY


def test_epoch_zero_creation_time_is_unset():
    data = _fx("sony_ilce")
    data["format"]["tags"]["creation_time"] = "1904-01-01T00:00:00.000000Z"
    r = VP.parse_probe(data)
    assert r["date_taken"] is None and r["date_source"] == "mtime"


# ---------------------------------------------------------------------------
# GPS (ISO 6709)
# ---------------------------------------------------------------------------

def test_pixel_location_tag_parses():
    r = VP.parse_probe(_fx("pixel_end_time"))
    assert (r["gps_lat"], r["gps_lon"]) == (37.8199, -122.4783)


def test_apple_iso6709_with_altitude():
    data = _fx("apple_live_hevc_rot")
    data["format"]["tags"]["com.apple.quicktime.location.ISO6709"] = "+45.9763+007.6586+3214.000/"
    r = VP.parse_probe(data)
    assert (r["gps_lat"], r["gps_lon"], r["gps_alt"]) == (45.9763, 7.6586, 3214.0)


@pytest.mark.parametrize("s,expect", [
    ("+37.3317-122.0307/", (37.3317, -122.0307, None)),
    ("-33.8568+151.2153+005.0/", (-33.8568, 151.2153, 5.0)),
    ("+4012.5-07530.0/", (40.208333, -75.5, None)),           # DDMM.m
    ("+401230-0753000/", (40.208333, -75.5, None)),           # DDMMSS
    ("+00.0000+000.0000/", None),                              # null island
    ("+95.0-122.0/", None),
    ("garbage", None), ("", None), (None, None),
])
def test_parse_iso6709(s, expect):
    got = VP.parse_iso6709(s)
    if expect is None:
        assert got is None
    else:
        assert got[0] == pytest.approx(expect[0], abs=1e-5)
        assert got[1] == pytest.approx(expect[1], abs=1e-5)
        assert got[2] == expect[2]


# ---------------------------------------------------------------------------
# Errors: zip magic, moov missing, truncated, no video stream
# ---------------------------------------------------------------------------

def test_zip_magic():
    assert VP.is_zip_magic(b"PK\x03\x04\x14\x00\x00\x00")
    assert not VP.is_zip_magic(b"\x00\x00\x00\x18ftypmp42")
    assert not VP.is_zip_magic(b"")


def test_classify_probe_error():
    assert VP.classify_probe_error(None) is None
    assert VP.classify_probe_error(VP.ZIP_WRAPPED) == "zip_wrapped"
    assert VP.classify_probe_error("unreadable: moov atom not found") == "unreadable"
    assert VP.classify_probe_error("moov atom not found") == "unreadable"  # pre-2b rows


def test_moov_missing_stderr_is_a_verdict(monkeypatch):
    stderr = (FIX / "moov_missing.stderr").read_text()
    assert VP.classify_ffprobe_failure(stderr) == "corrupt"

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 1, stdout="{\n\n}\n", stderr=stderr)

    out = VP.run_ffprobe("http://nas/x", ffprobe="/bin/ffprobe", run=fake_run)
    assert out == {"_verdict": "unreadable: moov atom not found"}
    assert VP.classify_probe_error(out["_verdict"]) == "unreadable"


def test_truncated_tail_moov_file_is_the_same_verdict(monkeypatch):
    """A moov-at-tail MP4 cut short (measured: head -c 3MB of the 2016 1080p
    fixture) fails exactly like the moov-missing one."""
    stderr = ("[mov,mp4,m4a,3gp,3g2,mj2 @ 0x11b7ba00] moov atom not found\n"
              "/tmp/trunc.mp4: Invalid data found when processing input\n")
    out = VP.run_ffprobe("u", ffprobe="x", run=lambda cmd, **kw:
                         subprocess.CompletedProcess(cmd, 1, "{}", stderr))
    assert out["_verdict"].startswith("unreadable: moov atom not found")


def test_truncated_faststart_file_still_parses():
    """A faststart file cut mid-mdat still has its header: ffprobe succeeds
    (measured) and so does the probe — it is playable as far as it goes."""
    data = _fx("generic_2016")
    r = VP.parse_probe(data)
    assert r["probe_error"] is None


@pytest.mark.parametrize("stderr", [
    "http://nas/x: Server returned 503 Service Unavailable",
    "Connection to tcp://nas:8000 failed: Connection refused",
    "http://nas/x: Input/output error",
])
def test_transport_failures_raise_and_are_never_verdicts(stderr):
    with pytest.raises(VP.ProbeTransportError):
        VP.run_ffprobe("u", ffprobe="x", run=lambda cmd, **kw:
                       subprocess.CompletedProcess(cmd, 1, "", stderr))


def test_timeout_is_transport():
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    with pytest.raises(VP.ProbeTransportError):
        VP.run_ffprobe("u", ffprobe="x", run=boom)


def test_unknown_failure_is_a_tool_error():
    with pytest.raises(VP.ProbeToolError):
        VP.run_ffprobe("u", ffprobe="x", run=lambda cmd, **kw:
                       subprocess.CompletedProcess(cmd, 1, "", "something odd"))


def test_missing_ffprobe_is_a_tool_error(monkeypatch):
    monkeypatch.setattr(VP, "find_ffprobe", lambda: None)
    with pytest.raises(VP.ProbeToolError):
        VP.run_ffprobe("u")


def test_ffprobe_command_reads_over_http_with_worker_headers():
    seen = {}

    def fake(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, json.dumps(_fx("kodak_pixpro")), "")
    out = VP.run_ffprobe("http://nas:8000/api/photos/5/full?purpose=worker",
                         ffprobe="ffprobe", run=fake)
    assert out["format"]["format_name"].startswith("mov")
    cmd = seen["cmd"]
    assert cmd[-1].endswith("/api/photos/5/full?purpose=worker")
    assert "-show_format" in cmd and "-show_streams" in cmd
    assert any("X-Photosearch-Source: worker" in c for c in cmd)


def test_no_video_stream_is_unreadable():
    data = _fx("generic_2016")
    data["streams"] = [s for s in data["streams"] if s["codec_type"] != "video"]
    r = VP.parse_probe(data)
    assert r["probe_error"] == "unreadable: no video stream"


def test_find_ffprobe_env_override(monkeypatch, tmp_path):
    exe = tmp_path / "ffprobe"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setenv("PHOTOSEARCH_FFPROBE", str(exe))
    assert VP.find_ffprobe() == str(exe)
    monkeypatch.setenv("PHOTOSEARCH_FFPROBE", str(tmp_path / "nope"))
    assert VP.find_ffprobe() is None


# ---------------------------------------------------------------------------
# Posters
# ---------------------------------------------------------------------------

def test_poster_jpegs_sizes_and_upload_body():
    np = pytest.importorskip("numpy")
    from PIL import Image
    import io
    frame = np.zeros((1440, 1080, 3), dtype=np.uint8)
    frame[:, :, 0] = 200
    jpegs = VP.poster_jpegs(frame)
    assert set(jpegs) == {"thumb", "preview", "full"}
    dims = {k: Image.open(io.BytesIO(v)).size for k, v in jpegs.items()}
    assert dims["thumb"] == (450, 600)            # long edge = web._THUMB_SIZE
    assert dims["preview"] == (1080, 1440)        # never upscaled
    assert dims["full"] == (1080, 1440)
    body = VP.poster_assets_body(jpegs)
    assert body["version"] == VP.PROBE_VERSION
    assert {a["size"] for a in body["assets"]} == {"thumb", "preview", "full"}
    assert all(a["kind"] == "poster" for a in body["assets"])


def test_poster_sizes_match_the_web_thumbnail_sizes():
    from photosearch import web
    assert VP.POSTER_SIZES["thumb"] == web._THUMB_SIZE
    assert VP.POSTER_SIZES["preview"] == web._PREVIEW_SIZE


# ---------------------------------------------------------------------------
# The worker's per-video flow (mocked transport, ffprobe and cv2)
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status=206, body=b"", text=""):
        self.status_code, self.text = status, text
        self.ok = 200 <= status < 300

        class _Raw:
            def read(self_inner, n):
                return body[:n]
        self.raw = _Raw()

    def close(self):
        pass


class _Client:
    server_url = "http://nas:8000"

    def __init__(self, head=b"\x00\x00\x00\x18ftypmp42", head_status=206, upload_status=200):
        self.head, self.head_status, self.upload_status = head, head_status, upload_status
        self.uploads = []

    def _request(self, method, url, **kw):
        if method == "GET":
            assert kw["headers"]["Range"] == "bytes=0-11"
            assert "purpose=worker" in url
            return _Resp(self.head_status, self.head)
        self.uploads.append((url, kw["json"]))
        return _Resp(self.upload_status, text="nope")


def _photos(*ids):
    return [{"id": i, "filename": f"V{i}.MOV", "filepath": f"x/V{i}.MOV"} for i in ids]


def _patch_tools(monkeypatch, data=None, frame=True):
    monkeypatch.setattr(VP, "run_ffprobe", lambda url: copy.deepcopy(data or _fx("apple_live_hevc_rot")))
    monkeypatch.setattr(VP, "grab_first_frame",
                        lambda url: (("FRAME", 0.0) if frame else (None, None)))
    monkeypatch.setattr(VP, "poster_jpegs",
                        lambda f: {"thumb": b"\xff\xd8\xffT", "preview": b"\xff\xd8\xffP",
                                   "full": b"\xff\xd8\xffF"})


def test_worker_ok_uploads_poster_then_returns_result(monkeypatch):
    from photosearch import worker as W
    _patch_tools(monkeypatch)
    c = _Client()
    out = W._process_video_probe(c, _photos(7))
    assert len(c.uploads) == 1 and c.uploads[0][0].endswith("/api/videos/7/assets")
    assert c.uploads[0][1]["version"] == VP.PROBE_VERSION
    (r,) = out
    assert r["photo_id"] == 7 and r["probe_version"] == VP.PROBE_VERSION
    assert r["date_taken"] == "2025-06-24 14:16:05" and r["poster_t"] == 0.0
    assert "error" not in r and r["probe_error"] is None
    # the worker never sends fields VideoProbeResult does not declare
    from photosearch.worker_api import VideoProbeResult
    VideoProbeResult(**r)
    assert set(r) <= set(VideoProbeResult.model_fields)


def test_worker_zip_is_a_verdict_and_never_probed(monkeypatch):
    from photosearch import worker as W
    monkeypatch.setattr(VP, "run_ffprobe", lambda url: pytest.fail("ffprobe on a ZIP"))
    c = _Client(head=b"PK\x03\x04\x14\x00rest")
    (r,) = W._process_video_probe(c, _photos(3))
    assert r["probe_error"] == VP.ZIP_WRAPPED and not c.uploads


def test_worker_unreadable_is_a_verdict_without_poster(monkeypatch):
    from photosearch import worker as W
    monkeypatch.setattr(VP, "run_ffprobe", lambda url: {"_verdict": "unreadable: moov atom not found"})
    monkeypatch.setattr(VP, "grab_first_frame", lambda url: pytest.fail("decoded a corrupt file"))
    c = _Client()
    (r,) = W._process_video_probe(c, _photos(3))
    assert r["probe_error"] == "unreadable: moov atom not found" and not c.uploads


def test_worker_transport_error_omits_the_video(monkeypatch):
    from photosearch import worker as W

    def boom(url):
        raise VP.ProbeTransportError("Server returned 503")
    monkeypatch.setattr(VP, "run_ffprobe", boom)
    assert W._process_video_probe(_Client(), _photos(3)) == []


def test_worker_404_and_rejected_poster_are_failure_rows(monkeypatch):
    from photosearch import worker as W
    _patch_tools(monkeypatch)
    (r,) = W._process_video_probe(_Client(head_status=404), _photos(3))
    assert W._is_failure_row(r)
    (r,) = W._process_video_probe(_Client(upload_status=400), _photos(4))
    assert W._is_failure_row(r) and "poster upload" in r["error"]


def test_worker_undecodable_first_frame_still_stores_the_probe(monkeypatch):
    from photosearch import worker as W
    _patch_tools(monkeypatch, frame=False)
    c = _Client()
    (r,) = W._process_video_probe(c, _photos(9))
    assert not c.uploads and r["probe_error"] is None and r["poster_t"] is None
    assert r["probe_json"]["poster_error"]


def test_short_clip_gets_a_poster_too(monkeypatch):
    """A <= 4 s clip may or may not pair with a still — that is decided
    server-side. An unpaired one (a 2 s Sony clip) must not be left posterless."""
    from photosearch import worker as W
    _patch_tools(monkeypatch, data=_fx("sony_ilce"))
    c = _Client()
    W._process_video_probe(c, _photos(1))
    assert len(c.uploads) == 1


def test_video_probe_claims_25():
    from photosearch import worker as W
    assert W.PASS_CLAIM_LIMITS["video-probe"] == 25

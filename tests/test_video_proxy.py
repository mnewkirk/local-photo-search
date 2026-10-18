"""The video-proxy pass (photosearch/video_proxy.py, D12 fallback): the
needs-a-proxy rule against real ffprobe shapes, the ffmpeg command builder
(HDR, rotation, interlacing, audio), output verification, the streaming
upload endpoint, Range serving, the claim predicate, the submit branch,
clear-pass, cleanup of stale files, the v39 migration and batch_state.

ffmpeg is never run here (CI has none): the transcode is exercised through an
injected `run`."""

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from photosearch import video_proxy as VPx
from photosearch import web, worker_api
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.video_probe import PROBE_VERSION, parse_probe
from photosearch.video_registry import register_video

FIX = Path(__file__).parent / "fixtures" / "video_proxy"
MP4 = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2avc1mp41" + b"\x00" * 4000
DIR = "2091/2091-10-18_phone-matt"


def _meta(name: str) -> dict:
    """A video_meta-shaped dict for one real fixture's (trimmed) ffprobe."""
    data = json.loads((FIX / f"{name}.json").read_text())
    r = parse_probe(data)
    return {"codec": r["codec"], "container": r["container"], "width": r["width"],
            "height": r["height"], "has_audio": r["has_audio"],
            "duration_s": r["duration_s"], "probe_json": data}


# ---------------------------------------------------------------------------
# needs_proxy — the measured playback matrix (owner's Chrome/Edge, 2026-10-18)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    # did NOT play
    ("hdr_live_short", True),       # HEVC Main10 HLG iPhone .mov
    ("ilce_8k_short", True),        # HEVC Main10 8K Sony .mp4
    ("canon_mvi", True),            # MPEG-4 ASP .avi
    ("sample", True),               # H.264 in .mts
    # played
    ("kodak_pixpro", False),        # H.264 yuv420p .mov
    ("iphone_portrait_rot", False), # H.264 yuvj420p, rotated, 1440x1080
    ("facebook_takeout", False),
    ("y2016_1080p_normal", False),
    ("y2016_720p", False),
])
def test_needs_proxy_matches_the_measured_browser(name, expected):
    assert VPx.needs_proxy(_meta(name)) is expected


def _synthetic(**kw):
    stream = {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p",
              "profile": "High", "level": 40}
    stream.update(kw.pop("stream", {}))
    meta = {"codec": "h264", "container": "mp4", "width": 1920, "height": 1080,
            "probe_json": {"streams": [stream]}}
    meta.update(kw)
    return meta


@pytest.mark.parametrize("kw,expected", [
    ({}, False),
    ({"container": "m4v"}, False),
    ({"container": "mov"}, False),
    ({"stream": {"pix_fmt": "yuvj420p"}}, False),
    ({"width": 3840, "height": 2160}, False),
    ({"width": 4096, "height": 2160}, True),          # over 3840
    ({"width": 2160, "height": 4096}, True),          # long edge, either axis
    ({"stream": {"pix_fmt": "yuv420p10le"}}, True),   # 10-bit h264
    ({"stream": {"pix_fmt": "yuv422p"}}, True),       # 4:2:2
    ({"container": "mpegts"}, True),
    ({"container": "avi"}, True),
    ({"codec": "hevc"}, True),
    ({"codec": "mpeg4"}, True),
    ({"probe_json": None}, True),                     # unknown pix_fmt -> proxy
    ({"width": None, "height": None}, True),          # unknown dims -> proxy
])
def test_needs_proxy_rule(kw, expected):
    assert VPx.needs_proxy(_synthetic(**kw)) is expected


def test_needs_proxy_reads_probe_json_text_and_sqlite_rows(tmp_path):
    meta = _synthetic(stream={"pix_fmt": "yuv420p10le"})
    meta["probe_json"] = json.dumps(meta["probe_json"])
    assert VPx.needs_proxy(meta) is True
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT 'h264' AS codec, 'mp4' AS container, 1280 AS width, "
                       "720 AS height, ? AS probe_json",
                       (json.dumps({"streams": [{"codec_type": "video",
                                                  "pix_fmt": "yuv420p"}]}),)).fetchone()
    assert VPx.needs_proxy(row) is False


def test_playback_info_hdr_and_depth():
    info = VPx.playback_info(_meta("hdr_live_short"))
    assert info["hdr"] is True and info["bit_depth"] == 10
    assert info["codec"] == "hevc" and info["profile"] == "Main 10"
    assert info["long_edge"] == 1920
    info = VPx.playback_info(_meta("kodak_pixpro"))
    assert info["hdr"] is False and info["bit_depth"] == 8


# ---------------------------------------------------------------------------
# The ffmpeg command
# ---------------------------------------------------------------------------

def _cmd(name, **kw):
    info = VPx.transcode_info(_meta(name))
    return VPx.build_transcode_cmd("ffmpeg", "/in", "/out.mp4", info, **kw)


def _vf(cmd):
    return cmd[cmd.index("-vf") + 1]


def test_cmd_hdr_is_tone_mapped_to_bt709():
    cmd = _cmd("hdr_live_short")
    vf = _vf(cmd)
    assert "zscale=tin=arib-std-b67:pin=bt2020:min=bt2020nc:t=linear" in vf
    assert "tonemap=tonemap=mobius" in vf
    assert vf.index("tonemap") < vf.index("format=yuv420p")
    assert vf.endswith("format=yuv420p")
    assert cmd[cmd.index("-colorspace") + 1] == "bt709"


def test_cmd_sdr_10bit_is_not_tone_mapped_but_is_8bit():
    vf = _vf(_cmd("ilce_8k_short"))      # bt709 HEVC Main10
    assert "tonemap" not in vf and "zscale" not in vf
    assert vf.endswith("format=yuv420p")


def test_cmd_scales_by_long_edge_without_upscaling():
    vf = _vf(_cmd("hdr_live_short", long_edge=1280))
    assert "min(1280,iw)" in vf and "min(1280,ih)" in vf and "-2" in vf


def test_cmd_rotation_is_baked_by_autorotate():
    # The rotated iPhone clip gets NO -noautorotate and no transpose of our
    # own: ffmpeg's default autorotate turns the frames upright before the
    # scale (which therefore sees display orientation) and writes no matrix.
    cmd = _cmd("hdr_live_short")
    assert "-noautorotate" not in cmd
    assert "transpose" not in _vf(cmd)


def test_cmd_audio_and_no_audio():
    cmd = _cmd("kodak_pixpro")
    assert cmd[cmd.index("-c:a") + 1] == "aac" and "128k" in cmd and "-ac" in cmd
    assert "0:a:0?" in cmd
    info = VPx.transcode_info(dict(_meta("kodak_pixpro"), has_audio=False))
    cmd = VPx.build_transcode_cmd("ffmpeg", "/in", "/out.mp4", info)
    assert "-an" in cmd and "-c:a" not in cmd and "0:a:0?" not in cmd


def test_cmd_h264_high_crf_faststart():
    cmd = _cmd("canon_mvi")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-profile:v") + 1] == "high"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-crf") + 1] == str(VPx.PROXY_CRF)
    assert cmd[cmd.index("-preset") + 1] == VPx.PROXY_PRESET
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd[-1] == "/out.mp4" and cmd[cmd.index("-i") + 1] == "/in"


def test_cmd_interlaced_mts_is_deinterlaced():
    assert _vf(_cmd("sample")).startswith("bwdif")
    assert "bwdif" not in _vf(_cmd("canon_mvi"))


def test_long_edge_env(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE", "961")
    assert VPx.proxy_long_edge() == 960
    assert "min(960,iw)" in _vf(_cmd("canon_mvi"))
    monkeypatch.setenv("PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE", "nope")
    assert VPx.proxy_long_edge() == VPx.DEFAULT_LONG_EDGE


def test_timeout_scales_with_duration():
    assert VPx.transcode_timeout(None) == VPx.TIMEOUT_BASE_S
    assert VPx.transcode_timeout(60) == VPx.TIMEOUT_BASE_S + 60 * VPx.TIMEOUT_PER_SECOND


# ---------------------------------------------------------------------------
# Output verification + transcode (fake run)
# ---------------------------------------------------------------------------

def _probe_out(codec="h264", pix="yuv420p", dur="12.6"):
    return {"format": {"duration": dur},
            "streams": [{"codec_type": "video", "codec_name": codec, "pix_fmt": pix,
                         "width": 1280, "height": 720}]}


def test_check_output_accepts_and_rejects():
    assert VPx.check_output(_probe_out(), 12.64) == \
        {"width": 1280, "height": 720, "duration_s": 12.6}
    with pytest.raises(VPx.ProxyToolError, match="codec"):
        VPx.check_output(_probe_out(codec="hevc"), 12.6)
    with pytest.raises(VPx.ProxyToolError, match="pix_fmt"):
        VPx.check_output(_probe_out(pix="yuv420p10le"), 12.6)
    with pytest.raises(VPx.ProxyToolError, match="source"):
        VPx.check_output(_probe_out(dur="10.0"), 12.6)
    with pytest.raises(VPx.ProxyToolError, match="no video"):
        VPx.check_output({"format": {}, "streams": []}, 1)


class _Proc:
    def __init__(self, rc=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = rc, stdout, stderr


def test_transcode_success_and_failures(tmp_path, monkeypatch):
    dst = tmp_path / "out.mp4"
    monkeypatch.setattr(VPx, "probe_local", lambda p, *a, **k: _probe_out(dur="12.6"))

    def ok(cmd, **kw):
        assert kw["timeout"] == VPx.transcode_timeout(12.638)
        Path(cmd[-1]).write_bytes(MP4)
        return _Proc()

    out = VPx.transcode("/in.avi", str(dst), _meta("canon_mvi"), ffmpeg="ffmpeg", run=ok)
    assert out["bytes"] == len(MP4) and out["width"] == 1280

    with pytest.raises(VPx.ProxyToolError, match="exit 1"):
        VPx.transcode("/in", str(tmp_path / "x.mp4"), _meta("canon_mvi"), ffmpeg="ffmpeg",
                      run=lambda c, **k: _Proc(1, stderr="Conversion failed!"))

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    with pytest.raises(VPx.ProxyToolError, match="timed out"):
        VPx.transcode("/in", str(tmp_path / "y.mp4"), _meta("canon_mvi"), ffmpeg="ffmpeg",
                      run=slow)
    monkeypatch.setenv("PHOTOSEARCH_FFMPEG", str(tmp_path / "missing-ffmpeg"))
    with pytest.raises(VPx.ProxyToolError, match="not found"):
        VPx.transcode("/in", str(tmp_path / "z.mp4"), _meta("canon_mvi"))


# ---------------------------------------------------------------------------
# Server: upload, serving, detail
# ---------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    db_path = str(tmp_path / "data" / "photo_index.db")
    os.makedirs(os.path.dirname(db_path))
    db = PhotoDB(db_path, photo_root=str(root))
    db.set_photo_root(str(root))
    monkeypatch.setattr(web, "_db_path", db_path)
    monkeypatch.setattr(web, "_photo_root", str(root))
    monkeypatch.setattr(web, "_nas_url", None)
    monkeypatch.setattr(worker_api, "_db_path", db_path)
    monkeypatch.setattr(worker_api, "_shutting_down", False)

    def video(name, meta_name=None, **cols):
        p = root / DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        vid = register_video(db, str(p), date_taken="2091-10-18 09:00:00")
        if meta_name:
            m = _meta(meta_name)
            db.conn.execute(
                "UPDATE video_meta SET probe_version = ?, kind = 'clip', codec = ?, "
                "container = ?, width = ?, height = ?, has_audio = ?, duration_s = ?, "
                "probe_json = ?, proxy_needed = ? WHERE photo_id = ?",
                (PROBE_VERSION, m["codec"], m["container"], m["width"], m["height"],
                 1 if m["has_audio"] else 0, m["duration_s"], json.dumps(m["probe_json"]),
                 1 if VPx.needs_proxy(m) else 0, vid))
        if cols:
            db.conn.execute(f"UPDATE video_meta SET {', '.join(f'{k} = ?' for k in cols)} "
                            f"WHERE photo_id = ?", (*cols.values(), vid))
        db.conn.commit()
        return vid

    yield {"db": db, "client": TestClient(web.app), "video": video,
           "db_path": db_path, "root": root}
    db.close()


def _put(c, vid, body=MP4, version=1, **params):
    q = "&".join([f"version={version}"] + [f"{k}={v}" for k, v in params.items()])
    return c.put(f"/api/videos/{vid}/proxy?{q}", content=body,
                 headers={"Content-Type": "video/mp4"})


def test_upload_streams_atomically_and_records(env):
    db, c = env["db"], env["client"]
    vid = env["video"]("IMG_1.MOV", "hdr_live_short")
    r = _put(c, vid, w=720, h=1280)
    assert r.status_code == 200, r.text
    path = VPx.proxy_path(env["db_path"], vid, 1)
    assert path.read_bytes() == MP4
    assert not [f for f in path.parent.iterdir() if f.name.endswith(".tmp")]
    row = db.conn.execute("SELECT proxy_version, proxy_bytes, proxy_w, proxy_h, proxy_at "
                          "FROM video_meta WHERE photo_id = ?", (vid,)).fetchone()
    assert tuple(row)[:4] == (1, len(MP4), 720, 1280) and row["proxy_at"]


def test_upload_rejects_bad_magic_oversize_and_non_video(env, monkeypatch):
    db, c = env["db"], env["client"]
    vid = env["video"]("IMG_2.MOV", "hdr_live_short")
    r = _put(c, vid, body=b"\xff\xd8\xff\xe0 not an mp4" * 10)
    assert r.status_code == 400 and "ftyp" in r.text
    monkeypatch.setattr(VPx, "MAX_PROXY_BYTES", 1000)
    r = _put(c, vid)
    assert r.status_code == 413
    root = VPx.proxies_root(env["db_path"])
    assert not any(root.iterdir())                     # nothing left behind
    assert db.conn.execute("SELECT proxy_version FROM video_meta WHERE photo_id = ?",
                           (vid,)).fetchone()[0] is None
    still = db.add_photo(filepath=f"{DIR}/IMG_9.JPG", filename="IMG_9.JPG")
    assert _put(c, still).status_code == 404
    assert _put(c, 999999).status_code == 404
    assert _put(c, vid, version=0).status_code == 422


def test_upload_refused_on_replica_and_503_during_shutdown(env, monkeypatch):
    vid = env["video"]("IMG_3.MOV", "hdr_live_short")
    monkeypatch.setattr(web, "_nas_url", "http://nas:8000")
    assert _put(env["client"], vid).status_code == 409
    monkeypatch.setattr(web, "_nas_url", None)
    monkeypatch.setattr(worker_api, "_shutting_down", True)
    r = _put(env["client"], vid)
    assert r.status_code == 503 and r.headers["retry-after"]


def test_serve_proxy_with_range(env):
    c = env["client"]
    vid = env["video"]("IMG_4.MOV", "hdr_live_short")
    assert c.get(f"/api/videos/{vid}/proxy").status_code == 404
    assert _put(c, vid).status_code == 200
    r = c.get(f"/api/videos/{vid}/proxy?v=1")
    assert r.status_code == 200 and r.content == MP4
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes"
    assert "max-age" in r.headers["cache-control"]
    r = c.get(f"/api/videos/{vid}/proxy", headers={"Range": "bytes=4-11"})
    assert r.status_code == 206 and r.content == MP4[4:12]
    assert r.headers["content-range"] == f"bytes 4-11/{len(MP4)}"
    r = c.get(f"/api/videos/{vid}/proxy", headers={"Range": f"bytes={len(MP4)}-"})
    assert r.status_code == 416


def test_replica_streams_the_proxy_from_the_nas(env, monkeypatch):
    import requests
    vid = env["video"]("IMG_5.MOV", "hdr_live_short", proxy_version=1)
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")
    seen = {}

    class Up:
        status_code = 206
        headers = {"Content-Type": "video/mp4", "Content-Range": "bytes 0-2/9",
                   "Content-Length": "3", "Accept-Ranges": "bytes"}

        def iter_content(self, chunk_size=None):
            yield b"abc"

        def close(self):
            seen["closed"] = True

    def fake_get(url, params=None, headers=None, stream=False, timeout=None):
        seen.update(url=url, params=params, headers=headers)
        return Up()

    monkeypatch.setattr(requests, "get", fake_get)
    r = env["client"].get(f"/api/videos/{vid}/proxy?v=1", headers={"Range": "bytes=0-2"})
    assert r.status_code == 206 and r.content == b"abc"
    assert seen["url"] == f"http://fake-nas:8000/api/videos/{vid}/proxy"
    assert seen["params"] == {"v": 1}
    assert seen["headers"]["Range"] == "bytes=0-2"
    assert seen["headers"]["X-Photosearch-Intent"]
    assert seen["closed"]


def test_detail_carries_proxy_and_playback(env):
    c = env["client"]
    hdr = env["video"]("IMG_6.MOV", "hdr_live_short")
    ok = env["video"]("KODAK.MOV", "kodak_pixpro")
    v = c.get(f"/api/photos/{hdr}").json()["video"]
    assert v["needs_proxy"] is True and v["proxy_url"] is None and v["proxy"] is None
    assert v["playback"]["codec"] == "hevc" and v["playback"]["hdr"] is True
    assert "probe_json" not in v
    _put(c, hdr, w=720, h=1280)
    v = c.get(f"/api/photos/{hdr}").json()["video"]
    assert v["proxy_url"] == f"/api/videos/{hdr}/proxy?v=1"
    assert v["proxy"] == {"version": 1, "width": 720, "height": 1280, "bytes": len(MP4)}
    v = c.get(f"/api/photos/{ok}").json()["video"]
    assert v["needs_proxy"] is False and v["playback"]["pix_fmt"] == "yuv420p"


def test_request_intents_for_proxy_routes():
    from photosearch.request_intent import infer_intent
    assert "upload the playable" in infer_intent("PUT", "/api/videos/7/proxy", "")
    assert "Play the H.264 copy of video 7" in infer_intent("GET", "/api/videos/7/proxy", "")


# ---------------------------------------------------------------------------
# Claim, submit, clear-pass
# ---------------------------------------------------------------------------

def _claimable(db):
    return {r["id"] for r in db.get_unprocessed_photos("video-proxy", limit=100)}


def test_claim_predicate(env):
    db = env["db"]
    hdr = env["video"]("A.MOV", "hdr_live_short")
    avi = env["video"]("B.AVI", "canon_mvi")
    ok = env["video"]("C.MOV", "kodak_pixpro")                       # plays as-is
    live = env["video"]("D.MOV", "hdr_live_short", kind="live")       # never analysed
    zipw = env["video"]("E.MOV", "hdr_live_short", probe_error="zip_wrapped")
    bad = env["video"]("F.MOV", "hdr_live_short", probe_error="unreadable: x")
    unprobed = env["video"]("G.MOV")
    done = env["video"]("H.MOV", "hdr_live_short", proxy_version=VPx.PROXY_VERSION)
    stale = env["video"]("I.MOV", "hdr_live_short", proxy_version=VPx.PROXY_VERSION - 1
                         if VPx.PROXY_VERSION > 1 else None)
    got = _claimable(db)
    assert got == {hdr, avi, stale}
    assert db.count_unprocessed_photos("video-proxy") == 3
    assert db.count_unprocessed_photos("video-proxy", photo_ids=[hdr, ok, done]) == 1
    for pid in (ok, live, zipw, bad, unprobed, done):
        assert pid not in got
    # attempts cap
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'video-proxy', ?)", (avi, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    assert avi not in _claimable(db)


def test_claim_response_carries_proxy_input(env):
    vid = env["video"]("A.MOV", "hdr_live_short")
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-proxy", "limit": 1}).json()
    assert [p["id"] for p in r["photos"]] == [vid]
    pi = r["photos"][0]["proxy_input"]
    assert pi["codec"] == "hevc" and pi["has_audio"] == 1
    assert VPx.transcode_info(pi)["hdr"] is True


def _submit(c, results=None, failures=None):
    body = {"batch_id": "b-1", "pass_type": "video-proxy",
            "video_proxy_results": results or []}
    if failures:
        body["failures"] = failures
    r = c.post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_submit_confirms_the_upload_and_writes_no_ledger(env):
    db, c = env["db"], env["client"]
    vid = env["video"]("A.MOV", "hdr_live_short")
    assert _put(c, vid).status_code == 200
    out = _submit(c, [{"photo_id": vid, "proxy_version": VPx.PROXY_VERSION}])
    assert out["written"] == 1 and out["processed"] == 1
    assert vid not in _claimable(db)
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed WHERE "
                           "pass_type = 'video-proxy'").fetchone()[0] == 0


def test_submit_without_upload_and_failures_spend_attempts(env):
    db, c = env["db"], env["client"]
    vid = env["video"]("A.MOV", "hdr_live_short")
    out = _submit(c, [{"photo_id": vid, "proxy_version": VPx.PROXY_VERSION}])
    assert out["written"] == 0 and out["processed"] == 1
    for _ in range(MAX_PROCESS_ATTEMPTS - 1):
        _submit(c, failures=[{"photo_id": vid, "error": "ffmpeg exit 1"}])
    assert vid not in _claimable(db)                      # three strikes
    err = db.conn.execute("SELECT message FROM index_errors WHERE pass_type = "
                          "'video-proxy' ORDER BY id DESC").fetchone()[0]
    assert "ffmpeg exit 1" in err


def test_clear_pass_requeues(env):
    db, c = env["db"], env["client"]
    vid = env["video"]("A.MOV", "hdr_live_short")
    _put(c, vid)
    _submit(c, [{"photo_id": vid, "proxy_version": VPx.PROXY_VERSION}])
    assert vid not in _claimable(db)
    r = c.post("/api/worker/clear-pass", json={"pass_type": "video-proxy",
                                               "photo_ids": [vid]})
    assert r.status_code == 200 and r.json()["cleared"] == 1
    assert vid in _claimable(db)
    # ...and the file nothing points at any more is stale
    stale = VPx.stale_proxy_files(db.conn, VPx.proxies_root(env["db_path"]))
    assert [f.name for f in stale] == [f"{vid}.v1.mp4"]


def test_stale_proxy_files(env, tmp_path):
    db = env["db"]
    a = env["video"]("A.MOV", "hdr_live_short", proxy_version=2)
    root = VPx.proxies_root(env["db_path"])
    root.mkdir(parents=True, exist_ok=True)
    for name in (f"{a}.v2.mp4", f"{a}.v1.mp4", "424242.v1.mp4", "README"):
        (root / name).write_bytes(b"x")
    old_tmp = root / f".{a}.v2.mp4.1.2.tmp"
    old_tmp.write_bytes(b"x")
    os.utime(old_tmp, (1, 1))
    new_tmp = root / f".{a}.v3.mp4.1.3.tmp"
    new_tmp.write_bytes(b"x")
    names = sorted(f.name for f in VPx.stale_proxy_files(db.conn, root))
    assert names == sorted([old_tmp.name, f"{a}.v1.mp4", "424242.v1.mp4"])


def test_probe_store_decides_proxy_needed(env):
    from photosearch.video_probe_store import apply_probe_result
    db = env["db"]
    for name, expected in (("hdr_live_short", 1), ("kodak_pixpro", 0), ("sample", 1)):
        vid = env["video"](f"{name}.MOV")
        r = parse_probe(json.loads((FIX / f"{name}.json").read_text()))
        r.update(photo_id=vid, probe_version=PROBE_VERSION)
        apply_probe_result(db.conn, r)
        db.conn.commit()
        got = db.conn.execute("SELECT proxy_needed FROM video_meta WHERE photo_id = ?",
                              (vid,)).fetchone()[0]
        assert got == expected, name
    # a probe ERROR leaves no decision
    vid = env["video"]("bad.MOV", "hdr_live_short")
    apply_probe_result(db.conn, {"photo_id": vid, "probe_version": PROBE_VERSION,
                                 "probe_error": "unreadable: moov atom not found"})
    assert db.conn.execute("SELECT proxy_needed FROM video_meta WHERE photo_id = ?",
                           (vid,)).fetchone()[0] is None


def test_scan_of_a_zip_marks_it_needing_a_proxy(env):
    """video-scan clears a ZIP's probe_error by applying the inner probe; the
    file on the NAS is still the ZIP, so it needs a proxy whatever the inner
    codec (here h264 that would otherwise play)."""
    from photosearch.video_scan_store import apply_scan_result
    db = env["db"]
    vid = env["video"]("Z.MOV", "kodak_pixpro", probe_error="zip_wrapped")
    inner = parse_probe(json.loads((FIX / "kodak_pixpro.json").read_text()))
    apply_scan_result(db.conn, {"photo_id": vid, "scan_error": "unreadable: test",
                                "probe": inner})
    db.conn.commit()
    row = db.conn.execute("SELECT probe_error, proxy_needed FROM video_meta "
                          "WHERE photo_id = ?", (vid,)).fetchone()
    assert tuple(row) == (None, 1)
    assert vid in _claimable(db)


def test_v38_db_migrates_to_v39_proxy_columns(tmp_path):
    from photosearch.db import SCHEMA_VERSION
    path = str(tmp_path / "v38.db")
    with PhotoDB(path) as db:
        hdr = db.add_photo(filepath="v/a.mov", filename="a.mov", media_type="video")
        ok = db.add_photo(filepath="v/b.mov", filename="b.mov", media_type="video")
        bad = db.add_photo(filepath="v/c.mov", filename="c.mov", media_type="video")
        for pid, name, err in ((hdr, "hdr_live_short", None), (ok, "kodak_pixpro", None),
                               (bad, "kodak_pixpro", "unreadable: x")):
            m = _meta(name)
            db.conn.execute(
                "INSERT INTO video_meta (photo_id, probe_version, probe_error, codec, "
                "container, width, height, probe_json) VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
                (pid, err, m["codec"], m["container"], m["width"], m["height"],
                 json.dumps(m["probe_json"])))
        db.conn.commit()
    conn = sqlite3.connect(path)
    for col in ("proxy_needed", "proxy_version", "proxy_bytes", "proxy_w", "proxy_h",
                "proxy_at"):
        conn.execute(f"ALTER TABLE video_meta DROP COLUMN {col}")
    conn.execute("UPDATE schema_info SET value = '38' WHERE key = 'version'")
    conn.commit()
    conn.close()
    with PhotoDB(path) as db:
        got = {r[0]: r[1] for r in db.conn.execute(
            "SELECT photo_id, proxy_needed FROM video_meta")}
        assert got == {hdr: 1, ok: 0, bad: None}
        assert db.conn.execute("SELECT proxy_version FROM video_meta LIMIT 1").fetchone()[0] is None
        v = db.conn.execute("SELECT value FROM schema_info WHERE key = 'version'").fetchone()[0]
        assert int(v) == SCHEMA_VERSION == 39


# ---------------------------------------------------------------------------
# batch_state
# ---------------------------------------------------------------------------

def test_batch_state_video_proxy_step(env):
    from photosearch.batch_state import batch_state
    from photosearch.ingest_batches import register_batch
    db, c = env["db"], env["client"]
    hdr = env["video"]("A.MOV")
    ok = env["video"]("B.MOV", "kodak_pixpro")
    bid = register_batch(db, DIR)

    def step():
        return next(s for s in batch_state(db, bid)["steps"] if s["step"] == "video-proxy")

    s = step()
    assert s["state"] == "waiting" and s["waiting_on"] == "video-probe"
    m = _meta("hdr_live_short")
    db.conn.execute("UPDATE video_meta SET probe_version = ?, kind = 'clip', codec = 'hevc', "
                    "container = 'mov', width = 1080, height = 1920, probe_json = ?, "
                    "proxy_needed = 1 WHERE photo_id = ?",
                    (PROBE_VERSION, json.dumps(m["probe_json"]), hdr))
    db.conn.commit()
    s = step()
    assert s["state"] == "needs_queue" and s["remaining"] == 1 and s["done"] == 1
    assert s["detail"] == "1 play as-is"
    _put(c, hdr)
    s = step()
    assert s["state"] == "completed" and s["detail"] == "1 proxied, 1 play as-is"
    # three failures read blocked, not needs_queue
    db.conn.execute("UPDATE video_meta SET proxy_version = NULL WHERE photo_id = ?", (hdr,))
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'video-proxy', ?)", (hdr, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    s = step()
    assert s["state"] == "blocked" and s["failed"] == 1
    assert ok


# ---------------------------------------------------------------------------
# The worker
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, status=200, text="", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def close(self):
        pass


class _Session:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.bodies = []

    def put(self, url, params=None, data=None, timeout=None, headers=None):
        self.bodies.append((url, params, data.read()))
        return _Resp(self.statuses.pop(0), headers={"Retry-After": "1"})


class _Client:
    server_url = "http://nas"

    def __init__(self, statuses=(200,)):
        self.session = _Session(statuses)


def _worker_setup(monkeypatch, tmp_path, transcode=None, download=None):
    from photosearch import worker as W

    def dl(client, pid, dest):
        if download:
            return download(client, pid, dest)
        Path(dest).write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 64)
        return {"status": 200, "sha256": "x", "bytes": 80}

    def tc(src, dst, meta):
        if transcode:
            return transcode(src, dst, meta)
        Path(dst).write_bytes(MP4)
        return {"width": 720, "height": 1280, "duration_s": 3.0, "bytes": len(MP4),
                "seconds": 0.5}

    monkeypatch.setattr(W, "_download_hashed", dl)
    monkeypatch.setattr(VPx, "transcode", tc)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    return W


def test_worker_success_uploads_and_returns_a_result(monkeypatch, tmp_path):
    W = _worker_setup(monkeypatch, tmp_path)
    client = _Client(statuses=(503, 200))          # one back-off, then stored
    photos = [{"id": 7, "filename": "IMG.MOV", "proxy_input": _meta("hdr_live_short")}]
    out = W._process_video_proxy(client, photos, str(tmp_path))
    assert out == [{"photo_id": 7, "proxy_version": VPx.PROXY_VERSION, "bytes": len(MP4),
                    "width": 720, "height": 1280, "seconds": 0.5}]
    assert len(client.session.bodies) == 2
    url, params, body = client.session.bodies[-1]
    assert url == "http://nas/api/videos/7/proxy" and body == MP4   # reopened, full body
    assert params == {"version": VPx.PROXY_VERSION, "w": 720, "h": 1280}
    assert not list(tmp_path.iterdir())             # temp dir cleaned


def test_worker_ffmpeg_failure_is_a_failure_row(monkeypatch, tmp_path):
    def boom(src, dst, meta):
        raise VPx.ProxyToolError("ffmpeg exit 1: Invalid data")
    W = _worker_setup(monkeypatch, tmp_path, transcode=boom)
    out = W._process_video_proxy(_Client(), [{"id": 7, "proxy_input": _meta("canon_mvi")}],
                                 str(tmp_path))
    assert out == [{"photo_id": 7, "error": "ffmpeg exit 1: Invalid data"}]


def test_worker_transport_is_omitted(monkeypatch, tmp_path):
    import requests

    def drop(client, pid, dest):
        raise requests.exceptions.ConnectionError("reset")
    W = _worker_setup(monkeypatch, tmp_path, download=drop)
    out = W._process_video_proxy(_Client(), [{"id": 7, "proxy_input": _meta("canon_mvi")}],
                                 str(tmp_path))
    assert out == []


def test_worker_upload_rejected_and_missing_input(monkeypatch, tmp_path):
    W = _worker_setup(monkeypatch, tmp_path)
    out = W._process_video_proxy(_Client(statuses=(400,)),
                                 [{"id": 7, "proxy_input": _meta("canon_mvi")},
                                  {"id": 8, "proxy_input": None}], str(tmp_path))
    assert [r["photo_id"] for r in out] == [7, 8]
    assert "HTTP 400" in out[0]["error"] and "proxy_input" in out[1]["error"]


def test_worker_drops_the_pass_without_ffmpeg(monkeypatch, capsys):
    from photosearch import worker as W
    monkeypatch.setenv("PHOTOSEARCH_FFMPEG", "/nonexistent/ffmpeg")
    W.run_worker("http://nas", ["video-proxy"])
    assert "video-proxy needs ffmpeg" in capsys.readouterr().err


def test_pass_lists_place_proxy_after_probe():
    from photosearch import rerun
    from photosearch.batch_state import DEPENDS_ON, WORKER_PASSES
    for lst in (WORKER_PASSES, worker_api._ALL_PASSES, rerun.ALL_PASSES):
        assert lst.index("video-proxy") == lst.index("video-probe") + 1
    assert DEPENDS_ON["video-proxy"] == "video-probe"
    from photosearch import worker as W
    assert W.PASS_CLAIM_LIMITS["video-proxy"] == 1 and "video-proxy" in W.SELF_READ_PASSES

"""video_decode: OpenCV returns BLACK (or a stale buffer) for interlaced
H.264 in MPEG-TS (AVCHD .m2ts) instead of failing, so the decode falls back
to an ffmpeg rawvideo pipe — and the targeted re-run CLI
(`video-redecode-black`). cv2 and ffmpeg are mocked: the decision logic is
what CI pins; tests/test_video_decode_real.py-style checks run locally."""

import json
import sqlite3

import numpy as np
import pytest
from click.testing import CliRunner

from photosearch import video_decode as VD
from photosearch.video_describe_store import requeue_describe
from photosearch.video_scan_store import SCAN_VERSION, black_decode_candidates
from test_video_scan_submit import _meta, env  # noqa: F401


# --- choose_backend -----------------------------------------------------------

@pytest.mark.parametrize("opened,cv2m,ffm,inter,want", [
    (True, [0.0, 0.0, 120.0], None, False, "cv2"),          # any lit frame: cv2 works
    (True, [0.0] * 12, [200.0], False, "ffmpeg"),           # cv2 black, ffmpeg lit
    (True, [0.0] * 12, [0.0, 0.0, 0.0, 0.0], False, "cv2"),  # genuinely black clip
    (True, [0.0] * 12, None, False, "unconfirmed"),         # no ffmpeg to confirm
    (True, [0.0] * 12, [None, None], False, "cv2"),         # ffmpeg fails: keep cv2
    (False, [], [180.0], False, "ffmpeg"),                  # cv2 cannot open
    (False, [], [None], False, "unreadable"),
    (False, [], None, False, "unreadable"),
    (True, [], [None], False, "unreadable"),                # cv2 opens, no frames
    (True, [], [5.0], False, "ffmpeg"),                     # ffmpeg's dark frame beats none
    # Interlaced: cv2 is never trusted, even when its frame looks lit (a
    # stale buffer from the previous video is exactly that).
    (True, [130.0], [0.0], True, "ffmpeg"),
    (True, [130.0], None, True, "unconfirmed"),
    (True, [130.0], [None], True, "unreadable"),
])
def test_choose_backend(opened, cv2m, ffm, inter, want):
    assert VD.choose_backend(opened, cv2m, ffm, inter) == want


def test_is_interlaced():
    assert VD.is_interlaced("tt") and VD.is_interlaced("BB")
    assert not VD.is_interlaced("progressive") and not VD.is_interlaced(None)


def test_frame_is_black():
    assert VD.frame_is_black(None)
    assert VD.frame_is_black(np.full((4, 4, 3), VD.BLACK_MAX - 1, np.uint8))
    assert not VD.frame_is_black(np.full((4, 4, 3), VD.BLACK_MAX, np.uint8))


# --- open_video ----------------------------------------------------------------

class _Cap:
    def release(self):
        self.released = True


@pytest.fixture
def stub(monkeypatch):
    """Drive open_video with canned cv2 / ffprobe / ffmpeg answers."""
    st = {"head": [], "opened": True, "fo": None, "ff": {}, "ff_calls": [], "exe": "/x/ffmpeg"}
    monkeypatch.setattr(VD, "ffprobe_basics",
                        lambda p: {"fps": 30.0, "duration": 20.0, "field_order": st["fo"]})
    monkeypatch.setattr(VD, "_cv2_head_maxes", lambda p: list(st["head"]))
    monkeypatch.setattr(VD, "_cv2_open", lambda p: _Cap() if st["opened"] else None)
    monkeypatch.setattr(VD, "_cv2_props", lambda cap: (30.0, 600.0))
    monkeypatch.setattr(VD, "find_ffmpeg", lambda: st["exe"])

    def ff(path, t, exe=None):
        st["ff_calls"].append(t)
        v = st["ff"].get(t, st["ff"].get("*"))
        return None if v is None else np.full((4, 6, 3), v, np.uint8)
    monkeypatch.setattr(VD, "ffmpeg_frame", ff)
    return st


def test_open_video_normal_clip_never_calls_ffmpeg(stub):
    stub["head"] = [90.0]
    src = VD.open_video("v.mp4")
    assert src.backend == "cv2" and stub["ff_calls"] == []


def test_open_video_black_cv2_falls_back_to_ffmpeg(stub):
    stub["head"] = [0.0] * VD.CHECK_FRAMES
    stub["ff"] = {"*": 140}
    msgs = []
    src = VD.open_video("v.m2ts", log=msgs.append)
    assert src.backend == "ffmpeg" and (src.width, src.height) == (6, 4)
    assert stub["ff_calls"] == [0.0]          # stops at the first lit frame
    assert "black" in msgs[0]


def test_open_video_genuinely_black_clip_stays_on_cv2(stub):
    stub["head"] = [0.0] * VD.CHECK_FRAMES
    stub["ff"] = {"*": 0}
    src = VD.open_video("black.mp4", duration=20.0)
    assert src.backend == "cv2"
    assert stub["ff_calls"] == [0.0, 5.0, 10.0, 15.0]   # confirmed across the clip


def test_open_video_interlaced_goes_straight_to_ffmpeg(stub):
    stub["fo"] = "tt"
    stub["head"] = [130.0]                    # a stale buffer must not win
    stub["ff"] = {"*": 0}                     # even a black first frame: ffmpeg
    src = VD.open_video("v.m2ts")
    assert src.backend == "ffmpeg" and stub["ff_calls"] == [0.0]


def test_open_video_interlaced_without_ffmpeg_raises_tool_error(stub):
    stub["fo"], stub["exe"] = "bb", None
    with pytest.raises(VD.DecodeToolError):
        VD.open_video("v.m2ts")


def test_open_video_unreadable(stub):
    stub["opened"] = False
    stub["ff"] = {}
    with pytest.raises(VD.DecodeOpenError):
        VD.open_video("garbage.mov")


# --- FfmpegSource ----------------------------------------------------------------

class _Proc:
    def __init__(self, data):
        import io
        self.stdout = io.BytesIO(data)
        self.returncode = None

    def poll(self):
        return self.returncode

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_ffmpeg_source_streams_frames_with_sample_times():
    w, h = 3, 2
    frames = [np.full((h, w, 3), v, np.uint8) for v in (10, 20, 30)]
    seen = {}

    def popen(cmd, **kw):
        seen["cmd"] = cmd
        return _Proc(b"".join(f.tobytes() for f in frames) + b"\x01\x02")  # + a torn tail

    src = VD.FfmpegSource("in.m2ts", "/x/ffmpeg", w, h, fps=30.0, nframes=90, popen=popen)
    out = list(src.samples(4.0))
    assert [round(t, 3) for t, _, _ in out] == [0.0, 0.25, 0.5]
    assert [n for _, n, _ in out] == [0, 8, 15]
    assert [int(f[0, 0, 0]) for _, _, f in out] == [10, 20, 30]
    vf = seen["cmd"][seen["cmd"].index("-vf") + 1]
    assert "yadif=deint=interlaced" in vf and "fps=fps=4" in vf and "scale=3:2" in vf
    assert seen["cmd"][-5:] == ["-f", "rawvideo", "-pix_fmt", "bgr24", "-"]


def test_ffmpeg_source_with_no_frames_is_unreadable():
    src = VD.FfmpegSource("x", "/x/ffmpeg", 2, 2, 30.0, 0, popen=lambda c, **k: _Proc(b""))
    with pytest.raises(VD.DecodeOpenError):
        list(src.samples(4.0))


# --- grab_frame -------------------------------------------------------------------

def test_grab_frame_interlaced_never_touches_cv2(monkeypatch):
    monkeypatch.setattr(VD, "ffmpeg_frame", lambda p, t, exe=None: np.full((2, 2, 3), 99, np.uint8))
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: pytest.fail("cv2 used on interlaced"),
                        raising=False)
    frame, t = VD.grab_frame("http://127.0.0.1:8000/x", 0.0, seek=False, interlaced=True)
    assert int(frame.max()) == 99 and t == 0.0


def test_probe_poster_passes_the_interlace_verdict(monkeypatch):
    from photosearch import video_probe as VP
    seen = {}

    def grab(url, t=0.0, *, seek=True, interlaced=None):
        seen["interlaced"] = interlaced
        return np.zeros((2, 2, 3), np.uint8), 0.0
    monkeypatch.setattr(VD, "grab_frame", grab)
    data = {"streams": [{"codec_type": "audio"},
                        {"codec_type": "video", "field_order": "tt"}]}
    assert VP.video_stream_interlaced(data)
    VP.grab_first_frame("u", interlaced=VP.video_stream_interlaced(data))
    assert seen["interlaced"] is True
    assert not VP.video_stream_interlaced({"streams": [{"codec_type": "video",
                                                        "field_order": "progressive"}]})


# --- video-redecode-black ----------------------------------------------------------

def _probe(fo):
    return json.dumps({"format": {}, "streams": [{"codec_type": "video", "field_order": fo}]})


def _three(env):
    db = env["db"]
    m2ts = env["video"]("00012.M2TS")
    m2ts_lit = env["video"]("00013.MTS")
    mp4 = env["video"]("C.MP4")
    live = env["video"]("L.MOV", kind="live")
    for pid, fo, subj in ((m2ts, "tt", []), (m2ts_lit, "bb", [{"t": 1}]),
                          (mp4, "progressive", []), (live, "tt", [])):
        db.conn.execute(
            "UPDATE video_meta SET probe_json=?, scan_version=?, subjects_version=1, "
            "subjects_json=? WHERE photo_id=?",
            (_probe(fo), SCAN_VERSION, json.dumps({"subjects": subj}), pid))
    db.conn.execute("UPDATE photos SET description='a dark room', categories='[\"x\"]', "
                    "keywords='[\"y\"]' WHERE id IN (?, ?, ?)", (m2ts, m2ts_lit, mp4))
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'keywords', 1)", (m2ts,))
    db.conn.commit()
    return m2ts, m2ts_lit, mp4, live


def test_candidates_are_interlaced_scanned_non_live(env):
    m2ts, m2ts_lit, mp4, live = _three(env)
    conn = sqlite3.connect(f"file:{env['db_path']}?mode=ro", uri=True)
    try:
        info = black_decode_candidates(conn)
        assert info["photo_ids"] == [m2ts, m2ts_lit]
        assert info["by_ext"]["m2ts"]["subjects_empty"] == 1
        assert info["by_ext"]["mts"]["subjects_empty"] == 0
        assert black_decode_candidates(conn, empty_only=True)["photo_ids"] == [m2ts]
        assert black_decode_candidates(conn, ["mp4"])["photo_ids"] == [m2ts, m2ts_lit, mp4]
    finally:
        conn.close()


def test_redecode_cli_dry_run_writes_nothing(env):
    import cli
    m2ts, *_ = _three(env)
    before = _meta(env["db"], m2ts)
    r = CliRunner().invoke(cli.cli, ["video-redecode-black", "--db", env["db_path"]])
    assert r.exit_code == 0, r.output
    assert "m2ts" in r.output and "total        2" in r.output and "Dry run" in r.output
    assert _meta(env["db"], m2ts) == before


def test_redecode_cli_apply_requeues_scan_and_describe(env):
    import cli
    db = env["db"]
    m2ts, m2ts_lit, mp4, live = _three(env)
    r = CliRunner().invoke(cli.cli, ["video-redecode-black", "--db", env["db_path"],
                                     "--save-collection", "black", "--apply"])
    assert r.exit_code == 0, r.output
    for pid in (m2ts, m2ts_lit):
        assert _meta(db, pid)["scan_version"] is None
        row = db.conn.execute("SELECT description, categories, keywords FROM photos "
                              "WHERE id=?", (pid,)).fetchone()
        assert tuple(row) == (None, None, None)
    assert _meta(db, mp4)["scan_version"] == SCAN_VERSION
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (mp4,)).fetchone()[0] == "a dark room"
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed WHERE photo_id=? AND "
                           "pass_type='keywords'", (m2ts,)).fetchone()[0] == 0
    assert sorted(p["id"] for p in db.get_unprocessed_photos("video-scan", limit=9)) == \
        [m2ts, m2ts_lit]
    c = db.get_collection_by_name("black")
    assert sorted(r[0] for r in db.conn.execute(
        "SELECT photo_id FROM collection_photos WHERE collection_id=?", (c["id"],))) == \
        [m2ts, m2ts_lit]


def test_requeue_describe_of_no_ids_touches_nothing(env):
    m2ts, *_ = _three(env)
    assert requeue_describe(env["db"].conn, []) == 0

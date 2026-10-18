"""Video serving (docs/plans/video-support.md step 2): posters/placeholders,
Range + mime on /full, the rate-capped worker stream, the replica streaming
proxy, the worker asset upload, the /api/search Media param and the
unreadable-videos list."""

import base64
import os

import pytest
from fastapi.testclient import TestClient

from photosearch import video_api, web, worker_api
from photosearch.db import PhotoDB
from photosearch.video_registry import register_video

JPEG = b"\xff\xd8\xff\xe0" + b"poster" * 10
VIDEO_BYTES = bytes(range(256)) * 40  # 10,240 bytes, every offset identifiable


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    (root / "2026" / "d").mkdir(parents=True)
    db_path = str(tmp_path / "data" / "photo_index.db")
    os.makedirs(os.path.dirname(db_path))
    db = PhotoDB(db_path, photo_root=str(root))
    db.set_photo_root(str(root))

    still = db.add_photo(filepath="2026/d/IMG_1.JPG", filename="IMG_1.JPG",
                         date_taken="2026-09-26 10:00:00")
    ids = {"still": still}
    for key, name in (("clip", "CLIP.MP4"), ("mov", "IPHONE.MOV"),
                      ("avi", "OLD.AVI"), ("mts", "CAM.MTS"),
                      ("live", "IMG_1.MOV"), ("bad", "BAD.MP4"),
                      ("new", "NEW.MP4")):
        path = root / "2026" / "d" / name
        path.write_bytes(VIDEO_BYTES)
        ids[key] = register_video(db, str(path), date_taken="2026-09-26 10:00:00")
    for key in ("clip", "mov", "avi", "mts"):
        db.conn.execute("UPDATE video_meta SET probe_version=1, kind='clip', "
                        "duration_s=42.4 WHERE photo_id=?", (ids[key],))
    db.conn.execute("UPDATE video_meta SET probe_version=1, kind='live', "
                    "live_of_photo_id=?, duration_s=2.1 WHERE photo_id=?",
                    (still, ids["live"]))
    db.conn.execute("UPDATE video_meta SET probe_version=1, "
                    "probe_error='moov atom not found' WHERE photo_id=?", (ids["bad"],))
    db.conn.commit()
    db.close()

    thumbs, previews = tmp_path / "data" / "thumbnails", tmp_path / "data" / "previews"
    thumbs.mkdir()
    previews.mkdir()
    (thumbs / f"{still}_thumb.jpg").write_bytes(JPEG + b"still-thumb")
    (previews / f"{still}_preview.jpg").write_bytes(JPEG + b"still-preview")

    monkeypatch.setattr(web, "_db_path", db_path)
    monkeypatch.setattr(web, "_photo_root", str(root))
    monkeypatch.setattr(web, "_thumb_dir", str(thumbs))
    monkeypatch.setattr(web, "_preview_dir", str(previews))
    monkeypatch.setattr(web, "_nas_url", None)
    monkeypatch.setattr(worker_api, "_shutting_down", False)
    monkeypatch.delenv("PHOTOSEARCH_VIDEO_STREAM_MBPS", raising=False)
    client = TestClient(web.app)
    return {"client": client, "ids": ids, "root": root, "db_path": db_path,
            "thumbs": thumbs, "previews": previews}


# ---------------------------------------------------------------------------
# Thumbnails / previews: the four states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,size", [("thumbnail", "thumb"), ("preview", "preview")])
def test_video_image_states(env, kind, size):
    c, ids = env["client"], env["ids"]
    # paired Live clip -> the still's image
    r = c.get(f"/api/photos/{ids['live']}/{kind}")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content.endswith(f"still-{size}".encode())
    # unreadable -> static placeholder
    r = c.get(f"/api/photos/{ids['bad']}/{kind}")
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert b"Unreadable" in r.content
    # not yet probed -> generic placeholder
    r = c.get(f"/api/photos/{ids['new']}/{kind}")
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert b"Video" in r.content and b"Unreadable" not in r.content
    # probed but no poster uploaded yet -> generic placeholder, never a decode
    r = c.get(f"/api/photos/{ids['clip']}/{kind}")
    assert r.headers["content-type"].startswith("image/svg+xml")
    # poster uploaded -> served from the usual cache name
    d = env["thumbs"] if size == "thumb" else env["previews"]
    (d / f"{ids['clip']}_{size}.jpg").write_bytes(JPEG + b"poster")
    r = c.get(f"/api/photos/{ids['clip']}/{kind}")
    assert r.headers["content-type"] == "image/jpeg" and r.content.endswith(b"poster")


def test_replica_caches_only_a_real_poster(env, monkeypatch):
    c, ids = env["client"], env["ids"]
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")
    answers = {"data": b"<svg>placeholder</svg>"}
    monkeypatch.setattr(web, "_fetch_from_nas", lambda pid, kind, timeout=30.0: answers["data"])
    cache = env["thumbs"] / f"{ids['clip']}_thumb.jpg"
    r = c.get(f"/api/photos/{ids['clip']}/thumbnail")
    assert r.headers["content-type"].startswith("image/svg+xml")
    assert not cache.exists()  # a placeholder must never stick in the cache
    answers["data"] = JPEG + b"nas-poster"
    r = c.get(f"/api/photos/{ids['clip']}/thumbnail")
    assert r.headers["content-type"] == "image/jpeg"
    assert cache.read_bytes() == JPEG + b"nas-poster"


# ---------------------------------------------------------------------------
# /full: mime + Range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,mime", [
    ("clip", "video/mp4"), ("mov", "video/quicktime"),
    ("avi", "video/x-msvideo"), ("mts", "video/mp2t"),
])
def test_full_mime_and_whole_file(env, key, mime):
    r = env["client"].get(f"/api/photos/{env['ids'][key]}/full")
    assert r.status_code == 200
    assert r.headers["content-type"] == mime
    assert r.headers["accept-ranges"] == "bytes"
    assert r.content == VIDEO_BYTES


@pytest.mark.parametrize("header,start,end", [
    ("bytes=0-99", 0, 99),
    ("bytes=1000-", 1000, len(VIDEO_BYTES) - 1),
    ("bytes=-500", len(VIDEO_BYTES) - 500, len(VIDEO_BYTES) - 1),
    ("bytes=10000-99999", 10000, len(VIDEO_BYTES) - 1),
])
def test_full_range_206(env, header, start, end):
    r = env["client"].get(f"/api/photos/{env['ids']['clip']}/full",
                          headers={"Range": header})
    assert r.status_code == 206
    assert r.headers["content-range"] == f"bytes {start}-{end}/{len(VIDEO_BYTES)}"
    assert int(r.headers["content-length"]) == end - start + 1
    assert r.content == VIDEO_BYTES[start:end + 1]


def test_full_range_not_satisfiable(env):
    r = env["client"].get(f"/api/photos/{env['ids']['clip']}/full",
                          headers={"Range": f"bytes={len(VIDEO_BYTES)}-"})
    assert r.status_code == 416
    assert r.headers["content-range"] == f"bytes */{len(VIDEO_BYTES)}"


def test_full_download_attaches(env):
    r = env["client"].get(f"/api/photos/{env['ids']['avi']}/full?download=1")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]


def test_photo_full_is_unchanged(env):
    jpg = env["root"] / "2026" / "d" / "IMG_1.JPG"
    jpg.write_bytes(JPEG)
    r = env["client"].get(f"/api/photos/{env['ids']['still']}/full",
                          headers={"Range": "bytes=0-1"})
    # Stills keep the old FileResponse path: image/jpeg, not the video route.
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content.startswith(b"\xff\xd8")


# ---------------------------------------------------------------------------
# The token bucket and purpose=worker
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept += s
        self.t += s


def test_token_bucket_caps_the_rate():
    fc = FakeClock()
    rate = 1_000_000
    b = video_api.TokenBucket(rate, burst=100_000, clock=fc.clock, sleep=fc.sleep)
    total = 0
    for _ in range(40):
        b.consume(100_000)
        total += 100_000
    # 4 MB at 1 MB/s with a 100 KB burst: ~3.9 s of enforced waiting.
    assert fc.slept == pytest.approx((total - 100_000) / rate, rel=1e-6)
    assert (total - 100_000) / fc.t <= rate * 1.0001  # beyond the burst, never faster


def test_stream_rate_env(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_VIDEO_STREAM_MBPS", "12.5")
    assert video_api.stream_rate_bps() == 12_500_000
    monkeypatch.setenv("PHOTOSEARCH_VIDEO_STREAM_MBPS", "junk")
    assert video_api.stream_rate_bps() == 30_000_000
    monkeypatch.delenv("PHOTOSEARCH_VIDEO_STREAM_MBPS")
    assert video_api.stream_rate_bps() == 30_000_000


@pytest.mark.parametrize("range_header", [None, "bytes=100-4195"])
def test_worker_purpose_goes_through_the_bucket(env, monkeypatch, range_header):
    consumed = []

    class SpyBucket:
        def consume(self, n):
            consumed.append(n)

    monkeypatch.setattr(video_api, "worker_bucket", lambda: SpyBucket())
    headers = {"Range": range_header} if range_header else {}
    url = f"/api/photos/{env['ids']['clip']}/full"
    r = env["client"].get(url + "?purpose=worker", headers=headers)
    want = VIDEO_BYTES if not range_header else VIDEO_BYTES[100:4196]
    assert r.content == want
    assert sum(consumed) == len(want)  # ranged reads are capped too
    consumed.clear()
    env["client"].get(url, headers=headers)
    assert consumed == []  # the browser path is not throttled


def test_worker_stream_is_rate_limited_end_to_end(env, monkeypatch):
    fc = FakeClock()
    monkeypatch.setattr(video_api, "worker_bucket", lambda: video_api.TokenBucket(
        1000, burst=1000, clock=fc.clock, sleep=fc.sleep))
    monkeypatch.setattr(video_api, "CHUNK", 1024)
    r = env["client"].get(f"/api/photos/{env['ids']['clip']}/full?purpose=worker")
    assert r.content == VIDEO_BYTES
    assert fc.slept == pytest.approx((len(VIDEO_BYTES) - 1000) / 1000, rel=1e-6)


# ---------------------------------------------------------------------------
# Replica proxy
# ---------------------------------------------------------------------------

class FakeUpstream:
    def __init__(self, status, headers, chunks):
        self.status_code = status
        self.headers = headers
        self._chunks = chunks
        self.closed = False
        self.iterated = False

    def iter_content(self, chunk_size=None):
        self.iterated = True
        yield from self._chunks

    def close(self):
        self.closed = True

    @property
    def content(self):  # pragma: no cover — must never be touched
        raise AssertionError("proxy must stream, not read the whole body")

    def read(self, *a):  # pragma: no cover
        raise AssertionError("proxy must stream, not read the whole body")


def test_replica_proxy_forwards_range_and_streams(env, monkeypatch):
    import requests
    os.remove(env["root"] / "2026" / "d" / "CLIP.MP4")  # the replica has no files
    monkeypatch.setattr(web, "_nas_url", "http://fake-nas:8000")
    seen = {}
    up = FakeUpstream(206, {"Content-Type": "video/mp4", "Content-Length": "6",
                            "Content-Range": "bytes 0-5/10240",
                            "Accept-Ranges": "bytes", "X-Other": "no"},
                      [b"abc", b"def"])

    def fake_get(url, params=None, headers=None, stream=False, timeout=None):
        seen.update(url=url, params=params, headers=headers, stream=stream)
        return up

    monkeypatch.setattr(requests, "get", fake_get)
    pid = env["ids"]["clip"]
    r = env["client"].get(f"/api/photos/{pid}/full?purpose=worker",
                          headers={"Range": "bytes=0-5"})
    assert r.status_code == 206 and r.content == b"abcdef"
    assert r.headers["content-range"] == "bytes 0-5/10240"
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-length"] == "6"
    assert "x-other" not in r.headers
    assert seen["url"] == f"http://fake-nas:8000/api/photos/{pid}/full"
    assert seen["params"] == {"purpose": "worker"}
    assert seen["headers"]["Range"] == "bytes=0-5"
    assert seen["headers"]["X-Photosearch-Intent"]  # outbound_headers
    assert seen["stream"] is True
    assert up.iterated and up.closed


# ---------------------------------------------------------------------------
# POST /api/videos/{id}/assets
# ---------------------------------------------------------------------------

def _b64(data):
    return base64.b64encode(data).decode()


def test_assets_store_and_install_poster(env):
    c, ids = env["client"], env["ids"]
    pid = ids["clip"]
    body = {"version": 2, "assets": [
        {"kind": "poster", "size": "thumb", "data_b64": _b64(JPEG + b"T")},
        {"kind": "poster", "size": "preview", "data_b64": _b64(JPEG + b"P")},
        {"kind": "poster", "size": "full", "data_b64": _b64(JPEG + b"F")},
        {"kind": "keyframe", "name": "seg003_t12.50.jpg", "data_b64": _b64(JPEG + b"K")},
        {"kind": "crop", "name": "ident02.jpg", "data_b64": _b64(JPEG + b"C")},
    ]}
    r = c.post(f"/api/videos/{pid}/assets", json=body)
    assert r.status_code == 200, r.text
    vdir = os.path.join(os.path.dirname(env["db_path"]), "video_frames", str(pid), "v2")
    assert sorted(os.listdir(vdir)) == sorted([
        "poster_thumb.jpg", "poster_preview.jpg", "poster_full.jpg",
        "keyframe_seg003_t12.50.jpg", "crop_ident02.jpg"])
    assert (env["thumbs"] / f"{pid}_thumb.jpg").read_bytes() == JPEG + b"T"
    assert (env["previews"] / f"{pid}_preview.jpg").read_bytes() == JPEG + b"P"
    assert open(os.path.join(os.path.dirname(vdir), "poster.jpg"), "rb").read() == JPEG + b"F"
    assert not [n for n in os.listdir(vdir) if n.endswith(".tmp")]
    # The grid now gets the real poster.
    t = c.get(f"/api/photos/{pid}/thumbnail")
    assert t.headers["content-type"] == "image/jpeg" and t.content == JPEG + b"T"


@pytest.mark.parametrize("asset", [
    {"kind": "keyframe", "name": "../../evil.jpg"},
    {"kind": "keyframe", "name": "sub/evil.jpg"},
    {"kind": "keyframe", "name": ".hidden.jpg"},
    {"kind": "keyframe", "name": "x.png"},
    {"kind": "poster", "size": "huge"},
    {"kind": "script", "name": "a.jpg"},
])
def test_assets_reject_bad_names_and_kinds(env, asset):
    pid = env["ids"]["clip"]
    good = {"kind": "keyframe", "name": "ok.jpg", "data_b64": _b64(JPEG)}
    body = {"version": 1, "assets": [good, dict(asset, data_b64=_b64(JPEG))]}
    r = env["client"].post(f"/api/videos/{pid}/assets", json=body)
    assert r.status_code == 400
    # Validated before the first write: not even the good asset landed.
    assert not os.path.exists(os.path.join(
        os.path.dirname(env["db_path"]), "video_frames", str(pid)))


@pytest.mark.parametrize("body", [
    {"version": 1, "assets": [{"kind": "crop", "name": "a.jpg",
                               "data_b64": _b64(b"GIF89a....")}]},
    {"version": 1, "assets": [{"kind": "crop", "name": "a.jpg", "data_b64": "!!notb64"}]},
    {"version": 0, "assets": [{"kind": "crop", "name": "a.jpg", "data_b64": _b64(JPEG)}]},
    {"version": 1, "assets": []},
])
def test_assets_reject_bad_payloads(env, body):
    r = env["client"].post(f"/api/videos/{env['ids']['clip']}/assets", json=body)
    assert r.status_code == 400


def test_assets_reject_oversize(env, monkeypatch):
    monkeypatch.setattr(video_api, "MAX_ASSET_BYTES", 10)
    body = {"version": 1, "assets": [{"kind": "crop", "name": "a.jpg",
                                      "data_b64": _b64(JPEG)}]}
    assert env["client"].post(f"/api/videos/{env['ids']['clip']}/assets",
                              json=body).status_code == 400


def test_assets_reject_non_video_and_missing(env):
    body = {"version": 1, "assets": [{"kind": "poster", "size": "thumb",
                                      "data_b64": _b64(JPEG)}]}
    c = env["client"]
    assert c.post(f"/api/videos/{env['ids']['still']}/assets", json=body).status_code == 404
    assert c.post("/api/videos/999999/assets", json=body).status_code == 404
    # The still's real thumbnail was not overwritten.
    assert (env["thumbs"] / f"{env['ids']['still']}_thumb.jpg").read_bytes().endswith(
        b"still-thumb")


def test_assets_503_during_shutdown(env, monkeypatch):
    monkeypatch.setattr(worker_api, "_shutting_down", True)
    r = env["client"].post(f"/api/videos/{env['ids']['clip']}/assets", json={})
    assert r.status_code == 503


def test_atomic_write_leaves_nothing_on_failure(tmp_path, monkeypatch):
    dest = tmp_path / "x" / "a.jpg"
    dest.parent.mkdir()
    dest.write_bytes(b"old")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(video_api.os, "replace", boom)
    with pytest.raises(OSError):
        video_api.atomic_write(dest, b"new")
    assert dest.read_bytes() == b"old"
    assert os.listdir(dest.parent) == ["a.jpg"]


# ---------------------------------------------------------------------------
# Serializers, search param, unreadable list, request intents
# ---------------------------------------------------------------------------

def test_search_returns_video_fields_and_media_param(env):
    c, ids = env["client"], env["ids"]
    res = c.get("/api/search?media=video").json()
    got = {r["id"]: r for r in res["results"]}
    # live + unreadable hidden; the rest are playable videos
    assert set(got) == {ids[k] for k in ("clip", "mov", "avi", "mts", "new")}
    assert got[ids["clip"]]["media_type"] == "video"
    assert got[ids["clip"]]["duration_s"] == pytest.approx(42.4)
    assert got[ids["clip"]]["video_kind"] == "clip"
    assert got[ids["new"]]["duration_s"] is None
    photos = c.get("/api/search?media=photo&date_from=2026-09-26").json()["results"]
    assert [r["id"] for r in photos] == [ids["still"]]
    assert photos[0]["media_type"] is None
    assert c.get("/api/search?media=films&date_from=2026-09-26").status_code == 400


def test_photo_detail_carries_video_block(env):
    c, ids = env["client"], env["ids"]
    d = c.get(f"/api/photos/{ids['avi']}").json()
    assert d["media_type"] == "video" and d["duration_s"] == pytest.approx(42.4)
    assert d["video"]["browser_playable"] is False
    assert c.get(f"/api/photos/{ids['clip']}").json()["video"]["browser_playable"] is True
    assert c.get(f"/api/photos/{ids['still']}").json()["video"] is None


def test_unreadable_videos_list(env):
    res = env["client"].get("/api/admin/unreadable-videos").json()
    assert res["total"] == 1
    row = res["results"][0]
    assert row["id"] == env["ids"]["bad"] and row["probe_error"] == "moov atom not found"
    assert row["filepath"].endswith("BAD.MP4")


def test_request_intents_for_video_routes():
    from photosearch import request_intent as ri
    assert "upload poster/keyframes of video 7" in ri.infer_intent(
        "POST", "/api/videos/7/assets", "", None)
    assert ri.infer_intent("GET", "/api/admin/unreadable-videos", "", None) == \
        "List unreadable videos"
    assert "Worker fleet" in ri.infer_intent("GET", "/api/photos/3/full",
                                             "purpose=worker", None)
    assert ri.infer_intent("GET", "/api/photos/3/full", "", None) == \
        "Fetch the original of photo 3"
    assert ri.classify_source("/api/videos/7/assets", {}) == "worker"

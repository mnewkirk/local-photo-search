"""Server side of video-probe (docs/plans/video-support.md step 2b): the
submit branch, Live pairing, the claim predicate, deferral on a lock, the
stored-error-never-reclaimed rule, and the poster becoming visible."""

import base64
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from photosearch import web, worker_api
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.video_probe import PROBE_VERSION, ZIP_WRAPPED
from photosearch.video_probe_store import candidate_folder_ranges
from photosearch.video_registry import register_video

JPEG = b"\xff\xd8\xff\xe0" + b"poster" * 10
DIR = "2091/2091-06-24_phone-nicole"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "photos"
    db_path = str(tmp_path / "data" / "photo_index.db")
    os.makedirs(os.path.dirname(db_path))
    db = PhotoDB(db_path, photo_root=str(root))
    db.set_photo_root(str(root))
    thumbs, previews = tmp_path / "data" / "thumbnails", tmp_path / "data" / "previews"
    thumbs.mkdir()
    previews.mkdir()
    monkeypatch.setattr(web, "_db_path", db_path)
    monkeypatch.setattr(web, "_photo_root", str(root))
    monkeypatch.setattr(web, "_thumb_dir", str(thumbs))
    monkeypatch.setattr(web, "_preview_dir", str(previews))
    monkeypatch.setattr(web, "_nas_url", None)
    monkeypatch.setattr(worker_api, "_db_path", db_path)
    monkeypatch.setattr(worker_api, "_shutting_down", False)

    def video(rel, date="2091-06-24 09:00:00"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        return register_video(db, str(p), date_taken=date)

    def still(rel, date):
        return db.add_photo(filepath=rel, filename=os.path.basename(rel), date_taken=date)

    yield {"db": db, "client": TestClient(web.app), "video": video, "still": still,
           "thumbs": thumbs, "db_path": db_path}
    db.close()


def _result(pid, **kw):
    r = {"photo_id": pid, "probe_version": PROBE_VERSION, "duration_s": 2.9,
         "fps": 30.0, "codec": "hevc", "container": "mov", "width": 1080,
         "height": 1920, "rotation": 90, "bitrate": 10_000_000, "has_audio": True,
         "poster_t": 0.0, "camera_make": "Apple", "camera_model": "iPhone 15 Pro",
         "date_taken": "2091-06-24 14:16:05", "date_source": "apple_cd",
         "probe_json": {"format": {"format_name": "mov,mp4"}}}
    r.update(kw)
    return r


def _submit(client, results, failures=None, batch="b-1"):
    body = {"batch_id": batch, "pass_type": "video-probe",
            "video_probe_results": results}
    if failures:
        body["failures"] = failures
    r = client.post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _meta(db, pid):
    return dict(db.conn.execute("SELECT * FROM video_meta WHERE photo_id=?", (pid,)).fetchone())


def _photo(db, pid):
    return dict(db.conn.execute("SELECT * FROM photos WHERE id=?", (pid,)).fetchone())


def _claimable(db):
    return {r["id"] for r in db.get_unprocessed_photos("video-probe", limit=100)}


# ---------------------------------------------------------------------------
# The ordinary result
# ---------------------------------------------------------------------------

def test_ok_result_fills_meta_and_photo_columns(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/IMG_0469.MOV")
    assert vid in _claimable(db)
    out = _submit(c, [_result(vid, duration_s=58.2, gps_lat=45.9, gps_lon=7.6)])
    assert out["written"] == 1 and out["processed"] == 1
    m = _meta(db, vid)
    assert m["probe_version"] == PROBE_VERSION and m["probe_error"] is None
    assert (m["codec"], m["width"], m["height"], m["rotation"], m["has_audio"]) == \
        ("hevc", 1080, 1920, 90, 1)
    assert m["kind"] == "clip" and m["live_of_photo_id"] is None
    assert '"format_name"' in m["probe_json"]
    p = _photo(db, vid)
    assert p["date_taken"] == "2091-06-24 14:16:05"           # policy date replaces mtime
    assert (p["image_width"], p["image_height"]) == (1080, 1920)
    assert (p["camera_make"], p["camera_model"]) == ("Apple", "iPhone 15 Pro")
    assert (p["gps_lat"], p["gps_lon"], p["location_source"]) == (45.9, 7.6, "exif")
    assert p["place_name"] is None   # the nightly geocode stage labels it
    # a successful probe is its own marker: no ledger row, never re-claimed
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed "
                           "WHERE pass_type='video-probe'").fetchone()[0] == 0
    assert vid not in _claimable(db)


def test_unknown_family_keeps_the_provisional_date(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/MOVIE.MP4", date="2091-06-20 08:00:00")
    _submit(c, [_result(vid, date_taken=None, date_source="mtime",
                        camera_make=None, camera_model=None)])
    p = _photo(db, vid)
    assert p["date_taken"] == "2091-06-20 08:00:00"
    assert p["camera_make"] is None


def test_gps_never_overwrites_a_manual_location(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/A.MOV")
    db.conn.execute("UPDATE photos SET gps_lat=1, gps_lon=2, place_name='Home', "
                    "location_source='manual' WHERE id=?", (vid,))
    db.conn.commit()
    _submit(c, [_result(vid, gps_lat=45.0, gps_lon=7.0)])
    p = _photo(db, vid)
    assert (p["gps_lat"], p["gps_lon"], p["place_name"], p["location_source"]) == \
        (1, 2, "Home", "manual")


def test_a_still_id_is_rejected_and_spends_an_attempt(env):
    db, c = env["db"], env["client"]
    sid = env["still"](f"{DIR}/IMG_1.HEIC", "2091-06-24 10:00:00")
    out = _submit(c, [_result(sid)])
    assert out["written"] == 0 and out["processed"] == 1
    assert db.conn.execute("SELECT attempts FROM worker_processed WHERE photo_id=? "
                           "AND pass_type='video-probe'", (sid,)).fetchone()[0] == 1
    assert _photo(db, sid)["camera_make"] is None


# ---------------------------------------------------------------------------
# Stored errors are results — and are never re-claimed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("err", [ZIP_WRAPPED, "unreadable: moov atom not found"])
def test_stored_error_is_not_reclaimed(env, err):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/BAD.MOV")
    _submit(c, [{"photo_id": vid, "probe_version": PROBE_VERSION, "probe_error": err}])
    m = _meta(db, vid)
    assert m["probe_error"] == err and m["probe_version"] == PROBE_VERSION
    assert m["kind"] is None and m["duration_s"] is None
    assert vid not in _claimable(db)
    assert db.count_unprocessed_photos("video-probe") == 0


def test_a_probe_version_bump_requeues(env, monkeypatch):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/OLD.MOV")
    _submit(c, [_result(vid)])
    assert vid not in _claimable(db)
    import photosearch.video_probe as VP
    monkeypatch.setattr(VP, "PROBE_VERSION", PROBE_VERSION + 1)
    assert vid in _claimable(db)


def test_worker_failures_spend_attempts_until_retired(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/FLAKY.MOV")
    for _ in range(MAX_PROCESS_ATTEMPTS):
        assert vid in _claimable(db)
        _submit(c, [], failures=[{"photo_id": vid, "error": "HTTP 404"}])
    assert vid not in _claimable(db)
    assert db.count_unprocessed_photos("video-probe") == 0


def test_a_lock_defers_without_spending_an_attempt(env, monkeypatch):
    db, c = env["db"], env["client"]
    a = env["video"](f"{DIR}/A.MOV")
    b = env["video"](f"{DIR}/B.MOV")
    from photosearch import video_probe_store as S
    real = S.apply_probe_result

    def flaky(conn, r):
        if r["photo_id"] == b:
            raise sqlite3.OperationalError("database is locked")
        return real(conn, r)
    monkeypatch.setattr(S, "apply_probe_result", flaky)
    out = _submit(c, [_result(a), _result(b)])
    assert out["deferred_photo_ids"] == [b] and out["written"] == 1
    assert _meta(db, b)["probe_version"] is None
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed "
                           "WHERE photo_id=?", (b,)).fetchone()[0] == 0
    assert b in _claimable(db) and a not in _claimable(db)


def test_a_failed_result_writes_nothing_half_way(env, monkeypatch):
    """One transaction per video: a failure after the video_meta UPDATE must
    not leave the probe columns behind."""
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/HALF.MOV")
    from photosearch import video_probe_store as S

    def explode(*a, **kw):
        raise sqlite3.IntegrityError("boom")
    monkeypatch.setattr(S, "find_live_still", explode)
    out = _submit(c, [_result(vid)])
    assert out["written"] == 0
    assert _meta(db, vid)["probe_version"] is None
    assert _photo(db, vid)["camera_make"] is None


# ---------------------------------------------------------------------------
# Live Photo pairing
# ---------------------------------------------------------------------------

def test_live_pairs_with_same_stem_same_folder(env):
    db, c = env["db"], env["client"]
    sid = env["still"](f"{DIR}/IMG_0457.HEIC", "2091-06-24 14:16:04")
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, duration_s=3.0)])
    m = _meta(db, vid)
    assert (m["kind"], m["live_of_photo_id"]) == ("live", sid)


def test_live_pairs_across_a_plus_minus_one_day_sibling_folder(env):
    db, c = env["db"], env["client"]
    # still filed under the previous day (a midnight / timezone edge)
    sid = env["still"]("2091/2091-06-23_phone-nicole/IMG_0900.JPG", "2091-06-24 00:00:01")
    vid = env["video"]("2091/2091-06-24_phone-nicole/IMG_0900.MOV")
    _submit(c, [_result(vid, duration_s=2.5, date_taken="2091-06-24 00:00:02")])
    assert _meta(db, vid)["live_of_photo_id"] == sid


def test_live_pairs_across_new_year(env):
    db, c = env["db"], env["client"]
    sid = env["still"]("2090/2090-12-31_phone-matt/PXL_1.JPG", "2090-12-31 23:59:59")
    vid = env["video"]("2091/2091-01-01_phone-matt/PXL_1.MP4")
    _submit(c, [_result(vid, duration_s=1.5, date_taken="2091-01-01 00:00:01")])
    assert _meta(db, vid)["live_of_photo_id"] == sid


@pytest.mark.parametrize("kw,why", [
    ({"duration_s": 4.5}, "longer than 4 s"),
    ({"date_taken": "2091-06-24 14:16:10"}, "more than 3 s apart"),
])
def test_live_negatives(env, kw, why):
    db, c = env["db"], env["client"]
    env["still"](f"{DIR}/IMG_0457.HEIC", "2091-06-24 14:16:05")
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, **{"duration_s": 3.0, **kw})])
    assert _meta(db, vid)["kind"] == "clip", why


def test_live_negative_different_stem_and_two_days_away(env):
    db, c = env["db"], env["client"]
    env["still"](f"{DIR}/IMG_0458.HEIC", "2091-06-24 14:16:05")             # other stem
    env["still"]("2091/2091-06-22_phone-nicole/IMG_0457.HEIC", "2091-06-24 14:16:05")  # 2 days
    env["still"]("2091/2091-06-25_x/IMG_0457.HEIC", "2091-06-24 14:16:05")  # +1 day: in range
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, duration_s=3.0)])
    m = _meta(db, vid)
    assert m["kind"] == "live"
    assert _photo(db, m["live_of_photo_id"])["folder"] == "2091/2091-06-25_x"


def test_live_never_pairs_with_another_video(env):
    db, c = env["db"], env["client"]
    other = env["video"](f"{DIR}/IMG_0457.MP4", date="2091-06-24 14:16:05")
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, duration_s=3.0)])
    assert _meta(db, vid)["kind"] == "clip"
    assert other


def test_live_stem_with_like_wildcards_is_exact(env):
    db, c = env["db"], env["client"]
    env["still"](f"{DIR}/IMGX0457.HEIC", "2091-06-24 14:16:05")  # '_' as a LIKE wildcard would match
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, duration_s=3.0)])
    assert _meta(db, vid)["kind"] == "clip"


def test_paired_live_is_hidden_from_default_search_and_shows_its_still(env):
    db, c = env["db"], env["client"]
    sid = env["still"](f"{DIR}/IMG_0457.HEIC", "2091-06-24 14:16:04")
    vid = env["video"](f"{DIR}/IMG_0457.MOV")
    _submit(c, [_result(vid, duration_s=3.0)])
    from photosearch.db import HIDDEN_VIDEO_IDS_SQL
    hidden = {r[0] for r in db.conn.execute(HIDDEN_VIDEO_IDS_SQL)}
    assert vid in hidden and sid not in hidden


def test_candidate_folder_ranges():
    r = candidate_folder_ranges("2091/2091-01-01_phone-matt")
    prefixes = [lo for lo, _ in r]
    assert "2091/2091-01-01_phone-matt" in prefixes
    assert {"2090/2090-12-31", "2091/2091-01-01", "2091/2091-01-02"} <= set(prefixes)
    assert candidate_folder_ranges("Facebook takeout/videos") == [
        ("Facebook takeout/videos", "Facebook takeout/videos\x00")]


# ---------------------------------------------------------------------------
# The poster becomes visible once the probe lands
# ---------------------------------------------------------------------------

def test_poster_served_only_after_the_probe(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/CLIP.MOV")
    body = {"version": PROBE_VERSION, "assets": [
        {"kind": "poster", "size": s, "data_b64": base64.b64encode(JPEG + s.encode()).decode()}
        for s in ("thumb", "preview", "full")]}
    assert c.post(f"/api/videos/{vid}/assets", json=body).status_code == 200
    r = c.get(f"/api/photos/{vid}/thumbnail")
    assert r.headers["content-type"].startswith("image/svg")   # not yet probed
    _submit(c, [_result(vid, duration_s=20.0)])
    r = c.get(f"/api/photos/{vid}/thumbnail")
    assert r.headers["content-type"] == "image/jpeg" and r.content == JPEG + b"thumb"


# ---------------------------------------------------------------------------
# claim + clear-pass + status
# ---------------------------------------------------------------------------

def test_claim_batch_offers_only_unprobed_videos(env):
    db, c = env["db"], env["client"]
    env["still"](f"{DIR}/IMG_1.HEIC", "2091-06-24 10:00:00")
    v1 = env["video"](f"{DIR}/V1.MOV")
    v2 = env["video"](f"{DIR}/V2.MOV")
    _submit(c, [_result(v2)])
    r = c.post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-probe", "limit": 25}).json()
    assert [p["id"] for p in r["photos"]] == [v1]
    st = c.get("/api/worker/status?passes=video-probe").json()
    # the queue count does not subtract claims: v1 is still unprobed
    assert st["queue_depth"]["video-probe"] == 1


def test_clear_pass_requeues_the_probe(env):
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/V.MOV")
    _submit(c, [{"photo_id": vid, "probe_version": PROBE_VERSION, "probe_error": ZIP_WRAPPED}])
    r = c.post("/api/worker/clear-pass", json={"pass_type": "video-probe",
                                               "photo_ids": [vid]})
    assert r.status_code == 200 and r.json()["cleared"] == 1
    m = _meta(db, vid)
    assert m["probe_version"] is None and m["probe_error"] is None
    assert vid in _claimable(db)


# ---------------------------------------------------------------------------
# batch_state
# ---------------------------------------------------------------------------

def _step(db, bid, name):
    from photosearch.batch_state import batch_state
    return next(s for s in batch_state(db, bid)["steps"] if s["step"] == name)


def test_batch_state_video_probe_states_and_unreadable_detail(env):
    from photosearch.batch_state import DEPENDS_ON, fleet_launch_passes, batch_state
    from photosearch.ingest_batches import register_batch
    db, c = env["db"], env["client"]
    ok = env["video"](f"{DIR}/OK.MOV")
    z = env["video"](f"{DIR}/ZIP.MOV")
    bad = env["video"](f"{DIR}/BAD.MOV")
    bid = register_batch(db, DIR)
    st = _step(db, bid, "video-probe")
    assert (st["state"], st["total"], st["remaining"]) == ("needs_queue", 3, 3)
    assert "video-probe" not in DEPENDS_ON
    assert "video-probe" in fleet_launch_passes(batch_state(db, bid))

    _submit(c, [_result(ok),
                {"photo_id": z, "probe_version": PROBE_VERSION, "probe_error": ZIP_WRAPPED},
                {"photo_id": bad, "probe_version": PROBE_VERSION,
                 "probe_error": "unreadable: moov atom not found"}])
    st = _step(db, bid, "video-probe")
    assert (st["state"], st["done"], st["remaining"], st["failed"]) == ("completed", 3, 0, 0)
    assert st["detail"] == "2 unreadable (1 zip-wrapped)"


def test_batch_state_video_probe_blocked_when_retired(env):
    from photosearch.ingest_batches import register_batch
    db, c = env["db"], env["client"]
    vid = env["video"](f"{DIR}/FLAKY.MOV")
    bid = register_batch(db, DIR)
    for _ in range(MAX_PROCESS_ATTEMPTS):
        _submit(c, [], failures=[{"photo_id": vid, "error": "HTTP 404"}])
    st = _step(db, bid, "video-probe")
    assert (st["state"], st["failed"], st["remaining"]) == ("blocked", 1, 0)


def test_batch_with_no_videos_reads_video_probe_complete(env):
    from photosearch.ingest_batches import register_batch
    db = env["db"]
    env["still"](f"{DIR}/IMG_1.HEIC", "2091-06-24 10:00:00")
    bid = register_batch(db, DIR)
    st = _step(db, bid, "video-probe")
    assert (st["state"], st["total"]) == ("completed", 0)

"""Server side of video-scan (docs/plans/video-support.md step 3): the claim
predicate, the submit branch through _SubmitOutcome, one faces row per
identity, vec0 rows replaced (never duplicated), the D13 label carry-over,
terminal scan errors, the ZIP inner probe, poster versioning, crops served
without touching the video, clear-pass, and cleanup-orphans' stale dirs."""

import base64
import io
import os
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photosearch import web, worker_api
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.video_probe import PROBE_VERSION, ZIP_WRAPPED
from photosearch.video_registry import register_video
from photosearch.video_scan_store import (CARRY_MAX_L2, SCAN_VERSION, crop_name,
                                          match_labels, rep_name, stale_frame_dirs)

DIR = "2091/2091-07-04_phone-matt"


def _jpeg(w=320, h=240, color=(120, 80, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "JPEG", quality=80)
    return buf.getvalue()


def _unit(i, j=None, eps=0.0):
    v = np.zeros(512)
    v[i] = 1.0
    if j is not None:
        v[j] = eps
    return (v / np.linalg.norm(v)).tolist()


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

    def video(name, probe=True, kind="clip", probe_error=None, duration=12.0):
        p = root / DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        pid = register_video(db, str(p), date_taken="2091-07-04 10:00:00")
        if probe:
            db.conn.execute(
                "UPDATE video_meta SET probe_version = ?, kind = ?, probe_error = ?, "
                "duration_s = ? WHERE photo_id = ?",
                (PROBE_VERSION, None if probe_error else kind, probe_error, duration, pid))
            db.conn.commit()
        return pid

    yield {"db": db, "client": TestClient(web.app), "video": video,
           "db_path": db_path, "data": tmp_path / "data", "thumbs": thumbs}
    db.close()


def _ident(ix, enc, bbox=(40, 200, 160, 80), **kw):
    d = {"ident_ix": ix, "bbox": list(bbox), "det_score": 0.9, "encoding": enc,
         "rep_t": 1.0 + ix, "rep_frame": 4 + ix, "t_start": 0.5, "t_end": 9.0,
         "n_det": 20, "presence_s": 8.0, "focus_s": 6.0 - ix,
         "presence_intervals": [[0.0, 8.0]], "focus_intervals": [[0.0, 6.0 - ix]],
         "mean_salience": 0.05}
    d.update(kw)
    return d


def _result(pid, idents, **kw):
    r = {"photo_id": pid, "scan_version": SCAN_VERSION, "file_hash": "ab" * 32,
         "file_size": 64, "shake": 1.2, "noise": 0.4, "focus_consistency": 0.8,
         "quality_json": {"frames": {"keyframes": [], "segments": []}},
         "clip_embedding": _unit(7), "identities": idents,
         "poster_uploaded": False, "poster_t": None}
    r.update(kw)
    return r


def _submit(client, results, failures=None, batch="b-1"):
    body = {"batch_id": batch, "pass_type": "video-scan", "video_scan_results": results}
    if failures:
        body["failures"] = failures
    r = client.post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _upload(client, pid, assets, version=SCAN_VERSION):
    body = {"version": version, "assets": [
        dict(a, data_b64=base64.b64encode(a.pop("data")).decode()) for a in assets]}
    return client.post(f"/api/videos/{pid}/assets", json=body)


def _faces(db, pid):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM faces WHERE photo_id = ? ORDER BY id", (pid,))]


def _n_enc(db, ids):
    if not ids:
        return 0
    ph = ",".join("?" * len(ids))
    return db.conn.execute(f"SELECT COUNT(*) FROM face_encodings WHERE face_id IN ({ph})",
                           ids).fetchone()[0]


def _meta(db, pid):
    return dict(db.conn.execute("SELECT * FROM video_meta WHERE photo_id=?", (pid,)).fetchone())


def _ledger(db, pid):
    row = db.conn.execute("SELECT attempts FROM worker_processed WHERE photo_id=? "
                          "AND pass_type='video-scan'", (pid,)).fetchone()
    return row[0] if row else 0


# --- claim predicate -------------------------------------------------------

def test_only_probed_playable_non_live_videos_are_claimed(env):
    db, video = env["db"], env["video"]
    ok = video("A.MOV")
    zipped = video("Z.MOV", probe_error=ZIP_WRAPPED)
    video("U.MOV", probe_error="unreadable: moov atom not found")
    video("L.MOV", kind="live")
    video("N.MOV", probe=False)
    still = db.add_photo(filepath=f"{DIR}/IMG.JPG", filename="IMG.JPG")
    got = {r["id"] for r in db.get_unprocessed_photos("video-scan", limit=50)}
    assert got == {ok, zipped}
    assert still not in got
    assert db.count_unprocessed_photos("video-scan") == 2


def test_claim_response_carries_media_type_and_duration(env):
    pid = env["video"]("A.MOV", duration=42.5)
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-scan", "limit": 1})
    photo = r.json()["photos"][0]
    assert photo["id"] == pid
    assert photo["media_type"] == "video" and photo["duration_s"] == 42.5


# --- the happy path ----------------------------------------------------------

def test_one_faces_row_per_identity_tracks_clip_meta(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    resp = _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1))])])
    assert resp["processed"] == 1 and resp["written"] == 1
    faces = _faces(db, pid)
    assert len(faces) == 2
    assert (faces[0]["bbox_top"], faces[0]["bbox_right"],
            faces[0]["bbox_bottom"], faces[0]["bbox_left"]) == (40, 200, 160, 80)
    assert _n_enc(db, [f["id"] for f in faces]) == 2
    tracks = db.conn.execute("SELECT * FROM video_tracks WHERE photo_id=? ORDER BY ident_ix",
                             (pid,)).fetchall()
    assert [t["face_id"] for t in tracks] == [f["id"] for f in faces]
    assert tracks[0]["presence_intervals"] == "[[0.0, 8.0]]"
    assert db.conn.execute("SELECT COUNT(*) FROM clip_embeddings WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 1
    m = _meta(db, pid)
    assert m["scan_version"] == SCAN_VERSION and m["scan_error"] is None
    assert m["shake"] == 1.2 and m["focus_consistency"] == 0.8 and m["scanned_at"]
    assert db.conn.execute("SELECT file_hash FROM photos WHERE id=?",
                           (pid,)).fetchone()[0] == "ab" * 32
    # success is its own marker: no ledger row, and never claimed again
    assert _ledger(db, pid) == 0
    assert db.get_unprocessed_photos("video-scan", limit=10) == []


def test_rescan_replaces_tracks_and_vec0_rows_not_duplicates(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1))])])
    old_ids = [f["id"] for f in _faces(db, pid)]
    _submit(client, [_result(pid, [_ident(0, _unit(2))], clip_embedding=_unit(9))])
    faces = _faces(db, pid)
    assert len(faces) == 1 and faces[0]["id"] not in old_ids
    assert _n_enc(db, old_ids) == 0, "old vec0 face rows must be deleted explicitly"
    assert _n_enc(db, [faces[0]["id"]]) == 1
    assert db.conn.execute("SELECT COUNT(*) FROM video_tracks WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM clip_embeddings WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 1


def test_manual_labels_carry_over_and_unmatched_are_reported(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1)),
                                   _ident(2, _unit(2))])])
    f0, f1, f2 = [f["id"] for f in _faces(db, pid)]
    calvin = db.conn.execute("INSERT INTO persons (name) VALUES ('Calvin')").lastrowid
    ellie = db.conn.execute("INSERT INTO persons (name) VALUES ('Ellie')").lastrowid
    db.assign_face_to_person(f0, calvin, match_source="manual")
    db.conn.execute("UPDATE faces SET match_source='rejected' WHERE id=?", (f1,))
    db.assign_face_to_person(f2, ellie, match_source="manual")
    db.conn.execute("INSERT INTO face_person_exclusions (face_id, person_id, reason) "
                    "VALUES (?, ?, 'resolve_dups')", (f0, ellie))
    db.conn.commit()

    # Re-scan: identity 0 drifts a little (still <= 0.6 from Calvin's face),
    # identity 1 is the rejected one, Ellie's identity is gone.
    near0 = _unit(0, 5, 0.3)
    assert np.linalg.norm(np.array(near0) - np.array(_unit(0))) <= CARRY_MAX_L2
    resp = _submit(client, [_result(pid, [_ident(0, _unit(1)), _ident(1, near0),
                                          _ident(2, _unit(3))])])
    faces = {f["id"]: f for f in _faces(db, pid)}
    by_person = {f["person_id"]: f for f in faces.values() if f["person_id"]}
    assert set(by_person) == {calvin}
    new_calvin = by_person[calvin]
    assert new_calvin["match_source"] == "manual"
    assert [f["match_source"] for f in faces.values()].count("rejected") == 1
    # exclusions travel with the label
    assert db.conn.execute("SELECT COUNT(*) FROM face_person_exclusions WHERE face_id=? "
                           "AND person_id=?", (new_calvin["id"], ellie)).fetchone()[0] == 1
    info = resp["video_scan"][0]
    assert {c["old_face_id"] for c in info["carried"]} == {f0, f1}
    assert info["unmatched_labels"] == [
        {"face_id": f2, "person_id": ellie, "match_source": "manual"}]


def test_match_labels_is_one_to_one_and_greedy_by_distance():
    old = [{"face_id": 1, "encoding": _unit(0)}, {"face_id": 2, "encoding": _unit(0, 1, 0.1)}]
    got, unmatched = match_labels(old, [_unit(0, 1, 0.1)])
    assert got[0]["face_id"] == 2 and [o["face_id"] for o in unmatched] == [1]
    got, unmatched = match_labels(old, [_unit(9)])
    assert got == {} and len(unmatched) == 2


def test_strict_and_temporal_labels_are_not_carried(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    f0 = _faces(db, pid)[0]["id"]
    p = db.conn.execute("INSERT INTO persons (name) VALUES ('Matt')").lastrowid
    db.assign_face_to_person(f0, p, match_source="strict")
    db.conn.commit()
    resp = _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    assert _faces(db, pid)[0]["person_id"] is None
    assert resp["video_scan"][0]["unmatched_labels"] == []


# --- errors ------------------------------------------------------------------

def test_terminal_scan_error_is_stored_and_never_reclaimed(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    resp = _submit(client, [_result(pid, [], scan_error="unreadable: cv2 could not open the file")])
    assert resp["processed"] == 1
    m = _meta(db, pid)
    assert m["scan_error"].startswith("unreadable") and m["scan_version"] == SCAN_VERSION
    assert db.get_unprocessed_photos("video-scan", limit=10) == []
    assert db.count_unprocessed_photos("video-scan") == 0


def test_lock_defers_without_spending_an_attempt(env, monkeypatch):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    import photosearch.video_scan_store as store

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(store, "apply_scan_result", locked)
    resp = _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    assert resp["deferred"] == 1 and resp["deferred_photo_ids"] == [pid]
    assert _ledger(db, pid) == 0
    assert [r["id"] for r in db.get_unprocessed_photos("video-scan", limit=5)] == [pid]


def test_non_lock_write_error_spends_an_attempt_and_rolls_back(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    bad = _ident(1, _unit(1), bbox=(1, 2, 3))      # 3-value bbox raises mid-write
    resp = _submit(client, [_result(pid, [_ident(0, _unit(0)), bad])])
    assert resp["processed"] == 1 and resp["written"] == 0
    assert _ledger(db, pid) == 1
    assert _faces(db, pid) == [], "a half-applied scan must roll back"
    assert _meta(db, pid)["scan_version"] is None


def test_worker_failure_spends_one_attempt_and_the_cap_retires_it(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    for i in range(MAX_PROCESS_ATTEMPTS):
        assert [r["id"] for r in db.get_unprocessed_photos("video-scan", limit=5)] == [pid]
        _submit(client, [], failures=[{"photo_id": pid, "error": "detection raised"}],
                batch=f"b-{i}")
    assert _ledger(db, pid) == MAX_PROCESS_ATTEMPTS
    assert db.get_unprocessed_photos("video-scan", limit=5) == []


# --- ZIP-wrapped, hash ---------------------------------------------------------

def test_zip_inner_probe_clears_probe_error_and_fills_dims(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("Z.MOV", probe_error=ZIP_WRAPPED)
    probe = {"probe_version": PROBE_VERSION, "duration_s": 105.4, "fps": 29.97,
             "codec": "h264", "container": "mp4", "width": 1080, "height": 1920,
             "rotation": 0, "date_taken": None, "probe_json": {"format": {}}}
    _submit(client, [_result(pid, [_ident(0, _unit(0))], probe=probe)])
    m = _meta(db, pid)
    assert m["probe_error"] is None and m["codec"] == "h264" and m["kind"] == "clip"
    row = db.conn.execute("SELECT image_width, image_height FROM photos WHERE id=?",
                          (pid,)).fetchone()
    assert tuple(row) == (1080, 1920)


def test_duplicate_hash_is_reported_not_deleted(env):
    db, client = env["db"], env["client"]
    a = env["video"]("A.MOV")
    b = env["video"]("B.MOV")
    db.conn.execute("UPDATE photos SET file_hash = ? WHERE id = ?", ("cd" * 32, a))
    db.conn.commit()
    resp = _submit(client, [_result(b, [], file_hash="cd" * 32)])
    assert resp["video_scan"][0]["duplicate_of"] == a
    assert db.conn.execute("SELECT COUNT(*) FROM photos WHERE id IN (?, ?)",
                           (a, b)).fetchone()[0] == 2


# --- assets, crops, poster -------------------------------------------------------

def test_crops_installed_and_served_without_the_video(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    assets = [{"kind": "keyframe", "name": rep_name(4), "data": _jpeg()}]
    for size in (120, 200):
        assets.append({"kind": "crop", "name": crop_name(0, size), "data": _jpeg(size, size)})
    assert _upload(client, pid, assets).status_code == 200
    _submit(client, [_result(pid, [_ident(0, _unit(0)), _ident(1, _unit(1))])])
    f0, f1 = [f["id"] for f in _faces(db, pid)]
    cache = env["thumbs"] / "face_crops"
    assert (cache / f"{f0}_120.jpg").exists() and (cache / f"{f0}_200.jpg").exists()
    r = client.get(f"/api/faces/crop/{f0}?size=200")
    assert r.status_code == 200 and r.content[:3] == b"\xff\xd8\xff"
    # identity 0's other size and identity 1 have no uploaded crop: rendered
    # from the uploaded REP FRAME (the video file is 64 zero bytes)
    r = client.get(f"/api/faces/crop/{f0}?size=300")
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (300, 300)
    r = client.get(f"/api/faces/crop/{f1}?size=200")
    assert r.status_code == 404      # its rep frame (f00005) was never uploaded


def test_a_bad_asset_rejects_the_whole_upload(env):
    pid = env["video"]("A.MOV")
    r = _upload(env["client"], pid, [
        {"kind": "keyframe", "name": "s000.jpg", "data": _jpeg()},
        {"kind": "crop", "name": "../evil.jpg", "data": _jpeg()}])
    assert r.status_code == 400
    assert not (env["data"] / "video_frames" / str(pid)).exists()


def test_poster_upload_bumps_poster_version_and_replica_name(env):
    from photosearch import video_api
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    db.conn.execute("UPDATE video_meta SET poster_version = 1 WHERE photo_id = ?", (pid,))
    db.conn.commit()
    before = video_api.replica_poster_name(pid, "thumb", video_api.video_state(db.conn, pid))
    _submit(client, [_result(pid, [], poster_uploaded=True, poster_t=3.5)])
    st = video_api.video_state(db.conn, pid)
    assert st["poster_version"] == 2
    assert video_api.replica_poster_name(pid, "thumb", st) != before
    assert _meta(db, pid)["poster_t"] == 3.5


def test_frames_route_serves_the_current_scan_version(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    assert client.get(f"/api/videos/{pid}/frames/keyframe_s000.jpg").status_code == 404
    _upload(client, pid, [{"kind": "keyframe", "name": "s000.jpg", "data": _jpeg()}])
    _submit(client, [_result(pid, [])])
    r = client.get(f"/api/videos/{pid}/frames/keyframe_s000.jpg")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/videos/{pid}/frames/nope.txt").status_code == 400


# --- clear-pass, warm, cleanup ---------------------------------------------------

def test_clear_pass_video_scan_requeues_but_keeps_labels(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    r = client.post("/api/worker/clear-pass", json={"pass_type": "video-scan",
                                                    "photo_ids": [pid]})
    assert r.json()["cleared"] == 1
    assert _meta(db, pid)["scan_version"] is None
    assert len(_faces(db, pid)) == 1      # kept until the re-scan replaces them
    assert [p["id"] for p in db.get_unprocessed_photos("video-scan", limit=5)] == [pid]


def test_a_still_pass_clear_never_touches_video_faces(env):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    for p in ("faces", "clip"):
        r = client.post("/api/worker/clear-pass", json={"pass_type": p, "photo_ids": [pid]})
        assert r.status_code == 200 and r.json()["cleared"] == 0
    assert len(_faces(db, pid)) == 1
    assert db.conn.execute("SELECT COUNT(*) FROM clip_embeddings WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 1


def test_warm_crops_is_a_no_op_for_video_faces(env):
    from photosearch.face_crop import warm_crops
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _submit(client, [_result(pid, [_ident(0, _unit(0))])])
    summary = warm_crops(db, photo_ids=[pid])
    assert summary["found"] == 0


def test_stale_frame_dirs_keep_current_scan_and_probe_versions(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    db.conn.execute("UPDATE video_meta SET scan_version = 3 WHERE photo_id = ?", (pid,))
    db.conn.commit()
    root = env["data"] / "video_frames"
    for v in (PROBE_VERSION, 2, 3):
        (root / str(pid) / f"v{v}").mkdir(parents=True, exist_ok=True)
    (root / "999999" / "v1").mkdir(parents=True)          # photo deleted
    stale = stale_frame_dirs(db.conn, root)
    want = {str(root / str(pid) / "v2"), str(root / "999999" / "v1")}
    if PROBE_VERSION == 2:
        want.discard(str(root / str(pid) / "v2"))
    assert set(stale) == want

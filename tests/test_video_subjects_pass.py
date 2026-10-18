"""The video-subjects pass wiring (docs/plans/video-support.md step 3b): the
claim predicate (needs a clean current scan, never Live / unreadable,
re-claimed after a re-scan), the claim's subjects input, the submit branch
through _SubmitOutcome (atomic replace, face_ix -> faces.id, generations row
with the effective model, lock defers / other errors spend the attempt, a
stale scan defers), the worker's <= 20% segment-failure tolerance and the
ContextOverflow stop, the read endpoint (a relabel in /faces moves the name
with no recompute), the per-segment face boxes video-scan now records, and
the schema v37 migration."""

import io
import os
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photosearch import web, worker_api
from photosearch import worker as W
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.video_probe import PROBE_VERSION
from photosearch.video_registry import register_video
from photosearch.video_scan_store import SCAN_VERSION
from photosearch.video_subjects_store import SUBJECTS_VERSION

DIR = "2091/2091-07-05_phone-matt"
W_, H_ = 1280, 720


def _unit(i):
    v = np.zeros(512)
    v[i] = 1.0
    return v.tolist()


def _jpeg(color=(30, 90, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buf, "JPEG", quality=80)
    return buf.getvalue()


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

    def video(name, kind="clip", probe_error=None):
        p = root / DIR / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        pid = register_video(db, str(p), date_taken="2091-07-05 10:00:00")
        db.conn.execute("UPDATE video_meta SET probe_version = ?, kind = ?, probe_error = ?, "
                        "duration_s = 9.0 WHERE photo_id = ?",
                        (PROBE_VERSION, None if probe_error else kind, probe_error, pid))
        db.conn.commit()
        return pid

    yield {"db": db, "client": TestClient(web.app), "video": video}
    db.close()


# Three 3 s segments. Identity 0 (the "kid") is present 0-6 s, its face box
# inside the person box the fake VLM returns for segments 0 and 1.
SEG_MANIFEST = [
    {"name": "keyframe_s000.jpg", "start": 0.0, "end": 3.0, "t": 1.5, "frame_ix": 3,
     "faces": [{"ix": 0, "t": 1.5, "bbox": [600.0, 200.0, 680.0, 280.0], "salience": 0.05}]},
    {"name": "keyframe_s001.jpg", "start": 3.0, "end": 6.0, "t": 4.5, "frame_ix": 9,
     "faces": [{"ix": 0, "t": 4.5, "bbox": [600.0, 200.0, 680.0, 280.0], "salience": 0.05}]},
    {"name": "keyframe_s002.jpg", "start": 6.0, "end": 9.0, "t": 7.5, "frame_ix": 15,
     "faces": []},
]
PERSON_BOX = {"label": "boy", "category": "person", "bbox": [0.4, 0.2, 0.6, 0.9],
              "area_frac": 0.14}
DOG_BOX = {"label": "the dog", "category": "animal", "bbox": [0.1, 0.5, 0.5, 0.95],
           "area_frac": 0.18}


def _scan(env, pid, idents=1):
    """Land a video-scan result (one identity per `idents`) via the real
    submit, so faces / video_tracks / the manifest are what production
    writes."""
    identities = [{
        "ident_ix": i, "bbox": [200, 680, 280, 600], "det_score": 0.9,
        "encoding": _unit(10 + i), "rep_t": 1.5, "rep_frame": 3, "t_start": 0.0,
        "t_end": 6.0, "n_det": 12, "presence_s": 6.0, "focus_s": 6.0,
        "presence_intervals": [[0.0, 6.0]], "focus_intervals": [[0.0, 6.0]],
        "mean_salience": 0.05} for i in range(idents)]
    res = {"photo_id": pid, "scan_version": SCAN_VERSION, "file_hash": None,
           "quality_json": {"width": W_, "height": H_, "duration": 9.0, "sample_fps": 2.0,
                            "frames": {"keyframes": [], "segments": SEG_MANIFEST}},
           "clip_embedding": _unit(7), "identities": identities}
    r = env["client"].post("/api/worker/submit-results", json={
        "batch_id": "scan", "pass_type": "video-scan", "video_scan_results": [res]})
    assert r.status_code == 200 and r.json()["written"] == 1, r.text


def _ledger(db, pid):
    row = db.conn.execute("SELECT attempts FROM worker_processed WHERE photo_id=? "
                          "AND pass_type='video-subjects'", (pid,)).fetchone()
    return row[0] if row else 0


def _claimable(db):
    return {r["id"] for r in db.get_unprocessed_photos("video-subjects", limit=50)}


# --- fake worker client ------------------------------------------------------

class _Resp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class FakeFrames:
    server_url = "http://nas"

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.urls = []

    def _request(self, method, url, **kw):
        self.urls.append((url, kw.get("params")))
        name = url.rsplit("/", 1)[-1]
        return _Resp(404) if name in self.missing else _Resp(200, _jpeg())


def _ground(answers):
    """A fake ground_frame_subjects returning answers[i] per call; an
    exception instance is raised."""
    calls = {"n": 0}

    def fake(image, model="m", *, with_category=True, raise_unusable=False, prompt=None):
        a = answers[calls["n"] % len(answers)]
        calls["n"] += 1
        if isinstance(a, BaseException):
            raise a
        return a
    fake.calls = calls
    return fake


def _claimed_photo(env, pid):
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-subjects", "limit": 2})
    assert r.status_code == 200, r.text
    photos = {p["id"]: p for p in r.json()["photos"]}
    return r.json()["batch_id"], photos[pid]


def _worker_result(env, pid, monkeypatch, answers, missing=()):
    from photosearch import subjects
    monkeypatch.setattr(subjects, "ground_frame_subjects", _ground(answers))
    batch, photo = _claimed_photo(env, pid)
    out = W._process_video_subjects(FakeFrames(missing), [photo], model="qwen2.5-vl")
    return batch, out


def _submit(env, batch, results, failures=None):
    body = {"batch_id": batch, "pass_type": "video-subjects",
            "video_subjects_results": results}
    if failures:
        body["failures"] = failures
    r = env["client"].post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


# --- claim predicate ---------------------------------------------------------

def test_claim_needs_a_clean_current_scan_and_never_live_or_unreadable(env):
    db, video = env["db"], env["video"]
    scanned = video("A.MOV")
    _scan(env, scanned)
    unscanned = video("B.MOV")
    live = video("L.MOV", kind="live")
    db.conn.execute("UPDATE video_meta SET scan_version = ? WHERE photo_id = ?",
                    (SCAN_VERSION, live))
    bad = video("C.MOV")
    db.conn.execute("UPDATE video_meta SET scan_version = ?, scan_error = 'unreadable: x' "
                    "WHERE photo_id = ?", (SCAN_VERSION, bad))
    unreadable = video("U.MOV", probe_error="unreadable: moov atom not found")
    db.conn.commit()
    assert _claimable(db) == {scanned}
    assert db.count_unprocessed_photos("video-subjects") == 1
    assert unscanned and unreadable


def test_stored_result_retires_the_video_and_a_rescan_requeues_it(env, monkeypatch):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[PERSON_BOX], [PERSON_BOX], [DOG_BOX]])
    _submit(env, batch, out)
    assert _claimable(db) == set()
    assert db.conn.execute("SELECT COUNT(*) FROM video_subjects WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 2
    # A re-scan replaces the face rows the subjects pointed at: subjects are
    # dropped in the same transaction and the video is claimable again.
    _scan(env, pid)
    assert _claimable(db) == {pid}
    vm = db.conn.execute("SELECT subjects_version, subjects_json FROM video_meta "
                         "WHERE photo_id=?", (pid,)).fetchone()
    assert vm[0] is None and vm[1] is None
    assert db.conn.execute("SELECT COUNT(*) FROM video_subjects WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 0


def test_exhausted_attempts_are_not_claimed(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'video-subjects', ?)", (pid, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    assert _claimable(db) == set()


def test_claim_response_carries_segments_and_tracks(env):
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _, photo = _claimed_photo(env, pid)
    inp = photo["subjects_input"]
    assert inp["scan_version"] == SCAN_VERSION and (inp["W"], inp["H"]) == (W_, H_)
    assert [s["name"] for s in inp["segments"]] == [s["name"] for s in SEG_MANIFEST]
    fid = env["db"].conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]
    assert inp["tracks"] == [{"ident_ix": 0, "face_id": fid,
                              "presence_intervals": [[0.0, 6.0]]}]


# --- submit --------------------------------------------------------------------

def test_submit_maps_idents_to_faces_and_logs_one_generation(env, monkeypatch):
    db = env["db"]
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lmstudio.invalid/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_SUBJECT_MODEL", "qwen2.5-vl-7b-instruct")
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[PERSON_BOX], [PERSON_BOX], [DOG_BOX]])
    # the worker stamps the EFFECTIVE model, resolved through the role
    W._provenance_kwargs("video-subjects", "qwen2.5-vl", out)
    assert out[0]["model"] == "qwen2.5-vl-7b-instruct" and out[0]["model_version"] == "lmstudio"
    resp = _submit(env, batch, out)
    assert resp["written"] == 1 and resp["status"] == "ok"

    fid = db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]
    rows = [dict(r) for r in db.conn.execute(
        "SELECT * FROM video_subjects WHERE photo_id=? ORDER BY subject_ix", (pid,))]
    assert [(r["category"], r["face_id"], r["focus_s"], r["share"]) for r in rows] == [
        ("person", fid, 6.0, round(6 / 9, 4)), ("animal", None, 3.0, round(3 / 9, 4))]
    assert rows[1]["label"] == "dog"          # normalised: "the dog" -> "dog"
    assert rows[0]["rep_seg"] in (0, 1) and rows[0]["rep_frame"] in (3, 9)
    gens = db.conn.execute("SELECT text_type, model_used, model_version FROM generations "
                           "WHERE photo_id=?", (pid,)).fetchall()
    assert [tuple(g) for g in gens] == [("subjects", "qwen2.5-vl-7b-instruct", "lmstudio")]
    vm = db.conn.execute("SELECT subjects_version, subjects_json FROM video_meta "
                         "WHERE photo_id=?", (pid,)).fetchone()
    assert vm[0] == SUBJECTS_VERSION and '"groundings"' in vm[1]
    # success is its own marker: no ledger row
    assert _ledger(db, pid) == 0

    # a second result REPLACES the rows (no duplicates)
    batch2 = "again"
    _submit(env, batch2, out)
    assert db.conn.execute("SELECT COUNT(*) FROM video_subjects WHERE photo_id=?",
                           (pid,)).fetchone()[0] == 2


def test_a_result_for_a_replaced_scan_is_deferred_not_written(env, monkeypatch):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[PERSON_BOX]])
    out[0]["scan_version"] = SCAN_VERSION + 1
    resp = _submit(env, batch, out)
    assert resp["written"] == 0 and resp["deferred_photo_ids"] == [pid]
    assert _ledger(db, pid) == 0 and _claimable(db) == {pid}


def test_a_lock_defers_and_any_other_error_spends_the_attempt(env, monkeypatch):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[DOG_BOX]])
    from photosearch import video_subjects_store as S

    def locked(conn, r):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(S, "apply_subjects_result", locked)
    resp = _submit(env, batch, out)
    assert resp["deferred_photo_ids"] == [pid] and _ledger(db, pid) == 0

    def broken(conn, r):
        raise ValueError("malformed subject")
    monkeypatch.setattr(S, "apply_subjects_result", broken)
    _submit(env, "b2", out)
    assert _ledger(db, pid) == 1
    assert db.conn.execute("SELECT COUNT(*) FROM video_subjects").fetchone()[0] == 0


def test_a_worker_failure_row_spends_one_attempt(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _submit(env, "b", [], failures=[{"photo_id": pid, "error": "2/3 segments unusable"}])
    assert _ledger(db, pid) == 1 and _claimable(db) == {pid}


# --- worker ----------------------------------------------------------------------

def test_up_to_20_percent_of_failed_segments_is_tolerated(env, monkeypatch):
    from photosearch.describe import UnusableAnswer
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    # 5 segments, 1 unusable (20%): stored, that segment is "no subject"
    segs5 = [dict(SEG_MANIFEST[2], name=f"keyframe_s{i:03d}.jpg", start=3.0 * i,
                  end=3.0 * (i + 1), t=3.0 * i + 1.5, faces=[]) for i in range(5)]
    photo = {"id": pid, "filename": "A.MOV", "subjects_input": {
        "scan_version": SCAN_VERSION, "W": W_, "H": H_, "sample_fps": 2.0,
        "segments": segs5, "tracks": []}}
    from photosearch import subjects
    monkeypatch.setattr(subjects, "ground_frame_subjects",
                        _ground([[DOG_BOX], UnusableAnswer("x"), [DOG_BOX], [DOG_BOX], [DOG_BOX]]))
    out = W._process_video_subjects(FakeFrames(), [photo], model="m")
    assert "error" not in out[0]
    a = out[0]["analysis"]
    assert a["n_ungrounded"] == 1 and len(a["failed_segments"]) == 1
    assert out[0]["subjects"][0]["share"] == pytest.approx(12 / 15)

    # 2 of 5 (40%): the whole video is a failure row
    monkeypatch.setattr(subjects, "ground_frame_subjects",
                        _ground([[DOG_BOX], UnusableAnswer("x"), UnusableAnswer("y"),
                                 [DOG_BOX], [DOG_BOX]]))
    out = W._process_video_subjects(FakeFrames(), [photo], model="m")
    assert W._is_failure_row(out[0]) and "2/5" in out[0]["error"]
    # a missing keyframe counts like an unusable answer
    monkeypatch.setattr(subjects, "ground_frame_subjects", _ground([[DOG_BOX]]))
    out = W._process_video_subjects(FakeFrames(missing={"keyframe_s003.jpg"}),
                                    [photo], model="m")
    assert "error" not in out[0] and out[0]["analysis"]["failed_segments"][0]["seg"] == 3
    assert db


def test_no_answer_defers_the_video(env, monkeypatch):
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _, out = _worker_result(env, pid, monkeypatch, [[DOG_BOX], None])
    assert out == []          # omitted: re-claimed for free


def test_context_overflow_halts_the_pass_without_spending_attempts(env, monkeypatch):
    from photosearch.describe import ContextOverflow
    a = env["video"]("A.MOV")
    b = env["video"]("B.MOV")
    _scan(env, a)
    _scan(env, b)
    from photosearch import subjects
    monkeypatch.setattr(subjects, "ground_frame_subjects",
                        _ground([[DOG_BOX], [DOG_BOX], [DOG_BOX], ContextOverflow("ctx")]))
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-subjects", "limit": 2})
    photos = sorted(r.json()["photos"], key=lambda p: p["id"])
    with pytest.raises(W.PassHalted) as ei:
        W._process_video_subjects(FakeFrames(), photos, model="m")
    # the first video finished and is handed back; the second is NOT a failure
    assert [x["photo_id"] for x in ei.value.results] == [photos[0]["id"]]
    assert not any(W._is_failure_row(x) for x in ei.value.results)


def test_the_worker_loop_stops_claiming_a_halted_pass(monkeypatch, tmp_path):
    from photosearch.describe import ContextOverflow  # noqa: F401
    submitted = []

    class Client:
        worker_id = "w"
        n = 0

        def claim_batch(self, pass_type, **kw):
            Client.n += 1
            return {"batch_id": f"b{Client.n}", "photos": [{"id": Client.n}], "remaining": 5}

        def get_status(self, **kw):
            return {"queue_depth": {}, "active_claims": []}

        def renew_claim(self, *a, **k):
            return {}

        def submit_results(self, batch_id, pass_type, **kw):
            submitted.append(kw)
            return {"written": len(kw.get("video_subjects_results") or []), "processed": 0}

    def halt(client, photos, model, credit=None):
        raise W.PassHalted("ContextOverflow: too big", [])
    monkeypatch.setattr(W, "WorkerClient", lambda *a, **k: Client())
    monkeypatch.setattr(W, "_process_video_subjects", halt)
    monkeypatch.setattr(W, "_unload_pass_models", lambda p: None)
    monkeypatch.setattr(W, "_flush_caches", lambda: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda **k: str(tmp_path))

    class _HB:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(W, "_ClaimHeartbeat", _HB)
    W.run_worker(server="http://fake", passes=["video-subjects"])
    assert Client.n == 1                     # claimed once, then stopped
    assert submitted == [{"video_subjects_results": []}]   # claim released, no failures


def test_without_per_segment_face_boxes_people_stay_unidentified():
    from photosearch import video_subjects as VSub
    old = [{k: v for k, v in s.items() if k != "faces"} for s in SEG_MANIFEST]
    assert VSub.face_layer_from_manifest(old, [{"ident_ix": 0, "presence_intervals": []}],
                                         W_, H_, 2.0) is None
    layer = VSub.face_layer_from_manifest(SEG_MANIFEST, [
        {"ident_ix": 0, "presence_intervals": [[0.0, 6.0]]}], W_, H_, 2.0)
    an = VSub.score_subjects([(s["start"], s["end"]) for s in SEG_MANIFEST],
                             [[PERSON_BOX], [PERSON_BOX], [DOG_BOX]], faces=layer,
                             W=W_, H=H_, keyframe_times=[s["t"] for s in SEG_MANIFEST])
    assert an.subjects[0].face_ix == 0 and an.subjects[0].focus_s == 6.0
    # S2 credit reads the stored presence intervals
    an2 = VSub.score_subjects([(0.0, 3.0), (3.0, 9.0)], [[PERSON_BOX], [PERSON_BOX]],
                              faces=layer, W=W_, H=H_, keyframe_times=[1.5, 4.5],
                              params=VSub.SubjectParams(credit="S2"))
    assert an2.subjects[0].focus_s == pytest.approx(6.0)


def test_segment_face_boxes_take_the_nearest_detection_in_the_window():
    from photosearch.video_analysis import Identity
    from photosearch.video_subjects import segment_face_boxes

    def ident(ix, ts):
        return Identity(ix=ix, n_det=len(ts), track_count=1, presence_s=1, presence_share=0,
                        focus_s=0, focus_share=0, presence_intervals=[], focus_intervals=[],
                        mean_salience=0, rep_t=0, rep_frame_ix=0, rep_bbox=[0, 0, 1, 1],
                        rep_det_score=1, first_t=0, last_t=0, encoding=None,
                        detections=[{"t": t, "bbox": [t, 0, t + 1, 1], "salience": t}
                                    for t in ts])
    got = segment_face_boxes([ident(0, [0.5, 1.4, 2.6]), ident(1, [5.0])], 1.5, 1.0)
    assert got == [{"ix": 0, "t": 1.4, "bbox": [1.4, 0.0, 2.4, 1.0], "salience": 1.4}]


# --- read endpoint -----------------------------------------------------------------

def test_read_endpoint_resolves_names_at_read_time(env, monkeypatch):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[PERSON_BOX], [PERSON_BOX], [DOG_BOX]])
    _submit(env, batch, out)
    fid = db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]

    j = client.get(f"/api/videos/{pid}/subjects").json()
    assert [s["display"] for s in j["subjects"]] == ["unnamed person", "dog"]
    assert j["subjects"][0]["presence_intervals"] == [[0.0, 6.0]]
    assert [(t["start"], t["end"], t["display"]) for t in j["timeline"]] == [
        (0.0, 6.0, "unnamed person"), (6.0, 9.0, "dog")]
    assert j["people"] == []

    # Name the face in /faces: the subject reads the new name, no recompute.
    cal = db.add_person("Calvin")
    db.conn.execute("UPDATE faces SET person_id=?, match_source='manual' WHERE id=?",
                    (cal, fid))
    db.conn.commit()
    j = client.get(f"/api/videos/{pid}/subjects").json()
    assert j["subjects"][0]["display"] == "Calvin"
    assert j["people"] == [{"person_id": cal, "name": "Calvin", "focus_s": 6.0,
                            "share": round(6 / 9, 4), "face_ids": [fid]}]
    assert j["timeline"][0]["display"] == "Calvin"

    assert client.get("/api/videos/999999/subjects").status_code == 404


def test_clear_pass_requeues_and_drops_rows(env, monkeypatch):
    db, client = env["db"], env["client"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    batch, out = _worker_result(env, pid, monkeypatch, [[DOG_BOX]])
    _submit(env, batch, out)
    r = client.post("/api/worker/clear-pass", json={"pass_type": "video-subjects",
                                                     "photo_ids": [pid]})
    assert r.status_code == 200 and r.json()["cleared"] == 1
    assert _claimable(db) == {pid}
    assert db.conn.execute("SELECT COUNT(*) FROM video_subjects").fetchone()[0] == 0


# --- plumbing ------------------------------------------------------------------------

def test_pass_lists_and_roles_agree():
    from photosearch import batch_state, describe, rerun
    for lst in (batch_state.WORKER_PASSES, rerun.ALL_PASSES, worker_api._ALL_PASSES):
        # after the scan it reads; right after aesthetics, whose model
        # (qwen2.5-vl) it shares, so the fleet loads it once (step 5 order)
        assert lst.index("video-subjects") > lst.index("video-scan")
        assert lst.index("video-subjects") == lst.index("aesthetics") + 1
    assert batch_state.DEPENDS_ON["video-subjects"] == "video-scan"
    assert describe.PASS_ROLES["video-subjects"] == "subject"
    assert rerun._PASS_LLM["video-subjects"][0] == "subject"
    assert rerun.FLEET_ROLE_MODELS["subject"] == rerun.FLEET_ROLE_MODELS["aesthetics"]
    assert "video-subjects" in W.SELF_READ_PASSES


def test_fleet_env_sets_the_subject_model(monkeypatch):
    from photosearch import admin_api
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.delenv("PHOTOSEARCH_FLEET_SUBJECT_MODEL", raising=False)
    monkeypatch.setenv("PHOTOSEARCH_LLM_SUBJECT_MODEL", "stray")
    assert admin_api._fleet_env()["PHOTOSEARCH_LLM_SUBJECT_MODEL"] == "qwen2.5-vl-7b-instruct"


def test_release_llm_models_knows_the_subject_role(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_SUBJECT_MODEL", "qwen2.5-vl-7b-instruct")
    pm = {"video-subjects": [("qwen2.5-vl", "subject")]}
    assert W._llm_models_for("video-subjects", pm) == {"qwen2.5-vl-7b-instruct"}


def test_v36_db_migrates_to_v37_subjects_columns(tmp_path):
    from photosearch.db import SCHEMA_VERSION
    path = str(tmp_path / "v36.db")
    with PhotoDB(path) as db:
        db.add_photo(filepath="v/a.mov", filename="a.mov", media_type="video")
    conn = sqlite3.connect(path)
    for col in ("subjects_version", "subjects_at", "subjects_json"):
        conn.execute(f"ALTER TABLE video_meta DROP COLUMN {col}")
    for col in ("share", "rep_seg"):
        conn.execute(f"ALTER TABLE video_subjects DROP COLUMN {col}")
    conn.execute("UPDATE schema_info SET value = '36' WHERE key = 'version'")
    conn.commit()
    conn.close()
    with PhotoDB(path) as db:
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(video_meta)")}
        assert {"subjects_version", "subjects_at", "subjects_json"} <= cols
        cols = {r["name"] for r in db.conn.execute("PRAGMA table_info(video_subjects)")}
        assert {"share", "rep_seg"} <= cols
        v = db.conn.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()[0]
        assert int(v) == SCHEMA_VERSION == 38

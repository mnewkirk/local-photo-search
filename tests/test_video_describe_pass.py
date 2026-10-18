"""The video-describe pass wiring (docs/plans/video-support.md step 5): the
claim predicate (a clean current scan, no description, never Live / unreadable
/ scan error, attempts-capped, NOT gated on video-subjects), the claim's
describe input (keyframes of the current scan + a subjects timeline with
person NAMES resolved at claim time), the worker's outcomes (unusable /
truncated -> failure row, no answer -> omitted, ContextOverflow / no role
model -> the pass halts with no attempt spent), the submit (description +
generations with the effective model + categories/keywords re-queued, verify
untouched), verify never claiming a video, the stale-description tooling, and
batch_state for the now "any" text passes."""

import io
import json
import os
import urllib.request

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from photosearch import describe as D
from photosearch import web, worker_api
from photosearch import worker as W
from photosearch.batch_state import batch_state
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.ingest_batches import register_batch
from photosearch.video_probe import PROBE_VERSION
from photosearch.video_registry import register_video
from photosearch.video_scan_store import SCAN_VERSION

DIR = "2091/2091-07-06_phone-matt"
GOOD = ("A boy jumps on a trampoline in a back garden while a dog runs around "
        "the edge. He lands, bounces higher and waves at the camera.")

_ROLE_VARS = ("PHOTOSEARCH_LLM_VIDEO_MODEL", "PHOTOSEARCH_TEXT_LLM_URL",
              "PHOTOSEARCH_TEXT_LLM_MODEL", "PHOTOSEARCH_LLM_VISUAL_MODEL",
              "PHOTOSEARCH_LLM_REASONING_EFFORT")


def _unit(i):
    v = np.zeros(512)
    v[i] = 1.0
    return v.tolist()


def _jpeg(color=(30, 90, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 36), color).save(buf, "JPEG", quality=80)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in _ROLE_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(D, "_RETRY_DELAY", 0)


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
        pid = register_video(db, str(p), date_taken="2091-07-06 10:00:00")
        db.conn.execute("UPDATE video_meta SET probe_version = ?, kind = ?, probe_error = ?, "
                        "duration_s = 9.0 WHERE photo_id = ?",
                        (PROBE_VERSION, None if probe_error else kind, probe_error, pid))
        db.conn.commit()
        return pid

    yield {"db": db, "client": TestClient(web.app), "video": video, "root": root}
    db.close()


KEYFRAMES = [{"name": "keyframe_k00.jpg", "t": 1.0, "frame_ix": 2},
             {"name": "keyframe_k01.jpg", "t": 4.5, "frame_ix": 9},
             {"name": "keyframe_k02.jpg", "t": 8.0, "frame_ix": 16}]


def _scan(env, pid, idents=1, keyframes=KEYFRAMES):
    """Land a video-scan result through the real submit."""
    identities = [{
        "ident_ix": i, "bbox": [200, 680, 280, 600], "det_score": 0.9,
        "encoding": _unit(10 + i), "rep_t": 1.5, "rep_frame": 3, "t_start": 0.0,
        "t_end": 6.0, "n_det": 12, "presence_s": 6.0, "focus_s": 6.0,
        "presence_intervals": [[0.0, 6.0]], "focus_intervals": [[0.0, 6.0]],
        "mean_salience": 0.05} for i in range(idents)]
    res = {"photo_id": pid, "scan_version": SCAN_VERSION, "file_hash": None,
           "quality_json": {"width": 1280, "height": 720, "duration": 9.0,
                            "sample_fps": 2.0,
                            "frames": {"keyframes": keyframes, "segments": []}},
           "clip_embedding": _unit(7), "identities": identities}
    r = env["client"].post("/api/worker/submit-results", json={
        "batch_id": "scan", "pass_type": "video-scan", "video_scan_results": [res]})
    assert r.status_code == 200 and r.json()["written"] == 1, r.text


def _subjects(db, pid):
    """A stored subjects result: identity 0 (a person) 0-6 s, a dog 6-9 s."""
    analysis = {"segments": [[0.0, 3.0], [3.0, 6.0], [6.0, 9.0]],
                "primaries": [0, 0, 1],
                "subjects": [{"ix": 0, "label": "boy", "category": "person", "face_ix": 0},
                             {"ix": 1, "label": "dog", "category": "animal",
                              "face_ix": None}],
                "summary": "0–6 s person (face_ix 0), 6–9 s dog"}
    db.conn.execute("UPDATE video_meta SET subjects_version = 1, subjects_json = ? "
                    "WHERE photo_id = ?", (json.dumps(analysis), pid))
    db.conn.commit()


def _claimable(db):
    return {r["id"] for r in db.get_unprocessed_photos("video-describe", limit=50)}


def _ledger(db, pid, pass_type="video-describe"):
    row = db.conn.execute("SELECT attempts FROM worker_processed WHERE photo_id=? "
                          "AND pass_type=?", (pid, pass_type)).fetchone()
    return row[0] if row else 0


def _claim(env, pid):
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-describe", "limit": 4})
    assert r.status_code == 200, r.text
    photos = {p["id"]: p for p in r.json()["photos"]}
    return r.json()["batch_id"], photos[pid]


def _submit(env, batch, results=None, failures=None):
    body = {"batch_id": batch, "pass_type": "video-describe",
            "video_describe_results": results or []}
    if failures:
        body["failures"] = failures
    r = env["client"].post("/api/worker/submit-results", json=body)
    assert r.status_code == 200, r.text
    return r.json()


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


def _fake_describe(monkeypatch, answer):
    calls = []

    def fake(frames, timestamps, duration_s, model=None, prompt=None,
             raise_unusable=False, subject_summary=None):
        calls.append({"frames": frames, "timestamps": timestamps,
                      "duration_s": duration_s, "model": model,
                      "raise_unusable": raise_unusable,
                      "subject_summary": subject_summary})
        if isinstance(answer, BaseException):
            raise answer
        return answer
    monkeypatch.setattr(D, "describe_video", fake)
    return calls


# --- claim predicate -----------------------------------------------------------

def test_claim_needs_a_clean_scan_and_no_description(env):
    db, video = env["db"], env["video"]
    scanned = video("A.MOV")
    _scan(env, scanned)                 # no subjects: still claimable
    described = video("D.MOV")
    _scan(env, described)
    db.conn.execute("UPDATE photos SET description = 'x' WHERE id = ?", (described,))
    unscanned = video("B.MOV")
    live = video("L.MOV", kind="live")
    db.conn.execute("UPDATE video_meta SET scan_version = ? WHERE photo_id = ?",
                    (SCAN_VERSION, live))
    bad = video("C.MOV")
    db.conn.execute("UPDATE video_meta SET scan_version = ?, scan_error = 'unreadable: x' "
                    "WHERE photo_id = ?", (SCAN_VERSION, bad))
    video("U.MOV", probe_error="unreadable: moov atom not found")
    db.conn.commit()
    assert _claimable(db) == {scanned}
    assert db.count_unprocessed_photos("video-describe") == 1
    assert unscanned


def test_exhausted_attempts_are_not_claimed(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'video-describe', ?)", (pid, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    assert _claimable(db) == set()


def test_claim_carries_keyframes_and_no_summary_without_subjects(env):
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _, photo = _claim(env, pid)
    inp = photo["describe_input"]
    assert inp["scan_version"] == SCAN_VERSION and inp["duration"] == 9.0
    assert inp["keyframes"] == [{"name": k["name"], "t": k["t"]} for k in KEYFRAMES]
    assert inp["subject_summary"] is None


def test_summary_names_people_at_claim_time(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _subjects(db, pid)
    _, photo = _claim(env, pid)
    s = photo["describe_input"]["subject_summary"]
    assert "an unnamed person" in s and "dog" in s and "face_ix" not in s
    # a /faces label reaches the NEXT claim with no recompute
    person = db.add_person("Calvin")
    db.conn.execute("UPDATE faces SET person_id = ? WHERE photo_id = ?", (person, pid))
    db.conn.execute("DELETE FROM worker_claims")
    db.conn.commit()
    _, photo = _claim(env, pid)
    s = photo["describe_input"]["subject_summary"]
    assert "Calvin" in s and "dog" in s and "person" not in s
    assert s.index("Calvin") < s.index("dog")


# --- worker --------------------------------------------------------------------

def test_worker_describes_with_keyframes_and_summary(env, monkeypatch):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _subjects(db, pid)
    calls = _fake_describe(monkeypatch, GOOD)
    _, photo = _claim(env, pid)
    fc = FakeFrames()
    out = W._process_video_describe(fc, [photo], model="qwen2.5-vl")
    assert out == [{"photo_id": pid, "description": GOOD, "scan_version": SCAN_VERSION,
                    "scan_generation": None}]
    (c,) = calls
    assert len(c["frames"]) == 3 and c["timestamps"] == [1.0, 4.5, 8.0]
    assert c["duration_s"] == 9.0 and c["raise_unusable"] is True
    assert "dog" in c["subject_summary"]
    assert all(p == {"v": SCAN_VERSION} for _, p in fc.urls)


@pytest.mark.parametrize("exc", [D.UnusableAnswer("video-describe: ends mid-sentence"),
                                 D.TruncatedOutput("cut off", text="A boy and")])
def test_unusable_or_truncated_is_a_failure_row(env, monkeypatch, exc):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _fake_describe(monkeypatch, exc)
    batch, photo = _claim(env, pid)
    out = W._process_video_describe(FakeFrames(), [photo], model="m")
    kwargs = W._submit_kwargs("video_describe_results", out)
    assert kwargs.get("video_describe_results") in (None, [])
    assert [f["photo_id"] for f in kwargs["failures"]] == [pid]
    _submit(env, batch, kwargs.get("video_describe_results"), kwargs["failures"])
    assert _ledger(db, pid) == 1
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (pid,)).fetchone()[0] is None


def test_a_missing_keyframe_is_a_failure_row(env, monkeypatch):
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    calls = _fake_describe(monkeypatch, GOOD)
    _, photo = _claim(env, pid)
    out = W._process_video_describe(FakeFrames(missing={"keyframe_k01.jpg"}), [photo],
                                    model="m")
    assert out[0]["photo_id"] == pid and "keyframe_k01.jpg" in out[0]["error"]
    assert not calls


@pytest.mark.parametrize("exc", [D.ContextOverflow("too big", detail="Context size"),
                                 D.RoleModelNotConfigured("no video model")])
def test_config_failures_halt_the_pass_without_spending_attempts(env, monkeypatch, exc):
    db = env["db"]
    a = env["video"]("A.MOV")
    b = env["video"]("B.MOV")
    _scan(env, a)
    _scan(env, b)
    _fake_describe(monkeypatch, exc)
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "video-describe", "limit": 4})
    body = r.json()
    with pytest.raises(W.PassHalted) as ei:
        W._process_video_describe(FakeFrames(), body["photos"], model="m")
    assert ei.value.results == []
    _submit(env, body["batch_id"], [])
    assert _ledger(db, a) == 0 and _ledger(db, b) == 0
    assert _claimable(db) == {a, b}


def test_tiny_max_tokens_is_a_failure_row_not_stored_truncated_text(env, monkeypatch):
    """The real describe_video over the OpenAI route: every call stops at the
    token limit (a deliberately tiny budget) -> TruncatedOutput after the
    double-budget retry and the 0.4 retry -> a failure row, nothing stored."""
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    monkeypatch.setitem(D._OPENAI_MAX_TOKENS, "video", 8)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    bodies = []

    class _R:
        def __init__(self, p): self.p = p
        def read(self): return json.dumps(self.p).encode()

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        return _R({"choices": [{"message": {"content": "A boy jumps on a"},
                                "finish_reason": "length"}], "usage": {}})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    batch, photo = _claim(env, pid)
    out = W._process_video_describe(FakeFrames(), [photo], model="vid-m")
    kwargs = W._submit_kwargs("video_describe_results", out)
    assert [f["photo_id"] for f in kwargs["failures"]] == [pid]
    assert {b["max_tokens"] for b in bodies} == {8, 16}
    _submit(env, batch, kwargs.get("video_describe_results"), kwargs["failures"])
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (pid,)).fetchone()[0] is None
    assert _ledger(db, pid) == 1


def test_fleet_drops_the_pass_without_a_video_model(monkeypatch, capsys):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")   # never a fallback

    def boom(*a, **k):
        raise AssertionError("must not connect before the role check")
    monkeypatch.setattr(W, "WorkerClient", boom)
    W.run_worker(server="http://fake", passes=["video-describe"])
    assert "PHOTOSEARCH_LLM_VIDEO_MODEL" in capsys.readouterr().err


# --- submit --------------------------------------------------------------------

def test_submit_writes_description_generation_and_requeues_dependents(env, monkeypatch):
    db = env["db"]
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lmstudio.invalid/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "qwen/qwen3.5-9b")
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    # stale derived output from an earlier description
    db.conn.execute("UPDATE photos SET categories = '[\"x\"]', keywords = '[\"y\"]' "
                    "WHERE id = ?", (pid,))
    for p in ("category-content", "keywords"):
        db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                        "VALUES (?, ?, 1)", (pid, p))
    db.conn.commit()
    _fake_describe(monkeypatch, GOOD)
    batch, photo = _claim(env, pid)
    out = W._process_video_describe(FakeFrames(), [photo], model="qwen2.5-vl")
    W._provenance_kwargs("video-describe", "qwen2.5-vl", out)
    assert out[0]["model"] == "qwen/qwen3.5-9b" and out[0]["model_version"] == "lmstudio"
    resp = _submit(env, batch, W._submit_kwargs("video_describe_results", out)
                   ["video_describe_results"])
    assert resp["written"] == 1
    row = db.conn.execute("SELECT description, categories, keywords, verified_at "
                          "FROM photos WHERE id=?", (pid,)).fetchone()
    assert tuple(row) == (GOOD, None, None, None)
    g = db.conn.execute("SELECT text_type, generated_text, model_used, model_version "
                        "FROM generations WHERE photo_id=?", (pid,)).fetchall()
    assert [tuple(x) for x in g] == [("describe", GOOD, "qwen/qwen3.5-9b", "lmstudio")]
    # the description is its own marker; dependents' ledger rows were dropped
    assert _ledger(db, pid) == 0
    assert _ledger(db, pid, "category-content") == 0 and _ledger(db, pid, "keywords") == 0
    assert _claimable(db) == set()
    # category-content / keywords now claim the video; verify never does (D18)
    assert pid in {r["id"] for r in db.get_unprocessed_photos("category-content", limit=9)}
    assert pid in {r["id"] for r in db.get_unprocessed_photos("keywords", limit=9)}
    assert pid not in {r["id"] for r in db.get_unprocessed_photos("verify", limit=9)}
    assert db.count_unprocessed_photos("verify") == 0


def test_text_pass_claim_carries_the_video_description(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    db.conn.execute("UPDATE photos SET description = ? WHERE id = ?", (GOOD, pid))
    db.conn.commit()
    r = env["client"].post("/api/worker/claim-batch", json={
        "worker_id": "w", "pass_type": "keywords", "limit": 4})
    (p,) = r.json()["photos"]
    assert p["id"] == pid and p["description"] == GOOD


def test_a_result_for_a_still_spends_the_attempt(env):
    db = env["db"]
    still = db.add_photo(filepath=f"{DIR}/IMG.JPG", filename="IMG.JPG")
    resp = _submit(env, "b", [{"photo_id": still, "description": GOOD}])
    assert resp["written"] == 0
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (still,)).fetchone()[0] is None
    assert _ledger(db, still) == 1


def test_clear_pass_requeues_the_video_and_its_text_passes(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    still = db.add_photo(filepath=f"{DIR}/IMG.JPG", filename="IMG.JPG", description="s")
    db.conn.execute("UPDATE photos SET description = ?, categories = '[]', keywords = '[]' "
                    "WHERE id = ?", (GOOD, pid))
    db.conn.commit()
    r = env["client"].post("/api/worker/clear-pass", json={
        "pass_type": "video-describe", "photo_ids": [pid, still]})
    assert r.status_code == 200 and r.json()["cleared"] == 1
    row = db.conn.execute("SELECT description, categories, keywords FROM photos WHERE id=?",
                          (pid,)).fetchone()
    assert tuple(row) == (None, None, None)
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (still,)).fetchone()[0] == "s"
    assert _claimable(db) == {pid}


# --- stale-description tooling ------------------------------------------------

def test_stale_descriptions_never_flags_a_video_for_verify(env):
    from photosearch.stale_descriptions import find_stale
    db = env["db"]
    pid = env["video"]("A.MOV")
    db.conn.execute("UPDATE photos SET description = ?, verified_at = '2091-01-01 00:00:00' "
                    "WHERE id = ?", (GOOD, pid))
    db.conn.execute("INSERT INTO generations (photo_id, text_type, generated_text, "
                    "created_at) VALUES (?, 'describe', ?, '2091-07-01T00:00:00Z')",
                    (pid, GOOD))
    db.conn.commit()
    assert pid not in find_stale(db.conn)["verify"]


# --- batch_state ---------------------------------------------------------------

def _finish_stills(db, ids):
    for i, pid in enumerate(ids):
        db.add_clip_embedding(pid, _unit(i))
        db.conn.execute(
            "UPDATE photos SET aesthetic_score = 5, aesthetic_concepts = '[]', "
            "aes_overall = 6, aes_overall_pct = 50, description = 'd', "
            "verified_at = datetime('now'), visual_tags = '[]', "
            "categories = '[]', keywords = '[]' WHERE id = ?", (pid,))
    db.mark_processed(ids, "faces", terminal=True)
    db.conn.commit()


def _steps(db, bid):
    return {s["step"]: s for s in batch_state(db, bid)["steps"]}


def test_batch_with_a_described_video_needs_the_text_passes(env):
    db = env["db"]
    stills = [db.add_photo(filepath=f"{DIR}/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
                           date_taken="2091-07-06 09:00:00") for i in range(2)]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _finish_stills(db, stills)
    bid = register_batch(db, DIR)
    st = _steps(db, bid)
    assert st["video-describe"]["state"] == "needs_queue"
    for p in ("category-content", "keywords"):
        assert st[p]["state"] == "waiting" and st[p]["waiting_on"] == "video-describe"
        assert st[p]["total"] == 3
    db.conn.execute("UPDATE photos SET description = ? WHERE id = ?", (GOOD, pid))
    db.conn.commit()
    st = _steps(db, bid)
    assert st["video-describe"]["state"] == "completed"
    for p in ("category-content", "keywords"):
        assert st[p]["state"] == "needs_queue" and st[p]["remaining"] == 1
    assert st["verify"]["state"] == "completed"


@pytest.mark.parametrize("how", ["scan_error", "live", "unreadable"])
def test_an_undescribable_video_does_not_hold_the_text_passes(env, how):
    db = env["db"]
    stills = [db.add_photo(filepath=f"{DIR}/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
                           date_taken="2091-07-06 09:00:00") for i in range(2)]
    if how == "unreadable":
        pid = env["video"]("U.MOV", probe_error="unreadable: moov atom not found")
    else:
        pid = env["video"]("A.MOV", kind="live" if how == "live" else "clip")
        db.conn.execute("UPDATE video_meta SET scan_version = ?, scan_error = ? "
                        "WHERE photo_id = ?",
                        (SCAN_VERSION, "unreadable: x" if how == "scan_error" else None, pid))
        db.conn.commit()
    _finish_stills(db, stills)
    bid = register_batch(db, DIR)
    st = _steps(db, bid)
    assert st["video-describe"]["state"] == "completed"
    for p in ("category-content", "keywords"):
        assert st[p]["state"] == "completed", st[p]
        assert st[p]["total"] == 2


# --- plumbing ------------------------------------------------------------------

def test_pass_order_roles_and_models():
    from photosearch import batch_state as bs
    from photosearch import rerun
    for lst in (bs.WORKER_PASSES, rerun.ALL_PASSES, worker_api._ALL_PASSES):
        # right after describe (one qwen3.5 load), after the scan it reads,
        # before the text passes that read its output
        assert lst.index("video-describe") == lst.index("describe") + 1
        assert lst.index("video-describe") > lst.index("video-scan")
        assert lst.index("video-describe") < lst.index("category-content")
        assert lst.index("video-describe") < lst.index("keywords")
    assert tuple(bs.WORKER_PASSES) == tuple(rerun.ALL_PASSES) == tuple(worker_api._ALL_PASSES)
    assert bs.DEPENDS_ON["video-describe"] == "video-scan"
    assert "video-subjects" not in (bs.DEPENDS_ON["video-describe"],)
    assert D.PASS_ROLES["video-describe"] == "video"
    assert rerun._PASS_LLM["video-describe"][0] == "video"
    assert rerun.FLEET_ROLE_MODELS["video"] == rerun.FLEET_ROLE_MODELS["describe"]
    assert "video-describe" in W.SELF_READ_PASSES


def test_fleet_env_sets_the_video_model(monkeypatch):
    from photosearch import admin_api
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.delenv("PHOTOSEARCH_FLEET_VIDEO_MODEL", raising=False)
    assert admin_api._fleet_env()["PHOTOSEARCH_LLM_VIDEO_MODEL"] == "qwen/qwen3.5-9b"


def test_rerun_resolve_model_is_strict_for_video(monkeypatch):
    from photosearch import rerun
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "vis-m")
    with pytest.raises(D.RoleModelNotConfigured):
        rerun._resolve_model("video-describe")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VIDEO_MODEL", "vid-m")
    assert rerun._resolve_model("video-describe") == "vid-m"


# --- review fixes (2026-10-17) ------------------------------------------------

def _script_chat(monkeypatch, answers):
    calls = []
    q = list(answers)

    def chat(**kw):
        calls.append(kw)
        a = q.pop(0)
        if isinstance(a, BaseException):
            raise a
        return a
    monkeypatch.setattr(D, "HAS_OLLAMA", True)
    monkeypatch.setattr(D, "_ollama_chat_with_retry", chat)
    return calls


FR, TS3 = [b"\xff\xd8a", b"\xff\xd8b"], [0.0, 3.0]


def test_transport_failure_raises_a_connection_error_for_the_worker(monkeypatch):
    _script_chat(monkeypatch, [TimeoutError("read timed out")])
    with pytest.raises(D.VideoTransportError) as ei:
        D.describe_video(FR, TS3, 4.0, raise_unusable=True)
    assert isinstance(ei.value, ConnectionError)
    assert isinstance(ei.value, W._TRANSIENT)
    # the eval / non-worker caller still gets None
    _script_chat(monkeypatch, [TimeoutError("read timed out")])
    assert D.describe_video(FR, TS3, 4.0) is None


@pytest.mark.parametrize("bad", ["", None])
def test_empty_or_non_description_is_retried_then_unusable(monkeypatch, bad):
    calls = _script_chat(monkeypatch, [bad, bad])
    with pytest.raises(D.UnusableAnswer):
        D.describe_video(FR, TS3, 4.0, raise_unusable=True)
    assert len(calls) == 2
    assert calls[1]["options"]["temperature"] == 0.4


def test_empty_then_good_answer_recovers_on_the_retry(monkeypatch):
    _script_chat(monkeypatch, ["", GOOD])
    assert D.describe_video(FR, TS3, 4.0, raise_unusable=True) == GOOD


def test_worker_none_is_a_failure_row_and_transport_is_omitted(env, monkeypatch):
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _fake_describe(monkeypatch, None)
    _, photo = _claim(env, pid)
    out = W._process_video_describe(FakeFrames(), [photo], model="m")
    assert out[0]["photo_id"] == pid and "error" in out[0]
    _fake_describe(monkeypatch, D.VideoTransportError("refused"))
    assert W._process_video_describe(FakeFrames(), [photo], model="m") == []


def test_llm_models_for_skips_an_unconfigured_strict_role(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    pm = {"video-describe": [("qwen2.5-vl", "video")],
          "describe": [("llama3.2-vision", "describe")]}
    monkeypatch.setenv("PHOTOSEARCH_LLM_DESCRIBE_MODEL", "qwen/qwen3.5-9b")
    assert W._llm_models_for("video-describe", pm) == set()

    class C:
        worker_id = "me"

        def get_status(self, **kw):
            return {"active_claims": [{"worker_id": "other",
                                       "pass_type": "video-describe"}]}
    unloaded = []
    monkeypatch.setattr(D, "unload_openai_models", lambda m: unloaded.append(set(m)) or [])
    # retiring describe while another worker holds a video-describe claim
    W._release_llm_models(C(), "describe", [], pm, {})
    assert unloaded == [{"qwen/qwen3.5-9b"}]


def test_subjects_result_for_a_replaced_generation_is_deferred(env):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    db.conn.execute("UPDATE video_meta SET scan_generation = 'genbbbbb' WHERE photo_id = ?",
                    (pid,))
    db.conn.commit()
    from photosearch.video_subjects_store import subjects_input
    assert subjects_input(db.conn, pid)["scan_generation"] == "genbbbbb"
    r = env["client"].post("/api/worker/submit-results", json={
        "batch_id": "b", "pass_type": "video-subjects", "video_subjects_results": [
            {"photo_id": pid, "scan_version": SCAN_VERSION,
             "scan_generation": "genaaaaa", "subjects": [], "analysis": {}}]})
    assert r.status_code == 200 and r.json()["written"] == 0
    assert pid in r.json()["deferred_photo_ids"]
    assert db.conn.execute("SELECT subjects_version FROM video_meta WHERE photo_id=?",
                           (pid,)).fetchone()[0] is None
    assert _ledger(db, pid, "video-subjects") == 0


def test_describe_result_for_a_replaced_generation_is_deferred(env, monkeypatch):
    db = env["db"]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    db.conn.execute("UPDATE video_meta SET scan_generation = 'genaaaaa' WHERE photo_id = ?",
                    (pid,))
    db.conn.commit()
    _fake_describe(monkeypatch, GOOD)
    batch, photo = _claim(env, pid)
    assert photo["describe_input"]["scan_generation"] == "genaaaaa"
    fc = FakeFrames()
    out = W._process_video_describe(fc, [photo], model="m")
    assert all(p == {"v": SCAN_VERSION, "g": "genaaaaa"} for _, p in fc.urls)
    # a clear-pass re-scan at the same version lands meanwhile
    db.conn.execute("UPDATE video_meta SET scan_generation = 'genbbbbb' WHERE photo_id = ?",
                    (pid,))
    db.conn.commit()
    resp = _submit(env, batch, out)
    assert resp["written"] == 0 and pid in resp["deferred_photo_ids"]
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (pid,)).fetchone()[0] is None
    assert _ledger(db, pid) == 0


def test_an_exhausted_video_does_not_hold_the_stills_text_passes(env):
    from photosearch.batch_state import fleet_launch_passes
    db = env["db"]
    stills = [db.add_photo(filepath=f"{DIR}/IMG_{i}.JPG", filename=f"IMG_{i}.JPG",
                           date_taken="2091-07-06 09:00:00") for i in range(2)]
    pid = env["video"]("A.MOV")
    _scan(env, pid)
    _finish_stills(db, stills)
    db.conn.execute("UPDATE photos SET categories = NULL, keywords = NULL WHERE id IN (?, ?)",
                    stills)
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'video-describe', ?)", (pid, MAX_PROCESS_ATTEMPTS))
    db.conn.commit()
    bid = register_batch(db, DIR)
    state = batch_state(db, bid)
    st = {s["step"]: s for s in state["steps"]}
    assert st["video-describe"]["state"] == "blocked"
    for p in ("category-content", "keywords"):
        assert st[p]["total"] == 2 and st[p]["state"] == "needs_queue", st[p]
    launch = fleet_launch_passes(state)
    assert "category-content" in launch and "keywords" in launch


def test_rerun_ignores_the_legacy_single_model_var_for_video(monkeypatch):
    from photosearch import rerun
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_MODEL", "legacy-m")
    with pytest.raises(D.RoleModelNotConfigured):
        rerun._resolve_model("video-describe")
    # the request path agrees (same strict resolver)
    with pytest.raises(D.RoleModelNotConfigured):
        D.effective_model("x", "video")

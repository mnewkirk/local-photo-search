"""Video eval harness + blind labelling (docs/plans/video-support.md step 4).

Every test points PHOTOSEARCH_MODEL_EVAL_DIR at tmp_path: nothing here may
touch the owner's real labels."""

import importlib.util
import json
import os
import random
import sys
import types

import pytest

from photosearch import model_eval as me

API = "/api/eval/models"
SECRETS = ("secretvariantA", "secretvariantB", "vendor/secret-model", "holdout")
VIDEO_BYTES = bytes(range(256)) * 40


def _load_harness():
    path = os.path.join(os.path.dirname(__file__), "..", "evals", "video_eval.py")
    spec = importlib.util.spec_from_file_location("video_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ve = _load_harness()


@pytest.fixture(autouse=True)
def _eval_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "model-eval"))
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)


def _sample(ids, durations=None, strata=None, splits=None):
    photos = []
    for i, pid in enumerate(ids):
        photos.append({"photo_id": pid, "stratum": (strata or {}).get(pid, "family-multi"),
                       "split": (splits or {}).get(pid, "holdout"), "filename": f"V{pid}.MP4",
                       "duration_s": (durations or {}).get(pid, 9.0), "width": 1920, "height": 1080})
    return me.save_video_sample({"seed": 1, "photos": photos, "probe_check": [],
                                 "recheck": list(ids[:1])})


def _cache_original(pid, data=VIDEO_BYTES):
    d = me.originals_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / str(pid)).write_bytes(data)


# --------------------------------------------------------------------------
# Split + segmentation
# --------------------------------------------------------------------------

def test_holdout_split_is_deterministic_stratified_and_order_free():
    entries = [{"photo_id": i, "stratum": "a" if i < 30 else "b"} for i in range(1, 46)]
    s1 = me.holdout_split(entries, seed=1)
    shuffled = entries[:]
    random.Random(9).shuffle(shuffled)
    assert me.holdout_split(shuffled, seed=1) == s1
    assert sum(1 for i in range(1, 30) if s1[i] == "holdout") == round(29 / 3)
    assert sum(1 for i in range(30, 46) if s1[i] == "holdout") == round(16 / 3)
    # Adding another stratum never moves a video between splits.
    more = entries + [{"photo_id": 100 + i, "stratum": "c"} for i in range(9)]
    s2 = me.holdout_split(more, seed=1)
    assert all(s2[k] == v for k, v in s1.items())
    assert me.holdout_split(entries, seed=2) != s1


def test_label_segments_are_the_production_schedule_coarsened_to_12():
    segs = me.video_label_segments(1751.0)           # the 29-min clip: 40 prod segs
    assert len(segs) == 12 and segs[0][0] == 0 and abs(segs[-1][1] - 1751.0) < 1e-6
    prod = me.video_production_segments(1751.0)
    assert len(prod) == 40
    assert {round(a, 6) for a, _ in segs} <= {round(a, 6) for a, _ in prod}
    assert len(me.video_label_segments(9.0)) == 3


def test_coarsen_keys_takes_the_majority_time():
    prod = [(0, 3), (3, 6), (6, 9), (9, 12)]
    keys = [["p", 1], ["p", 1], ["s", "dog"], ["none"]]
    assert me.coarsen_keys(prod, keys, [(0, 9), (9, 12)]) == [["p", 1], ["none"]]


# --------------------------------------------------------------------------
# Labels keyed by (pid, seg_version, seg_ix)
# --------------------------------------------------------------------------

def test_subject_labels_are_keyed_by_pid_seg_version_and_ix(monkeypatch):
    v = me.video_seg_version()
    me.save_video_subject_label(7, seg_version=v, n_segments=3, people=[4],
                                segments={0: {"kind": "person", "person_id": 4},
                                          "2": {"kind": "subject", "label": " The  Dog "}})
    data = me.load_video_subject_labels()
    assert set(data["segments"]) == {f"7:{v}:0", f"7:{v}:2"}
    assert data["segments"][f"7:{v}:2"]["label"] == "the dog"
    assert me.video_segment_labels(data, 7, v, 3)[1] is None
    # A different segmentation sees none of them (orphaned, not re-scored).
    assert me.video_segment_labels(data, 7, v + "x", 3) == [None, None, None]
    with pytest.raises(ValueError):
        me.save_video_subject_label(7, seg_version="stale", n_segments=3)
    with pytest.raises(ValueError):          # a person not in "who"
        me.save_video_subject_label(7, seg_version=v, n_segments=3, people=[],
                                    segments={0: {"kind": "person", "person_id": 4}})
    with pytest.raises(ValueError):
        me.save_video_subject_label(7, seg_version=v, n_segments=3,
                                    segments={3: {"kind": "none"}})


def test_quality_labels_three_state_and_recheck_file():
    with pytest.raises(ValueError):
        me.save_video_quality_label(1, {"in_focus": "maybe"})
    with pytest.raises(ValueError):
        me.save_video_quality_label(1, {"in_focus": "yes"}, done=True)
    me.save_video_quality_label(1, {"in_focus": "debatable", "shaky": "no", "noisy": "no",
                                    "keeper": "meh"}, label_set="recheck")
    assert me.load_video_quality_labels("main") == {}
    assert me.load_video_quality_labels("recheck")["1"]["in_focus"] == "debatable"


# --------------------------------------------------------------------------
# The page is blind
# --------------------------------------------------------------------------

@pytest.fixture
def video_eval_env(client, db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 3")]
    _sample(ids, strata={ids[2]: "live-paired"})
    for pid in ids:
        _cache_original(pid)
    v = me.video_seg_version()
    for variant, label in (("secretvariantA", "dog"), ("secretvariantB", "mountain lake")):
        run = me.open_run(ve.PASS_SUBJECT, variant, approach="S1",
                          grounding_model="vendor/secret-model")
        run["items"][str(ids[0])] = {"seg_version": v, "pred": [["s", label], ["none"], ["p", 1]],
                                     "label_segments": [[0, 3], [3, 6], [6, 9]], "present": [1]}
        me.save_run(ve.PASS_SUBJECT, variant, run)
        drun = me.open_run(ve.PASS_DESCRIBE, variant, effective_model="vendor/secret-model",
                           approach="D1", k=6, px=768)
        text = f"A {label} by the water, then it runs off."
        drun["items"][str(ids[0])] = {"status": "ok", "text": text, "text_sha": me.text_sha(text)}
        me.save_run(ve.PASS_DESCRIBE, variant, drun)
    me.save_disputes({str(ids[0]): {"shas": [], "model": "vendor/secret-model", "questions": [
        {"key": "k1", "question": "What animal?", "answers": {"x": "dog", "y": "cat"}}]}},
        ve.PASS_DESCRIBE)
    return client, ids


def test_video_page_responses_never_name_a_variant_model_or_split(video_eval_env):
    client, ids = video_eval_env
    bodies = [client.get(API + "/video/subject").text,
              client.get(API + "/video/subject?set=recheck").text,
              client.get(API + "/video/quality").text,
              client.get(API + "/video/temporal").text,
              client.get(API + "/describe/disputes?pass=video-describe").text]
    for body in bodies:
        for s in SECRETS:
            assert s not in body, s
    subj = client.get(API + "/video/subject").json()
    first = next(v for v in subj["videos"] if v["photo_id"] == ids[0])
    assert sorted(first["segments"][0]["choices"]) == ["dog", "mountain lake"]
    assert first["segments"][2]["choices"] == []            # a person key is not a chip
    live = next(v for v in subj["videos"] if v["photo_id"] == ids[2])
    assert live["kind"] == "live" and live["segments"] == []
    assert "main-subject" in subj["notes"]
    d = client.get(API + "/describe/disputes?pass=video-describe").json()
    assert d["photos"][0]["media"] == "video"
    assert client.get(API + "/describe/disputes?pass=bogus").status_code == 400


def test_subject_put_round_trip_and_validation(video_eval_env):
    client, ids = video_eval_env
    v = me.video_seg_version()
    url = f"{API}/video/subject/{ids[0]}"
    # Done with a segment unanswered -> refused.
    r = client.put(url, json={"seg_version": v, "people": [], "segments": {"0": {"kind": "none"}}})
    assert r.status_code == 400
    r = client.put(url, json={"seg_version": "old", "people": [], "segments": {
        str(i): {"kind": "none"} for i in range(3)}})
    assert r.status_code == 400
    r = client.put(url, json={"seg_version": v, "people": [1], "someone_else": True, "segments": {
        "0": {"kind": "subject", "label": "dog"}, "1": {"kind": "cant_tell"},
        "2": {"kind": "person", "person_id": 1}}})
    assert r.status_code == 200, r.text
    got = client.get(API + "/video/subject").json()
    first = next(x for x in got["videos"] if x["photo_id"] == ids[0])
    assert first["done"] and first["segments"][2]["label"] == {"kind": "person", "person_id": 1}
    assert f"{ids[0]}:{v}:1" in me.load_video_subject_labels()["segments"]
    # The live clip only needs the pairing answer.
    r = client.put(f"{API}/video/subject/{ids[2]}", json={"seg_version": v, "live_pair": "yes"})
    assert r.status_code == 200
    # Recheck labels go to their own file; the first answer is not served there.
    rc = client.get(API + "/video/subject?set=recheck").json()
    assert [x["photo_id"] for x in rc["videos"]] == [ids[0]]
    assert rc["videos"][0]["who"] is None


def test_original_is_range_served_from_the_local_cache(video_eval_env):
    client, ids = video_eval_env
    r = client.get(f"{API}/video/original/{ids[0]}", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206 and r.content == VIDEO_BYTES[10:20]
    assert r.headers["content-type"].startswith("video/mp4")
    assert client.get(f"{API}/video/original/999999").status_code == 404


def test_temporal_label_keyed_by_text(video_eval_env):
    client, ids = video_eval_env
    t = client.get(API + "/video/temporal").json()["videos"][0]["texts"][0]
    r = client.put(f"{API}/video/temporal/{t['sha']}", json={"photo_id": ids[0], "answer": "partly"})
    assert r.status_code == 200
    assert me.load_temporal_labels()[t["sha"]]["answer"] == "partly"
    assert client.put(f"{API}/video/temporal/{t['sha']}",
                      json={"photo_id": ids[0], "answer": "maybe"}).status_code == 400


def test_video_labeller_notes_exist():
    from photosearch.eval_api import LABELLER_NOTES
    for k in ("main-subject", "in_focus", "shaky", "noisy", "keeper", "temporal"):
        assert f"video:{k}" in LABELLER_NOTES


# --------------------------------------------------------------------------
# TransportError never cached; context overflow cached as rejected
# --------------------------------------------------------------------------

@pytest.fixture
def frames(monkeypatch, tmp_path):
    from PIL import Image
    jpg = tmp_path / "f.jpg"
    Image.new("RGB", (32, 24), (90, 90, 90)).save(jpg)
    monkeypatch.setattr(me, "cached_frame_jpeg", lambda pid, t, px=768, grab=None: jpg)
    _sample([11, 12])
    return jpg


class _HTTP400(Exception):
    def __init__(self):
        super().__init__("HTTP Error 400: Bad Request")

    def read(self):
        return b'{"error": "Context size has been exceeded."}'


def test_describe_transport_error_is_not_cached_overflow_is_rejected(frames, monkeypatch):
    from photosearch import describe

    def chat(model, messages, **kw):
        if "images" in messages[0] and len(messages[0]["images"]) > 3:
            raise _HTTP400()
        raise ConnectionError("Connection refused")
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", chat)
    run = ve.run_describe("d1", "D1", k=6, log=lambda *a: None)
    assert {k: v["status"] for k, v in run["items"].items()} == {"11": "rejected", "12": "rejected"}
    run = ve.run_describe("d1k3", "D1", k=3, log=lambda *a: None)
    assert run["items"] == {}                               # transport: nothing cached
    assert me.load_run(ve.PASS_DESCRIBE, "d1k3")["items"] == {}

    calls = []

    def ok(model, messages, **kw):
        calls.append(1)
        return "A dog runs along the beach and then jumps into the water."
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", ok)
    run = ve.run_describe("d1k3", "D1", k=3, log=lambda *a: None)
    assert run["items"]["11"]["status"] == "ok" and run["items"]["11"]["text_sha"]
    # Cached: a re-run makes no calls.
    n = len(calls)
    ve.run_describe("d1k3", "D1", k=3, log=lambda *a: None)
    assert len(calls) == n


def test_grounding_transport_failure_is_not_cached(frames, fake_chat):
    run = ve.run_grounding("g", grounder=lambda path, model: None, log=lambda *a: None)
    assert run["items"] == {}
    run = ve.run_grounding("g", grounder=_fake_grounder, log=lambda *a: None)
    assert set(run["items"]) == {"11", "12"}


def _fake_grounder(path, model):
    from photosearch import describe
    describe._ollama_chat_with_retry(model=model, messages=[], role="aesthetics")
    return [{"label": "a dog", "bbox": [0.3, 0.3, 0.7, 0.8]}]


@pytest.fixture
def fake_chat(monkeypatch):
    from photosearch import describe
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", lambda *a, **k: "ok")


def test_classify_failure():
    assert me.classify_failure(_HTTP400()) == "rejected"
    assert me.classify_failure(RuntimeError("Context size has been exceeded")) == "rejected"
    assert me.classify_failure(ConnectionError("refused")) == "transport"
    assert me.classify_failure(me.TransportError("HTTP Error 400: Bad Request")) == "transport"


# --------------------------------------------------------------------------
# run-faces against the video_scan contract (stubbed ScanData)
# --------------------------------------------------------------------------

def _det(x, emb):
    return {"bbox": (x, 300, x + 300, 600), "det_score": 0.9, "embedding": emb, "sharpness": 10.0}


@pytest.fixture
def stub_scan(monkeypatch):
    import numpy as np
    e = np.zeros(512)
    e[0] = 1.0

    class ScanData:
        W, H, duration, fps = 1920.0, 1080.0, 9.0, 30.0
        frames = [{"t": (i + 0.5) / 4, "faces": [_det(810, e.tolist())]} for i in range(36)]
        measures = {"shake": 0.4, "noise": 2.0, "tenengrad_max": 900.0}

    calls = []
    mod = types.ModuleType("photosearch.video_scan")

    def scan_video(path, params=None, detect=None, want_frames=True):
        calls.append(path)
        return ScanData()
    mod.scan_video = scan_video
    monkeypatch.setitem(sys.modules, "photosearch.video_scan", mod)
    import photosearch
    monkeypatch.setattr(photosearch, "video_scan", mod, raising=False)
    return calls


def test_run_faces_and_quality_use_the_scan_contract_and_detection_cache(stub_scan, frames, fake_chat):
    for pid in (11, 12):
        _cache_original(pid)
    run = ve.run_faces("F1", "F1", conn=None, log=lambda *a: None)
    it = run["items"]["11"]
    assert it["pred"] == [["else"]] * 3 and it["seg_version"] == me.video_seg_version()
    assert len(stub_scan) == 2
    ve.run_faces("F0", "F0", conn=None, log=lambda *a: None)
    assert len(stub_scan) == 2                  # rescored from the detection cache
    fr, meta = me.load_detections(11, ve.DET_TAG)
    assert len(fr) == 36 and meta["W"] == 1920.0 and len(fr[0]["faces"][0]["embedding"]) == 512
    q = ve.run_quality("Qm", "Qm", log=lambda *a: None)
    m = q["items"]["11"]["measures"]
    assert m["shake"] == 0.4 and m["noise"] == 2.0 and m["focus_consistency"] == 1.0
    # S1 over a stubbed grounder: the dog is the subject of every segment.
    s = ve.run_faces("S1", "S1", conn=None, grounding="g", grounder=_fake_grounder,
                     log=lambda *a: None)
    assert s["items"]["11"]["pred"] == [["s", "dog"]] * 3


# --------------------------------------------------------------------------
# Report gate math on synthetic labels
# --------------------------------------------------------------------------

def _label(pid, people, kinds):
    v = me.video_seg_version()
    me.save_video_subject_label(pid, seg_version=v, n_segments=len(kinds), people=people,
                                segments=dict(enumerate(kinds)))


def _subject_run(name, approach, preds, present):
    run = me.open_run(ve.PASS_SUBJECT, name, approach=approach)
    for pid, pred in preds.items():
        run["items"][str(pid)] = {"seg_version": me.video_seg_version(), "pred": pred,
                                  "label_segments": [[0, 3], [3, 6], [6, 9]],
                                  "present": present.get(pid, []), "latency_s": 2.0}
    me.save_run(ve.PASS_SUBJECT, name, run)
    return run


def test_subject_metrics_and_gates_on_synthetic_labels():
    ids = [1, 2, 3, 4]
    _sample(ids)
    P = {"kind": "person", "person_id": 5}
    DOG = {"kind": "subject", "label": "the dog"}
    NONE = {"kind": "none"}
    _label(1, [5], [P, P, P])
    _label(2, [5], [P, DOG, NONE])
    _label(3, [], [DOG, DOG, {"kind": "cant_tell"}])
    _label(4, [5], [DOG, DOG, DOG])
    perfect = {1: [["p", 5]] * 3, 2: [["p", 5], ["s", "dog"], ["none"]],
               3: [["s", "dogs"], ["s", "dog"], ["none"]], 4: [["s", "dog"]] * 3}
    run = _subject_run("perfect", "S1", perfect, {1: [5], 2: [5], 4: [5]})
    s = ve.summarize_subject(ve.subject_metrics(run, me.load_video_subject_labels(),
                                                me.load_video_sample()))
    assert s["videos"] == 4 and s["seg_acc"]["all"] == 1.0 and s["seg_n"]["all"] == 11
    assert s["seg_acc"]["person"] == 1.0 and s["seg_acc"]["non-person"] == 1.0
    assert s["top"] == 1.0 and s["mae"] == 0.0 and s["presence_p"] == 1.0
    g = ve.subject_gates(s)
    assert g["ship_subject_filter"] and g["ships"] == "subject filter"

    # Video 1 misses the person entirely; video 4 names someone not in it.
    worse = dict(perfect)
    worse[1] = [["none"]] * 3
    run = _subject_run("worse", "F1", worse, {2: [5], 4: [5, 9]})
    s = ve.summarize_subject(ve.subject_metrics(run, me.load_video_subject_labels(),
                                                me.load_video_sample()))
    assert s["seg_acc"]["person"] == pytest.approx(1 / 4)       # 1 of 4 person segments
    assert s["seg_acc"]["non-person"] == 1.0
    assert s["top"] == 0.75
    # tp: videos 2 and 4; fp: person 9 on video 4; fn: video 1.
    assert s["presence_p"] == pytest.approx(2 / 3) and s["presence_r"] == pytest.approx(2 / 3)
    # Share pairs: video 1 {person: 1 vs 0, none: 0 vs 1}, video 2 three exact
    # thirds, videos 3 and 4 one exact pair each -> 2 errors of 1.0 over 7.
    assert s["mae"] == pytest.approx(2 / 7)
    g = ve.subject_gates(s)
    assert not g["ship_subject_filter"] and not g["seg_acc_person"] and not g["share_mae"]
    assert g["seg_acc_non_person"] and not g["named_presence_p"] and g["ships"] == "nothing"

    # A label the comparer judged the same counts as right.
    me.write_pass_file(me.VIDEO, "label_judgements.json", {"dog|puppy": {"same": True}})
    same = ve.make_same(me.read_pass_file(me.VIDEO, "label_judgements.json", {}))
    assert same(["s", "dog"], ["s", "puppy"]) and not same(["s", "dog"], ["s", "cat"])
    assert not same(["p", 5], ["else"])



def test_dev_split_never_reaches_the_holdout_numbers():
    _sample([1, 2], splits={1: "dev", 2: "holdout"})
    _label(1, [5], [{"kind": "person", "person_id": 5}] * 3)
    _label(2, [5], [{"kind": "person", "person_id": 5}] * 3)
    run = _subject_run("x", "F1", {1: [["none"]] * 3, 2: [["p", 5]] * 3}, {})
    hold = ve.summarize_subject(ve.subject_metrics(run, me.load_video_subject_labels(),
                                                   me.load_video_sample(), split="holdout"))
    dev = ve.summarize_subject(ve.subject_metrics(run, me.load_video_subject_labels(),
                                                  me.load_video_sample(), split="dev"))
    assert hold["seg_acc"]["all"] == 1.0 and dev["seg_acc"]["all"] == 0.0


def test_n_below_3_never_passes_a_gate():
    s = ve.summarize_subject(ve.subject_metrics({"items": {}}, {"who": {}, "segments": {}},
                                                {"photos": []}))
    assert s["top"] is None and s["mae"] is None
    assert not any(v for k, v in ve.subject_gates(s).items() if isinstance(v, bool))


def test_pick_winner_tie_band_prefers_the_simpler_approach():
    scores = {"s1": {"approach": "S1", "seg_acc": {"all": 0.80}},
              "s2": {"approach": "S2", "seg_acc": {"all": 0.84}},
              "f1": {"approach": "F1", "seg_acc": {"all": 0.70}}}
    w, tied = ve.pick_winner(scores, ["s1", "s2"])
    assert w == "s1" and tied == ["s1", "s2"]
    scores["s2"]["seg_acc"]["all"] = 0.86
    assert ve.pick_winner(scores, ["s1", "s2"])[0] == "s2"
    scores["s1"]["seg_acc"]["person"] = 0.80
    scores["f1"]["seg_acc"]["person"] = 0.77
    assert ve.f1_vs_s1(scores, "f1", "s1") is False


def test_describe_gate_vs_poster_only():
    win = {"temporal_yes": 0.70, "right_rate": 0.80, "per_video": {1: (3, 0), 2: (2, 1)}}
    d0 = {"temporal_yes": 0.45, "right_rate": 0.79, "per_video": {1: (2, 1), 2: (2, 1)}}
    g = ve.describe_gate(win, d0)
    assert g["pass"] and g["temporal_delta"] == pytest.approx(0.25)
    win["temporal_yes"] = 0.50
    assert not ve.describe_gate(win, d0)["pass"]          # +0.01 right, n too small for p<0.1
    assert ve.sign_test_p(8, 0) < 0.1 and ve.sign_test_p(1, 1) == 1.0


def test_quality_threshold_fitted_on_dev_scored_on_holdout():
    ids = list(range(1, 13))
    splits = {i: ("dev" if i <= 6 else "holdout") for i in ids}
    _sample(ids, splits=splits)
    shake = {i: (2.0 if i % 2 else 0.1) for i in ids}
    for i in ids:
        me.save_video_quality_label(i, {"in_focus": "yes", "shaky": "yes" if i % 2 else "no",
                                        "noisy": "debatable", "keeper": "keep"})
    run = me.open_run(ve.PASS_QUALITY, "Qm", approach="Qm")
    for i in ids:
        run["items"][str(i)] = {"status": "ok", "measures": {"shake": shake[i], "noise": 1.0,
                                                             "focus_consistency": 0.9}}
    me.save_run(ve.PASS_QUALITY, "Qm", run)
    q = ve.quality_scores({"Qm": run}, me.load_video_quality_labels(), me.load_video_sample())
    c = q["shaky"]["Qm"]
    assert c["p"] == 1.0 and c["r"] == 1.0 and c["gate"] and c["threshold"] == 2.0
    assert q["noisy"]["Qm"]["p"] is None and not q["noisy"]["Qm"]["gate"]   # debatable = neither
    rho, ci = ve.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    assert rho == pytest.approx(1.0)
    rho, ci = ve.spearman([1, 2, 3, 4, 5, 6], [1, 3, 2, 5, 4, 6])
    assert 0 < rho < 1 and ci[0] < rho < ci[1]


def test_agreement_kappa():
    assert ve.cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert ve.cohen_kappa([]) is None
    _sample([1, 2])
    for ls in ("main", "recheck"):
        me.save_video_quality_label(1, {"in_focus": "yes", "shaky": "no", "noisy": "no",
                                        "keeper": "keep"}, label_set=ls)
    ag = ve.self_agreement()
    assert ag["quality"]["shaky"]["n"] == 1 and ag["quality"]["shaky"]["agreement"] == 1.0


# --------------------------------------------------------------------------
# Sampling from a read-only DB
# --------------------------------------------------------------------------

def test_sample_draws_strata_from_prescan_facts(tmp_path):
    from photosearch.db import PhotoDB
    from photosearch.video_registry import register_video
    root = tmp_path / "photos"
    folder = root / "2026" / "2026-09-12_phone-matt"
    folder.mkdir(parents=True)
    db_path = str(tmp_path / "p.db")
    db = PhotoDB(db_path, photo_root=str(root))
    db.set_photo_root(str(root))
    db.add_photo(filepath="2026/2026-09-12_phone-matt/IMG_1.JPG", filename="IMG_1.JPG",
                 date_taken="2026-09-12 10:00:00")
    db.conn.execute("UPDATE photos SET categories = ? WHERE filename = 'IMG_1.JPG'",
                    (json.dumps(["soccer", "field"]),))
    ids = []
    for i, dur in enumerate((30.0, 8.0, 25.0)):
        p = folder / f"V{i}.MP4"
        p.write_bytes(b"x")
        pid = register_video(db, str(p), date_taken="2026-09-12 10:00:00")
        db.conn.execute("UPDATE video_meta SET probe_version=1, kind='clip', duration_s=?, "
                        "width=1920, height=1080 WHERE photo_id=?", (dur, pid))
        ids.append(pid)
    db.conn.commit()
    db.close()
    conn = me.open_db_readonly(db_path)
    data = ve.draw_sample(conn, strata={"sports-long": 5, "sports-short": 5})
    got = {p["photo_id"]: p["stratum"] for p in data["photos"]}
    assert got == {ids[0]: "sports-long", ids[1]: "sports-short", ids[2]: "sports-long"}
    assert data["shortfall"] == {"sports-long": 3, "sports-short": 4}
    assert all(p["split"] in ("dev", "holdout") for p in data["photos"])
    assert ve.draw_sample(conn, strata={"sports-long": 5, "sports-short": 5}) == data


def test_report_marks_out_of_scope_gates_not_evaluated_never_failed():
    """Quality is out of scope by owner decision (2026-10-25, D11), and the
    describe gate has no labels: both print "not evaluated", neither a fail."""
    _sample([1, 2])
    me.save_video_quality_label(1, {"in_focus": "yes", "shaky": "no", "noisy": "no",
                                    "keeper": "keep"})
    rep = ve.build_report()
    assert set(rep["not_evaluated"]) == {"quality", "describe"}
    assert "owner decision 2026-10-25" in rep["not_evaluated"]["quality"]
    assert rep["quality"] == {} and rep["keeper"] == {}
    assert rep["describe_winner"]["gate_vs_D0"] == {"not_evaluated": ve.DESCRIBE_NO_LABELS}
    assert rep["agreement"] == {"not_measured": "recheck set not labelled"}
    lines = []
    ve.render(rep, log=lambda *a: lines.append(" ".join(map(str, a))))
    out = "\n".join(lines)
    assert "not evaluated (owner decision 2026-10-25" in out
    assert "gate vs D0: not evaluated" in out and "not measured" in out
    assert "fail" not in out.split("== Description")[1]

    # Opting back in scores the quality gate as before.
    rep = ve.build_report(not_evaluated={})
    assert "quality" not in rep["not_evaluated"]
    assert set(rep["quality"]) == set(me.QUALITY_FLAGS)


def test_describe_gate_is_evaluated_once_it_has_labels():
    _sample([1])
    me.save_temporal_label("abc", 1, "yes")
    rep = ve.build_report()
    assert "describe" not in rep["not_evaluated"]

"""evals/describe_eval.py — sampling, the run cache, and the automatic screens.

No model and no network: the chat entry point and the image fetch are stubbed,
and the real `describe.describe_photo` runs on top of the stub so retry,
degeneration recovery and the llava fallback are exercised as they ship."""

import importlib.util
import json
import os
import sqlite3

import pytest

from photosearch import describe
from photosearch import model_eval as me


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "describe_eval.py")
    spec = importlib.util.spec_from_file_location("describe_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()

GOOD = ("A sailboat with a white sail glides across a calm blue bay under a clear "
        "sky, with green hills along the far shore and a small dock in front.")
LOOP = "the boat " * 40


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(tmp_path / "vt"))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    for v in ("PHOTOSEARCH_LLM_DESCRIBE_MODEL", "PHOTOSEARCH_LLM_VISUAL_MODEL",
              "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.setenv(v, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(v)
    monkeypatch.setattr(describe, "HAS_OLLAMA", True)
    monkeypatch.setattr(describe, "_encode_image_for_ollama", lambda p: "b64")
    monkeypatch.setattr(me, "lmstudio_loaded", lambda *a, **kw: None)


def _db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, description TEXT)")
    conn.execute("CREATE TABLE generations (id INTEGER PRIMARY KEY, photo_id, "
                 "text_type, model_used)")
    conn.executemany("INSERT INTO photos VALUES (?, ?)", [
        (1, "A dog on a beach."),
        (2, 'A cafe sign reads "Blue Door" above the entrance.'),
        (3, "A menu board lists coffee prices."),
        (4, "Mountains at dusk."),
        (5, "A banner hangs over the street."),
    ])
    conn.execute("INSERT INTO generations (photo_id, text_type, model_used) "
                 "VALUES (1, 'describe', 'qwen/qwen3.5-9b')")
    conn.commit()
    return str(path)


def _visual_sample(ids):
    return {"photos": [{"photo_id": i, "stratum": "random"} for i in ids]}


def test_sample_adds_a_text_stratum_without_duplicates(tmp_path):
    conn = me.open_db_readonly(_db(tmp_path / "p.db"))
    photos = D.draw_sample(conn, text_n=2, exclude=[5], text_ids=[2],
                           visual_sample=_visual_sample([1, 4]))
    strata = {p["photo_id"]: p["stratum"] for p in photos}
    assert strata[1] == "random" and strata[4] == "random"
    assert strata[2] == "text"                     # the pinned id goes first
    text = [p for p in photos if p["stratum"] == "text"]
    assert len(text) == 2 and 5 not in strata      # vetoed
    assert {p["photo_id"] for p in text} <= {2, 3}


def _chat(monkeypatch, script):
    """Stub the chat entry point with a queue of answers (or exceptions),
    recording the model each call asked for."""
    q = list(script)
    asked = []

    def chat(*a, **kw):
        asked.append(kw.get("model"))
        nxt = q.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", chat)
    return asked


def _setup(tmp_path, ids):
    me.save_sample("describe", [{"photo_id": i, "stratum": "random"} for i in ids])
    for i in ids:
        (me.originals_dir()).mkdir(parents=True, exist_ok=True)
        (me.originals_dir() / str(i)).write_bytes(b"jpeg")


def test_run_caches_real_answers_and_skips_transport(tmp_path, monkeypatch):
    _setup(tmp_path, [1, 2, 3])
    _chat(monkeypatch, [
        GOOD,                                   # 1: clean
        LOOP, LOOP, LOOP, GOOD,                 # 2: degenerate x3, llava rescues
        ConnectionError("refused"),             # 3: LM Studio down
    ])
    run = D.run_variant("v", model="gemma", log=lambda *_: None)
    assert os.environ["PHOTOSEARCH_LLM_DESCRIBE_MODEL"] == "gemma"
    assert run["effective_model"] == "gemma"
    items = run["items"]
    assert set(items) == {"1", "2"}             # 3 not cached — re-run fills it
    assert items["1"]["text"] == GOOD and not items["1"]["retried"]
    two = items["2"]
    assert two["degenerate_first"] and two["retried"] and two["fallback_called"]
    assert two["text"] == GOOD and two["calls"] == 4
    assert items["1"]["text_sha"] == me.text_sha(GOOD)

    # Re-run fills the gap only.
    _chat(monkeypatch, [GOOD])
    run = D.run_variant("v", model="gemma", log=lambda *_: None)
    assert set(run["items"]) == {"1", "2", "3"}


def test_an_answered_but_invalid_description_is_cached_as_null(monkeypatch, tmp_path):
    _setup(tmp_path, [1])
    _chat(monkeypatch, [""])                     # model answered nothing usable
    run = D.run_variant("v", model="gemma", log=lambda *_: None)
    assert run["items"]["1"]["text"] is None


def test_summarize_and_screen(tmp_path):
    run = {"variant": "v", "effective_model": "m", "items": {
        "1": {"text": GOOD, "latency_s": 9.0},
        "2": {"text": None, "latency_s": 1.0},
        "3": {"text": GOOD, "truncated": True, "retried": True, "latency_s": 2.0},
        "4": {"text": GOOD, "fallback_called": True, "latency_s": 3.0},
    }}
    r = D.summarize(run, use_clip=False)
    assert (r["n"], r["answered"], r["truncated"], r["retried"], r["fallback"]) == \
        (4, 3, 1, 1, 1)
    assert r["lat_p50"] == 2.0                   # JIT-loading first call dropped
    assert r["screen_out"] is True               # 2 bad of 4
    assert "SCREEN OUT" in D.render([r])


def test_text_accuracy():
    recall, invented = D.text_accuracy(
        'A sign reads "Blue Door Cafe" and a poster says "Grand Sale".',
        "BLUE DOOR CAFE open daily")
    assert recall == pytest.approx(3 / 5)        # blue door cafe of blue door cafe open daily
    assert invented == 1                         # "Grand Sale" is not there


def test_report_includes_stored_with_its_model(tmp_path, monkeypatch):
    db = _db(tmp_path / "p.db")
    me.save_sample("describe", [{"photo_id": 1, "stratum": "random"},
                                {"photo_id": 2, "stratum": "text"}])
    me.write_pass_file("describe", "text_truth.json",
                       {"2": {"text": "Blue Door", "done": True}})
    rows = D.build_report(db=db, use_clip=False)
    stored = [r for r in rows if r["variant"] == "stored"][0]
    assert stored["n"] == 2 and stored["answered"] == 2
    assert stored["text_n"] == 1 and stored["text_recall"] == 1.0
    run = D.stored_run(me.open_db_readonly(db))
    assert run["items"]["1"]["stored_model"] == "qwen/qwen3.5-9b"


CUT = "A sailboat with a white sail glides across a calm blue bay under a clear sky and"


def test_unusable_answers_are_cached_with_their_reason(tmp_path, monkeypatch):
    _setup(tmp_path, [1, 2])
    _chat(monkeypatch, [CUT, CUT, CUT, CUT,     # 1: cut off every time, llava too
                        ""])                    # 2: empty
    run = D.run_variant("v", model="gemma", log=lambda *_: None)
    one, two = run["items"]["1"], run["items"]["2"]
    assert one["text"] is None and "ends mid-sentence" in one["unusable"]
    assert two["text"] is None and two["unusable"] == "not a description"
    r = D.summarize(run, use_clip=False)
    assert r["unusable_reasons"] == {"describe: ends mid-sentence": 1, "not a description": 1}
    # An old run's bare null still loads.
    old = {"variant": "o", "items": {"1": {"text": None}}}
    assert D.summarize(old, use_clip=False)["unusable_reasons"] == {"reason not recorded": 1}


def test_run_records_spec_and_refuses_a_stale_resume(tmp_path, monkeypatch):
    _setup(tmp_path, [1, 2])
    _chat(monkeypatch, [GOOD])
    run = D.run_variant("v", model="gemma", limit=1, log=lambda *_: None)
    assert run["spec_hash"] and me.spec_status("describe", run)["stale"] is False
    monkeypatch.setattr(describe, "DESCRIBE_PROMPT", describe.DESCRIBE_PROMPT + " Be brief.")
    with pytest.raises(SystemExit, match="changed: prompt"):
        D.run_variant("v", model="gemma", log=lambda *_: None)
    assert me.spec_status("describe", me.load_run("describe", "v"))["why"] == \
        "spec changed: prompt"
    _chat(monkeypatch, [GOOD])
    run = D.run_variant("v", model="gemma", force_stale=True, log=lambda *_: None)
    assert set(run["items"]) == {"1", "2"} and me.spec_status("describe", run)["stale"]
    assert "STALE" in D.render([D.summarize(run, use_clip=False)])


def test_report_replays_current_guards_over_cached_texts():
    """A text cached before `ends mid-sentence` existed is bad today."""
    run = {"variant": "v", "effective_model": "m", "items": {
        str(i): {"text": CUT if i < 2 else GOOD, "latency_s": 1.0} for i in range(10)}}
    r = D.summarize(run, use_clip=False)
    assert r["bad"] == 0 and not r["screen_out"]                 # as stored
    assert r["replay_bad"] == 2 and r["replay_screen_out"]       # 20% > 10%
    assert r["replay_reasons"] == {"ends mid-sentence": 2}
    # Format problems count in the stored screen AND in the summary's bad count.
    run["items"]["5"]["text"] = "**Search Index Description:** " + GOOD
    r = D.summarize(run, use_clip=False)
    assert r["format"] == 1 and r["bad"] == D.bad_outputs(r) == 1
    assert "replayed" in D.render([r])


def test_fair_score_uses_original_questions_and_charges_unanswered_photos():
    a, b = me.text_sha(GOOD), me.text_sha("Other text.")
    me.save_disputes({
        "1": {"shas": [a, b], "questions": [
            {"key": "q1", "question": "Boat?", "answers": {a: "sailboat", b: "canoe"}},
            {"key": "q2", "question": "Dock?", "answers": {a: "yes", b: None},
             "added_by": "v"}]},
        "2": {"shas": [b], "questions": [
            {"key": "q3", "question": "Sky?", "answers": {b: "clear"}}]}})
    for k, c in (("q1", ["canoe"]), ("q2", ["yes"]), ("q3", ["clear"])):
        me.save_dispute_label(k, c)
    items = {"1": {"text": GOOD, "text_sha": a}, "2": {"text": None}}
    t = D.dispute_tally(items)
    assert (t["disp_right"], t["disp_wrong"], t["disp_orig_right"], t["disp_orig_wrong"]) == \
        (1, 1, 0, 1)
    # Photo 2 is unanswered: its one original question counts as wrong.
    assert D.fair_score(t["disp_records"], original=True) == 0.0
    assert D.fair_score(t["disp_records"], original=False) == pytest.approx(1 / 3)

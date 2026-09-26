"""evals/verify_eval.py — planting, the independence rule, the transport
guard, and the catch / false-rejection arithmetic."""

import importlib.util
import os
import random

import pytest

from photosearch import describe
from photosearch import model_eval as me


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "verify_eval.py")
    spec = importlib.util.spec_from_file_location("verify_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = _load()

CLEAN1 = "Two children play on a red slide in a park."
CLEAN2 = "A white sailboat crosses a calm bay."
CLEAN3 = "A dog sleeps on a porch."
WRONG = "A man rides a horse along a beach, with a lighthouse behind him."


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    for v in ("PHOTOSEARCH_LLM_VERIFY_MODEL", "PHOTOSEARCH_LLM_VISUAL_MODEL",
              "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.setenv(v, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(v)
    monkeypatch.setattr(me, "lmstudio_loaded", lambda *a, **kw: None)


def _describe_run(effective="qwen/qwen3.5-9b"):
    run = me.open_run("describe", "src", effective_model=effective, prompt_sha=None)
    for pid, t in ((1, CLEAN1), (2, CLEAN2), (3, CLEAN3), (4, WRONG), (5, "unlabelled.")):
        run["items"][str(pid)] = {"text": t, "text_sha": me.text_sha(t)}
    me.save_run("describe", "src", run)
    for pid, t in ((1, CLEAN1), (2, CLEAN2), (3, CLEAN3)):
        me.save_claim(me.text_sha(t), pid, len(me.segment_claims(t)), [])
    me.save_claim(me.text_sha(WRONG), 4, len(me.segment_claims(WRONG)), [1])
    for pid in range(1, 6):
        me.originals_dir().mkdir(parents=True, exist_ok=True)
        (me.originals_dir() / str(pid)).write_bytes(b"jpeg")


def test_planters_are_deterministic_and_mark_their_span():
    rnd = random.Random(1)
    text, spans = V.plant_colour(CLEAN1, rnd)
    assert "red slide" not in text and spans[0].endswith("slide")
    text, spans = V.plant_count(CLEAN1, random.Random(1))
    assert text.startswith("Four children") and spans == ["four children"]
    text, spans = V.plant_object(CLEAN3, random.Random(1))
    assert text.startswith(CLEAN3) and spans[0] in text and spans[0] not in CLEAN3
    assert V.plant_colour(CLEAN3, rnd) is None and V.plant_count(CLEAN3, rnd) is None
    # Same seed, same plant.
    assert V.plant_object(CLEAN3, random.Random(1)) == V.plant_object(CLEAN3, random.Random(1))


def test_build_sets(tmp_path):
    _describe_run()
    data = V.build_sets("src")
    kinds = [(it["kind"], it["photo_id"]) for it in data["items"]]
    assert ("real", 4) in kinds and ("clean", 1) in kinds
    assert not any(pid == 5 for _, pid in kinds)              # unlabelled: skipped
    planted = [it for it in data["items"] if it["kind"] == "planted"]
    assert len(planted) == 3 and {p["type"] for p in planted} <= {"object", "colour", "count"}
    real = [it for it in data["items"] if it["kind"] == "real"][0]
    assert real["spans"] == ["with a lighthouse behind him."]
    # Unconfirmed plants are not scored.
    assert all(it["kind"] != "planted" for it in V.scored_items(data))


def test_run_refuses_the_describe_model_as_verifier():
    _describe_run(effective="google/gemma-4-e2b")
    V.build_sets("src")
    with pytest.raises(SystemExit, match="independent"):
        V.run_variant("x", model="google/gemma-4-e2b", log=lambda *_: None)


def test_a_dead_backend_is_not_scored_as_all_correct(monkeypatch):
    _describe_run()
    V.build_sets("src")
    monkeypatch.setattr(describe, "_ollama_chat_with_retry",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    run = V.run_variant("e2b", model="google/gemma-4-e2b", log=lambda *_: None)
    assert run["items"] == {}           # llm_verify_description returned [] — not cached


def test_scoring(monkeypatch):
    _describe_run()
    data = V.build_sets("src")
    for it in data["items"]:
        if it["kind"] == "planted":
            me.confirm_planted(it["id"], it["photo_id"] != 3)   # photo 3's plant rejected
    answers = {}
    sets = me.load_verify_sets()
    for it in sets["items"]:
        if it["kind"] == "planted":
            answers[it["text"]] = "WRONG: " + it["spans"][0]      # catches every plant
        elif it["kind"] == "real":
            answers[it["text"]] = "WRONG: a lighthouse"
        elif it["photo_id"] == 1:
            answers[it["text"]] = "WRONG: the park"               # a false rejection
        else:
            answers[it["text"]] = "ALL CORRECT"

    def chat(*a, messages=None, **kw):
        text = messages[0]["content"]
        # Longest first: an object plant APPENDS to a clean text.
        for k in sorted(answers, key=len, reverse=True):
            if k in text:
                return answers[k]
        raise AssertionError("unexpected text")
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", chat)
    V.run_variant("e2b", model="google/gemma-4-e2b", log=lambda *_: None)
    sets, rows = V.build_report()
    r = rows[0]
    assert r["planted"] == [2, 2, 2]           # 2 confirmed plants, both caught + matched
    assert r["real"] == [1, 1, 1]
    assert r["clean"] == [3, 1]                # 1 of 3 clean descriptions flagged
    assert "planted caught" in V.render(sets, rows)


def test_planted_api_round_trip(client):
    _describe_run()
    V.build_sets("src")
    got = client.get("/api/eval/models/verify/planted").json()
    assert got["progress"] == {"done": 0, "total": 3}
    iid = got["items"][0]["id"]
    assert client.put(f"/api/eval/models/verify/planted/{iid}",
                      json={"confirmed": True}).status_code == 200
    assert client.put("/api/eval/models/verify/planted/nope",
                      json={"confirmed": True}).status_code == 404
    assert client.get("/api/eval/models/verify/planted").json()["progress"]["done"] == 1

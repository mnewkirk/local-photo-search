"""Describe 'Differences': the facts a photo's descriptions disagree on,
extracted by a text model, labelled by the owner, scored per description."""

import importlib.util
import json
import os

import pytest

from photosearch import model_eval as me


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "describe_eval.py")
    spec = importlib.util.spec_from_file_location("describe_eval_disputes_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load()

A = "Four boys play soccer on a green turf field."
B = "Two boys and a girl play field hockey on grass, with a parked car behind them."


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    for v in ("PHOTOSEARCH_LLM_TEXT_MODEL", "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.setenv(v, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(v)


def _runs(ids):
    me.save_sample("describe", [{"photo_id": i, "stratum": "random"} for i in ids])
    for v, text in (("va", A), ("vb", B)):
        run = me.open_run("describe", v, effective_model=v, prompt_sha=None)
        for i in ids:
            run["items"][str(i)] = {"text": text, "text_sha": me.text_sha(text)}
        me.save_run("describe", v, run)


def test_parse_keeps_real_disputes_only():
    raw = """```json
    [{"question": "What sport?", "answers": {"D1": "soccer", "D2": "field hockey"}},
     {"question": "Grass colour?", "answers": {"D1": "green", "D2": "Green."}},
     {"question": "Is there a car?", "answers": {"D1": "not mentioned", "D2": "parked car"}},
     {"question": "bad", "answers": "x"}]
    ```"""
    qs = D._parse_disputes(raw, ["D1", "D2"])
    assert [q["question"] for q in qs] == ["What sport?", "Is there a car?"]
    assert qs[1]["answers"] == {"D1": None, "D2": "parked car"}
    assert D._parse_disputes("I cannot compare these.", ["D1"]) is None
    assert D._parse_disputes("[]", ["D1", "D2"]) == []


def test_build_disputes_is_blind_resumable_and_keyed_to_the_texts():
    _runs([1, 2])
    prompts = []

    def chat(model=None, messages=None, **kw):
        prompts.append(messages[0]["content"])
        return json.dumps([{"question": "What sport?",
                            "answers": {"D1": "soccer", "D2": "field hockey"}}])
    D.build_disputes(model="judge", chat=chat, log=lambda *_: None)
    assert len(prompts) == 2
    assert "va" not in prompts[0] and "vb" not in prompts[0]      # blind
    d = me.load_disputes()["1"]
    assert d["shas"] == sorted([me.text_sha(A), me.text_sha(B)])
    q = d["questions"][0]
    assert set(q["answers"]) == set(d["shas"])                     # labels mapped back
    D.build_disputes(model="judge", chat=chat, log=lambda *_: None)
    assert len(prompts) == 2                                        # resumable


def test_scoring_right_wrong_silent_and_cant_tell():
    sa, sb = me.text_sha(A), me.text_sha(B)
    assert me.dispute_verdict("Soccer.", {"correct": ["soccer"]}) == "right"
    assert me.dispute_verdict("field hockey", {"correct": ["soccer"]}) == "wrong"
    assert me.dispute_verdict(None, {"correct": ["soccer"]}) == "silent"
    assert me.dispute_verdict("x", {"correct": [], "cant_tell": True}) is None
    assert me.dispute_verdict("x", None) is None

    _runs([1])
    me.save_disputes({"1": {"shas": sorted([sa, sb]), "questions": [
        {"key": "k1", "question": "Sport?", "answers": {sa: "soccer", sb: "field hockey"}},
        {"key": "k2", "question": "Car?", "answers": {sa: None, sb: "parked car"}}]}})
    me.save_dispute_label("k1", ["soccer"])
    me.save_dispute_label("k2", [])                                 # "None right": no car
    rows = {r["variant"]: r for r in D.build_report(use_clip=False)}
    assert (rows["va"]["disp_right"], rows["va"]["disp_wrong"], rows["va"]["disp_silent"]) == (1, 0, 1)
    assert (rows["vb"]["disp_right"], rows["vb"]["disp_wrong"]) == (0, 2)
    assert "disputed pts" in D.render(list(rows.values()))


def test_format_problem():
    assert D.format_problem("A dog.\n\n**Search Index Description:**\n- A dog")
    assert D.format_problem("Tags:\n1. dog")
    assert not D.format_problem("A dog - a small one - sleeps on 2 rugs.")


def test_disputes_api_is_blind_and_validates(client):
    sa, sb = me.text_sha(A), me.text_sha(B)
    me.save_sample("describe", [{"photo_id": 7, "stratum": "text"}])
    me.save_disputes({"7": {"shas": [sa, sb], "model": "vendor/judge", "questions": [
        {"key": "k1", "question": "Sport?", "answers": {sa: "soccer", sb: "field hockey"}},
        {"key": "k2", "question": "Car?", "answers": {sa: None, sb: "parked car"}}]}})
    resp = client.get("/api/eval/models/describe/disputes")
    assert "vendor" not in resp.text
    photo = resp.json()["photos"][0]
    assert photo["questions"][0]["options"] == ["field hockey", "soccer"]
    assert photo["questions"][1]["options"] == ["parked car"]     # silence is not an option
    assert client.put("/api/eval/models/describe/disputes/k1",
                      json={"correct": ["soccer"]}).status_code == 200
    assert client.put("/api/eval/models/describe/disputes/k1",
                      json={"correct": ["cricket"]}).status_code == 400
    assert client.put("/api/eval/models/describe/disputes/nope",
                      json={"correct": []}).status_code == 404
    assert client.put("/api/eval/models/describe/disputes/k2",
                      json={"cant_tell": True}).status_code == 200
    got = client.get("/api/eval/models/describe/disputes").json()
    assert got["photos"][0]["done"] and got["progress"]["points_done"] == 2

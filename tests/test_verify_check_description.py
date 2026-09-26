"""verify.check_description — the worker's three-stage check pulled out so the
model eval measures exactly what ships. These pin both the function and the
worker's behaviour on top of it (statuses, flags, the printed stage)."""

import json

import pytest

from photosearch import describe, verify
from photosearch import clip_embed


EMB = [1.0] + [0.0] * 511


@pytest.fixture
def stubs(monkeypatch):
    """CLIP scores come from a dict; the LLM answer from a list."""
    state = {"desc": [], "tags": [], "sim": {}, "llm": [], "llm_calls": 0}
    monkeypatch.setattr(verify, "clip_score_description", lambda emb, d: list(state["desc"]))
    monkeypatch.setattr(verify, "clip_score_tags", lambda emb, t: list(state["tags"]))

    def embed_text(text):
        noun = text.replace("a photo of ", "")
        return [state["sim"].get(noun, 0.0)] + [0.0] * 511
    monkeypatch.setattr(clip_embed, "embed_text", embed_text)

    def chat(*a, **kw):
        state["llm_calls"] += 1
        return state["llm"].pop(0)
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", chat)
    return state


def _scores(**sims):
    return [{"noun": n, "similarity": s} for n, s in sims.items()]


def test_clip_clean_never_asks_the_llm(stubs):
    stubs["desc"] = _scores(boat=0.30, water=0.28)
    r = verify.check_description("x.jpg", "A boat on water.", [], EMB)
    assert r["stage"] == "clip_clean" and stubs["llm_calls"] == 0


def test_llm_cleared(stubs):
    stubs["desc"] = _scores(boat=0.30, water=0.28, dog=0.05)
    stubs["llm"] = ["ALL CORRECT"]
    r = verify.check_description("x.jpg", "A boat, water and a dog.", [], EMB)
    assert r["stage"] == "llm_cleared"
    assert [f["noun"] for f in r["clip_flags"]] == ["dog"]


def test_clip_override_and_confirmed(stubs):
    stubs["desc"] = _scores(boat=0.30, water=0.28, dog=0.05)
    stubs["sim"] = {"dog": 0.29, "cat": 0.01}          # median of scores is 0.28
    stubs["llm"] = ["WRONG: dog"]
    assert verify.check_description("x.jpg", "d", [], EMB)["stage"] == "clip_override"
    stubs["llm"] = ["WRONG: cat\nWRONG: dog"]
    r = verify.check_description("x.jpg", "d", [], EMB)
    assert r["stage"] == "confirmed"
    assert [c["noun"] for c in r["confirmed"]] == ["cat"]
    assert [c["noun"] for c in r["llm_items"]] == ["cat", "dog"]


def test_llm_all_skips_both_clip_stages(stubs):
    stubs["desc"] = _scores(boat=0.30)                  # would be clip_clean
    stubs["sim"] = {"dog": 0.99}                        # would be overridden
    stubs["llm"] = ["WRONG: dog"]
    r = verify.check_description("x.jpg", "d", [], EMB, llm_all=True)
    assert r["stage"] == "confirmed" and [c["noun"] for c in r["confirmed"]] == ["dog"]


def test_no_embedding_goes_straight_to_the_llm(stubs):
    stubs["llm"] = ["WRONG: dog"]
    r = verify.check_description("x.jpg", "d", [], None)
    assert r["stage"] == "confirmed" and r["clip_flags"] == []


def test_worker_statuses_are_unchanged(stubs, monkeypatch, capsys):
    from photosearch import worker as W
    monkeypatch.setattr(describe, "check_available", lambda model: None)
    monkeypatch.setattr(describe, "describe_photo", lambda *a, **kw: "A clean redo.")
    monkeypatch.setattr(describe, "tag_visual_photo", lambda *a, **kw: ["sunny"])

    class Client:
        def get_photo_detail(self, pid):
            return {"description": "A boat, water and a dog.", "tags": "[]",
                    "clip_embedding": EMB}

    stubs["desc"] = _scores(boat=0.30, water=0.28, dog=0.05)
    stubs["sim"] = {"dog": 0.01}
    stubs["llm"] = ["ALL CORRECT", "WRONG: dog"]
    out = W._process_verify([({"id": 1, "filename": "a.jpg"}, "/x/a.jpg"),
                             ({"id": 2, "filename": "b.jpg"}, "/x/b.jpg")],
                            client=Client())
    printed = capsys.readouterr().out
    assert [r["status"] for r in out] == ["pass", "regenerated"]
    assert "pass (LLM cleared)" in printed and "REGENERATED" in printed
    assert json.loads(out[0]["hallucination_flags"])[0]["noun"] == "dog"   # clip flag kept
    assert json.loads(out[1]["hallucination_flags"]) == [{"noun": "dog", "llm_says": "NO"}]
    assert out[1]["description"] == "A clean redo."

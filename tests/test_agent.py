"""Tests for the in-app LLM agent loop (photosearch/agent.py — M24b).

The chat client (`agent._chat`) is monkeypatched with a scripted sequence so
the loop is deterministic — no real LLM needed. Tool calls run for real
against the conftest `db` fixture (Alex: 3 photos, Jamie: 2, Sam: 1).
"""

import pytest

from photosearch import agent


def _script(*replies):
    """Return a fake _chat that yields the given replies in order."""
    calls = {"n": 0}
    seq = list(replies)

    def fake_chat(messages, tools, temperature=0.0, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        return seq[min(i, len(seq) - 1)]
    fake_chat.calls = calls
    return fake_chat


def _tc(name, args, cid="c1"):
    return {"content": None, "tool_calls": [{"id": cid, "name": name, "arguments": args}]}


def _answer(text):
    return {"content": text, "tool_calls": []}


def _run(db, message, **kw):
    return list(agent.run_agent(db, message, **kw))


def _types(events):
    return [e["type"] for e in events]


# ---------------------------------------------------------------------------
# Full tool-calling loop
# ---------------------------------------------------------------------------

def test_loop_search_then_answer(db, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("search_photos", {"people": ["Alex"]}),
        _answer("Found 3 photos of Alex."),
    ))
    events = _run(db, "show me photos of Alex")
    assert [t for t in _types(events) if t != "step"] == ["tool_call", "tool_result", "photos", "answer"]
    assert any(e["type"] == "step" and e["n"] == 1 for e in events)  # round progress emitted
    photos = next(e for e in events if e["type"] == "photos")
    assert photos["total"] == 3
    assert len(photos["results"]) == 3
    assert events[-1]["text"] == "Found 3 photos of Alex."


def test_daily_highlights_results_render(db, monkeypatch):
    """daily_highlights results must reach the on-page `photos` event — the
    regression where the UI showed the prose answer but 0 photos because the
    agent only promoted a hardcoded set of tools' results."""
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("daily_highlights", {"per_day": 20, "window_minutes": 10}),
        _answer("Here is the day-by-day highlight reel."),
    ))
    events = _run(db, "best photos per day, chronological, no duplicates")
    assert [t for t in _types(events) if t != "step"] == \
        ["tool_call", "tool_result", "photos", "answer"]
    photos = next(e for e in events if e["type"] == "photos")
    assert len(photos["results"]) > 0
    assert photos["total"] == len(photos["results"])  # falls back to `returned`


def test_loop_grounds_then_searches(db, monkeypatch):
    # Model lists people first, then searches — two tool steps, then answers.
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("list_people", {}),
        _tc("search_photos", {"people": ["Alex", "Jamie"]}),
        _answer("Two photos have both Alex and Jamie."),
    ))
    events = _run(db, "photos with Alex and Jamie")
    assert [t for t in _types(events) if t != "step"] == [
        "tool_call", "tool_result",   # list_people
        "tool_call", "tool_result",   # search_photos
        "photos", "answer",
    ]
    assert [e["n"] for e in events if e["type"] == "step"] == [1, 2, 3]  # 2 tool rounds + answer round
    photos = next(e for e in events if e["type"] == "photos")
    assert photos["total"] == 2


def test_unknown_tool_is_reported_not_fatal(db, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("bogus_tool", {"x": 1}),
        _answer("done"),
    ))
    events = _run(db, "anything")
    tr = next(e for e in events if e["type"] == "tool_result")
    assert "unknown tool" in tr["summary"]
    assert events[-1]["type"] == "answer"


def test_empty_turn_after_grounding_is_nudged(db, monkeypatch):
    # Reproduce qwen3: ground via a tool, then return an EMPTY turn (no content,
    # no tool calls). The loop should nudge it to search rather than give up.
    empty = {"content": "", "tool_calls": []}
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("get_library_overview", {}),     # ground
        empty,                                # stall — should trigger a nudge
        _tc("search_photos", {"people": ["Alex"]}),  # nudged → searches
        _answer("Found Alex's photos."),
    ))
    events = _run(db, "show me photos of Alex")
    types = _types(events)
    assert any(e.get("tool") == "_nudge" for e in events if e["type"] == "tool_result")
    assert "photos" in types
    photos = next(e for e in events if e["type"] == "photos")
    assert photos["total"] == 3


def test_persistent_empty_turns_give_up_gracefully(db, monkeypatch):
    # If the model keeps returning empty turns past _MAX_NUDGES, end cleanly.
    monkeypatch.setattr(agent, "_chat", _script({"content": "", "tool_calls": []}))
    events = _run(db, "x")
    assert events[-1]["type"] == "answer"  # no crash, terminal answer


def test_step_cap_emits_partial(db, monkeypatch):
    # Model loops forever issuing tool calls; cap stops it and emits photos.
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("search_photos", {"people": ["Alex"]}),
    ))  # always returns a tool call → never answers
    events = _run(db, "loop", max_steps=3)
    assert events[-1]["type"] == "answer"
    assert "Stopped after 3 steps" in events[-1]["text"]
    # The last good search result set is still surfaced.
    assert any(e["type"] == "photos" for e in events)


def test_should_abort_stops_loop(db, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _script(_tc("search_photos", {"people": ["Alex"]})))
    events = _run(db, "x", should_abort=lambda: True)
    assert events == [{"type": "error", "message": "cancelled"}]


def test_empty_message():
    events = list(agent.run_agent(None, "   "))
    assert events == [{"type": "error", "message": "empty message"}]


def test_system_prompt_injects_library_facts(db, monkeypatch):
    agent._CONTEXT_CACHE.clear()
    monkeypatch.setenv("PHOTOSEARCH_AGENT_HINTS", "The kids are Alex and Jamie.")
    p = agent._system_prompt(db)
    assert "LIBRARY FACTS" in p
    assert "Alex" in p and "Jamie" in p          # registered people injected
    assert "landscape" in p or "people" in p      # categories injected
    assert "USER NOTES" in p and "The kids are Alex and Jamie." in p


def test_write_gate_on_by_default(db, monkeypatch):
    # Behind Tailscale → writes are ON unless explicitly disabled.
    monkeypatch.delenv("PHOTOSEARCH_ALLOW_WRITES", raising=False)
    assert agent._writes_enabled() is True
    # The write guidance is only appended when writes are enabled.
    assert "WRITES" not in agent._system_prompt(db, allow_writes=False)
    assert "WRITES" in agent._system_prompt(db, allow_writes=True)


def test_write_gate_can_be_disabled(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_ALLOW_WRITES", "0")
    assert agent._writes_enabled() is False
    monkeypatch.setenv("PHOTOSEARCH_ALLOW_WRITES", "off")
    assert agent._writes_enabled() is False
    monkeypatch.setenv("PHOTOSEARCH_ALLOW_WRITES", "1")
    assert agent._writes_enabled() is True


# ---------------------------------------------------------------------------
# Single-shot fallback
# ---------------------------------------------------------------------------

def test_single_shot_env_flag(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_AGENT_SINGLE_SHOT", "1")
    monkeypatch.setattr(agent, "_chat", _script(_answer('{"people": ["Alex"]}')))
    events = _run(db, "alex photos")
    assert _types(events) == ["tool_call", "tool_result", "photos", "answer"]
    assert next(e for e in events if e["type"] == "photos")["total"] == 3


def test_single_shot_strips_code_fence(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_AGENT_SINGLE_SHOT", "1")
    monkeypatch.setattr(agent, "_chat", _script(
        _answer('```json\n{"category": "landscape"}\n```')))
    events = _run(db, "landscapes")
    assert next(e for e in events if e["type"] == "photos")["total"] == 1


def test_single_shot_unparseable_is_graceful(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_AGENT_SINGLE_SHOT", "1")
    monkeypatch.setattr(agent, "_chat", _script(_answer("I am not JSON at all")))
    events = _run(db, "???")
    assert _types(events) == ["answer"]
    assert "couldn't" in events[0]["text"].lower()


def test_chat_failure_falls_back_to_single_shot(db, monkeypatch):
    # First _chat raises (e.g. endpoint can't tool-call) → single-shot kicks in.
    state = {"n": 0}

    def flaky_chat(messages, tools, temperature=0.0, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("400 tools not supported")
        return _answer('{"people": ["Alex"]}')
    monkeypatch.setattr(agent, "_chat", flaky_chat)
    events = _run(db, "alex")
    assert any(e["type"] == "photos" and e["total"] == 3 for e in events)
    assert events[-1]["type"] == "answer"


def test_deadline_returns_gracefully_without_hanging(db, monkeypatch):
    # A zero time budget trips the wall-clock guard before any LLM call, so an
    # over-complex query returns a clear message instead of churning (the 158s
    # hang). _chat must never be invoked.
    monkeypatch.setenv("PHOTOSEARCH_AGENT_DEADLINE_S", "0")
    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("_chat should not be called past the deadline")
    monkeypatch.setattr(agent, "_chat", boom)
    events = _run(db, "something far too complex for the budget")
    assert called["n"] == 0
    assert events[-1]["type"] == "answer"
    assert "time budget" in events[-1]["text"].lower()


# ---------------------------------------------------------------------------
# Locked filters — the structured Search filter bar pinned as HARD Ask inputs
# ---------------------------------------------------------------------------

def test_normalize_locked_drops_empty():
    out = agent._normalize_locked({"camera": "X", "location": "", "people": [],
                                   "date_from": None, "bogus": "y", "min_quality": 7})
    assert out == {"camera": "X", "min_quality": 7}


def test_merge_locked_overrides_scalars_and_unions_people():
    merged = agent._merge_locked(
        "search_photos",
        {"people": ["Alex"], "camera": "SonyA"},
        {"camera": "KODAK", "people": ["Jamie"], "date_from": "2026-06-28"})
    # people unions; camera hard-overrides; date_from injected.
    assert set(merged["people"]) == {"Alex", "Jamie"}
    assert merged["camera"] == "KODAK"
    assert merged["date_from"] == "2026-06-28"


def test_merge_locked_only_filter_tools():
    # rerank_photos takes ids, not filters — locked filters must not leak in.
    args = {"photo_ids": [1, 2], "criteria": "x"}
    assert agent._merge_locked("rerank_photos", args, {"camera": "KODAK"}) == args


def test_locked_prompt_mentions_pinned_values():
    p = agent._locked_prompt({"camera": "KODAK PIXPRO WPZ2", "date_from": "2026-06-28"})
    assert "ACTIVE FILTERS" in p and "KODAK PIXPRO WPZ2" in p


def test_locked_camera_is_enforced_even_when_model_omits_it(db, monkeypatch):
    # The model plans a plain people search; the pinned (non-matching) camera
    # must still be applied server-side, yielding zero results — the KODAK bug.
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("search_photos", {"people": ["Alex"]}),
        _answer("No photos from that camera."),
    ))
    events = _run(db, "photos of Alex", locked_filters={"camera": "nope-cam"})
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["arguments"]["camera"] == "nope-cam"   # enforced into the call
    photos = next(e for e in events if e["type"] == "photos")
    assert photos["total"] == 0


def test_locked_camera_match_keeps_results(db, monkeypatch):
    monkeypatch.setattr(agent, "_chat", _script(
        _tc("search_photos", {"people": ["Alex"]}),
        _answer("Three photos of Alex on that camera."),
    ))
    events = _run(db, "photos of Alex", locked_filters={"camera": "ILCE-7M4"})
    photos = next(e for e in events if e["type"] == "photos")
    assert photos["total"] == 3

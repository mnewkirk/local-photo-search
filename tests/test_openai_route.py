"""Tests for the OpenAI-compatible (LM Studio) LLM route in describe.py."""
import time
import urllib.request

import pytest

from photosearch.describe import _openai_chat_with_retry


def test_wall_clock_cap_aborts_trickling_server(monkeypatch):
    """A server that never returns (or trickles tokens) must be abandoned at the
    wall-clock timeout, not run unbounded.

    Regression for the LM Studio stall where one verify call ran 966s: urllib's
    timeout is a socket-*idle* timeout, so a slow-trickling slot never trips it.
    The route must enforce a true wall-clock cap per attempt.
    """
    def hung_urlopen(req, timeout=None):
        time.sleep(5)  # far longer than our 0.5s wall-clock cap
        raise AssertionError("urlopen should have been abandoned by the wall-clock cap")

    monkeypatch.setattr(urllib.request, "urlopen", hung_urlopen)

    t0 = time.time()
    with pytest.raises(Exception) as exc:
        _openai_chat_with_retry(
            "http://stub", "m",
            [{"role": "user", "content": "hi"}],
            retries=1, timeout=0.5,
        )
    elapsed = time.time() - t0

    assert elapsed < 3, f"call ran {elapsed:.1f}s — wall-clock cap did not fire"
    assert "exceed" in str(exc.value).lower() or "timeout" in str(exc.value).lower()


def _capture_payload(monkeypatch):
    import io, json, urllib.request
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        seen.update(json.loads(req.data))
        return _Resp(json.dumps({"choices": [{"message": {"content": "sunny"}}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_reasoning_effort_is_sent_only_when_configured(monkeypatch):
    # gemma-4 / qwen3.x think by default and spend the whole budget on hidden
    # reasoning, returning ''. The switch must reach the wire when set, and a
    # backend that has never heard of it must never see the field.
    from photosearch import describe
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm/v1")
    monkeypatch.delenv("PHOTOSEARCH_LLM_REASONING_EFFORT", raising=False)
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(base_url="http://lm/v1", model="m", messages=[{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in seen
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(base_url="http://lm/v1", model="m", messages=[{"role": "user", "content": "x"}])
    assert seen["reasoning_effort"] == "none"


def test_reasoning_effort_skipped_for_known_non_reasoning_models(monkeypatch):
    """qwen2.5-vl-7b-instruct (the aesthetics/subject role model) has no
    reasoning mode at all. LM Studio logs a WARN on every call that carries
    `reasoning_effort` to it ("No valid custom reasoning fields found ...").
    Sending it anyway is a no-op plus log spam, so the field must be skipped
    for a model on the known non-reasoning list even when the env var is set.
    """
    from photosearch import describe
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(
        base_url="http://lm/v1", model="qwen2.5-vl-7b-instruct",
        messages=[{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in seen


def test_reasoning_effort_still_sent_for_a_reasoning_model(monkeypatch):
    """A reasoning model (gemma-4*, qwen3.x, minicpm-v-4_5 — not on the
    skip list) must still get the field; it is the whole point of the env
    var and must not be silently dropped by an over-broad pattern."""
    from photosearch import describe
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(
        base_url="http://lm/v1", model="google/gemma-4-12b-qat",
        messages=[{"role": "user", "content": "x"}])
    assert seen["reasoning_effort"] == "none"


def test_no_reasoning_models_list_is_extendable_via_env(monkeypatch):
    """PHOTOSEARCH_LLM_NO_REASONING_MODELS lets an operator add a model to
    the skip list (substring match, case-insensitive) without editing code."""
    from photosearch import describe
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    monkeypatch.setenv("PHOTOSEARCH_LLM_NO_REASONING_MODELS", "my-custom-model")
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(
        base_url="http://lm/v1", model="My-Custom-Model-7B",
        messages=[{"role": "user", "content": "x"}])
    assert "reasoning_effort" not in seen


@pytest.mark.parametrize("role,expects_reasoning_effort", [
    ("describe", True),      # qwen/qwen3.5-9b — thinks by default
    ("verify", True),        # google/gemma-4-12b-qat — thinks by default
    ("visual", True),        # minicpm-v-4_5 — thinks by default
    ("aesthetics", False),   # qwen2.5-vl-7b-instruct — no reasoning mode
    ("text", True),          # google/gemma-4-12b-qat — thinks by default
    ("subject", False),      # qwen2.5-vl-7b-instruct — no reasoning mode
    ("video", True),         # qwen/qwen3.5-9b — thinks by default
])
def test_fleet_role_models_get_expected_reasoning_effort_behaviour(
        monkeypatch, role, expects_reasoning_effort):
    """Pin each of rerun.FLEET_ROLE_MODELS to whether it should receive
    `reasoning_effort` — so a model swap in that table can't silently drop
    the field from a thinking model (or silently add noisy log spam back for
    a non-reasoning one) without this test going red."""
    from photosearch import describe
    from photosearch.rerun import FLEET_ROLE_MODELS
    model = FLEET_ROLE_MODELS[role]
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    seen = _capture_payload(monkeypatch)
    describe._openai_chat_with_retry(
        base_url="http://lm/v1", model=model,
        messages=[{"role": "user", "content": "x"}])
    if expects_reasoning_effort:
        assert seen.get("reasoning_effort") == "none", (
            f"role={role!r} model={model!r} should still get reasoning_effort")
    else:
        assert "reasoning_effort" not in seen, (
            f"role={role!r} model={model!r} should NOT get reasoning_effort")

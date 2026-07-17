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

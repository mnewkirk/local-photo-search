"""describe.check_available must ride out an LM Studio JIT model load.

On 2026-09-26 a worker died mid-fleet: category-content started right after
its model had been ejected, LM Studio stopped answering while it re-loaded the
model, and the once-per-batch reachability check (outside every per-photo
retry) raised on its first 10 s timeout, killing the process.
"""

import pytest

from photosearch import describe


class _Resp:
    def read(self):
        return b"{}"


def _patch(monkeypatch, failures):
    calls = {"n": 0, "slept": []}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        if calls["n"] <= failures:
            raise TimeoutError("timed out")
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(describe.time, "sleep", lambda s: calls["slept"].append(s))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lmstudio:1234/v1")
    return calls


def test_transient_timeouts_are_retried_until_it_answers(monkeypatch):
    calls = _patch(monkeypatch, failures=3)
    describe.check_available("any")          # must not raise
    assert calls["n"] == 4
    assert calls["slept"] == list(describe._REACHABILITY_BACKOFF_S[:3])


def test_gives_up_with_the_old_error_after_the_backoff(monkeypatch):
    calls = _patch(monkeypatch, failures=99)
    with pytest.raises(RuntimeError, match="Cannot reach OpenAI-compatible LLM"):
        describe.check_available("any")
    assert calls["n"] == 1 + len(describe._REACHABILITY_BACKOFF_S)


def test_healthy_backend_does_not_sleep(monkeypatch):
    calls = _patch(monkeypatch, failures=0)
    describe.check_available("any")
    assert calls["n"] == 1 and calls["slept"] == []

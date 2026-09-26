"""photosearch/model_eval.py — the storage + instrumentation contract every
model-eval harness shares. No network and no model: the chat call and the
HTTP layer are stubbed."""

import json
import os
import urllib.request

import pytest

from photosearch import describe
from photosearch import model_eval as me


@pytest.fixture(autouse=True)
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "model-evals"
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(d))
    for var in ("PHOTOSEARCH_TEXT_LLM_URL", "PHOTOSEARCH_LLM_VISUAL_MODEL",
                "PHOTOSEARCH_LLM_AESTHETICS_MODEL", "PHOTOSEARCH_LLM_DESCRIBE_MODEL",
                "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.setenv(var, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(var)
    return d


# --------------------------------------------------------------------------
# Atomic writes + run cache
# --------------------------------------------------------------------------

def test_atomic_write_leaves_old_file_on_failure(eval_dir, monkeypatch):
    path = eval_dir / "x.json"
    me.write_json_atomic(path, {"a": 1})

    def boom(*a, **kw):
        raise OSError("disk full")
    monkeypatch.setattr(me.json, "dump", boom)
    with pytest.raises(OSError):
        me.write_json_atomic(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
    assert [p.name for p in eval_dir.iterdir()] == ["x.json"]   # no temp left


def test_open_run_resumes_and_refuses_a_different_identity():
    run = me.open_run("describe", "v1", effective_model="m1", prompt_sha=None)
    run["items"]["5"] = {"text": "hi"}
    me.save_run("describe", "v1", run)

    again = me.open_run("describe", "v1", effective_model="m1", prompt_sha=None)
    assert again["items"] == {"5": {"text": "hi"}}
    with pytest.raises(SystemExit, match="effective_model"):
        me.open_run("describe", "v1", effective_model="m2", prompt_sha=None)
    fresh = me.open_run("describe", "v1", force=True, effective_model="m2", prompt_sha=None)
    assert fresh["items"] == {}
    assert me.list_variants("describe") == ["v1"]


@pytest.mark.parametrize("bad", ["", "stored", "../x", "a b"])
def test_bad_variant_names_are_refused(bad):
    with pytest.raises(SystemExit):
        me.open_run("describe", bad)


# --------------------------------------------------------------------------
# Pinning
# --------------------------------------------------------------------------

def test_pin_role_model_sets_the_role_var_and_beats_the_visual_fallback(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen")
    # Unpinned, aesthetics resolves to the VISUAL model — the trap.
    assert describe.effective_model("anything", "aesthetics") == "qwen"
    me.pin_role_model("aesthetics", "gemma")
    assert os.environ["PHOTOSEARCH_LLM_AESTHETICS_MODEL"] == "gemma"
    assert describe.effective_model("anything", "aesthetics") == "gemma"


def test_pin_role_model_refuses_without_the_lmstudio_route():
    with pytest.raises(SystemExit, match="PHOTOSEARCH_TEXT_LLM_URL"):
        me.pin_role_model("describe", "gemma")
    me.pin_role_model("describe", None)   # no --model: nothing to do, no error


# --------------------------------------------------------------------------
# Originals cache
# --------------------------------------------------------------------------

def test_original_is_fetched_once_and_failures_are_not_cached():
    calls = []

    def fetch(server, pid, kind):
        calls.append(pid)
        if pid == 2:
            raise RuntimeError("502 Bad Gateway")
        return b"jpeg-%d" % pid

    p = me.original_path(1, fetch=fetch)
    assert p.read_bytes() == b"jpeg-1"
    me.original_path(1, fetch=fetch)
    assert calls == [1]

    with pytest.raises(RuntimeError):
        me.original_path(2, fetch=fetch)
    assert me.cached_original(2) is None

    have, got, bad = me.prefetch([1, 2, 3], fetch=fetch, log=lambda *_: None)
    assert (have, got, bad) == (1, 1, 1)


# --------------------------------------------------------------------------
# Latency label
# --------------------------------------------------------------------------

def test_latency_label():
    assert me.latency_label("m", ["m"], ["m"]) == "solo"
    assert me.latency_label("m", ["m", "q"], ["m"]) == "shared(q)"
    assert me.latency_label("m", None, ["m"]) == "unknown"
    assert me.latency_label("m", None, None, solo_flag=True) == "solo (asserted)"


def test_lmstudio_loaded_reads_state(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://h:1234/v1")
    seen = {}

    class R:
        def __init__(self, body): self.body = body
        def read(self): return self.body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake(url, timeout=None):
        seen["url"] = url
        return R(json.dumps({"data": [{"id": "a", "state": "loaded"},
                                      {"id": "b", "state": "not-loaded"}]}).encode())
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    assert me.lmstudio_loaded() == ["a"]
    assert seen["url"] == "http://h:1234/api/v0/models"

    def dead(url, timeout=None):
        raise OSError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", dead)
    assert me.lmstudio_loaded() is None


# --------------------------------------------------------------------------
# Recorder + the per-attempt hook
# --------------------------------------------------------------------------

def test_recorder_tells_transport_from_a_useless_answer(monkeypatch):
    monkeypatch.setattr(describe, "_ollama_chat_with_retry",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("refused")))
    rec = me.Recorder()
    with rec.active():
        try:
            describe._ollama_chat_with_retry(model="m", messages=[], role="verify")
        except ConnectionError:
            pass
    with pytest.raises(me.TransportError, match="refused"):
        rec.check(result_is_empty=True)

    monkeypatch.setattr(describe, "_ollama_chat_with_retry", lambda *a, **kw: "garbage")
    rec = me.Recorder()
    with rec.active():
        describe._ollama_chat_with_retry(model="m", messages=[], role="verify")
    rec.check(result_is_empty=True)          # answered: a real (bad) result
    assert rec.calls[0]["raw"] == "garbage" and rec.calls[0]["role"] == "verify"

    with pytest.raises(me.TransportError, match="no model call"):
        me.Recorder().check(result_is_empty=True)


class _Resp:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()


def test_attempt_hook_sees_a_recovered_timeout_and_truncation(monkeypatch):
    monkeypatch.setattr(describe, "_RETRY_DELAY", 0)
    n = {"i": 0}

    def fake_urlopen(req, timeout=None):
        n["i"] += 1
        if n["i"] == 1:
            raise TimeoutError("timed out")
        return _Resp({"choices": [{"message": {"content": "cut off mid"},
                                   "finish_reason": "length"}],
                      "usage": {"completion_tokens": 768}})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    rec = me.Recorder()
    with rec.active():
        out = describe._ollama_chat_with_retry(
            model="m", messages=[{"role": "user", "content": "hi"}],
            role="describe", timeout=5)
    assert out == "cut off mid"
    assert describe._ATTEMPT_HOOK is None          # only live inside the recorder
    assert [a["outcome"] for a in rec.attempts()] == ["timeout", "ok"]
    assert rec.count("timeout") == 1
    assert rec.truncated() is True


def test_a_broken_hook_cannot_change_the_result(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _Resp(
        {"choices": [{"message": {"content": "fine"}, "finish_reason": "stop"}],
         "usage": {}}))

    def bad_hook(**kw):
        raise ValueError("observer bug")
    monkeypatch.setattr(describe, "_ATTEMPT_HOOK", bad_hook)
    assert describe._openai_chat_with_retry("http://x/v1", "m", [], timeout=5) == "fine"


def test_failure_counter_warns_on_a_streak():
    lines = []
    fc = me.FailureCounter(log=lines.append)
    for i in range(me.CONSECUTIVE_FAIL_WARN):
        fc.fail(i, "400 Bad Request")
    fc.summary()
    assert any("in a row" in l for l in lines)
    assert any("NOT cached" in l for l in lines)

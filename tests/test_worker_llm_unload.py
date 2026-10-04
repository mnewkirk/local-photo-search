"""A retiring LLM pass unloads its model from LM Studio.

LM Studio keeps every JIT-loaded model resident, so a fleet walking
describe -> category-visual -> category-content -> keywords -> verify ended
with all of them in VRAM at once (2026-10-04: four 16k-context models on one
24 GB card). The owner runs one model at a time, passes in sequence.

Three rules, each pinned here:
- the pass up next keeps a model it shares (gemma serves category-content,
  keywords AND verify — unloading it between them is pure reload cost);
- a model another worker still holds a live claim on is NOT unloaded (that
  would fail its in-flight requests) — the last worker out unloads;
- anything uncertain (status call fails, Ollama route) unloads nothing.
"""

import io
import json

import pytest

from photosearch import describe as D
from photosearch import worker as W

ROLE_ENV = {
    "PHOTOSEARCH_LLM_DESCRIBE_MODEL": "qwen/qwen3.5-9b",
    "PHOTOSEARCH_LLM_VERIFY_MODEL": "google/gemma-4-12b-qat",
    "PHOTOSEARCH_LLM_VISUAL_MODEL": "minicpm-v-4_5",
    "PHOTOSEARCH_LLM_AESTHETICS_MODEL": "qwen2.5-vl-7b-instruct",
    "PHOTOSEARCH_LLM_TEXT_MODEL": "google/gemma-4-12b-qat",
}

PASS_MODELS = {
    "describe":         [("x", "describe")],
    "verify":           [("x", "verify"), ("x", "describe")],
    "category-content": [("x", "text")],
    "keywords":         [("x", "text")],
    "category-visual":  [("x", "visual")],
    "aesthetics":       [("x", "aesthetics")],
}


@pytest.fixture
def lmstudio(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm:1234/v1")
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_MODEL", raising=False)
    for k, v in ROLE_ENV.items():
        monkeypatch.setenv(k, v)
    unloaded: list = []
    monkeypatch.setattr(D, "unload_openai_models",
                        lambda models: (unloaded.extend(sorted(models)), sorted(models))[1])
    return unloaded


class Client:
    worker_id = "me"

    def __init__(self, claims=(), fail=False):
        self.claims = list(claims)
        self.fail = fail

    def get_status(self, **kw):
        if self.fail:
            raise ConnectionError("nas down")
        return {"active_claims": self.claims, "queue_depth": {}}


def release(client, retired, keep):
    return W._release_llm_models(client, retired, keep, PASS_MODELS,
                                 {"collection_id": None, "directory": None,
                                  "filters": None})


def test_retired_pass_unloads_its_model(lmstudio):
    release(Client(), "category-visual", ["category-content"])
    assert lmstudio == ["minicpm-v-4_5"]


def test_model_shared_with_the_next_pass_stays_loaded(lmstudio):
    release(Client(), "category-content", ["keywords"])
    assert lmstudio == []
    release(Client(), "keywords", ["verify"])
    assert lmstudio == []


def test_last_pass_unloads_everything_it_used(lmstudio):
    release(Client(), "verify", [])
    assert lmstudio == ["google/gemma-4-12b-qat", "qwen/qwen3.5-9b"]


def test_another_workers_live_claim_keeps_the_model(lmstudio):
    other = [{"worker_id": "sibling", "pass_type": "category-visual"}]
    release(Client(other), "category-visual", ["category-content"])
    assert lmstudio == []


def test_our_own_stale_claim_does_not_block(lmstudio):
    mine = [{"worker_id": "me", "pass_type": "category-visual"}]
    release(Client(mine), "category-visual", ["category-content"])
    assert lmstudio == ["minicpm-v-4_5"]


def test_status_failure_unloads_nothing(lmstudio):
    release(Client(fail=True), "category-visual", [])
    assert lmstudio == []


def test_ollama_route_is_untouched(lmstudio, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL")
    release(Client(), "category-visual", [])
    assert lmstudio == []


def test_torch_passes_have_no_llm(lmstudio):
    release(Client(), "clip", [])
    assert lmstudio == []


# --- unload_openai_models: the LM Studio REST calls ---------------------------

class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): pass


def test_unload_posts_only_loaded_wanted_instances(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm:1234/v1")
    listing = {"models": [
        {"key": "minicpm-v-4_5", "loaded_instances": [{"id": "minicpm-v-4_5"}]},
        {"key": "google/gemma-4-12b-qat", "loaded_instances": [{"id": "google/gemma-4-12b-qat"}]},
        {"key": "qwen/qwen3.5-9b", "loaded_instances": []},
    ]}
    posts = []

    def urlopen(req, timeout=None):
        if isinstance(req, str):
            assert req == "http://lm:1234/api/v1/models"
            return _Resp(json.dumps(listing).encode())
        posts.append((req.full_url, json.loads(req.data)))
        return _Resp(b"{}")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    done = D.unload_openai_models({"minicpm-v-4_5", "qwen/qwen3.5-9b"})
    assert done == ["minicpm-v-4_5"]
    assert posts == [("http://lm:1234/api/v1/models/unload",
                      {"instance_id": "minicpm-v-4_5"})]


def test_unload_never_raises_when_backend_lacks_the_api(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://llama-server:8080/v1")

    def urlopen(req, timeout=None):
        raise OSError("404")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert D.unload_openai_models({"anything"}) == []


# --- the loop: retirement triggers it, in sequential order --------------------

def test_sequential_fleet_unloads_each_model_as_its_passes_finish(monkeypatch, tmp_path, lmstudio):
    from tests.test_worker_pass_lifecycle import FakeClient
    client = FakeClient({"category-visual": 1, "category-content": 1, "keywords": 1})
    monkeypatch.setattr(W, "WorkerClient", lambda *a, **k: client)
    monkeypatch.setattr(W, "_download_batch", lambda c, photos, d: {p["id"]: "x" for p in photos})
    for name in ("_process_category_visual", "_process_category_content", "_process_keywords"):
        monkeypatch.setattr(W, name, lambda d, **k: [])
    monkeypatch.setattr(W, "_provenance_kwargs", lambda *a, **k: {})
    monkeypatch.setattr(W, "_unload_pass_models", lambda p: None)
    monkeypatch.setattr(W, "_flush_caches", lambda: None)
    monkeypatch.setattr(W.time, "sleep", lambda s: None)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda **k: str(tmp_path))
    monkeypatch.setattr(W.shutil, "rmtree", lambda *a, **k: None)

    class _HB:
        def __init__(self, *a, **k): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr(W, "_ClaimHeartbeat", _HB)

    W.run_worker(server="http://fake",
                 passes=["category-visual", "category-content", "keywords"])
    # minicpm goes when visual drains; gemma survives content -> keywords and
    # goes only when keywords (the last pass) drains.
    assert lmstudio == ["minicpm-v-4_5", "google/gemma-4-12b-qat"]

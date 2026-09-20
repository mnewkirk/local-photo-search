"""Provenance: `generations.model_used` must name the model that actually ran.

It did not. On the live library 159,647 of 159,650 `category-visual` rows say
`llava` — the NOMINAL CLI default — because the worker logged the name it was
configured with rather than the id `_resolve_openai_model` resolved for the LM
Studio route. `model_version` was NULL for every Jun-Sep row. Nobody could tell
which model tagged anything.

The fix is one shared helper pair, used by the worker fleet and the M28 re-run
path alike.
"""

import pytest

from photosearch import describe as D


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_ollama_route_reports_the_name_it_was_given(monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    assert D.effective_model("llava", "visual") == "llava"


def test_openai_route_reports_the_role_resolved_model(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://example.invalid/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen/qwen2.5-vl-7b")
    assert D.effective_model("llava", "visual") == "qwen/qwen2.5-vl-7b"


def test_openai_route_without_a_role_env_falls_back_like_the_call_does(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://example.invalid/v1")
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_DESCRIBE_MODEL", raising=False)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_MODEL", "legacy-single")
    # Whatever the chat call would send is what gets logged — same function.
    assert D.effective_model("llama3.2-vision", "describe") == \
        D._resolve_openai_model("llama3.2-vision", "describe") == "legacy-single"


def test_model_version_is_a_static_marker_on_the_openai_route(monkeypatch):
    """There is no Ollama to query, and asking anyway blocks ~80 s retrying
    localhost:11434 once per pass."""
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://example.invalid/v1")
    assert D.effective_model_version("anything") == "lmstudio"


def test_model_version_uses_the_ollama_digest_otherwise(monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    from photosearch import worker as W
    monkeypatch.setattr(W, "_model_version", lambda m: "deadbeef1234")
    assert D.effective_model_version("llava") == "deadbeef1234"


def test_every_llm_pass_has_a_role():
    from photosearch.rerun import ALL_PASSES, _PASS_LLM

    assert set(D.PASS_ROLES) == set(_PASS_LLM)
    assert set(D.PASS_ROLES) <= set(ALL_PASSES)
    for pass_type, role in D.PASS_ROLES.items():
        assert _PASS_LLM[pass_type][0] == role, pass_type


def test_rerun_shares_the_helper(monkeypatch):
    """rerun.py already resolved this correctly; it must not keep a second
    copy of the logic that can drift."""
    import inspect

    from photosearch import rerun as R
    src = inspect.getsource(R._model_version)
    assert "effective_model_version" in src


# ---------------------------------------------------------------------------
# The worker reports the effective model for every LLM pass
# ---------------------------------------------------------------------------

_LLM_KWARG_PASSES = {
    "describe": ("describe_results", "describe"),
    "verify": ("verify_results", "describe"),
    "category-content": ("category_content_results", "text"),
    "category-visual": ("category_visual_results", "visual"),
    "keywords": ("keywords_results", "text"),
    "aesthetics": ("aesthetics_results", "aesthetics"),
}


@pytest.mark.parametrize("pass_type", sorted(_LLM_KWARG_PASSES))
def test_worker_builds_kwargs_with_the_effective_model(monkeypatch, pass_type):
    from photosearch import worker as W

    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://example.invalid/v1")
    role = _LLM_KWARG_PASSES[pass_type][1]
    monkeypatch.setenv(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL", f"real-{role}")

    results = [{"photo_id": 1}]
    kwargs = W._provenance_kwargs(pass_type, "nominal-cli-default", results,
                                  role=role)

    reported = kwargs.get("model") or results[0].get("model")
    assert reported == f"real-{role}", \
        f"{pass_type} logged the nominal default, not the model that ran"
    version = kwargs.get("model_version") or results[0].get("model_version")
    assert version == "lmstudio"


def test_worker_per_result_passes_stamp_every_row(monkeypatch):
    from photosearch import worker as W

    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.setattr(W, "_model_version", lambda m: "abc123")
    results = [{"photo_id": 1}, {"photo_id": 2}]
    W._provenance_kwargs("category-visual", "llava", results)
    assert all(r["model"] == "llava" and r["model_version"] == "abc123"
               for r in results)


def test_worker_batch_level_passes_use_top_level_kwargs(monkeypatch):
    """describe / verify carry provenance on the request, not per row."""
    from photosearch import worker as W

    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.setattr(W, "_model_version", lambda m: "abc123")
    kwargs = W._provenance_kwargs("describe", "llama3.2-vision", [{"photo_id": 1}])
    assert kwargs == {"model": "llama3.2-vision", "model_version": "abc123"}


def test_non_llm_passes_get_no_provenance(monkeypatch):
    from photosearch import worker as W

    for pass_type in ("clip", "faces", "quality"):
        assert W._provenance_kwargs(pass_type, "whatever", [{"photo_id": 1}]) == {}


def test_every_llm_branch_in_the_worker_loop_uses_the_helper():
    """The loop must not hand-roll provenance for any pass — that is exactly
    how category-visual came to log a model that never ran."""
    import inspect

    from photosearch import worker as W
    src = inspect.getsource(W.run_worker)
    assert "_model_version(" not in src, \
        "a branch still stamps the raw digest instead of the shared helper"
    assert src.count("_provenance_kwargs(") == len(D.PASS_ROLES)


def test_verify_logs_the_regen_model_not_the_verifier():
    """The verify pass's artifact is the REGENERATED description, written by
    the describe-role model."""
    import inspect

    from photosearch import worker as W
    src = inspect.getsource(W.run_worker)
    assert 'role="describe"' in src

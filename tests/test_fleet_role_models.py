"""The UI worker fleet's per-pass models (rerun.FLEET_ROLE_MODELS), chosen by
the model evals. Every role is set explicitly: the fleet used to inherit the
server's PHOTOSEARCH_LLM_VISUAL_MODEL (meant for rerank) and fall back to it
for describe/verify/aesthetics, so every vision pass silently ran on one model."""

from photosearch import admin_api, rerun


def _clean(monkeypatch):
    for role in rerun.FLEET_ROLE_MODELS:
        monkeypatch.delenv(f"PHOTOSEARCH_LLM_{role.upper()}_MODEL", raising=False)
        monkeypatch.delenv(f"PHOTOSEARCH_FLEET_{role.upper()}_MODEL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_MODEL", raising=False)


def test_fleet_gets_the_eval_chosen_model_per_role(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    # The server's own VISUAL (rerank) must not leak into the fleet.
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen2.5-vl-7b-instruct")
    env = admin_api._fleet_env()
    assert env["PHOTOSEARCH_LLM_VERIFY_MODEL"] == "google/gemma-4-12b-qat"
    assert env["PHOTOSEARCH_LLM_VISUAL_MODEL"] == "minicpm-v-4_5"
    assert env["PHOTOSEARCH_LLM_DESCRIBE_MODEL"] == "qwen/qwen3.5-9b"
    assert env["PHOTOSEARCH_LLM_AESTHETICS_MODEL"] == "qwen2.5-vl-7b-instruct"
    assert env["PHOTOSEARCH_LLM_TEXT_MODEL"] == "google/gemma-4-12b-qat"


def test_one_role_can_be_overridden(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    monkeypatch.setenv("PHOTOSEARCH_FLEET_VERIFY_MODEL", "google/gemma-4-e2b")
    env = admin_api._fleet_env()
    assert env["PHOTOSEARCH_LLM_VERIFY_MODEL"] == "google/gemma-4-e2b"
    assert env["PHOTOSEARCH_LLM_VISUAL_MODEL"] == "minicpm-v-4_5"


def test_ollama_route_is_untouched(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    env = admin_api._fleet_env()
    assert "PHOTOSEARCH_LLM_VERIFY_MODEL" not in env


def test_verify_and_describe_differ():
    """The verify check must be independent of the model that wrote the text."""
    assert rerun.FLEET_ROLE_MODELS["verify"] != rerun.FLEET_ROLE_MODELS["describe"]

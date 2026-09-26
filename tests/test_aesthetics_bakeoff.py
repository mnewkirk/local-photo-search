"""evals/aesthetics_bakeoff.py — pinning, the v2 run cache, and the metrics.

No model, no network: the production scorer is replaced by a stub that goes
through the (stubbed) chat entry point, so the Recorder sees real calls."""

import importlib.util
import json
import math
import os
import sqlite3

import pytest

from photosearch import describe


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "aesthetics_bakeoff.py")
    spec = importlib.util.spec_from_file_location("aesthetics_bakeoff_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load()


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    for v in ("PHOTOSEARCH_LLM_AESTHETICS_MODEL", "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.delenv(v, raising=False)
    # The trap: the shell exports the VISUAL model.
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen-visual")
    import urllib.request

    def no_network(*a, **kw):
        raise OSError("no network in tests")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)


def _scorer_from(answers):
    """A scorer that asks the stubbed chat once per answer until one parses."""
    def scorer(path, model):
        name = os.path.basename(path)
        for raw in answers[name]:
            got = describe._ollama_chat_with_retry(model=model, messages=[],
                                                   role="aesthetics")
            if got and got.startswith("{"):
                return {"overall": json.loads(got)["overall"]}
        return None
    return scorer


def _chat_from(answers, monkeypatch):
    seq = {k: list(v) for k, v in answers.items()}
    current = {}

    def chat(*a, **kw):
        raw = seq[current["name"]].pop(0)
        if isinstance(raw, Exception):
            raise raw
        return raw
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", chat)
    return current


def test_run_vlm_pins_the_model_and_caches_only_real_results(tmp_path, monkeypatch):
    answers = {"a.jpg": ['{"overall": 7.0}'],
               "b.jpg": ["prose", '{"overall": 5.0}'],     # parse-fail, then ok
               "c.jpg": ["prose", "prose"],                  # parse-fail both
               "d.jpg": [ConnectionError("refused")]}        # transport
    current = _chat_from(answers, monkeypatch)
    real_scorer = _scorer_from(answers)
    # The stub answers are not full rubric JSON; parse the way the stub scorer does.
    from photosearch import aesthetics
    monkeypatch.setattr(aesthetics, "parse_aesthetics_response",
                        lambda raw: json.loads(raw) if raw.startswith("{") else None)

    def scorer(path, model):
        current["name"] = os.path.basename(path)
        try:
            return real_scorer(path, model)
        except ConnectionError:
            return None                                       # production swallows it

    photos = [(n, str(tmp_path / n)) for n in answers]
    v2 = tmp_path / "scores-v2.json"
    variant, entry = B.run_vlm("gemma-x", photos, str(v2), scorer=scorer,
                               log=lambda *_: None)

    assert os.environ["PHOTOSEARCH_LLM_AESTHETICS_MODEL"] == "gemma-x"
    assert entry["effective_model"] == "gemma-x"          # not qwen-visual
    items = entry["items"]
    assert set(items) == {"a.jpg", "b.jpg", "c.jpg"}      # d not cached
    assert items["a.jpg"]["first_parse_ok"] is True
    assert items["b.jpg"] == {**items["b.jpg"], "overall": 5.0, "first_parse_ok": False,
                              "calls": 2}
    assert items["c.jpg"]["overall"] is None
    assert entry["latency_label"] == "unknown"
    assert json.loads(v2.read_text())[variant]["items"].keys() == items.keys()

    # Resuming under another effective model is refused.
    monkeypatch.setenv("PHOTOSEARCH_LLM_AESTHETICS_MODEL", "other")
    with pytest.raises(SystemExit, match="scored by"):
        B.run_vlm("other", photos, str(v2), variant=variant, scorer=scorer,
                  log=lambda *_: None)


def test_vlm_summary_counts_parse_failures_and_rho():
    entry = {"items": {
        "a": {"overall": 8.0, "first_parse_ok": True, "latency_s": 9.0},
        "b": {"overall": 6.0, "first_parse_ok": False, "latency_s": 1.0},
        "c": {"overall": None, "first_parse_ok": False, "latency_s": 2.0},
        "d": {"overall": 4.0, "first_parse_ok": True, "latency_s": 3.0},
        "pid:9": {"overall": 1.0, "first_parse_ok": True, "latency_s": 1.0}}}
    gt = {"a": 3, "b": 2, "d": 1}                 # higher = better, agrees fully
    r = B.vlm_summary(entry, gt)
    assert r["n"] == 4                            # selection photos excluded
    assert (r["first_fail"], r["final_fail"]) == (2, 1)
    assert r["rho"] == pytest.approx(1.0)
    assert r["lat_median"] == 2.0                 # first (JIT) call dropped


def test_spread_catches_squashed_scores():
    sq = B.spread_for([7.0, 7.0, 7.1, 6.9, 7.0])
    wide = B.spread_for([2.0, 4.0, 6.0, 8.0, 9.5])
    assert sq["near_median"] == 1.0 and sq["distinct"] == 3
    assert wide["near_median"] == pytest.approx(0.2)
    assert wide["iqr"] > sq["iqr"]


def test_spearman_ci_brackets_rho_and_is_wide_at_n28():
    lo, hi = B.spearman_ci(0.70, 28)
    assert lo < 0.70 < hi
    # Bonett-Wright by hand.
    se = math.sqrt((1 + 0.49 / 2) / 25)
    assert lo == pytest.approx(math.tanh(math.atanh(0.7) - 1.96 * se))
    assert hi - lo > 0.3
    assert all(v != v for v in B.spearman_ci(0.5, 3))


def test_selection_groups_and_agreement(tmp_path):
    db = tmp_path / "p.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE review_selections (photo_id, directory, selected, cluster_id)")
    conn.executemany("INSERT INTO review_selections VALUES (?,?,?,?)", [
        (1, "d", 1, 10), (2, "d", 0, 10), (3, "d", 0, 10),   # usable
        (4, "d", 1, 11), (5, "d", 1, 11),                    # no not-kept: dropped
        (6, "d", 0, None)])                                   # no cluster: dropped
    conn.commit()
    groups = B.selection_groups(B.open_db_readonly(str(db)))
    assert groups == [([1], [2, 3])]
    agree, total = B.selection_agreement(groups, {1: 7.0, 2: 5.0, 3: 7.0})
    assert (agree, total) == (1.5, 2)


def test_one_vlm_per_process_on_the_lmstudio_route(monkeypatch):
    monkeypatch.setattr("sys.argv", ["x", "--vlm", "a", "--vlm", "b",
                                     "--photos-dir", "."])
    with pytest.raises(SystemExit, match="One --vlm per process"):
        B.main()

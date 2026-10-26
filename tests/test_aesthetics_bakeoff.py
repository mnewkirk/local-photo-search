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
        monkeypatch.setenv(v, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(v)
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


def test_score_vlm_stores_dimension_and_sub_scores_and_old_entries_still_read(tmp_path):
    def scorer(path, model):
        describe._ollama_chat_with_retry(model=model, messages=[], role="aesthetics")
        return {"overall": 6.5, "dim_scores": {"technical": 7.0},
                "sub_scores": {"sharpness": 7.5}, "critiques": {"technical": "ok"}}

    import photosearch.describe as d
    orig = d._ollama_chat_with_retry
    d._ollama_chat_with_retry = lambda *a, **kw: '{"x": 1}'
    try:
        entry = B.score_vlm("m", [("a.jpg", str(tmp_path / "a.jpg"))], {"items": {}},
                            scorer=scorer, log=lambda *_: None)
    finally:
        d._ollama_chat_with_retry = orig
    item = entry["items"]["a.jpg"]
    assert item["overall"] == 6.5
    assert item["dim_scores"] == {"technical": 7.0}
    assert item["sub_scores"] == {"sharpness": 7.5}
    assert "critiques" not in item                       # scores only
    # An entry from before the breakdowns were stored still summarises.
    old = {"items": {"a": {"overall": 7.0, "first_parse_ok": True, "latency_s": 1.0}}}
    assert B.vlm_summary(old, {})["n"] == 1


def test_ground_truth_ranks_reports_missing_and_duplicate_ranks(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("filename,rank\na,1\nb,2\nc,4\nd,4\ne,6\n")
    info = B.ground_truth_ranks(str(p))
    assert info["n"] == 5 and info["missing"] == [3, 5] and info["duplicates"] == [4]
    assert B.ground_truth_ranks(str(tmp_path / "nope.csv")) is None


def test_baseline_scores_read_file_size_and_resolution(tmp_path):
    from PIL import Image
    small, big = tmp_path / "s.jpg", tmp_path / "b.jpg"
    Image.new("RGB", (100, 50)).save(small)
    Image.new("RGB", (400, 300), (200, 10, 10)).save(big)
    b = B.baseline_scores([("s", str(small)), ("b", str(big))])
    assert b["baseline:megapixels"] == {"s": pytest.approx(0.005), "b": pytest.approx(0.12)}
    assert b["baseline:file-size"]["b"] > b["baseline:file-size"]["s"]


def test_gt_analysis_splits_halves_and_marks_ties():
    names = [f"p{i}" for i in range(12)]
    gt = {n: -i for i, n in enumerate(names)}            # p0 best
    perfect = {n: -i for i, n in enumerate(names)}
    # Right on the bottom half only: the headline ρ hides a random top half.
    tail = {n: (v if i >= 6 else [3, -5, 1, 0, -2, 4][i])
            for i, (n, v) in enumerate(perfect.items())}
    analysis = B.gt_analysis({"perfect": perfect, "tail": tail,
                              "baseline:size": tail}, gt, reference="perfect", n_boot=300)
    r = analysis["rows"]
    assert analysis["halves"] == (6, 6) and analysis["best_baseline"] == "baseline:size"
    assert r["perfect"]["rho"] == pytest.approx(1.0) and r["perfect"]["top"] == pytest.approx(1.0)
    assert r["tail"]["bottom"] == pytest.approx(1.0) and r["tail"]["top"] < 0.5
    assert r["perfect"]["vs_reference"] is None
    # Identical scorers can never be told apart.
    assert B._tie(r["tail"]["vs_best_baseline"]).startswith("TIE")
    lo, hi = r["tail"]["ci"]
    assert lo <= r["tail"]["rho"] <= hi
    lines = B.gt_analysis_lines(analysis, {"n": 12, "ranks": list(range(1, 13)),
                                           "missing": [], "duplicates": []}, {})
    assert any("NO stored results" in line for line in lines)

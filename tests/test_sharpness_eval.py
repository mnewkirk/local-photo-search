"""Scoring / sweep / gate logic of evals/sharpness_eval.py -- synthetic labels
and measurements, no network, no DB."""

import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "sharpness_eval", os.path.join(_HERE, "..", "evals", "sharpness_eval.py"))
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _lab(yes=(), debatable=(), done=True, **kw):
    return {"yes": list(yes), "debatable": list(debatable), "done": done, **kw}


def test_truth_is_three_state_and_respects_offered():
    assert E.truth_of(_lab(yes=["blurry"]), "blurry") is True
    assert E.truth_of(_lab(), "blurry") is False
    assert E.truth_of(_lab(debatable=["blurry"]), "blurry") is None
    assert E.truth_of(_lab(offered=["moody"]), "blurry") is None


def test_counts_skip_debatable_and_missing_prediction_is_negative():
    truth = {1: True, 2: False, 3: None, 4: True}
    pred = {1: True, 2: True, 3: True}          # 4 missing -> negative
    c = E.counts(pred, truth)
    assert c == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}


def test_fmt_ratio_hides_tiny_denominators():
    assert E.fmt_ratio(1, 2) == "n<3"
    assert E.fmt_ratio(2, 4) == "0.50"


def _world():
    """10 blurry (low metric) and 10 sharp photos, strata night/bokeh/other."""
    values, truth, strata = {}, {}, {}
    for i in range(20):
        blurry = i < 10
        values[i] = float(i)            # blurry ones have the lowest values
        truth[i] = blurry
        strata[i] = ("night-iso" if i % 3 == 0 else "bokeh-portrait" if i % 3 == 1
                     else "random")
    return values, truth, strata


def test_blurry_sweep_finds_the_perfect_threshold_and_passes():
    values, truth, strata = _world()
    groups = {"night": E.group_ids(strata, ["night"]), "bokeh": E.group_ids(strata, ["bokeh"])}
    ev = E.evaluate_blurry(values, truth, groups)
    assert ev["pass"]
    g = ev["best_gate"]
    assert g["threshold"] == 9.0                 # <= 9 catches exactly 0..9
    assert g["precision"] == 1.0 and g["recall"] == 1.0
    # lower thresholds still pass the overall gate at R >= 0.5 but recall less
    assert any(r["overall_ok"] and r["recall"] < 1 for r in ev["sweep"])


def test_blurry_gate_fails_when_a_stratum_is_imprecise():
    values, truth, strata = _world()
    # Make every sharp BOKEH photo read lowest -> high-FP in bokeh.
    for i in range(10, 20):
        if strata[i].startswith("bokeh"):
            values[i] = -1.0
    groups = {"night": E.group_ids(strata, ["night"]), "bokeh": E.group_ids(strata, ["bokeh"])}
    ev = E.evaluate_blurry(values, truth, groups)
    # overall P at the perfect-recall threshold is 10/13 = 0.77 < 0.8 anyway;
    # every threshold includes the 3 bokeh FPs, so bokeh precision fails.
    assert not ev["pass"]
    assert all(not r["passes"] for r in ev["sweep"])


def test_gate_fails_when_a_stratum_has_no_labels():
    values, truth, strata = _world()
    groups = {"night": E.group_ids(strata, ["night"]), "bokeh": []}
    assert not E.evaluate_blurry(values, truth, groups)["pass"]


def test_stratum_with_few_predictions_passes_only_without_fp():
    assert E.stratum_ok({"tp": 1, "fp": 0, "fn": 0, "tn": 3}, 0.6) == (True, "n<3 (1/1)")
    assert E.stratum_ok({"tp": 1, "fp": 1, "fn": 0, "tn": 3}, 0.6)[0] is False
    assert E.stratum_ok({"tp": 3, "fp": 1, "fn": 0, "tn": 0}, 0.6) == (True, "0.75")


def test_none_values_never_fire():
    truth = {1: True, 2: True, 3: False, 4: True}
    values = {1: 1.0, 2: None, 3: 5.0, 4: 2.0}
    ev = E.evaluate_blurry(values, truth, {})
    best = ev["best_f1"]
    assert best["threshold"] == 2.0
    assert best["counts"] == {"tp": 2, "fp": 0, "fn": 1, "tn": 1}


def test_sharp_keep_and_retire_rules():
    # 10 sharp with high values, 10 not sharp low: a threshold at 10 fires on
    # exactly 50% -> above the 40% cap, so only a stricter threshold keeps it.
    values = {i: float(i) for i in range(20)}
    truth = {i: i >= 10 for i in range(20)}
    ev = E.evaluate_sharp(values, truth)
    assert ev["keep"]
    k = ev["best_keep"]
    assert k["firing"] <= 0.40 and k["precision"] >= 0.8
    assert k["threshold"] == 12.0          # 8/20 = 40% firing, max recall allowed
    # A metric that cannot separate them -> retire.
    flat = {i: float(i % 2) for i in range(20)}
    assert not E.evaluate_sharp(flat, truth)["keep"]


def test_load_label_sets_merges_filters_and_reads_sibling_sample(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    (a / "labels.json").write_text(json.dumps({"labels": {
        "1": _lab(yes=["blurry"], updated_at="2026-09-20T00:00:00"),
        "2": _lab(done=False),
        "3": _lab(updated_at="2026-09-27T00:00:00")}}))
    (a / "sample.json").write_text(json.dumps({"photos": [
        {"photo_id": 1, "stratum": "night"}, {"photo_id": 3, "stratum": "bokeh"}]}))
    (b / "labels.json").write_text(json.dumps({"labels": {
        "4": _lab(yes=["sharp"], updated_at="2026-09-28T00:00:00")}}))
    labels, strata = E.load_label_sets([str(a / "labels.json"), str(b / "labels.json")],
                                       log=lambda *_: None)
    assert sorted(labels) == [1, 3, 4]              # un-done photo dropped
    assert strata == {1: "night", 3: "bokeh"}
    labels, _ = E.load_label_sets([str(a / "labels.json"), str(b / "labels.json")],
                                  labelled_after="2026-09-26", log=lambda *_: None)
    assert sorted(labels) == [3, 4]


def _fake_result(score, frame_med, error=False):
    if error:
        return {"score": None, "detail": {"error": "boom"}, "version": 1, "seconds": 0.1}
    reg = {f: None for f in E.sharpness.REGION_FEATURES}
    reg.update({"lap_nc_max": score, "lap_max": score, "lap_median": frame_med,
                "boxes": 1, "boxes_skipped": 0, "tiles": 4, "flat_skipped": 0})
    return {"score": score, "version": 1, "seconds": 0.2,
            "detail": {"noise_sigma": 1.0, "noise_sigma_raw": 1.2, "source": "frame",
                       "regions": {"frame": reg}}}


def test_build_and_render_end_to_end_without_network():
    labels, strata, rows, cache = {}, {}, {}, {}
    for i in range(24):
        blurry = i < 8
        labels[i] = _lab(yes=["blurry"] if blurry else (["sharp"] if i > 18 else []))
        strata[i] = ["night-iso", "bokeh", "random"][i % 3]
        rows[i] = {"camera_model": "ILCE-7M4" if i % 2 else "Pixel 7 Pro",
                   "aes_sharpness": 2 if i < 5 else 7,
                   "visual_tags": json.dumps(["blurry"] if i in (0, 9) else [])}
        cache[str(i)] = _fake_result(10.0 + 100 * (i >= 8) + i, 5.0)
    cache["23"] = _fake_result(None, None, error=True)
    rep = E.build(labels, strata, rows, cache)
    assert rep["n_measured"] == 24 and rep["errors"] == 1
    assert rep["blurry"]["score"]["pass"]
    assert rep["baselines"]["blurry"]["stored blurry"]["counts"]["tp"] == 1
    assert rep["baselines"]["blurry"]["aes_sharpness<=2"]["counts"]["tp"] == 5
    text = E.render(rep)
    assert "BLURRY GATE: PASS" in text
    assert "SHARP:" in text
    assert "Pixel 7 Pro" in text and "night-iso" in text


def test_measure_all_caches_by_version_and_skips_cached(tmp_path):
    cache_path = str(tmp_path / "m.json")
    calls = []

    def fake_measure(path, faces, subjects):
        calls.append((path, len(faces), subjects))
        return {"score": 1.0, "detail": {}, "version": E.sharpness.SHARPNESS_VERSION}

    rows = {1: {"faces": [{"bbox_left": 0}], "subject_boxes": '[{"bbox": [0,0,1,1]}]'},
            2: {}}
    E.measure_all([1, 2], rows, cache_path, "http://x", fetch_path=lambda p: f"/f/{p}",
                  measure=fake_measure, log=lambda *_: None)
    assert [c[0] for c in calls] == ["/f/1", "/f/2"]
    assert calls[0][1] == 1 and calls[0][2] == [{"bbox": [0, 0, 1, 1]}]
    saved = json.load(open(cache_path))
    assert set(saved[f"v{E.sharpness.SHARPNESS_VERSION}"]) == {"1", "2"}
    E.measure_all([1, 2, 3], rows, cache_path, "http://x", fetch_path=lambda p: f"/f/{p}",
                  measure=fake_measure, log=lambda *_: None)
    assert [c[0] for c in calls][2:] == ["/f/3"]


def test_fetch_failure_is_not_cached(tmp_path):
    cache_path = str(tmp_path / "m.json")

    def bad(_pid):
        raise OSError("502")
    E.measure_all([1], {}, cache_path, "http://x", fetch_path=bad,
                  measure=lambda *a: pytest.fail("should not measure"), log=lambda *_: None)
    assert E.load_cache(cache_path) == {}

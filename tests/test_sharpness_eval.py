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


def test_truth_is_three_state():
    assert E.truth_of(_lab(yes=["blurry"]), "blurry") is True
    assert E.truth_of(_lab(), "blurry") is False
    assert E.truth_of(_lab(debatable=["blurry"]), "blurry") is None


def test_default_strata_match_what_the_sharpness_sampler_writes():
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "visual_tags_eval", os.path.join(_HERE, "..", "evals", "visual_tags_eval.py"))
    V = iu.module_from_spec(spec)
    spec.loader.exec_module(V)
    names = [n for n, _q, _p in V.build_sharpness_strata()]
    for needles in (E.DEFAULT_NIGHT, E.DEFAULT_BOKEH):
        assert any(n in s for n in needles for s in names), needles
    assert E.group_ids({1: "iso>=3200-or-night", 2: "bokeh-portrait", 3: "stored-blurry"},
                       E.DEFAULT_NIGHT) == [1]
    assert E.group_ids({1: "iso>=3200-or-night", 2: "bokeh-portrait", 3: "stored-blurry"},
                       E.DEFAULT_BOKEH) == [2]


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


@pytest.fixture
def eval_dir(tmp_path, monkeypatch):
    """A real label tree under a tmp PHOTOSEARCH_VISUAL_EVAL_DIR: the visual
    sample + labels at the top, the sharpness ones under sharpness/."""
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(tmp_path))
    (tmp_path / "sharpness").mkdir()
    (tmp_path / "sample.json").write_text(json.dumps({"photos": [
        {"photo_id": 10, "stratum": "low-light"},
        {"photo_id": 11, "stratum": "indoor"}]}))
    (tmp_path / "labels.json").write_text(json.dumps({"labels": {
        # measured: the chips were shown and the answer was "neither"
        "10": _lab(yes=["moody"], measured=True),
        # labelled before the chips existed: silence is NOT a no -> skipped
        "11": _lab(yes=["sunny"])}}))
    (tmp_path / "sharpness" / "sample.json").write_text(json.dumps({"photos": [
        {"photo_id": 1, "stratum": "iso>=3200-or-night"},
        {"photo_id": 2, "stratum": "bokeh-portrait"},
        {"photo_id": 3, "stratum": "stored-blurry"},
        {"photo_id": 4, "stratum": "random"}]}))
    (tmp_path / "sharpness" / "labels.json").write_text(json.dumps({"labels": {
        "1": _lab(yes=["blurry"], measured=True),
        "2": _lab(yes=["sharp"], debatable=["blurry"], measured=True),
        "3": _lab(yes=["blurry"], done=False, measured=True),     # not done
        "4": _lab(yes=["peaceful"])}}))                             # unmeasured
    return tmp_path


def test_load_labels_uses_measured_labels_and_skips_unmeasured(eval_dir):
    labels, strata = E.load_labels()
    assert sorted(labels) == [1, 2, 10]         # 11 + 4 unmeasured, 3 not done
    assert strata == {1: "iso>=3200-or-night", 2: "bokeh-portrait", 10: "low-light"}
    assert labels[2] == {"yes": ["sharp"], "debatable": ["blurry"], "set": "sharpness"}
    assert labels[10]["yes"] == [] and labels[10]["set"] == "main"
    assert E.truth_of(labels[1], "blurry") is True
    assert E.truth_of(labels[2], "blurry") is None
    assert E.truth_of(labels[10], "blurry") is False
    only, _ = E.load_labels(["sharpness"])
    assert sorted(only) == [1, 2]
    assert E.default_cache_path() == str(eval_dir / "sharpness" / "measurements.json")


def test_main_reads_the_real_label_path_end_to_end(eval_dir, monkeypatch, capsys):
    import sqlite3
    db = eval_dir / "p.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, camera_model TEXT, aes_sharpness REAL,"
        " visual_tags TEXT, subject_boxes TEXT, iso INTEGER);"
        "CREATE TABLE faces (photo_id INTEGER, bbox_left INT, bbox_top INT,"
        " bbox_right INT, bbox_bottom INT);"
        "INSERT INTO photos (id, camera_model) VALUES (1,'ILCE-7M4'),(2,'ILCE-7M4'),(10,'phone');")
    conn.commit()
    conn.close()
    E.save_cache(E.default_cache_path(), {
        "1": _fake_result(5.0, 1.0, upsampled=True),
        "2": _fake_result(90.0, 9.0), "10": _fake_result(80.0, 8.0)})
    assert E.main(["report", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "3 done + measured photos" in out
    assert "main=1, sharpness=2" in out
    assert "upsampled (original < normalised edge): 1" in out
    assert "resolution" in out and "upsampled" in out


def _fake_result(score, frame_med, error=False, upsampled=False):
    if error:
        return {"score": None, "detail": {"error": "boom"}, "version": 1, "seconds": 0.1}
    reg = {f: None for f in E.sharpness.REGION_FEATURES}
    reg.update({"lap_nc_max": score, "lap_max": score, "lap_median": frame_med,
                "boxes": 1, "boxes_skipped": 0, "tiles": 4, "flat_skipped": 0})
    return {"score": score, "version": 1, "seconds": 0.2,
            "detail": {"noise_sigma": 1.0, "noise_sigma_raw": 1.2, "source": "frame",
                       "upsampled": upsampled, "regions": {"frame": reg}}}


def test_build_and_render_end_to_end_without_network():
    labels, strata, rows, cache = {}, {}, {}, {}
    for i in range(24):
        blurry = i < 8
        labels[i] = _lab(yes=["blurry"] if blurry else (["sharp"] if i > 18 else []))
        strata[i] = ["iso>=3200-or-night", "bokeh-portrait", "random"][i % 3]
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
    assert "Pixel 7 Pro" in text and "iso>=3200-or-night" in text


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

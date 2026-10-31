"""The eval's scoring and gates, on synthetic labels (no model, no DB)."""
from photosearch import people_bind_eval as E


def _slot(text, *fids):
    return {"slot": "s1", "kind": "person" if len(fids) == 1 else "group",
            "spans": [{"start": 0, "end": len(text), "text": text}],
            "faces": [{"face_id": f, "bbox": [0, 1, 1, 0]} for f in fids]}


LABELS = {
    "1": {"split": "holdout", "stratum": "single",
          "mentions": [{"phrase": "A boy", "gold": [10]}]},
    "2": {"split": "holdout", "stratum": "multi_2_3",
          "mentions": [{"phrase": "A boy in blue", "gold": [20]},
                       {"phrase": "a girl", "gold": "NONE"},
                       {"phrase": "the keeper", "gold": "AMBIGUOUS"}]},
    "3": {"split": "dev", "stratum": "adversarial",
          "mentions": [{"phrase": "A man", "gold": "NONE"}]},
}


def test_outcomes_per_mention():
    preds = {"1": {"method": "rule", "slots": [_slot("A boy", 10)]},
             "2": {"method": "vlm", "slots": [_slot("A boy in blue", 21), _slot("a girl", 22)]},
             "3": {"method": "vlm", "slots": []}}
    r = E.score(LABELS, preds)
    t = r["total"]
    assert t == {"correct_bind": 1, "wrong_bind": 2, "missed_abstain": 0, "right_abstain": 2}
    assert {c["phrase"] for c in r["wrong_cases"]} == {"A boy in blue", "a girl"}
    assert r["by_method"]["rule"]["correct_bind"] == 1
    assert r["by_stratum"]["adversarial"]["right_abstain"] == 1


def test_split_filter_and_deferred_photos_are_not_scored():
    preds = {"1": {"method": "rule", "slots": [_slot("A boy", 10)]},
             "2": {"method": None, "slots": None}}
    r = E.score(LABELS, preds, split="holdout")
    assert r["photos"] == 1 and r["mentions"] == 1


def test_a_bound_phrase_nobody_labelled_is_counted():
    preds = {"1": {"method": "vlm", "slots": [_slot("A boy", 10), _slot("another kid", 11)]}}
    assert E.score(LABELS, preds)["unscored_bind"] == 1


def test_a_correct_box_on_a_wrongly_labelled_face_is_a_user_visible_wrong_name():
    labels = {"1": dict(LABELS["1"], bad_labels=[10])}
    preds = {"1": {"method": "rule", "slots": [_slot("A boy", 10)]}}
    r = E.score(labels, preds, face_sources={10: "strict"})
    assert r["total"]["correct_bind"] == 1
    assert r["visible_wrong"] == {"by_source": {"strict": 1}, "total": 1}
    gate = {n: ok for n, ok, _ in E.gates(r)}
    assert gate["strict-caused user-visible wrong names == 0 (keeps `strict` in the name sources)"] is False


def test_gates_fail_on_one_wrong_bind_and_are_incomplete_without_enough_data():
    preds = {"1": {"method": "rule", "slots": [_slot("A boy", 10)]},
             "2": {"method": "vlm", "slots": [_slot("A boy in blue", 99)]}}
    rows = E.gates(E.score(LABELS, preds, split="holdout"))
    assert E.verdict(rows) == "FAIL"
    clean = {"1": {"method": "rule", "slots": [_slot("A boy", 10)]}}
    assert E.verdict(E.gates(E.score(LABELS, clean, split="holdout"))) == "INCOMPLETE"   # < 150 bound


def test_all_gates_pass_on_a_big_clean_synthetic_set():
    labels, preds = {}, {}
    for i in range(160):
        stratum = "single" if i % 2 else "multi_2_3"
        labels[str(i)] = {"split": "holdout", "stratum": stratum,
                          "mentions": [{"phrase": f"A boy {i}", "gold": [i]},
                                       {"phrase": f"a girl {i}", "gold": "NONE"}]}
        preds[str(i)] = {"method": "vlm", "slots": [_slot(f"A boy {i}", i)]}
    labels["adv"] = {"split": "holdout", "stratum": "adversarial",
                     "mentions": [{"phrase": "A man", "gold": "NONE"}]}
    preds["adv"] = {"method": "vlm", "slots": []}
    rows = E.gates(E.score(labels, preds, split="holdout"))
    assert E.verdict(rows) == "PASS", rows


def test_split_is_stable_and_roughly_a_third_holdout():
    assert E.split_of(248080) == E.split_of(248080)
    share = sum(E.split_of(i) == "holdout" for i in range(3000)) / 3000
    assert 0.25 < share < 0.42


# --- the CLI harness end to end, rule tier only (no model, no network) ---------------

def test_harness_sample_run_report_rule_tier(tmp_path, monkeypatch, capsys):
    import importlib.util, json, sys
    from pathlib import Path
    from photosearch.db import PhotoDB
    spec = importlib.util.spec_from_file_location(
        "people_bind_eval_cli", Path(__file__).resolve().parent.parent / "evals" / "people_bind_eval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for name in ("HERE", "SAMPLE", "LABELS", "RUNS", "SHEET"):
        monkeypatch.setattr(cli, name, tmp_path / "pb" / (getattr(cli, name).name))
    monkeypatch.setattr(cli, "HERE", tmp_path / "pb")

    db_path = str(tmp_path / "t.db")
    with PhotoDB(db_path) as d:
        pid = d.add_photo(
            filepath="2026/x/a.JPG", filename="a.JPG", date_taken="2026-10-03 12:00:00",
            description="A young boy with reddish hair walks across a field.",
            image_width=6528, image_height=4352,
            subject_boxes=json.dumps([{"label": "person", "bbox": [0.4, 0.4, 0.62, 0.93],
                                       "area_frac": 0.1}]))
        zack = d.add_person("Zack Azerad")
        fid = d.add_face(pid, (1936, 3387, 2228, 3153), [], person_id=zack, det_score=0.85)
        d.conn.execute("UPDATE faces SET match_source='manual' WHERE id=?", (fid,))
        d.conn.commit()

    ns = lambda **kw: type("A", (), kw)()
    cli.cmd_sample(ns(db=db_path, date_from="2026-01-01", seed=1, single=5, multi_2_3=5,
                      multi_4_6=5, add=[]))
    assert list(json.loads(cli.SAMPLE.read_text())) == [str(pid)]
    cli.cmd_run(ns(db=db_path, variant="rule", tier="rule", model=None, force=False,
                   server="x"))
    # hand labels
    cli.LABELS.write_text(json.dumps({str(pid): {
        "split": "holdout", "stratum": "single",
        "mentions": [{"phrase": "A young boy with reddish hair", "gold": [fid]}],
        "bad_labels": []}}))
    cli.cmd_report(ns(db=db_path, split="holdout"))
    out = capsys.readouterr().out
    assert "correct_bind" in out and "INCOMPLETE" in out          # 1 bound << 150
    assert "WRONG" not in out


# --- phrase boundaries: the model picks its own -------------------------------------------

def test_a_bind_counts_when_its_span_overlaps_the_labelled_phrase():
    desc = "One child in blue is dribbling while others chase."
    labels = {"7": {"split": "dev", "stratum": "multi_2_3",
                    "mentions": [{"phrase": "One child", "gold": [5]}]}}
    longer = {"slot": "s1", "kind": "person",
              "spans": [{"start": 0, "end": 17, "text": "One child in blue"}],
              "faces": [{"face_id": 5, "bbox": [0, 1, 1, 0]}]}
    preds = {"7": {"method": "vlm", "slots": [longer]}}
    r = E.score(labels, preds, descriptions={"7": desc})
    assert r["total"]["correct_bind"] == 1 and r["unscored_bind"] == 0
    # exact-text matching (no descriptions) would call the same answer an unscored bind
    r = E.score(labels, preds)
    assert r["unscored_bind"] == 1 and r["total"]["missed_abstain"] == 1


# --- label validation (the `check` step) ---------------------------------------------------

def test_check_labels_reports_every_kind_of_mistake():
    desc = "A goalkeeper in orange stands near one child in green."
    sample = {"9": {"split": "dev", "stratum": "adversarial"}, "10": {"split": "dev", "stratum": "single"}}
    photos = {"9": {"description": desc, "offered": {1, 2}, "faces": {1, 2, 3}}}
    ok = {"9": {"split": "dev", "stratum": "adversarial", "bad_labels": [3], "reviewed": True,
                "mentions": [{"phrase": "A goalkeeper", "gold": [1]},
                             {"phrase": "one child in green", "gold": None}]}}
    r = E.check_labels(ok, sample, photos)
    assert r["errors"] == [] and r["labelled"] == 1 and r["unlabelled"] == 1
    assert r["missing"] == ["10"] and r["unreviewed"] == []
    bad = {"9": {"split": "holdout", "stratum": "adversarial", "bad_labels": [99],
                 "mentions": [{"phrase": "A Goalkeeper", "gold": [1]},          # case differs
                              {"phrase": "A goalkeeper in orange", "gold": [3]},  # not drawn
                              {"phrase": "A goalkeeper", "gold": "MAYBE"},        # overlaps; bad word
                              {"phrase": "one child", "gold": [1, 1]}]}}         # duplicate face
    errs = "\n".join(E.check_labels(bad, sample, photos)["errors"])
    for needle in ("split", "exact substring", "not drawn", "overlap", "MAYBE",
                   "distinct", "bad_labels"):
        assert needle in errs, needle


# --- the CLI's run step: a transport failure is never cached --------------------------------

def _cli(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "people_bind_eval_cli2", Path(__file__).resolve().parent.parent / "evals" / "people_bind_eval.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    for name in ("SAMPLE", "LABELS", "RUNS", "SHEET"):
        monkeypatch.setattr(cli, name, tmp_path / "pb" / (getattr(cli, name).name))
    monkeypatch.setattr(cli, "_original", lambda pid, server: tmp_path / "img.jpg")
    return cli


def _two_face_db(tmp_path):
    import json
    from photosearch.db import PhotoDB
    db_path = str(tmp_path / "t.db")
    with PhotoDB(db_path) as d:
        pid = d.add_photo(filepath="2026/x/a.JPG", filename="a.JPG",
                          date_taken="2026-10-03 12:00:00",
                          description="A boy in blue and a girl in red play.",
                          image_width=4000, image_height=3000)
        p = d.add_person("Zack Azerad")
        f1 = d.add_face(pid, (100, 400, 400, 100), [], person_id=p, det_score=0.9)
        d.add_face(pid, (100, 900, 400, 600), [], det_score=0.9)
        d.conn.execute("UPDATE faces SET match_source='manual' WHERE id=?", (f1,))
        d.conn.commit()
    (tmp_path / "pb").mkdir()
    (tmp_path / "pb" / "sample.json").write_text(json.dumps(
        {str(pid): {"stratum": "multi_2_3", "split": "dev"}}))
    return db_path, pid


def test_run_never_caches_a_transport_failure_but_records_an_unusable_answer(tmp_path, monkeypatch):
    import json
    from photosearch import people_bind as PB
    from photosearch.describe import UnusableAnswer
    cli = _cli(tmp_path, monkeypatch)
    db_path, pid = _two_face_db(tmp_path)
    # setenv first so teardown removes what cmd_run's setdefault writes
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "x")
    monkeypatch.delenv("PHOTOSEARCH_LLM_REASONING_EFFORT")
    ns = lambda **kw: type("A", (), dict(dict(db=db_path, variant="v", tier="vlm", model=None,
                                              force=False, server="x"), **kw))()
    monkeypatch.setattr(PB, "bind_with_model", lambda *a, **k: None)          # no answer
    cli.cmd_run(ns())
    run = json.loads((tmp_path / "pb" / "runs" / "v.json").read_text()) \
        if (tmp_path / "pb" / "runs" / "v.json").exists() else {"preds": {}}
    assert run["preds"] == {}
    import os
    assert os.environ["PHOTOSEARCH_LLM_REASONING_EFFORT"] == "none"          # the fleet's setting

    def unusable(*a, **k):
        raise UnusableAnswer("people-bind: not a JSON array")
    monkeypatch.setattr(PB, "bind_with_model", unusable)
    cli.cmd_run(ns())
    run = json.loads((tmp_path / "pb" / "runs" / "v.json").read_text())
    pred = run["preds"][str(pid)]
    assert pred["slots"] == [] and "UnusableAnswer" in pred["error"] and pred["calls"] == 2
    assert run["spec"]["max_px"] == PB._OVERLAY_MAX_PX and run["spec"]["second_call"] is True


def test_run_refuses_to_resume_under_a_different_spec(tmp_path, monkeypatch):
    import pytest
    from photosearch import people_bind as PB
    cli = _cli(tmp_path, monkeypatch)
    db_path, pid = _two_face_db(tmp_path)
    monkeypatch.setenv("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    monkeypatch.setattr(PB, "bind_with_model", lambda *a, **k: [])
    ns = lambda **kw: type("A", (), dict(dict(db=db_path, variant="v", tier="vlm", model=None,
                                              force=False, server="x"), **kw))()
    cli.cmd_run(ns())
    with pytest.raises(SystemExit):
        cli.cmd_run(ns(max_px=1536))                 # an ablation needs its own variant name
    cli.cmd_run(ns(max_px=1536, force=True))         # or --force to start over


def test_one_call_ablation_skips_the_reshuffled_second_call(monkeypatch):
    from photosearch import people_bind as PB
    calls = []
    monkeypatch.setattr(PB, "draw_overlay", lambda path, faces, order, max_px=0: calls.append(max_px) or "b64")
    monkeypatch.setattr(PB, "ask_model", lambda *a, **k: [])
    faces = [{"face_id": 1, "bbox": [0, 100, 100, 0], "det_score": 0.9},
             {"face_id": 2, "bbox": [0, 300, 100, 200], "det_score": 0.9}]
    PB.bind_with_model("x.jpg", "A boy.", faces, 7, "m")
    assert calls == [PB._OVERLAY_MAX_PX] * 2
    calls.clear()
    PB.bind_with_model("x.jpg", "A boy.", faces, 7, "m", max_px=1536, second_call=False)
    assert calls == [1536]

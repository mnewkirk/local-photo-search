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

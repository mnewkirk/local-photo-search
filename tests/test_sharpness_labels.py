"""Sharpness eval, step 1 (docs/plans/sharpness-measurement.md): labels first.

`sharp` / `blurry` are MEASURED tags — labelled by the owner as ground truth
for a native-resolution measurement, and never scored against a VLM variant.
The sharpness sample lives beside the visual one (`sharpness/`), so drawing it
must never orphan the visual sample, its labels or its cached runs.

No real DB anywhere: a tmp sqlite fixture, opened read-only by the sampler.
"""

import hashlib
import importlib.util
import json
import os
import sqlite3

import pytest

from photosearch import visual_tag_eval as store
from photosearch.visual_tags_derive import FROZEN_TAGS, PERCEIVED_VOCABULARY

API = "/api/eval/visual-tags"


def _load_harness():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "visual_tags_eval.py")
    spec = importlib.util.spec_from_file_location("visual_tags_eval_sharpness_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_harness()


@pytest.fixture(autouse=True)
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "visual-tags"
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(d))
    monkeypatch.delenv("PHOTOSEARCH_DB", raising=False)
    return d


def _lab(yes=(), deb=(), measured=True):
    return {"yes": list(yes), "debatable": list(deb), "done": True, "measured": measured}


# ---------------------------------------------------------------------------
# The measured group
# ---------------------------------------------------------------------------

def test_measured_tags_are_the_frozen_pair_and_outside_every_scored_group():
    assert set(store.MEASURED_TAGS) == {"sharp", "blurry"} == set(FROZEN_TAGS)
    assert not set(store.MEASURED_TAGS) & set(store.CANDIDATE_TAGS)
    assert not set(store.MEASURED_TAGS) & set(PERCEIVED_VOCABULARY)


def test_measured_tags_can_be_labelled_and_imply_measured():
    entry = store.save_label(1, ["blurry"], ["sharp"])
    assert entry["yes"] == ["blurry"] and entry["debatable"] == ["sharp"]
    assert entry["measured"] is True
    # Nothing labelled, not asked: an ordinary visual label stays unmeasured.
    assert store.save_label(2, ["sunny"], [])["measured"] is False
    # Asked and answered "neither" is measured, with no tag at all.
    assert store.save_label(3, [], [], measured=True)["measured"] is True


def test_score_never_scores_a_vlm_on_measured_tags_in_either_direction():
    labels = {1: _lab(["blurry", "sunny"]), 2: _lab(["sharp"]), 3: _lab()}
    predicted = {1: ["sunny"],               # labelled blurry, not predicted
                 2: ["blurry"],              # predicted the wrong measured tag
                 3: ["sharp", "blurry"]}     # predicted measured tags on a "neither"
    res = store.score(predicted, labels)
    assert "sharp" not in res["per_tag"] and "blurry" not in res["per_tag"]
    assert res["overall"] == {**res["overall"], "tp": 1, "fp": 0, "fn": 0}
    # Even if a caller tries to offer them as extra vocabulary.
    res = store.score(predicted, labels, extra_tags=["sharp", "blurry"])
    assert "blurry" not in res["per_tag"]
    assert res["overall"]["fp"] == 0 and res["overall"]["fn"] == 0


def test_harness_report_ignores_measured_tags_for_stored_and_model_variants():
    labels = {1: _lab(["blurry"]), 2: _lab(["sunny"])}
    sample = {"photos": [{"photo_id": 1, "stratum": "a"}, {"photo_id": 2, "stratum": "b"}]}
    rep = H.build_report({"stored": {1: ["blurry"], 2: ["sunny", "sharp"]},
                          "gemma": {1: [], 2: ["sunny"]}}, labels, sample)
    for name in ("stored", "gemma"):
        r = rep["results"][name]
        assert "blurry" not in r["per_tag"] and "sharp" not in r["per_tag"]
        assert r["overall"]["tp"] == 1 and r["overall"]["fp"] == 0 and r["overall"]["fn"] == 0


def test_stored_predictions_drop_measured_tags(tmp_path):
    p = str(tmp_path / "s.db")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, visual_tags TEXT)")
    conn.execute("INSERT INTO photos VALUES (1, ?)", (json.dumps(["blurry", "sunny"]),))
    conn.commit(); conn.close()
    assert H.stored_predictions(H.open_db_readonly(p), [1]) == {1: ["sunny"]}


# ---------------------------------------------------------------------------
# measured_labels + agreement
# ---------------------------------------------------------------------------

def test_measured_labels_skip_visual_labels_that_predate_the_chips():
    store.save_sample([{"photo_id": 1, "stratum": "indoor"},
                       {"photo_id": 2, "stratum": "random"}], seed=1)
    store.save_sample([{"photo_id": 10, "stratum": "stored-blurry"}], seed=1,
                      sample="sharpness")
    store.save_label(1, ["sunny"], [])                            # old: never asked
    store.save_label(2, ["sunny"], [], measured=True)             # asked: neither
    store.save_label(10, ["blurry"], [], label_set="sharpness")
    store.save_label(11, ["blurry"], [], label_set="sharpness", done=False)
    got = store.measured_labels()
    assert set(got) == {2, 10}
    assert got[2] == {"yes": [], "debatable": [], "stratum": "random", "set": "main"}
    assert got[10] == {"yes": ["blurry"], "debatable": [], "stratum": "stored-blurry",
                       "set": "sharpness"}
    with pytest.raises(ValueError):
        store.measured_labels(["sharpness-recheck"])


def test_blurry_self_agreement_on_the_sharpness_recheck():
    for pid, first, second in [(1, ["blurry"], ["blurry"]), (2, [], []),
                               (3, ["blurry"], []), (4, [], ["blurry"])]:
        store.save_label(pid, first, [], label_set="sharpness", measured=True)
        store.save_label(pid, second, [], label_set="sharpness-recheck", measured=True)
    res = store.self_agreement(["sharpness-recheck"])
    c = res["per_tag"]["blurry"]
    assert res["photos"] == 4
    assert (c["n"], c["agree_yes"], c["agree_no"], c["yes_then_no"], c["no_then_yes"]) \
        == (4, 1, 1, 1, 1)
    assert c["kappa"] == pytest.approx(0.0)
    # The visual pair is unaffected, and nothing is pooled unless asked.
    assert store.self_agreement()["photos"] == 0
    assert store.self_agreement(["recheck", "sharpness-recheck"])["photos"] == 4
    with pytest.raises(ValueError):
        store.self_agreement(["main"])


def test_agreement_never_reads_an_unasked_blurry_as_a_no():
    store.save_label(1, ["sunny"], [])                         # before the chips
    store.save_label(1, ["sunny", "blurry"], [], label_set="recheck")
    res = store.self_agreement(["recheck"])
    assert "blurry" not in res["per_tag"]                      # no evidence either way
    assert res["per_tag"]["sunny"]["agree_yes"] == 1


def test_sharpness_recheck_is_its_own_seeded_persisted_draw(eval_dir):
    store.save_sample([{"photo_id": i, "stratum": "x"} for i in range(1, 61)], seed=1)
    store.save_sample([{"photo_id": i, "stratum": "y"} for i in range(100, 160)],
                      seed=1, sample="sharpness")
    visual = store.recheck_ids()
    sharp = store.recheck_ids(sample="sharpness")
    assert len(visual) == store.RECHECK_N and len(sharp) == store.SHARPNESS_RECHECK_N
    assert all(100 <= i < 160 for i in sharp)
    assert (eval_dir / "sharpness" / "recheck.json").exists()
    assert store.recheck_ids(sample="sharpness") == sharp            # persisted
    assert [p["photo_id"] for p in store.set_entries("sharpness-recheck")] == sharp


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

def _make_db(path, n=600):
    """Every stratum reachable; ids by residue so predicates are checkable."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, filepath TEXT, folder TEXT, "
        "visual_tags TEXT, categories TEXT, keywords TEXT, aes_sharpness REAL, "
        "subject_boxes TEXT, date_taken TEXT, camera_make TEXT, camera_model TEXT, "
        "exposure_time TEXT, f_number TEXT, iso INTEGER, image_width INTEGER, "
        "image_height INTEGER)")
    conn.execute("CREATE TABLE faces (id INTEGER PRIMARY KEY, photo_id INTEGER)")
    cams = [("SONY", "ILCE-7RM6"), ("SONY", "ILCE-7M4"), ("Apple", "iPhone 15 Pro"),
            ("GoPro", "HERO11 Black"), ("Google", "Pixel 8"), ("Canon", "EOS R5")]
    for i in range(1, n + 1):
        make, model = cams[i % len(cams)]
        conn.execute(
            "INSERT INTO photos VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, f"2026/d/{i}.{'HEIC' if i % 50 == 0 else 'jpg'}", "2026/d",
             json.dumps(["blurry"] if i % 11 == 0 else ["sunny"]),
             json.dumps(["indoor", "basketball"] if i % 13 == 0 else ["outdoor"]),
             json.dumps([]),
             1.5 if i % 17 == 0 else 7.0,
             "[]" if i % 5 == 0 else json.dumps([{"label": "x", "bbox": [0, 0, 1, 1]}]),
             f"2026-09-01T{'23' if i % 19 == 0 else '12'}:00:00",
             make, model,
             "2" if i % 23 == 0 else "1/500",
             "1.8" if i % 7 == 0 else "8",
             6400 if i % 29 == 0 else 100, 6000, 4000))
        if i % 7 == 0 or i % 3 == 0:
            conn.execute("INSERT INTO faces (photo_id) VALUES (?)", (i,))
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    p = str(tmp_path / "fixture.db")
    _make_db(p)
    return p


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_sharpness_sample_strata_honour_their_predicates(db_path):
    conn = H.open_db_readonly(db_path)
    sample = H.choose_sharpness_sample(conn, n=60, seed=1)
    assert len(sample) == 60 and len({p["photo_id"] for p in sample}) == 60
    by = {}
    for p in sample:
        by.setdefault(p["stratum"], []).append(p["photo_id"])
    for name, _q, _p in H.build_sharpness_strata():
        assert by.get(name), name
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM photos")}
    faces = {r[0] for r in conn.execute("SELECT photo_id FROM faces")}
    assert all("blurry" in json.loads(rows[i]["visual_tags"]) for i in by["stored-blurry"])
    assert all(rows[i]["aes_sharpness"] <= 2 for i in by["aes-sharpness<=2"])
    assert all(rows[i]["exposure_time"] == "2" for i in by["long-exposure"])
    assert all("basketball" in rows[i]["categories"] for i in by["indoor-sports"])
    assert all(rows[i]["iso"] >= 3200 or "T23" in rows[i]["date_taken"]
               for i in by["iso>=3200-or-night"])
    assert all(i in faces and rows[i]["f_number"] == "1.8" for i in by["bokeh-portrait"])
    assert all(rows[i]["subject_boxes"] == "[]" and i not in faces
               for i in by["no-subject-landscape"])
    assert all(rows[i]["camera_model"] == "ILCE-7RM6" for i in by["camera:a7RVI"])
    assert all(rows[i]["camera_model"] == "ILCE-7M4" for i in by["camera:a7IV"])
    assert all(rows[i]["camera_make"] in ("Apple", "Google") for i in by["camera:phone"])
    assert all(rows[i]["camera_make"] == "GoPro" for i in by["camera:gopro"])
    # The loupe cannot draw HEIC, so it is never offered for labelling.
    assert not any(rows[p["photo_id"]]["filepath"].endswith(".HEIC") for p in sample)
    # Oversampled: the likely-blurry strata get the lion's share.
    likely = sum(len(by[s]) for s in ("stored-blurry", "aes-sharpness<=2",
                                      "long-exposure", "indoor-sports"))
    assert likely >= 25


def test_sharpness_sample_is_deterministic_and_excludes_the_visual_sample(db_path):
    conn = H.open_db_readonly(db_path)
    a = H.choose_sharpness_sample(conn, n=60, seed=1)
    assert a == H.choose_sharpness_sample(conn, n=60, seed=1)
    assert a != H.choose_sharpness_sample(conn, n=60, seed=2)
    visual = [p["photo_id"] for p in a[:20]]
    b = H.choose_sharpness_sample(conn, n=60, seed=1, exclude=visual)
    assert not {p["photo_id"] for p in b} & set(visual)


def test_sampler_tolerates_a_db_without_the_newer_columns(tmp_path):
    p = str(tmp_path / "old.db")
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, filepath TEXT)")
    conn.executemany("INSERT INTO photos VALUES (?, ?)", [(i, f"{i}.jpg") for i in range(1, 11)])
    conn.commit(); conn.close()
    sample = H.choose_sharpness_sample(H.open_db_readonly(p), n=60, seed=1)
    assert len(sample) == 10 and {s["stratum"] for s in sample} == {"random"}


def test_sample_sharpness_command_is_read_only_and_leaves_the_visual_eval_alone(
        db_path, eval_dir):
    # An existing visual eval: sample, labels, recheck subset and a cached run.
    store.save_sample([{"photo_id": i, "stratum": "x"} for i in range(1, 61)], seed=1)
    store.save_label(3, ["sunny"], [])
    store.recheck_ids()
    run = eval_dir / "runs" / "production.json"
    run.parent.mkdir(parents=True)
    run.write_text('{"predictions": {}}')
    before = {p: _digest(p) for p in eval_dir.rglob("*.json")}
    db_before = _digest(db_path)

    H.main(["sample-sharpness", "--db", db_path, "--n", "40", "--seed", "3"])

    assert _digest(db_path) == db_before                         # mode=ro
    assert {p: _digest(p) for p in before} == before             # visual eval untouched
    sharp = store.load_sample("sharpness")
    assert sharp["seed"] == 3 and len(sharp["photos"]) == 40
    assert not {p["photo_id"] for p in sharp["photos"]} & set(range(1, 61))
    assert (eval_dir / "sharpness" / "sample.json").exists()
    # Labels are keyed to it: a second draw needs --force.
    with pytest.raises(SystemExit, match="--force"):
        H.main(["sample-sharpness", "--db", db_path])
    H.main(["sample-sharpness", "--db", db_path, "--n", "10", "--force"])
    assert len(store.load_sample("sharpness")["photos"]) == 10


def test_sample_sharpness_refuses_a_missing_db(tmp_path):
    missing = str(tmp_path / "nope.db")
    with pytest.raises(SystemExit):
        H.main(["sample-sharpness", "--db", missing])
    assert not os.path.exists(missing)                           # no stub created


def test_agreement_command_takes_a_set(capsys):
    store.save_label(1, ["blurry"], [], label_set="sharpness")
    store.save_label(1, ["blurry"], [], label_set="sharpness-recheck")
    H.main(["agreement", "--set", "sharpness"])
    assert "blurry" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        H.main(["agreement"])                                     # visual pair: empty


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@pytest.fixture
def both_samples(db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 4")]
    assert len(ids) == 4
    store.save_sample([{"photo_id": ids[0], "stratum": "sports"},
                       {"photo_id": ids[1], "stratum": "random"}], seed=1)
    store.save_sample([{"photo_id": ids[2], "stratum": "stored-blurry"},
                       {"photo_id": ids[3], "stratum": "bokeh-portrait"}], seed=1,
                      sample="sharpness")
    return ids


def test_sharpness_set_serves_its_own_sample_with_only_measured_chips(client, both_samples):
    data = client.get(API + "?set=sharpness").json()
    assert data["set"] == "sharpness"
    assert [p["photo_id"] for p in data["photos"]] == both_samples[2:]
    sections = data["vocabulary"]["sections"]
    assert len(sections) == 1 and sections[0]["measured"]
    assert [t["tag"] for t in sections[0]["tags"]] == ["sharp", "blurry"]
    assert all(t["note"] for t in sections[0]["tags"])            # labeller definitions


def test_main_set_offers_the_measured_chips_too(client, both_samples):
    sections = client.get(API).json()["vocabulary"]["sections"]
    assert [t["tag"] for s in sections if s.get("measured") for t in s["tags"]] \
        == ["sharp", "blurry"]


def test_put_to_the_sharpness_set_writes_only_its_own_file(client, both_samples, eval_dir):
    pid = both_samples[2]
    resp = client.put(f"{API}/{pid}?set=sharpness",
                      json={"yes": ["blurry"], "debatable": [], "measured": True})
    assert resp.status_code == 200
    assert resp.json()["progress"] == {"done": 1, "total": 2}
    assert resp.json()["measured_done"] == 1
    assert store.load_labels("sharpness")[pid]["yes"] == ["blurry"]
    assert store.load_labels("main") == {}
    assert not (eval_dir / "labels.json").exists()
    # A visual-sample photo is not in the sharpness sample, and vice versa.
    assert client.put(f"{API}/{both_samples[0]}?set=sharpness",
                      json={"yes": []}).status_code == 404
    assert client.put(f"{API}/{pid}", json={"yes": []}).status_code == 404


def test_main_set_counts_labels_that_still_need_sharpness(client, both_samples):
    pid = both_samples[0]
    client.put(f"{API}/{pid}", json={"yes": ["sunny"], "debatable": []})    # old client
    data = client.get(API).json()
    assert data["progress"]["done"] == 1 and data["measured_done"] == 0
    client.put(f"{API}/{pid}", json={"yes": ["sunny"], "measured": True})
    assert client.get(API).json()["measured_done"] == 1


def test_sharpness_hint_names_the_sharpness_sampler(client):
    data = client.get(API + "?set=sharpness").json()
    assert data["photos"] == []
    assert "sample-sharpness --db" in data["hint"]


def test_sharpness_recheck_is_blind(client, both_samples, monkeypatch):
    pid = both_samples[2]
    store.save_label(pid, ["blurry"], [], label_set="sharpness")
    monkeypatch.setattr(store, "recheck_ids", lambda *a, **k: [pid])
    data = client.get(API + "?set=sharpness-recheck").json()
    assert [p["photo_id"] for p in data["photos"]] == [pid]
    assert data["photos"][0]["label"] is None
    assert client.put(f"{API}/{both_samples[3]}?set=sharpness-recheck",
                      json={"yes": []}).status_code == 404


def test_page_has_the_loupe_on_the_original():
    page = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "frontend", "dist", "eval_visual_tags.html"),
                encoding="utf-8").read()
    assert "PS.Loupe" in page and "/full'" in page
    assert "'sharpness'" in page and "'sharpness-recheck'" in page

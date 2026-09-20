"""evals/visual_tags_eval.py — sampling, the run cache, and the report math.

No network, no model, no real DB: a tiny sqlite fixture, a stub tagger and a
stub fetch. The harness's numbers decide which prompt and which model ship, so
the arithmetic is pinned against a hand-computed example.
"""

import hashlib
import importlib.util
import json
import os
import sqlite3

import pytest

from photosearch import visual_tag_eval as store


def _load_harness():
    # evals/ is a directory of scripts, not a package.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "visual_tags_eval.py")
    spec = importlib.util.spec_from_file_location("visual_tags_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load_harness()


@pytest.fixture(autouse=True)
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "visual-tags"
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(d))
    # `_pin_model` writes this one; keep a test's pin out of the next test.
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_MODEL", raising=False)
    # `--db` defaults to it; a dev shell that exports it must not leak in.
    monkeypatch.delenv("PHOTOSEARCH_DB", raising=False)
    return d


# ---------------------------------------------------------------------------
# Fixture DB
# ---------------------------------------------------------------------------

def _make_db(path, n=400):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE photos (id INTEGER PRIMARY KEY, filepath TEXT, folder TEXT, "
        "visual_tags TEXT, categories TEXT, country TEXT, exposure_time TEXT, "
        "f_number TEXT, iso INTEGER, image_width INTEGER, image_height INTEGER)")
    tag_cycle = [["peaceful", "sunny"], ["moody"], ["close-up", "centered"],
                 ["wide-angle"], ["symmetrical", "colorful"], ["macro"],
                 ["aerial"], ["reflection"], [], ["sunny", "low-light"]]
    folders = ["2026/2026-08-29_ILCE-7RM6", "2026/2026-09-12_ILCE-7RM6",
               "2026/2026-09-19_ILCE-7RM6", "2019/2019-05-01", "2021/2021-07-04_phone-matt"]
    for i in range(1, n + 1):
        untagged = i % 9 == 0
        night = i % 7 == 0
        conn.execute(
            "INSERT INTO photos VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (i, f"x/{i}.jpg", folders[i % len(folders)],
             None if untagged else json.dumps(tag_cycle[i % len(tag_cycle)]),
             json.dumps([["indoor"], ["landscape", "outdoor"], ["soccer"]][i % 3]),
             [None, "US", "FR"][i % 3],
             # EV100 ~ 1.3 at night (low-light derives); ~14 by day.
             "1/15" if night else "1/1000", "2.0" if night else "8.0",
             3200 if night else 100, 6000, 4000))
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


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def test_sample_is_deterministic_for_a_seed(db_path):
    conn = H.open_db_readonly(db_path)
    a = H.choose_sample(conn, n=60, seed=1)
    b = H.choose_sample(conn, n=60, seed=1)
    c = H.choose_sample(conn, n=60, seed=2)
    assert a == b
    assert [p["photo_id"] for p in a] != [p["photo_id"] for p in c]
    assert len(a) == 60 and len({p["photo_id"] for p in a}) == 60


def test_sample_covers_the_failure_strata_and_honours_their_predicates(db_path):
    conn = H.open_db_readonly(db_path)
    sample = H.choose_sample(conn, n=60, seed=1)
    by = {}
    for p in sample:
        by.setdefault(p["stratum"], []).append(p["photo_id"])
    # The RARE strata come from the prompt's own RARE section.
    for name in ("sports:2026-08-29", "sports:2026-09-12", "sports:2026-09-19",
                 "rare:close-up", "rare:macro", "rare:wide-angle", "rare:aerial",
                 "rare:centered", "rare:symmetrical", "rare:other",
                 "mood:peaceful", "mood:moody", "low-light", "indoor",
                 "landscape", "travel", "empty", "random"):
        assert by.get(name), name
    assert sum(len(v) for v in by.values()) == 60

    rows = {r["id"]: r for r in conn.execute("SELECT * FROM photos")}
    # Never an untagged photo: NULL means "not yet tagged", nothing to score.
    assert all(rows[p["photo_id"]]["visual_tags"] is not None for p in sample)
    assert all(rows[i]["folder"].startswith("2026/2026-09-12") for i in by["sports:2026-09-12"])
    assert all("close-up" in json.loads(rows[i]["visual_tags"]) for i in by["rare:close-up"])
    assert all(rows[i]["visual_tags"] == "[]" for i in by["empty"])
    assert all(rows[i]["country"] == "FR" for i in by["travel"])
    # low-light is decided by EXIF, not by whatever the column claims.
    assert all(rows[i]["iso"] == 3200 for i in by["low-light"])


def test_small_db_yields_what_it_has_without_duplicates(tmp_path):
    p = str(tmp_path / "small.db")
    _make_db(p, n=20)
    sample = H.choose_sample(H.open_db_readonly(p), n=60, seed=1)
    ids = [s["photo_id"] for s in sample]
    assert len(ids) == len(set(ids)) == 18          # 20 minus the 2 untagged


def test_sample_command_writes_via_the_store_and_refuses_to_overwrite(db_path, eval_dir):
    H.main(["sample", "--db", db_path, "--n", "30", "--seed", "4"])
    first = store.load_sample()
    assert first["seed"] == 4 and len(first["photos"]) == 30

    with pytest.raises(SystemExit) as e:
        H.main(["sample", "--db", db_path, "--n", "30", "--seed", "5"])
    assert "--force" in str(e.value)
    assert store.load_sample() == first             # untouched

    H.main(["sample", "--db", db_path, "--n", "30", "--seed", "5", "--force"])
    assert store.load_sample()["seed"] == 5


def test_db_is_opened_read_only_and_never_through_photodb(db_path, monkeypatch):
    import photosearch.db as dbmod

    def forbidden(*a, **kw):
        raise AssertionError("PhotoDB migrates on open — the eval must not use it")

    monkeypatch.setattr(dbmod, "PhotoDB", forbidden)
    before = _digest(db_path)
    H.main(["sample", "--db", db_path])
    store.save_label(store.load_sample()["photos"][0]["photo_id"], [], [])
    H.main(["report", "--db", db_path])
    assert _digest(db_path) == before
    assert not os.path.exists(db_path + "-wal") and not os.path.exists(db_path + "-journal")

    with pytest.raises(sqlite3.OperationalError):
        H.open_db_readonly(db_path).execute("UPDATE photos SET folder = 'x'")


def test_a_mistyped_db_path_is_an_error_not_a_new_stub(tmp_path):
    missing = str(tmp_path / "nope.db")
    with pytest.raises(SystemExit):
        H.open_db_readonly(missing)
    assert not os.path.exists(missing)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _seed_labels(done=(1, 2, 3), undone=(4,)):
    ids = list(done) + list(undone) + [5]            # 5 stays unlabelled
    store.save_sample([{"photo_id": i, "stratum": "random"} for i in ids], seed=1)
    for i in done:
        store.save_label(i, ["sunny"], [])
    for i in undone:
        store.save_label(i, [], [], done=False)


class _Tagger:
    def __init__(self, answers):
        self.answers, self.seen = answers, []

    def __call__(self, path, model):
        assert os.path.exists(path)                  # the pixels were written
        pid = int(os.path.basename(path).split(".")[0])
        self.seen.append(pid)
        ans = self.answers[pid]
        if isinstance(ans, Exception):
            raise ans
        return ans, [{"raw": "stub"}]


def _fetch(server, pid, kind):
    return b"\xff\xd8 not really a jpeg"


def test_run_predicts_only_done_photos_and_keeps_empty_distinct_from_none():
    _seed_labels()
    tagger = _Tagger({1: ["sunny", "moody"], 2: [], 3: None})
    H.run_variant("v1", tagger=tagger, fetch=_fetch, log=lambda *_: None)
    assert tagger.seen == [1, 2, 3]                  # not the un-done 4, not 5

    raw = json.loads(H.run_path("v1").read_text())
    assert raw["predictions"]["2"]["tags"] == []     # the model answered "none"
    assert raw["predictions"]["3"]["tags"] is None   # no usable answer
    assert H.predictions_of(H.load_run("v1")) == {1: ["sunny", "moody"], 2: [], 3: None}
    assert raw["predictions"]["1"]["latency_s"] >= 0
    assert raw["predictions"]["1"]["effective_model"] == raw["effective_model"]

    r = store.score(H.predictions_of(raw))
    assert r["unanswered"] == 1 and r["photos_scored"] == 3


def test_run_is_resumable_and_does_not_cache_failures():
    _seed_labels()
    tagger = _Tagger({1: ["sunny"], 2: H.TransportError("LM Studio is down"), 3: []})
    H.run_variant("v1", tagger=tagger, fetch=_fetch, log=lambda *_: None)
    # A dead backend says nothing about the photo — caching it as unanswered
    # would quietly score as zero recall.
    assert set(H.load_run("v1")["predictions"]) == {"1", "3"}

    again = _Tagger({2: ["sunny"]})
    H.run_variant("v1", tagger=again, fetch=_fetch, log=lambda *_: None)
    assert again.seen == [2]                         # 1 and 3 came from the cache
    assert set(H.load_run("v1")["predictions"]) == {"1", "2", "3"}

    forced = _Tagger({1: [], 2: [], 3: []})
    H.run_variant("v1", tagger=forced, fetch=_fetch, force=True, log=lambda *_: None)
    assert forced.seen == [1, 2, 3]


def test_run_limit_and_a_failed_fetch():
    _seed_labels()
    tagger = _Tagger({1: [], 2: [], 3: []})
    H.run_variant("v1", tagger=tagger, fetch=_fetch, limit=2, log=lambda *_: None)
    assert tagger.seen == [1, 2]

    def bad_fetch(server, pid, kind):
        raise OSError("connection refused")

    H.run_variant("v1", tagger=tagger, fetch=bad_fetch, log=lambda *_: None)
    assert tagger.seen == [1, 2]                     # never reached the tagger
    assert "3" not in H.load_run("v1")["predictions"]


def test_run_cache_is_written_atomically(eval_dir):
    _seed_labels()
    H.run_variant("v1", tagger=_Tagger({1: [], 2: [], 3: []}), fetch=_fetch,
                  log=lambda *_: None)
    assert sorted(os.listdir(eval_dir / "runs")) == ["v1.json"]   # no stray .tmp


def test_a_variant_cannot_silently_mix_two_prompts_or_use_a_reserved_name():
    _seed_labels()
    H.run_variant("v1", tagger=_Tagger({1: [], 2: [], 3: []}), fetch=_fetch,
                  limit=1, log=lambda *_: None)
    with pytest.raises(SystemExit) as e:
        H.run_variant("v1", prompt_text="something else", tagger=_Tagger({}),
                      fetch=_fetch, log=lambda *_: None)
    assert "prompt_sha" in str(e.value)
    for bad in ("stored", "../escape", ""):
        with pytest.raises(SystemExit):
            H.run_variant(bad, tagger=_Tagger({}), fetch=_fetch, log=lambda *_: None)


def test_effective_model_is_recorded_not_the_configured_name(monkeypatch):
    """On the LM Studio route the name passed to the call is ignored, so
    --model must be pinned as the role model or the bake-off scores the wrong
    model under the right name."""
    _seed_labels()
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lmstudio:1234/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen/already-configured")

    run = H.run_variant("default", tagger=_Tagger({1: [], 2: [], 3: []}),
                        fetch=_fetch, log=lambda *_: None)
    assert run["model"] == "llava"
    assert run["effective_model"] == "qwen/already-configured"

    run = H.run_variant("gemma", model="google/gemma-4-e2b",
                        tagger=_Tagger({1: [], 2: [], 3: []}),
                        fetch=_fetch, log=lambda *_: None)
    assert run["effective_model"] == "google/gemma-4-e2b"
    from photosearch import describe
    assert describe.effective_model("llava", "visual") == "google/gemma-4-e2b"


def _jpeg(tmp_path):
    from PIL import Image
    p = str(tmp_path / "p.jpg")
    Image.new("RGB", (64, 48), (10, 120, 200)).save(p, "JPEG")
    return p


def test_prompt_file_reaches_the_real_tagger_and_production_is_restored(tmp_path, monkeypatch):
    from photosearch import describe
    monkeypatch.setattr(describe, "HAS_OLLAMA", True)
    prompts = []

    def fake_chat(model, messages, **kw):
        prompts.append(messages[0]["content"])
        assert kw["role"] == "visual" and messages[0]["images"]
        return "Tags: sunny, low-light, colorful"

    monkeypatch.setattr(describe, "_ollama_chat_with_retry", fake_chat)
    production = describe._build_visual_prompt(H.PERCEIVED_VOCABULARY)

    with H.prompt_override("ALTERNATE PROMPT"):
        tags, calls = H.production_tagger(_jpeg(tmp_path), "llava")
    # The real parser ran: the label was stripped, the derived term dropped.
    assert tags == ["sunny", "colorful"]
    assert prompts == ["ALTERNATE PROMPT"]
    assert calls[0]["raw"].startswith("Tags:") and calls[0]["temperature"] == 0

    assert describe._build_visual_prompt(H.PERCEIVED_VOCABULARY) == production
    assert describe._ollama_chat_with_retry is fake_chat       # recorder removed
    H.production_tagger(_jpeg(tmp_path), "llava")
    assert prompts[-1] == production


def test_production_tagger_tells_a_dead_backend_from_a_useless_answer(tmp_path, monkeypatch):
    from photosearch import describe
    monkeypatch.setattr(describe, "HAS_OLLAMA", True)

    def dead(*a, **kw):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(describe, "_ollama_chat_with_retry", dead)
    with pytest.raises(H.TransportError):
        H.production_tagger(_jpeg(tmp_path), "llava")

    monkeypatch.setattr(describe, "_ollama_chat_with_retry",
                        lambda *a, **kw: "I cannot help with that request.")
    tags, calls = H.production_tagger(_jpeg(tmp_path), "llava")
    assert tags is None and len(calls) == 2          # asked, retried, still prose

    monkeypatch.setattr(describe, "_ollama_chat_with_retry", lambda *a, **kw: "none")
    assert H.production_tagger(_jpeg(tmp_path), "llava")[0] == []


def test_probe_verdicts():
    sent = {}

    def sees(**kw):
        sent.update(kw)
        return "Red."

    assert H.probe("m", chat=sees)["verdict"] == "sees"
    assert sent["role"] == "visual" and sent["options"]["temperature"] == 0
    assert len(sent["messages"][0]["images"]) == 1
    # Accepting the payload is not seeing it.
    assert H.probe("m", chat=lambda **kw: "I see no image.")["verdict"] == "blind"

    def rejects(**kw):
        raise RuntimeError("400 model does not support images")

    res = H.probe("m", chat=rejects)
    assert res["verdict"] == "rejected" and "400" in res["error"]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _report_fixture(tmp_path):
    """Four labelled photos in two strata; photo 5 sampled but un-done.

    variant `a`, by hand:
      1  pred sunny,centered   yes sunny,colorful        tp sunny  fp centered  fn colorful
      2  pred centered,moody   yes moody  deb peaceful   tp moody  fp centered
      3  pred None             yes sunny                 fn sunny  (unanswered)
      4  pred centered,peaceful yes -     deb peaceful   fp centered  debatable peaceful
      => tp 2, fp 3, fn 2, debatable 1
    """
    store.save_sample([{"photo_id": 1, "stratum": "sports"},
                       {"photo_id": 2, "stratum": "sports"},
                       {"photo_id": 3, "stratum": "indoor"},
                       {"photo_id": 4, "stratum": "indoor"},
                       {"photo_id": 5, "stratum": "indoor"}], seed=1)
    store.save_label(1, ["sunny", "colorful"], [])
    store.save_label(2, ["moody"], ["peaceful"])
    store.save_label(3, ["sunny"], [])
    store.save_label(4, [], ["peaceful"])
    store.save_label(5, ["sunny"], [], done=False)
    a = {1: ["sunny", "centered"], 2: ["centered", "moody"], 3: None,
         4: ["centered", "peaceful"], 5: ["macro"]}
    H.save_run("a", {"variant": "a", "effective_model": "m", "prompt_file": None,
                     "predictions": {str(k): {"tags": v} for k, v in a.items()}})

    db = str(tmp_path / "stored.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, visual_tags TEXT)")
    conn.executemany("INSERT INTO photos VALUES (?,?)", [
        (1, '["colorful", "low-light", "sharp", "sunny"]'),   # derived/frozen ignored
        (2, '["peaceful"]'),
        (3, None),                                            # untagged = unanswered
        (4, "[]"),
        (5, '["macro"]')])
    conn.commit()
    conn.close()
    return db


def test_report_math_matches_the_hand_computed_example(tmp_path):
    db = _report_fixture(tmp_path)
    conn = H.open_db_readonly(db)
    labels = store.scoreable_labels()
    stored = H.stored_predictions(conn, labels)
    assert stored == {1: ["colorful", "sunny"], 2: ["peaceful"], 3: None, 4: []}

    rep = H.build_report({"stored": stored, "a": H.predictions_of(H.load_run("a"))})
    assert rep["labelled"] == 4 and rep["sampled"] == 5

    a = rep["results"]["a"]
    assert a["photos_scored"] == 4 and a["unanswered"] == 1
    o = a["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["debatable"]) == (2, 3, 2, 1)
    assert o["precision"] == pytest.approx(2 / 5) and o["recall"] == pytest.approx(2 / 4)
    assert a["avg_tags"] == pytest.approx(6 / 3)     # unanswered photo excluded
    assert a["per_tag"]["centered"]["fp"] == 3
    assert a["per_tag"]["macro"]["fp"] == 0          # photo 5 is not `done`
    s = a["strata"]
    assert (s["sports"]["tp"], s["sports"]["fp"], s["sports"]["fn"]) == (2, 2, 1)
    assert (s["indoor"]["tp"], s["indoor"]["fp"], s["indoor"]["fn"]) == (0, 1, 1)
    assert s["indoor"]["photos_scored"] == 2 and s["indoor"]["unanswered"] == 1

    st = rep["results"]["stored"]
    so = st["overall"]
    # 1: tp sunny, tp colorful. 2: debatable peaceful, fn moody. 3: fn sunny.
    assert (so["tp"], so["fp"], so["fn"], so["debatable"]) == (2, 0, 2, 1)
    assert st["unanswered"] == 1
    assert st["avg_tags"] == pytest.approx(3 / 3)


def test_small_n_ratios_are_withheld():
    assert H.fmt_ratio(0, 0) == "—"
    assert H.fmt_ratio(2, 2) == "n<3 (2/2)"          # never a confident 1.00
    assert H.fmt_ratio(2, 3) == "0.67 (2/3)"


def test_report_tables_sort_by_false_positives_and_apply_the_n_rule(tmp_path):
    _report_fixture(tmp_path)
    rep = H.build_report({"a": H.predictions_of(H.load_run("a"))})
    tables = {title: (head, rows) for title, head, rows in H.report_tables(rep)}

    head, rows = tables["Per tag (sorted by false positives)"]
    assert [r[0] for r in rows] == ["centered", "colorful", "sunny", "moody", "peaceful"]
    by_tag = {r[0]: r for r in rows}
    assert by_tag["centered"][1:] == ["0/3/0/0", "0.00 (0/3)", "—"]
    assert by_tag["sunny"][1:] == ["1/0/1/0", "n<3 (1/1)", "n<3 (1/2)"]
    assert by_tag["peaceful"][1:] == ["0/0/0/1", "—", "—"]
    # Untouched vocabulary is left out rather than padding the table.
    assert "aerial" not in by_tag

    _, overall = tables["Overall"]
    assert overall[0][:6] == ["a", 4, 1, "2.00", "0.40 (2/5)", "0.50 (2/4)"]


def test_report_command_side_by_side_with_stored_and_html(tmp_path, capsys):
    db = _report_fixture(tmp_path)
    out_html = str(tmp_path / "r.html")
    H.main(["report", "--db", db, "--html", out_html])
    text = capsys.readouterr().out
    assert "stored: tp/fp/fn/deb" in text and "a: tp/fp/fn/deb" in text
    assert "=== Per stratum ===" in text and "n<3" in text
    page = open(out_html, encoding="utf-8").read()
    assert "centered" in page and "class=low" in page

    # Without a DB the model runs still report; asking for `stored` by name
    # without one is an error.
    H.main(["report"])
    assert "skipping the `stored`" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        H.main(["report", "--variants", "stored,a"])
    with pytest.raises(SystemExit):
        H.main(["report", "--variants", "missing"])

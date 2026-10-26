"""evals/text_passes_eval.py + the /eval/models text tabs.

The real extractors run on top of a stubbed OpenAI transport, so the
per-attempt hook, the 10 s timeout path and the parser are exercised as they
ship. No network, no model."""

import importlib.util
import json
import os
import sqlite3
import urllib.request

import pytest

from photosearch import describe
from photosearch import model_eval as me


def _load():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "text_passes_eval.py")
    spec = importlib.util.spec_from_file_location("text_passes_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = _load()

SAIL = "A white sailboat crosses a calm bay while two men in life jackets trim the sails."


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "me"))
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    for v in ("PHOTOSEARCH_LLM_TEXT_MODEL", "PHOTOSEARCH_TEXT_LLM_MODEL"):
        monkeypatch.setenv(v, "x")  # recorded, so teardown removes a pin
        monkeypatch.delenv(v)
    monkeypatch.setattr(describe, "HAS_OLLAMA", True)
    monkeypatch.setattr(describe, "_RETRY_DELAY", 0)
    monkeypatch.setattr(me, "lmstudio_loaded", lambda *a, **kw: None)


class _Resp:
    def __init__(self, content):
        self.body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 5}}

    def read(self):
        return json.dumps(self.body).encode()


def _transport(monkeypatch, script):
    q = list(script)

    def fake(req, timeout=None):
        nxt = q.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return _Resp(nxt)
    monkeypatch.setattr(urllib.request, "urlopen", fake)


def test_supported_catches_the_soccer_failure():
    assert X.supported("sailboat", SAIL)
    assert X.supported("sails", SAIL)                     # plural / stem
    assert not X.supported("soccer", SAIL)
    assert not X.supported("adult male", SAIL)
    assert X.supported("adult male", SAIL, lenient=True)  # men -> adult, male
    assert not X.supported("soccer", SAIL, lenient=True)
    assert not X.supported("boat", SAIL) and X.supported("boat", SAIL, lenient=True)


def _inputs(tmp_path):
    db = tmp_path / "p.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, description TEXT, "
                 "categories TEXT, keywords TEXT)")
    conn.executemany("INSERT INTO photos VALUES (?,?,?,?)", [
        (1, SAIL, json.dumps(["boat", "soccer"]), json.dumps(["sailboat"])),
        (2, "A dog runs on a beach.", None, None),
        (3, None, None, None)])
    conn.commit()
    me.save_sample("describe", [{"photo_id": i, "stratum": "random"} for i in (1, 2, 3)])
    return str(db)


def test_freeze_snapshots_stored_and_refuses_to_overwrite(tmp_path):
    db = _inputs(tmp_path)
    data = X.freeze(db=db)
    assert set(data["items"]) == {"1", "2"}               # 3 has no description
    assert data["items"]["1"]["text_sha"] == me.text_sha(SAIL)
    with pytest.raises(SystemExit, match="already frozen"):
        X.freeze(db=db)


def test_run_counts_recovered_timeouts_defers_and_skips_transport(tmp_path, monkeypatch):
    X.freeze(db=_inputs(tmp_path))
    from photosearch import vocab_content
    good = [t for t in ("boat", "sailing", "water", "people") if t in vocab_content.CONTENT_VOCABULARY][:2]
    assert good, "vocabulary changed — pick two real terms"
    _transport(monkeypatch, [
        TimeoutError("timed out"), ", ".join(good + ["not-a-term"]),   # photo 1: recovers
        TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError("timed out"),  # 2
    ])
    run = X.run_variant("category-content", "v", model="gemma-e4b", log=lambda *_: None)
    assert run["effective_model"] == "gemma-e4b"
    one, two = run["items"]["1"], run["items"]["2"]
    assert one["tags"] == good and one["timeouts"] == 1 and one["off_vocab"] == 1
    assert two["deferred"] is True and two["timeouts"] == 3   # a real outcome: cached

    # Connection refused is transport: not cached.
    _transport(monkeypatch, [ConnectionRefusedError("refused")] * 3)
    run = X.run_variant("keywords", "v", model="gemma-e4b", limit=1, log=lambda *_: None)
    assert run["items"] == {}

    rows = X.build_report(db=None)
    row = [r for r in rows if r["pass"] == "category-content"][0]
    assert (row["n"], row["deferred"], row["attempts"], row["timeouts"]) == (2, 1, 5, 4)


def test_label_scoring_splits_describe_faults_from_text_faults(tmp_path, monkeypatch):
    X.freeze(db=_inputs(tmp_path))
    inputs = me.load_inputs()
    run = me.open_run("category-content", "v", effective_model="m",
                      input_source=me.inputs_identity(inputs))
    run["items"]["1"] = {"tags": ["boat", "soccer", "people"], "deferred": False,
                         "text_sha": me.text_sha(SAIL)}
    me.save_run("category-content", "v", run)
    # Owner: boat is right, a flag is missing; soccer and people are wrong.
    from photosearch import vocab_content
    extra = [t for t in vocab_content.CONTENT_VOCABULARY if t not in ("boat", "soccer", "people")][0]
    me.save_category_label(me.label_key(1, me.text_sha(SAIL)), ["boat", extra])
    r = X.summarize("category-content", me.load_run("category-content", "v"), inputs)
    assert (r["labelled"], r["tp"], r["fp"], r["fn"]) == (1, 1, 2, 1)
    # "people" is supported by "men" (lenient) -> describe's fault; soccer is the text model's.
    assert r["fp_from_description"] == 1
    # Strict is whole words: "boat" is not a word of "sailboat". Lenient accepts
    # the compound and "men" -> people, leaving only soccer.
    assert r["unsupported"] == 3 and r["unsupported_lenient"] == 1


def test_category_label_must_be_in_vocabulary():
    with pytest.raises(ValueError):
        me.save_category_label("1:abc", ["definitely not a category"])


def test_keyword_label_api_pool_and_validation(client, db, monkeypatch):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 2")]
    me.save_sample("describe", [{"photo_id": i, "stratum": "random"} for i in ids])
    me.save_inputs("main", "somevariant", "m", {ids[0]: SAIL})
    inputs = me.load_inputs()
    for v, kws in (("secretA", ["sailboat", "bay"]), ("secretB", ["sailboat", "soccer"])):
        run = me.open_run("keywords", v, effective_model=v,
                          input_source=me.inputs_identity(inputs))
        run["items"][str(ids[0])] = {"tags": kws, "deferred": False,
                                     "text_sha": me.text_sha(SAIL)}
        me.save_run("keywords", v, run)
    resp = client.get("/api/eval/models/text/keywords")
    assert "secret" not in resp.text
    photo = resp.json()["photos"][0]
    assert photo["pool"] == ["bay", "sailboat", "soccer"]
    ok = client.put(f"/api/eval/models/text/keywords/{ids[0]}", json={"wrong": ["soccer"]})
    assert ok.status_code == 200 and ok.json()["label"]["judged"] == ["bay", "sailboat", "soccer"]
    assert client.put(f"/api/eval/models/text/keywords/{ids[0]}",
                      json={"wrong": ["nope"]}).status_code == 400
    assert client.put(f"/api/eval/models/text/keywords/{ids[1]}",
                      json={"wrong": []}).status_code == 404
    rows = [r for r in X.build_report() if r["pass"] == "keywords"]
    by = {r["variant"]: r for r in rows}
    assert (by["secretA"]["kw_right"], by["secretA"]["kw_wrong"]) == (2, 0)
    assert (by["secretB"]["kw_right"], by["secretB"]["kw_wrong"]) == (1, 1)


def test_category_api_adds_vocabulary_terms(client, db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 1")]
    me.save_inputs("main", "somevariant", "m", {ids[0]: SAIL})
    data = client.get("/api/eval/models/text/categories").json()
    term = data["vocabulary"][0]
    assert client.put(f"/api/eval/models/text/categories/{ids[0]}",
                      json={"yes": [term]}).status_code == 200
    photo = client.get("/api/eval/models/text/categories").json()["photos"][0]
    assert term in photo["pool"] and photo["label"]["done"]
    # A term the vocabulary lacks is kept as a gap, never scored.
    r = client.put(f"/api/eval/models/text/categories/{ids[0]}",
                   json={"yes": [term, "Sail  Boat"]})
    assert r.status_code == 200
    assert r.json()["label"]["yes"] == [term] and r.json()["label"]["missing"] == ["sail boat"]
    photo = client.get("/api/eval/models/text/categories").json()["photos"][0]
    assert "sail boat" in photo["pool"]
    assert X.vocabulary_gaps() == [("sail boat", 1)]


SOCCER = "Two kids kick a soccer ball across a grass field outdoors."


def _frozen(texts):
    me.save_inputs("main", "somevariant", "m", texts)
    return me.load_inputs()


def _cat_run(variant, inputs, tags_by_pid, created=None, raw=None):
    run = me.open_run("category-content", variant, effective_model=variant,
                      input_source=me.inputs_identity(inputs))
    if created:
        run["created"] = created
    for pid, tags in tags_by_pid.items():
        run["items"][str(pid)] = {"tags": tags, "deferred": tags is None,
                                  "text_sha": inputs["items"][str(pid)]["text_sha"],
                                  **({"raw": raw[pid]} if raw else {})}
    me.save_run("category-content", variant, run)
    return run


def test_unseen_tags_are_unjudged_not_false_positives():
    inputs = _frozen({1: SOCCER})
    key = me.label_key(1, inputs["items"]["1"]["text_sha"])
    early = _cat_run("early", inputs, {1: ["soccer", "outdoor"]}, created="2026-09-01T00:00:00+00:00")
    me.save_category_label(key, ["soccer", "field"])          # no pool: an old-style label
    late = _cat_run("late", inputs, {1: ["soccer", "outdoor", "grass"]},
                    created="2099-01-01T00:00:00+00:00")
    pool_runs = [early, late]
    e = X.summarize("category-content", early, inputs, pool_runs)
    assert (e["tp"], e["fp"], e["fn"], e["unjudged"], e["precision_approx"]) == (1, 1, 1, 0, False)
    # "grass" was never on screen (the late run came after the label): unjudged,
    # and the reconstructed checklist makes the precision approximate.
    lt = X.summarize("category-content", late, inputs, pool_runs)
    assert (lt["tp"], lt["fp"], lt["unjudged"], lt["precision_approx"]) == (1, 1, 1, True)
    # A label that RECORDS its pool is exact.
    me.save_category_label(key, ["soccer", "field"], pool=["soccer", "outdoor", "field"])
    lt = X.summarize("category-content", late, inputs, pool_runs)
    assert (lt["fp"], lt["unjudged"], lt["precision_approx"]) == (1, 1, False)


def test_full_sample_denominator_and_f1():
    inputs = _frozen({1: SOCCER, 2: SAIL})
    run = _cat_run("v", inputs, {1: ["soccer", "outdoor"], 2: None})   # 2 deferred
    for pid, yes in ((1, ["soccer"]), (2, ["boat"])):
        me.save_category_label(me.label_key(pid, inputs["items"][str(pid)]["text_sha"]), yes)
    r = X.summarize("category-content", run, inputs, [run])
    assert (r["labelled"], r["answered_labelled"], r["tp"], r["fp"], r["fn"]) == (2, 1, 1, 1, 1)
    p, rc, f1 = X.category_scores(r["records"])
    assert (p, rc, f1) == (0.5, 0.5, 0.5)
    # Keywords: an unanswered photo scores 0 in the per-photo mean.
    assert X.keyword_photo_precision([{"answered": True, "right": 3, "wrong": 1},
                                      {"answered": False, "right": 0, "wrong": 0},
                                      {"answered": True, "right": 0, "wrong": 0}]) == 0.375


def test_replay_applies_current_grounding_to_cached_raw(monkeypatch):
    def no_model(*a, **kw):
        raise AssertionError("replay must not call a model")
    monkeypatch.setattr(urllib.request, "urlopen", no_model)
    inputs = _frozen({1: SOCCER})
    raw = {1: "soccer, basketball, outdoor, field"}
    run = _cat_run("v", inputs, {1: ["soccer", "basketball", "outdoor", "field"]}, raw=raw)
    real_chat = describe._text_chat_with_retry
    rep = X.replay_item("category-content", SOCCER, run["items"]["1"])
    assert rep["tags"] == ["soccer", "outdoor", "field"]          # basketball contradicted
    assert describe._text_chat_with_retry is real_chat            # stub restored
    assert X.replay_item("category-content", SOCCER, {"tags": None, "deferred": True}) is None
    me.save_category_label(me.label_key(1, inputs["items"]["1"]["text_sha"]),
                           ["soccer", "outdoor", "field"])
    rows = [r for r in X.build_report() if r["pass"] == "category-content"]
    r = rows[0]
    assert r["fp"] == 1 and r["replay"]["fp"] == 0 and r["replay"]["replayed"] == 1
    assert "replayed" in X.render(rows) and "STALE" in X.render(rows)   # no spec hash
    # An answer today's guards reject is unusable on replay, and counts as unanswered.
    rep = X.replay_item("keywords", SOCCER, {"raw": "i can't see any image, please provide one"})
    assert rep["tags"] is None and "refusal" in rep["unusable"]


def test_text_run_records_spec(tmp_path, monkeypatch):
    X.freeze(db=_inputs(tmp_path))
    _transport(monkeypatch, ["sailboat, bay"])
    run = X.run_variant("keywords", "v", model="gemma-e4b", limit=1, log=lambda *_: None)
    assert run["spec_hash"] and not me.spec_status("keywords", run)["stale"]
    monkeypatch.setattr(describe, "_KEYWORDS_MAX_PLAUSIBLE", 99)
    with pytest.raises(SystemExit, match="force-stale"):
        X.run_variant("keywords", "v", model="gemma-e4b", log=lambda *_: None)


def test_category_api_records_the_pool_shown(client, db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 1")]
    inputs = _frozen({ids[0]: SOCCER})
    _cat_run("secretA", inputs, {ids[0]: ["soccer", "basketball"]})
    assert client.put(f"/api/eval/models/text/categories/{ids[0]}",
                      json={"yes": ["soccer", "field"]}).status_code == 200
    lab = me.load_category_labels()[me.label_key(ids[0], inputs["items"][str(ids[0])]["text_sha"])]
    assert set(lab["pool"]) >= {"soccer", "basketball", "field"}

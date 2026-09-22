"""evals/visual_tags_unsplash.py — the Unsplash photographer-keyword recall set.

No network, no model, no real dataset: a tiny fake dataset dir (tsv files) and
a fake thumbs dir in tmp_path, a stub tagger for `run`. The report math is
pinned against a hand-computed example because the numbers decide whether a
prompt change helped recall or just moved it.
"""

import importlib.util
import json
import os

import pytest

from photosearch import visual_tag_eval as store


def _load_module():
    # evals/ is a directory of scripts, not a package.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "visual_tags_unsplash.py")
    spec = importlib.util.spec_from_file_location("visual_tags_unsplash_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_module()

# The fake dataset: (photo_id, keyword, suggested_by_user). Human keywords
# are chosen so the mapping has case, multi-keyword and two-tag cases to bite
# on; AI rows carry a mapped keyword to prove they never count.
_ROWS = [
    ("p1", "Sunny", "t"), ("p1", "beach", "t"), ("p1", "overcast", "f"),
    ("p2", "overcast", "t"), ("p2", "sky", "t"),
    ("p3", "Golden Hour", "t"), ("p3", "golden-hour", "t"),
    ("p4", "monochrome", "t"),                       # -> monochromatic AND black-and-white
    ("p5", "sunny", "t"), ("p5", "calm", "t"),       # sunny + peaceful
    ("p6", "sunny", "f"),                            # AI only: never a positive
    ("p7", "sunny", "t"),                            # no thumb: never sampled
    ("p8", "action", "t"),                           # candidate tag
    ("p9", "overcast", "t"), ("p10", "overcast", "t"), ("p11", "overcast", "t"),
]
_NO_THUMB = {"p7"}


def _write_dataset(d):
    d.mkdir(parents=True, exist_ok=True)
    head = ("photo_id\tkeyword\tai_service_1_confidence\tai_service_2_confidence\t"
            "suggested_by_user\tuser_suggestion_source\tsuggested_by_ai_service_3\t"
            "confirmed_by_ai_service_3\n")
    with open(d / "keywords.tsv000", "w", encoding="utf-8") as f:
        f.write(head)
        for pid, kw, human in _ROWS:
            src = "photographer" if human == "t" else ""
            f.write(f"{pid}\t{kw}\t\t\t{human}\t{src}\t\t\n")
    with open(d / "photos.tsv000", "w", encoding="utf-8") as f:
        f.write("photo_id\tphoto_url\tphoto_image_url\n")
        for pid in sorted({r[0] for r in _ROWS}):
            f.write(f"{pid}\thttps://unsplash.com/photos/{pid}\thttps://images/{pid}\n")


def _write_thumbs(d):
    d.mkdir(parents=True, exist_ok=True)
    for pid in sorted({r[0] for r in _ROWS} - _NO_THUMB):
        (d / f"{pid}.jpg").write_bytes(b"\xff\xd8 not really a jpeg")


@pytest.fixture(autouse=True)
def fake_world(tmp_path, monkeypatch):
    dataset, thumbs = tmp_path / "dataset", tmp_path / "thumbs"
    _write_dataset(dataset)
    _write_thumbs(thumbs)
    monkeypatch.setattr(U, "DATASET", str(dataset))
    monkeypatch.setattr(U, "THUMBS", str(thumbs))
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(tmp_path / "visual-tags"))
    # `_pin_model` writes this one; keep a test's pin out of the next test.
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    return {"dataset": dataset, "thumbs": thumbs}


# ---------------------------------------------------------------------------
# Dataset + mapping
# ---------------------------------------------------------------------------

def test_only_human_keywords_count():
    kws = U.load_human_keywords()
    assert kws["p1"] == ["Sunny", "beach"]          # the AI `overcast` row is gone
    assert "p6" not in kws                           # AI-only photo has no keywords at all
    assert U.photos_by_tag(kws)["sunny"].keys() == {"p1", "p5", "p7"}


def test_mapping_is_case_insensitive_multi_keyword_and_two_tag():
    by_tag = U.photos_by_tag(U.load_human_keywords())
    assert by_tag["sunny"]["p1"] == ["Sunny"]                        # case-insensitive
    assert by_tag["golden-hour"]["p3"] == ["Golden Hour", "golden-hour"]   # both spellings kept
    assert by_tag["peaceful"]["p5"] == ["calm"]                      # a synonym maps
    # `monochrome` is deliberately a positive for both.
    assert "p4" in by_tag["monochromatic"] and "p4" in by_tag["black-and-white"]
    assert "p2" not in by_tag["sunny"]


def test_mapping_only_names_scorable_tags():
    for tag in U.MAPPING:
        assert tag in U.PERCEIVED_VOCABULARY or tag in store.CANDIDATE_TAGS, tag


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def test_sample_is_deterministic_capped_and_skips_photos_without_a_thumb():
    by_tag = U.photos_by_tag(U.load_human_keywords())
    a, counts = U.choose_sample(by_tag, per_tag=2, seed=1)
    b, _ = U.choose_sample(by_tag, per_tag=2, seed=1)
    assert a == b
    assert counts["overcast"] == (4, 2)              # 4 available, capped at 2
    assert counts["sunny"] == (2, 2)                 # p7 has no thumb: not available
    assert "p7" not in a
    assert all(os.path.exists(U.thumb_path(pid)) for pid in a)

    big, counts = U.choose_sample(by_tag, per_tag=40, seed=1)
    assert counts["overcast"] == (4, 4)
    assert big["p5"]["tags"] == ["peaceful", "sunny"]           # union across tags, sorted
    assert big["p5"]["keywords"] == ["calm", "sunny"]
    assert big["p4"]["tags"] == ["black-and-white", "monochromatic"]

    # A different seed draws a different overcast pair (4 choose 2 = 6 ways).
    draws = {tuple(sorted(p for p, e in U.choose_sample(by_tag, per_tag=2, seed=s)[0].items()
                          if "overcast" in e["tags"])) for s in range(1, 12)}
    assert len(draws) > 1


def test_per_tag_rng_means_one_tag_cannot_reshuffle_another():
    by_tag = U.photos_by_tag(U.load_human_keywords())
    before, _ = U.choose_sample(by_tag, per_tag=2, seed=1)
    # Change the overcast pool: the sunny draw must be untouched.
    by_tag["overcast"] = {k: v for k, v in by_tag["overcast"].items() if k != "p2"}
    after, _ = U.choose_sample(by_tag, per_tag=2, seed=1)
    sunny = lambda s: sorted(p for p, e in s.items() if "sunny" in e["tags"])  # noqa: E731
    assert sunny(before) == sunny(after)


def test_sample_command_writes_the_store_and_refuses_to_overwrite(capsys):
    U.main(["sample", "--per-tag", "2", "--seed", "3"])
    first = U.load_sample()
    assert first["seed"] == 3 and first["per_tag"] == 2
    assert first["photos"]["p1"]["human_keywords"] == ["Sunny", "beach"]
    assert "available" in capsys.readouterr().out
    assert sorted(os.listdir(U.unsplash_dir())) == ["sample.json"]   # no stray .tmp

    with pytest.raises(SystemExit) as e:
        U.main(["sample", "--per-tag", "2", "--seed", "4"])
    assert "--force" in str(e.value)
    assert U.load_sample() == first

    U.main(["sample", "--per-tag", "2", "--seed", "4", "--force"])
    assert U.load_sample()["seed"] == 4


def test_the_dataset_and_thumbs_are_never_written(fake_world):
    def snapshot(d):
        return {p.name: p.read_bytes() for p in d.iterdir()}
    ds, th = snapshot(fake_world["dataset"]), snapshot(fake_world["thumbs"])
    U.main(["sample", "--per-tag", "40"])
    U.run_variant("v1", tagger=_Tagger(_all([])), log=lambda *_: None)
    U.main(["sheet", "--tag", "sunny"])
    assert snapshot(fake_world["dataset"]) == ds
    assert snapshot(fake_world["thumbs"]) == th


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class _Tagger:
    def __init__(self, answers):
        self.answers, self.seen = answers, []

    def __call__(self, path, model):
        assert os.path.exists(path)                  # the thumb itself, in place
        pid = os.path.basename(path).rsplit(".", 1)[0]
        self.seen.append(pid)
        ans = self.answers[pid]
        if isinstance(ans, Exception):
            raise ans
        return ans, [{"raw": "stub"}]


def _all(tags):
    return {pid: tags for pid in {r[0] for r in _ROWS}}


def test_run_reads_thumbs_in_place_and_keeps_empty_distinct_from_none():
    U.main(["sample", "--per-tag", "40"])
    answers = _all([])
    answers.update({"p1": ["sunny", "moody"], "p2": [], "p3": None})
    tagger = _Tagger(answers)
    U.run_variant("v1", tagger=tagger, log=lambda *_: None)
    assert "p7" not in tagger.seen and "p6" not in tagger.seen
    assert tagger.seen == sorted(tagger.seen)        # id order, so resumes are predictable

    raw = json.loads(U.run_path("v1").read_text())
    assert raw["predictions"]["p2"]["tags"] == []    # the model answered "none"
    assert raw["predictions"]["p3"]["tags"] is None  # no usable answer
    assert raw["pixels"] == "unsplash-thumb-400px"
    preds = U.predictions_of(U.load_run("v1"))
    assert preds["p1"] == ["sunny", "moody"] and preds["p2"] == [] and preds["p3"] is None
    assert raw["predictions"]["p1"]["effective_model"] == raw["effective_model"]


def test_run_is_resumable_does_not_cache_failures_and_honours_limit():
    U.main(["sample", "--per-tag", "40"])
    answers = _all([])
    answers["p2"] = U.owner.TransportError("LM Studio is down")
    U.run_variant("v1", tagger=_Tagger(answers), log=lambda *_: None)
    cached = set(U.load_run("v1")["predictions"])
    assert "p2" not in cached and "p1" in cached

    again = _Tagger({"p2": ["overcast"]})
    U.run_variant("v1", tagger=again, log=lambda *_: None)
    assert again.seen == ["p2"]                      # everything else came from the cache

    limited = _Tagger(_all([]))
    U.run_variant("v2", tagger=limited, limit=2, log=lambda *_: None)
    assert limited.seen == ["p1", "p10"]
    assert sorted(os.listdir(U.unsplash_dir() / "runs")) == ["v1.json", "v2.json"]

    forced = _Tagger(_all([]))
    U.run_variant("v1", tagger=forced, force=True, log=lambda *_: None)
    assert len(forced.seen) == len(U.load_sample()["photos"])


def test_a_variant_cannot_silently_mix_two_prompts_or_use_a_reserved_name():
    U.main(["sample", "--per-tag", "40"])
    U.run_variant("v1", tagger=_Tagger(_all([])), limit=1, log=lambda *_: None)
    with pytest.raises(SystemExit) as e:
        U.run_variant("v1", prompt_text="something else", tagger=_Tagger({}),
                      log=lambda *_: None)
    assert "prompt_sha" in str(e.value)
    with pytest.raises(SystemExit) as e:
        U.run_variant("v1", prompt_text=None, extra_vocab=["action"],
                      tagger=_Tagger({}), log=lambda *_: None)
    assert "--prompt-file" in str(e.value)
    for bad in ("stored", "../escape", ""):
        with pytest.raises(SystemExit):
            U.run_variant(bad, tagger=_Tagger({}), log=lambda *_: None)


def test_run_without_a_sample_is_an_error():
    with pytest.raises(SystemExit) as e:
        U.run_variant("v1", tagger=_Tagger({}), log=lambda *_: None)
    assert "sample" in str(e.value)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _hand_sample():
    # 4 overcast positives, 2 sunny, 1 monochrome (both tags), 1 action.
    return {"photos": {
        "o1": {"tags": ["overcast"], "keywords": ["overcast"]},
        "o2": {"tags": ["overcast"], "keywords": ["overcast"]},
        "o3": {"tags": ["overcast"], "keywords": ["overcast"]},
        "o4": {"tags": ["overcast"], "keywords": ["overcast"]},
        "s1": {"tags": ["sunny"], "keywords": ["sunny"]},
        "s2": {"tags": ["sunny"], "keywords": ["sunny"]},
        "m1": {"tags": ["black-and-white", "monochromatic"], "keywords": ["monochrome"]},
        "a1": {"tags": ["action"], "keywords": ["action"]},
    }}


def test_report_recall_and_contradiction_match_a_hand_computed_example():
    predicted = {
        "o1": ["overcast"],                 # found
        "o2": ["sunny"],                    # missed AND contradicted
        "o3": ["overcast", "sunny"],        # found AND contradicted (the guard's job, but count it)
        "o4": None,                         # unanswered: a miss, not a contradiction
        "s1": ["sunny", "low-light"],       # found; the derived term is ignored
        "s2": [],                           # the model said none: a miss
        "m1": ["black-and-white", "colorful"],   # found for b&w, contradicted for it too
        "a1": ["action"],
    }
    r = U.score(predicted, _hand_sample())
    o = r["per_tag"]["overcast"]
    assert (o["positives"], o["scored"], o["found"], o["contradiction"], o["unanswered"]) \
        == (4, 4, 2, 2, 1)
    assert o["recall"] == 0.5 and o["contradiction_rate"] == 0.5
    s = r["per_tag"]["sunny"]
    assert (s["found"], s["contradiction"]) == (1, 0) and s["recall"] == 0.5
    bw = r["per_tag"]["black-and-white"]
    assert (bw["found"], bw["contradiction"]) == (1, 1)
    assert r["per_tag"]["monochromatic"]["found"] == 0
    # The candidate was not offered: absent, not scored as a miss.
    assert "action" not in r["per_tag"]
    assert r["photos_scored"] == 8 and r["unanswered"] == 1
    # tags/photo counts only allowed tags over answered photos — low-light
    # (derived) and the unoffered `action` drop out: 1,1,2,1,0,2,0 -> 7/7.
    assert r["avg_tags"] == pytest.approx(1.0)

    offered = U.score(predicted, _hand_sample(), extra_tags=["action"])
    assert offered["per_tag"]["action"]["found"] == 1
    assert offered["avg_tags"] == pytest.approx(8 / 7)

    # A photo the run never reached is not scored at all.
    partial = U.score({"o1": ["overcast"]}, _hand_sample())
    assert partial["per_tag"]["overcast"]["scored"] == 1
    assert partial["per_tag"]["overcast"]["positives"] == 4


def test_report_withholds_ratios_under_min_n_and_states_recall_only(capsys):
    predicted = {pid: ["overcast"] for pid in _hand_sample()["photos"]}
    report = U.build_report({"v": predicted}, _hand_sample())
    text = U.render_text(report)
    assert "RECALL ONLY" in text and "NO PRECISION" in text
    lines = {l.split()[0]: l for l in text.splitlines() if l.strip()}
    assert "1.00 (4/4)" in lines["overcast"]
    assert "n<3 (0/2)" in lines["sunny"]             # 2 positives: withheld
    assert "not offered" in lines["action"]
    # Sorted by positives: overcast (4) before sunny (2).
    assert text.index("\novercast") < text.index("\nsunny")
    assert "RECALL ONLY" in U.render_html(report)


def test_report_command_reads_cached_runs_side_by_side(tmp_path, capsys):
    U.main(["sample", "--per-tag", "40"])
    U.run_variant("a", tagger=_Tagger(_all(["overcast"])), log=lambda *_: None)
    U.run_variant("b", tagger=_Tagger(_all([])), log=lambda *_: None)
    out = str(tmp_path / "r.html")
    U.main(["report", "--html", out])
    text = capsys.readouterr().out
    assert "a: found/scored" in text and "b: found/scored" in text
    assert os.path.exists(out)
    with pytest.raises(SystemExit):
        U.main(["report", "--variants", "nope"])


# ---------------------------------------------------------------------------
# sheet
# ---------------------------------------------------------------------------

def test_sheet_writes_html_naming_the_tag_with_keywords_and_predictions(tmp_path):
    U.main(["sample", "--per-tag", "40"])
    answers = _all([])
    answers.update({"p2": ["sunny"], "p11": ["overcast"], "p10": None})
    U.run_variant("v1", tagger=_Tagger(answers), log=lambda *_: None)
    out = str(tmp_path / "sheet.html")
    U.main(["sheet", "--tag", "overcast", "--n", "3", "--out", out])
    doc = open(out, encoding="utf-8").read()
    assert "<code>overcast</code>" in doc
    # --n honoured: 4 positives, the first 3 in id order (p10, p11, p2).
    assert doc.count("<div class=card>") == 3 and "p9.jpg" not in doc
    assert "p10.jpg" in doc and "UNANSWERED" in doc
    assert 'class=bad>sunny' in doc                    # the contradiction, in red
    assert 'class=hit>overcast' in doc
    assert "https://unsplash.com/photos/p2" in doc     # linked to the source page
    assert "<b>overcast</b>" in doc and "sky" in doc   # matching keyword bold, others listed

    # Default location, and an unknown / empty tag is an error.
    U.main(["sheet", "--tag", "sunny"])
    assert (U.unsplash_dir() / "sheet-sunny.html").exists()
    with pytest.raises(SystemExit):
        U.main(["sheet", "--tag", "not-a-tag"])
    with pytest.raises(SystemExit):
        U.main(["sheet", "--tag", "harsh-light"])      # nobody in the fake set

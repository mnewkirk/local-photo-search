"""Storage + scoring for the visual-tag labelled eval.

`photosearch/visual_tag_eval.py` is the contract between the labelling page
and `evals/visual_tags_eval.py`, so these pin the file formats and — the part
that decides every number the eval prints — the THREE-state label semantics:
a debatable tag is neither a hit nor a miss, and a photo that is not `done`
has no opinion at all.
"""

import json
import os

import pytest

from photosearch import visual_tag_eval as E


@pytest.fixture(autouse=True)
def eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "visual-tags"
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(d))
    return d


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def test_eval_dir_follows_the_env_var(eval_dir):
    assert E.eval_dir() == eval_dir


def test_missing_files_read_as_empty(eval_dir):
    assert not eval_dir.exists()
    assert E.load_sample() == {"created": None, "seed": None, "photos": []}
    assert E.load_labels() == {}
    assert E.scoreable_labels() == {}
    # Reading must not create anything — a mistyped dir should stay absent.
    assert not eval_dir.exists()


def test_sample_round_trip_and_format(eval_dir):
    E.save_sample([{"photo_id": "7", "stratum": "empty", "extra": "dropped"},
                   {"photo_id": 3, "stratum": "random"}], seed=5)
    on_disk = json.loads((eval_dir / "sample.json").read_text())
    assert on_disk["seed"] == 5 and on_disk["created"]
    # Normalised to exactly the two contract keys, order preserved.
    assert on_disk["photos"] == [{"photo_id": 7, "stratum": "empty"},
                                 {"photo_id": 3, "stratum": "random"}]
    assert E.load_sample() == on_disk


def test_label_round_trip_sorts_dedupes_and_keys_by_int(eval_dir):
    entry = E.save_label(12, ["sunny", "colorful", "sunny"], ["moody"])
    assert entry["yes"] == ["colorful", "sunny"]
    assert entry["debatable"] == ["moody"]
    assert entry["done"] is True and entry["updated_at"]
    assert E.load_labels() == {12: entry}
    # JSON keys are strings on disk; the API hands back ints.
    assert list(json.loads((eval_dir / "labels.json").read_text())["labels"]) == ["12"]


def test_saving_one_label_keeps_the_others_and_replaces_its_own():
    E.save_label(1, ["sunny"], [])
    E.save_label(2, ["moody"], [])
    E.save_label(1, [], ["peaceful"])
    labels = E.load_labels()
    assert labels[1]["yes"] == [] and labels[1]["debatable"] == ["peaceful"]
    assert labels[2]["yes"] == ["moody"]


@pytest.mark.parametrize("tag", ["long-exposure", "low-light", "panoramic",  # derived
                                 "sharp", "blurry",                          # frozen
                                 "motion-blur",                              # retired
                                 "not-a-tag"])
def test_non_perceived_tags_are_rejected_in_either_list(tag):
    with pytest.raises(ValueError, match="not perceived"):
        E.save_label(1, [tag], [])
    with pytest.raises(ValueError, match="not perceived"):
        E.save_label(1, [], [tag])
    assert E.load_labels() == {}          # nothing half-written


def test_a_tag_cannot_be_both_yes_and_debatable():
    with pytest.raises(ValueError, match="both"):
        E.save_label(1, ["sunny", "moody"], ["moody"])
    assert E.load_labels() == {}


def test_scoreable_labels_are_only_the_done_ones():
    E.save_label(1, ["sunny"], [], done=True)
    E.save_label(2, ["sunny"], [], done=False)
    assert set(E.load_labels()) == {1, 2}
    assert set(E.scoreable_labels()) == {1}


def test_failed_write_leaves_the_old_labels_and_no_temp_file(eval_dir, monkeypatch):
    E.save_label(1, ["sunny"], [])
    before = (eval_dir / "labels.json").read_text()

    def boom(*a, **kw):
        raise OSError("disk full")

    # Die mid-dump: the hand labels already on disk are the thing that cannot
    # be regenerated, so the old file must survive byte-for-byte.
    monkeypatch.setattr(E.json, "dump", boom)
    with pytest.raises(OSError):
        E.save_label(2, ["moody"], [])
    assert (eval_dir / "labels.json").read_text() == before
    assert sorted(os.listdir(eval_dir)) == ["labels.json"]


def test_write_goes_through_a_rename_in_the_same_directory(eval_dir, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(E.os, "replace",
                        lambda src, dst: (seen.append((str(src), str(dst))), real(src, dst)))
    E.save_label(1, ["sunny"], [])
    (src, dst), = seen
    assert dst == str(eval_dir / "labels.json")
    # Same directory = same filesystem, which is what makes the rename atomic.
    assert os.path.dirname(src) == str(eval_dir)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def _lab(yes=(), debatable=(), done=True):
    return {"yes": list(yes), "debatable": list(debatable), "done": done}


def test_score_hand_computed():
    labels = {1: _lab(["sunny", "colorful"]),
              2: _lab(["moody"], ["peaceful"]),
              3: _lab([])}
    predicted = {1: ["sunny", "centered"],       # tp sunny, fp centered, fn colorful
                 2: ["peaceful", "moody"],       # debatable peaceful, tp moody
                 3: ["centered"]}                # fp centered
    r = E.score(predicted, labels)
    assert r["photos_scored"] == 3 and r["unanswered"] == 0
    assert r["per_tag"]["sunny"] == {"tp": 1, "fp": 0, "fn": 0, "debatable": 0,
                                     "precision": 1.0, "recall": 1.0}
    assert r["per_tag"]["centered"]["fp"] == 2
    assert r["per_tag"]["centered"]["precision"] == 0.0
    assert r["per_tag"]["centered"]["recall"] is None      # never labelled yes
    assert r["per_tag"]["colorful"]["fn"] == 1
    assert r["per_tag"]["colorful"]["precision"] is None   # never predicted
    o = r["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["debatable"]) == (2, 2, 1, 1)
    assert o["precision"] == pytest.approx(2 / 4)
    assert o["recall"] == pytest.approx(2 / 3)


def test_debatable_is_neither_hit_nor_miss_in_either_direction():
    labels = {1: _lab([], ["moody"]), 2: _lab([], ["moody"])}
    r = E.score({1: ["moody"], 2: []}, labels)
    c = r["per_tag"]["moody"]
    # Predicting it is not a tp or fp; NOT predicting it is not a fn.
    assert (c["tp"], c["fp"], c["fn"], c["debatable"]) == (0, 0, 0, 1)
    assert c["precision"] is None and c["recall"] is None


def test_undone_photos_are_not_scored(eval_dir):
    E.save_label(1, ["sunny"], [], done=True)
    E.save_label(2, [], [], done=False)
    # Photo 2's absent tags are NOT a "no" yet, so `centered` on it is no fp.
    r = E.score({1: ["sunny"], 2: ["centered"]})
    assert r["photos_scored"] == 1
    assert r["per_tag"]["centered"]["fp"] == 0
    assert r["overall"]["tp"] == 1


def test_none_prediction_is_unanswered_and_misses_every_yes_tag():
    labels = {1: _lab(["sunny", "moody"]), 2: _lab(["sunny"])}
    r = E.score({1: None, 2: []}, labels)
    assert r["photos_scored"] == 2
    # None = no usable answer; [] = the model answered "none". Both miss the
    # yes tags, but only one is counted unanswered.
    assert r["unanswered"] == 1
    assert r["per_tag"]["sunny"]["fn"] == 2
    assert r["per_tag"]["moody"]["fn"] == 1
    assert r["overall"]["recall"] == 0.0


def test_photos_without_a_prediction_are_skipped_not_counted_as_misses():
    labels = {1: _lab(["sunny"]), 2: _lab(["sunny"])}
    r = E.score({1: ["sunny"]}, labels)
    assert r["photos_scored"] == 1
    assert r["per_tag"]["sunny"] == {"tp": 1, "fp": 0, "fn": 0, "debatable": 0,
                                     "precision": 1.0, "recall": 1.0}


def test_non_perceived_predicted_tags_are_ignored():
    labels = {1: _lab(["sunny"])}
    r = E.score({1: ["sunny", "low-light", "sharp", "motion-blur", "bogus"]}, labels)
    assert r["overall"]["fp"] == 0 and r["overall"]["tp"] == 1
    assert "low-light" not in r["per_tag"] and "bogus" not in r["per_tag"]


def test_per_tag_covers_exactly_the_perceived_vocabulary():
    from photosearch.visual_tags_derive import PERCEIVED_VOCABULARY
    r = E.score({}, {})
    assert set(r["per_tag"]) == set(PERCEIVED_VOCABULARY)
    assert r["overall"]["precision"] is None and r["overall"]["recall"] is None


def test_duplicate_predicted_tags_count_once():
    r = E.score({1: ["sunny", "sunny"]}, {1: _lab(["sunny"])})
    assert r["per_tag"]["sunny"]["tp"] == 1


# --- candidate tags: labelled now, scored only for variants that offered them


def _cand_env(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(tmp_path))
    from photosearch import visual_tag_eval as v
    return v


def test_candidate_label_round_trips(monkeypatch, tmp_path):
    v = _cand_env(monkeypatch, tmp_path)
    v.save_label(1, ["action", "sunny"], [])
    assert v.load_labels()[1]["yes"] == ["action", "sunny"]


def test_candidate_is_invisible_to_a_variant_that_never_offered_it(monkeypatch, tmp_path):
    # Production never mentions `action`; a labelled `action` must not become
    # a miss against it, and a stray prediction must not become a hit.
    v = _cand_env(monkeypatch, tmp_path)
    v.save_label(1, ["action", "sunny"], [])
    res = v.score({1: ["sunny", "action"]})
    assert "action" not in res["per_tag"]
    assert res["overall"] == {**res["overall"], "tp": 1, "fp": 0, "fn": 0}


def test_candidate_is_scored_for_a_variant_that_offered_it(monkeypatch, tmp_path):
    v = _cand_env(monkeypatch, tmp_path)
    v.save_label(1, ["action"], [])
    v.save_label(2, [], [])
    v.save_label(3, [], ["action"])
    res = v.score({1: [], 2: ["action"], 3: ["action"]}, extra_tags=["action"])
    c = res["per_tag"]["action"]
    assert (c["tp"], c["fp"], c["fn"], c["debatable"]) == (0, 1, 1, 1)


def test_extra_tags_outside_the_candidate_list_are_ignored(monkeypatch, tmp_path):
    v = _cand_env(monkeypatch, tmp_path)
    v.save_label(1, [], [])
    assert "sharp" not in v.score({1: ["sharp"]}, extra_tags=["sharp"])["per_tag"]

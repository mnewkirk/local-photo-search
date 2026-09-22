"""Tests for photosearch/eval_api.py — the API behind `/eval/visual-tags`.

Labels live in files under `visual_tag_eval.eval_dir()`, never the DB, so every
test points PHOTOSEARCH_VISUAL_EVAL_DIR at tmp_path: nothing here may touch the
owner's real `evals/visual-tags/` labels.
"""

import json

import pytest

from photosearch import visual_tag_eval
from photosearch.visual_tags_derive import (
    CAPTURE_FACT_TAGS, FROZEN_TAGS, PERCEIVED_GLOSS, PERCEIVED_VOCABULARY,
    RETIRED_TAGS,
)

API = "/api/eval/visual-tags"


@pytest.fixture(autouse=True)
def _eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "visual-eval"
    monkeypatch.setenv("PHOTOSEARCH_VISUAL_EVAL_DIR", str(d))
    return d


@pytest.fixture
def sample(db):
    """Three real photo ids from the shared fixture DB, one with stored tags."""
    ids = [r[0] for r in db.conn.execute(
        "SELECT id FROM photos ORDER BY id LIMIT 3").fetchall()]
    assert len(ids) == 3
    db.conn.execute("UPDATE photos SET visual_tags = ? WHERE id = ?",
                    (json.dumps(["peaceful", "sunny"]), ids[0]))
    db.conn.execute("UPDATE photos SET visual_tags = NULL WHERE id = ?", (ids[1],))
    db.conn.commit()
    visual_tag_eval.save_sample(
        [{"photo_id": ids[0], "stratum": "sports"},
         {"photo_id": ids[1], "stratum": "landscape"},
         {"photo_id": ids[2], "stratum": "random"}], seed=7)
    return ids


def test_empty_sample_returns_hint(client):
    resp = client.get(API)
    assert resp.status_code == 200
    data = resp.json()
    assert data["photos"] == []
    assert data["progress"] == {"done": 0, "total": 0}
    assert "python evals/visual_tags_eval.py sample --db" in data["hint"]


def test_get_shape(client, sample):
    data = client.get(API).json()
    assert data["hint"] is None
    assert data["sample"]["seed"] == 7
    assert [p["photo_id"] for p in data["photos"]] == sample
    assert data["progress"] == {"done": 0, "total": 3}

    first, second = data["photos"][0], data["photos"][1]
    assert first["stratum"] == "sports"
    assert first["label"] is None
    assert first["stored_tags"] == ["peaceful", "sunny"]
    assert first["in_db"] is True
    # NULL column = "not tagged", which is not the same answer as [].
    assert second["stored_tags"] is None


def test_vocabulary_is_exactly_the_perceived_terms(client):
    sections = client.get(API).json()["vocabulary"]["sections"]
    assert len(sections) >= 2 and all(s["title"] for s in sections)
    # Shipped sections are exactly the perceived vocabulary; trial tags sit in
    # their own flagged section so the two can never blur together.
    from photosearch.visual_tag_eval import CANDIDATE_TAGS
    trial = [t["tag"] for s in sections if s.get("candidate") for t in s["tags"]]
    assert set(trial) == set(CANDIDATE_TAGS)
    assert all(t["gloss"] is None for s in sections if s.get("candidate")
               for t in s["tags"])   # a candidate has no model-facing text
    tags = [t["tag"] for s in sections if not s.get("candidate") for t in s["tags"]]
    assert len(tags) == len(set(tags))
    assert set(tags) == set(PERCEIVED_VOCABULARY)
    # Capture facts are EXIF's call; frozen/retired terms are never asked.
    assert not set(tags) & (set(CAPTURE_FACT_TAGS) | FROZEN_TAGS | RETIRED_TAGS)
    # The labeller must read the same definition the model is given.
    gloss = {t["tag"]: t["gloss"] for s in sections for t in s["tags"]}
    for tag, text in PERCEIVED_GLOSS.items():
        assert gloss[tag] == text


def test_photo_missing_from_db_is_still_listed(client, sample):
    visual_tag_eval.save_sample(
        [{"photo_id": 987654, "stratum": "random"}], seed=1)
    photo = client.get(API).json()["photos"][0]
    assert photo["photo_id"] == 987654
    assert photo["in_db"] is False and photo["stored_tags"] is None


def test_put_round_trip(client, sample):
    pid = sample[0]
    resp = client.put(f"{API}/{pid}",
                      json={"yes": ["sunny"], "debatable": ["moody"], "done": True})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"]["yes"] == ["sunny"]
    assert body["label"]["debatable"] == ["moody"]
    assert body["progress"] == {"done": 1, "total": 3}

    data = client.get(API).json()
    label = data["photos"][0]["label"]
    assert (label["yes"], label["debatable"], label["done"]) == (["sunny"], ["moody"], True)
    assert data["progress"] == {"done": 1, "total": 3}
    assert visual_tag_eval.scoreable_labels()[pid]["yes"] == ["sunny"]


def test_resave_replaces_and_draft_is_not_done(client, sample):
    pid = sample[0]
    client.put(f"{API}/{pid}", json={"yes": ["sunny"], "debatable": [], "done": True})
    resp = client.put(f"{API}/{pid}", json={"yes": ["foggy"], "debatable": [], "done": False})
    assert resp.status_code == 200
    assert resp.json()["progress"]["done"] == 0
    assert visual_tag_eval.load_labels()[pid]["yes"] == ["foggy"]
    assert pid not in visual_tag_eval.scoreable_labels()


def test_empty_label_is_a_valid_done_answer(client, sample):
    pid = sample[2]
    resp = client.put(f"{API}/{pid}", json={"yes": []})
    assert resp.status_code == 200
    assert resp.json()["label"] == {**resp.json()["label"],
                                    "yes": [], "debatable": [], "done": True}
    assert visual_tag_eval.scoreable_labels()[pid]["yes"] == []


@pytest.mark.parametrize("tag", ["long-exposure", "sharp", "motion-blur", "not-a-tag"])
def test_non_perceived_tag_is_400(client, sample, tag):
    resp = client.put(f"{API}/{sample[0]}", json={"yes": [tag], "debatable": []})
    assert resp.status_code == 400
    assert tag in resp.json()["detail"]
    assert visual_tag_eval.load_labels() == {}


def test_tag_in_both_lists_is_400(client, sample):
    resp = client.put(f"{API}/{sample[0]}",
                      json={"yes": ["sunny"], "debatable": ["sunny"]})
    assert resp.status_code == 400
    assert "both" in resp.json()["detail"]
    assert visual_tag_eval.load_labels() == {}


def test_photo_outside_sample_is_404(client, sample, db):
    other = db.conn.execute(
        "SELECT id FROM photos WHERE id NOT IN (?,?,?) LIMIT 1", sample).fetchone()
    pid = other[0] if other else 987654
    resp = client.put(f"{API}/{pid}", json={"yes": [], "debatable": []})
    assert resp.status_code == 404
    assert visual_tag_eval.load_labels() == {}


def test_never_proxies_to_the_nas_in_replica_mode(client, sample, monkeypatch):
    """Labels are local files; an unreachable NAS must not matter."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://127.0.0.1:9")
    assert client.get(API).status_code == 200
    assert client.put(f"{API}/{sample[0]}", json={"yes": ["sunny"]}).status_code == 200


def test_page_route_serves_html(client):
    resp = client.get("/eval/visual-tags")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "/api/eval/visual-tags" in resp.text


def test_labeller_notes_never_leak_into_the_production_prompt():
    # The notes define terms for the human only. Moving one into
    # PERCEIVED_GLOSS changes the shipped prompt, which needs an A/B first.
    from photosearch import eval_api
    from photosearch.visual_tags_derive import PERCEIVED_GLOSS, PERCEIVED_VOCABULARY
    from photosearch.describe import _build_visual_prompt
    prompt = _build_visual_prompt(list(PERCEIVED_VOCABULARY))
    for tag, note in eval_api.LABELLER_NOTES.items():
        assert tag in PERCEIVED_VOCABULARY
        assert note not in prompt
        assert PERCEIVED_GLOSS.get(tag) != note


def test_recheck_set_withholds_the_first_answer_and_writes_its_own_file(client, sample, monkeypatch):
    from photosearch import visual_tag_eval as v
    ids = [p["photo_id"] for p in v.load_sample()["photos"]]
    v.save_label(ids[0], ["sunny"], [])                      # first answer, main set
    monkeypatch.setattr(v, "recheck_ids", lambda *a, **k: [ids[0]])
    r = client.get(API + "?set=recheck").json()
    assert r["set"] == "recheck" and [p["photo_id"] for p in r["photos"]] == [ids[0]]
    assert r["photos"][0]["label"] is None                   # blind
    assert client.put(f"{API}/{ids[0]}?set=recheck", json={"yes": [], "debatable": [], "done": True}).status_code == 200
    assert v.load_labels("main")[ids[0]]["yes"] == ["sunny"]  # untouched
    assert v.load_labels("recheck")[ids[0]]["yes"] == []
    assert client.get(API + "?set=bogus").status_code == 400
    # a photo outside the recheck subset is refused for that set
    if len(ids) > 1:
        assert client.put(f"{API}/{ids[1]}?set=recheck", json={"yes": [], "debatable": []}).status_code == 404

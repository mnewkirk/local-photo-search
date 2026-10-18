"""Tests for photosearch/category_grounding_eval_api.py — the API behind
`/eval/category-grounding`.

Labels live in the holdout sample FILE under
`category_grounding_eval_api.eval_dir()`, never the DB, so every test points
PHOTOSEARCH_CATEGORY_EVAL_DIR at tmp_path: nothing here may touch the owner's
real `evals/category-grounding/` labels.
"""

import json

import pytest

from photosearch import category_grounding_eval_api as api

API = "/api/eval/category-grounding"


@pytest.fixture(autouse=True)
def _eval_dir(tmp_path, monkeypatch):
    d = tmp_path / "category-grounding-eval"
    monkeypatch.setenv("PHOTOSEARCH_CATEGORY_EVAL_DIR", str(d))
    return d


def _write_sample(eval_dir_path, groups: dict, seed=11, note="test sample"):
    eval_dir_path.mkdir(parents=True, exist_ok=True)
    path = eval_dir_path / api.HOLDOUT_FILE
    path.write_text(json.dumps({"_note": note, "seed": seed, "groups": groups}))
    return path


@pytest.fixture
def sample(_eval_dir, db):
    """One drop per group, using real photo ids from the shared fixture DB.
    DSC04922 (`categories` holds ["landscape", "people"]) carries the "people"
    drop so a non-empty, real stored-categories list is exercised."""
    ids = db._test_photo_ids
    pid_922 = ids["DSC04922.JPG"]
    pid_894 = ids["DSC04894.JPG"]
    groups = {
        "sport": [
            {"photo_id": pid_922, "term": "people",
             "reason": "sport: soccer cued, basketball not",
             "evidence": "[soccer] Two boys play soccer.", "label": None},
        ],
        "setting": [
            {"photo_id": pid_894, "term": "landscape",
             "reason": "setting: indoor cued, outdoor not",
             "evidence": "[indoor] Inside a kitchen.", "label": None},
        ],
        "light": [], "weather": [],
    }
    _write_sample(_eval_dir, groups)
    return {"pid_922": pid_922, "pid_894": pid_894}


def test_empty_sample_returns_hint(client):
    resp = client.get(f"{API}/items")
    assert resp.status_code == 200
    data = resp.json()
    assert all(data["groups"][g] == [] for g in api.GROUPS)
    assert data["progress"] == {"done": 0, "total": 0}
    assert "python evals/category_grounding_eval.py" in data["hint"]


def test_items_merges_current_description_and_categories(client, sample, db):
    data = client.get(f"{API}/items").json()
    assert data["hint"] is None
    assert data["seed"] == 11

    sport = data["groups"]["sport"]
    assert len(sport) == 1
    item = sport[0]
    assert item["photo_id"] == sample["pid_922"]
    assert item["term"] == "people"
    assert item["group"] == "sport"
    assert item["label"] is None
    assert item["in_db"] is True
    # Pulled live from the DB, not frozen in the sample file.
    assert item["description"] == "Sunset over the Pacific with silhouettes of two people on the cliff."
    assert item["categories"] == ["landscape", "people"]

    setting = data["groups"]["setting"]
    assert setting[0]["photo_id"] == sample["pid_894"]
    assert setting[0]["categories"] is None   # DSC04894 has no stored categories

    assert data["group_rules"].keys() == set(api.GROUPS)
    assert all(data["group_rules"][g] for g in api.GROUPS)
    assert data["progress"] == {"done": 0, "total": 2}


def test_photo_missing_from_db_is_still_listed(client, _eval_dir):
    _write_sample(_eval_dir, {"sport": [{"photo_id": 987654, "term": "basketball",
                                         "reason": "r", "evidence": "e", "label": None}],
                              "setting": [], "light": [], "weather": []})
    item = client.get(f"{API}/items").json()["groups"]["sport"][0]
    assert item["photo_id"] == 987654
    assert item["in_db"] is False
    assert item["description"] is None and item["categories"] is None


def test_label_round_trips_to_the_file(client, sample, _eval_dir):
    resp = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": "correct"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == {"photo_id": sample["pid_922"], "term": "people", "group": "sport",
                    "label": "correct", "progress": {"done": 1, "total": 2}}

    on_disk = json.loads((_eval_dir / api.HOLDOUT_FILE).read_text())
    assert on_disk["groups"]["sport"][0]["label"] == "correct"
    # Nothing else in the row was rewritten.
    assert on_disk["groups"]["sport"][0]["reason"] == "sport: soccer cued, basketball not"

    data = client.get(f"{API}/items").json()
    assert data["groups"]["sport"][0]["label"] == "correct"
    assert data["progress"] == {"done": 1, "total": 2}


def test_relabel_overwrites_and_null_clears(client, sample):
    client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": "wrong"})
    r2 = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": "debatable"})
    assert r2.json()["label"] == "debatable"

    r3 = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": None})
    assert r3.status_code == 200
    assert r3.json()["label"] is None
    assert r3.json()["progress"]["done"] == 0


def test_write_is_atomic_no_tmp_file_left_behind(client, sample, _eval_dir):
    client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": "correct"})
    names = [p.name for p in _eval_dir.iterdir()]
    assert names == [api.HOLDOUT_FILE]   # no .tmp leftovers


@pytest.mark.parametrize("label", ["yes", "no", "CORRECT", ""])
def test_bad_label_is_400(client, sample, label):
    resp = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": label})
    assert resp.status_code == 400


def test_bad_group_is_400(client, sample):
    resp = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "nonsense", "label": "correct"})
    assert resp.status_code == 400


def test_unknown_drop_is_404(client, sample):
    # Right group, wrong term — not the same drop.
    resp = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "not-a-term", "group": "sport", "label": "correct"})
    assert resp.status_code == 404
    # Right term, wrong group.
    resp2 = client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "weather", "label": "correct"})
    assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# Score math
# ---------------------------------------------------------------------------

def _group_of(n_correct, n_wrong, n_debatable, n_unlabelled):
    items = []
    pid = 1
    for _ in range(n_correct):
        items.append({"photo_id": pid, "term": "t", "label": "correct"}); pid += 1
    for _ in range(n_wrong):
        items.append({"photo_id": pid, "term": "t", "label": "wrong"}); pid += 1
    for _ in range(n_debatable):
        items.append({"photo_id": pid, "term": "t", "label": "debatable"}); pid += 1
    for _ in range(n_unlabelled):
        items.append({"photo_id": pid, "term": "t", "label": None}); pid += 1
    return items


def test_score_precision_excludes_debatable_and_unlabelled():
    data = {"groups": {"sport": _group_of(18, 2, 5, 0),
                       "setting": [], "light": [], "weather": []}}
    score = api.compute_score(data)
    sport = score["per_group"]["sport"]
    assert sport == {"n": 25, "labelled": 25, "unlabelled": 0, "correct": 18, "wrong": 2,
                     "debatable": 5, "precision": 0.9, "passes": True}


def test_score_floor_requires_20_judged():
    # 90% precision but only 10 judged (9 correct, 1 wrong) — below the floor.
    data = {"groups": {"sport": _group_of(9, 1, 0, 15),
                       "setting": [], "light": [], "weather": []}}
    score = api.compute_score(data)
    sport = score["per_group"]["sport"]
    assert sport["precision"] == 0.9
    assert sport["passes"] is False       # judged == 10 < MIN_JUDGED


def test_score_below_precision_bar_fails_even_with_enough_judged():
    data = {"groups": {"sport": _group_of(17, 3, 0, 5),
                       "setting": [], "light": [], "weather": []}}
    score = api.compute_score(data)
    sport = score["per_group"]["sport"]
    assert sport["precision"] == 0.85
    assert sport["passes"] is False


def test_score_no_judged_drops_has_no_precision():
    data = {"groups": {"sport": _group_of(0, 0, 3, 2),
                       "setting": [], "light": [], "weather": []}}
    score = api.compute_score(data)
    sport = score["per_group"]["sport"]
    assert sport["precision"] is None
    assert sport["passes"] is False


def test_score_overall_pools_every_group():
    data = {"groups": {
        "sport": _group_of(18, 2, 0, 0),
        "setting": _group_of(18, 2, 0, 0),
        "light": _group_of(18, 2, 0, 0),
        "weather": _group_of(18, 2, 0, 0),
    }}
    score = api.compute_score(data)
    overall = score["overall"]
    assert overall["n"] == 80 and overall["correct"] == 72 and overall["wrong"] == 8
    assert overall["precision"] == 0.9
    assert overall["passes"] is True


def test_score_endpoint_matches_compute_score(client, sample):
    client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport", "label": "correct"})
    resp = client.get(f"{API}/score")
    assert resp.status_code == 200
    body = resp.json()
    assert body == api.compute_score()
    assert body["per_group"]["sport"]["correct"] == 1
    assert body["bar"] == api.PASS_PRECISION and body["min_judged"] == api.MIN_JUDGED


# ---------------------------------------------------------------------------
# Local-only / replica mode
# ---------------------------------------------------------------------------

def test_never_proxies_to_the_nas_in_replica_mode(client, sample, monkeypatch):
    """Labels are local files; an unreachable NAS must not matter."""
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://127.0.0.1:9")
    assert client.get(f"{API}/items").status_code == 200
    assert client.get(f"{API}/score").status_code == 200
    assert client.post(f"{API}/label", json={
        "photo_id": sample["pid_922"], "term": "people", "group": "sport",
        "label": "correct"}).status_code == 200


def test_page_route_serves_html(client):
    resp = client.get("/eval/category-grounding")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "/api/eval/category-grounding" in resp.text


# ---------------------------------------------------------------------------
# Consistency with photosearch.category_grounding
# ---------------------------------------------------------------------------

def test_groups_match_category_grounding_module():
    from photosearch import category_grounding as cg
    assert set(api.GROUPS) == {g.name for g in cg.GROUPS}


def test_group_rules_cover_every_group():
    assert set(api.GROUP_RULES.keys()) == set(api.GROUPS)
    assert all(isinstance(v, str) and v for v in api.GROUP_RULES.values())


# ---------------------------------------------------------------------------
# request_intent classification
# ---------------------------------------------------------------------------

def test_request_intent_classifies_the_three_endpoints():
    from photosearch import request_intent as ri
    assert ri.infer_intent("GET", f"{API}/items", "") == "Load the category-grounding eval sample"
    assert ri.infer_intent("POST", f"{API}/label", "") == "Label a category-grounding drop"
    assert ri.infer_intent("GET", f"{API}/score", "") == "Score the category-grounding eval"


def test_request_log_records_the_inferred_intent(client, sample, tmp_path, monkeypatch):
    log_path = tmp_path / "request_log.jsonl"
    monkeypatch.setenv("PHOTOSEARCH_REQUEST_LOG", str(log_path))
    from photosearch import request_log
    request_log.reset()
    client.get(f"{API}/items")
    request_log.reset()
    lines = [json.loads(l) for l in log_path.read_text().splitlines()]
    rec = next(r for r in lines if r["path"] == f"{API}/items")
    assert rec["intent"] == "Load the category-grounding eval sample"
    request_log.reset()

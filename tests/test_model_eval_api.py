"""photosearch/model_eval_api.py — the API behind `/eval/models`, plus the
describe label store and scoring it feeds.

Every test points PHOTOSEARCH_MODEL_EVAL_DIR at tmp_path: nothing here may
touch the owner's real labels."""

import importlib.util
import os

import pytest

from photosearch import model_eval as me

API = "/api/eval/models"

T1 = "A red kayak rests on the sand, with two paddles beside it. A dog sleeps nearby."
T2 = "A blue kayak floats on a lake; a man waves from the shore."


@pytest.fixture(autouse=True)
def _eval_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_MODEL_EVAL_DIR", str(tmp_path / "model-eval"))


def _run(variant, model, texts):
    run = me.open_run("describe", variant, effective_model=model, prompt_sha=None)
    for pid, t in texts.items():
        run["items"][str(pid)] = {"text": t, "text_sha": me.text_sha(t)}
    me.save_run("describe", variant, run)


@pytest.fixture
def sample(db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id LIMIT 3")]
    me.save_sample("describe", [{"photo_id": ids[0], "stratum": "random"},
                                {"photo_id": ids[1], "stratum": "random"},
                                {"photo_id": ids[2], "stratum": "text"}])
    _run("secretmodelA", "vendor/secret-a", {ids[0]: T1, ids[1]: T1})
    _run("secretmodelB", "vendor/secret-b", {ids[0]: T2, ids[1]: T1})
    return ids


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    T1, T2,
    'The sign reads "Dock 5." Salt and pepper sit nearby. Mr. Smith waves.',
    "one sentence with no end",
])
def test_segment_claims_is_lossless(text):
    assert "".join(me.segment_claims(text)) == text


def test_segment_claims_splits_clauses_not_lists():
    segs = me.segment_claims(T1)
    assert segs == ["A red kayak rests on the sand, ", "with two paddles beside it. ",
                    "A dog sleeps nearby."]
    assert len(me.segment_claims("Salt and pepper sit on a table.")) == 1
    assert len(me.segment_claims("Mr. Smith waves.")) == 1


# --------------------------------------------------------------------------
# Claims API — blind, de-duplicated, keyed by text
# --------------------------------------------------------------------------

def test_claims_are_blind_and_deduplicated(client, sample):
    resp = client.get(API + "/describe/claims")
    assert resp.status_code == 200
    body = resp.text
    for leak in ("secretmodel", "vendor/secret"):
        assert leak not in body
    photos = {p["photo_id"]: p for p in resp.json()["photos"]}
    assert len(photos[sample[0]]["texts"]) == 2
    assert len(photos[sample[1]]["texts"]) == 1           # both wrote T1
    assert photos[sample[2]]["texts"] == []
    # Stable order across reloads.
    again = client.get(API + "/describe/claims").json()["photos"]
    assert [t["sha"] for t in again[0]["texts"]] == \
        [t["sha"] for t in photos[sample[0]]["texts"]]


def test_claim_label_round_trip_and_validation(client, sample):
    sha = me.text_sha(T1)
    ok = client.put(API + f"/describe/claims/{sha}",
                    json={"photo_id": sample[1], "wrong": [2], "done": True})
    assert ok.status_code == 200 and ok.json()["label"]["wrong"] == [2]
    assert client.put(API + f"/describe/claims/{sha}",
                      json={"photo_id": sample[1], "wrong": [9]}).status_code == 400
    assert client.put(API + f"/describe/claims/{me.text_sha(T2)}",
                      json={"photo_id": sample[1], "wrong": []}).status_code == 404
    assert client.put(API + f"/describe/claims/{sha}",
                      json={"photo_id": 999999, "wrong": []}).status_code == 404
    photos = {p["photo_id"]: p for p in client.get(API + "/describe/claims").json()["photos"]}
    assert photos[sample[1]]["done"] is True
    assert client.get(API + "/describe/claims").json()["progress"]["done"] == 1


def test_a_shared_text_label_counts_for_every_variant_that_wrote_it(sample):
    me.save_claim(me.text_sha(T1), sample[1], 3, [2])
    labels = me.load_claims()
    # secretmodelA wrote T1 for both photos; the label applies wherever T1 is.
    assert me.claim_errors(labels[me.text_sha(T1)]) == 1
    assert me.claim_errors(None) is None
    assert me.claim_errors({"done": False, "wrong": [1]}) is None


# --------------------------------------------------------------------------
# Pairs + text truth
# --------------------------------------------------------------------------

def _harness():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "describe_eval.py")
    spec = importlib.util.spec_from_file_location("describe_eval_api_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pairs_are_blind_and_scored(client, sample):
    D = _harness()
    data = D.build_pairs("secretmodelA", ["secretmodelB"], n=40)
    assert len(data["pairs"]) == 1                          # photo 2's texts are identical
    resp = client.get(API + "/describe/pairs")
    assert "secretmodel" not in resp.text
    pair = resp.json()["pairs"][0]
    b_side = "a" if pair["a"]["text"] == T2 else "b"        # T2 is secretmodelB's
    assert client.put(API + "/describe/pairs/" + pair["key"],
                      json={"choice": b_side}).status_code == 200
    assert client.put(API + "/describe/pairs/" + pair["key"],
                      json={"choice": "left"}).status_code == 400
    rows = D.pairwise_table()
    assert rows[0][:5] == ("secretmodelB", "secretmodelA", 1, 0, 0)
    assert rows[0][5] == 1.0


def test_sign_test_and_bootstrap():
    D = _harness()
    assert D.sign_test_p(8, 2) == pytest.approx(0.109375)
    assert D.sign_test_p(0, 0) is None
    lo, hi = D.bootstrap_mean_ci([0, 0, 1, 2, 0, 1])
    assert lo <= 4 / 6 <= hi


def test_text_truth_only_for_the_text_stratum(client, sample):
    assert client.put(API + f"/describe/text-truth/{sample[2]}",
                      json={"text": "BLUE DOOR"}).status_code == 200
    assert client.put(API + f"/describe/text-truth/{sample[0]}",
                      json={"text": "x"}).status_code == 404
    got = client.get(API + "/describe/text-truth").json()
    assert got["progress"] == {"done": 1, "total": 1}
    assert got["photos"][0]["truth"]["text"] == "BLUE DOOR"


def test_original_serves_the_local_cache(client, sample, tmp_path):
    from PIL import Image
    d = me.originals_dir()
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3000, 2000), "red").save(d / str(sample[0]), format="JPEG")
    resp = client.get(API + f"/original/{sample[0]}?max_px=600")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/jpeg"
    import io
    assert max(Image.open(io.BytesIO(resp.content)).size) == 600
    assert client.get(API + f"/original/{sample[1]}").status_code == 404

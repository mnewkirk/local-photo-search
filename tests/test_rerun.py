"""Tests for photosearch/rerun.py (M28 re-run index passes).

Network-free: the NAS HTTP layer (urllib / WorkerClient / worker._process_*) is
monkeypatched, so these lock in the orchestration logic — pass dispatch, submit
kwargs, mirror application — without a live NAS or LM Studio.
"""

import io
import json

import pytest

from photosearch import rerun


# ---------------------------------------------------------------------------
# Guardrails / pure helpers
# ---------------------------------------------------------------------------

def test_all_passes_matches_worker_api():
    from photosearch.worker_api import _ALL_PASSES
    assert set(rerun.ALL_PASSES) == set(_ALL_PASSES)


def test_requeue_rejects_unknown_pass(db):
    with pytest.raises(ValueError):
        rerun.requeue_passes([1], ["bogus"])


def test_requeue_rejects_empty_ids(db):
    with pytest.raises(ValueError):
        rerun.requeue_passes([], ["describe"])


def test_run_pass_sync_rejects_unknown_pass(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    with pytest.raises(ValueError):
        rerun.run_pass_sync(db, 1, "bogus")


def test_run_pass_sync_requires_server(db, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    pid = next(iter(db._test_photo_ids.values()))
    with pytest.raises(RuntimeError):
        rerun.run_pass_sync(db, pid, "describe")


def test_mirror_photos_noop_without_nas(db, monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    out = rerun.mirror_photos(db, [1])
    assert out["mirrored"] == 0
    assert "skipped" in out


def test_resolve_model_lmstudio_role(monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://lm:1234/v1")
    monkeypatch.setenv("PHOTOSEARCH_LLM_DESCRIBE_MODEL", "qwen2.5-vl-7b-instruct")
    monkeypatch.setenv("PHOTOSEARCH_LLM_TEXT_MODEL", "llama-3.2-3b-instruct")
    assert rerun._resolve_model("describe") == "qwen2.5-vl-7b-instruct"
    assert rerun._resolve_model("category-content") == "llama-3.2-3b-instruct"


def test_resolve_model_ollama_default(monkeypatch):
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_DESCRIBE_MODEL", raising=False)
    assert rerun._resolve_model("describe") == "llama3.2-vision"


# ---------------------------------------------------------------------------
# _apply_mirror — pure DB logic
# ---------------------------------------------------------------------------

def test_apply_mirror_text_columns(db):
    pid = db._test_photo_ids["DSC04894.JPG"]
    rerun._apply_mirror(db, pid, {
        "description": "a sailboat with kids in life jackets",
        "categories": json.dumps(["boat", "water"]),
        "keywords": json.dumps(["sailboat"]),
        "verified_at": "2026-06-23T00:00:00",
    })
    row = db.get_photo(pid)
    assert row["description"] == "a sailboat with kids in life jackets"
    assert json.loads(row["categories"]) == ["boat", "water"]
    assert row["verified_at"] == "2026-06-23T00:00:00"


def test_apply_mirror_replaces_clip_embedding(db):
    pid = db._test_photo_ids["DSC04894.JPG"]
    new_emb = [0.5] * 512
    rerun._apply_mirror(db, pid, {"clip_embedding": new_emb})
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM clip_embeddings WHERE photo_id=?", (pid,)).fetchone()
    assert row["n"] == 1  # replaced, not duplicated


def test_apply_mirror_clears_clip_when_null(db):
    pid = db._test_photo_ids["DSC04894.JPG"]
    rerun._apply_mirror(db, pid, {"clip_embedding": None})
    row = db.conn.execute(
        "SELECT COUNT(*) AS n FROM clip_embeddings WHERE photo_id=?", (pid,)).fetchone()
    assert row["n"] == 0


def test_apply_mirror_replaces_faces(db):
    pid = db._test_photo_ids["DSC04894.JPG"]  # starts with 2 faces
    rerun._apply_mirror(db, pid, {"faces": [
        {"bbox": [10, 90, 80, 20], "encoding": [0.1] * 512, "det_score": 0.9},
    ]})
    rows = db.conn.execute(
        "SELECT bbox_top, bbox_right, bbox_bottom, bbox_left, det_score "
        "FROM faces WHERE photo_id=?", (pid,)).fetchall()
    assert len(rows) == 1
    assert (rows[0]["bbox_top"], rows[0]["bbox_right"],
            rows[0]["bbox_bottom"], rows[0]["bbox_left"]) == (10, 90, 80, 20)
    # encoding stored too
    fid = db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()["id"]
    enc = db.conn.execute(
        "SELECT COUNT(*) AS n FROM face_encodings WHERE face_id=?", (fid,)).fetchone()
    assert enc["n"] == 1


# ---------------------------------------------------------------------------
# mirror_photos — monkeypatched NAS HTTP
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = json.dumps(payload).encode()
        self.status = status
    def read(self):
        return self._payload
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_mirror_photos_applies_remote_fields(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    pid = db._test_photo_ids["DSC04894.JPG"]

    payload = {
        "description": "mirrored desc",
        "categories": json.dumps(["mirrored"]),
        "clip_embedding": [0.25] * 512,
        "faces": [],
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=30: _FakeResp(payload))

    out = rerun.mirror_photos(db, [pid])
    assert out == {"mirrored": 1, "errors": 0, "missing": 0}
    row = db.get_photo(pid)
    assert row["description"] == "mirrored desc"
    # faces were cleared (payload had empty list)
    n = db.conn.execute("SELECT COUNT(*) AS n FROM faces WHERE photo_id=?", (pid,)).fetchone()["n"]
    assert n == 0


# ---------------------------------------------------------------------------
# run_pass_sync — dispatch + submit kwargs (mock worker layer)
# ---------------------------------------------------------------------------

def test_run_pass_sync_describe_dispatch(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    pid = db._test_photo_ids["DSC04894.JPG"]

    from photosearch import worker as W

    submitted = {}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

    def fake_download(client, photos, tmpdir):
        return [(photos[0], "/tmp/fake.jpg")]

    def fake_describe(downloaded, model="x"):
        return [{"photo_id": downloaded[0][0]["id"], "description": "new desc"}]

    def fake_submit(self, batch_id, pass_type, **kwargs):
        submitted["pass_type"] = pass_type
        submitted["kwargs"] = kwargs
        return {"written": 1, "processed": 1}

    monkeypatch.setattr(W, "WorkerClient", _FakeClient)
    monkeypatch.setattr(W, "_model_version", lambda m: "deadbeef")  # avoid ollama.list()
    monkeypatch.setattr(W, "_download_batch", fake_download)
    monkeypatch.setattr(W, "_process_describe", fake_describe)
    monkeypatch.setattr(_FakeClient, "submit_results", fake_submit, raising=False)
    monkeypatch.setattr(rerun, "mirror_photos", lambda db, ids, server=None: {"mirrored": 1})

    res = rerun.run_pass_sync(db, pid, "describe")
    assert submitted["pass_type"] == "describe"
    assert submitted["kwargs"]["describe_results"][0]["description"] == "new desc"
    assert "model" in submitted["kwargs"] and "model_version" in submitted["kwargs"]
    assert res["written"] == 1 and res["mirrored"] == 1 and res["authority"] == "nas"


def test_run_pass_sync_text_pass_skips_download(db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_NAS_URL", "http://nas:8000")
    pid = db._test_photo_ids["DSC04894.JPG"]
    from photosearch import worker as W

    calls = {"downloaded": False}

    class _FakeClient:
        def __init__(self, *a, **k): pass

    def fake_download(client, photos, tmpdir):
        calls["downloaded"] = True
        return []

    def fake_kw(photos, model="x"):
        return [{"photo_id": photos[0]["id"], "keywords": ["k1", "k2"]}]

    def fake_submit(self, batch_id, pass_type, **kwargs):
        calls["kwargs"] = kwargs
        return {"written": 1, "processed": 1}

    monkeypatch.setattr(W, "WorkerClient", _FakeClient)
    monkeypatch.setattr(W, "_model_version", lambda m: "deadbeef")  # avoid ollama.list()
    monkeypatch.setattr(W, "_download_batch", fake_download)
    monkeypatch.setattr(W, "_process_keywords", fake_kw)
    monkeypatch.setattr(_FakeClient, "submit_results", fake_submit, raising=False)
    monkeypatch.setattr(rerun, "mirror_photos", lambda db, ids, server=None: {"mirrored": 0})

    rerun.run_pass_sync(db, pid, "keywords")
    assert calls["downloaded"] is False  # text-only pass needs no image
    # per-result model stamped (matches worker submit shape)
    assert calls["kwargs"]["keywords_results"][0]["model"]

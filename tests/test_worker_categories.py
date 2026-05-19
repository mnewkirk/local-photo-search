import json
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Mirrors the existing test_web_vocab pattern: setenv PHOTOSEARCH_DB AND
    patch web._db_path because web.py captures the path at module load.
    Also patches worker_api._db_path so claim/submit endpoints use the tmp DB.
    Resets _shutting_down to False so tests are order-independent — earlier
    TestClient teardowns trigger the FastAPI shutdown handler which sets the
    flag True, and it persists across test functions in the same process."""
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web, worker_api
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_shutting_down", False, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        db.conn.executemany(
            "INSERT INTO photos (id, filepath, filename, description) VALUES (?, ?, ?, ?)",
            [(1, "a.jpg", "a.jpg", "a dog at the beach"),
             (2, "b.jpg", "b.jpg", "a misty mountain morning")],
        )
        db.conn.commit()
    return TestClient(web.app)


def test_claim_batch_returns_description_for_text_passes(client):
    r = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["pass_type"] == "category-content"
    descs = {p["id"]: p.get("description") for p in body["photos"]}
    assert descs == {1: "a dog at the beach", 2: "a misty mountain morning"}


def test_submit_category_content_writes_to_photos_and_generations(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    }).json()
    batch_id = claim["batch_id"]
    r = client.post("/api/worker/submit-results", json={
        "batch_id": batch_id,
        "worker_id": "w1",
        "pass_type": "category-content",
        "category_content_results": [
            {"photo_id": 1, "categories": ["beach", "dog"], "model": "llama3.2:3b"},
            {"photo_id": 2, "categories": ["mountain"], "model": "llama3.2:3b"},
        ],
    })
    assert r.status_code == 200, r.text

    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        rows = {
            row["id"]: json.loads(row["categories"])
            for row in db.conn.execute("SELECT id, categories FROM photos").fetchall()
        }
        assert rows == {1: ["beach", "dog"], 2: ["mountain"]}
        text_types = {r[0] for r in db.conn.execute(
            "SELECT DISTINCT text_type FROM generations"
        ).fetchall()}
        assert "category-content" in text_types


def test_submit_keywords_writes_lowercased(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "keywords", "limit": 10,
    }).json()
    r = client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"],
        "worker_id": "w1",
        "pass_type": "keywords",
        "keywords_results": [
            {"photo_id": 1, "keywords": ["dog", "pacific ocean", "stinson beach"],
             "model": "llama3.2:3b"},
        ],
    })
    assert r.status_code == 200, r.text
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT keywords FROM photos WHERE id=1").fetchone()
        assert json.loads(row[0]) == ["dog", "pacific ocean", "stinson beach"]


def test_submit_category_visual_writes(client):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-visual", "limit": 10,
    }).json()
    r = client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"],
        "worker_id": "w1",
        "pass_type": "category-visual",
        "category_visual_results": [
            {"photo_id": 1, "visual_tags": ["dramatic", "foggy"], "model": "llava"},
        ],
    })
    assert r.status_code == 200, r.text
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT visual_tags FROM photos WHERE id=1").fetchone()
        assert json.loads(row[0]) == ["dramatic", "foggy"]


def test_empty_results_still_mark_processed(client):
    """Parse-empty must mark processed so the photo isn't re-claimed forever."""
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10,
    }).json()
    client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"],
        "worker_id": "w1",
        "pass_type": "category-content",
        "category_content_results": [
            {"photo_id": 1, "categories": [], "model": "llama3.2:3b"},
            {"photo_id": 2, "categories": [], "model": "llama3.2:3b"},
        ],
    })
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        rows = db.conn.execute(
            "SELECT photo_id, attempts FROM worker_processed WHERE pass_type='category-content'"
        ).fetchall()
        assert {r[0]: r[1] for r in rows} == {1: 1, 2: 1}


def test_clear_pass_category_content_nulls_column_and_resets_attempts(client):
    """clear-pass for category-content should null the column and remove worker_processed rows."""
    # Seed a non-empty value + a processed marker.
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET categories = ? WHERE id = 1",
                        (json.dumps(["beach"]),))
        db.mark_processed([1], "category-content")
        db.conn.commit()
    r = client.post("/api/worker/clear-pass", json={
        "pass_type": "category-content",
        "photo_ids": [1],
    })
    assert r.status_code == 200, r.text
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT categories FROM photos WHERE id=1").fetchone()
        assert row[0] is None
        rows = db.conn.execute(
            "SELECT 1 FROM worker_processed WHERE pass_type='category-content' AND photo_id=1"
        ).fetchall()
        assert rows == []

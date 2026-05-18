import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_data(tmp_path, monkeypatch):
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    # /data dir for json files.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("PHOTOSEARCH_DATA_DIR", str(data_dir))

    # Seed candidates file.
    (data_dir / "vocab_candidates.json").write_text(json.dumps({
        "source_count": 3,
        "min_count": 1,
        "candidates": [{"term": "beach", "count": 3}, {"term": "dog", "count": 2}],
    }))

    # Seed a couple of described photos for coverage preview.
    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        db.conn.executemany(
            "INSERT INTO photos (filepath, filename, description) VALUES (?, ?, ?)",
            [("a.jpg", "a.jpg", "a dog at the beach"),
             ("b.jpg", "b.jpg", "an empty mountain road")],
        )
        db.conn.commit()

    # Patch web._db_path directly (env var is set at module import; must override the global).
    from photosearch import web
    original_db_path = web._db_path
    web._db_path = db_path
    try:
        yield TestClient(web.app)
    finally:
        web._db_path = original_db_path


def test_get_candidates_returns_seeded_file(client_with_data):
    r = client_with_data.get("/api/admin/vocab/candidates")
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"][0]["term"] == "beach"


def test_get_draft_returns_empty_when_missing(client_with_data):
    r = client_with_data.get("/api/admin/vocab/draft")
    assert r.status_code == 200
    assert r.json() == {"content": [], "visual": [], "expansions": {}}


def test_put_draft_persists_then_reads_back(client_with_data):
    draft = {
        "content": ["beach", "dog"],
        "visual": ["dramatic"],
        "expansions": {"sea": ["beach"]},
    }
    r = client_with_data.put("/api/admin/vocab/draft", json=draft)
    assert r.status_code == 200
    r2 = client_with_data.get("/api/admin/vocab/draft")
    assert r2.json() == draft


def test_coverage_preview_uses_draft_against_descriptions(client_with_data):
    # Both photos contain at least one content term.
    draft = {"content": ["beach", "mountain"], "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/coverage-preview",
                              json={"draft": draft, "sample_size": 10})
    body = r.json()
    assert body["sample_size"] == 2
    assert body["covered_count"] == 2
    assert body["coverage_pct"] == 100.0


def test_test_photo_returns_matched_terms(client_with_data):
    draft = {"content": ["beach", "dog"], "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/test-photo/1", json={"draft": draft})
    body = r.json()
    assert set(body["matched_categories"]) == {"beach", "dog"}


def test_compile_rejects_empty_visual(client_with_data, tmp_path):
    draft = {"content": ["x"] * 60, "visual": [], "expansions": {}}
    r = client_with_data.post("/api/admin/vocab/compile",
                              json={"draft": draft, "repo_dir": str(tmp_path)})
    assert r.status_code == 400
    assert "visual" in r.json()["detail"].lower()

"""Tests for Bundle F: text_match param rename, new filter params, and stats split."""

import json
import pytest


# ---------------------------------------------------------------------------
# Search signature tests (no DB needed)
# ---------------------------------------------------------------------------

def test_search_combined_accepts_text_match_param():
    """text_match replaces tag_match; tag_match no longer in signature."""
    from photosearch.search import search_combined
    import inspect
    sig = inspect.signature(search_combined)
    assert "text_match" in sig.parameters
    assert "tag_match" not in sig.parameters
    assert "category" in sig.parameters
    assert "visual_tag" in sig.parameters
    assert "keyword" in sig.parameters


def test_search_semantic_accepts_text_match_param():
    from photosearch.search import search_semantic
    import inspect
    sig = inspect.signature(search_semantic)
    assert "text_match" in sig.parameters
    assert "tag_match" not in sig.parameters


# ---------------------------------------------------------------------------
# API tests (FastAPI TestClient)
# ---------------------------------------------------------------------------

def test_api_search_translates_legacy_tag_to_category(tmp_path, monkeypatch):
    """?tag=beach should still work as ?category=beach (legacy alias)."""
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        db.conn.executemany(
            "INSERT INTO photos (id, filepath, filename, categories) VALUES (?, ?, ?, ?)",
            [(1, "a.jpg", "a.jpg", json.dumps(["beach", "dog"])),
             (2, "b.jpg", "b.jpg", json.dumps(["mountain"]))],
        )
        db.conn.commit()

    from fastapi.testclient import TestClient
    client = TestClient(web.app)
    r = client.get("/api/search?tag=beach")
    assert r.status_code == 200, r.text
    body = r.json()
    ids = [item["id"] for item in body.get("results", [])]
    assert 1 in ids
    assert 2 not in ids


def test_api_stats_has_three_new_counters(tmp_path, monkeypatch):
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        db.conn.execute(
            "INSERT INTO photos (id, filepath, filename, categories, visual_tags, keywords) "
            "VALUES (1, 'a.jpg', 'a.jpg', ?, ?, ?)",
            (json.dumps(["beach"]), json.dumps(["dramatic"]), json.dumps(["sunset"])),
        )
        db.conn.commit()

    from fastapi.testclient import TestClient
    client = TestClient(web.app)
    r = client.get("/api/stats")
    body = r.json()
    assert "category_tagged" in body
    assert "visual_tagged" in body
    assert "keyword_tagged" in body
    assert "tagged" not in body
    assert body["category_tagged"] == 1
    assert body["visual_tagged"] == 1
    assert body["keyword_tagged"] == 1

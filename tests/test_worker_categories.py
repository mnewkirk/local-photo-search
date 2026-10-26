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


def test_clear_pass_category_visual_nulls_column_and_resets_attempts(client):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET visual_tags = ? WHERE id = 1",
                        (json.dumps(["dramatic"]),))
        db.mark_processed([1], "category-visual")
        db.conn.commit()
    r = client.post("/api/worker/clear-pass", json={
        "pass_type": "category-visual",
        "photo_ids": [1],
    })
    assert r.status_code == 200, r.text
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT visual_tags FROM photos WHERE id=1").fetchone()
        assert row[0] is None
        rows = db.conn.execute(
            "SELECT 1 FROM worker_processed WHERE pass_type='category-visual' AND photo_id=1"
        ).fetchall()
        assert rows == []


def test_clear_pass_keywords_nulls_column_and_resets_attempts(client):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET keywords = ? WHERE id = 1",
                        (json.dumps(["beach"]),))
        db.mark_processed([1], "keywords")
        db.conn.commit()
    r = client.post("/api/worker/clear-pass", json={
        "pass_type": "keywords",
        "photo_ids": [1],
    })
    assert r.status_code == 200, r.text
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute("SELECT keywords FROM photos WHERE id=1").fetchone()
        assert row[0] is None
        rows = db.conn.execute(
            "SELECT 1 FROM worker_processed WHERE pass_type='keywords' AND photo_id=1"
        ).fetchall()
        assert rows == []


# ---------------------------------------------------------------------------
# Server-side grounding (docs/plans/category-grounding.md, step 4): the server
# re-grounds every category-content answer against the CURRENT description,
# because an older worker, a description rewritten since the claim, or an M28
# re-run can all hand it an ungrounded list.
# ---------------------------------------------------------------------------

def _seed(photo_id, description):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute(
            "INSERT INTO photos (id, filepath, filename, description) VALUES (?, ?, ?, ?)",
            (photo_id, f"{photo_id}.jpg", f"{photo_id}.jpg", description))
        db.conn.commit()


def _submit_categories(client, results):
    r = client.post("/api/worker/submit-results", json={
        "batch_id": "no-such-claim", "pass_type": "category-content",
        "category_content_results": results,
    })
    assert r.status_code == 200, r.text
    return r.json()


def _stored_and_logged(photo_id):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        stored = db.conn.execute("SELECT categories FROM photos WHERE id=?",
                                 (photo_id,)).fetchone()[0]
        logged = [r[0] for r in db.conn.execute(
            "SELECT generated_text FROM generations WHERE photo_id=? "
            "AND text_type='category-content' ORDER BY id", (photo_id,))]
    return stored, logged


def test_an_old_workers_list_is_repaired_and_the_kept_list_logged(client):
    """An older worker sends `basketball` for a soccer description; the server
    stores and logs it without, so stored == latest generation still holds."""
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    _submit_categories(client, [{
        "photo_id": 3, "model": "google/gemma-4-12b-qat",
        "categories": ["soccer", "basketball", "basketball hoop", "grass",
                       "artificial lighting"]}])
    stored, logged = _stored_and_logged(3)
    assert json.loads(stored) == ["soccer", "grass"]
    assert logged == [stored]


def test_grounding_uses_the_description_stored_now_not_at_claim(client):
    """verify rewrote the description between claim and submit."""
    _seed(3, "Children play indoor soccer inside a gym.")
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-content", "limit": 10}).json()
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        db.conn.execute("UPDATE photos SET description=? WHERE id=3",
                        ("Children play soccer on an outdoor turf field.",))
        db.conn.commit()
    client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"], "pass_type": "category-content",
        "category_content_results": [
            {"photo_id": 3, "categories": ["soccer", "indoor", "gym"]}]})
    stored, _ = _stored_and_logged(3)
    assert json.loads(stored) == ["soccer"]


def test_a_null_description_stores_the_list_unchanged(client):
    _seed(3, None)
    _submit_categories(client, [{"photo_id": 3,
                                 "categories": ["basketball", "soccer"]}])
    stored, logged = _stored_and_logged(3)
    assert json.loads(stored) == ["basketball", "soccer"]
    assert logged == [stored]


def test_an_explicit_empty_answer_is_still_done(client):
    """'[]' (the model answered "none") is a result, not an emptied list."""
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    body = _submit_categories(client, [{"photo_id": 3, "categories": []}])
    stored, _ = _stored_and_logged(3)
    assert stored == "[]"
    assert body["processed"] == 1


# ---------------------------------------------------------------------------
# The answer before grounding is kept (category-content-raw), so grounding can
# still be measured on data written after it shipped: the grounded list alone
# re-grounds to itself.
# ---------------------------------------------------------------------------

def _raw_logged(photo_id):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        return [(r[0], json.loads(r[1])) for r in db.conn.execute(
            "SELECT id, generated_text FROM generations WHERE photo_id=? "
            "AND text_type='category-content-raw' ORDER BY id", (photo_id,))]


def _grounded_id(photo_id):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        return db.conn.execute(
            "SELECT MAX(id) FROM generations WHERE photo_id=? "
            "AND text_type='category-content'", (photo_id,)).fetchone()[0]


def test_the_workers_raw_answer_is_grounded_and_logged(client):
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    raw = ["soccer", "basketball", "grass", "artificial lighting"]
    _submit_categories(client, [{"photo_id": 3, "model": "m", "categories": ["soccer", "grass"],
                                 "raw_categories": raw}])
    stored, _ = _stored_and_logged(3)
    assert json.loads(stored) == ["soccer", "grass"]
    [(rid, logged)] = _raw_logged(3)
    assert logged == raw
    # The eval pairs the two rows by id (raw is logged immediately before).
    assert rid == _grounded_id(3) - 1


def test_the_server_grounds_the_raw_answer_not_the_workers_list(client):
    """An older worker's rules let `basketball` through; the server's current
    rules decide from the raw answer."""
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    _submit_categories(client, [{"photo_id": 3, "categories": ["soccer", "basketball"],
                                 "raw_categories": ["soccer", "basketball"]}])
    stored, _ = _stored_and_logged(3)
    assert json.loads(stored) == ["soccer"]


def test_an_old_worker_without_raw_still_gets_its_drops_recorded(client):
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    _submit_categories(client, [{"photo_id": 3, "categories": ["soccer", "basketball"]}])
    assert [r for _, r in _raw_logged(3)] == [["soccer", "basketball"]]


def test_nothing_dropped_logs_no_raw_row(client):
    _seed(3, "Two boys play soccer on an outdoor grass field.")
    _submit_categories(client, [{"photo_id": 3, "categories": ["soccer", "grass"]}])
    assert _raw_logged(3) == []


def test_the_worker_sends_raw_categories_only_when_grounding_changed_them(monkeypatch):
    from photosearch import describe, worker

    answers = {1: (["soccer", "grass"], ["soccer", "basketball", "grass"]),
               2: (["beach"], ["beach"])}

    def fake_extract(description, model=None, report=None):
        kept, raw = answers[int(description)]
        report.update(raw=len(raw), raw_terms=list(raw), dropped={})
        return kept
    monkeypatch.setattr(describe, "extract_categories_from_description", fake_extract)
    monkeypatch.setattr(describe, "check_available", lambda model: None)
    rows = worker._process_category_content(
        [{"id": 1, "filename": "a", "description": "1"},
         {"id": 2, "filename": "b", "description": "2"}], model="m")
    assert rows[0] == {"photo_id": 1, "categories": ["soccer", "grass"],
                       "raw_categories": ["soccer", "basketball", "grass"]}
    assert rows[1] == {"photo_id": 2, "categories": ["beach"]}


def test_the_extractor_reports_the_terms_before_grounding(monkeypatch):
    from photosearch import describe

    monkeypatch.setattr(describe, "HAS_OLLAMA", True)
    monkeypatch.setattr(describe, "_text_pass_answer",
                        lambda *a, **k: ["soccer", "basketball", "grass"])
    report: dict = {}
    kept = describe.extract_categories_from_description(
        "Two boys play soccer on an outdoor grass field.", model="m", report=report)
    assert kept == ["soccer", "grass"]
    assert report["raw_terms"] == ["soccer", "basketball", "grass"]
    assert report["raw"] == 3 and "basketball" in report["dropped"]

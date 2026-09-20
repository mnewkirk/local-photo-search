"""Every path that persists `visual_tags` must go through the derive merge.

The merge is applied SERVER-SIDE (where the photo's EXIF is in the DB), not in
the worker — a worker only ever sees the pixels, which is the whole reason the
capture-fact tags were taken away from it.

Covers:
  * `worker_api.submit_results` category-visual branch — the worker fleet AND
    the M28 `rerun.run_pass_sync` path, which submits through this same
    endpoint.
  * the two in-process `index.py` writers (directory mode + collection mode).
  * the `photosearch derive-visual-tags` backfill CLI.
"""

import json
import os

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from photosearch import visual_tags_derive as V


# A full-sun frame: 1/800 s, f/5.6, ISO 160 -> EV100 ~ 13.2, 3:2 aspect.
DAYLIGHT = dict(exposure_time="1/800", f_number="28/5", iso=160,
                image_width=6000, image_height=4000)
# A 4 s tripod night frame: f/8, ISO 100 -> EV100 4.0.
NIGHT = dict(exposure_time="4", f_number="8", iso=100,
             image_width=6000, image_height=4000)


def _insert(db, pid, year, **exif):
    cols = ["id", "filepath", "filename", "date_taken"] + list(exif)
    vals = [pid, f"{year}/p{pid}.jpg", f"p{pid}.jpg",
            f"{year}-05-04T12:00:00"] + [exif[k] for k in exif]
    db.conn.execute(
        f"INSERT INTO photos ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)


# ---------------------------------------------------------------------------
# worker_api.submit_results
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "x.db")
    monkeypatch.setenv("PHOTOSEARCH_DB", db_path)
    from photosearch import web, worker_api
    monkeypatch.setattr(web, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_db_path", db_path, raising=False)
    monkeypatch.setattr(worker_api, "_shutting_down", False, raising=False)

    from photosearch.db import PhotoDB
    with PhotoDB(db_path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        _insert(db, 2, 2091, **NIGHT)
        _insert(db, 3, 2090)          # no EXIF at all
        db.conn.commit()
    return TestClient(web.app)


def _submit(client, results):
    claim = client.post("/api/worker/claim-batch", json={
        "worker_id": "w1", "pass_type": "category-visual", "limit": 10,
    }).json()
    return client.post("/api/worker/submit-results", json={
        "batch_id": claim["batch_id"], "worker_id": "w1",
        "pass_type": "category-visual", "category_visual_results": results,
    })


def _stored(pid):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute(
            "SELECT visual_tags FROM photos WHERE id=?", (pid,)).fetchone()
    return None if row[0] is None else json.loads(row[0])


def test_submit_strips_capture_facts_the_exif_contradicts(client):
    r = _submit(client, [{"photo_id": 1, "model": "llava",
                          "visual_tags": ["long-exposure", "low-light",
                                          "motion-blur", "joyful"]}])
    assert r.status_code == 200, r.text
    assert _stored(1) == ["joyful"]


def test_submit_adds_derived_capture_facts_the_model_missed(client):
    r = _submit(client, [{"photo_id": 2, "model": "llava",
                          "visual_tags": ["peaceful"]}])
    assert r.status_code == 200, r.text
    assert _stored(2) == ["long-exposure", "low-light", "peaceful"]


def test_submit_with_no_exif_only_strips(client):
    r = _submit(client, [{"photo_id": 3, "model": "llava",
                          "visual_tags": ["long-exposure", "centered"]}])
    assert r.status_code == 200, r.text
    assert _stored(3) == ["centered"]


def test_submit_persists_empty_list_when_the_merge_empties_it(client):
    """An empty result is a legitimate outcome — it must still persist '[]' so
    the photo is marked done in one pass (_SubmitOutcome semantics)."""
    r = _submit(client, [{"photo_id": 1, "model": "llava",
                          "visual_tags": ["long-exposure", "motion-blur"]}])
    assert r.status_code == 200, r.text
    assert _stored(1) == []
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        n = db.conn.execute(
            "SELECT COUNT(*) FROM worker_processed "
            "WHERE photo_id=1 AND pass_type='category-visual'").fetchone()[0]
    assert n == 1, "an emptied merge must still mark the photo processed"


def test_submit_logs_the_merged_array_to_generations(client):
    _submit(client, [{"photo_id": 2, "model": "llava", "model_version": "abc",
                      "visual_tags": ["peaceful"]}])
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        row = db.conn.execute(
            "SELECT generated_text FROM generations "
            "WHERE photo_id=2 AND text_type='category-visual'").fetchone()
    assert json.loads(row[0]) == ["long-exposure", "low-light", "peaceful"]


def test_submit_writes_no_generation_when_the_merge_leaves_only_derived(client):
    """Derived tags are not LLM artifacts. A photo whose only surviving tags
    are derived must not claim the model produced them."""
    _submit(client, [{"photo_id": 2, "model": "llava",
                      "visual_tags": ["motion-blur"]}])
    assert _stored(2) == ["long-exposure", "low-light"]
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        n = db.conn.execute(
            "SELECT COUNT(*) FROM generations "
            "WHERE photo_id=2 AND text_type='category-visual'").fetchone()[0]
    assert n == 0


# ---------------------------------------------------------------------------
# index.py in-process writers
# ---------------------------------------------------------------------------

def test_index_module_routes_visual_tags_through_the_merge():
    """Both in-process writers must call the shared helper, not json.dumps the
    raw model answer. Pinned by source inspection because the surrounding
    functions need a live Ollama + real image files to execute."""
    import inspect
    from photosearch import index as I

    src = inspect.getsource(I)
    # No writer may persist the raw VLM list.
    assert "visual_tags=_json.dumps(vtags)" not in src, \
        "index.py still writes the raw model answer; route it through merge_for_row"
    assert src.count("_merge_visual_tags(") >= 2, \
        "both index.py visual-tag writers must go through the shared merge"
    assert "merge_for_row" in inspect.getsource(I._merge_visual_tags)


def test_index_merge_helper_reads_the_photo_row(db):
    from photosearch.index import _merge_visual_tags

    db.conn.execute(
        "UPDATE photos SET exposure_time=?, f_number=?, iso=? WHERE id=?",
        ("1/800", "28/5", 160, 1))
    db.conn.commit()
    assert _merge_visual_tags(db, 1, ["long-exposure", "peaceful"]) == ["peaceful"]
    # An unknown photo id degrades to strip-only rather than raising.
    assert _merge_visual_tags(db, 10 ** 9, ["long-exposure", "peaceful"]) == ["peaceful"]


# ---------------------------------------------------------------------------
# derive-visual-tags backfill CLI
# ---------------------------------------------------------------------------

@pytest.fixture
def backfill_db(tmp_path):
    from photosearch.db import PhotoDB
    path = str(tmp_path / "b.db")
    with PhotoDB(path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        _insert(db, 2, 2091, **NIGHT)
        _insert(db, 3, 2090)
        _insert(db, 4, 2090, **DAYLIGHT)
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["long-exposure", "low-light", "joyful"]),))
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=2",
                        (json.dumps(["peaceful"]),))
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=3",
                        (json.dumps(["centered"]),))
        # id 4 stays NULL: never tagged, drives the worker queue.
        db.conn.commit()
    return path


def _run(path, *args):
    from cli import cli
    return CliRunner().invoke(cli, ["derive-visual-tags", "--db", path, *args])


def _all(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        return {r[0]: (None if r[1] is None else json.loads(r[1]))
                for r in db.conn.execute("SELECT id, visual_tags FROM photos")}


def test_backfill_is_dry_run_by_default(backfill_db):
    res = _run(backfill_db)
    assert res.exit_code == 0, res.output
    assert "Dry run" in res.output
    assert _all(backfill_db)[1] == ["long-exposure", "low-light", "joyful"]


def test_backfill_dry_run_prints_a_before_after_frequency_table(backfill_db):
    res = _run(backfill_db)
    assert "long-exposure" in res.output
    assert "low-light" in res.output
    assert "before" in res.output.lower() and "after" in res.output.lower()


def test_backfill_apply_rewrites_only_the_rows_that_change(backfill_db):
    res = _run(backfill_db, "--apply")
    assert res.exit_code == 0, res.output
    rows = _all(backfill_db)
    assert rows[1] == ["joyful"]
    assert rows[2] == ["long-exposure", "low-light", "peaceful"]
    assert rows[3] == ["centered"]


def test_backfill_never_touches_a_null_visual_tags_row(backfill_db):
    _run(backfill_db, "--apply")
    assert _all(backfill_db)[4] is None


def test_backfill_is_idempotent(backfill_db):
    _run(backfill_db, "--apply")
    first = _all(backfill_db)
    res = _run(backfill_db, "--apply")
    assert _all(backfill_db) == first
    assert "0 photo" in res.output or "would change 0" in res.output.lower()


def test_backfill_guards_each_update_on_the_old_value(backfill_db):
    """A concurrent fleet write between the read and the write must win, not be
    clobbered. The UPDATE carries `WHERE id=? AND visual_tags IS ?`."""
    import inspect
    import cli as C

    src = inspect.getsource(C.derive_visual_tags.callback)
    assert "visual_tags IS ?" in src, "the UPDATE must be guarded on the old value"


def test_backfill_writes_in_bounded_chunks(backfill_db):
    import inspect
    import cli as C

    src = inspect.getsource(C.derive_visual_tags.callback)
    assert "_DERIVE_VISUAL_CHUNK" in src, "the write loop must be chunked"


def test_backfill_logs_nothing_to_generations(backfill_db):
    _run(backfill_db, "--apply")
    from photosearch.db import PhotoDB
    with PhotoDB(backfill_db) as db:
        n = db.conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    assert n == 0, "derived tags are not LLM artifacts"


def test_backfill_skips_malformed_json_rather_than_crashing(backfill_db):
    from photosearch.db import PhotoDB
    with PhotoDB(backfill_db) as db:  # noqa: F811 — local import keeps the fixture lazy
        db.conn.execute("UPDATE photos SET visual_tags='not json' WHERE id=3")
        db.conn.commit()
    res = _run(backfill_db, "--apply")
    assert res.exit_code == 0, res.output
    with PhotoDB(backfill_db) as db:
        raw = db.conn.execute(
            "SELECT visual_tags FROM photos WHERE id=3").fetchone()[0]
    assert raw == "not json", "an unparseable row is left exactly as it was"

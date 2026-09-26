"""Review follow-ups on the derived-visual-tags work.

  * a transient lock on the EXIF read must DEFER, not silently under-derive
  * the contradiction table must contain only genuinely exclusive pairs
  * `merge_tags` must not iterate a JSON string character by character
  * `generations` must record what the VLM produced, not the merged array
  * an unparseable answer must not be persisted as '[]'
"""

import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from photosearch import visual_tags_derive as V

DAYLIGHT = dict(exposure_time="1/800", f_number="28/5", iso=160,
                image_width=6000, image_height=4000)
NIGHT = dict(exposure_time="4", f_number="8", iso=100,
             image_width=6000, image_height=4000)


def _insert(db, pid, year, **exif):
    cols = ["id", "filepath", "filename", "date_taken"] + list(exif)
    vals = [pid, f"{year}/p{pid}.jpg", f"p{pid}.jpg",
            f"{year}-05-04T12:00:00"] + [exif[k] for k in exif]
    db.conn.execute(
        f"INSERT INTO photos ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        vals)


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


def _row(pid, col="visual_tags"):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        return db.conn.execute(
            f"SELECT {col} FROM photos WHERE id=?", (pid,)).fetchone()[0]


def _attempts(pid):
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        r = db.conn.execute(
            "SELECT attempts FROM worker_processed "
            "WHERE photo_id=? AND pass_type='category-visual'", (pid,)).fetchone()
    return 0 if r is None else r[0]


# ---------------------------------------------------------------------------
# FINDING 2 — a transient lock on the EXIF read must defer
# ---------------------------------------------------------------------------

def test_a_locked_exif_read_defers_instead_of_under_deriving(client, monkeypatch):
    """Writing the row strip-only would drop the derived tags with no trace,
    spend the attempt and mark the photo processed — permanently
    under-derived. The transient-only deferral rule (50003c2) must hold here
    too."""
    from photosearch import worker_api as WA

    real = WA._merge_visual_tags

    def boom(db, photo_id, perceived):
        if photo_id == 2:
            raise sqlite3.OperationalError("database is locked")
        return real(db, photo_id, perceived)

    monkeypatch.setattr(WA, "_merge_visual_tags", boom)
    r = _submit(client, [{"photo_id": 2, "visual_tags": ["peaceful"],
                          "model": "llava"}])
    assert r.status_code == 200, r.text
    assert 2 in r.json().get("deferred_photo_ids", [])
    assert _row(2) is None, "a deferred photo must not be written"
    assert _attempts(2) == 0, "a transient lock must not spend an attempt"


def test_a_non_transient_read_error_degrades_to_strip_only(client, monkeypatch):
    from photosearch import worker_api as WA

    monkeypatch.setattr(
        WA, "_read_derive_row",
        lambda db, pid: (_ for _ in ()).throw(sqlite3.DatabaseError("corrupt")))
    r = _submit(client, [{"photo_id": 2, "model": "llava",
                          "visual_tags": ["long-exposure", "peaceful"]}])
    assert r.status_code == 200, r.text
    assert json.loads(_row(2)) == ["peaceful"]


# ---------------------------------------------------------------------------
# FINDING 1 (server side) — an unparseable answer is not an empty result
# ---------------------------------------------------------------------------

def test_a_null_result_leaves_the_column_null_but_spends_an_attempt(client):
    r = _submit(client, [{"photo_id": 1, "visual_tags": None, "model": "llava"}])
    assert r.status_code == 200, r.text
    assert _row(1) is None, "the photo must stay claimable"
    assert _attempts(1) == 1, "but a repeatable failure must be bounded"
    assert r.json()["written"] == 0


def test_an_empty_list_result_still_persists_and_marks_done(client):
    r = _submit(client, [{"photo_id": 1, "visual_tags": [], "model": "llava"}])
    assert r.status_code == 200, r.text
    assert json.loads(_row(1)) == []
    assert _attempts(1) == 1


# ---------------------------------------------------------------------------
# MINOR — generations records what the VLM produced
# ---------------------------------------------------------------------------

def test_generations_logs_the_perceived_tags_not_the_merged_array(client):
    _submit(client, [{"photo_id": 2, "model": "llava", "model_version": "v1",
                      "visual_tags": ["peaceful"]}])
    assert json.loads(_row(2)) == ["long-exposure", "low-light", "peaceful"]
    from photosearch.db import PhotoDB
    with PhotoDB(os.environ["PHOTOSEARCH_DB"]) as db:
        text = db.conn.execute(
            "SELECT generated_text FROM generations "
            "WHERE photo_id=2 AND text_type='category-visual'").fetchone()[0]
    assert json.loads(text) == ["peaceful"], \
        "llava did not produce long-exposure or low-light; EXIF did"


def test_index_logs_only_the_perceived_tags():
    import inspect

    from photosearch import index as I
    src = inspect.getsource(I)
    assert src.count('log_generation(photo_id, "visual_tags", perceived_json') == 2


# ---------------------------------------------------------------------------
# FINDING 3 — the contradiction table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pair", [
    ("close-up", "wide-angle"),     # an environmental portrait is ordinary
    ("foggy", "sunny"),             # sun through fog is a classic shot
    ("joyful", "moody"),            # subject emotion vs how it is lit
    ("colorful", "monochromatic"),  # a blazing orange sunset is both
    ("aerial", "close-up"),         # a tight drone crop is both
    ("macro", "wide-angle"),        # close-focus wide-angle is a real technique
    ("dramatic", "peaceful"),       # a still sunset over the sea is both
    ("colorful", "muted"),          # muted is the light, colorful the hue count
])
def test_pairs_that_are_not_genuinely_exclusive_are_gone(pair):
    normalised = {tuple(sorted(p)) for p in V.CONTRADICTORY_PAIRS}
    assert tuple(sorted(pair)) not in normalised


@pytest.mark.parametrize("pair", [
    ("joyful", "melancholy"),
    ("overcast", "sunny"),
    ("harsh-light", "soft-light"),
    ("muted", "vibrant"),
    ("black-and-white", "colorful"),
    ("black-and-white", "vibrant"),
    ("aerial", "macro"),
])
def test_the_genuinely_exclusive_pairs_remain(pair):
    normalised = {tuple(sorted(p)) for p in V.CONTRADICTORY_PAIRS}
    assert tuple(sorted(pair)) in normalised


def test_the_table_is_exactly_those_pairs():
    assert len(V.CONTRADICTORY_PAIRS) == 7


# ---------------------------------------------------------------------------
# MINOR — merge_tags and a JSON string
# ---------------------------------------------------------------------------

def test_merge_tags_accepts_a_json_array_string():
    assert V.merge_tags('["peaceful", "sunny"]', []) == ["peaceful", "sunny"]
    assert V.merge_tags("[]", ["low-light"]) == ["low-light"]


def test_merge_tags_rejects_any_other_string_loudly():
    with pytest.raises(TypeError):
        V.merge_tags("peaceful, sunny", [])
    with pytest.raises(TypeError):
        V.merge_tags("peaceful", [])


def test_merge_for_row_accepts_a_json_array_string():
    row = {"exposure_time": "4", "f_number": "8", "iso": 100,
           "image_width": None, "image_height": None, "aes_sharpness": None}
    assert V.merge_for_row('["peaceful"]', row) == \
        ["long-exposure", "low-light", "peaceful"]


# ---------------------------------------------------------------------------
# MINOR — the backfill dry run opens the DB read-only
# ---------------------------------------------------------------------------

def test_dry_run_opens_the_database_read_only(tmp_path):
    from click.testing import CliRunner

    from cli import cli
    from photosearch.db import PhotoDB

    path = str(tmp_path / "b.db")
    with PhotoDB(path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["long-exposure", "joyful"]),))
        db.conn.commit()
    before = os.path.getmtime(path)

    res = CliRunner().invoke(cli, ["derive-visual-tags", "--db", path])
    assert res.exit_code == 0, res.output
    assert "read-only" in res.output.lower()
    assert os.path.getmtime(path) == before, "a dry run must not touch the file"


def test_dry_run_on_a_read_only_file_still_works(tmp_path):
    from click.testing import CliRunner

    from cli import cli
    from photosearch.db import PhotoDB

    path = str(tmp_path / "ro.db")
    with PhotoDB(path) as db:
        _insert(db, 1, 2090, **DAYLIGHT)
        db.conn.execute("UPDATE photos SET visual_tags=? WHERE id=1",
                        (json.dumps(["long-exposure", "joyful"]),))
        db.conn.commit()
    os.chmod(path, 0o444)
    try:
        res = CliRunner().invoke(cli, ["derive-visual-tags", "--db", path])
        assert res.exit_code == 0, res.output
        assert "long-exposure" in res.output
    finally:
        os.chmod(path, 0o644)

"""`photosearch ground-categories` — the category-grounding backfill.

docs/plans/category-grounding.md, step 5. It applies the ONE rule
(photosearch/category_grounding.py) to `photos.categories` already stored.
What must hold: a dry run touches nothing; NULL rows, hand-edited rows,
recitations and lists grounding would empty are never written; every UPDATE
is guarded on the value it was computed from; commits happen between chunks.
"""
import json
import os

import pytest
from click.testing import CliRunner

SOCCER = "Two boys play soccer on an outdoor grass field."
GEMMA = "google/gemma-4-12b-qat"
LLAMA = "llama3.2:3b"


def _photo(db, pid, description, categories, generated=None, model=GEMMA):
    """A photo whose categories came from `model`. `generated` defaults to
    the stored list (stored == generation, the normal state)."""
    db.conn.execute(
        "INSERT INTO photos (id, filepath, filename, folder, description, categories) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (pid, f"2090/2090-01-01_x/{pid}.jpg", f"{pid}.jpg", "2090/2090-01-01_x",
         description, None if categories is None else json.dumps(categories)))
    gen = categories if generated is None else generated
    if gen:
        db.log_generation(pid, "category-content", json.dumps(gen), model, "lmstudio")
    db.conn.commit()


@pytest.fixture
def path(tmp_path):
    from photosearch.db import PhotoDB
    p = str(tmp_path / "g.db")
    with PhotoDB(p) as db:
        _photo(db, 1, SOCCER, ["soccer", "basketball", "grass"])
        _photo(db, 2, SOCCER, ["soccer", "grass"])                    # nothing to drop
    return p


def _run(*args):
    from cli import cli
    return CliRunner().invoke(cli, ["ground-categories", *args])


def _cats(path, pid):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        raw = db.conn.execute("SELECT categories FROM photos WHERE id=?",
                              (pid,)).fetchone()[0]
    return None if raw is None else json.loads(raw)


def _recital(description):
    from photosearch import category_grounding as cg
    from photosearch.vocab_content import CONTENT_VOCABULARY
    grouped = {t for g in cg.GROUPS for s in g.sides for t in s.terms}
    terms = [t for t in CONTENT_VOCABULARY if t in cg.OBJECT_TERMS
             and t not in grouped and not cg.is_supported(description, t)][:30]
    assert cg.is_recitation(description, terms)
    return terms


def test_apply_grounds_and_logs_nothing(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        gens = db.conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 1) == ["soccer", "grass"]
    assert _cats(path, 2) == ["soccer", "grass"]
    assert "Updated 1 photos" in res.output
    with PhotoDB(path) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0] == gens


def test_dry_run_writes_nothing_and_opens_read_only(path):
    before = os.path.getmtime(path)
    res = _run("--db", path)
    assert res.exit_code == 0, res.output
    assert "read-only" in res.output.lower()
    assert "basketball" in res.output and "1 photo rows would change" in res.output
    assert os.path.getmtime(path) == before
    assert _cats(path, 1) == ["soccer", "basketball", "grass"]


def test_a_missing_database_is_an_error_never_a_stub(tmp_path):
    p = str(tmp_path / "nope.db")
    for args in (["--db", p], ["--db", p, "--apply"]):
        res = _run(*args)
        assert res.exit_code != 0
        assert "not found" in res.output
        assert not os.path.exists(p)


def test_db_comes_from_the_environment(path, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", path)
    res = _run()
    assert res.exit_code == 0, res.output
    assert "1 photo rows would change" in res.output


def test_null_rows_are_never_touched(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        _photo(db, 3, SOCCER, None, generated=["soccer", "basketball"])
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 3) is None


def test_a_hand_edited_row_is_skipped_and_reported(path):
    """set_photo_tags / bulk-set-tags: the stored list holds a term the model
    never produced, so a human chose it and it wins."""
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        _photo(db, 3, SOCCER, ["soccer", "basketball", "trophy"],
               generated=["soccer", "basketball"])
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 3) == ["soccer", "basketball", "trophy"]
    assert "left alone: 1; e.g. 3" in res.output


def test_rows_with_no_generation_or_another_model_are_skipped(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        _photo(db, 3, SOCCER, ["soccer", "basketball"], generated=[])       # pre-logging
        _photo(db, 4, SOCCER, ["soccer", "basketball"], model=LLAMA)
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 3) == ["soccer", "basketball"]
    assert _cats(path, 4) == ["soccer", "basketball"]
    assert "1 with no category-content generation" in res.output
    res = _run("--db", path, "--apply", "--model", "llama")
    assert _cats(path, 4) == ["soccer"]


def test_folder_scopes_the_run(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        db.conn.execute("UPDATE photos SET folder='2091/other' WHERE id=1")
        db.conn.commit()
    res = _run("--db", path, "--apply", "--folder", "2090")
    assert res.exit_code == 0, res.output
    assert _cats(path, 1) == ["soccer", "basketball", "grass"]
    _run("--db", path, "--apply", "--folder", "2091/other")
    assert _cats(path, 1) == ["soccer", "grass"]


def test_a_guarded_update_loses_to_a_concurrent_write(path, monkeypatch):
    """A worker re-extracted photo 1 between the read and the write: the
    worker's value must survive and the race is counted."""
    import sqlite3

    from photosearch import category_grounding as cg
    real = cg.ground

    def racing(description, categories):
        if "basketball" in categories:
            other = sqlite3.connect(path)
            other.execute("UPDATE photos SET categories=? WHERE id=1",
                          (json.dumps(["soccer", "turf"]),))
            other.commit()
            other.close()
        return real(description, categories)
    monkeypatch.setattr(cg, "ground", racing)
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 1) == ["soccer", "turf"]
    assert "Updated 0 photos (1 changed underneath us, left alone)" in res.output


def test_commits_between_chunks(path, monkeypatch):
    import cli as C
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        for pid in range(3, 8):
            _photo(db, pid, SOCCER, ["soccer", "basketball court"])
    monkeypatch.setattr(C, "_GROUND_CATEGORIES_CHUNK", 2)
    seen = []
    real_init = PhotoDB.__init__

    def traced(self, *a, **kw):
        real_init(self, *a, **kw)
        self.conn.set_trace_callback(seen.append)
    monkeypatch.setattr(PhotoDB, "__init__", traced)
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    shape = ["U" if s.startswith("UPDATE photos SET categories") else "C"
             for s in seen if s.startswith("UPDATE photos SET categories") or s == "COMMIT"]
    assert shape == ["U", "U", "C", "U", "U", "C", "U", "U", "C"]
    assert res.output.count("... committed") == 3


def test_recitations_are_untouched_without_the_flag_and_requeued_with_it(path):
    from photosearch.db import PhotoDB
    recital = _recital(SOCCER)
    with PhotoDB(path) as db:
        _photo(db, 3, SOCCER, recital, model=LLAMA)
        db.mark_processed([3], "category-content")

    def ledger():
        with PhotoDB(path) as db:
            return db.conn.execute(
                "SELECT COUNT(*) FROM worker_processed WHERE photo_id=3 "
                "AND pass_type='category-content'").fetchone()[0]

    res = _run("--db", path, "--apply", "--model", "llama")
    assert res.exit_code == 0, res.output
    assert "Vocabulary recitations" in res.output
    assert _cats(path, 3) == recital and ledger() == 1

    res = _run("--db", path, "--model", "llama", "--requeue-recitations")   # dry run
    assert _cats(path, 3) == recital and ledger() == 1

    res = _run("--db", path, "--apply", "--model", "llama", "--requeue-recitations")
    assert res.exit_code == 0, res.output
    assert "Requeued 1 recitation rows" in res.output
    assert _cats(path, 3) is None and ledger() == 0


def test_an_emptied_list_is_never_written_as_empty(path):
    from photosearch.db import PhotoDB
    with PhotoDB(path) as db:
        _photo(db, 3, SOCCER, ["basketball", "basketball hoop"])
    res = _run("--db", path, "--apply")
    assert res.exit_code == 0, res.output
    assert _cats(path, 3) == ["basketball", "basketball hoop"]
    assert "Would EMPTY" in res.output

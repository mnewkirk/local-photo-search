"""Scoped worker queue counts drive from an index, not per-id rowid lookups.

`count_unprocessed_photos(photo_ids=...)` backs /api/worker/status for a
collection/directory/filter scope. With `id IN (...)` SQLite looked every id
up by rowid — one leaf of the wide photos table each — which cost 19-33 s on
the NAS for a 16k-photo collection. Above `_SCOPE_DRIVE_FROM_INDEX_AT` ids the
test is written `+id IN (...)`; these pin that the answer is unchanged and the
plan no longer searches by primary key.
"""

import pytest

from photosearch import db as dbmod
from photosearch.worker_api import _ALL_PASSES


@pytest.fixture
def varied(db):
    ids = [r[0] for r in db.conn.execute("SELECT id FROM photos ORDER BY id")]
    db.conn.execute("UPDATE photos SET description = NULL WHERE id = ?", (ids[0],))
    db.conn.execute("UPDATE photos SET verified_at = NULL")
    db.conn.execute("UPDATE photos SET verified_at = '2026-10-09' WHERE id = ?", (ids[1],))
    db.conn.execute("UPDATE photos SET aesthetic_concepts = NULL WHERE id = ?", (ids[2],))
    db.conn.execute("UPDATE photos SET categories = NULL, keywords = NULL, "
                    "visual_tags = NULL, aes_overall = NULL WHERE id IN (?, ?)",
                    (ids[3], ids[4]))
    db.conn.execute("DELETE FROM faces WHERE photo_id = ?", (ids[3],))
    db.conn.execute("DELETE FROM clip_embeddings WHERE photo_id = ?", (ids[3],))
    db.conn.execute(
        "INSERT INTO worker_processed (photo_id, pass_type, attempts) VALUES (?, 'keywords', 3)",
        (ids[3],))
    db.conn.commit()
    return db, ids


@pytest.mark.parametrize("pass_type", _ALL_PASSES)
def test_both_forms_give_the_same_count(varied, monkeypatch, pass_type):
    db, ids = varied
    scope = ids[:4]
    monkeypatch.setattr(dbmod, "_SCOPE_DRIVE_FROM_INDEX_AT", 10_000)
    rowid_form = db.count_unprocessed_photos(pass_type, photo_ids=scope)
    monkeypatch.setattr(dbmod, "_SCOPE_DRIVE_FROM_INDEX_AT", 0)
    index_form = db.count_unprocessed_photos(pass_type, photo_ids=scope)
    assert rowid_form == index_form


@pytest.mark.parametrize("pass_type", _ALL_PASSES)
def test_large_scope_does_not_search_by_rowid(varied, monkeypatch, pass_type):
    db, ids = varied
    monkeypatch.setattr(dbmod, "_SCOPE_DRIVE_FROM_INDEX_AT", 0)
    statements = []
    db.conn.set_trace_callback(statements.append)
    try:
        db.count_unprocessed_photos(pass_type, photo_ids=ids)
    finally:
        db.conn.set_trace_callback(None)
    sql = [s for s in statements if "COUNT(*)" in s][-1]
    plan = " | ".join(r[3] for r in db.conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert "INTEGER PRIMARY KEY" not in plan, plan


def test_small_scope_keeps_rowid_lookups():
    assert dbmod._scope_col("id", [1, 2, 3]) == "id"
    assert dbmod._scope_col("p.id", list(range(5000))) == "+p.id"

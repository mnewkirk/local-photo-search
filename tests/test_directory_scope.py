"""get_directory_photo_ids — the scope resolver behind `worker -d` and
`/api/worker/status?directory=`.

It sits on the fleet's PER-CLAIM path, and it used to be
`filepath LIKE 'dir/%'`: unindexable, so a full scan of `photos` on the N100 for
every batch a worker claimed. It now ranges over the indexed `folder` column.
These tests pin that the answer did not change — and that the index is used.
"""

import pytest


def _add(db, relpath):
    return db.add_photo(filepath=relpath, filename=relpath.rsplit("/", 1)[-1])


@pytest.fixture
def lib(db):
    ids = {
        "leaf_a": _add(db, "2091/2091-09-19_ILCE-7RM6/DSC0001.JPG"),
        "leaf_b": _add(db, "2091/2091-09-19_ILCE-7RM6/DSC0002.JPG"),
        "sub": _add(db, "2091/2091-09-19_ILCE-7RM6/edits/DSC0001-2.JPG"),
        # Shares the leaf's name as a PREFIX — must never match it.
        "sibling": _add(db, "2091/2091-09-19_ILCE-7RM6-extra/DSC0009.JPG"),
        "other_day": _add(db, "2091/2091-09-12_ILCE-7RM6/DSC0500.JPG"),
        "other_year": _add(db, "2090/2090-01-01/IMG_1.JPG"),
    }
    return db, ids


def test_leaf_folder_includes_subfolders_but_not_prefix_siblings(lib):
    db, ids = lib
    got = set(db.get_directory_photo_ids("2091/2091-09-19_ILCE-7RM6"))
    assert got == {ids["leaf_a"], ids["leaf_b"], ids["sub"]}


def test_year_level_directory_still_works(lib):
    db, ids = lib
    got = set(db.get_directory_photo_ids("2091"))
    assert got == {ids["leaf_a"], ids["leaf_b"], ids["sub"],
                   ids["sibling"], ids["other_day"]}


@pytest.mark.parametrize("spelling", [
    "2091/2091-09-12_ILCE-7RM6",
    "2091/2091-09-12_ILCE-7RM6/",
    "./2091/2091-09-12_ILCE-7RM6",
])
def test_input_spellings_are_equivalent(lib, spelling):
    db, ids = lib
    assert db.get_directory_photo_ids(spelling) == [ids["other_day"]]


def test_unknown_directory_is_empty(lib):
    db, _ = lib
    assert db.get_directory_photo_ids("2031/nope") == []


def test_case_mismatch_falls_back_to_the_old_like_semantics(lib):
    # LIKE is ASCII case-insensitive; a range over `folder` is not. Callers may
    # have relied on that, so a miss retries the old way rather than 404ing.
    db, ids = lib
    assert db.get_directory_photo_ids("2091/2091-09-12_ilce-7rm6") == [ids["other_day"]]


def test_the_lookup_uses_the_folder_index(lib):
    db, _ = lib
    sql, params = db._directory_scope_sql("2091/2091-09-19_ILCE-7RM6")
    plan = " ".join(r[-1] for r in db.conn.execute("EXPLAIN QUERY PLAN " + sql, params))
    assert "idx_photos_folder" in plan
    assert "SCAN photos" not in plan

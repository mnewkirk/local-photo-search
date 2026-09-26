"""A face de-duplicated away from a person must never be re-matched to THAT person.

The nightly maintenance sweep ran `match_faces` then `resolve_dups` and the log
showed, night after night, the two stages applying the same number
(6134/6134, 6129/6129, 5262/5262) while the unmatched pool grew 202k -> 213k
over eight runs. `resolve_dups` unmatches the loser of a (photo, person)
duplicate with `match_source='dedupe_unmatched'`, which is DELIBERATELY still
matchable (the loser may be a different person) — but nothing remembered WHICH
person it lost to, so the matcher re-applied the same label the next night and
the resolver stripped it again. Forever, on the sweep's heaviest CPU stage.

`face_person_exclusions` is the memory: per (face, person), so the face stays
matchable to anyone else.
"""

import struct

import pytest

from photosearch.db import FACE_DIMENSIONS, PhotoDB


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _vec(*head):
    """A 512-dim encoding whose leading components are `head`, rest zero."""
    v = [0.0] * FACE_DIMENSIONS
    for i, x in enumerate(head):
        v[i] = float(x)
    return v


def _add_reference(db, person_id, enc):
    cur = db.conn.execute(
        "INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
        (person_id, "test"))
    db.conn.execute(
        "INSERT INTO face_ref_encodings (ref_id, encoding) VALUES (?, ?)",
        (cur.lastrowid, struct.pack(f"{FACE_DIMENSIONS}f", *enc)))
    db.conn.commit()


@pytest.fixture
def mdb(tmp_path):
    """A DB with two registered people and faces at known ArcFace distances.

    Ann's reference is e0, Bea's is 1.2*e1 — far enough apart that a face
    sitting on Ann is outside BOTH the strict (1.15) and temporal (1.45)
    tolerances of Bea.

      dup_a / dup_b  (photo 1) — ~0 from Ann, ~1.5 from Bea.
                                 Only Ann is in tolerance at all.
      near_both      (photo 2) — 0.632 from Ann, 1.0 from Bea.
                                 Ann wins, but Bea is inside tolerance too.
    """
    db = PhotoDB(str(tmp_path / "m.db"))
    db.set_photo_root("/photos")
    p1 = db.add_photo(filepath="2026/a.jpg", filename="a.jpg",
                      date_taken="2026-09-12 10:00:00")
    p2 = db.add_photo(filepath="2026/b.jpg", filename="b.jpg",
                      date_taken="2026-09-12 10:05:00")
    ann = db.add_person("Ann")
    bea = db.add_person("Bea")
    _add_reference(db, ann, _vec(1))
    _add_reference(db, bea, _vec(0, 1.2))

    dup_a = db.add_face(p1, (10, 60, 60, 10), _vec(1), det_score=0.95)
    dup_b = db.add_face(p1, (10, 160, 60, 110), _vec(0.98, 0.02), det_score=0.80)
    near_both = db.add_face(p2, (10, 60, 60, 10), _vec(0.8, 0.6), det_score=0.90)
    db.conn.commit()

    db.ids = {"p1": p1, "p2": p2, "ann": ann, "bea": bea,
              "dup_a": dup_a, "dup_b": dup_b, "near_both": near_both}
    yield db
    db.close()


def _excl(db):
    return {(r[0], r[1]) for r in db.conn.execute(
        "SELECT face_id, person_id FROM face_person_exclusions")}


# ---------------------------------------------------------------------------
# schema (the v30 -> v31 migration itself lives in tests/test_db.py, beside
# the v29 -> v30 one)
# ---------------------------------------------------------------------------

def test_migration_is_fine_when_face_dedupe_undo_does_not_exist(db, tmp_db_path):
    """`face_dedupe_undo` is created on demand, so most DBs have never had one.
    The migration's backfill must not assume it."""
    db.conn.execute("DROP TABLE IF EXISTS face_dedupe_undo")
    db.conn.execute("DROP TABLE IF EXISTS face_person_exclusions")
    db.conn.execute("UPDATE schema_info SET value = '30' WHERE key = 'version'")
    db.conn.commit()
    db.close()
    with PhotoDB(tmp_db_path) as reopened:
        assert reopened.conn.execute(
            "SELECT COUNT(*) FROM face_person_exclusions").fetchone()[0] == 0


def test_migration_backfills_from_face_dedupe_undo(db, tmp_db_path):
    """The ~5,531 already-de-duplicated faces must stop churning on deploy,
    without an operator step."""
    face_id = db._test_face_ids["alex_907"]
    alex = db._test_person_ids["Alex"]
    labelled = db._test_face_ids["jamie_922"]
    jamie = db._test_person_ids["Jamie"]

    db.conn.execute(
        "CREATE TABLE IF NOT EXISTS face_dedupe_undo ("
        "face_id INTEGER PRIMARY KEY, person_id INTEGER, match_source TEXT, "
        "unmatched_at TEXT DEFAULT (datetime('now')))")
    db.conn.execute("INSERT INTO face_dedupe_undo(face_id, person_id, match_source) "
                    "VALUES (?, ?, 'strict')", (face_id, alex))
    # A snapshot row whose face was since re-labelled: not a live exclusion.
    db.conn.execute("INSERT INTO face_dedupe_undo(face_id, person_id, match_source) "
                    "VALUES (?, ?, 'strict')", (labelled, jamie))
    db.conn.execute("UPDATE faces SET person_id = NULL, "
                    "match_source = 'dedupe_unmatched' WHERE id = ?", (face_id,))
    db.conn.execute("DROP TABLE IF EXISTS face_person_exclusions")
    db.conn.execute("UPDATE schema_info SET value = '30' WHERE key = 'version'")
    db.conn.commit()
    db.close()

    with PhotoDB(tmp_db_path) as reopened:
        assert _excl(reopened) == {(face_id, alex)}

    # Re-running the backfill by hand adds nothing (PK, INSERT OR IGNORE).
    with PhotoDB(tmp_db_path) as reopened:
        from photosearch.db import backfill_exclusions_from_dedupe_undo
        n = backfill_exclusions_from_dedupe_undo(reopened.conn, apply=True)
        assert n == 0
        assert _excl(reopened) == {(face_id, alex)}


def test_backfill_skips_an_undo_row_whose_person_was_deleted(db, tmp_db_path):
    """`INSERT OR IGNORE` does NOT suppress a FOREIGN KEY violation, and this
    runs inside `_init_schema` — one stale person_id would make the PhotoDB
    constructor throw, so the web server and every CLI container would fail to
    start and the schema would never reach 31. The live snapshot has 2,940 of
    8,919 orphan face_ids, proving these rows do go stale."""
    from photosearch.db import backfill_exclusions_from_dedupe_undo

    face_id = db._test_face_ids["alex_907"]
    db.conn.execute(
        "CREATE TABLE IF NOT EXISTS face_dedupe_undo ("
        "face_id INTEGER PRIMARY KEY, person_id INTEGER, match_source TEXT, "
        "unmatched_at TEXT DEFAULT (datetime('now')))")
    db.conn.execute("INSERT INTO face_dedupe_undo(face_id, person_id, match_source) "
                    "VALUES (?, 999999, 'strict')", (face_id,))
    db.conn.execute("UPDATE faces SET person_id = NULL, "
                    "match_source = 'dedupe_unmatched' WHERE id = ?", (face_id,))
    db.conn.commit()

    assert backfill_exclusions_from_dedupe_undo(db.conn, apply=True) == 0
    assert _excl(db) == set()


def test_a_failing_backfill_never_blocks_the_migration(db, tmp_db_path, caplog):
    """A backfill that raises must not take the whole process down with it —
    the schema upgrade completes and the CLI command remains the manual retry."""
    import photosearch.db as dbmod

    db.conn.execute("DROP TABLE IF EXISTS face_person_exclusions")
    db.conn.execute("UPDATE schema_info SET value = '30' WHERE key = 'version'")
    db.conn.commit()
    db.close()

    def boom(conn, apply=False):
        raise RuntimeError("stale undo row")

    original = dbmod.backfill_exclusions_from_dedupe_undo
    dbmod.backfill_exclusions_from_dedupe_undo = boom
    try:
        with caplog.at_level("WARNING", logger="photosearch.db"):
            with PhotoDB(tmp_db_path) as reopened:
                assert reopened.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='face_person_exclusions'").fetchone()
                version = reopened.conn.execute(
                    "SELECT value FROM schema_info WHERE key = 'version'"
                ).fetchone()["value"]
                assert int(version) == dbmod.SCHEMA_VERSION
    finally:
        dbmod.backfill_exclusions_from_dedupe_undo = original

    assert any("exclusion" in r.message.lower() or "backfill" in r.message.lower()
               for r in caplog.records), "the failure was swallowed silently"


def test_unmatch_does_not_stamp_over_an_already_unmatched_face(mdb):
    """The UPDATE loop ran over every id passed in, including ones skipped as
    already unmatched — so a `rejected` marker (a human 'no') could be
    overwritten with `dedupe_unmatched`, and the return value over-counted."""
    from photosearch.db import unmatch_faces_as_duplicates
    from photosearch.faces import REJECTED_MATCH_SOURCE

    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                     "WHERE id = ?", (mdb.ids["ann"], mdb.ids["dup_a"]))
    mdb.conn.execute("UPDATE faces SET person_id = NULL, match_source = ? "
                     "WHERE id = ?", (REJECTED_MATCH_SOURCE, mdb.ids["dup_b"]))
    mdb.conn.commit()

    n = unmatch_faces_as_duplicates(mdb.conn, [mdb.ids["dup_a"], mdb.ids["dup_b"]])
    mdb.conn.commit()

    assert n == 1, "an already-unmatched face was counted as unmatched"
    assert mdb.conn.execute(
        "SELECT match_source FROM faces WHERE id = ?",
        (mdb.ids["dup_b"],)).fetchone()["match_source"] == REJECTED_MATCH_SOURCE
    assert _excl(mdb) == {(mdb.ids["dup_a"], mdb.ids["ann"])}


def test_exclusions_do_not_outlive_their_face_or_person(mdb):
    """PRAGMA foreign_keys is ON, so both sides cascade."""
    from photosearch.db import record_face_person_exclusions
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="test")
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["near_both"], mdb.ids["bea"])], reason="test")
    mdb.conn.commit()
    assert len(_excl(mdb)) == 2

    mdb.delete_face(mdb.ids["dup_b"])
    mdb.conn.execute("DELETE FROM face_references WHERE person_id = ?", (mdb.ids["bea"],))
    mdb.conn.execute("DELETE FROM faces WHERE person_id = ?", (mdb.ids["bea"],))
    mdb.conn.execute("DELETE FROM persons WHERE id = ?", (mdb.ids["bea"],))
    mdb.conn.commit()
    assert _excl(mdb) == set()


# ---------------------------------------------------------------------------
# the matchers
# ---------------------------------------------------------------------------

def test_strict_matcher_skips_the_excluded_pairing_only(mdb):
    """Excluded from Ann, and Bea is too far -> the face stays unmatched.
    Every OTHER face still matches Ann normally."""
    from photosearch.db import record_face_person_exclusions
    from photosearch.faces import match_faces_to_persons

    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    match_faces_to_persons(mdb)
    rows = {r["id"]: r["person_id"] for r in mdb.conn.execute(
        "SELECT id, person_id FROM faces")}
    assert rows[mdb.ids["dup_b"]] is None
    assert rows[mdb.ids["dup_a"]] == mdb.ids["ann"]


def test_strict_matcher_falls_through_to_the_next_best_person(mdb):
    """The correct behaviour is NEXT best inside the same tolerance, not
    dropping the face: `near_both` is 0.632 from Ann and 0.894 from Bea."""
    from photosearch.db import record_face_person_exclusions
    from photosearch.faces import match_faces_to_persons

    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["near_both"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    match_faces_to_persons(mdb)
    row = mdb.conn.execute("SELECT person_id, match_source FROM faces WHERE id = ?",
                           (mdb.ids["near_both"],)).fetchone()
    assert row["person_id"] == mdb.ids["bea"]
    assert row["match_source"] == "strict"


def test_temporal_matcher_skips_the_excluded_pairing_only(mdb):
    """The bug was one query, not both — temporal must honour it too."""
    from photosearch.db import record_face_person_exclusions
    from photosearch.faces import match_faces_temporal

    # Ann is confirmed in this session (an auto-matched face in photo 1).
    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                     "WHERE id = ?", (mdb.ids["ann"], mdb.ids["dup_a"]))
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    match_faces_temporal(mdb)
    assert mdb.conn.execute("SELECT person_id FROM faces WHERE id = ?",
                            (mdb.ids["dup_b"],)).fetchone()["person_id"] is None


def test_temporal_matcher_does_NOT_fall_through_to_the_next_best_person(mdb):
    """The asymmetry with strict. Temporal's `min_gap` check is a judgement
    about the WHOLE field, so removing the barred person from the ranking
    before it would hand the runner-up an unopposed win the unfiltered field
    never gave them. Temporal therefore ranks unfiltered and, if the top
    choice is barred, leaves the face alone."""
    from photosearch.db import record_face_person_exclusions
    from photosearch.faces import match_faces_temporal

    # Bea is confirmed in this session so the temporal presence check passes,
    # and she is a clearly-separated second (0.632 Ann vs 1.0 Bea).
    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                     "WHERE id = ?", (mdb.ids["bea"], mdb.ids["dup_a"]))
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["near_both"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    match_faces_temporal(mdb)
    assert mdb.conn.execute(
        "SELECT person_id FROM faces WHERE id = ?",
        (mdb.ids["near_both"],)).fetchone()["person_id"] is None


def test_temporal_exclusion_cannot_promote_an_ambiguous_sibling(tmp_path):
    """THE finding. Two siblings 1.20 / 1.25 from a face is a 0.05 gap, under
    TEMPORAL_MIN_GAP (0.08) — the old code REFUSED it as ambiguous. Barring
    the nearer sibling must not turn the other into an unopposed top with a
    wide gap: that converts self-cancelling nightly churn into a PERSISTENT
    wrong label, and it is the Calvin<->Ellie case the docs call the ArcFace
    limit. Measured on the 260k-face replica: of 1,500 sampled
    backfill-eligible faces, 34 gained a new label under fall-through, 17 of
    them sibling swaps.
    """
    from photosearch.db import record_face_person_exclusions
    from photosearch.faces import match_faces_temporal

    db = PhotoDB(str(tmp_path / "sib.db"))
    db.set_photo_root("/photos")
    p1 = db.add_photo(filepath="2026/s1.jpg", filename="s1.jpg",
                      date_taken="2026-09-12 10:00:00")
    p2 = db.add_photo(filepath="2026/s2.jpg", filename="s2.jpg",
                      date_taken="2026-09-12 10:05:00")
    cal = db.add_person("Cal")
    elle = db.add_person("Elle")
    _add_reference(db, cal, _vec(1))
    _add_reference(db, elle, _vec(0, 1))

    # 1.200 from Cal, 1.250 from Elle -> gap 0.050 < TEMPORAL_MIN_GAP.
    ambiguous = db.add_face(p1, (10, 60, 60, 10), _vec(0.5, 0.43875, 0.99875),
                            det_score=0.9)
    # Both siblings confirmed in the session, so Check 3 passes either way.
    anchor_c = db.add_face(p2, (10, 60, 60, 10), _vec(1), det_score=0.9)
    anchor_e = db.add_face(p2, (10, 160, 60, 110), _vec(0, 1), det_score=0.9)
    db.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                    "WHERE id = ?", (cal, anchor_c))
    db.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                    "WHERE id = ?", (elle, anchor_e))
    db.conn.commit()

    # Sanity: without any exclusion the gap guard already refuses this face.
    assert match_faces_temporal(db) == 0

    record_face_person_exclusions(db.conn, [(ambiguous, cal)], reason="resolve_dups")
    db.conn.commit()

    match_faces_temporal(db)
    got = db.conn.execute("SELECT person_id FROM faces WHERE id = ?",
                          (ambiguous,)).fetchone()["person_id"]
    assert got is None, (
        "barring the nearer sibling promoted the other past the min_gap guard")
    db.close()


def test_both_matchers_use_the_shared_exclusion_loader():
    """Same precedent as MATCHABLE_SQL: a fix applied to one candidate query
    would have looked complete."""
    import inspect
    from photosearch import faces

    for fn in (faces.match_faces_to_persons, faces.match_faces_temporal):
        assert "load_match_exclusions" in inspect.getsource(fn), (
            f"{fn.__name__} does not consult the shared exclusion loader")


# ---------------------------------------------------------------------------
# recording the exclusion — every duplicate-unmatch path
# ---------------------------------------------------------------------------

def test_maintenance_resolve_dups_records_the_exclusion(mdb):
    from photosearch.maintenance import _stage_resolve_dups

    for fid in (mdb.ids["dup_a"], mdb.ids["dup_b"]):
        mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                         "WHERE id = ?", (mdb.ids["ann"], fid))
    mdb.conn.commit()

    out = _stage_resolve_dups(mdb, True, lambda e: None, lambda: False)
    assert out["applied"] == 1
    assert _excl(mdb) == {(mdb.ids["dup_b"], mdb.ids["ann"])}
    assert mdb.conn.execute(
        "SELECT match_source FROM faces WHERE id = ?",
        (mdb.ids["dup_b"],)).fetchone()["match_source"] == "dedupe_unmatched"


def test_the_duplicate_unmatch_write_lives_in_one_place():
    """cli.py and maintenance.py each had their own copy of snapshot-then-null.
    A fix to one would have left the other churning."""
    import inspect

    import cli
    from photosearch import maintenance

    for mod in (cli, maintenance):
        src = inspect.getsource(mod)
        assert "match_source = 'dedupe_unmatched'" not in src, (
            f"{mod.__name__} still writes the dedupe marker itself instead of "
            "calling the shared primitive")
        assert "unmatch_faces_as_duplicates" in src


def test_restore_unmatched_faces_clears_the_exclusion(mdb):
    """A restore puts the person back; leaving the exclusion would make the
    face permanently unmatchable to them."""
    from click.testing import CliRunner

    import cli
    from photosearch.db import unmatch_faces_as_duplicates

    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                     "WHERE id = ?", (mdb.ids["ann"], mdb.ids["dup_b"]))
    unmatch_faces_as_duplicates(mdb.conn, [mdb.ids["dup_b"]])
    mdb.conn.commit()
    assert _excl(mdb) == {(mdb.ids["dup_b"], mdb.ids["ann"])}
    path = mdb.db_path
    mdb.close()

    res = CliRunner().invoke(cli.cli, ["restore-unmatched-faces", "--db", path, "--apply"])
    assert res.exit_code == 0, res.output
    with PhotoDB(path) as reopened:
        assert _excl(reopened) == set()
        assert reopened.conn.execute(
            "SELECT person_id FROM faces WHERE id = ?",
            (mdb.ids["dup_b"],)).fetchone()["person_id"] is not None


# ---------------------------------------------------------------------------
# a human overrules the machine
# ---------------------------------------------------------------------------

def test_manual_assignment_clears_the_exclusion(mdb):
    from photosearch.db import record_face_person_exclusions
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    mdb.assign_face_to_person(mdb.ids["dup_b"], mdb.ids["ann"], match_source="manual")
    mdb.conn.commit()
    assert _excl(mdb) == set()


def test_an_automatic_assignment_does_not_clear_the_exclusion(mdb):
    """Only a human overrules it — otherwise the first matcher write would
    delete the very memory that is supposed to stop it."""
    from photosearch.db import record_face_person_exclusions
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    mdb.assign_face_to_person(mdb.ids["dup_b"], mdb.ids["ann"], match_source="strict")
    mdb.conn.commit()
    assert _excl(mdb) == {(mdb.ids["dup_b"], mdb.ids["ann"])}


def test_api_assign_clears_the_exclusion(client, db):
    from photosearch.db import record_face_person_exclusions
    face_id = db._test_face_ids["alex_907"]
    alex = db._test_person_ids["Alex"]
    record_face_person_exclusions(db.conn, [(face_id, alex)], reason="resolve_dups")
    db.conn.commit()

    r = client.post(f"/api/faces/{face_id}/assign", params={"name": "Alex"})
    assert r.status_code == 200
    assert _excl(db) == set()


def test_accepting_a_merge_clears_the_exclusion(client, db):
    """The merge-accept path writes the label with its own UPDATE rather than
    through db.assign_face_to_person, so it has to clear them itself."""
    from photosearch.db import record_face_person_exclusions
    face_id = db._test_face_ids["unknown_878"]        # cluster 99, unmatched
    alex = db._test_person_ids["Alex"]
    record_face_person_exclusions(db.conn, [(face_id, alex)], reason="resolve_dups")
    db.conn.commit()

    r = client.post("/api/faces/merges", json={
        "source": {"type": "cluster", "id": 99},
        "target": {"type": "person", "id": alex}})
    assert r.status_code == 200, r.text
    assert r.json()["moved_face_count"] == 1
    assert _excl(db) == set()


def test_correct_face_clears_the_exclusion(mdb):
    """`correct-face` wrote a raw UPDATE that never reached the shared
    primitive, so a hand-corrected face stayed barred from the person the
    human had just named it."""
    from click.testing import CliRunner

    import cli
    from photosearch.db import record_face_person_exclusions

    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()
    path = mdb.db_path
    mdb.close()

    # dup_b is the second face of a.jpg (ordered by id).
    res = CliRunner().invoke(cli.cli, ["correct-face", "--db", path, "a.jpg", "2", "Ann"])
    assert res.exit_code == 0, res.output
    with PhotoDB(path) as reopened:
        assert _excl(reopened) == set()
        row = reopened.conn.execute(
            "SELECT person_id, match_source FROM faces WHERE id = ?",
            (mdb.ids["dup_b"],)).fetchone()
        assert row["person_id"] == mdb.ids["ann"]
        assert row["match_source"] == "manual"


def test_bulk_unmatch_restore_clears_the_exclusion(mdb):
    """`restore-unmatch`'s snapshot restore is the other undo path, and it has
    the same rule as restore-unmatched-faces: the restored pairing's exclusion
    is spent."""
    from photosearch import bulk_unmatch
    from photosearch.db import record_face_person_exclusions

    mdb.conn.execute("UPDATE faces SET person_id = NULL, match_source = ? "
                     "WHERE id = ?", (bulk_unmatch.REJECTED_MATCH_SOURCE,
                                      mdb.ids["dup_b"]))
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    restored, _skipped = bulk_unmatch.restore(mdb, [
        {"face_id": mdb.ids["dup_b"], "person_id": mdb.ids["ann"],
         "match_source": "manual"}])
    assert restored == 1
    assert _excl(mdb) == set()


def test_api_bulk_assign_clears_the_exclusion(client, db):
    from photosearch.db import record_face_person_exclusions
    face_id = db._test_face_ids["alex_907"]
    alex = db._test_person_ids["Alex"]
    record_face_person_exclusions(db.conn, [(face_id, alex)], reason="resolve_dups")
    db.conn.commit()

    r = client.post("/api/faces/bulk-assign",
                    json={"face_ids": [face_id], "person_name": "Alex"})
    assert r.status_code == 200
    assert _excl(db) == set()


# ---------------------------------------------------------------------------
# the replica push
# ---------------------------------------------------------------------------

def test_face_state_apply_cannot_recreate_an_excluded_pairing(mdb, tmp_path):
    """The replica recomputes matches off a synced copy, so its file can carry
    exactly the pairing the NAS de-duplicated. Applying it would restart the
    loop by the long route."""
    from photosearch.db import record_face_person_exclusions
    from photosearch.face_state import apply_face_state, export_face_state

    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' "
                     "WHERE id IN (?, ?)",
                     (mdb.ids["ann"], mdb.ids["dup_b"], mdb.ids["near_both"]))
    mdb.conn.commit()
    path = str(tmp_path / "fs.db")
    export_face_state(mdb, path)

    # Both faces come back unmatched; only dup_b is excluded from Ann.
    mdb.conn.execute("UPDATE faces SET person_id = NULL, match_source = NULL "
                     "WHERE id IN (?, ?)", (mdb.ids["dup_b"], mdb.ids["near_both"]))
    record_face_person_exclusions(
        mdb.conn, [(mdb.ids["dup_b"], mdb.ids["ann"])], reason="resolve_dups")
    mdb.conn.commit()

    apply_face_state(mdb, path, apply=True)
    rows = {r["id"]: r["person_id"] for r in mdb.conn.execute(
        "SELECT id, person_id FROM faces")}
    assert rows[mdb.ids["dup_b"]] is None, "a push re-created an excluded pairing"
    assert rows[mdb.ids["near_both"]] == mdb.ids["ann"]


# ---------------------------------------------------------------------------
# the loop itself
# ---------------------------------------------------------------------------

def test_match_then_resolve_twice_is_stable(mdb):
    """THE regression. Two faces of one photo both match Ann; the resolver
    keeps one. On the current code the second night re-applies and re-strips
    the same label, which is the 6134/6134 in the log."""
    from photosearch.faces import match_faces_to_persons
    from photosearch.maintenance import _stage_resolve_dups

    first_match = match_faces_to_persons(mdb)
    assert first_match >= 2
    first_resolve = _stage_resolve_dups(mdb, True, lambda e: None, lambda: False)
    assert first_resolve["applied"] == 1

    second_match = match_faces_to_persons(mdb)
    second_resolve = _stage_resolve_dups(mdb, True, lambda e: None, lambda: False)

    assert second_match == 0, "the matcher re-applied the de-duplicated label"
    assert second_resolve["applied"] == 0, "the resolver stripped it again"

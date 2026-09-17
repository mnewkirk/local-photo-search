"""Bulk-removing one person's labels from a whole match_source."""
import json

import numpy as np
import pytest

from photosearch import bulk_unmatch
from photosearch.db import PhotoDB
from photosearch.faces import REJECTED_MATCH_SOURCE


def _vec(seed, dim=512):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near(base, seed, spread=0.02):
    """Calibrated: spread 0.02 lands ~0.42 from `base`, while an unrelated
    _vec() lands ~1.40 — so a 1.0 gate cleanly separates them. Raising it much
    past 0.05 stops meaning "same person" at all."""
    rng = np.random.RandomState(seed)
    v = np.asarray(base, np.float32) + spread * rng.randn(len(base)).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def udb(tmp_path):
    db = PhotoDB(str(tmp_path / "unmatch.db"))
    if not db.conn.execute("SELECT 1 FROM sqlite_master "
                           "WHERE name='face_encodings'").fetchone():
        pytest.skip("sqlite-vec unavailable")
    db._pat = db.add_person("Pat")
    db._base = _vec(1)
    db._n = 0

    def face(enc, source, day=1, sec=0):
        db._n += 1
        ph = db.add_photo(filepath=f"/p/{db._n}.jpg", filename=f"{db._n}.jpg",
                          date_taken=f"2026-05-{day:02d} 10:00:{sec:02d}")
        fid = db.add_face(ph, (10, 90, 90, 10), enc, det_score=0.9)
        db.assign_face_to_person(fid, db._pat, match_source=source)
        return fid
    db._face = face

    # Six hand-made references, minutes apart so no burst exclusion.
    for k in range(6):
        face(_near(db._base, 100 + k), "manual", day=1 + k)
    db.conn.commit()
    yield db
    db.close()


def test_selects_only_the_named_source(udb):
    good = udb._face(_near(udb._base, 7), "strict", day=20)
    bad = udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()

    rows, stats = bulk_unmatch.select(udb, person="Pat")
    assert [r["face_id"] for r in rows] == [bad]
    assert good not in [r["face_id"] for r in rows]
    assert stats["references"] == 6


def test_cleared_faces_are_rejected_not_null(udb):
    """A plain NULL is indistinguishable from never-matched, so the next
    match-faces sweep puts the same wrong label straight back."""
    bad = udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()

    rows, _ = bulk_unmatch.select(udb, person="Pat")
    assert bulk_unmatch.apply(udb, rows) == 1
    row = udb.conn.execute("SELECT person_id, match_source FROM faces WHERE id=?",
                           (bad,)).fetchone()
    assert row["person_id"] is None
    assert row["match_source"] == REJECTED_MATCH_SOURCE


def test_min_dist_keeps_the_ones_the_bad_source_got_right(udb):
    """temporal measured ~4% accurate, not 0% — a blanket clear loses those."""
    right = udb._face(_near(udb._base, 42), "temporal", day=22)
    wrong = udb._face(_vec(999), "temporal", day=23)
    udb.conn.commit()

    rows, stats = bulk_unmatch.select(udb, person="Pat", min_dist=1.0)
    ids = [r["face_id"] for r in rows]
    assert wrong in ids and right not in ids
    assert stats["candidates"] == 2 and stats["selected"] == 1


def test_unmeasurable_faces_survive_a_gate(udb):
    """'unknown' is not 'far'. A face with no usable reference must not be
    cleared by a filter that is a claim about distance."""
    orphan = udb._face(_vec(555), "temporal", day=1, sec=0)   # same burst as a ref
    udb.conn.execute("UPDATE faces SET person_id=NULL WHERE match_source='manual'")
    udb.conn.commit()
    rows, _ = bulk_unmatch.select(udb, person="Pat", min_dist=0.5)
    assert orphan not in [r["face_id"] for r in rows]


def test_date_scope(udb):
    inside = udb._face(_vec(998), "temporal", day=15)
    outside = udb._face(_vec(997), "temporal", day=25)
    udb.conn.commit()
    rows, _ = bulk_unmatch.select(udb, person="Pat",
                                  date_from="2026-05-14", date_to="2026-05-16")
    assert [r["face_id"] for r in rows] == [inside]
    assert outside not in [r["face_id"] for r in rows]


def test_restore_puts_back_exactly_the_snapshot(udb):
    bad = udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()
    rows, _ = bulk_unmatch.select(udb, person="Pat")
    bulk_unmatch.apply(udb, rows)

    restored, skipped = bulk_unmatch.restore(udb, [
        {"face_id": r["face_id"], "person_id": r["person_id"],
         "match_source": r["match_source"]} for r in rows])
    assert (restored, skipped) == (1, 0)
    row = udb.conn.execute("SELECT person_id, match_source FROM faces WHERE id=?",
                           (bad,)).fetchone()
    assert row["person_id"] == udb._pat and row["match_source"] == "temporal"


def test_restore_never_overwrites_a_hand_label(udb):
    """The snapshot is an undo, not a replay: re-applying a temporal guess over
    a human correction made since the sweep would be a silent regression."""
    bad = udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()
    rows, _ = bulk_unmatch.select(udb, person="Pat")
    snap = [{"face_id": r["face_id"], "person_id": r["person_id"],
             "match_source": r["match_source"]} for r in rows]
    bulk_unmatch.apply(udb, rows)

    other = udb.add_person("Quinn")
    udb.assign_face_to_person(bad, other, match_source="manual")
    udb.conn.commit()

    restored, skipped = bulk_unmatch.restore(udb, snap)
    assert (restored, skipped) == (0, 1)
    row = udb.conn.execute("SELECT person_id, match_source FROM faces WHERE id=?",
                           (bad,)).fetchone()
    assert row["person_id"] == other and row["match_source"] == "manual"


def test_apply_requires_a_snapshot(udb, tmp_path):
    """--apply without --snapshot is refused: the snapshot is the only precise
    undo, and restore-unmatched-faces would restore every unmatch ever made."""
    from click.testing import CliRunner
    import cli as cli_mod
    udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()
    path = udb.conn.execute("PRAGMA database_list").fetchone()["file"]
    res = CliRunner().invoke(cli_mod.cli, [
        "unmatch-person", "--db", path, "--person", "Pat", "--apply"])
    assert res.exit_code != 0
    assert "--snapshot" in res.output


def test_cli_roundtrip_writes_a_usable_snapshot(udb, tmp_path):
    from click.testing import CliRunner
    import cli as cli_mod
    bad = udb._face(_vec(999), "temporal", day=21)
    udb.conn.commit()
    path = udb.conn.execute("PRAGMA database_list").fetchone()["file"]
    udb.close()
    snap = str(tmp_path / "snap.json")

    r = CliRunner().invoke(cli_mod.cli, [
        "unmatch-person", "--db", path, "--person", "Pat",
        "--snapshot", snap, "--apply"])
    assert r.exit_code == 0, r.output
    assert json.load(open(snap))["faces"][0]["face_id"] == bad

    r2 = CliRunner().invoke(cli_mod.cli, [
        "restore-unmatch", "--db", path, "--from", snap, "--apply"])
    assert r2.exit_code == 0, r2.output
    assert "Restored 1" in r2.output

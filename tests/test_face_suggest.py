""""More of this kid" — ranked, never-written candidate faces for one person."""
import numpy as np
import pytest

from photosearch import face_suggest
from photosearch.db import PhotoDB
from photosearch.faces import REJECTED_MATCH_SOURCE


def _vec(seed, dim=512):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near(base, seed, spread=0.02):
    """~0.42 from `base`; an unrelated _vec() lands ~1.40 (see test_bulk_unmatch)."""
    rng = np.random.RandomState(seed)
    v = np.asarray(base, np.float32) + spread * rng.randn(len(base)).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def sdb(tmp_path):
    db = PhotoDB(str(tmp_path / "suggest.db"))
    if not db.conn.execute("SELECT 1 FROM sqlite_master "
                           "WHERE name='face_encodings'").fetchone():
        pytest.skip("sqlite-vec unavailable")
    db._n = 0

    def face(enc, person_id=None, source=None, day="2026-09-26", photo=None):
        if photo is None:
            db._n += 1
            photo = db.add_photo(filepath=f"/p/{db._n}.jpg", filename=f"{db._n}.jpg",
                                 date_taken=f"{day} 10:00:00")
        fid = db.add_face(photo, (10, 90, 90, 10), enc, det_score=0.9)
        if person_id is not None:
            db.assign_face_to_person(fid, person_id, match_source=source)
        return fid, photo
    db._face = face
    db._kid = db.add_person("Kid")
    db._other = db.add_person("Other")
    db._kbase, db._obase = _vec(1), _vec(2)
    face(_near(db._kbase, 10), db._kid, "manual")
    face(_near(db._kbase, 11), db._kid, "manual", day="2025-06-01")  # older shoot
    face(_near(db._obase, 20), db._other, "manual")
    db.conn.commit()
    yield db
    db.close()


def test_ranks_the_days_unmatched_faces_nearest_first(sdb):
    near, _ = sdb._face(_near(sdb._kbase, 30))
    far, _ = sdb._face(_vec(999))
    off_day, _ = sdb._face(_near(sdb._kbase, 31), day="2026-09-20")
    sdb.conn.commit()

    rows, stats = face_suggest.suggest(sdb, person="Kid",
                                       date_from="2026-09-26", date_to="2026-09-26")
    ids = [r["face_id"] for r in rows]
    assert ids == [near, far]                       # nearest first, scoped to the day
    assert off_day not in ids
    assert stats["references"] == 2                  # references are library-wide
    assert rows[0]["dist"] < 0.6 < rows[1]["dist"]


def test_never_writes(sdb):
    fid, _ = sdb._face(_near(sdb._kbase, 30))
    sdb.conn.commit()
    face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26")
    assert sdb.conn.execute("SELECT person_id FROM faces WHERE id=?",
                            (fid,)).fetchone()[0] is None


def test_max_dist_is_the_tunable_cutoff(sdb):
    near, _ = sdb._face(_near(sdb._kbase, 30))
    sdb._face(_vec(999))
    sdb.conn.commit()
    rows, stats = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26",
                                       max_dist=0.8)
    assert [r["face_id"] for r in rows] == [near]
    assert stats["within"]["0.8"] == 1 and stats["measured"] == 2


def test_a_face_closer_to_another_labelled_kid_is_flagged(sdb):
    """On a field of lookalikes a candidate can be nearer a rival than to the
    person — flagged by default, dropped on request, never silently kept."""
    theirs, _ = sdb._face(_near(sdb._obase, 40))
    sdb.conn.commit()
    rows, _ = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26")
    row = next(r for r in rows if r["face_id"] == theirs)
    assert row["closer_to_other"] and row["rival"] == "Other"
    rows, _ = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26",
                                   exclude_closer_to_other=True)
    assert theirs not in [r["face_id"] for r in rows]


def test_a_rival_labelled_on_another_shoot_still_counts(sdb):
    lookalike = sdb.add_person("Teammate")
    tbase = _vec(3)
    sdb._face(_near(tbase, 70), lookalike, "manual", day="2025-01-01")
    theirs, _ = sdb._face(_near(tbase, 71))
    sdb.conn.commit()
    rows, _ = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26")
    row = next(r for r in rows if r["face_id"] == theirs)
    assert row["closer_to_other"] and row["rival"] == "Teammate"


def test_person_already_in_the_photo_is_flagged(sdb):
    _, photo = sdb._face(_near(sdb._kbase, 50), sdb._kid, "manual")
    second, _ = sdb._face(_near(sdb._kbase, 51), photo=photo)
    sdb.conn.commit()
    rows, _ = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26")
    assert next(r for r in rows if r["face_id"] == second)["in_photo"] is True


def test_rejected_and_labelled_faces_are_not_candidates(sdb):
    rej, _ = sdb._face(_near(sdb._kbase, 60))
    sdb.conn.execute("UPDATE faces SET match_source=? WHERE id=?",
                     (REJECTED_MATCH_SOURCE, rej))
    sdb.conn.commit()
    rows, _ = face_suggest.suggest(sdb, person="Kid", date_from="2026-09-26")
    ids = [r["face_id"] for r in rows]
    assert rej not in ids
    assert all(r["face_id"] not in ids for r in sdb.conn.execute(
        "SELECT id AS face_id FROM faces WHERE person_id IS NOT NULL").fetchall())


def test_unknown_person_raises(sdb):
    with pytest.raises(ValueError):
        face_suggest.suggest(sdb, person="Nobody", date_from="2026-09-26")


def test_endpoint_requires_a_date_scope():
    from fastapi.testclient import TestClient
    from photosearch.web import app
    r = TestClient(app).get("/api/faces/suggest-person?person=Kid")
    assert r.status_code == 400

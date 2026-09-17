"""The rival test: another face in the SAME photo claims the label better.

A person appears in a photo at most once, so a better claimant sitting in the
same frame settles the question — and unlike the margin test, the claimant does
not have to be a registered person. That is the case this exists for: on photo
244260 (2026-09-12) "Franklin" was on the wrong boy and the real Franklin was
UNLABELLED two faces to the right, so no comparison against named people could
have named the right face.
"""
import numpy as np
import pytest

from photosearch import face_verify
from photosearch.db import PhotoDB


def _vec(seed, dim=512):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _near(base, seed, spread=0.12):
    """A vector near `base` — same identity, different frame."""
    rng = np.random.RandomState(seed)
    v = np.asarray(base, dtype=np.float32) + spread * rng.randn(len(base)).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def rival_db(tmp_path):
    """Two people with reference sets, plus one photo where P's label sits on
    Q's face and P's own (unlabelled) face is right beside it."""
    db = PhotoDB(str(tmp_path / "rivals.db"))
    if not db.conn.execute("SELECT 1 FROM sqlite_master "
                           "WHERE name='face_encodings'").fetchone():
        pytest.skip("sqlite-vec unavailable; encodings cannot be stored")

    ident = {"Pat": _vec(1), "Quinn": _vec(2)}
    pid = {n: db.add_person(n) for n in ident}
    db._ident, db._person_id, db._photo = ident, pid, {}

    def photo(day, hh, mm, ss, tag=""):
        return db.add_photo(filepath=f"/p/{day}-{hh}-{mm}-{ss}{tag}.jpg",
                            filename=f"{day}.jpg",
                            date_taken=f"2026-05-{day:02d} {hh:02d}:{mm:02d}:{ss:02d}")
    db._mk_photo = photo

    # Reference sets: one face per photo, minutes apart so no burst exclusion,
    # and a photo per PERSON — sharing them would put both people in every
    # reference frame and quietly break the one-person-per-photo premise the
    # rival test rests on.
    # Seeds are positional, NOT hash(name): PYTHONHASHSEED is randomized per
    # process, so a hash-derived seed makes the whole fixture — and therefore
    # every distance in these tests — different on every run.
    for pi, (n, base) in enumerate(sorted(ident.items())):
        for k in range(12):
            ph = photo(1, 10, k * 5, 0, tag="-" + n)
            f = db.add_face(ph, (10, 90, 90, 10), _near(base, 300 + pi * 100 + k),
                            det_score=0.9)
            db.assign_face_to_person(f, pid[n], match_source="manual")
    db.conn.commit()
    yield db
    db.close()


def _findings(db, **kw):
    f, skipped, stats = face_verify.verify_labels(
        db, date_from="2026-05-01", date_to="2026-05-31", **kw)
    return f, stats


def test_label_on_the_wrong_face_is_flagged_with_the_right_one(rival_db):
    """The motivating shape: P's label on Q's face, P's real face UNLABELLED."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    wrong = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 77), det_score=0.9)
    db.assign_face_to_person(wrong, db._person_id["Pat"], match_source="manual")
    real = db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Pat"], 88), det_score=0.9)
    db.conn.commit()

    findings, stats = _findings(db)
    hit = [f for f in findings if f["face_id"] == wrong]
    assert hit, "the mislabelled face should be flagged"
    r = hit[0]["rival"]
    assert r is not None
    assert r["rival_face_id"] == real
    assert r["rival_label"] is None          # the point: an unlabelled claimant
    assert r["d_rival"] < r["radius"] < r["d_self"]
    assert r["displaced"] > 0
    assert stats["rivals"] == 1


def test_rivals_sort_first(rival_db):
    """A rival names the right face, so it outranks findings that only accuse."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    wrong = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 77), det_score=0.9)
    db.assign_face_to_person(wrong, db._person_id["Pat"], match_source="manual")
    db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Pat"], 88), det_score=0.9)
    # A second, rival-free mislabel elsewhere. 'temporal' so it stays out of
    # Pat's reference set — a manual one would be a reference, and this test is
    # about ordering, not about how much pollution the calibration tolerates.
    ph2 = db._mk_photo(3, 12, 0, 0)
    lone = db.add_face(ph2, (10, 90, 90, 10), _near(db._ident["Quinn"], 55), det_score=0.9)
    db.assign_face_to_person(lone, db._person_id["Pat"], match_source="temporal")
    db.conn.commit()

    findings, _ = _findings(db)
    assert findings[0]["face_id"] == wrong
    assert findings[0]["rival"] is not None


def test_correct_label_beside_a_stranger_is_left_alone(rival_db):
    """The false-positive that matters: a CORRECT label must survive a photo-mate."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    good = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Pat"], 91), det_score=0.9)
    db.assign_face_to_person(good, db._person_id["Pat"], match_source="manual")
    db.add_face(ph, (10, 190, 90, 110), _vec(4242), det_score=0.9)   # unrelated person
    db.conn.commit()

    findings, stats = _findings(db)
    assert stats["rivals"] == 0
    assert all(f["rival"] is None for f in findings)


def test_rival_that_is_better_explained_by_someone_else_is_not_stolen(rival_db):
    """Gate 4. Q's own correctly-labelled face must not be taken for P."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    wrong = db.add_face(ph, (10, 90, 90, 10), _vec(999), det_score=0.9)   # nobody
    db.assign_face_to_person(wrong, db._person_id["Pat"], match_source="manual")
    q = db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Quinn"], 33), det_score=0.9)
    db.assign_face_to_person(q, db._person_id["Quinn"], match_source="manual")
    db.conn.commit()

    findings, _ = _findings(db)
    hit = [f for f in findings if f["face_id"] == wrong]
    assert all(f["rival"] is None for f in hit), \
        "Quinn's own face is not a candidate for Pat's label"


def test_one_rival_cannot_satisfy_two_labels(rival_db):
    """The assignment is one-to-one — a per-label argmin would hand the same
    face to both, and 'swap' would then write two people onto one face."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    a = db.add_face(ph, (10, 90, 90, 10), _vec(4001), det_score=0.9)
    b = db.add_face(ph, (10, 190, 90, 110), _vec(4002), det_score=0.9)
    db.assign_face_to_person(a, db._person_id["Pat"], match_source="manual")
    db.assign_face_to_person(b, db._person_id["Quinn"], match_source="manual")
    # One face that is a plausible Pat AND the only other face present.
    shared = db.add_face(ph, (10, 290, 90, 210), _near(db._ident["Pat"], 64), det_score=0.9)
    db.conn.commit()

    findings, _ = _findings(db)
    claimed = [f["rival"]["rival_face_id"] for f in findings if f["rival"]]
    assert claimed.count(shared) <= 1


def test_burst_siblings_are_linked(rival_db):
    """The same mistake in consecutive frames is grouped, so it is one action."""
    db = rival_db
    faces = []
    for sec in (0, 1):
        ph = db._mk_photo(2, 12, 0, sec)
        w = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 70 + sec), det_score=0.9)
        db.assign_face_to_person(w, db._person_id["Pat"], match_source="manual")
        db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Pat"], 80 + sec), det_score=0.9)
        faces.append(w)
    db.conn.commit()

    findings, _ = _findings(db)
    hits = {f["face_id"]: f for f in findings if f["rival"]}
    assert set(faces) <= set(hits), "both frames should be flagged independently"
    sibs = [b["face_id"] for b in hits[faces[0]]["rival"]["burst_siblings"]]
    assert faces[1] in sibs


def test_rivals_can_be_switched_off(rival_db):
    """rivals=False must not merely hide them — it must skip the extra scan."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    wrong = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 77), det_score=0.9)
    db.assign_face_to_person(wrong, db._person_id["Pat"], match_source="manual")
    db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Pat"], 88), det_score=0.9)
    db.conn.commit()

    findings, _ = _findings(db, rivals=False)
    assert all(f["rival"] is None for f in findings)


# ---------------------------------------------------------------------------
# Which labels you are ASKED about (see TRUSTED_LABEL_SOURCES).
# ---------------------------------------------------------------------------

def test_temporal_labels_are_not_adjudicated_by_default(rival_db):
    """A panel of per-face verdicts is the wrong shape for a ~4%-accurate
    matcher: on 2026-09-12 temporal buried the real findings 76:1."""
    db = rival_db
    ph = db._mk_photo(3, 12, 0, 0)
    t = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 55), det_score=0.9)
    db.assign_face_to_person(t, db._person_id["Pat"], match_source="temporal")
    db.conn.commit()

    findings, stats = _findings(db)
    assert all(f["face_id"] != t for f in findings)
    assert stats["hidden_by_source"] == {"temporal": 1}
    assert stats["hidden_by_person"] == {"Pat": 1}


def test_hidden_labels_are_counted_not_silently_dropped(rival_db):
    """The caller must be able to say what it is not showing."""
    db = rival_db
    for k in range(3):
        ph = db._mk_photo(4, 12, k, 0)
        t = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 200 + k), det_score=0.9)
        db.assign_face_to_person(t, db._person_id["Pat"], match_source="temporal")
    db.conn.commit()

    # Anchored to the unfiltered call rather than a literal: the count must be
    # exactly what the caller would have been shown, whatever that is.
    every, _ = _findings(db, label_sources=None)
    expected = sum(1 for f in every if f["match_source"] == "temporal")
    _, stats = _findings(db)
    assert expected == 3, "fixture should produce a finding for each temporal face"
    assert stats["hidden_by_source"]["temporal"] == expected


def test_sources_none_reports_everything(rival_db):
    db = rival_db
    ph = db._mk_photo(3, 12, 0, 0)
    t = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 55), det_score=0.9)
    db.assign_face_to_person(t, db._person_id["Pat"], match_source="temporal")
    db.conn.commit()

    findings, stats = _findings(db, label_sources=None)
    assert any(f["face_id"] == t for f in findings)
    assert stats["hidden_by_source"] == {}


def test_hiding_a_source_does_not_change_the_numbers(rival_db):
    """Filtering happens LAST. A hidden temporal face still competes in its
    photo's assignment and still shapes the calibration — otherwise the answer
    would depend on which rows you asked to see."""
    db = rival_db
    ph = db._mk_photo(2, 12, 0, 0)
    wrong = db.add_face(ph, (10, 90, 90, 10), _near(db._ident["Quinn"], 77), det_score=0.9)
    db.assign_face_to_person(wrong, db._person_id["Pat"], match_source="manual")
    db.add_face(ph, (10, 190, 90, 110), _near(db._ident["Pat"], 88), det_score=0.9)
    ph2 = db._mk_photo(5, 12, 0, 0)
    t = db.add_face(ph2, (10, 90, 90, 10), _near(db._ident["Quinn"], 55), det_score=0.9)
    db.assign_face_to_person(t, db._person_id["Pat"], match_source="temporal")
    db.conn.commit()

    trusted = {f["face_id"]: f for f in _findings(db)[0]}
    every = {f["face_id"]: f for f in _findings(db, label_sources=None)[0]}
    assert trusted[wrong]["d_own"] == every[wrong]["d_own"]
    assert trusted[wrong]["rival"] == every[wrong]["rival"]

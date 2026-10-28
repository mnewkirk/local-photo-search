"""Owner decisions 2026-10-28 for video faces.

1. Margin rule: the strict matcher names a face of a `media_type='video'` row
   only when its best person is within MATCH_TOLERANCE AND the nearest OTHER
   person is >= VIDEO_MATCH_MARGIN farther (round-2 eval: identity precision
   0.33 -> 0.97 dev). Stills are unchanged. A barred best (face_person_exclusions)
   falls through to the next best, which must pass tolerance + margin against
   the REMAINING field.
2. Presence: a person is present in a video only with >= VIDEO_PRESENCE_MIN_S
   (2 s) of tracked presence, summed over their identities. Applies to the
   person search filter, the subjects endpoint and the eval. `faces` rows are
   unchanged.
"""

import math
import struct

import numpy as np
import pytest

from photosearch import faces as F
from photosearch.db import FACE_DIMENSIONS, PhotoDB
from photosearch.search import search_combined
from photosearch.video_presence import (VIDEO_PRESENCE_MIN_S, present_face_sql,
                                        present_people)


def _vec(*head):
    v = [0.0] * FACE_DIMENSIONS
    for i, x in enumerate(head):
        v[i] = float(x)
    return v


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _add_reference(db, person_id, enc):
    cur = db.conn.execute(
        "INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
        (person_id, "test"))
    db.conn.execute(
        "INSERT INTO face_ref_encodings (ref_id, encoding) VALUES (?, ?)",
        (cur.lastrowid, struct.pack(f"{FACE_DIMENSIONS}f", *enc)))


ANN, BEA, CAT = _vec(1), _vec(0, 1), _vec(0, 0, 1)
CLOSE = _vec(0.5, 0.45, 0.6)      # Ann 0.901, Bea 0.955: gap 0.054
CLEAR = _vec(0.9, 0.1, 0.3)       # Ann 0.33, Bea 1.31
NEAR_AB = _vec(0.6, 0.5, 0.1)     # Ann 0.648, Bea 0.787, Cat 1.19
NEAR_ABC = _vec(0.6, 0.5, 0.45)   # Ann 0.783, Bea 0.901, Cat 0.955


@pytest.fixture
def mdb(tmp_path):
    db = PhotoDB(str(tmp_path / "m.db"))
    still = db.add_photo(filepath="2026/a.jpg", filename="a.jpg",
                         date_taken="2026-09-12 10:00:00")
    video = db.add_photo(filepath="2026/v.mp4", filename="v.mp4", media_type="video",
                         date_taken="2026-09-12 10:01:00")
    ann, bea, cat = db.add_person("Ann"), db.add_person("Bea"), db.add_person("Cat")
    _add_reference(db, ann, ANN)
    _add_reference(db, bea, BEA)
    db.conn.commit()
    db.ids = {"still": still, "video": video, "ann": ann, "bea": bea, "cat": cat}
    yield db
    db.close()


def _person(db, face_id):
    return db.conn.execute("SELECT person_id FROM faces WHERE id = ?",
                           (face_id,)).fetchone()[0]


# ---------------------------------------------------------------------------
# the pure rule
# ---------------------------------------------------------------------------

def test_rule_needs_tolerance_and_margin():
    assert F.pick_person_with_margin({1: 0.5, 2: 0.7}, None, 1.15, 0.15) == 1
    assert F.pick_person_with_margin({1: 0.5, 2: 0.6}, None, 1.15, 0.15) is None
    assert F.pick_person_with_margin({1: 1.2, 2: 1.9}, None, 1.15, 0.15) is None


def test_no_other_person_passes_the_margin():
    """One person with references (or every rival barred) cannot be ambiguous."""
    assert F.pick_person_with_margin({1: 0.9}, None, 1.15, 0.15) == 1
    assert F.pick_person_with_margin({1: 0.9, 2: 0.91}, {2}, 1.15, 0.15) == 1


def test_barred_best_falls_through_judged_against_the_remaining_field():
    d = {1: 0.648, 2: 0.787, 3: 1.19}
    assert F.pick_person_with_margin(d, None, 1.15, 0.15) is None   # 1 vs 2 too close
    assert F.pick_person_with_margin(d, {1}, 1.15, 0.15) == 2       # 2 vs 3: 0.40
    assert F.pick_person_with_margin({1: 0.78, 2: 0.90, 3: 0.955},
                                     {1}, 1.15, 0.15) is None       # 2 vs 3: 0.055


# ---------------------------------------------------------------------------
# the matcher
# ---------------------------------------------------------------------------

def test_a_video_face_needs_the_margin_a_still_does_not(mdb):
    assert _dist(CLOSE, BEA) - _dist(CLOSE, ANN) < F.VIDEO_MATCH_MARGIN
    s_close = mdb.add_face(mdb.ids["still"], (0, 10, 10, 0), CLOSE)
    v_close = mdb.add_face(mdb.ids["video"], (0, 10, 10, 0), CLOSE)
    v_clear = mdb.add_face(mdb.ids["video"], (20, 30, 30, 20), CLEAR)
    mdb.conn.commit()
    assert F.match_faces_to_persons(mdb) == 2
    assert _person(mdb, s_close) == mdb.ids["ann"]
    assert _person(mdb, v_close) is None
    assert _person(mdb, v_clear) == mdb.ids["ann"]
    src = mdb.conn.execute("SELECT match_source FROM faces WHERE id = ?",
                           (v_clear,)).fetchone()[0]
    assert src == "strict"


def _legacy_strict(db, tolerance=F.MATCH_TOLERANCE):
    """The pre-margin matcher, verbatim in effect: nearest reference within
    tolerance, walking past barred persons."""
    from photosearch.db import load_face_person_exclusions
    refs = F._load_reference_encodings(db.conn)
    excl = load_face_person_exclusions(db.conn)
    out = {}
    for (fid,) in db.conn.execute(f"SELECT f.id FROM faces f WHERE {F.MATCHABLE_SQL}"):
        enc = struct.unpack(f"{FACE_DIMENSIONS}f", db.conn.execute(
            "SELECT encoding FROM face_encodings WHERE face_id = ?", (fid,)).fetchone()[0])
        for i, d in F.match_face(list(enc), [r[2] for r in refs], tolerance):
            if refs[i][0] in excl.get(fid, set()):
                continue
            out[fid] = refs[i][0]
            break
    return out


def test_still_matching_is_byte_identical_to_the_legacy_rule(mdb):
    from photosearch.db import record_face_person_exclusions
    _add_reference(mdb, mdb.ids["cat"], CAT)
    rng = np.random.default_rng(7)
    ids = []
    for k in range(60):
        v = np.zeros(FACE_DIMENSIONS)
        v[:3] = rng.uniform(0, 1, 3)
        v[3] = rng.uniform(0, 0.4)
        ids.append(mdb.add_face(mdb.ids["still"], (k, 10, 10, 0), list(v)))
    record_face_person_exclusions(mdb.conn, [(ids[0], mdb.ids["ann"]),
                                             (ids[1], mdb.ids["bea"])], reason="test")
    mdb.conn.commit()
    expected = _legacy_strict(mdb)
    before = mdb.conn.execute("SELECT * FROM faces ORDER BY id").fetchall()
    F.match_faces_to_persons(mdb)
    got = {r[0]: r[1] for r in mdb.conn.execute(
        "SELECT id, person_id FROM faces WHERE person_id IS NOT NULL")}
    assert got == expected and len(expected) > 20
    after = mdb.conn.execute("SELECT * FROM faces ORDER BY id").fetchall()
    for b, a in zip(before, after):
        if a["id"] not in expected:
            assert tuple(a) == tuple(b)


def test_video_exclusion_falls_through_to_the_next_best(mdb):
    from photosearch.db import record_face_person_exclusions
    _add_reference(mdb, mdb.ids["cat"], CAT)
    ab = mdb.add_face(mdb.ids["video"], (0, 10, 10, 0), NEAR_AB)
    abc = mdb.add_face(mdb.ids["video"], (20, 30, 30, 20), NEAR_ABC)
    plain = mdb.add_face(mdb.ids["video"], (40, 50, 50, 40), NEAR_AB)
    record_face_person_exclusions(mdb.conn, [(ab, mdb.ids["ann"]),
                                             (abc, mdb.ids["ann"])], reason="test")
    mdb.conn.commit()
    F.match_faces_to_persons(mdb)
    assert _person(mdb, ab) == mdb.ids["bea"]       # Bea beats Cat by 0.40
    assert _person(mdb, abc) is None                # Bea vs Cat: 0.054
    assert _person(mdb, plain) is None              # Ann vs Bea: 0.139


def test_report_counts_without_writing(mdb, tmp_path):
    named = mdb.add_face(mdb.ids["video"], (0, 10, 10, 0), CLOSE)
    mdb.conn.execute("UPDATE faces SET person_id = ?, match_source = 'strict' WHERE id = ?",
                     (mdb.ids["ann"], named))
    mdb.add_face(mdb.ids["video"], (20, 30, 30, 20), CLOSE)
    mdb.add_face(mdb.ids["video"], (40, 50, 50, 40), CLEAR)
    mdb.add_face(mdb.ids["still"], (0, 10, 10, 0), CLOSE)
    mdb.conn.commit()
    r = F.video_match_margin_report(mdb.conn)
    assert r["video_faces"] == 3 and r["named"] == {"strict": 1}
    assert r["strict_would_unname"] == 1 and r["strict_keep"] == 0
    assert (r["unnamed_matchable"], r["legacy_would_name"], r["margin_would_name"],
            r["left_unnamed_by_margin"]) == (2, 2, 1, 1)
    assert mdb.conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL"
                            ).fetchone()[0] == 1


def test_report_cli_is_read_only(mdb):
    from click.testing import CliRunner
    from cli import cli
    path = mdb.db_path
    res = CliRunner().invoke(cli, ["video-match-margin-report", "--db", path])
    assert res.exit_code == 0, res.output
    assert "video faces: 0" in res.output


# ---------------------------------------------------------------------------
# presence >= 2 s
# ---------------------------------------------------------------------------

def _track(db, vid, face_id, presence_s, ix):
    db.conn.execute("INSERT INTO video_tracks (photo_id, face_id, ident_ix, presence_s, n_det) "
                    "VALUES (?, ?, ?, ?, 4)", (vid, face_id, ix, presence_s))


@pytest.fixture
def pres(tmp_path):
    db = PhotoDB(str(tmp_path / "p.db"))
    kid = db.add_person("Kid")
    still = db.add_photo(filepath="2026/s.jpg", filename="s.jpg",
                         date_taken="2026-09-26 09:00:00")
    db.add_face(still, (0, 10, 10, 0), [], person_id=kid)
    ids = {"kid": kid, "still": still}
    # brief: one 1.5 s identity. split: two 1.2 s identities (2.4 s summed).
    # long: 5 s. Each video also has an unnamed stranger.
    for name, pres_list in (("brief", [1.5]), ("split", [1.2, 1.2]), ("long", [5.0])):
        vid = db.add_photo(filepath=f"2026/{name}.mp4", filename=f"{name}.mp4",
                           media_type="video", date_taken="2026-09-26 10:00:00")
        db.conn.execute("INSERT INTO video_meta (photo_id, probe_version, kind, duration_s) "
                        "VALUES (?, 1, 'clip', 30)", (vid,))
        for ix, p in enumerate(pres_list):
            fid = db.add_face(vid, (ix, 10, 10, 0), [], person_id=kid)
            _track(db, vid, fid, p, ix)
        stranger = db.add_face(vid, (90, 99, 99, 90), [])
        _track(db, vid, stranger, 9.0, 9)
        ids[name] = vid
    db.conn.commit()
    db.ids = ids
    yield db
    db.close()


def test_default_floor_is_two_seconds():
    assert VIDEO_PRESENCE_MIN_S == 2.0
    assert present_people([{"person_id": 1, "presence_s": 1.2},
                           {"person_id": 1, "presence_s": 0.8},
                           {"person_id": 2, "presence_s": 1.9},
                           {"person_id": None, "presence_s": 50}]) == [1]


@pytest.mark.parametrize("kw", [dict(person="Kid"), dict(person="Kid", media="video"),
                                dict(person="Kid", date_from="2026-09-26",
                                     date_to="2026-09-26"),
                                dict(person_ids=None, query="Kid")])
def test_person_search_needs_two_seconds_in_a_video(pres, kw):
    page, _ = search_combined(pres, limit=100, with_total=True, **kw)
    got = {r["id"] for r in page}
    want = {pres.ids["split"], pres.ids["long"]}
    if kw.get("media") != "video":
        want.add(pres.ids["still"])
    assert got == want


def test_ask_tools_person_filter_needs_two_seconds(pres):
    from photosearch import tools
    where, params = tools._build_filter_sql(pres, {"people": ["Kid"], "media": "all"})
    ids = {r[0] for r in pres.conn.execute(
        f"SELECT photos.id FROM photos WHERE {where or '1'}", params)}
    assert pres.ids["brief"] not in ids
    assert {pres.ids["split"], pres.ids["long"], pres.ids["still"]} <= ids


def test_presence_predicate_is_index_driven(pres):
    plan = " | ".join(r[3] for r in pres.conn.execute(
        "EXPLAIN QUERY PLAN SELECT f.photo_id FROM faces f WHERE f.person_id = 1 AND "
        + present_face_sql("f")))
    assert "idx_faces_person_photo" in plan, plan
    assert "idx_video_tracks_face" in plan, plan
    # the per-person sum is a two-column point seek on faces
    assert ("(person_id=? AND photo_id=?)" in plan
            or "(photo_id=? AND person_id=?)" in plan), plan
    assert "SCAN video_tracks" not in plan and "SCAN fp" not in plan, plan


def test_subjects_endpoint_lists_present_people(pres):
    from photosearch.video_subjects_store import read_subjects
    assert read_subjects(pres.conn, pres.ids["brief"])["present"] == []
    got = read_subjects(pres.conn, pres.ids["split"])
    assert [(p["name"], p["presence_s"]) for p in got["present"]] == [("Kid", 2.4)]
    assert got["presence_min_s"] == 2.0


def test_eval_presence_rederives_from_identities():
    import sys
    sys.path.insert(0, "evals")
    import video_eval as ve
    it = {"present": [1, 2], "identities": [
        {"person_id": 1, "presence_s": 0.5}, {"person_id": 2, "presence_s": 3.0}]}
    assert ve.item_present(it) == [2]
    assert ve.item_present(it, 0.0) == [1, 2]
    assert ve.item_present({"present": [7]}) == [7]

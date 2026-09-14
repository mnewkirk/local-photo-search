"""Tests for /api/faces/person/{id}/inspect — the person-audit view.

This is the tool for "X is tagged on other kids at Saturday's match": it
sub-clusters one person's faces and scores each by distance to that person's
core, so wrong faces can be bulk-unassigned. The date scope (added 2026-09-13)
is what makes it usable on a single shoot rather than a whole library.
"""

import numpy as np
import pytest


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _unit(seed: int) -> list:
    """A deterministic unit-norm 512-d encoding. add_face truth-tests its
    `encoding` argument, so this must be a list, not a numpy array."""
    return _vec(seed).tolist()


def _near(seed: int, jitter: float, jseed: int) -> list:
    """A unit vector close to _unit(seed) — same 'person', slightly different."""
    v = _vec(seed) + jitter * _vec(jseed)
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture
def two_day_person(db):
    """A person with trusted faces on 2026-03-13 and suspect faces on 2026-09-12.

    Mirrors the real shape: a solid history, plus one shoot where temporal
    matching tagged other people onto them.
    """
    pid = db.add_person("Casey")
    photos = db._test_photo_ids

    # Day 1 (2026-03-13, the fixture date): 3 trusted faces = a real core.
    day1 = [photos["DSC04878.JPG"], photos["DSC04880.JPG"], photos["DSC04894.JPG"]]
    for i, ph in enumerate(day1):
        fid = db.add_face(ph, (10, 100, 90, 20), _near(1, 0.15, 500 + i), person_id=pid)
        db.assign_face_to_person(fid, pid, "strict")

    # Day 2 (2026-09-12): re-date two photos, then add temporal faces —
    # one genuinely Casey, one obviously a different person.
    day2 = [photos["DSC04907.JPG"], photos["DSC04922.JPG"]]
    for ph in day2:
        db.conn.execute("UPDATE photos SET date_taken = '2026-09-12T14:00:00' "
                        "WHERE id = ?", (ph,))
    same = db.add_face(day2[0], (10, 100, 90, 20), _near(1, 0.2, 600), person_id=pid)
    db.assign_face_to_person(same, pid, "temporal")
    other = db.add_face(day2[1], (10, 100, 90, 20), _unit(999), person_id=pid)
    db.assign_face_to_person(other, pid, "temporal")
    db.conn.commit()
    return {"person_id": pid, "same": same, "other": other,
            "day1_faces": 3, "day2_faces": 2}


def _inspect(client, pid, **params):
    r = client.get(f"/api/faces/person/{pid}/inspect", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_unscoped_returns_every_face_and_keeps_library_defaults(client, two_day_person):
    d = _inspect(client, two_day_person["person_id"])
    assert d["person"]["face_count"] == 5
    assert d["scope"]["in_scope"] == 5 and d["scope"]["total"] == 5
    assert d["scope"]["date_from"] is None
    # The library-wide defaults must not shift just because a scope exists.
    assert d["params"]["eps"] == 0.50
    assert d["params"]["min_samples"] == 3


def test_date_scope_limits_the_faces_reviewed(client, two_day_person):
    d = _inspect(client, two_day_person["person_id"],
                 date_from="2026-09-12", date_to="2026-09-12")
    ids = {f["face_id"] for f in d["faces"]}
    assert ids == {two_day_person["same"], two_day_person["other"]}
    assert d["scope"] == {"date_from": "2026-09-12", "date_to": "2026-09-12",
                          "in_scope": 2, "total": 5}


def test_scoped_call_loosens_the_dbscan_defaults(client, two_day_person):
    """One team on one day is a sparse space: at the library's 0.50 a scoped
    call groups nothing (measured: 212/212 noise on a real shoot)."""
    d = _inspect(client, two_day_person["person_id"], date_from="2026-09-12")
    assert d["params"]["eps"] == 0.90
    assert d["params"]["min_samples"] == 2


def test_explicit_eps_beats_the_adaptive_default(client, two_day_person):
    d = _inspect(client, two_day_person["person_id"],
                 date_from="2026-09-12", eps=0.65, min_samples=3)
    assert d["params"]["eps"] == 0.65
    assert d["params"]["min_samples"] == 3


def test_reference_core_is_NOT_scoped(client, two_day_person):
    """THE invariant. Distances must be measured against the person's full
    trusted set, not the scoped subset.

    Scoping asks "which of these faces aren't really Casey?". Every trusted
    face here is on day 1, so a scoped core would fall back to day 2's own
    faces — i.e. the very faces under suspicion — and the impostor would score
    as core. Pinning dist equality across scoped/unscoped proves the core did
    not move.
    """
    full = _inspect(client, two_day_person["person_id"])
    scoped = _inspect(client, two_day_person["person_id"], date_from="2026-09-12")

    assert full["params"]["ref_source"] == "trusted"
    assert scoped["params"]["ref_source"] == "trusted"

    full_dist = {f["face_id"]: f["dist"] for f in full["faces"]}
    for f in scoped["faces"]:
        assert f["dist"] == pytest.approx(full_dist[f["face_id"]], abs=1e-6), (
            "scoped distances drifted — the reference core was scoped too")

    # And the check has teeth: the impostor really is far from the core.
    scoped_dist = {f["face_id"]: f["dist"] for f in scoped["faces"]}
    assert scoped_dist[two_day_person["other"]] > scoped_dist[two_day_person["same"]]


def test_faces_with_no_date_drop_out_of_a_scoped_review(client, db, two_day_person):
    """A NULL date_taken can't satisfy a date filter, so it must not default
    into the scope — it would be silently bulk-unassigned along with the rest."""
    photos = db._test_photo_ids
    db.conn.execute("UPDATE photos SET date_taken = NULL WHERE id = ?",
                    (photos["DSC04907.JPG"],))
    db.conn.commit()
    d = _inspect(client, two_day_person["person_id"], date_from="2026-09-12")
    assert {f["face_id"] for f in d["faces"]} == {two_day_person["other"]}


def test_empty_scope_is_an_empty_result_not_an_error(client, two_day_person):
    d = _inspect(client, two_day_person["person_id"],
                 date_from="2030-01-01", date_to="2030-01-02")
    assert d["faces"] == [] and d["sub_clusters"] == []
    assert d["scope"]["in_scope"] == 0 and d["scope"]["total"] == 5


def test_scope_reports_the_denominator(client, two_day_person):
    """The panel bulk-unassigns, so it has to be able to say '2 of 5' rather
    than implying the review covered the person's whole library."""
    d = _inspect(client, two_day_person["person_id"], date_from="2026-09-12")
    assert d["scope"]["total"] == 5
    assert d["scope"]["in_scope"] == 2


# ---------------------------------------------------------------------------
# /api/faces/group/{type}/{id}/photos — content filters apply to EVERY group
# type, not just "unclustered" (added 2026-09-13)
# ---------------------------------------------------------------------------

def _photos(client, gtype, gid, **params):
    r = client.get(f"/api/faces/group/{gtype}/{gid}/photos", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_person_photos_respect_the_date_filter(client, two_day_person):
    """The whole point: clicking a name from a one-day /faces view must not
    open that person's entire library. Calvin has 18,633 photos; before this
    the filtered view opened all of them."""
    pid = two_day_person["person_id"]
    all_p = _photos(client, "person", pid)
    day2 = _photos(client, "person", pid, date_from="2026-09-12", date_to="2026-09-12")

    assert len(day2["photos"]) < len(all_p["photos"])
    assert {p["id"] for p in day2["photos"]} < {p["id"] for p in all_p["photos"]}
    for p in day2["photos"]:
        assert (p["date_taken"] or "").startswith("2026-09-12")


def test_response_reports_whether_it_is_filtered_and_the_whole_size(client, two_day_person):
    """The UI needs both, or a narrowed list is indistinguishable from a person
    who simply has few photos."""
    pid = two_day_person["person_id"]
    unfiltered = _photos(client, "person", pid)
    filtered = _photos(client, "person", pid, date_from="2026-09-12")

    assert unfiltered["filtered"] is False
    assert filtered["filtered"] is True
    # total_unfiltered is the WHOLE group either way — it's the denominator.
    assert filtered["total_unfiltered"] == unfiltered["total_unfiltered"]
    assert filtered["total_unfiltered"] >= len(filtered["photos"])


def test_widening_returns_everything_again(client, two_day_person):
    """The 'Show all' path is just the same call without the filter."""
    pid = two_day_person["person_id"]
    assert (len(_photos(client, "person", pid)["photos"])
            == _photos(client, "person", pid)["total_unfiltered"])


def test_cluster_photos_respect_the_date_filter_too(client, db):
    """Unknown clusters get the same treatment — the filter used to be accepted
    and silently ignored for both person AND cluster."""
    fid = db._test_face_ids["unknown_878"]      # cluster_id = 99
    row = db.conn.execute("SELECT cluster_id FROM faces WHERE id = ?", (fid,)).fetchone()
    cid = row["cluster_id"]
    wide = _photos(client, "cluster", cid)
    narrow = _photos(client, "cluster", cid, date_from="2030-01-01")
    assert wide["photos"] and narrow["photos"] == []
    assert narrow["total_unfiltered"] == wide["total_unfiltered"]


def test_unclustered_still_requires_a_filter(client):
    """Unchanged: that bucket is DEFINED by the filter, and without one the
    query would return every unclustered face in the library."""
    r = client.get("/api/faces/group/unclustered/0/photos")
    assert r.status_code == 400


def test_a_filter_matching_nothing_returns_empty_not_everything(client, two_day_person):
    """The dangerous failure mode: an over-narrow filter falling back to the
    unfiltered set would silently re-open all 18k photos."""
    d = _photos(client, "person", two_day_person["person_id"],
                date_from="2030-01-01", date_to="2030-01-02")
    assert d["photos"] == []
    assert d["filtered"] is True
    assert d["total_unfiltered"] == 5

"""A human rejection must survive auto-matching.

On 2026-09-14 a routine `match_faces` sweep re-applied 164 Calvin temporal
matches that had been deliberately removed the day before, because the only
marker available ('dedupe_unmatched') is not consulted by the matchers and a
plain NULL is indistinguishable from never-matched.
"""

import pytest

from photosearch.faces import MATCHABLE_SQL, REJECTED_MATCH_SOURCE


def test_rejected_is_distinct_from_the_dedupe_marker():
    """They mean different things and must not be conflated.

    'dedupe_unmatched' is the duplicate-resolver's automatic tie-break — a
    machine keeping one face per (photo, person). The losing face may still
    legitimately match a DIFFERENT person later, so it stays matchable.
    'rejected' is a human saying "this is not that person".
    """
    assert REJECTED_MATCH_SOURCE == "rejected"
    assert REJECTED_MATCH_SOURCE != "dedupe_unmatched"
    assert "rejected" in MATCHABLE_SQL
    assert "dedupe_unmatched" not in MATCHABLE_SQL


def test_both_matchers_use_the_shared_predicate():
    """Strict and temporal must not drift — the bug was one query, not both."""
    import inspect
    from photosearch import faces
    src = inspect.getsource(faces)
    body = src.split("MATCHABLE_SQL = (", 1)[1]
    # Every candidate-selection query filters on the shared fragment rather
    # than a bare `person_id IS NULL`.
    assert body.count("{MATCHABLE_SQL}") >= 4, (
        "a matcher candidate query is not using MATCHABLE_SQL")


def test_clearing_a_face_marks_it_rejected(client, db):
    face_id = db._test_face_ids["alex_894"]
    r = client.post(f"/api/faces/{face_id}/clear")
    assert r.status_code == 200
    row = db.conn.execute(
        "SELECT person_id, match_source FROM faces WHERE id = ?", (face_id,)).fetchone()
    assert row["person_id"] is None
    assert row["match_source"] == REJECTED_MATCH_SOURCE


def test_bulk_clear_marks_rejected(client, db):
    ids = [db._test_face_ids["alex_907"], db._test_face_ids["jamie_922"]]
    r = client.post("/api/faces/bulk-assign",
                    json={"face_ids": ids, "person_name": None})
    assert r.status_code == 200
    for fid in ids:
        row = db.conn.execute(
            "SELECT person_id, match_source FROM faces WHERE id = ?", (fid,)).fetchone()
        assert row["person_id"] is None
        assert row["match_source"] == REJECTED_MATCH_SOURCE


def test_temporal_matching_skips_rejected_faces(db):
    """THE regression. A rejected face must not be re-tagged by a later sweep."""
    from photosearch import faces as F

    face_id = db._test_face_ids["alex_894"]
    alex = db._test_person_ids["Alex"]
    db.conn.execute("UPDATE faces SET person_id = NULL, match_source = ? WHERE id = ?",
                    (REJECTED_MATCH_SOURCE, face_id))
    db.conn.commit()

    candidates = [r["id"] for r in db.conn.execute(
        f"SELECT f.id FROM faces f WHERE {MATCHABLE_SQL}")]
    assert face_id not in candidates, "a rejected face was offered to the matcher"

    # An ordinary unmatched face IS still offered.
    other = db._test_face_ids["unknown_878"]
    assert other in candidates
    assert alex is not None


def test_dedupe_unmatched_faces_remain_matchable(db):
    """The duplicate-resolver's marker must NOT block re-matching: its losing
    face may belong to a different person entirely. Filtering it was the
    tempting one-line fix and would have frozen ~9.6k faces from June."""
    face_id = db._test_face_ids["alex_907"]
    db.conn.execute("UPDATE faces SET person_id = NULL, match_source = 'dedupe_unmatched' "
                    "WHERE id = ?", (face_id,))
    db.conn.commit()
    candidates = [r["id"] for r in db.conn.execute(
        f"SELECT f.id FROM faces f WHERE {MATCHABLE_SQL}")]
    assert face_id in candidates


def test_face_state_apply_skips_rejected(client, db, monkeypatch):
    """The replica push must respect a rejection too, not just the dedup marker."""
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    from photosearch.face_state import export_face_state, apply_face_state

    face_id = db._test_face_ids["unknown_878"]
    alex = db._test_person_ids["Alex"]
    db.conn.execute("UPDATE faces SET person_id = ?, match_source = 'temporal' WHERE id = ?",
                    (alex, face_id))
    db.conn.commit()
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "fs.db")
        export_face_state(db, path)
        db.conn.execute("UPDATE faces SET person_id = NULL, match_source = ? WHERE id = ?",
                        (REJECTED_MATCH_SOURCE, face_id))
        db.conn.commit()
        apply_face_state(db, path, apply=True)
    row = db.conn.execute("SELECT person_id, match_source FROM faces WHERE id = ?",
                          (face_id,)).fetchone()
    assert row["person_id"] is None, "face-state apply re-filled a rejected face"
    assert row["match_source"] == REJECTED_MATCH_SOURCE


# ---------------------------------------------------------------------------
# face_verify — the calibrated-margin label check
# ---------------------------------------------------------------------------

def test_leave_one_burst_out_is_what_makes_it_work():
    """THE bug the first implementation had, pinned.

    Two mislabelled faces from one burst alibi each other: on the real
    2026-09-12 case, the disputed face's nearest 'Oliver' reference was the
    OTHER impostor 1s away at 0.579, which collapsed the margin below zero and
    hid the error entirely. Excluding the burst gives 1.156 and it is decisive.
    Leave-one-FACE-out silently defeats the whole method.
    """
    import inspect
    from photosearch import face_verify
    src = inspect.getsource(face_verify.verify_labels)
    assert "burst_window" in src
    assert "own[\"ts\"][i]" in src, "burst exclusion is not consulting timestamps"
    assert face_verify.BURST_WINDOW_SECONDS > 0


def test_separation_uses_a_percentile_not_a_raw_min():
    """A single bad label on either side drags min() to ~0 and silently
    disables calibration for that pair — surfacing as 'no conflicts found',
    the most dangerous failure mode available."""
    from photosearch import face_verify
    assert 0 < face_verify.SEPARATION_PERCENTILE < 50
    import inspect
    assert "percentile" in inspect.getsource(face_verify.verify_labels)


def test_temporal_faces_are_never_references():
    """Temporal measured ~4% accurate on these shoots; using it to define who
    someone looks like poisons the baseline the whole test rests on."""
    from photosearch.face_verify import REFERENCE_SOURCES
    assert set(REFERENCE_SOURCES) == {"manual", "strict"}


def test_a_scope_is_required(db):
    from photosearch import face_verify
    with pytest.raises(ValueError, match="scope is required"):
        face_verify.verify_labels(db)


def test_people_with_too_few_references_are_reported_not_scored(db):
    """A 1-2 face person can't calibrate anything. Saying so beats guessing."""
    from photosearch import face_verify
    findings, skipped, stats = face_verify.verify_labels(
        db, date_from="2026-01-01", date_to="2027-01-01", min_references=3)
    assert any("too few trusted references" in s["reason"] for s in skipped)
    assert isinstance(findings, list)


def test_alibi_excluded_is_never_negative(db):
    """It counts same-burst references WITHHELD, so it cannot be negative.

    The first version computed it as len(refs) - 1 - len(kept), which assumes
    the face under test is itself in the reference set. A `temporal` face never
    is (references are manual/strict only), so every temporal finding reported
    -1 — visible nonsense in the UI and a wrong signal to the reviewer.
    """
    from photosearch import face_verify
    findings, _, _ = face_verify.verify_labels(
        db, date_from="2026-01-01", date_to="2027-01-01", min_references=2)
    assert all(f["alibi_excluded"] >= 0 for f in findings)


def test_findings_carry_what_the_review_ui_needs(db):
    """The panel renders three strips and posts a reassignment without a
    second round-trip, so each finding must carry both reference sets."""
    from photosearch import face_verify
    findings, _, _ = face_verify.verify_labels(
        db, date_from="2026-01-01", date_to="2027-01-01", min_references=2)
    for f in findings:
        assert isinstance(f["own_refs"], list)
        assert isinstance(f["candidate_refs"], list)
        assert f["face_id"] not in f["own_refs"], "a face cannot be its own reference"


def test_verify_labels_endpoint_requires_a_scope(client):
    assert client.get("/api/faces/verify-labels").status_code == 400


def test_verify_labels_endpoint_resolves_person_ids(client, db, monkeypatch):
    monkeypatch.setenv("PHOTOSEARCH_DB", db.db_path)
    r = client.get("/api/faces/verify-labels",
                   params={"date_from": "2026-01-01", "date_to": "2027-01-01",
                           "min_references": 2})
    assert r.status_code == 200
    d = r.json()
    assert "findings" in d and "stats" in d and "decisive_count" in d
    for f in d["findings"]:
        assert f["person_id"] is not None


def test_mirror_writes_the_same_marker_the_nas_does(monkeypatch, db, tmp_path):
    """The replica mirror must not invent its own semantics for a clear.

    It duplicates the authoritative write rather than reading it back, so a
    divergence is SILENT: the NAS recorded 'rejected' while the replica showed a
    never-matched face — and a replica-side face-state export would then ship
    that back up as "no opinion" and quietly undo the rejection.
    """
    from photosearch import web
    monkeypatch.setattr(web, "_db_path", db.db_path)
    fid = db._test_face_ids["alex_894"]
    web._mirror_face_labels([fid], None)
    row = db.conn.execute("SELECT person_id, match_source FROM faces WHERE id = ?",
                          (fid,)).fetchone()
    assert row["person_id"] is None
    assert row["match_source"] == REJECTED_MATCH_SOURCE


def test_mirror_still_writes_manual_for_an_assignment(monkeypatch, db):
    from photosearch import web
    monkeypatch.setattr(web, "_db_path", db.db_path)
    fid = db._test_face_ids["unknown_878"]
    web._mirror_face_labels([fid], "Alex")
    row = db.conn.execute("SELECT person_id, match_source FROM faces WHERE id = ?",
                          (fid,)).fetchone()
    assert row["person_id"] == db._test_person_ids["Alex"]
    assert row["match_source"] == "manual"

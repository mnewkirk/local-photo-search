"""The sweep's match_faces stage is STRICT-ONLY unless temporal is asked for.

The nightly maintenance-sweep used to run the temporal matcher on every run. On
2026-09-19 it swept a shoot ingested hours earlier and wrote 466 temporal labels
(Calvin 382, Ellie 149) beside 65 strict ones — on exactly the kind of shoot
where temporal is ~4% accurate (it tags one kid across both teams). Every new
shoot was being polluted before anyone had reviewed it. Temporal is now an
explicit opt-in.
"""

import pytest

from photosearch import faces as faces_mod
from photosearch.maintenance import _stage_match_faces, run_maintenance_sweep


@pytest.fixture
def matchable(db, monkeypatch):
    """A DB with one unmatched face and one person reference; matchers faked."""
    pid = db.add_photo(filepath="2091/2091-01-01/a.jpg", filename="a.jpg")
    db.add_face(pid, (0, 10, 10, 0), [0.0] * 512)
    person = db.add_person("Someone")
    db.conn.execute("INSERT INTO face_references (person_id, source_path) VALUES (?, ?)",
                    (person, "ref.jpg"))
    db.conn.commit()

    calls = []
    monkeypatch.setattr(faces_mod, "check_available", lambda: None)
    monkeypatch.setattr(faces_mod, "match_faces_to_persons",
                        lambda _db, **kw: calls.append("strict") or 3)
    monkeypatch.setattr(faces_mod, "match_faces_temporal",
                        lambda _db, **kw: calls.append("temporal") or 40)
    return db, calls


def _noop(*_a, **_k):
    return None


def test_the_stage_is_strict_only_by_default(matchable):
    db, calls = matchable
    res = _stage_match_faces(db, True, _noop, _noop)
    assert calls == ["strict"]
    assert res["applied"] == 3
    assert res["status"] == "done"


def test_temporal_runs_only_when_asked_for(matchable):
    db, calls = matchable
    res = _stage_match_faces(db, True, _noop, _noop, temporal=True)
    assert calls == ["strict", "temporal"]
    assert res["applied"] == 43


def test_the_sweep_does_not_run_temporal_by_default(matchable):
    db, calls = matchable
    run_maintenance_sweep(db, apply=True, stages=["match_faces"])
    assert "temporal" not in calls
    assert "strict" in calls


def test_the_sweep_passes_the_opt_in_through(matchable):
    db, calls = matchable
    run_maintenance_sweep(db, apply=True, stages=["match_faces"], match_temporal=True)
    assert calls == ["strict", "temporal"]


def test_a_dry_run_calls_neither_matcher(matchable):
    db, calls = matchable
    res = _stage_match_faces(db, False, _noop, _noop, temporal=True)
    assert calls == []
    assert res["status"] == "preview"

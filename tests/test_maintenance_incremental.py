"""The nightly sweep must be nearly free on an unchanged library.

Two stages did full work every night on the NAS (log
/var/log/photo-maintenance.log, 2026-09-19 .. 2026-09-26):

* ``normalize_aesthetics`` — ``would 1373, applied 158111``. 1,373 scored
  photos have no ``date_taken`` AND no ``date_created`` (Facebook takeout,
  old scans), so ``_normalize_by_day`` can never give them a per-day
  percentile, while the gate asked ``aes_overall_day_pct IS NULL`` —
  permanently true for them. ``applied`` was the scored-row count, not the
  rows written, so the log could not show that nothing had changed.
* ``stacking`` — ``would 157815, applied 27165``. The stage called an
  UNSCOPED ``run_stacking``: a full-library re-detect that cleared and
  re-created every stack each night. ``applied`` was the total stack count.

These tests pin the missing-only replacements.
"""

import numpy as np
import pytest

from photosearch import maintenance, stacking
from photosearch.db import PhotoDB
from photosearch.stacking import detect_stacks, run_incremental_stacking


@pytest.fixture
def pdb(tmp_path):
    db = PhotoDB(str(tmp_path / "inc.db"))
    yield db
    db.close()


def _noop(*_a, **_k):
    return None


# ---------------------------------------------------------------------------
# normalize_aesthetics / normalize_subject_aesthetics
# ---------------------------------------------------------------------------

def _scored(db, name, score, date_taken=None, subject=None):
    pid = db.add_photo(filepath=f"{name}.jpg", filename=f"{name}.jpg",
                       date_taken=date_taken)
    db.conn.execute(
        "UPDATE photos SET aes_overall=?, aes_subject_overall=?, "
        "date_created=NULL WHERE id=?", (score, subject, pid))
    db.conn.commit()
    return pid


def test_undated_scored_photo_does_not_hold_the_gate_open(pdb):
    _scored(pdb, "a", 7.0, "2026-01-01 10:00:00")
    _scored(pdb, "b", 5.0, "2026-01-01 11:00:00")
    undated = _scored(pdb, "c", 6.0, None)

    first = maintenance._stage_normalize_aesthetics(pdb, True, _noop, _noop)
    assert first["status"] == "done"
    # The undated photo got a library percentile but can never get a day one.
    row = pdb.conn.execute(
        "SELECT aes_overall_pct, aes_overall_day_pct FROM photos WHERE id=?",
        (undated,)).fetchone()
    assert row[0] is not None and row[1] is None

    # Old gate: this was 1 forever -> a full re-rank every night.
    second = maintenance._stage_normalize_aesthetics(pdb, True, _noop, _noop)
    assert second["would"] == 0
    assert second["status"] == "skipped"
    assert second["applied"] == 0


def test_undated_photo_missing_its_LIBRARY_percentile_still_counts(pdb):
    _scored(pdb, "a", 7.0, "2026-01-01 10:00:00")
    _scored(pdb, "b", 6.0, None)
    res = maintenance._stage_normalize_aesthetics(pdb, False, _noop, _noop)
    assert res["would"] == 2 and res["status"] == "preview"


def test_date_created_counts_as_a_capture_day(pdb):
    """_normalize_by_day falls back to date_created; the gate must too, or a
    photo with only date_created would be wrongly excluded."""
    pid = _scored(pdb, "a", 7.0, None)
    pdb.conn.execute("UPDATE photos SET date_created='2026-01-01 10:00:00', "
                     "aes_overall_pct=50 WHERE id=?", (pid,))
    pdb.conn.commit()
    res = maintenance._stage_normalize_aesthetics(pdb, False, _noop, _noop)
    assert res["would"] == 1
    maintenance._stage_normalize_aesthetics(pdb, True, _noop, _noop)
    assert pdb.conn.execute("SELECT aes_overall_day_pct FROM photos WHERE id=?",
                            (pid,)).fetchone()[0] is not None


def test_applied_is_rows_written_not_rows_considered(pdb):
    for i in range(5):
        _scored(pdb, f"p{i}", 3.0 + i, "2026-01-01 10:00:00")
    first = maintenance._stage_normalize_aesthetics(pdb, True, _noop, _noop)
    assert first["applied"] == 10          # 5 library + 5 per-day, all new
    assert first["considered"] == 5

    # Nothing moved: a forced re-rank computes everything and writes nothing.
    again = maintenance._stage_normalize_aesthetics(
        pdb, True, _noop, _noop, force=True)
    assert again["status"] == "done"
    assert again["applied"] == 0
    assert again["unchanged"] == 10
    assert again["considered"] == 5


def test_subject_stage_has_the_same_undated_gate(pdb):
    _scored(pdb, "a", 7.0, "2026-01-01 10:00:00", subject=7.0)
    _scored(pdb, "b", 5.0, None, subject=5.0)
    first = maintenance._stage_normalize_subject_aesthetics(pdb, True, _noop, _noop)
    assert first["status"] == "done" and first["applied"] == 3  # 2 lib + 1 day
    second = maintenance._stage_normalize_subject_aesthetics(pdb, True, _noop, _noop)
    assert second["status"] == "skipped" and second["would"] == 0


# ---------------------------------------------------------------------------
# stacking
# ---------------------------------------------------------------------------

def _vec(seed, dim=512):
    v = np.random.RandomState(seed).randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _near(base, seed, noise=0.001):
    v = base + np.random.RandomState(seed).randn(base.shape[0]).astype(np.float32) * noise
    return v / np.linalg.norm(v)


_n = [0]


def _photo(db, when, vec, score=5.0):
    _n[0] += 1
    pid = db.add_photo(filepath=f"s{_n[0]}.jpg", filename=f"s{_n[0]}.jpg",
                       date_taken=when, aesthetic_score=score)
    db.add_clip_embedding(pid, vec.tolist())
    db.conn.commit()
    return pid


def _burst(db, hhmm, base_seed, n=3):
    base = _vec(base_seed)
    return [_photo(db, f"2026-03-13T{hhmm}:{i:02d}", _near(base, base_seed * 100 + i),
                   score=5.0 + i) for i in range(n)]


def _stacks(db):
    """{frozenset(members): stack_id}"""
    out = {}
    for sid, pid in db.conn.execute("SELECT stack_id, photo_id FROM stack_members"):
        out.setdefault(sid, set()).add(pid)
    return {frozenset(m): sid for sid, m in out.items()}


def test_first_run_then_unchanged_library_is_a_skip(pdb):
    a = _burst(pdb, "10:00", 1)
    b = _burst(pdb, "11:00", 2)
    first = maintenance._stage_stacking(pdb, True, _noop, _noop)
    assert first["status"] == "done"
    assert first["would"] == 6 and first["applied"] == 2
    before = _stacks(pdb)
    assert set(before) == {frozenset(a), frozenset(b)}

    second = maintenance._stage_stacking(pdb, True, _noop, _noop)
    assert second == {"stage": "stacking", "would": 0, "applied": 0,
                      "status": "skipped"}
    assert _stacks(pdb) == before  # same stacks, same ids


def test_first_run_over_existing_full_detect_writes_nothing(pdb):
    """Deploy day: the ledger is empty but last night's full run already left
    the right stacks. The catch-up run re-detects everything and writes only
    the diff — here, none."""
    _burst(pdb, "10:00", 1)
    _burst(pdb, "11:00", 2)
    stacking.run_stacking(pdb)  # the old nightly behaviour
    before = _stacks(pdb)
    res = run_incremental_stacking(pdb)
    assert res["dirty"] == 6 and res["applied"] == 0
    assert res["stacks_unchanged"] == 2
    assert _stacks(pdb) == before


def test_new_frame_joins_its_burst_and_only_that_stack_is_rewritten(pdb):
    a = _burst(pdb, "10:00", 1)
    b = _burst(pdb, "11:00", 2)
    run_incremental_stacking(pdb)
    before = _stacks(pdb)

    new = _photo(pdb, "2026-03-13T10:00:03", _near(_vec(1), 999))
    res = run_incremental_stacking(pdb)
    assert res["dirty"] == 1
    assert res["scope_photos"] == 4       # the one 10:00 session, not the library
    assert res["stacks_removed"] == 1 and res["stacks_created"] == 1
    assert res["applied"] == 2
    after = _stacks(pdb)
    assert frozenset(a + [new]) in after
    assert after[frozenset(b)] == before[frozenset(b)]  # untouched, same id


def test_isolated_new_photo_is_considered_but_writes_nothing(pdb):
    _burst(pdb, "10:00", 1)
    run_incremental_stacking(pdb)
    before = _stacks(pdb)
    _photo(pdb, "2026-03-13T15:00:00", _vec(77))
    res = maintenance._stage_stacking(pdb, True, _noop, _noop)
    assert res["would"] == 1 and res["applied"] == 0 and res["scope_photos"] == 1
    assert _stacks(pdb) == before
    assert maintenance._stage_stacking(pdb, True, _noop, _noop)["status"] == "skipped"


def test_dry_run_writes_nothing(pdb):
    _burst(pdb, "10:00", 1)
    res = maintenance._stage_stacking(pdb, False, _noop, _noop)
    assert res["status"] == "preview" and res["would"] == 3
    assert pdb.conn.execute("SELECT COUNT(*) FROM stacking_seen").fetchone()[0] == 0
    assert pdb.conn.execute("SELECT COUNT(*) FROM stack_members").fetchone()[0] == 0


def test_never_clears_the_library_or_runs_unscoped(pdb, monkeypatch):
    """The two ways stacks have been wiped before: clear_stacks(), and
    detect_stacks with photo_ids=[] (read as 'no scope' -> whole library)."""
    _burst(pdb, "10:00", 1)
    _burst(pdb, "11:00", 2)
    run_incremental_stacking(pdb)

    def _boom(*a, **k):
        raise AssertionError("clear_stacks must never be called")
    monkeypatch.setattr(pdb, "clear_stacks", _boom)

    seen_scopes = []
    real = stacking.detect_stacks

    def _spy(db, *a, photo_ids=None, **k):
        assert photo_ids, "detect_stacks called without a non-empty scope"
        seen_scopes.append(list(photo_ids))
        return real(db, *a, photo_ids=photo_ids, **k)
    monkeypatch.setattr(stacking, "detect_stacks", _spy)

    _photo(pdb, "2026-03-13T10:00:04", _near(_vec(1), 555))
    run_incremental_stacking(pdb)
    assert len(seen_scopes) == 1 and len(seen_scopes[0]) == 4

    # An undated-only change (nothing sessionable) must not call detect at all.
    pid = pdb.add_photo(filepath="nodate.jpg", filename="nodate.jpg",
                        date_taken="garbage")
    pdb.add_clip_embedding(pid, _vec(5).tolist())
    pdb.conn.commit()
    res = run_incremental_stacking(pdb)
    assert res["dirty"] == 1 and res["scope_photos"] == 0
    assert len(seen_scopes) == 1


def test_retimed_photo_leaves_its_old_stack(pdb):
    a = _burst(pdb, "10:00", 1)
    run_incremental_stacking(pdb)
    pdb.conn.execute("UPDATE photos SET date_taken='2026-03-13T18:00:00' WHERE id=?",
                     (a[0],))
    pdb.conn.commit()
    res = run_incremental_stacking(pdb)
    assert res["dirty"] == 1
    assert set(_stacks(pdb)) == {frozenset(a[1:])}


def test_stack_left_with_one_member_is_removed(pdb):
    a = _burst(pdb, "10:00", 1, n=2)
    run_incremental_stacking(pdb)
    pdb.conn.execute("DELETE FROM photos WHERE id=?", (a[0],))
    pdb.conn.commit()
    assert pdb.conn.execute("SELECT COUNT(*) FROM stack_members").fetchone()[0] == 1
    res = run_incremental_stacking(pdb)
    assert res["stacks_removed"] == 1
    assert _stacks(pdb) == {}


def test_hand_picked_top_survives_an_unchanged_stack(pdb):
    a = _burst(pdb, "10:00", 1)
    _burst(pdb, "10:30", 3)
    run_incremental_stacking(pdb)
    sid = _stacks(pdb)[frozenset(a)]
    pdb.set_stack_top(sid, a[0])  # not the aesthetic best
    # New photo in a DIFFERENT session nearby in the day — irrelevant to a.
    _photo(pdb, "2026-03-13T10:15:00", _vec(42))
    run_incremental_stacking(pdb)
    top = pdb.conn.execute(
        "SELECT photo_id FROM stack_members WHERE stack_id=? AND is_top=1",
        (sid,)).fetchone()[0]
    assert top == a[0]


def test_incremental_batches_converge_to_the_full_detect_answer(pdb):
    """Adding photos over several nights must land on exactly what one full
    re-detect of the final library would produce."""
    rng = np.random.RandomState(7)
    bases = [_vec(1000 + k) for k in range(6)]
    batches = []
    for night in range(4):
        ids = []
        for j in range(12):
            k = int(rng.randint(0, 6))
            sec = int(rng.randint(0, 40))
            ids.append(_photo(pdb, f"2026-03-13T12:{k:02d}:{sec:02d}",
                              _near(bases[k], 5000 + night * 100 + j, noise=0.002),
                              score=float(rng.rand())))
        batches.append(ids)
        run_incremental_stacking(pdb)
    incremental = set(_stacks(pdb))
    full = {frozenset(s) for s in detect_stacks(pdb)}
    assert incremental == full
    assert full  # the scenario actually produced stacks

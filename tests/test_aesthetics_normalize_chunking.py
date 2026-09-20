"""The percentile refresh must not hold the write lock for a minute.

`normalize_overall` wrote ~158k rows through a single `executemany` + one
commit, so the nightly maintenance sweep's `normalize_aesthetics` stage held
SQLite's single writer lock for the whole rewrite. PhotoDB sets
`busy_timeout=60000`, and on 2026-09-20 that stage ran long enough for the
GPU fleet's concurrent `submit-results` writes to blow past 60 s and log
`database is locked`.

The percentiles are still computed from ONE consistent snapshot of the
scores — chunking only changes how the answer is WRITTEN, never what it is.
"""

import sqlite3

import pytest

from photosearch import aesthetics as A


def _seed(db, rows):
    """rows: [(id, aes_overall|None, day|None, aes_subject_overall|None)]"""
    for pid, overall, day, subject in rows:
        db.conn.execute(
            "INSERT INTO photos (id, filepath, filename, date_taken) VALUES (?,?,?,?)",
            (pid, f"{pid}.jpg", f"{pid}.jpg", (day + " 10:00:00") if day else None))
        db.conn.execute(
            "UPDATE photos SET aes_overall=?, aes_subject_overall=? WHERE id=?",
            (overall, subject, pid))
    db.conn.commit()


# A deliberately awkward fixture: ties (5.0 three times), a NULL score that
# must be skipped entirely, two capture days, and one undated row.
FIXTURE = [
    (1, 3.0, "2090-06-28", 2.0),
    (2, 5.0, "2090-06-28", 5.0),
    (3, 5.0, "2090-06-29", None),
    (4, 5.0, "2090-06-29", 8.0),
    (5, 9.0, "2090-06-29", 1.0),
    (6, None, "2090-06-29", 4.0),   # unscored — never percentiled
    (7, 7.0, None, 6.0),            # undated — library pct yes, day pct no
]


@pytest.fixture
def adb(tmp_path):
    from photosearch.db import PhotoDB
    with PhotoDB(str(tmp_path / "pct.db")) as db:
        _seed(db, FIXTURE)
        yield db


def _col(db, col):
    return {r[0]: r[1] for r in db.conn.execute(
        f"SELECT id, {col} FROM photos").fetchall()}


# ---------------------------------------------------------------------------
# Chunking must not change the answer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [1, 2, 3, 10_000])
def test_chunked_percentiles_match_the_unchunked_computation(adb, monkeypatch, chunk):
    scored = [(pid, s) for pid, s, _, _ in FIXTURE if s is not None]
    expected = dict(zip([p for p, _ in scored],
                        A.percentile_ranks([s for _, s in scored])))

    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", chunk)
    assert A.normalize_overall(adb, apply=True) == len(scored)
    got = _col(adb, "aes_overall_pct")
    assert {k: v for k, v in got.items() if v is not None} == expected
    assert got[6] is None, "an unscored row must never get a percentile"
    # ties share a rank — the property chunking is most likely to break
    assert expected[2] == expected[3] == expected[4]


@pytest.mark.parametrize("chunk", [1, 3, 10_000])
def test_chunked_per_day_percentiles_match(adb, monkeypatch, chunk):
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", chunk)
    A.normalize_overall_by_day(adb, apply=True)
    d = _col(adb, "aes_overall_day_pct")
    assert d[1] < d[2]                      # day 2090-06-28
    assert d[3] == d[4] < d[5]              # day 2090-06-29, tie preserved
    assert d[6] is None and d[7] is None    # unscored / undated


def test_subject_percentiles_chunk_too(adb, monkeypatch):
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    scored = [(pid, s) for pid, _, _, s in FIXTURE if s is not None]
    expected = dict(zip([p for p, _ in scored],
                        A.percentile_ranks([s for _, s in scored])))
    assert A.normalize_subject_overall(adb, apply=True) == len(scored)
    got = _col(adb, "aes_subject_overall_pct")
    assert {k: v for k, v in got.items() if v is not None} == expected


# ---------------------------------------------------------------------------
# ...and it must actually commit between chunks
# ---------------------------------------------------------------------------

def test_more_than_one_chunk_commits(adb, monkeypatch):
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    commits = {"n": 0}

    class CountingConn:
        """sqlite3.Connection.commit is read-only, so wrap the connection."""

        def __init__(self, inner):
            self._inner = inner

        def commit(self):
            commits["n"] += 1
            self._inner.commit()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(adb, "conn", CountingConn(adb.conn))
    A.normalize_overall(adb, apply=True)
    # 6 scored rows / chunk of 2 = 3 chunks, each its own transaction.
    assert commits["n"] >= 3, commits


def test_another_writer_can_interleave_between_chunks(adb, monkeypatch, tmp_path):
    """The whole point: a concurrent writer (the fleet's submit-results) must
    be able to take the write lock partway through the refresh."""
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    other = sqlite3.connect(adb.db_path, timeout=2.0)
    interleaved = {"n": 0}

    def between_chunks(done, total):
        if done < total:                     # not the final chunk
            other.execute("UPDATE photos SET description='written by a rival' WHERE id=1")
            other.commit()
            interleaved["n"] += 1

    A.normalize_overall(adb, apply=True, on_chunk=between_chunks)
    other.close()
    assert interleaved["n"] >= 1, "no between-chunk window existed"
    assert adb.conn.execute(
        "SELECT description FROM photos WHERE id=1").fetchone()[0] == "written by a rival"


# ---------------------------------------------------------------------------
# Don't rewrite 158k rows to change nothing
# ---------------------------------------------------------------------------

def test_a_second_run_rewrites_nothing(adb):
    A.normalize_overall(adb, apply=True)
    before = adb.conn.total_changes
    n = A.normalize_overall(adb, apply=True)
    assert adb.conn.total_changes == before, "unchanged percentiles were rewritten"
    assert n == 6, "the return value still counts the scored rows"


def test_a_second_by_day_run_rewrites_nothing(adb):
    A.normalize_overall_by_day(adb, apply=True)
    before = adb.conn.total_changes
    A.normalize_overall_by_day(adb, apply=True)
    assert adb.conn.total_changes == before


def test_a_changed_score_is_rewritten_on_the_next_run(adb):
    A.normalize_overall(adb, apply=True)
    adb.conn.execute("UPDATE photos SET aes_overall=1.0 WHERE id=5")
    adb.conn.commit()
    A.normalize_overall(adb, apply=True)
    pcts = _col(adb, "aes_overall_pct")
    assert pcts[5] == min(v for v in pcts.values() if v is not None)


# ---------------------------------------------------------------------------
# The snapshot guard
# ---------------------------------------------------------------------------

def test_a_score_changed_mid_run_is_not_given_a_stale_percentile(adb, monkeypatch):
    """Percentiles come from one snapshot taken before the first chunk. If a
    row's score moves before its chunk lands, the value computed for it is
    already wrong — skip it rather than write a stale number."""
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    other = sqlite3.connect(adb.db_path, timeout=2.0)
    done_once = {"v": False}

    def between_chunks(done, total):
        if not done_once["v"]:
            done_once["v"] = True
            # photo 5 is the top score and sorts into a later chunk
            other.execute("UPDATE photos SET aes_overall=0.5 WHERE id=5")
            other.commit()

    A.normalize_overall(adb, apply=True, on_chunk=between_chunks)
    other.close()
    assert _col(adb, "aes_overall_pct")[5] is None, \
        "the guarded UPDATE must skip a row whose score moved"


def test_skipped_rows_are_counted(adb, monkeypatch, caplog):
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    other = sqlite3.connect(adb.db_path, timeout=2.0)

    def between_chunks(done, total):
        other.execute("UPDATE photos SET aes_overall=0.5 WHERE id=5")
        other.commit()

    with caplog.at_level("INFO", logger="photosearch.aesthetics"):
        A.normalize_overall(adb, apply=True, on_chunk=between_chunks)
    other.close()
    assert "skipped" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Unchanged contract
# ---------------------------------------------------------------------------

def test_dry_run_still_writes_nothing(adb, monkeypatch):
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 1)
    assert A.normalize_overall(adb, apply=False) == 6
    assert A.normalize_subject_overall(adb, apply=False) == 6
    assert A.normalize_overall_by_day(adb, apply=False) == 5
    assert all(v is None for v in _col(adb, "aes_overall_pct").values())
    assert all(v is None for v in _col(adb, "aes_subject_overall_pct").values())
    assert all(v is None for v in _col(adb, "aes_overall_day_pct").values())


# ---------------------------------------------------------------------------
# The sweep stage uses the between-chunk gaps for abort + progress
# ---------------------------------------------------------------------------

def test_the_sweep_stage_can_abort_between_chunks(adb, monkeypatch):
    from photosearch import maintenance
    monkeypatch.setattr(A, "PERCENTILE_CHUNK_ROWS", 2)
    events = []
    calls = {"n": 0}

    def check_abort():
        calls["n"] += 1
        if calls["n"] > 1:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        maintenance._stage_normalize_aesthetics(
            adb, True, events.append, check_abort, force=True)
    assert any(e.get("total") for e in events), \
        "per-chunk progress must reach the SSE stream"


def test_chunk_size_is_a_sane_module_constant():
    assert isinstance(A.PERCENTILE_CHUNK_ROWS, int)
    assert 100 <= A.PERCENTILE_CHUNK_ROWS <= 10_000

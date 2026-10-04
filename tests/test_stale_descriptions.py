"""stale_descriptions.find_stale / requeue — the traps it exists to get right."""

import json

import pytest

from photosearch import stale_descriptions as SD
from photosearch.db import PhotoDB


@pytest.fixture
def db(tmp_path):
    with PhotoDB(str(tmp_path / "s.db")) as d:
        yield d


def _photo(db, pid, folder="2090/2090-01-01_x", cats='["a"]', kws='["k"]', verified=None):
    db.conn.execute(
        "INSERT INTO photos (id, filepath, filename, folder, description, categories, "
        "keywords, verified_at, verification_status) VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, f"{folder}/{pid}.jpg", f"{pid}.jpg", folder, "d", cats, kws,
         verified, "pass" if verified else None))


def _gen(db, pid, text_type, at):
    db.conn.execute(
        "INSERT INTO generations (photo_id, text_type, generated_text, created_at) "
        "VALUES (?,?,?,?)", (pid, text_type, "x", at))


def test_verify_rewrite_after_text_passes_is_stale(db):
    _photo(db, 1)
    _gen(db, 1, "describe", "2090-01-01 00:00:00")
    _gen(db, 1, "category-content", "2090-01-01 01:00:00")
    _gen(db, 1, "keywords", "2090-01-01 01:00:01")
    _gen(db, 1, "verify", "2090-01-01 02:00:00")     # rewrite AFTER both
    _photo(db, 2)
    _gen(db, 2, "describe", "2090-01-01 00:00:00")
    _gen(db, 2, "verify", "2090-01-01 00:30:00")     # rewrite BEFORE both
    _gen(db, 2, "category-content", "2090-01-01 01:00:00")
    _gen(db, 2, "keywords", "2090-01-01 01:00:01")
    f = SD.find_stale(db.conn)
    assert f["category-content"] == [1] and f["keywords"] == [1]


def test_mixed_timestamp_spellings_compare_correctly(db):
    """'2090-01-01T03:00:00' sorts after '2090-01-01 05:00:00' as raw text."""
    _photo(db, 1)
    _gen(db, 1, "verify", "2090-01-01T03:00:00")
    _gen(db, 1, "category-content", "2090-01-01 05:00:00")
    _gen(db, 1, "keywords", "2090-01-01 05:00:00")
    assert SD.find_stale(db.conn)["category-content"] == []


def test_verify_staleness_tolerates_the_worker_timezone(db):
    """verified_at is worker-local, generations are UTC: a describe at 17:00
    UTC is BEFORE a 10:00 PST (18:00 UTC) verification, not a re-describe."""
    _photo(db, 1, verified="2090-01-01T10:00:00")
    _gen(db, 1, "describe", "2090-01-01 17:00:00")
    _photo(db, 2, verified="2090-01-01T10:00:00")
    _gen(db, 2, "describe", "2090-01-03 00:00:00")   # genuinely re-described
    assert SD.find_stale(db.conn)["verify"] == [2]


def test_redescribe_an_hour_after_verification_is_caught(db):
    """The case a timezone MARGIN hid: verify normally runs within the hour,
    so an 8 h slack swallowed every real re-describe. 10:00 PDT = 17:00 UTC."""
    _photo(db, 1, verified="2090-07-01T10:00:00")
    _gen(db, 1, "describe", "2090-07-01 16:30:00")   # before: describe -> verify
    _photo(db, 2, verified="2090-07-01T10:00:00")
    _gen(db, 2, "describe", "2090-07-01 18:00:00")   # 1 h after the verification
    assert SD.find_stale(db.conn)["verify"] == [2]


def test_explicit_utc_stamps_are_read_exactly(db):
    _photo(db, 1, verified="2090-07-01T10:00:00Z")
    _gen(db, 1, "describe", "2090-07-01 10:30:00")
    assert SD.find_stale(db.conn)["verify"] == [1]


def test_verified_utc_conversion():
    assert SD._verified_utc("2026-07-01T10:00:00") == "2026-07-01 17:00:00"   # PDT
    assert SD._verified_utc("2026-12-01T10:00:00") == "2026-12-01 18:00:00"   # PST
    assert SD._verified_utc("2026-07-01T10:00:00Z") == "2026-07-01 10:00:00"
    assert SD._verified_utc("2026-07-01T10:00:00+00:00") == "2026-07-01 10:00:00"
    assert SD._verified_utc("garbage") is None and SD._verified_utc(None) is None


def test_worker_stamps_verified_at_in_explicit_utc(monkeypatch):
    from photosearch import worker as W
    monkeypatch.setattr(W.time, "gmtime", lambda: __import__("time").struct_time(
        (2090, 7, 1, 17, 0, 0, 0, 182, 0)))
    assert W._utc_stamp() == "2090-07-01T17:00:00Z"


def test_output_with_no_generation_row_is_counted_not_flagged(db):
    _photo(db, 1)
    _gen(db, 1, "describe", "2090-01-01 00:00:00")
    f = SD.find_stale(db.conn)
    assert f["category-content"] == [] and f["unknown_category-content"] == 1


def test_folder_scope_includes_subfolders_only(db):
    for pid, folder in ((1, "2090/a"), (2, "2090/a/sub"), (3, "2090/ab")):
        _photo(db, pid, folder=folder)
        _gen(db, pid, "category-content", "2090-01-01 00:00:00")
        _gen(db, pid, "verify", "2090-01-02 00:00:00")
    assert SD.find_stale(db.conn, folder="2090/a")["category-content"] == [1, 2]


def test_requeue_clears_only_the_stale_pass(db):
    _photo(db, 1, verified="2090-01-01T00:00:00")
    for p in SD.TEXT_PASSES:
        db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                        "VALUES (1, ?, 1)", (p,))
    db.conn.commit()
    SD.requeue(db.conn, {"category-content": [1], "keywords": [], "verify": []})
    row = db.conn.execute("SELECT categories, keywords, verified_at FROM photos").fetchone()
    assert row[0] is None and json.loads(row[1]) == ["k"] and row[2] is not None
    ledger = {r[0] for r in db.conn.execute("SELECT pass_type FROM worker_processed")}
    assert ledger == {"keywords", "verify"}


# --- the content check: keywords should be words of the description ----------

def test_keyword_match_ratio_tolerates_paraphrase():
    d = "A woman is wearing a blue hoodie while holding two dogs on the beach."
    assert SD.keyword_match_ratio(d, ["woman's hoodie", "dogs", "beach"]) == 1.0
    assert SD.keyword_match_ratio(d, ["beach", "sailboat"]) == 0.5
    assert SD.keyword_match_ratio(d, []) is None
    assert SD.keyword_match_ratio(d, ["a", "k"]) is None       # nothing judgeable


def test_mismatch_flags_what_timestamps_cannot_see(db):
    """Keywords from some OTHER text, with no generations at all to compare."""
    _photo(db, 1, kws=json.dumps(["soccer", "goal", "field"]))
    db.conn.execute("UPDATE photos SET description='A sailboat on a calm lake.' WHERE id=1")
    _photo(db, 2, kws=json.dumps(["sailboat", "lake"]))
    db.conn.execute("UPDATE photos SET description='A sailboat on a calm lake.' WHERE id=2")
    f = SD.find_stale(db.conn)
    assert f["keyword_mismatch"] == [1]
    assert f["keywords"] == [1] and f["category-content"] == [1]   # both re-queued
    assert f["by_timestamp"]["keywords"] == []


def test_mismatch_reasons(db):
    rows = {
        1: ("A dog on a beach.", ["i couldn't find any text to extract keywords from"]),
        2: ("A dog on a beach.", ["formal family garden blooming flowers wedding event"]),
        3: ("A white desk with", ["office furniture", "chair"]),
    }
    for pid, (d, k) in rows.items():
        _photo(db, pid, kws=json.dumps(k))
        db.conn.execute("UPDATE photos SET description=? WHERE id=?", (d, pid))
    f = SD.find_stale(db.conn)
    assert f["mismatch_reasons"] == {"refusal stored as keywords": 1,
                                     "whole list stored as one keyword": 1,
                                     "description cut off": 1}
    # a cut-off description is reported, not re-queued: same text, same miss
    assert 3 not in f["keywords"] and 3 in f["truncated_description"]

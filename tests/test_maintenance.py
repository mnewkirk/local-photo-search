"""Tests for the M25 maintenance sweep + data validation/repair.

Covers:
  - the sweep's missing-only predicates select the right rows (dry-run counts)
  - dry-run writes nothing to the DB
  - validate_data flags a corrupt date_taken row
  - repair_data --apply repairs it (folder-name date cascade)
  - light TestClient smoke tests for the SSE endpoint + validate GET
"""

import json

import pytest

from photosearch.db import PhotoDB
from photosearch.maintenance import (
    run_maintenance_sweep,
    validate_data,
    repair_data,
)


# ---------------------------------------------------------------------------
# Fixture DB — precise, minimal control over the predicate inputs
# ---------------------------------------------------------------------------

def _seed(db):
    db.set_photo_root("/photos")
    # pA: GPS present, no place_name, no structured cols, no colors, valid date.
    #     -> counted by geocode, normalize AND colors stages.
    pa = db.add_photo(
        filepath="2021/2021-05-01_trip/a.jpg", filename="a.jpg",
        date_taken="2021-05-01 10:00:00",
        gps_lat=47.6, gps_lon=-122.3,
    )
    # pB: GPS + place_name + structured cols + colors -> excluded everywhere.
    pb = db.add_photo(
        filepath="2021/2021-05-01_trip/b.jpg", filename="b.jpg",
        date_taken="2021-05-01 10:05:00",
        gps_lat=47.6, gps_lon=-122.3,
        place_name="Seattle, WA", country="US", admin1="Washington",
        admin2="King", locality="Seattle",
        dominant_colors=json.dumps(["#ffffff"]),
    )
    # pC: no GPS, no colors, valid date -> only colors stage counts it.
    pc = db.add_photo(
        filepath="2021/2021-05-01_trip/c.jpg", filename="c.jpg",
        date_taken="2021-05-01 10:10:00",
        dominant_colors=json.dumps(["#000000"]),  # has colors -> excluded
    )
    db.conn.commit()
    return pa, pb, pc


@pytest.fixture
def seeded_db(tmp_db_path):
    db = PhotoDB(tmp_db_path)
    _seed(db)
    yield db
    db.close()


def test_force_normalize_aesthetics_reranks_when_nothing_missing(seeded_db):
    # Two scored photos that ALREADY have percentiles (nothing missing) — the
    # default missing-only stage skips, but force re-ranks the whole library.
    from photosearch import maintenance
    c = seeded_db.conn
    ids = [r[0] for r in c.execute("SELECT id FROM photos ORDER BY id LIMIT 2").fetchall()]
    # Seed both the library-wide AND per-day percentile so the missing-only
    # predicate finds nothing (day_pct is part of it since v28).
    c.execute("UPDATE photos SET aes_overall=?, aes_overall_pct=?, aes_overall_day_pct=? WHERE id=?",
              (7.0, 50.0, 50.0, ids[0]))
    c.execute("UPDATE photos SET aes_overall=?, aes_overall_pct=?, aes_overall_day_pct=? WHERE id=?",
              (5.0, 50.0, 50.0, ids[1]))
    seeded_db.conn.commit()

    skipped = maintenance._stage_normalize_aesthetics(seeded_db, True, lambda e: None, lambda: None)
    assert skipped["status"] == "skipped"  # nothing missing → default no-op

    forced = maintenance._stage_normalize_aesthetics(
        seeded_db, True, lambda e: None, lambda: None, force=True)
    assert forced["status"] == "done" and forced["applied"] == 2
    pcts = dict(c.execute(
        "SELECT id, aes_overall_pct FROM photos WHERE id IN (?,?)", (ids[0], ids[1])).fetchall())
    assert pcts[ids[0]] != pcts[ids[1]]  # re-ranked apart


# ---------------------------------------------------------------------------
# Sweep predicates (dry-run)
# ---------------------------------------------------------------------------

def test_dry_run_predicates_select_right_rows(seeded_db):
    res = run_maintenance_sweep(
        seeded_db, apply=False, do_colors=True, do_stacking=False,
        do_recluster=False,
    )
    by_stage = {s["stage"]: s for s in res["stages"]}

    # geocode: only pA (GPS, place_name NULL)
    assert by_stage["geocode"]["would"] == 1
    assert by_stage["geocode"]["applied"] == 0
    assert by_stage["geocode"]["status"] == "preview"

    # normalize: only pA (GPS, country/admin1 NULL); pB fully populated
    assert by_stage["normalize"]["would"] == 1
    assert by_stage["normalize"]["applied"] == 0

    # colors: only pA (dominant_colors NULL); pB + pC have colors
    assert by_stage["colors"]["would"] == 1
    assert by_stage["colors"]["applied"] == 0


def test_dry_run_writes_nothing(seeded_db):
    pa, pb, pc = (
        seeded_db.get_photo_by_path("2021/2021-05-01_trip/a.jpg")["id"],
        None, None,
    )
    before = seeded_db.conn.execute(
        "SELECT place_name, country, admin1, dominant_colors FROM photos WHERE id=?",
        (pa,),
    ).fetchone()

    run_maintenance_sweep(
        seeded_db, apply=False, do_colors=True, do_stacking=False,
    )

    after = seeded_db.conn.execute(
        "SELECT place_name, country, admin1, dominant_colors FROM photos WHERE id=?",
        (pa,),
    ).fetchone()
    assert tuple(after) == tuple(before)
    assert after["place_name"] is None
    assert after["country"] is None
    assert after["dominant_colors"] is None


def test_recluster_off_by_default(seeded_db):
    res = run_maintenance_sweep(seeded_db, apply=False, do_stacking=False)
    stages = {s["stage"] for s in res["stages"]}
    assert "recluster" not in stages
    res2 = run_maintenance_sweep(
        seeded_db, apply=False, do_stacking=False, do_recluster=True,
    )
    stages2 = {s["stage"] for s in res2["stages"]}
    assert "recluster" in stages2


def test_should_abort_stops_sweep(seeded_db):
    with pytest.raises(InterruptedError):
        run_maintenance_sweep(
            seeded_db, apply=False, do_stacking=False,
            should_abort=lambda: True,
        )


# ---------------------------------------------------------------------------
# validate_data / repair_data
# ---------------------------------------------------------------------------

def test_validate_flags_corrupt_date(seeded_db):
    # Insert a row with a control-byte date_taken (the M24a corruption shape).
    seeded_db.add_photo(
        filepath="2019/2019-07-04_party/bad.jpg", filename="bad.jpg",
        date_taken="\x18u garbage",
    )
    seeded_db.conn.commit()

    report = validate_data(seeded_db)
    assert report["corrupt_date_taken"]["count"] == 1
    assert len(report["corrupt_date_taken"]["sample"]) == 1
    # The good rows (pA/pB/pC) are not flagged.
    assert report["corrupt_date_taken"]["sample"][0]["id"] is not None


def test_validate_flags_malformed_json(seeded_db):
    pid = seeded_db.add_photo(
        filepath="2021/2021-05-01_trip/j.jpg", filename="j.jpg",
        date_taken="2021-05-01 12:00:00",
        tags="{not valid json",
    )
    seeded_db.conn.commit()
    report = validate_data(seeded_db)
    assert report["malformed_json"]["tags"]["count"] == 1
    assert pid in report["malformed_json"]["tags"]["sample"]


def test_repair_dry_run_does_not_write(seeded_db):
    pid = seeded_db.add_photo(
        filepath="2019/2019-07-04_party/bad.jpg", filename="bad.jpg",
        date_taken="\x18u garbage",
    )
    seeded_db.conn.commit()

    summary = repair_data(seeded_db, apply=False)
    assert summary["apply"] is False
    assert summary["corrupt_date_taken"]["count"] == 1
    # Nothing repaired; the bad value is still there.
    row = seeded_db.get_photo(pid)
    assert row["date_taken"] == "\x18u garbage"


def test_repair_apply_fixes_corrupt_date_from_folder(seeded_db):
    pid = seeded_db.add_photo(
        filepath="2019/2019-07-04_party/bad.jpg", filename="bad.jpg",
        date_taken="\x18u garbage",
    )
    seeded_db.conn.commit()

    summary = repair_data(seeded_db, apply=True)
    assert summary["apply"] is True
    assert summary["corrupt_date_taken"]["count"] == 1
    # File doesn't exist + no EXIF -> folder-name date wins.
    assert summary["corrupt_date_taken"]["from_folder"] == 1

    row = seeded_db.get_photo(pid)
    assert row["date_taken"] == "2019-07-04 00:00:00"
    # The repaired value now passes the validator.
    report = validate_data(seeded_db)
    assert report["corrupt_date_taken"]["count"] == 0


def test_repair_fixes_zero_and_ddmmyyyy_dates(seeded_db):
    # A well-formatted-but-impossible date (0000-00-00) + a DD/MM/YYYY import —
    # both should be flagged and repaired from the YYYY-MM-DD folder name.
    z = seeded_db.add_photo(filepath="2011/2011-03-09_x/z.jpg", filename="z.jpg",
                            date_taken="0000-00-00 00:00:00")
    d = seeded_db.add_photo(filepath="2015/2015-11-29_x/d.jpg", filename="d.jpg",
                            date_taken="29/11/2015")
    seeded_db.conn.commit()
    assert validate_data(seeded_db)["corrupt_date_taken"]["count"] == 2

    repair_data(seeded_db, apply=True)
    assert seeded_db.get_photo(z)["date_taken"] == "2011-03-09 00:00:00"
    assert seeded_db.get_photo(d)["date_taken"] == "2015-11-29 00:00:00"
    assert validate_data(seeded_db)["corrupt_date_taken"]["count"] == 0


def test_repair_apply_nulls_bad_gps(seeded_db):
    pid = seeded_db.add_photo(
        filepath="2021/2021-05-01_trip/g.jpg", filename="g.jpg",
        date_taken="2021-05-01 13:00:00",
        gps_lat=999.0, gps_lon=0.0,
    )
    seeded_db.conn.commit()
    report = validate_data(seeded_db)
    assert report["bad_gps"]["count"] == 1

    summary = repair_data(seeded_db, apply=True)
    assert summary["bad_gps"]["nulled"] == 1
    row = seeded_db.get_photo(pid)
    assert row["gps_lat"] is None and row["gps_lon"] is None


# ---------------------------------------------------------------------------
# SSE endpoint + validate GET smoke tests (light)
# ---------------------------------------------------------------------------

def test_validate_endpoint_smoke(client):
    r = client.get("/api/admin/validate-data")
    assert r.status_code == 200
    body = r.json()
    assert "corrupt_date_taken" in body
    assert "bad_gps" in body
    assert "malformed_json" in body


def test_maintenance_sweep_endpoint_smoke(client):
    # Dry-run, heavy stages off, so it never touches CLIP/torch/stacking.
    r = client.post("/api/admin/maintenance-sweep", json={
        "apply": False, "do_colors": False, "do_stacking": False,
        "do_recluster": False,
    })
    assert r.status_code == 200
    # Parse the SSE body for a terminal "done" event.
    events = []
    for chunk in r.text.split("\n\n"):
        chunk = chunk.strip()
        if chunk.startswith("data:"):
            events.append(json.loads(chunk[len("data:"):].strip()))
    types = [ev.get("type") for ev in events]
    assert "done" in types
    done = next(ev for ev in events if ev.get("type") == "done")
    assert done["apply"] is False
    assert isinstance(done["stages"], list)


def test_maintenance_sweep_endpoint_validates_params(client):
    r = client.post("/api/admin/maintenance-sweep", json={"window_minutes": 0})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Duplicate-photo detection + dedup stage
# ---------------------------------------------------------------------------

def test_find_duplicate_photo_plan_groups_and_picks_canonical(seeded_db):
    from photosearch.maintenance import find_duplicate_photo_plan
    db = seeded_db
    keep = db.add_photo(filepath="2021/x/keep.jpg", filename="keep.jpg",
                        date_taken="2021-05-01 10:00:00", file_hash="DEADBEEF",
                        description="a real description")   # has_desc -> canonical
    drop = db.add_photo(filepath="2021/x/dup.jpg", filename="dup.jpg",
                        date_taken="2021-05-01 10:00:00", file_hash="DEADBEEF")
    db.conn.commit()
    plan = find_duplicate_photo_plan(db)
    assert plan["n_groups"] == 1
    assert plan["redundant_ids"] == [drop]
    assert plan["groups"][0]["keep"] == keep


def test_dedup_stage_off_by_default(seeded_db):
    from photosearch.maintenance import run_maintenance_sweep
    res = run_maintenance_sweep(seeded_db, apply=False, do_colors=False,
                                do_stacking=False, do_match=False)
    assert not any(s["stage"] == "dedup_photos" for s in res["stages"])


def test_dedup_stage_runs_first_and_previews(seeded_db):
    from photosearch.maintenance import run_maintenance_sweep
    db = seeded_db
    db.add_photo(filepath="2021/x/k.jpg", filename="k.jpg",
                 date_taken="2021-05-01 10:00:00", file_hash="H1")
    db.add_photo(filepath="2021/x/d.jpg", filename="d.jpg",
                 date_taken="2021-05-01 10:00:00", file_hash="H1")
    db.conn.commit()
    res = run_maintenance_sweep(db, apply=False, do_colors=False, do_stacking=False,
                                do_match=False, do_dedup=True)
    assert res["stages"][0]["stage"] == "dedup_photos"   # runs first
    dd = res["stages"][0]
    assert dd["status"] == "preview" and dd["would"] == 1 and dd["applied"] == 0
    # dry-run mutated nothing
    assert db.conn.execute("SELECT COUNT(*) FROM photos WHERE file_hash='H1'").fetchone()[0] == 2


def test_dedup_stage_apply_deletes_redundant(seeded_db):
    from photosearch.maintenance import _stage_dedup_photos
    db = seeded_db
    keep = db.add_photo(filepath="2021/x/k.jpg", filename="k.jpg",
                        date_taken="2021-05-01 10:00:00", file_hash="H2",
                        description="keep me")
    drop = db.add_photo(filepath="2021/x/d.jpg", filename="d.jpg",
                        date_taken="2021-05-01 10:00:00", file_hash="H2")
    db.conn.commit()
    out = _stage_dedup_photos(db, apply=True, emit=lambda e: None, check_abort=lambda: None)
    assert out["status"] == "done" and out["applied"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM photos WHERE id=?", (drop,)).fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM photos WHERE id=?", (keep,)).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Re-queue stuck worker passes
# ---------------------------------------------------------------------------

def test_requeue_off_by_default(seeded_db):
    from photosearch.maintenance import run_maintenance_sweep
    res = run_maintenance_sweep(seeded_db, apply=False, do_colors=False,
                                do_stacking=False, do_match=False)
    assert not any(s["stage"] == "requeue" for s in res["stages"])


def _seed_requeue_cases(db):
    """Three describe photos: one silently stuck via empty-string, one stuck
    via the attempts cap, one legitimately still-claimable (should be left)."""
    from photosearch.db import MAX_PROCESS_ATTEMPTS
    empty = db.add_photo(filepath="2021/x/empty.jpg", filename="empty.jpg",
                         date_taken="2021-05-01 10:00:00", description="")
    capped = db.add_photo(filepath="2021/x/capped.jpg", filename="capped.jpg",
                          date_taken="2021-05-01 10:00:00")  # description NULL
    for _ in range(MAX_PROCESS_ATTEMPTS):
        db.mark_processed([capped], "describe")
    fresh = db.add_photo(filepath="2021/x/fresh.jpg", filename="fresh.jpg",
                         date_taken="2021-05-01 10:00:00")  # NULL, no attempts
    db.conn.commit()
    return empty, capped, fresh


def test_requeue_preview_targets_only_stuck(seeded_db):
    from photosearch.maintenance import _stage_requeue
    db = seeded_db
    empty, capped, fresh = _seed_requeue_cases(db)
    out = _stage_requeue(db, apply=False, emit=lambda e: None,
                         check_abort=lambda: None, passes=("describe",))
    # empty-string + attempts-capped are stuck; the fresh NULL is still queued.
    assert out["status"] == "preview"
    assert out["would"] == 2
    assert out["by_pass"]["describe"] == 2
    # dry-run wrote nothing
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed "
                           "WHERE photo_id=? AND pass_type='describe'",
                           (capped,)).fetchone()[0] == 1


def test_requeue_apply_clears_markers_and_reenters_queue(seeded_db):
    from photosearch.maintenance import _stage_requeue
    db = seeded_db
    empty, capped, fresh = _seed_requeue_cases(db)
    before = db.count_unprocessed_photos("describe")
    out = _stage_requeue(db, apply=True, emit=lambda e: None,
                         check_abort=lambda: None, passes=("describe",))
    assert out["status"] == "done" and out["applied"] == 2
    # empty-string reset to NULL, capped attempts row deleted
    assert db.conn.execute("SELECT description FROM photos WHERE id=?",
                           (empty,)).fetchone()[0] is None
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed "
                           "WHERE photo_id=? AND pass_type='describe'",
                           (capped,)).fetchone()[0] == 0
    # both now show up in the claimable queue (fresh was already there)
    assert db.count_unprocessed_photos("describe") == before + 2


def test_requeue_skips_description_blocked_content(seeded_db):
    """category-content on a photo with no description is blocked upstream on
    describe, not stuck — requeue must not touch it."""
    from photosearch.maintenance import _stage_requeue
    db = seeded_db
    blocked = db.add_photo(filepath="2021/x/nodesc.jpg", filename="nodesc.jpg",
                           date_taken="2021-05-01 10:00:00", categories="")
    for _ in range(3):
        db.mark_processed([blocked], "category-content")
    db.conn.commit()
    out = _stage_requeue(db, apply=False, emit=lambda e: None,
                         check_abort=lambda: None, passes=("category-content",))
    assert out["would"] == 0 and out["status"] == "skipped"


def test_validate_reports_duplicate_photos(seeded_db):
    from photosearch.maintenance import validate_data
    db = seeded_db
    db.add_photo(filepath="d/1.jpg", filename="1.jpg",
                 date_taken="2021-05-01 10:00:00", file_hash="HX")
    db.add_photo(filepath="d/2.jpg", filename="2.jpg",
                 date_taken="2021-05-01 10:00:00", file_hash="HX")
    db.conn.commit()
    rep = validate_data(db)
    assert "duplicate_photos" in rep
    assert rep["duplicate_photos"]["count"] >= 1

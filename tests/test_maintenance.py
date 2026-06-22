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

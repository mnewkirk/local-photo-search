"""CLI tests for the infer-locations command."""

import pytest
from click.testing import CliRunner

from cli import cli


def _seed(db):
    """Seed a DB with one GPS anchor + two no-GPS photos in window."""
    db.add_photo(filepath="/a.jpg", filename="a.jpg",
                 date_taken="2020-06-15T10:00:00",
                 gps_lat=47.6, gps_lon=-122.3)
    db.add_photo(filepath="/b.jpg", filename="b.jpg",
                 date_taken="2020-06-15T10:10:00")
    db.add_photo(filepath="/c.jpg", filename="c.jpg",
                 date_taken="2020-06-15T10:20:00")
    db.conn.commit()


def test_cli_dry_run_reports_candidates(tmp_db_path):
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as db:
        _seed(db)

    runner = CliRunner()
    result = runner.invoke(cli, ["infer-locations", "--db", tmp_db_path])
    assert result.exit_code == 0
    assert "Candidates:" in result.output
    # Two no-GPS photos within the window with cascade on -> both anchored.
    assert "2" in result.output
    assert "Re-run with --apply" in result.output


def test_cli_apply_writes_with_inferred_source(tmp_db_path, monkeypatch):
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as db:
        _seed(db)

    # Stub the reverse geocoder so the test doesn't hit GeoNames data.
    from photosearch import infer_location as il_mod
    monkeypatch.setattr(
        "photosearch.geocode.reverse_geocode_batch",
        lambda coords: ["Seattle, Washington, US"] * len(coords),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["infer-locations", "--db", tmp_db_path, "--apply"])
    assert result.exit_code == 0
    assert "Transaction committed" in result.output

    with PhotoDB(tmp_db_path) as db:
        rows = db.conn.execute(
            "SELECT filename, gps_lat, gps_lon, place_name, "
            "       location_source, location_confidence "
            "FROM photos ORDER BY filename"
        ).fetchall()
    anchor = next(r for r in rows if r["filename"] == "a.jpg")
    inferred = [r for r in rows if r["filename"] in ("b.jpg", "c.jpg")]

    assert anchor["location_source"] == "exif"
    for r in inferred:
        assert r["gps_lat"] == pytest.approx(47.6)
        assert r["gps_lon"] == pytest.approx(-122.3)
        assert r["place_name"] == "Seattle, Washington, US"
        assert r["location_source"] == "inferred"
        assert 0 < r["location_confidence"] <= 1.0


def test_cli_apply_does_not_overwrite_existing_gps(tmp_db_path, monkeypatch):
    """Rows that already have gps_lat must not be touched."""
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as db:
        db.add_photo(filepath="/a.jpg", filename="a.jpg",
                     date_taken="2020-06-15T10:00:00",
                     gps_lat=47.6, gps_lon=-122.3)
        db.add_photo(filepath="/b.jpg", filename="b.jpg",
                     date_taken="2020-06-15T10:10:00",
                     gps_lat=10.0, gps_lon=20.0,  # pre-existing, different
                     place_name="Pre-existing")
        db.conn.commit()

    monkeypatch.setattr(
        "photosearch.geocode.reverse_geocode_batch",
        lambda coords: ["Never called"] * len(coords),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["infer-locations", "--db", tmp_db_path, "--apply"])
    assert result.exit_code == 0

    with PhotoDB(tmp_db_path) as db:
        row_b = db.conn.execute(
            "SELECT gps_lat, gps_lon, place_name FROM photos WHERE filename='b.jpg'"
        ).fetchone()
    assert row_b["gps_lat"] == pytest.approx(10.0)
    assert row_b["place_name"] == "Pre-existing"

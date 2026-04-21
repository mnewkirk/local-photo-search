"""Unit tests for photosearch.infer_location."""

import pytest


def test_haversine_us_seattle_portland():
    from photosearch.infer_location import haversine_km
    # Seattle (47.6205, -122.3493) to Portland (45.5152, -122.6784)
    d = haversine_km(47.6205, -122.3493, 45.5152, -122.6784)
    assert 230 <= d <= 240


def test_haversine_japan_tokyo_osaka():
    from photosearch.infer_location import haversine_km
    # Tokyo (35.68, 139.69) to Osaka (34.69, 135.50)
    d = haversine_km(35.68, 139.69, 34.69, 135.50)
    assert 390 <= d <= 400


def test_haversine_europe_paris_berlin():
    from photosearch.infer_location import haversine_km
    # Paris (48.85, 2.35) to Berlin (52.52, 13.41)
    d = haversine_km(48.85, 2.35, 52.52, 13.41)
    assert 870 <= d <= 885


def test_haversine_southern_sydney_melbourne():
    from photosearch.infer_location import haversine_km
    # Sydney (-33.87, 151.21) to Melbourne (-37.81, 144.96)
    d = haversine_km(-33.87, 151.21, -37.81, 144.96)
    assert 705 <= d <= 720


def test_haversine_date_line_crossing():
    from photosearch.infer_location import haversine_km
    # Two points straddling the date line — naive lon-diff would give ~24900km
    d = haversine_km(51.50, 179.0, 51.50, -179.0)
    assert 135 <= d <= 145


def test_haversine_same_point_is_zero():
    from photosearch.infer_location import haversine_km
    assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)


@pytest.fixture
def empty_db(tmp_db_path):
    """A PhotoDB with the schema created but no photos inserted."""
    from photosearch.db import PhotoDB
    pdb = PhotoDB(tmp_db_path)
    pdb.set_photo_root("/photos")
    yield pdb
    pdb.close()


def _add(db, *, filepath, date_taken, lat=None, lon=None):
    """Shorthand for inserting a test photo."""
    return db.add_photo(
        filepath=filepath,
        filename=filepath.rsplit("/", 1)[-1],
        date_taken=date_taken,
        gps_lat=lat,
        gps_lon=lon,
    )


def test_scan_photos_sorts_by_date_and_counts_no_date(empty_db):
    from photosearch.infer_location import _scan_photos

    _add(empty_db, filepath="/p1.jpg", date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    _add(empty_db, filepath="/p2.jpg", date_taken="2020-06-15T09:00:00")
    _add(empty_db, filepath="/p3.jpg", date_taken=None)  # no_date

    photos, no_date_count = _scan_photos(empty_db)
    assert no_date_count == 1
    assert [p["filepath"] for p in photos] == ["/p2.jpg", "/p1.jpg"]
    assert photos[0]["date_taken_dt"].year == 2020


def test_scan_photos_accepts_space_and_T_separators(empty_db):
    """date_taken strings come in two flavors ('YYYY-MM-DD HH:MM:SS' and
    'YYYY-MM-DDTHH:MM:SS'). Both must parse."""
    from photosearch.infer_location import _scan_photos

    _add(empty_db, filepath="/a.jpg", date_taken="2020-06-15 10:00:00")
    _add(empty_db, filepath="/b.jpg", date_taken="2020-06-15T11:00:00")

    photos, _ = _scan_photos(empty_db)
    assert len(photos) == 2
    assert all("date_taken_dt" in p for p in photos)

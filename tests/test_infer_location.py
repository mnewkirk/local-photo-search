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


def test_infer_empty_db(empty_db):
    from photosearch.infer_location import infer_locations
    result = infer_locations(empty_db, cascade=False)
    assert result["candidates"] == []
    assert result["summary"]["total_photos"] == 0
    assert result["summary"]["candidate_count"] == 0


def test_infer_no_gps_anchors(empty_db):
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/a.jpg", date_taken="2020-06-15T10:00:00")
    _add(empty_db, filepath="/b.jpg", date_taken="2020-06-15T10:10:00")
    result = infer_locations(empty_db, cascade=False)
    assert result["candidates"] == []
    assert result["summary"]["no_gps_count"] == 2
    assert result["summary"]["gps_count"] == 0


def test_infer_single_anchor_one_sided(empty_db):
    """One GPS anchor + one no-GPS photo after it (within window)
    -> one-sided inference with sides_factor=0.7."""
    from photosearch.infer_location import infer_locations
    anchor = _add(empty_db, filepath="/anchor.jpg",
                  date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    target = _add(empty_db, filepath="/target.jpg",
                  date_taken="2020-06-15T10:10:00")  # 10 min later
    result = infer_locations(empty_db, window_minutes=30, cascade=False)
    assert len(result["candidates"]) == 1
    c = result["candidates"][0]
    assert c["photo_id"] == target
    assert c["lat"] == pytest.approx(47.6)
    assert c["lon"] == pytest.approx(-122.3)
    assert c["sides"] in ("left", "right")
    assert c["hop_count"] == 1
    assert c["source_photo_id"] == anchor
    # base_decay = 1 - 10/30 = 0.667; sides_factor = 0.7 -> 0.467
    assert c["confidence"] == pytest.approx(2.0 / 3.0 * 0.7, rel=1e-3)


def test_infer_flanking_anchors_agree(empty_db):
    """Two flanking anchors within max_drift_km -> two-sided,
    no sides penalty, nearest-time wins."""
    from photosearch.infer_location import infer_locations
    a1 = _add(empty_db, filepath="/a1.jpg",
              date_taken="2020-06-15T10:00:00", lat=47.60, lon=-122.30)
    target = _add(empty_db, filepath="/target.jpg",
                  date_taken="2020-06-15T10:10:00")
    a2 = _add(empty_db, filepath="/a2.jpg",
              date_taken="2020-06-15T10:25:00", lat=47.61, lon=-122.31)
    result = infer_locations(empty_db, window_minutes=30,
                              max_drift_km=5.0, cascade=False)
    assert len(result["candidates"]) == 1
    c = result["candidates"][0]
    assert c["sides"] == "both"
    assert c["source_photo_id"] == a1  # 10 min vs 15 min -> a1 wins
    # base_decay = 1 - 10/30 = 0.667; sides_factor = 1.0
    assert c["confidence"] == pytest.approx(2.0 / 3.0, rel=1e-3)


def test_infer_movement_guard_fires(empty_db):
    """Flanking anchors >max_drift_km apart -> skipped."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/seattle.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.60, lon=-122.30)
    _add(empty_db, filepath="/mid.jpg",
         date_taken="2020-06-15T11:00:00")
    _add(empty_db, filepath="/portland.jpg",
         date_taken="2020-06-15T12:00:00", lat=45.52, lon=-122.68)
    result = infer_locations(empty_db, window_minutes=90,
                              max_drift_km=25.0, cascade=False)
    assert result["candidates"] == []
    assert result["summary"]["skipped"]["movement_guard"] == 1


def test_infer_gap_equals_window_filtered(empty_db):
    """When time_gap == window_minutes, base_decay == 0 -> confidence 0
    -> excluded (min_confidence default 0.0 filters <= 0.0)."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/anchor.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    _add(empty_db, filepath="/target.jpg",
         date_taken="2020-06-15T10:30:00")  # exactly 30 min
    result = infer_locations(empty_db, window_minutes=30, cascade=False)
    assert result["candidates"] == []
    assert result["summary"]["skipped"]["below_min_confidence"] == 1


def test_cascade_three_hop_chain(empty_db):
    """Real at t=0, no-GPS at t=20/40/60, window=30. Cascade should
    anchor all three with decaying confidence and hop_count 1/2/3."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/anchor.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    t1 = _add(empty_db, filepath="/t1.jpg", date_taken="2020-06-15T10:10:00")
    t2 = _add(empty_db, filepath="/t2.jpg", date_taken="2020-06-15T10:20:00")
    t3 = _add(empty_db, filepath="/t3.jpg", date_taken="2020-06-15T10:30:00")

    result = infer_locations(empty_db, window_minutes=30, cascade=True)

    by_id = {c["photo_id"]: c for c in result["candidates"]}
    assert set(by_id) == {t1, t2, t3}
    assert by_id[t1]["hop_count"] == 1
    assert by_id[t2]["hop_count"] == 2
    assert by_id[t3]["hop_count"] == 3
    # 10 min past the previous anchor each time -> base_decay=2/3 per hop,
    # sides_factor=0.7 (one-sided, no right anchor yet).
    assert by_id[t1]["confidence"] == pytest.approx(2/3 * 0.7, rel=1e-3)
    assert by_id[t2]["confidence"] == pytest.approx((2/3 * 0.7) ** 2, rel=1e-3)
    assert result["summary"]["cascade_rounds_used"] >= 3


def test_cascade_terminates_on_isolated_cluster(empty_db):
    """No-GPS cluster with no real anchor in reach stays unanchored;
    loop exits cleanly without hitting max_cascade_rounds."""
    from photosearch.infer_location import infer_locations
    # Connected: real anchor at t=0, target at t=15.
    _add(empty_db, filepath="/anchor.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    _add(empty_db, filepath="/connected.jpg",
         date_taken="2020-06-15T10:15:00")
    # Isolated pair: two no-GPS photos with no anchor within reach.
    _add(empty_db, filepath="/iso1.jpg", date_taken="2021-01-01T00:00:00")
    _add(empty_db, filepath="/iso2.jpg", date_taken="2021-01-01T00:10:00")

    result = infer_locations(empty_db, window_minutes=30,
                              cascade=True, max_cascade_rounds=10)
    assert len(result["candidates"]) == 1
    assert result["summary"]["skipped"]["no_anchor"] == 2
    # Must terminate well before max_cascade_rounds.
    assert result["summary"]["cascade_rounds_used"] < 10


def test_cascade_disabled_matches_direct_only(empty_db):
    """With cascade=False, the 3-hop chain from above only gets t1."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/anchor.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    t1 = _add(empty_db, filepath="/t1.jpg", date_taken="2020-06-15T10:20:00")
    _add(empty_db, filepath="/t2.jpg", date_taken="2020-06-15T10:40:00")
    _add(empty_db, filepath="/t3.jpg", date_taken="2020-06-15T11:00:00")

    result = infer_locations(empty_db, window_minutes=30, cascade=False)
    assert [c["photo_id"] for c in result["candidates"]] == [t1]
    assert result["summary"]["skipped"]["no_anchor"] == 2


def test_cascade_movement_guard_transitive(empty_db):
    """Real anchors 250km apart flank a no-GPS region. The middle
    photo sees flanking *inferred* anchors still 250km apart -> skip."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/seattle.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.60, lon=-122.30)
    _add(empty_db, filepath="/t30.jpg",  date_taken="2020-06-15T10:25:00")
    _add(empty_db, filepath="/t60.jpg",  date_taken="2020-06-15T10:50:00")
    _add(empty_db, filepath="/portland.jpg",
         date_taken="2020-06-15T11:15:00", lat=45.52, lon=-122.68)
    _add(empty_db, filepath="/t90.jpg",  date_taken="2020-06-15T11:40:00")

    result = infer_locations(empty_db, window_minutes=30,
                              max_drift_km=25.0, cascade=True)

    filenames = {c["filepath"] for c in result["candidates"]}
    # t30 anchors to Seattle, t90 to Portland (one-sided each).
    # t60 has flanking *inferred* anchors ~250km apart -> movement_guard.
    assert "/t30.jpg" in filenames
    assert "/t90.jpg" in filenames
    assert "/t60.jpg" not in filenames
    assert result["summary"]["skipped"]["movement_guard"] >= 1

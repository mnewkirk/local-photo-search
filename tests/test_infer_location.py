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

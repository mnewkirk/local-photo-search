"""Tests for the GeoNames rich-dataset builder.

Doesn't actually download anything; uses fixture strings shaped like
the real GeoNames allCountries.txt / admin*Codes.txt lines.
"""

import csv
import io
import os
from pathlib import Path

import pytest


# Fixture data — shaped like the real GeoNames files but trimmed to
# three rows each so the filter behaviour is easy to verify.

_FIXTURE_ALLCOUNTRIES = "\n".join([
    # (Columns 0-14 are what the parser reads.)
    #
    # Inverness — small populated place (P / PPL), should be kept.
    "\t".join([
        "5359593",             # 0 geonameid
        "Inverness",           # 1 name
        "Inverness",           # 2 asciiname
        "",                    # 3 alternatenames
        "38.1018",             # 4 lat
        "-122.8539",           # 5 lon
        "P",                   # 6 feature class
        "PPL",                 # 7 feature code
        "US",                  # 8 country code
        "",                    # 9 cc2
        "CA",                  # 10 admin1 code
        "041",                 # 11 admin2 code
        "",                    # 12 admin3
        "",                    # 13 admin4
        "1421",                # 14 population
    ]),
    # Point Reyes National Seashore — protected area (L / PRK), kept
    # by the feature-code allowlist even though class L isn't auto-kept.
    "\t".join([
        "5381001",
        "Point Reyes National Seashore",
        "Point Reyes National Seashore",
        "",
        "38.0500",
        "-122.9000",
        "L",
        "PRK",
        "US",
        "",
        "CA",
        "041",
        "",
        "",
        "0",
    ]),
    # A random railway station (S / RSTN) — should be dropped since
    # RSTN isn't in KEEP_FEATURE_CODES.
    "\t".join([
        "9999999",
        "Random Station",
        "Random Station",
        "",
        "40.0000",
        "-100.0000",
        "S",
        "RSTN",
        "US",
        "",
        "NV",
        "001",
        "",
        "",
        "0",
    ]),
])

_FIXTURE_ADMIN1 = "US.CA\tCalifornia\tCalifornia\t5332921\n"
_FIXTURE_ADMIN2 = "US.CA.041\tMarin County\tMarin County\t5372253\n"


def test_build_rich_dataset_filters_and_joins(tmp_path, monkeypatch):
    """Downloads are mocked; the builder should pass the already-present
    fixture files through the filter + admin-code join unchanged."""
    cache = tmp_path
    (cache / "allCountries.txt").write_text(_FIXTURE_ALLCOUNTRIES)
    (cache / "admin1CodesASCII.txt").write_text(_FIXTURE_ADMIN1)
    (cache / "admin2Codes.txt").write_text(_FIXTURE_ADMIN2)
    # Pre-create an allCountries.zip stub so the unzip step is skipped.
    (cache / "allCountries.zip").write_bytes(b"stub")

    from photosearch import geonames_rich
    # Force _download_with_progress to raise if called — the fixtures
    # are already in place, so it shouldn't be.
    monkeypatch.setattr(geonames_rich, "_download_with_progress",
                        lambda *a, **k: pytest.fail(
                            "download called despite fixture presence"))

    out = geonames_rich.build_rich_dataset(cache_dir=str(cache))

    rows = list(csv.reader(open(out)))
    # Header + 2 kept rows (Inverness + Point Reyes; station dropped).
    assert rows[0] == ["lat", "lon", "name", "admin1", "admin2", "cc"]
    names = {r[2] for r in rows[1:]}
    assert names == {"Inverness", "Point Reyes National Seashore"}

    # Admin code join worked: Marin County appears as the admin2
    # value on both rows.
    for r in rows[1:]:
        assert r[3] == "California"
        assert r[4] == "Marin County"
        assert r[5] == "US"


def test_build_is_idempotent(tmp_path, monkeypatch):
    """Second call returns the existing CSV without re-processing."""
    cache = tmp_path
    (cache / "rg_rich.csv").write_text("lat,lon,name,admin1,admin2,cc\n")

    from photosearch import geonames_rich
    monkeypatch.setattr(geonames_rich, "_download_with_progress",
                        lambda *a, **k: pytest.fail("should not re-download"))

    out = geonames_rich.build_rich_dataset(cache_dir=str(cache))
    assert out == str(cache / "rg_rich.csv")


def test_get_rich_geocoder_returns_none_when_absent(tmp_path, monkeypatch):
    """Fallback path: no dataset → None, so callers know to use stock."""
    from photosearch import geonames_rich
    monkeypatch.setenv("PHOTOSEARCH_GEONAMES_DIR", str(tmp_path))
    assert geonames_rich.get_rich_geocoder() is None

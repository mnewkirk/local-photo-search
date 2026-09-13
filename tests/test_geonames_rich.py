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


# ---------------------------------------------------------------------------
# Population gate (_MIN_POPULATION) — added after the geocode OOM, 2026-09-13
# ---------------------------------------------------------------------------

def _row(geonameid, name, lat, lon, fclass, fcode, population):
    """One allCountries.txt line; only columns 0-14 are read by the parser."""
    return "\t".join([geonameid, name, name, "", lat, lon, fclass, fcode,
                      "US", "", "CA", "041", "", "", population])


_FIXTURE_POPULATION = "\n".join([
    # Kept: populated place with a real population.
    _row("1", "Real Town", "38.10", "-122.85", "P", "PPL", "1421"),
    # Dropped: GeoNames records population 0 for 4.72M of its 5.2M class-P
    # rows, and carrying them is what OOM-killed the NAS.
    _row("2", "Pop Zero Hamlet", "38.11", "-122.86", "P", "PPL", "0"),
    # Dropped: empty population field parses as 0.
    _row("3", "Blank Pop Hamlet", "38.12", "-122.87", "P", "PPL", ""),
    # Dropped: non-numeric population must not crash the build.
    _row("4", "Junk Pop Hamlet", "38.13", "-122.88", "P", "PPL", "not-a-number"),
    # KEPT despite population 0 — a named POI. This is the whole point of the
    # rich dataset, and the gate must never reach it.
    _row("5", "Big Sur Beach", "38.14", "-122.89", "H", "BCH", "0"),
    _row("6", "Mount Tam", "38.15", "-122.90", "T", "MT", "0"),
    # KEPT: class P *and* a POI code, population 0. The is_poi check runs
    # first, so this survives; reversing that order silently drops it.
    _row("7", "Park Village", "38.16", "-122.91", "P", "PRK", "0"),
])


def _build(tmp_path, monkeypatch, allcountries):
    (tmp_path / "allCountries.txt").write_text(allcountries)
    (tmp_path / "admin1CodesASCII.txt").write_text(_FIXTURE_ADMIN1)
    (tmp_path / "admin2Codes.txt").write_text(_FIXTURE_ADMIN2)
    (tmp_path / "allCountries.zip").write_bytes(b"stub")
    from photosearch import geonames_rich
    monkeypatch.setattr(geonames_rich, "_download_with_progress",
                        lambda *a, **k: pytest.fail("must not download"))
    out = geonames_rich.build_rich_dataset(cache_dir=str(tmp_path), force=True)
    return {r[2] for r in list(csv.reader(open(out)))[1:]}


def test_population_gate_drops_pop_zero_places_but_never_pois(tmp_path, monkeypatch):
    names = _build(tmp_path, monkeypatch, _FIXTURE_POPULATION)
    assert names == {"Real Town", "Big Sur Beach", "Mount Tam", "Park Village"}


def test_population_gate_ignores_unparseable_population(tmp_path, monkeypatch):
    """A junk population field must be treated as 0, not crash the build."""
    names = _build(tmp_path, monkeypatch,
                   _row("4", "Junk Pop", "38.1", "-122.8", "P", "PPL", "xyz"))
    assert names == set()


def test_poi_code_beats_the_population_gate(tmp_path, monkeypatch):
    """A class-P row whose feature CODE is a POI is kept at population 0.

    Pinned separately because it is the one ordering bug in this filter that
    produces no error — just quietly missing park/village labels.
    """
    names = _build(tmp_path, monkeypatch,
                   _row("7", "Park Village", "38.1", "-122.8", "P", "PRK", "0"))
    assert names == {"Park Village"}


def test_force_rebuilds_the_csv_without_redownloading(tmp_path, monkeypatch):
    """`force` re-filters from the source on disk. It used to imply a fresh
    400 MB pull, so re-filtering cost a download that could not change the
    answer — `refresh_source` is now the flag for that."""
    (tmp_path / "rg_rich.csv").write_text("lat,lon,name,admin1,admin2,cc\nstale\n")
    names = _build(tmp_path, monkeypatch, _FIXTURE_POPULATION)
    assert "Real Town" in names, "force must have rebuilt the stale CSV"


def test_rich_geocoder_streams_the_file_rather_than_slurping_it(tmp_path, monkeypatch):
    """get_rich_geocoder must hand RGeocoder the FILE OBJECT.

    The old StringIO(f.read()) held the 370 MB CSV twice as UCS-2 text —
    1.44 GB of transient waste that helped OOM-kill the NAS. csv.DictReader
    iterates line by line, so nothing needed to be materialized at all.
    """
    csv_path = tmp_path / "rg_rich.csv"
    csv_path.write_text(
        "lat,lon,name,admin1,admin2,cc\n"
        "38.05,-122.80,Point Reyes,California,Marin County,US\n")
    monkeypatch.setenv("PHOTOSEARCH_GEONAMES_DIR", str(tmp_path))

    import io as _io
    from photosearch import geonames_rich

    seen = {}
    rg = pytest.importorskip("reverse_geocoder")
    real = rg.RGeocoder

    # A delegating wrapper, not a subclass: conftest mocks the heavy deps, so
    # RGeocoder may not be a real class here.
    def spy(*a, stream=None, **kw):
        seen["type"] = type(stream)
        return real(*a, stream=stream, **kw)

    monkeypatch.setattr(rg, "RGeocoder", spy)
    geo = geonames_rich.get_rich_geocoder()

    assert geo is not None
    assert not issubclass(seen["type"], _io.StringIO), \
        "the whole CSV was slurped into memory again"
    assert geo.search([(38.04, -122.79)])[0]["name"] == "Point Reyes"

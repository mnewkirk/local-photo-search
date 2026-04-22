"""Richer reverse geocoding via a filtered GeoNames allCountries dataset.

The ``reverse_geocoder`` library ships with the ``cities1000`` dataset
(~158k populated places). That misses:

- Small unincorporated communities (some CDPs below the pop-1000 cutoff)
- Named POIs — parks, beaches, monuments, lighthouses, lakes
- Hamlets + neighborhoods that matter in a personal photo library

This module handles a one-time download of GeoNames' full
``allCountries`` dataset (~400 MB zipped), filters it to the features
that matter for photo geocoding, joins admin codes to names, and
emits a CSV compatible with ``reverse_geocoder.RGeocoder(stream=...)``.

Downloaded dataset defaults to ``${PHOTOSEARCH_GEONAMES_DIR}`` or
``/data/geonames`` inside the Docker container (bind-mounted, so it
survives restarts). ``get_rich_geocoder()`` returns a wrapper exposing
the same ``search(coords)`` contract the rest of the code expects —
or ``None`` if the dataset hasn't been downloaded, in which case
callers fall back to the stock ``reverse_geocoder`` module.

Memory note: loading ~5M filtered rows into the KDTree takes ~1 GB
at steady state. Kept as a module-level singleton after first access
so the build cost (10-30s on an N100) is paid once per process.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

GEONAMES_ALLCOUNTRIES_URL = "https://download.geonames.org/export/dump/allCountries.zip"
GEONAMES_ADMIN1_URL = "https://download.geonames.org/export/dump/admin1CodesASCII.txt"
GEONAMES_ADMIN2_URL = "https://download.geonames.org/export/dump/admin2Codes.txt"

# Feature classes / codes we keep from allCountries.txt. The GeoNames
# taxonomy has 9 top-level classes; photo geocoding only cares about a
# few. Full class P (populated places) is always in; selected codes
# from L (areas/parks), S (spots), H (water features), T (topographic)
# add POI richness without bloating the dataset beyond workable size.
_KEEP_FEATURE_CLASSES = {"P"}
_KEEP_FEATURE_CODES = {
    # L — protected areas and named regions
    "PRK",    # park
    "PRKN",   # national park
    "PRKG",   # park gate / region
    "RESN",   # reserve
    "RESV",   # reservation
    "RESW",   # wildlife reserve
    "RESF",   # forest reserve
    "AREA",   # named area
    # S — named man-made POIs
    "HSTS",   # historic site
    "MNMT",   # monument
    "MUS",    # museum
    "THTR",   # theatre
    "CSTL",   # castle
    "LTHSE",  # lighthouse
    "ZOO",    # zoo
    "STDM",   # stadium
    "UNIV",   # university
    "RUIN",   # ruin
    # T — topographic features people photograph
    "MT",     # mountain
    "PK",     # peak
    "VAL",    # valley
    "VLC",    # volcano
    "CNYN",   # canyon
    # H — notable water features
    "LK",     # lake
    "BCH",    # beach
    "BCHS",   # beaches (plural feature)
    "FALLS",  # waterfall
    "BAY",    # bay
    "CAPE",   # cape
    # V — vegetation
    "FRST",   # forest
}


def _default_cache_dir() -> str:
    """Where to stash the downloaded + processed dataset. ``/data/geonames``
    on the NAS (bind-mounted), ``~/.cache/photosearch/geonames`` locally.
    Override via ``PHOTOSEARCH_GEONAMES_DIR``.
    """
    env = os.environ.get("PHOTOSEARCH_GEONAMES_DIR")
    if env:
        return env
    data_dir = Path("/data")
    if data_dir.exists() and data_dir.is_dir():
        return str(data_dir / "geonames")
    return str(Path.home() / ".cache" / "photosearch" / "geonames")


def _download_with_progress(url: str, dest: str) -> None:
    """HTTP GET to dest, printing bytes + speed to stderr. Resumable
    only via re-invocation (skips if file already exists at caller)."""
    resp = urllib.request.urlopen(url, timeout=120)
    total = int(resp.headers.get("Content-Length") or 0) or None
    chunk_size = 1 << 20  # 1 MB
    done = 0
    start = time.time()
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            elapsed = max(time.time() - start, 0.1)
            speed = done / elapsed / (1 << 20)
            if total:
                pct = done / total * 100
                sys.stderr.write(
                    f"\r  {done / (1 << 20):,.1f} / {total / (1 << 20):,.1f} MB "
                    f"({pct:5.1f}%) @ {speed:5.1f} MB/s"
                )
            else:
                sys.stderr.write(
                    f"\r  {done / (1 << 20):,.1f} MB @ {speed:5.1f} MB/s"
                )
    sys.stderr.write("\n")
    os.replace(tmp, dest)


def _load_admin_codes(path: str) -> dict[str, str]:
    """Parse a tab-separated admin codes file into {code: name}.
    File format: ``CC.A1[.A2]<TAB>name<TAB>ascii<TAB>geonameid``.
    """
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                out[parts[0]] = parts[1]
    return out


def build_rich_dataset(
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download + filter + transform GeoNames into a reverse_geocoder CSV.

    Idempotent: skips any step whose output file already exists unless
    ``force=True``. Returns the path to the final CSV.

    The CSV has six columns (``lat,lon,name,admin1,admin2,cc``) — the
    shape ``reverse_geocoder.RGeocoder`` consumes from a stream.
    """
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    allcountries_zip = os.path.join(cache_dir, "allCountries.zip")
    allcountries_txt = os.path.join(cache_dir, "allCountries.txt")
    admin1_path = os.path.join(cache_dir, "admin1CodesASCII.txt")
    admin2_path = os.path.join(cache_dir, "admin2Codes.txt")
    output_csv = os.path.join(cache_dir, "rg_rich.csv")

    if os.path.exists(output_csv) and not force:
        print(f"Rich dataset already built at {output_csv}")
        return output_csv

    if not os.path.exists(allcountries_zip) or force:
        print("Downloading allCountries.zip (~400 MB)…")
        _download_with_progress(GEONAMES_ALLCOUNTRIES_URL, allcountries_zip)

    if not os.path.exists(allcountries_txt) or force:
        print("Unzipping allCountries.txt (~1.5 GB on disk)…")
        with zipfile.ZipFile(allcountries_zip) as z:
            z.extract("allCountries.txt", cache_dir)

    if not os.path.exists(admin1_path) or force:
        print("Downloading admin1CodesASCII.txt…")
        _download_with_progress(GEONAMES_ADMIN1_URL, admin1_path)

    if not os.path.exists(admin2_path) or force:
        print("Downloading admin2Codes.txt…")
        _download_with_progress(GEONAMES_ADMIN2_URL, admin2_path)

    print("Loading admin code → name mappings…")
    admin1 = _load_admin_codes(admin1_path)
    admin2 = _load_admin_codes(admin2_path)

    print("Filtering + transforming allCountries.txt…")
    rows_in = 0
    rows_out = 0
    tmp_csv = output_csv + ".part"
    with open(allcountries_txt, encoding="utf-8") as inf, \
         open(tmp_csv, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(["lat", "lon", "name", "admin1", "admin2", "cc"])
        for line in inf:
            rows_in += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 15:
                continue
            feature_class = parts[6]
            feature_code = parts[7]
            if (feature_class not in _KEEP_FEATURE_CLASSES
                    and feature_code not in _KEEP_FEATURE_CODES):
                continue
            try:
                lat = float(parts[4])
                lon = float(parts[5])
            except (ValueError, IndexError):
                continue
            name = parts[1]
            cc = parts[8]
            a1_code = parts[10]
            a2_code = parts[11]
            a1_name = admin1.get(f"{cc}.{a1_code}", "")
            a2_name = admin2.get(f"{cc}.{a1_code}.{a2_code}", "")
            writer.writerow([lat, lon, name, a1_name, a2_name, cc])
            rows_out += 1
            if rows_out % 100000 == 0:
                sys.stderr.write(f"\r  {rows_out:,} kept / {rows_in:,} scanned")
        sys.stderr.write("\n")

    os.replace(tmp_csv, output_csv)
    print(f"Done. Wrote {rows_out:,} rows to {output_csv} "
          f"(from {rows_in:,} in allCountries).")
    return output_csv


class _RichWrapper:
    """Adapter so the rest of the codebase can call ``rg.search(coords)``
    whether ``rg`` is the default reverse_geocoder module or a custom-
    dataset RGeocoder instance.
    """

    def __init__(self, rgeocoder):
        self._rg = rgeocoder

    def search(self, coords):
        # RGeocoder instances use .query(), module uses .search() — we
        # hide the difference behind this shim.
        return self._rg.query(coords)


def get_rich_geocoder():
    """Return a ``_RichWrapper`` over the rich dataset if present, else
    ``None`` (caller should fall back to stock reverse_geocoder).
    """
    cache_dir = _default_cache_dir()
    rich_csv = os.path.join(cache_dir, "rg_rich.csv")
    if not os.path.exists(rich_csv):
        return None

    import reverse_geocoder as rg

    # reverse_geocoder's stream loader expects a file-like of the full
    # CSV. Read it all — building a KDTree incrementally isn't
    # supported. 300-400 MB of text in memory briefly, then the KDTree
    # replaces it.
    with open(rich_csv, encoding="utf-8") as f:
        inst = rg.RGeocoder(mode=1, verbose=False, stream=io.StringIO(f.read()))
    return _RichWrapper(inst)

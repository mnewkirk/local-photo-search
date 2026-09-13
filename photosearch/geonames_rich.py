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

Memory note — measured 2026-09-13, and the reason for `_MIN_POPULATION`:
``RGeocoder.load()`` keeps **one Python dict per row**, which costs ~745
bytes/row, not the "~1 GB for the KDTree" this docstring used to claim. The
original unfiltered build kept 6.58M rows = **4.9 GB**, plus 1.44 GB of
transient text, against 4.5 GB free on the N100 — so the geocode maintenance
stage OOM-killed the whole web container (SIGKILL, exit 137) every time it ran.

Two changes keep it inside the budget, and both matter:

- ``_MIN_POPULATION`` drops class-P rows GeoNames records as population 0.
  That is **4.72M of 5.2M populated places** — overwhelmingly hamlets and
  localities — while every named POI is kept regardless of population (parks
  and peaks are always pop 0, and they are the whole point of this dataset).
  6.58M rows -> 1.86M, 4.9 GB -> ~1.4 GB.
- ``get_rich_geocoder`` streams the CSV instead of ``StringIO(f.read())``,
  which used to hold it twice (UCS-2, because the file contains CJK
  punctuation) for 1.44 GB of pure waste.

Still a per-process cost, so it stays a module-level singleton. The durable
fix is a SQLite R-tree instead of an in-memory KDTree — see
``docs/plans/geocode-rtree.md``.
"""

from __future__ import annotations

import csv
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
# few. Class P (populated places) is in, subject to _MIN_POPULATION below;
# selected codes from L (areas/parks), S (spots), H (water features), T
# (topographic) add POI richness and are kept unconditionally.
_KEEP_FEATURE_CLASSES = {"P"}

# Class-P rows below this population are dropped. GeoNames records population 0
# for 4.72M of its 5.2M populated places, and carrying them cost 3.5 GB of the
# 4.9 GB that OOM-killed the NAS (see the module docstring).
#
# This threshold applies to class P ONLY. Named POIs — parks, peaks, beaches,
# monuments — are matched by feature CODE and kept whatever their population,
# which is always 0. Raising this to filter "small towns" would therefore not
# touch the POIs; it would only trade away the long tail of real villages, and
# the measured saving past 1 is small (pop>=1000 is 1.52M rows vs 1.86M, since
# the 1.38M POIs are the floor).
_MIN_POPULATION = 1
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
    refresh_source: bool = False,
) -> str:
    """Download + filter + transform GeoNames into a reverse_geocoder CSV.

    Idempotent: skips any step whose output file already exists. Returns the
    path to the final CSV.

    ``force`` rebuilds the derived CSV from the source files already on disk —
    which is what you want after changing ``_MIN_POPULATION`` or the feature
    filter. It deliberately does NOT re-download: `force` used to imply a fresh
    400 MB pull, so re-filtering an existing dataset cost a download that could
    not change the answer. Pass ``refresh_source=True`` for that separately,
    when the point is newer data from GeoNames.

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

    if os.path.exists(output_csv) and not (force or refresh_source):
        print(f"Rich dataset already built at {output_csv}")
        return output_csv

    if not os.path.exists(allcountries_zip) or refresh_source:
        print("Downloading allCountries.zip (~400 MB)…")
        _download_with_progress(GEONAMES_ALLCOUNTRIES_URL, allcountries_zip)

    if not os.path.exists(allcountries_txt) or refresh_source:
        print("Unzipping allCountries.txt (~1.5 GB on disk)…")
        with zipfile.ZipFile(allcountries_zip) as z:
            z.extract("allCountries.txt", cache_dir)

    if not os.path.exists(admin1_path) or refresh_source:
        print("Downloading admin1CodesASCII.txt…")
        _download_with_progress(GEONAMES_ADMIN1_URL, admin1_path)

    if not os.path.exists(admin2_path) or refresh_source:
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
            is_poi = feature_code in _KEEP_FEATURE_CODES
            if feature_class not in _KEEP_FEATURE_CLASSES and not is_poi:
                continue
            # Population gate on class P only — never on the named POIs, whose
            # population is always 0. `is_poi` is checked FIRST so a feature
            # that is both (e.g. a class-P row whose code is in the POI set)
            # survives: dropping it would be a silent regression in exactly the
            # labels this dataset exists to provide.
            if not is_poi:
                try:
                    population = int(parts[14] or 0)
                except ValueError:
                    population = 0
                if population < _MIN_POPULATION:
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

    # Hand RGeocoder the FILE OBJECT, not StringIO(f.read()).
    #
    # Its load() is `csv.DictReader(stream)`, which iterates line by line, so a
    # plain file object is all it ever needed. The old StringIO(f.read()) held
    # the 370 MB CSV twice — and as a *Python str*, which is UCS-2 here because
    # the file contains CJK punctuation (max codepoint U+FF1F), so 360M chars
    # cost 2 bytes each: 0.72 GB per copy, 1.44 GB for the pair, all of it
    # transient waste on a box with 4.5 GB free. Do not "simplify" this back.
    with open(rich_csv, encoding="utf-8") as f:
        inst = rg.RGeocoder(mode=1, verbose=False, stream=f)
    return _RichWrapper(inst)

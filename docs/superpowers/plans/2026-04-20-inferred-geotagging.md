# Inferred Geotagging (M19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill `gps_lat`/`gps_lon`/`place_name` on photos lacking GPS by copying from GPS-bearing temporal neighbors, with a cascade so inferred photos become anchors for still-further photos. Writes stamped `location_source='inferred'` + confidence, exposed via a CLI (`infer-locations`) and a `/status` tuning panel.

**Architecture:** Pure-functional inference core in `photosearch/infer_location.py` that takes a time-sorted list of photos and returns candidates + a summary. Schema v16 → v17 adds two columns + backfills `location_source='exif'` for existing rows. CLI is a thin click wrapper; API endpoints (`/api/geocode/infer-preview`, `/api/geocode/infer-apply`) wrap the same core synchronously. Frontend is a React component on `/status` alongside `StackingForm`.

**Tech Stack:** Python 3.11, SQLite, Click (CLI), FastAPI (API), vanilla React UMD (UI). Reuses existing `photosearch.geocode.reverse_geocode_batch` (offline GeoNames). No new Python deps.

**Source of truth:** `docs/superpowers/specs/2026-04-20-inferred-geotagging-design.md`

---

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `photosearch/infer_location.py` | Haversine, photo scan, direct inference, cascade loop, public `infer_locations()` |
| `tests/test_infer_location.py` | Unit tests for the inference core (Haversine, direct, cascade, edge cases) |
| `tests/test_web_geocode.py` | Integration tests for `/api/geocode/infer-preview` and `/api/geocode/infer-apply` |
| `tests/test_cli_infer.py` | CLI test via `click.testing.CliRunner` |

**Modify:**

| File | Change |
|---|---|
| `photosearch/db.py` | Bump `SCHEMA_VERSION` 16→17; add ALTER TABLE migration + exif backfill |
| `cli.py` | Add `infer-locations` command |
| `photosearch/web.py` | Add two endpoints |
| `frontend/dist/status.html` | Add `InferLocationForm` React component alongside `StackingForm` |

---

## Task 1 — Schema migration v16 → v17

**Files:**
- Modify: `photosearch/db.py:54` (`SCHEMA_VERSION`), `photosearch/db.py:~360` (new migration block)
- Test: `tests/test_db.py` (new test function, append to file)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_db.py`:

```python
def test_migration_v17_adds_location_columns_and_backfills_exif(tmp_path):
    """v16 → v17 adds location_source + location_confidence, backfills 'exif'
    on pre-existing GPS rows, and leaves no-GPS rows' location_source NULL."""
    import sqlite3
    from photosearch.db import PhotoDB

    db_path = str(tmp_path / "v16.db")

    # Build a minimal v16-shaped DB by hand (no location_source column).
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_info VALUES ('version', '16');
        CREATE TABLE photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            gps_lat REAL, gps_lon REAL, place_name TEXT,
            date_taken TEXT
        );
        INSERT INTO photos (filepath, filename, gps_lat, gps_lon, place_name)
            VALUES ('/a.jpg', 'a.jpg', 47.6, -122.3, 'Seattle, WA, US');
        INSERT INTO photos (filepath, filename)
            VALUES ('/b.jpg', 'b.jpg');
    """)
    conn.commit()
    conn.close()

    # Opening with the current code should migrate.
    with PhotoDB(db_path) as pdb:
        row_a = pdb.conn.execute(
            "SELECT location_source, location_confidence FROM photos WHERE filename='a.jpg'"
        ).fetchone()
        row_b = pdb.conn.execute(
            "SELECT location_source, location_confidence FROM photos WHERE filename='b.jpg'"
        ).fetchone()
        version = pdb.conn.execute(
            "SELECT value FROM schema_info WHERE key='version'"
        ).fetchone()[0]

    assert row_a["location_source"] == "exif"
    assert row_a["location_confidence"] is None
    assert row_b["location_source"] is None
    assert row_b["location_confidence"] is None
    assert int(version) >= 17
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_db.py::test_migration_v17_adds_location_columns_and_backfills_exif -v
```

Expected: FAIL with `sqlite3.OperationalError: no such column: location_source`.

- [ ] **Step 3: Bump `SCHEMA_VERSION` and add the migration block**

In `photosearch/db.py`, change line 54:

```python
SCHEMA_VERSION = 17
```

Then find the existing migration block near line 357 (the `date_created` migration, which is currently the v16 migration). **Immediately after** that block and before the next table's CREATE, insert:

```python
        # Migration: location_source + location_confidence columns (schema v17).
        # Stamps provenance on gps_lat/gps_lon so inferred writes (M19) can be
        # filtered, audited, or bulk-reverted. Existing GPS-bearing rows are
        # backfilled as 'exif' since that was the only prior source.
        try:
            cur.execute("SELECT location_source FROM photos LIMIT 1")
        except sqlite3.OperationalError:
            cur.execute("ALTER TABLE photos ADD COLUMN location_source TEXT")
            cur.execute("ALTER TABLE photos ADD COLUMN location_confidence REAL")
            cur.execute(
                "UPDATE photos SET location_source='exif' "
                "WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL"
            )
```

Then in the index-creation section near line 496, add:

```python
        cur.execute("CREATE INDEX IF NOT EXISTS idx_photos_location_source ON photos(location_source)")
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_db.py::test_migration_v17_adds_location_columns_and_backfills_exif -v
```

Expected: PASS.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```
pytest tests/ -x --ignore=tests/test_face_matching.py -q
```

(`test_face_matching.py` requires real CLIP; skip it in the plan's automated runs.)

Expected: all other tests pass.

- [ ] **Step 6: Commit**

```bash
git add photosearch/db.py tests/test_db.py
git commit -m "Schema v17: add location_source + location_confidence columns

Columns are populated by M19 inferred-geotagging writes. Backfill stamps
'exif' on every existing row with gps_lat set — the only prior GPS source."
```

---

## Task 2 — Haversine helper

**Files:**
- Create: `photosearch/infer_location.py`
- Create: `tests/test_infer_location.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_infer_location.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_infer_location.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'photosearch.infer_location'`.

- [ ] **Step 3: Create the module with `haversine_km`**

Create `photosearch/infer_location.py`:

```python
"""Temporal-neighbor GPS inference (M19).

Walks photos sorted by date_taken and copies coordinates from GPS-bearing
neighbors within a time window. Supports cascading — inferred photos
become anchors for further inference — with multiplicative confidence
decay so chains self-limit.

Called by the `infer-locations` CLI and the /api/geocode/infer-preview +
/api/geocode/infer-apply endpoints. Pure-functional core: no DB writes
happen inside infer_locations(); the caller decides.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points.

    Uses the standard haversine formula; correctly handles the
    International Date Line because sin^2 is symmetric around ±π.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_infer_location.py -v
```

Expected: all six Haversine tests PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/infer_location.py tests/test_infer_location.py
git commit -m "Add infer_location module with Haversine helper

Multi-region fixture tests (US / Japan / Europe / Southern / date-line)
verify global correctness and the date-line sign handling."
```

---

## Task 3 — Photo scan + `_parse_date` helper

**Files:**
- Modify: `photosearch/infer_location.py` (append)
- Modify: `tests/test_infer_location.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_infer_location.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_infer_location.py::test_scan_photos_sorts_by_date_and_counts_no_date -v
```

Expected: FAIL with `ImportError: cannot import name '_scan_photos'`.

- [ ] **Step 3: Add `_parse_date` and `_scan_photos` to the module**

Append to `photosearch/infer_location.py`:

```python
def _parse_date(s: str) -> datetime:
    """Parse a date_taken string. Python 3.11+ fromisoformat accepts both
    'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS'."""
    return datetime.fromisoformat(s)


def _scan_photos(db) -> tuple[list[dict], int]:
    """Return (time_sorted_photos, no_date_count).

    Each photo dict contains: id, filepath, date_taken (original str),
    date_taken_dt (parsed datetime), gps_lat, gps_lon.
    Rows whose date_taken fails to parse are counted as no_date.
    """
    rows = db.conn.execute(
        "SELECT id, filepath, date_taken, gps_lat, gps_lon "
        "FROM photos WHERE date_taken IS NOT NULL"
    ).fetchall()
    photos: list[dict] = []
    parse_failures = 0
    for r in rows:
        try:
            dt = _parse_date(r["date_taken"])
        except (ValueError, TypeError):
            parse_failures += 1
            continue
        photos.append({
            "id": r["id"],
            "filepath": r["filepath"],
            "date_taken": r["date_taken"],
            "date_taken_dt": dt,
            "gps_lat": r["gps_lat"],
            "gps_lon": r["gps_lon"],
        })
    photos.sort(key=lambda p: p["date_taken_dt"])

    no_date_row = db.conn.execute(
        "SELECT COUNT(*) AS cnt FROM photos WHERE date_taken IS NULL"
    ).fetchone()
    return photos, int(no_date_row["cnt"]) + parse_failures
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_infer_location.py -v
```

Expected: all tests so far PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/infer_location.py tests/test_infer_location.py
git commit -m "Add _scan_photos + _parse_date helpers for inference core

Sorts photos by date_taken and counts rows with NULL/unparseable dates
as a skipped bucket. Parses both 'YYYY-MM-DD HH:MM:SS' and
'YYYY-MM-DDTHH:MM:SS' via Python 3.11 fromisoformat."
```

---

## Task 4 — Direct inference (round 1, no cascade)

**Files:**
- Modify: `photosearch/infer_location.py` (append)
- Modify: `tests/test_infer_location.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_infer_location.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_infer_location.py -v
```

Expected: six new tests FAIL with `ImportError: cannot import name 'infer_locations'`.

- [ ] **Step 3: Implement direct inference**

Append to `photosearch/infer_location.py`:

```python
def _find_flanking_anchors(
    photos: list[dict],
    idx: int,
    window_minutes: float,
    anchor_ids: set[int],
    anchor_data: dict[int, dict],
) -> tuple[Optional[dict], Optional[dict]]:
    """For the photo at photos[idx], walk outward and return the nearest
    anchor on each side within window_minutes. Anchors are photos whose
    id is in anchor_ids; their coords + confidence live in anchor_data.

    Returns (left_anchor, right_anchor) where each is either a dict or None.
    """
    target_dt = photos[idx]["date_taken_dt"]
    left = None
    for j in range(idx - 1, -1, -1):
        gap = (target_dt - photos[j]["date_taken_dt"]).total_seconds() / 60.0
        if gap > window_minutes:
            break
        if photos[j]["id"] in anchor_ids:
            left = {"photo": photos[j], "gap_min": gap}
            break
    right = None
    for j in range(idx + 1, len(photos)):
        gap = (photos[j]["date_taken_dt"] - target_dt).total_seconds() / 60.0
        if gap > window_minutes:
            break
        if photos[j]["id"] in anchor_ids:
            right = {"photo": photos[j], "gap_min": gap}
            break
    return left, right


def _infer_one_round(
    photos: list[dict],
    unanchored_indices: list[int],
    anchor_ids: set[int],
    anchor_data: dict[int, dict],
    *,
    window_minutes: float,
    max_drift_km: float,
    min_confidence: float,
) -> tuple[list[dict], dict[str, int]]:
    """One inference pass over `unanchored_indices` using the current
    anchor set. Returns (new_candidates, skip_counters)."""
    candidates: list[dict] = []
    skipped = {"no_anchor": 0, "movement_guard": 0, "below_min_confidence": 0}

    for idx in unanchored_indices:
        p = photos[idx]
        left, right = _find_flanking_anchors(
            photos, idx, window_minutes, anchor_ids, anchor_data
        )
        if left is None and right is None:
            skipped["no_anchor"] += 1
            continue

        if left is not None and right is not None:
            la = anchor_data[left["photo"]["id"]]
            ra = anchor_data[right["photo"]["id"]]
            drift = haversine_km(la["lat"], la["lon"], ra["lat"], ra["lon"])
            if drift > max_drift_km:
                skipped["movement_guard"] += 1
                continue
            # Nearest-time wins; tie-break to left.
            if left["gap_min"] <= right["gap_min"]:
                chosen, chosen_gap, sides = left["photo"], left["gap_min"], "both"
            else:
                chosen, chosen_gap, sides = right["photo"], right["gap_min"], "both"
            sides_factor = 1.0
        elif left is not None:
            chosen, chosen_gap = left["photo"], left["gap_min"]
            drift = 0.0
            sides = "left"
            sides_factor = 0.7
        else:
            chosen, chosen_gap = right["photo"], right["gap_min"]
            drift = 0.0
            sides = "right"
            sides_factor = 0.7

        anchor = anchor_data[chosen["id"]]
        base_decay = max(0.0, 1.0 - chosen_gap / window_minutes)
        confidence = base_decay * sides_factor * anchor["confidence"]

        if confidence <= min_confidence:
            skipped["below_min_confidence"] += 1
            continue

        candidates.append({
            "photo_id": p["id"],
            "filepath": p["filepath"],
            "lat": anchor["lat"],
            "lon": anchor["lon"],
            "confidence": confidence,
            "hop_count": anchor["hop_count"] + 1,
            "sides": sides,
            "time_gap_min": chosen_gap,
            "drift_km": drift,
            "source_photo_id": chosen["id"],
        })

    return candidates, skipped


def infer_locations(
    db,
    *,
    window_minutes: int = 30,
    max_drift_km: float = 25.0,
    min_confidence: float = 0.0,
    cascade: bool = True,
    max_cascade_rounds: int = 10,
) -> dict:
    """Scan photos lacking GPS and return inferred (lat, lon) candidates
    pulled from temporal GPS neighbors.

    Returns {
        "candidates": [
            {photo_id, filepath, lat, lon, confidence, hop_count, sides,
             time_gap_min, drift_km, source_photo_id},
            ...
        ],
        "summary": {
            "total_photos", "no_gps_count", "gps_count",
            "candidate_count", "cascade_rounds_used",
            "skipped": {"no_anchor", "movement_guard",
                        "no_date_taken", "below_min_confidence"}
        }
    }
    """
    photos, no_date_count = _scan_photos(db)
    total_photos = len(photos) + no_date_count

    # Build initial anchor set from real GPS rows.
    anchor_ids: set[int] = set()
    anchor_data: dict[int, dict] = {}
    unanchored_indices: list[int] = []
    for i, p in enumerate(photos):
        if p["gps_lat"] is not None and p["gps_lon"] is not None:
            anchor_ids.add(p["id"])
            anchor_data[p["id"]] = {
                "lat": p["gps_lat"],
                "lon": p["gps_lon"],
                "confidence": 1.0,
                "hop_count": 0,
            }
        else:
            unanchored_indices.append(i)

    total_skipped = {
        "no_anchor": 0,
        "movement_guard": 0,
        "no_date_taken": no_date_count,
        "below_min_confidence": 0,
    }
    all_candidates: list[dict] = []
    rounds_used = 0

    # Round 1: direct inference only.
    new_candidates, skipped = _infer_one_round(
        photos, unanchored_indices, anchor_ids, anchor_data,
        window_minutes=window_minutes,
        max_drift_km=max_drift_km,
        min_confidence=min_confidence,
    )
    all_candidates.extend(new_candidates)
    for k in ("no_anchor", "movement_guard", "below_min_confidence"):
        total_skipped[k] += skipped[k]
    rounds_used = 1 if photos else 0

    # (Task 5 will add cascade rounds here.)

    return {
        "candidates": all_candidates,
        "summary": {
            "total_photos": total_photos,
            "no_gps_count": len(unanchored_indices) + no_date_count,
            "gps_count": len(anchor_ids),
            "candidate_count": len(all_candidates),
            "cascade_rounds_used": rounds_used,
            "skipped": total_skipped,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_infer_location.py -v
```

Expected: all direct-inference tests PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/infer_location.py tests/test_infer_location.py
git commit -m "Add direct (round 1) inference with movement guard

Implements _find_flanking_anchors, _infer_one_round, and the public
infer_locations() wrapper. Confidence = base_decay * sides_factor *
anchor.confidence with sides_factor=0.7 for one-sided anchors. Movement
guard compares flanking anchors' Haversine distance to max_drift_km.
Cascade stub — iterates only round 1 for now."
```

---

## Task 5 — Cascade loop

**Files:**
- Modify: `photosearch/infer_location.py` (amend `infer_locations`)
- Modify: `tests/test_infer_location.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_infer_location.py`:

```python
def test_cascade_three_hop_chain(empty_db):
    """Real at t=0, no-GPS at t=20/40/60, window=30. Cascade should
    anchor all three with decaying confidence and hop_count 1/2/3."""
    from photosearch.infer_location import infer_locations
    _add(empty_db, filepath="/anchor.jpg",
         date_taken="2020-06-15T10:00:00", lat=47.6, lon=-122.3)
    t1 = _add(empty_db, filepath="/t1.jpg", date_taken="2020-06-15T10:20:00")
    t2 = _add(empty_db, filepath="/t2.jpg", date_taken="2020-06-15T10:40:00")
    t3 = _add(empty_db, filepath="/t3.jpg", date_taken="2020-06-15T11:00:00")

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
    _add(empty_db, filepath="/t30.jpg",  date_taken="2020-06-15T10:30:00")
    _add(empty_db, filepath="/t60.jpg",  date_taken="2020-06-15T11:00:00")
    _add(empty_db, filepath="/t90.jpg",  date_taken="2020-06-15T11:30:00")
    _add(empty_db, filepath="/portland.jpg",
         date_taken="2020-06-15T12:00:00", lat=45.52, lon=-122.68)

    result = infer_locations(empty_db, window_minutes=30,
                              max_drift_km=25.0, cascade=True)

    filenames = {c["filepath"] for c in result["candidates"]}
    # t30 anchors to Seattle, t90 to Portland (one-sided each).
    # t60 has flanking *inferred* anchors ~250km apart -> movement_guard.
    assert "/t30.jpg" in filenames
    assert "/t90.jpg" in filenames
    assert "/t60.jpg" not in filenames
    assert result["summary"]["skipped"]["movement_guard"] >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_infer_location.py -v
```

Expected: the four new cascade tests FAIL (the 3-hop chain stops at 1 hop, etc.).

- [ ] **Step 3: Implement the cascade loop**

In `photosearch/infer_location.py`, replace the `# (Task 5 will add cascade rounds here.)` comment and everything up to the `return` with:

```python
    # Cascade rounds: newly-inferred photos become anchors for the next
    # round. Confidence compounds multiplicatively, so chains self-limit.
    # Promote round-1 candidates into the anchor set.
    for c in new_candidates:
        anchor_ids.add(c["photo_id"])
        anchor_data[c["photo_id"]] = {
            "lat": c["lat"],
            "lon": c["lon"],
            "confidence": c["confidence"],
            "hop_count": c["hop_count"],
        }
    still_unanchored = [i for i in unanchored_indices if photos[i]["id"] not in anchor_ids]

    if cascade:
        while rounds_used < max_cascade_rounds and still_unanchored:
            round_candidates, round_skipped = _infer_one_round(
                photos, still_unanchored, anchor_ids, anchor_data,
                window_minutes=window_minutes,
                max_drift_km=max_drift_km,
                min_confidence=min_confidence,
            )
            if not round_candidates:
                break  # fixpoint reached
            rounds_used += 1
            all_candidates.extend(round_candidates)
            for c in round_candidates:
                anchor_ids.add(c["photo_id"])
                anchor_data[c["photo_id"]] = {
                    "lat": c["lat"],
                    "lon": c["lon"],
                    "confidence": c["confidence"],
                    "hop_count": c["hop_count"],
                }
            still_unanchored = [
                i for i in still_unanchored if photos[i]["id"] not in anchor_ids
            ]
            # Skip counters from cascade rounds aren't meaningful for
            # "did not get anchored" buckets — those photos might anchor
            # in a later round. Only add the final round's skipped totals
            # at the end.
        # After loop exits (fixpoint or round cap), anything still
        # unanchored is genuinely no-anchor or movement-guarded.
        # Recompute final skip reasons on the remaining set.
        final_skipped = {"no_anchor": 0, "movement_guard": 0, "below_min_confidence": 0}
        _, final_skipped = _infer_one_round(
            photos, still_unanchored, anchor_ids, anchor_data,
            window_minutes=window_minutes,
            max_drift_km=max_drift_km,
            min_confidence=min_confidence,
        )
        total_skipped["no_anchor"] = final_skipped["no_anchor"]
        total_skipped["movement_guard"] = final_skipped["movement_guard"]
        total_skipped["below_min_confidence"] = final_skipped["below_min_confidence"]
    # (If cascade=False, round-1 skipped counts already in total_skipped.)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_infer_location.py -v
```

Expected: all unit tests PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/infer_location.py tests/test_infer_location.py
git commit -m "Add cascade loop: inferred photos become anchors

Newly-anchored photos are added to the anchor pool at the end of each
round; confidence compounds multiplicatively through the chain. Loop
exits on fixpoint or max_cascade_rounds. Movement guard transfers
through the cascade — flanking inferred anchors get the same Haversine
check as real ones.

Skip counts are recomputed from the still-unanchored set after the loop
so 'no_anchor' reflects genuinely unreachable photos, not transient
ones that would anchor in a later round."
```

---

## Task 6 — UTF-8 place_name roundtrip (spec test 13)

**Files:**
- Modify: `tests/test_infer_location.py` (append)

This is a regression guard rather than a code change — `place_name TEXT`
already stores UTF-8. The test pins the behavior so a future change
can't regress it.

- [ ] **Step 1: Write the test**

Append to `tests/test_infer_location.py`:

```python
def test_utf8_place_name_roundtrip(empty_db):
    """Non-ASCII place_name (e.g. Japanese, accented European) must
    round-trip through write + read unchanged."""
    pid = empty_db.add_photo(
        filepath="/kyoto.jpg",
        filename="kyoto.jpg",
        gps_lat=35.0116,
        gps_lon=135.7681,
        place_name="Kyōto, Kyoto, JP",
    )
    empty_db.conn.commit()
    row = empty_db.conn.execute(
        "SELECT place_name FROM photos WHERE id = ?", (pid,)
    ).fetchone()
    assert row["place_name"] == "Kyōto, Kyoto, JP"
```

- [ ] **Step 2: Run test**

```
pytest tests/test_infer_location.py::test_utf8_place_name_roundtrip -v
```

Expected: PASS (first try — SQLite handles UTF-8 natively).

- [ ] **Step 3: Commit**

```bash
git add tests/test_infer_location.py
git commit -m "Pin UTF-8 place_name roundtrip for global library support"
```

---

## Task 7 — CLI command `infer-locations`

**Files:**
- Modify: `cli.py` (append a new `@cli.command("infer-locations")`)
- Create: `tests/test_cli_infer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_infer.py`:

```python
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
                     gps_lat=47.6, gps_lon=-122.3,
                     place_name="Seattle, Washington, US")
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_cli_infer.py -v
```

Expected: FAIL with `Error: No such command 'infer-locations'`.

- [ ] **Step 3: Implement the CLI command**

In `cli.py`, locate a convenient spot (e.g., right before `@cli.command("cleanup-orphans")` near line 962). Append:

```python
@cli.command("infer-locations")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB",
              help="Path to the SQLite database file.")
@click.option("--window-minutes", default=30, type=int,
              help="Max time gap to a GPS anchor.")
@click.option("--max-drift-km", default=25.0, type=float,
              help="Reject if flanking anchors disagree by more than this.")
@click.option("--min-confidence", default=0.0, type=float,
              help="Filter inferences below this confidence (0..1).")
@click.option("--cascade/--no-cascade", default=True,
              help="Let inferred photos serve as anchors for further inference.")
@click.option("--max-cascade-rounds", default=10, type=int,
              help="Defensive ceiling on cascade iterations.")
@click.option("--apply", is_flag=True, default=False,
              help="Write inferences. Without it, prints a dry-run report.")
def infer_locations_cmd(db, window_minutes, max_drift_km, min_confidence,
                         cascade, max_cascade_rounds, apply):
    """Infer gps_lat/gps_lon for photos lacking GPS from temporal neighbors.

    Read-only without --apply. With --apply, reverse-geocodes the inferred
    coords via the offline GeoNames geocoder and writes
    gps_lat/gps_lon/place_name/location_source='inferred'/location_confidence
    in a single transaction.
    """
    from collections import Counter
    from photosearch.db import PhotoDB
    from photosearch.infer_location import infer_locations
    from photosearch.geocode import reverse_geocode_batch

    with PhotoDB(db) as pdb:
        result = infer_locations(
            pdb,
            window_minutes=window_minutes,
            max_drift_km=max_drift_km,
            min_confidence=min_confidence,
            cascade=cascade,
            max_cascade_rounds=max_cascade_rounds,
        )
        candidates = result["candidates"]
        summary = result["summary"]

        click.echo(
            f"Scanning {summary['total_photos']:,} photos... "
            f"{summary['no_gps_count']:,} have no GPS, "
            f"{summary['gps_count']:,} have GPS."
        )
        click.echo(
            f"Inferring with window={window_minutes}min, "
            f"max_drift={max_drift_km}km, min_conf={min_confidence}, "
            f"cascade={'on' if cascade else 'off'}..."
        )
        click.echo("")
        click.echo(
            f"Cascade: {summary['cascade_rounds_used']} rounds to fixpoint."
        )
        click.echo(
            f"Candidates: {summary['candidate_count']:,} of "
            f"{summary['no_gps_count']:,} no-GPS photos"
        )
        click.echo("Skipped:")
        for k, v in summary["skipped"].items():
            click.echo(f"  {k:<22} {v:,}")

        if candidates:
            # Confidence buckets
            def _bucket(c: float) -> str:
                if c >= 0.90:   return ">=0.90"
                if c >= 0.75:   return "0.75-0.90"
                if c >= 0.50:   return "0.50-0.75"
                if c >= 0.25:   return "0.25-0.50"
                return "<0.25"
            buckets = Counter(_bucket(c["confidence"]) for c in candidates)
            click.echo("\nConfidence distribution:")
            for name in (">=0.90", "0.75-0.90", "0.50-0.75", "0.25-0.50", "<0.25"):
                click.echo(f"  {name:<12} {buckets.get(name, 0):,}")

            # Hop distribution
            hops = Counter(c["hop_count"] for c in candidates)
            click.echo("\nHop distribution:")
            for h in sorted(hops):
                click.echo(f"  {h:<4} {hops[h]:,}")

            click.echo("\nSample inferences:")
            import random
            for c in random.sample(candidates, min(10, len(candidates))):
                click.echo(
                    f"  {c['filepath']}\n"
                    f"    -> {c['lat']:.4f}, {c['lon']:.4f}  "
                    f"gap={c['time_gap_min']:.0f}min, "
                    f"drift={c['drift_km']:.1f}km, "
                    f"confidence={c['confidence']:.2f}, "
                    f"hops={c['hop_count']}"
                )

        if not apply:
            click.echo("\nRe-run with --apply to write these.")
            return

        # --apply path
        if not candidates:
            click.echo("\nNothing to apply.")
            return

        click.echo(f"\nWriting {len(candidates):,} inferences...")
        coords = [(c["lat"], c["lon"]) for c in candidates]
        places = reverse_geocode_batch(coords)
        geocoded = sum(1 for p in places if p)
        click.echo(f"  Reverse-geocoded {geocoded:,} place_names.")

        updated = 0
        cur = pdb.conn.cursor()
        for c, place in zip(candidates, places):
            # Guard: only fill rows that still have gps_lat IS NULL.
            cur.execute(
                "UPDATE photos "
                "SET gps_lat=?, gps_lon=?, place_name=COALESCE(place_name, ?), "
                "    location_source='inferred', location_confidence=? "
                "WHERE id=? AND gps_lat IS NULL",
                (c["lat"], c["lon"], place, c["confidence"], c["photo_id"]),
            )
            updated += cur.rowcount
        pdb.conn.commit()
        click.echo(f"  Transaction committed. Wrote {updated:,} inferences.")
```

- [ ] **Step 4: Run the tests**

```
pytest tests/test_cli_infer.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Run the full unit-test suite**

```
pytest tests/test_infer_location.py tests/test_cli_infer.py tests/test_db.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add cli.py tests/test_cli_infer.py
git commit -m "Add photosearch infer-locations CLI

Dry-run by default. --apply reverse-geocodes inferred coords and writes
gps_lat/gps_lon/place_name/location_source='inferred'/location_confidence
in a single transaction. Only fills rows where gps_lat IS NULL — existing
GPS is never overwritten."
```

---

## Task 8 — API endpoints

**Files:**
- Modify: `photosearch/web.py` (add two routes)
- Create: `tests/test_web_geocode.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_geocode.py`:

```python
"""Integration tests for /api/geocode/* endpoints."""

import pytest
from photosearch.db import PhotoDB


def _seed_inferrable(db_path):
    """Seed the DB so there's exactly one inferrable no-GPS photo."""
    with PhotoDB(db_path) as db:
        db.add_photo(filepath="/anchor.jpg", filename="anchor.jpg",
                     date_taken="2020-06-15T10:00:00",
                     gps_lat=47.6, gps_lon=-122.3)
        db.add_photo(filepath="/target.jpg", filename="target.jpg",
                     date_taken="2020-06-15T10:10:00")
        db.conn.commit()


def test_preview_returns_counts_and_samples(client, db):
    _seed_inferrable(db.db_path)
    r = client.post(
        "/api/geocode/infer-preview",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["candidate_count"] >= 1
    assert "confidence_buckets" in body
    assert "hop_distribution" in body
    assert len(body["samples"]) >= 1
    sample = body["samples"][0]
    for key in ("photo_id", "filepath", "inferred_lat", "inferred_lon",
                "confidence", "hop_count", "time_gap_min", "drift_km",
                "sides", "source_photo_id", "thumbnail_url"):
        assert key in sample


def test_apply_requires_confirm(client, db):
    _seed_inferrable(db.db_path)
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10},
    )
    assert r.status_code == 400
    assert "confirm" in r.text.lower()


def test_apply_writes_inferred_source(client, db, monkeypatch):
    _seed_inferrable(db.db_path)
    monkeypatch.setattr(
        "photosearch.geocode.reverse_geocode_batch",
        lambda coords: ["Seattle, Washington, US"] * len(coords),
    )
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.0, "cascade": True,
              "max_cascade_rounds": 10, "confirm": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["updated_count"] >= 1

    with PhotoDB(db.db_path) as pdb:
        row = pdb.conn.execute(
            "SELECT gps_lat, location_source, location_confidence "
            "FROM photos WHERE filename='target.jpg'"
        ).fetchone()
    assert row["location_source"] == "inferred"
    assert 0 < row["location_confidence"] <= 1.0
    assert row["gps_lat"] == pytest.approx(47.6)


def test_apply_untouched_when_no_candidates(client, db):
    """With min_confidence=0.99 no inference will pass the threshold, so
    the apply path should be a clean no-op even though the db fixture
    has no-GPS photos present."""
    r = client.post(
        "/api/geocode/infer-apply",
        json={"window_minutes": 30, "max_drift_km": 25.0,
              "min_confidence": 0.99, "cascade": True,
              "max_cascade_rounds": 10, "confirm": True},
    )
    assert r.status_code == 200
    assert r.json()["updated_count"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```
pytest tests/test_web_geocode.py -v
```

Expected: FAIL with `404 Not Found` on `/api/geocode/infer-preview`.

- [ ] **Step 3: Add the endpoints**

In `photosearch/web.py`, find a location near the other similar endpoints
(e.g., after the last `/api/stats/*` route). Follow the project convention
of `data: dict` POST bodies (see `api_bulk_collect_faces` at line ~957 for
a reference). Add:

```python
# ---------------------------------------------------------------------------
# Inferred geotagging (M19)
# ---------------------------------------------------------------------------

from collections import Counter
import random


_INFER_DEFAULTS = {
    "window_minutes": 30,
    "max_drift_km": 25.0,
    "min_confidence": 0.0,
    "cascade": True,
    "max_cascade_rounds": 10,
}


def _parse_infer_params(data: dict) -> dict:
    """Pull infer params from a POST body, falling back to defaults.
    Coerces numeric types defensively (JSON-over-HTTP sometimes sends
    ints where we expect floats and vice versa)."""
    out = dict(_INFER_DEFAULTS)
    if "window_minutes" in data: out["window_minutes"] = int(data["window_minutes"])
    if "max_drift_km" in data:   out["max_drift_km"] = float(data["max_drift_km"])
    if "min_confidence" in data: out["min_confidence"] = float(data["min_confidence"])
    if "cascade" in data:        out["cascade"] = bool(data["cascade"])
    if "max_cascade_rounds" in data:
        out["max_cascade_rounds"] = int(data["max_cascade_rounds"])
    return out


def _bucket_confidence(c: float) -> str:
    if c >= 0.90:   return ">=0.90"
    if c >= 0.75:   return "0.75-0.90"
    if c >= 0.50:   return "0.50-0.75"
    if c >= 0.25:   return "0.25-0.50"
    return "<0.25"


@app.post("/api/geocode/infer-preview")
def api_infer_preview(data: dict):
    from .infer_location import infer_locations

    params = _parse_infer_params(data)
    with _get_db() as db:
        result = infer_locations(db, **params)
        candidates = result["candidates"]
        summary = result["summary"]

        buckets = Counter(_bucket_confidence(c["confidence"]) for c in candidates)
        hops = Counter(c["hop_count"] for c in candidates)

        sample_rows = random.sample(candidates, min(10, len(candidates)))
        # Pre-geocode just the sampled rows so the UI can show place_name.
        from .geocode import reverse_geocode_batch
        sample_places = reverse_geocode_batch(
            [(c["lat"], c["lon"]) for c in sample_rows]
        ) if sample_rows else []

        samples = []
        for c, place in zip(sample_rows, sample_places):
            samples.append({
                "photo_id": c["photo_id"],
                "filepath": c["filepath"],
                "thumbnail_url": f"/api/photos/{c['photo_id']}/thumbnail",
                "inferred_lat": c["lat"],
                "inferred_lon": c["lon"],
                "place_name": place,
                "confidence": c["confidence"],
                "hop_count": c["hop_count"],
                "time_gap_min": c["time_gap_min"],
                "drift_km": c["drift_km"],
                "sides": c["sides"],
                "source_photo_id": c["source_photo_id"],
            })

        return {
            "total_photos": summary["total_photos"],
            "no_gps_count": summary["no_gps_count"],
            "gps_count": summary["gps_count"],
            "candidate_count": summary["candidate_count"],
            "cascade_rounds_used": summary["cascade_rounds_used"],
            "skipped": summary["skipped"],
            "confidence_buckets": [
                {"bucket": b, "count": buckets.get(b, 0)}
                for b in (">=0.90", "0.75-0.90", "0.50-0.75",
                          "0.25-0.50", "<0.25")
            ],
            "hop_distribution": [
                {"hops": h, "count": hops[h]} for h in sorted(hops)
            ],
            "samples": samples,
        }


@app.post("/api/geocode/infer-apply")
def api_infer_apply(data: dict):
    import time
    from .infer_location import infer_locations
    from .geocode import reverse_geocode_batch

    if not data.get("confirm"):
        raise HTTPException(
            status_code=400,
            detail="confirm=true required to apply inferences",
        )

    params = _parse_infer_params(data)
    start = time.perf_counter()
    with _get_db() as db:
        result = infer_locations(db, **params)
        candidates = result["candidates"]
        if not candidates:
            return {"updated_count": 0, "rounds_used": 0, "duration_seconds": 0.0}

        coords = [(c["lat"], c["lon"]) for c in candidates]
        places = reverse_geocode_batch(coords)

        cur = db.conn.cursor()
        updated = 0
        for c, place in zip(candidates, places):
            cur.execute(
                "UPDATE photos "
                "SET gps_lat=?, gps_lon=?, place_name=COALESCE(place_name, ?), "
                "    location_source='inferred', location_confidence=? "
                "WHERE id=? AND gps_lat IS NULL",
                (c["lat"], c["lon"], place, c["confidence"], c["photo_id"]),
            )
            updated += cur.rowcount
        db.conn.commit()

    return {
        "updated_count": updated,
        "rounds_used": result["summary"]["cascade_rounds_used"],
        "duration_seconds": time.perf_counter() - start,
    }
```

`HTTPException` is already imported at the top of `web.py`, so no new
top-level imports are needed.

- [ ] **Step 4: Run the tests to verify they pass**

```
pytest tests/test_web_geocode.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Run the full unit + integration suite**

```
pytest tests/test_infer_location.py tests/test_cli_infer.py \
       tests/test_web_geocode.py tests/test_db.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add photosearch/web.py tests/test_web_geocode.py
git commit -m "Add /api/geocode/infer-preview and /api/geocode/infer-apply

Synchronous endpoints wrapping infer_locations(). Preview returns
candidate counts, confidence buckets, hop distribution, and 10 samples
with thumbnails + pre-geocoded place_name. Apply requires confirm=true,
reverse-geocodes the full candidate set, and writes in one transaction
with a WHERE gps_lat IS NULL guard against concurrent overwrites."
```

---

## Task 9 — Frontend `InferLocationForm` panel

**Files:**
- Modify: `frontend/dist/status.html`

No automated frontend tests in this project — the plain-React-UMD
setup has no build/test toolchain. Verification is manual in a browser.

- [ ] **Step 1: Read the existing `StackingForm` component as a reference**

Run:

```
grep -n "StackingForm\b" frontend/dist/status.html | head
```

Note its location (the component definition, where it's rendered, and
its styling hooks). The new component follows the exact same shape:
params → Preview → results summary → Apply with confirm dialog.

- [ ] **Step 2: Add the `InferLocationForm` component**

In `frontend/dist/status.html`, immediately after the `StackingForm`
component definition (before the top-level `App` component), add:

```javascript
function InferLocationForm({ onDone }) {
  const [params, setParams] = React.useState({
    window_minutes: 30,
    max_drift_km: 25.0,
    min_confidence: 0.0,
    cascade: true,
    max_cascade_rounds: 10,
  });
  const [preview, setPreview] = React.useState(null);
  const [previewParamsHash, setPreviewParamsHash] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);

  const paramsHash = JSON.stringify(params);
  const applyReady = preview && previewParamsHash === paramsHash && preview.candidate_count > 0;

  async function runPreview() {
    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/geocode/infer-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setPreview(await r.json());
      setPreviewParamsHash(paramsHash);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runApply() {
    if (!applyReady) return;
    const msg = `Apply ${preview.candidate_count.toLocaleString()} inferences?\n\n` +
                `window=${params.window_minutes}min, max_drift=${params.max_drift_km}km, ` +
                `min_conf=${params.min_confidence}, cascade=${params.cascade}`;
    if (!window.confirm(msg)) return;

    setBusy(true);
    setError(null);
    try {
      const r = await fetch('/api/geocode/infer-apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...params, confirm: true }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      alert(`Applied ${body.updated_count.toLocaleString()} inferences in ${body.duration_seconds.toFixed(1)}s.`);
      // Re-preview so the next tier is reflected.
      setPreview(null);
      setPreviewParamsHash(null);
      await runPreview();
      if (onDone) onDone();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const setParam = (k) => (e) => {
    const raw = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    let val = raw;
    if (e.target.type === 'number') {
      val = raw === '' ? '' : Number(raw);
    }
    setParams({ ...params, [k]: val });
  };

  function renderHistogram(rows, labelKey, valueKey) {
    const max = Math.max(1, ...rows.map(r => r[valueKey]));
    return rows.map((r, i) => React.createElement(
      'div', { key: i, style: { display: 'flex', gap: 8, fontFamily: 'monospace', fontSize: 13 } },
      React.createElement('div', { style: { width: 110 } }, String(r[labelKey])),
      React.createElement('div', {
        style: {
          width: Math.round((r[valueKey] / max) * 200),
          background: '#4a90e2', height: 14,
        }
      }),
      React.createElement('div', null, r[valueKey].toLocaleString()),
    ));
  }

  return React.createElement(
    'div', { className: 'panel' },
    React.createElement('h3', null, 'Infer Locations'),

    // Inputs row
    React.createElement('div', { style: { display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'end', marginBottom: 8 } },
      React.createElement('label', null, 'Window (min)',
        React.createElement('input', { type: 'number', value: params.window_minutes, onChange: setParam('window_minutes'), min: 1, style: { width: 70, display: 'block' } })),
      React.createElement('label', null, 'Max drift (km)',
        React.createElement('input', { type: 'number', value: params.max_drift_km, onChange: setParam('max_drift_km'), min: 0, step: 0.5, style: { width: 90, display: 'block' } })),
      React.createElement('label', null, 'Min conf',
        React.createElement('input', { type: 'number', value: params.min_confidence, onChange: setParam('min_confidence'), min: 0, max: 1, step: 0.05, style: { width: 70, display: 'block' } })),
      React.createElement('label', null,
        React.createElement('input', { type: 'checkbox', checked: params.cascade, onChange: setParam('cascade') }), ' Cascade'),
      React.createElement('label', null, 'Max rounds',
        React.createElement('input', { type: 'number', value: params.max_cascade_rounds, onChange: setParam('max_cascade_rounds'), min: 1, max: 50, style: { width: 70, display: 'block' } })),
    ),

    // Buttons
    React.createElement('div', { style: { display: 'flex', gap: 8, marginBottom: 12 } },
      React.createElement('button', { onClick: runPreview, disabled: busy }, busy ? 'Working…' : 'Preview'),
      React.createElement('button',
        { onClick: runApply, disabled: busy || !applyReady },
        preview && preview.candidate_count > 0
          ? `Apply ${preview.candidate_count.toLocaleString()} inferences`
          : 'Apply'
      ),
    ),

    error && React.createElement('div', { style: { color: '#b00', marginBottom: 8 } }, error),

    // Preview results
    preview && React.createElement('div', null,
      React.createElement('div', { style: { fontWeight: 600, marginBottom: 4 } },
        `${preview.candidate_count.toLocaleString()} candidates out of ${preview.no_gps_count.toLocaleString()} no-GPS photos (cascade ${preview.cascade_rounds_used} rounds).`),
      React.createElement('div', { style: { fontSize: 13, color: '#555', marginBottom: 8 } },
        `Skipped: ` + Object.entries(preview.skipped).map(([k, v]) => `${k} ${v.toLocaleString()}`).join(' · ')),

      React.createElement('div', { style: { marginBottom: 8 } },
        React.createElement('div', { style: { fontWeight: 500 } }, 'Confidence'),
        renderHistogram(preview.confidence_buckets, 'bucket', 'count')),

      React.createElement('div', { style: { marginBottom: 8 } },
        React.createElement('div', { style: { fontWeight: 500 } }, 'Hops'),
        renderHistogram(preview.hop_distribution, 'hops', 'count')),

      preview.samples.length > 0 && React.createElement('div', null,
        React.createElement('div', { style: { fontWeight: 500, margin: '8px 0 4px' } }, 'Samples'),
        ...preview.samples.map(s => React.createElement(
          'div', { key: s.photo_id, style: { display: 'flex', gap: 8, alignItems: 'center', padding: '4px 0' } },
          React.createElement('img', { src: s.thumbnail_url, style: { width: 48, height: 48, objectFit: 'cover' } }),
          React.createElement('div', null,
            React.createElement('div', { style: { fontFamily: 'monospace', fontSize: 12 } }, s.filepath),
            React.createElement('div', { style: { fontSize: 12, color: '#555' } },
              `→ ${s.place_name || `${s.inferred_lat.toFixed(3)}, ${s.inferred_lon.toFixed(3)}`} · ` +
              `conf ${s.confidence.toFixed(2)} · ${s.time_gap_min.toFixed(0)}min · ` +
              `drift ${s.drift_km.toFixed(1)}km · ${s.hop_count} hop${s.hop_count === 1 ? '' : 's'}`
            ))
        ))
      )
    ),
  );
}
```

- [ ] **Step 3: Render the component in the status page**

Find where `StackingForm` is rendered (likely inside the top-level `App`
component's `React.createElement` tree). Add a sibling render
immediately after:

```javascript
React.createElement(InferLocationForm, { onDone: () => load() }),
```

(`load()` is the status page's existing refetch function for
`/api/stats` — check its name and pass the matching callback.)

- [ ] **Step 4: Rebuild the Docker image and verify in a browser**

```
docker compose -f docker-compose.nas.yml build photosearch
docker compose -f docker-compose.nas.yml up -d photosearch
```

Then:
- Open `/status` in a browser.
- Confirm the "Infer Locations" panel appears alongside the stacking form.
- Click **Preview** — expect a summary with histograms + samples in
  under ~10s on the real library.
- Tweak `window_minutes` or `max_drift_km` — confirm **Apply** disables
  until you re-click **Preview**.
- Spot-check sample rows: do the inferred place_names look right? Are
  confidences plausible?
- (Optional for first deploy) Click **Apply** on a high-confidence
  subset (e.g., set `min_confidence=0.75`). Verify via
  `/api/stats` that photos now have GPS and via a direct SQL query that
  `location_source='inferred'` is stamped.

- [ ] **Step 5: Commit**

```bash
git add frontend/dist/status.html
git commit -m "Add InferLocationForm panel to /status

Wraps /api/geocode/infer-preview and /api/geocode/infer-apply. Users
tune window_minutes / max_drift_km / min_confidence / cascade, click
Preview to see aggregate stats + 10 sample inferences with thumbnails,
then Apply (with a confirm prompt). Apply is disabled until the current
params have been previewed."
```

---

## Task 10 — Documentation pass

**Files:**
- Modify: `CLAUDE.md`
- Modify: `.claude/skills/photo-search/SKILL.md`

The project's skill and CLAUDE.md files are load-bearing for future
work — they describe current capabilities, CLI commands, and schema
versions. Not updating them would make the next session miss M19.

- [ ] **Step 1: Update the schema version mention in CLAUDE.md**

Find the "Database" section in `CLAUDE.md` and change:

```
File is `photo_index.db` (not `photos.db`). Schema version 15. Key tables: ...
```

(Currently — check the actual version there; it may say 15 or 16.) Update to:

```
File is `photo_index.db` (not `photos.db`). Schema version 17. Key tables: ...
```

- [ ] **Step 2: Add an "Inferred geotagging" section to CLAUDE.md**

After the "Name extraction in search queries" section (or wherever the
other feature paragraphs live), add:

```markdown
## Inferred geotagging (M19)

`photos` now has `location_source` (`'exif' | 'inferred' | NULL`) and
`location_confidence` (`NULL | (0,1]`) columns stamped on every GPS-bearing
row. `photosearch infer-locations` scans photos with `gps_lat IS NULL` and
copies coordinates from temporal GPS neighbors within a window (default
30min), with a cascade that promotes inferred photos into the anchor set
for subsequent rounds. A movement guard refuses to infer when flanking
anchors disagree by more than `--max-drift-km` (default 25km).

```bash
$DC run --rm photosearch infer-locations [--window-minutes 30] [--max-drift-km 25] \
    [--min-confidence 0.0] [--no-cascade] [--apply]
```

Dry-run (default) prints candidates + a confidence + hop histogram.
`--apply` reverse-geocodes via the offline GeoNames database and writes
`gps_lat`/`gps_lon`/`place_name`/`location_source='inferred'`/
`location_confidence` in one transaction. Rows with pre-existing
`gps_lat` are never overwritten.

Live tuning UI: `/status` has an **Infer Locations** panel that
wraps `POST /api/geocode/infer-preview` and `POST /api/geocode/infer-apply`
for parameter-tuning without re-shelling into the container.

Rollback:

```sql
UPDATE photos
   SET gps_lat=NULL, gps_lon=NULL, place_name=NULL,
       location_source=NULL, location_confidence=NULL
 WHERE location_source='inferred' AND location_confidence < 0.5;
```
```

- [ ] **Step 3: Add the same summary to the skill**

Append an "Inferred geotagging (M19)" section to
`.claude/skills/photo-search/SKILL.md` mirroring the CLAUDE.md entry,
with additional detail: the
`photosearch/infer_location.py` module, the two endpoints, the
`InferLocationForm` component in `frontend/dist/status.html`, and the
schema v17 columns.

- [ ] **Step 4: Update the planned-milestones list**

In both CLAUDE.md and SKILL.md, remove M19 from `docs/plans/*`-style
"Planned milestones" entries (if any still mention it as pending), or
mark it as shipped. Leave the parent `docs/plans/bulk-set-location.md`
entry intact — the manual half is still unimplemented.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md .claude/skills/photo-search/SKILL.md
git commit -m "Document M19 inferred geotagging in CLAUDE.md + skill"
```

---

## Verification checklist (before declaring done)

- [ ] `pytest tests/test_infer_location.py tests/test_cli_infer.py tests/test_web_geocode.py tests/test_db.py -v` — all PASS
- [ ] Full suite: `pytest tests/ -x --ignore=tests/test_face_matching.py -q` — no new failures
- [ ] `/status` panel renders, Preview returns results, Apply writes rows
- [ ] Spot-check a SQL query: `SELECT COUNT(*) FROM photos WHERE location_source='inferred';` returns the expected count
- [ ] Spot-check rollback query on a small subset works as documented
- [ ] On the real NAS library, a Preview run completes in < 30s (validates the no-SSE design choice)

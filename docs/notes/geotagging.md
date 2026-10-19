# Geotagging — history

Inferred geotagging (M19), `/geotag`, `/map`. Condensed rules: `.claude/rules/search.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 2198) on 2026-10-19 -->
## Inferred geotagging (M19)

`photos` now has `location_source` (`'exif' | 'inferred' | NULL`) and
`location_confidence` (`NULL | (0,1]`) columns stamped on every GPS-bearing
row. `add_photo` auto-sets `location_source='exif'` whenever a caller
provides `gps_lat`/`gps_lon` without an explicit source, so provenance is
complete from the first indexing run forward.

`photosearch infer-locations` scans photos with `gps_lat IS NULL` and copies
coordinates from temporal GPS neighbors within a window (default 30min).
The cascade promotes each successful inference into the anchor set for
subsequent photos (sequential time-ordered promote-as-you-go), so chains
form naturally and each photo picks its nearest anchor. A movement guard
refuses to infer when two flanking anchors disagree by more than
`--max-drift-km` (default 25km).

```bash
$DC run --rm photosearch infer-locations [--window-minutes 30] [--max-drift-km 25] \
    [--min-confidence 0.0] [--no-cascade] [--apply]
```

Dry-run (default) prints a candidate summary + confidence + hop histograms
+ 10 sample inferences. `--apply` reverse-geocodes via the offline
GeoNames database and writes
`gps_lat`/`gps_lon`/`place_name`/`location_source='inferred'`/
`location_confidence` in one transaction. The UPDATE carries a
`WHERE gps_lat IS NULL` guard so rows with pre-existing GPS are never
overwritten.

Live tuning UI: `/status` has an **Infer Locations** panel that wraps
`POST /api/geocode/infer-preview` and `POST /api/geocode/infer-apply`.
Apply is disabled until the current params have been previewed, and
re-enables only after a successful preview with matching params.

Manual bulk geotagging: `/geotag` is a 3-panel UI (folder list →
thumbnails with multi-select → typeahead picker) for filling GPS on
photos M19 couldn't reach. Typeahead merges **library places** (distinct
`place_name` values from photos already tagged, via
`/api/geotag/known-places`, instant, no external call) with **Nominatim
forward-geocode results** (`/api/geocode/search`, proxied through the
NAS with a 30-day cache in the `geocode_cache` table). Selecting a
library place resolves its lat/lon via one cached Nominatim call at
apply time. Writes `location_source='manual'` in a single transaction
via `POST /api/photos/bulk-set-location`; overwrite is off by default
(guards existing exif/inferred GPS unless the user explicitly toggles).
The photo panel's **Show** selector picks which photos are listed:
missing GPS (default), + inferred, or **All** (`show_located=true`, which
also reveals fully-tagged folders). All is how you correct *camera* GPS —
a phone-linked camera (Sony via the Creators' App) stamps the phone's
position, so a school field can come out as the neighbouring park. Such
rows are `location_source='exif'`, so replacing them also needs
**Overwrite** ticked; the skipped-count toast says so.

Map view: `/map` plots every GPS-bearing photo (exif + inferred) on a
Leaflet map with marker clustering. Sidebar filters by source
(exif/inferred) and year. Clicking a marker or cluster opens a preview
pane (up to 9 thumbnails) with:
- **Zoom** — fits the map to the cluster's bounds (or zooms to 16 on a
  single marker).
- **Search** — deep-links to `/?location=<suffix>` using the longest
  trailing `"Locality, Admin1, CC"` suffix that covers the majority
  (≥50%) of the cluster's photos. Broadens naturally with altitude:
  tight cluster keeps the full place_name, country-wide cluster
  collapses to a 2-letter country code (prefixed with `", "` so the
  LIKE match anchors to the country slot and doesn't false-positive
  on locality names like "Esterzili"). The header shows coverage when
  below 100% (e.g., "ES — 96% of 938 photos").
Clicking a thumbnail still opens `PS.PhotoModal`. Backed by the
`GET /api/photos/geojson` endpoint (compact tuple format
`[id, lat, lon, source, year, place_name]`, ~700KB gzipped on a
50k-GPS library; place_name is included in the dump so cluster clicks
don't need per-photo fetches).

Rollback (delete every inferred write below a confidence floor):

```sql
UPDATE photos
   SET gps_lat=NULL, gps_lon=NULL, place_name=NULL,
       location_source=NULL, location_confidence=NULL
 WHERE location_source='inferred' AND location_confidence < 0.5;
```

Module: `photosearch/infer_location.py` (`infer_locations`,
`_find_flanking_anchors`, `_infer_one_round`, `haversine_km`).

<!-- moved verbatim from SKILL.md (line 618) on 2026-10-19 -->
## Inferred geotagging (M19)

Fills `gps_lat`/`gps_lon` for photos with missing GPS by interpolating
between temporal GPS neighbors.

**Schema v17 columns** (in `photos`):
- `location_source` — `'exif' | 'inferred' | NULL`. `add_photo()` auto-
  stamps `'exif'` whenever a caller passes `gps_lat`/`gps_lon` without an
  explicit source, so provenance is complete from the first index forward.
- `location_confidence` — `NULL | (0,1]`. Only non-null for inferred
  rows. `1.0 × 0.7^hops × time_decay × side_bonus` roughly.

**Algorithm** (`photosearch/infer_location.py`):
1. `_scan_photos` sorts all photos by `date_taken`. Rows with no
   parseable date are counted and skipped (reported under
   `skipped.no_date_taken`).
2. `_find_flanking_anchors` walks left/right from each no-GPS photo to
   find the nearest anchor on each side within the window. Works
   bidirectionally so cascade chains resolve correctly regardless of
   scan order.
3. Cascade path is **sequential time-ordered promote-as-you-go**: each
   successful inference becomes an anchor for subsequent photos. Each
   new inference picks its NEAREST anchor (typically its just-inferred
   predecessor), so chains compound confidence multiplicatively rather
   than anchoring the whole chain to a single distant real-GPS photo.
4. **Movement guard**: if two flanking anchors disagree by more than
   `max_drift_km` (default 25), inference is refused. `skipped.movement_guard`
   surfaces this.

**Surfaces:**
- **CLI:** `photosearch infer-locations [--window-minutes 30]
  [--max-drift-km 25] [--min-confidence 0.0] [--no-cascade] [--apply]`.
  Dry-run default prints summary + histograms + 10 samples.
- **API:** `/api/geocode/infer-preview` + `/api/geocode/infer-apply`
  (see above).
- **UI:** `/status` has an **Infer Locations** panel
  (`InferLocationForm` in `frontend/dist/status.html`) that wraps the
  two endpoints. Apply is disabled until the current params have been
  previewed, and re-disables on any param edit until a re-preview.

**Rollback:** every inferred write below a confidence floor can be
nulled in one query, since `location_source='inferred'` is only ever
stamped by `infer-apply`:

```sql
UPDATE photos
   SET gps_lat=NULL, gps_lon=NULL, place_name=NULL,
       location_source=NULL, location_confidence=NULL
 WHERE location_source='inferred' AND location_confidence < 0.5;
```

Tests:
`tests/test_infer_location.py` (engine, 20 cases including UTF-8
`place_name` roundtrip), `tests/test_cli_infer.py` (CLI),
`tests/test_web_geocode.py` (API), plus 3 cases in `tests/test_db.py`
for the `add_photo` auto-stamp.

---

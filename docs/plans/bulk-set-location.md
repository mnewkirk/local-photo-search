# Bulk set location + inferred geotagging (future milestone — absorbs M19)

Let photos that lack GPS get a location from one of two sources — a user-
chosen address (manual) or a GPS-bearing photo nearby in time (inferred,
formerly M19). Both paths land in the same DB columns through the same
endpoint, so they're developed together. Typical targets are photos from
older cameras, film scans, or cameras in airplane mode — today the only way
to attach a place is to re-shoot with GPS or hand-edit the DB.

## Two sources, one sink

| | Manual (bulk-set) | Inferred (was M19) |
|---|---|---|
| Trigger | User selects photos, types address | Scanner finds no-GPS photo within N minutes of a GPS-bearing photo |
| Needs forward geocoding | Yes (address → lat/lon) | No (lat/lon copied from neighbor) |
| Confidence | Exact (user-confirmed) | Score based on time gap + neighbor count |
| Undo-ability | Same | Same |

Both paths call `POST /api/photos/bulk-set-location` under the hood and
write the same columns. The inferred path additionally stamps a
`location_source='inferred'` marker and a confidence score so the UI can
distinguish and let the user review before accepting.

## Desired UX

- Multi-select on `/` (search results) and `/collections/{id}` — the existing
  selection bar gets a **"Set location…"** action alongside the current bulk
  operations.
- The input is a single text field with **autocomplete**. User types an address
  or place name (`"Olympic National Park"`, `"1600 Pennsylvania Ave"`,
  `"Bondi Beach, Sydney"`) and picks a suggestion. Pasting a `lat,lon` pair
  should also work and skip autocomplete.
- On confirm: every selected photo gets `gps_lat`, `gps_lon`, and `place_name`
  set from the chosen suggestion. Photos that already have GPS keep theirs
  unless the user checks **"Overwrite existing"**.

## Geocoding provider

The project's current reverse geocoding (`photosearch/geocode.py`) is
offline-only via the `reverse_geocoder` package — lat/lon → nearest city.
Forward geocoding (address → lat/lon) needs more than GeoNames can offer.

Options, ordered by how well they fit the local-first posture:

1. **Nominatim (OpenStreetMap) self-hosted** — Docker image exists, but the
   planet extract is ~150 GB. Overkill for a home NAS. Regional extracts
   (e.g. North America) are ~20 GB and workable.
2. **Nominatim public API** — free, rate-limited to 1 req/sec, requires a
   descriptive User-Agent. Fine for interactive autocomplete where the user
   is typing at human speed. No account required.
3. **Photon** (OSM-based, Komoot-hosted) — purpose-built for autocomplete,
   public endpoint, no key. Same terms-of-use considerations as Nominatim.
4. **Pelias** — self-hosted alternative to Nominatim, more autocomplete-
   focused, but heavier stack (Elasticsearch).

Recommendation: start with **public Nominatim** (debounced typeahead, server-
side proxy so the User-Agent is consistent and we can cache), with a config
flag to swap in a self-hosted endpoint later. This keeps the initial change
small while leaving the door open for fully-offline mode.

**Privacy note**: proxying through the backend means the NAS's IP hits
Nominatim, not the user's browser. The project's posture is "photos never
leave the machine" — the text the user types in the location search box
would leave. Worth calling out in the UI.

## Backend sketch

New module `photosearch/geocode_forward.py`:

```python
def search_places(query: str, limit: int = 5) -> list[dict]:
    """Forward-geocode a query to candidate places.
    Returns [{display_name, lat, lon, country, admin1, admin2, locality,
    type, importance}]."""
```

New module `photosearch/infer_location.py`:

```python
def infer_locations(db, window_minutes: int = 30,
                    max_distance_km: float = 50.0) -> list[dict]:
    """For every photo with NULL gps, find GPS-bearing neighbors within
    window_minutes of date_taken. Returns [{photo_id, lat, lon,
    confidence, neighbor_ids}]. confidence decays with time gap and
    drops if neighbors span >max_distance_km (you moved during the
    window)."""
```

New endpoints (`photosearch/web.py`):

- `GET /api/geocode/search?q=...&limit=5` — forward geocode proxy + cache.
  Response cached in a new `geocode_cache` table keyed by lowercased query,
  TTL ~30 days.
- `POST /api/photos/bulk-set-location` — body `{photo_ids: [...], lat, lon,
  place_name, country?, admin1?, admin2?, locality?, source:
  'manual'|'inferred', confidence?: float, overwrite: bool}`. Returns
  `{updated_count}`. Single transaction. Respects `overwrite=false` by
  skipping photos where `gps_lat IS NOT NULL`.
- `GET /api/geocode/infer-preview?window_minutes=30` — runs the inference
  scanner read-only and returns candidate assignments for review. The UI
  shows them grouped by neighbor-photo so the user can accept/reject in
  batches.
- `POST /api/geocode/infer-apply` — accepts a list of inference results
  and writes them via `bulk-set-location` with `source='inferred'`.

New CLI:

- `photosearch infer-locations [--window-minutes 30] [--dry-run] [--apply]`
  — scanner that walks no-GPS photos and finds GPS-bearing temporal
  neighbors. `--dry-run` prints candidates; `--apply` writes them. Useful
  for one-shot backfills after importing a batch of smartphone photos
  (M20 / Google Photos import) that fill in the GPS-bearing neighbor set.

## DB changes

`gps_lat`, `gps_lon`, `place_name` already exist on `photos`
(`db.py:218-220`), but `place_name` today is a single flat string from the
offline reverse geocoder (usually city-level). To unlock future map view and
region-scoped queries like *"beach near southwest France"*, we want to also
store the structured place hierarchy that Nominatim returns. New columns on
`photos`:

```sql
ALTER TABLE photos ADD COLUMN country TEXT;             -- "France"
ALTER TABLE photos ADD COLUMN admin1  TEXT;             -- "Nouvelle-Aquitaine"
ALTER TABLE photos ADD COLUMN admin2  TEXT;             -- "Gironde"
ALTER TABLE photos ADD COLUMN locality TEXT;            -- "Bordeaux"
ALTER TABLE photos ADD COLUMN location_source TEXT;     -- NULL | 'exif' | 'manual' | 'inferred'
ALTER TABLE photos ADD COLUMN location_confidence REAL; -- NULL for exif/manual; [0,1] for inferred
CREATE INDEX IF NOT EXISTS idx_photos_country ON photos(country);
CREATE INDEX IF NOT EXISTS idx_photos_admin1  ON photos(admin1);
```

Both `place_name` (exact Nominatim display string) and the structured fields
get filled in — the display string preserves user intent; the structured
fields enable hierarchical filtering. `gps_lat`/`gps_lon` are still the
source of truth for any distance math.

Backfill path: a `normalize-places` CLI that re-runs the (forward or reverse)
geocoder on every existing row that has lat/lon but empty structured fields.
Safe to run repeatedly; uses the `geocode_cache` to avoid hammering Nominatim.

A small `geocode_cache` table for the autocomplete proxy:

```sql
CREATE TABLE IF NOT EXISTS geocode_cache (
    query TEXT PRIMARY KEY,       -- lowercased user input
    results_json TEXT NOT NULL,   -- JSON array of suggestions
    fetched_at TEXT NOT NULL      -- ISO timestamp
);
```

Bump `SCHEMA_VERSION` and add migration in `_init_schema()`.

## Frontend sketch

- Reuse `PS.SharedHeader`'s selection-count pattern from collections.html.
  Add a **"Set location…"** button to the bulk action bar on
  `frontend/dist/index.html` and `collections.html`.
- New modal (`PS.SetLocationModal` in `shared.js`) — text input bound to
  `GET /api/geocode/search` with 250ms debounce. Shows up to 5 suggestions
  with the OSM display name and a small map thumbnail (tile server URL
  directly — no build step). Includes an **"Overwrite existing GPS"**
  checkbox, defaults off. Also includes an **"Auto-fill from temporal
  neighbors"** option that calls `infer-preview` for the selected photos
  and pre-populates the confirmation with the inferred place (fall-through
  to manual input if no neighbor is found).
- On confirm, call `POST /api/photos/bulk-set-location` and refresh the
  grid. Toast shows `"Set location for N photos"`.
- New standalone page **`/geotag-review`** — surfaces all inference
  candidates across the whole library (not just a current selection).
  Renders neighbor-grouped cards (like `/merges`) so a user can sweep
  through a trip in one sitting. Keyboard shortcuts mirror `/merges`:
  `A` accept, `D` dismiss, `J/K` next/prev.

## Open questions

- ~~Should bulk-set also regenerate `place_name` via the offline reverse
  geocoder (for consistency with GPS-tagged photos) or use the exact
  Nominatim display name?~~ **Resolved**: store both. `place_name` keeps
  the exact display string; new structured columns (`country`, `admin1`,
  `admin2`, `locality`) give uniform hierarchy for filtering.
- Undo: the previous (lat, lon, place_name) is lost once overwritten.
  Either skip undo (acceptable — user selected the photos deliberately)
  or stash the pre-change tuple in a `location_history` table for a
  single-level undo. Start without; add if users ask.
- Paste format for `lat,lon` — accept `"47.6, -122.3"`, `"47.6°N 122.3°W"`,
  Google Maps URL paste (`"...@47.6,-122.3,..."`)? Start with the first
  two; URL paste is a nice-to-have.

## Related / downstream milestones

Once every photo has lat/lon + a structured hierarchy, several things
become straightforward:

- **Map view** — a new `/map` page that plots every photo with non-null
  `gps_lat`/`gps_lon`. Cluster at low zoom (Leaflet + marker clustering,
  no build step so it fits the project's plain-React posture). Click a
  cluster → filters the search grid to that bbox.
- **Radius search** — `?location_near=47.6,-122.3&radius_km=5` filters
  by Haversine distance. Cheap in SQLite via a bbox prefilter on
  `gps_lat`/`gps_lon` indexes, then exact distance in Python.
- **Region-scoped semantic queries** — *"beach near southwest France"*.
  The query parser (see `extract_location_from_query`) would recognize
  "southwest France" as a region descriptor, translate it to a
  bbox/admin1 filter (`country='France' AND admin1 IN (...)`), then run
  the residual CLIP query ("beach") against that subset. Requires a
  small hand-curated region→admin1 map for phrases like "southwest",
  "new england", "pacific northwest".
- **Place-hierarchy browse** — sidebar tree on `/` that shows
  `country > admin1 > locality` with photo counts, clicking any node
  filters the grid.

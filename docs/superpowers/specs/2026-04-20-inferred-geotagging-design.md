# Inferred geotagging (M19) — design

Date: 2026-04-20
Status: design approved, awaiting implementation plan
Supersedes: the "inferred" half of `docs/plans/bulk-set-location.md`

## Summary

Backfill `gps_lat` / `gps_lon` / `place_name` on photos that lack GPS by copying
coordinates from GPS-bearing photos nearby in time (temporal neighbors). Writes
are stamped with a new `location_source='inferred'` marker and a
`location_confidence` score so they can be audited, filtered, or rolled back
later.

Motivated by the recent phone-photo import: phone photos carry dense GPS, and
many camera photos were taken at the same trips/events but lack GPS. Temporal
inference turns the phone-photo GPS anchors into locations for those camera
photos.

Shipped as a CLI + a `/status`-page panel for parameter tuning. No per-photo
review UI in this slice.

## Non-goals

This spec deliberately excludes (to be picked up by the parent
`bulk-set-location.md` plan or later milestones):

- **Manual bulk-set location** (user types an address, forward geocoding via
  Nominatim). The parent plan bundles this with inferred; we are scoping to
  inferred only.
- **Structured location hierarchy columns** (`country`, `admin1`, `admin2`,
  `locality`). `place_name` (produced by the existing offline
  `reverse_geocode_batch`) is enough for current filtering; hierarchical
  columns are deferred until a downstream feature (map view, region-scoped
  queries) actually consumes them.
- **Per-photo accept/reject review page** (`/geotag-review` from the parent
  plan). Replaced in this slice by aggregate tuning in the `/status` panel —
  see Frontend.
- **Overwriting existing GPS.** Inferred writes only fill rows where
  `gps_lat IS NULL`.
- **Undo / location history.** Rollback is via
  `UPDATE photos SET gps_lat=NULL ... WHERE location_source='inferred'`
  (or a confidence filter on that set).

## Data model

Schema bump `16 → 17`. Add two columns to `photos`:

```sql
ALTER TABLE photos ADD COLUMN location_source     TEXT;  -- NULL | 'exif' | 'inferred'
ALTER TABLE photos ADD COLUMN location_confidence REAL;  -- NULL for exif; (0,1] for inferred
CREATE INDEX IF NOT EXISTS idx_photos_location_source ON photos(location_source);
```

Migration backfills `location_source='exif'` for every existing row with
`gps_lat IS NOT NULL` (EXIF is the only current source of GPS). `place_name`
remains populated via the existing post-index `reverse_geocode_batch` path
and is written in the same transaction as inferred coords.

## Algorithm

Pure-functional core in new module `photosearch/infer_location.py`:

```python
def infer_locations(
    db,
    *,
    window_minutes: int = 30,
    max_drift_km: float = 25.0,
    min_confidence: float = 0.0,
    cascade: bool = True,
    max_cascade_rounds: int = 10,
) -> Iterator[dict]:
    """Yield one dict per inferable no-GPS photo.
    Each dict: {photo_id, lat, lon, place_name=None, confidence,
                hop_count, time_gap_min, drift_km, sides,
                source_photo_id}
    Read-only — caller decides whether to write."""
```

### Scan

One SQL query, sorted by `date_taken`:

```sql
SELECT id, date_taken, gps_lat, gps_lon
FROM photos
WHERE date_taken IS NOT NULL
ORDER BY date_taken
```

Photos with `date_taken IS NULL` are counted in the summary as
`skipped.no_date_taken` and ignored. A time-sorted walk maintains flanking
anchor pointers so each photo sees its left/right anchors in O(1) after an
O(N) scan.

### Direct inference (round 1)

For each no-GPS photo:

1. Find the nearest GPS-bearing anchors on each side within
   `window_minutes`. If neither exists → `skipped.no_anchor`.
2. If **both** exist, compute Haversine distance between them. If
   `> max_drift_km` → `skipped.movement_guard`. (You were moving during the
   window — refuse to guess.)
3. Pick the anchor with the smaller time gap. (Tie-break: earlier side.)
4. Compute confidence:

   ```
   base_decay   = max(0, 1 - time_gap_minutes / window_minutes)
   sides_factor = 1.0 if two flanking anchors else 0.7
   confidence   = base_decay * sides_factor * anchor.confidence
   ```

   Real GPS anchors have `confidence = 1.0`.

5. If `confidence <= min_confidence` → `skipped.below_min_confidence`.
   (At the default `min_confidence=0.0`, this excludes exact-zero
   confidence, which happens when the time gap equals the window.)
6. Yield `{photo_id, lat, lon, confidence, hop_count=1, sides,
   time_gap_min, drift_km, source_photo_id}`. `source_photo_id` is the
   immediate anchor used — for cascade hops this is the previous-round
   inferred photo, not the ultimate real-GPS root.

### Cascade (rounds 2+, default on)

After round 1, newly-inferred photos become eligible anchors for round 2, and
so on. Each round repeats the direct-inference pass over the still-no-anchor
set, but the anchor pool now includes the previous round's results.

- Anchor carries its `confidence` forward; compounding formula above makes a
  5-hop chain at gap=20min/window=30 land at `0.33^5 ≈ 0.004`, so chains
  self-limit long before hitting `max_cascade_rounds`.
- **Movement guard transfers** — the drift check compares the flanking
  anchors' `lat`/`lon` regardless of whether those anchors are real or
  inferred. "Were you stationary during this window" is a fact about space,
  not provenance.
- Loop terminates when a round adds zero new anchors, or at
  `max_cascade_rounds` (default 10) as a defensive ceiling.
- `hop_count` = 1 + the source anchor's hop_count. Real GPS anchors start at
  hop 0; direct inferences are hop 1; cascade inferences are 2+.

### Haversine

Hand-implemented using `math.radians` / `math.sin` / `math.cos` — no new
dependency. Earth radius 6371 km.

## CLI

New command in `cli.py`:

```
photosearch infer-locations [OPTIONS]

  --window-minutes INTEGER       Default: 30
  --max-drift-km FLOAT           Default: 25.0
  --min-confidence FLOAT         Default: 0.0  (filter below this)
  --cascade / --no-cascade       Default: --cascade
  --max-cascade-rounds INTEGER   Default: 10
  --apply                        Write inferences. Without it, prints dry-run.
  --db TEXT                      envvar=PHOTOSEARCH_DB (default: photo_index.db)
```

**Dry-run output:**

```
Scanning 312,401 photos... 89,204 have no GPS, 223,197 have GPS.
Inferring with window=30min, max_drift=25km, min_conf=0.0, cascade=on...

Cascade: 4 rounds to fixpoint.
Candidates: 41,883 of 89,204 no-GPS photos (46.9%)
Skipped:
  no_anchor              42,011
  movement_guard          5,310
  no_date_taken              0
  below_min_confidence       0

Confidence distribution:
  >=0.90        18,402  ████████████████
  0.75-0.90      9,831  ████████
  0.50-0.75      8,004  ███████
  0.25-0.50      4,211  ████
  <0.25          1,435  █

Hop distribution:
  1 (direct)    28,401  ███████████████
  2              9,112  █████
  3              3,004  ██
  4+             1,366  █

Sample inferences (10 random):
  /photos/2018/2018-07-14/IMG_0041.CR2
    -> 47.6205, -122.3493 (Seattle, Washington, US)
       gap=3min, drift=1.2km, confidence=0.90, hops=1
  ...

Re-run with --apply to write these.
```

**`--apply` output:** same summary, plus:

```
Writing 41,883 inferences...
  Reverse-geocoded 41,880 place_names.
  Transaction committed.
```

`--apply` performs reverse geocoding once on the full candidate set via
`reverse_geocode_batch`, then writes everything in a single transaction
(`gps_lat`, `gps_lon`, `place_name`, `location_source='inferred'`,
`location_confidence`). Rows without a resolved place string still get the
coords — `place_name` stays NULL for those ~0.01% edge cases.

Exits non-zero if the DB has zero GPS-bearing photos (nothing to infer
from — likely a misconfigured DB or PHOTOSEARCH_DB path).

## API

Two new endpoints in `photosearch/web.py`, synchronous. Inference is an
O(N) scan in Python — seconds on a 500k-photo library. If measurement later
shows otherwise, add SSE using the stacking pattern.

```
POST /api/geocode/infer-preview
  body: {window_minutes, max_drift_km, min_confidence,
         cascade, max_cascade_rounds}
  200: {
    total_photos, no_gps_count, gps_count,
    candidate_count,
    cascade_rounds_used,
    skipped: {no_anchor, movement_guard, no_date_taken, below_min_confidence},
    confidence_buckets: [{bucket: ">=0.90", count: N}, ...],
    hop_distribution:   [{hops: 1, count: N}, ...],
    samples: [  // 10 random candidates
      {photo_id, thumbnail_url, filepath, inferred_lat, inferred_lon,
       place_name, confidence, hop_count, time_gap_min, drift_km, sides,
       source_photo_id}
    ]
  }

POST /api/geocode/infer-apply
  body: same fields as preview + {confirm: true}
  200: {updated_count, rounds_used, duration_seconds}
  400:  if confirm != true
```

Preview is read-only — safe to spam while tuning sliders. Apply requires
`confirm: true` to guard against accidental fires with a stale form.

Thumbnails in samples reuse `/api/photos/{id}/thumbnail` — no new image
route.

## Frontend

New React component `InferLocationForm` in `frontend/dist/status.html`,
rendered as a sibling panel to the existing `StackingForm`. Follows the
same shape (inline form + params + preview/apply pattern, no separate
page).

```
┌─ Infer Locations ───────────────────────────────────────────┐
│  Window (min) [30]   Max drift (km) [25]   Min conf [0.0]   │
│  [x] Cascade   Max rounds [10]                              │
│  [ Preview ]    [ Apply 41,883 inferences ]                 │
├─────────────────────────────────────────────────────────────┤
│  Preview results                        41,883 candidates   │
│  Skipped: no_anchor 42,011 · movement 5,310 · ...           │
│  Confidence   >=0.90 ████████████ 18,402                    │
│               0.75+  ████████      9,831                    │
│               ...                                           │
│  Hops         1 ████████████ 28,401                         │
│               2 ████           9,112                        │
│               3+ ███           4,370                        │
│                                                             │
│  Samples (10 random):                                       │
│  ┌──────┐ /photos/2018/IMG_0041.CR2                         │
│  │ img  │ -> Seattle, WA (47.62, -122.35)                   │
│  └──────┘    conf 0.90 · 3min · 1.2km · 1 hop               │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

**Behaviors:**

- Params start at the CLI defaults.
- **Preview** button calls `POST /api/geocode/infer-preview`, renders
  results into the summary area.
- **Apply** button is disabled until a preview has been rendered with the
  current param values. Any param change re-disables **Apply** and requires
  another **Preview**.
- **Apply** pops a confirm dialog showing candidate count and the params,
  then POSTs `infer-apply` with `confirm: true`. On success, shows a toast
  (`Applied N inferences in Ks`) and re-runs **Preview** so the next batch
  (lower confidence tier, wider window) can be reviewed against the new
  baseline.
- No cancel button (synchronous, fast).

No changes to `shared.js` (no new nav link — the panel lives on
`/status`).

## Testing

**Unit tests — new file `tests/test_infer_location.py`:**

1. Empty DB → `[]`.
2. No GPS anchors in the DB → zero candidates, clean.
3. Single GPS anchor + one no-GPS photo within window → one-sided
   inference; `confidence = 0.7 * base_decay`; `hop_count=1`; `sides` field
   records `'left'`/`'right'`.
4. Two flanking anchors agreeing within `max_drift_km` → nearest-time
   wins; `sides='both'`; no one-sided penalty.
5. Two flanking anchors disagreeing > `max_drift_km` → skipped.movement_guard.
6. Gap equals `window_minutes` → base_decay = 0 → filtered (confidence must
   be > 0).
7. Cascade 3-hop chain (real at t=0, no-GPS at t=20/40/60, window=30) →
   confidences `0.33`, `0.11`, `0.036`; `hop_count` 1/2/3; rounds_used=3.
8. Cascade fixpoint terminates without hitting `max_cascade_rounds` when
   isolated no-GPS clusters exist.
9. `--no-cascade` parity: same chain as (7) only anchors t=20; t=40/t=60
   land in `skipped.no_anchor`.
10. Movement-guard transitivity: real at t=0 (Seattle) and t=120 (Portland),
    window=30, max_drift=25km. t=30/t=90 anchor one-sided. t=60 has
    flanking inferred anchors ~250km apart → movement_guard fires.
11. `date_taken IS NULL` rows counted separately, not crashed on.
12. Haversine fixture test: (47.62, -122.35) ↔ (45.52, -122.68) ≈ 234km.

**Integration tests — new file `tests/test_web_geocode.py`:**

13. `POST /api/geocode/infer-preview` with seeded DB returns expected
    counts + at least one sample.
14. `POST /api/geocode/infer-apply` without `confirm: true` → 400.
15. Apply with `confirm: true` writes: `gps_lat`/`gps_lon` populated,
    `place_name` via reverse_geocode_batch, `location_source='inferred'`,
    `location_confidence` in `(0, 1]`. Rows with pre-existing `gps_lat` are
    untouched.
16. Schema migration test: open a fixture DB at `SCHEMA_VERSION=16`,
    verify v17 migration adds columns and backfills
    `location_source='exif'` on pre-existing GPS rows.

**Manual (not regression-gated):**

- UI panel on a real NAS library in a browser.
- End-to-end timing on the full library (validates the "seconds, not
  minutes" premise underlying the no-SSE decision).

## Rollout notes

1. Deploy schema v17 — the migration is additive (two nullable columns +
   a backfill on existing GPS-bearing rows). Safe to deploy on a running
   DB; a fresh `docker compose up -d photosearch` runs `_init_schema()` on
   boot.
2. Tune parameters via the `/status` panel.
3. Apply inferences in passes: run at high min_confidence first (e.g.,
   0.75), verify samples, then lower threshold for subsequent passes on
   the remaining no-GPS set.
4. Rollback at any point:
   ```sql
   UPDATE photos
      SET gps_lat=NULL, gps_lon=NULL, place_name=NULL,
          location_source=NULL, location_confidence=NULL
    WHERE location_source='inferred';
   ```
   (Or with a `location_confidence < X` predicate to only revert the weak
   inferences.)

## Open follow-ups

- **Per-photo review page (`/geotag-review`)** — deferred. Re-evaluate
  after the first real apply: does aggregate tuning give enough control,
  or do you want per-photo accept/reject before writing?
- **Structured hierarchy columns** (`country`/`admin1`/`admin2`/`locality`)
  — picked up by whichever downstream feature needs them first (map view,
  region search).
- **Manual bulk-set** — the other half of `bulk-set-location.md`, still
  open.
- **Phone-dedup + inferred geotag pipeline** — after M20 (Takeout import),
  run `infer-locations` once to backfill camera photos bracketed by newly
  imported phone GPS.

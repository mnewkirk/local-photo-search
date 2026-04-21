# Photo Search

Local, Docker-deployed photo search system. Photos never leave the machine.
Runs on a UGREEN NAS (Intel N100, no GPU), developed on Mac.

## Tech Stack

Python 3.11, FastAPI, SQLite (WAL) + sqlite-vec, CLIP (ViT-B/16) for semantic search,
InsightFace for faces, LLaVA via Ollama for descriptions, CLIP ViT-L/14 for quality scoring.
Frontend is plain React (UMD, no build step) in `frontend/dist/`. Docker Compose for deployment.

## Key Patterns — Follow These

1. **Streaming generators** for batch jobs — yield results incrementally, save per-batch
2. **`envvar="PHOTOSEARCH_DB"`** on every CLI command's `--db` option
3. **Frontend has no build step** — edit HTML directly, use `React.createElement` not JSX.
   Changes require Docker image rebuild (`COPY frontend/ frontend/`)
4. **Docker cache** — `requirements.txt` changes = ~5min rebuild; Python changes = ~10s
5. **SSE for long-running API ops** — use `asyncio.get_running_loop()` (not `get_event_loop()`),
   always include terminal events (`done`, `fatal`, `cancelled`)
6. **Per-file ledger writes** for cancel safety (uploads, etc.)

## Database

File is `photo_index.db` (not `photos.db`). Schema version 17. Key tables: photos, faces,
persons, face_references, collections, collection_photos, photo_stacks, stack_members,
review_selections, google_photos_uploads, ignored_clusters, schema_info.

Schema migrations run automatically via `_init_schema()`. Bump `SCHEMA_VERSION` in `db.py`
when adding tables/columns.

## NAS Docker Commands

```bash
DC="docker compose -f docker-compose.nas.yml"

$DC build photosearch              # Rebuild after code changes
$DC up -d photosearch              # Restart web server
$DC run --rm photosearch <cmd>     # Run CLI command

# Ad-hoc Python / shell inside the container — MUST override entrypoint.
# docker-entrypoint.sh shell-routes every non-"serve"/"index" arg to
# `python cli.py …`, so `… photosearch python -c "…"` becomes
# `python cli.py python -c "…"` and fails with "No such command 'python'".
# --entrypoint is a `docker compose run` flag and MUST appear before the
# service name (photosearch), so it doesn't fit inside the $DC alias — use
# dedicated aliases instead:
DCPY="docker compose -f docker-compose.nas.yml run --rm --entrypoint python"
DCSH="docker compose -f docker-compose.nas.yml run --rm --entrypoint bash"
$DCPY photosearch -c "<snippet>"
$DCSH photosearch -c "<shell command>"

# IMPORTANT: PhotoDB() with no args defaults to the RELATIVE path "photo_index.db",
# which resolves to /app/photo_index.db inside the container (an empty stub DB).
# The PHOTOSEARCH_DB env var is only wired into cli.py's --db defaults, NOT into
# PhotoDB.__init__. Every ad-hoc snippet must pass it explicitly, or you'll get
# zero-row results and a bogus /app/photo_index.db will be created:
$DCPY photosearch -c "
import os
from photosearch.db import PhotoDB
with PhotoDB(os.environ['PHOTOSEARCH_DB']) as db:
    ...
"

# Git pull (Alpine workaround for UGOS ownership)
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo pull"
```

## Distributed Indexing (Worker)

Offload heavy indexing to a fast laptop while the NAS keeps the DB + photos.
Worker claims batches via HTTP, downloads photos, processes locally, POSTs results back.

**Docker worker fleet (recommended)** — avoids PyTorch MPS memory leak on macOS:
```bash
./run-workers.sh -s http://<NAS-IP>:8000 -p clip -d /photos/2026 -n 4
./run-workers.sh --status    # containers + memory + progress
./run-workers.sh --logs      # tail all workers live
./run-workers.sh --stop      # stop all workers
```
Uses CPU-only PyTorch with 3GB hard memory limit per container. Use NAS IP address
(not hostname) — Docker containers can't resolve local DNS names.

**Bare-metal (single quick test only):**
```bash
python cli.py worker -s http://nas.local:8000 -p clip,quality -d /photos/2026
```

Key files: `photosearch/worker.py` (client), `photosearch/worker_api.py` (server endpoints),
`run-workers.sh` (Docker fleet launcher). API routes under `/api/worker/*`.
Claims have TTL (default 30min) for crash recovery.

## Vec0 orphan cleanup

`clip_embeddings` and `face_encodings` are sqlite-vec `vec0` virtual tables.
Virtual tables can't participate in foreign-key constraints, so
`ON DELETE CASCADE` on `photos`/`faces` doesn't reach them — when a photo or
face row is deleted, its vector row stays behind. The status page will show
`>100% embedded` when this happens. AUTOINCREMENT on `photos.id` / `faces.id`
prevents ID reuse, so the dangling rows can never silently re-attach to a
different photo.

Also: `add_clip_embedding` uses explicit DELETE+INSERT (not `INSERT OR
REPLACE`) because vec0 doesn't honor `OR REPLACE` — the PK conflict fires
first and raises UNIQUE, which causes worker submits to fail when a claim
TTL expired and another worker re-processed the same photo.

```bash
$DC photosearch cleanup-orphans [--dry-run]
```

## Adding Features

- **New CLI command:** Add to `cli.py`, always include `envvar="PHOTOSEARCH_DB"` on `--db`
- **New search type:** `search.py` → `search_combined()` → `web.py` param → `index.html` UI
- **New indexing pass:** Add to both `index_directory()` and `_index_collection()` in `index.py`,
  add `--flag`/`--force-flag` to `cli.py`, use streaming generator pattern
- **New API endpoint:** `web.py` with `_get_db()`, SSE for long ops
- **Schema change:** Bump `SCHEMA_VERSION`, add migration SQL in `_init_schema()`

## Name extraction in search queries

`search.py:_extract_persons_from_query` parses registered person names out of
the free-text `q=` parameter and turns them into AND-intersected person
filters. `?q=Calvin and Ellie` matches photos containing both people
(case-insensitive, word-bounded, longest-first so "Matt Newkirk" wins over
"Matt"), strips connector tokens ("and", "with", "&", ","), and sends any
residual to CLIP. `(?<!-)` lookbehind keeps `-Calvin` as a CLIP exclusion.
Stacks cleanly with dates/locations/explicit `person=` via `search_combined`'s
existing result-set intersection — no special casing.

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

Rollback (delete every inferred write below a confidence floor):

```sql
UPDATE photos
   SET gps_lat=NULL, gps_lon=NULL, place_name=NULL,
       location_source=NULL, location_confidence=NULL
 WHERE location_source='inferred' AND location_confidence < 0.5;
```

Module: `photosearch/infer_location.py` (`infer_locations`,
`_find_flanking_anchors`, `_infer_one_round`, `haversine_km`).

## Status page live actions

`/status` now runs jobs and monitors the worker fleet directly, not just as
copy-paste bash snippets:

- **Workers panel** (above the activity chart) polls `/api/worker/status`
  every 5s. Shows total active workers + queued photos, per-pass queue pills
  (clip / faces / quality / describe / tags / verify), and one row per active
  claim with pass type, worker_id, photo count, live TTL (yellow <2min, red
  when expired). A 1s tick re-renders TTLs between fetches.
- **Tags card** — `/api/stats` gained a `tagged` field; the card uses the
  same progress-bar pattern as CLIP/quality/etc.
- **Stacking form** — runs `POST /api/stacks/detect/stream` (SSE) with
  editable `time_window_sec` / `clip_threshold` / `max_stack_span_sec`, plus
  `clear` + `dry_run` toggles. Live progress card shows phase label
  ("Loading CLIP embeddings", "Comparing pairs", "Saving stacks"), a progress
  bar with done/total, pair count, and elapsed seconds. Cancel calls
  `AbortController.abort()`; server sees `request.is_disconnected()`, flips
  a `threading.Event`, and stacking.py's hot loop checks `should_abort` every
  ~1000 iterations before raising `InterruptedError`. Blocking
  `POST /api/stacks/detect` is kept for scripts/curl.

The `on_progress` + `should_abort` callback pair added to `stacking.py` is
the reference shape for instrumenting other long-running jobs (reclustering,
describe, etc.) with SSE progress + cancel.

## Face clustering

New faces land with `cluster_id = NULL` — per-batch clustering was removed from
`index_directory()`, `_index_collection()`, and `worker_api.submit_results()`
because every batch independently restarted IDs at 0 and collided
("Unknown #0" was the union of the first face of every batch).

Grouping is an on-demand step:

```bash
$DC run --rm photosearch recluster-faces \
  [--eps 0.55] [--min-samples 3] \
  [--no-session-stacking] [--session-eps 0.50] [--session-window 60] \
  [--dry-run]
```

Runs global DBSCAN over every `person_id IS NULL` encoding, then (by default)
a **session-stacking second pass**: union-find over the DBSCAN noise points,
linking pairs whose L2 distance is within `--session-eps` AND whose
`date_taken` is within `--session-window` minutes. Components of size ≥ 2
become new clusters continuing past the DBSCAN id range, recovering
same-person-same-event groups that min_samples=3 had discarded. Pass
`--no-session-stacking` for DBSCAN-only behavior. `ignored_clusters` is
cleared in the same transaction (IDs are fully renumbered). Loader uses
`np.frombuffer` over concatenated BLOBs — 120k faces decode in seconds on
an N100; DBSCAN at 512-dim with ball_tree is the dominant cost after that.

## Splitting attractor clusters (M18)

Large muddled clusters (observed on the NAS: one cluster with 126 faces
across 469 days) pull in multiple distinct people during the eps=0.55
global recluster. Splitting them is a per-cluster DBSCAN re-run with
tighter params:

```bash
$DC run --rm photosearch split-cluster 15 --dry-run       # preview
$DC run --rm photosearch split-cluster 15 --eps 0.45      # apply
```

Also available from the /faces detail sidebar: "Split" button on any
unknown cluster, preview shows the histogram of resulting sub-cluster
sizes before you apply. API: `POST /api/faces/clusters/{id}/split` with
body `{eps, min_samples, dry_run}`. New cluster_ids are minted past the
current max so they never collide. Faces that fall out as DBSCAN noise
go to `cluster_id=NULL`. Implementation: `photosearch/faces.py:split_cluster`.

## Face merge suggestions (M18)

Two-step flow for merging unknown-face groups into named persons or into
other unknown clusters:

**1. Generate suggestions** (CLI, read-only):

```bash
$DC run --rm photosearch suggest-face-merges \
  --json-out /data/suggestions.json \
  [--base-url http://<nas>:8000] \
  [--centroid-cutoff 0.95] [--min-pair-cutoff 0.60] \
  [--max-members 60] [--min-group-size 1] \
  [--verify-pair 'cluster:X=person:Name' ...]
```

For every candidate pair, `photosearch/face_merge.py` computes two L2
metrics on the ArcFace 512-dim encodings: **centroid_dist** (between the
two groups' normalized mean vectors) and **min_pair_dist** (minimum over
all member-to-member pairs). Both must be below their cutoffs for a
suggestion. `--verify-pair` reports TP recall / FP avoidance against
known-ground-truth pairs (`=` for should-merge, `!=` for should-not).

**2. Review and act** on the `/merges` page (reads the JSON, applies via
`POST /api/faces/merges`):
- Each card shows both rep-face crops, labels, face counts, date ranges,
  and the three scores (min_pair, centroid, overlap).
- Accept → `POST /api/faces/merges` performs the merge transactionally.
  cluster→person sets person_id + clears cluster_id + stamps
  `match_source='merge_review'`. cluster→cluster updates cluster_id.
- Dismiss → stored in localStorage (client-side only for now; persistent
  rejection table deferred).
- Top nav link is global. Suggestions whose source cluster has been
  accepted or emptied are filtered out live in the API response.

Cluster IDs are ephemeral (every `recluster-faces` run renumbers them),
but the merge flow is the intended consumer — review within the same
recluster cycle. Face IDs (`rep_face_id`) are stable, so localStorage
dismissals survive JSON regeneration.

The /merges toolbar also has a **Regenerate** button (`POST
/api/faces/suggestions/regenerate`) that re-runs the engine with
user-selected cutoffs and writes fresh JSON — useful after splitting
attractor clusters or after a batch of accepts changes the grouping
landscape. Blocks ~15–30s on an N100.

`/api/faces/groups` is paginated (`limit`/`offset`/`filter`/`total`/`counts`),
hides singletons by default (`?include_singletons=1` to restore), and
auto-downgrades its O(N²) similarity sort to count-sort above 500 groups.
`/api/faces/crop/{id}` disk-caches at `thumbnails/face_crops/{id}_{size}.jpg`.

Remaining work in `docs/plans/faces-clustering-and-perf.md`: persist
`det_score` + bbox area and filter low-quality faces before clustering,
swap to HDBSCAN for varying density, and materialize a `face_groups` table.

## Planned milestones (see `docs/plans/`)

- `docs/plans/bulk-set-location.md` — bulk-assign location to photos lacking
  GPS. **M19 (temporal-neighbor inference) shipped** — see the
  "Inferred geotagging (M19)" section above. Remaining work from that
  plan: manual address picker (forward geocoding, Nominatim) and the
  `country`/`admin1`/`admin2`/`locality` columns that unlock map view,
  radius search, and region-scoped queries like "beach near southwest
  France".
- `docs/plans/infer-location-refinements.md` — post-M19 cascade fixes
  surfaced on the 127k NAS library. Cap hop depth (cascade ran 776
  deep, 71% of candidates below 0.25 confidence), downweight inferred
  anchors on re-scan so decay protects cross-run compounding, drop the
  dead `--max-cascade-rounds` flag. All localized to
  `photosearch/infer_location.py`.
- `docs/plans/google-photos-import.md` — M20. Takeout-based import of
  ~200K smartphone photos. Google Photos API read scopes were deprecated for
  third-party apps in March 2025, so Takeout is the only path. Incremental
  per-year export, composite dedup (photoTakenTime + device + filename stem),
  lands in `/photos/YYYY/YYYY-MM-DD_gphotos/`. Phone GPS amplifies the
  inferred-geotag recall on camera photos.
- `docs/plans/faces-clustering-and-perf.md` — remaining face work: persist
  `det_score` + bbox area, HDBSCAN, materialize a `face_groups` table.

## Detailed Reference

For full API endpoint list, indexing pass details, Google Photos integration, stacking
parameters, face management commands, and troubleshooting, see `.claude/skills/photo-search/SKILL.md`.

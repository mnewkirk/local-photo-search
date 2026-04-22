---
name: photo-search
description: >
  Expert knowledge for developing and operating the local-photo-search service — a
  self-hosted photo library with CLIP semantic search, face recognition, LLaVA scene
  descriptions, aesthetic quality scoring, Google Photos upload, and shoot review/culling,
  deployed via Docker on a UGREEN NAS.

  Use this skill whenever the user asks about: adding features to photo-search, running
  indexing jobs, troubleshooting the NAS deployment, understanding the codebase
  architecture, modifying the CLI or web API, updating the status page, managing face
  references, finding duplicate photos, working with collections, stacking burst photos,
  uploading to Google Photos, or any other task related to this project.

  Trigger even for vague requests like "how do I index 2024?" or "why is search slow?"
  or "I want to add a new filter" or "upload these to Google" — all of these relate
  to this service.
---

# Photo Search Skill

This service is a fully local, Docker-deployed photo search system. Photos are never
sent to any external API. It runs on a UGREEN NAS (Intel N100 CPU, no GPU) and is
developed on a Mac.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Database | SQLite (WAL mode) with sqlite-vec for vector search |
| Semantic search | CLIP (ViT-B/16, 512-dim) via open-clip-torch — CPU only |
| Face recognition | InsightFace (buffalo_l) — ArcFace embeddings |
| Scene descriptions | LLaVA via Ollama sidecar container |
| Aesthetic quality | CLIP ViT-L/14 + linear scorer (sac+logos+ava1) |
| Frontend | Plain React (UMD, no build step) in `frontend/dist/` |
| Containerization | Docker Compose (multi-stage build) |
| Google Photos | OAuth2 with photoslibrary.appendonly scope |

---

## Project Layout

```
local-photo-search/
├── cli.py                  # All CLI commands (Click)
├── Dockerfile              # Multi-stage: builder + runtime
├── docker-compose.yml      # Local dev
├── docker-compose.nas.yml  # NAS production
├── run-workers.sh          # Docker worker fleet launcher (Mac)
├── requirements.txt        # Python deps (CPU torch pinned)
├── references.yml          # Face reference config (person → photos)
├── GOOGLE_PHOTOS_SETUP.md  # Step-by-step Google Photos OAuth setup
├── photosearch/
│   ├── db.py               # PhotoDB class, schema v17, all queries
│   ├── infer_location.py   # M19 — temporal GPS inference (haversine, cascade)
│   ├── index.py            # index_directory() + _index_collection() — indexing pipeline
│   ├── search.py           # search_combined() + all search types
│   ├── clip_embed.py       # CLIP text/image embedding (streaming)
│   ├── faces.py            # InsightFace detection, encoding, matching,
│   │                       #   recluster + session-stacking, split_cluster
│   ├── face_merge.py       # M18 — merge-suggestion engine (load_groups,
│   │                       #   compute_suggestions, score_pair)
│   ├── quality.py          # Aesthetic scoring + concept analysis (streaming)
│   ├── describe.py         # LLaVA scene descriptions via Ollama
│   ├── stacking.py         # Burst/bracket stack detection (union-find)
│   ├── verify.py           # Hallucination detection (CLIP + cross-model LLM)
│   ├── google_photos.py    # Google Photos OAuth2, upload, album management
│   ├── web.py              # FastAPI app, 50+ /api/* endpoints
│   ├── worker.py           # Remote worker client — claims batches from NAS, processes locally
│   ├── worker_api.py       # Worker API endpoints (/api/worker/*) for distributed indexing
│   ├── exif.py             # EXIF extraction
│   ├── colors.py           # Dominant color extraction
│   ├── geocode.py          # Offline reverse geocoding
│   ├── date_parse.py       # Natural language date parsing from queries
│   └── cull.py             # Shoot review / culling logic
└── frontend/dist/          # Static HTML/JS served by FastAPI
    ├── index.html          # Main search UI
    ├── faces.html          # Face browser + Split cluster inline form
    ├── merges.html         # M18 — merge-suggestion review page
    ├── collections.html    # Collections UI + Google Photos upload modal
    ├── map.html            # Leaflet map view of GPS-bearing photos
    ├── geotag.html         # Manual bulk geotag picker (folder-first UI)
    ├── review.html         # Shoot review / culling UI
    ├── status.html         # Indexing status + run commands
    └── shared.js           # Shared components: PS.SharedHeader (with /merges
                            #   link), PS.PhotoModal, PS.GooglePhotosButton,
                            #   PS.formatFocalLength, etc.
```

---

## Database Schema (v17)

The database file is `photo_index.db` (not `photos.db`). Key tables:

| Table | Purpose |
|---|---|
| photos | Main photo records — path, date_taken, EXIF, description, tags, scores |
| faces | Detected faces per photo (bbox, encoding, quality) |
| persons | Named persons for face matching |
| face_references | Reference photos/encodings for each person |
| collections | Named photo collections/albums |
| collection_photos | Junction table with sort_order for manual ordering |
| photo_stacks | Burst/bracket groups detected by time + visual similarity |
| stack_members | Photos belonging to each stack |
| review_selections | Culling selections per folder |
| google_photos_uploads | Upload ledger (album_id, filepath, media_item_id) |
| ignored_clusters | Face clusters marked to ignore |
| schema_info | Schema version + photo_root path |

Important columns on `photos`: `date_taken` (TEXT, "YYYY-MM-DD HH:MM:SS", indexed),
`aesthetic_score` (REAL, 1-10), `description` (TEXT, LLaVA-generated), `tags` (JSON array),
`place_name` (TEXT, reverse-geocoded), `dominant_colors` (JSON array of hex values),
`location_source` (`'exif'|'inferred'|NULL`, v17), `location_confidence`
(`NULL|(0,1]`, v17, only non-null for inferred rows), `date_created` (v16
file-mtime fallback sort key).

Schema migrations run automatically on DB open via `_init_schema()` with version checks.
Bump `SCHEMA_VERSION` when adding tables or columns, and add migration SQL in the
appropriate position (after any table it depends on).

---

## Key Patterns

These patterns were established through iteration — always follow them.

### 1. Streaming generators for long-running batch jobs
Any function that processes thousands of photos must yield results incrementally.
Progress is saved to the DB per-batch (survives cancellation) and printed to the log.

```python
# The established pattern in quality.py / clip_embed.py:
def score_photos_stream(image_paths, batch_size=8):
    for batch_start in range(0, total, batch_size):
        # ... process batch ...
        for idx, result in enumerate(batch_results):
            yield batch_start + idx, result
```

### 2. PHOTOSEARCH_DB env var for --db defaults
Every CLI command's `--db` option must include `envvar="PHOTOSEARCH_DB"`.

### 3. Filename auto-detection in search
Queries that look like camera filenames (e.g., `DSC06241`) are detected by
`_looks_like_filename()` in `search.py` and routed to SQL LIKE instead of CLIP.

### 4. File exclusions in find_photos()
`index.py:find_photos()` skips macOS AppleDouble sidecars (`._*`) and numbered
copies (`DSC_1.JPG` when `DSC.JPG` exists).

### 5. Frontend — no build step
`frontend/dist/` contains plain HTML with React loaded from CDN (UMD build). Edit HTML
files directly. Use vanilla JS (`React.createElement`) not JSX. Changes require a Docker
image rebuild since files are baked in via `COPY frontend/ frontend/`.

### 6. Docker cache awareness
`requirements.txt` changes invalidate the pip install layer (~5 min on N100). Python file
changes only invalidate `COPY` (~10s). CPU-only PyTorch is pinned via
`--extra-index-url https://download.pytorch.org/whl/cpu` at the top of `requirements.txt`.

### 7. SSE for long-running API operations
Google Photos upload uses Server-Sent Events (SSE) via `StreamingResponse` with
`text/event-stream`. Cross-thread communication uses `asyncio.Queue` + `threading.Event`
for cancellation. Key gotcha: use `asyncio.get_running_loop()` (not `get_event_loop()`)
in async endpoints, and always include terminal events (`done`, `fatal`, `cancelled`)
in the generate() function to close the stream.

### 8. Per-file ledger writes for cancel safety
Operations that can be cancelled mid-way (like uploads) write per-file results to the
DB immediately rather than batching at the end. This ensures partial progress survives
cancellation.

### 9. Face IDs are stable, cluster IDs are not
`faces.id` is `INTEGER PRIMARY KEY AUTOINCREMENT` — never reused, survives every
`recluster-faces`. `cluster_id` is fully renumbered on each recluster, so anything
that needs to persist across reclusters (merge dismissals, suggestion-file
identity) keys on `rep_face_id` pairs, not cluster_ids. `/merges` localStorage
dismissals and the `resolveTarget()` rewrite logic both exploit this. Same rule
applies when designing future tables: use `face_id` as the stable anchor if you
need cross-recluster persistence.

---

## API Endpoints (50+)

### Search & Photos
- `GET /api/search` — Combined search (CLIP semantic, color, face, place, date, filename)
  - Params: q, person, color, place, limit, min_score, min_quality, sort_quality,
    tag_match, date_from, date_to, location, match_source
  - **Name extraction from `q`** — `search.py:_extract_persons_from_query`
    pulls registered person names out of the free-text query and turns
    them into AND-intersected person filters via the same `result_sets`
    logic that handles `person=`. `?q=Calvin and Ellie` matches photos
    with both people; connector tokens (`and`, `with`, `&`, `,`) are
    stripped, the residual goes to CLIP. Matching is case-insensitive,
    word-bounded, longest-first ("Matt Newkirk" beats "Matt"). Lookbehind
    `(?<!-)` keeps `-Calvin` as a CLIP exclusion rather than a filter.
- `GET /api/photos/{id}` — Photo detail
- `GET /api/photos/{id}/thumbnail` — Cached thumbnail
- `GET /api/photos/{id}/full` — Full resolution
- `GET /api/photos/{id}/preview` — Preview size

### Faces
- `GET /api/faces/groups` — All face groupings (paginated, similarity-sorted)
- `GET /api/faces/group-info?cluster_id=N | person_id=N` — Single group metadata
  (used by `/faces?cluster_id=N` auto-open; works for hidden singletons + unloaded pages)
- `GET /api/faces/group/{type}/{id}/photos` — Photos for a person or cluster
- `GET /api/faces/crop/{face_id}` — Face crop image (disk-cached)
- `GET /api/faces/face-detail/{face_id}` — Photo id + bbox + dimensions for overlays
- `POST /api/faces/{face_id}/assign` — Assign face to person
- `POST /api/faces/{face_id}/clear` — Clear assignment
- `POST /api/faces/bulk-collect` — Bulk assign unassigned faces
- `POST /api/faces/ignore` / `POST /api/faces/unignore` — Ignore/restore clusters
- `POST /api/faces/clusters/{id}/split` — Re-run DBSCAN with tighter eps on
  one cluster (body `{eps, min_samples, dry_run}`)
- `GET /api/faces/manual-assignments` / `POST /api/faces/import-assignments`
  — Export/import manual face-to-person assignments

### Merge suggestions (M18)
- `GET /api/faces/suggestions` — Read cached JSON (via `PHOTOSEARCH_SUGGESTIONS_JSON`
  or `/data/suggestions.json`), augments each side with up to 4 sample_face_ids,
  filters rows whose source cluster has been collapsed
- `POST /api/faces/suggestions/regenerate` — Re-run the engine with params
  `{centroid_cutoff, min_pair_cutoff, max_members, min_group_size,
  include_ignored}` and overwrite the cached JSON
- `POST /api/faces/merges` — Apply a merge, body `{source: {type:"cluster",id},
  target: {type:"cluster"|"person",id}}`. Returns `{moved_face_count}`. Stamps
  `faces.match_source='merge_review'`.

### Persons
- `GET /api/persons` — List persons with photo counts

### Collections
- `GET /api/collections` — List all
- `POST /api/collections` — Create
- `GET /api/collections/{id}` — Detail (includes photos with sort_order)
- `PUT /api/collections/{id}` — Rename
- `DELETE /api/collections/{id}` — Delete
- `POST /api/collections/{id}/photos` — Add photos
- `POST /api/collections/{id}/photos/remove` — Remove photos

### Stacks (Burst/Bracket Groups)
- `GET /api/stacks` — List all
- `GET /api/stacks/{id}` — Detail with members
- `PUT /api/stacks/{id}/top` — Set top photo
- `DELETE /api/stacks/{id}` — Delete stack
- `POST /api/photos/{id}/unstack` — Remove from stack
- `POST /api/stacks/{id}/add` — Add to stack
- `GET /api/photos/{id}/nearby-stacks` — Find nearby stacks
- `POST /api/stacks/detect` — Run detection synchronously. Body
  `{time_window_sec?, clip_threshold?, max_stack_span_sec?, clear?, dry_run?}`
  (all optional, defaults match the CLI: 5.0 / 0.05 / 10.0). Returns
  `{stacks_created, photos_stacked, cleared, dry_run, duration_seconds, params}`.
- `POST /api/stacks/detect/stream` — SSE variant used by the status-page
  Stacking form. Same body; streams `start` / `progress` / `done` /
  `cancelled` / `fatal` events. Client disconnect flips a `threading.Event`
  that `stacking.py` checks inside its hot loops (≤1s abort latency on 500k
  libraries). Progress events carry `phase` ∈ `{scan, load_embeddings,
  pairs, group, save, cleared}` plus phase-specific counters.

### Review (Culling)
- `GET /api/review/folders` — Available folders (returns `{path, max_date}` objects, sorted by most recent photo first)
- `GET /api/review/run` — Run culling algorithm
- `GET /api/review/load` — Load saved selections
- `POST /api/review/toggle/{id}` — Toggle photo selection

### Google Photos
- `GET /api/google/status` — OAuth status (configured + authenticated)
- `GET /api/google/authorize` — Start OAuth flow
- `POST /api/google/exchange-code` — Manual code exchange
- `GET /api/google/callback` — OAuth callback
- `DELETE /api/google/disconnect` — Revoke + clear tokens
- `POST /api/google/albums` — Create album
- `POST /api/google/upload-status` — Check which photos are already uploaded to an album
- `POST /api/google/upload` — Upload with SSE streaming progress

### Utility
- `GET /api/stats` — Database statistics for status page. Response includes
  `photos`, `clip_embedded`, `faces`, `persons`, `described`, `quality_scored`
  (+ `quality_stats`), `concepts_analyzed`, `tagged`, `stacks`, `stacked_photos`,
  `verify_passed`/`verify_failed`/`verify_regenerated`.
- `GET /api/stats/activity` — Hourly index activity, 3-day window
- `GET /api/stats/errors` — Recent indexing errors
- `GET /api/worker/status` — (see Worker System below) polled by the status
  page Workers panel every 5s for per-pass queue depth + active claims

### Map view
- `GET /api/photos/geojson` — compact point dump of every GPS-bearing
  photo. Returns `{count, points: [[id, lat, lon, source, year], ...]}`
  — tuple format keeps 50k points under 500KB gzipped. `source` is
  `'exif' | 'inferred' | None`; `year` is parsed from `date_taken[:4]`
  or None. Consumed by `/map` (Leaflet + markercluster UMD, no build
  step). **Must be declared before `/api/photos/{photo_id}` in
  `web.py`** or FastAPI's path matcher takes `geojson` as a photo id
  and returns 422.

### Manual bulk geotagging
- `GET /api/geotag/folders` — folder summary with no-GPS/inferred/exif
  counts per folder, sorted by no-GPS descending. Hides fully-tagged
  folders unless `?include_fully_tagged=true`. Folder path is derived
  from `os.path.dirname`-style rfind of `/` in `filepath`.
- `GET /api/geotag/folder-photos?folder=...&show_inferred=false` —
  photos in one folder. By default returns `gps_lat IS NULL` rows only;
  with `show_inferred=true` also returns `location_source='inferred'`
  rows so users can correct M19 misfires. Never returns `exif` rows.
- `GET /api/geotag/known-places?q=...` — distinct `place_name` values
  from the library matching q (case-insensitive substring), sorted by
  photo count descending. Feeds the "library" section of the /geotag
  typeahead.
- `GET /api/geocode/search?q=...&limit=5` — forward-geocode via
  Nominatim (public OSM endpoint) with a persistent 30-day cache in
  `geocode_cache`. User-Agent: `"local-photo-search/1.0 (…)"`.
  Response shape `{results: [{display_name, lat, lon, country,
  admin1, admin2, locality, type, importance}], source: 'cache'|
  'nominatim'}`.
- `POST /api/photos/bulk-set-location` — body `{photo_ids, lat, lon,
  place_name, overwrite?}`. Writes `gps_lat/gps_lon/place_name/
  location_source='manual'/location_confidence=NULL` in one transaction.
  `overwrite=false` (default) keeps existing GPS; `overwrite=true`
  replaces any source including exif. Returns
  `{updated_count, skipped_count}`.

### Inferred geotagging (M19)
- `POST /api/geocode/infer-preview` — read-only. Body takes
  `{window_minutes, max_drift_km, min_confidence, cascade, max_cascade_rounds}`
  (all optional, defaults match the CLI: 30 / 25.0 / 0.0 / true / 10).
  Returns candidate counts, `confidence_buckets` (fixed 5-entry list),
  `hop_distribution` (sorted), `skipped` reasons, and up to 10 sample
  candidates with `thumbnail_url` + pre-reverse-geocoded `place_name`.
- `POST /api/geocode/infer-apply` — write path. Same body plus
  `confirm: true` (400 without it). Reverse-geocodes the full candidate
  set and writes in one transaction with
  `WHERE id=? AND gps_lat IS NULL` so rows a concurrent indexer just
  populated are never overwritten. Returns
  `{updated_count, rounds_used, duration_seconds}`.

---

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

## Google Photos Integration

### Scope & Limitations
- Only `photoslibrary.appendonly` scope is available (read/sharing deprecated March 2025)
- This is write-only: cannot list album contents, cannot read media items
- `batchAddMediaItems` returns 400 "invalid media item id" for photos removed from albums
  via Google Photos UI — the media_item_id becomes invalid for album operations
- Only option for re-adding removed photos: full re-upload of bytes
- Token stored in `google_photos_token.json` (not in the database)
- Client credentials in `client_secret.json` alongside DB

### Upload Flow
1. Raw bytes POSTed to Google → returns uploadToken
2. `batchCreate` with uploadTokens (batch size 50) → creates mediaItems with album assignment
3. Per-file SSE events: `start` → `begin` → `bytes_sent` → `progress` → `done`
4. Upload ledger tracks (album_id, filepath, media_item_id) per file
5. Selective re-upload: `force_reupload_ids` param targets specific photos

### Album ID Note
The API album ID (from `albums.create`) differs from the ID visible in Google Photos URLs.
Always use the API ID stored in the database.

---

## Stacking System

Burst/bracket detection using union-find over time-sorted photos:
- Two photos linked if taken within `time_window_sec` (default 5.0) AND
  CLIP cosine distance < `clip_threshold` (default 0.05)
- Span enforcement: max `max_stack_span_sec` (default 10.0) from earliest
  to latest member
- Top photo selected by highest aesthetic score
- Full CRUD API for manual stack management

Three invocation paths:
1. **CLI** — `photosearch stack` (see command table below)
2. **Blocking API** — `POST /api/stacks/detect`
3. **SSE API** — `POST /api/stacks/detect/stream` (used by the status-page
   Stacking form). Streams `progress` events with `phase` labels
   (`scan`, `load_embeddings`, `pairs`, `group`, `save`) throttled to
   ~0.25s, terminates with `done` / `cancelled` / `fatal`.

`stacking.py:run_stacking`, `detect_stacks`, `save_stacks`, and
`_load_embeddings_bulk` all accept `on_progress: Callable[[dict], None]`
and `should_abort: Callable[[], bool]`. Abort is checked at phase
transitions and every ~1000 photos inside the pairs loop, so a `cancel_event`
set by `request.is_disconnected()` stops the run within ~1s even on a
500k-photo library. The pair of hooks is the reference shape for adding
SSE progress + cancel to other long-running jobs.

---

## Shoot Review / Culling

Adaptive clustering of CLIP embeddings to select representative photos:
- Target ~10% selection from a shoot folder
- Agglomerative clustering with quality-weighted representative selection
- "Represent-all-then-trim" strategy ensures tag diversity
- Review selections persisted per folder in `review_selections` table

---

## Shared Frontend Components (shared.js)

- `PS.SharedHeader` — Consistent nav header across all pages. activePage values:
  `'search' | 'review' | 'faces' | 'merges' | 'collections' | 'status'`.
- `PS.PhotoModal` — Unified photo detail modal with configurable features:
  showFaces, showCollections, showLocation, showSearchScore, showAesthetics.
  Includes face editing, collection management, stacking UI, keyboard navigation
  (arrow keys), and mobile swipe navigation (swipe left/right on touch devices).
  Slots: fetchDetail, headerChildren, footerChildren.
- `PS.GooglePhotosButton` — Upload single photo to Google Photos from modal sidebar
- `PS.formatFocalLength()` / `PS.formatFNumber()` — EXIF display helpers

### /status page

The status page is both a dashboard and a control panel:

- **Stat grid** — one card per pass (CLIP, faces, descriptions, quality,
  concepts, **tags**, stacks, verify). Each card reads its count from
  `/api/stats`; the Tags card uses the `tagged` field
  (`COUNT(*) WHERE tags IS NOT NULL AND tags != '[]'`).
- **Workers panel** — fetched from `/api/worker/status` on mount and every
  5s after. Renders total active workers, total queued photos across all
  passes, a row of per-pass queue-depth pills, and one row per active claim
  with pass type, worker_id, photo count, and a live TTL (yellow under
  2 min, red once expired). A separate 1s tick re-renders TTLs between
  poll cycles. Main `/api/stats` is only re-fetched on the Refresh
  button, since its COUNTs are expensive.
- **Stacking form** — editable `time_window_sec` / `clip_threshold` /
  `max_stack_span_sec` fields (pre-filled with CLI defaults 5.0 / 0.05 /
  10.0, with a Reset button), plus "Clear existing stacks first" (with
  a confirm prompt) and "Dry run" toggles. Submit calls
  `POST /api/stacks/detect/stream`, reads the SSE via
  `response.body.getReader()` and `\n\n`-split chunks (EventSource can't
  POST), renders a phase label + progress bar + pair count + elapsed
  seconds while running, and swaps in a Cancel button that calls
  `AbortController.abort()`. On `done`, calls the parent's `load()` so
  the Stacks stat card refreshes immediately.
- **Activity chart** — hourly buckets for the last 72h, color-coded by
  pass type, includes a "cleared (--force)" series.
- **Verify failures / Errors log** — unchanged from M12 shipping.

The Workers panel + Stacking form establish the pattern for other
"run this job from the UI" actions: fetch-based SSE, throttled progress
emission inside `on_progress` callbacks, `should_abort` pair for cancel,
keepalive every 0.5s on the server side.

### /faces URL params (deep-linkable)

- `?name=<Person>` — open named-person detail
- `?person_id=N` — open person by id (works for hidden/unloaded pages)
- `?cluster_id=N` — open unknown cluster by id (works for hidden singletons)
- `?face_id=N` — open the group containing this rep face
- `?filter=named|unknown|ignored|all` — pre-filter the grid

`cluster_id` / `person_id` fall back to `/api/faces/group-info` when the group
isn't already in the loaded page, so the suggest-face-merges CLI emits these
with `--base-url` and they always resolve.

### /merges page architecture

- Reads cached suggestions from `/api/faces/suggestions` (not computed on page
  load — the engine takes ~20s)
- Per-card face strip of up to 4 crops (via `sample_face_ids` that the API
  augments onto the JSON — no per-card fetch)
- Click any crop → `/api/faces/face-detail/{face_id}` → photo-preview overlay
  with the bbox highlighted on the full photo
- Confidence tier (Strong/Probable/Borderline) calibrated from the observed
  FP pattern: `min_pair < 0.50` strong, `< 0.55` probable, `≥ 0.55` borderline
- Risk badges: "attractor risk" (big target cluster), "long overlap"
  (cluster↔cluster spanning >180 days), "edge score" (min_pair ≥ 0.55),
  "rewritten after merge" (see below)
- Keyboard: `J/↓` next, `K/↑` prev, `A/Enter` accept, `D` dismiss, `Esc` blur.
  Accept does NOT advance focus — the accepted card is filtered out, so the
  next row slides into the same slot naturally
- **Merge chain rewriting**: when you accept `A → B`, any pending `C → A`
  becomes `C → B` (using stored target info from the accepted merge). The
  original min_pair is still a valid lower bound because the faces that
  matched C are now in B. `resolveTarget()` handles chains (`A → B`, then
  `B → Person`) with cycle protection. Source-emptied suggestions
  (`A → X` where A was merged away) are tagged stale, not rewritten.
- Dismissals persist in `localStorage` keyed by the sorted rep_face_id pair
  (stable across reclusters). Accepts persist in the DB.

---

## NAS Operations

### Docker commands
All commands use: `docker compose -f docker-compose.nas.yml`

```bash
# Rebuild after code changes
docker compose -f docker-compose.nas.yml build photosearch

# Restart web server
docker compose -f docker-compose.nas.yml up -d photosearch

# Run a CLI command
docker compose -f docker-compose.nas.yml run --rm photosearch <command>

# Ad-hoc Python or shell inside the container — MUST override the entrypoint.
# docker-entrypoint.sh has a `case` that routes every non-"serve"/"index" first
# arg to `python cli.py <arg> …`. So `… photosearch python -c "…"` becomes
# `python cli.py python -c "…"` and cli.py rejects it ("No such command 'python'").
#
# `--entrypoint` is a `docker compose run` flag, so it MUST come *before* the
# service name (photosearch). It cannot be appended after $DC. Use dedicated
# aliases so the copy-paste works:
DCPY="docker compose -f docker-compose.nas.yml run --rm --entrypoint python"
DCSH="docker compose -f docker-compose.nas.yml run --rm --entrypoint bash"
$DCPY photosearch -c "<snippet>"
$DCSH photosearch -c "<shell cmd>"

# IMPORTANT: PhotoDB() with no args defaults to the RELATIVE path "photo_index.db",
# which resolves to /app/photo_index.db (an empty stub) inside the container. The
# PHOTOSEARCH_DB env var is wired only into cli.py's --db defaults, NOT into
# PhotoDB.__init__. Every ad-hoc snippet must pass the env var explicitly, or
# you'll see all-zero stats and leave a bogus /app/photo_index.db behind:
$DCPY photosearch -c "
import os
from photosearch.db import PhotoDB
with PhotoDB(os.environ['PHOTOSEARCH_DB']) as db:
    ...
"

# Background indexing job
nohup docker compose -f docker-compose.nas.yml run --rm \
  -e PYTHONUNBUFFERED=1 photosearch index /photos/YEAR --clip --no-colors \
  > /tmp/clip_YEAR.log 2>&1 &

# Git pull (Alpine workaround for UGOS ownership issue)
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo pull"
```

### Volume layout
```
/data/                      # Persistent Docker volume
  photo_index.db            # SQLite database
  thumbnails/               # Generated JPEG thumbnails
  google_photos_token.json  # Google OAuth token
  .insightface/             # InsightFace model cache (~300MB)
  .cache/photosearch/       # CLIP + aesthetic model cache
/photos/                    # Photo library (read-only mount)
  2026/2026-01-15/DSC*.JPG  # Year/date-named folders
/references/                # Face reference photos
  references.yml            # Person → photo mapping
```

### Full indexing sequence for a new year
```bash
DC="docker compose -f docker-compose.nas.yml run --rm"
NOHUP="nohup docker compose -f docker-compose.nas.yml run --rm -e PYTHONUNBUFFERED=1"

$NOHUP photosearch index /photos/YEAR --clip --no-colors > /tmp/clip_YEAR.log 2>&1 &
$NOHUP photosearch index /photos/YEAR --faces --no-colors > /tmp/faces_YEAR.log 2>&1 &
$NOHUP photosearch index /photos/YEAR --quality --no-colors > /tmp/quality_YEAR.log 2>&1 &

$DC -v /home/cantimatt/docker/photosearch/references:/references:ro \
  photosearch add-persons --config /references/references.yml
$DC photosearch match-faces --temporal
$DC photosearch recluster-faces          # group remaining unknowns via DBSCAN
$DC photosearch stack --directory /photos/YEAR
```

New faces land with `cluster_id = NULL` and are invisible on `/faces` until
`recluster-faces` runs — it's the only thing that forms "Unknown #N" groups.
Run it after each face-indexing pass (or batch thereof). Warning: every run
renumbers every unknown cluster_id and clears `ignored_clusters`, so any
"ignore" decisions on unknown clusters need to be reapplied afterward.

---

## Indexing Types — Detailed Reference

The `index` command has two modes: **directory mode** (scan a folder for new photos and
index them) and **collection mode** (re-index existing photos in a collection). Collection
mode skips EXIF extraction/insertion since photos are already in the DB, making it ideal
for re-running specific passes (e.g. testing a different LLM model for descriptions).

```
# Directory mode — scan and index new photos
photosearch index <photo_dir> [OPTIONS]

# Collection mode — re-index existing photos by collection ID
photosearch index --collection <ID> [OPTIONS]

Common options:
  --clip / --force-clip        CLIP semantic embeddings (ViT-B/16, 512-dim)
  --faces / --force-faces      Face detection + ArcFace encoding (InsightFace buffalo_l)
  --quality / --force-quality  Aesthetic scoring (ViT-L/14 + MLP) + concept analysis
  --describe / --force-describe  LLaVA scene descriptions (requires Ollama)
  --describe-model MODEL       Ollama model name (default: llava; alt: moondream)
  --tags / --force-tags        Semantic tags from fixed ~60-tag vocabulary (requires Ollama)
  --no-colors                  Disable dominant color extraction (on by default)
  --full                       Enable all: --clip --faces --describe --quality --tags
  --verify                     Run hallucination verification after other passes
  --batch-size N               Batch size for CLIP/quality (default: 8)
  --db PATH                    Database path (default: PHOTOSEARCH_DB env var)

Collection-only options:
  --collection ID              Re-index photos in this collection (replaces PHOTO_DIR)
  --expand-stacks              Also include other photos from the same stacks
```

### Pass details

| Pass | Flag | Model | Speed on N100 | Dependencies |
|------|------|-------|---------------|-------------|
| EXIF + hash | (always) | — | Fast, ~1000/min | None |
| CLIP embeddings | `--clip` | ViT-B/16 (512-dim, ~330 MB) | ~1000 photos/hr | None |
| Dominant colors | (default on) | ColorThief | Fast, ~1000/min | None |
| Face detection | `--faces` | InsightFace buffalo_l (~300 MB) | ~0.5-2s/photo | None |
| Quality scoring | `--quality` | ViT-L/14 (768-dim) + MLP | ~1000 photos/hr | None |
| Concept analysis | (auto with quality) | Same ViT-L/14 | Runs after scoring | Quality pass |
| Descriptions | `--describe` | LLaVA 7B via Ollama | 30-200s/photo | Ollama running |
| Tags | `--tags` | Same as describe model | 30-200s/photo | Ollama running |
| Critique | (auto) | Same as describe model | 30-200s/photo | Quality + describe |
| Stacking | (auto after CLIP) | — | Fast (DB only) | CLIP embeddings |
| Geocoding | (auto) | GeoNames (offline) | Fast | GPS data in EXIF |
| Verification | `--verify` | minicpm-v + llava | Slow (2 LLM passes) | Descriptions exist |

### Parallelization

These can run as separate `nohup` processes simultaneously (different models, no conflicts):
- `--clip`, `--faces`, `--quality`

These must run sequentially (share Ollama's single model slot):
- `--describe`, `--tags`, critiques

### Collection-based re-indexing

All indexing commands support `--collection <ID>` to scope work to a specific collection
instead of an entire directory. This is useful for A/B testing models, re-running a pass
on a curated set, or iterating on a small batch without re-processing thousands of photos.

```bash
# Re-describe collection 2 with a different model
photosearch index --collection 2 --describe --force-describe --describe-model moondream

# Re-score quality for collection 2, including burst stack members
photosearch index --collection 2 --quality --force-quality --expand-stacks

# Match faces only within a collection
photosearch match-faces --collection 2 --temporal

# Detect stacks only among a collection's photos
photosearch stack --collection 2
```

Collection mode in `index` skips EXIF extraction and photo insertion (photos are already
in the DB) and runs all selected passes over the collection's photos. The `--expand-stacks`
flag expands the photo set to include other members of the same stacks.

**Implementation:** `index_directory()` dispatches to `_index_collection()` when
`collection_id` is set. The collection path uses `PhotoDB.get_collection_photo_pairs()`
to get `(photo_id, abs_path)` tuples. `match_faces_to_persons()`, `match_faces_temporal()`,
`detect_stacks()`, and `run_stacking()` all accept an optional `photo_ids` parameter for
scoping. `PhotoDB.expand_to_stacks()` expands photo IDs to include stack members.

### Key model details

- **Search CLIP (ViT-B/16):** 512-dim embeddings, different from quality CLIP. Force-regen
  with `--force-clip` when switching models to avoid stale embeddings.
- **Quality CLIP (ViT-L/14):** 768-dim embeddings fed to linear MLP scorer. Score range
  in practice: 3.68–5.99 (mean 4.81). Both models unloaded after each pass to free RAM.
- **InsightFace:** Downloads buffalo_l to `INSIGHTFACE_HOME` (~300 MB). Images >3500px
  long-edge are downsampled before detection. 512-dim ArcFace embeddings.
- **LLaVA/moondream:** Images resized to 1024px max before sending to Ollama. moondream is
  5-10x faster but significantly worse quality — hallucinates frequently and ignores
  structured prompts (especially critique prompts). Use llava for anything quality-sensitive.

### Stacking parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `time_window_sec` | 5.0 | Max seconds between consecutive burst shots |
| `clip_threshold` | 0.05 | Max CLIP cosine distance (very tight = same scene) |
| `max_stack_span_sec` | 10.0 | Hard cap on total stack duration |

The `stack` CLI command also supports `--collection ID` and `--expand-stacks` to scope
detection to a specific collection's photos.

**Progress + cancel hooks** — `detect_stacks`, `_load_embeddings_bulk`,
`save_stacks`, and `run_stacking` all take
`on_progress: Callable[[dict], None] | None` and
`should_abort: Callable[[], bool] | None`. Progress events are dicts with
a `phase` key (`scan` / `load_embeddings` / `pairs` / `group` / `save`)
plus phase-specific counters (`loaded`/`processed`/`saved` vs `total`,
`pairs_found`, `stacks_detected`). Emission inside hot loops is throttled
to `_PROGRESS_INTERVAL = 0.25s`; phase transitions bypass the throttle.
`should_abort()` is checked every ~1000 iterations and at every phase
transition; returning `True` causes `InterruptedError` to propagate out.
The SSE endpoint `/api/stacks/detect/stream` wires these up to an
`asyncio.Queue` + `threading.Event` — the exact pattern to copy when
adding cancel + progress to any other long-running job (recluster,
describe, quality, verify).

### Face management commands (post-indexing)

```bash
add-persons --config <yaml>                 # Batch register from YAML
match-faces [--tolerance 1.15] [--temporal] # Match faces to persons
  --collection ID                           # Scope to collection photos only
  --expand-stacks                           # Include stack members with --collection
  --temporal-tolerance 1.45                 # Looser threshold for temporal
  --temporal-window 30                      # Session context window (minutes)
recluster-faces [--eps 0.55] [--min-samples 3] \
                [--no-session-stacking] \
                [--session-eps 0.50] [--session-window 60] \
                [--dry-run]
                                            # Global DBSCAN over all person_id IS NULL
                                            # encodings, then (default-on) session-
                                            # stacking second pass that unions DBSCAN
                                            # noise pairs within (session_eps L2 +
                                            # session_window minutes). Renumbers every
                                            # unknown cluster and clears
                                            # ignored_clusters atomically.
split-cluster CLUSTER_ID [--eps 0.45] [--min-samples 2] [--dry-run]
                                            # Re-run DBSCAN on one cluster with
                                            # tighter params to split "attractor"
                                            # clusters. New cluster_ids minted
                                            # past the current max; noise → NULL.
                                            # Also available from the /faces
                                            # detail sidebar's Split button.
suggest-face-merges [--centroid-cutoff 0.95] [--min-pair-cutoff 0.60] \
                    [--max-members 60] [--min-group-size 1] \
                    [--include-ignored] [--limit N | --all] \
                    [--json-out FILE] \
                    [--verify-pair 'cluster:X=person:Name' ...]
                                            # Read-only dry-run. Surfaces likely merges
                                            # between every pair of groups (named +
                                            # unknown). Uses centroid + min-pair L2
                                            # distances. --verify-pair reports
                                            # TP/FP coverage against known examples.
list-persons                                # Show persons and counts
face-clusters                               # Show unidentified clusters
correct-face <filename> <face_num> <name>   # Manual correction
clear-matches <dir> [--person] [--all-faces]
export-face-assignments / import-face-assignments
```

### Unknown-face clustering & merge suggestions (M18)

Four cooperating tools improve unknown-face grouping:

1. **Session stacking** in `recluster-faces` — after global DBSCAN, a second
   pass runs union-find over the noise points: pairs whose L2 distance is
   within `--session-eps` (default 0.50) and whose `date_taken` is within
   `--session-window` minutes (default 60) get linked. Components of size ≥
   2 become new clusters continuing past the DBSCAN id range. Recovers
   same-person-same-event groups that `min_samples=3` had thrown away.
   Pass `--no-session-stacking` to restore DBSCAN-only behavior.

2. **`suggest-face-merges`** — read-only CLI that finds likely merges
   between any two face groups (cluster↔cluster and cluster↔named). For
   each candidate pair it computes `centroid_dist` (between the two groups'
   normalized mean encodings) and `min_pair_dist` (min across all
   member-to-member pairs). Suggests when both are under their cutoffs.
   `--verify-pair` takes known TP (`=`) and FP (`!=`) examples and prints
   whether each would be caught — use it to tune thresholds against the
   real library. `--base-url` emits clickable links next to each suggestion.

3. **`/merges` review page** (see the "/merges page architecture" section
   above) — renders the JSON output from suggest-face-merges and lets the
   user accept/dismiss with keyboard shortcuts. Can regenerate the JSON
   directly from the UI via `POST /api/faces/suggestions/regenerate`.

4. **`split-cluster`** (CLI + API + /faces button) — re-runs DBSCAN on a
   single cluster with tighter eps to break apart "attractor" clusters that
   lumped multiple people together during the eps=0.55 global recluster.
   New cluster_ids are minted past the current max so they never collide.
   Dry-run previews show the histogram of resulting sub-cluster sizes.

Implementation: `photosearch/face_merge.py` (`load_groups`,
`compute_suggestions`, `score_pair`) for suggestions;
`photosearch/faces.py:split_cluster` for splitting. Suggestion per-pair cost
is O(K²) where K is `--max-members` (default 60, biggest-bbox faces sampled
first). named↔named pairs are never suggested.

### Merge review page (`/merges`) — summary

After generating suggestions (CLI `--json-out /data/suggestions.json` or the
in-UI Regenerate button), the `/merges` page renders each suggestion
side-by-side with face crops, labels, face counts, and scores. Full
architecture is documented in the "Shared Frontend Components" section
above. Key actions:

- **Accept** → `POST /api/faces/merges` with `{source, target}`. For
  cluster→person: updates `faces.person_id`, clears `cluster_id`, stamps
  `match_source='merge_review'`. For cluster→cluster: updates
  `faces.cluster_id` only. Chain-rewrites downstream suggestions so
  `C → A` becomes `C → B` when `A → B` was just accepted.
- **Dismiss** → localStorage-only (keyed by the sorted `rep_face_id` pair,
  which is stable across reclusters). Persistent rejection table is
  future work.
- **Regenerate** → `POST /api/faces/suggestions/regenerate` with
  cutoff params; blocks ~15–30s on an N100 while the engine reloads groups
  and rescores.
- Each card links to `/faces?cluster_id=N` / `?person_id=N` for deep
  verification before accepting.

API:
- `GET /api/faces/suggestions` — reads `PHOTOSEARCH_SUGGESTIONS_JSON`
  (default `/data/suggestions.json`). Filters out rows whose source
  cluster has been collapsed since the JSON was written.
- `POST /api/faces/merges` — body `{"source": {type, id}, "target":
  {type, id}}`. Source must be `cluster`; target can be `person` or
  `cluster`. Returns `{ok, moved_face_count, source, target}`.

Suggestions file is on the Docker volume at `/data/suggestions.json` by
default — re-run the CLI whenever you want fresh candidates.

### Vec0 orphan cleanup

`clip_embeddings` and `face_encodings` are sqlite-vec `vec0` virtual tables. SQLite virtual
tables cannot participate in foreign-key constraints, so `ON DELETE CASCADE` on the parent
`photos` / `faces` tables does not reach them — every historical photo or face deletion
left a dangling vector row behind. Symptoms: status page shows `>100% embedded`, and CLIP
worker submits occasionally fail with `UNIQUE constraint failed on clip_embeddings primary key`
(since vec0 does not honor `INSERT OR REPLACE` — the PK conflict is raised first).

`AUTOINCREMENT` on `photos.id` and `faces.id` guarantees the orphan IDs can never be reissued
to different rows, so deletion is always safe.

```bash
photosearch cleanup-orphans [--dry-run]
```

Also: `add_clip_embedding` now uses explicit `DELETE` + `INSERT` instead of `INSERT OR REPLACE`,
making re-submits idempotent when a worker's claim TTL expires and the photo is re-claimed by
another worker before the original submit lands.

### Frontend stack filtering behavior

- **Search page:** Filename searches bypass stack filtering — matched photos always visible
  regardless of stack position. Other searches respect stack collapse/expand.
- **Collection page:** All collection photos always visible (no stack filtering for
  explicitly-added photos). Expand Stacks shows additional stack members from the API.
- **Result counts:** Show visible count, with "(N with stacks)" if stacking hides any.
- **Review page:** Folder picker is searchable with partial path matching
  ("2026-04" matches "2026/2026-04-06"), sorted by most recent photos first.
- **Static assets:** Served with `Cache-Control: no-cache` to prevent stale JS after deploys.

---

## Distributed Indexing (Worker System)

The worker system lets you offload heavy indexing passes (CLIP, faces, quality,
describe, tags, verify) to a fast laptop/desktop while the NAS remains the single
source of truth (DB + photos).

### Architecture

- **NAS** runs the web server with `/api/worker/*` endpoints (`worker_api.py`)
- **Worker** (laptop) runs `python cli.py worker`, claims batches via HTTP,
  downloads photo bytes, processes locally, and POSTs results back
- Claims have a TTL (default 30 min) — if a worker dies, photos are auto-reclaimed
- Multiple workers can run concurrently on different passes

### Docker Worker Fleet (Recommended for Mac)

Running workers bare-metal on macOS causes PyTorch's MPS (Metal) allocator to leak
memory over time, eventually crashing the machine. The Docker fleet avoids this by
using CPU-only PyTorch with hard memory limits per container.

```bash
# Launch 4 workers for CLIP embeddings:
./run-workers.sh -s http://<NAS-IP>:8000 -p clip -d /photos/2026 -n 4

# Multiple passes:
./run-workers.sh -s http://<NAS-IP>:8000 -p clip,quality,faces -d /photos/2026

# Fewer workers for heavier passes:
./run-workers.sh -s http://<NAS-IP>:8000 -p describe --collection 3 -n 2

# More memory per worker if needed:
./run-workers.sh -s http://<NAS-IP>:8000 -p quality -d /photos/2026 -m 4g -n 3

# Monitor:
./run-workers.sh --status    # containers + memory usage + recent progress
./run-workers.sh --logs      # tail all worker logs live (Ctrl-C to stop)
./run-workers.sh --stop      # stop all workers
```

**Important:** Use the NAS IP address (not hostname) — Docker containers run in an
isolated network and cannot resolve local DNS names like `nas.local` or mDNS hostnames.

Key options: `-n` (number of workers, default 4), `-m` (memory limit, default 3g),
`--batch-size`, `--model-batch-size`, `--ttl`, `--force`, `--describe-model`, `--verify-model`.

CPU-only inference is ~2-3x slower per worker than MPS, but 4 containers running
concurrently with stable memory is faster than 1-2 MPS workers that eventually crash.

#### Ollama for describe/tags/verify passes

If `--passes` includes `describe`, `tags`, or `verify`, the script checks
`localhost:11434` and either reuses an existing Ollama or starts a managed
container. It then pre-pulls the required models into the Ollama volume:

- `describe` / `tags` → pulls `${DESCRIBE_MODEL:-llava}`
- `verify` → pulls **both** `${VERIFY_MODEL:-minicpm-v}` (verifier) **and**
  `${DESCRIBE_MODEL:-llava}` (regeneration model used when a description fails
  verification). Ollama does not auto-pull on request, so both must be present
  before the pass runs.

**Prefer native Ollama.** Running `ollama serve` directly on the Mac host avoids
Docker Desktop VM memory oversubscription. When native Ollama is reachable at
`localhost:11434`, `run-workers.sh` detects and reuses it (and only *warns*
about missing models rather than pulling them, since pulling multi-GB models
into someone else's Ollama unannounced is rude).

**If you must use the managed Ollama container:** raise Docker Desktop memory
(Settings → Resources → Memory) to at least ~24 GiB for the default fleet
(4 workers × 3 GiB = 12 GiB + Ollama ~4-5 GiB + daemon overhead). ~16 GiB has
been observed to still OOM-kill the llama runner. Restart Docker Desktop fully
after changing the limit.

**Diagnostic hint:** `photosearch/describe.py` detects the
`"llama runner process has terminated"` error pattern and prints a one-time
hint to stderr explaining the likely cause and fixes (see
`_maybe_print_runner_oom_hint()`).

### Bare-Metal Worker CLI (Single Quick Tests Only)

```bash
# Run CLIP embeddings for a directory:
python cli.py worker -s http://nas.local:8000 -p clip -d /photos/2026/2026-04-09

# Run full pipeline on a directory:
python cli.py worker -s http://nas.local:8000 -p clip,faces,quality,describe,tags,verify -d /photos/2026

# Run CLIP + quality for a collection:
python cli.py worker -s http://nas.local:8000 -p clip,quality --collection 3

# Run descriptions with moondream model:
python cli.py worker -s http://nas.local:8000 -p describe --describe-model moondream

# Quick test — one batch only:
python cli.py worker -s http://localhost:8000 -p clip -d /photos/2026/2026-04-09 --one-shot

# Force re-process (clears existing data first):
python cli.py worker -s http://nas.local:8000 -p clip --collection 3 --force
```

Key options: `--batch-size` (photos per claim, default 16), `--model-batch-size`
(inference batch, default 8), `--ttl` (claim TTL minutes, default 30), `--one-shot`
(single batch then exit), `--force` (clear + reprocess, requires --collection or --directory).

### Worker API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/worker/claim-batch` | Claim N unprocessed photos for a pass type |
| `POST /api/worker/submit-results` | Submit processing results for a claimed batch |
| `POST /api/worker/clear-pass` | Clear processing state to allow re-processing |
| `GET /api/worker/photo-detail/{id}` | Get photo metadata + CLIP embedding (for verify) |
| `GET /api/worker/status` | Queue depth and active claims |

### Worker Loop

1. Claims a batch of unprocessed photos from the server
2. Downloads photo bytes to a temp directory
3. Runs the specified indexing pass locally (fast GPU/CPU)
4. POSTs results back to the server
5. Cleans up temp files, repeats until queue is empty

Network retries are built in (exponential backoff on transient errors like
sleep/wake disconnections). If submit fails after retries, photos are released
after TTL expires and will be reclaimed by the next worker.

### Key Files

- `run-workers.sh` — Docker fleet launcher script (start/status/logs/stop)
- `photosearch/worker.py` — Client-side worker loop + per-pass processing functions
- `photosearch/worker_api.py` — Server-side FastAPI router (`/api/worker/*`)
- `cli.py` — `worker` command with all CLI options

### DB Tables (Worker)

| Table | Purpose |
|-------|---------|
| `worker_claims` | Active batch claims (batch_id, worker_id, photo_ids, expires_at) |
| `worker_processed` | Per-photo processing ledger (photo_id, pass_type) for faces/describe/tags |

---

## Troubleshooting

**"Error: No such command 'python'" / "'sh' / '-c'" when running ad-hoc commands** —
`docker-entrypoint.sh` routes every non-"serve"/"index" first arg to `python cli.py <arg> …`,
so `docker compose … run --rm photosearch python -c "…"` becomes `python cli.py python -c "…"`
and Click rejects it. Override the entrypoint — but note `--entrypoint` is a `docker compose
run` flag, so it must come *before* the service name (it can't be tacked onto an existing
`$DC="docker compose … run --rm photosearch"` alias). Either re-expand the full command
or use dedicated aliases:
```bash
docker compose -f docker-compose.nas.yml run --rm --entrypoint python photosearch -c "<snippet>"
docker compose -f docker-compose.nas.yml run --rm --entrypoint bash   photosearch -c "<shell cmd>"

# Or as aliases:
DCPY="docker compose -f docker-compose.nas.yml run --rm --entrypoint python"
DCSH="docker compose -f docker-compose.nas.yml run --rm --entrypoint bash"
$DCPY photosearch -c "<snippet>"
```
Same pattern for sqlite3, pip, etc. — anything that isn't a `cli.py` subcommand needs
`--entrypoint`. This trips up diagnostics constantly; reach for it first when an in-container
one-liner fails.

**Empty / all-zero results from `PhotoDB()` ad-hoc snippets** — `PhotoDB()` with no args
defaults to the relative path `"photo_index.db"`, which inside the container resolves to
`/app/photo_index.db` (a fresh empty DB), not the real `/data/photo_index.db`. The
`PHOTOSEARCH_DB` env var is only honored by `cli.py`'s `--db` defaults, not by the `PhotoDB`
constructor. Always pass it explicitly:
```python
import os
from photosearch.db import PhotoDB
with PhotoDB(os.environ['PHOTOSEARCH_DB']) as db:
    ...
```
If you hit this, also `rm -f /app/photo_index.db` afterwards — the empty stub gets
persisted into the container's writable layer on first access.

**"database is locked"** — Another process holds a write lock. Check `docker ps` for
concurrent jobs. WAL mode + `PRAGMA busy_timeout=60000` resolves short locks automatically.

**"Database not found: photo_index.db"** — `PHOTOSEARCH_DB` env var not set. Confirm
Dockerfile has `ENV PHOTOSEARCH_DB=/data/photo_index.db`.

**InsightFace downloading every run** — `/data/.insightface/` not persisted. Confirm Docker
volume mount and `ENV INSIGHTFACE_HOME=/data/.insightface`.

**No SSE progress reaching browser** — Three common causes: (1) using `asyncio.get_event_loop()`
instead of `asyncio.get_running_loop()` in async endpoints, (2) missing terminal event
type in the generate() stream check, (3) `InterruptedError` swallowed by generic
`except Exception` instead of being re-raised.

**Worker containers exit immediately with "Connection refused"** — Docker containers
can't resolve local hostnames (mDNS, `.local`, NAS hostnames). Use the NAS IP address
instead: `./run-workers.sh -s http://192.168.x.x:8000 ...` Find it with `ping nas-hostname`.

**Worker memory leak / Mac crash** — PyTorch MPS allocator leaks memory in long-running
workers. Use the Docker fleet (`./run-workers.sh`) instead of bare `python cli.py worker`
for sustained multi-worker runs. Docker forces CPU-only with hard memory limits.

**"llama runner process has terminated: %!w(<nil>)" (status 500)** — The llama
runner subprocess inside Ollama was OOM-killed, almost always because the managed
Docker Ollama container is sharing Docker Desktop's VM memory with worker containers
and LLaVA's ~4.3 GiB working set doesn't fit. Fixes (easiest first):
1. Use native `ollama serve` on the host. Stop the fleet (`./run-workers.sh --stop`),
   run `ollama serve` (or launch Ollama.app), `ollama pull llava && ollama pull minicpm-v`,
   then relaunch the fleet — it will detect and reuse the native Ollama.
2. Raise Docker Desktop memory to ~24 GiB (Settings → Resources → Memory) and
   restart Docker Desktop fully. ~16 GiB is too tight for the default 4×3g fleet
   plus Ollama.
3. Reduce fleet pressure: `./run-workers.sh -n 2 -m 2g ...`.

`photosearch/describe.py` prints a one-time diagnostic hint when this error pattern
is detected, so you'll see this guidance in the worker logs as well.

**Google Photos upload shows 0 uploaded, N re-synced** — `batchAddMediaItems` silently
fails for photos removed via Google Photos UI. Use "Re-upload" with specific photos selected.

---

## Planned milestones (see `docs/plans/`)

Living roadmap entries — each is a design doc the next contributor can
pick up and implement without re-deriving the shape:

- **`docs/plans/infer-location-refinements.md`** — post-M19 cascade
  fixes surfaced by the first 127k-library apply. Three ordered
  changes: (1) cap hop depth at ~6 (observed chain ran 776 deep), (2)
  downweight inferred anchors on re-scan so decay protects cross-run
  compounding, (3) delete the dead `--max-cascade-rounds` flag and
  rename `cascade_rounds_used` → `max_hop_count` to match reality.
  All localized to `photosearch/infer_location.py`.
- **`docs/plans/google-photos-import.md`** — **M20**. Takeout-based
  import of ~200K smartphone photos. The Library API read scopes were
  deprecated for third-party apps on March 31, 2025, so the API path is
  closed; Takeout is the only route. Incremental per-year export,
  composite dedup (photoTakenTime + camera model + filename stem + phash),
  lands in `/photos/YYYY/YYYY-MM-DD_gphotos/` to keep origin visible and
  rollback trivial. Adds `phash`, `import_source`, `google_photo_id`
  columns. New CLI `photosearch takeout-import` using a reviewable
  ndjson plan ledger for resumability. Phone GPS amplifies inferred-
  geotag recall on camera photos — run after each year's import.
**Shipped, kept for reference:**
- **`docs/plans/bulk-set-location.md`** — both halves done (M19
  inferred + `/geotag` manual). Future potential: structured
  `country`/`admin1`/`admin2`/`locality` columns to unlock radius
  search and region-scoped queries.
- **`docs/plans/faces-clustering-and-perf.md`** — M18 clustering
  overhaul + merge suggestions + accept/reject UI all shipped. Future
  potential: quality pre-filter (`det_score` + bbox area), HDBSCAN
  for varying density, materialized `face_groups` table. Pick up only
  if a concrete pain surfaces — the current clustering has been good
  enough on 120k+ faces in day-to-day use.

When picking up any active plan, read the doc first — it contains the
shape the schema migrations should take and the UX calls that have
already been made.

---

## Adding New Features

### New CLI command
```python
@cli.command("my-command")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
def my_command(db):
    """One-line description."""
    with PhotoDB(db) as photo_db:
        pass
```

### New search type
1. Add `search_by_X()` in `search.py`
2. Wire into `search_combined()` — add to `result_sets`
3. Add query param to `web.py:api_search()`
4. Add UI control in `frontend/dist/index.html`

### New indexing pass
1. Add processing function in `photosearch/` module
2. Use streaming generator pattern (Key Patterns §1)
3. Add `enable_X` / `force_X` params to `index_directory()` in `index.py`
4. Add the same pass to `_index_collection()` in `index.py` (collection mode)
5. Add `--X` / `--force-X` flags to `index` command in `cli.py`
6. Add to status page run-commands in `status.html`

### New API endpoint
1. Add route in `web.py` with appropriate method + path
2. Use `with _get_db() as db:` for database access
3. For long-running ops, consider SSE pattern (Key Patterns §7). Reference
   implementations:
   - **Google Photos upload** (`web.py:api_upload_google`) — per-file
     progress over slow network + per-file ledger writes
   - **Stacking detect stream** (`web.py:api_detect_stacks_stream`) — pure
     CPU job with multiple phases. Pass `on_progress` + `should_abort`
     through to the worker function (see `stacking.py:run_stacking`);
     throttle emission to ~0.25s; check abort at every phase transition
     and every ~1000 iterations inside hot loops; raise `InterruptedError`
     on abort; terminal events `done` / `cancelled` / `fatal`.
4. Add frontend integration in the appropriate HTML file. For POST+SSE,
   use `fetch` + `response.body.getReader()` + `\n\n`-split parsing
   (EventSource can't POST); keep an `AbortController` around so the UI
   can cancel. See `frontend/dist/status.html:StackingForm` for a
   minimal template.

### Schema changes
1. Bump `SCHEMA_VERSION` in `db.py`
2. Add `CREATE TABLE IF NOT EXISTS` or `ALTER TABLE` in `_init_schema()`
3. Ensure migration SQL appears after any table it depends on
4. Add test in `tests/test_db.py` that creates a minimal old-version DB and verifies
   the migration runs correctly

### New frontend page
1. Create `frontend/dist/<page>.html` using the pattern from existing pages
   (plain React UMD, `React.createElement`, no build step)
2. Include `<script src="/shared.js"></script>` and use `PS.SharedHeader`
   with a new `activePage` id
3. Add a route in `photosearch/web.py` near the other `@app.get("/<page>")`
   block that serves the HTML with `Cache-Control: no-cache`
4. Add the page's nav id to `PS.SharedHeader`'s `navLinks` in `shared.js`
   so the link appears on every page
5. Styles live inline in each `<page>.html`; if you're reproducing a pattern
   from merges.html / faces.html (sticky toolbar, kbd hints, face-strip,
   etc.), copy the CSS rather than inventing

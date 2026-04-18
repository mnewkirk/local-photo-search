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

File is `photo_index.db` (not `photos.db`). Schema version 12. Key tables: photos, faces,
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

## Face clustering

New faces land with `cluster_id = NULL` — per-batch clustering was removed from
`index_directory()`, `_index_collection()`, and `worker_api.submit_results()`
because every batch independently restarted IDs at 0 and collided
("Unknown #0" was the union of the first face of every batch).

Grouping is an on-demand step:

```bash
$DC run --rm photosearch recluster-faces [--eps 0.55] [--min-samples 3] [--dry-run]
```

Runs global DBSCAN over every `person_id IS NULL` encoding, renumbers all
unknown clusters from scratch, and clears `ignored_clusters` in the same
transaction (IDs change, so ignore flags would otherwise mis-apply). Loader
uses `np.frombuffer` over concatenated BLOBs — 120k faces decode in seconds
on an N100; DBSCAN at 512-dim with ball_tree is the dominant cost after that.

`/api/faces/groups` is paginated (`limit`/`offset`/`filter`/`total`/`counts`),
hides singletons by default (`?include_singletons=1` to restore), and
auto-downgrades its O(N²) similarity sort to count-sort above 500 groups.
`/api/faces/crop/{id}` disk-caches at `thumbnails/face_crops/{id}_{size}.jpg`.

Remaining work in `docs/plans/faces-clustering-and-perf.md`: persist
`det_score` + bbox area and filter low-quality faces before clustering,
swap to HDBSCAN for varying density, and materialize a `face_groups` table.

## Detailed Reference

For full API endpoint list, indexing pass details, Google Photos integration, stacking
parameters, face management commands, and troubleshooting, see `.claude/skills/photo-search/SKILL.md`.

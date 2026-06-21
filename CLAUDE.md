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

File is `photo_index.db` (not `photos.db`). Schema version 22. Key tables: photos, faces,
persons, face_references, collections, collection_photos, photo_stacks, stack_members,
review_selections, google_photos_uploads, ignored_clusters, generations, schema_info.

Schema migrations run automatically via `_init_schema()`. Bump `SCHEMA_VERSION` in `db.py`
when adding tables/columns.

`worker_processed` (schema v22) gained an `attempts INTEGER` column. The claim
path filters `attempts >= MAX_PROCESS_ATTEMPTS` (default 3) instead of `NOT
EXISTS`, so transient failures (HEIC files indexed pre-pillow-heif, runner-OOM
blips) auto-retry on the next pass; truly broken files stop being claimed after
N tries. `mark_processed` UPSERT-increments. This replaces the manual
`retry-failed-describe` workflow for newly-stuck photos — that CLI is still
useful for clearing historical pre-v22 markers (which all migrated to
attempts=1) when you want to give them a clean re-attempt counter.

`generations` (schema v21) is a provenance log — one row per LLM text artifact
(describe, category-content/visual, keywords, verify), with `photo_id`,
`text_type`, `generated_text`, `model_used`,
`model_version`, `created_at`. The `photos` table still holds the current
"promoted" description/tags; `generations` is the full history, so outputs from
different models can coexist with provenance. `worker_api.submit_results` logs a
row per artifact via `db.log_generation()`. Backfill existing data with
`photosearch backfill-generations` (idempotent).

## Richer reverse geocoding (optional)

Default reverse-geocoding uses the `reverse_geocoder` library's
bundled `cities1000` dataset (~158k populated places). That labels
Point Reyes photos as "Inverness", Muir Woods photos as "Mill
Valley", etc. For richer labels (named parks, beaches, monuments,
smaller towns), download the filtered GeoNames `allCountries`
dataset:

```bash
$DC run --rm photosearch download-geonames
```

One-time ~400 MB download to `/data/geonames/`. After it finishes,
re-label existing photos:

```bash
$DC run --rm photosearch normalize-places --force
```

Runtime cost: ~1 GB RAM at steady state (the KDTree), ~10-30 s
build on first query each process. Fully falls back to stock
`reverse_geocoder` if the dataset isn't present — nothing breaks if
you skip this.

Feature-code filter covers populated places (class P) plus named
POIs: parks (PRK, PRKN), reserves (RESN, RESW, RESF), monuments
(MNMT), beaches (BCH), lakes (LK), waterfalls, historic sites,
mountains, peaks, capes, forests. Tweak `_KEEP_FEATURE_CODES` in
`photosearch/geonames_rich.py` to add more.

## Debugging against the prod DB locally

`./debug-db.sh` pulls `/data/photo_index.db` from the NAS via rsync
and exposes it for read-only SQL debugging on the Mac. Avoids the
multi-line-python-paste headache of running diagnostics through
`$DCPY photosearch -c "…"` and keeps the prod DB untouched.

```bash
./debug-db.sh pull                       # rsync DB + -wal/-shm locally
./debug-db.sh shell                      # sqlite3 -readonly shell
./debug-db.sh query "SELECT ..."         # one-shot SQL
./debug-db.sh person Calvin "Lucas Valley"   # person-coverage CLI
./debug-db.sh stats                      # stats CLI
./debug-db.sh clean                      # delete local copy
```

Env overrides (`NAS_HOST`, `NAS_DATA_DIR`, `LOCAL_DB`) let you point
it at a different NAS / path / local filename. The local copy goes
under `.gitignore` as `photo_index.db.local*`.

Related: `photosearch person-coverage NAME [--place-like PATTERN]`
(added as a proper CLI command) prints face-match coverage so you
can diagnose "X has many photos at Y but Person-X search only returns
N" queries — tells you whether the gap is face detection, face
matching, or elsewhere. The `debug-db.sh person` subcommand wraps
this against the local copy.

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

# Git on the NAS — UGOS owns this dir as a different uid, so bare `git`
# fails with "dubious ownership". Two workarounds; pick whichever fits:
#
# 1. Host-side, single command — pass `-c safe.directory=...` per-call.
#    Works for status / diff / log / add / commit / push / etc.
git -c safe.directory=/volume1/docker/photosearch status
git -c safe.directory=/volume1/docker/photosearch diff --cached
git -c safe.directory=/volume1/docker/photosearch commit -m "..."
#
# 2. Alpine container — only needed when you'd otherwise want to set
#    persistent `git config` (user.email/name) without polluting the host.
docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo pull"

# Push: the remote is HTTPS (https://github.com/mnewkirk/local-photo-search.git)
# but the host has no stored HTTPS credential — `gh` CLI holds the token.
# Wire gh in as the credential helper for the push call:
git -c safe.directory=/volume1/docker/photosearch \
    -c credential.helper='!gh auth git-credential' push origin main
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

**Bare-metal / native** — required when the worker has a GPU (the Docker fleet
is CPU-only). Two ways to launch:

```bash
# Easier: same command on every machine; auto-picks native on WSL2, Docker on Mac/Linux.
./run-workers.sh -s http://<NAS-IP>:8000 -p clip,faces,quality -d /photos/2026 -n 3
# Override with --native / --docker. --ollama-host URL for non-default Ollama.

# Or run cli.py directly:
python cli.py worker -s http://<NAS-IP>:8000 -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -d /photos/2026
```

**`-d` gotcha:** `-d` must be a real subdirectory (e.g. `/photos/2026` or
`/photos/2026/2026-04-09`). `-d /photos` — the photo *root* — always 404s
("No photos found in directory /photos"): `get_directory_photo_ids` strips
the photo_root prefix, leaving an empty prefix that matches no relative DB
path. To process the whole library, **omit `-d` entirely.**

Native mode launches `cli.py worker` processes from the project venv (GPU
auto-detected via torch+rocm / cuda) with `HSA_ENABLE_DXG_DETECTION=1` and
`OLLAMA_HOST` resolved to the Windows-host gateway on WSL2. Worker logs land
in `/tmp/photosearch-worker-fleet/worker-N.log`; `--status` / `--logs` /
`--stop` work in both modes.

Key files: `photosearch/worker.py` (client), `photosearch/worker_api.py` (server endpoints),
`run-workers.sh` (Docker fleet launcher). API routes under `/api/worker/*`.
Claims have TTL (default 30min) for crash recovery. `submit_results` for the
LLM passes (describe / category-* / keywords / verify) also logs to the
`generations` table — the worker sends `model` + `model_version` so provenance
is captured per artifact.

### Worker passes

The valid worker passes (`-p`/`--passes`, enforced by `valid_passes` in
`cli.py`) are: `clip`, `faces`, `quality`, `describe`, `category-content`,
`category-visual`, `keywords`, `verify`. The full LLM-backed pass list is:

```
clip,faces,quality,describe,category-content,category-visual,keywords,verify
```

**The old `tags` pass was removed** and split into three first-class passes —
passing `--passes tags` now errors with "unknown pass type". The split:

- **`category-visual`** (vision) — visual-quality tags from the image
  (inherits the old `tags` task; `--category-visual-model`, default `llava`).
- **`category-content`** (text-only) — content categories from the existing
  description (`--category-content-model`, default `llama3.2:3b`).
- **`keywords`** (text-only) — free-form keywords from the description
  (`--keywords-model`, default `llama3.2:3b`).

### Per-pass model strategy (Ollama defaults)

No single vision model wins every pass, so each has its own default
(all overridable: `--describe-model`, `--verify-model`,
`--category-visual-model`, `--category-content-model`, `--keywords-model`):

- **describe / regen → `llama3.2-vision`** — won a 100-image bakeoff over llava
  on free-form description quality (esp. text/OCR-heavy photos).
- **category-visual → `llava`** — the constrained tag task is a different shape;
  llama3.2-vision degenerates and over-selects on it. The
  regurgitation guard (`_MAX_PLAUSIBLE_TAGS`) drops responses echoing the
  vocabulary. `photosearch clean-garbage-tags` clears historical regurgitated
  tag sets so they get re-tagged.
- **verify → `llava`** — must differ from the describe/regen model for an
  independent cross-check.

`describe.py` has model-aware Ollama options (llama3.2-vision gets
`temperature` + `repeat_penalty` to suppress repetition loops) plus a
`_is_degenerate` detector → up to 2 retries → llava fallback in `describe_photo`.

These defaults apply when passes run through **Ollama**. When the fleet routes
to an OpenAI-compatible backend (LM Studio — see below), per-pass models are
selected by role env var instead.

### Routing LLM passes to LM Studio (OpenAI-compatible)

`describe.py` can route **all** LLM passes (text *and* vision) to an
OpenAI-compatible `/chat/completions` endpoint instead of Ollama. Set
`PHOTOSEARCH_TEXT_LLM_URL` (base must end in `/v1`) and the per-pass model is
chosen by **role** env var (not by model name):

| Pass | role | Env var |
|---|---|---|
| describe (+ regen) | `describe` | `PHOTOSEARCH_LLM_DESCRIBE_MODEL` |
| verify | `verify` | `PHOTOSEARCH_LLM_VERIFY_MODEL` |
| category-visual | `visual` | `PHOTOSEARCH_LLM_VISUAL_MODEL` |
| category-content / keywords | `text` | `PHOTOSEARCH_LLM_TEXT_MODEL` |

`run-workers.sh` has no flag for these — `export` them before a `--native`
launch (the native launcher inherits exported env):

```bash
export PHOTOSEARCH_TEXT_LLM_URL=http://<host>:1234/v1
export PHOTOSEARCH_LLM_DESCRIBE_MODEL=qwen/qwen3.5-9b
export PHOTOSEARCH_LLM_VERIFY_MODEL=google/gemma-4-e2b
export PHOTOSEARCH_LLM_VISUAL_MODEL=google/gemma-4-e2b
export PHOTOSEARCH_LLM_TEXT_MODEL=llama-3.2-3b-instruct
./run-workers.sh --native -s http://<NAS-IP>:8000 \
    -p clip,faces,quality,describe,category-content,category-visual,keywords,verify -n 2
```

This was adopted because Ollama proved unstable on a single 24 GB AMD GPU
(`model runner has unexpectedly stopped` under VRAM contention). LM Studio
caveats: enable JIT loading + max-loaded-models ≥3 + TTL off, and **raise each
model's context length above the 4096 JIT default** (LM Studio splits context
across parallel slots, so a vision describe request 400s with `Context size has
been exceeded`; qwen3.5-9b→16384, gemma→8192 worked).

### Ollama stall on the text passes — tight per-call timeout

The text-only category passes (`category-content`, `keywords` on
`llama3.2:3b`) intermittently trigger a multi-minute Ollama stall under
sustained `NUM_PARALLEL=1` load: the runner serves nothing for minutes,
then bulk-releases. Investigated 2026-05-20 — it is **not** GPU/ROCm
(a CPU-only WSL2 Ollama froze identically), **not** poison input (stuck
photos have normal descriptions), and **not** runaway generation (the
stuck photos replay in ~1s, `eval_count` ~20). Root cause is still open
(some Ollama-internal stall common to 0.23.x-CPU and 0.24-GPU). The
describe/`category-visual` vision passes have not shown it.

Mitigation in `describe.py`: `_TEXT_OLLAMA_TIMEOUT_S = 10` is passed as
`timeout=` on the `extract_categories_from_description` /
`extract_keywords_from_description` calls, so a stall aborts in 10s
(vs the 120s `_DEFAULT_OLLAMA_TIMEOUT_S` the vision passes keep) and the
photo is retried later instead of blocking the single Ollama slot. This
caps blast radius, not the root cause. A restart-watchdog was rejected:
it crashes workers (the un-retry-wrapped `check_available`) and only
treats the symptom.

### GPU acceleration

CLIP / quality / faces run on the worker's GPU automatically — `clip_embed.py`
and `faces.py` auto-detect CUDA/ROCm (faces is env-gated via
`PHOTOSEARCH_DEVICE`, mirroring clip_embed). The LLM passes (describe /
category-* / keywords / verify) run through Ollama by default — point
`OLLAMA_HOST` at whichever host has a GPU-capable Ollama — or through an
OpenAI-compatible backend (LM Studio) via `PHOTOSEARCH_TEXT_LLM_URL` (see
"Routing LLM passes to LM Studio" above). See the photo-search SKILL.md "GPU
acceleration" section for per-machine setup (Mac Metal, WSL2 + AMD via
librocdxg, WSL2 + NVIDIA).

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

## Phone-photo daily ingest

`/photos/_incoming/<source>/` is the staging area for camera-roll syncs from
phones (Syncthing on Android, Möbius Sync on iPhone). Each top-level subdir
under `_incoming/` is one "source" label that ends up in the destination
folder name. `_incoming` is in `_BUILTIN_EXCLUDED_DIRS` (see
`photosearch/index.py`), so a `photosearch index /photos` never walks the
staging tree even if it's mid-sync.

The daily sweep:

```bash
$DC run --rm photosearch ingest-incoming [--dry-run] [--no-index]
```

For each photo under `_incoming/<source>/`:
1. SHA-256 the file (`photosearch.index.file_hash`).
2. `SELECT id FROM photos WHERE file_hash = ?` — if found, move source to
   `_incoming/<source>/.processed/<rel-subpath>` and skip. The `.processed/`
   archive is dedup-only — successful imports are *moved out* of `_incoming/`,
   not copied, so there's no doubled disk usage for new photos.
3. Otherwise extract EXIF (or mtime fallback), move to
   `/photos/YYYY/YYYY-MM-DD_<suffix>/<filename>` (the suffix mirrors M20's
   `_gphotos` convention so origin stays visible at the path level). The
   suffix is `phone-<source>` for human source labels (`matt`, `wife`) and a
   **bare `<source>`** for camera-model labels (`_folder_suffix` /
   `_looks_like_camera_model` in `ingest.py` — uppercase-alnum-with-a-digit
   like `ILCE-7RM6` → `2026-06-19_ILCE-7RM6/`, no `phone-` prefix). Override
   the heuristic with `--bare-source LABEL` / `--phone-source LABEL` (also
   `PHOTOSEARCH_INGEST_BARE_SOURCES` / `PHOTOSEARCH_INGEST_PHONE_SOURCES`,
   comma-separated). Photos with no parseable date land in `/photos/_undated/<suffix>/`.

**RAW + video are "companions":** `ingest` relocates them into the same dated
folder (so every file reaches the NAS) but never adds a DB row or indexes them
— `is_photo = ext in INGEST_EXTENSIONS` gates the DB-dedup + CLIP path;
`COMPANION_EXTENSIONS` (RAW + video) take a move-only path with **destination
dedup** (same name + same hash already present → archive to `.processed`)
since they have no `photos.file_hash` row. Companion folders are kept out of
the returned `new_dirs`, so the follow-up CLIP index pass skips them. Video
with no EXIF date routes by file mtime (photos still go to `_undated`).

The SD-card import path feeds this: `D:\Photos\import-photos-safe.ps1` (on the
Windows workstation) reads each photo's/RAW's EXIF camera model and pushes all
media (JPEG/HEIC + RAW + video) to `\\nas\Photos\_incoming\<CameraModel>\`,
keeping its local `D:\Photos\Archive` backup, then ssh-triggers
`ingest-incoming`. Video has no readable model, so it rides along with the
shoot's body (the batch's single camera model) or lands under `unknown-camera`
(a default-bare suffix). The `/status` **Ingest incoming** card (`POST
/api/admin/ingest-incoming`, SSE) runs the same sweep on demand.
4. By default, runs `index_directory(photo_dir=<new dated folder>)` for
   CLIP + colors at the end. `--no-colors` drops the color pass (CLIP only)
   — lighter/faster for the daily cron and avoids the memory-heavy color
   extraction; colors are *not* a worker pass, so backfill them later with
   `photosearch index <dir>`. Faces / quality / describe / tags get picked
   up by the existing worker fleet on its next claim — no special wiring.

`.processed/` is safe to `rm -rf` at any time; it only holds deduped
sources kept for audit. Module: `photosearch/ingest.py`. Tests:
`tests/test_ingest.py` (12 cases — routing, dedup, archive layout,
mtime fallback, undated bucket, filename-collision suffix, HEIC,
AppleDouble skip, hidden-source-dir skip).

Cron entry on the NAS:

```cron
0 3 * * * cd /volume1/docker/photosearch && docker compose -f docker-compose.nas.yml run --rm photosearch ingest-incoming --no-colors >> /var/log/photo-ingest.log 2>&1
```

`--no-colors` keeps the daily sweep CLIP-only: color extraction is the slow,
memory-heavy part (~4 s/photo) and a first-time backfill once OOM-cut a run
mid-index. The move phase always completes first (files land safely on disk),
so a killed index pass only leaves photos un-embedded — re-run
`photosearch index /photos` (idempotent) to backfill CLIP, and a plain
`photosearch index <dir>` later for colors (colors are **not** a worker pass).

### Scheduling / crontab perms on the NAS

The cron job lives in **root's crontab** (`sudo crontab -l`), not
`/etc/cron.d` — a normal `crontab -l` as `cantimatt` is empty. The job needs no
root (cantimatt is in the `docker` group; the container moves files as PUID
1000), so it can move to the user crontab — but UGOS ships the cron subsystem
**without** the standard Debian setgid setup, so a non-root `crontab` fails with
`/var/spool/cron: mkstemp: Permission denied`. Fix once as root (a firmware
update may revert it):

```bash
sudo chown root:crontab /usr/bin/crontab && sudo chmod 2755 /usr/bin/crontab
sudo chown root:crontab /var/spool/cron/crontabs && sudo chmod 1730 /var/spool/cron/crontabs
sudo chown cantimatt:admin /var/log/photo-ingest.log   # so the non-root job can append
```

Then load the user crontab via a temp file — the
`( crontab -l; echo '...' ) | crontab -` one-liner is paste-fragile (mangled
line-continuations → `"-":1: bad minute`):

```bash
crontab -l 2>/dev/null > /tmp/mycron
echo '0 3 * * * cd /volume1/docker/photosearch && docker compose -f docker-compose.nas.yml run --rm photosearch ingest-incoming --no-colors >> /var/log/photo-ingest.log 2>&1' >> /tmp/mycron
crontab /tmp/mycron && rm /tmp/mycron
```

Firmware-update-proof fallback (keep it as root, just drop the password prompt):
`echo 'cantimatt ALL=(ALL) NOPASSWD: /usr/bin/crontab' | sudo tee /etc/sudoers.d/crontab-nopasswd && sudo chmod 440 /etc/sudoers.d/crontab-nopasswd`.

### Syncthing receiver

`docker-compose.nas.yml` runs a `linuxserver/syncthing` service
(`photosearch-syncthing`, web GUI on :8384) as a **receive-only** folder
landing phone camera-rolls in `./photos/_incoming/<source>/` (shared host path
with the photosearch container, mounted rw at `/photos/_incoming`). Because
`ingest-incoming` *moves* synced files out into the library, the receive-only
folder records those as local deletions and the **phone permanently shows
<100%** — this is expected, not a fault (all photos reach the NAS; the
`db/completion` for the phone device reads 100%). Do **not** click "Revert Local
Changes" in the Syncthing GUI — it re-downloads every already-ingested photo.
The apikey for the REST API is in `syncthing-config/config.xml` (gitignored
along with `photos/`).

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

## Deploy panel (Version / Build / Restart)

`/status` has a Deployment card backed by `photosearch/admin_api.py`. It
replaces the manual git-pull + compose-build + compose-up cycle. Endpoints:

- `GET /api/admin/version` — deployed SHA (from `/app/BUILD_SHA`) vs HEAD
- `POST /api/admin/git-fetch|git-pull` — git with `-c safe.directory=*`
- `POST /api/admin/docker-build` — SSE stream with explicit
  `--build-arg GIT_SHA=$(git rev-parse HEAD)` (env-var expansion in the
  compose file is unreliable)
- `POST /api/admin/restart` — **runs compose in a detached helper container**

**Critical invariant — never call `docker compose up` for the photosearch
service from inside the photosearch container itself.** Every process inside
a container shares one PID namespace; when compose kills the old container,
every process inside dies — including the compose subprocess that's
mid-restart, leaving the new container stranded in Created state with a
rename-suffix name. `setsid`/`nohup` don't help (different problem).

The restart endpoint does this instead:

```python
docker run --rm --detach \
  --name photosearch-restart-helper \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $HOST_REPO_DIR:/repo \
  -w /repo docker:cli \
  compose -p photosearch -f docker-compose.nas.yml up -d --no-deps photosearch
```

The helper has its own PID namespace and lifecycle. `HOST_REPO_DIR` is set
in docker-compose.nas.yml as `${REPO_DIR:-/volume1/docker/photosearch}` so
the helper knows the **host** path (in-container `/repo` is useless to a
sibling container).

**Shutdown handshake** (no worker-fleet churn during restart):

1. `/api/admin/restart` first runs `UPDATE worker_claims SET expires_at =
   datetime('now', '+15 minutes')` — workers' in-flight batches survive
   the swap gap even if `renew-claim` times out during downtime.
2. Helper launched; compose SIGTERMs the old photosearch.
3. `@app.on_event("shutdown")` in `web.py` flips
   `worker_api._shutting_down = True`.
4. Middleware returns `503 Retry-After: 30` on `/api/worker/*` and
   `/api/photos/<id>/full`. `WorkerClient._request()` sleeps the
   Retry-After window then loops — transparent to call sites. Browser
   UI paths keep serving normally.
5. `stop_grace_period: 60s` lets in-flight transfers drain cleanly.
6. Old dies, helper completes `compose up`, new container binds :8000.

**Workers built before `577a3b3` will crash on the 503** (old client
raised on any non-2xx). Pull + restart the worker fleet after server-side
deploy.

**Recovery from a wedged restart** (extremely rare with helper-container):
```bash
docker rm -f <stuck-container-id>
docker rm -f photosearch-restart-helper   # if it didn't self-clean
unset REPO_DIR   # bash ${VAR:-default} falls back only when UNSET, not empty
docker compose -f docker-compose.nas.yml up -d --no-deps photosearch
```
Walk `docker events --since 10m --until 30s 2>&1 | grep photosearch` to see
exactly which step failed.

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
  [--min-det-score 0.65] [--min-bbox-edge 60] \
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

### Quality pre-filter (schema v20)

`faces.det_score` stores InsightFace's detection confidence `[0, 1]`,
stamped on every new face from `detect_faces()` onward. `recluster-faces`
and `split-cluster` both filter out low-quality detections via a SQL
WHERE clause before DBSCAN runs:

```sql
WHERE (det_score IS NULL OR det_score >= :min_det_score)
  AND MIN(bbox_bottom - bbox_top, bbox_right - bbox_left) >= :min_bbox_edge
```

Defaults (tunable via `--min-det-score` / `--min-bbox-edge` or
`CLUSTER_MIN_DET_SCORE` / `CLUSTER_MIN_BBOX_EDGE` in `faces.py`):

- `min_det_score=0.65` — InsightFace det_score below this is usually a
  misfire (off-angle head, partial face, false positive).
- `min_bbox_edge=60` — shorter bbox edge in pixels. Smaller faces
  produce noisier encodings that cluster together by "junkness"
  rather than identity.

**NULL is grandfathered** — faces indexed before v20 have
`det_score IS NULL` and pass the filter unconditionally, so libraries
upgrading in place keep clustering as before. As users re-index, new
rows get real det_score values and the filter tightens. No forced
backfill.

`split-cluster` applies the same filter: faces below threshold drop to
`cluster_id=NULL` (not left in the source cluster) so a split leaves no
junk residue. The summary includes `filtered_out_count` alongside
`noise_count`.

Evidence motivating this filter: a 434-face cluster on the NAS split
cleanly at `eps=0.50` into one 75-face real-person sub-cluster plus
182 NULL'd noise faces. Low-quality faces were the primary attractor
cause; filtering them pre-DBSCAN prevents the attractor from forming
in the first place.

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

## Face over-matching cleanup & the one-person-per-photo invariant

Temporal matching (`match-faces --temporal`, `match_source='temporal'`) boosts
recall but over-matches: it tags the same person on 2+ faces in one photo
(rampant on kids at events — ~1600 photos had 2+ "Calvin" faces). A person
appears in a photo at most once, so any such photo has >=1 false match.

**`dedupe-person-faces --person NAME --references <photo-ids> [--min-gap 0.15]
[--report PATH] [--apply]`** — reference-based, **source-gated** cleanup for one
person. The reference set is a small curated list of photo-ids with one clear
face of them each, ideally spanning years (Calvin used one-per-year 2016-2026;
handles aging). Per double-person photo: **1 trusted (strict/manual) face →
keep it, drop the temporal extras** (decided by source, no distance, no
true-positive risk); **all-temporal → keep the face closest to the references,
drop the rest if the distance gap > `--min-gap`**; **2+ trusted → leave for
review**. `--report` writes an HTML keep-vs-remove **face-crop** gallery (crops
from `PHOTOSEARCH_NAS_URL`), least-confident first.

**`resolve-duplicate-persons [--apply]`** — universal safety net guaranteeing
the invariant for **every** person: for each (photo, person) with 2+ faces, keep
one by priority **manual > strict > temporal**, tie-break **det_score then bbox
area**; unmatch the rest. Run the reference-based dedupe first for precise
people, then this to sweep everyone.

**`restore-unmatched-faces [--apply]`** — both commands above snapshot
`(face_id, person_id, match_source)` into an on-demand **`face_dedupe_undo`**
table before nulling `person_id` (unmatch sets `match_source='dedupe_unmatched'`),
so the bulk cleanup is **reversible**. Restore re-applies the snapshot to faces
still unmatched.

Applied to the NAS 2026-06-20: 889 (Calvin reference) + 5382 (global) = **6271
faces unmatched, 4948 → 0 duplicate-person groups**, all reversible.

**Embedder/VLM swap is a tested dead-end — don't re-explore.** The hard
"wrong-green" cases (a photo where *neither* face is really the person, both
within temporal tolerance) are NOT separable by a better recognition head:
ArcFace (current), AdaFace IR-101, MagFace, QMagFace, and a VLM (qwen2.5-vl)
all fail on the same controlled set (`evals/adaface_compare.py`,
`magface_compare.py`, `vlm_face_compare.py`). MagFace quality is *inverted*
(wrong faces are higher quality than awkward-angle correct ones). Root cause is
identity ambiguity between similar-looking kids at the same event. The
within-photo **relative** comparison is the only robust signal. Per-year
references are worse than all-time.

### Desktop-as-client face recompute (heavy clustering off the NAS)

The N100 can't do the global DBSCAN at scale (~230k unmatched faces after the
quality filter), but the desktop can. Compute on the desktop off the synced
read-replica, push only the assignments to the NAS:

**`export-face-state --out FILE`** dumps `(face_id, cluster_id, person_id,
match_source)` for every face into a small (~9MB) SQLite file.
**`apply-face-state --db <nas> --from FILE [--overwrite-persons] [--apply]`**
ATTACHes it and applies set-based: person matches **additive** by default (only
fill faces unmatched on the target — preserves curation, and **skips
`dedupe_unmatched` faces** so a re-run of temporal matching can't undo the
dedup); clusters applied to still-unmatched faces; faces absent from the file
left untouched. (Distinct from the JSON `export-face-assignments`, which is
manual-label rebuild-preservation.)

Pipeline (`/tmp/face_recompute_pipeline.sh` is the reference):
```
sync replica → (on desktop) match-faces --temporal + recluster-faces
→ export-face-state → scp to NAS → (NAS) git pull
→ apply-face-state --apply → resolve-duplicate-persons --apply
```
`recluster-faces` writes by default (has `--dry-run`, no `--apply`);
`match-faces` writes directly. DBSCAN at 512-dim scales ~O(n²) (curse of
dimensionality) — ~230k faces takes tens of minutes even on a fast desktop.

### Face detection / matching gotchas

- **Big backlog**: clustering + matching are on-demand and fell far behind after
  the big imports (Google Takeout / phone ingest). At 2026-06-20: of 331k faces,
  only 51k (15%) matched to a person, 77k unmatched-but-clustered, and **202k
  never clustered** (`person_id IS NULL AND cluster_id IS NULL`, shown as bare
  "Unknown #" since the label is `"Unknown #" + cluster_id`).
- **Babies don't auto-match**: infant faces ARE detected and cluster fine, but
  ArcFace can't bridge baby→child, so `match-faces` won't attach them to the
  grown person. Recover by reclustering then **manually assigning the baby
  clusters** in `/faces`/`/merges`. (A baby face also can't seed a reference set
  until it's tagged.)
- **Some photos are genuinely undetectable**: e.g. photo 153553 returns 0 faces
  even re-detecting at det_size up to 2048px — InsightFace just can't find the
  (turned-away/occluded) face. Not a resolution problem; nothing recovers it.
- **Duplicate detections exist**: occasional identical-bbox face rows (one face
  detected twice). Harmless while unmatched; once both get a person they become
  a duplicate-person photo that `resolve-duplicate-persons` cleans.

## Planned milestones (see `docs/plans/`)

- `docs/plans/infer-location-refinements.md` — post-M19 cascade fixes
  surfaced on the 127k NAS library. Cap hop depth (cascade ran 776
  deep, 71% of candidates below 0.25 confidence), downweight inferred
  anchors on re-scan so decay protects cross-run compounding, drop the
  dead `--max-cascade-rounds` flag. All localized to
  `photosearch/infer_location.py`.
- `docs/plans/search-accuracy-improvements.md` — next-milestone
  search work. Quick-wins bundle: Reciprocal Rank Fusion across
  filters + recency decay on final ranking + structured
  `country`/`admin1`/`admin2`/`locality` columns (fixes "Calvin in
  France ranks ancient photos first" and "Marin County as a query
  returns nothing" simultaneously, ~200 lines + schema v19 + one
  backfill CLI). Later work: fuzzy name/place matching, ambiguity
  disambiguation via photo GPS priors, LLM query rewriter, VLM
  re-ranking, self-hosted Nominatim.
- `docs/plans/google-photos-import.md` — M20. Takeout-based import of
  ~200K smartphone photos. Google Photos API read scopes were deprecated for
  third-party apps in March 2025, so Takeout is the only path. Incremental
  per-year export, composite dedup (photoTakenTime + device + filename stem),
  lands in `/photos/YYYY/YYYY-MM-DD_gphotos/`. Phone GPS amplifies the
  inferred-geotag recall on camera photos.
- `docs/plans/llm-driven-search.md` — M24 **(SHIPPED)**. Stop hand-assembling
  search filters; let an LLM be the query planner. One **shared tool layer**
  (`photosearch/tools.py`: `search_photos` / `list_people` / `list_places`
  / `list_vocab` / `get_photo` / `get_photo_image` / `get_library_overview`)
  consumed by two adapters — **M24a** a streamable-HTTP MCP server
  (`photosearch/mcp_server.py`, low-level `Server`) as the `photosearch-mcp`
  NAS container (`mcp>=1.2`), and **M24b** an in-app `POST /api/ask` SSE agent
  loop (`photosearch/agent.py`) on the **local** LM Studio/Ollama backend
  (nothing leaves the NAS), surfaced as an "✨ Ask" mode toggle on the search
  page. Agent model via `PHOTOSEARCH_LLM_AGENT_MODEL` (falls back to the text
  role); tool-calling loop capped at 6 steps with a single-shot NL→filters
  fallback (`PHOTOSEARCH_AGENT_SINGLE_SHOT=1`) for non-tool-calling models.
  Image returns gated by `PHOTOSEARCH_MCP_ALLOW_IMAGES` (default off). Only
  existing-code change: a `person_ids` path in `search_combined`. No schema
  bump. Tests: `tests/test_tools.py`, `tests/test_agent.py`.

- `docs/plans/backfill-maintenance-sweep.md` — M25. **Queued: do NOT start
  until M24b ships.** Many derived-data passes get lost on new photos —
  structured location columns (`normalize-places`), inferred GPS
  (`infer-locations`), face clustering/matching (`recluster-faces` /
  `match-faces`), colors (cron runs `--no-colors`), and cross-folder
  stacking are all manual CLIs nothing schedules; only `ingest-incoming`
  runs daily. Proposes one idempotent `maintenance-sweep` (missing-only
  predicates, dependency-ordered, cron + `/status` button) over the
  lightweight CPU backfills, leaving heavy GPU passes to the worker fleet.
  Also a `validate-data`/`repair-data` tool for invalid rows — e.g. the
  corrupt `date_taken` control-byte values found in M24a (re-extract from
  EXIF/folder/mtime, else NULL), bad GPS, malformed JSON columns.

- `docs/plans/local-replica-and-writes.md` — M26. Pivot for the asymmetric
  hardware (weak N100 NAS, strong GPU desktop): run the whole experience
  (web UI + `/api/ask` agent + MCP) on the desktop off a **local
  read-replica** of the SQLite DB, with the local LM Studio — search compute
  (CLIP embed, KNN, SQL) moves to the fast machine. **M26a** syncs the DB +
  thumbnail mirror (rsync, nightly + a "sync now" button; thumbs ≈12 GB,
  append-only so tiny deltas; DB ≈1 GB; previews/full proxied from the NAS).
  **M26b** adds **write tools** (tags / location / collections) with a
  read-local / **write-NAS-authoritative / mirror-local** dual-write model
  (NAS applies + returns canonical values → mirror those to the local DB so
  search updates instantly without a full sync; nightly sync reconciles) and
  mandatory guardrails (explicit id-set scoping, dry-run→confirm, affected-
  count cap, reversible+audited via `location_source='manual'` /
  `generations`). Needs one new NAS endpoint (`bulk-set-tags`). Independent
  of M25. Reuses `debug-db.sh`'s rsync approach.

**Shipped, kept for reference:**
`docs/plans/bulk-set-location.md` (M19 inferred + `/geotag` manual) and
`docs/plans/faces-clustering-and-perf.md` (M18 clustering overhaul +
merge suggestions) are both complete. Each doc has a status header and
a "Future potential improvements" section for the genuinely-nice-to-
have items (structured location columns; HDBSCAN, quality pre-filter,
materialized face_groups). Pick those up only if concrete pain
surfaces.

## Detailed Reference

For full API endpoint list, indexing pass details, Google Photos integration, stacking
parameters, face management commands, and troubleshooting, see `.claude/skills/photo-search/SKILL.md`.

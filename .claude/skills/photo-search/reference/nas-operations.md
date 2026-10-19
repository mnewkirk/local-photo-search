# NAS operations

Read before running anything on the NAS: Docker aliases, volumes, debug-db.sh, geocoding data, ingest cron, Syncthing, scheduling.

<!-- moved verbatim from CLAUDE.md (line 69) on 2026-10-19 -->
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

Runtime cost: **~1.47 GB RAM** at steady state (measured on the NAS after the
fix below; it was ~6.3 GB peak before), **~7 s** build on first query each
process. Fully falls back to stock `reverse_geocoder` if the dataset isn't
present — nothing breaks if you skip this.

Feature-code filter covers populated places (class P, **subject to
`_MIN_POPULATION`**) plus named POIs: parks (PRK, PRKN), reserves (RESN, RESW,
RESF), monuments (MNMT), beaches (BCH), lakes (LK), waterfalls, historic sites,
mountains, peaks, capes, forests. Tweak `_KEEP_FEATURE_CODES` in
`photosearch/geonames_rich.py` to add more.

<!-- moved verbatim from CLAUDE.md (line 217) on 2026-10-19 -->
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

<!-- moved verbatim from CLAUDE.md (line 244) on 2026-10-19 -->
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

<!-- moved verbatim from CLAUDE.md (line 2128) on 2026-10-19 -->
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
echo '0 4 * * * cd /volume1/docker/photosearch && flock -n /tmp/photo-ingest.lock docker compose -f docker-compose.nas.yml run --rm photosearch ingest-incoming --no-colors >> /var/log/photo-ingest.log 2>&1' >> /tmp/mycron
crontab /tmp/mycron && rm /tmp/mycron
```

Firmware-update-proof fallback (keep it as root, just drop the password prompt):
`echo 'cantimatt ALL=(ALL) NOPASSWD: /usr/bin/crontab' | sudo tee /etc/sudoers.d/crontab-nopasswd && sudo chmod 440 /etc/sudoers.d/crontab-nopasswd`.

<!-- moved verbatim from CLAUDE.md (line 2157) on 2026-10-19 -->
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

<!-- moved verbatim from SKILL.md (line 1054) on 2026-10-19 -->
## NAS Operations

<!-- moved verbatim from SKILL.md (line 1056) on 2026-10-19 -->
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

<!-- moved verbatim from SKILL.md (line 1104) on 2026-10-19 -->
### Volume layout
```
/data/                      # Persistent Docker volume
  photo_index.db            # SQLite database
  thumbnails/               # Generated JPEG thumbnails
  google_photos_token.json  # Google OAuth token
  .insightface/             # InsightFace model cache (~300MB)
  .cache/photosearch/       # CLIP + aesthetic model cache
/photos/                    # Photo library (rw mount — ingest moves files in)
  2026/2026-01-15/DSC*.JPG  # Year/date-named folders
  YYYY/YYYY-MM-DD_phone-<source>/  # phone imports (ingest-incoming)
  _incoming/<source>/       # Syncthing staging (excluded from indexing)
  _undated/phone-<source>/  # phone photos with no parseable date
/references/                # Face reference photos
  references.yml            # Person → photo mapping
```

> The `/photos` mount is **rw** (not the older ro) so `ingest-incoming` can
> move phone uploads from `_incoming/` into the year/date tree. `_incoming` is
> in `_BUILTIN_EXCLUDED_DIRS`, so a `photosearch index /photos` never walks it.

<!-- moved verbatim from SKILL.md (line 1125) on 2026-10-19 -->
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

New faces land with `cluster_id = NULL`, and `recluster-faces` is the only
thing that forms "Unknown #N" groups. Run it after each face-indexing pass (or
batch thereof). Warning: every run renumbers every unknown cluster_id and
clears `ignored_clusters`, so any "ignore" decisions on unknown clusters need
to be reapplied afterward.

Until it runs, those faces are ungrouped — but they are **not** unreachable:
`/api/faces/groups` surfaces them as a single `type: "unclustered"` group
whenever a content filter (date / place / query / camera) is active, so "review
the faces from that day" works without a global recluster. Unfiltered, the
bucket is suppressed (it would be a six-figure blob) and
`GET /api/faces/group/unclustered/0/photos` without filters returns 400. The
bucket has no Rename/Merge/Split/Ignore and no multi-select — those reassign
every face in a group, which is wrong when the group is many different people.

<!-- moved verbatim from SKILL.md (line 1156) on 2026-10-19 -->
### Phone-photo ingest (Syncthing + `ingest-incoming`)

Phones sync their camera roll into `/photos/_incoming/<source>/` via a
`linuxserver/syncthing` service (`photosearch-syncthing`, web GUI :8384) defined
in `docker-compose.nas.yml`. The folder is **receive-only**; each top-level
subdir under `_incoming/` is one source label (`matt`, `nicole`, …). The daily
sweep evacuates them into the library:

```bash
$DC photosearch ingest-incoming [--dry-run] [--no-index] [--no-colors]
```

Per file: SHA-256 → if it matches an existing `photos.file_hash`, move the
source to `_incoming/<source>/.processed/` (dedup audit) and skip; otherwise
EXIF-date it and **move** (not copy) to `/photos/YYYY/YYYY-MM-DD_<suffix>/`
(undated → `/photos/_undated/<suffix>/`). Then it indexes each new dated
folder (CLIP + colors). Module `photosearch/ingest.py`; 15 tests in
`tests/test_ingest.py`.

- **Folder suffix** (`_folder_suffix`): `phone-<source>` for human labels
  (`matt`, `wife`), bare `<source>` for camera-model labels (uppercase alnum
  with a digit, e.g. `ILCE-7RM6` → `2026-06-19_ILCE-7RM6/`). The SD-card
  importer (`scripts/windows-import/import-photos-safe.ps1`, run from a repo checkout on the Windows box) names `_incoming/<model>/` dirs
  after the EXIF camera model, so camera bodies get clean per-model folders
  with zero config. Force either behavior with `--bare-source` /
  `--phone-source` (or `PHOTOSEARCH_INGEST_BARE_SOURCES` /
  `PHOTOSEARCH_INGEST_PHONE_SOURCES`, comma-separated).
- **RAW + video companions:** ingest relocates RAW (`RAW_EXTENSIONS`) and video
  (`VIDEO_EXTENSIONS`) into the dated library folder too (every file reaches
  the NAS) but **never indexes them** — no DB row, no CLIP, and their folders
  are kept out of `new_dirs`. They dedup at the destination (same name + hash)
  instead of via `photos.file_hash`. Video with no EXIF routes by mtime. The
  importer reads RAW's EXIF model (RAW co-locates with its JPEG); model-less
  video rides the batch's single camera model or lands under `unknown-camera`
  (default-bare). `stats` gains `companions_moved` / `companions_deduped`.
- **On-demand run:** the `/status` **Ingest incoming** card hits
  `POST /api/admin/ingest-incoming` (SSE; `?dry_run=true` supported), which
  streams `docker compose run --rm --no-deps photosearch ingest-incoming
  --no-colors` — the same job as the cron, in a throwaway sibling container so
  the CLIP import never runs inside the web process.

- **`--no-colors`** runs the post-move index CLIP-only. Color extraction is the
  slow, memory-heavy part (~4 s/photo) and a first-time backfill once OOM-cut a
  run mid-index. The **move phase always completes first** (files land on disk),
  so a killed index pass only leaves photos un-embedded — `photosearch index
  /photos` (idempotent) backfills CLIP; colors are **not** a worker pass, so a
  plain `photosearch index <dir>` backfills those. Faces/quality/describe/tags
  are left to the worker fleet either way.
- **Re-filing historical `unknown-camera` folders:**

  ```bash
  $DC photosearch refile-unknown-camera [--only DIR] [--limit N] \
      [--include-indexed] [--infer-from-sibling] [--apply --audit /data/refile.csv]
  $DC photosearch refile-unknown-camera --undo /data/refile.csv --apply
  ```

  Moves `YYYY/YYYY-MM-DD_unknown-camera/*` onto the body each file's **own**
  EXIF names, reusing `ingest._file_suffix` verbatim. Dry run by default (and
  read-only on the DB); `--apply` refuses without `--audit` (the CSV is the undo,
  written intent-then-confirm and fsynced per file) and takes ingest's
  `_sweep_lock`. The move is `os.link` + `unlink`, so it **cannot** overwrite —
  the nightly ingest writes into these same dated folders. Never deletes, never
  guesses a model from the date or a sibling folder (two bodies were in use on
  several of these days); files with no readable model are left in place. The DB
  gate matches on exact `filepath`, not the `folder` column, which some writers
  leave stale. Module `photosearch/refile.py`; tests `tests/test_refile.py`.
  See the CLAUDE.md section for the full operator sequence.
- **Stuck-phone gotcha:** because ingest *moves* files out, the receive-only
  Syncthing folder records them as local deletions and the **phone permanently
  shows <100%** — expected, not a fault (all photos reach the NAS; the REST
  `db/completion?device=<phone>` reads 100%). Do **not** click "Revert Local
  Changes" — it re-downloads every already-ingested photo. REST apikey is in
  `syncthing-config/config.xml` (gitignored, along with `photos/`).

<!-- moved verbatim from SKILL.md (line 1230) on 2026-10-19 -->
### Cron / scheduling on the NAS

The ingest job runs from cron:

```cron
0 4 * * * cd /volume1/docker/photosearch && flock -n /tmp/photo-ingest.lock docker compose -f docker-compose.nas.yml run --rm photosearch ingest-incoming --no-colors >> /var/log/photo-ingest.log 2>&1
```

It lives in **root's crontab** (`sudo crontab -l`), not `/etc/cron.d` — a normal
`crontab -l` as `cantimatt` is empty. The job needs no root (cantimatt is in the
`docker` group; the container writes as PUID 1000), so it can move to the user
crontab — but UGOS ships cron **without** the standard Debian setgid setup, so a
non-root `crontab` fails `/var/spool/cron: mkstemp: Permission denied`. One-time
fix as root (a firmware update may revert it):

```bash
sudo chown root:crontab /usr/bin/crontab && sudo chmod 2755 /usr/bin/crontab
sudo chown root:crontab /var/spool/cron/crontabs && sudo chmod 1730 /var/spool/cron/crontabs
sudo chown cantimatt:admin /var/log/photo-ingest.log
```

Load the user crontab via a temp file — the `( crontab -l; echo '...' ) |
crontab -` one-liner is paste-fragile (mangled `\` continuations →
`"-":1: bad minute`):

```bash
crontab -l 2>/dev/null > /tmp/mycron
echo '0 3 * * * ...full line...' >> /tmp/mycron
crontab /tmp/mycron && rm /tmp/mycron
```

Firmware-proof fallback (keep it root-owned, just drop the password prompt):
`echo 'cantimatt ALL=(ALL) NOPASSWD: /usr/bin/crontab' | sudo tee /etc/sudoers.d/crontab-nopasswd && sudo chmod 440 /etc/sudoers.d/crontab-nopasswd`.

<!-- moved verbatim from SKILL.md (line 1264) on 2026-10-19 -->
### Scheduling + replica-mode maintenance (2026-07-17)

`maintenance-sweep` **runs nightly on the NAS from the NAS user's own crontab**
(not root's), at **18:30 host-local** (Pacific) — verified on the box
2026-09-19 with `crontab -l`:

```cron
0 4 * * *   … flock -n /tmp/photo-ingest.lock      … ingest-incoming --no-colors …
30 18 * * * … flock -n /tmp/photo-maintenance.lock … maintenance-sweep --apply --no-colors …
```

Both are `flock`-guarded, so a run that overlaps the previous one exits instead
of stacking. Host-local time means it drifts an hour against UTC with DST
(01:30 UTC in summer, 02:30 in winter); the `CRON_TZ=UTC` variant was never
adopted. Log: `/var/log/photo-maintenance.log`. `--recluster` /
`--dedup-photos` stay OFF (recluster clears `ignored_clusters`; dedup DELETEs
photos). (This section used to say the entry was "not yet installed", in root's
crontab, at 01:00 UTC, with ingest at 03:00 — all four were stale.)

**The sweep's `match_faces` stage is STRICT-ONLY.** It used to run the temporal
matcher too, every night, over shoots nobody had reviewed. On 2026-09-19 it
swept a shoot ingested that afternoon and wrote **466 temporal labels (Calvin
382, Ellie 149)** beside 65 strict ones — on exactly the kind of shoot where
temporal is ~4% accurate and tags one kid across both teams. Temporal is now an
explicit opt-in: `maintenance-sweep --match-temporal`, `match_temporal=True` on
`run_maintenance_sweep`, `"match_temporal": true` on the API. Hand-run
`match-faces --temporal` is unchanged. Tests: `tests/test_maintenance_match_faces.py`.

**Known, NOT fixed — a nightly match/unmatch churn loop.** The log shows the
stage "applying" ~6,000 matches every night while the unmatched pool never
shrinks (202k → 213k over eight runs), because `resolve_dups` strips
~5,300–6,100 of them the *same night*. The ~5,500 `dedupe_unmatched` faces are
deliberately still matchable (they may belong to a *different* person), so they
are re-matched to the *same* person and re-stripped forever — on the sweep's
heaviest CPU stage. Strict-only should shrink it; it has not been measured
since. The durable fix is to remember *which person* a face was de-duplicated
away from and skip only that pairing.

**Replica mode is now gated.** A sweep writes to whatever `PHOTOSEARCH_DB`
points at, and `sync-replica.sh` replaces the replica's DB wholesale — so a
replica-side apply used to be silently destroyed on the next sync. Now:

- `apply=true` on the replica with the NAS unreachable → **503, refuses to run**.
- `colors` / `dedup_photos` / `requeue` → **400** in replica mode (colors needs
  pixels the replica doesn't have; dedup DELETEs; requeue is a no-op because
  the fleet claims from the NAS).
- Everything else: pre-flight fingerprint → auto-sync if drifted → compute →
  one push at the end. Push modes (`maintenance_sync.push_mode`): cheap
  deterministic stages are **re-triggered** on the NAS; `stacking` is
  **transferred** as rows; `match_faces` / `recluster` are **face_state** —
  the replica exports a face-state file (`photosearch/face_state.py`, the same
  code as the `export-face-state` / `apply-face-state` CLI pair) and POSTs it
  to `POST /api/admin/maintenance-apply-face-state`. The leg order is transfer
  → face_state → trigger, and a landed match adds `resolve_dups` to the
  triggers.
- face_state apply on the NAS applies **only what the replica recomputed**:
  match → additive person fill (skips `dedupe_unmatched`), no cluster copy (a
  sync-time cluster id would revert a split made on the NAS since); recluster
  → cluster ids for unmatched faces, stale ids cleared on named faces and on
  faces absent from the file, and `ignored_clusters` **remapped** — a new
  cluster stays ignored when >50% of its faces sat in an ignored cluster
  before. Known window: a label cleared on the NAS between sync and push
  (`/clear` sets `match_source=NULL`, indistinguishable) is re-filled by a
  match push.

`maintenance_runs` (schema v29) holds the per-stage watermark on both machines.
The `/status` Replica card and `/admin_maintenance` show per-stage drift;
**"Replica ahead — unpushed"** means local work will be lost on the next sync.

Module: `photosearch/maintenance_sync.py`. Spec:
`docs/superpowers/specs/2026-07-17-maintenance-push-up-design.md`.

---

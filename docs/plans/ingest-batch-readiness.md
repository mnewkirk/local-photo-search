# Ingest-batch readiness — automation + per-batch status flow

**Status:** planned (2026-09-19). Produced by a two-planner debate (two critique
rounds); the `## Decisions` table records what the planners could not settle and what
the owner chose.

## Goal

A fresh ingest should reach **"ready to review"** — reviewable on `/faces` for face
matching, and rankable for top photos — with one action, and the owner should see at a
glance where a batch is: every step, in exactly one state, with the next action obvious.

## Why now — the 2026-09-19 incident

During an SD-card ingest the NAS web app wedged completely: all 40 request threads in
disk wait, 20+ minutes per request, ingest moving **one file every ~3 minutes**. Root
cause was an unrelated ~50 MB/s SMB bulk read (a desktop backup) saturating the 2-disk
mirror; stopping it took ingest to ~2 files/s and load from 44 to 5 within minutes.

Two lessons shape this design:

- **The status page made it worse.** `PipelineFunnel` and the Workers panel poll with
  `setInterval`, which fires whether or not the previous poll returned. Against a starved
  disk, two open tabs stacked requests until the pool was full. A status view that can
  wedge the box it reports on is a failure.
- **"Is ingest running, stalled, or done?" was unanswerable** without SSH. Nothing
  persists ingest progress; files-per-minute — the one number that would have shown the
  stall — did not exist.

## Model

**A batch is one dated folder** (`photos.folder`, the indexed v25 column). A card dump
spanning three days is three shoots, so three batches; a `run_id` groups the folders
that came from one sweep, for display.

**Membership is the folder column itself — nothing is materialized.** No per-batch
collection (the 03:00 phone cron ingests daily; auto-collections would pile up on the
Collections page by the hundred).

**Late arrivals re-open the batch.** Because membership and state are both derived live,
photos added to the same folder later simply join it: `register_batch` upserts on
`directory`, updates `photo_count`, clears `ready_at` / `dismissed_at`, and the affected
steps flip from *completed* back to *needs queue* for just the new photos. Every pass is
missing-only, so nothing already done is redone. This is a requirement, with a test.

**Ingest liveness lives in the DB.** A sweep row carries `status`
(`moving → indexing → registered`, or `failed`), `heartbeat_at`, `files_seen`,
`files_moved`. It counts **files moved**, not photos: `ingest.py`'s move loop never mints
photo ids (`index_directory()` does, later, per new dir from `cli.py`). `moving` with a
heartbeat older than 5 minutes renders as **stalled**; files/min is shown.

## Steps

### 1. Schema v30 + batch module
`photosearch/db.py` (`SCHEMA_VERSION = 30`, additive `CREATE TABLE IF NOT EXISTS` only),
new `photosearch/ingest_batches.py`.

- `ingest_sweeps(run_id PK, status, started_at, finished_at, heartbeat_at, files_seen,
  files_moved, error)`
- `ingest_batches(id, directory UNIQUE, source, run_id, created_at, updated_at,
  photo_count, ready_at, dismissed_at)`
- `ingest_batch_jobs(batch_id, step, job_kind, opened_at, expires_at, closed_at,
  PRIMARY KEY(batch_id, step))` — **job intent only**, never derived state.

*Verify:* `tests/test_db.py` v29→v30 is additive + idempotent;
`tests/test_ingest_batches.py` upsert idempotency and **re-open on late arrival**.

### 2. Ingest writes the sweep + batches
`photosearch/ingest.py`: create the sweep row on the first file moved (no empty sweeps),
throttled heartbeat (~1/s) in the move loop, terminal status in a `finally` so a crash
leaves `failed`, never a phantom `moving`. `cli.py:ingest_incoming_cmd`: flip to
`indexing`, then `register_batch()` per `new_dir` after its `index_directory()`;
register even under `--no-index` (the diagram just shows `clip: needs queue`).

*Verify:* extend `tests/test_ingest.py` — sweep counters, crash leaves `failed`,
companions/dedup-skips don't create batches.

### 3. Indexed folder fast path
`db.get_directory_photo_ids`: try `folder = ?` equality first, fall back to the existing
`filepath LIKE 'dir/%'` scan when it matches nothing (year-level dirs like `/photos/2026`
must keep working). This helper is on the fleet's **per-claim** path, so it also speeds up
every existing `-d` run.

*Verify:* tests for leaf dir (fast path), year dir (fallback), photo root (still 404s).

### 4. State derivation — `photosearch/batch_state.py`
Steps: `ingest`, the 9 worker passes (`clip`, `faces`, `quality`, `aesthetics`,
`describe`, `category-content`, `category-visual`, `keywords`, `verify`), `stacking`,
`normalize_aesthetics`, `match_faces`, `resolve_dups`, `warm_crops`, `rank_measure`.

Per step compute `total / eligible / done / remaining / failed`. **`remaining == 0` does
not mean done** — both planners verified two ways it lies:

- `count_unprocessed_photos` already excludes `attempts >= MAX_PROCESS_ATTEMPTS`, so a
  permanently stuck pass counts down to 0.
- `category-content` / `keywords` / `verify` gate on `description IS NOT NULL`, so before
  describe runs they count 0.

| state | rule |
|---|---|
| **completed** | `done == total` (so `eligible == total` and `failed == 0`) |
| **running** | an unexpired `worker_claims` row for the pass intersects the batch's ids |
| **queued** | an open, unexpired `ingest_batch_jobs` row |
| **needs queue** | `remaining > 0`, neither of the above |
| **waiting** | `eligible < total` — upstream step incomplete (shows `waiting on: describe`) |
| **blocked** | `remaining == 0` and `failed > 0` — a human must look |

`remaining` reuses `worker_api._count_scoped` — the exact claim predicate, so the number
shown is the number the fleet will claim. "Queued" comes from the job-intent row, **not**
from parsing `/workers/fleet-status` (it shells `run-workers.sh --status`, which itself
curls the heavy `/api/worker/status`, and is blind to hand-launched fleets and to the
other machine); fleet-status may only *clear* a stale job row. `warm_crops` and
`rank_measure` derive from job rows only — stat-ing crop files is per-file I/O per poll.
CLIP has no attempts cap, so its unloadable rows surface as `blocked` via a no-progress
rule rather than sitting in `needs queue` forever.

*Verify:* `tests/test_batch_state.py` — one test per transition, plus the describe-gate
(`keywords` must read `waiting`, never `completed`) and exhausted-attempts (`blocked`).

### 5. Cheap status API — `photosearch/batch_api.py`
`GET /api/batches` (pure SQL over the batch/sweep tables, no derivation) and
`GET /api/batches/{id}` (full state, embeds queue depth + claims — one request per
render). Derivation sits behind a 30 s **in-memory** memo guarded by
`threading.Lock` acquired non-blocking: **a poll that finds the lock held returns the
cached snapshot immediately** instead of queueing. Responses carry `computed_at` / `stale`.
Derived state is never persisted.

*Verify:* `tests/test_batch_api.py` — two concurrent requests → one derivation; the
cached path makes zero `count_unprocessed` calls.

### 6. `batch-advance` — the one action
`photosearch/batch_advance.py`, CLI `photosearch batch-advance --batch N [--apply]`
(`envvar="PHOTOSEARCH_DB"`), SSE `POST /api/admin/batch-advance`
(`done` / `fatal` / `cancelled`).

- **NAS-side** (sibling container, shares `_ingest_lock`): `run_stacking(photo_ids=…)` —
  id-scoped, **never `--clear`** (that is what wiped the library's stacks twice);
  `normalize-aesthetics`; **strict** `match_faces_to_persons`;
  `resolve_duplicate_persons`; `warm_face_crops` for the batch's dates.
- **Desktop-side** (when `PHOTOSEARCH_NAS_URL` is set): launch the native fleet via the
  existing `/workers/start` (add a `directory` field) for passes in `needs queue`, in
  dependency order, writing a job row per pass (`expires_at = now + 6h`);
  `rank_measure` runs here only.
- **Not in "ready":** temporal matching (~4% accurate on these shoots; it manufactures
  Bulk-unmatch work), `recluster-faces` (clears `ignored_clusters`, ~an hour), colors,
  dedup. Unknown faces stay reviewable per day via the existing *Unclustered* bucket.
- **Trigger:** manual one-click. `scripts/batch-autopilot.sh` ships **off** (marker/PID
  file, never `pgrep -f`); `rank_measure` never runs unattended.

*Verify:* `tests/test_batch_advance.py` with fake stage callables — ordering, job rows
opened/closed, `--apply` gating, abort path.

### 7. `/batches` page + poll discipline
New `frontend/dist/batches.html` (React UMD, `React.createElement`): sweep banner
(moving / stalled / indexing, files/min), batch picker, then the flow diagram —
dependency-ordered nodes, one colour per state, `done/total` plus `failed` and
`waiting on`, and one **Advance batch** button. Nav link via `PS.SharedHeader`; the
maintenance funnel links out.

Add `PS.poll(fn, ms)` to `shared.js` — a `setTimeout` chain that re-arms **only after the
previous fetch settles**, with `AbortController` and a `document.hidden` pause — and
**convert `PipelineFunnel` and the Workers panel to it**. Fixing only the new page would
leave the incident's actual trigger in place.

*Verify:* `node scripts/check-frontend-refs.js`; `frontend/__tests__/poll.test.js`
asserts no overlapping calls.

### 8. Docs + CI
CLAUDE.md section, SKILL.md endpoints + schema note. `./scripts/test-like-ci.sh` from the
main checkout (worktree gitdir caveat). **Deploy the NAS before the desktop** — same shape
as `/mirror-fields`; the conductor degrades to "NAS out of date" per step, not a 503.

## Estimate
Steps 1–3 ~5 h · 4 ~6 h · 5 ~3 h · 6 ~8 h · 7 ~7 h · 8 ~2 h — **~4 days**.

## Risks
- Step 3 touches a helper on the shared claim path → fallback-to-LIKE is mandatory and tested.
- Job rows leaking `queued` if a fleet dies → `expires_at`, and a live claim or falling
  `remaining` supersedes.
- Hand-made subfolders under a dated dir break `folder =` equality → `register_batch`
  refuses a directory with no `folder`-matching rows.
- v30 rollout order (NAS first).

## Decisions

| Topic | Option A (Planner A, Sonnet) | Option B (Planner B, Opus) | Owner's choice | Date |
|---|---|---|---|---|
| Batch membership + fleet scope | `photos.folder` only, plus an indexed `folder = ?` fast path in `get_directory_photo_ids` | Materialize each batch into an auto-collection; folder as fallback | **A — folder only.** Owner asked what happens when photos land in the same folder later → made explicit: late arrivals re-open the batch | 2026-09-19 |
| Where the flow diagram lives | New dedicated `/batches` page | Card inside `/admin/maintenance` | **A — new `/batches` page** | 2026-09-19 |

Settled in debate without the owner: dated-folder granularity (B conceded); the
`eligible`/`failed` split with `waiting` + `blocked` (A conceded — verified bug);
job-intent rows as the source of "queued" (A conceded); non-blocking-lock cache (A
conceded); no persisted derived state (B conceded); strict-only matching in "ready"
(A conceded); manual trigger with autopilot off (both).

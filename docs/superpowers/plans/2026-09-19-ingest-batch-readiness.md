# Ingest-batch readiness — execution plan

Design + rationale + owner decisions: `docs/plans/ingest-batch-readiness.md`. Read its
"Model" and "State derivation" sections if a task brief leaves a *why* unanswered; the
task text below is the *what*. Already shipped (do not redo): `PS.poll` in
`frontend/dist/shared.js`; the indexed `db.get_directory_photo_ids`.

## Global Constraints

- Read `CLAUDE.md` "Key Patterns" first. Python 3.11, FastAPI, SQLite. Frontend is plain
  React UMD with **no build step**: `React.createElement` only, never JSX.
- **TDD.** Write the failing test first. Run Python tests with
  `venv/bin/python -m pytest <files> -q -p no:cacheprovider`
  (run from a checkout that has a venv — a git worktree may not have its own). Frontend:
  `cd frontend && npx jest` and
  `node scripts/check-frontend-refs.js` from the repo root. Never `source` an activate script.
- The shared `db` pytest fixture is **pre-seeded** with photos under `2026/…`. Use years
  `2090`/`2091` in new fixtures so nothing collides.
- **Privacy:** never write a hostname, IP, MAC, or username into a tracked file. Use
  `<nas-host>` / env vars.
- **A batch is one dated folder**: `ingest_batches.directory == photos.folder` (relative to
  photo_root, e.g. `2091/2091-09-19_ILCE-7RM6`). Membership is **derived live** as
  `SELECT id FROM photos WHERE folder = ?` — never materialized, never a collection.
- **Derived state is never persisted.** Only identity, lifecycle, sweep progress and
  job *intent* are stored.
- **No per-file filesystem I/O in any polled path.** Status endpoints read the DB only.
- `count_unprocessed == 0` does **not** mean "done" (see Task 3). Any code that treats it
  so is a defect.
- Every CLI `--db` option carries `envvar="PHOTOSEARCH_DB"`. SSE streams end with a
  terminal event (`done` / `fatal` / `cancelled`). Use `asyncio.get_running_loop()`.
- Frontend polling uses `PS.poll`, never `setInterval`, for anything that hits the server.
- Shared frontend helpers go on `PS.*` in `shared.js`; never copy a helper between pages.
- Do not run `git push`. Do not contact any network host. Commit your work on the current
  branch with a conventional message ending in the line
  `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add` the
  `frontend/node_modules` symlink or `.superpowers/`.

## Frozen interfaces (tasks depend on these exact names and shapes)

```python
# photosearch/ingest_batches.py  (Task 1)
STALL_SECONDS = 300
def start_sweep(db) -> str                                   # returns run_id (uuid4 hex)
def heartbeat_sweep(db, run_id, *, files_seen, files_moved) -> None
def set_sweep_status(db, run_id, status, error=None) -> None  # moving|indexing|registered|failed
def get_active_sweep(db) -> dict | None      # newest row with status in (moving, indexing)
def register_batch(db, directory, *, source=None, run_id=None) -> int   # upsert; RE-OPENS
def list_batches(db, include_dismissed=False, limit=50) -> list[dict]
def get_batch(db, batch_id) -> dict | None
def batch_photo_ids(db, batch) -> list[int]
def mark_ready(db, batch_id) -> None
def dismiss_batch(db, batch_id) -> None
def open_job(db, batch_id, step, job_kind, ttl_seconds=21600) -> None
def close_job(db, batch_id, step) -> None
def open_jobs(db, batch_id) -> dict[str, dict]   # step -> row; unexpired AND not closed only
def closed_jobs(db, batch_id) -> set[str]        # steps whose job row has closed_at set

# photosearch/batch_state.py  (Task 3)
WORKER_PASSES = ("clip","faces","quality","aesthetics","describe",
                 "category-visual","category-content","keywords","verify")
NAS_STEPS     = ("stacking","normalize_aesthetics","match_faces","resolve_dups","warm_crops")
DESKTOP_STEPS = ("rank_measure",)
STEP_ORDER    = ("ingest",) + WORKER_PASSES + NAS_STEPS + DESKTOP_STEPS
DEPENDS_ON    = {"category-content":"describe","keywords":"describe","verify":"describe",
                 "normalize_aesthetics":"aesthetics","match_faces":"faces",
                 "resolve_dups":"match_faces","warm_crops":"faces","rank_measure":"faces"}
STATES = ("completed","running","queued","needs_queue","waiting","blocked")
def batch_state(db, batch_id) -> dict
# -> {"batch": {...}, "sweep": {...}|None, "ready": bool, "next_action": str|None,
#     "steps": [{"step","kind"("ingest"|"worker"|"nas"|"desktop"),"state","total",
#                "eligible","done","remaining","failed","waiting_on"(str|None),
#                "detail"(str|None)}, ...]}   # in STEP_ORDER
```

---

## Task 1: Schema v30 + `ingest_batches` module

**Files:** `photosearch/db.py`, new `photosearch/ingest_batches.py`, new
`tests/test_ingest_batches.py`, `tests/test_db.py`.

Bump `SCHEMA_VERSION` to 30. In `_init_schema()` add, as additive
`CREATE TABLE IF NOT EXISTS` (no `ALTER`, no backfill), for both fresh and migrating DBs:

```sql
CREATE TABLE IF NOT EXISTS ingest_sweeps (
  run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
  started_at TEXT NOT NULL DEFAULT (datetime('now')), finished_at TEXT,
  heartbeat_at TEXT NOT NULL DEFAULT (datetime('now')),
  files_seen INTEGER NOT NULL DEFAULT 0, files_moved INTEGER NOT NULL DEFAULT 0,
  error TEXT);
CREATE TABLE IF NOT EXISTS ingest_batches (
  id INTEGER PRIMARY KEY AUTOINCREMENT, directory TEXT NOT NULL UNIQUE,
  source TEXT, run_id TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  photo_count INTEGER NOT NULL DEFAULT 0, ready_at TEXT, dismissed_at TEXT);
CREATE TABLE IF NOT EXISTS ingest_batch_jobs (
  batch_id INTEGER NOT NULL, step TEXT NOT NULL, job_kind TEXT NOT NULL,
  opened_at TEXT NOT NULL DEFAULT (datetime('now')), expires_at TEXT NOT NULL,
  closed_at TEXT, PRIMARY KEY (batch_id, step));
CREATE INDEX IF NOT EXISTS idx_ingest_batches_created ON ingest_batches(created_at);
```

Implement every function in the "Frozen interfaces" block for `ingest_batches.py`. Rules:

- `register_batch` normalizes `directory` exactly as `db.get_directory_photo_ids` does
  (strip, drop leading `./`, re-root an absolute path under `db.photo_root`, strip `/`).
  It sets `photo_count = COUNT(*) FROM photos WHERE folder = ?`. It **raises `ValueError`**
  if that count is 0 (guards hand-made subfolders and typos).
- **Re-open rule (owner requirement):** calling `register_batch` for an existing directory
  updates `photo_count` + `updated_at`, and if the count **grew** clears `ready_at` and
  `dismissed_at`. If the count did not grow, `ready_at`/`dismissed_at` are left alone.
  `source`/`run_id` are overwritten only when the new value is not None.
- `get_active_sweep` adds a computed `stalled: bool` = status is `moving` and
  `heartbeat_at` is older than `STALL_SECONDS`; and `files_per_min: float` computed from
  `files_moved` over `(heartbeat_at - started_at)` (0.0 when the span is < 1s).
- `open_job` upserts (re-opening clears `closed_at`, resets `opened_at`/`expires_at`).
  `open_jobs` returns only rows with `closed_at IS NULL AND expires_at > datetime('now')`.
- All writes `db.conn.commit()`.

**Tests:** v29→v30 migration is additive and idempotent (open a DB, force
`schema_info.version=29` after dropping the three tables, reopen, assert tables exist and
version is 30 — mirror an existing migration test in `tests/test_db.py`); upsert
idempotency; re-open on growth clears `ready_at`+`dismissed_at`; no growth preserves them;
`ValueError` on an empty directory; directory spellings (`./x/`, absolute) normalize to
one row; job open/close/expiry; `stalled` + `files_per_min`.

## Task 2: Ingest writes the sweep heartbeat and registers batches

**Files:** `photosearch/ingest.py`, `cli.py` (`ingest_incoming_cmd`), `tests/test_ingest.py`.

In `ingest_incoming()`:

- Lazily `start_sweep(db)` on the **first file actually moved or archived** (a sweep that
  finds nothing creates no row). Skip entirely when `dry_run`.
- `heartbeat_sweep` throttled to at most once per second (`time.monotonic()`), counting
  `files_seen` (every file scanned) and `files_moved` (imported + companions_moved).
  Always emit a final heartbeat when the loop ends.
- Wrap the sweep loop so an exception sets `set_sweep_status(run_id, "failed", error=str(exc))`
  and re-raises — a crash must never leave a phantom `moving` row.
- Return `run_id` in the result dict as `result["run_id"]` (None when no sweep was created).
  On normal completion leave status `moving`; the CLI advances it.

In `cli.py:ingest_incoming_cmd`, after `ingest_incoming()` returns a `run_id`:
`set_sweep_status(..., "indexing")` before the per-`new_dir` index loop; after each dir's
`index_directory()` call `register_batch(db, new_dir, source=<that dir's source label>,
run_id=run_id)` (catch `ValueError` and print a warning — never fail the sweep over it);
finally `set_sweep_status(..., "registered")`. With `--no-index`, still register each
`new_dir` only if it already has photo rows (the `ValueError` path covers the rest), then
set `registered`. Wrap indexing so an exception sets `failed`.

**Tests** (extend `tests/test_ingest.py`; `extract_exif` is already monkeypatched there):
sweep row created with correct counters after a real move; **no** sweep row on an empty
`_incoming`; none on `dry_run`; an exception raised mid-loop (monkeypatch `shutil.move` or
`file_hash` to raise on the 2nd file) leaves status `failed`; `result["run_id"]` present.

## Task 3: `batch_state.py` — the state derivation

**Files:** new `photosearch/batch_state.py`, new `tests/test_batch_state.py`.

Implement the `batch_state.py` block of "Frozen interfaces". `ids = batch_photo_ids(...)`;
`total = len(ids)`. For each **worker** pass compute:

- `remaining` = `worker_api._count_scoped(db, pass, ids)` — reuse it, do not reimplement.
- `failed` = photos in `ids` with a `worker_processed` row for that pass whose
  `attempts >= db.MAX_PROCESS_ATTEMPTS` **and** whose output is still missing. Read
  `db.count_unprocessed_photos` to learn each pass's "output missing" predicate and reuse
  the same column tests. `clip` has no attempts ledger → `failed = 0`.
- `eligible` = `total`, except `category-content` / `keywords` / `verify`, where it is the
  count of `ids` with `description IS NOT NULL AND description != ''`.
- `done` = `eligible - remaining - failed`, floored at 0.

Worker-pass state, first match wins:
1. `running` — an unexpired `worker_claims` row with that `pass_type` whose JSON
   `photo_ids` intersects `ids` (use a set; the claims table is tiny).
2. `queued` — `open_jobs(db, batch_id)` has this step.
3. `completed` — `total > 0 and done == total`.
4. `waiting` — `eligible < total` and the `DEPENDS_ON` step is not `completed`;
   set `waiting_on`.
5. `blocked` — `remaining == 0 and failed > 0`.
6. `needs_queue` — otherwise.

Two tests are mandatory because they are the bugs this module exists to prevent:
**(a)** before any description exists, `keywords`/`category-content`/`verify` must read
`waiting` with `waiting_on == "describe"` — **never `completed`**; **(b)** a pass whose
every remaining photo has exhausted its attempts must read `blocked` — **never `completed`**.

The `ingest` step: `completed` if the batch exists and no active sweep references its
`run_id`; `running` while that sweep is `moving`/`indexing` (`detail` = "stalled" when
`stalled`); `blocked` if that sweep is `failed`.

NAS/desktop steps — cheap predicates, **no filesystem access**:
- `stacking`: `completed` when no batch photo with `date_taken IS NOT NULL` lacks a
  `stack_members` row **or** a closed job row exists for it (a shoot with no bursts has
  legitimately nothing to stack — so a *closed* `ingest_batch_jobs` row counts as done;
  use `ingest_batches.closed_jobs` for this).
- `normalize_aesthetics`: `remaining` = batch photos with `aes_overall IS NOT NULL AND
  aes_overall_pct IS NULL`; `waiting` on `aesthetics` until that pass is `completed`.
- `match_faces`, `resolve_dups`, `warm_crops`, `rank_measure`: `completed` iff a closed
  job row exists; `waiting` until their `DEPENDS_ON` step is `completed`; else
  `queued` (open job) or `needs_queue`.

`ready` = every step in `WORKER_PASSES + NAS_STEPS` is `completed` (`rank_measure` is
optional and does not gate `ready`). `next_action`: `"wait_ingest"` | `"launch_fleet"`
(any worker pass `needs_queue`) | `"advance_nas"` (any NAS step `needs_queue`) |
`"review_blocked"` | `"wait"` (only running/queued/waiting left) | `None` when ready.

**Tests:** one per state for a worker pass; (a) and (b) above; running via an intersecting
claim and *not* via a non-intersecting one; an expired claim is ignored; `ready` +
`next_action` transitions; the shape (every `STEP_ORDER` step present, in order).

## Task 4: Cheap status API

**Files:** new `photosearch/batch_api.py`, `photosearch/web.py` (mount the router next to
the others), new `tests/test_batch_api.py`.

- `GET /api/batches` → `{"sweep": get_active_sweep()|None, "batches": list_batches()}`.
  Pure SQL, **no derivation**. `?include_dismissed=1` supported.
- `GET /api/batches/{id}` → `batch_state(db, id)` plus `computed_at` (ISO) and
  `stale: bool`; 404 for an unknown id.
- `POST /api/batches/{id}/dismiss`, `POST /api/batches/{id}/ready`.
- `POST /api/batches/register` body `{directory, source?}` → `register_batch` (400 on
  `ValueError`). Lets an existing folder be adopted without re-ingesting.

**The cache is the point of this task.** Module-level `_memo: dict[int, tuple[float, dict]]`
and one `threading.Lock`. On `GET /api/batches/{id}`: if a memo entry is younger than
`_TTL_SECONDS = 30`, return it (`stale: False`). Otherwise try
`lock.acquire(blocking=False)`: on success recompute, store, release in `finally`; **on
failure return the existing memo immediately with `stale: True`** (or, if there is no memo
yet, a minimal `{"batch": get_batch(...), "steps": [], "stale": True, "computing": True}`)
— a request must **never wait** on another request's derivation. This is the direct answer
to the 2026-09-19 incident, where stacked status polls filled all 40 request threads. Any
write endpoint above invalidates that batch's memo entry. Expose `_reset_cache()` for tests.

**Tests** (FastAPI `TestClient`, follow `tests/test_api.py` for app/DB fixture wiring):
two sequential GETs inside the TTL call `batch_state` once (monkeypatch + counter); a GET
while the lock is held by another thread returns promptly with `stale: True` and does not
call `batch_state`; dismiss invalidates; 404; register 400 on empty directory.

## Task 5: `/batches` page — the flow diagram

**Files:** new `frontend/dist/batches.html`, `photosearch/web.py` (route, served with
`Cache-Control: no-cache` like `/merges`), `frontend/dist/shared.js` (`navLinks` entry
`{ href: '/batches', label: 'Batches', id: 'batches' }`), new
`frontend/dist/batch-flow.js` + `frontend/__tests__/batch-flow.test.js`,
`frontend/dist/admin_maintenance.html` (a "Per-batch view →" link in the PipelineFunnel
header).

Put the **pure** presentation logic in `batch-flow.js` (browser global `PS.BatchFlow`
*and* CommonJS export, same dual pattern as `split-geometry.js`) so it is unit-testable:

- `STATE_META` — for each of the 6 states: `label` ("Completed", "Running", "Queued",
  "Needs to be queued", "Waiting", "Blocked"), a CSS class, and a glyph.
- `layout(steps)` → rows for the diagram: row 0 `ingest`; row 1 `clip`; row 2 the parallel
  group `faces, quality, aesthetics, describe, category-visual`; row 3 `category-content,
  keywords, verify` (under describe) ; row 4 the NAS steps; row 5 `rank_measure`.
  Unknown steps go in a trailing row rather than being dropped.
- `summarize(state)` → `{done, total, headline}` where headline is a human sentence for
  `next_action` ("Launch the worker fleet for 5 passes", "Ingest is running — 412 files
  moved, 38/min", "Stalled: no file moved for 6 min", "Ready to review", …).
- `stepCaption(step)` → e.g. `"1,143 / 1,373"`, `"waiting on describe"`, `"12 failed"`.

`batches.html`: `PS.SharedHeader({activePage:'batches'})`; a **sweep banner** when
`sweep` is non-null (status, files moved, files/min, red when `stalled`); a batch list
(directory, photo count, created, a ready/dismissed pill; newest first; `?batch=<id>` in
the URL selects one, default the newest); the **flow diagram** for the selected batch —
boxes in the rows above, connected top-to-bottom, each box coloured by state with its
caption, plus a legend of all six states; the `summarize` headline above it; a `stale`
indicator ("updating…") when the API says so; Dismiss / Mark-ready buttons; and links
"Review faces →" `/faces?date_from=D&date_to=D` and "Search this day →" when the
directory's basename starts with a `YYYY-MM-DD` date. Poll `GET /api/batches` and the
selected batch with **`PS.poll`** at 10 s. Dark theme: reuse the CSS variables
(`--surface2`, `--border`, …) and card styles from `admin_maintenance.html`. Must be
usable at phone width. Reserve a disabled "Advance batch" button slot labelled
"(coming in the next step)" — Task 6 wires it.

**Tests:** `batch-flow.test.js` — `layout` places every `STEP_ORDER` step exactly once and
keeps unknown steps; `summarize` for each `next_action` incl. stalled; `stepCaption` for
completed / waiting / blocked / needs_queue. `node scripts/check-frontend-refs.js` passes.

## Task 6: `batch-advance` — the one action

**Files:** new `photosearch/batch_advance.py`, `cli.py`, `photosearch/admin_api.py`,
`frontend/dist/batches.html`, new `tests/test_batch_advance.py`.

`advance_nas_steps(db, batch_id, *, apply=False, on_progress=None, should_abort=None,
runners=None) -> dict`: for each step in `NAS_STEPS`, in order, whose derived state is
`needs_queue` (skip `completed`; **stop** at the first `waiting`/`blocked`): `open_job`,
run it, `close_job` on success (leave the job open to expire on failure, and report the
error). `runners` maps step → callable for tests; defaults:

- `stacking` → `stacking.run_stacking` scoped by the batch's `photo_ids`. **Never pass
  `clear=True`** — scoped stacking with clear has twice wiped the whole library's stacks.
  Read `run_stacking`'s signature and assert in a test that `clear` is falsy.
- `normalize_aesthetics` → `maintenance._stage_normalize_aesthetics`.
- `match_faces` → `faces.match_faces_to_persons` (strict) scoped by `photo_ids`.
  **Never the temporal matcher.**
- `resolve_dups` → `maintenance._stage_resolve_dups`.
- `warm_crops` → the existing warm-face-crops implementation, scoped to the batch's date.

Dry-run (`apply=False`, the default) reports what *would* run and writes nothing, including
no job rows. `should_abort` is checked between steps → `InterruptedError`.

CLI `photosearch batch-advance --batch N [--apply]` with `--db` + `envvar="PHOTOSEARCH_DB"`.
SSE `POST /api/admin/batch-advance` body `{batch_id, apply}`: follow
`admin_ingest_incoming` — throwaway sibling container, shares `_ingest_lock` (409 when
held), terminal `done`/`fatal`/`cancelled`. In replica mode (`PHOTOSEARCH_NAS_URL` set)
proxy the stream to the NAS instead of running locally.

`POST /api/admin/batch-launch-fleet` body `{batch_id, count?}`: only valid where the fleet
can launch (return 400 with a clear message when `run-workers.sh` is missing). Computes the
worker passes currently `needs_queue`, orders them by `WORKER_PASSES`, calls the existing
fleet-start logic scoped to the batch's **directory** (add an optional `directory` field to
`WorkersStartRequest`, mutually exclusive with `collection`/`filters`, passed as `-d`; it
must be the absolute `/photos/...` form — never the photo root), `sequential=True`, and
`open_job(..., job_kind="fleet")` per pass on the **authoritative** DB (in replica mode,
POST to the NAS's new `POST /api/batches/{id}/jobs` `{steps:[...], job_kind}` endpoint —
add it to `batch_api.py`).

`batches.html`: replace the reserved slot with **Advance batch** — calls
`batch-launch-fleet` when `next_action == "launch_fleet"`, `batch-advance` (apply) when
`"advance_nas"`, streams SSE progress into an inline log using `PS.parseSSEChunk`, and is
disabled with an explanatory label for `wait` / `wait_ingest` / `review_blocked`.

**Tests:** fake `runners` — ordering; stops at first `waiting`; job rows opened then closed;
failure leaves the job open and surfaces the error; dry-run writes nothing; abort path;
`clear` never truthy; the launch endpoint picks only `needs_queue` passes in order and 400s
without `run-workers.sh`; `WorkersStartRequest` rejects `directory` + `filters` together.

## Task 7: Docs

**Files:** `CLAUDE.md`, `.claude/skills/photo-search/SKILL.md`,
`docs/plans/ingest-batch-readiness.md`.

`CLAUDE.md`: a new `## Ingest batches (/batches)` section in the house style of the M28/M31
sections — what a batch is and why it is folder-derived (incl. the re-open rule); the six
states and the two ways `count_unprocessed == 0` lies; where "queued" comes from and why not
`fleet-status`; the non-blocking-lock cache and the incident it answers; what is
deliberately **not** in "ready" (temporal matching, recluster) and why; the endpoints and
the CLI; "deploy the NAS before the desktop". Update the schema-version mentions (29 → 30)
and the "Planned milestones" entry to "shipped". `SKILL.md`: API endpoints list, schema
table, `/batches` page note. Set the design doc's status header to shipped. Verify every
file, function and endpoint you name actually exists (`grep` for each) before writing it.

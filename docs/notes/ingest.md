# Ingest — history and traps

Phone/SD ingest, `refile-unknown-camera`, and the `/batches` readiness model. Condensed rules: `.claude/rules/ingest.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 1821) on 2026-10-19 -->
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

**`unknown-camera` is a fallback label, not a fact** — `ingest._file_suffix`
overrides it per file with the file's own EXIF model when that passes the
camera-model shape check. The Windows importer reads the model through the shell
property store, which has no codec for a new body's RAWs: on 2026-09-19 every
ILCE-7RM6 `.ARW` arrived as `unknown-camera` and was filed away from its JPEG.
Only the fallback defers to EXIF — a person's label (`nicole`) or a named model
dir is never overridden.

The SD-card import path feeds this: `scripts/windows-import/import-photos-safe.ps1`
(run on the Windows workstation from a checkout of this repo; its NAS share +
SSH target live in a git-ignored `import-config.local.ps1` — see the README there) reads each photo's/RAW's EXIF camera model and pushes all
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
   `photosearch index <dir>`. `--no-clip` goes further: the index pass only
   **registers** rows (hash/EXIF/geocode) and leaves CLIP to the fleet's `clip`
   pass — the SD-card importer uses it. **Never `--no-index` for that**:
   ingest itself inserts no rows, so `--no-index` leaves moved photos with no
   DB row at all — unclaimable by the fleet, unsearchable, and invisible to the
   next sweep's hash dedup. Faces / quality / describe / tags get picked
   up by the existing worker fleet on its next claim — no special wiring.

`.processed/` is safe to `rm -rf` at any time; it only holds deduped
sources kept for audit. Module: `photosearch/ingest.py`. Tests:
`tests/test_ingest.py` (12 cases — routing, dedup, archive layout,
mtime fallback, undated bucket, filename-collision suffix, HEIC,
AppleDouble skip, hidden-source-dir skip).

<!-- moved verbatim from CLAUDE.md (line 1897) on 2026-10-19 -->
### Repairing the folders already misfiled — `refile-unknown-camera`

`_file_suffix` only fixes files ingested *from now on*. The ones already on
disk (25 folders `2026/2026-MM-DD_unknown-camera`, ~9,600 files, 2026-06-28 →
2026-09-19, overwhelmingly `.ARW`) are moved onto the right body by
`photosearch refile-unknown-camera`. Module `photosearch/refile.py`
(`refile_unknown_camera`, `undo_refile`, `render_report`); tests
`tests/test_refile.py`.

**The model comes only from the file's OWN EXIF** — via
`ingest._file_suffix("unknown-camera", "unknown-camera", meta)`, imported, never
re-implemented, so the destination is exactly what ingest would choose today.
Never from the date, never from a sibling folder, never from a same-stem JPEG:
on several of these days **two bodies were in use** (`…_ILCE-7M4` *and*
`…_ILCE-7RM6` folders exist for the same date), so a date-based guess silently
mixes two cameras' files — an error nobody would notice afterwards. A file whose
EXIF names no usable model (video, unreadable RAW) is **left exactly where it
is** and counted `no_model`. `--infer-from-sibling` is the one opt-in exception
and is gated hard: the same-stem sibling must sit in the ONLY non-unknown-camera
folder for that date.

**The date comes only from the source folder's name.** Ingest already dated
these files; re-deriving it from EXIF could split a shoot across a
midnight/timezone edge.

Collisions at the destination, the dangerous part — **nothing is ever
overwritten, renamed, or deleted**: same name + identical hash → `duplicate_left`
(source kept in place; clear those by hand); same name + different bytes →
`conflict`, both paths in the audit, source kept. Hashing happens **only** on a
name collision **during an apply** — whole-file reads across 9,600 RAWs on a
spinning NAS disk are the thing this tool must not do, and repeated Sony DSC
names make collisions the expected case. A **dry run never hashes**: it reports
`would_collide` with both sizes (differing sizes are certainly a conflict; equal
sizes need a hash at apply time). `extract_exif` reads the header only
(`exifread.process_file(..., details=False)`, `photosearch/exif.py:55`).

**The move is `os.link` + `unlink`, and that is not a style choice.** Both
`os.rename` and `shutil.copy2` **silently overwrite** the destination, and the
check-then-move window is real, not theoretical: today's date is one of the 25
target folders, the nightly `ingest-incoming` cron and the SD-card importer
write into `YYYY-MM-DD_<Model>/` while this runs, Sony DSC filenames repeat
across cards, and `ingest._unique_target_path` has its own race on the other
side. A hardlink fails with `EEXIST` instead of clobbering, and preserves
mtime/permissions for free. `EXDEV`/`EPERM`/`EMLINK`/`ENOTSUP` (and only those)
fall back to a copy that opens the destination `O_CREAT|O_EXCL|O_WRONLY` — never
`copy2` onto a path — fsyncs the file *and* its directory before unlinking the
source, and removes its own partial output on any failure so an ENOSPC can't
leave a truncated file among real photos. An `EEXIST` at move time is handled
exactly like a pre-detected collision, never retried under another name. A crash
between link and unlink leaves both names on one inode, which the next run reads
as an identical-hash `duplicate_left`.

**The DB gate looks rows up by exact `filepath`, never by `folder`.** `folder` is
derived, and `relocate-into-year-dirs` (cli.py) and `db.remap_paths` (db.py)
used to `UPDATE photos SET filepath = ?` and leave it stale. **Fixed:** every
filepath writer now goes through **`db.set_photo_filepath(conn, id, path,
raw_filepath=...)`**, which re-derives `folder` in the same UPDATE, and
`PhotoDB.update_photo(filepath=...)` does the same — so never write a bare
`UPDATE photos SET filepath`. But a DB that ran the old writers can still hold
stale values (heal with `photosearch backfill-folders --force --apply`), so the
gate keeps trusting only `filepath`: a stale `folder` would hide the row, the
file would look unindexed, get moved by default, and orphan the row where
`_heal_folder` could not see it either. `_db_links` therefore does a prefix
**range scan** on the UNIQUE `filepath` index (`>= 'dir/'`, `< 'dir0'`).

That moves the risk from the column to the **spelling**, so two more guards:

- **A pre-flight refusal** (`_preflight_paths`, both dry run and apply, **no
  override**). Any `filepath`/`raw_filepath` stored absolute, `./`-prefixed,
  backslashed, or with a doubled slash is counted, sampled, and the run stops:
  a string lookup that cannot be trusted is not a lookup. (Checked read-only on
  the live DB 2026-09-19: **zero** such rows.)
- **A photo-root identity check** (`_preflight_root`) — the hole the spelling
  pre-flight **cannot see**. `relative_filepath` swallows a mismatch: when the
  path is not under `photo_root`, `Path.relative_to` raises and it returns the
  **absolute** path instead. So a DB whose stored `photo_root` is a symlink,
  another mount spelling, or simply a different directory leaves every stored
  path looking perfectly canonical while **every** link lookup misses —
  measured: stored `2026/2026-06-19_unknown-camera/DSC01.JPG`, pre-flight green,
  `would_move=1 skipped_indexed=0`, an indexed file read as unindexed. So when
  the DB **does** store a root, the run refuses unless
  `Path(stored).resolve() == root`. `_ReadOnlyDB` keeps the DB's own value
  separately (`db_photo_root`), because an explicit `--photo-root` would
  otherwise mask exactly the mismatch being tested for.
- **A mandatory round-trip proof** (`_preflight_roundtrip`) — and *this*, not
  where the root is configured, is what makes the gate trustworthy. Up to 25
  rows spread across the id range (lowest, highest, middle — not the first row
  alone; different eras of the indexer sit at different ends) are each fed to
  the **production** lookup `_resolve_link`, the exact call the gate makes, and
  must come back. At least one sampled row must also exist on disk, which is
  what proves `root` is where the library actually lives. Any failure refuses
  with "path mapping between the DB and the photo root is broken", naming the
  root and where it came from. An **empty** `photos` table returns 0 instead of
  refusing: with no rows the gate cannot produce a false negative.

  "At least one of 25", not "all", is deliberate: `_heal_folder` exists
  *because* rows whose file has moved are a normal state, so one stale row must
  not lock the operator out.

**The NAS stores no `photo_root` at all** — `schema_info` holds only
`('version', N)` and the container supplies `PHOTO_ROOT=/photos`; stored paths
are canonical relative ones like `2026/2026-09-19_ILCE-7RM6/DSC09999.JPG`. An
earlier revision refused on a missing stored root and so **refused to run on
the only system it exists for**. `_effective_root` now mirrors
`PhotoDB.__init__`'s precedence — `--photo-root` > `PHOTO_ROOT` env > the DB's
stored value — and refuses only when none of the three exists. The report
header states which one was used, e.g.
`photo root: /photos (from PHOTO_ROOT env; DB stores none) — mapping verified
on 25 rows`.
- **A per-file re-query** (`_resolve_link`) immediately before each move, on
  both the relative and absolute spellings, against the UNIQUE index — so a row
  inserted *since* the up-front scan still blocks the move. `raw_filepath` has
  **no index** (confirmed in `db.py`), so it is scanned once up front and keyed
  by absolute path; the per-file check hits that set instead.

**One inode under two names is finished, not re-hashed.** A kill between
`os.link` and `os.unlink` leaves exactly that. `_same_inode` (`st_dev` +
`st_ino`, two stats, no hashing) proves identity rather than equal content, so
the source is unlinked and the move recorded `moved` with
`completed_interrupted_link`. Classifying it `duplicate_left` instead would
strand the source forever, keep the folder undeletable, and whole-file hash
both names on every future run.

**That unlink requires `st_nlink >= 2`**, re-read immediately before it along
with a second `_same_inode` confirmation. `nlink >= 2` is outright proof that
another name still holds the bytes, so the unlink cannot destroy data; taking it
late closes the check→unlink window; and it defeats a synthetic-inode false
positive, since SMB/NFS/FUSE can repeat `st_ino` while reporting `nlink` 1. If
either check fails the source is **not** unlinked — the file falls through to
ordinary collision handling and the audit records `unlink_refused` with the
reason.

Other behaviour worth knowing:

- **Only the top level** of each folder is processed. A nested subdirectory is
  left untouched and reported — flattening it would invent collisions. Such a
  folder therefore never ends up empty and is never removed. Same for leftovers:
  the folder is `rmdir`'d (never `rmtree`) only when literally nothing remains,
  and the report names what is still there (`.DS_Store`, `@eaDir`, …).
- **Symlinks are never moved or followed.** A source link is counted
  `skipped_symlink`; the destination check uses `os.path.lexists`, because a
  **dangling** symlink is a taken name that `exists()` reports as free.
- **Indexed photos are skipped by default** (`skipped_indexed`). RAW/video
  companions have no `photos` row (`ingest.py` gates the DB path on `is_photo =
  ext in INGEST_EXTENSIONS`), but a JPEG in one of these folders may, and a RAW
  may be some JPEG's `raw_filepath`. `--include-indexed` moves them and updates
  `filepath` + `folder` (+ `raw_filepath` refs) — thumbnails/previews are keyed
  by photo **id**, so they survive a path change untouched.
- **Move first, write the row second.** They cannot be one transaction. This
  order leaves a row naming a gone file (recoverable) rather than a row naming a
  file that was never created. A re-run heals it: `_heal_folder` finds rows in
  the folder whose file is missing and repoints only on a **`file_hash` match**
  against that date's sibling folders. No hash to check against — a row with a
  NULL `file_hash`, or any stale `raw_filepath` (which stores no hash at all, so
  "only one candidate" could attach the *other* body's identically-named RAW) —
  is counted `heal_unverifiable` and printed for hand repair. A dry run never
  hashes; it just reports how many rows would need healing.
- `_undated/unknown-camera` is **skipped with a message** — no date, so no
  destination folder can be computed. Sort it by hand.
- **A dry run opens the DB read-only** (`mode=ro`) and fails if the file is
  missing, so a mistyped `--db` cannot create a stub or run migrations.
- **`--apply` takes ingest's `_sweep_lock`** and aborts with
  `IngestAlreadyRunning` if a sweep holds it. An `--audit` path resolving inside
  the photo root is refused (it would count as a leftover and block `rmdir`) —
  put it on `/data`.
- Idempotent and resumable: a second run after a full apply does nothing.

The audit CSV is the undo, so `--apply` refuses without `--audit`. Each move
writes **two** rows — an `intent` before the syscall and a `moved` after it —
each `fsync`ed, so neither crash window loses the record. `--undo` resolves an
unconfirmed `intent` by looking at the disk: source still present means the move
never started (`never_started`); destination present at the recorded size with
the source gone means it completed and is undone.

**Operator procedure** (`/data` is the writable persistent volume):

1. **Pause the nightly ingest cron and hold the SD-card importer** for the
   window. Today's-date folder genuinely contends; the lock enforces it too.
2. Full **dry run** to a file, then check it: **the pre-flight passed** (it
   refuses outright otherwise) and the header names the expected root and
   source (`/photos`, from `PHOTO_ROOT` env) with the mapping verified;
   files ≈ 9,600; `skipped_indexed` matches
   expectation — on *this* library that is **0**, verified read-only on
   2026-09-19: no `photos` rows and no `raw_filepath` refs live under any
   `*_unknown-camera` folder, since the misfiled files are all RAW companions.
   (It is the pre-flight, not a non-zero count, that proves the gate works.)
   `would_collide` and errors 0; `no_model` small and explained; **both** bodies
   appear on the known two-body dates, and never a model that isn't yours.
3. **Smallest folder first** with `--apply --audit /data/…`.
4. **Verify on disk**: listing, sizes, mtimes, an `exiftool`/`stat` spot-check.
5. **Rehearse the undo** on that folder, confirm the tree is back, then redo it.
6. Then the rest, **one audit file per batch**.
7. A final dry run should report nothing left.
8. Leave `--include-indexed` and `--infer-from-sibling` **off** for the first pass.

```bash
$DC run --rm photosearch refile-unknown-camera > /data/refile-dryrun.txt     # 2
$DC run --rm photosearch refile-unknown-camera \
    --only 2026-06-28_unknown-camera --apply --audit /data/refile-0628.csv   # 3
$DC run --rm photosearch refile-unknown-camera --undo /data/refile-0628.csv --apply   # 5
$DC run --rm photosearch refile-unknown-camera --apply --audit /data/refile-all.csv   # 6
```

Undo restores a row only when the destination is still the file this tool put
there — **size**, plus **hash for collision rows** (the only rows that carry one;
recording a digest for every move would mean reading ~576 GB) — and the source
path is free. Anything else is `refused` and left alone. Any DB row the tool
repointed is pointed back too, and the restore uses the same non-overwriting
primitive as the forward move. An undo **dry run** opens the DB read-only, like
the forward one.

Known and deliberately left alone: `EACCES`/`ENOSYS` are not in the hardlink
fallback list, so they are a hard error and no move happens (fail-safe); the
copy fallback preserves mode and mtime but **not owner/group** — irrelevant on
the NAS, where `/photos` and `/data` share a device and the link path is always
taken; and an `intent` row carries no hash, so undoing a crash-window move
verifies size only.

Cron entry on the NAS:

```cron
0 4 * * * cd /volume1/docker/photosearch && flock -n /tmp/photo-ingest.lock docker compose -f docker-compose.nas.yml run --rm photosearch ingest-incoming --no-colors >> /var/log/photo-ingest.log 2>&1
```

`--no-colors` keeps the daily sweep CLIP-only: color extraction is the slow,
memory-heavy part (~4 s/photo) and a first-time backfill once OOM-cut a run
mid-index. The move phase always completes first (files land safely on disk),
so a killed index pass only leaves photos un-embedded — re-run
`photosearch index /photos` (idempotent) to backfill CLIP, and a plain
`photosearch index <dir>` later for colors (colors are **not** a worker pass).

<!-- moved verbatim from CLAUDE.md (line 3850) on 2026-10-19 -->
## Ingest batches (`/batches`)

One-glance per-shoot readiness, replacing "SSH in and eyeball `/api/stats`"
with a page that names the next action. Built after the 2026-09-19 incident
where an unrelated ~50 MB/s SMB backup starved the NAS's disks and the
**status page itself made it worse** — `PipelineFunnel`'s `setInterval`
polling stacked requests faster than the starved server could answer them,
filling all 40 request threads. `/batches` (`docs/plans/ingest-batch-readiness.md`)
is built against a repeat: every read behind it is a fixed handful of indexed
`COUNT` queries, behind a cache that never lets two requests wait on the same
derivation.

**A batch is one dated folder — `ingest_batches.directory == photos.folder`.**
Membership is never materialized: `ingest_batches.batch_photo_ids` is always
`SELECT id FROM photos WHERE folder = ?` (`photosearch/ingest_batches.py`), so
a batch row holds only identity, lifecycle timestamps, sweep progress, and job
*intent* — never derived state. `register_batch` upserts on `directory` and
**re-opens an existing batch**: if a later sweep lands more photos in today's
folder (the 03:00 phone cron, a second card dump), `photo_count` grows and
`ready_at`/`dismissed_at` are cleared, so the newly-arrived photos' steps flip
back to `needs_queue` while everything already `completed` (missing-only by
construction) stays that way. `source`/`run_id` are only overwritten when the
caller passes a non-`None` value, so a plain re-scan never clobbers who
registered the batch. Registering a directory with zero matching `photos`
rows raises `ValueError` — guards a typo'd or hand-made subfolder against
becoming a phantom batch.

<!-- moved verbatim from CLAUDE.md (line 3877) on 2026-10-19 -->
### The six states, and the two ways `count_unprocessed == 0` lies

`photosearch/batch_state.py` reports exactly one of `completed / running /
queued / needs_queue / waiting / blocked` for every step — the 9 worker
passes (`clip`, `faces`, `quality`, `aesthetics`, `describe`,
`category-visual`, `category-content`, `keywords`, `verify`) and the 6 NAS
steps (`stacking`, `normalize_aesthetics`, `match_faces`, `resolve_dups`,
`warm_crops`, `rank_measure`). The module exists because
`db.count_unprocessed_photos` — the fleet's **claim predicate**, not a
progress bar — returns 0 in two situations that are not "done":

1. every remaining photo has `worker_processed.attempts >=
   MAX_PROCESS_ATTEMPTS` (3), so the claim path skips it forever — this reads
   `blocked`, never `completed`.
2. `category-content` / `keywords` / `verify` are gated on `description IS
   NOT NULL`, so before `describe` has run they count 0 unprocessed **and** 0
   eligible — this reads `waiting`, never `completed`.

So every step carries `total / eligible / done / remaining / failed`, and
`completed` is `done == total`, never `remaining == 0`. Every worker pass's
claim predicate now excludes exhausted photos, so `failed` and `remaining` are
disjoint: `done = eligible - remaining - failed` and `blocked` is `remaining
== 0 and failed > 0` (`_REMAINING_FILTERS_ATTEMPTS`, all `True`). `clip`,
`quality` and `verify` used to carry **no attempts filter** — `failed` sat
inside `remaining`, so they needed `done = eligible - remaining` and `blocked =
remaining == failed > 0` — until they joined the ledger (see "A failed
SERVER-side write must not burn a retry attempt" and "Non-image rows"). The
table stays: a future pass added without the cap must say `False` there, or
`done` double-subtracts `failed`.

**A poison `clip` photo now reads `blocked`.** It used to be the exception:
clip kept no ledger, so `_worker_step` hardcoded `failed = 0` and a
clip-unloadable photo (a ZIP-wrapped `.JPG` Live Photo) left the batch's `clip`
step at `remaining > 0` **indefinitely** — `next_action` stuck on
`launch_fleet`, never `ready`, never `blocked`. `_OUTPUT_MISSING["clip"]` is
now the embedding test and the step counts exhausted photos like any other.
Such rows should still be rare for new batches (`index.py:is_real_image()`
gates row creation at ingest); `purge-nonimage-photos` removes old ones, and
**Mark ready** remains the manual escape.

Worker-pass precedence is **running > completed > blocked > queued > waiting
> needs_queue** — `completed` deliberately outranks an open job row, because
nothing ever *closes* a worker pass's job row (the job-only NAS steps are the
mirror image: a *closed* row is their only proof of success).
Precedence used to put `queued` first; a pass the fleet had already finished
then read `queued` for the row's full 6h TTL, so the batch could never reach
`ready` and the launch button's "already running" 409 stayed armed long after
the fleet had exited.

<!-- moved verbatim from CLAUDE.md (line 3926) on 2026-10-19 -->
### `faces` is the one pass where "no output" means done

A photo with nobody facing the camera has **no `faces` rows after a perfectly
successful run**. The server now records that as terminal — attempts set
straight to `MAX_PROCESS_ATTEMPTS` on the first clean empty result (it used to
re-try it three times). Either way it ends exhausted with no rows, which the
claim predicate cannot tell from three failures. `batch_state._EMPTY_OUTPUT_IS_DONE` therefore counts an
exhausted no-face photo toward `done`, not `failed`, and reports the count in the
step's `detail` ("113 with no detectable face"), which the page shows on the
completed box.

This was found by the feature on its **first real batch**, within a minute of
deploying it (2026-09-19, 1,373 photos): `faces` read `blocked — 113 failed` and
held `match_faces` / `warm_crops` / `rank_measure` in `waiting`. Checked against
the live DB: all 113 sat at exactly `attempts=3` with no face rows — players
facing away, distant shots — while 3,696 faces had been found in the other 1,260.
Every per-task review and the whole-branch review missed it, because nothing in
the code says `worker_processed` for `faces` means *attempted*, not *failed*.
A genuinely corrupt file ends in the same place and is indistinguishable here;
it is rare, and the count is shown rather than hidden. Don't extend the rule to
other passes without the same evidence: for `describe` or `aesthetics`, no output
after three tries really is a failure.

<!-- moved verbatim from CLAUDE.md (line 3949) on 2026-10-19 -->
### Where "queued" comes from — never `/workers/fleet-status`

"Queued" is an open, unexpired `ingest_batch_jobs` row, full stop — never a
parse of `GET /workers/fleet-status` (which shells `run-workers.sh --status`,
which itself curls the heavier `/api/worker/status`, and is blind to a
hand-launched fleet or one running on the other machine). A job row's 6h TTL
is what reclaims it if a fleet dies mid-run. `warm_crops`, `match_faces`,
`resolve_dups` and `rank_measure` write no per-photo column this module can
count at all, so for them a **closed** job row is the *only* completion
evidence there is (`_job_only_step`) — which is why a failed or aborted NAS
step **deletes** its job row (`ingest_batches.delete_job`) rather than
closing it or leaving it open: closing it would falsely prove success, and
leaving it open would read `queued` and block a retry for the full TTL with
no recovery short of editing the table by hand.

**A growing batch invalidates that evidence.** A closed row proves the step
ran over *the photos that were in the batch at the time*. When a later sweep
lands more files in today's folder, `register_batch` clears
`ready_at`/`dismissed_at` and every **derived** step re-reads the new photos
as unfinished by construction — but the job-only four would have kept reading
`completed` although the arrivals have no matched faces, no warmed crops and
no sharpness measurement. So growth now also deletes those steps' **closed**
rows (`ingest_batches._clear_job_only_completion`, keyed on
`batch_state.JOB_ONLY_STEPS`); open rows are left to whoever owns them. One
more `Advance batch` fills the gap, cheaply — every one of those runners is
missing-only or resumable.

<!-- moved verbatim from CLAUDE.md (line 3976) on 2026-10-19 -->
### The non-blocking-lock cache — the incident it answers

`GET /api/batches/{id}` (`photosearch/batch_api.py`) memoizes `batch_state`
for 30s (`_TTL_SECONDS`) behind **one lock for the whole module**, acquired
non-blocking. A poll past the TTL that can't get the lock never waits — it
returns the existing memo with `stale: true`, or a `{computing: true}`
placeholder if nothing is memoized yet. This is the direct fix for
2026-09-19: one lock for every batch, not one per batch, mirrors the
incident's actual shape on purpose — "all 40 threads wedged," not "one
batch's derivation is slow." The memo key is `(db_path, batch_id)`, not a
bare `batch_id` — `sync-replica.sh` swaps the replica's DB file wholesale,
and every fresh test DB mints batch id 1, so a bare-id key could serve one
DB's cached state against a different DB for up to 30s. Every write
(`dismiss`/`ready`/`jobs`/`register`) invalidates its key through a
generation counter, not a bare cache-pop — an in-flight recompute that
started before the write must not resurrect the pre-write snapshot under a
fresh timestamp once it finishes (race documented in the module docstring).

<!-- moved verbatim from CLAUDE.md (line 3994) on 2026-10-19 -->
### `rank_measure` is a NAS step, and it is OPTIONAL

It was originally labelled **desktop-only**, on the theory that it is heavy.
Heavy it is — `scripts/rank_shoot.py --measure` decodes every photo at **full
native resolution** to take the Laplacian variance of each face crop (1,260
photos with faces ≈ 10 min on the N100) — but it is heavy **where the pixels
are**: only the NAS holds the originals, the desktop replica has the DB and
the thumbnails and no files at all. As a desktop step it had **no runner
anywhere** — not in `batch_advance`, not in the fleet launcher — and its state
derived from an `ingest_batch_jobs` row nothing ever wrote, so the box read
"Needs to be queued 0 / 1,373" forever. It is now the last entry in
`NAS_STEPS` with a runner (`batch_advance._run_rank_measure`), so
**`Advance batch` runs it**.

Two things keep that from changing the rest of the flow:

- **`OPTIONAL_STEPS`.** `ready` is "every step in `WORKER_PASSES +
  NAS_STEPS`", so moving `rank_measure` in would silently have made it a
  readiness gate; `ready` subtracts `OPTIONAL_STEPS` explicitly.
- **`next_action` still offers it.** `if ready: return None` would have left
  it exactly as dead as before — a box that needs queueing and no button that
  queues it. So `advance_nas` *is* returned when a batch is otherwise ready
  and only an optional step is runnable, and the page says "Ready to review —
  optional: measure sharpness for ranking" with an enabled "Advance batch — 1
  optional step". It is offered **only** from that state: a blocked batch, or
  one with a pass still in flight, has something more important to say.

<!-- moved verbatim from CLAUDE.md (line 4021) on 2026-10-19 -->
### What's deliberately not in "ready" — and why

**A waiting step is deferred, not a stop** (2026-09-26). `batch-advance` used
to halt at the first `waiting` step — and `normalize_aesthetics` (waiting on the
slow VLM `aesthetics` pass) sits before `match_faces`, so face matching could
not run until aesthetics finished even though it needs only `faces`. Now a
waiting/blocked step is recorded `deferred` and later steps whose own
dependency is satisfied still run; dependents of a deferred step derive as
waiting on it and are deferred too. A step that FAILS still stops the run.
`stopped_at` still names the first deferred step. Run `Advance batch` once
after clip+faces (stacking, face matching, crops) and again after aesthetics.

`ready` is `all(step == completed for step in WORKER_PASSES + NAS_STEPS if
step not in OPTIONAL_STEPS)`. `batch-advance`
(`photosearch/batch_advance.py`) runs the NAS steps in order and
deliberately leaves two things out of "ready":

- **Temporal face matching** (`match_faces_temporal`) — ~4% accurate on these
  shoots ("Bulk-undoing one person's bad labels" above); pouring it into a
  fresh batch would manufacture Bulk-unmatch work instead of saving any. Only
  `faces.match_faces_to_persons` (the **strict** matcher) runs, scoped to the
  batch's photo ids — not `maintenance._stage_match_faces`, which runs both.
  Unknown faces stay reviewable per day via the existing Unclustered bucket.
- **`recluster-faces`** — clears `ignored_clusters` and takes on the order of
  an hour; a per-batch action can't justify that cost or that blast radius.

Also load-bearing: `run_stacking` has **no `clear` flag** — the clear scope
is whatever restricted the run, so a call scoped by `photo_ids=<batch>` only
clears stacks overlapping that batch. The real hazard is an **unscoped** call
or an **empty** `photo_ids` list: `detect_stacks` treats `[]` as "no scope
given" and falls through to the whole library, which is how the library's
stacks have been wiped twice before. `_run_stacking` guards both — a batch
that has emptied out (a dedup prune, a purge, a clock retime rewriting
`folder`) short-circuits with `skipped: "empty batch"` before reaching
`detect_stacks` at all. `warm_crops` is scoped by the batch's `photo_ids`,
not its calendar day — two folders can share a date (a phone sync and a card
dump landing the same day), and a photo with no `date_taken` would fall out
of a date scope while still belonging to the batch. `normalize_aesthetics`
and `resolve_dups` have no scope parameter at all — they're library-wide
maintenance stages reused as-is, so advancing one batch can touch
duplicate-person rows or aesthetic percentiles elsewhere in the library (both
reversible: `face_dedupe_undo` for the former, a re-run for the latter). In
replica mode, `warm_crops` runs on the **NAS** (where the photo files live),
so the desktop's own face-crop cache only warms on the next thumbnail sync —
the step can read `completed` on the desktop's `/batches` page while the
replica's face grids are still serving cold (slow, first-decode) crops.

**One click launches the whole worker pipeline.** `POST
/api/admin/batch-launch-fleet` doesn't just queue passes that are
`needs_queue` right now — `batch_state.fleet_launch_passes` also includes
every `waiting` pass whose dependency is already underway or itself in the
launch set (one forward walk, since `WORKER_PASSES` is already a valid
topological order). Without this, `category-content` / `keywords` / `verify`
are `waiting` on `describe` at click time and a second launch mid-run is
refused (409 — it would kill the running fleet), so they were never queued by
the button at all. Mirrored in JS as `PS.BatchFlow.fleetLaunchPasses`
(`frontend/dist/batch-flow.js`) so the Advance button's "N passes" count
matches what the server will actually launch — keep the two pinned to the
same cases, the way `split-geometry.js`'s worked example pins the JS and
Python geometry together. Launch itself is refused (409) if any worker pass
already reads `queued` or `running` for the batch, since a second
`run-workers.sh --name ui` kills and restarts the fleet — exactly what a
double click or a second open tab would otherwise trigger silently.

<!-- moved verbatim from CLAUDE.md (line 4085) on 2026-10-19 -->
### Endpoints and CLI

- `GET /api/batches` — pure SQL: the active sweep (if any) + the batch list.
- `GET /api/batches/{id}` — cached derived state (`batch_state`), carries
  `computed_at`/`stale`.
- `POST /api/batches/{id}/dismiss` / `POST /api/batches/{id}/ready` —
  lifecycle writes.
- `POST /api/batches/{id}/jobs` — record job intent from a *remote* launcher
  (the desktop, launching a fleet against the NAS in replica mode). Steps are
  validated against `STEP_ORDER` before anything is written
  (`batch_advance.open_step_job`) — `ingest_batch_jobs.step` is unchecked
  TEXT, so a typo'd write would read `needs_queue` forever with no error
  anywhere.
- `POST /api/batches/register` — manual registration; mirrors what
  `ingest-incoming` does automatically per new dated folder.
- `POST /api/admin/batch-advance` — SSE; runs the NAS steps. `photosearch
  batch-advance --batch N [--apply]` is the same thing from the CLI
  (`envvar="PHOTOSEARCH_DB"`); dry-run by default, writing nothing — not even
  a job row.
- `POST /api/admin/batch-launch-fleet` — launches `run-workers.sh --native`
  scoped to the batch's directory for exactly the passes it still needs.
- `/batches` (`frontend/dist/batches.html` + `batch-flow.js`) — sweep banner
  (moving/stalled/indexing, files/min), batch picker, the flow diagram, one
  Advance button. Polls both list and detail with `PS.poll`, never
  `setInterval`. Linked from `/admin/maintenance`.

<!-- moved verbatim from CLAUDE.md (line 4111) on 2026-10-19 -->
### Deploy the NAS before the desktop

Same shape as M28's `/mirror-fields` gotcha. In replica mode
(`PHOTOSEARCH_NAS_URL` set) **every** `/api/batches` route proxies to the NAS
with no local fallback — `batch_api._proxy` re-raises whatever status the NAS
returns as-is (a 502 names the host when it's unreachable at all). An old NAS
without this schema/code 404s every one of those calls, so the desktop
`/batches` page and `batch-launch-fleet` fail loudly instead of silently
reading a stale or absent local copy. That matters more here than in most
M26b write paths: a stale local read would let `batch-launch-fleet`'s
409-avoidance check pass on stale job rows and double-launch a fleet that's
already running on the authoritative side.

### Corrections found while condensing (2026-10-19)

Checked against the code while writing `.claude/rules/`; the text above is kept verbatim.

- Video is no longer move-only: since schema v35 `ingest.py` gives video a `photos` row (`media_type='video'`), dedups it against the DB and adds its folder to `new_dirs`. It is still never CLIP-indexed. RAW stays move-only.
- `_EMPTY_OUTPUT_IS_DONE` is Python-only; `batch-flow.js` only mentions it in a comment. `fleet_launch_passes` ↔ `PS.BatchFlow.fleetLaunchPasses` is the real mirror (`tests/test_batch_state.py`).

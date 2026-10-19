---
paths:
  - "photosearch/ingest*.py"
  - "photosearch/refile.py"
  - "photosearch/batch_*.py"
  - "frontend/dist/batches.html"
  - "frontend/dist/batch-flow.js"
---
# Ingest and batches — traps

Full history and measurements: docs/notes/ingest.md (cite by heading). Runbooks: .claude/skills/photo-search/reference/nas-operations.md.

- **Phone-photo daily ingest** — `_incoming/<source>/` is staging and stays in `index._BUILTIN_EXCLUDED_DIRS`. Imports are moved, not copied; DB-hash duplicates go to `.processed/`. A non-decodable image (`index.is_real_image`) gets no row.
- **RAW + video are "companions"** — RAW is move-only (no row) with destination dedup: same name + same hash goes to `.processed`. Video also gets a `photos` row (`media_type='video'`), dedups against the DB, and is never CLIP-indexed. Keep `is_photo = ext in INGEST_EXTENSIONS` as the gate for the photo path.
- **Never `--no-index` for the SD importer** — ingest itself inserts no rows, so `--no-index` leaves moved files unclaimable, unsearchable and invisible to hash dedup. Use `--no-clip` (register rows, leave CLIP to the fleet).
- **`unknown-camera` is a fallback label, not a fact** — `ingest._file_suffix` overrides only that fallback with the file's own EXIF model. Never override a person label or a named model dir.
- **Repairing the folders already misfiled — `refile-unknown-camera`** (`refile.py`):
  - Take the model ONLY from the file's own EXIF, through `ingest._file_suffix` (import it, don't copy it). Never from the date, a sibling folder or a same-stem JPEG: two bodies shoot the same day. No model → leave the file, count `no_model`.
  - Take the date ONLY from the source folder name.
  - Move with `os.link` + `unlink`. Never `os.rename` or `shutil.copy2`: both silently overwrite. The copy fallback only covers `EXDEV`/`EPERM`/`EMLINK`/`ENOTSUP` and opens `O_CREAT|O_EXCL`. Never overwrite, rename or delete on a collision (`duplicate_left` / `conflict`). Hash only on a name collision during an apply; a dry run never hashes.
  - Unlink a one-inode-two-names source only after `_same_inode` AND `st_nlink >= 2`, re-checked right before the unlink; otherwise record `unlink_refused`.
  - The DB gate looks rows up by exact `filepath` (`_db_links`, `_resolve_link` re-query per file), never by `folder`. Keep all three preflights: `_preflight_paths`, `_preflight_root`, `_preflight_roundtrip`. Root precedence is `_effective_root`: `--photo-root` > `PHOTO_ROOT` env > the DB's stored value (the NAS stores none).
  - Move first, write the row second. `_heal_folder` repoints only on a `file_hash` match, else `heal_unverifiable`.
  - The audit CSV is the undo: `--apply` requires `--audit` (outside the photo root), writes fsynced `intent` + `moved` rows, and takes ingest's `_sweep_lock` (`IngestAlreadyRunning`). A dry run opens the DB `mode=ro`. Never follow symlinks; check destinations with `os.path.lexists`; `rmdir` only, never `rmtree`.
- **Filepath writes** — always use `db.set_photo_filepath(...)` or `PhotoDB.update_photo(filepath=...)`, which re-derive `folder`. Never write a bare `UPDATE photos SET filepath`.
- **A batch is one dated folder** — `ingest_batches.directory == photos.folder`. Never materialize membership: `batch_photo_ids` is always a `folder = ?` query. `register_batch` upserts and re-opens a batch (clears `ready_at`/`dismissed_at`); a zero-photo directory raises `ValueError`.
- **The six states, and the two ways `count_unprocessed == 0` lies** — `db.count_unprocessed_photos` is a claim predicate, not progress. All exhausted reads `blocked`; no description yet reads `waiting`. `completed` means `done == total`, never `remaining == 0`. A new pass without the attempts cap must say `False` in `_REMAINING_FILTERS_ATTEMPTS`. Precedence: running > completed > blocked > queued > waiting > needs_queue.
- **`faces` is the one pass where "no output" means done** — `_EMPTY_OUTPUT_IS_DONE` counts an exhausted no-face photo as done and reports the count in `detail`. Don't extend it to other passes without evidence.
- **Where "queued" comes from — never `/workers/fleet-status`** — "queued" means an open, unexpired `ingest_batch_jobs` row, nothing else. For `JOB_ONLY_STEPS` a closed row is the only proof of success (`_job_only_step`), so a failed or aborted NAS step deletes its row (`ingest_batches.delete_job`); never close it or leave it open. When a batch grows, `_clear_job_only_completion` deletes those closed rows.
- **The non-blocking-lock cache** — `batch_api` uses one module `_lock`, acquired non-blocking: never wait, return a `stale` memo or `{computing: true}`. Key the memo on `(db_path, batch_id)`. Writes invalidate through the per-key generation counter, never a bare pop.
- **`rank_measure` is a NAS step, and it is OPTIONAL** — `ready` subtracts `OPTIONAL_STEPS`. `next_action` still returns `advance_nas` when only an optional step is runnable.
- **A waiting step is deferred, not a stop** — `batch-advance` marks waiting/blocked steps deferred and runs later steps whose own dependency is met. Only a FAILED step stops the run. Never add temporal matching or `recluster-faces` to a batch.
- **Stacking scope** — never call `run_stacking`/`detect_stacks` unscoped or with `photo_ids=[]`: `[]` means the whole library and has wiped all stacks. `_run_stacking` skips an `"empty batch"`. Scope `warm_crops` by `photo_ids`, not by date.
- **One click launches the whole worker pipeline** — `batch_state.fleet_launch_passes` is mirrored as `PS.BatchFlow.fleetLaunchPasses` in `batch-flow.js`; change both and keep their tests pinned. Launch is refused with 409 while any pass reads `queued`/`running`, because a second launch kills the fleet. Validate job steps against `STEP_ORDER` (`open_step_job`).
- **Poll `/batches` with `PS.poll`**, never `setInterval`.
- **Deploy the NAS before the desktop** — in replica mode every `/api/batches` route proxies through `batch_api._proxy` with no local fallback. Never add a local fallback: a stale local read can double-launch a fleet.

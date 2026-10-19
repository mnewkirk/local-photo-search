---
paths:
  - "photosearch/worker*.py"
  - "photosearch/sharpness*.py"
  - "photosearch/clip_embed.py"
  - "photosearch/quality.py"
  - "run-workers.sh"
  - "setup-wsl-gpu.sh"
---
# Workers — traps

Full history and measurements: docs/notes/workers.md, docs/notes/sharpness.md (cite by heading). Runbook: .claude/skills/photo-search/reference/workers.md.

## Fleet client and server

- **Worker passes** — `valid_passes` in `cli.py` is the source of truth: `clip, faces, quality, describe, verify, category-content, category-visual, keywords, aesthetics` plus `video-probe, video-proxy, video-scan, video-subjects, video-describe`. The `tags` pass is gone; `--passes tags` must keep erroring. A new pass must also get a `db.PASS_MEDIA` entry.
- **A failed SERVER-side write must not burn a retry attempt** — every `worker_api.submit_results` branch routes through `_SubmitOutcome`. Spend an attempt only when the result was persisted or the worker reported a real per-photo outcome. `'[]'` / no description is a result: done in one pass.
- **Transient lock only** — defer only when `_is_transient_db_error` says so (`sqlite3.OperationalError` mentioning "locked"/"busy"). Never widen it to a bare `except Exception`: that uncaps the pass and re-claims a poison photo every TTL forever. Every other failure spends the attempt and logs. One classifier serves per-row, batch-commit and `mark_processed`.
- **`MAX_PROCESS_ATTEMPTS`** (`db.py`, 3) — `mark_processed` UPSERT-increments `worker_processed.attempts`; every claim/count predicate excludes `attempts >= MAX_PROCESS_ATTEMPTS` (`_CLIP_NOT_EXHAUSTED`, `_QUALITY_NOT_EXHAUSTED`, `_VERIFY_NOT_EXHAUSTED`). A pass without this filter re-claims failures forever.
- **Worker-reported failures are a separate `failures` list** — `FailedResult{photo_id, error}`, built by `worker._submit_kwargs` from `_failure_row`s: one attempt, no column written, an `index_errors` row. Never fold failures into per-pass rows (old servers 422 or store them as real). Transient errors (`worker._TRANSIENT`) are omitted from the payload, never reported.
- **A clean empty `faces` result is TERMINAL** — `db.mark_processed(..., terminal=True)` jumps to the cap. Only safe because a detection *exception* is a failure row, never `faces: []`. Keep `index.py`'s in-process faces writers on the same contract.
- **`quality` and `verify` are on the ledger now** — `submit_results` marks both processed and `clear-pass` deletes their ledger rows. Don't let a pass silently drop a photo it could not score.
- **Deferred COMMIT** — `begin_batch` defers the commit, so SQLITE_BUSY surfaces at COMMIT: defer the batch (`status: "deferred"`, `deferred_photo_ids`, claim released). Don't claim "nothing landed"; `db.log_error` commits.
- **Deploy the server and the workers together** — an old worker sends `faces: []` on error and the new server retires the photo in one shot; an old server ignores `failures`.
- **`-d` gotcha** — `-d` must be a real subdirectory; `-d /photos` (the root) matches nothing, so the server answers 404 "No photos found in directory /photos" (`worker_api.py`) and the fleet does no work. To process the whole library, omit `-d` entirely. `get_directory_photo_ids` runs per claim, so it ranges over indexed `photos.folder` (`folder = p OR (folder >= 'p/' AND folder < 'p0')`). Never `filepath LIKE` (full scan) or `folder = ?` (drops subfolders). `tests/test_directory_scope.py` checks the plan uses `idx_photos_folder`.
- **Startup probe is `GET /api/health`** — no DB. Never probe `/api/stats` (full-library counts, the whole fleet at once). A 404 counts as reachable (older server). Single-result submitters use `WorkerClient(..., probe=False)`.
- **503 Retry-After** — `WorkerClient._request` sleeps `Retry-After` and retries on 503 (server restarting, video stream slots full). Don't raise on it; connection errors are not retried there.
- **Workers drain passes sequentially by default** — `sequential=True` drains each pass in `-p` order before the next; `--round-robin` restores the interleave. Keep sequential the default.
- **Scoped claims** — filter scopes resolve server-side (`worker_api._resolve_scope_ids`); `_unprocessed_scoped` / `_count_scoped` chunk ids under the SQLite variable cap, and `db._scope_col` writes `+id IN (...)` above 2,000 ids.

## LLM routing and GPU

- **Each pass unloads its model from LM Studio when it drains** — `worker._release_llm_models` → `describe.unload_openai_models`. Keep a model the next pass shares or another worker still holds a live claim on; a failed status call or the Ollama route unloads nothing.
- **Routing LLM passes to LM Studio (OpenAI-compatible)** — models are chosen by role env var, not by name. Fleet role models live only in `rerun.FLEET_ROLE_MODELS`; reasoning models need `PHOTOSEARCH_LLM_REASONING_EFFORT=none` (`rerun.FLEET_REASONING_EFFORT`) or they return `''`. Log provenance through `describe.effective_model` / `worker._provenance_kwargs`, never the nominal CLI default.
- **GPU acceleration** — `clip_embed.py` / `quality.py` / `faces.py` auto-detect CUDA/ROCm, gated by `PHOTOSEARCH_DEVICE`; the Docker fleet is CPU-only, so GPU and video passes need `--native`.

## Watching jobs

- **Watching a long job: never `pgrep -f` a bare string** — it matches the watcher itself. Poll a marker file, `kill -0` the PID you started, or grep the log for the terminal SSE event (`"type": "done"`).

## Measured sharpness (schema v33) — column + throttled backfill

`photosearch/sharpness_backfill.py` is the only writer of `photos.sharpness*`.

- **Missing-only** — `sharpness_version IS NULL OR < SHARPNESS_VERSION`; only a version bump re-measures. Keep `ORDER BY +p.id` so the planner stays on `idx_photos_sharpness_version`.
- **Faces first** — a photo is a candidate only after its faces pass is done (face row, or faces attempts at the cap); measuring earlier stamps a frame-only answer that is never revisited.
- **A decode error is STORED** — `sharpness` NULL, error in `sharpness_json`, version set, so a bad file is never re-decoded. An absent file writes nothing (`not_local`) — the backfill only runs where the originals are (the NAS); on the replica every photo is `not_local`.
- **Guarded UPDATE** — `WHERE id = ? AND sharpness_version IS ?` (old value); a lost race counts `raced`.
- **`photo_ids=[]` is NO photos** — never the whole library; a `--folder` matching nothing is empty too.
- **Throttles** — `pause_s` between photos, commit every 25 (`COMMIT_EVERY`), `lower_priority()` (nice + IDLE ioprio). Return `status: "refused"` while the ingest `_sweep_lock` is held (probe with `ingest.sweep_lock_held`, never hold it) or an open batch-advance NAS job row exists; re-check per chunk and stop as `"yielded"`.
- **Not a batch step any more** — nothing reads the column; don't re-add it to `/batches` until something consumes it. Dry runs open the DB `mode=ro`.

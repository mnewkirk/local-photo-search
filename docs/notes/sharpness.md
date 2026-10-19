# Measured sharpness — history

Schema v33 sharpness columns and the throttled NAS backfill. Condensed rules: `.claude/rules/workers.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 4124) on 2026-10-19 -->
## Measured sharpness (schema v33) — column + throttled backfill

`docs/plans/sharpness-measurement.md` steps 4–5. `photos.sharpness` (REAL,
the headline score; NULL when unmeasurable), `sharpness_json` (every candidate
feature from `sharpness.measure_photo`, or `{"error": …}`), `sharpness_version`
(the `SHARPNESS_VERSION` that produced the row) and `sharpness_scored_at`, plus
`idx_photos_sharpness_version`. The **only writer** is
`photosearch/sharpness_backfill.py`. Nothing derives a tag from it yet: `sharp`
/ `blurry` stay FROZEN until the labelled eval (step 3) passes.

**Where:** only where the originals are — the NAS. On the replica
`photo_root` is unset, stored paths stay relative, and every photo counts
`not_local`: nothing is written, nothing is recorded as failed.

**Rules not to simplify away:**

- **Missing-only** = `sharpness_version IS NULL OR < SHARPNESS_VERSION`; a
  version bump re-measures, nothing else does. The candidate query orders by
  `+p.id` so the planner uses the index (MULTI-INDEX OR) — measured on a
  161k-row replica copy: 0.21 s table scan → 0.002 s, and the nightly "nothing
  left" check 0.17 s → ~0. The scan reads the whole wide `photos` table, the
  disk traffic a starved N100 can't afford.
- **Faces first:** a photo is a candidate only once its faces pass is done
  (a face row, or faces `attempts >= MAX`). Face boxes are the headline
  region; measuring before detection would stamp a frame-only answer with the
  current version that missing-only never revisits.
- **A decode error is STORED** (`sharpness` NULL, the error in
  `sharpness_json`, the version set) so a bad file is never re-decoded forever
  (the CLIP re-claim trap). An **absent** file records nothing (`not_local`).
- Every UPDATE is guarded `WHERE id = ? AND sharpness_version IS ?` (the old
  value, as the chunked percentile refresh does); a lost race counts `raced`.
- **`photo_ids=[]` is NO photos**, never the whole library; a `--folder` that
  matches nothing is empty too.

**Throttles (the 2026-09-19 disk-starvation incident):** `pause_s` (0.2 s)
between photos; commit every 25; `os.nice(10)` + the Linux IDLE I/O class via a
raw `ioprio_set` syscall (no psutil; skipped where unavailable; per-THREAD on
Linux, so inside the web server it only lowers the sweep/advance worker
thread); **refuses to start** (`status: "refused"`) while `ingest-incoming`
holds its lock — **or `unwrap-zip-videos --apply` / `refile-unknown-camera
--apply`, which hold the same lock**; `_sweep_lock(holder=…)` records the job
and the refusal names it (it used to call every holder "an ingest sweep") —
`ingest.sweep_lock_held` is a shared non-blocking PROBE,
never a hold, so it can't make the 04:00 cron fail — or while any open,
unexpired batch-advance **NAS** job row exists (a crashed advance therefore
blocks it for up to the 6 h TTL), and re-checks every chunk, stopping as
`"yielded"` if one starts mid-run.

**Where it's wired — all opt-in:**

- CLI: `photosearch sharpness [--limit N] [--apply] [--pause 0.2] [--folder F]`.
  Dry run by default, opened `mode=ro` (a missing `--db` is an error, never a
  stub); on a pre-v33 DB it says to migrate first.
- Maintenance stage `sharpness`: OFF by default — `maintenance-sweep
  --sharpness [--sharpness-limit 5000]`, API `"sharpness": true` (or
  `do_sharpness`) + `"sharpness_limit"`, a checkbox on `/admin/maintenance`.
  Needs pixels, so like `colors` it is EXCLUDED on the replica (400).
- **Not a batch step any more** (removed 2026-10-18, owner decision). It was
  an optional box on `/batches`, but nothing reads `photos.sharpness` — the
  `blurry` tag it was built for failed its eval and the plan is CLOSED — so
  every batch showed "Needs to be queued" for N100 work feeding nothing.
  Bring it back (or make it a desktop worker pass, since the fleet already
  pulls originals) only once something consumes the column. A leftover
  `ingest_batch_jobs` row for it is ignored.
- `mirror-fields` / `rerun._MIRROR_COLUMNS` carry the four columns.

Tests: `tests/test_sharpness_backfill.py`, `TestSharpnessIsNotABatchStep` in
`tests/test_batch_state.py`, `test_v32_db_migrates_to_v33_sharpness_columns`.

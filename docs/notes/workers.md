# Workers — history and traps

Worker passes, long-job watchers, server-side write failures vs retry attempts. Condensed rules: `.claude/rules/workers.md`. Runbook: the skill's `reference/workers.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 298) on 2026-10-19 -->
### Watching a long job: never `pgrep -f` a bare string

A watcher that polls `pgrep -f "<pattern>"` **matches itself**, because its own
command line contains the pattern it is searching for. The loop then never
terminates and — worse — anything else polling the same pattern believes the
job is still running.

This has bitten three separate times in one session:

- `pkill -f "analyze.py"` killed the very shell running the command (the shell's
  argv contained `analyze.py`), so a heredoc half-wrote its script.
- A recluster watcher (`pgrep -f "cli.py recluster-faces"`) spun for its full
  50-minute budget after the job had finished.
- A Google Photos upload watcher matched itself, so a chained follow-up job sat
  waiting on a process that had exited ~40 minutes earlier — and the status it
  reported was its own liveness, not the upload's.

Use a signal the watcher cannot be confused with:

```bash
# GOOD — the job writes a marker; the watcher polls the marker
nohup sh -c 'long-job …; echo done > /tmp/job.done' &
until [ -f /tmp/job.done ]; do sleep 10; done

# GOOD — watch the PID you actually started
nohup long-job & echo $! > /tmp/job.pid
while kill -0 "$(cat /tmp/job.pid)" 2>/dev/null; do sleep 10; done

# GOOD — completion is in the output; grep for it, not for the process
until grep -q '"type": "done"' /tmp/job.log; do sleep 10; done

# BAD — matches the watcher itself, forever
until ! pgrep -f "long-job" >/dev/null; do sleep 10; done
```

If a pattern really is the only option, exclude self with `pgrep -f pat | grep -v $$`
or bracket a character (`"[l]ong-job"`) — but a marker file or PID is safer and
reads better. And when a job streams SSE, prefer its terminal event
(`"type": "done"`) over process liveness: the client can stay open on the stream
long after the work has finished, which is exactly how the upload watcher
misreported a completed job as running.

<!-- moved verbatim from CLAUDE.md (line 441) on 2026-10-19 -->
### Worker passes

The valid worker passes (`-p`/`--passes`, enforced by `valid_passes` in
`cli.py`) are: `clip`, `faces`, `quality`, `describe`, `category-content`,
`category-visual`, `keywords`, `verify`, `aesthetics`, plus the four video
passes `video-probe`, `video-scan`, `video-subjects`, `video-describe` (see
"## Video"). The full LLM-backed still-photo pass list is:

```
clip,faces,quality,describe,category-content,category-visual,keywords,verify,aesthetics
```

**The old `tags` pass was removed** and split into three first-class passes —
passing `--passes tags` now errors with "unknown pass type". The split:

- **`category-visual`** (vision) — visual-quality tags from the image
  (inherits the old `tags` task; `--category-visual-model`, default `llava`).
- **`category-content`** (text-only) — content categories from the existing
  description (`--category-content-model`, default `llama3.2:3b`).
- **`keywords`** (text-only) — free-form keywords from the description
  (`--keywords-model`, default `llama3.2:3b`).
- **`aesthetics`** (vision) — rich VLM aesthetic scoring (`--aesthetics-model`,
  default `qwen2.5-vl`; role `aesthetics` →
  `PHOTOSEARCH_LLM_AESTHETICS_MODEL`). See below.

<!-- moved verbatim from CLAUDE.md (line 1446) on 2026-10-19 -->
### A failed SERVER-side write must not burn a retry attempt

`worker_api.submit_results` counted a photo as processed *before* attempting
its write, and `mark_processed` UPSERT-increments `worker_processed.attempts`
while the claim path stops claiming at `MAX_PROCESS_ATTEMPTS` (3). So a DB
error on the NAS punished the PHOTO: two sharp, ordinary frames reached
attempts=5 for `category-visual` with `visual_tags` still NULL, abandoned for
good, one batch of GPU work thrown away per collision. Seven branches had the
shape (`describe`, `tags`, `category-content`, `category-visual`, `keywords`,
`aesthetics`, `faces`); all now route through `_SubmitOutcome`.

An attempt is now spent only when the result was **persisted**, or when the
**worker** reported a genuine per-photo outcome. The deliberate semantics are
unchanged and must stay: a photo with no description or an empty tag list is
*done* in one pass (`'[]'` is written), and a worker-side timeout defers by
being omitted from the payload entirely.

**Worker-reported failures are a separate `failures` list** on the submit
payload (`FailedResult{photo_id, error}`), built by `worker._submit_kwargs`
from `_failure_row`s. Each spends ONE attempt, writes no column, and is logged
to `index_errors` — so a poison photo (an unloadable image, a detection or
verification that raises every time) is retired by the cap instead of being
re-claimed forever. It is deliberately NOT folded into the per-pass rows: an
older server ignores the unknown field and behaves as before, whereas
`aesthetic_score=None` would 422 an old `QualityResult` and a verify
`status="error"` row would be written as a real verification. Timeouts /
network errors (`worker._TRANSIENT`) are still omitted, never reported.

**`quality` and `verify` are on the ledger now.** Their claim/count
predicates carry the same `attempts >= MAX_PROCESS_ATTEMPTS` exclusion as the
other passes (`db._QUALITY_NOT_EXHAUSTED` / `_VERIFY_NOT_EXHAUSTED`; quality
keeps its either-column self-heal), `submit_results` marks both processed, and
`clear-pass` deletes their ledger rows. Before, the worker silently dropped a
photo it could not score or verify and nothing recorded it; a quality photo
whose concept analysis failed rewrote its score and stayed claimable — both
infinite re-claims. Existing libraries have no quality/verify ledger rows, so
deploying changes nothing until a photo actually fails.

**A clean empty `faces` result is TERMINAL.** `submit_results` marks it with
`db.mark_processed(..., terminal=True)`, which sets `attempts` straight to
`MAX_PROCESS_ATTEMPTS` (never lowering a higher count; no schema change). It
used to be `+1`, so every photo with nobody in frame was detected three times —
67,853 such photos, ~136k wasted InsightFace runs. This is only safe because
the worker no longer sends `faces: []` for a detection **exception**: that is
now a failure row (one attempt, still retried). A non-empty result is
unchanged. The in-process `index.py` faces writers (directory + collection
mode) follow the same contract — terminal on empty, one attempt on an error,
skip exhausted photos, and `--force-faces` clears the ledger.

**Deploy the server and the workers together.** A worker older than this
change still sends `faces: []` on a detection error, which the new server
reads as "nobody here" and retires in one shot — a transiently-failing photo
would lose its two retries (recover with `clear-pass faces` on the affected
scope). The reverse (new worker, old server) is safe but uncapped: the old
server ignores `failures`, so a failing photo is re-claimed as it always was.

Deferral is for a **transient lock only** — `_is_transient_db_error`: a
`sqlite3.OperationalError` whose message mentions "locked" or "busy".
**Do not widen this back to a bare `except Exception`.** Deferring every
error uncaps the pass: a deterministically-failing photo never reaches
MAX_PROCESS_ATTEMPTS, is re-claimed every TTL forever, and pays for a model
run each cycle — the pathology documented below for the formerly un-capped
`clip` claim. (The first version of this fix did exactly that; review caught it.)
`faces` is the worst case, because its predicate is
`NOT EXISTS(faces) AND attempts < MAX` with no column to heal it and a
malformed payload (short bbox, missing `encoding`, non-512 vector) raises
*before* any INSERT. Every non-lock failure therefore still counts the
attempt and logs the reason. One classifier serves the per-row,
batch-commit and mark_processed paths so they cannot drift.

`begin_batch` defers the commit, so SQLITE_BUSY usually surfaces at
**COMMIT**. A transient commit failure defers the batch (200 with
`status: "deferred"`, claim released — it used to be a 500 that left the claim
held); a non-lock one (disk full, read-only FS) counts the attempt and logs at
ERROR — re-claiming is unavoidable there, so only the cap bounds it. What is
knowable is that the *final flush* did not land: `db.log_error` commits
unconditionally, so earlier rows may be on disk. That is harmless — a
landed-but-unmarked row is never offered again by a column-guarded predicate —
but don't let the message claim "nothing landed". The response gained
`deferred` / `deferred_photo_ids` (additive; older workers read only
`written`/`processed`). Deferrals mean a long-running writer (the sweep), not a
bad photo. Tests: `tests/test_worker_submit_resilience.py`.

### Corrections found while condensing (2026-10-19)

Checked against the code while writing `.claude/rules/`; the text above is kept verbatim.

- `valid_passes` in `cli.py` has five video passes: `video-probe`, `video-proxy`, `video-scan`, `video-subjects`, `video-describe`.

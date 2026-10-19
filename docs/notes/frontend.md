# Frontend — history and traps

Status page, M28 re-runs, the deploy panel, the no-linter check, `PS.poll`. Condensed rules: `.claude/rules/frontend.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 2283) on 2026-10-19 -->
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

<!-- moved verbatim from CLAUDE.md (line 2309) on 2026-10-19 -->
## Replica freshness card + "Sync from NAS" (M26a UI)

`/status` renders a **Replica** card — drift pill (local vs NAS photo counts
via `GET /api/admin/replica-status`), last-sync age, and a **Sync from NAS**
button that streams `POST /api/admin/replica-sync` (= `sync-replica.sh`:
consistent `dump-db` pull + atomic swap) into an inline log, same SSE shape as
the deploy-page buttons. The card only renders when `replica_mode` is true
(`PHOTOSEARCH_NAS_URL` set), so the NAS's own /status is unchanged. This is
the "sync now" button from the M26a plan; the nightly pull still reconciles.

<!-- moved verbatim from CLAUDE.md (line 2319) on 2026-10-19 -->
## On-demand pass re-runs + status-page fleet launcher (M28)

Re-run any index pass on a photo from the image view, and launch a worker
fleet for chosen passes from `/status`. The NAS (weak N100, no GPU) can't
compute, so the desktop always does the model work; results are written
**NAS-authoritative then mirrored into the local replica DB** (the M26b
read-local / write-NAS / mirror-local model — don't wait for the nightly
`sync-replica.sh` full pull). Module: `photosearch/rerun.py`.

- **Image view** — `PS.PhotoModal` (`frontend/dist/shared.js`) has a "Re-run
  passes" section: checkboxes for all 8 passes, **Run now** (synchronous) vs
  **Queue for fleet**, plus a **Refresh** button (mirror + re-read). Checking
  `describe` auto-selects `category-content` + `keywords` — re-describing alone
  leaves those stale (the exact failure that mislabeled a sailboat/life-jacket
  photo as "soccer": truncated describe → text-only category pass hallucinated
  soccer tags, and verify never ran).
- **`POST /api/admin/rerun-passes`** `{photo_ids, passes, mode}` —
  `mode=sync` computes each pass in-process (reusing `worker._process_*` + a
  synthetic-batch submit; `submit-results` applies by `photo_id` even without a
  live claim) and submits to the NAS, then mirrors. `mode=queue` calls the NAS
  `clear-pass` (which already accepts `photo_ids`) so the fleet re-processes it.
- **`POST /api/admin/mirror-photos`** `{photo_ids}` + NAS
  **`GET /api/photos/{id}/mirror-fields`** — targeted per-photo refresh of
  text/scalar columns + CLIP embedding + face rows (no full sync). `rerun.py`
  applies it (`_apply_mirror`).
- **Status page** — a "Worker fleet" panel launches the **native** fleet via
  `run-workers.sh --native --name ui -s <NAS> -p <passes> -n <N>`, exporting LM
  Studio role models (defaults filled when `PHOTOSEARCH_TEXT_LLM_URL` is set).
  Endpoints `POST /api/admin/workers/start|stop`, `GET /workers/fleet-status`
  (process status — distinct from `/api/worker/status`'s queue depth).

`WorkerClient(..., probe=False)` skips the `/api/stats` connectivity probe
(heavy count scan — times out when the NAS is busy) for single-result submits.

**Mirror needs the NAS running M28 code** — `/mirror-fields` 404s on older
NAS images, so `mirrored:0` until the NAS is redeployed; the sync compute +
authoritative write still work, and the local UI catches up on the next sync.

<!-- moved verbatim from CLAUDE.md (line 2739) on 2026-10-19 -->
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

<!-- moved verbatim from CLAUDE.md (line 3308) on 2026-10-19 -->
## Frontend has no linter — `scripts/check-frontend-refs.js`

The pages are plain React UMD with no build step, so a helper referenced from
the wrong page is a `ReferenceError` that only appears when someone clicks the
button. **A syntax check does not catch it** — `new Function(src)` happily
accepts a call to an undefined name, which is exactly how a bare
`parseSSEChunk` (defined only in `admin_maintenance.html`) shipped in
`faces.html`.

```bash
node scripts/check-frontend-refs.js     # or: cd frontend && npm run check:refs
```

It flags bare `name(` calls nothing in that script binds, plus `PS.*` calls
naming something `shared.js` never defines. Wired into CI in the jest job and
into pytest (`tests/test_frontend_refs.py`). Two notes for anyone editing it:
comments and strings are blanked in **one left-to-right scan**, not layered
regexes (either order invents phantom names); and `shared.js` is an **IIFE**, so
only `window.PS` escapes — the first version collected its inner names as
globals and therefore *passed the very bug it was written for*.

**Shared frontend helpers go on `PS.*` in `shared.js`**, never copied between
pages.

<!-- moved verbatim from CLAUDE.md (line 3332) on 2026-10-19 -->
### Poll with `PS.poll`, never `setInterval`

`setInterval` fires whether or not the previous request came back. On 2026-09-19
the NAS's disks were saturated by an unrelated ~50 MB/s SMB bulk read (a desktop
backup); each status poll then took *minutes*, and two open `/admin/maintenance`
tabs kept issuing more until **all 40 request threads were stuck in disk wait**
— 20+ minutes per request, 38 stacked `maintenance-fingerprint` calls in the
log. The status page wedged the box it was reporting on, and ingest crawled at
one file per ~3 minutes until the backup was stopped (then ~2 files/s, load
44 → 5).

`PS.poll(fn, ms)` re-arms only **after** `fn`'s promise settles, pauses while the
tab is hidden, and returns `stop()` (which aborts the in-flight call). So a slow
server is polled slower, never harder. `fn` must **return** its fetch chain
(`Promise.all` for several) — a fire-and-forget fetch inside it defeats the
point. A plain `setInterval` is still right for a pure UI clock tick. Tests:
`frontend/__tests__/poll.test.js`.

Symptom to recognise next time: `load average` far above the core count while
every container shows near-zero CPU = threads in uninterruptible I/O wait.
`vmstat 1` (`b` and `wa` columns) confirms it; `ss -tni` on port 445 names the
SMB client doing the reading.

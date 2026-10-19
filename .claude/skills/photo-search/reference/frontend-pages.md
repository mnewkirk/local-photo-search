# Frontend pages

Read when working on a page: shared.js components, /status, /faces panels, /merges, review/culling, stack filtering.

<!-- moved verbatim from SKILL.md (line 729) on 2026-10-19 -->
## Shoot Review / Culling

Adaptive clustering of CLIP embeddings to select representative photos:
- Target ~10% selection from a shoot folder
- Agglomerative clustering with quality-weighted representative selection
- "Represent-all-then-trim" strategy ensures tag diversity
- Review selections persisted per folder in `review_selections` table

---

<!-- moved verbatim from SKILL.md (line 739) on 2026-10-19 -->
## Shared Frontend Components (shared.js)

- `PS.SharedHeader` — Consistent nav header across all pages. activePage values:
  `'search' | 'review' | 'faces' | 'merges' | 'collections' | 'status'`.
- `PS.PhotoModal` — Unified photo detail modal with configurable features:
  showFaces, showCollections, showLocation, showSearchScore, showAesthetics.
  Includes face editing, collection management, stacking UI, keyboard navigation
  (arrow keys), and mobile swipe navigation (swipe left/right on touch devices).
  Slots: fetchDetail, headerChildren, footerChildren.
- `PS.GooglePhotosButton` — Upload single photo to Google Photos from modal sidebar
- `PS.formatFocalLength()` / `PS.formatFNumber()` — EXIF display helpers

<!-- moved verbatim from SKILL.md (line 751) on 2026-10-19 -->
### /status page

The status page is both a dashboard and a control panel:

- **Stat grid** — one card per pass (CLIP, faces, descriptions, quality,
  concepts, **tags**, stacks, verify). Each card reads its count from
  `/api/stats`; the Tags card uses the `tagged` field
  (`COUNT(*) WHERE tags IS NOT NULL AND tags != '[]'`).
- **Workers panel** — fetched from `/api/worker/status` on mount and every
  5s after. Renders total active workers, total queued photos across all
  passes, a row of per-pass queue-depth pills, and one row per active claim
  with pass type, worker_id, photo count, and a live TTL (yellow under
  2 min, red once expired). A separate 1s tick re-renders TTLs between
  poll cycles. Main `/api/stats` is only re-fetched on the Refresh
  button, since its COUNTs are expensive.
- **Stacking form** — editable `time_window_sec` / `clip_threshold` /
  `max_stack_span_sec` fields (pre-filled with CLI defaults 5.0 / 0.05 /
  10.0, with a Reset button), plus "Clear existing stacks first" (with
  a confirm prompt) and "Dry run" toggles. Submit calls
  `POST /api/stacks/detect/stream`, reads the SSE via
  `response.body.getReader()` and `\n\n`-split chunks (EventSource can't
  POST), renders a phase label + progress bar + pair count + elapsed
  seconds while running, and swaps in a Cancel button that calls
  `AbortController.abort()`. On `done`, calls the parent's `load()` so
  the Stacks stat card refreshes immediately.
- **Activity chart** — hourly buckets for the last 72h, color-coded by
  pass type, includes a "cleared (--force)" series.
- **Verify failures / Errors log** — unchanged from M12 shipping.

The Workers panel + Stacking form establish the pattern for other
"run this job from the UI" actions: fetch-based SSE, throttled progress
emission inside `on_progress` callbacks, `should_abort` pair for cancel,
keepalive every 0.5s on the server side.

#### Deploy panel (Version / Fetch / Build / Restart)

The Deployment card on `/status` replaces the manual
`git pull && docker compose build && docker compose up -d` cycle. It hits
`/api/admin/*` endpoints under `photosearch/admin_api.py`:

- `GET /version` — reads `/app/BUILD_SHA` (baked by the Dockerfile from a
  `--build-arg GIT_SHA=...`) and runs `git log/status` in the mounted
  `/repo` to report dirty files, ahead/behind, and whether the deployed
  SHA matches HEAD.
- `POST /git-fetch`, `POST /git-pull` — `git -c safe.directory=*` under
  `/repo` (UGOS owns it as a different uid; the global flag is required).
- `POST /docker-build` — SSE stream of `docker compose build
  --build-arg GIT_SHA=$(git rev-parse HEAD)`. The explicit `--build-arg`
  is load-bearing — relying on env-var expansion in the compose file
  occasionally resolved to "unknown" even with `GIT_SHA` exported on the
  subprocess. The UI flips to the green pulsing "Restart" CTA on the
  SSE `done` event (NOT on connection close — uvicorn keeps the SSE
  channel open briefly after `done`).
- `POST /restart` — **see helper-container section below.**

**Helper-container restart pattern (critical):** the Restart endpoint does
NOT call `docker compose up` in-process. It launches a detached helper:

```python
docker run --rm --detach \
  --name photosearch-restart-helper \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v $HOST_REPO_DIR:/repo \
  -w /repo docker:cli \
  compose -p photosearch -f docker-compose.nas.yml up -d --no-deps photosearch
```

The helper has its own PID namespace, so it survives the photosearch
container's SIGTERM/SIGKILL and reliably reaches the "start new
container" step. Doing this in-process would put the compose subprocess
inside the very container being recreated — when docker kills that
container's PID namespace, every process inside it dies, INCLUDING the
compose subprocess that's mid-restart. The result is a new container
stranded in Created state with a rename-suffix name and no way for the
panel to recover (since the panel container is now dead too).

`HOST_REPO_DIR` env var (set in docker-compose.nas.yml,
`${REPO_DIR:-/volume1/docker/photosearch}`) tells the helper the
**host** path to mount as `/repo` — the in-container `/repo` path is
useless to a sibling container. `setsid`/`nohup` don't fix this either;
they detach from process groups, not PID namespaces.

`--no-deps` was originally required because compose followed `depends_on`
and tried to recreate ollama too, hitting a name conflict on
`photosearch-ollama`. The ollama service was removed from
docker-compose.nas.yml on 2026-07-17 (nothing used it — see the compose
header), so that conflict is gone; `--no-deps` is kept because the panel
should only ever swap the photosearch container itself.

**Graceful shutdown handshake.** When the panel triggers restart:
1. `/api/admin/restart` first runs
   `UPDATE worker_claims SET expires_at = datetime('now', '+15 minutes')`
   so workers' in-flight batches survive the swap gap even if their
   `renew-claim` calls time out during the brief downtime.
2. Helper container is launched (detached).
3. Compose sends SIGTERM to the old photosearch container.
4. uvicorn enters shutdown; FastAPI fires `@app.on_event("shutdown")`
   which sets `worker_api._shutting_down = True`.
5. The middleware in `web.py` starts returning `503 Retry-After: 30`
   on `/api/worker/*` and `/api/photos/<id>/full`. Workers'
   `WorkerClient._request()` sees 503, sleeps Retry-After seconds, and
   loops. Browser UI paths (thumbnails, stats, search) keep serving so
   the drain only waits on actual in-flight worker downloads.
6. `stop_grace_period: 60s` in the compose file gives uvicorn time to
   finish those in-flight transfers.
7. Old dies, helper finishes its `compose up`, new container binds :8000.

**If the helper gets wedged** (rare, e.g. a previous helper crashed
without `--rm` succeeding), the panel returns
`"...is already in use by container..."` with a hint:
```bash
docker rm -f photosearch-restart-helper
```
Then retry from the panel.

**Manual restart recovery (when panel is unreachable):**

```bash
# 1. Clean up any stuck rename-suffix containers from a partial recreate.
docker ps -a | grep photosearch
docker rm -f <any-container-with-rename-suffix-name>

# 2. CRITICAL: REPO_DIR must not be set to empty string. Bash's ${VAR:-default}
#    falls back to default only if UNSET, not if set-but-empty — so an empty
#    REPO_DIR resolves to an empty volume mount source and docker silently
#    drops the /repo mount, breaking the deploy panel on the next boot.
unset REPO_DIR

cd /volume1/docker/photosearch
GIT_SHA=$(docker run --rm -v /volume1/docker/photosearch:/repo alpine sh -c \
  "apk add -q git && git config --global --add safe.directory /repo && git -C /repo rev-parse HEAD")
docker compose -f docker-compose.nas.yml build --build-arg GIT_SHA=$GIT_SHA photosearch
docker compose -f docker-compose.nas.yml up -d --no-deps photosearch
sleep 5
docker ps | grep photosearch
curl -s http://localhost:8000/api/admin/version | python3 -m json.tool | head -15
```

<!-- moved verbatim from SKILL.md (line 889) on 2026-10-19 -->
### /faces review panels (M32)

`/faces` round-trips its content filters through the **URL**
(`date_from`/`date_to`/`location`/`q`/`camera`), so a shoot review survives a
reload and is shareable:

```
/faces?filter=named&date_from=2026-09-12&date_to=2026-09-12
```

They did not used to: a reload dropped you back to the unfiltered library, and
reviewing one shoot meant re-typing the date every refresh. Apply/Clear use
`replaceState` so the history does not fill up, and a navigation-only
`updateUrl` falls back to the URL's existing filters so opening a cluster
cannot silently drop the shoot.

Three panels, all opened from the filter bar:

- **Inspect** (on a person group) — sub-clusters that person's faces, scores
  each by distance to their core, bulk unassign/reassign. Inherits the page's
  date filter; states its scope ("showing 196 of 20,575") because it
  bulk-unassigns and a hidden scope would be the wrong thing to hide. Radius
  dropdown, since the useful eps is shoot-dependent.
- **🔍 Verify labels** — any date range. One row per suspect: the disputed crop,
  the identity it CARRIES, the identity it LOOKS LIKE, nearest references first
  on each side. Reassign / Reject / Keep. Reject records `rejected`, so
  auto-matching cannot put the label back. A **RIVAL IN PHOTO** row additionally
  shows the face in the same photo that claims the label better, and offers
  **Swap** (one click, fixes both faces) plus **Swap all N frames** when the
  same mistake was independently flagged in neighbouring burst frames. Rivals
  sort first; a "rivals only" toggle isolates them. A source dropdown
  (trusted / all / temporal-only) decides which labels are adjudicated —
  trusted by default, with a banner naming what it is not showing.
- **📊 Label health** — the only panel NOT gated on a date filter, because
  seeing it without picking a day first is the point. Ranks everyone by how many
  temporal labels they carry, measures one person on click, and turns each of
  their worst days into a button that opens the Bulk-unmatch grid scoped to that
  person + day.
- **✂ Bulk unmatch** — any date range. Remove a whole bad `match_source` from
  one person: pick the person, optionally `+strict` and a distance gate, then a
  farthest-first crop grid with the usual multi-select (click / shift-range /
  Select all / > 1.15 / > 1.00). The grid IS the dry run and the selection is
  the confirmation. A **date scope is required** — library-wide is thousands of
  faces. Applies via `bulk-assign`, so cleared faces are `rejected` and
  auto-matching cannot put them back. Undo is in-session only; for a durable
  one use the CLI's `--snapshot`.

Before browsing any of these panels on a cold shoot, warm the crops — each one
is a full image decode on first view (~2 s on the N100), so a 120-face grid
takes minutes to fill. `/admin/maintenance` → **Warm face crops** (date range +
optional person), or:

```bash
$DC run --rm photosearch warm-face-crops --all \
    --date-from 2026-09-12 --date-to 2026-09-12
```

It defaults to every face, not just matched ones — the panels browse unknown
clusters too — so a date scope is what keeps it affordable.

- **⚽ Review team faces** — single day only (it learns ONE jersey colour; a
  range would average two kits). Groups the day's unknown team faces so you
  name a handful of groups instead of a thousand crops. The hue is learned from
  faces already named that day, so **clean the day's labels first** or it learns
  from garbage; there is a manual hue override. The "Ungrouped" bucket is
  rendered but deliberately NOT assignable — it is DBSCAN noise, many different
  people.

<!-- moved verbatim from SKILL.md (line 957) on 2026-10-19 -->
### Face-encoding cache (suggest-person, verify-labels)

`PhotoDB.get_face_encodings_cached` — process-wide LRU of float32 encodings
keyed by (DB real path, face id), ~40k faces / ~80 MB. Used by
`face_suggest` (➕ More of this kid) and `face_verify` (🔍 Verify labels).
No invalidation is needed: encodings are only INSERTed or DELETEd and
`faces.id` is AUTOINCREMENT, so a relabel changes which ids are asked for,
never an id's encoding. Koa over Sep 1–27: 12.3 s → 1.9 s first call, 0.4 s
after (replica copy); NAS repeat calls 0.5–1.9 s. The first calls after a
container restart are still slow (69 s measured, cold disk + MCP startup).
**If an in-place encoding update is ever added, it must evict that face.**
Tests: `tests/test_face_encoding_cache.py`.

<!-- moved verbatim from SKILL.md (line 970) on 2026-10-19 -->
### Frontend has no linter — run `scripts/check-frontend-refs.js`

No build step and no bundler, so a helper referenced from the wrong page is a
runtime `ReferenceError` nothing catches. A syntax check does **not** find it:
`new Function(src)` accepts calls to undefined names, which is how a bare
`parseSSEChunk` (defined only in `admin_maintenance.html`) shipped in
`faces.html`. `node scripts/check-frontend-refs.js` (also
`cd frontend && npm run check:refs`) flags those plus `PS.*` calls naming
something `shared.js` never defines. In CI via the jest job and pytest.
**Shared helpers belong on `PS.*` in `shared.js`, never copied between pages.**

<!-- moved verbatim from SKILL.md (line 981) on 2026-10-19 -->
### /faces URL params (deep-linkable)

- `?name=<Person>` — open named-person detail
- `?person_id=N` — open person by id (works for hidden/unloaded pages)
- `?cluster_id=N` — open unknown cluster by id (works for hidden singletons)
- `?face_id=N` — open the group containing this rep face
- `?filter=named|unknown|ignored|all` — pre-filter the grid

`cluster_id` / `person_id` fall back to `/api/faces/group-info` when the group
isn't already in the loaded page, so the suggest-face-merges CLI emits these
with `--base-url` and they always resolve.

<!-- moved verbatim from SKILL.md (line 993) on 2026-10-19 -->
### /merges page architecture

- Reads cached suggestions from `/api/faces/suggestions` (not computed on page
  load — the engine takes ~20s)
- Per-card face strip of up to 4 crops (via `sample_face_ids` that the API
  augments onto the JSON — no per-card fetch)
- Click any crop → `/api/faces/face-detail/{face_id}` → photo-preview overlay
  with the bbox highlighted on the full photo
- Confidence tier (Strong/Probable/Borderline) calibrated from the observed
  FP pattern: `min_pair < 0.50` strong, `< 0.55` probable, `≥ 0.55` borderline
- Risk badges: "attractor risk" (big target cluster), "long overlap"
  (cluster↔cluster spanning >180 days), "edge score" (min_pair ≥ 0.55),
  "rewritten after merge" (see below)
- Keyboard: `J/↓` next, `K/↑` prev, `A/Enter` accept, `D` dismiss, `Esc` blur.
  Accept does NOT advance focus — the accepted card is filtered out, so the
  next row slides into the same slot naturally
- **Merge chain rewriting**: when you accept `A → B`, any pending `C → A`
  becomes `C → B` (using stored target info from the accepted merge). The
  original min_pair is still a valid lower bound because the faces that
  matched C are now in B. `resolveTarget()` handles chains (`A → B`, then
  `B → Person`) with cycle protection. Source-emptied suggestions
  (`A → X` where A was merged away) are tagged stale, not rewritten.
- Dismissals persist in `localStorage` keyed by the sorted rep_face_id pair
  (stable across reclusters). Accepts persist in the DB.

---

<!-- moved verbatim from SKILL.md (line 1702) on 2026-10-19 -->
### Frontend stack filtering behavior

- **Search page:** Filename searches bypass stack filtering — matched photos always visible
  regardless of stack position. Other searches respect stack collapse/expand.
- **Collection page:** All collection photos always visible (no stack filtering for
  explicitly-added photos). Expand Stacks shows additional stack members from the API.
- **Result counts:** Show visible count, with "(N with stacks)" if stacking hides any.
- **Review page:** Folder picker is searchable with partial path matching
  ("2026-04" matches "2026/2026-04-06"), sorted by most recent photos first.
- **Static assets:** Served with `Cache-Control: no-cache` to prevent stale JS after deploys.

---

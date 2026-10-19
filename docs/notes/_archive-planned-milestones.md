# ARCHIVE — the old "Planned milestones" sections

Frozen copies of the milestone lists CLAUDE.md and SKILL.md carried until 2026-10-19. STALE: plan status lives in `docs/plans/README.md` only.

<!-- moved verbatim from CLAUDE.md (line 4193) on 2026-10-19 -->
## Planned milestones (see `docs/plans/`)

- `docs/plans/video-support.md` — **BUILT 2026-10-17, deployed 2026-10-18,
  backfill not started** (commits `9015b61`…`8eb246c`, schema v39). See
  "## Video" above; the plan's Step 8 is the copy-paste backfill runbook.

- `docs/plans/search-indexes.md` — **SHIPPED 2026-10-04.** Schema v34 search
  indexes; `search_combined` composes date/camera/people into one id scope;
  id-first people and LIKE queries; SQL-paginated aesthetics/quality browses;
  people-only searches read full rows for the page only. Same day: the
  persistent API request log with source + intent (`request-stats`) and the
  face-encoding cache for suggest-person / verify-labels. Measured: camera +
  2 days >120 s → 0.4 s cold on the NAS; Calvin + camera + place + 2 days
  2.2 GB → 1.3 MB read. The `idx_photos_aes_technical/composition/impact`
  indexes are deliberately KEPT (owner decision 2026-10-04): the SQL-paginated
  browse floors use them. Open: date pushdown into the remaining standalone
  filters.

- `docs/plans/ingest-batch-readiness.md` — **SHIPPED 2026-09-19.** Per-batch
  readiness + status flow: one dated folder = one batch, a `/batches` page
  with a per-step flow diagram (the six states above) and a one-click
  `batch-advance` / fleet launch. See "## Ingest batches (`/batches`)" above
  for the shape and the traps; `docs/plans/ingest-batch-readiness.md` itself
  now carries a "What changed during the build" section for what the plan
  got wrong along the way.

- `docs/plans/infer-location-refinements.md` — post-M19 cascade fixes
  surfaced on the 127k NAS library. Cap hop depth (cascade ran 776
  deep, 71% of candidates below 0.25 confidence), downweight inferred
  anchors on re-scan so decay protects cross-run compounding, drop the
  dead `--max-cascade-rounds` flag. All localized to
  `photosearch/infer_location.py`.
- `docs/plans/search-accuracy-improvements.md` — next-milestone
  search work. Quick-wins bundle: Reciprocal Rank Fusion across
  filters + recency decay on final ranking + structured
  `country`/`admin1`/`admin2`/`locality` columns (fixes "Calvin in
  France ranks ancient photos first" and "Marin County as a query
  returns nothing" simultaneously, ~200 lines + schema v19 + one
  backfill CLI). Later work: fuzzy name/place matching, ambiguity
  disambiguation via photo GPS priors, LLM query rewriter, VLM
  re-ranking, self-hosted Nominatim.
- `docs/plans/google-photos-import.md` — M20. Takeout-based import of
  ~200K smartphone photos. Google Photos API read scopes were deprecated for
  third-party apps in March 2025, so Takeout is the only path. Incremental
  per-year export, composite dedup (photoTakenTime + device + filename stem),
  lands in `/photos/YYYY/YYYY-MM-DD_gphotos/`. Phone GPS amplifies the
  inferred-geotag recall on camera photos.
- `docs/plans/llm-driven-search.md` — M24 **(SHIPPED)**. Stop hand-assembling
  search filters; let an LLM be the query planner. One **shared tool layer**
  (`photosearch/tools.py`) consumed by two adapters — **M24a** a
  streamable-HTTP MCP server (`photosearch/mcp_server.py`, low-level `Server`)
  as the `photosearch-mcp` NAS container (`mcp>=1.2`), and **M24b** an in-app
  `POST /api/ask` SSE agent loop (`photosearch/agent.py`) on the **local** LM
  Studio/Ollama backend (nothing leaves the NAS), surfaced as an "✨ Ask" mode
  toggle on the search page. Agent model via `PHOTOSEARCH_LLM_AGENT_MODEL`
  (falls back to the text role); tool-calling loop capped at 6 steps with a
  single-shot NL→filters fallback (`PHOTOSEARCH_AGENT_SINGLE_SHOT=1`) for
  non-tool-calling models. Image returns gated by
  `PHOTOSEARCH_MCP_ALLOW_IMAGES` (default off). Tests: `tests/test_tools.py`,
  `tests/test_agent.py`.

  **The registry has grown well past the original 7 tools** — the current set
  (`all_tools()` order) is: `get_library_overview`, `list_people`,
  `list_places`, `list_vocab`, `search_photos`, `summarize` (facet/COUNT by
  year/month/location/person/camera_model), `representatives` (top-N best per
  bucket — "one per year", `max_buckets` cap, `rank_by=quality|subject`),
  `daily_highlights` (best-per-day, burst/near-dup collapsed, chronological),
  `group_into_chapters` + `daily_scene_breakdown` + `suggest_layout` (the
  **photo-book curation tools** — geographic chapters, per-day sub-scenes, and
  a house-style spread/layout plan), `get_photo`, `rerank_photos` (VLM
  re-rank), `get_photo_image` (image-gated), plus the M26b **write** tools
  `set_photo_location` / `set_photo_tags` / `add_to_collection` (gated by
  `PHOTOSEARCH_ALLOW_WRITES`, dry-run→confirm). All the search-like tools share
  one filter vocabulary via `_build_filter_sql` / the `_CURATION_FILTER_PROPS`
  schema fragment, so they stay in lockstep with structured Search — including
  `camera`, `min_aesthetic` (VLM percentile), `style_tag`, and `match_source`.

  **Filters-as-Ask-inputs:** the "✨ Ask" mode now KEEPS the structured filter
  bar visible; whatever the user pins there is sent to `/api/ask` as a
  `filters` object and **enforced server-side** — `agent._normalize_locked` /
  `_merge_locked` inject them into every filter-consuming tool call (people
  union; scalars hard-override) and `_locked_prompt` tells the model they're
  pinned. So a pinned filter is a HARD input to the agent's queries, not a
  post-filter on its results (fixes "camera=KODAK PIXPRO WPZ2 returns non-Kodak
  photos" — the tool schemas simply hadn't exposed `camera`, so the model
  dropped it). Compact search hits now carry `camera_model` so the shared grid
  📷 badge renders on Ask cards too.

  **Grouped photobook results + per-day aesthetics:** the book tools now drive a
  **sectioned** results view — `agent._grouping_for` emits `group_field`+`groups`
  on the `photos` event (chapters/scenes/day/bucket) and `PhotoGrid` renders
  labeled sections (like Review's Timeline) instead of a flat grid, so chapters
  are actually visible. Ask also mirrors its query+pinned-filters into the URL.
  New `min_day_aesthetic` filter floors on the PER-DAY percentile
  (`COALESCE(aes_subject_overall_day_pct, aes_overall_day_pct)`) rather than the
  library-wide `aes_overall_pct` of `min_aesthetic` — "best of each day" on a
  trip. Threaded through `search_combined`, `/api/search`, the Ask tools, and a
  "Min day %" UI input.

- `docs/plans/backfill-maintenance-sweep.md` — M25. **SHIPPED (2026-06-21).**
  `maintenance-sweep` / `validate-data` / `repair-data` CLIs + `photosearch/
  maintenance.py` (dependency-ordered, missing-only, dry-run default, on_progress/
  should_abort), SSE `POST /api/admin/maintenance-sweep` + `GET /api/admin/
  validate-data`, and a Maintenance-sweep card on `/status`. Tests in
  `tests/test_maintenance.py`. Heavy GPU passes stay with the worker fleet;
  recluster is opt-in (it clears `ignored_clusters`). Original plan below:
  Many derived-data passes get lost on new photos —
  structured location columns (`normalize-places`), inferred GPS
  (`infer-locations`), face clustering/matching (`recluster-faces` /
  `match-faces`), colors (cron runs `--no-colors`), and cross-folder
  stacking are all manual CLIs nothing schedules; only `ingest-incoming`
  runs daily. Proposes one idempotent `maintenance-sweep` (missing-only
  predicates, dependency-ordered, cron + `/status` button) over the
  lightweight CPU backfills, leaving heavy GPU passes to the worker fleet.
  Also a `validate-data`/`repair-data` tool for invalid rows — e.g. the
  corrupt `date_taken` control-byte values found in M24a (re-extract from
  EXIF/folder/mtime, else NULL), bad GPS, malformed JSON columns.

- `docs/plans/review-folders-perf.md` — M27. **Queued.** The `/review`
  folder picker is slow: `/api/review/folders` loads all ~163k photo rows and
  runs `Path(fp).parent` per row in Python on every load. Quick win: swap
  `Path` for `fp.rsplit('/',1)[0]`. Durable fix: add an indexed `folder`
  column (dirname of filepath) populated in `add_photo` + backfill, then
  `GROUP BY folder` — also speeds `/geotag`'s folder summary. Optional caching
  invalidated on ingest.

- `docs/plans/local-replica-and-writes.md` — M26. Pivot for the asymmetric
  hardware (weak N100 NAS, strong GPU desktop): run the whole experience
  (web UI + `/api/ask` agent + MCP) on the desktop off a **local
  read-replica** of the SQLite DB, with the local LM Studio — search compute
  (CLIP embed, KNN, SQL) moves to the fast machine. **M26a** syncs the DB +
  thumbnail mirror (rsync, nightly + a "sync now" button; thumbs ≈12 GB,
  append-only so tiny deltas; DB ≈1 GB; previews/full proxied from the NAS).
  **M26b** adds **write tools** (tags / location / collections) with a
  read-local / **write-NAS-authoritative / mirror-local** dual-write model
  (NAS applies + returns canonical values → mirror those to the local DB so
  search updates instantly without a full sync; nightly sync reconciles) and
  mandatory guardrails (explicit id-set scoping, dry-run→confirm, affected-
  count cap, reversible+audited via `location_source='manual'` /
  `generations`). Needs one new NAS endpoint (`bulk-set-tags`). Independent
  of M25. Reuses `debug-db.sh`'s rsync approach.

**Shipped, kept for reference:**
`docs/plans/bulk-set-location.md` (M19 inferred + `/geotag` manual) and
`docs/plans/faces-clustering-and-perf.md` (M18 clustering overhaul +
merge suggestions) are both complete. Each doc has a status header and
a "Future potential improvements" section for the genuinely-nice-to-
have items (structured location columns; HDBSCAN, quality pre-filter,
materialized face_groups). Pick those up only if concrete pain
surfaces.

<!-- moved verbatim from SKILL.md (line 2485) on 2026-10-19 -->
## Planned milestones (see `docs/plans/`)

Living roadmap entries — each is a design doc the next contributor can
pick up and implement without re-deriving the shape:

- **`docs/plans/infer-location-refinements.md`** — post-M19 cascade
  fixes surfaced by the first 127k-library apply. Three ordered
  changes: (1) cap hop depth at ~6 (observed chain ran 776 deep), (2)
  downweight inferred anchors on re-scan so decay protects cross-run
  compounding, (3) delete the dead `--max-cascade-rounds` flag and
  rename `cascade_rounds_used` → `max_hop_count` to match reality.
  All localized to `photosearch/infer_location.py`.
- **`docs/plans/search-accuracy-improvements.md`** — ranking +
  structural fixes for the search pipeline. Quick-wins bundle
  (~200 lines + schema v19 + one backfill CLI): RRF across filters,
  recency decay on the final sort, and structured
  `country`/`admin1`/`admin2`/`locality` columns so "Marin County"
  becomes a real filter and Nominatim round-trips are only needed
  for truly-new places. Iterate items: fuzzy name/place match,
  multi-candidate ambiguity resolution (GPS-prior tiebreaker), LLM
  query rewriter. Future: VLM re-ranking, self-hosted Nominatim.
- **`docs/plans/google-photos-import.md`** — **M20**. Takeout-based
  import of ~200K smartphone photos. The Library API read scopes were
  deprecated for third-party apps on March 31, 2025, so the API path is
  closed; Takeout is the only route. Incremental per-year export,
  composite dedup (photoTakenTime + camera model + filename stem + phash),
  lands in `/photos/YYYY/YYYY-MM-DD_gphotos/` to keep origin visible and
  rollback trivial. Adds `phash`, `import_source`, `google_photo_id`
  columns. New CLI `photosearch takeout-import` using a reviewable
  ndjson plan ledger for resumability. Phone GPS amplifies inferred-
  geotag recall on camera photos — run after each year's import.
- **`docs/plans/llm-driven-search.md`** — **M24 (SHIPPED)**. LLM-driven search:
  stop hand-assembling structured filters and let an LLM plan the query.
  Keystone is a **shared tool layer** (`photosearch/tools.py`) — a registry
  of `PhotoDB` wrappers (`search_photos`, `list_people`, `list_places`,
  `list_vocab`, `get_photo`, `get_photo_image`, `get_library_overview`)
  with JSON Schemas, consumed by two adapters: **M24a** a `FastMCP`
  streamable-HTTP MCP server in a new `photosearch-mcp` NAS container
  (reuses the photosearch image for CLIP; the "first step"), and **M24b**
  an in-app `POST /api/ask` SSE agent loop on the **local** LLM
  (`PHOTOSEARCH_TEXT_LLM_URL` + `PHOTOSEARCH_LLM_AGENT_MODEL`) surfaced as
  an "Ask" mode on the search page — fully local, nothing leaves the NAS.
  Image returns gated by `PHOTOSEARCH_MCP_ALLOW_IMAGES` (default off).
  Single-shot NL→filters fallback when the local model can't do multi-turn
  tool calls. Only existing-code change: an optional `person_ids` path in
  `search_combined`; no schema bump. Adds `mcp>=1.2` (~5-min rebuild).
- **`docs/plans/backfill-maintenance-sweep.md`** — **M25. Queued: do NOT
  start until M24b ships.** Derived-data passes get lost on new photos.
  Verified trigger map: `ingest-incoming` runs CLIP + (folder-scoped)
  stacking + geocode but `--no-colors` on the cron; the worker fleet does
  faces/quality/describe/tags only. So `normalize-places` (structured
  location cols), `infer-locations`, `recluster-faces`, `match-faces`,
  colors, and library-wide stacking are manual CLIs nothing schedules.
  Proposes one idempotent `maintenance-sweep` (dependency-ordered,
  missing-only SQL predicates, cron stage + `/status` SSE button,
  cancellable via the stacking `on_progress`/`should_abort` shape) that
  orchestrates the existing CLIs over the lightweight CPU backfills —
  leaving heavy GPU passes to the worker fleet. Note: auto-`recluster-faces`
  would wipe `ignored_clusters` nightly, so gate it to a slower cadence.
  Also `validate-data`/`repair-data` for invalid rows: corrupt `date_taken`
  control bytes found in M24a (repair cascade EXIF→folder→mtime→NULL), bad
  GPS, malformed JSON columns; reports orphaned-vec / garbage-tag rows and
  points at the existing `cleanup-orphans` / `clean-garbage-tags`.
- **`docs/plans/local-replica-and-writes.md`** — **M26**. Run the whole stack
  (web UI + `/api/ask` agent + MCP) on the strong GPU desktop off a **local
  read-replica** of the SQLite DB + a thumbnail mirror, with local LM Studio —
  because search is read-only, the NAS becomes a sync-only source of truth and
  all search compute moves to the fast machine. **M26a:** rsync the DB (≈1 GB)
  + `thumbnails/` (≈12 GB measured, append-only → tiny deltas; previews/full
  proxied from NAS), nightly + a "sync now" SSE endpoint/button; desktop has no
  originals so thumbnails are mirrored-or-lazy-proxied, never generated locally.
  **M26b:** write tools (set_photo_tags/location, add_to_collection) with a
  **read-local / write-NAS-authoritative / mirror-local** dual-write — the NAS
  endpoint applies and returns canonical values, the tool mirrors *those* to the
  local DB so search updates instantly, nightly sync reconciles (NAS write fails
  → abort, don't touch local; mirror fails → safe on NAS, lands next sync).
  Guardrails: explicit `photo_ids` scoping (no internal re-search),
  dry-run→`confirm=true`, affected-count cap, reversible+audited
  (`location_source='manual'` / `generations`). New NAS endpoint:
  `POST /api/photos/bulk-set-tags`. Reuses `debug-db.sh`'s rsync approach.
**Shipped, kept for reference:**
- **`docs/plans/bulk-set-location.md`** — both halves done (M19
  inferred + `/geotag` manual). Future potential: structured
  `country`/`admin1`/`admin2`/`locality` columns to unlock radius
  search and region-scoped queries.
- **`docs/plans/faces-clustering-and-perf.md`** — M18 clustering
  overhaul + merge suggestions + accept/reject UI all shipped. Future
  potential: quality pre-filter (`det_score` + bbox area), HDBSCAN
  for varying density, materialized `face_groups` table. Pick up only
  if a concrete pain surfaces — the current clustering has been good
  enough on 120k+ faces in day-to-day use.

When picking up any active plan, read the doc first — it contains the
shape the schema migrations should take and the UX calls that have
already been made.

---

# Handoff — 2026-10-09: search performance, request log, encoding cache

Written to start a fresh session cleanly. Everything under "Verified state" was
checked on 2026-10-09, not recalled. The work itself ran 2026-10-03 → 10-06.

The full design, measurements and decisions are in
**`docs/plans/search-indexes.md`**. The operating notes are in CLAUDE.md
("API request timing log", "Face-encoding cache", the v34 line under
Database) and in the `/photo-search` skill ("Search performance", Key pattern 10,
Troubleshooting → "Request log").

---

## What this work did, in one paragraph each

**Why it started.** On 2026-10-03, `/?camera=ILCE-7RM6&date_from=2026-09-26&date_to=2026-09-27`
took over 120 s on the NAS. `camera_model` had no index, and `search_combined` ran
every structured filter over the **whole library** as its own `SELECT *`, then
intersected the results in Python. With a cold cache on the N100's spinning disk,
each `SCAN photos` reads about 545 MB.

**Schema v34 (`31f87c7`).** 17 indexes (`db._SEARCH_INDEXES`) and 9 superseded
ones dropped (`_SUPERSEDED_INDEXES`). `_compose_scope` combines date, camera and
people into a single id query, and every other filter runs over those ids.
Dates are pushed into SQL, and `substr()` date filters are rewritten as ranges.
On the NAS the migration took 25 s, run in a one-off container with the web
container stopped. A backup was taken first: `/data/photo_index.db.bak-v33-20261004`.

**Phase 2 (`686869e`, `befe57e`).** People and LIKE queries look up ids from a
narrow index first, then read each row once. The aesthetics and `min_quality`
browses are paginated in SQL. People-only searches select `_NARROW_COLUMNS` and
read full rows for the returned page only.

**Request log (`95b6b9a`, `7eaa3c6`, `d1e7f25`).** Every `/api/*` request is
written to `request_log.jsonl` beside the DB; MCP tool calls go to
`request_log.mcp.jsonl`. Each line carries `source` and `intent`. Intent is
stated by Claude (header or MCP `intent` argument), or taken from the Ask
question, or inferred from the page, endpoint and parameters. Calls the replica
makes to the NAS are labelled `replica` and chain the cause:
`… - for: <the UI request>`. Read it with `photosearch request-stats`.

**Face-encoding cache (`7eaa3c6`).** `PhotoDB.get_face_encodings_cached` is a
process-wide LRU used by `face_suggest` and `face_verify`. It needs no
invalidation, because encodings are insert/delete-only and face ids are never
reused.

**Measured on the NAS, cold, right after a restart:**

| request | before | after |
|---|---|---|
| camera + 2 days | >120 s | 0.14–0.4 s |
| Calvin alone | 1.55 s | 0.35 s |
| More of this kid, Robert, one day | 6.1 s | 0.58 s |

Answers were checked against the old code: 19 + 37 + 9 cases on a replica copy.
They match, apart from photos with identical timestamps swapping places.

---

## Verified state (2026-10-09)

- **NAS:** `photosearch` runs `8803977`. That is the code at HEAD; `cbae933` on
  top of it is docs only. Up 46 h. `photosearch-mcp` is the same image, up 3
  days. Schema **34**.
- **Replica** (`localhost:8001`, systemd user unit `photosearch-replica`):
  process started 2026-10-06 16:27. DB synced 2026-10-09 03:32, schema 34.
  **It runs the code that was checked out when it started**, so after a pull,
  run `systemctl --user restart photosearch-replica`.
- **Request logs:** `/data/request_log.jsonl` (+ `.mcp.jsonl`) on the NAS,
  `./request_log.jsonl` on the replica. From 10-07 00:35 to 10-09 00:50 the NAS
  logged 25,023 requests: worker 24,953, claude 66, ui 3, replica 1. The UI was
  barely used in that window, so there is no real user-latency data yet.
- Other sessions have since shipped visual-tags / LLM-pass work (`6e7b4fd` →
  `cbae933`). None of it touches the search, index or log code.

---

## Backlog, in the order I'd take it

1. **DONE 2026-10-09** (worker probe → `/api/health`; `/api/stats` memoized,
   one scan instead of four; replica card → fingerprint). Original note:
   **`/api/stats` is the slowest thing in the NAS log, and the worker fleet
   causes it.** `WorkerClient.__init__` (`worker.py:~235`) checks the connection
   with `GET /api/stats`, which runs full-library COUNT/MIN/MAX scans. Every
   worker in a fleet calls it at the same moment. On 2026-10-07 07:59 that was
   **3 concurrent × 117 s**; other launches cost 3 × 6.7 s and 3 × 8.7 s. Fix:
   point the check at a cheap endpoint (`/api/admin/version`, or add a trivial
   `/api/health`). Separately, make `/api/stats` cheaper or cached, since the
   status page and replica-status (12.5 s on 10-09) use it too.
2. **PARTLY DONE 2026-10-09.** The NAS log showed every slow call omitted
   `passes=` (all 9 counts): Claude's idle checks (unscoped, 18–82 s) and
   collection-58 checks (16k ids, 19–33 s). Worker calls send `passes=` and
   take 0.01–0.2 s; the replica panel's polls were ~absent that week. Fixed
   the scoped case: above `db._SCOPE_DRIVE_FROM_INDEX_AT` (2000) ids the count
   writes `+id IN (...)`, driving from the need-index instead of a rowid
   lookup per id (replica, cold, 16k ids: 1.1 s → 0.00 s describe/verify/
   quality, 1.2 → 0.17 s clip/faces). The unscoped counts were already
   index-only (worst: faces 1.45 s cold on the replica); the 82 s was not
   reproduced — re-check `request-stats` after deploy before doing more.
   **When checking the queue by hand, pass `passes=`.** Original note:
   **`/api/worker/status` is slow cold: 8.6, 18.5, 43 and 82 s measured.** It
   counts nine worker queues over the whole library. Only describe, verify and
   quality have partial "work remaining" indexes. The likely costly ones are
   `clip` (`NOT IN clip_embeddings`, a vec0 read) and `faces`
   (`NOT EXISTS faces`). The replica's `/admin/maintenance` Workers panel polls
   it every 5 s through `/api/admin/workers/queue-status` (~4.1k polls in the
   week to 10-04). Profile each pass's count first, then cache it or give each
   count an index-only form.
3. **First calls after a restart are still slow:** suggest-person took 69 s and
   verify-labels 26 s cold, partly because the MCP container reads the library
   at the same moment. Options: warm the encoding cache in the background at
   startup, or persist it beside the DB. Only worth it if restarts turn out to
   matter in practice.
4. **Let a week of real use accumulate, then look at it:**
   `request-stats --source ui --slowest 20` and `--source claude-mcp`. The
   Ask/MCP path (`tools._build_filter_sql`) composes filters in SQL but still
   uses `LIKE` on the JSON tag columns. Check its real latencies before
   changing it.
5. **Push dates into the remaining standalone filters** in `search_combined`.
   This is the one open item in the plan doc. It is low value now that the
   composed scope covers the common combinations.

**Decided — don't relitigate:** the `idx_photos_aes_technical` / `_composition`
/ `_impact` indexes are **kept**. The SQL browse floors use them (owner,
2026-10-04).

---

## Traps worth not rediscovering

- **Expression and partial indexes match the query text.** Keep
  `COALESCE(aes_overall, aesthetic_score)`, `COALESCE(aes_subject_overall_pct,
  aes_overall_pct)` and the worker-count predicates verbatim.
  `tests/test_search_indexes.py` EXPLAINs the SQL production actually runs; run
  it after touching any of this.
- **The `+date_taken` / `+id` in the tag queries is deliberate.** Without it,
  a wide date range makes SQLite use `idx_photos_date` and read 399 MB.
- **Don't use the id-first form inside `tools._build_filter_sql`.** It made a
  broad place + one day read 259 MB instead of 8.7 MB.
- **Date bounds are `>= from` and `<= to || ' 23:59:59'`.** That is exact only
  because every `date_taken` is `YYYY-MM-DD hh:mm:ss` (161,309 of 161,309,
  checked). `cull.py` uses `|| '~'`, which also tolerates a `T` separator.
- **`_NARROW_COLUMNS`:** if the post-filter pipeline starts reading another
  column, it must be added there. The narrow-vs-full test catches a miss.
- **Encoding cache:** if anything ever updates an encoding in place, it must
  evict that face from the cache.
- **Replica → NAS calls must send `request_intent.outbound_headers(purpose)`.**
  New background threads or pools must go through `_carry_context` /
  `carry_context`, or their NAS calls lose the "for:" part. Test fakes for
  `requests.get` / `request` must accept `headers=`.
- **Claude sessions must state their intent** on every API call they make:
  `-H 'X-Photosearch-Source: claude' -H 'X-Photosearch-Intent: …'`. MCP calls
  fill the `intent` argument instead. This is in CLAUDE.md and in memory.
- **A schema-changing NAS deploy:**
  1. Check the fleet is idle.
  2. Stop the web container, so old code can't run its own migration against a
     half-migrated DB.
  3. Migrate in a one-off container.
  4. `up -d photosearch photosearch-mcp`; both run the same image.
  5. Then sync the replica and restart it.
- **The first `curl` to `/api/worker/status` after an idle spell can exceed
  60 s.** That is a slow response, not an outage (see backlog item 2).
- **NAS container stdout is wiped on every recreate.** Use the request log for
  history, never `docker logs`.

---

## Key files

| file | what |
|---|---|
| `photosearch/db.py` | `_SEARCH_INDEXES`, `_SUPERSEDED_INDEXES`, `_create_search_indexes`, `_FaceEncodingCache` |
| `photosearch/search.py` | `_compose_scope`, `_scope_clause`, `_date_bounds`, `_sql_page`, `_sort_sql`, `_aesthetic_floor_sql`, `_NARROW_COLUMNS`, `_hydrate` |
| `photosearch/request_log.py` | queue-backed rotating JSONL writer, `read_records` |
| `photosearch/request_intent.py` | `classify_source`, `infer_intent` (`_RULES`), `outbound_headers`, `carry_context` |
| `photosearch/web.py` | `_log_request_timing` middleware, `_carry_context` |
| `photosearch/mcp_server.py` | `_with_intent`, `call_tool_logged` |
| `cli.py` | `request-stats` |
| `tests/test_search_indexes.py`, `test_request_log.py`, `test_face_encoding_cache.py` | the guards |

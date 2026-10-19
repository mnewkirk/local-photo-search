# Search — history and traps

Database overview, v34 indexes, the API request log, name extraction. Condensed rules: `.claude/rules/search.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 23) on 2026-10-19 -->
## Database

File is `photo_index.db` (not `photos.db`). Schema version 39 (`SCHEMA_VERSION` in `db.py` is the source of truth;
v35–v39 are video — see "## Video").
v34 is indexes only (`db._SEARCH_INDEXES`; `docs/plans/search-indexes.md`):
camera+date, file_hash, covering tag composites, NOCASE location, /map GPS,
quality expression indexes, worker-count partials, faces `(photo_id, person_id)`
/ `(person_id, photo_id)`. Expression/partial indexes match the query TEXT —
`tests/test_search_indexes.py` EXPLAINs the SQL production runs, so keep those
expressions verbatim. `search_combined` composes date/camera/people into one id
scope (`_compose_scope`) before running the other filters, instead of running
each over the whole library and intersecting in Python. Key tables: photos, faces,
persons, face_references, face_person_exclusions, collections, collection_photos,
photo_stacks, stack_members, stacking_seen, review_selections, google_photos_uploads,
ignored_clusters, generations, schema_info, ingest_sweeps, ingest_batches,
ingest_batch_jobs.
(v23 split `tags` into `categories`/`visual_tags`/`keywords` + `tags_v22_backup`.)
(v25 added `photos.folder` — indexed dirname of `filepath`, populated in
`add_photo` + backfilled on migration — so the `/review` and `/geotag` folder
pickers `GROUP BY folder` instead of grouping every row in Python (M27). Note:
some deployed DBs were stamped `version=24` with no v24 column change, so the
folder migration is keyed v25 — the `try/except SELECT folder` adds the column
for both v23 and that stray-v24 state. Manual re-derive: `photosearch
backfill-folders [--apply] [--force]`.)

Schema migrations run automatically via `_init_schema()`. Bump `SCHEMA_VERSION` in `db.py`
when adding tables/columns.

`worker_processed` (schema v22) gained an `attempts INTEGER` column. The claim
path filters `attempts >= MAX_PROCESS_ATTEMPTS` (default 3) instead of `NOT
EXISTS`, so transient failures (HEIC files indexed pre-pillow-heif, runner-OOM
blips) auto-retry on the next pass; truly broken files stop being claimed after
N tries. `mark_processed` UPSERT-increments. This replaces the manual
`retry-failed-describe` workflow for newly-stuck photos — that CLI is still
useful for clearing historical pre-v22 markers (which all migrated to
attempts=1) when you want to give them a clean re-attempt counter.

`generations` (schema v21) is a provenance log — one row per LLM text artifact
(describe, category-content/visual, keywords, verify), with `photo_id`,
`text_type`, `generated_text`, `model_used`,
`model_version`, `created_at`. The `photos` table still holds the current
"promoted" description/tags; `generations` is the full history, so outputs from
different models can coexist with provenance. `worker_api.submit_results` logs a
row per artifact via `db.log_generation()`. Backfill existing data with
`photosearch backfill-generations` (idempotent).

<!-- moved verbatim from CLAUDE.md (line 155) on 2026-10-19 -->
## API request timing log

Every `/api/*` request is appended as one JSON line (ts, method, path, query,
status, ms, client) to `request_log.jsonl` **beside the DB**:
`/data/request_log.jsonl` on the NAS and `./request_log.jsonl` on the replica.
The container's stdout log is discarded on every redeploy, so this file is the
only lasting record of how real requests performed. It rotates at 20 MB with 5
backups. Set `PHOTOSEARCH_REQUEST_LOG` to a path to move it, or to `0` to turn
it off; the test suite turns it off in `conftest.py`. Writes go through a
queue to a background thread, so a starved disk never stalls a request. A
logging failure never fails the request. For SSE endpoints (`"streaming": true`),
`ms` is the time to the response headers, not to the end of the stream.

```bash
$DC run --rm photosearch request-stats [--last 500] [--since 2026-10-05] [--include-polling] [--slowest 10]
./.venv/bin/python cli.py request-stats --db photo_index.db.local   # replica
```

Page polling (`/batches`, maintenance status, worker queues) and the worker
fleet's own traffic are hidden unless `--include-polling` is given.

**Every line says who made the request and why** (`photosearch/request_intent.py`):

- `source`: an `X-Photosearch-Source` header wins. Otherwise `worker` for
  `/api/worker/*`, `ui` when the browser sent a `Referer`, `replica` for a
  `photosearch-*` User-Agent, `script` for any other Python HTTP client, and
  `other` for anything else.
- **Calls the replica makes to the NAS are `replica`, with a chained intent.**
  Every such call site sends `request_intent.outbound_headers(purpose)`. The
  intent is its own purpose plus the intent of the request that caused it,
  e.g. `Fetch preview of photo 248304 (not cached on the replica) - for: Review
  team faces (from the faces page)`. The middleware puts the current request's
  intent in a contextvar. Background threads and pools do not inherit it, so
  wrap them in `carry_context` (`web._carry_context` for the SSE threads,
  face_review's preview pool). A new replica→NAS call must use
  `outbound_headers`; a new background thread should be started through
  `_carry_context`. Workers label their session `worker`.
- `intent`: an `X-Photosearch-Intent` header wins. Next, a handler can set
  `request.state.log_intent`; `/api/ask` sets it to `Ask: <question>`, and each
  tool call the agent makes is logged with that intent too (`source: agent`).
  Otherwise the intent is inferred from the page, the endpoint and its
  parameters, and the line is marked `intent_inferred`. For example:
  `Search photos: person Calvin, camera ILCE-7RM6, 2026-09-26 to 2026-09-27
  (from the search page)`.
- **MCP tool calls** go to `request_log.mcp.jsonl` (`source: claude-mcp`): the
  MCP server is a separate process, and two processes must not rotate one file.
  Every tool advertises an optional `intent` argument. The server instructions
  ask Claude to fill it in, and it is stripped before the tool runs.
  `request-stats` merges both files.

**When a Claude session calls the HTTP API itself (curl, scripts), it MUST state
its intent**, because nothing can infer it:

```bash
curl -H 'X-Photosearch-Source: claude' \
     -H 'X-Photosearch-Intent: Time the camera+date search after the v34 deploy' \
     "http://<nas>:8000/api/search?..."
```

`request-stats --recent 50` lists recent requests with their source and
intent. `--source claude` shows only one source.

<!-- moved verbatim from CLAUDE.md (line 2187) on 2026-10-19 -->
## Name extraction in search queries

`search.py:_extract_persons_from_query` parses registered person names out of
the free-text `q=` parameter and turns them into AND-intersected person
filters. `?q=Calvin and Ellie` matches photos containing both people
(case-insensitive, word-bounded, longest-first so "Matt Newkirk" wins over
"Matt"), strips connector tokens ("and", "with", "&", ","), and sends any
residual to CLIP. `(?<!-)` lookbehind keeps `-Calvin` as a CLIP exclusion.
Stacks cleanly with dates/locations/explicit `person=` via `search_combined`'s
existing result-set intersection — no special casing.

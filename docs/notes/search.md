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

## Ask agent prompt

### 2026-10-26 — Ask eval baseline and mutation testing (evals/ask_eval)

Baseline at tag `pre-ask-diet`, qwen/qwen3.5-9b, 16,384-token context, fleet
idle: 154/189 pass (63 items x 3). First-round prompt 11,697 tokens on
average, largest round 16,318, and 9 runs failed on LM Studio "Context size has
been exceeded" with nothing else running: the multi-round chapter asks
(b01, b02, b05) overflow on their own. Items are 3/3 or 0/3 (temperature 0),
so items, not runs, are the sample.

Failing on the shipped prompt: no `sort=subject` for "in the foreground"; no
rerank for eyes/mouth or short hair; "last year" read as the past 12 months;
"top 40% per day" sent as `min_day_aesthetic` 40; pinned dates ignored;
no retry after a 0-hit search; "last summer" taken from the example as last
year's (owner: the most recent summer — example fixed after this run).

Mutation testing (one rep per mutant, all items) killed 23/23, but four items
(r10, w01, w02, x04) also flip under control edits that remove nothing (a
newline, a reworded first sentence, two examples swapped). Discounting them:
18/23. Survivors: rule.framing-flags (680 chars), rule.summarize (338),
rule.mood-visual-tag (252), grounding (220) — their behaviour passes without
them, likely because the tool schemas say the same — and rule.explain-answer
(271), which no item grades (a coverage gap, not a cut). x01, b04 and b06
still recur as killers of unrelated mutants and are probably fragile too;
firm attribution needs 3 reps per mutant. The two big tool trims both cost
behaviour: dropping all parameter descriptions (16,288 chars) broke 6 items,
de-duplicating the filter descriptions (6,520) broke d05 (and x01).

### 2026-10-27 — the Ask prompt trim, measured (evals/ask_eval)

Shipped: (1) `_lean_for_model` caps chapter / day / scene / bucket summaries
at 20 compact entries — the real cause of the overflows (one chapters call
sent 114 summaries, ~8k tokens); (2) the three long shared filter
descriptions (`min_aesthetic`, `min_day_aesthetic`, `style_tag`) are full on
search_photos only and hand-shortened elsewhere (`tools.SHORT_DESCRIPTIONS`),
each keeping its routing hint; (3) "last summer" = the most recent summer.
Result, 63 items x 3, qwen3.5-9b: 154/189 -> 162/189, context errors 9 -> 0,
first round 11,697 -> 10,918 tokens, peak 16,318 -> 15,724. Worse: only
x04 (fragile under control edits). Better: b01, b02, b05 (no overflow, and
"top 40% of each day" now = 60), s05.

Rejected, with evidence: single-search-tool mode (`--consolidated`) — 7,244
first-round tokens but 9 items regressed (134/189, sign p 0.057). Automatic
first-clause shortening of shared descriptions — dropped "a plain 'best' is
sort='quality_desc' with NO floor" from five schemas and moved d02/d05 off
the flat search. Cutting the four rules the mutation run marked unneeded
(framing flags, summarize, mood tag, grounding; ~1.5k chars) — a
single-factor run showed it cost b06 (Lake Como recovery) and d01. Items flip
chaotically under small prompt edits at temperature 0: localise every change
with one-factor configs before believing it.

### 2026-10-28 — Ask ablations on the shipped prompt; local model comparison

qwen3.5-9b, 23 deletions + 6 controls on HEAD: 10 of 63 items flip under
edits that remove nothing (newline, one word, two rules swapped, tool order
reversed) — about 16% of asks depend on incidental prompt details at
temperature 0. Discounting them, 12/23 deletions are caught: the tool
descriptions, the rules block, examples, facts, hints, and the foreground /
read-total / representatives / rerank rules. Eleven single rules do not move
qwen beyond noise; it routes mainly from the tool descriptions.

gemma-4-12b-qat as the Ask model: 77/189 vs qwen 162/189, 17.7 s vs 3.4 s
per Ask. 129 of its 506 tool calls (25%) arrive with invalid argument keys —
gemma's native tool-call tokens (`<|"|>`, `<|tool_call>`) leak through
LM Studio's parser (`{"people=[<|"|>Nicole<|"|>,...": 1}`); 45 of its 112
failing runs contain one, the rest are worse routing. Do not use gemma-4 for
tool calling on this LM Studio setup; it stays fine for the text passes,
which use no tools. Its ablation series was stopped as uninformative.

# Adding features

Read before adding a CLI command, search type, indexing pass, API endpoint, schema change or frontend page, or when running the tests like CI.

<!-- moved verbatim from CLAUDE.md (line 2357) on 2026-10-19 -->
## Running the tests the way CI does

`./scripts/test-like-ci.sh` — use this before pushing. A plain `pytest` run on a
dev box diverges from GitHub Actions in three ways, and **each one has hidden a
real failure**:

1. **No Topaz CLI on CI.** With Topaz installed, `POST
   /api/admin/upscale-photo` returns 200; on CI `cli_path()` raises and it
   returns **503**. A test asserting 200 passed locally and failed on CI for
   five commits. Tests that exercise that route must take the `fake_cli`
   fixture.
2. **Minimal dependencies.** CI installs no torch/cv2/insightface —
   `tests/conftest.py` mocks them — so code paths differ from a dev venv that
   has the real thing. The script builds a venv with CI's exact list.
3. **Nothing ignored.** `--ignore=` is tempting when a file fails for local
   reasons, and it silently stops testing that file. CI ignores nothing.

Known local-only failure: `tests/test_admin_deploy_mode.py` reads HEAD through
git, and a **git worktree created from Windows** has a `\wsl.localhost` UNC
gitdir that git cannot resolve from inside WSL — so those two tests fail in a
worktree and pass on CI. The script detects it and says so. Run from the main
checkout for a clean result.

<!-- moved verbatim from SKILL.md (line 245) on 2026-10-19 -->
## Key Patterns

These patterns were established through iteration — always follow them.

<!-- moved verbatim from SKILL.md (line 249) on 2026-10-19 -->
### 1. Streaming generators for long-running batch jobs
Any function that processes thousands of photos must yield results incrementally.
Progress is saved to the DB per-batch (survives cancellation) and printed to the log.

```python
# The established pattern in quality.py / clip_embed.py:
def score_photos_stream(image_paths, batch_size=8):
    for batch_start in range(0, total, batch_size):
        # ... process batch ...
        for idx, result in enumerate(batch_results):
            yield batch_start + idx, result
```

<!-- moved verbatim from SKILL.md (line 262) on 2026-10-19 -->
### 2. PHOTOSEARCH_DB env var for --db defaults
Every CLI command's `--db` option must include `envvar="PHOTOSEARCH_DB"`.

<!-- moved verbatim from SKILL.md (line 265) on 2026-10-19 -->
### 3. Filename auto-detection in search
Queries that look like camera filenames (e.g., `DSC06241`) are detected by
`_looks_like_filename()` in `search.py` and routed to SQL LIKE instead of CLIP.

<!-- moved verbatim from SKILL.md (line 269) on 2026-10-19 -->
### 4. File exclusions in find_photos()
`index.py:find_photos()` skips macOS AppleDouble sidecars (`._*`) and numbered
copies (`DSC_1.JPG` when `DSC.JPG` exists).

<!-- moved verbatim from SKILL.md (line 273) on 2026-10-19 -->
### 5. Frontend — no build step
`frontend/dist/` contains plain HTML with React loaded from CDN (UMD build). Edit HTML
files directly. Use vanilla JS (`React.createElement`) not JSX. Changes require a Docker
image rebuild since files are baked in via `COPY frontend/ frontend/`.

<!-- moved verbatim from SKILL.md (line 278) on 2026-10-19 -->
### 6. Docker cache awareness
`requirements.txt` changes invalidate the pip install layer (~5 min on N100). Python file
changes only invalidate `COPY` (~10s). CPU-only PyTorch is pinned via
`--extra-index-url https://download.pytorch.org/whl/cpu` at the top of `requirements.txt`.

<!-- moved verbatim from SKILL.md (line 283) on 2026-10-19 -->
### 7. SSE for long-running API operations
Google Photos upload uses Server-Sent Events (SSE) via `StreamingResponse` with
`text/event-stream`. Cross-thread communication uses `asyncio.Queue` + `threading.Event`
for cancellation. Key gotcha: use `asyncio.get_running_loop()` (not `get_event_loop()`)
in async endpoints, and always include terminal events (`done`, `fatal`, `cancelled`)
in the generate() function to close the stream.

<!-- moved verbatim from SKILL.md (line 290) on 2026-10-19 -->
### 8. Per-file ledger writes for cancel safety
Operations that can be cancelled mid-way (like uploads) write per-file results to the
DB immediately rather than batching at the end. This ensures partial progress survives
cancellation.

<!-- moved verbatim from SKILL.md (line 295) on 2026-10-19 -->
### 9. Face IDs are stable, cluster IDs are not
`faces.id` is `INTEGER PRIMARY KEY AUTOINCREMENT` — never reused, survives every
`recluster-faces`. `cluster_id` is fully renumbered on each recluster, so anything
that needs to persist across reclusters (merge dismissals, suggestion-file
identity) keys on `rep_face_id` pairs, not cluster_ids. `/merges` localStorage
dismissals and the `resolveTarget()` rewrite logic both exploit this. Same rule
applies when designing future tables: use `face_id` as the stable anchor if you
need cross-recluster persistence.

---

<!-- moved verbatim from SKILL.md (line 306) on 2026-10-19 -->
### 10. Claude states the intent of every API call it makes

Every `/api/*` request is logged with its timing, `source` and `intent`
(`request_log.jsonl`, see "Request log" under Troubleshooting). UI intent is
inferred; Claude's cannot be. So any curl/script call from a Claude session
sends:

```bash
curl -H 'X-Photosearch-Source: claude' \
     -H 'X-Photosearch-Intent: <one sentence: what you are finding out and why>' ...
```

MCP tool calls fill the optional `intent` argument instead.

<!-- moved verbatim from SKILL.md (line 2583) on 2026-10-19 -->
## Adding New Features

<!-- moved verbatim from SKILL.md (line 2585) on 2026-10-19 -->
### New CLI command
```python
@cli.command("my-command")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
def my_command(db):
    """One-line description."""
    with PhotoDB(db) as photo_db:
        pass
```

<!-- moved verbatim from SKILL.md (line 2595) on 2026-10-19 -->
### New search type
1. Add `search_by_X()` in `search.py`, taking `scope=None` and applying it
   with `_scope_clause` so it runs over the composed id scope, not the
   whole library
2. Wire into `search_combined()` — add to `result_sets`, count it in
   `n_structured`, and add it to the `narrow` exclusion list (only people
   filters may run narrow)
3. If the new filter has an index-backed exact form (equality / range),
   consider adding it to `_compose_scope`; push any date range into its SQL
   with `_date_bounds`
4. Add query param to `web.py:api_search()` and a rule to
   `request_intent._filters` so its intent reads well in the request log
5. Add UI control in `frontend/dist/index.html`
6. Add a case to `tests/test_search_indexes.py` (`COMBOS`) and EXPLAIN its SQL

<!-- moved verbatim from SKILL.md (line 2610) on 2026-10-19 -->
### New indexing pass
1. Add processing function in `photosearch/` module
2. Use streaming generator pattern (Key Patterns §1)
3. Add `enable_X` / `force_X` params to `index_directory()` in `index.py`
4. Add the same pass to `_index_collection()` in `index.py` (collection mode)
5. Add `--X` / `--force-X` flags to `index` command in `cli.py`
6. Add to status page run-commands in `status.html`

<!-- moved verbatim from SKILL.md (line 2618) on 2026-10-19 -->
### New API endpoint
1. Add route in `web.py` with appropriate method + path
2. Use `with _get_db() as db:` for database access
3. For long-running ops, consider SSE pattern (Key Patterns §7). Reference
   implementations:
   - **Google Photos upload** (`web.py:api_upload_google`) — per-file
     progress over slow network + per-file ledger writes
   - **Stacking detect stream** (`web.py:api_detect_stacks_stream`) — pure
     CPU job with multiple phases. Pass `on_progress` + `should_abort`
     through to the worker function (see `stacking.py:run_stacking`);
     throttle emission to ~0.25s; check abort at every phase transition
     and every ~1000 iterations inside hot loops; raise `InterruptedError`
     on abort; terminal events `done` / `cancelled` / `fatal`.
4. Add a rule to `request_intent._RULES` so the request log can say what a
   UI call to it was for (the fallback is just `METHOD /path`)
5. Add frontend integration in the appropriate HTML file. For POST+SSE,
   use `fetch` + `response.body.getReader()` + `\n\n`-split parsing
   (EventSource can't POST); keep an `AbortController` around so the UI
   can cancel. See `frontend/dist/status.html:StackingForm` for a
   minimal template.

<!-- moved verbatim from SKILL.md (line 2639) on 2026-10-19 -->
### Schema changes
1. Bump `SCHEMA_VERSION` in `db.py` (currently 34). New indexes go in
   `db._SEARCH_INDEXES` (built one commit at a time); replaced ones in
   `_SUPERSEDED_INDEXES`
2. Add `CREATE TABLE IF NOT EXISTS` or `ALTER TABLE` in `_init_schema()`
3. Ensure migration SQL appears after any table it depends on
4. Add test in `tests/test_db.py` that creates a minimal old-version DB and verifies
   the migration runs correctly

<!-- moved verbatim from SKILL.md (line 2648) on 2026-10-19 -->
### New frontend page
1. Create `frontend/dist/<page>.html` using the pattern from existing pages
   (plain React UMD, `React.createElement`, no build step)
2. Include `<script src="/shared.js"></script>` and use `PS.SharedHeader`
   with a new `activePage` id
3. Add a route in `photosearch/web.py` near the other `@app.get("/<page>")`
   block that serves the HTML with `Cache-Control: no-cache`
4. Add the page's nav id to `PS.SharedHeader`'s `navLinks` in `shared.js`
   so the link appears on every page
5. Styles live inline in each `<page>.html`; if you're reproducing a pattern
   from merges.html / faces.html (sticky toolbar, kbd hints, face-strip,
   etc.), copy the CSS rather than inventing

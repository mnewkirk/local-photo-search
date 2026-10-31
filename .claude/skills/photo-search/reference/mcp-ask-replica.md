# MCP server, Ask agent, local replica

Read when working on the MCP server (M24a), the in-app Ask agent (M24b) or the desktop read-replica (M26a).

<!-- moved verbatim from SKILL.md (line 2249) on 2026-10-19 -->
## MCP server (M24a)

LLM-driven search over the Model Context Protocol. Lets any MCP client on
the LAN plan a photo search instead of hand-assembling structured filters.
Full design: `docs/plans/llm-driven-search.md`.

**Shared tool layer** — `photosearch/tools.py` is the single source of truth:
a registry of `ToolSpec`s (name + LLM-facing description + JSON Schema +
`fn(db, args)` handler) over `PhotoDB`, dispatched by `call_tool(db, name,
args)`. `search` (hence CLIP/torch) is imported lazily inside handlers, so
the module imports cheap and is unit-testable without the CLIP stack. The
same registry feeds both the MCP server (`mcp_tools()`) and the in-app
`/api/ask` agent (`openai_tools()`).

**Intent logging:** the MCP server adds an optional `intent` property to every
tool's schema (`mcp_server._with_intent`) and its instructions ask Claude to
fill it in. `call_tool_logged` strips it before the tool runs and logs the call
(timing, args, intent) to `request_log.mcp.jsonl` with `source: claude-mcp`.
The agent's tools never see `intent`.

**Current tools** (`all_tools()` order): `get_library_overview`,
`list_people`, `list_places`, `list_vocab`, `search_photos`, `summarize`
(facet/COUNT by year/month/location/person/camera_model), `representatives`
(top-N best per bucket — "one per year"; `max_buckets` cap;
`rank_by=quality|subject`), `daily_highlights` (best-per-day, burst/near-dup
collapsed, chronological), **`group_into_chapters`** (split a trip into
place-based book chapters), **`daily_scene_breakdown`** (split one day into
sub-scenes), **`suggest_layout`** (a house-style spread/layout plan for an
explicit photo set), `get_photo`, `get_photo_audit` (change history of a photo or collection), `rerank_photos` (VLM re-rank of a candidate
set), `get_photo_image` (image-gated), plus the M26b **write** tools
`set_photo_location` / `set_photo_tags` / `add_to_collection` (gated by
`PHOTOSEARCH_ALLOW_WRITES`; dry-run→`confirm=true`). The last three
book-curation tools are the "Shutterfly photo-book tools for Ask" — chapter
grouping mirrors the trip-book structure, scene breakdown the "sizing to the
day" cut, and `suggest_layout` the archetypes in
`~/shutterfly-proofs/BOOK_STYLE_GUIDE.md` (matched 2-up / asymmetric collage /
dense grid / full-spread panorama, hero = highest-aesthetic frame).

**One shared filter vocabulary.** Every search-like tool honors the same
structured filters as `search_combined`, via `_build_filter_sql` and the
`_CURATION_FILTER_PROPS` JSON-Schema fragment spliced into each tool's
`properties` — so `search_photos`/`representatives`/`daily_highlights`/
`summarize`/`group_into_chapters`/`daily_scene_breakdown` all accept
`camera`, `min_aesthetic` (VLM percentile), `style_tag` (and `search_photos`
also `match_source`), not just the original people/location/date/vocab set.
Adding a filter here means adding it to `_build_filter_sql` + the fragment,
and to `search_combined` for the `search_photos` path. Tests:
`tests/test_tools.py`.

`search_photos` resolves `people` names → ids and passes them to
`search_combined`'s `person_ids` param (routes to `search_by_all_persons` for
AND-intersection). It returns compact hits (`description` truncated to 240
chars, plus `camera_model` for the grid 📷 badge) + a `total` so the model
can broaden/narrow.

**Server** — `photosearch/mcp_server.py` uses the SDK's **low-level
`Server`** (not `FastMCP`) so the `tools.py` JSON Schemas are fed in verbatim
via `mcp_tools()` rather than re-derived from function signatures. Streamable
HTTP, mounted at `/mcp`, stateless. `requirements.txt` adds `mcp>=1.2`
(~5-min image rebuild).

**Container** — `photosearch-mcp` in `docker-compose.nas.yml`. Reuses the
photosearch service's built image (`image: photosearch-photosearch`, no
build block of its own → no redundant rebuild) with `command: ["mcp"]`. That
routes through a new `mcp)` case in `docker-entrypoint.sh` which execs
`python -m photosearch.mcp_server` (the `*)` branch would mis-route it to
`cli.py mcp`). Mounts `/photos:ro` + the shared `photosearch-data` volume
(DB + thumbnail cache). Port `${MCP_PORT:-8848}`.

```bash
DC="docker compose -f docker-compose.nas.yml"
$DC build photosearch        # build the shared image (mcp reuses it)
$DC up -d photosearch-mcp     # start the MCP server
# client connects to http://<nas-ip>:8848/mcp
```

**Image gating** — `get_photo_image` returns thumbnail bytes and is the only
path that emits pixels. **Off by default**: `PHOTOSEARCH_MCP_ALLOW_IMAGES`
unset/0 means the tool is neither advertised in `list_tools` nor callable
(enforced at both the projection and the call boundary). Set to `1` only if
you accept thumbnails leaving the NAS to whatever client is connected. The
text tools always return derived metadata (descriptions, tags, places,
dates, scores) — that text leaves if an off-box client consumes it.

**Env:** `PHOTOSEARCH_DB` (required), `PHOTO_ROOT`, `PHOTOSEARCH_MCP_HOST`
(default 0.0.0.0), `PHOTOSEARCH_MCP_PORT` (default 8848),
`PHOTOSEARCH_MCP_ALLOW_IMAGES` (default 0). Two processes now open the DB
(web + MCP); both read-only for search, WAL handles concurrent readers — no
schema change.

---

<!-- moved verbatim from SKILL.md (line 2341) on 2026-10-19 -->
## In-app "Ask" agent (M24b)

Natural-language search on the **local** LLM — an "✨ Ask" toggle on the
search page (`/`). The user describes what they want; a server-side agent
loop plans the search with the same tool layer the MCP server exposes.
Nothing leaves the NAS.

- **`photosearch/agent.py`** — `run_agent(db, message, history, should_abort,
  locked_filters=None)` is a synchronous generator yielding event dicts
  (`tool_call` / `tool_result` / `photos` / `answer` / `error`). It has its
  **own** tool-calling chat client (`_chat`) because describe.py's client is
  text-only — routes to `PHOTOSEARCH_TEXT_LLM_URL` (OpenAI-compatible, the LM
  Studio path) with `tools=`/`tool_choice=auto`, else Ollama. Loop capped at
  `_MAX_STEPS=6`. The `photos` event now also fires for `representatives` /
  `daily_highlights` / `group_into_chapters` / `daily_scene_breakdown` /
  `suggest_layout` results, not just `search_photos`.
- **Filters as HARD Ask inputs (`locked_filters`):** the structured Search
  filter bar is fed in as `filters` on `POST /api/ask` and ENFORCED, not used
  as a post-filter on results. `_normalize_locked` keeps the recognized keys;
  `_merge_locked` injects them into every filter-consuming tool call (`people`
  unions, scalars hard-override) BEFORE dispatch; `_locked_prompt` adds an
  "ACTIVE FILTERS" note to the system prompt so the model plans the rest of the
  request around them. This is why a pinned camera is guaranteed applied even
  if the model omits it (the "KODAK PIXPRO WPZ2 → non-Kodak photos" bug: the
  tool schemas hadn't exposed `camera`, so the model silently dropped it).
  Filter-consuming tools: `search_photos`, `representatives`,
  `daily_highlights`, `summarize`, `group_into_chapters`,
  `daily_scene_breakdown` (`_FILTER_TOOLS`). Pinned filters are logged in the
  per-query Ask log.
- **Model:** `PHOTOSEARCH_LLM_AGENT_MODEL` (role-resolved, falls back to
  `PHOTOSEARCH_LLM_TEXT_MODEL` → `PHOTOSEARCH_TEXT_LLM_MODEL` → `llama3.1`).
  Needs a **tool-calling-capable** model (qwen2.5-instruct / qwen3 /
  llama-3.1+ with LM Studio tool use on).
- **Single-shot fallback:** `PHOTOSEARCH_AGENT_SINGLE_SHOT=1` forces a one-shot
  NL→`search_photos`-args JSON completion (no loop). Also kicks in
  automatically if the first tool-calling round-trip throws or the model
  returns neither tool calls nor prose — so the feature degrades gracefully on
  a model that can't tool-call.
- **`POST /api/ask`** (web.py, SSE) bridges the sync generator through the
  thread→`asyncio.Queue` pattern (same as the stacking stream), with
  `is_disconnected()` cancellation and a terminal `{"type":"done"}`. Body
  `{message, history?, filters?}` — `filters` is the pinned filter bar (see
  `locked_filters` above).
- **Frontend:** the "✨ Ask" toggle in `index.html` swaps the search bar to an
  NL box but **keeps the structured filter bar visible** (labeled "✨ Pinned to
  Ask"); `runAsk` collects the filter state into `askFilters` and sends it as
  `filters`, and mirrors both the query and the pinned filters into the URL
  (`?ask=…&camera=…&min_quality=…`) so an Ask is reloadable/shareable.
  Narration streams into a status panel and the final `photos` event feeds the
  **existing** `PhotoGrid` via `setResults` (compact hits share the grid's photo
  shape, including `camera_model` for the 📷 badge).
- **Grouped ("photobook") results.** The book/curation tools tag each hit with a
  grouping key and return an ordered group summary; `agent._grouping_for` turns
  that into `group_field` + `groups` on the `photos` event
  (`group_into_chapters`→`chapter`, `daily_scene_breakdown`→`scene`,
  `daily_highlights`→`day`, `representatives`→`bucket`). `PhotoGrid` renders
  labeled sections (header = label + sublabel like a date span / places) instead
  of a flat grid — so chapters/days/scenes are actually visible, like Review's
  Timeline. **Structured search has the same sectioning** via a client-side
  "Group by" control (None / Day / Month / Location / Camera, `?group_by=`): when
  `groups` isn't supplied (no Ask tool grouping), `PhotoGrid` auto-derives
  sections from the loaded results in encounter order, keyed by the chosen
  attribute (`day`/`month` derived from `date_taken`, else a raw column like
  `place_name`/`camera_model`). In Ask mode a tool's grouping wins; a "Group by"
  selection is the fallback when no tool grouped.
- **Per-day aesthetic filter.** `min_day_aesthetic` (0-100) floors on the
  PER-DAY percentile `COALESCE(aes_subject_overall_day_pct, aes_overall_day_pct)`
  — how a photo ranks among others taken the SAME day — vs `min_aesthetic`'s
  library-wide `aes_overall_pct`. Threaded through `search_combined`
  (`_filter_aesthetic`), `/api/search`, the tool filter set, and the "Min day %"
  UI input; pinnable into Ask. Use it for "best of each day" on a trip whose
  whole days sit below the library median.
- **Tests:** `tests/test_agent.py` (10 cases, chat client mocked; tool dispatch
  runs for real against the fixture DB). **Still TODO:** a live smoke test
  against the LM Studio backend once deployed.

---

<!-- moved verbatim from SKILL.md (line 2419) on 2026-10-19 -->
## Running off a local read-replica (M26a)

Run the whole app (web UI + ✨ Ask agent + optional MCP) on the strong GPU
machine off a synced read-replica — CLIP search runs on the local GPU; the NAS
stays the source of truth. Image routes proxy thumbnails from the NAS on demand
(`PHOTOSEARCH_NAS_URL`), so **no photo files are needed locally**. Design:
`docs/plans/local-replica-and-writes.md`.

**Prereqs:** the project venv with the real ML stack (the worker machines
already have torch + open-clip + sqlite-vec); ssh access to the NAS (same as
`debug-db.sh`); LM Studio running with a **tool-calling-capable** model loaded.

**Turnkey launcher** — `run-local-replica.sh` (cross-platform; auto-resolves the
LM Studio endpoint):

```bash
# WSL2 (this dev box) — LM Studio runs on the WINDOWS host; the script finds it
# via the default-route gateway automatically.
./run-local-replica.sh --sync --model qwen/qwen2.5-7b-instruct
#   --sync       pulls a fresh replica (sync-replica.sh) first (~1.6 GB / ~128 s)
#   then serves http://localhost:8001 off ./photo_index.db.local
#   (gitignored). Subsequent runs: drop --sync to start instantly on the
#   existing replica.

# macOS — same command; the script defaults LM Studio to http://localhost:1234.
./run-local-replica.sh --sync --model qwen/qwen2.5-7b-instruct
```

Key flags: `-p PORT` (default 8001), `--db PATH`, `--nas URL` (default
`http://dxp4800-f976:8000`), `--lm URL` (override the auto endpoint), `--model`.

**Manual equivalent** (what the launcher exports):

```bash
export PHOTOSEARCH_DB=./photo_index.db.local
export PHOTOSEARCH_NAS_URL=http://dxp4800-f976:8000          # image proxy + drift
export PHOTOSEARCH_TEXT_LLM_URL=http://<lm-studio-host>:1234/v1   # WSL2: Windows-host gateway IP; Mac: localhost
export PHOTOSEARCH_LLM_AGENT_MODEL=<tool-capable model loaded in LM Studio>
./.venv/bin/python cli.py serve --db "$PHOTOSEARCH_DB" --host 0.0.0.0 --port 8001
```

On WSL2 the Windows-host IP is `ip route show default | awk '{print $3}'` (it can
change across WSL restarts — the launcher resolves it each run). **Do NOT set
`PHOTO_ROOT`** — leaving it unset means the originals resolve to non-existent
paths, which triggers the NAS image proxy (intended). Phones reach the machine
over Tailscale.

**Nightly sync** — schedule `sync-replica.sh`:
- WSL2 / Linux cron: `0 4 * * * cd /home/mattn/local-photo-search && PHOTOSEARCH_DB=./photo_index.db.local ./sync-replica.sh >> /tmp/replica-sync.log 2>&1`
- macOS: a launchd agent (or `cron`) running the same line.
- On-demand: the **Local replica** card on `/status` ("Sync now" →
  `POST /api/admin/replica-sync`), or just `./run-local-replica.sh --sync`.

**Notes:** first view of a photo proxies its thumbnail from the NAS (slightly
slower), then it's cached locally. The atomic DB swap means a running `serve`
picks up a fresh sync on its next request. If the Ask agent's model can't
tool-call, it auto-falls-back to single-shot NL→filters.

---

**Replica code vs replica data.** `sync-replica.sh` replaces the DB (schema
and indexes come with it); the server only picks up new **code** on restart
(`systemctl --user restart photosearch-replica`, port **8001**). The service
starts without `--sync`, so after a NAS deploy that changes the schema: sync,
then restart. The replica writes its own `./request_log.jsonl`.

### Corrections found while condensing (2026-10-19)

Checked against the code while writing `.claude/rules/`; the text above is kept verbatim.

- Write tools are ON unless `PHOTOSEARCH_ALLOW_WRITES` is set falsy (`mcp_server._writes_allowed`, `agent.py`); "gated by" above overstates it.
- `PHOTOSEARCH_MCP_ALLOW_IMAGES` defaults off in `mcp_server.py`, but `docker-compose.nas.yml` sets it to `1`.

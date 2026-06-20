# LLM-Driven Search + MCP Server (M24)

**Status:** ✅ **SHIPPED.** M24a (shared tool layer + MCP server) and M24b
(in-app `/api/ask` agent + "Ask" mode) are both implemented and tested. M24a
was verified live against the 163k-photo NAS library; M24b ships with a
mocked-LLM test suite (`tests/test_agent.py`) and needs a smoke test against
the live LM Studio backend once deployed.
**Decisions locked (2026-06-20):**
- **Consumer model:** local-only is the daily driver. The headline UX is an
  in-app "Ask" box backed by a *server-side agent loop on the existing local
  LLM* (LM Studio / Ollama via `PHOTOSEARCH_TEXT_LLM_URL`). Nothing — metadata
  or pixels — leaves the NAS on that path. This preserves CLAUDE.md line 1
  ("Photos never leave the machine").
- **MCP transport:** a streamable-HTTP MCP server runs as a **new container in
  `docker-compose.nas.yml`**, reachable on the LAN. It is the "first step"
  deliverable and shares the same tool layer as the in-app agent.
- **Image returns:** tools *can* return thumbnail bytes, but image-returning is
  **off by default and gated** (`PHOTOSEARCH_MCP_ALLOW_IMAGES`). Local agent =
  always safe to look at pixels; MCP cloud-ish clients only get pixels if the
  operator flips the flag.

---

## 1. Motivation

Search today forces the caller to map intent onto a fixed set of structured
filters — `person=`, `location=`, `date_from`/`date_to`, `category=`,
`visual_tag=`, `keyword=`, `color=`, `min_quality=`, `sort=` — plus a free-text
`q` that `search_combined()` heroically post-processes (it already extracts
registered names, dates, and place names from `q` via
`_extract_persons_from_query`, `date_parse`, and `geocode.extract_location_from_query`).

That heuristic extraction is good but brittle and one-shot: it can't *ground*
itself ("is there a person named Cal or Calvin?"), can't *iterate* ("that
returned 4000 photos, tighten to the Italy trip"), and can't *verify* ("of these
30 candidates, which actually show a birthday cake?"). The user wants to stop
hand-assembling filters and just describe what they want.

The reframe: **let an LLM be the query planner.** It reads the natural-language
request, grounds itself against the library (what people/places/tags exist),
composes the structured filters, inspects result counts, refines, and —
optionally — looks at candidate thumbnails to re-rank. The user talks; the model
drives the existing search machinery.

## 2. Architecture — one tool layer, two adapters

The keystone is a **shared tool layer** (`photosearch/tools.py`): a registry of
plain Python functions over `PhotoDB`, each paired with a JSON Schema describing
its arguments. Defined once, consumed twice:

```
                       photosearch/tools.py
              (TOOLS registry + call_tool(db, name, args))
                          /                    \
                         /                      \
        photosearch/mcp_server.py        web.py  POST /api/ask
        (FastMCP, streamable HTTP,       (SSE agent loop on the
         new NAS container)               local LLM, in-process)
                 |                                  |
        any MCP client on the LAN         "Ask" box in index.html
        (off-box only if operator         (nothing leaves the NAS)
         opts in to images)
```

Both adapters call the *same* `call_tool()` dispatch, so there is exactly one
definition of what "search" means to an LLM, and the JSON Schemas that the MCP
server advertises are byte-identical to the ones the in-app agent hands its
model in the OpenAI `tools=[...]` field.

The in-app agent calls the tool layer **in-process** (no MCP network hop, no
double serialization) — MCP is one adapter, not a required middle layer.

### 2.1 The tools

Descriptions below are for an LLM audience — in the real code they get
expanded, opinionated docstrings (the LLM's behavior is only as good as these).

| Tool | Purpose | Returns |
|---|---|---|
| `get_library_overview` | Ground the model: total photos, `date_taken` min/max, counts described / tagged / with-faces / with-GPS. Always call this first. | small dict |
| `list_people` | Registered persons + photo counts. The model MUST consult this before filtering by a name — it can't know "Calvin"/"Ellie" exist otherwise. | `[{name, photo_count}]` |
| `list_places` | Distinct `place_name` + the structured `country`/`admin1`/`admin2`/`locality` values with counts. Lets "our Italy trip" resolve to real strings the DB holds. | `[{value, kind, count}]` |
| `list_vocab` | Distinct `categories` / `visual_tags` / `keywords` with counts. Stops the model from guessing tags that don't exist. | `{categories, visual_tags, keywords}` |
| `search_photos` | The workhorse. Accepts the full structured filter set (see below) and returns compact hits + a `total`. The model fills whichever filters it inferred and reads `total` to decide whether to broaden/narrow. | `{total, results:[…]}` |
| `get_photo` | Full detail on one id: description, categories/keywords, EXIF, GPS/place, who's in it (faces→persons), stack info. For drill-down. | dict |
| `get_photo_image` | Thumbnail **bytes** for one id (for vision re-rank/verify). Gated — see §5. | image content |

`search_photos` arguments (a thin, LLM-friendly projection of
`search_combined`):

```jsonc
{
  "query":       "free-text semantic query (CLIP); omit if pure-filter search",
  "people":      ["Calvin", "Ellie"],      // AND-intersection; resolved to person_ids
  "location":    "Marin County",            // place_name / structured-column match
  "date_from":   "2025-06-01",              // YYYY-MM-DD
  "date_to":     "2025-06-30",
  "color":       "blue",
  "category":    "landscape",
  "visual_tag":  "golden hour",
  "keyword":     "birthday",
  "min_quality": 6.0,
  "sort":        "relevance",               // date_desc|date_asc|quality_desc|relevance
  "limit":       30
}
```

`results` items stay compact (the model pays for every token): `id`,
`filename`, `date_taken`, `place_name`, `description` (truncated ~240 chars),
`categories`, `aesthetic_score`, `score`, `thumbnail_url`. The `thumbnail_url`
lets the *frontend* render the grid without the model ever holding pixels.

### 2.2 One tiny `search_combined` extension

`search_photos`'s `people` list maps cleanest to a new optional
`person_ids: list[int]` path in `search_combined` that routes to the existing
`search_by_all_persons` (AND-intersection) — instead of stuffing names back into
`query` and relying on `_extract_persons_from_query` to re-parse them. The tool
layer resolves names→ids via `list_people` data and passes `person_ids`. This is
~15 lines in `search.py` and is the only change to existing search code; all
other tools are read-only wrappers.

## 3. Milestone M24a — MCP server (the "first step")

**Goal:** a standard MCP server exposing the tool layer over streamable HTTP,
running as a NAS container. Once this exists, *any* MCP-capable local client (or
the MCP Inspector) can drive LLM search end-to-end — proving the tool layer
before we build any UI.

**Work:**

1. **`photosearch/tools.py`** — the registry + `call_tool()` + JSON Schemas +
   the `search_combined` `person_ids` extension. Unit tests in
   `tests/test_tools.py` (each tool against a small fixture DB; schema validity;
   name→id resolution; truncation).
2. **`photosearch/mcp_server.py`** — `FastMCP` server (official `mcp` SDK). On
   startup opens `PhotoDB(os.environ["PHOTOSEARCH_DB"])`. Registers each tool
   from the registry. `get_photo_image` is only registered (or only returns
   bytes) when `PHOTOSEARCH_MCP_ALLOW_IMAGES` is truthy; otherwise it returns a
   clear "image returns disabled by operator" error. Runs streamable-HTTP on a
   configurable port.
3. **`requirements.txt`** — add `mcp>=1.2` (the official Python SDK with
   `FastMCP` + streamable-HTTP). ⚠️ requirements change = ~5 min image rebuild.
4. **`docker-compose.nas.yml`** — new `photosearch-mcp` service. **Reuses the
   existing photosearch image** (same Dockerfile — the CLIP text encoder is
   needed for `search_photos`' semantic path, so this is not a lightweight
   image; that's fine, the model is already cached in `/data`). Override the
   command to `python -m photosearch.mcp_server`. Mounts: `/data` (DB +
   thumbnails, rw for thumbnail cache) and `/photos:ro`. Map the MCP port.
   `PHOTOSEARCH_DB=/data/photo_index.db`, `PHOTOSEARCH_MCP_ALLOW_IMAGES` default
   unset.
5. **Concurrency:** two processes now open the SQLite DB (web + MCP). Both are
   read-only for search; WAL + the existing `busy_timeout` handle concurrent
   readers. No schema change, no writer contention.
6. **Docs:** SKILL.md "MCP server" section + a short connect recipe (client
   config pointing at `http://<nas-ip>:<port>/mcp`).

**Auth note:** streamable HTTP on the LAN. v1 ships open on the trusted LAN; a
later hardening item adds a static bearer token (env-configured header check) if
the endpoint is ever exposed beyond the LAN. Tracked in §6.

**Exit criteria:** drive a full "find photos of Calvin at the beach last summer"
session through the MCP Inspector (or a local client) and get correct ids back,
with the model grounding via `list_people`/`get_library_overview` first.

## 4. Milestone M24b — in-app local agent (`/api/ask`)

**Goal:** the privacy-preserving daily driver — a natural-language box in the
web UI that runs the agent loop entirely on the local LLM and renders the
result grid inline.

**Work:**

1. **Agent loop** (`photosearch/agent.py`): given the user message (+ short
   history), call the local OpenAI-compatible chat endpoint
   (reuse the LLM client plumbing in `describe.py`; base URL from
   `PHOTOSEARCH_TEXT_LLM_URL`, model from a new `PHOTOSEARCH_LLM_AGENT_MODEL`
   role var). Pass `tools=` built from the registry's JSON Schemas. Loop:
   model emits `tool_calls` → `call_tool()` → append tool results → repeat until
   the model returns a final answer or an iteration cap (~6) is hit. Collect the
   final photo-id set across the session.
2. **`POST /api/ask`** (SSE, in `web.py`): follows the established SSE pattern
   (`asyncio.get_running_loop()`, terminal events). Streams:
   - `tool_call` — narration ("searching: people=[Calvin], location=beach…"),
   - `tool_result` — counts ("412 matches, narrowing…"),
   - `photos` — the final compact result set for the grid,
   - `answer` — the model's NL summary,
   - terminal `done` / `cancelled` / `fatal`.
   `AbortController` → `request.is_disconnected()` → stop the loop (same shape as
   the stacking endpoint).
3. **Frontend:** an **"Ask" mode toggle on the existing search page**
   (`index.html`) — additive, not a rewrite. The NL box streams the narration as
   a thin status line and renders the final `photos` payload with the *existing*
   result-card components. The structured filters stay for power use.
4. **Vision verify (optional, local-safe):** because everything is local, the
   agent may call `get_photo_image` to let a vision-capable local model confirm
   ambiguous hits before answering. Gated behind a per-request flag so the cheap
   text-only loop stays the default.

**Model requirement & fallback tier:** the loop needs a *tool-calling-capable*
local model (qwen2.5-instruct / qwen3 / llama-3.1+ served by LM Studio with tool
use, or an Ollama tool-capable model). If none is configured, degrade gracefully
to a **single-shot NL→filters mode**: one constrained JSON completion that fills
the `search_photos` arguments, runs one search, and returns — no agent loop, no
iteration, but still "type a sentence, get results." This guarantees the feature
works even on a model that can't do multi-turn tool calls, and is a clean A/B
baseline for the full loop.

**Exit criteria:** from the web UI, "show me the best landscape shots from our
Italy trip in 2024" returns a correct, quality-sorted grid with zero structured
filters touched, and the network tab confirms no request left the NAS.

## 5. Privacy posture (explicit)

- **In-app `/api/ask`:** fully local. Metadata and pixels stay on the NAS. This
  is the recommended default path and the one surfaced in the UI.
- **MCP server, text tools:** return descriptions, tags, place names, dates,
  scores — the *derived* metadata. If a client off the NAS consumes them, that
  text leaves. Operator's choice of client governs this.
- **MCP server, `get_photo_image`:** the only path that emits pixels. **Default
  off** (`PHOTOSEARCH_MCP_ALLOW_IMAGES` unset). Local agent is unaffected (it can
  always look, because "local"). Flipping the flag is a deliberate operator act
  with a one-line consequence: thumbnails become available to whatever client is
  connected.

## 6. Risks & open questions

- **Local tool-calling quality** is the main risk. Mitigation: the single-shot
  NL→filters fallback (§4) and a documented short-list of known-good local
  models.
- **Latency:** each `search_photos` does a CLIP text embed (~the cost of one
  web search on the N100) plus the LLM round-trips. Cap iterations; show
  streaming narration so the wait is legible.
- **Two DB readers:** fine under WAL (both read-only for search), but worth a
  smoke test under a concurrent worker-fleet write load.
- **MCP image weight:** the MCP container carries the full torch/CLIP image. If
  a future lightweight "text-tools-only" MCP variant is wanted (no semantic
  search → no CLIP), it could drop torch, but that's a separate, optional image.
- **MCP auth** if ever exposed beyond the LAN (static bearer token via header).
- **No schema change** is required. If we later want to log agent sessions for
  debugging/eval, add an `agent_sessions` table then (deferred — not v1).

## 7. Sequencing

1. **Tool layer** (`tools.py` + `search_combined` `person_ids` + tests) — both
   adapters depend on it.
2. **M24a MCP server** + compose service + SKILL docs — the "first step";
   provably drives LLM search before any UI exists.
3. **M24b in-app agent** (`agent.py` + `/api/ask` SSE + "Ask" mode in
   index.html) — the private daily driver, mostly frontend + the loop once the
   tool layer is in place.

## 8. Future

- **VLM re-rank** as a first-class loop step (dovetails with the "VLM
  re-ranking" item in `docs/plans/search-accuracy-improvements.md`, which this
  milestone partly subsumes — the "LLM query rewriter" there is exactly the
  single-shot fallback tier here).
- **Write tools** (add-to-collection, tag, set-location) so the model can *act*
  on results, not just find them — turns search into an assistant.
- **Saved/explained searches** — persist a session and its tool trace so "why
  these photos?" is answerable.
- **MCP auth + remote access** if the user ever wants to drive it from off-LAN.

## Related plans

- `docs/plans/search-accuracy-improvements.md` — RRF / recency / structured
  location columns (already partly shipped) and the LLM-query-rewriter idea this
  milestone generalizes.

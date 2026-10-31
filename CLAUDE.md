# Photo Search

Local, Docker-deployed photo search system. Photos never leave the machine.
Runs on a UGREEN NAS (Intel N100, no GPU), developed on Mac and a Windows/WSL2 desktop with a 24 GB AMD GPU.

## Tech Stack

Python 3.11, FastAPI, SQLite (WAL) + sqlite-vec, CLIP (ViT-B/16) for semantic search,
InsightFace for faces, vision/text LLMs via LM Studio (OpenAI-compatible) or Ollama for descriptions and tags,
a VLM for aesthetic scoring. Frontend is plain React (UMD, no build step) in `frontend/dist/`. Docker Compose for deployment.

## Key Patterns — Follow These

1. **Streaming generators** for batch jobs — yield results incrementally, save per batch.
2. **`envvar="PHOTOSEARCH_DB"`** on every CLI command's `--db` option. `PhotoDB()` with no argument opens a relative `photo_index.db` (an empty stub in the container): ad-hoc code must pass the path.
3. **Frontend has no build step** — edit `frontend/dist/` directly, `React.createElement` not JSX; changes need a Docker image rebuild. Shared helpers go on `PS.*` in `shared.js`, never copied between pages. Run `node scripts/check-frontend-refs.js`. Poll with `PS.poll`, never `setInterval`.
4. **Docker cache** — `requirements.txt` changes ≈ 5 min rebuild; Python changes ≈ 10 s.
5. **SSE for long-running API ops** — `asyncio.get_running_loop()` (not `get_event_loop()`); always send a terminal event (`done`, `fatal`, `cancelled`).
6. **Per-file ledger writes** for cancel safety (uploads etc.).
7. **The NAS is the only writer.** The desktop replica's DB is swapped wholesale by `sync-replica.sh`, so replica-side writes go NAS-authoritative, then mirror locally. Deploy the NAS before the replica or the workers.
8. **Face ids are stable; cluster ids are not** — every `recluster-faces` renumbers clusters. Store face ids.
9. **State your intent on every API call you make** (curl, scripts): `-H 'X-Photosearch-Source: claude' -H 'X-Photosearch-Intent: <why>'`. MCP tools take an `intent` argument.
10. **Watching a long job: never `pgrep -f` a bare string** — it matches the watcher itself. Poll a marker file, a saved PID (`kill -0`), or the job's terminal SSE event.
11. **Bad model output is a failure, not a result** — guards raise, the worker records a failure row, attempts are capped (`MAX_PROCESS_ATTEMPTS`). Never widen a transient-error check to a bare `except`.

## Where knowledge goes (tests/test_context_budget.py enforces the caps)

- CLAUDE.md (≤ 24 KB): rules every task needs. One line each. No dates, no narratives, no `###`.
- A trap specific to one subsystem → `.claude/rules/<subsystem>.md` (path-scoped, ≤ 8 KB). One bullet per trap; bold the trap's name.
- How to operate something → `.claude/skills/photo-search/reference/<topic>.md`.
- What happened, measurements, why a decision was made → `docs/notes/<subsystem>.md` under its own dated `###` heading, then ONE line in the rule or reference file that names it.
- Plan or milestone status → `docs/plans/README.md` only.

Test before appending here: would a session editing an unrelated file (say `cli.py`) need this line? If not, it goes elsewhere.

## Subsystem traps

Path-scoped rules load only when you open matching code; a `PreToolUse` hook (`scripts/hooks/inject_rules.py`) also injects one once per session when a Bash/Grep/Glob call names its subsystem. A session that only answers questions gets neither, so before answering or running anything in a subsystem below, Read its rule file first:

- faces: before answering or running anything about faces, clusters, people labels or merges, Read `.claude/rules/faces.md` (one-person-per-photo · `rejected` labels · rival test · trusted labels only · temporal matches are unreliable)
- llm-passes: before answering or running anything about describe/verify/category/keywords/aesthetics passes or LM Studio/Ollama models, Read `.claude/rules/llm-passes.md` (fleet role models · reasoning effort · category-visual derived vs perceived · grounding · provenance · stop reasons)
- ingest: before answering or running anything about ingest, the SD importer, `_incoming`, batches or refile, Read `.claude/rules/ingest.md` (companions · refile-unknown-camera · six batch states · job rows · deploy NAS first)
- maintenance: before answering or running anything about maintenance sweeps, geocoding, stacking, `sync-replica.sh` or the replica, Read `.claude/rules/maintenance.md` (geocode memory per row · incremental stacking · STRICT-only matching · face exclusions · replica gating)
- workers: before answering or running anything about `run-workers.sh`, the worker fleet, `-d`, passes or `/api/worker/*`, Read `.claude/rules/workers.md` (retry attempts vs server write failures · `-d` gotcha · `/api/health` probe · sharpness backfill)
- people-bind: before answering or running anything about names in descriptions, `people-bind`, `description_subjects`/`description_display` or the display flag, Read `.claude/rules/people-bind.md` (raw text untouched · spans not templates · label-free pass · trusted sources only · not a batch step)
- video: before answering or running anything about video, `media_type`, `--native` or ffprobe/ffmpeg, Read `.claude/rules/video.md` (NULL `media_type` = still · `PASS_MEDIA` · the NAS never decodes · backfill not started)
- print: before answering or running anything about Topaz upscale, books, Shutterfly or split export, Read `.claude/rules/print.md` (Topaz CLI quirks · strength blend · split geometry exists twice · integer-dpi grid)
- frontend: before answering or running anything about `frontend/dist/`, the admin deploy panel or a restart, Read `.claude/rules/frontend.md` (no linter · `PS.poll` · deploy helper container · M28 re-run mirror)
- search: before answering or running anything about search, `search.py`, `web.py`, the MCP server or request logging, Read `.claude/rules/search.md` (indexes match query TEXT · `_compose_scope` · request log intent · `set_photo_filepath`)

## Ops facts you need without opening code

Copied from the rule files above; those stay the source of truth.

- **Worker `-d`**: `-d` must be a real subdirectory; `-d /photos` (the root) matches nothing, so the server answers 404 "No photos found in directory /photos" and the fleet does no work. To process the whole library, omit `-d` entirely.
- **Worker startup probe**: `GET /api/health` (no DB). Never probe `/api/stats` (full-library counts, the whole fleet at once). A 404 counts as reachable (older server).
- **Reasoning models** (every fleet model thinks): set `PHOTOSEARCH_LLM_REASONING_EFFORT=none` (`rerun.FLEET_REASONING_EFFORT`) or they spend all of `max_tokens` on reasoning and return `''`. The field is skipped for a model matching `describe._NO_REASONING_MODEL_PATTERNS` (qwen2.5-vl, llava, ...) — extend via `PHOTOSEARCH_LLM_NO_REASONING_MODELS`.
- **Video passes need `--native`**: ffprobe is not in the Docker worker image; the static ffprobe segfaults resolving `localhost`, so point workers at `127.0.0.1`.
- **SD importer**: never `--no-index` (moved files become unclaimable, unsearchable and invisible to hash dedup); use `--no-clip` to register rows and leave CLIP to the fleet.
- **Restarting photosearch**: never `docker compose up` it from inside its own container (it strands the new container in Created state); `POST /api/admin/restart` launches the detached restart helper. Git on the NAS needs per-call `-c safe.directory=...` (see NAS Docker Commands).

History for every trap: `docs/notes/<subsystem>.md`. Runbooks: `.claude/skills/photo-search/reference/`. Plan status: `docs/plans/README.md`.

## Database

File is `photo_index.db` (not `photos.db`). `SCHEMA_VERSION` in `photosearch/db.py` is the source of truth (v41).
Schema change: bump `SCHEMA_VERSION`, add the migration in `_init_schema()`; new indexes go in `db._SEARCH_INDEXES`.
Write `filepath` only through `db.set_photo_filepath` (it re-derives `folder`).
Table/column reference: `.claude/skills/photo-search/reference/schema.md`.

## NAS Docker Commands

```bash
DC="docker compose -f docker-compose.nas.yml"
$DC build photosearch && $DC up -d photosearch     # rebuild + restart after code changes
$DC run --rm photosearch <cmd>                     # CLI command
# Ad-hoc Python/shell: --entrypoint must come BEFORE the service name.
DCPY="docker compose -f docker-compose.nas.yml run --rm --entrypoint python"
DCSH="docker compose -f docker-compose.nas.yml run --rm --entrypoint bash"
$DCPY photosearch -c "import os; from photosearch.db import PhotoDB; db = PhotoDB(os.environ['PHOTOSEARCH_DB'])"
# Git on the NAS: bare git fails "dubious ownership" (UGOS owns the dir) → per-call safe.directory; push via gh's credential helper.
git -c safe.directory=/volume1/docker/photosearch status
git -c safe.directory=/volume1/docker/photosearch -c credential.helper='!gh auth git-credential' push origin main
```

Everything else (debug-db.sh, volumes, cron, Syncthing, geocoding data): `.claude/skills/photo-search/reference/nas-operations.md`.

## Running the tests the way CI does

`./scripts/test-like-ci.sh` before pushing. Plain `pytest` on a dev box differs from CI three ways:
no Topaz CLI on CI (tests of that route take the `fake_cli` fixture); CI installs no torch/cv2/insightface
(`tests/conftest.py` mocks them); CI ignores no test file. In a git worktree created from Windows,
`tests/test_admin_deploy_mode.py` fails locally only (UNC gitdir).

## Adding Features

- **New CLI command:** add to `cli.py`, always include `envvar="PHOTOSEARCH_DB"` on `--db`.
- **New search type:** `search.py` → `search_combined()` → `web.py` param → `index.html` UI. Take `scope=None`
  and apply `_scope_clause`, push dates into SQL with `_date_bounds`, count it in `n_structured`, keep people-only
  `narrow` mode correct (`_NARROW_COLUMNS`), and add a `COMBOS` case to `tests/test_search_indexes.py`.
- **New indexing pass:** add to both `index_directory()` and `_index_collection()` in `index.py`, add
  `--flag`/`--force-flag` to `cli.py`, use the streaming generator pattern. A new worker pass also needs a
  `db.PASS_MEDIA` entry.
- **New API endpoint:** `web.py` with `_get_db()`, SSE for long ops; add a rule to `request_intent._RULES`.
- **Schema change:** bump `SCHEMA_VERSION`, add migration SQL in `_init_schema()`. New indexes go in
  `db._SEARCH_INDEXES` (one commit per index), replaced ones in `_SUPERSEDED_INDEXES`.

More: `.claude/skills/photo-search/reference/adding-features.md`.

## Detailed Reference

The photo-search skill (`.claude/skills/photo-search/SKILL.md`) is a navigator: API endpoints, indexing passes,
NAS operations, the worker fleet, face management, MCP/Ask/replica and troubleshooting each have a file in
`.claude/skills/photo-search/reference/`.

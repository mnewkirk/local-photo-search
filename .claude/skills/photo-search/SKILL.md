---
name: photo-search
description: >
  Expert knowledge for developing and operating the local-photo-search service — a
  self-hosted photo library with CLIP semantic search, face recognition, LLaVA scene
  descriptions, aesthetic quality scoring, Google Photos upload, and shoot review/culling,
  deployed via Docker on a UGREEN NAS.

  Use this skill whenever the user asks about: adding features to photo-search, running
  indexing jobs, troubleshooting the NAS deployment, understanding the codebase
  architecture, modifying the CLI or web API, updating the status page, managing face
  references, finding duplicate photos, working with collections, stacking burst photos,
  uploading to Google Photos, search performance and indexes, the API request log
  (latency, who made a request and why), or any other task related to this project.

  Trigger even for vague requests like "how do I index 2024?" or "why is search slow?"
  or "what were my last queries?" or "I want to add a new filter" or "upload these to Google" — all of these relate
  to this service.
---

# Photo Search Skill

A fully local, Docker-deployed photo search system: photos never go to an external API.
It runs on a UGREEN NAS (Intel N100, no GPU); the heavy model passes run on a desktop
worker fleet (WSL2 + 24 GB AMD GPU, LM Studio). CLAUDE.md holds the universal rules;
this file is a navigator. **Read the reference file for your topic before answering or
acting** — the detail lives there, not here.

## Where to look

| Topic | Read | What's in it |
|---|---|---|
| Indexing a folder/year, what each pass does | `reference/indexing-passes.md` | pass details, models, parallelism, collection re-index, stacking parameters |
| Running anything on the NAS | `reference/nas-operations.md` | Docker aliases, volumes, full indexing sequence, phone ingest, cron, Syncthing, geocoding data, debug-db.sh |
| Worker fleet | `reference/workers.md` | `run-workers.sh` (Docker vs `--native`), LM Studio role models, GPU setup, worker API and loop |
| Faces from the CLI | `reference/face-management-cli.md` | references, recluster, split, merges, unmatch-person, rank_shoot, desktop recompute, over-matching cleanup |
| Reviewing faces in the UI | `reference/faces-review.md` | person inspector, team review, More of this kid, Label health |
| API routes | `reference/api-endpoints.md` | every `/api/*` route by area (search, video, faces, batches, geotag, Google, …) |
| Frontend pages | `reference/frontend-pages.md` | shared.js components, /status, /faces panels, /merges, review/culling |
| Schema and search performance | `reference/schema.md` | tables, columns, v34 indexes and how search uses them |
| "Why was that slow / what was that for?" | `reference/troubleshooting.md` | `request-stats` and the request log |
| MCP server, Ask agent, local replica | `reference/mcp-ask-replica.md` | M24a/M24b tool layer, locked filters, M26a replica sync |
| Google Photos upload | `reference/google-photos.md` | scope, upload flow, album ids |
| Model evals | `reference/evals.md` | `/eval/models` harnesses and their traps |
| Adding a feature | `reference/adding-features.md` | CLI/search/pass/endpoint/schema/page checklists, key patterns, CI parity |

Subsystem traps (read before changing or operating that subsystem): `.claude/rules/<faces|llm-passes|ingest|maintenance|workers|video|print|frontend|search>.md`.
History, measurements and decisions: `docs/notes/<subsystem>.md`. Plan status: `docs/plans/README.md` only.

## Before you operate

- Read the matching `.claude/rules/*.md` — rules load automatically only when you open code, not when you run commands.
- Every curl/script call to the API sends `X-Photosearch-Source: claude` and `X-Photosearch-Intent: <why>`.
- Writes go to the NAS (the authoritative DB); the desktop replica mirrors them. Never write to the replica DB and expect it to last.
- Heavy passes (CLIP, faces, LLMs, video) run on the desktop fleet, never on the N100.
- New narratives go to `docs/notes/<subsystem>.md`, not to CLAUDE.md or this file.

## Project Layout

```
local-photo-search/
├── cli.py                  # All CLI commands (Click)
├── docker-compose.nas.yml  # NAS production (photosearch, photosearch-mcp, syncthing)
├── run-workers.sh          # Worker fleet launcher (Docker CPU or --native GPU)
├── run-local-replica.sh    # Desktop replica server (M26a)
├── sync-replica.sh         # Pull the NAS DB + thumbnails to the replica
├── photosearch/
│   ├── db.py               # PhotoDB, schema (SCHEMA_VERSION), all queries
│   ├── index.py            # index_directory() + _index_collection()
│   ├── search.py           # search_combined() + search types
│   ├── web.py              # FastAPI app and most /api/* routes
│   ├── worker.py / worker_api.py   # fleet client / server (/api/worker/*)
│   ├── describe.py         # LLM routing (Ollama / OpenAI-compatible), describe + tag passes
│   ├── aesthetics.py, quality.py   # VLM aesthetics, LAION prior
│   ├── faces.py, face_*.py # detection, matching, clustering, verify, suggest, review
│   ├── ingest.py, ingest_batches.py, batch_*.py, refile.py   # ingest + /batches
│   ├── maintenance*.py     # maintenance sweep + replica push
│   ├── video_*.py          # video passes, stores, API
│   ├── tools.py, agent.py, mcp_server.py   # shared tool layer, Ask agent, MCP
│   ├── upscale.py, split_export.py, book*.py   # Topaz, Split, photobooks
│   └── rerun.py            # M28 re-runs; FLEET_ROLE_MODELS
├── frontend/dist/          # Static pages + shared.js (PS.*), no build step
├── evals/                  # model and context eval harnesses
├── docs/plans/ docs/notes/ # plans (status in README.md) and subsystem history
└── tests/
```

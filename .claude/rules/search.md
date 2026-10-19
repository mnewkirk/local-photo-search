---
paths:
  - "photosearch/search.py"
  - "photosearch/db.py"
  - "photosearch/tools.py"
  - "photosearch/agent.py"
  - "photosearch/mcp_server.py"
  - "photosearch/request_*.py"
  - "photosearch/infer_location.py"
  - "photosearch/web.py"
  - "tests/test_search_indexes.py"
---
# Search, schema and the request log — traps

Full history: docs/notes/search.md, docs/notes/geotagging.md (cite by heading). Reference: .claude/skills/photo-search/reference/schema.md, api-endpoints.md, mcp-ask-replica.md.

- **Database: schema changes** — `SCHEMA_VERSION` in `db.py` is the source of truth. Bump it for any new table/column and add the migration to `_init_schema()`. New indexes go in `db._SEARCH_INDEXES` (built one commit per index by `_create_search_indexes`); indexes they replace go in `_SUPERSEDED_INDEXES`.
- **Database: expression and partial indexes match the query TEXT** — keep `COALESCE(aes_overall, aesthetic_score)`, `COALESCE(aes_subject_overall_pct, aes_overall_pct)` and the worker-count predicates verbatim, or the planner stops using the index. Tag queries write `+date_taken` / `+id` on purpose. `tests/test_search_indexes.py` EXPLAINs the SQL production runs; run it after touching any query.
- **Database: date bounds** — filter dates with `_date_bounds` (`date_taken >= from AND date_taken <= to || ' 23:59:59'`). Never `substr(date_taken, 1, 10)`: it cannot use an index.
- **Database: composed scope** — `search_combined` folds date/camera/people into one id scope (`_compose_scope`) when `n_structured >= 2`, then runs every other filter over it. A new search type takes `scope=None`, applies `_scope_clause`, pushes dates into SQL with `_date_bounds`, counts toward `n_structured`, and gets a `COMBOS` case in `tests/test_search_indexes.py`.
- **Database: people-only searches read narrow rows** — they select `_NARROW_COLUMNS` and `_hydrate` only the returned page. A column the post-filter pipeline starts reading must be added to `_NARROW_COLUMNS`, or those searches return rows missing it.
- **Database: kept aesthetic indexes** — `idx_photos_aes_technical` / `_composition` / `_impact` serve the `min_technical`/`min_composition`/`min_impact` floors (`_aesthetic_floor_sql`). Do not drop them as unused.
- **Database: `photos.folder` is derived** — never write a bare `UPDATE photos SET filepath`. Use `db.set_photo_filepath(conn, id, path, raw_filepath=...)` or `PhotoDB.update_photo(filepath=...)`, which re-derive `folder`. The folder migration is keyed v25 because some DBs carry a stray `version=24`; keep its `try/except SELECT folder`.
- **Database: `worker_processed.attempts`** — claim paths exclude `attempts >= MAX_PROCESS_ATTEMPTS`; `mark_processed` UPSERT-increments. A new pass predicate must carry the same exclusion, or a failing photo is re-claimed forever.
- **Database: `generations` is the provenance log** — one row per LLM text artifact via `db.log_generation()`; `photos` holds only the promoted current value. Log the model that ran, not the one configured.
- **`PhotoDB()` with no args opens a stub** — the default `db_path` is the relative `photo_index.db`, not `PHOTOSEARCH_DB`. Always pass the path (`web._get_db()` in handlers).
- **API request timing log** — every `/api/*` request goes to `request_log.jsonl` beside the DB through a queue to a background thread. Logging must never stall or fail a request. `PHOTOSEARCH_REQUEST_LOG` moves it or turns it off (`0`); tests turn it off in `conftest.py`. MCP calls log to the separate `request_log.mcp.jsonl` (`call_tool_logged`), because two processes must not rotate one file.
- **API request timing log: replica→NAS calls** — every call the replica makes to the NAS sends `request_intent.outbound_headers(purpose)`. Background threads and pools lose the intent contextvar: start them through `web._carry_context` (or `request_intent.carry_context`).
- **API request timing log: new endpoints** — add a rule to `request_intent._RULES` so a UI call's intent is inferred rather than logged blank. A handler may set `request.state.log_intent` instead.
- **API request timing log: Claude's own calls** — when a Claude session curls or scripts the HTTP API, send `X-Photosearch-Source: claude` and `X-Photosearch-Intent: <why>`; nothing can infer them. MCP calls fill the tool's `intent` argument.
- **Name extraction in search queries** — `_extract_persons_from_query` matches registered names longest-first (so "Matt Newkirk" beats "Matt"), word-bounded, AND-intersects them, strips connectors and sends the rest to CLIP. Its `(?<!-)` lookbehind keeps `-Calvin` a CLIP exclusion; do not drop it.
- **Inferred geotagging (M19)** — `add_photo` stamps `location_source='exif'` whenever it gets GPS with no source. Inferred writes set `location_source='inferred'` + `location_confidence` and carry `WHERE ... gps_lat IS NULL` so real GPS is never overwritten. Manual writes (`POST /api/photos/bulk-set-location`) stamp `'manual'` and skip located rows unless `overwrite=true`. Keep provenance on every GPS write: the rollback SQL keys on `location_source='inferred'`.
- **Ask locked filters are enforced server-side** — `agent._normalize_locked` / `_merge_locked` inject pinned filters into every `_FILTER_TOOLS` call before dispatch (people union, scalars override). A new filter must be added to `tools._build_filter_sql` and `_CURATION_FILTER_PROPS` (and `search_combined` for `search_photos`), or the model silently drops it.
- **Write tools are ON by default** — `set_photo_location` / `set_photo_tags` / `add_to_collection` run unless `PHOTOSEARCH_ALLOW_WRITES` is falsy. Keep the dry-run → `confirm=true` gate and the `PHOTOSEARCH_WRITE_MAX_ROWS` cap (`confirm_large`). `get_photo_image` is off in code unless `PHOTOSEARCH_MCP_ALLOW_IMAGES` is truthy, but `docker-compose.nas.yml` defaults it to `1`.

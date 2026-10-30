# Database schema

Read when you need table/column details or how search uses the v34 indexes.

<!-- moved verbatim from SKILL.md (line 102) on 2026-10-19 -->
## Database Schema (v40)

> Version note: this section documents the v23 baseline; later migrations added
> structured location columns (v19 in CLAUDE.md's numbering), `photos.folder`
> (v25), the VLM aesthetics `aes_*` columns (v26), per-day aesthetic
> percentile normalization (v28), `maintenance_runs` (v29), and
> `ingest_sweeps`/`ingest_batches`/`ingest_batch_jobs` (v30),
> `face_person_exclusions` (v31), `stacking_seen` (v32), the measured
> `sharpness*` columns (v33), the **search indexes (v34, indexes only — see
> "Search performance" below)**, and **video** (v35 `photos.media_type` +
> `video_meta`/`video_tracks`/`video_subjects`; v36 `poster_version`; v37
> subject shares; v38 `scan_generation`; v39 `video_meta.proxy_*` playback
> proxies — see "Video" below and CLAUDE.md "## Video"; v40 `photo_audit`, the
> per-photo change log, `docs/notes/audit.md`). `SCHEMA_VERSION` in
> `db.py` is the source of truth — currently **40**. See CLAUDE.md for the
> aesthetics + folder + ingest-batch + face-exclusion details.

The database file is `photo_index.db` (not `photos.db`). Key tables:

| Table | Purpose |
|---|---|
| photos | Main photo records — path, date_taken, EXIF, description, tags, scores |
| faces | Detected faces per photo (bbox, encoding, quality) |
| persons | Named persons for face matching |
| face_references | Reference photos/encodings for each person |
| face_person_exclusions | (face, person) pairs the duplicate resolver unmatched — both matchers skip that pairing only (v31) |
| collections | Named photo collections/albums |
| collection_photos | Junction table with sort_order for manual ordering |
| photo_stacks | Burst/bracket groups detected by time + visual similarity |
| stack_members | Photos belonging to each stack |
| review_selections | Culling selections per folder |
| google_photos_uploads | Upload ledger (album_id, filepath, media_item_id) |
| ignored_clusters | Face clusters marked to ignore |
| generations | Provenance log for LLM text artifacts — describe / category-* / keywords / verify (v21) |
| ingest_sweeps | One row per ingest-incoming move+index sweep run (v30) |
| ingest_batches | One row per dated ingest folder — identity + lifecycle, membership derived live (v30) |
| ingest_batch_jobs | Per-(batch, step) job intent (worker fleet / NAS stage), TTL-gated (v30) |
| stacking_seen | Which photos incremental stacking has considered, at which date_taken (v32) |
| photo_audit | Per-photo change log — one row per collection add/remove, face label, tags, location, file move, photo add/delete, with old/new value, source and intent (v40). Append-only, no FKs; written only via `photosearch/audit.py`. See `docs/notes/audit.md` |
| schema_info | Schema version + photo_root path |

Important columns on `photos`: `date_taken` (TEXT, "YYYY-MM-DD HH:MM:SS", indexed),
`aesthetic_score` (REAL, 1-10), `description` (TEXT, vision-model-generated), `tags` (JSON array),
`place_name` (TEXT, reverse-geocoded), `dominant_colors` (JSON array of hex values),
`location_source` (`'exif'|'inferred'|NULL`, v17), `location_confidence`
(`NULL|(0,1]`, v17, only non-null for inferred rows), `date_created` (v16
file-mtime fallback sort key).

`generations` columns (v21): `photo_id`, `text_type` (`'describe'|'tags'|'verify'`),
`generated_text` (the description string, or a JSON array for tags), `model_used`,
`model_version` (Ollama digest, best-effort), `created_at`. The `photos` table
holds the current "promoted" description/tags; `generations` is the full history
so artifacts from different models coexist with provenance. Written by
`db.log_generation()` from `worker_api.submit_results`. Backfill historical data
with `photosearch backfill-generations` (idempotent — `NOT EXISTS` guard, NULL
model for pre-v21 rows). Related maintenance command: `photosearch
clean-garbage-tags` clears regurgitated tag sets (photos with ≥16 of the 78
vocabulary tags) from both `photos.tags` and `generations`, so they get
re-tagged — see the regurgitation guard in `describe.py:tag_photo`.

Schema migrations run automatically on DB open via `_init_schema()` with version checks.
Bump `SCHEMA_VERSION` when adding tables or columns, and add migration SQL in the
appropriate position (after any table it depends on).

**v23** (current): `photos` gained `categories` / `visual_tags` / `keywords`
columns (the old single `tags` pass split three ways — see Indexing Types) plus
a `tags_v22_backup` snapshot of the pre-split `tags`. **v22:** `worker_processed`
gained `attempts INTEGER`; the claim path filters `attempts >=
MAX_PROCESS_ATTEMPTS` (default 3) instead of `NOT EXISTS`, so transient failures
auto-retry and truly-broken files stop being claimed after N tries (replaces the
manual `retry-failed-describe` flow for new failures). **v20:** `faces.det_score`
stores InsightFace detection confidence, used by the clustering quality
pre-filter. Note: `face_dedupe_undo` (created on demand by the dedup commands
below) is a utility table, NOT part of the migrated schema.

<!-- moved verbatim from SKILL.md (line 175) on 2026-10-19 -->
### Search performance — indexes (v34) and how search uses them

Full plan, measurements and decisions: `docs/plans/search-indexes.md`. On the
NAS the cost of a query is **bytes read from a spinning disk with a cold
cache**, and any `SCAN photos` reads ~545 MB (the table is 520 MB, ~3.5 KB a
row). The trigger: a camera + two-day search took >120 s on 2026-10-03.

**Indexes** — `db._SEARCH_INDEXES`, built by `PhotoDB._create_search_indexes`
(one commit per index, after a lock-free warm-up read):

| index | serves |
|---|---|
| `idx_photos_camera_date (camera_model, date_taken)` | camera alone, camera + dates |
| `idx_photos_file_hash` | ingest dedup (was a full scan per incoming file) |
| `idx_photos_{visual_tags,categories,keywords} (col, date_taken)` | tag filters scan a 10–25 MB covering index, not the table |
| `idx_photos_{country,admin1,admin2,locality}_nc (col COLLATE NOCASE)` | structured location (`col = ? COLLATE NOCASE`) |
| `idx_photos_gps_cover` (partial, `gps_lat IS NOT NULL`) | `/api/photos/geojson` (/map) |
| `idx_photos_raw_quality`, `idx_photos_subject_aes` (expressions) | `min_quality`, subject-aesthetic sort |
| `idx_photos_need_{describe,verify,quality}` (partial) | worker queue counts |
| `idx_faces_photo_person`, `idx_faces_person_photo` | "is P in photo X" (one seek) and "P's photos" |

Dropped in v34 (`_SUPERSEDED_INDEXES`): the single-column faces indexes, the
BINARY location indexes (never used behind `LOWER()`), and three exact
duplicates of PRIMARY KEY / UNIQUE autoindexes.

**Kept on purpose:** `idx_photos_aes_technical` / `_composition` / `_impact`.
Dropping them was approved once, then reversed (owner, 2026-10-04): the browse's
`min_technical` / `min_composition` / `min_impact` floors are SQL `col >= ?`
predicates (`_aesthetic_floor_sql`) that use them. Do not drop them as
"unused".

**How `search_combined` uses them:**

- **Composed scope** (`_compose_scope`). With 2+ structured filters, date /
  camera / people are first combined into ONE id query (people as `EXISTS`
  on `faces(photo_id, person_id)`); every other filter then runs only over
  those ids (`_scope_clause`). Above `_SCOPE_MAX_IDS` (20k) the scope is
  dropped. The Python intersection + `_filter_by_date` still run, so the scope
  is a pure optimisation. Date × person × camera × location: ~1.9 GB → ~1 MB.
- **id-first** people and substring-LIKE queries: resolve ids from a narrow
  index, then read each row once. **Not** inside `tools._build_filter_sql`
  (there the planner loses the date index: 8.7 → 259 MB measured).
- **SQL pagination** for the aesthetics and `min_quality` browses
  (`_sql_page`, `_sort_sql`, `_aesthetic_floor_sql`): floors, dates and every
  sort mode translated exactly (incl. `or -1` on 0 and the stable-sort
  tie-break); `COUNT(*)` for the total, only the page read. `style_tag` keeps
  the Python path.
- **People-only searches** select `_NARROW_COLUMNS` and `_hydrate` full rows
  for the returned page only (Calvin, 17.6k photos: 223 → 75 MB, 3.1 → 0.07 s).

**Traps:**

- Expression and partial indexes match the query **text**. Keep
  `COALESCE(aes_overall, aesthetic_score)`, `COALESCE(aes_subject_overall_pct,
  aes_overall_pct)` and the worker-count predicates verbatim.
- Tag queries write `+date_taken` (and `+id`) on purpose: without the `+` the
  planner picks `idx_photos_date` for a wide range and reads 399 MB.
- Date bounds are `date_taken >= from AND date_taken <= to || ' 23:59:59'`
  (`_date_bounds`) — exact because every stored `date_taken` is
  `YYYY-MM-DD hh:mm:ss` (verified). Never `substr(date_taken, 1, 10)`: it
  cannot use an index.
- A column the post-filter pipeline starts reading must be added to
  `_NARROW_COLUMNS`, or people-only searches return rows missing it.
- `tests/test_search_indexes.py` EXPLAINs the SQL production actually runs
  (`set_trace_callback`) and compares scoped vs unscoped, SQL-paged vs
  Python-paged, and narrow vs full-row answers. Run it after touching any of
  this.

---

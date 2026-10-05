# Search indexes (schema v34)

**Status:** Phase 1 (schema v34) and Phase 2 implemented 2026-10-04. Open: the aes_* index drop, and date pushdown into the remaining standalone filters.
Planned 2026-10-03. Produced by a two-planner debate (Sonnet + Opus, two
critique rounds); every point below was agreed by both.

## Why

On 2026-10-03, `/api/search?camera=ILCE-7RM6&date_from=2026-09-26&date_to=2026-09-27&sort=date_desc`
took over 120 s on the NAS. Date-only took 5 s, and the same search runs in 0.9 s warm.
Two things caused it: the web container had just restarted, so the page cache was cold,
and one-off CLI containers were hitting the disk at the same time.

The camera filter suffered most because it has no index. `search.py:1809` runs
`SELECT * FROM photos WHERE camera_model = ?`, a full scan of the 520 MB `photos`
table, and only then filters the dates in Python.

On the NAS, cost means **bytes read from a spinning disk with a cold cache**, not warm
time. All numbers below are bytes SQLite read, measured on a replica copy with
`cache_size=-2000` and `mmap_size=0`. Any `SCAN photos` reads about 545 MB. The faces
table is only 9.6 MB, so nearly all the cost is in `photos`.

`sqlite_stat1` does not exist (ANALYZE has never run). Python's bundled SQLite has no
STAT4, so each fix below is pinned by an EXPLAIN QUERY PLAN test rather than by
statistics.

## Measured before / after

| query (where) | today | with plan |
|---|---|---|
| camera + date, the incident (`search.py:1809`) | 544 MB, scan + Python date filter | **8.7 MB** |
| camera only | 544 MB | 43 MB |
| `file_hash = ?` (`ingest.py:542`, **once per ingested file**) | 544 MB | **~0** |
| visual_tag / category / keyword filter loops | 545 MB each | 10 / 13 / 25 MB (covering) |
| tag filter with a wide date range, without `+date_taken` | 399 MB | 10 MB with `+date_taken` |
| min_quality only (`search.py:1843`, `tools.py:947`) | 545 MB + temp sort | 1.5 MB |
| structured location, rare place (`LOWER()` stops the index being used) | 561 MB | 1.0 MB |
| common place ("Marin County", 67k rows) | 561 MB | 571 MB (no regression; cost is the 67k-row `SELECT *`) |
| `/api/photos/geojson` (/map) | 454 MB | 10.7 MB (covering) |
| worker queue counts: describe / verify / quality | 544 MB each | ~0 |
| worker queue counts: category-* / keywords | 544 MB | ~0.1 MB (from the tag indexes) |

## Phase 1: schema v34

### 1. Migration shape (`db.py`)
- `SCHEMA_VERSION = 34`. Add the DDL to the "Indexes for common queries" block as
  `CREATE INDEX IF NOT EXISTS`, so fresh DBs get the indexes too. Remove the
  `CREATE INDEX` lines for every index this plan drops, or the next migration recreates
  them.
- **Commit immediately before and after each `CREATE INDEX` on `photos`.** Earlier DML
  in the non-fast path, such as the v31 exclusion backfill, leaves a transaction open.
  Without these commits, every index build would happen inside one write lock.
- **Warm the cache read-only before the first build:** one `SELECT sum(length(...))`
  over the indexed columns. In WAL a reader doesn't block writers, so the slow cold read
  of 520 MB happens outside the write lock.
- Stamp the version only after the last index has committed.

### 2. The incident: camera composite + date pushdown
```sql
CREATE INDEX IF NOT EXISTS idx_photos_camera_date ON photos(camera_model, date_taken);  -- 5.9 MB
```
Push `date_taken >= ? AND date_taken <= ? || ' 23:59:59'` into the camera SQL when a
date is given. The index alone still pulls 12k wide rows. `_filter_by_date` stays, so
the results are identical. The composite also serves camera-only searches; a plain
`camera_model` index would add nothing.

`camera_model` is never UPDATEd after ingest, so fleet writes don't pay for this index.

The three `substr(date_taken,1,10) >= ?` sites in `web.py` become range comparisons:
`_face_filter_photo_ids` (~1003), geotag folders (~3893) and folder-photos (~3955).
This is exact: all 160,049 dated rows are `YYYY-MM-DD hh:mm:ss`.

### 3. file_hash
```sql
CREATE INDEX IF NOT EXISTS idx_photos_file_hash ON photos(file_hash);  -- 11.5 MB, not UNIQUE (~3.3k dup hashes)
```
Ingest dedup currently scans the whole table once per incoming photo, so a 1,373-photo
card dump costs 1,373 full scans. The index also makes dedup's `GROUP BY file_hash` a
covering scan.

### 4. JSON tag columns: covering composites
```sql
CREATE INDEX IF NOT EXISTS idx_photos_visual_tags ON photos(visual_tags, date_taken);
CREATE INDEX IF NOT EXISTS idx_photos_categories  ON photos(categories,  date_taken);
CREATE INDEX IF NOT EXISTS idx_photos_keywords    ON photos(keywords,    date_taken);
```
The three loops at `search.py:1762-1806` still parse the JSON in Python. These indexes
cut the disk reads, not the per-row `json.loads` cost. When a date is set, add
`AND +date_taken >= ? AND +date_taken <= ?`. **The unary `+` is required:** without it
the planner picks `idx_photos_date` for a wide range and reads 399 MB. Replace the
per-match `db.get_photo(id)` with chunked `SELECT * ... WHERE id IN (...)` (≤ 900 ids
per chunk).

### 5. Structured location: swap unused BINARY indexes for NOCASE ones
`search.py:1536` and `tools.py:918` compare `LOWER(col) = LOWER(?)`, so the four
existing indexes are never used.
```sql
DROP INDEX IF EXISTS idx_photos_country;   DROP INDEX IF EXISTS idx_photos_admin1;
DROP INDEX IF EXISTS idx_photos_admin2;    DROP INDEX IF EXISTS idx_photos_locality;
CREATE INDEX IF NOT EXISTS idx_photos_country_nc  ON photos(country  COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_photos_admin1_nc   ON photos(admin1   COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_photos_admin2_nc   ON photos(admin2   COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_photos_locality_nc ON photos(locality COLLATE NOCASE);
```
Rewrite the comparisons to `col = ? COLLATE NOCASE`. Results don't change, because
NOCASE folds only ASCII, exactly like `LOWER()` without ICU. Net write cost is zero
(four indexes swapped for four). The maintenance gate (`maintenance.py:165`) still gets
MULTI-INDEX OR.

### 6. /map covering index
```sql
CREATE INDEX IF NOT EXISTS idx_photos_gps_cover
  ON photos(gps_lat, gps_lon, location_source, date_taken, place_name)
  WHERE gps_lat IS NOT NULL;  -- 10.2 MB
```
This matches the geojson SELECT list exactly. It is **not** a fix for the bbox search:
measured on the home bbox, that `SELECT * ORDER BY date_taken` gets no better.

### 7. Quality / subject-aesthetic expression indexes
```sql
CREATE INDEX IF NOT EXISTS idx_photos_raw_quality
  ON photos(COALESCE(aes_overall, aesthetic_score));
CREATE INDEX IF NOT EXISTS idx_photos_subject_aes_coalesce
  ON photos(COALESCE(aes_subject_overall_pct, aes_overall_pct));
```
Push the date range into the min_quality SQL. In the `subject_aesthetic_desc` branch,
change the WHERE guard to `COALESCE(...) IS NOT NULL`. That changes no results: on the
replica, 0 of 72,167 subject-scored photos lack `aes_overall_pct`.

**Expression indexes only match textually.** Comment every site that uses one, and
cover it with a plan test.

### 8. Partial "work remaining" indexes for worker counts
```sql
CREATE INDEX IF NOT EXISTS idx_photos_need_describe ON photos(id) WHERE description IS NULL;
CREATE INDEX IF NOT EXISTS idx_photos_need_verify   ON photos(id)
  WHERE verified_at IS NULL AND description IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_photos_need_quality  ON photos(id)
  WHERE aesthetic_score IS NULL OR aesthetic_concepts IS NULL;
```
These hold almost no rows on a drained library. A row leaves the index once, when its
pass completes. The partial WHERE must match the predicate in `db.py` term for term;
SQLite still uses the index when the real query ANDs an extra `NOT EXISTS(...)`
(verified). Pin each one with a plan test.

### 9. Faces: both composites, replacing the single-column indexes (REQUIRED, see 9a)
```sql
CREATE INDEX IF NOT EXISTS idx_faces_photo_person ON faces(photo_id, person_id);  -- replaces idx_faces_photo (same prefix)
DROP INDEX IF EXISTS idx_faces_photo;
CREATE INDEX IF NOT EXISTS idx_faces_person_photo ON faces(person_id, photo_id);  -- replaces idx_faces_person
DROP INDEX IF EXISTS idx_faces_person;
```
**`(photo_id, person_id)` is not optional.** Step 9a depends on it. Without it, SQLite
answers "is person P in this photo" through `idx_faces_person`, walking all of
Calvin's 17,828 faces for every candidate photo.

Measured on a replica copy, 2026-10-04, Calvin + two days:

| | bytes read | time |
|---|---|---|
| without the index | 16 GB | 10 s |
| with it | **1.3 MB** | ~0 s |

The index builds in 0.06 s, and the net index count doesn't change. Pin the plan with
a test (`SEARCH f USING COVERING INDEX idx_faces_photo_person (photo_id=? AND
person_id=?)`). If the planner still picks the other index, use `INDEXED BY`.

### 9a. Compose the structured filters into ONE query in `search_combined` (highest value)
Today `search_combined` runs each structured filter over the **whole library** as its
own `SELECT *`. It then intersects the result sets in Python and filters dates in
Python. So date × person × camera × location costs the sum of every filter's full
result. Cold-cache reads, measured on the replica:

| filter, run alone over the whole library | bytes read |
|---|---|
| person = Calvin (17,828 photos) | 223 MB |
| camera = ILCE-7RM6 | 552 MB |
| location, `place_name LIKE` | 566 MB |
| location, structured `LOWER()` | 570 MB |
| **total for date × person × camera × location** | **~1.9 GB** |

Composed into one WHERE clause, the date (or camera+date) index narrows first. Each
other filter is then a cheap check on that small set, with person tested as
`EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id AND f.person_id = ?)`:

| composed query | rows | bytes read |
|---|---|---|
| 2 days × person | 276 | 1.3 MB |
| 2 days × person × camera | 276 | 1.3 MB |
| 3 years × person | 9,965 | 46.6 MB |
| 3 years × person × camera | 1,099 | 5.6 MB |

Any combination works this way. It needs **no per-combination composite index**:
SQLite uses one index to narrow, then checks the other filters row by row.

`tools._build_filter_sql` (Ask/MCP) already composes filters into one query. Reuse it,
or extract its clause builder, so the two paths cannot drift. Keep the in-Python
intersection only for things SQL can't express: the CLIP query, color and face-image
searches. Run those over the composed id set instead of the whole library. Use the
`EXISTS` form, not `id IN (SELECT photo_id ... WHERE person_id = ?)`. The `IN` form
walks the person's whole history (78 MB for both the 2-day and the 3-year range).

### 10. Drop exact duplicates
```sql
DROP INDEX IF EXISTS idx_stack_members_photo;     -- = UNIQUE(photo_id)
DROP INDEX IF EXISTS idx_stack_members_stack;     -- = PK(stack_id, photo_id) prefix
DROP INDEX IF EXISTS idx_collection_photos_coll;  -- = PK(collection_id, photo_id) prefix
```
`idx_collection_photos_photo` is **not** a duplicate; keep it. Add a test that
`get_photo_stack`, which runs once per result row, still uses the autoindexes.

### 11. ANALYZE: optional, separate, later
Run it outside the migration's write lock, as a separate step after deploy. No fix in
this plan depends on statistics: with no STAT4, ANALYZE can't see that one place has
67k photos and another 290, and the key plans were unchanged before and after it.

## Phase 2: query rewrites (separate change, after v34 is verified on the NAS)

**Shipped 2026-10-04.** Measured old vs new on a replica copy (cold-cache proxy):
aesthetics browse `sort=aesthetic_desc` 856 → 4 MB; `min_aesthetic=90` 856 → 1–86 MB
depending on sort; a 10-day `min_quality=6` 434 → 25 MB; filename search 566 → 9 MB;
`location=Varenna` 567 → 7 MB; three people 24 → 8 MB. All 37 combinations returned the
same totals and the same pages, except that photos with identical timestamps can swap
places. The browses translate every sort mode and `_filter_aesthetic`'s floors into
SQL (`_sort_sql`, `_aesthetic_floor_sql`), including the `or -1` treatment of 0 and
the stable-sort tie-break. `tests/test_search_indexes.py` compares them against the
Python path, which they replace, for every sort × floor × offset. A browse with
`style_tag` keeps the Python path, because that match runs in Python.

**People-only searches (2026-10-04):** when the only filters are people, the person
queries select `search._NARROW_COLUMNS` (what dedupe, the filters and the sorts read),
and full rows are read only for the returned page (`_hydrate`). Calvin, 17,638 photos:
223 → 75 MB, 3.1 → 0.07 s, with identical rows on 9 cases. A column the post-filter
pipeline starts reading must be added to `_NARROW_COLUMNS`. The test compares narrow
pages against full-row pages, so it fails if one is missing. Person results now
tie-break on `p.id`, so the relevance rank, and which duplicate copy survives dedupe,
no longer depend on the query plan.

With a date range, SQLite starts from the date index. That is the right choice for a
trip-sized range. Over several years it can read more than starting from the quality
index (125 → 520 MB for `min_quality=7` over 3 years), but it is still faster.
- `search_by_all_persons`: switch to an `id IN (SELECT photo_id ... GROUP BY photo_id
  HAVING COUNT(DISTINCT person_id)=?)` subquery. 331 → 45 MB.
- Look up ids first for `_search_by_location`'s `place_name LIKE` and for
  `search_by_filename` (560 → 7–9 MB), **in `search.py` only.** Inside
  `tools._build_filter_sql` the planner then loses the date index: Marin County plus one
  day went from 8.7 MB to 259 MB.
- Paginate the aesthetics-only browse (`search.py:1867`, 846 MB) in SQL: floors in SQL,
  `COUNT(*)` for the total, `SELECT *` for the requested page only. This changes the
  `with_total` contract. No index fixes it.
- Later: push dates into every standalone filter in `search_combined`.

## Not indexed, and why
- `description`: infix `LIKE` and CLIP can't use a B-tree. FTS5 would be a separate
  project.
- `aes_style_tags`: rewritten on every aesthetics submit, little payoff.
- A single-column `camera_model` index: the composite covers it.
- A bare `gps_lat` / `gps_lon` index: measured worse than today on the home bbox.
- `worker_processed`: every predicate already uses its primary key.
- More faces indexes: the table is 9.6 MB, and recluster rewrites 230k rows.
- Owner call, separate change: dropping `idx_photos_aes_technical`, `_composition` and
  `_impact`. They appear in no SQL predicate (those floors are applied in Python) but are
  rewritten on every aesthetics submit. Keep the `*_pct` and `*_day_pct` indexes; the
  sweep gate uses them.

Net: about **+95 MB** of indexes, roughly 4.5% of the DB. The `derive-visual-tags`
backfill will run somewhat slower, because each row update now also touches one more
B-tree.

## Risks
1. **The first index build exceeds the 60 s busy_timeout on a cold cache.** Fleet
   submits defer without burning an attempt, but the sweep and CLI containers can fail
   with `database is locked`. Mitigations: the warm-up read, committing after each index,
   and the deploy procedure below.
2. **The table drops out of the page cache between builds** (1.75 GB free, 2.6 GB
   swapped). Each build would then be another cold scan. Worst case is about 15–25 min.
3. **A harmless-looking SQL reformat silently undoes an expression or partial index.**
   The plan tests are the guard.
4. **About 100 MB of new index pages goes through the WAL.** Sync the replica after a
   checkpoint.

## Verification
- New `tests/test_search_indexes.py`, in the style of `test_directory_scope.py`, on a
  synthetic DB created through `PhotoDB`. **Capture the SQL production actually runs**
  (`set_trace_callback`) and EXPLAIN that, not a copied string. For each index, assert
  its name is in the plan and that a bare `SCAN photos` is not.
- Behaviour tests: camera+date results identical, including NULL dates and a date given
  on only one side; NOCASE matches `LOWER()` on mixed-case data.
- Migration test `test_v33_db_migrates_to_v34_indexes`: new indexes present, dropped
  ones absent, version 34.
- `./scripts/test-like-ci.sh`.
- NAS:
  1. Stop the fleet, and stay clear of the 18:30 sweep and the 04:00 ingest.
  2. `$DC build`, then `$DC run --rm photosearch stats` to run the migration in a one-off
     container and time it.
  3. Check that `PRAGMA index_list(photos)` and the version are right.
  4. `up -d`, then re-run the incident URL straight away (cold).
  5. Watch the per-file time on the next ingest and the `/admin/maintenance` poll
     latency.
  6. Restart the fleet and check `index_errors` for `locked`.

## Estimate
Phase 1 is about 1 day. The NAS migration takes 2–5 min, or 15–25 min worst case.
Phase 2 is about 1 day.

## What shipped (2026-10-04)

| commit | what |
|---|---|
| `31f87c7` | Phase 1: schema v34 indexes, composed scope (9a), date pushdown, `substr()` date filters rewritten |
| `686869e` | Phase 2: id-first people/LIKE queries, SQL-paginated aesthetics and quality browses |
| `befe57e` | People-only searches read full rows for the returned page only (`_NARROW_COLUMNS`) |
| `95b6b9a` | Persistent API request log + `request-stats` |
| `7eaa3c6` | Face-encoding cache (suggest-person, verify-labels); request log records `source` + `intent` |

**NAS migration:** the backup `/data/photo_index.db.bak-v33-20261004` was taken
first and verified (`quick_check` ok, 163,286 photos, 53 s). The migration ran
in a one-off container with the web container stopped: 25 s, `quick_check` ok.

**NAS, cold cache, straight after a restart:**

| request | before | after |
|---|---|---|
| camera + 2 days (the incident) | >120 s | 0.14–0.41 s |
| Calvin + camera + place + 2 days | — | 0.11 s |
| Calvin + ILCE-7M4, 2024–2026 | — | 1.24 s |
| browse by aesthetic score | — | 0.87 s |
| `min_quality=6`, 10 days | — | 0.35 s |
| filename search | — | 0.20 s |
| Calvin alone (17,638 photos) | 1.55 s | 0.35 s |
| More of this kid, Alan, Aug 15–Sep 27 | 8.1 s | 2.4 s first call, 1.9 s repeat |
| More of this kid, Robert, one day | 6.1 s | 0.58 s |

Answers were checked against the previous code on a replica copy. There were
19 Phase 1 combinations, 37 Phase 2 combinations and 9 people-only cases. Totals
and pages match. The only exception is photos with identical timestamps
swapping places.

### Request log with source and intent

There was no record of how real searches performed, because the NAS
container's stdout is discarded on every redeploy. Every `/api/*` request is
now logged to `request_log.jsonl` beside the DB:
- `source`: ui / claude / claude-mcp / agent / script / worker / other.
- `intent`: stated by Claude in the `X-Photosearch-Intent` header or the MCP
  `intent` argument; for the Ask agent, the question; otherwise inferred from
  the page, the endpoint and its parameters.

`photosearch request-stats` summarises it. See CLAUDE.md, "API request timing
log". Only 320 user-initiated requests were recoverable from before this, all
from the replica's journal. Replayed against both servers, every one returned
200. Apart from suggest-person, all were under about 3 s.

### Face-encoding cache

`PhotoDB.get_face_encodings_cached`: an LRU of float32 encodings in process
memory, about 40k faces / 80 MB. "More of this kid" used to fetch about 10k
trusted-label encodings from the vec0 table on every call: 1.7 s warm, about
8 s cold. It needs no invalidation, because encodings are insert/delete-only
and face ids are never reused. Outputs were identical to the old loader in 5
sample calls. On a replica copy, Koa over Sep 1–27 went from 12.3 s to 1.9 s
on the first call and 0.4 s on repeats; verify for one day went from 2.0 s to
0.13–0.2 s. The first calls after a container restart are still slow (69 s
measured): the disk is cold, and the MCP server reads the library at the same
moment.

## Decisions

The planners resolved every point between them, so the owner had no disagreement to
decide.

| topic | option A | option B | user's choice | date |
|---|---|---|---|---|
| (none unresolved) | — | — | — | 2026-10-03 |

Points where the planners started out apart and converged:

| topic | Planner A (Sonnet) initially | Planner B (Opus) initially | agreed |
|---|---|---|---|
| ANALYZE | inside the migration | never | optional, separate, after deploy |
| JSON tag columns | not indexable | covering composites | covering composites, with the `json.loads` caveat |
| file_hash, NOCASE swap, worker partials, /map, duplicate drops | not in scope | proposed | adopted |
| subject-aesthetic expression index | proposed | missed | adopted (count check: 0 affected) |
| query rewrites | — | Phase 2 | separate Phase 2 after v34 is verified |

Owner decisions, 2026-10-04:
- **Dropping the `aes_technical` / `aes_composition` / `aes_impact` indexes:** yes, as a
  separate change. *Not done yet, and worth re-asking:* since Phase 2, the
  browse's `min_technical` / `min_composition` / `min_impact` floors are SQL
  `col >= ?` predicates that can use them.
- **Timing:** start Phase 1 (DB backup, migration, tests) only once the separate
  session fixing the 2026-10-03 batch run reports the batch complete.
- **Added after the debate:** step 9a (compose filters in `search_combined`) and the
  `faces(photo_id, person_id)` index. These came from the owner's frequent
  date × person × camera × location searches, measured above. Step 9a moves into
  Phase 1, because it is the fix for exactly those searches.

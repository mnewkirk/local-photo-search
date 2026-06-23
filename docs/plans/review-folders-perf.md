# Review folder-list performance (M27 — queued)

**Status:** quick-win #1 **shipped 2026-06-22** — `api_review_folders` now uses
the `_folder_of` rsplit helper instead of `Path().parent`. Verified against the
163k-row replica: identical 4328-folder output, 360ms → 38ms (9.5× faster) on
the Python grouping loop. The durable fix (#3, a `folder` column) remains queued.
Surfaced 2026-06-21 — the `/review` folder picker is sometimes slow to load.

## Problem

`GET /api/review/folders` (`web.py:api_review_folders`) builds the folder list
by loading **every** photo row and grouping in Python:

```python
rows = db.conn.execute("SELECT filepath, date_taken FROM photos WHERE filepath IS NOT NULL").fetchall()
for row in rows:
    parent = str(Path(fp).parent)            # ← Path() per row, ~163k times
    if parent not in dir_dates or dt > dir_dates[parent]:
        dir_dates[parent] = dt
```

On the ~163k-photo NAS library this means: pull 163k rows over the wire, build
163k `Path` objects, and group — every time the picker opens. `Path(...).parent`
is the dominant cost (Path construction is slow); the full-table scan is the
rest. It also recomputes from scratch on every request (no caching) even though
the folder set only changes on ingest.

## Fix options (cheapest first)

1. **Drop `Path` for string ops (quick win, ~minutes):**
   `parent = fp.rsplit('/', 1)[0] if '/' in fp else ''`. Avoids 163k `Path`
   allocations — likely the bulk of the wall-clock with no schema/behavior change.

2. **Group in SQL:** `SELECT <dirname(filepath)>, MAX(date_taken) FROM photos
   GROUP BY <dirname>`. SQLite has no `dirname`, so either compute the prefix
   with `substr(...instr(...))` (no built-in `reverse`), or — cleaner —

3. **Add a `folder` column + index (best, schema bump):** populate `folder =
   dirname(filepath)` in `add_photo` (and a one-time backfill), then
   `SELECT folder, MAX(date_taken) FROM photos GROUP BY folder ORDER BY 2 DESC`
   is a fast indexed aggregate. Also benefits `/geotag` folders and any future
   folder-scoped feature. This is the durable fix.

4. **Cache the result** (orthogonal): the folder list changes only on ingest, so
   memoize with invalidation on index/ingest (or a short TTL). Layer on top of
   any of the above.

**Recommendation:** ship #1 immediately (trivial, big relative win), then do #3
(`folder` column) as the durable fix and reuse it for `/geotag`'s folder summary
(`api_geotag_folders` does a similar per-row dirname loop and would benefit too).

## Scope

`web.py:api_review_folders` (+ `api_geotag_folders` for #3), `db.py` schema +
`add_photo` for #3, a backfill CLI for the column. No UI change. Verify against
the replica (`photo_index.db.local`).

# Geocode: replace the in-memory KDTree with a SQLite R-tree

**Status:** queued. The OOM that motivated it is already mitigated (see
"What shipped instead" below) — this is the durable fix, not an emergency.

## The problem, measured

`geonames_rich.get_rich_geocoder()` builds an in-memory
`reverse_geocoder.RGeocoder`. That library's `load()` keeps **one Python dict
per row**, and the cost is dominated by Python object overhead, not by the
KDTree the old docstring blamed:

| | rows | measured |
|---|---|---|
| bytes per row (`locations` + `geo_coords`) | — | **745 B** |
| original unfiltered dataset | 6,581,251 | **4.90 GB** |
| transient text (`StringIO(f.read())`, UCS-2 ×2) | — | **1.44 GB** |
| NAS host memory available | — | **4.5 GB** |

So the `geocode` maintenance stage SIGKILLed the whole photosearch web
container (`exitCode=137`), reproducibly, ~4-5 minutes into the stage. Because
a replica sweep defers every trigger stage to the push, this made **every**
replica→NAS push report `partial` / `unreachable`, and it would have killed the
web server nightly once the maintenance cron was installed.

## What shipped instead (2026-09-13)

Two changes that fit in the existing design, taking peak ~6.3 GB → ~1.5 GB:

- `_MIN_POPULATION = 1` drops class-P rows GeoNames records as population 0 —
  **4.72M of 5.2M populated places**. Named POIs are matched by feature *code*
  and kept unconditionally, so parks/peaks/beaches (always population 0) are
  untouched. 6.58M rows → 1.86M, 4.90 GB → ~1.4 GB.
- `get_rich_geocoder` streams the CSV instead of `StringIO(f.read())`,
  removing 1.44 GB of transient text.

That is enough to stop the OOM, but it leaves two real costs:

1. **~1.4 GB resident** in every process that geocodes, for the whole process
   lifetime (it is a module-level singleton). On a 7.7 GB box shared with CLIP,
   the workers and SQLite, that is a lot to spend on a lookup table.
2. **~10-30 s build on first query, per process** — paid again by every CLI
   invocation, every container, every worker.
3. It bought headroom by **throwing away 4.72M rows**. Some of those are real
   named places whose population GeoNames simply doesn't know.

## The proposal

Store the dataset in a **SQLite R-tree** and query it, instead of loading it.
`rtree` is compiled into the SQLite on both machines (verified: NAS container
3.46.1, desktop venv 3.45.1) — no new dependency.

```sql
CREATE VIRTUAL TABLE geo_rtree USING rtree(id, minlat, maxlat, minlon, maxlon);
CREATE TABLE geo_place (id INTEGER PRIMARY KEY, name TEXT, admin1 TEXT,
                        admin2 TEXT, cc TEXT, lat REAL, lon REAL);
```

Nearest-neighbour is a bounding-box probe that widens until it finds a hit,
then an exact haversine over the (small) candidate set:

```sql
SELECT p.* FROM geo_rtree r JOIN geo_place p ON p.id = r.id
 WHERE r.minlat >= :lat - :d AND r.maxlat <= :lat + :d
   AND r.minlon >= :lon - :d AND r.maxlon <= :lon + :d;
```

Expected: **~0 resident memory**, **no per-process load**, ~500 MB on disk, and
**all 6.58M rows retained** — so `_MIN_POPULATION` can go back to 0 and the
hamlets return.

### Work

1. `build_rich_dataset` gains a `--format sqlite` (or a sibling
   `build_rich_sqlite`) writing the two tables above from `allCountries.txt`.
   One pass, streaming, bounded memory.
2. A `_SqliteGeocoder` with the same `search(coords) -> [dict]` contract
   `_RichWrapper` already exposes, so no caller changes.
3. `get_rich_geocoder()` prefers `rg_rich.sqlite`, falls back to `rg_rich.csv`,
   then to stock `reverse_geocoder`. Three tiers, each independently testable.
4. Batch the probe: callers pass many coordinates at once
   (`normalize-places --batch-size 2000`), so the query should take the batch,
   not one point per call.
5. Tests: correctness against the CSV path on a fixture set (the two must agree
   on the same input), the widening-radius edge cases (poles, the ±180°
   antimeridian, a point in the middle of an ocean with no hit for many
   widenings), and a memory assertion that the loader does not go resident.

### Watch out for

- **The antimeridian and the poles.** A naive `lon ± d` box is wrong across
  ±180°, which the KDTree handled implicitly. Needs either two boxes or a
  projected coordinate. Easy to get silently wrong — most photo libraries never
  exercise it, so a test has to.
- **Degrees are not distance.** A 0.1° box is ~11 km in latitude but ~4 km in
  longitude at 60°N. Widening must account for the cosine, or high-latitude
  lookups will scan far more than they need (or miss).
- **Ranking must stay haversine.** Whichever candidate the box returns first is
  not the nearest; sort the candidate set properly, as the KDTree did.
- **Verify against the current labels before switching.** Re-run
  `normalize-places` into a scratch column on a sample and diff against the
  existing `place_name` values — a silent change in labels across the whole
  library would be worse than the memory it saves.

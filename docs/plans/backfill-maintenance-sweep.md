# Backfill / Maintenance Sweep + Data Integrity (M25)

**Status:** ✅ **SHIPPED 2026-06-21.** `maintenance-sweep` / `validate-data` /
`repair-data` CLIs + `photosearch/maintenance.py`, the SSE
`POST /api/admin/maintenance-sweep` + `GET /api/admin/validate-data` endpoints,
and the `/status` Maintenance-sweep card all landed. Tests in
`tests/test_maintenance.py`. (Status authority: see
[the roadmap index](README.md). Original plan below, kept for reference.)

## 1. Problem

Indexing a photo is not one step — it's a constellation of derived-data
passes, and only a subset run automatically when new photos arrive. The rest
are manual CLIs that nobody schedules, so freshly-ingested photos sit in a
half-enriched state until someone remembers to run each command. The user
hit this directly: recent phone photos with `place_name: None`, faces not
clustered, structured location columns empty.

### Current trigger map (verified 2026-06-20)

| Derived data | `ingest-incoming` (daily cron) | Worker fleet (`submit_results`) | Manual CLI | Gets lost when… |
|---|---|---|---|---|
| CLIP embeddings | ✅ `index_directory(enable_clip=True)` | ✅ `clip` pass | `index` | — (covered) |
| Reverse-geocode `place_name` | ✅ inside `index_directory` (GPS-bearing rows) | ❌ | (part of `index`) | GPS arrives *later* (infer/manual) → never re-geocoded |
| Dominant colors | ❌ cron runs `--no-colors` | ❌ (not a worker pass) | `index <dir>` | **every daily run** — colors always deferred |
| Stacking | ✅ but **scoped to the new dated folder** | ❌ | `stack` | cross-folder / cross-day bursts; worker-only photos |
| Structured loc cols (`country`/`admin1`/`admin2`/`locality`, v19) | ❌ | ❌ | `normalize-places` | **all new photos** — never auto-populated |
| Inferred GPS (`location_source='inferred'`) | ❌ | ❌ | `infer-locations --apply` | **all new no-GPS photos** |
| Face→person matching | ❌ | ❌ | `match-faces [--temporal]` | new faces never matched to known people |
| Face clustering (Unknown #N) | ❌ (per-batch clustering was removed) | ❌ | `recluster-faces` | **all new faces** — invisible on `/faces` until run |
| Faces / quality / describe / tags / verify | ❌ | ✅ (worker passes) | `index --full` | — (worker fleet covers) |
| Generations provenance | n/a | ✅ per-artifact | `backfill-generations` (historical only) | — (covered) |

The only scheduled job is `ingest-incoming --no-colors`. Everything in the
"❌ / Manual CLI" rows is a step a human has to remember — and they don't.
Each pass is *already idempotent and already exists as a CLI*; the gap is
**orchestration + scheduling**, not new algorithms.

## 2. Proposed design — a single idempotent sweep

### 2a. `photosearch maintenance-sweep`

One command that runs each backfill pass over **only the rows missing it**, in
dependency order, with per-pass on/off flags and a `--dry-run` that reports
how many rows each pass *would* touch. It orchestrates the existing CLIs/
functions rather than reimplementing them.

Dependency order matters (later passes consume earlier outputs):

```
1. reverse-geocode any GPS-bearing rows with place_name IS NULL
2. normalize-places         → structured country/admin1/admin2/locality cols
3. infer-locations --apply  → fill GPS for no-GPS rows from temporal neighbors
4. reverse-geocode + normalize the newly-inferred rows   (re-run 1–2 scoped)
5. dominant colors          → rows with dominant_colors IS NULL
6. stacking                 → library-wide (not folder-scoped) so cross-folder
                              bursts group; idempotent re-detect
7. match-faces --temporal   → attach new faces to known persons
8. recluster-faces          → group remaining unknowns into Unknown #N
```

Each step is gated by a "what's missing" SQL predicate so a sweep over a
fully-enriched library is nearly free (counts come back zero, nothing runs).
Reuse the `on_progress` + `should_abort` callback pair (the stacking
reference shape) so the sweep is cancellable and streams progress.

**Worker-fleet boundary:** the heavy GPU passes (faces detection, quality,
describe, category, keywords, verify) stay on the worker fleet — the sweep
does NOT do those. It handles only the lightweight CPU-side backfills that
have no worker pass and currently fall through the cracks (geocode, colors,
structured cols, infer, stacking, match/recluster).

### 2b. Scheduling + surfacing

- **Cron:** add a second daily stage after `ingest-incoming` (or fold it into
  the same cron line) that runs `maintenance-sweep`. Ingest already lands +
  CLIP-indexes new photos; the sweep then enriches them. Keep it `--no-colors`
  optional vs. a slower weekly full sweep — TBD (see open questions).
- **`/status` button:** a "Run maintenance sweep" card (SSE, dry-run toggle),
  same pattern as the Ingest-incoming and Stacking cards. Shows per-pass
  counts as it goes.
- **Caveat to honor:** `recluster-faces` renumbers every unknown cluster and
  clears `ignored_clusters`. A nightly auto-recluster would wipe "ignore"
  decisions and churn the `/merges` landscape daily. So either (a) gate
  recluster behind a separate, less-frequent cadence, or (b) make it opt-in
  in the sweep. Decide before wiring it to cron.

## 3. Data-integrity validation + repair (the corrupt `date_taken` problem)

Surfaced live in M24a: a few production rows have **corrupt `date_taken`**
(stray control bytes, e.g. `'\x18u'`), which poisoned `MIN/MAX(date_taken)`
in `get_library_overview` and skews any date sort. We worked around it with a
GLOB filter, but the bad data should be *fixed*, not just filtered.

### `photosearch validate-data` / `repair-data`

A read-only validator (`--dry-run` default) that scans for invalid rows and a
`--apply` mode that repairs them:

- **Corrupt `date_taken`** — anything not matching `YYYY-MM-DD HH:MM:SS`.
  Repair cascade: re-extract from EXIF → fall back to folder-name date →
  fall back to file mtime (`date_created`) → if all fail, NULL it (so it
  sorts to the tail instead of emitting control bytes). Relates to the
  existing `backfill-dates` command (which fills `date_created` from mtime).
- **Out-of-range / zero GPS** — `gps_lat`/`gps_lon` of 0,0 or outside
  [-90,90]/[-180,180]; null them + clear `place_name`/`location_source`.
- **Malformed JSON columns** — `tags`/`categories`/`visual_tags`/`keywords`/
  `dominant_colors` that don't parse as arrays.
- **Orphaned vec rows** — already handled by `cleanup-orphans`; the validator
  should *report* them and point at that command (don't duplicate).
- **Garbage tag sets** — already handled by `clean-garbage-tags`; report +
  point.

Output a per-category count in dry-run; `--apply` repairs in one transaction
per category with the same streaming/idempotent shape. This becomes another
optional stage of the maintenance sweep (or stays a separate ad-hoc tool —
see open questions).

## 4. Open questions (resolve at kickoff)

- One mega-command (`maintenance-sweep` with `--no-colors`/`--skip-recluster`
  flags) vs. a thin orchestrator that shells the existing CLIs? Leaning
  single in-process function that calls the existing module functions, for
  shared DB connection + progress.
- Cron cadence: nightly light sweep (geocode/normalize/colors/infer/stack)
  + weekly heavy (recluster)? Or event-driven off ingest's `new_dirs`?
- Should the sweep be scoped to "rows touched since last sweep" (a watermark)
  for speed on a 163k library, or always full-scan with missing-only
  predicates? Full-scan is simplest and the predicates already make it cheap;
  measure before optimizing.
- Is `validate-data`/`repair-data` part of the sweep or a separate
  manually-run safety tool? (Repairs are destructive-ish; maybe keep apply
  manual.)

## 5. Why this is queued after M24b

M24b (the in-app `/api/ask` agent + "Ask" mode) is mid-flight and higher
value to finish first. This plan is pure operational hardening — it makes
existing passes reliable, adds no user-facing search capability — so it
waits. Pick it up once M24b ships.

## Related

- `docs/plans/llm-driven-search.md` — M24 (a shipped, b in progress).
- `docs/plans/infer-location-refinements.md` — cascade fixes that the
  infer-locations step of the sweep should incorporate.
- `docs/plans/search-accuracy-improvements.md` — structured location columns
  (the `normalize-places` output this sweep would keep current).

# Search accuracy & ranking improvements

Next-milestone plan for the search pipeline. Some of the pain below was
partially addressed inline during M19 / /map / /geotag rollouts (country
codes, Nominatim bbox + padding, cache self-heal, pagination, regex
tolerance for commas/hyphens). What remains are the ranking and
structural issues that need a deliberate pass.

## Current pipeline

```
query string
  └─ regex extraction  → location, date, person names
  └─ filter calls      → search_by_person, _search_by_location,
                         search_by_color, search_by_date, …
                         (each returns its own set; uncapped prefetch)
  └─ set intersection  (by photo_id)
  └─ dedup by file_hash
  └─ sort              by first filter's data (usually date_taken ASC)
  └─ [offset:offset+limit]
```

Each filter is binary (photo matches or doesn't). Scores are never
combined. CLIP semantic is the only ranked signal, and only when it's
the primary filter.

## Observed pain

From real use on the 144k-photo NAS library:

1. **"Newest first" is actually "newest of the oldest 1000".** The
   backend paginates `merged[offset:offset+limit]` from whatever order
   the first filter set produced. `search_by_person` and
   `_search_by_location` both do `ORDER BY date_taken LIMIT ?`, and
   SQLite sorts NULLs **first** in ASC — so the returned page is
   biased toward NULL-date + oldest photos. The frontend's
   `PS.applySortOrder(results, 'date_desc')` then sorts those 1000
   newest-first, but they're already the oldest slice of the real
   set. Concrete symptom: `?q=Calvin` with "Newest first" doesn't
   show 2026 photos; NULL-date photos (like id 41435 with no EXIF)
   surface at the top because NULLs sort first in SQLite ASC.
   Family searches almost always want recent-first, so this is
   currently wrong by default, not just unranked.
2. **Single-filter ranking.** For `"Calvin at the beach"`, a photo
   where Calvin's face matches at 95% and CLIP scores "beach" at 0.85
   ranks the same as one at 50% / 0.40. Both pass the intersection;
   downstream sort ignores the confidence spread.
3. **No typo tolerance.** `"Calvvin"` silently returns zero — no "did
   you mean Calvin?" fallback. Same for mis-spelled places.
4. **Ambiguity silently resolved.** `"in Paris"` → Nominatim's #1
   (Paris, France) wins even if the user's library is 90% Paris, TX
   photos. No disambiguation UI, no GPS-prior tiebreaker.
5. **`place_name` is a flat string.** We paper over this with Nominatim
   bbox unions, but `"Marin County"` as a query still fails because
   `reverse_geocoder`'s output shape is `Locality, Admin1, CC` — admin2
   never appears, so substring-matching for a county name produces 0
   + whatever Nominatim's county bbox happens to overlap (small).
6. **CLIP is image-level.** Can't distinguish "people at the beach"
   from "people near a beach-themed poster". The existing description +
   tag boosts help but weights are hardcoded.
7. **Nominatim is a hard external dependency** for non-country
   location queries. If the NAS loses internet, location search
   degrades to substring-only (18 results instead of 5,946 for the San
   Rafael case from the rollout). Cache helps for repeat queries, not
   cold ones.

## Ordered action items

### Ship soon

1. **Reciprocal Rank Fusion across filters.** For each result set,
   assign ranks; combine via `score = Σ 1/(k + rank_i)` with
   `k = 60` (textbook). Replaces the current "use result_sets[0] for
   ranking" heuristic. ~50 lines in `search.py`, no new dependencies.
   Immediately fixes pain #2.

2. **Sort-before-slice + recency decay.** Two coupled fixes for
   pain #1:

   - **Sort before pagination**, not inside the SQL of the first
     filter. Today `search_combined` returns `merged[offset:offset
     + limit]` where `merged` preserves filter-set insertion order —
     which is whatever the first SQL `ORDER BY` produced. Apply the
     caller-requested sort (`date_desc`, `date_asc`,
     `quality_desc`, `relevance`) to `merged` BEFORE slicing, with
     NULL-date photos always at the tail regardless of direction.
     One-line change in `api_search` to accept a `sort` param +
     ~10 lines in `search_combined` for the sort helpers.
   - **Recency decay** on relevance-mode scores. Once RRF lands,
     multiply each photo's fused score by `exp(-years_ago * 0.1)`
     (tunable). Users searching `"beach"` get recent beach photos
     first; strict-relevance seekers can opt out with
     `sort=relevance_strict`. Makes family searches feel "native"
     without forcing the user to pick a sort.
   - **Plumb the frontend sort dropdown through** to
     `/api/search?sort=...`. Today the sortBy state is only applied
     client-side to the 1000-photo page; switching to "Newest
     first" doesn't re-fetch with the right order. Trivial:
     `doSearch` / `loadMore` append `&sort=${sortBy}`.

3. **Structured location columns.** Add `country`, `admin1`, `admin2`,
   `locality` to `photos`. Schema v19. Backfill via a new CLI
   (`photosearch normalize-places`) that re-runs reverse_geocoder
   per-row and splits the result. `_search_by_location` gains a
   pre-step: if the query matches a known country / admin1 / admin2 /
   locality value in the DB, filter on that column directly instead
   of substring LIKE. Fixes pain #5 and reduces Nominatim round-trips
   to only truly new places. Also unlocks map view's "filter by
   region" and radius search (already sketched in
   `docs/plans/bulk-set-location.md`).

### Iterate

4. **Fuzzy name + place matching.** Enable SQLite's `fts5` virtual
   table over `persons.name` and `photos.place_name`, or ship a
   trigram similarity Python helper (library size: zero extra deps,
   can use `difflib` for short lists). On an extracted-person or
   location term, fall back to nearest-neighbour if exact match is
   empty. Catches `"Calvvin"`, `"San Raphael"`. Fixes pain #3.

5. **Ambiguity disambiguation via photo priors.** When `forward_geocode`
   returns multiple high-importance candidates (e.g. Paris, FR and
   Paris, TX, both at importance > 0.4), score each by how many
   library photos fall in its bbox. Pick the one with more photos; if
   they're close, surface a "Did you mean…?" chip in the UI. Fixes
   pain #4. Requires the map data already present.

6. **LLM query rewriter (optional).** A 3-7B local model
   (Qwen2.5-3B, Llama-3.2-3B) parses free text into structured intent:
   `{person, location, date_range, semantic, sort_by, exclude}`.
   Handles "Calvin around Tahoe during the 2022 summer but not the
   lake house" much more robustly than regex. Requires an Ollama
   model slot and adds ~1-2s latency per query. Behind a
   `smart_parse=true` flag at first.

### Future

7. **VLM re-ranking.** Top-100 from fast search get rescored by
   LLaVA/Qwen2-VL against the raw query string. Slow (2-5s on N100)
   but dramatically improves precision on nuanced queries
   ("people looking sad in snow"). Cacheable per (query, photo_id)
   pair. Fixes pain #6 for queries where the user is willing to
   wait.

8. **Self-hosted Nominatim.** Removes external dependency entirely,
   lets us return admin2 and POI matches without a round-trip, enables
   offline operation. ~20GB for a North America extract; less for
   regional. Fixes pain #7. Biggest cost is the initial bulk import
   and keeping it updated, but a one-off snapshot works fine for a
   personal library.

## Tradeoffs / risks

- **RRF changes the default ranking.** Users who relied on "oldest
  first" from the implicit ASC sort may see a different order after
  a deploy. Add a per-request `sort=date_asc|date_desc|relevance`
  param so the behaviour is explicit.

- **Structured columns are a backfill burden.** 144k photos × one
  `reverse_geocoder.search()` call each ≈ 10-20 minutes on an N100.
  Worker-offloadable. But schema migrations that touch every row
  need a stable code path — run the backfill after the schema bump
  commits, not inline with the migration.

- **Fuzzy matching can hide real zero-result states.** `"Calvvin"` →
  "did you mean Calvin?" is great, but `"Calvvvvin"` → Calvin might
  be wrong. Cap edit distance at 2 and require prefix overlap, or
  gate the suggestion behind a UI chip the user has to accept.

- **LLM rewriter + VLM rerank both want Ollama's single slot.** The
  existing describe/tag pipeline already competes for it. Either add
  request priorities in the Ollama wrapper or serve these queries
  via a lighter dedicated model.

- **Self-hosted Nominatim is operational overhead.** Do it only after
  the usage justifies it — currently one cached round-trip per new
  location covers 99% of searches.

## Non-goals (kept out of scope)

- Replacing CLIP with a better vision backbone (ViT-L, SigLip,
  DINO-v2). The current gap is ranking/fusion, not embedding quality.
  Re-visit only if VLM re-ranking can't close specific precision
  gaps.
- Supporting non-English queries. The regex and country-name table
  are English-only; most filters work independently of language, but
  `"Calvin à Paris"` won't extract the location. Fine for this user.
- Sharding / distributed search. The single-node SQLite setup
  handles 150k photos well; no need.

## Critical files

- `photosearch/search.py` — `search_combined`, `_search_by_location`,
  `search_by_person`, the intersection / sort / limit pipeline
- `photosearch/geocode.py` — regex extraction, country-name mapping,
  Nominatim proxy, cache
- `photosearch/db.py` — schema migrations, `add_photo` stamping,
  reverse_geocoder hookup
- `photosearch/web.py` — `/api/search` endpoint, request logging
- `frontend/dist/index.html` — search page UX, pagination, result
  count / ranking surfacing

## Quick wins bundle

If picking just one ship, do **sort-before-slice + RRF + recency
decay + structured location columns** together:

- ~250 lines net across `search.py`, `db.py`, `cli.py`,
  `frontend/dist/index.html`
- One schema migration (v19) + one backfill CLI
- Fixes "Newest first isn't newest first" (pain #1) — the most
  visible regression today
- Fixes multi-filter ranking incoherence (pain #2)
- Fixes "Marin County as a query returns nothing" (pain #5)
- Reduces Nominatim round-trips to genuinely-new places only
- No new external dependencies

Order within the bundle (land one commit at a time, each a real
improvement even if later ones slip):

1. **Sort param + sort-before-slice** — smallest change, most user-
   visible win. Immediately fixes the "Calvin → NULL dates on top"
   symptom. No schema or ranking math.
2. **RRF across filters** — replaces the "use result_sets[0] order"
   heuristic with a proper fused score. Enables sane multi-filter
   ranking.
3. **Recency decay** — one-line multiplier on the RRF score. Makes
   family searches feel native.
4. **Structured location columns + backfill** — schema migration,
   CLI, and `_search_by_location` pre-step that checks structured
   columns before falling back to substring + Nominatim. Biggest
   change in isolation but landed independently.

Subsequent work (fuzzy, disambiguation, LLM rewriter, VLM rerank,
self-hosted Nominatim) is independent and can be picked up one at a
time as specific pain surfaces.

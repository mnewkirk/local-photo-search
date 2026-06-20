# Search / LLM-driven-search improvement backlog

Running, prioritized list of improvements to natural-language photo search,
seeded by the M24 agent bake-off (`evals/bakeoff.py`) findings. Roughly ordered
by impact-per-effort. Items move out to their own `docs/plans/` doc when picked
up.

## In progress / done

- **Prompt improvements (one pass) — DONE.** `photosearch/agent.py`:
  pre-inject **library facts** (people, places, categories, visual_tags, date
  span — cached) into the system prompt so models plan straight to
  `search_photos` without spending tool calls/nudges to discover them; few-shot
  examples; result-count discipline (relax-on-0 / narrow-on-huge); group-term
  resolution; configurable household glossary via `PHOTOSEARCH_AGENT_HINTS`
  (e.g. "the kids = Calvin, Ellie"). Validated in bake-off round 4.
- Earlier agent fixes: stringified-array coercion, empty-turn nudge,
  best→quality, exclusion (`-people`) syntax.

## Next (committed order)

1. **`summarize` / faceting tool.** The bake-off's p10 ("which year were we in
   both NY and France?") failed for ALL models because the tools expose
   retrieval but not **aggregation** — there's no way to get "the set of years
   for a location." Add a tool like `summarize(filters, group_by="year"|"month"
   |"location"|"person")` returning counts per bucket. Unlocks "which year /
   how many / when / how often" questions and makes multi-hop set-intersection
   solvable. Small, high-leverage.
2. **VLM re-ranking of top-K.** The biggest lever for "find THE photo." Cheaply
   retrieve 30-50 candidates (CLIP + filters), then have a local vision model
   look at each thumbnail and score it against the query; return the best.
   This is the `get_photo_image` path turned into a rerank stage — makes "the
   one where Ellie's blowing out candles" work. Gated/optional per request
   (latency). Pairs with the agent.

## Backlog (impact-ordered)

0a. **Subject-prominence ranking** (representatives/search enhancement). "Best
   photo of Matt, one per year — make sure Matt is the *primary* subject
   (foreground, not background), and in group shots fewer than ~4-5 people."
   Today `representatives` ranks each bucket by `aesthetic_score`, so it can
   surface shots where Matt is incidental/background (observed bad for 2009,
   2012, 2013; e.g. DSCN1692, "Christmas Party 037"). We HAVE the data to fix
   it: `faces` has per-face `bbox_top/bottom/left/right` + `det_score` +
   `person_id`, and `photos` has `image_width/height`. Add a `rank_by=subject`
   mode that scores each candidate by the filtered person's **face-area
   fraction** (their bbox area ÷ image area → foreground vs. background) with a
   penalty for high total face count (favor ≤4-5 people) and low det_score,
   optionally blended with aesthetic. Requires a person filter (you rank by
   *that* person's prominence). Then "best of Matt per year" returns shots
   where Matt is actually the subject.

0. **Top-N-per-bucket / diversified results** (quick, high-utility). Requests
   like "best photo of Matt, one per year for the last 10 years" or "a few from
   each trip" can't be expressed today — `search_photos` returns a flat
   quality-ranked list, so the agent returns the global top-50, not one per
   year. `summarize` counts per bucket but returns no photos. Add either a
   `representatives(filters, bucket=year|month|location, n=1)` tool returning
   the top-N (by quality) photos *per bucket*, or a `one_per`/`diversify` option
   on `search_photos`. SQL window-function (`ROW_NUMBER() OVER (PARTITION BY
   bucket ORDER BY aesthetic_score DESC)`) makes this cheap. Observed live
   2026-06-20 ("one per year" returned 50 of 3360).

3. **Events / trips metadata.** Cluster photos by time + GPS + place into
   events; "our France trip" becomes a first-class entity with a date range.
   Also the clean way to answer multi-hop year questions (read two trips'
   years, intersect) without LLM date arithmetic. Extend stacking's union-find
   to multi-day. Highest-value *new metadata*.
4. **OCR / text-in-image.** Birthday banners, cake text, jersey numbers, signs,
   scoreboards, dates. Directly improves "birthday", "soccer", specific-event
   queries. One GPU extraction pass → a searchable field.
5. **Description embeddings (hybrid text+image retrieval).** ~157k LLaVA
   descriptions are matched today by `LIKE`/keyword scoring. Embed them in a
   vector index and RRF-fuse image-CLIP + description-text + filters. Big recall
   win on content queries, reusing existing data. (sqlite-vec already present —
   no new store needed.)
6. **Upgrade the embedding model.** CLIP ViT-B/16 is small/old; SigLIP /
   EVA-CLIP / a larger CLIP improves raw semantic recall across the board (one
   re-embed pass + `--force-clip`).
7. **Plant & animal identification (Google Lens style).** Identify species in
   photos — "show me the hawk photos", "which flowers did we see in Hawaii".
   Approach: a specialized classifier (iNaturalist / PlantNet-style model) or a
   vision-LLM with a species taxonomy, producing a `species` / `taxa` tag set
   per photo. **If a reference species database is needed, we can build/host one**
   (iNaturalist taxonomy is open). Stored like the other tag passes; searchable
   via a new `species` filter or folded into keywords. Complements the generic
   object/category tags with fine-grained natural-history labels.
8. **Quality-scoring evaluation harness.** Sanity-check the current aesthetic
   scorer (CLIP ViT-L/14 + linear MLP, observed range ~3.68-5.99 — suspiciously
   narrow). Gather a reference set of known high-quality images (National
   Geographic / photo-contest winners, plus some deliberately poor shots) and
   run the scorer against them: do winners score near the top, snapshots near
   the bottom, is the spread meaningful? Tells us whether `min_quality` /
   `sort=quality_desc` actually discriminate, and whether to retrain/replace the
   scorer. Mirror the agent `evals/` pattern (a script + an HTML report).
9. **Visual-style fingerprint + style clustering** (a colleague's approach,
   worth adopting). Have the vision model write a *rich visual-style*
   description per photo — lighting, mood, composition, tonal character — then
   embed that text (e.g. nomic-embed-text via the local LLM) into a vector, and:
   (a) cosine-similarity for "find photos with similar visual *character*"
   ("more like this"), (b) K-means / clustering to auto-group by style/mood.
   This is the aesthetic/style analog of item 5 (description embeddings),
   specialized to *style* rather than *content*. Notes for our stack: we already
   generate descriptions and already have **sqlite-vec** (no need for DuckDB) —
   so this is a style-description pass + an embedding pass + a similarity/cluster
   surface. Enables a "similar vibe" button and mood-based albums.
10. **Household / relationships table.** Promote the `PHOTOSEARCH_AGENT_HINTS`
    glossary to structured data (who's who, family groups) for deterministic
    "the kids" / "the whole family" / "Calvin's grandparents" resolution across
    the app, not just the agent prompt.
11. **Finish structured location columns** (`country/admin1/admin2/locality`).
    Partly there (`normalize-places`); fully backfill so region queries ("Marin
    County") are reliable. See `search-accuracy-improvements.md`.
12. **People attributes** — group-vs-solo, person count, rough age (kid vs
    adult) — for "group photos" / "the kids" without a roster.
13. **Calendar / holiday enrichment** — map dates to holidays / known birthdays
    → "Christmas morning", "Calvin's birthday".
14. **ANN index at scale.** sqlite-vec brute-force is fine at 163k; past ~1-2M
    photos, an HNSW index (hnswlib/faiss) keeps latency flat. Not urgent.

## Related plans
- `docs/plans/llm-driven-search.md` — M24 (shipped): the agent + tool layer.
- `docs/plans/local-replica-and-writes.md` — M26: run it on the GPU box.
- `docs/plans/search-accuracy-improvements.md` — RRF / recency / structured
  location columns.
- `evals/bakeoff.py` — the model bake-off harness + HTML report.

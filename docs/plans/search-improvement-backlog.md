# Search / LLM-driven-search improvement backlog

Running, prioritized list of improvements to natural-language photo search,
seeded by the M24 agent bake-off (`evals/bakeoff.py`) findings. Roughly ordered
by impact-per-effort. Items move out to their own `docs/plans/` doc when picked
up.

## In progress / done

- **`summarize` / faceting tool — SHIPPED.** `tools.py:_h_summarize` +
  `summarize` ToolSpec, agent system-prompt rule, 7 tests
  (`test_tools.py::test_summarize_*`). Counts by year/month/location/person/
  camera_model; the multi-hop "which year in both X and Y" path is documented in
  the prompt. (Was "Next #1".)
- **VLM re-ranking — SHIPPED (core).** `tools.py:_h_rerank_photos` + `rerank_photos`
  ToolSpec (per-image vision scoring, parallel, `top_n`, no-model fallback),
  agent prompt rules, MCP exposure, 4 tests. Open: precision eval +
  vision-model bakeoff — see `docs/plans/vlm-reranking.md`. (Was "Next #2".)
- **Subject-prominence + top-N-per-bucket — SHIPPED.** `representatives` tool
  (`rank_by=quality|subject`, dedupe) + `search_photos(sort='subject')` flat
  variant. Ranks by the named person's face-area fraction (foreground vs
  background) with a prominence sweet-spot band. Tests + agent prompt rules in.
  (Was backlog #0 / #0a.)
- **Prompt improvements (one pass) — DONE.** `photosearch/agent.py`:
  pre-inject **library facts** (people, places, categories, visual_tags, date
  span — cached) into the system prompt so models plan straight to
  `search_photos` without spending tool calls/nudges to discover them; few-shot
  examples; result-count discipline (relax-on-0 / narrow-on-huge); group-term
  resolution; configurable household glossary via `PHOTOSEARCH_AGENT_HINTS`
  (e.g. "the kids = Calvin, Ellie"). Validated in bake-off round 4.
- **Face-framing metadata filters — SHIPPED (2026-06-23).**
  `only_these_people` (total detected faces == named-people count → no extras)
  and `faces_in_frame` (no face bbox at the image edge; scoped to the named
  people when given) on `search_photos`/`summarize`/`representatives`, wired
  through both the post-filter and `_build_filter_sql` paths. Reframes "only
  the four of us / nobody cropped" as metadata, not vision — agent applies the
  flags first and reserves `rerank_photos` for true visual criteria
  (eyes/smiles). 7 tests. From the Ask-log review (the query that previously
  timed out at 92s on per-result `get_photo` fan-out now runs in ~13s).
- **`representatives` `max_buckets` — SHIPPED (2026-06-23).** Completes backlog
  #0: caps the result to the best K buckets (ranked by each bucket's top photo)
  and orders output best-first, so "top N photos, no more than 1 from each
  location" = `bucket='location', n=1, max_buckets=N`. Before, that was
  inexpressible and the agent stuffed N into `n` (per-bucket) → ~10 per
  location. 4 tests. From the Ask-log review.
- Earlier agent fixes: stringified-array coercion, empty-turn nudge,
  best→quality, exclusion (`-people`) syntax.

## Next (committed order)

The original top two here (`summarize`, VLM re-ranking) both **SHIPPED** — see
"In progress / done" above. What's actually next:

1. **VLM re-ranking validation.** The tool/agent/MCP/tests are in; the precision
   eval (`evals/rerank_eval.py`, top-1/top-3 vs baseline) and the vision-model
   bakeoff to pick `PHOTOSEARCH_LLM_VISUAL_MODEL` are the open items. Needs a
   vision model loaded in the local LM Studio. See `docs/plans/vlm-reranking.md`.
2. **M26b write tools** (`docs/plans/local-replica-and-writes.md`) — genuinely
   not started; the next big buildable search/agent feature.

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

0. **Top-N-per-bucket / diversified results** — **SHIPPED.** The
   `representatives(filters, bucket=year|month|location|person|camera_model, n)`
   tool returns the top-N (by quality, or `rank_by=subject`) photos *per bucket*
   via `ROW_NUMBER() OVER (PARTITION BY bucket …)`. The "top-N overall, one per
   bucket" cap (`max_buckets`) landed 2026-06-23 — see "In progress / done".
   (Original motivation: "one per year" returned 50 of 3360, observed
   2026-06-20.)

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

# Categories + keywords redesign

Status: **design** — written 2026-05-17. Phases 0.1–0.2, 1, 2.1–2.4, 3.1–3.3 implemented; remaining phases pending vocab curation + backfill.

Phase 0 keyword model: **llama3.2:3b** (bakeoff 2026-05-18, N=30 sample, head-to-head vs qwen2.5:3b: 8.5 vs 12 median kw, both ~0.44s/call, llama 8.4% strict-substring "hallucinations" — mostly paraphrases — vs qwen 13.6% real ones including tokenizer-mashed compounds like "grasshillside"; llama wins on multi-word phrase preservation).

Replaces the M9 single-`tags` column (78-word fixed vocab, LLaVA
vision-pass) with a richer two-axis design: structured `categories`
mined from your actual library plus free-form `keywords` extracted
from descriptions.

## Why this exists

On the 135k-photo NAS library as of 2026-05-17:

- 133,406 photos (98%) have descriptions.
- 43,533 photos have any tags.
- **89,873 photos have descriptions but no tags.**

The gap is structural, not transient. The current `--tags` pass sends
every photo to LLaVA with a "pick from this 78-word list" prompt, then
`_parse_tag_response` discards every word not in the vocabulary. The
two dominant failure modes:

1. **Parse-empty** — LLaVA returned a fluent description-style
   response with no in-vocab words. Tagger returns `None`,
   `worker_api.py:362` skips the write, photo stays NULL.
2. **Regurgitation** — LLaVA echoed most of the vocabulary back. The
   `_MAX_PLAUSIBLE_TAGS` guard catches it and returns `None`.

Of the ~24k photos that have been attempted (under the retry cap), the
vast majority are parse-empty silent failures. Only 12 photos have hit
the 3-attempt exhaustion cap. The `index_errors` table for `tags` has
24 rows total, all "database is locked" — the failure mode isn't
crashing, it's just silently producing nothing.

The user's intent: tags should be **richer than a 78-word vocab can
express**, and should ideally derive from the description (which is
already 98% complete and contains the real information about each
photo). At the same time, a controlled vocabulary still has value for
faceted filtering, search query expansion, and UI tag clouds. The
design splits these two needs into two fields rather than fighting one.

## Goal

Every described photo should land in `categories` with at least one
term (the "coverage net"). Specifically-named subjects, breeds, places,
and objects should appear in `keywords`. Visual-quality tags (mood,
light, composition) should appear in `visual_tags`.

Target coverage on the curation sanity-check (1000-photo sample):
≥95% of described photos get ≥1 category.

## Design

### Schema (v22 → v23)

| Column | Type | Source | Notes |
|---|---|---|---|
| `categories` | TEXT (JSON array) | Renamed from `tags`. Written by `category-content` text-pass | Old 78-vocab values get nulled at migration; mandatory re-tag |
| `visual_tags` | TEXT (JSON array, new) | Written by `category-visual` vision-pass | ~25-word vocab: mood / light / composition |
| `keywords` | TEXT (JSON array, new) | Written by `keywords` text-pass | Free-form, 5-15 per photo, lowercased |
| `tags_v22_backup` | TEXT (JSON array, new, temporary) | Snapshot of pre-migration `tags` content | Dropped in Phase 7 after a week of confidence |

Migration is wrapped in a single transaction. `tags_v22_backup` makes
rollback a one-line UPDATE.

`generations` gets three new `text_type` values: `'category-content'`,
`'category-visual'`, `'keywords'`. Existing `'tags'` rows migrate to
`'category-content-legacy'` for provenance continuity.

`worker_processed.pass_type` gains three values: `'category-content'`,
`'category-visual'`, `'keywords'`. The old `'tags'` value stays in
historical rows but no new code claims against it; the
`MAX_PROCESS_ATTEMPTS` exclusion still applies per-pass.

Two columns instead of one merged `categories` because each pass owns
its column entirely — no merge logic, no race conditions, no per-pass
provenance sidecar.

### Three new passes

#### `category-content` (text-only)

- **Input:** `photos.description` (skip rows where description IS NULL)
- **Model:** Llama 3.2 text-only via Ollama (default; Phase 0 bakeoff
  may swap to llama3.2-vision text-mode)
- **Prompt:** "Given this description, return tags from this
  vocabulary that apply: [content vocab]. Comma-separated, only tags
  from the list, no others."
- **Output:** JSON array, written to `photos.categories`
- **Batching:** ~32 photos per worker batch (text is fast)
- **Failure mode:** parse-empty → `[]` (not None) + processed.
  No regurgitation guard required — text-mode LLMs handle constrained
  prompts reliably. Verify in Phase 0 bakeoff.

#### `category-visual` (vision)

- **Input:** the image file
- **Model:** `llava` (same as the current visual tags pass)
- **Prompt:** "Pick visual-quality tags from this list: [visual
  vocab ~25 words]. Examples: dramatic, peaceful, foggy, silhouette,
  close-up, aerial."
- **Output:** JSON array, written to `photos.visual_tags`
- **Batching:** Same as the current tags pass (GPU-bound)
- **Failure mode:** Inherits the regurgitation guard from the existing
  `tag_photo`. Guard threshold drops from 16 to 12 (smaller vocab).

#### `keywords` (text-only)

- **Input:** `photos.description`
- **Model:** TBD per Phase 0 bakeoff — `llama3.2:3b` text-only vs
  `llama3.2-vision` in text mode
- **Prompt:** "Extract 5-15 keywords/phrases from this description.
  Include proper nouns, multi-word phrases (e.g. 'golden retriever',
  'pacific ocean'), breeds, locations, named subjects."
- **Output:** JSON array, lowercased (sentence-initial / proper-noun
  case is too inconsistent across the LLM's outputs to be worth
  preserving — see "Case handling" below)
- **Batching:** ~32 per batch

All three passes follow the existing `worker_processed` retry pattern:
`attempts >= MAX_PROCESS_ATTEMPTS` exclusion at claim time,
mark-processed on empty result (parse-empty doesn't burn a retry).

`category-content` and `keywords` both gate on `description IS NOT NULL`.
`category-visual` has no dependency.

The current `--tags` CLI flag, `tag_photo`, and `TAG_VOCABULARY` get
deleted. The status-page tags stat card, the `tagged` field on
`/api/stats`, the `_QUERY_TO_TAGS` table in `search.py`, and the
`_tags_match_query` function are all replaced.

### Vocab construction (one-time)

Two files produced: `photosearch/vocab_content.py` (~225 words) and
`photosearch/vocab_visual.py` (~25 words). The synonym/expansion map
(`photosearch/vocab_query_expansion.py`) is generated alongside them.

**Step 1: Mine candidates from descriptions**

New CLI: `photosearch mine-vocab --out /data/vocab_candidates.json`.
Reads all 133k descriptions, runs spaCy noun-chunk extraction +
lemmatization, emits a frequency-sorted list of noun/adjective phrases
with count ≥ 50. Expected output: 2,000-4,000 candidate terms.

**Step 2: Auto-group with an LLM**

Llama 3.2 text-only with a "group these terms into semantic
categories" prompt, chunked since the candidate list won't fit in one
context window. Outputs `/data/vocab_proposal.json` with a draft
hierarchy: `{animals: [...], landscapes: [...], ...}`.

**Step 3: Manual curation via the `/admin/vocab` UI**

Two-pane layout:

- Left pane: all candidate terms grouped by the LLM's buckets, with
  frequency counts. Filterable, searchable. Click toggles
  include/exclude.
- Right pane: the in-progress vocab, with `content` and `visual` tabs.
  Terms can be moved between tabs, removed, or have synonym groups
  edited.

Live coverage preview at the top: "If committed, X% of 1000-photo
sample would get ≥1 content tag." Recomputed on demand, not real-time.

Three persistence actions:

- `Save draft` — writes `/data/vocab_draft.json`. Resumable.
- `Compile` — generates `photosearch/vocab_content.py`,
  `photosearch/vocab_visual.py`, and `photosearch/vocab_query_expansion.py`.
  Writes to the repo workdir for human commit.
- `Test on photo` — input a photo ID, see what categories would land
  on it with the current draft.

Backend endpoints under `/api/admin/vocab/`:

| Endpoint | Purpose |
|---|---|
| `GET /candidates` | Read `/data/vocab_candidates.json` |
| `GET/PUT /draft` | Read/write `/data/vocab_draft.json` |
| `POST /coverage-preview` | Run draft vocab against N sample photos, return % covered |
| `POST /compile` | Generate the Python modules |
| `POST /test-photo/{id}` | Apply draft to a single photo |

**Step 4: Coverage sanity-check**

Run the new `category-content` pass against a 1,000-photo sample.
Verify ≥95% of photos get ≥1 category. If not, iterate in the
curator UI.

Expected human time: 1-2 hours of curation in Step 3. Steps 1, 2, and
4 are scripted.

### Search-side changes

Three new match functions in `search.py`:

```python
_categories_match_query(categories_json, query) -> float
_visual_match_query(visual_tags_json, query) -> float
_keywords_match_query(keywords_json, query) -> float
```

- `_categories_match_query` — exact-match + `_QUERY_TO_CATEGORIES`
  expansion (regenerated by the curator's `Compile` step)
- `_visual_match_query` — exact-match against the visual vocab; tiny
  expansion map (`peaceful` / `calm` / `serene` → `peaceful`)
- `_keywords_match_query` — token-overlap; multi-word phrase tokens
  match if the query contains the whole phrase; no vocab expansion
  (keywords are already specific)

Ranking weights: `keywords > categories > visual > description literal`.
Keywords get the strongest boost because they're the most specific
signal. Exact constants land in code review.

The existing `tag_match` parameter on `/api/search` is renamed to
`text_match` with values:

- `"all"` (default) — union all four signals
- `"categories"`, `"keywords"`, `"visual"`, `"dict"` — single-source
  diagnostic modes
- `"off"` — no text relevance, pure CLIP

Three new optional filters on `/api/search`:

- `category=beach` — must have `beach` in `categories`
- `keyword=tahoe` — substring match against `keywords`
- `visual_tag=dramatic` — must have `dramatic` in `visual_tags`

All three AND-intersect with existing filters via the `result_sets`
pattern.

**Person extraction (`_extract_persons_from_query`) keeps precedence
over keywords.** A query of "Calvin at the beach" still routes
"Calvin" to the persons filter (registered person → face matches), not
to keywords. Only unregistered names fall through to keywords.

**Filename auto-detection (`_looks_like_filename`) unchanged** — camera
filenames (`DSC06241`) still route to SQL LIKE before any of this.

`/api/search` and `/api/photos/{id}` responses gain `categories`,
`visual_tags`, `keywords` fields. The old `tags` field is removed
cleanly — no compat shim. Bookmarked deep links (`/?tag=beach`) get a
redirect rule in the search API (`tag=X` → `category=X`).

### Frontend changes

#### `PS.PhotoModal` (`shared.js`)

Replaces the single "Tags" row with three:

- `Categories` — chips, clickable to search `?category=<term>`
- `Visual` — chips, clickable to search `?visual_tag=<term>`
- `Keywords` — chips (smaller, looser style to signal free-form),
  clickable to search `?keyword=<term>`

Each row hides itself when its field is empty.

#### Search page (`index.html`)

Three new filter controls above the result grid:

- Category multi-select (populated from `/api/admin/vocab/content`)
- Visual tag multi-select (populated from `/api/admin/vocab/visual`)
- Keyword text input (free-form, substring match)

All AND-intersect with existing filters. The existing free-text query
keeps its current behavior.

#### `/status` page

The Tags stat card splits into three: Categories, Visual tags,
Keywords. All three reuse the existing progress-bar pattern.
`/api/stats` gains `category_tagged`, `visual_tagged`, `keyword_tagged`
fields; old `tagged` field removed.

The Workers panel needs no change — it reads `worker_processed.pass_type`
dynamically. The three new pass types auto-appear in the per-pass
queue pills on first claim.

Run-command snippets on the status page: replace the single `--tags`
row with three rows (one per new pass) + a "Run all three" row that
backgrounds them in parallel.

#### `/admin/vocab` page (new)

Curator UI sketched above.

#### Header / nav

`PS.SharedHeader` gains an "Admin" entry linking to a thin admin
landing page (links to `/status` and `/admin/vocab`). Alternative:
just add `/admin/vocab` directly to main nav under its own entry —
simpler if no more admin tools land soon.

### Case handling for keywords

Store lowercase, match case-insensitively. The hard cases:

- **Sentence-initial capitals** — "Beach scene with a child." Is
  "Beach" a proper noun or just sentence-start? LLM produces both
  `"Beach"` and `"beach"` across 133k descriptions; deduping at query
  time is wasted work.
- **Multi-word proper nouns** — "Pacific Ocean" may come back as one
  phrase or two terms. Lowercased to `"pacific ocean"`, searchable
  but loses the proper-noun cue.
- **Stylized brand caps** — `iPhone` → `iphone` (searchable, looks
  slightly bad).
- **Acronyms** — `BMX` → `bmx` (same).
- **Names disambiguation** — "Mike" the person vs "mike" the
  equipment is real but rare; person identity in this library is
  canonically in the face data, not the description.

The whole reason to add `keywords` is recall, not display. If the UI
later wants pretty case, a post-process display rule (title-case
multi-word phrases; small special-case lookup for `iPhone` /
`eBay` / `USA` etc.) is cheap to bolt on.

## Error handling

- **Text passes (category-content, keywords):** Ollama exception →
  `index_errors` row + `[]` result + processed. Parse-empty → `[]` +
  processed.
- **Visual pass (category-visual):** regurgitation guard inherited
  from `tag_photo`, threshold dropped to 12. Parse-empty → `None`
  (still doesn't write — same as today).
- **Vocab curator API:** compile rejects when content or visual vocab
  is empty, when a term appears in both lists, when fewer than 50
  content terms (sanity floor).
- **Schema migration:** wrapped in a single transaction. Backs up
  `photos.tags` to `photos.tags_v22_backup` before nulling so a botched
  re-tag is recoverable.

## Testing

- `tests/test_search.py` — extend with cases for
  `_categories_match_query`, `_visual_match_query`,
  `_keywords_match_query`, and the new `text_match` modes.
- `tests/test_vocab_mining.py` (new) — noun-phrase mining over
  fixture descriptions.
- `tests/test_vocab_admin.py` (new) — curator API endpoints: draft
  save/load, coverage preview, compile pre-conditions.
- `tests/test_worker_api.py` — extend with end-to-end coverage of all
  three new pass types.
- `tests/test_db.py` — test that creates a v22 DB, runs the migration,
  asserts column shape + backup table contents.

## Rollout phasing

| Phase | What | Reversible? |
|---|---|---|
| 0 | Keyword extraction bakeoff: run both candidate models against 30 sample descriptions side-by-side. Pick one. | N/A |
| 1 | Schema v22→v23 migration. New pass types claimable but no workers yet. | Yes — restore from `tags_v22_backup` |
| 2 | Mining CLI + vocab curator page built. Run mining + LLM grouping → `vocab_proposal.json`. | Yes — output files only |
| 3 | Curate vocab in `/admin/vocab`. Compile produces Python modules. Commit. | Yes — repo revert |
| 4 | Backend code for the three passes (`worker.py`, `worker_api.py`, `index.py`, `cli.py`). Search-side updates (`search.py`, `web.py`). Deploy. | Yes — code revert + null out new columns |
| 5 | Backfill: three worker fleets in parallel. Text passes finish in hours; visual pass ~3h with 3 Mac-native workers. | Yes — null + re-claim |
| 6 | Frontend updates (PhotoModal, search filters, status page split). Deploy. | Yes — code revert |
| 7 | Drop `tags_v22_backup` column after a week of confidence. | One-way after this |

Phases 0–4 land in one branch. Phase 5 is operational, no code change.
Phases 6–7 follow once backfill has produced enough data to validate
the new fields render.

**Guardrails:**

- `photosearch dump-db --to /data/pre-v23.db` runs *before* the
  migration. (Reuses existing `dump-db` mechanism; one SSH.)
- Restart panel pauses worker traffic during migration via the
  existing shutdown handshake (commit `577a3b3`).
- Vocab compile writes Python modules with a header comment containing
  generating timestamp + draft hash, so the active vocab is traceable
  back to a specific curation session.

## Future potential improvements

Land only if concrete pain surfaces:

- **Visual-quality folded into describe prompt** — if the describe
  prompt is extended to surface mood/light/composition reliably,
  `category-visual` becomes redundant and the column can be deprecated.
  Note kept here so we remember the option.
- **Pretty-case display rule for keywords** — title-case multi-word
  phrases + special-case lookup for branded caps. Cheap to add when
  the UI starts caring.
- **Synonym normalization on extraction** — collapse `puppy` / `pup`
  / `young dog` to one term at extraction time. Currently handled at
  search-time via `_QUERY_TO_CATEGORIES`.
- **Per-keyword photo-count facet** in the search UI — like the
  existing place / person facets. Useful only if keyword cardinality
  stays bounded enough to render.

## Open questions

These got deferred during brainstorming and don't block implementation:

- Admin landing page vs direct `/admin/vocab` nav link. Defer until
  a second admin tool is needed.
- Whether `category-visual` regurgitation guard at 12 is right —
  validate empirically during Phase 5 backfill.
- Whether mining should also include adjective phrases ("blue dress",
  "stormy sky") or just nouns. Defer until Step 1 output makes the
  call obvious.

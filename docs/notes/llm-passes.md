# LLM passes — history and traps

describe / verify / category-* / keywords / aesthetics: model strategy, guards, provenance, the category-visual split, category-content grounding. Condensed rules: `.claude/rules/llm-passes.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 466) on 2026-10-19 -->
### VLM aesthetic scoring (the `aesthetics` pass)

Replaces the compressed **LAION CLIP `aesthetic_score`** (in `quality.py`) as
the *primary* quality signal — that predictor's 1–10 scale tops out ~6.9 even on
curated Unsplash, so the library never exceeds ~6.2. The LAION `quality` pass +
column are **kept as a cheap prior**; the VLM `aesthetics` pass is primary.

`photosearch/aesthetics.py` prompts a vision LLM (qwen*-VL via the existing
`describe._ollama_chat_with_retry(role="aesthetics")` routing — Ollama or LM
Studio) for a rubric-anchored JSON critique: three dimensions — **Technical
Excellence** (sharpness/exposure/depth_of_field/white_balance), **Composition**
(framing/leading_lines/rule_of_thirds/balance), **Impact & Storytelling**
(emotion/originality/wow) — each the mean of its sub-attributes, plus a free-text
style critique (lighting/mood/tonal_character/color) and controlled style tags.

Schema v26 adds `aes_*` columns to `photos`: `aes_overall` (weighted mean,
default weights Technical .30 / Composition .30 / Impact .40 in
`DIMENSION_WEIGHTS`, **retunable without a VLM re-run**), the 3 dimension + 11
sub-attribute scores, `aes_style`/`aes_style_tags` (JSON), and
`aes_overall_pct`. The `aes_technical_iqa`/`aes_overall_iqa` columns were
reserved for an optional objective-IQA anchor (pyiqa MUSIQ/TOPIQ /
VisualQuality-R1). **Phase-0 bakeoff (2026-07-09) rejected the anchor:** on a
28-photo hand-ranked sample, qwen2.5-vl-7b beat both IQA metrics on Spearman vs
the human ranking (ρ **0.70** vs topiq_nr 0.61, musiq 0.55), and the two IQA
metrics correlated with each other (ρ 0.82) more than with the human — i.e. they
measure a technical-cleanliness axis partly orthogonal to actual preference. So
**the VLM (qwen2.5-vl) is the sole primary signal; no IQA anchor.** This keeps
`pyiqa`/MUSIQ out of the production path entirely (it only ever ran in the
bakeoff eval — and was the OOM-killer there until `evals/aesthetics_bakeoff.py`
gained the resolution cap). The `aes_*_iqa` columns stay as unused/reserved
(harmless NULLs); revisit only if a qwen+IQA blend is ever shown to beat qwen
alone. Bakeoff harness + report: `evals/aesthetics_bakeoff.py`.

**The 6.2-ceiling fix is percentile normalization.** VLMs cluster raw scores
6–8 just like LAION compresses, so ranking/search use `aes_overall_pct` — the
library-relative percentile of `aes_overall`. Two no-VLM backfills maintain it:

```bash
photosearch recompute-aesthetic-overall --apply   # after retuning weights
photosearch normalize-aesthetics --apply           # refresh percentiles
```

`normalize-aesthetics` is also a `maintenance-sweep` stage (percentile refresh
only — the VLM scoring stays with the worker fleet). Search: `sort=aesthetic_desc`,
`min_aesthetic` (percentile), `min_technical`/`min_composition`/`min_impact`,
`style_tag`. The `min_quality` filter is the **raw** counterpart of
`min_aesthetic`: it floors on `COALESCE(aes_overall, aesthetic_score)` — the raw
VLM `aes_overall` (1-10) shown on the photo modal, falling back to the legacy
LAION `aesthetic_score` for photos the VLM hasn't scored — so "min quality 5.5"
matches the 5.5 the modal displays (it used to floor on the legacy score alone).
The `/status` "Aesthetics (VLM)" card + `PhotoModal` "Aesthetic
Evaluation" breakdown + M28 re-run checkbox surface it in the UI.

<!-- moved verbatim from CLAUDE.md (line 519) on 2026-10-19 -->
### The percentile refresh is chunked — and must stay that way

`normalize_overall` / `normalize_subject_overall` / `_normalize_by_day` used
to write every scored row through one `executemany` + one commit — 158,111
rows. That holds SQLite's single writer lock for the whole rewrite, and on
2026-09-20 the nightly sweep's `normalize_aesthetics` stage held it long
enough for the GPU fleet's concurrent `submit-results` writes to exhaust
their 60 s `busy_timeout` and fail with `database is locked` — which then
burned those photos' retry attempts (see "A failed SERVER-side write must not
burn a retry attempt"). Symptom: `index_errors` full of `database is locked`
for one pass, timestamped inside a sweep.

Writes now go in `aesthetics.PERCENTILE_CHUNK_ROWS` (2000) chunks with a
commit between each. Three invariants:

- **Every percentile comes from ONE snapshot read**, taken before the first
  chunk. Chunking changes how the answer is written, never what it is — the
  per-day variant likewise computes all days before writing anything. Do not
  "simplify" this into a read-compute-write loop per chunk.
- **The UPDATE is guarded on the score it was derived from**
  (`WHERE id=? AND aes_overall IS ?`). A row rescored by the fleet between
  the snapshot and its chunk is skipped, not given a stale percentile; it
  corrects itself on the next run. Skips are counted and logged. (The float
  `IS ?` round-trip was verified experimentally: 505/505 rows matched, 0
  spurious rewrites on a second run.)
- **Unchanged rows are not rewritten.** The nightly run rewrote all 158k rows
  even when nothing had moved — most of the lock time for no change.

Return values and the `apply=False` contract are unchanged, so the CLI and
the sweep stages are unaffected. The stages pass `on_chunk(done, total)`,
which checks abort and emits SSE progress in the between-chunk gaps. Tests:
`tests/test_aesthetics_normalize_chunking.py`.

**The stage's gate excludes rows that can never be filled.** The nightly log
read `normalize_aesthetics done: would 1373, applied 158111` every night on an
unchanged library. `would 1373` was **1,373 scored photos with no capture day
at all** — `date_taken` AND `date_created` both NULL (Facebook takeout, old
scans; verified read-only on the replica 2026-09-26: 1,373 of 1,373 missing a
day pct, 0 missing a library pct, none from the 2026-09-19 1,373-photo batch —
the equal count is a coincidence). `_normalize_by_day` can never give such a
photo a per-day percentile, but the gate asked `aes_overall_day_pct IS NULL`,
so it stayed true forever and re-ranked the library nightly. The gate
(`maintenance._aesthetics_missing_sql`) now counts a missing day pct only when
`COALESCE(date_taken, date_created) IS NOT NULL` — the same predicate the
writer uses. Same for `normalize_subject_aesthetics` (399 such photos).

**`applied` is rows WRITTEN**, summed across the library and per-day passes
(the `stats=` dict the `normalize_*` functions fill; their int return is
unchanged). It used to be the scored-row count, which is why the log could not
show the skip-unchanged logic working — measured on a replica copy, that
nightly `applied 158111` was really **0 rows written**. The stage result also
carries `considered` (the old number), `unchanged` and `rescored_mid_run`.
Tests: `tests/test_maintenance_incremental.py`.

<!-- moved verbatim from CLAUDE.md (line 573) on 2026-10-19 -->
### Per-pass model strategy (Ollama defaults)

No single vision model wins every pass, so each has its own default
(all overridable: `--describe-model`, `--verify-model`,
`--category-visual-model`, `--category-content-model`, `--keywords-model`):

- **describe / regen → `llama3.2-vision`** — won a 100-image bakeoff over llava
  on free-form description quality (esp. text/OCR-heavy photos).
- **category-visual → `llava`** — the constrained tag task is a different shape;
  llama3.2-vision degenerates and over-selects on it. The
  regurgitation guard (`_VISUAL_MAX_PLAUSIBLE_TAGS`) drops responses echoing
  the vocabulary. `photosearch clean-garbage-tags` clears historical
  regurgitated tag sets (from the legacy `tags` column, whose own threshold is
  `_MAX_PLAUSIBLE_TAGS`) so they get re-tagged. See "derived capture facts vs
  perceived qualities" below for what this pass is now allowed to decide.
- **verify → `llava`** — must differ from the describe/regen model for an
  independent cross-check.

`describe.py` has model-aware Ollama options (llama3.2-vision gets
`temperature` + `repeat_penalty` to suppress repetition loops) plus a
`_is_degenerate` detector → up to 2 retries → llava fallback in `describe_photo`.

These defaults apply when passes run through **Ollama**. When the fleet routes
to an OpenAI-compatible backend (LM Studio — see below), per-pass models are
selected by role env var instead.

<!-- moved verbatim from CLAUDE.md (line 599) on 2026-10-19 -->
### `category-visual`: derived capture facts vs perceived qualities

The visual vocabulary used to be one flat 36-word checklist, and the pass was
measurably broken on half of it. On the live library (159,526 tagged photos)
and one 1,373-frame daytime soccer folder (shutter 1/125–1/800, ISO median 160):

| tag | how often | reality |
|---|---|---|
| `long-exposure` | 877/1,373 of that folder (64%) | slowest frame was 1/125 s → precision **0%**. Library-wide it is **anti-correlated**: on 10.1% of photos faster than 1/250 s but only 25.2% of genuine ≥1/4 s exposures (precision ≈3%, recall 25%) |
| `motion-blur` | 48% | median `aes_sharpness` **7.0 tagged and untagged** — no signal at all |
| `low-light` | 33% | 92% of those at ISO ≤ 400 in full sun |
| `peaceful` | 24% | co-occurs with `dramatic` 216× |

Contradictory pairs were routine (`long-exposure`+`sunny` 565,
`low-light`+`golden-hour` 283, `low-light`+`vibrant` 302), and the model was
largely ignoring the image: 1,373 photos produced only **163 distinct tag
sets**, the top 8 covering 56%, with one verbatim 8-tag set on **101** photos.

**Why: a VLM cannot read a shutter speed off a 336-px tile. EXIF can.** The
vocabulary mixed physically measurable CAPTURE FACTS with mood and
composition, and asked one model to answer both. Note `sunny` and `low-light`
*do* track EXIF library-wide, so it was selectively broken, not useless —
which is why the fix is a split, not a new model.

**The split** (`photosearch/visual_tags_derive.py`) is FOUR disjoint groups
covering the compiled vocabulary exactly:

- **DERIVED** (3) — `long-exposure`, `low-light`, `panoramic`. Computed from
  EXIF, never asked of the model, authoritative over it.
- **RETIRED** (1) — `motion-blur`. Dropped from *both* halves **and stripped
  from existing rows**: a scalar cannot tell motion blur from defocus (a long
  exposure on a tripod is sharp), and the VLM had zero signal. No rule was
  defensible, so there is none and nothing replaces it.
- **FROZEN** (2) — `sharp`, `blurry`. Not derived, not asked, **not deleted**.
  See below.
- **PERCEIVED** (30) — everything else, presented in four prompt sections
  (light-and-weather, colour, mood, and a RARE section for the terms the model
  over-applies). `PROMPT_SECTIONS` must partition this list exactly.

**The merge rule is that derived wins.** It strips *every* capture-fact term
the model emitted — including when nothing was derived, because an absent
capture fact is a positive statement ("the EXIF says this is not a long
exposure") — then adds the derived ones, passes perceived terms through, and
returns a sorted, deduped list. **Missing EXIF derives nothing; it never
guesses.**

**Frozen terms make the input's provenance part of the API**, because the two
paths need opposite behaviour. Two named entry points, over one primitive:

- **`merge_vlm_answer(perceived, row, existing=...)`** — the WRITE path
  (`worker_api.submit_results`, both `index.py` writers). Fresh model output:
  frozen terms the model volunteered are **dropped**, and any frozen term
  already on the photo is **carried across** from `existing` (the current
  `visual_tags`). So a re-tag replaces the perceived half — that is the point —
  without deleting a `sharp`/`blurry` nothing has re-decided.
- **`merge_stored_tags(stored, row)`** — the BACKFILL path. An array already in
  the column: frozen terms **pass through** untouched.

`merge_tags(..., source="vlm"|"stored")` and `merge_for_row(...)` are the
primitive underneath; `source` is validated, so a new call site cannot silently
pick the wrong semantics. Getting this backwards in either direction is a
data-loss bug, which is why both directions are tested.

Storage is unchanged: one `photos.visual_tags` JSON column, so the search blob,
the `visual_tag=` filter, `list_vocab` and the Ask tools all keep working, and
`list_vocab` lists the derived terms because it aggregates the column.

**Thresholds, and the numbers behind them** (measured on the read-only July
library copy, 151,291 tagged rows):

| derived tag | rule | evidence |
|---|---|---|
| `long-exposure` | shutter ≥ **1/4 s** | VLM agreement climbs monotonically as the bar tightens — ≥1/15 s: 6.9% of those photos were VLM-tagged, ≥1/8 s: 19.0%, ≥1/4 s: 25.1%. 1/4 s is 1.0% of the library (tripod/streaking territory); 1/15 s would be 4.3% and sweep in ordinary handheld indoor frames |
| `low-light` | **EV100 < 5.0**, where EV100 = log2(N²/t) − log2(ISO/100) | cross-checked against clock time: fires on **44.9%** of photos shot 21:00–06:00 vs **3.6%** of 09:00–16:00 (12.5:1). EV100 < 6 halves that ratio to 6.6:1 for 11 more points of night recall |
| `panoramic` | long/short edge ≥ **2.0** | 283 photos library-wide. The VLM claimed 4,290, of which **2.3%** had AR ≥ 2.0 |

#### `sharp` / `blurry` are FROZEN — and why that reversed a shipped decision

They were briefly DERIVED from `aes_sharpness` (≥ 9 → `sharp`, ≤ 2 →
`blurry`), on the cleanest-looking separation in the July validation copy:
**47.0%** of photos scoring ≤ 2 carried the VLM's own `blurry` tag against
**0.00%** of every bucket at 3 and above. **That was agreement between two
models, which is not ground truth** — and the July copy had only **2.4%**
coverage of `aes_sharpness`, so the rule barely fired there and the weakness
did not show.

On the live DB coverage is **98.9%** (158,111 rows) and the rule bites hard:
`blurry` **1,284 → 10,103 (+8,819)**, `sharp` **3,410 → 15,076 (+11,666)**.
Four photos scoring ≤ 2 were then inspected by hand: two were genuinely blurry
(defocused / motion-blurred indoor sports) and **two were false positives** — a
tack-sharp phone photo of construction formwork with the wood grain clearly
resolved, and a dark, noisy GoPro night street scene that is not blurry at all.
`aes_sharpness` evidently conflates sharpness with general technical quality;
noise and low light drag it down. Four photos is a tiny sample, but a plausible
~50% false-positive rate is disqualifying for a backfill that would stamp
`blurry` on ten thousand photos.

So the terms are **frozen in all three directions**: `derive_tags` never emits
them, the prompt never offers them (and the parser drops them if a model
volunteers one), and **nothing deletes the values already stored**. Deleting
would be a second unvalidated decision on top of the first; they stay exactly
as they are until a **hand-labelled eval** decides. The backfill's before/after
table therefore shows both at **delta 0**.

**A native-resolution Laplacian was tried as the source, and it FAILED**
(2026-09-26, `docs/plans/sharpness-measurement.md`). The eval used 208
owner-labelled photos, 80 of them blurry, judged at 100%. It tested
face-, subject- and frame-tiled Laplacian and Tenengrad, noise-corrected, at
several scales. No measured feature reached P ≥ 0.8 at R ≥ 0.5. Gradient energy
cannot tell a soft *scene* from soft *focus*.

The best idea looked good only in-sample: `aes_sharpness ≤ 2 AND
frame.ten_max ≤ 4000` scored 0.86, but fell to **0.55** on a 40-photo random
holdout. The labels and the measurement cache are kept in `evals/visual-tags/`,
so a new signal can be scored with `evals/sharpness_eval.py` without
relabelling anything. Face-crop Laplacian is still the right *within-shoot
ranking* signal (`rank_shoot`): comparing frames of one burst cancels the
scene. It is not a library-wide tag.

**Deliberately NOT derived**, because the EXIF cannot decide them: `macro`
(needs subject distance / magnification, which is not stored), `aerial` (no
altitude column), `wide-angle` (focal length is a mix of full-frame and phone
actual mm with no 35 mm-equivalent column to normalise against), and any
shallow-depth-of-field tag — the vocabulary has no `bokeh`-type term at all, so
there was nothing to derive. These stay PERCEIVED.

**The merge is applied SERVER-SIDE**, where the EXIF lives, through one shared
call — the worker only ever sees pixels. Every write path routes through it:
`worker_api.submit_results` (the fleet, **and** the M28 `rerun.run_pass_sync`
path, which submits through that same endpoint), plus both in-process
`index.py` writers (directory mode and collection mode). An emptied merge still
persists `'[]'` so `_SubmitOutcome` marks the photo done in one pass. A row
whose only survivors are derived logs **no `generations` entry** — derived tags
are not LLM artifacts.

`tools.set_photo_tags` / `POST /api/photos/bulk-set-tags` (the M26b agent/user
write) deliberately **bypass** the merge — a human asking for a tag should get
it. **But the next `derive-visual-tags --apply` will strip a hand-set
capture-fact tag**, because the backfill treats EXIF as authoritative over
every row. Hand-setting `long-exposure` on a 1/800 s frame does not survive a
backfill; hand-setting `peaceful` does.

**Two kinds of empty, and conflating them is what made the original bug
permanent.** `tag_visual_photo` returns `[]` when the model *answered* "no
tags" (that is a result: the server writes `'[]'` and the photo is done), and
`None` when there is **no usable answer** — no response, unparseable prose,
regurgitation, or a guard that removed everything. On `None` the server leaves
`visual_tags` **NULL**, so the photo stays claimable, but still marks it
processed so a *repeatable* failure is retired by `MAX_PROCESS_ATTEMPTS`
instead of being re-claimed forever (the CLIP-style infinite re-claim). This is
the same shape as the aesthetics pass's empty-scores row.
`CategoryVisualResult.visual_tags` is therefore `Optional`; older workers
always send a list, so `None` never arrives from one.

**`_parse_visual_response` is deliberately tolerant.** It splits on commas,
newlines and semicolons; strips bullets (`-`, `*`, `•`, `1.`, `1)`), a leading
`Label:` prefix, code fences, quotes, brackets and emphasis; and reads a JSON
array. The first version split on commas alone, so `RIGHT: sunny` parsed to
`[]`, `Tags: sunny, peaceful` lost `sunny`, and a bullet list parsed to
nothing — each of which then looked like "no visual qualities" and retired the
photo. **Don't simplify it back to a comma split**: the prompt asks for one
shape and small vision models supply six, and the cost of not reading one is a
silently mistagged photo that never comes back. For the same reason every
answer the prompt *demonstrates* is a bare tag line — a labelled `WRONG:` /
`RIGHT:` example teaches the model to emit a label. Pinned by
`tests/test_visual_parse_tolerance.py`.

A transient `database is locked` on the server-side **EXIF read** is re-raised
so `_record_write_failure` defers the photo. Swallowing it would write the row
strip-only — derived tags silently absent — spend the attempt and mark the
photo processed, leaving it permanently under-derived with no trace. Same rule
as `50003c2`, applied to reads.

#### The prompt: no axes to fill, no worked examples

**"Pick at most ONE tag from each group" was read as "pick one FROM EACH
group".** The first rewrite grouped the perceived terms into six AXES and said
exactly that. Live, the cap held (3–5 tags) and 28 of 29 early tag sets were
distinct — but nobody got fewer than 3 tags, and `close-up` / `symmetrical` /
`wide-angle` turned up on photo after photo: a person walking down stairs got
`close-up, sunny, symmetrical`; an artichoke close-up got `close-up …
wide-angle`. The model was filling every section because the prompt implied
there was one to fill.

A/B on **11 photos the controller labelled by hand** (qwen2.5-vl-7b-instruct
via the OpenAI route, 1024-px JPEG q85, temperature 0, the same parser;
debatable tags scored as neither):

| variant | right | wrong | tags | note |
|---|---|---|---|---|
| **A** axes prompt | 16 | **5** | 36 (avg 3.3) | called two defocused **indoor gym** photos `sunny`; `close-up, symmetrical` on the stairs photo |
| **B** RARE section, no examples | 14 | **0** | 20 (avg 1.8) | answered `none` for both gym shots and a construction snapshot — correct: nothing notable about their look |
| **C** = B + two examples | 14 | 1 | 18 | the example containing `close-up` leaked `close-up` onto a baseball photo |

**B shipped.** What makes it work is the **RARE section**: instead of a 3–6
word gloss, each over-applied term states the exact situation it needs *and
what it is not* — "people seen full-length are never close-up", "candid photos
of people are never symmetrical". Plus a standing instruction that "most photos
have NO viewpoint tag and NO composition tag at all", and permission to answer
`none`.

**C is why there are no worked examples.** A small model parrots the shape and
the content it is shown; the example's own `close-up` came back on an unrelated
photo. That is the same example-leak the earlier review flagged when a
`WRONG:` / `RIGHT:` label taught the model to emit a label.

**Caveats, honestly:** 11 photos, labelled by the controller, and B was written
after seeing A's failures on some of them — so it is partly overfitted to them.
`centered` is still somewhat eager under B (3 of 11). The owner's labelled eval
is what settles this.

**The labelled eval that settles it** — `evals/visual_tags_eval.py` +
`/eval/visual-tags` (linked from `/admin/maintenance`). `sample` draws a
stratified, seeded 60 (sports folders, each RARE term, `peaceful`/`moody`,
EXIF low-light, indoor, landscape, travel, stored-`[]`, random) from a DB
opened `mode=ro`; the page collects **three-state** labels (yes / debatable /
no — debatable scores as neither, because forcing a verdict on `moody`
manufactures disagreement that is not the model's fault); `run --variant`
calls the **production** `tag_visual_photo` (real encode, parser, guard, retry)
with pixels from `/api/photos/{id}/full`, the fleet's own endpoint;
`report` gives per-tag precision/recall, per stratum, side by side, plus a
`stored` pseudo-variant scoring what is in the column today with no model run.
`--prompt-file` swaps only the prompt; `--model` pins
`PHOTOSEARCH_LLM_VISUAL_MODEL` for the process, because on the LM Studio route
the call-site name is ignored and the bake-off would otherwise score the
configured model under another's name. `--probe` answers "does this model see
images" with one synthetic frame — 2026-09-20: `qwen2.5-vl-7b-instruct`,
`gemma-4-e2b`, `qwen3.5-9b` and `gemma-4-26b-a4b` **all do**.

**Candidate tags** (`visual_tag_eval.CANDIDATE_TAGS`) are labelled by the owner
but are NOT in the shipped vocabulary or prompt — collecting the label now is
nearly free, re-opening 60 photos later is not. A candidate is scored **only
for a variant that offered it** (`run --extra-vocab action --prompt-file …`,
which also lets the parser accept it); against any other variant it is
invisible in both directions, so production is never charged a miss for a tag
it was never asked about. First candidate: **`action`** — the owner read a
dribble and a keeper's save as `dramatic`. Fair reading, wrong tag: `dramatic`
is the LOOK (contrast / visual tension, contradicts `peaceful`) and on a sports
shoot would saturate like `sunny` (98.7%); action is the MOMENT, nothing in
categories/keywords carries it (1,251 of 1,373 frames say "soccer", none say
what is happening), and it is the signal `rank_shoot` says it lacks.
`eval_api.LABELLER_NOTES` likewise holds human-only definitions
(`harsh-light`, `soft-light`, `overexposed`, `backlit` — judged on the
CLEAREST SUBJECT, not the largest figure); a test asserts none of that text
reaches the production prompt. Promote either into `visual_tags_derive` only
after the A/B.

Also candidates since 2026-09-26: **`blue-sky`** and **`cloudy`** — not
exclusive with each other or with `sunny`/`overcast` (no pair). qwen already
volunteered `blue-sky, fluffy-clouds` unprompted, which the parser dropped.
`foggy`/`hazy` got labeller definitions (fog = the air itself visible, near
things fade; haze = only the distance washes out). The variant that offers all
of it is `evals/prompts/visual_defs_sky.txt` (`--extra-vocab
action,blue-sky,cloudy`). The parser maps **`misty` → `foggy`**
(`describe._VISUAL_SYNONYMS`, true synonyms only). The Unsplash sample predates
the two new mappings, so it has no positives for them until re-drawn with
`sample --force` — which orphans the cached runs.

**The labeller is measured too.** `/eval/visual-tags?set=recheck` re-serves a
seeded 15-photo subset (`recheck.json`) with the first answer withheld; labels
go to `labels-recheck.json`; `visual_tags_eval.py agreement` prints per-tag
self-agreement and Cohen's kappa. A tag the owner disagrees with themself on
(kappa < ~0.4) is not tightly enough defined to score a model against — fix
the definition (`eval_api.LABELLER_NOTES`), not the prompt.

**Sharpness labels ride on the same page** (`docs/plans/sharpness-measurement.md`
step 1). `sharp`/`blurry` are `visual_tag_eval.MEASURED_TAGS`: labellable on
every set, outside `CANDIDATE_TAGS`, and never scored against a VLM variant in
either direction. They are judged in `PS.Loupe` — the original from `/full` at
one image pixel per device pixel — never from `/preview`. A separate
blurry-weighted sample lives in `<eval_dir>/sharpness/` (`sample-sharpness`,
`?set=sharpness`, `?set=sharpness-recheck`, `agreement --set all`) so drawing
it never orphans the visual sample. Labels carry `measured: true` once the
chips were shown; the first visual labels do not, and `measured_labels()`
skips them — a missing `blurry` there means "never asked".

**Second set — Unsplash recall** (`evals/visual_tags_unsplash.py`): the
Unsplash Research Dataset Lite's PHOTOGRAPHER-supplied keywords as positives
(up to 40 photos per tag from the 25k local thumbs in
`~/unsplash-quality-eval/thumbs/`, 400 px — not fleet resolution). Keywords are
positive-only, so it reports **recall and contradictions only, never
precision**; `sheet --tag X` writes a contact sheet of what photographers mean
by X. `centered` / `harsh-light` / `overexposed` have ≤2 photos there and
cannot be measured on it.

**Model bake-off 2026-09-21** (production prompt, 60 owner labels):
qwen2.5-vl-7b 0.51/0.40 precision/recall, **gemma-4-26b-a4b 0.70/0.41** (fp
65 → 30, `soft-light` recall 0.18 → 0.57, `centered`/`joyful` fp gone; only
`close-up` regresses; needs `PHOTOSEARCH_LLM_REASONING_EFFORT=none` and is 5×
slower at 3.5 s/photo), qwen3.5-9b 0.55/0.44, gemma-4-e2b unusable.

Labels and run caches are **files** in `evals/visual-tags/` (git-ignored,
`PHOTOSEARCH_VISUAL_EVAL_DIR`), not DB rows: `sync-replica.sh` swaps the
replica DB wholesale and hand labels are the one artifact that cannot be
regenerated. The API is local-only — it must never proxy to the NAS. Shared
contract: `photosearch/visual_tag_eval.py`. A photo that is not `done` has no
opinion and is never scored; ratios under 3 observations print `n<3`. The
page hides the model's stored tags by default (and skips `PS.PhotoModal`,
whose sidebar shows them) so the labeller is not anchored.

**Don't reintroduce axes-to-fill or worked examples without re-running the
A/B.** Both looked like obvious improvements and both cost accuracy. The guard
never enforced one-per-axis and must not start: `backlit, silhouette` describe
one photo honestly.

**Temperature stays at 0**: the collapse came from the prompt, not from greedy
decoding, so the prompt is the fix and the pass stays reproducible. The hard
cap of 5 stays in the guard.

**The guard** (`describe._visual_answer_problem`) was at `>= 12` tags,
inherited from the 78-term `tags` vocabulary — and the real failure's maximum
was **11**, median 5, so **it never fired**. It is now `_VISUAL_MAX_TAGS + 1`,
derived from the cap the prompt states so the two cannot drift. A second check
covers `visual_tags_derive.CONTRADICTORY_PAIRS`; on a contradiction it retries
once, then **drops both members of each bad pair** rather than rejecting the
response — when a model calls one photo peaceful *and* dramatic, nothing says
which half it meant, but the untouched tags are still good. Regurgitation
behaviour is unchanged: a retry that still over-selects drops the whole
response.

**Because the guard deletes BOTH members, a pair that is merely unusual
destroys correct tags.** The bar is therefore *could a competent
photographer's single frame honestly be both?* — and it rules out more than it
first looks like. The seven that survive contradict on the *same property*:
joyful×melancholy, overcast×sunny, harsh-light×soft-light, muted×vibrant,
black-and-white×colorful, black-and-white×vibrant, aerial×macro. Rejected,
with the reason:
close-up×wide-angle (an environmental portrait is ordinary), foggy×sunny (sun
through fog is a classic shot), joyful×moody (subject's emotion vs how the
frame is lit), colorful×monochromatic (a blazing orange sunset reads as both),
aerial×close-up (a tight drone crop is both), macro×wide-angle (close-focus
wide-angle is a real technique), and peaceful×moody (the library's biggest
co-occurrence at 19,056 — but a still, misty lake is honestly both),
and — removed by the owner 2026-09-26 off the Unsplash sheets —
dramatic×peaceful (a still sunset over the sea is both) and colorful×muted
(muted is the *light*, colorful the *number of hues*; the pair turned gemma's
`muted, colorful` on a rose photo into no answer at all, because deleting both
members left nothing). There is
no sharp×blurry entry: both are derived now and cannot co-occur.

**Backfill** (no VLM, no `generations` rows):

```bash
photosearch derive-visual-tags            # dry run: before/after frequency table
photosearch derive-visual-tags --apply
```

The **dry run opens the DB read-only** (`mode=ro`), so it is safe against a
live or write-protected copy — `PhotoDB` migrates on open and would write.
`--apply` reads once, then writes in 2,000-row chunks with a commit and a
progress line between, skips rows whose array would not change, and guards
every UPDATE on the old value (`WHERE id=? AND visual_tags IS ?`) so a
concurrent fleet write is not clobbered. Idempotent. **It never touches a photo
whose `visual_tags` is NULL** — NULL means "not yet tagged" and is what drives
the category-visual worker queue.

Dry run over the July copy: **106,120 of 151,291 rows change** (70.1%) —
45,442 change tags, the other 60,678 are re-ordering into the new
deterministic sort. `low-light` 33,951 → 15,173, `long-exposure` 7,885 →
1,551, `panoramic` 4,290 → 283, `motion-blur` 2,378 → 0, and — the point of
freezing them — `sharp` 3,247 → 3,247 and `blurry` 1,001 → 1,001, **delta 0**.
Only those four terms ever move.

The 58,158 ordering-only rewrites are **intentional and a one-shot cost**:
`merge_tags` sorts, which is what makes the stored JSON a deterministic
function of (answer, row) — and that is what lets the backfill skip unchanged
rows and lets the `WHERE visual_tags IS ?` guard mean anything. Don't "optimise"
them away by preserving the model's emission order.

Expect the `/status` **"visual tagged"** card to **drop** after the backfill:
it counts `visual_tags IS NOT NULL AND != '[]'`, and a photo whose only tags
were capture facts the EXIF refutes now legitimately holds `[]`. That is the
fix working, not a regression.

**A blanket VLM re-run is NOT recommended.** At the measured ~1 s/photo it is
~44 h of GPU time for the whole library, and the backfill already fixes the
half that was actually wrong without touching the model. Re-run
`category-visual` only on a **targeted cohort** — the folders where the
perceived tags are visibly collapsed (sports shoots: `centered` 88–94%) — via
`clear-pass` + the fleet, or M28's per-photo re-run.

**Don't simplify any of this back.** The prompt cannot fix the capture facts —
the information is not in the image. The flat list, the missing cap and the
12-tag guard each looked harmless and each was load-bearing in the failure.

<!-- moved verbatim from CLAUDE.md (line 1112) on 2026-10-19 -->
### A new description re-queues what was derived from it

category-content and keywords are extracted FROM `photos.description`, and
verify checks it. Until 2026-10-04 the fleet ran **verify last**, so each verify
rewrite (~16% of a batch) left categories/keywords extracted from text verify
had just judged hallucinated — **13,637 photos** library-wide. Three fixes:

- **Pass order** (`batch_state.WORKER_PASSES` = `rerun.ALL_PASSES` =
  `worker_api._ALL_PASSES`, mirrored in `batch-flow.js`): `clip, faces,
  quality, video-probe, video-scan, aesthetics, video-subjects, describe,
  video-describe, verify, category-content, keywords, category-visual`
  (video-describe must drain before the text passes that read its output).
  `/workers/start` sorts any
  hand-picked `-p` into it. Also one load per model: qwen3.5 → +gemma → gemma →
  minicpm. `DEPENDS_ON` is unchanged (text passes still wait on `describe`, not
  `verify`), so a blocked verify never starves them.
- **The server re-queues on every description write**
  (`PhotoDB.invalidate_description_dependents`): a describe result NULLs
  categories/keywords/verification + their ledger rows; a verify rewrite NULLs
  categories/keywords only (its own verdict is the result). Order alone is not
  enough — workers overlap at a pass boundary, and M28's re-run checks
  `describe` or `verify` → auto-selects the text passes too. **Server-side:
  deploy the NAS.**
- **Verify no longer re-tags.** The rewrite path ran the visual tagger into the
  legacy `tags` column (NULL since v23, read by nothing current), which also
  pulled minicpm into VRAM beside gemma + qwen3.5. `visual_tags` come from the
  pixels, so a rewrite doesn't stale them. An older worker's `tags` field is
  ignored.

**Historical cleanup — `photosearch stale-description-passes`**
(`photosearch/stale_descriptions.py`). Dry run (read-only) reports, per pass,
photos whose output predates the current description, from `generations`
timestamps. `--folder` / `--collection` scope it; `--save-collection` queues
the set as a collection; `--requeue` clears exactly the stale pass per photo
(chunked commits). Run writes **on the NAS**. Traps it handles: `created_at`
comes as both `…T…` and `… …` (normalized); `verified_at` is worker-local while
generations are UTC — read zone-less stamps as Pacific (exact for every
fleet stamp since May, measured off the ~14k rewrite anchors; the lone UTC era,
the 2026-04-12 NAS run, gets the later reading so it can never be flagged
wrongly). **Not a margin**: verify runs within the hour of describe, so any
slack wide enough to absorb the offset also hides real re-describes. New stamps
are explicit UTC (`worker._utc_stamp`, `…Z`);
output with no generation row predates logging and is counted, never flagged
(1,128 categories).

**It also reads the content.** Keywords are extracted from the description, so
their words should be in it (`keyword_match_ratio`, possessives/plurals/stopwords
folded). Timestamp-current photos: 92% match >= 90%, 99.8% >= 50%; below 50%
(`KEYWORD_MATCH_MIN`) is a real defect, not paraphrase — on 2026-10-04, 286
photos the timestamps missed: whole keyword list stored as ONE string, a
refusal stored as keywords ("i couldn't find any text…"), etc. A mismatch
re-queues keywords AND categories (same source text). Categories get no content
check of their own — they're a fixed vocabulary, not the description's words.
A mismatch caused by a **cut-off description** is reported but not re-queued
(re-extracting from the same truncated text repeats it); the report also counts
all mid-sentence descriptions (4,350) — those need a re-describe.

<!-- moved verbatim from CLAUDE.md (line 1169) on 2026-10-19 -->
### Bad LLM output is a failure, not a result

Every LLM pass used to store whatever came back. Neither backend's stop reason
was read, so a generation cut off at the token limit was stored as a finished
answer: **4,350** live descriptions end mid-sentence (llava at `num_predict`
150, qwen3.5 with reasoning on eating `max_tokens`). Keywords held refusals and
unsplit lists, and llama3.2's category answers recited the vocabulary
(53-232 tags). Each looked like a success until a later audit.

- **Both chat helpers read the stop reason** (`finish_reason` / `done_reason`
  == `"length"`), retry **once at double the budget**, then raise
  `describe.TruncatedOutput`. An **empty** answer cut off mid-reasoning is a
  truncation too.
- **Per-role `max_tokens`** on the LM Studio route (`_OPENAI_MAX_TOKENS`:
  describe/verify 512, text 256, visual 128, others 768). It used to be a flat
  768 for everything.
- **Answer guards** raise `describe.UnusableAnswer` after one retry
  (temperature 0.4):
  - describe: cut off, repetition loop, or ends mid-sentence
    (`ends_mid_sentence` = `stale_descriptions.is_truncated`). A degenerate
    answer is no longer returned as a last resort.
  - category-content: empty, nothing in the vocabulary, a **vocabulary
    recitation** (`category_grounding.is_recitation`, below), more than
    `category_grounding.MAX_PLAUSIBLE` (120) categories, or an answer that
    grounding emptied.
  - keywords: refusal, a whole list as one keyword (> 5 words), more than 30
    keywords, or under `KEYWORD_MATCH_MIN` of them using the description's
    words.
  An explicit `none` is still a real empty result.
- **The worker turns `UnusableAnswer` into a failure row**: one attempt spent
  and an `index_errors` entry, nothing written. Describe and category-visual
  opt in with `raise_unusable=True`; other callers still get `None`. Text
  passes used to **defer** these, retrying forever at no cost, or store them.
  A transport stall still defers.
- **A timeout on the double-budget retry is a truncation, not a stall**
  (`_expanded_retry`). The first call already proved the answer overruns. On
  2026-10-06, gemma looped `railing, railing, …` at temperature 0 on one photo
  (IMAG2074). The 512-token retry overran the 10 s text cap, read as a stall,
  and the photo was re-claimed ~90 times in 1.5 h with no attempt spent. Now
  it spends attempts, and the 0.4 retry escaped the loop on attempt 3.
  Connection errors on that retry are still transport.
- **A cut-off verify verdict propagates** instead of returning `[]` (which
  reads as ALL CORRECT and stamps the photo verified).
- The eval `Recorder` records a truncation as `unusable`: a model outcome to
  cache, not a `TransportError`.

**There is no 60-category cap any more.** It was set just above gemma's
then-maximum of 52 and just under llama3.2's recitation floor of 53, and
gemma-4-12b (the fleet text model) averages **25.7** categories, accurate but
exhaustive. **Owner decision 2026-10-07: the long lists and some
near-duplicates (grass / grassy field) are fine.** On 2026-10-18 the cap
retired DSC05775 (photo 251357, a daylight outdoor soccer frame) after four
answers of 84, 83, 62 and 65 categories that were NOT recitations: about two
thirds right, the rest padding and contradictions (basketball court,
artificial lighting, indoor, kitchen). The count was the wrong test; see
"category-content grounding" below for what replaced it. The 120 backstop is
judged on the REPAIRED list and exists for the Ollama route.

Worker-side guards need a fleet restart; the server-side grounding repair
needs a NAS redeploy. Tests: `tests/test_text_pass_safeguards.py`.

<!-- moved verbatim from CLAUDE.md (line 1230) on 2026-10-19 -->
### category-content grounding: the description decides

`photosearch/category_grounding.py` (`docs/plans/category-grounding.md`,
built 2026-10-18 by plan-debate; **not deployed until the owner's holdout
labels pass**). The category-content pass used to store any in-vocabulary
term the model returned. Now one pure function, `ground(description,
categories) -> Grounding(kept, dropped, recitation)`, runs on the **worker**
(`describe.extract_categories_from_description`, so the fleet, both
`index.py` writers and M28 all get it) and again on the **server**
(`worker_api.submit_results`, against the CURRENT stored description, the
same precedent as `_merge_visual_tags`). Idempotent; output is a subset of
input in input order; a NULL description changes nothing.

**Four contradiction GROUPS, and the description picks the side.** sport
(soccer / basketball / baseball / volleyball terms), setting (indoor terms vs
`outdoor`), light (`artificial lighting` vs natural-light terms), weather
(clear-sky terms vs cloudy terms). Per group: a present side is dropped when
a rival side is cued in the description and its own is not, **including when
only one side was emitted** (`artificial lighting` alone on "outdoors under
trees"); both are dropped only when both were emitted and nothing is cued;
both are kept when both are cued. Drop-both as the default was rejected on
the data: it would delete 329 soccer tags from 124 soccer photos. Negated and
through-the-window spans are blanked before any cue runs (`SCRUB_PATTERNS`).
Owner decisions 2026-10-18: `basketball court` / `basketball hoop` are KEPT
when an indoor setting is cued (gym soccer: they are the venue, 18 of 67
drops); "partly cloudy" cues the cloudy side, so it contradicts `clear sky`
but not `cloudy sky`; a bare "cloud" is neutral. Traps pinned by tests:
"light blue", "daylight", "sunlight" are not artificial-light cues; "baseball
cap" is not a baseball cue; `collage` skips the setting group; `car interior`
/ `interior space` are not indoor terms (photos taken from inside a car).

**No retry when the description decided a drop.** The 0.4 retry on 251357 was
worse (84 tags). A repairable answer costs one model call. Retry once only
when both sides were emitted and neither cued, or on a recitation verdict.

**Unsupported padding is NOT dropped.** Only 41% of gemma tags appear
literally in the description ("vehicle" for cars, "child" for boys), so a
word match would delete correct tags. Required-cue FAMILIES (basketball needs
a hoop, kitchen a stove, water a water word, city, building, celebration,
coach, hat, wall, ...) are a diagnostic in the eval only; each is promoted
one at a time after >= 90% correct drops on a fresh owner-labelled holdout.

**The recitation verdict replaces the count cap:** vocabulary-order share
>= 0.9 AND >= 10 unsupported CONCRETE nouns AND they are >= 50% of the
answer's concrete nouns. Objects are the ~126 concrete nouns in
`_CONCRETE` only; the family terms are exactly gemma's padding vocabulary,
and counting them flagged 23 honest gemma rows. Measured on the replica:
141/141 llama recitations over 53 tags caught, 121 of the 702 llama lists at
20 to 52 tags (partial recitations the cap never saw: "beehive-like stone
structure, no people" stored `bleacher, bridge, chain link fence`), 1 gemma
row flagged out of 7,769. Order alone is NOT a signal: 72% of gemma lists
are >= 0.9 in vocabulary order.

**Measured on 7,769 current gemma rows:** 1,031 rows hit (13%), 1,228 of
163,106 tags dropped (0.75%): light 851 (`artificial lighting` 1,186 → 387),
sport 171, setting 129, weather 77. 251357 at temperature 0: 68 → 64, not a
recitation, so it would now be STORED. Eval: `evals/category_grounding_eval.py`
(read-only, no model calls, ~50 s) writes `evals/category-grounding/summary.md`;
the seed-7 samples were used to tune the cue lists, so the owner labels the
seed-11 holdout (`samples-holdout-seed11.json`, 25 drops per group) on
`/eval/category-grounding`; a group ships only at >= 90% correct drops.

**Backfill:** `photosearch ground-categories [--model gemma|llama|all]
[--folder F] [--apply] [--requeue-recitations]`. Dry run by default, opened
`mode=ro`; 2,000-row chunks with a commit between; `UPDATE ... WHERE id=? AND
categories IS ?`; unchanged rows skipped; never touches NULL (it drives the
worker queue); never writes `'[]'` over a non-empty list (33 llama rows would
be emptied: reported, not written); only rows whose stored set is a subset of
their latest `generations` row are eligible (hand-set rows skipped); no
`generations` row is written. Recitation rows (255 llama, 1 gemma) are only
re-queued behind `--requeue-recitations`. Default scope is gemma; 154k llama
rows are the bulk of the library and were not the target.

**Collapse across a shoot is flagged, not gated** (`photosearch/visual_collapse.py`).
A per-photo guard cannot see category-visual stamping one set on a whole shoot
(2026-10-03: 1,211 of 1,260 photos `colorful, sunny, vibrant`). The flag fires
when a folder has at least 50 tagged photos and its commonest set covers at
least 50% (the p90 of 540 folders; the median is 24%). Stored `'[]'` is not
counted. 59 folders library-wide trip it.
- `/batches`: the category-visual node shows `⚠ N% share one tag set`, with the
  set in the tooltip. Every step row now carries `warning` / `warning_detail`
  (null except here), and category-visual also carries `collapse` stats. The
  step's state is unchanged, because the pass did finish. Shows on the replica
  only once the NAS runs this code (`/api/batches` proxies).
- `photosearch visual-tag-collapse [--folder 2026] [--save-collection
  [--top-set-only]]` lists collapsed folders worst-first, read-only. The
  collection is a targeted re-run cohort; save it on the NAS.

**Re-tagging a flagged folder with minicpm does not fix it** (probe on
2026-10-07, 24 photos across three flagged folders, read against the photos).
minicpm-v-4_5, the fleet visual model, collapses to its own default set,
`sunny, colorful, vibrant`. It put that on indoor gym futsal and on an indoor
classroom party.
- **qwen2.5-vl** half-fixes it: it answered `[]` on most of the gym shots but
  still called classroom shots `sunny`.
- **gemma-4-26b-a4b** was right on all three folders: `[]` on the gym,
  `golden-hour, soft-light` at dusk. It is ~5.7 s/photo vs ~1 s for minicpm.
- **2026-10-03 itself is NOT a failure.** All three models agree on
  `sunny, colorful(, vibrant)` for a bright match in neon kits, so it is a
  legitimately uniform shoot (the flag's known false-positive case).

**The prompt now rules the attractor set out (2026-10-09)**, so minicpm stays
the fleet's visual model. `sunny` and `overcast` say "never indoors", the LIGHT
header says indoor or lamp-lit photos get no weather tag, and `colorful`,
`vibrant` and `muted` got definitions. Measured on minicpm (variant D,
`evals/prompts/visual_attractor_d.txt`, eval run `attrD-minicpm`):
- **Collapse probe** (`2026-01-24` gym, `2025-10-31` classroom, `2026-10-03`
  control): gym went from `sunny, colorful, vibrant` on 12 of 12 frames to 0
  `sunny`. The classroom's remaining `sunny` frames are the outdoor costume
  parade. The control kept `sunny` on 8 of 8.
- **60 owner labels**: false positives 82 -> 58, precision 0.54 -> 0.63,
  recall unchanged at 0.56. `vibrant` fp 30 -> 10, `sunny` fp 13 -> 3 at
  recall 0.96, `muted` recall 0.23 -> 0.54.

Variants that lost, and why (don't reintroduce them without re-running both
the probe and the eval):
- **B**: a blanket "most photos have NO colour tag" rule. Precision 0.61 but
  recall 0.49, mostly `colorful` (0.95 -> 0.68).
- **C**: D plus the labeller definitions of `harsh-light` and `soft-light`.
  Recall 0.58, but `sunny` fp went back up to 11 and the gym got 3 `sunny`.
  The `harsh-light` text ends "...is just sunny", which seems to steer the
  model toward `sunny`. `soft-light` recall barely moved (0.14 -> 0.21):
  minicpm rarely uses it whatever the definition.

Photos tagged before this keep their old tags. Re-tag a flagged folder with
`clear-pass category-visual` and the fleet.

**Done 2026-10-07/08 with gemma-4-26b as a one-off** (owner decision; the fleet
default stays minicpm). NAS collection 58 held 16,276 photos in 58 folders,
excluding 2026-10-03. The run took ~23 h at ~4.5 s/photo with 3 workers.
Results:
- 0 failures, all 183 frozen tags kept.
- 4,726 photos now hold `[]`.
- Flagged folders went from 59 library-wide to 19.
- The rest are genuinely uniform shoots: an outdoor graduation in full sun,
  `vibrant` neon kits in the gym, EXIF-derived overnight `long-exposure`.
- gemma over-uses `soft-light`, e.g. on hard midday shadows.

Clearing a folder also used to delete its frozen `sharp`/`blurry`: the re-tag
carried them over from the column `clear-pass` had just nulled. They are now
stashed first (`db.stash_frozen_visual_tags`, on-demand table
`visual_frozen_carry`) and read back by the write when the column is NULL.

**Cut-off descriptions already stored** (4,333 on 2026-10-06; re-described
2026-10-06/07 as NAS collection 57: 0 cut off afterwards, 207 rewritten by
verify, categories now 17 per photo on average, 3 failure rows in total):
`stale-description-passes --save-truncated` saves them as a collection and
clears nothing. Then `clear-pass describe` on that collection and run the fleet
with `-p describe,verify,category-content,keywords`. Each new description
re-queues its derived passes server-side.

<!-- moved verbatim from CLAUDE.md (line 1381) on 2026-10-19 -->
### Provenance: log the model that RAN, not the one configured

`generations.model_used` said `llava` for **159,647 of 159,650**
`category-visual` rows, and `model_version` was NULL for every Jun–Sep row — so
nobody could tell which model tagged anything. The cause: `worker.py` stamped
the **nominal CLI default** it was launched with. On the LM Studio route that
name is *ignored* — `describe._resolve_openai_model(model, role)` picks the
model by ROLE — so the log recorded a model that never executed.

One shared helper pair in `describe.py`, used by the worker fleet and the M28
re-run path alike:

- **`effective_model(model, role)`** — on the OpenAI-compatible route, exactly
  what `_resolve_openai_model(model, role)` returns (the *same* function the
  request uses, not a reimplementation); on Ollama, the name as given.
- **`effective_model_version(model)`** — `"lmstudio"` on the OpenAI route
  (there is no Ollama to query, and asking blocks ~80 s retrying
  localhost:11434 once per pass); otherwise the short Ollama digest.

`describe.PASS_ROLES` is the pass→role map, pinned by a test against
`rerun._PASS_LLM` so the two cannot disagree. `worker._provenance_kwargs`
stamps every LLM pass through it — batch-level for `describe`/`verify`,
per-result row for the rest. **`verify` passes `role="describe"` explicitly**:
its logged artifact is the *regenerated description*, written by the regen
model, not by the verifier. `rerun._model_version` is now a thin alias for the
shared helper.

Don't reintroduce a per-branch `_model_version(...)` call in the worker loop —
a wrong provenance string still looks like a string, so drift here is invisible
until someone asks which model tagged a photo. A test asserts the loop contains
none.

<!-- moved verbatim from CLAUDE.md (line 1413) on 2026-10-19 -->
### Ollama stall on the text passes — tight per-call timeout

The text-only category passes (`category-content`, `keywords` on
`llama3.2:3b`) intermittently trigger a multi-minute Ollama stall under
sustained `NUM_PARALLEL=1` load: the runner serves nothing for minutes,
then bulk-releases. Investigated 2026-05-20 — it is **not** GPU/ROCm
(a CPU-only WSL2 Ollama froze identically), **not** poison input (stuck
photos have normal descriptions), and **not** runaway generation (the
stuck photos replay in ~1s, `eval_count` ~20). Root cause is still open
(some Ollama-internal stall common to 0.23.x-CPU and 0.24-GPU). The
describe/`category-visual` vision passes have not shown it.

Mitigation in `describe.py`: `_TEXT_OLLAMA_TIMEOUT_S = 10` is passed as
`timeout=` on the `extract_categories_from_description` /
`extract_keywords_from_description` calls, so a stall aborts in 10s
(vs the 120s `_DEFAULT_OLLAMA_TIMEOUT_S` the vision passes keep) and the
photo is retried later instead of blocking the single Ollama slot. This
caps blast radius, not the root cause. A restart-watchdog was rejected:
it crashes workers (the un-retry-wrapped `check_available`) and only
treats the symptom.

### Corrections found while condensing (2026-10-19)

Checked against the code while writing `.claude/rules/`; the text above is kept verbatim.

- The four category-visual groups are `CAPTURE_FACT_TAGS` (derived), `RETIRED_TAGS`, `FROZEN_TAGS` and `PERCEIVED_VOCABULARY` in `visual_tags_derive.py`; "DERIVED"/"PERCEIVED" above are prose names.
- The over-select guard constant is `_VISUAL_MAX_PLAUSIBLE_TAGS = _VISUAL_MAX_TAGS + 1` (`describe.py`).
- The percentile UPDATE guard is generic: `WHERE id=? AND {score_col} IS ?` (overall and subject scores).


### people-bind: names in descriptions (2026-10-30)

Design decisions and the debate behind them live in `docs/plans/description-names.md`
(two independent planners, two critique rounds; three decisions put to the owner). What
was built before any model ran, and what is deliberately unmeasured:

- Built: schema v40 `description_subjects`; `people_bind.py` (rule tier, prompt, parser,
  decision, validator, overlay); `description_render.py` (read-time names, IoU re-pair,
  sweep); worker pass `people-bind` (role `bind`, non-batch); surfaces behind
  `PHOTOSEARCH_DISPLAY_DESCRIPTIONS` (photo detail, Ask/MCP via `call_tool`, book captions,
  M28 mirror, modal); eval harness `evals/people_bind_eval.py` + scoring + gates.
- Not measured: VLM accuracy on small soccer faces, seconds per call, which model wins,
  today's real scope and strict-vs-manual mix. The 29.5k-photo figure is from a stale
  July replica snapshot. The backfill is gated on the holdout eval (0 wrong binds over
  >= 150 bound mentions) and has not been started.
- Subject-box guard on VLM slots (plan: "face centre inside a subject box for the first
  sentence's subject") was NOT implemented: it can only lower coverage, and the could_be
  count + shuffled second call are the safety. Revisit if the eval shows a wrong bind it
  would have caught.
- Query-plan finding: the correlated `EXISTS (trusted face)` claim predicate scanned the
  whole `photos` table; `p.id IN (SELECT photo_id FROM faces WHERE <trusted>)` drives from
  the faces list instead (`tests/test_people_bind_pass.py`).

# Eval harnesses for aesthetics, describe, category-content, keywords, verify

Status: **built** (2026-09-26, commits `6856e4a`..step 7). Not yet run against
the new models — see "Runbook" at the end. Picks up from
`docs/HANDOFF-2026-09-26-model-evals.md` (Phases 1–4). Built by plan-debate
(two planners, two critique rounds). The three points they still disagreed on
went to the owner; see "Decisions" at the end.

Goal: compare `google/gemma-4-12b-qat`, `minicpm-v-4_5` and `google/gemma-4-e4b`
against the production role models on every LLM pass, and pick the best model
**per role**. The owner accepts running **one model loaded at a time**, with the
passes run in sequence (decided 2026-09-26, see Decisions), so the handoff's
"all role models fit in 24 GB together" constraint no longer applies. Each
model only has to fit on its own.

## What the code shows (read this before building)

1. **The aesthetics bake-off may have scored the wrong model.**
   `describe._resolve_openai_model` (describe.py:544-557) falls back from the
   `aesthetics` role to `PHOTOSEARCH_LLM_VISUAL_MODEL` when
   `PHOTOSEARCH_LLM_AESTHETICS_MODEL` is unset.
   - The advice in `aesthetics_bakeoff.py:15-16` ("DO NOT set …AESTHETICS_MODEL")
     is wrong whenever the fleet env exports VISUAL.
   - `scores.json` records no effective model.
   - So the old **ρ 0.70 cannot be trusted as qwen2.5-vl's score**. Re-run it as a
     new baseline, not as a regression check.
2. **The ground truth is `evals/aesthetics-bakeoff/ranked.csv`**: 28 rows that
   match `sample/`. The directory is untracked but not in `.gitignore`, so check
   that no personal photos get committed.
3. **Production code turns transport failures into answers:**
   - `describe_photo` swallows every error and returns None (describe.py:732).
   - `llm_verify_description` returns `[]` on error (verify.py:326-328), which
     is indistinguishable from "ALL CORRECT".
   - `score_photo_aesthetics` returns None for both a transport error and a
     parse failure.
   - The text extractors return None on timeout.

   Every harness therefore wraps the chat call with a Recorder that turns "no
   call answered" into a `TransportError`, and never caches it. This is the
   same pattern as `production_tagger` in `evals/visual_tags_eval.py:332-367`.
4. **Retries and timeouts happen inside `_openai_chat_with_retry`**
   (describe.py:471-526: 3 attempts with a 5 s sleep between them).
   - A `None` from a text extractor means three timeouts.
   - A timeout that recovered on a later attempt can't be seen from outside the
     loop.
   - Truncation (`max_tokens=768`, `finish_reason`) is invisible too.
5. **On LM Studio the `llava` fallback is really a third retry.** The fallback
   call (describe.py:722-729) resolves by role, so it runs the same describe
   model. Count it by the recorded nominal name `llava`, not by comparing model
   names.
6. **Production verify has three stages, written inline in
   `worker._process_verify`** (worker.py:809-870): CLIP gate → LLM `WRONG:`
   pass → CLIP override. `verify.verify_photo` writes to the DB, so the eval
   must not call it.
7. **The content vocabulary has 360 terms**, so the owner can't label against a
   full checklist. The checklist is pooled instead (step 5).

## Steps

### 1. Shared infrastructure (unblocks everything)

**New: `photosearch/model_eval.py`.** The single storage contract for the new
passes, namespaced per pass.
- `eval_dir()` reads `PHOTOSEARCH_MODEL_EVAL_DIR`, default
  `./evals/model-evals/`. Add that path to `.gitignore`.
- Atomic write and read. Lift `visual_tag_eval._write_atomic` and `_read`
  rather than copying them.
- Run cache `<dir>/<pass>/runs/<variant>.json`. Its header records `variant`,
  `model`, `effective_model`, `role`, `prompt_sha`, `input_source`,
  `loaded_models_start/end`, `created`, and `items{}`.
- **Resume guard:** a mismatch in effective model, prompt or input refuses to
  run ("use a new `--variant` or `--force`").
- `pin_role_model(role, model)` generalises `_pin_model` to any role. It refuses
  when `--model` is given but `PHOTOSEARCH_TEXT_LLM_URL` is unset.
- **Originals cache:** `<dir>/originals/<pid>`, fetched once through the
  replica's `/api/photos/{pid}/full` and written atomically. A failed fetch
  raises and caches nothing. A `fetch-originals` subcommand pulls the whole
  sample while the NAS is up.
- `lmstudio_loaded()`: a best-effort `GET /api/v0/models`, recorded at the start
  and end of each run. Latency is labelled `solo`, `shared(<others>)` or
  `unknown`; `--solo` lets the operator assert it.
- `Recorder` context manager: wraps `describe._ollama_chat_with_retry` and
  collects the per-attempt records.
- Helpers: `fmt_ratio`/`MIN_N`, and median latency that skips the first call
  (which pays the JIT load).
- Failed-photo line: "N photo(s) failed and were NOT cached — re-run without
  --force", with a loud warning when 5 or more photos in a row fail (the
  LM Studio drop).

**Modified: `photosearch/describe.py`, with no behaviour change.**
- Add an optional `_ATTEMPT_HOOK`, default None.
- `_openai_chat_with_retry` calls it after every attempt with `role`, `model`,
  `attempt`, `outcome` (ok, timeout or error), `elapsed`, `completion_tokens`
  and `finish_reason`.
- Exceptions raised by the hook are swallowed.
- Fallback if the owner rejects this change: patch `describe._llm_trace` and
  count `None` returns.

**Modified: `evals/visual_tags_eval.py`.** `fetch_image` goes through the
originals cache, on by default.

**Tests: `tests/test_model_eval_store.py`.** Atomic write, resume-guard
mismatch, the originals cache fetching once and not caching failures, the
pinning refusal, transport errors vs "answered but useless", and the hook
firing on timeout (stubbed `urlopen`).

### 2. Aesthetics (independent; starts right after step 1)

`evals/aesthetics_bakeoff.py` keeps its CLI and gains:
- `--vlm M` pins `PHOTOSEARCH_LLM_AESTHETICS_MODEL`, with one model per
  process.
- A new `scores-v2.json` with per-scorer `effective_model`, `loaded_models` and
  per-item `{overall, attempts, first_parse_ok, latency_s}`. The legacy
  `scores.json` is read-only and reported as "effective model unknown".
- Report columns: n, **parse-failure rate** (first attempt and final), **spread**
  (std, IQR, distinct values, share within ±0.5 of the median), **s/photo**
  (median and p90, labelled solo or shared), and ρ against `ranked.csv` with a
  Fisher-z 95% CI. A ρ gap under 0.1 is a tie.
- `--ground-truth` defaults to `ranked.csv`.
- Optional `--selections-gt --db <replica>`: within-cluster pairwise agreement
  with `review_selections` (opened `mode=ro`, ~150 clusters, 0 owner minutes).
  Report it with the caveat that culling picks mix sharpness and moment.
- A larger owner ranking is deferred.

Tests: `tests/test_aesthetics_bakeoff.py` (spread metrics, parse-failure
classification, legacy and v2 caches side by side, pinning, a hand-computed
ρ with its CI).

### 3. Describe: sample, run, automatic report

**New: `evals/describe_eval.py`**, with subcommands
`sample | fetch-originals | run | report | pairs`.
- **`sample`**: the 60 visual-tags sample ids with their strata, plus
  `--text-n 10` photos in a `text` stratum. Candidates come from stored
  descriptions matching sign/menu/label/quoted text, shown on a contact sheet;
  `--exclude` lets the owner veto some.
- **`run --variant V --model M [--prompt-file]`**: pins the `describe` role,
  uses the Recorder, and calls the **production** `describe.describe_photo`.
  Each item caches `text`, `text_sha`, `latency_s`, the calls and attempts,
  `retried`, `fallback_called` and `truncated` (`finish_reason=="length"` or
  `completion_tokens>=768`). An answered call with a None result is cached as
  `text: null`.
- **`report`** covers every variant plus a `stored` pseudo-variant: the current
  `photos.description`, with its model taken from `generations`. Columns:
  - answered %
  - degenerate %, on the first attempt and on the final text
  - truncated %, retry %, fallback %
  - words (median and p90)
  - latency (solo or shared)
  - **CLIP-unsupported nouns**: `verify._extract_nouns` →
    `clip_score_description` against the stored embedding → `_flag_by_clip`.
    This check only runs on the desktop.
  - **text accuracy** on the `text` stratum, against the owner's typed truth
  - a **screen verdict**: a model with more than 10% unanswered, degenerate or
    truncated output gets no owner time.

Tests: `tests/test_describe_eval_harness.py`.

### 4. Labelling API and page

**New: `photosearch/model_eval_api.py`.**
- Router `/api/eval/models`, with the page served at `/eval/models`. Mount it
  next to `eval_router` in `web.py`.
- Local only: it never proxies to the NAS.
- `GET /original/{pid}` serves the originals cache, downscaled to 1920 px.

**New: `frontend/dist/eval_models.html`.** One page with the tabs **Claims |
Pairwise | Visible text | Categories | Keywords | Planted**.
- Built with React UMD and `React.createElement`; its own shared component
  goes in `shared.js` as `PS.Chip` (see below).
- Keyboard navigation copied from `eval_visual_tags.html`.
- Link it from `admin_maintenance.html`.

**Describe claims** are stored in `claims.json`, **keyed by `text_sha`**, not by
variant.
- Each photo is shown once, with all its de-duplicated descriptions stacked in
  shuffled order.
- Each description is split into claim chips by a deterministic
  `segment_claims()`. The owner toggles the wrong chips, or marks the
  description "all correct"; "other wrong (unsegmented)" is a free flag.
- The API never sends a variant or model name; a test asserts this.
- Identical texts share one label, and a re-run that changes the text orphans
  its old label.

**Pairwise preference** (`pairs.json`, `prefs.json`) covers only the finalists
against the baseline: about 40 photos, at most 2 pairs each, with a seeded
left/right order. The report shows the win rate (a tie counts as ½), n and a
sign-test p-value.

**Visible text** (`text_truth.json`): the owner types the visible text once per
photo in the `text` stratum.

**Move `Chip` into `shared.js` as `PS.Chip`** and make
`eval_visual_tags.html` use it. `scripts/check-frontend-refs.js` must pass.

**Report additions:** clean-description rate, wrong claims per description
(bootstrap CI), and breakdowns per stratum.

Tests: `tests/test_model_eval_api.py` (the page is blind, out-of-sample
requests return 404, out-of-range segments return 400, labels survive a
re-run).

### 5. category-content and keywords (text-only; runs in parallel with 3–4)

**New: `evals/text_passes_eval.py`**, with subcommands `freeze | run | report`.
- **`freeze`** snapshots the descriptions of the 60 sample photos into
  `<dir>/text/inputs.json` as `{source, source_effective_model, items{pid:
  {text, text_sha}}}`.
  - The default source is **`stored`** (the production descriptions), so text
    runs don't wait for describe.
  - It refuses to overwrite without `--force`, because labels are keyed to
    `(pid, text_sha)`.
- **`run --pass category-content|keywords --variant X --model M`** pins the
  `text` role, uses the Recorder, and calls the production
  `extract_categories_from_description` / `extract_keywords_from_description`.
- **Automatic metrics:**
  - timeouts per 100 attempts, measured against the 10 s
    `_TEXT_OLLAMA_TIMEOUT_S`
  - None rate
  - p50 and p95 latency
  - **unsupported-tag rate**, strict and lenient; the lenient version uses a
    small `SUPPORT_SYNONYMS` map, and keywords are scored strict only
  - tags per description
  - off-vocabulary drops
- **Owner labels** (`<dir>/text/labels.json`, keyed `pid:text_sha`):
  - **Categories:** a **pooled** checklist (every variant's output plus the
    stored `photos.categories`, shuffled and unattributed), with a search box
    for any of the 360 terms. The report gives precision and **pooled recall**.
    Wrong categories are split into ones the description supports (the describe
    model's fault) and ones it doesn't (the text model's fault). That split is
    the direct measure of the "soccer" failure.
  - **Keywords:** the owner marks the wrong chips; the report gives precision
    only.
- **Chain check:** once a describe winner exists, run
  `freeze --from <winner> --inputs chain` and then the text winner on that
  input. This check is automatic metrics only.

Tests: `tests/test_text_passes_eval.py` (freeze immutability, a support matcher
that includes the sailboat-vs-soccer case, timeout counting from hook records,
pooled precision and recall).

### 6. Verify (depends on the step 4 labels)

**Modified: `photosearch/verify.py`.** Add
`check_description(image_path, description, tags, clip_embedding,
verify_model, llm_all=False)`, which is `worker.py:809-870` moved verbatim.
**`worker._process_verify`** calls it; regeneration stays in the worker. Add a
parity test in `tests/test_verify.py`. This mode can be deferred if review
objects to touching the worker.

**New: `evals/verify_eval.py`**, with subcommands `plant | run | report`.
- **`plant`** builds `sets.json`:
  - **clean**: descriptions the owner marked all-correct.
  - **planted**: one templated false claim per clean description, of three
    types. **object** appends an absent noun from a fixed list, chosen **without
    looking at CLIP**. **colour** swaps a colour word. **count** changes a
    number word. Each plant records its `planted_span`.
  - **real**: the wrong claims the owner flagged in step 4.

  On the **Planted** tab the owner gives a **quick yes/no that each planted
  claim is actually false** (~5 s each). Unconfirmed plants aren't scored.
- **`run --variant X --model M --mode llm|pipeline`** pins the `verify` role.
  - It **refuses when the verify model's effective model is the same as the
    source describe model**.
  - `llm` mode is the headline model comparison.
  - `pipeline` mode adds the CLIP gate and override, which is the number that
    would actually ship.
  - Transport errors are never cached.
- **`report`**:
  - catch rate on the planted set, overall, per type and **matched** (a flag
    must overlap the planted span)
  - catch rate on real errors
  - **false-rejection rate** on the clean set
  - s/photo
  - a table of describe source × verify model showing only legal pairs

Tests: `tests/test_verify_eval.py` (deterministic planting, the same-model
refusal, the matching rules, `[]`-on-error counted as a transport error).

### 7. Cross-pass summary

**New: `evals/model_eval_summary.py`.**
- Reads `<dir>/models.json`, which the owner fills in during Phase 0: VRAM when
  loaded on its own, context length, reasoning setting, whether the model sees
  images, and **swap time** (seconds to load it cold in LM Studio).
- Calls each harness's `build_report()`.
- Prints one table per role with the headline metric, screen flags and solo
  s/photo.
- Picks per role **independently**. The constraints are that each model fits
  in 24 GB on its own and that verify ≠ describe. It marks the Pareto-dominant
  choices per role and doesn't pick a winner.
- **Sequential-schedule estimate:** for a typical batch (default 1,373 photos,
  the 2026-09-19 shoot), fleet wall-clock = Σ over passes (photos × solo
  s/photo) + one swap per model change. Two roles sharing one model save a
  swap, so shared models show up as a cost saving rather than a requirement.
- There is **no all-models-loaded final run**, because production won't load
  them together.

**Production follow-up (outside the harness, not built here).** The fleet
currently runs every pass concurrently, with LM Studio keeping several models
resident (max-loaded ≥3, TTL off). Switching to one model at a time means:
- launching the passes in dependency order: describe → category-content +
  keywords (one text model) → verify → category-visual → aesthetics
- setting LM Studio to one loaded model with JIT loading, or unloading
  explicitly between passes

`/batches`' one-click fleet launch would need to learn that sequence. Plan it
once the winners are known.

## Ordering

```
1 infra ─┬─ 2 aesthetics
         ├─ fetch-originals (NAS up) ─ 3 describe runs ─ 4 labels (owner) ─ 6 verify
         └─ 5 text (freeze from stored) ─────────────── chain check after 4
                                              all ─ 7 summary
```

Run GPU jobs only with the worker fleet stopped, and with one model loaded at a
time. Watch jobs with a marker file or the log's final line, never
`pgrep -f`.

## Risks

- `/api/v0/models` is unverified. If it's missing, latency reads `unknown`
  unless the operator passes `--solo`.
- The describe hook and the verify extraction touch production code. Both are
  behaviour-neutral and tested. The verify extraction is the larger change and
  can be deferred, since `llm` mode doesn't need it.
- The CLIP noun check needs torch, so it only runs on the desktop; CI mocks
  `embed_text`.
- Splitting descriptions into claims is crude. The free "other wrong" flag
  mitigates it.
- Pooled recall misses terms that no model and no stored value proposed. The
  search box mitigates it.
- Templated plants may read less naturally than real hallucinations, which
  makes them easier to catch. The real-error set is the check on this.
- 28 photos is small for ρ. Report the CI, and treat a gap under 0.1 as a tie.

## Estimate

**Engineering: about 31 h.**

| step | hours |
|---|---|
| 1 infra + hook | 4 |
| 2 aesthetics | 2.5 |
| 3 describe | 4 |
| 4 API + page + `PS.Chip` | 7 |
| 5 text passes | 5 |
| 6 verify | 5 |
| 7 summary | 2 |
| test slack | 1.5 |

**Owner labelling: about 3 h.**

| task | minutes |
|---|---|
| visible text | 8 |
| claims (about 70 photo views) | ~70 |
| pairwise | ~25 |
| categories | ~40 |
| keywords | ~20 |
| planted yes/no | ~5 |

**GPU: about 2.5 h** of wall-clock time, including model swaps.

## Decisions

| topic | option A | option B | owner's choice | date |
|---|---|---|---|---|
| Code/page structure | Per-pass modules, routers and pages (`describe_eval`, `category_eval`, `verify_eval`; `eval_describe.html`, `eval_categories.html`) over a thin `eval_common` | One shared `photosearch/model_eval.py`, one `/api/eval/models` router, one `/eval/models` page with tabs; the CLIs stay per pass | **B: one shared module + one page** | 2026-09-26 |
| Review of planted errors | The owner approves each full planted description for naturalness (~90 s each, ~1 h) | Templated plants with a recorded span; the owner gives a yes/no that each claim is false (~5 s each) | **B: quick yes/no per claim** | 2026-09-26 |
| Inputs for the text passes | Freeze the chosen describe model's descriptions after Phase 2 (text work waits on describe) | Freeze the stored production descriptions now; run the describe→text chain on the winner later as a check | **B: freeze from production now** | 2026-09-26 |
| Loading models in production | All role models loaded at once in 24 GB (the handoff's goal), verified by a final all-loaded run | One model loaded at a time with the passes run in sequence; winners chosen per role, each only has to fit alone | **One at a time, sequential passes** (raised by the owner after the debate) | 2026-09-26 |

## Runbook (build done — this is the order to run it)

With the worker fleet stopped, and ONE model loaded in LM Studio at a time
(`--solo` if LM Studio's `/api/v0/models` isn't available):

```bash
export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1 PHOTOSEARCH_LLM_REASONING_EFFORT=none
DB=photo_index.db.local

# 0. sample + pixels, once, while the NAS is up
python evals/describe_eval.py sample --db $DB          # --list to see text candidates
python evals/describe_eval.py fetch-originals
python evals/text_passes_eval.py freeze --db $DB       # production descriptions

# 1. aesthetics (re-run the qwen baseline first: the old rho 0.70 is 'legacy')
python evals/aesthetics_bakeoff.py --photos-dir evals/aesthetics-bakeoff/sample --vlm <id>

# 2. describe, per model, then screen
python evals/describe_eval.py run --variant <short> --model <id>
python evals/describe_eval.py report --db $DB

# 3. text passes, per model (no GPU contention with describe needed — inputs are frozen)
python evals/text_passes_eval.py run --pass category-content --variant <short> --model <id>
python evals/text_passes_eval.py run --pass keywords         --variant <short> --model <id>

# 4. owner: /eval/models — Visible text, Claims, then Categories / Keywords
python evals/describe_eval.py pairs --baseline <prod> --variants <finalist>[,<finalist>]
#    then the Pairwise tab

# 5. verify, from one describe variant's labelled descriptions
python evals/verify_eval.py plant --source <describe variant>   # then the Planted tab
python evals/verify_eval.py run --variant <short> --model <id> [--mode pipeline --db $DB]

# 6. summary
python evals/model_eval_summary.py init-models          # fill in vram_gb + swap_s
python evals/model_eval_summary.py report --db $DB [--assign role=model,...]
```

Re-run any `run` without `--force` after a failure line: only the gaps are
filled.


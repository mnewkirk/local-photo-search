# Handoff — 2026-09-26: evaluating new LM Studio models across the LLM passes

Status: **planned, not started.** Two things come first: finish the
visual-tag reruns (below), then process today's photo run. Don't run the model
evals while the worker fleet is running: they share the GPU with it, so both
the latency numbers and LM Studio's stability suffer (see "Operational
traps").

**Models under test** (just installed in LM Studio):

- `google/gemma-4-12b-qat`
- `minicpm-v-4_5`
- `google/gemma-4-e4b`

**Baselines to beat** (current production role models):

| role | passes | current model |
|---|---|---|
| describe | describe, and verify's regeneration | `qwen/qwen3.5-9b` |
| verify | verify | `google/gemma-4-e2b` |
| visual | category-visual | `qwen2.5-vl-7b-instruct` (runner-up `gemma-4-26b-a4b`) |
| text | category-content, keywords | `llama-3.2-3b-instruct` |
| aesthetics | aesthetics | `qwen2.5-vl-7b-instruct` |

Confirm the table against the fleet's exported env before trusting it (the
`project_worker_fleet_lmstudio` memory has the launch recipe). The mapping from
pass to role lives in `describe.PASS_ROLES`.

**The goal is one configuration, not five separate winners.** All the role
models have to stay loaded in LM Studio at once on a single 24 GB GPU. So a set
of per-role winners that doesn't fit in memory together is not a valid answer.
Report the winning set, what it costs in memory, and whether it fits.

---

## Where each pass stands on evaluation

| pass | harness | ground truth | state |
|---|---|---|---|
| category-visual | `evals/visual_tags_eval.py` + `evals/visual_tags_unsplash.py` | 60 owner-labelled photos (precision and recall); 901 photos with photographer keywords (recall only) | **solid**, just point it at the new models |
| aesthetics | `evals/aesthetics_bakeoff.py` | a 28-photo hand ranking (Spearman ρ) | **usable but small**: qwen2.5-vl scored ρ 0.70 |
| describe | none (the llama3.2-vision vs llava 100-image bake-off was never saved to the repo) | none | **build** |
| category-content | none | none | **build** |
| keywords | none | none | **build** |
| verify | none | none | **build** |
| agent (Ask) | `evals/bakeoff.py`, `evals/mcp_bakeoff.py` | judged by eye | optional: only matters if one of these models can call tools |

---

## Phase 0 — check each model can run at all (~15 min)

For each of the three models:

1. `python evals/visual_tags_eval.py run --probe --model <id>` sends one
   synthetic image and reports whether the model can see it.
2. **Thinking mode.** The gemma-4 family thinks by default and returns `''`
   unless `PHOTOSEARCH_LLM_REASONING_EFFORT=none` is set. MiniCPM-V 4.5 also
   has a "deep thinking" mode. Assume all three need the setting until a run
   without it gives real answers. The symptom: "unanswered" answers, with the
   whole `max_tokens` budget spent on reasoning.
3. **Context length.** LM Studio's JIT default is 4096 tokens, split across
   parallel slots, so vision requests fail with `Context size has been
   exceeded`. Raise it per model before measuring (qwen3.5-9b needed 16384,
   gemma 8192).
4. Write down each model's VRAM use when loaded on its own.

## Phase 1 — passes that already have a harness (~1–2 h of GPU time)

**category-visual.** For each model:

```bash
export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1 PHOTOSEARCH_LLM_REASONING_EFFORT=none
python evals/visual_tags_eval.py     run --variant prod-<short> --model <id>
python evals/visual_tags_unsplash.py run --variant prod-<short> --model <id>
# and the prompt with definitions + trial tags, for the best one or two:
SKY="--prompt-file evals/prompts/visual_defs_sky.txt --extra-vocab action,blue-sky,cloudy"
python evals/visual_tags_eval.py     run --variant sky-<short> --model <id> $SKY
python evals/visual_tags_unsplash.py run --variant sky-<short> --model <id> $SKY
python evals/visual_tags_eval.py report
python evals/visual_tags_unsplash.py report
```

`--model` fixes the visual role model for that process only. On the LM Studio
route the model name at the call site is ignored, so without it the report
would score whatever model is configured under a different name.

**aesthetics.** `python evals/aesthetics_bakeoff.py --photos-dir
evals/aesthetics-bakeoff/sample --vlm <id> ... --ground-truth <the ranking
CSV>` (find the ranking file the 2026-07-09 run used before starting). Report
three things besides ρ:
- how often the JSON fails to parse
- how spread out the scores are (squashed scores were the original LAION
  problem)
- seconds per photo

28 photos is small. Treat a ρ difference under ~0.1 as a tie.

## Phase 2 — build the describe eval (the biggest gap)

Descriptions feed everything downstream. The text passes only ever see the
description. The sailboat photo that got tagged "soccer" started as a
truncated description that category-content then made things up from. So
describe is the pass where a model choice matters most, and it has no eval.

There's no single right description to score against, so measure the ways a
description goes wrong:

1. **Wrong claims (the main metric, needs the owner).** A labelling page in
   the style of `/eval/visual-tags` shows a photo plus one model's description,
   with the model's name hidden and the order randomized. The owner clicks any
   wrong claim, or marks it "all correct". The scores are the share of
   descriptions with no wrong claims, and wrong claims per description.
2. **Blind side-by-side preference (owner).** Two descriptions of the same
   photo from different models; pick the better one or call it a tie. Gives a
   win rate for each pair of models. Keep the count low (~40 photos) because
   this is the expensive part.
3. **Automatic checks (no owner needed), to screen before spending owner
   time:**
   - how often the output is degenerate (`describe._is_degenerate`) or cut off
   - length
   - how often it retries or falls back
   - latency
   - the CLIP noun check from verify's first pass (`verify.py`), which flags
     nouns the photo's CLIP embedding doesn't support
   - text-in-photo accuracy on a small subset where the owner writes down the
     visible text once

**Sample:** reuse the 60-photo visual-tags sample. It's already split into
categories (sports, indoor, low-light, travel and so on), the full-size images
are known to fetch, and describe then shares a photo set with category-visual.
Add ~10 photos with lots of visible text; that's what decided the last
describe bake-off.

**Build it on what the visual-tags eval already has:**
- labels and run caches kept as **files** under `evals/`, git-ignored, never
  in DB rows (`sync-replica.sh` replaces the replica DB wholesale)
- samples drawn read-only (`mode=ro`)
- one cache file per variant, recording the model that actually ran
- resumable; failed photos aren't cached
- the eval API only serves locally and never forwards to the NAS

Put the storage in a shared module (like `photosearch/visual_tag_eval.py`) so
the harness, the API and the page all agree on one format. Call the
**production** `describe_photo` so the real prompt, retry and fallback code
paths are what gets measured.

## Phase 3 — category-content and keywords (text-only)

The input is a description and the output is a list, so these are cheap to
run, and a small model like `gemma-4-e4b` may win outright on cost.

- **Freeze the inputs.** Take the Phase 2 descriptions from **one** fixed
  model (or the current production ones in `generations`), so the text-model
  comparison doesn't also swing with description quality. Then run the whole
  describe → text chain once for the winning describe model as a sanity check.
- **Automatic check for unsupported tags:** the share of categories and
  keywords with no word-level support in the description. That catches the
  "soccer" failure directly.
- **Owner labels:** the correct categories for ~60 descriptions (a vocabulary
  checklist, the same interaction as the visual-tags page) → precision and
  recall. For keywords, the owner marks the wrong ones → precision only;
  keywords are free-form, so there's no fixed answer list to score recall
  against.
- **Latency matters here:** these passes use a 10-second per-call timeout
  (`_TEXT_OLLAMA_TIMEOUT_S`), so report timeouts per model.

## Phase 4 — verify

Verify asks a vision model to confirm or reject specific claims in a
description, and regenerates the description when a claim is rejected. The
evaluation needs descriptions where we already know which claims are false:

- **Planted-error set:** take descriptions from Phase 2 that the owner
  confirmed correct, and plant one false claim in each (an object that isn't
  there, the wrong colour, the wrong count). Score how many planted errors are
  caught, and how often a clean description is wrongly rejected.
- **Real-error set:** the wrong claims the owner found in Phase 2.
- **Rule:** the verify model must be a different model from the describe
  model, to keep the check independent. Score only pairings that obey that.

## Phase 5 (optional) — agent

Only if a candidate model supports tool calling: `evals/bakeoff.py` and
`evals/mcp_bakeoff.py` with the model added.

---

## State at handoff (visual-tag reruns, 2026-09-26)

Changes made today (commits `a1857a3` and before):
- Dropped two contradiction rules: dramatic×peaceful and colorful×muted.
- Parser: `misty` now reads as `foggy`.
- Trial tags `blue-sky` and `cloudy` added.
- Labeller definitions for `foggy` vs `hazy`.
- New prompt `evals/prompts/visual_defs_sky.txt`.
- The Unsplash sample was redrawn: 901 photos (was 828), including 40
  `blue-sky` and 40 `cloudy`.
- Old run caches backed up to `evals/visual-tags/backup-2026-09-26/`.

qwen2.5-vl-7b, production prompt → sky prompt:
- Owner-labelled 60 photos: precision 0.51 → 0.52, recall 0.40 → 0.40.
- Unsplash recall: 0.39 → 0.39. Notable per-tag changes:
  - `foggy` 0.38 → 0.55
  - `moody` 0.40 → 0.65
  - `blue-sky` 0.70
  - `cloudy` **0.10**. Checked: on the 40 cloudy photos qwen mostly said
    `sunny` (17) and `blue-sky` (15), not `overcast` (9)
  - `hazy` still ≈ 0
  - `peaceful` 0.53 → 0.30
  - `monochromatic` 0.47 → 0.28

gemma-4-26b-a4b, production → sky prompt (all runs complete; the failed
photos were re-run):
- Owner-labelled 60 photos: production 0.69 precision / 0.39 recall; sky 0.61 /
  0.39. But 12 of sky's false positives are `blue-sky` (9) and `cloudy` (3),
  tags the owner hasn't labelled yet. Without those, sky's precision is
  **0.68**, level with production.
- Unsplash recall: production 0.40, sky **0.43**, the best of the four runs.
  Notable per-tag changes:
  - `hazy` 0.10 → **0.42**: the definition works for gemma, not for qwen
  - `foggy` 0.53 → 0.62
  - `cloudy` 0.35
  - `blue-sky` 0.72
  - `peaceful` 0.78
  - `soft-light` 0.92
  - `action` 0.39
  - costs: `colorful` recall on the owner set 0.53 → 0.18, `sunny` 0.28 → 0.35
    (still weak), `monochromatic` and `snowy` ≤ 0.17
- The same gemma model ran at 0.4 s/photo once the LM Studio restart left it
  loaded alone.

**So for category-visual:** gemma with the sky prompt is the best result so
far: same precision as production gemma and the highest recall. qwen stays
behind on precision (0.51). Label `blue-sky` / `cloudy` on the owner set
before deciding whether they ship.

**The owner's 60 labels predate `blue-sky` and `cloudy`.** So the report
counts any `blue-sky` a model emits as a false positive, even when it's right.
Label those two tags on `/eval/visual-tags` before reading them on the owner
set. The blind recheck set (`?set=recheck`) is still 0/15.

---

## Operational traps (all hit today)

- **NAS 502s fetching originals.** The owner-set runs fetch full-size photos
  through the replica, which pulls them from the NAS. During a NAS restart
  28 photos got `502 Bad Gateway`. They aren't cached, so re-running **without
  `--force`** fills just the gaps. Worth adding before Phase 2: a local cache
  of each sample's originals (read once, reuse for every model). Three models ×
  several passes would otherwise pull the same ~70 originals from the N100 over
  and over.
- **LM Studio stops responding and the run keeps going.** At 11:21 a connection
  dropped, then ~80 photos in a row got `400 Bad Request`; the run carried on
  and marked them failed. Check the `N photo(s) failed` line after every run
  and re-run to fill the gaps.
- **Speed depends on what else is loaded.** gemma-4-26b ran at **3.1 s/photo**
  with qwen2.5-vl also loaded and **0.45 s** alone: 7× faster, same model, same
  answers. Measure each model's speed with **only that model loaded** and say
  so in the report; otherwise the numbers compare memory pressure, not models.
  The whole set of role models still has to fit at once in production (see the
  goal above).
- **Watching a job:** use a marker file or the log's final line, never
  `pgrep -f` (CLAUDE.md, "Watching a long job").

# Model evals

Read before running or extending a model eval harness (`/eval/models`).

<!-- moved verbatim from CLAUDE.md (line 1077) on 2026-10-19 -->
### Model evals for every LLM pass (`/eval/models`)

`docs/plans/model-eval-harnesses.md` (plan + runbook). Harnesses:
`evals/aesthetics_bakeoff.py`, `describe_eval.py`, `text_passes_eval.py`,
`verify_eval.py`, `model_eval_summary.py`; storage in `photosearch/model_eval.py`
(files under `PHOTOSEARCH_MODEL_EVAL_DIR`, default `./evals/model-evals`,
git-ignored); owner labelling on `/eval/models` (`model_eval_api.py`, local-only).
Production is to run **one model loaded at a time, passes in sequence** (owner
decision 2026-09-26), so winners are picked per role.

Traps these exist to catch — don't undo them:

- **`--model` must PIN the role env var** (`model_eval.pin_role_model`). On the
  LM Studio route the call-site name is ignored and the vision roles fall back
  to `PHOTOSEARCH_LLM_VISUAL_MODEL` (since `4cfc202`, 2026-07-10). The
  2026-07-09 aesthetics "qwen ρ 0.70" predates that fallback and IS qwen's —
  verified from file times vs git history; it reports as `legacy:` only because
  scores.json records no effective model.
- **Production turns transport failures into answers**: `describe_photo` →
  None, `llm_verify_description` → `[]` (== ALL CORRECT), the text extractors →
  None. Every harness wraps the call in `model_eval.Recorder`, which raises
  `TransportError` (never cached) when the LAST call errored. A text call that
  timed out on every attempt IS cached, as `deferred` — that is what production
  does at the 10 s limit.
- **Retries hide timeouts and truncation**: `describe._ATTEMPT_HOOK` (None in
  production, exceptions swallowed) reports each attempt's outcome and
  `finish_reason`.
- **Labels are keyed by `text_sha`, never by variant**, so the page is blind by
  construction (a test asserts no variant/model name in any response).
- **Verify's pipeline lives in `verify.check_description`** (extracted from
  `worker._process_verify`, parity-tested); `llm_all=True` is the pure-model mode.
  The verify model must not be the describe model — `run` refuses.
- Pixels come from a local **originals cache** (`fetch-originals`), so neither a
  run nor the labelling page depends on the NAS staying up.
- **A run records the production spec it measured** (`model_eval.spec_parts` →
  `make_spec`: the prompt the real builders produce, guard/parser source,
  options, reasoning effort as sent). Resuming under a different spec refuses
  (`--force-stale` appends and stays STALE); every report and the summary mark
  a run whose hash is missing or differs as **STALE**. All runs before
  2026-10-25 are STALE. Reports also **replay current guards** over cached
  outputs (describe `_description_problem`; text passes re-run the extractor
  on the cached raw answer). Headlines in `model_eval_summary.py` are over the
  full sample with a seeded per-photo bootstrap CI; `≈` = tie with the best,
  and no Pareto star on a lead inside the CI. Category labels record the pool
  shown; tags outside it are unjudged. Verify `matched` uses shipped `flagged`
  nouns by head noun only. History: docs/plans/model-eval-harnesses.md
  ("Grading fixes (2026-10-25)").

### Ask prompt eval (`evals/ask_eval/`)

Before/after check for trimming the Ask agent's ~11.7k-token prompt (system prompt + tool list); runbook in `evals/ask_eval/README.md`. `harness.py run` refuses while the worker fleet is busy (local worker pids, worker containers, NAS active claims; fails closed) and has no override. `measure` calls no model.

### Visual-tag + aesthetics eval statistics (`visual_tags_eval.py`, `visual_tags_unsplash.py`, `aesthetics_bakeoff.py`)

- **A constant answer is the bar.** `report` adds `baseline:top1..3` (most-labelled tags on every photo), `recite-all` and `empty`; on the 60 labels `colorful, soft-light, sunny` scores micro-F1 0.517 and only the shipped prompt on minicpm (attrD) beats it (paired-bootstrap CI above 0). Read the "verdict" column, not P/R alone.
- **Rows compare only on one photo set**: by default every variant is scored on the intersection of cached photos (`--all-photos` opts out, with a warning). F1/macro/CIs are perceived-vocabulary only; macro averages tags with >= 5 positives.
- **STALE**: a run records `prompt_sha` (live `_build_visual_prompt` when no `--prompt-file`), `code_sha` (guard/parser source), temperature and reasoning effort; `run` refuses to resume on any mismatch, `report` marks runs that lack or mismatch them. All runs before 2026-10-25 are STALE.
- **The guard/parser are invisible in scored runs** (models answer clean comma lists), so `spec` / the report's spec section mutates cached answers through the production code; `tests/test_visual_tags_eval_harness.py` fails if the parser or guard regresses.
- **Labels**: recheck kappa < 0.4 flags a tag "unreliable label"; the report warns when main labels were re-saved after the blind recheck (true for all 60 as of 2026-09-27).
- **Unsplash is recall-only**: compare a variant with the constant answer of its own size (recite-all = recall 1.00).
- **Aesthetics**: the 28-photo ordering (ranks 10 and 28 absent) is between-shoot; read top-half/bottom-half ρ, the file-size/megapixels baselines and the Δρ ties. `scores-v2.json` items now also store `dim_scores`/`sub_scores` for offline `DIMENSION_WEIGHTS` tuning.

### Ablation findings (2026-10-28)

`evals/ablations-2026-10-28.md`: context-layer ablations on Haiku/Sonnet/Opus (docs mostly buy turns, six traps live only in docs, CLAUDE.md suppresses the skill) and Ask prompt ablations on the local models (16% of asks flip under no-op edits; gemma-4 tool calls malformed via LM Studio).

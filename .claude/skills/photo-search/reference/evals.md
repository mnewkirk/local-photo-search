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

### Ask prompt eval (`evals/ask_eval/`)

Before/after check for trimming the Ask agent's ~11.7k-token prompt (system prompt + tool list); runbook in `evals/ask_eval/README.md`. `harness.py run` refuses while the worker fleet is busy (local worker pids, worker containers, NAS active claims; fails closed) and has no override. `measure` calls no model.

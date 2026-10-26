---
paths:
  - "photosearch/describe.py"
  - "photosearch/verify.py"
  - "photosearch/rerun.py"
  - "photosearch/aesthetics.py"
  - "photosearch/visual_tags_derive.py"
  - "photosearch/visual_collapse.py"
  - "photosearch/visual_tag_eval.py"
  - "photosearch/stale_descriptions.py"
  - "photosearch/category_grounding*.py"
  - "photosearch/subjects.py"
  - "photosearch/model_eval*.py"
  - "photosearch/eval_api.py"
  - "photosearch/vocab_*.py"
  - "evals/*.py"
---
# LLM passes — traps

Full history and measurements: docs/notes/llm-passes.md (cite by heading). Runbooks: .claude/skills/photo-search/reference/workers.md (LM Studio routing), evals.md.

- **Routing LLM passes to LM Studio (OpenAI-compatible)**: `PHOTOSEARCH_TEXT_LLM_URL` (ends `/v1`) picks the model by ROLE env var, never the call-site name: describe → `PHOTOSEARCH_LLM_DESCRIBE_MODEL`, verify → `_VERIFY_`, category-visual → `_VISUAL_`, category-content/keywords → `_TEXT_`, aesthetics → `_AESTHETICS_` (all `PHOTOSEARCH_LLM_*_MODEL`). Change fleet models only in `rerun.FLEET_ROLE_MODELS`; `admin_api._fleet_env` sets every role from it (override: `PHOTOSEARCH_FLEET_<ROLE>_MODEL`).
- **Reasoning models need `PHOTOSEARCH_LLM_REASONING_EFFORT=none`**: every fleet model thinks; without it they spend all of `max_tokens` on reasoning and return `''`. The UI fleet sets it from `rerun.FLEET_REASONING_EFFORT`; keep it. The field is only sent when the RESOLVED model isn't on `describe._NO_REASONING_MODEL_PATTERNS` (qwen2.5-vl, qwen2.5, llama-3.2/llama3.2, llava) — extend via `PHOTOSEARCH_LLM_NO_REASONING_MODELS` rather than editing the list for a one-off model.
- **Each pass unloads its model from LM Studio when it drains** (`worker._release_llm_models` → `describe.unload_openai_models`). Keep a model the next pass or another worker's live claim still uses. A model loaded into full VRAM stays spilled to CPU: text calls all time out with 0 reasoning tokens.
- **Per-pass model strategy (Ollama defaults)**: describe `llama3.2-vision`, category-visual + verify `llava`. The verify model must differ from the describe model (independent check).
- **`category-visual`: derived capture facts vs perceived qualities**: `visual_tags_derive` splits the vocabulary into four disjoint groups: derived (`CAPTURE_FACT_TAGS`, from EXIF), `RETIRED_TAGS`, `FROZEN_TAGS` (`sharp`/`blurry`: never derived, never asked, never deleted), `PERCEIVED_VOCABULARY`. `PROMPT_SECTIONS` must partition `PERCEIVED_VOCABULARY` exactly. Derived wins: strip every capture-fact term the model emitted, even when none derived; missing EXIF derives nothing.
- **Frozen terms make the input's provenance part of the API**: write path = `merge_vlm_answer` (drops frozen terms the model volunteered, carries stored ones from `existing`); backfill = `merge_stored_tags` (frozen pass through). Swapping them loses data. Apply the merge SERVER-side (`worker_api.submit_results`, both `index.py` writers). A transient lock on the EXIF read must re-raise, not write strip-only.
- **Two kinds of empty**: `tag_visual_photo` returns `[]` for an answered "no tags" (write `'[]'`, done) and `None` for no usable answer (leave `visual_tags` NULL, still mark processed). Keep `CategoryVisualResult.visual_tags` Optional.
- **`_parse_visual_response` is deliberately tolerant** (commas, newlines, semicolons, bullets, `Label:`, fences, JSON). Never reduce it to a comma split. Prompt examples must be bare tag lines. Test: `tests/test_visual_parse_tolerance.py`.
- **Don't reintroduce axes-to-fill or worked examples without re-running the A/B** (`evals/visual_tags_eval.py`): both leak into answers. **Temperature stays at 0.** Never enforce one-tag-per-axis.
- **The guard** (`describe._visual_answer_problem`): the over-select limit is `_VISUAL_MAX_PLAUSIBLE_TAGS = _VISUAL_MAX_TAGS + 1`; never hardcode it. On a `CONTRADICTORY_PAIRS` hit it retries once, then drops BOTH members, so add a pair only if no single honest frame could be both.
- **Backfill** (`derive-visual-tags`, `ground-categories`): dry run opens `mode=ro`; 2,000-row chunks; `UPDATE … WHERE id=? AND <col> IS ?`; skip unchanged rows; never touch NULL (NULL drives the worker queue). `merge_tags` sorts on purpose: deterministic output is what makes the guard work.
- **Collapse across a shoot is flagged, not gated** (`visual_collapse.py`). `clear-pass` must stash frozen tags first (`db.stash_frozen_visual_tags`).
- **The percentile refresh is chunked — and must stay that way** (`aesthetics.PERCENTILE_CHUNK_ROWS`): compute every percentile from ONE snapshot read before the first chunk; guard each UPDATE on the source score (`WHERE id=? AND {score_col} IS ?`); skip unchanged rows; `applied` = rows written. Gates must use the writer's predicate (`maintenance._aesthetics_missing_sql`). Test: `tests/test_aesthetics_normalize_chunking.py`.
- **VLM aesthetic scoring (the `aesthetics` pass)**: primary signal is the VLM; rank/search on `aes_overall_pct` (percentile), not raw scores. LAION `aesthetic_score` stays as a cheap prior. No IQA anchor (`aes_*_iqa` stay NULL). `DIMENSION_WEIGHTS` are retunable with `recompute-aesthetic-overall`, no VLM re-run.
- **A new description re-queues what was derived from it**: every description write calls `PhotoDB.invalidate_description_dependents`. Pass order (`batch_state.WORKER_PASSES` = `rerun.ALL_PASSES` = `worker_api._ALL_PASSES` = `batch-flow.js`) runs verify before category-content/keywords; change all four together. Verify never re-tags.
- **Bad LLM output is a failure, not a result**: chat helpers read `finish_reason`/`done_reason`, retry once at double budget, then raise `describe.TruncatedOutput`; guards raise `describe.UnusableAnswer` (one attempt + `index_errors`, nothing written). A timeout on the double-budget retry is a truncation (`_expanded_retry`), not a stall. Per-role caps: `_OPENAI_MAX_TOKENS`. A cut-off verify verdict must propagate, never return `[]` (= ALL CORRECT). Test: `tests/test_text_pass_safeguards.py`.
- **category-content grounding: the description decides**: `category_grounding.ground()` is pure and idempotent (subset of input, input order), and runs on the worker AND on the server against the current description. Four groups (sport, setting, light, weather); `SCRUB_PATTERNS` blank negations first. No retry when the description decided a drop. Never drop unsupported padding. `is_recitation` (vocab order + `_CONCRETE` nouns) replaces the count cap; `MAX_PLAUSIBLE` is only a backstop. When grounding changes an answer, the pre-grounding list is logged as `category-content-raw` immediately before the grounded row (the eval pairs them by `id - 1`); the server grounds the worker's `raw_categories`. A new cue or term needs a phrase in `tests/test_category_grounding_rules.py`.
- **Provenance: log the model that RAN, not the one configured**: stamp via `describe.effective_model(model, role)` / `effective_model_version` through `worker._provenance_kwargs`; `describe.PASS_ROLES` must match `rerun._PASS_LLM`. Verify logs with `role="describe"` (its artifact is the regen). No per-branch `_model_version(` call in the worker loop (`tests/test_llm_provenance.py`).
- **Ollama stall on the text passes — tight per-call timeout**: keep `_TEXT_OLLAMA_TIMEOUT_S` (10 s) on `extract_categories_from_description` / `extract_keywords_from_description`; a stall then defers. No restart watchdog.
- **Model evals**: `--model` must pin the role env var (`model_eval.pin_role_model`); a transport failure raises `TransportError`, never caches as an answer. Eval labels are files, never DB rows, and the eval API never proxies to the NAS.

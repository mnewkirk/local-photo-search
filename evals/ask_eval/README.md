# Ask eval: does a smaller Ask prompt still plan searches right?

The in-app ✨ Ask agent (`photosearch/agent.py`) sends the same block before
every round: a system prompt (routing rules, worked examples, library facts,
household notes) and the whole tool list (`tools.openai_tools()`). On
`pre-ask-diet` that is 44,545 characters, which LM Studio counts as **~11.7k
prompt tokens** for `qwen/qwen3.5-9b`. That model is loaded with a 16,384-token
context and the worker fleet sends its describe/video-describe calls to the same
instance. One Ask plus two fleet requests overflowed it on 2026-10-25 (LM
Studio: `Context size has been exceeded.`; the agent logged a bare `HTTP Error
400` and fell back to single-shot, which found 0 photos).

This harness is the before/after check for trimming that prompt, built like
`evals/context_eval/`. A **config** is one git ref. Rows append to
`results/<config>/<model>.jsonl` (git-ignored). `report` compares configs and
`regrade` re-scores stored rows without calling a model. It drives the real
`agent.run_agent` against the replica DB and LM Studio. Nothing is mocked except
the safety wrappers below.

## Never run it while the fleet is busy

`run` and `ablate` call `fleet_busy()` before they start and again before every
chunk (one model × one rep × one mutant). They refuse (exit 2) if any of these is true:

- a `cli.py worker` or `run-workers.sh` process is alive on this machine
  (scanned from `/proc`, not `pgrep -f`), or a live pid sits in a
  `/tmp/photosearch-worker-fleet*/worker-*.pid` file;
- a `docker ps` container name contains `worker` (other than the Ollama one);
- the NAS (`--nas`, default `$PHOTOSEARCH_NAS_URL` or `http://dxp4800-f976:8000`)
  reports active claims on `/api/worker/status?passes=clip`, **or cannot be
  reached** (the guard fails closed).

There is no override flag. The fleet shares LM Studio: an eval run during a
batch slows the batch and measures contention instead of the prompt.
`--dry-run` prints the plan and its cost either way and calls no model.

## Is the eval any good? Its "test coverage"

An eval needs the same question a test suite gets: would it fail if the thing it
guards broke? `coverage` and `mutation` answer it at five levels.

1. **Rule coverage (line coverage).** Every rule, block and worked example in
   the prompt has an item that declares it in `covers`. `coverage` lists each
   one with its size and the share of real asks that need it.
2. **Usage coverage (hot paths).** `coverage` reads the replica's Ask logs
   (`ask-logs/ask-*.md`) and classifies each real ask by the behaviours it needs
   (`mutants.USAGE`). It reports per-behaviour item counts against real shares,
   and **combination coverage**: the share of real asks for which ONE item
   exercises everything that ask needs together. Real asks combine three or more
   behaviours (a per-year spread + foreground subject + framing flags), which
   per-behaviour counts hide.
3. **Mutation score (mutation testing).** `ablate` deletes one named piece of
   the prompt (`mutants.py`: a rule line, a block, an example, a tool
   description, a tool's parameter descriptions, a whole tool, or the duplicated
   filter descriptions) and reruns the items. `mutation` marks a mutant
   **killed** when some item's pass rate drops by at least 50% against the
   baseline. A survivor is either a coverage gap or text the agent does not
   need. The trim itself is a sequence of mutations, so this is the number that
   says whether the eval will notice one going wrong. The survivors, sorted by
   size, are the first candidates to cut.
4. **Grader validity.** Replaying the June `evals/bakeoff.py` traces through the
   first graders found two false passes: a `min_quality: 6` floor for "good
   photos", and the whole family added to "our best photos last summer". Every
   item now carries an `allowed_args` list, so "fill ONLY the filters you
   inferred" is checked on every call (`test_allowed_args_catches_the_june_false_passes`).
5. **Power.** Temperature is 0, so the reps of one item are strongly correlated
   and the sample is the items, not the runs. `report` flags an item REGRESSED
   only on a drop of at least half its runs, and adds a paired sign test over
   items (worse vs better) for each model.

Coverage on `pre-ask-diet` (`harness.py coverage --ref pre-ask-diet`):

- rules and blocks covered: 20/20;
- worked examples covered: 9/9;
- combination coverage: 112 of 116 real asks (97%);
- tool parameters read by some check: 50/133.

The combination figure depends on the `covers` tags and a keyword classifier,
both written by hand, so it is a self-assessment. The mutation score is the
independent check. Neither has been run yet.

## Suites (`asks.jsonl`, 63 items)

| suite | items | what it measures |
|---|---|---|
| `real` | 23 | asks copied from the replica's Ask logs: per-year spreads with foreground subject and framing flags, per-day quotas with burst windows, trips chaptered by day with a per-day aesthetic floor, rerank for activities and visual details, a camera per day, the Lake Como timeline (no place name says "Como", so it also tests recovery) |
| `routing` | 13 | people, place, camera, media, mood, exclusion, quality bar and grounding routing (r01–r09 are `evals/bakeoff.py` p01–p09) |
| `curation` | 6 | chapters, scenes, one per year, per day over a trip, `suggest_layout` after a search |
| `counting` | 6 | `summarize` questions whose answer is checked against the replica DB |
| `dates` | 3 | today / an explicit year / no date when none was asked |
| `locked` | 3 | pinned filter-bar values survive a vague ask (`locked_filters`) |
| `followup` | 3 | a second turn keeps, swaps, or reshapes the first turn's search |
| `writes` | 3 | each write tool is called as a dry run, after a search, never with `confirm` |
| `honesty` | 3 | an unregistered person, a date before the library, and a 0-hit search that must relax a secondary filter without dropping the person |

An item passes only if every one of its checks passes, the run had no error,
and it did **not** fall back to single-shot. Each item also gets a partial score
(checks passed / total) in the report. Several real-ask items run 3–5 rounds, so
`max_prompt_tokens` reflects the peak round, where the overflow happens, not just
the first.

### Item fields and check grammar

Items are data, graded by `checks.py` in the harness's own tree, so a check
fixed later re-scores old rows with `regrade`.

- `checks`: `{name, op, tool, arg?, value?, min?, n?, match?, default?}` over the
  trace. `tool` may be a list. `arg` may be a list ("birthday" in `query`,
  `category` or `keyword`). `default` is the tool's own value for an omitted arg
  (representatives `bucket=year`, `n=1`; daily_highlights `per_day=20`,
  `window_minutes=10`), so leaving it out is not marked wrong.
  Ops: `called` (`min`), `not_called`, `count_ge` / `count_le` (`n`; only calls
  where `arg` equals `value`, or matches `match`), `max_calls`, `first_tool`,
  `called_after` (`tool` after one of `value`), `set_eq`, `set_has`,
  `set_disjoint`, `every_has` (every call), `eq`, `in`, `contains`,
  `contains_any`, `matches` (regex), `num_between`, `present`, `date_between`.
  The negative ops (`not_contains`, `absent`, `every_has`) need at least one
  call to the tool, so calling nothing never passes them.
- `allowed_args`: the only filters the search-family and `summarize` calls may
  set, besides `limit`, `dedupe`, `match_source` and the pinned keys.
- `covers`: rule, block, example and behaviour ids (`mutants.RULES`, `BLOCKS`,
  `example.*`, `USAGE`). Used only by `coverage` and `ablate --items covering`.
- Dates may use `{today}`, `{yesterday}`, `{week_ago}`, `{year}`, `{last_year}`,
  `{last_summer}`, `{month_start}`, `{last_month_start}`, `{last_month_end}`,
  resolved from the day the row was **run**.
- A call to the consolidated `search(mode=…)` tool grades as the family tool
  that mode routes to, so offering one search tool is a legitimate trim.
- `answer_expect` / `answer_any` / `answer_reject`: regexes on the answer.
  `answer_sql`: a scalar from the replica DB (read-only) that must appear in it.
- `results_min`; `results_sql`: SQL over the returned ids (`{ids}`) whose scalar
  must equal `expect` (usually a count of hits that break the ask).
- `facts`: plain English for the judge on answer checks prone to phrasing fails.

## Running

```bash
PY=./.venv/bin/python
# no model calls (safe while the fleet runs)
$PY evals/ask_eval/harness.py measure  --ref pre-ask-diet     # prompt component sizes
$PY evals/ask_eval/harness.py mutants  --ref pre-ask-diet     # every mutant and its size
$PY evals/ask_eval/harness.py coverage --ref pre-ask-diet     # rule / usage / parameter coverage
$PY evals/ask_eval/harness.py run    --config before --ref pre-ask-diet --dry-run
$PY evals/ask_eval/harness.py ablate --dry-run [--items covering]

# 1. baseline (63 items x 3 reps = 189 runs, ~1.5 h)
$PY evals/ask_eval/harness.py run --config before --ref pre-ask-diet --repeat 3
# 2. mutation testing on the baseline: the core mutants on their covering items
#    first (~460 runs, ~4 h), then the whole suite for the survivors
$PY evals/ask_eval/harness.py ablate --items covering
$PY evals/ask_eval/harness.py mutation --baseline before
$PY evals/ask_eval/harness.py ablate --mutant <survivor> --rerun
# 3. the trimmed prompt
$PY evals/ask_eval/harness.py run --config after --ref HEAD --repeat 3
$PY evals/ask_eval/harness.py report --configs before after [--failures]
$PY evals/ask_eval/harness.py regrade --configs before after --judge
```

`--ref` is a git ref, checked out as a detached worktree under `/tmp` and
removed by its exact path afterwards (never `git worktree prune`). `--ref
WORKTREE` runs the live checkout, uncommitted edits included; the row records
`dirty`. `ablate` defaults to `--ref pre-ask-diet`, `--repeat 1` and the core
set (`mutants.CORE`: every rule and block, the filter-description de-dup, all
parameter descriptions, the grounding tools); `--set all` adds every example,
tool description, tool's parameter descriptions and whole-tool drop. Each mutant
writes `results/abl-<id>/`, and finished mutants are skipped unless `--rerun`.

Defaults: `--model qwen/qwen3.5-9b`, `--lm http://localhost:1234/v1`,
`--reasoning-effort none` (as the replica and the fleet run), the replica DB
`./photo_index.db.local`, the household notes from `./.agent-hints` (what
`run-local-replica.sh` feeds the replica; `$PHOTOSEARCH_AGENT_HINTS` wins).
`--consolidated` sets `PHOTOSEARCH_CONSOLIDATED_SEARCH=1`.

## What a row records

The trace (tool calls and result summaries), returned ids, the answer, error,
single-shot fallback and its reason, wall / LLM / tool seconds, and per LLM round
the prompt and completion tokens, reasoning tokens and any error with **LM
Studio's response body** (the agent drops it). `first_prompt_tokens` is the fixed
block, `max_prompt_tokens` the largest round, `context_errors` the rounds whose
body mentions the context, `mutant` / `mutant_removed` the ablation applied.
`meta.json` holds, per run, the ref and sha, the `measure` output, and LM
Studio's `/api/v0/models` (each model's `loaded_context_length`). The report's
**headroom** column is `loaded context − largest prompt − 3,000` (the agent's
reply budget): what is left for the fleet's concurrent requests.

## Safety while it runs

`runner.py` runs inside the tree under test and wraps it:

- a write tool called with `confirm=true` is **not executed**; the call is
  recorded in `blocked_writes` and answered with an error. Dry runs execute (they
  write nothing). `PHOTOSEARCH_NAS_URL` is removed, so nothing proxies to the NAS.
- `rerank_photos` is faked unless `--allow-rerank`: the real handler runs with no
  vision model configured (its unranked fallback) and the result is dressed as a
  scored rerank, so the agent's next turn sees a normal result and the routing
  is tested without loading a vision model.
- a mutant is applied only to tool-calling requests; the single-shot fallback
  keeps its own prompt.
- the Ask log goes to `results/<config>/ask-logs/`, the request log is off.

## Caveats

- `tools.openai_tools()` splices the same filter-prop dicts into several tool
  schemas. `mutants.apply` deep-copies each tool separately; one deepcopy of the
  list keeps the aliasing and stripped every tool at once. A trim that edits
  schemas in place has the same trap.
- `mutants.RULES` and `BLOCKS` match the prompt's wording on `pre-ask-diet`
  (`test_every_registered_mutant_finds_its_text`). After a trim rewrites a rule,
  update its regex, or the mutant reports as absent.
- The replica DB is swapped nightly by `sync-replica.sh`; `meta.json` records its
  mtime. Compare configs run against the same snapshot, or at least the same week.
- Prompt-token counts depend on the tokenizer; compare a model with itself.
- One Ask runs at a time: fleet contention is not reproduced, and the headroom
  column stands in for it.

`tests/test_ask_eval_harness.py` covers the item shapes and `covers` ids, every
check op, defaults, the allowlist (including the June false passes), the fleet
guard (refuses, starts nothing, re-checks per chunk, fails closed), the write
block, the rerank fake, the LM Studio error body, every registered mutant against
the live prompt, the shared-schema aliasing, the mutation kill table, the sign
test, coverage parsing, and a scripted run_job → grade round trip. No model, LM
Studio or NAS is called.

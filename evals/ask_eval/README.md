# Ask eval: does a smaller Ask prompt still plan searches right?

The in-app ✨ Ask agent (`photosearch/agent.py`) sends the same block before
every round: a system prompt (routing rules, library facts, household notes)
and the whole tool list (`tools.openai_tools()`). On HEAD that is ~44.5k
characters, which LM Studio counts as **~11.7k prompt tokens** for
`qwen/qwen3.5-9b`. That model is loaded with a 16,384-token context and the
worker fleet sends its describe/video-describe calls to the same instance. One
Ask plus two fleet requests overflowed it on 2026-10-25 (LM Studio: `Context
size has been exceeded.`; the agent logged a bare `HTTP Error 400` and fell
back to single-shot, which found 0 photos).

This harness is the before/after check for trimming that prompt. It is the same
shape as `evals/context_eval/`: a **config** is one git ref, rows append to
`results/<config>/<model>.jsonl` (git-ignored), `report` compares configs and
`regrade` re-scores stored rows without calling a model. It drives the real
`agent.run_agent` against the replica DB and LM Studio. Nothing is mocked.

## Never run it while the fleet is busy

`run` calls `fleet_busy()` before it starts and again before every chunk (one
model × one rep). It refuses (exit 2) if any of these is true:

- a `cli.py worker` or `run-workers.sh` process is alive on this machine
  (scanned from `/proc`, not `pgrep -f`), or a live pid sits in a
  `/tmp/photosearch-worker-fleet*/worker-*.pid` file;
- a `docker ps` container name contains `worker` (other than the Ollama one);
- the NAS (`--nas`, default `$PHOTOSEARCH_NAS_URL` or `http://dxp4800-f976:8000`)
  reports active claims on `/api/worker/status?passes=clip`, **or cannot be
  reached** (the guard fails closed).

There is no override flag. The fleet shares LM Studio: an eval run during a
batch slows the batch and measures contention instead of the prompt.

## Suites (`asks.jsonl`, 37 items)

| suite | items | what it measures |
|---|---|---|
| `routing` | 14 | people, place, camera, media, mood, exclusion and quality routing into `search_photos` args (r01–r09 are `evals/bakeoff.py` p01–p09, re-checked against today's vocabulary) |
| `dates` | 4 | today / last month / an explicit year / no date when none was asked |
| `curation` | 5 | the book tools: chapters, daily highlights, one per year, scenes, `suggest_layout` |
| `counting` | 5 | `summarize` questions whose answer is checked against the replica DB (top year from SQL, Tokyo years, top camera, the New York ∩ France year) |
| `locked` | 3 | pinned filter-bar values survive a vague ask (`locked_filters`) |
| `followup` | 2 | a second turn with history keeps or swaps the right filter |
| `writes` | 2 | write tools are called as a dry run first (no `confirm`) |
| `honesty` | 2 | an unregistered person or an impossible date gets "none found", not other photos |

An item passes only if every one of its checks passes, the run had no error,
and it did **not** fall back to single-shot (the diet's goal is the full
loop). Each item also gets a partial score (checks passed / total) in the report.

### Check grammar

Items are data, graded by `checks.py` in the harness's own tree, so a
check fixed later re-scores old rows with `regrade`.

- `checks`: `{name, op, tool, arg?, value?, min?, n?}` over the trace. `tool` may
  be a list. `arg` may be a list ("birthday" in `query`, `category` or `keyword`).
  Ops: `called` (`min`), `not_called`, `count_ge`/`count_le` (`n`, optionally
  only calls where `arg` == `value`), `max_calls`, `first_tool`, `set_eq`,
  `set_has`, `set_disjoint`, `eq`, `in`, `contains`, `contains_any`, `present`,
  `date_between` (`[lo, hi]`). `not_contains` and `absent` need at least one
  call to the tool, so calling nothing never passes a negative check.
- Dates may use `{today}`, `{yesterday}`, `{week_ago}`, `{year}`, `{last_year}`,
  `{last_summer}`, `{month_start}`, `{last_month_start}`, `{last_month_end}`,
  resolved from the day the row was **run** (`ran_at`), not the day it is graded.
- A call to the consolidated `search(mode=…)` tool grades as the family tool
  that mode routes to (`chapters` → `group_into_chapters`, …), so offering one
  search tool is a legitimate diet move that the same items still score.
- `answer_expect` / `answer_any` / `answer_reject`: regexes on the final answer.
  `answer_sql`: a scalar from the replica DB (read-only) that must appear in the
  answer, with or without thousands separators.
- `results_min`; `results_sql`: SQL over the returned ids (`{ids}`) whose scalar
  must equal `expect`, usually a count of hits that break the ask (non-France
  hits for "France", off-camera hits for a pinned camera).
- `facts`: plain English for the judge, on the items whose answer checks are
  prone to phrasing false fails.

## Running

```bash
PY=./.venv/bin/python
# prompt component sizes for a ref — calls no model (safe while the fleet runs)
$PY evals/ask_eval/harness.py measure --ref HEAD [--consolidated] [--json]

# what would run, and whether the guard allows it; calls no model
$PY evals/ask_eval/harness.py run --config before --ref pre-ask-diet --dry-run

# baseline, then the trimmed prompt (one model loaded at a time)
$PY evals/ask_eval/harness.py run --config before --ref pre-ask-diet --repeat 3
$PY evals/ask_eval/harness.py run --config after  --ref HEAD --repeat 3
$PY evals/ask_eval/harness.py run --config after-gemma --ref HEAD --model google/gemma-4-12b-qat

# compare; then let a judge look at answer-only failures
$PY evals/ask_eval/harness.py report --configs before after [--failures]
$PY evals/ask_eval/harness.py regrade --configs before after --judge [--judge-model haiku]
```

`--ref` is a git ref, checked out as a detached worktree under `/tmp` and
removed by its exact path afterwards (never `git worktree prune`). `--ref
WORKTREE` runs the live checkout, uncommitted edits included, for iterating on a
trim; the row records `dirty`. Tag `pre-ask-diet` marks the tree before any trim.

Defaults: `--model qwen/qwen3.5-9b`, `--repeat 3`, `--lm http://localhost:1234/v1`,
`--reasoning-effort none` (as the replica and the fleet run), the replica DB
`./photo_index.db.local`, the household notes from `./.agent-hints` (what
`run-local-replica.sh` feeds the replica; `$PHOTOSEARCH_AGENT_HINTS` wins).
`--consolidated` sets `PHOTOSEARCH_CONSOLIDATED_SEARCH=1`. A full pass is 111
runs per model; at ~20–40 s per Ask that is roughly an hour.

## What a row records

Per run: the trace (tool calls and result summaries), returned ids, the answer,
error, single-shot fallback and its reason, wall / LLM / tool seconds, and per
LLM round the prompt and completion tokens, reasoning tokens and any error with
**LM Studio's response body** (the agent drops it). `first_prompt_tokens` is the
fixed block (system + tools + the ask), `max_prompt_tokens` the largest round.
`context_errors` lists rounds whose body mentions the context. `meta.json` holds,
per run, the ref and sha, the `measure` output for that tree, and LM Studio's
`/api/v0/models` (loaded state and `loaded_context_length` of every model).

The report's **headroom** column is `loaded context − largest prompt − 3,000`
(the agent's reply budget): what is left for the fleet's concurrent requests.

## Safety while it runs

`runner.py` runs inside the tree under test and wraps it:

- a write tool called with `confirm=true` is **not executed**; the call is
  recorded in `blocked_writes` and answered with an error. Dry runs execute (they
  write nothing). `PHOTOSEARCH_NAS_URL` is removed, so nothing proxies to the NAS.
- `rerank_photos` is stubbed (it loads a vision model and takes minutes, and the
  diet does not change it); `--allow-rerank` runs it for real.
- the Ask log goes to `results/<config>/ask-logs/`, the request log is off.

## Caveats

- The replica DB is swapped every night by `sync-replica.sh`. `meta.json` records
  its mtime; compare configs run against the same snapshot, or at least the same
  week. `d01-today` checks only the date args: the replica rarely holds today's
  photos.
- Temperature is 0, but LM Studio is still not deterministic across loads: use
  `--repeat 3` and compare rates.
- Prompt-token counts depend on the model's tokenizer; compare a model with
  itself. `measure` reports characters, and estimates tokens at 4 characters
  per token.
- The eval measures one Ask at a time. It does not reproduce fleet contention;
  the headroom column is the proxy for it.

`tests/test_ask_eval_harness.py` covers the item shapes, every check op, date
tokens, the fallback/error rules, the DB checks, the fleet guard (refuses and
starts nothing; re-checks per chunk; fails closed on an unreachable NAS), the
write block, the rerank stub, the LM Studio error body, and a scripted run_job
→ grade round trip. No model, LM Studio or NAS is called.

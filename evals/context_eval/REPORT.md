# Context diet — evaluation report (2026-10-19)

Does moving ~95% of the always-loaded context (CLAUDE.md, the photo-search SKILL.md body, the
auto-memory index) into path-scoped rules, `docs/notes/` and skill `reference/` files hurt Claude
Code's behaviour in this repo, and what does it save? Plan: `docs/plans/context-diet.md`.
Narrative handoff: `docs/HANDOFF-2026-10-19-context-diet.md`. Every test item, in plain English, with
per-item tables for each model: [SUITES.md](SUITES.md). Per-item reports (git-ignored):
`results/report.md` (Sonnet), `results/report-opus.md`, `results/report-haiku.md`.

**Bottom line.** Across Sonnet, Opus 5.5 and Haiku 4.5 the pass rate is unchanged (±3 of 231 runs)
while billed cost drops 60-71%, but cloud sessions get 6-34% *slower* wall-clock because answers
take extra lookup turns. On a local 12B model it is the reverse: 13× faster, because prompt
processing dominates there; its one real failure is a fact it had to look up and didn't.

## What changed

| File | Before | After |
|---|---|---|
| `CLAUDE.md` | 258,266 B | 7,803 B |
| `.claude/skills/photo-search/SKILL.md` | 147,932 B | 5,905 B (navigator; same trigger description) |
| `MEMORY.md` (auto-memory index) | 18,218 B, 62 lines | 6,719 B, 54 lines; 10 closed topics archived |
| `.claude/rules/*.md` (new, path-scoped) | — | 9 files, 5.5-7.1 KB each |

Every old section was copied verbatim to `docs/notes/` or `reference/` (168/168 headings,
`scripts/check_headings_preserved.py`); `tests/test_context_budget.py` keeps the caps.
Fixed tokens at session start (Opus, measured): ~142k → ~14k.

## Method

`harness.py` runs headless `claude -p` against four suites: **recall** (49 repo-fact questions),
**rules** (9 subsystem traps, after a forced Read of a subsystem file), **skills** (8 vague prompts
that should trigger the photo-search skill; pass = a Skill tool call), **tasks** (11 small coding
tasks in a throwaway worktree, graded on the diff). 3 reps each = 231 runs per config.
"before" = tag `pre-context-diet`, "after" = HEAD; Opus/Haiku/local runs use `--isolate` (every
suite in a worktree of `--ref`). Costs are what `claude -p` reports at API prices; the runs were
billed against a Max plan, so they are notional. Wall clock is each `claude -p` run's elapsed time;
totals are the per-run times summed (cloud configs ran 2 at a time, so elapsed was about half).
$ per second = notional cost ÷ wall clock (totals: total cost ÷ summed wall clock). Graders were fixed during the study (see
Insights 6) and all configs were re-scored with the final graders (`harness.py regrade`).

## Results

### Sonnet

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall (49 items) | 100% → 100% | 138k → 82k | $0.210 → $0.056 | 6.3 s → 10.0 s | $0.0332 → $0.0057 |
| rules (9) | 100% → 100% | 377k → 132k | $0.327 → $0.157 | 13.5 s → 14.2 s | $0.0243 → $0.0110 |
| skills (8) | 0% → 12% | 340k → 100k | $0.246 → $0.060 | 13.5 s → 12.5 s | $0.0182 → $0.0048 |
| tasks (11) | 100% → 100% | 649k → 302k | $0.569 → $0.169 | 26.1 s → 35.9 s | $0.0218 → $0.0047 |
| **total** | 207 → 210 of 231 | | **$64.43 → $19.52 (−70%)** | **41 → 56 min** (summed) | **$0.0260 → $0.0059** (4.4× less) |

### Opus 5.5

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall | 100% → 100% | 205k → 94k | $0.434 → $0.107 | 9.0 s → 12.1 s | $0.0480 → $0.0088 |
| rules | 100% → 100% | 392k → 160k | $0.519 → $0.239 | 13.7 s → 14.0 s | $0.0380 → $0.0171 |
| skills | 0% → 0% | 426k → 131k | $0.454 → $0.122 | 15.0 s → 21.5 s | $0.0303 → $0.0057 |
| tasks | 100% → 97% | 765k → 402k | $1.126 → $0.339 | 28.5 s → 37.2 s | $0.0395 → $0.0091 |
| **total** | 207 → 206 of 231 | | **$125.90 → $36.27 (−71%)** | **50 → 65 min** (summed) | **$0.0420 → $0.0093** (4.5× less) |

The one difference: a `task-register-pass` run hit the 25-turn cap while finding every place the
pass list is mirrored.

### Haiku 4.5

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall | 99% → 97% | 108k → 111k | $0.076 → $0.035 | 12.7 s → 15.0 s | $0.0060 → $0.0024 |
| rules | 100% → 96% | 280k → 99k | $0.151 → $0.064 | 17.9 s → 16.2 s | $0.0084 → $0.0039 |
| skills | 0% → 0% | 135k → 182k | $0.073 → $0.048 | 12.6 s → 19.8 s | $0.0058 → $0.0024 |
| tasks | 73% → 88% | 1,365k → 402k | $0.335 → $0.097 | 41.1 s → 33.8 s | $0.0081 → $0.0029 |
| **total** | 197 → 197 of 231 | | **$28.05 → $11.28 (−60%)** | **67 → 70 min** (summed) | **$0.0070 → $0.0027** (2.6× less) |

Haiku's pre-diet config never hit the prompt cache (~$0.18 every run), which inflates its "before"
cost; treat the −60% as an upper bound.

### Local: gemma-4-12b-qat via LM Studio (first 10 recall items, 1 rep)

| config | pass | wall per item (mean / median) | total wall | mean turns | tool calls |
|---|---|---|---|---|---|
| local-before | 10/10 | 874 s / 887 s | 146 min | 1.0 | none |
| local-after | 9/10 | 65 s / 33 s | 11 min | 1.5 | Grep/Read on 3 items |

Setup that was required: model loaded at ≥128k context (262k used, parallel 1), the keep-alive
proxy (`keepalive_proxy.py`; LM Studio sends no bytes during prefill and Claude Code drops the
stream at 5 min), and `CLAUDE_CODE_ATTRIBUTION_HEADER=0` (else the prefix cache never hits). The
harness sets the last two for any non-Anthropic `ANTHROPIC_BASE_URL`.

### Cross-model summary

| model | pass before → after | cost | summed wall clock before → after | $ per second before → after |
|---|---|---|---|---|
| Sonnet | 207 → 210 | −70% | 41 → 56 min (+34%) | $0.0260 → $0.0059 (4.4× less) |
| Opus 5.5 | 207 → 206 | −71% | 50 → 65 min (+30%) | $0.0420 → $0.0093 (4.5× less) |
| Haiku 4.5 | 197 → 197 | −60% | 67 → 70 min (+6%) | $0.0070 → $0.0027 (2.6× less) |
| gemma-4-12b (local, 10 items) | 10 → 9 | — (local) | 146 → 11 min (13× faster) | — (local) |

## Insights that generalize

1. **The fixed-context cut is model-independent; the accuracy effect is not.** The same files
   shrink the starting prompt ~90% for every model. What it costs in accuracy scales with how well
   the model *finds* things: Opus and Sonnet lost nothing; Haiku lost a few single reps on recall
   and rules; gemma-12b failed the one item whose answer lives only in a path-scoped rule.

2. **Preloaded context also hurts weaker models on long work.** Haiku's tasks went 73% → 88%:
   with the old 258 KB CLAUDE.md a task run carried ~1.4M input tokens, and the smaller context
   did better. The trade is "recall by lookup" (slightly worse for weak models) against "less noise
   during multi-step work" (better for weak models).

3. **More turns: much cheaper, but slower on cloud models.** Answers now take extra Grep/Read turns
   (Sonnet recall 1.0 → 3.4, tasks 6.2 → 12.1). Cost fell 60-71% because every turn re-sends a far
   smaller cached prefix, but each turn is a round trip, and cloud prefill of a cached 140k prompt
   is fast, so summed wall clock rose 34% (Sonnet), 30% (Opus), 6% (Haiku). Mean recall went
   6.3 s → 10.0 s on Sonnet. The diet trades seconds of latency for dollars and context headroom:
   per second of session, cost fell 2.6-4.5× overall ($0.026 → $0.006/s Sonnet, $0.042 → $0.009/s Opus,
   $0.007 → $0.003/s Haiku), because each turn now carries a far smaller prompt.
   One Opus task run hit the 25-turn cap, so lookup-heavy tasks need headroom.

4. **For local models the win is wall-clock, and prefill dominates.** Prompt processing slows as the
   prompt grows (~1,200 tok/s at the start, ~110 tok/s by mid-prompt), so a 100k-token prompt is
   ~15 min before the first token, and gemma's sliding-window attention prevents reusing the
   cache across items. A ~12k prompt is processed in seconds.

5. **Path-scoped rules only fire when code is opened.** Q&A and ops sessions never open files, so
   the CLAUDE.md index is the only route to a rule. Strong models follow "see rules/workers.md";
   gemma ran one Grep, missed, and confabulated. The Skill tool was essentially never invoked by
   any model, before or after (0-12%), but after the diet 7/8 (Sonnet) and 19/24 (Opus) skill
   answers cited the right `reference/` file: models route through CLAUDE.md pointers, not the
   skill mechanism. (Follow-up: a duplicate claude.ai skill was the likely cause; see below.)

6. **Regex graders produce false failures on correct paraphrases.** Over the study, correct answers
   were failed by a stale constant (the grader expected `SCHEMA_VERSION = 39`, written when the
   repo was at v38; the correct bump is now 39 → 40), by
   "don't do X" checks matching the model's own comment explaining why it avoided X, and by
   literal-phrase lists missing a synonym ("own process", "new file", `{score_col} IS ?`). Each
   was verified by reading the answer. Storing answers and added lines and re-scoring
   (`harness.py regrade`) made this cheap to correct without re-running.

7. **Eval hygiene matters as much as the change under test:**
   - The cwd is part of Claude Code's system prompt: one worktree per run defeated the prompt
     cache ($0.93 vs $0.04 per Opus recall run); share one per config.
   - The project's `.claude/settings.local.json` allowlist reaches eval worktrees. Combined with a
     "commit and push by default" preference, Haiku task runs pushed 12 commits to `origin/main`
     (reverted `f759330`). Every eval run now passes `--disallowedTools` for git writes and `gh`,
     and a task that moves HEAD fails.
   - Plan session limits arrive as ordinary text answers; the harness classes them as infra, stops
     after 3 in a row, and `--resume` continues without re-paying.
   - Gateway failure markers must only apply to failed runs; a correct answer can say
     "connection refused".

## Further suggestions

1. **Inline the highest-traffic operational facts as one-liners in CLAUDE.md** *(Acted on 2026-10-20: see Follow-up.)*, not just pointers.
   This also cuts the extra lookup turns that made cloud sessions slower.
   The gemma miss (`-d /photos` → 404, omit `-d`) and Haiku's `dubious ownership` miss were both
   facts a session can need without opening code. Candidates: the `-d` gotcha, `/api/health` probe,
   reasoning-effort for thinking models. Keep the budget test as the brake.
2. **Make the subsystem index imperative** *(Acted on 2026-10-20: see Follow-up.)*: "Before answering a question about workers, Read
   `.claude/rules/workers.md`" helps weaker models more than a topic list.
3. **Hook rules to Bash/Grep** *(Acted on 2026-10-20: see Follow-up.)*: a `PreToolUse` hook that injects the matching rule when a command
   names a subsystem (`run-workers.sh`, `maintenance-sweep`) would close the "rules only load on
   file open" gap for ops sessions.
4. **Decide the skill's role.** *(Acted on 2026-10-20: see Follow-up.)* It is almost never invoked. Either rewrite its description around
   concrete triggers and re-measure the skills suite, or drop the suite's Skill-call criterion and
   grade "cited/used the right reference file" instead. Also remove or update the claude.ai-synced
   duplicate (`anthropic-skills:photo-search`), which still carries the old 148 KB body.
5. **Extend the local run** to all 49 recall items plus rules (now ~1 min/item, so ~1 h), and try
   one non-sliding-window model so cache reuse across items can be measured.
6. **Investigate Haiku's pre-diet cache misses** before quoting its cost delta.
7. **Raise `--max-turns` for tasks** *(Acted on 2026-10-20: see Follow-up.)* (25 → 40) or report turn-cap hits separately; lookup-heavy
   tasks are where the diet adds turns.
8. **Replace brittle regexes with a rubric grader** *(Acted on 2026-10-20: see Follow-up.)* (an LLM judge with the expected facts) for
   recall/rules, keeping regex as a cheap pre-filter; at minimum keep re-reading every failure.
9. **Keep the guardrails**: `tests/test_context_budget.py`, the "Where knowledge goes" section,
   and the eval's git-write block for any new headless automation in this repo.
10. **Plan section 8 levers still open**: move the `photosearch` MCP server from user to project
    scope (~2-3k tokens per session), and the optional PostToolUse budget hook.
11. **Public-repo hygiene** (unrelated to the diet, found while checking): the repo already
    publishes children's names, another family's child's full name
    (`docs/HANDOFF-2026-09-18-rank-shoot.md`), the Tailscale IP and the NAS account name.

## Follow-up (2026-10-20): suggestions 1-4, 7, 8 acted on

Same-day before/after pairs on this Mac. "Old" = `c98beac` (the report above), "new" = the
commits named below. Both arms of each pair ran on the same CLI version (Haiku/Sonnet on
2.1.226, Opus on 2.1.304, which Opus 5.5 requires). Results stay in `results/<config>/`.

**Confound found first.** In the runs above, every skills session was offered two skills named
photo-search: the repo's and the claude.ai-synced `anthropic-skills:photo-search` (old 148 KB
body). It was disabled on 2026-10-20 before these runs (every row records only `photo-search`
offered). With it gone, the *old* description already triggered 81% on Haiku and Sonnet, against
0-12% above, so the duplicate is the likely cause of the near-zero skill use reported in
Insight 5. That was not measured directly (it would need the duplicate re-enabled).

**Suggestion 4: skill description rewritten around concrete triggers** (`a53f15a`). It now
opens "Invoke it FIRST, before Grep or Read" and lists the concrete request types (indexing,
slow search / request log, worker fleet, faces, Google upload, NAS/Docker/ingest/replica,
schema and tables, adding a filter/command/endpoint/pass/page, MCP/Ask, evals). The skills suite
gained 10 held-out prompts, written by a separate model from the `reference/` headings only,
without seeing SKILL.md or the 8 original prompts, so a tuned description can't overfit unseen.

| model | 8 original (24 runs) old → new | 10 held-out (30 runs) old → new | mean cost old → new | mean wall old → new |
|---|---|---|---|---|
| Haiku 4.5 | 20 → 22 | 24 → 29 | $0.050 → $0.053 | 16.2 s → 14.8 s |
| Sonnet | 18 → 22 | 26 → 30 | $0.153 → $0.148 | 18.9 s → 17.2 s |
| Opus 5.5 | — → 21 (old not re-run) | **0 → 23** | $0.136 → $0.159 | 17.1 s → 15.9 s |

The gains sit where the new text names something the old one did not: "what tables are in
photo_index.db?" (Sonnet 0 → 3) and "which table holds the face encodings?" (Haiku 0 → 2,
Sonnet 0 → 3). Opus still never invokes the skill for those two table questions (CLAUDE.md
already points at `reference/schema.md`), nor for "which endpoints handle stacks and
collections?". With the old description Opus invoked the skill on none of the held-out prompts.

**Suggestions 1-3: ops facts inlined, imperative subsystem index, rule-injection hook**
(`65e2558`, `a07cc0d`, merged at `18260ff`). CLAUDE.md grows 7.8 → 10.3 KB. The hook
(`scripts/hooks/inject_rules.py`, registered in the now-tracked `.claude/settings.json`) adds a
rule file's body once per session when a Bash/Grep/Glob call names its subsystem. A real Haiku
session that ran only `ls run-workers.sh` then quoted a `valid_passes` rule that exists only in
`.claude/rules/workers.md`, so the injection reaches the model. Haiku, old → new, judged:

| suite | pass | mean cost | mean wall | mean turns |
|---|---|---|---|---|
| rules (9 items × 3) | 24 → 26 of 27 | $0.068 → $0.076 | 11.1 s → 11.8 s | 2.5 → 2.4 |
| recall (49 items × 1) | 45 → 43 of 49 | $0.061 → $0.059 | 15.7 s → 15.0 s | 4.2 → 3.4 |

Recall at one rep is within noise (5 items flipped down and 2 up, different items each way).
The new tree also carries the new skill description: recall runs called the skill on 20 of
49 items, against 3. Not run: Sonnet/Opus recall and rules, and tasks.

**Suggestions 7-8: harness** (`60cbb8d`). `tasks` now default to 40 turns (read-only suites
keep 25), and `report` counts turn-cap hits as their own column. `regrade --judge` sends only
regex FAILs on recall/rules to a Haiku judge with the item's `facts` (added to 32 items) or a
rubric rendered from its regexes. A regex pass is never judged. On the four Haiku configs
above it made 15 calls ($0.013) and overturned 1 failure.

**Not done:** 5 (the local run, paused as asked), 6 (needs the original Haiku result files,
which are not on this Mac), 10 and 11 (owner decisions). Ad-hoc headless runs here also need
`env -i` with `USER`/`LOGNAME` and a quoted `PATH` (it contains a space), and a Mac with the
lid closed sleeps even under `caffeinate -i`.

## Reproduce

```bash
# cloud: one config per ref, same items, 3 reps
python3 evals/context_eval/harness.py run --config opus-before --ref pre-context-diet --isolate --model claude-opus-5-5 --repeat 3 --parallel 2
python3 evals/context_eval/harness.py run --config opus-after  --ref HEAD             --isolate --model claude-opus-5-5 --repeat 3 --parallel 2
# local (LM Studio at :1234, model loaded ≥128k context)
ANTHROPIC_BASE_URL=http://localhost:1234 ANTHROPIC_AUTH_TOKEN=lmstudio \
  python3 evals/context_eval/harness.py run --config local-after --ref HEAD --isolate \
  --model google/gemma-4-12b-qat --suite recall --limit 10 --repeat 1 --parallel 1 --timeout 1800
python3 evals/context_eval/harness.py regrade --configs opus-before opus-after
python3 evals/context_eval/harness.py report  --configs opus-before opus-after --failures
```

Run from a clean environment (`env -i HOME=$HOME PATH=$PATH …`) when launching from inside a Claude
session, and use `--resume` after an interruption.

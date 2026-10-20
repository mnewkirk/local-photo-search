# Handoff 2026-10-19: context diet

Plan: `docs/plans/context-diet.md` (steps 0-8 executed). Rollback point: tag `pre-context-diet`;
file copies in `~/context-diet-backup-2026-10-19/` (CLAUDE.md, the skill, the whole memory dir).

## What moved

| File | Before | After |
|---|---|---|
| `CLAUDE.md` | 258,266 B | 7,803 B |
| `.claude/skills/photo-search/SKILL.md` | 147,932 B | 5,905 B (navigator; same trigger description) |
| `MEMORY.md` (auto-memory index) | 18,218 B, 62 lines | 6,719 B, 54 lines; 10 closed topics in `memory/archive/` |
| `.claude/rules/*.md` (new, path-scoped) | — | 9 files, 5.5-7.1 KB each |

Every old section was copied verbatim to `docs/notes/<subsystem>.md` (history) or the skill's
`reference/<topic>.md` (runbooks); `scripts/check_headings_preserved.py` confirms 168/168 headings.
Where the old prose had drifted from the code, the rules follow the code and the notes carry a
dated "Corrections found while condensing" heading. `tests/test_context_budget.py` holds the caps.

## Measured (evals/context_eval, Sonnet, 3 reps, 231 runs per config)

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall (49 items) | 100% → 100% | 138k → 82k | $0.210 → $0.056 | 6.3 s → 10.0 s | $0.0332 → $0.0057 |
| rules (9) | 100% → 100% | 377k → 132k | $0.327 → $0.157 | 13.5 s → 14.2 s | $0.0243 → $0.0110 |
| skills (8) | 0% → 12% | 340k → 100k | $0.246 → $0.060 | 13.5 s → 12.5 s | $0.0182 → $0.0048 |
| tasks (11) | 100% → 100% | 649k → 302k | $0.569 → $0.169 | 26.1 s → 35.9 s | $0.0218 → $0.0047 |
| **total** | 207 → 210 of 231 | | **$64.43 → $19.52 (−70%)** | **41 → 56 min** (summed) | **$0.0260 → $0.0059** (4.4× less) |

Costs are notional (Max plan). Turns rose (recall 1.0 → 3.4, tasks 6.2 → 12.1): sessions now
read a rule or reference file instead of having everything preloaded, and it is still far cheaper.

Skills: Sonnet rarely calls the Skill tool, before or after, but after the diet 7 of 8 skill prompts
cite the right `reference/*.md`, reached through CLAUDE.md's pointers. The runbooks are reachable.

One real regression was found and fixed: the workers rule had dropped the 404 symptom of
`-d /photos`. Grader fixes along the way (schema-bump pinned to v39, comments counted as code,
over-literal regexes) are in `evals/context_eval/`.

## Repeated on Opus 5.5 (`opus-before` = tag `pre-context-diet`, `opus-after` = HEAD; `--isolate`)

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall | 100% → 100% | 205k → 94k | $0.434 → $0.107 | 9.0 s → 12.1 s | $0.0480 → $0.0088 |
| rules | 100% → 100% | 392k → 160k | $0.519 → $0.239 | 13.7 s → 14.0 s | $0.0380 → $0.0171 |
| skills | 0% → 0% | 426k → 131k | $0.454 → $0.122 | 15.0 s → 21.5 s | $0.0303 → $0.0057 |
| tasks | 100% → 97% | 765k → 402k | $1.126 → $0.339 | 28.5 s → 37.2 s | $0.0395 → $0.0091 |
| **total** | 207 → 206 of 231 | | **$125.90 → $36.27 (−71%)** | **50 → 65 min** (summed) | **$0.0420 → $0.0093** (4.5× less) |

The single difference: one `task-register-pass` run hit the 25-turn cap while finding every place
the pass list is mirrored (post-diet sessions take more turns: tasks 7.7 → 11.9). Opus never calls
the Skill tool in either config, but post-diet 19 of 24 skills answers cite the right
`reference/` or rule file (0 of 24 pre-diet, which answered from the preloaded CLAUDE.md).

Two more over-literal graders surfaced (both also failed pre-diet runs) and were fixed;
`harness.py regrade` re-scores stored rows without re-running, and all four configs were
re-scored with the final graders. Sonnet totals above are unchanged by it.

## Repeated on Haiku 4.5 (`haiku-before` / `haiku-after`, `--isolate`)

| suite | pass before → after | mean input tok before → after | mean cost before → after | mean wall per run before → after | $ per second before → after |
|---|---|---|---|---|---|
| recall | 99% → 97% | 108k → 111k | $0.076 → $0.035 | 12.7 s → 15.0 s | $0.0060 → $0.0024 |
| rules | 100% → 96% | 280k → 99k | $0.151 → $0.064 | 17.9 s → 16.2 s | $0.0084 → $0.0039 |
| skills | 0% → 0% | 135k → 182k | $0.073 → $0.048 | 12.6 s → 19.8 s | $0.0058 → $0.0024 |
| tasks | 73% → 88% | 1,365k → 402k | $0.335 → $0.097 | 41.1 s → 33.8 s | $0.0081 → $0.0029 |
| **total** | 197 → 197 of 231 | | **$28.05 → $11.28 (−60%)** | **67 → 70 min** (summed) | **$0.0070 → $0.0027** (2.6× less) |

Haiku is where the trade-off shows: recall/rules lose a few single reps (it has to find the
answer in a rule or reference file: recall turns 1.1 → 4.3), tasks gain clearly (73% → 88%;
a 1.4M-token task context hurt it). Net equal.

**Incident: eval task runs pushed to main.** The project's `.claude/settings.local.json`
allowlists `git commit`/`git push`, which reaches the eval worktrees, and the owner's standing
"commit and push" preference made Haiku commit and push 12 task answers (incl. a schema v40
bump) to `origin/main`, 18:55-19:10 PT. Reverted in `f759330` (NAS was at `dced247`, never
pulled). The harness now passes `--disallowedTools` for git commit/push/reset/rebase/merge/tag
and gh (tested against a bare remote with an allowlist), and fails a task whose worktree HEAD
moved. The Haiku task suites were discarded and re-run with the guard.

## Local model (gemma-4-12b-qat via LM Studio)

`local-before`: 10/10 recall, but ~15 min per item: the ~100k-token prompt is processed from scratch
each time (gemma's sliding-window cache can't reuse across items). Needs ≥128k context, the
keep-alive proxy (`evals/context_eval/keepalive_proxy.py`) and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`,
which the harness sets. `local-after` (same 10 items, `--isolate`): **9/10 in 11 min total vs 10/10 in
146 min** (median 33 s vs 887 s per item). Gemma does call Grep/Read through LM Studio; its one miss
(`worker-d-photo-root`) is a fact that lives only in the path-scoped workers rule, which it didn't find.
Full write-up with all tables, insights and suggestions: `evals/context_eval/REPORT.md`.

## Open

- The claude.ai-synced copy of the skill (`anthropic-skills:photo-search`, from the owner's claude.ai
  account) still carries the old 148 KB body. Sessions see two `photo-search` skills; removing or
  updating the synced copy is an owner action in claude.ai.
- Optional `PostToolUse` hook (plan 5d) not added.
- `photosearch` MCP server is user-scope (~2-3k tokens per session); plan section 8 lever.

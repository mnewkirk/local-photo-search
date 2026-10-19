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

| suite | pass before → after | mean input tok before → after | mean cost before → after |
|---|---|---|---|
| recall (49 items) | 100% → 100% | 138k → 82k | $0.210 → $0.056 |
| rules (9) | 100% → 100% | 377k → 132k | $0.327 → $0.157 |
| skills (8) | 0% → 12% | 340k → 100k | $0.246 → $0.060 |
| tasks (11) | 100% → 100% | 649k → 302k | $0.569 → $0.169 |
| **total** | 207 → 210 of 231 | | **$64.43 → $19.52 (−70%)** |

Costs are notional (Max plan). Turns rose (recall 1.0 → 3.4, tasks 6.2 → 12.1): sessions now
read a rule or reference file instead of having everything preloaded, and it is still far cheaper.

Skills: Sonnet rarely calls the Skill tool, before or after, but after the diet 7 of 8 skill prompts
cite the right `reference/*.md`, reached through CLAUDE.md's pointers. The runbooks are reachable.

One real regression was found and fixed: the workers rule had dropped the 404 symptom of
`-d /photos`. Grader fixes along the way (schema-bump pinned to v39, comments counted as code,
over-literal regexes) are in `evals/context_eval/`.

## Local model (gemma-4-12b-qat via LM Studio)

`local-before`: 10/10 recall, but ~15 min per item: the ~100k-token prompt is processed from scratch
each time (gemma's sliding-window cache can't reuse across items). Needs ≥128k context, the
keep-alive proxy (`evals/context_eval/keepalive_proxy.py`) and `CLAUDE_CODE_ATTRIBUTION_HEADER=0`,
which the harness sets. A `local-after` run would show the prompt-size win directly; not run yet.

## Open

- The claude.ai-synced copy of the skill (`anthropic-skills:photo-search`, from the owner's claude.ai
  account) still carries the old 148 KB body. Sessions see two `photo-search` skills; removing or
  updating the synced copy is an owner action in claude.ai.
- Optional `PostToolUse` hook (plan 5d) not added.
- `photosearch` MCP server is user-scope (~2-3k tokens per session); plan section 8 lever.

# Context eval: does a trimmed CLAUDE.md still steer Claude Code?

`docs/plans/context-diet.md` moves ~90% of CLAUDE.md and SKILL.md out of the
per-session context. Its own checks prove the text was *relocated*. This
harness checks that a fresh Claude Code session can still *find* the
knowledge and still *obeys* it, and records what each run cost, so the
before/after is judged on behaviour and billed tokens rather than file sizes.

It drives Claude Code headless (`claude -p --output-format stream-json`), so
runs are billed to the account running it. Nothing here is mocked.

## Suites

| suite | items | what it measures | graded by |
|---|---|---|---|
| `recall` | `recall.jsonl`, 50 | can a cold session answer one trap question per relocated section | regexes on the answer (`expect` all / `any` groups / `reject`) |
| `rules` | `rule_triggers.jsonl`, 9 | after a `Read` of a subsystem file, does the subsystem's trap knowledge load (the path-scoped `.claude/rules/*.md` once they exist) | same |
| `skills` | `skill_triggers.jsonl`, 8 | do the vague prompts in the skill description still trigger the photo-search skill | a `Skill` tool call naming photo-search |
| `tasks` | `tasks.jsonl`, 11 | small coding tasks where a documented trap bites; the real adherence test | deterministic checks on the diff's added lines in a throwaway worktree |

Every run also records `total_cost_usd`, token usage (input, cache creation,
cache read), turns, wall time, the model that answered, the git sha and the
byte sizes of CLAUDE.md / SKILL.md / rules files in the tree it ran against.

## Running

```bash
# list what would run, call nothing
python evals/context_eval/harness.py run --config before --dry-run

# baseline against the current CLAUDE.md (read-only suites run in the repo root)
python evals/context_eval/harness.py run --config before --repeat 3 --model sonnet --parallel 2

# one suite, a few items, once
python evals/context_eval/harness.py run --config smoke --suite recall --only cli-db-envvar --repeat 1

# after the restructure is COMMITTED (tasks run in a worktree of --ref, default HEAD)
python evals/context_eval/harness.py run --config after --repeat 3 --model sonnet --parallel 2

# compare
python evals/context_eval/harness.py report --configs before after [--failures] > evals/context_eval/results/report.md
```

Defaults: `--model sonnet`, `--repeat 3`, `--parallel 2`, `--max-turns 25`,
`--budget 1.50` per run, `--timeout 900` s. Results append to
`results/<config>/<suite>.jsonl` (git-ignored); `report` aggregates whatever
is there, so re-running a subset adds samples.

Rough cost: the read-only suites are one or two turns each; `tasks` are
5–20 turns. A full `--repeat 3` pass is ~235 runs. With the current 62k-token
CLAUDE.md every run pays that block (as a cache write once, then cache reads),
which is itself part of what the eval measures.

## Reading the report

- **Per suite**: pass rate, mean cost, mean input tokens, mean cache-read tokens,
  mean turns. The `after` config should match or beat `before` on pass rate and
  come in well under on input tokens.
- **Per item**: pass count per config with `REGRESSED` / `improved` flags. A
  regression on a `recall` item usually means the pointer that replaced the
  section is not findable; on a `tasks` item it means a rule that used to be
  in CLAUDE.md is now only path-scoped and the task never opened a matching
  file. Fix the pointer or the rule's `paths:`, not the eval.
- `--failures` prints each failing run's check detail and the first 800
  characters of its answer.

## Caveats

- Regex grading is lenient on purpose and still produces some false fails on
  phrasing. Compare pass *rates between configs*, and read `--failures` before
  acting on a single item.
- `tasks` graders check for the convention, not for working code. A task can
  pass with a broken implementation; the question is only whether the trap was
  avoided.
- Read-only suites run in the repo root so the real memory directory loads;
  `tasks` run in a `/tmp` worktree with no project memory. Both configs see
  the same treatment, so the comparison holds, but absolute numbers differ
  between suites.
- `skills` can legitimately read 0% on the `before` config: in the two smoke
  runs the `Skill` tool and both photo-search skills were offered, and the
  model still answered straight from the 62k-token CLAUDE.md without invoking
  one. Each row records `skill_tool_offered` and `photo_skills_offered` so
  "not offered" and "not triggered" stay distinguishable. After the diet the
  skill is the only route to the runbooks, so this suite matters for `after`.
- Headless sessions do not share a prompt cache with each other: the smoke
  runs each paid ~119k cache-creation tokens seconds apart (~$0.45 on sonnet
  before any work). That fixed block is the thing the diet removes, and it is
  why the baseline costs roughly $0.50 per run.
- `--strict-mcp-config` disables MCP servers so the photosearch MCP tools can
  not answer recall questions for the model.
- Permissions: read-only suites allow Read/Grep/Glob/Skill and read-only Bash
  patterns; `tasks` add Edit/Write under `--permission-mode acceptEdits`. Any
  other tool call is denied and the model adapts.
- Long-session compaction behaviour is out of scope here.

`tests/test_context_eval_harness.py` covers the grading and transcript parsing
without calling Claude.

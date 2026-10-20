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
| `recall` | `recall.jsonl`, 49 | can a cold session answer one trap question per relocated section | regexes on the answer (`expect` all / `any` groups / `reject`); optional LLM judge on regex fails (`facts`) |
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

Defaults: `--model sonnet`, `--repeat 3`, `--parallel 2`, `--max-turns` per
suite (40 for `tasks`, 25 for the read-only suites; an explicit `--max-turns N`
overrides every suite), `--budget 1.50` per run, `--timeout 900` s. Each row
records the cap it ran under (`max_turns`). Results append to
`results/<config>/<suite>.jsonl` (git-ignored); `report` aggregates whatever
is there, so re-running a subset adds samples.

Rough cost: the read-only suites are one or two turns each; `tasks` are
5–20 turns. A full `--repeat 3` pass is ~235 runs. With the current 62k-token
CLAUDE.md every run pays that block (as a cache write once, then cache reads),
which is itself part of what the eval measures.

## Reading the report

- **Per suite**: pass rate, mean cost, mean input tokens, mean cache-read tokens,
  mean turns, turn-cap hits. The `after` config should match or beat `before` on
  pass rate and come in well under on input tokens.
- **Turn-cap hits**: runs that ended on `--max-turns` (`subtype`
  `error_max_turns`). They are still FAILs in the pass rate, but counted in their
  own column, tagged `[N turn-cap]` in the per-item table, and listed under
  "Turn-cap hits" with `--failures`, so "the context lost a rule" and "the task
  needed more turns than the cap" stay distinguishable.
- **Judged runs**: once any row carries a judge verdict (below), the per-suite
  table shows the judged pass rate, the regex-only pass rate and the number of
  judged rows; per-item cells add `(regex p/n)` where the judge changed the
  count.
- **Per item**: pass count per config with `REGRESSED` / `improved` flags. A
  regression on a `recall` item usually means the pointer that replaced the
  section is not findable; on a `tasks` item it means a rule that used to be
  in CLAUDE.md is now only path-scoped and the task never opened a matching
  file. Fix the pointer or the rule's `paths:`, not the eval.
- `--failures` prints each failing run's check detail and the first 800
  characters of its answer.

## Regrading and the LLM judge

```bash
# regex only: re-score stored answers / added lines under the current graders
python evals/context_eval/harness.py regrade --configs before after

# plus the judge on recall/rules rows the regexes fail
python evals/context_eval/harness.py regrade --configs before after --judge [--judge-model haiku] [--rejudge]
```

`regrade` never re-runs the model under test. With `--judge`, only `recall` and
`rules` rows are eligible, and only those the regexes **fail**: a regex PASS is
never sent to the judge, and a judge never turns a pass into a fail. The judge
may overturn a regex fail.

- **Call**: `claude -p --model haiku --max-turns 1 --output-format json
  --tools ""` (no tools, skills or MCP, replaced system prompt, scratch cwd
  outside the repo so no CLAUDE.md loads), with the same cleaned environment
  and git-write/`gh` `--disallowedTools` block as the runs. It gets the
  question, the expected facts and the stored answer, and must reply with
  strict JSON `{"pass": bool, "reason": str}`.
- **Expected facts**: the item's optional `facts` string (plain English, written
  for the judge); otherwise the `expect` / `any` / `reject` regexes rendered as
  readable text. `facts` is set on the items most prone to paraphrase false
  fails (synonym lists, thresholds, "don't do X" explanations).
- **Recorded on the row**: `detail.regex_pass` (the regex verdict, always),
  `detail.judge` = `{pass, reason, model, cost_usd}` when the judge gave a usable
  verdict, and `pass` = the judge's verdict. The judge's answer is JSON only: a
  malformed reply, an auth/login failure, a non-zero exit or a timeout records
  `detail.judge_error` instead, keeps the regex verdict, and is never retried.
  After 3 judge errors in a row the run stops calling the judge.
- **Idempotent**: a stored judge verdict is reused by later `regrade` runs (with
  or without `--judge`) while the regex still fails, so paid verdicts are not
  lost or re-bought; `--rejudge` forces new calls.
- **Cost**: one haiku call per regex failure (a few thousand tokens, well under
  a cent), printed as a total at the end. Rows with `error`/`infra_error`/a
  turn-cap are skipped.

## Caveats

- Regex grading is lenient on purpose and still produces some false fails on
  phrasing. Compare pass *rates between configs*, read `--failures` before
  acting on a single item, and use `regrade --judge` to see how many fails are
  phrasing only. The judge is itself a model: report both numbers.
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

`tests/test_context_eval_harness.py` covers the grading, transcript parsing,
per-suite turn caps, turn-cap counting and the judge (prompt, JSON parsing,
overturn, failure handling) without calling Claude; an autouse fixture fails any
test that tries to start the real `claude` CLI.

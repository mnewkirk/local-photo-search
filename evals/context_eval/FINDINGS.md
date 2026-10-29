# Context eval: findings (start here)

Status as of 2026-10-28. This file sums up what the CLAUDE.md / context eval
has measured so far, what the results mean and how far to trust them. The
other files each have one job:

| file | job |
|---|---|
| [README.md](README.md) | how the harness works and how to run it |
| [SUITES.md](SUITES.md) | every test item in plain English, with per-item tables for the 2026-10-19 before/after runs |
| [REPORT.md](REPORT.md) | the dated lab notebook: the 2026-10-19 study, the 2026-10-20 follow-ups, the 2026-10-25 grader audit, the 2026-10-28 ablation table |
| [ABLATIONS.md](ABLATIONS.md) | the 2026-10-28 write-up covering both evals (context and Ask) |
| [ABLATION_LAYERS.md](ABLATION_LAYERS.md) | exactly which files each ablation deletes, and what stays |

Every number below comes from the stored rows in `results/<config>/*.jsonl`.
Those rows are git-ignored but present on the owner's desktop. The source is
named next to each table. A regrade of a scratch copy of all stored rows with
the current graders (`harness.regrade`, no judge, 2026-10-28) changed 0 rows,
so the stored verdicts match the graders in the tree today.

## The question

In 2026-10, the project cut the context Claude Code loads at session start by
about 90%. CLAUDE.md went from 258 KB to 8 KB (10.5 KB today). The detail moved
into three places: path-scoped rules (`.claude/rules/`), history notes
(`docs/notes/`) and skill runbooks (`.claude/skills/photo-search/reference/`).
File-size checks prove the text moved. They cannot show that a new session
still **finds** the knowledge and **obeys** it. This eval tests that, and
records what each session costs.

## Method in plain terms

The harness starts a fresh headless Claude Code session (`claude -p`) for
every item. MCP servers are off, so the photo-library tools cannot answer for
the model. A **run** is one session on one item. A **config** is one set of
runs: one model, one git ref, optionally one ablation. Four suites:

| suite | items | a run passes when |
|---|--:|---|
| recall | 49 | the answer to a trap question ("what happens with `-d /photos`?") matches the item's regexes |
| rules | 9 | same, after the model is told to Read a subsystem file first (this is what loads a path-scoped rule) |
| skills | 8 (+10 held-out since 2026-10-20) | the model invokes the photo-search skill on a vague prompt |
| tasks | 11 | a small code change in a throwaway worktree avoids a documented trap (checked on the added lines) |

A **turn** is one model call inside a session. Costs are what `claude -p`
reports at API prices. The runs were billed to a Max plan, so dollar figures
are notional but comparable. **Isolated** runs (`--isolate`) run every suite in
a `/tmp` worktree with no project memory. **Repo-root** runs answer the
read-only suites in the real checkout, where the owner's auto-memory also
loads.

## 1. The initial before/after (2026-10-19)

"Before" is tag `pre-context-diet` (258 KB CLAUDE.md). "After" is the slim
tree at that time. 77 items × 3 reps = 231 runs per config.

| model (configs) | pass before → after | turns/run | wall s/run | $/run |
|---|---|---|---|---|
| Sonnet 5.5, repo-root (`before`, `after`) | 207 → 210 of 231 | 2.2 → 4.7 | 10.7 → 14.4 | 0.279 → 0.085 |
| Opus 5.5, isolated (`opus-before`, `opus-after`) | 207 → 206 | 2.8 → 4.9 | 13.0 → 16.9 | 0.545 → 0.157 |
| Haiku 4.5, isolated (`haiku-before`, `haiku-after`) | 197 of 230 → 197 of 229 scored | 2.8 → 5.0 | 17.3 → 18.3 | 0.122 → 0.049 |
| gemma-4-12b, local, first 10 recall items × 1 (`local-*`) | 10 → 9 of 10 | 1.0 → 1.5 | 874 → 65 | — |

Source: `results/{before,after,opus-*,haiku-*,local-*}`. Means are over
scored runs. Infra rows (session limits) are unscored: 1 in `haiku-before`, 2
in `haiku-after`.

**What it established.** The slim context cut billed cost by 60-71% on every
cloud model. Overall pass counts moved by at most 3 of 231 runs. Sessions took
about twice as many turns, so cloud wall time per run rose (Sonnet +35%, Opus
+30%, Haiku +6%). On the local model it was the reverse: 13× faster, because
prompt processing dominates there.

**What it did not establish.**

- *That nothing was lost.* Sonnet and Opus passed every recall, rules and
  tasks run in **both** arms. A suite at 100% before and after cannot detect a
  loss. It shows only that nothing broke badly.
- *Anything about the skill.* Skills read 0-12% in every arm. A duplicate
  claude.ai copy of the skill was offered alongside the repo's. Once it was
  disabled (2026-10-20), the old description triggered 81% on Haiku and Sonnet.
  So these skills numbers measure the confound, not the diet (REPORT.md,
  "Follow-up").
- *The Haiku task gain (24 → 29 of 33).* See the next section.

## 2. What the grader audit changed (2026-10-25)

The graders were found to be **too lenient**. Hand-written wrong answers
("yes, it burns an attempt", "an empty list is a no-op") passed 51 of 114
recall/rules checks. 6 of 8 deliberately broken task diffs passed. After the
fix, both are 0. Every item now carries `counter` answers that its regexes must
fail and a real `pass_example` they must pass, and the unit tests check both.
Details are in REPORT.md, "Grader audit".

| earlier conclusion | after the audit |
|---|---|
| Sonnet and Opus before/after pass rates | **confirmed.** No stored PASS flipped to FAIL under the strict graders, so no headline number was inflated. |
| Haiku recall 99% → 97% | **corrected** to 100% → 98% (146/146 → 142/145). Three "failures" were session-limit messages. Fisher p = 0.12; the sign test over items is 3 down, 0 up (p = 0.25). Not significant. |
| 7 Haiku items and Opus `task-register-pass` "REGRESSED" | **withdrawn.** All are single-rep flips, two were infra rows, and Opus's run hit the turn cap with every check passing. Under the new rule (flag only on a drop of at least 2 of 3 reps), no item regressed for any model. |
| Haiku tasks 73% → 88%: "less preloaded context helps weak models on long work" | **weakened to unproven.** 5 of the 15 failed task runs passed every check but hit the turn cap. On the checks alone it is 85% → 91% (28 → 30 of 33). The scored change has Fisher p = 0.21. |

The lesson for future reading: before 2026-10-25, a PASS meant only "did not
contradict a lenient regex". Since then it means the answer states the fact,
and an answer that asserts the opposite fails.

## 3. Ablations across Haiku, Sonnet and Opus (2026-10-28)

**Why.** Section 1 hit a ceiling. The ablations ask a sharper question: which
layer is actually *needed*? Each config deletes one layer in an isolated
worktree and reruns all 87 items. [ABLATION_LAYERS.md](ABLATION_LAYERS.md)
lists the exact files and the hook change for each. In brief:

- **rules**: `.claude/rules/` plus the rule-injection hook
- **notes**: `docs/notes/`
- **runbooks**: the skill's `reference/` files (`SKILL.md` stays)
- **CLAUDE.md**: the file itself
- **bare**: all of the above

The baseline (`head`) ran 3 reps; each ablation ran 1 rep. Models: Haiku 5.5
(a newer Haiku than the 4.5 in section 1), Sonnet 5.5, Opus 5.5. Sources:
`results/ctx-<model>-<layer>`. Trees: Haiku `35dcc9e`, Sonnet and Opus
`1429bb9`.

| model | removed | recall /49 | rules /9 | skill use | tasks /11 | turns | wall s | $/run |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Haiku | none (3 reps) | 95% | 100% | 52% | 100% | 5.8 | 17.3 | 0.010 (rep 1: 0.011) |
| | rules | 49 | **8** | 61% | **10** | 6.3 | 15.9 | 0.008 |
| | notes | 48 | 9 | 50% | 11 | 5.7 | 16.2 | 0.011 |
| | rules + notes | 49 | 9 | 61% | 11 | 6.5 | 17.0 | 0.009 |
| | runbooks | 46 | 9 | 44% | 11 | 6.0 | 16.7 | 0.011 |
| | CLAUDE.md | 48 | 9 | **89%** | 11 | 6.2 | 16.0 | 0.008 |
| | bare | 47 | 9 | **94%** | 11 | **7.6** | 18.6 | 0.007 |
| Sonnet | none (3 reps) | 98% | 100% | 46% | 100% | 5.0 | 17.5 | 0.098 (rep 1: 0.133) |
| | rules | 48 | 9 | 33% | **10** | 5.3 | 15.8 | 0.115 |
| | notes | 47 | 9 | 39% | **10** | 4.6 | 18.8 | 0.127 |
| | rules + notes | 48 | 9 | 39% | **10** | 5.9 | 19.3 | 0.118 |
| | runbooks | 47 | 9 | 39% | 11 | 5.1 | 19.0 | 0.130 |
| | CLAUDE.md | 47 | 9 | **83%** | **10** | 4.7 | 18.8 | 0.111 |
| | bare | 46 | 9 | **89%** | **10** | **6.0** | 19.7 | 0.105 |
| Opus | none (3 reps) | 100% | 100% | 41% | 100% | 4.9 | 17.4 | 0.196 (rep 1: 0.272) |
| | rules | 49 | 9 | 22% | 11 | 5.2 | 17.7 | 0.243 |
| | notes | 48 | 9 | 33% | 11 | 4.9 | 16.5 | 0.264 |
| | rules + notes | 47 | 9 | 39% | 11 | 5.7 | 18.8 | 0.247 |
| | runbooks | 49 | 9 | 17% | 11 | 4.9 | 16.8 | 0.264 |
| | CLAUDE.md | 48 | 9 | **89%** | 11 | 4.9 | 17.7 | 0.230 |
| | bare | 46 | 9 | **89%** | **10** | **6.1** | 20.4 | 0.224 |

Baseline rows give rates over 3 reps (147 recall, 27 rules, 54 skills, 33
tasks runs). Ablation rows give counts over 1 rep. Skill use is the share of
the 18 skills prompts that invoked the photo-search skill. Turns, wall time and
cost are means over all scored runs of the config.

**On the cost column.** Don't compare the baseline's 3-rep mean cost with a
1-rep ablation. Reps 2 and 3 of the baseline find the prompt cache warm. Its
rep-1 cost is 1.1-1.4× its 3-rep mean, and it is at or above every ablation
on all three models. That fits the ablations' smaller starting context, but the
ablation runs are single samples, and cost also moves with turns.

### 3a. Correctness: docs mostly buy turns, not right answers

With every curated layer gone (`bare`), recall stays at 94-96% and rules at
100% on all three models. Tasks stay at 10 or 11 of 11. `bare` costs 1.0-1.8
more turns per run than the baseline. Wall time rises by 1-3 s per run, which
is within the run-to-run spread of the wall column. On these items, the
curated docs make sessions faster and more direct. They do not decide whether
the answer is right.

### 3b. The `bare` floor is not code-only

ABLATIONS.md says "the answers can be found in the code". The stored answers
show that is overstated. `bare` leaves about 1.4 MB of other prose in the tree:
`docs/plans/`, older plans in `docs/superpowers/`, eight `HANDOFF` files,
`README.md`, `NAS_SETUP.md`, and this eval's own files, including
`recall.jsonl`, which holds the expected patterns and passing answers
([ABLATION_LAYERS.md](ABLATION_LAYERS.md)). Recall answers that cite those
files:

| model | `head` answers citing `evals/context_eval/` | `bare` answers citing `evals/context_eval/` | `bare` answers citing other surviving docs |
|---|--:|--:|--:|
| Haiku | 0 / 147 | 8 / 49 | 21 / 49 |
| Sonnet | 0 / 147 | 4 / 49 | 24 / 49 |
| Opus | 1 / 147 | 6 / 49 | 16 / 49 |

Source: a regex over the `answer` field of `results/ctx-*/recall.jsonl`. The
rows store tool names but not tool inputs. Citations therefore undercount what
was read; they are a lower bound. One Haiku `bare` answer says outright: "The
answer key for this trap is `photodb-no-args` in
`evals/context_eval/recall.jsonl:6`." So `bare` measures "no curated context,
answer key reachable", not "code only". The 94-96% floor is an upper bound on
what code alone gives.

### 3c. Traps that only the docs carry

These items passed all 3 baseline reps, then failed once a layer was removed,
in more than one ablation or on more than one model:

| item | trap | failed under |
|---|---|---|
| `task-nas-git-pull` | git on the NAS needs `-c safe.directory` | Sonnet: notes, CLAUDE.md, bare |
| `detect-stacks-empty-list` | `detect_stacks(photo_ids=[])` means the whole library | Haiku: runbooks · Sonnet: CLAUDE.md, bare |
| `task-restack-folder` | restacking a folder must stay scoped | Haiku: rules · Opus: bare |
| `task-curl-intent-headers` | API calls send `X-Photosearch-Intent` | Sonnet: rules, rules + notes |
| `mirror-fields-needs-nas` | the replica's mirror needs the NAS's `/mirror-fields` | Sonnet and Opus: CLAUDE.md |
| `batch-queued-source` | batch "queued" comes only from open job rows | Sonnet and Opus: rules + notes |
| `maintenance-cron-time` | the sweep's cron time | Sonnet and Opus: bare |
| `video-passes-native` | video passes need `--native` | Opus: notes, bare (Sonnet's baseline is already mixed on it) |

Source: a read-only script over `results/ctx-*`, comparing each ablation's
failures with the item's baseline pass count. Each failure is one run. Read
the table as a list of candidates, not proof. The results are not monotone:
Haiku fails `task-restack-folder` without the rules, yet passes it when rules
and notes are both removed. Sonnet fails `task-nas-git-pull` without the notes,
yet passes it without rules and notes. At one rep, a single failure cannot be
told apart from noise (section 3e).

### 3d. CLAUDE.md suppresses skill use

This is the largest and most consistent effect in the data. With CLAUDE.md
present, the skill is invoked on 41-52% of skills prompts. With CLAUDE.md
removed, on 83-89%, on all three models:

| model | baseline (3 reps) | without CLAUDE.md (1 rep) | Fisher p | bare |
|---|--:|--:|--:|--:|
| Haiku | 28/54 | 16/18 | 0.005 | 17/18 |
| Sonnet | 25/54 | 15/18 | 0.007 | 16/18 |
| Opus | 22/54 | 16/18 | 0.0004 | 16/18 |

The likely mechanism: CLAUDE.md already answers or points to the runbook, so
the model skips the skill. The suite grades *invocation*, not whether the
answer was right. So this shows that CLAUDE.md competes with the skill. It
does not show that skipping the skill hurts. Removing the runbooks or the rules
lowered Opus's skill use to 17-22%. At one rep that is not significant (p ≈
0.09 for runbooks).

### 3e. Noise and robustness per model

| | Haiku | Sonnet | Opus |
|---|--:|--:|--:|
| baseline items (of 69 non-skill) with mixed results across 3 reps | 7 | 3 | 0 |
| baseline failed runs / 207 | 7 (3.4%) | 3 (1.4%) | 0 |
| failures expected by chance in one 69-run ablation | ~2.3 | ~1.0 | ~0 |
| failures observed per ablation: rules, notes, rules + notes, runbooks, CLAUDE.md, bare | 2, 1, 0, 3, 1, 2 | 2, 3, 2, 2, 3, 4 | 0, 1, 2, 0, 1, 4 |

- **Haiku**: its ablation failures are about what its baseline noise
  predicts. At one rep, this eval cannot tell which layers Haiku depends on.
- **Sonnet**: two to four times its noise in every ablation, and the most task
  failures (one task under five of six ablations). It is the model whose
  behaviour depends most on the docs.
- **Opus**: no baseline noise, every task passes under every single-layer
  ablation, recall loses at most 2 items, and the only clear drop is `bare`
  (4 failures). It is the most robust and the
  most expensive: about 2× Sonnet and about 25× Haiku per run.

## 4. Limitations

1. **Ceiling.** Most items pass under most configs. Recall and rules can show
   that something broke. They cannot rank two good contexts.
2. **Many items are answerable without the docs.** Answers come from code,
   tests and comments, and also from surviving plans, handoffs and the eval's
   own answer key (3b). Ablation drops therefore understate what a layer
   provides when that other prose is absent or stale.
3. **Single-rep ablations.** Every per-item ablation claim rests on one run.
   Only the skill-use effect (3d) is large enough to stand at one rep.
4. **Isolated vs repo-root.** Section 1's Sonnet arms ran the read-only
   suites in the repo root, where the owner's memory loads. Everything else
   ran isolated, without it. Don't compare absolute numbers across the two.
5. **Different models under one name.** "Haiku" is Haiku 4.5 in section 1 and
   Haiku 5.5 in section 3. The `sonnet` / `opus` aliases also moved between
   CLI versions (REPORT.md, follow-up).
6. **Staleness.** Section 1's "after" arms ran on 2026-10-19 trees (`e9ccd99`,
   `c21eba4`, `777916b`), before the 2026-10-20 inlined ops facts, imperative
   subsystem index and rule-injection hook. Don't quote them for HEAD. The
   ablation baselines (`35dcc9e` / `1429bb9`) are current. Since `1429bb9`,
   only eval write-ups have changed. The 2026-10-20 follow-up configs
   (`ctx2-*`) are not on this machine, so their numbers in REPORT.md could
   not be rechecked here.
7. **Skills grades invocation only.** A session that answers correctly from
   CLAUDE.md fails the skills suite.
8. **Wall time is noisy.** Haiku's baseline task wall time (48 s) is higher
   than every one of its ablations (32-41 s), which makes no causal sense.
   Treat wall-time differences under ~20% as noise.

## 5. Recommendations, ranked

1. **Hide the answer key from ablation runs.** Delete `evals/` (or at least
   `evals/context_eval/`) in every `--ablate` worktree, and consider
   `docs/plans`, `docs/superpowers` and `HANDOFF-*` as a separate "old prose"
   layer. Then re-run `bare` before quoting a code-only floor.
2. **Add discriminating items** whose answers exist only in the curated docs
   (owner decisions, operational history, measured thresholds). Seed them
   from the eight traps in 3c. Until then, the turns and cost columns are the
   most informative signal.
3. **Repeat the ablations at 3 reps** for Sonnet, at least for rules, notes
   and CLAUDE.md. Sonnet is the model where layers seem to matter, and its
   noise is low enough for 3 reps to resolve them.
4. **Decide what the skill is for, then grade that.** If the runbooks are the
   intended route, have CLAUDE.md point to the skill instead of answering
   directly (3d). Grade the skills suite on the answer or the reference file
   used, not on invocation alone.
5. **Keep the eight doc-only traps in CLAUDE.md or their rule files.** These
   are the lines that demonstrably earn their place. Several are already
   inlined.
6. **Re-run the before/after arms at HEAD** before quoting the diet's cost
   savings for the current tree. Compare cost at matched reps (rep 1 vs rep 1)
   or with the cache state recorded.
7. **Store tool inputs** (at least file paths read) on each row, so "which
   file answered this" can be measured instead of inferred from citations.

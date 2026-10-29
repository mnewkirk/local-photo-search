# What the context and the prompt actually do (ablations, 2026-10-28)

Two experiments:

- **CLAUDE.md eval** (`evals/context_eval/`): removes one documentation layer
  at a time and asks Haiku, Sonnet and Opus the same 87 questions and coding
  tasks.
- **Ask eval** (`evals/ask_eval/`): deletes one piece of the Ask agent's prompt
  at a time and replays 63 real-style photo searches on the local models.

## Summary

- **Docs buy speed, mostly.** With every doc removed, all three Claude models
  still answer 94–96% of recall questions and every rules question. The
  answers can be found in the code. Without docs they take 1–2 more turns.
- **A few traps live only in the docs.** Six items fail repeatedly when a
  layer is gone (listed below).
- **CLAUDE.md hides the skill.** Removing CLAUDE.md raises photo-search skill
  use from about 45% to about 85% on every model: CLAUDE.md answers first, so
  the skill is skipped.
- **The Ask model is sensitive.** At temperature 0, 10 of 63 asks change
  result after edits that remove nothing. Judge any prompt change on several
  items moving together.
- **gemma-4 can't do tool calls here.** 25% of gemma-12b's tool calls arrive
  malformed through LM Studio's parser. qwen3.5-9b stays the Ask model.

## CLAUDE.md eval: removing one layer at a time

Each row removes one context layer in an isolated worktree of HEAD
(`harness.py run --ablate <layer>`). `bare` removes all of them: CLAUDE.md,
the path-scoped rules and their injection hook, `docs/notes`, and the skill's
runbooks. The baseline ran 3 times per model; each ablation ran once. One run
of recall has a noise of about ±4 points; 91% on tasks is one task of 11.

| Model | Layer removed | Recall | Rules | Skill use | Tasks | Turns | $/run |
|---|---|--:|--:|--:|--:|--:|--:|
| Haiku | none | 95% | 100% | 52% | 100% | 5.8 | 0.010 |
| | rules | 100% | **89%** | 61% | **91%** | 6.3 | 0.008 |
| | notes | 98% | 100% | 50% | 100% | 5.7 | 0.011 |
| | rules + notes | 100% | 100% | 61% | 100% | 6.5 | 0.009 |
| | runbooks | 94% | 100% | 44% | 100% | 6.0 | 0.011 |
| | CLAUDE.md | 98% | 100% | **89%** | 100% | 6.2 | 0.008 |
| | everything (bare) | 96% | 100% | **94%** | 100% | **7.6** | 0.007 |
| Sonnet | none | 98% | 100% | 46% | 100% | 5.0 | 0.098 |
| | rules | 98% | 100% | 33% | **91%** | 5.3 | 0.115 |
| | notes | 96% | 100% | 39% | **91%** | 4.6 | 0.127 |
| | rules + notes | 98% | 100% | 39% | **91%** | 5.9 | 0.118 |
| | runbooks | 96% | 100% | 39% | 100% | 5.1 | 0.130 |
| | CLAUDE.md | 96% | 100% | **83%** | **91%** | 4.7 | 0.111 |
| | everything (bare) | 94% | 100% | **89%** | **91%** | **6.0** | 0.105 |
| Opus | none | 100% | 100% | 41% | 100% | 4.9 | 0.196 |
| | rules | 100% | 100% | 22% | 100% | 5.2 | 0.243 |
| | notes | 98% | 100% | 33% | 100% | 4.9 | 0.264 |
| | rules + notes | 96% | 100% | 39% | 100% | 5.7 | 0.247 |
| | runbooks | 100% | 100% | 17% | 100% | 4.9 | 0.264 |
| | CLAUDE.md | 98% | 100% | **89%** | 100% | 4.9 | 0.230 |
| | everything (bare) | 94% | 100% | **89%** | **91%** | **6.1** | 0.224 |

Costs are what `claude -p` reports at API prices; the runs were billed to a
Max plan. Skill use counts how often the model invoked the photo-search skill
on 18 prompts written to trigger it. Isolated runs don't load the owner's
memory directory, so these differ from the repo-root numbers in
`evals/context_eval/REPORT.md`.

## Where the docs are actually needed

These passed every baseline run and then failed once a layer was removed,
more than once across layers or models. Each is a trap the code doesn't
explain. All rest on single runs; confirm before acting on any one.

| Trap | Failed for |
|---|---|
| Git on the NAS needs `-c safe.directory` | Sonnet: without notes, without CLAUDE.md, bare |
| `detect_stacks(photo_ids=[])` means the whole library | Haiku: without runbooks · Sonnet: without CLAUDE.md, bare |
| Restacking a folder must stay scoped | Haiku: without rules · Opus: bare |
| API calls send `X-Photosearch-Intent` | Sonnet: without rules, without rules + notes |
| The replica's mirror needs the NAS's `/mirror-fields` | Sonnet and Opus: without CLAUDE.md |
| Batch "queued" comes only from open job rows | Sonnet and Opus: without rules + notes |

## How the three models differ

| | Mixed results across baseline runs | Doc dependence | Cost |
|---|---|---|---|
| Haiku | 7 of 69 items | one rules item and one task fail without the rules | ~$0.01 / run |
| Sonnet | 3 of 69 | three different tasks fail under different ablations | ~$0.10–0.13 / run |
| Opus | 0 of 69 | tasks stay at 100% under every ablation except bare | ~2× Sonnet, ~25× Haiku |

## Ask agent: deleting prompt pieces on the local model

The Ask agent runs on LM Studio, so library data never leaves the machine.
Each run deletes one piece of the shipped prompt and replays 63 asks on
qwen3.5-9b (`harness.py ablate --ref HEAD --prefix abl-head-`). Six control
edits that remove nothing (a newline, one word, two rules swapped, the tool
list reversed) measure the noise.

| Finding | Value |
|---|--:|
| Asks that change result under an edit that removes nothing | 10 of 63 |
| Deletions caught after discounting those | 12 of 23 |
| Single rules whose removal stays within the noise | 11 |

qwen relies on the tool descriptions, the rules block as a whole, the worked
examples, the library facts and household notes, and four specific rules
(subject in the foreground, read the total and relax, one per year, rerank
for visual checks). It routes mostly from the tool descriptions rather than
the rules text.

## Ask agent: qwen3.5-9b against gemma-4-12b

| | qwen3.5-9b | gemma-4-12b |
|---|--:|--:|
| Passing runs (63 asks × 3) | 162 / 189 | 77 / 189 |
| Seconds per Ask | 3.4 | 17.7 |
| Tool calls with invalid argument keys | 0 / 240 | 129 / 506 |
| Failing runs with a malformed call | 0 / 27 | 45 / 112 |

Gemma's own tool-call tokens leak through LM Studio's parser into the
arguments, for example `{"people=[<|"|>Nicole<|"|>,…": 1}`, so the search runs
with no usable filter. Its well-formed calls also route worse. It remains fine
for the text passes, which use no tools. Its ablation series was stopped as
uninformative.

## What this changes

- The CLAUDE.md eval needs items whose answers aren't in the code
  (operations, history, owner decisions) before it can show what the docs are
  worth. Until then its turns and cost columns are the honest signal.
- The six dependencies above are the doc lines that demonstrably earn their
  place.
- Decide whether the photo-search skill should be the entry point. If so,
  CLAUDE.md should point to it instead of answering directly.
- Judge any Ask prompt change on several items moving together, never on one
  item. Keep qwen3.5-9b as the Ask model.

## Sources

`evals/context_eval/REPORT.md` ("Context ablations across models"),
`docs/notes/search.md` (2026-10-26 to 10-28), results under
`evals/*/results/` (git-ignored). Models: claude-haiku-5-5 and the `sonnet` /
`opus` aliases; qwen/qwen3.5-9b and google/gemma-4-12b-qat at a 16,384-token
context.

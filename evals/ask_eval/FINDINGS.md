# Ask prompt eval: findings (start here)

Status as of 2026-10-28. This file sums up what the Ask prompt eval has
measured, what it means, and how far to trust it. [README.md](README.md)
covers how the harness works and how to run it. [MUTANTS.md](MUTANTS.md) shows
the exact text each ablation removes. The dated history is in
`docs/notes/search.md`, "Ask agent prompt". Every number below comes from
stored rows in `results/<config>/*.jsonl` (git-ignored, present on the owner's
desktop). They were regenerated with the offline commands `harness.py report`,
`mutation` and `coverage`, plus small read-only scripts where noted.

## Why the eval exists

The in-app Ask agent runs on a local model in LM Studio (qwen3.5-9b), so the
library never leaves the machine. Every round, it sends the same block: a
system prompt (routing rules, worked examples, library facts, household notes)
and 16 tool schemas. On `pre-ask-diet` that block was 44,545 characters, about
11.7k prompt tokens. The model is loaded with a 16,384-token context, and the
worker fleet sends describe requests to the same instance. On 2026-10-25, one
Ask plus two fleet requests overflowed it. The agent logged a bare HTTP 400,
fell back to single-shot, and found 0 photos. The goal was to trim the prompt
without breaking how the agent plans searches.

## Method in plain terms

63 **items** (asks), each with checks on the tool calls the agent makes: which
tool, which arguments, and which arguments it must *not* set (`allowed_args`).
23 items are copied from real Ask logs. The rest cover routing, curation,
counting, dates, pinned filters, follow-ups, write tools and honesty. An item
passes only if every check passes, there is no error, and there is no
single-shot fallback. Runs use temperature 0, 3 reps per item, so 189 runs per
config. The harness drives the real `agent.run_agent` against the replica DB
and LM Studio. Write tools never execute with `confirm=true`, and
`rerank_photos` is faked so that no vision model is needed.

At temperature 0, an item's reps almost always agree (3/3 or 0/3). So **items,
not runs, are the sample**. The report's paired sign test counts items that
got worse against items that got better.

## 1. Baseline and the shipped trim

| config | prompt | pass | context errors | first-round tokens | largest round | headroom |
|---|---|--:|--:|--:|--:|--:|
| `before` | `pre-ask-diet`, 2026-10-26 | 154/189 | 9 | 11,697 | 16,318 | −2,934 |
| `head` | pre-trim HEAD `bf25497`, 2026-10-27 | 153/189 | 9 | 11,718 | 16,141 | −2,757 |
| `trim4` | the shipped trim (worktree on `f048981`; same sizes as `1429bb9`) | **162/189** | **0** | 10,918 | 15,724 | −2,340 |

Source: `harness.py report --configs before head trim4`. **Headroom** = loaded
context (16,384) − largest prompt − the agent's 3,000-token reply budget. It
is what is left for concurrent fleet requests on the shared model.

**What the trim changed.** Three edits, from `docs/notes/search.md`:
(1) chapter, day, scene and bucket summaries fed back to the model are capped
at 20 compact entries; one chapters call had sent 114 summaries, about 8k
tokens. (2) Three long shared filter descriptions are kept in full on
`search_photos` only and hand-shortened elsewhere. (3) "Last summer" now means
the most recent summer.

**The evidence.**

- *Context errors 9 → 0.* In `before`, all 9 were the three multi-round
  chapter asks (b01, b02, b05), on every rep. They overflowed on their own,
  with nothing else running. All three pass 3/3 after the trim.
- *Pass 154 → 162.* Paired by item: 4 better (b01, b02, b05, s05) and 2 worse
  (x04 3/3 → 0/3, r10 1/3 → 0/3). Sign test p = 0.69. The gain is the three
  overflow items plus one more. The trim fixed the overflow; it did not
  measurably improve routing, and it did not measurably harm it.
- *The two worse items are both fragile.* Each flips under edits that remove
  nothing (section 3).
- *Headroom is still negative.* The largest round (15,724 tokens) plus the
  reply budget still exceeds 16,384. A long Ask running while the fleet sends
  a request can still overflow. The trim moved the peak by 594 tokens.

**A free replicate.** `before` and `head` differ by 86 characters (the
"last summer" example) and by one night's replica DB snapshot. Their totals
match (154 vs 153), but 4 items flipped: w01 and x01 went 3/3 → 0/3, r10 and
s05 went up. That is about the size of the trim's own paired change.

## 2. What was tried and rejected

| config | what it tested | pass | why rejected |
|---|---|--:|---|
| `head-consolidated` | one `search(mode=…)` tool instead of five search-family tools | 137/189 | 10 items worse, 3 better (sign p = 0.09). First round 7,244 tokens, the biggest cut. |
| `trim1` | earlier trim draft: rule cuts plus automatic first-clause shortening of shared descriptions | 151/189 | d02, d05 and b06 to 0/3. The shortening dropped "a plain 'best' is sort='quality_desc' with NO floor" from five schemas. |
| `trim2` | as `trim1`, with the grounding line | 150/189 | d01, d02, d05, b06 to 0/3 |
| `exp-c1` (1 rep) | compaction only | 51/63 | single-factor probe |
| `exp-c2` (1 rep) | compaction + cutting four "unneeded" rules + grounding line | 50/63 | b06 (Lake Como recovery) and d01 fail. The four rules were kept. |
| `exp-headcap` (1 rep) | the summary cap alone | 51/63 | still 2 context errors (b01, b02) in its one rep |

Source: `harness.py report`. `docs/notes/search.md` gives the consolidated run
as 134/189 with p = 0.057. The stored rows now read 137/189 and p = 0.09,
probably after a later regrade; the conclusion does not change. `trim1-4` ran
on dirty worktrees, and only their prompt sizes were recorded, not their
diffs. Their exact contents can't be reconstructed from the stored data.

The lesson from these runs: cuts that mutation testing marked "not needed"
still broke items once they were combined (`exp-c2`). Test every change with a
one-factor config before trusting it.

## 3. Mutation testing and the noise floor

**Mutation testing** deletes one named piece of the prompt per config and
reruns all 63 items once. The deletion can be a rule line, a block, the
examples, the filter descriptions, all parameter descriptions, or the four
grounding tools. A mutant is **killed** when some item's pass rate drops by at
least 50% against the 3-rep baseline. **Control** mutants remove nothing: a
trailing newline, "photos" → "pictures", one reworded sentence, two examples
swapped, two rules swapped, or the tool list reversed. An item that flips
under a control is **fragile**, and it is not counted as a killer. Exact edits:
[MUTANTS.md](MUTANTS.md).

| series | baseline | controls | fragile items | mutants killed (raw) | killed, fragile items discounted |
|---|---|--:|--:|--:|--:|
| pre-diet prompt (`abl-*`) | `before` | 3 | 4 | 23/23 | 18/23 |
| shipped prompt (`abl-head-*`) | `trim4` | 6 | **10** | 22/23 | **12/23** |
| gemma-4-12b (`abl-g12b-*`, stopped early) | `head-g12b` | 6 | 11 | — | 8/23 |

Source: `harness.py mutation --baseline trim4 --prefix abl-head-` and
`--baseline head-g12b --prefix abl-g12b-`. The pre-diet row comes from a
read-only script over the `abl-<mutant>` configs alone, because the
`--prefix abl-` filter also matches the other two series.

**The noise floor.** On the shipped prompt, each control flips 2-4 items. The
union over six controls is 10 of 63 items (16%): b01, d01, d02, d05, l03, r10,
s05, w01, w02, x04. On the pre-diet series, three controls gave 4. The count grows
with the number of controls, so 16% is a lower bound on how many asks depend on
incidental prompt details at temperature 0.

**One more probably-fragile item.** b06 (the Lake Como recovery) is a killer
for 7 of the 12 kills. For 4 of them it is the only killer: library facts, the
writes block, the grounding tools, and the read-total rule. The writes block
is about write tools and has no plausible link to b06, and b06 also flipped in
`trim1`, `trim2`, `exp-c2` and `head-consolidated`. No control happened to
flip it, but treat it as fragile. Without b06, the score is **8/23**: the
examples, household notes, the rules block, the foreground, representatives
and rerank rules, and both tool-description trims.

**What qwen relies on.** Removing all parameter descriptions (12,863 chars) or
the de-duplicated filter descriptions (3,095) breaks items. So do the rules
block as a whole, the worked examples and the household notes. Eleven single
rules survive their own deletion, among them framing flags, summarize, mood
tag, dates, exclusion and group terms. Their behaviour still passes, most
likely because the tool schemas say the same thing. That does **not** make
them safe to cut together: see `exp-c2` above.

**Items that never pass.** Six items fail on every qwen config: g01 and g02
(a person in the foreground in a given year), h03 (relax a secondary filter
after 0 hits), r03 ("best last summer"), s04 (eyes) and v02 (short hair);
s04 and v02 need rerank. These are capability gaps. A regression on them
cannot show up.

## 4. qwen3.5-9b vs gemma-4-12b

| | qwen3.5-9b (`trim4`) | gemma-4-12b (`head-g12b`) |
|---|--:|--:|
| passing runs | 162/189 | 77/189 |
| seconds per Ask | 3.4 | 17.7 |
| tool calls with argument keys not in the schema | 0 / 240 | 126 / 506 |
| failing runs with at least one such call | 0 / 27 | 42 / 112 |
| "empty model turn" nudges | 0 | 45 |

Source: `harness.py report`, plus a read-only script that checks each traced
call's argument keys against `tools.openai_tools()`. `docs/notes/search.md`
reports 129 and 45 from a slightly different check. Both runs used the same
prompt (system 9,613 chars, tools 31,584).

Gemma's own tool-call tokens leak through LM Studio's parser into the
arguments, for example `{"people=[<|"|>Nicole<|"|>,...": 1}`. The search then
runs with no usable filter. Its well-formed calls also route worse: 30 items
that qwen passes 3/3 fail 0/3 on gemma. Gemma beats qwen on only two items:
s04, which qwen never passes, and x04. Its 45 nudges are rounds where it spent the whole 3,000-token reply
budget without calling a tool. Its ablation series is uninformative: its
baseline is low, and 11 items flip under controls. The series was stopped
early (`control.tool-order` has 38 of 63 runs).

The conclusion is limited to tool calling through LM Studio's parser as
configured on 2026-10-28. Gemma stays fine for the text passes, which use no
tools.

## 5. Limitations

1. **Mutants ran 1 rep against a 3-rep baseline.** A kill is a 3/3 item going
   0/1. With 10-11 fragile items, a single kill is weak evidence.
2. **Before and after ran on different replica DB snapshots** (2026-10-26 vs
   2026-10-27; `sync-replica.sh` swaps it nightly). Counting items read the DB,
   so a few flips could come from the data rather than the prompt.
3. **Coverage is self-assessed.** `coverage` at HEAD reports all 20 rules and
   blocks covered, all 9 examples covered, and one item covering the whole
   combination for 112 of 116 real asks. Those numbers rest on hand-written
   `covers` tags and a keyword classifier. Mutation testing is the
   independent check, and with the noise above it is a coarse one.
4. **The trim drafts are not reproducible** from stored data (dirty
   worktrees, sizes only).
5. **No fleet contention.** One Ask runs at a time. Headroom stands in for
   the concurrent requests that caused the original overflow.
6. **One local model per comparison.** Token counts depend on the tokenizer,
   so compare a model only with itself.

## 6. Recommendations, ranked

1. **Fix the headroom, not the wording.** The peak round still leaves
   −2,340 tokens. Loading the Ask model at a 32k context (or giving the fleet
   its own instance) removes the overflow risk. Further prompt trims cannot
   reach that margin without breaking items.
2. **Judge any prompt change on several items moving together.** At least
   one in six asks flips under a no-op edit. Add b06 to the fragile list.
   Never act on a single item.
3. **Run 3 reps for any mutant you intend to act on**, and run the six
   controls with every new series so that its noise floor is measured, not
   assumed.
4. **Keep qwen3.5-9b as the Ask model.** Re-test gemma-4 only after an LM
   Studio parser update, and check the malformed-key count first.
5. **Turn the never-pass items into work items** (foreground person per year,
   relax-after-zero, "last summer", rerank for eyes and hair). They are where
   the agent is actually wrong today.
6. **Record the prompt diff** (or commit it) for every config, so that trims
   stay reproducible.

## 7. Later checks

### 2026-10-31: the `get_photo_audit` tool (`b3d9a23`)

Plan: [AUDIT_TOOL_PLAN.md](AUDIT_TOOL_PLAN.md). Desktop, the usual Ask model
loaded alone at 16,384 context, 3 reps, fleet idle. `measure` showed only the
new tool (tools 31,584 → 32,853 chars, 16 → 17 tools); `measure --ref f3254cd`
matched `trim4` exactly.

| config | ref | pass | context errors | fallbacks | first-round tokens | largest round | headroom |
|---|---|--:|--:|--:|--:|--:|--:|
| `trim4` | (2026-10-27) | 162/189 | 0 | 0 | 10,918 | 15,724 | −2,340 |
| `audit-before` | `f3254cd` | 162/189 | 0 | 0 | 10,918 | 16,109 | −2,725 |
| `audit-after` | `b3d9a23` | 162/189 | 0 | 0 | 11,223 | 15,542 | −2,158 |

Source: `harness.py report --configs audit-before audit-after --failures`.

- *Baseline.* The shortcut applied (sizes matched, snapshot 3 days old), but a
  run took about 12 minutes, not 1.5 h (189 asks at ~3.7 s), so `audit-before`
  ran too, on the same snapshot as `audit-after`. Both are compared below.
- *Flips.* `audit-before` → `audit-after`: 2 worse (l03 3/3 → 0/3, flagged
  REGRESSED; b01 1/3 → 0/3), 2 better (x04 0/3 → 3/3, w01 2/3 → 3/3), sign
  test p = 1.0. Against `trim4`: 1 worse (b01), 1 better (x04), p = 1.0. Every
  moved item is on the fragile list; **no non-fragile item moved**. In l03 the
  agent now asks which dates to use instead of taking the pinned ones.
- *Noise replicate.* `trim4` → `audit-before` is the same prompt on a newer
  snapshot: 3 fragile items flipped (b01, l03, w01), as many as the tool did.
- *Tool bleed.* 0 `get_photo_audit` calls in 189 `audit-after` rows.
- *Headroom.* Each round costs +305 tokens (median over items, first round and
  same-length runs alike). The report's peak fell only because w01, which sets
  the peak in all three configs, took 4 rounds instead of 5. Like for like, the
  headroom is about −2,645 against `trim4`'s −2,340, as the plan estimated.
- *Side notes.* In `audit-before`, w01 r2 called `set_photo_tags` with
  `confirm=true` (blocked by the harness). g01, a never-pass item, now fails by
  calling `representatives` instead of `search_photos` without `sort=subject`.
  The replica DB was already schema v40 (an empty `photo_audit` table) before
  either run: the Step 1 `measure --ref b3d9a23` opened it with the new code.
  `f3254cd` ignores the table.

**Decision** (plan's table): clean on context errors, regressions, tool bleed
and the aggregate, with headroom as predicted. Keep the tool as is. The
positive items in the plan's "gap" section are still untested.

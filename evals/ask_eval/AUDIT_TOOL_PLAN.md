# Ask eval: test plan for the `get_photo_audit` tool

Written 2026-10-31. Harness manual: [README.md](README.md). Results so far: [FINDINGS.md](FINDINGS.md).

**Status: run 2026-10-31 on the desktop; clean, keep the tool as is.** `audit-before` and `audit-after` both 162/189, 0 context errors, 2 worse / 2 better (all fragile, sign p = 1.0), no `get_photo_audit` call on any item, +305 tokens per round. Each run took about 12 minutes, not 1.5 h. Details: FINDINGS.md, section 7. The "gap" items below are still not added.

## What `asks.jsonl` is

`asks.jsonl` is the eval's question set: 63 items, one JSON object per line. Each item is a question a person might type into the in-app Ask box ("Show me photos of Calvin."), plus a list of `checks` on the tool calls the agent makes in reply: which tool it called, which arguments it set, and which it must not set. The harness asks the real Ask agent each question against the replica DB, records the trace, and grades it with `checks.py`. Nothing reads the answer text unless an item asks for that. 23 items are copied from real Ask logs; the rest probe routing, counting, dates, pinned filters, follow-ups, write tools and honesty. See "Suites" in the README for the table. Items are the unit of statistics, not runs: at temperature 0 the three reps of an item almost always agree.

## What this plan tests

Commit `b3d9a23` added a read tool, `get_photo_audit`, to the registry. Ask and MCP both advertise it. Ask's tool list is eval-tuned, so the question is whether one more tool changes how Ask plans searches.

| Risk | How it would show up |
|---|---|
| **Routing:** an ask like "what happened to…" or "removed from…" pulls Ask away from `search_photos` or `summarize`, or the tool gets called on an unrelated ask | Existing items fail, or a trace calls `get_photo_audit` on an ask that has nothing to do with history |
| **Context:** the tool schema adds 1,267 characters (about 300 tokens) to a prompt that already overflowed a 16,384-token context once (2026-10-25) | Context errors above 0, or headroom getting worse |
| **Noise:** the eval flips items even under edits that remove nothing | Reading a flip as a regression |

The reference to beat is `trim4` in FINDINGS.md: 162/189, 0 context errors, first round 10,918 tokens, largest round 15,724, headroom −2,340. Estimate for this change (not measured): first round about 11.2k, largest round about 16.0k, headroom about −2.6k. Headroom is loaded context (16,384) − largest prompt − the 3,000-token reply budget.

## Preconditions

1. `git pull` on the desktop so `HEAD` contains `b3d9a23`.
2. The fleet is idle: no local workers, no worker containers, no active NAS claims. The harness refuses to start otherwise and has no override.
3. Nothing else uses LM Studio during the run (other sessions, the Ask page, describe jobs).
4. `qwen/qwen3.5-9b` is the only model loaded, at **16384 context**. A model that unloads mid-run is JIT-reloaded at LM Studio's default context and every later request fails with HTTP 400. On a 4096-context load every run falls back to single-shot with a ~235-token prompt: that is a setup error, not a result.
5. Do not run `sync-replica.sh` between the two configs: both must see the same replica snapshot (`meta.json` records its mtime).
6. The desktop must not sleep during the run.

```bash
lms ps    # one model, qwen/qwen3.5-9b, CONTEXT 16384
```

## Step 1: checks that call no model (minutes)

```bash
PY=./.venv/bin/python
$PY -m pytest tests/test_ask_eval_harness.py tests/test_tools.py tests/test_photo_audit.py -q
$PY evals/ask_eval/harness.py measure --ref f3254cd
$PY evals/ask_eval/harness.py measure --ref b3d9a23
$PY evals/ask_eval/harness.py run --config audit-after --ref b3d9a23 --repeat 3 --dry-run
```

Pass: tests green, and `measure` shows only the new tool added (`tools_chars` +1,267, `n_tools` 16 → 17).

Shortcut: if `measure --ref f3254cd` matches what `results/trim4/meta.json` recorded and the replica snapshot is within about a week of the `trim4` run, use `trim4` as the baseline and skip the `audit-before` run below (saves about 1.5 h).

## Step 2: the runs (about 3 h; 1.5 h with the shortcut)

Run as a background script that writes a marker file when done. Watch the marker or a saved PID (`kill -0`), never `pgrep -f` on a bare string. `--nas` takes the NAS API URL; use `$PHOTOSEARCH_NAS_URL`.

```bash
PY=./.venv/bin/python
$PY evals/ask_eval/harness.py run --config audit-before --ref f3254cd --repeat 3 --nas "$PHOTOSEARCH_NAS_URL"
$PY evals/ask_eval/harness.py run --config audit-after  --ref b3d9a23 --repeat 3 --nas "$PHOTOSEARCH_NAS_URL"
$PY evals/ask_eval/harness.py report --configs audit-before audit-after --failures
```

- Run `audit-before` first. The "after" code opens the replica DB through `PhotoDB`, which migrates it from schema v39 to v40 and adds an empty `photo_audit` table. That is harmless, but the order keeps the baseline on the schema it was built against. Note that Step 1's `measure --ref b3d9a23` (and the `--dry-run`) already opens the DB with the "after" code, so the migration happens there, before either run; on 2026-10-31 both configs saw v40.
- Sanity check after about 5 minutes: no row says `FALLBACK`, and prompt tokens are about 11–13k, not about 235. If not, stop and fix the context before spending the hours.
- Optional afterwards: `harness.py regrade --configs audit-before audit-after --judge` re-scores answer checks with the judge model.

## Step 3: reading the result

In this order:

1. **Context errors:** must be 0 in both configs.
2. **Real regressions:** `report` marks an item REGRESSED when its pass rate drops by at least half. Ignore flips among the known fragile items, which move under edits that remove nothing: b01, d01, d02, d05, l03, r10, s05, w01, w02, x04, and b06 (FINDINGS.md, section 3). A REGRESSED item outside that set is real.
3. **Tool bleed:** search the `audit-after` rows for `get_photo_audit` calls. It should be called only on history questions, and no existing item is one. A call on any routing, counting, curation, dates, locked, followup, writes or honesty item is a routing regression even if the item still passes.
4. **Aggregate:** pass count against `audit-before` and the paired sign test over items. Expect within about ±4 flips and p well above 0.05. FINDINGS.md's "free replicate" saw 4 flips from a one-line edit plus a new snapshot.
5. **Headroom:** compare the report's headroom column. Expect about −300 tokens against the baseline. It is already negative, so this is a judgement call, not a gate.

## The gap: nothing tests the tool itself

No item in the 63 should call `get_photo_audit`. A clean run shows the tool does not hurt routing, not that Ask can use it. To close the gap, add 3–4 items to `asks.jsonl` and run them against both configs (the baseline is expected to fail the positives; they have no tool to call). These were **not** added, because changing the item set also changes the sample the baseline was built on, and the grading cannot be validated without a model run.

Proposed items (grade with `called` / `first_tool` and an `arg` check from the README's check grammar):

- "Which photo was removed from the Best of March collection?" expects `get_photo_audit` with `collection` set.
- "What happened to photo 250154?" expects `get_photo_audit` with `photo_id` 250154.
- "Has anyone changed the face labels on photo 250154?" expects `get_photo_audit` with `action` in `face_assigned`, `face_cleared`, `face_unmatched`, or no `action`.
- Negative: "show me photos removed from my vacation" must still be a search (`search_photos` first), not an audit call.

Item shape and `covers` ids are checked by `tests/test_ask_eval_harness.py`; copy an existing item and run that test.

## Decisions

| Outcome | Action |
|---|---|
| Clean on steps 3.1–3.4, headroom as predicted | Keep the tool as is |
| Tool bleed, or a non-fragile item regresses | Shorten the tool description (target under 600 characters) and re-run `audit-after` alone (about 1.5 h) |
| Context errors above 0 | Shorten the description, or hide the tool from Ask and keep it on MCP only |
| Borderline | Re-run `audit-before` once more (about 1.5 h) to get a replicate and see how much of the flipping is noise |

## What to bring back

The `report --failures` output, any non-fragile item that moved, and the result of the `get_photo_audit` search on non-history items. Record the outcome in FINDINGS.md under a new dated heading.

# Category grounding: make `photos.categories` trustworthy

**Status: PLANNED 2026-10-18** (plan-debate, two planners, two critique rounds;
owner decisions below). Nothing built yet.

## Problem

The `category-content` pass hands the model `photos.description` plus the
377-term `CONTENT_VOCABULARY` (`photosearch/vocab_content.py`) and stores any
in-vocabulary term it returns. Nothing checks that the description supports
the term. The only guard is `_CATEGORY_MAX_PLAUSIBLE = 60` in
`describe.extract_categories_from_description`, a count cap set for
llama3.2:3b's vocabulary recitations (53 to 232 alphabetical terms).

What that looks like on the live library (replica copy, read-only, 2026-10-18):

- Photo 251357 (DSC05775, `2026/2026-10-17_ILCE-7RM6`): a daylight outdoor
  boys' soccer description. gemma-4-12b-qat returned 68 in-vocabulary tags at
  temperature 0 and 84 on the 0.4 retry. About 44 were supported, about 14
  were unsupported padding (building, celebration, city, coach, court, guest,
  hat, road, wall, urban environment), and about 10 were contradicted by the
  text (basketball, basketball court, basketball hoop, artificial lighting
  against "bright daylight", soccer goal; at 0.4 also indoor, gym, gymnasium,
  kitchen, mountain, water). The answer was not alphabetical and not in
  vocabulary order: it was not a recitation. The cap retired the photo after
  three attempts. Right outcome, wrong reason, and 44 good tags lost.
- Neighbours 251356 and 251358 stored 28 and 21 tags; 251356 holds
  `artificial lighting` and `house` for the same daylight field. The leak is
  in accepted rows too, at small scale.
- gemma era (8,034 generations since 2026-10-01; 7,769 photos whose stored
  list equals the generation): median 21, p99 44, max 60. 133 rows carry two
  or more sport groups and 90 of those descriptions say "soccer" (247937 is a
  soccer team photo stored with basketball, basketball court, basketball hoop).
  `artificial lighting` appears alone in 1,143 rows; 633 to 794 of those
  descriptions mention daylight or outdoors and contain no lighting word.
- llama era: 158,668 `llama3.2:3b` + 2,238 `llama-3.2-3b-instruct`
  generations; 154,446 stored rows are still llama output. 122 stored rows
  hold more than 60 categories (live recitations). Of 702 llama rows with 20
  to 52 tags, 126 are alphabetical lists of mostly unsupported concrete nouns
  (41317: "beehive-like stone structure, no people" stored `bleacher, bridge,
  chain link fence, cityscape, grassy field`; 11387: "tree with pink
  blossoms" stored `bird, bouquet, cluster`). The count cap never sees these.
- Vocabulary order is NOT a recitation signal on its own: 72% of gemma lists
  are at least 0.9 ascending in vocabulary order (median 0.96, 43.5% fully
  monotonic) against a median of 0.98 for llama's recitations.
- Literal word match is NOT a support test on its own: only 41% of gemma tags
  appear literally in the description ("vehicle" for cars, "child" for boys;
  `artificial lighting` is literal 1% of the time).

Owner decision 2026-10-07 still stands: long gemma lists and near-duplicates
(grass / grassy field) are fine; the prompt and vocabulary do not change
without an A/B.

## Design

### One pure rule module

`photosearch/category_grounding.py`:

```
ground(description, categories) -> Grounding(kept: list[str],
                                              dropped: dict[str, str],   # term -> reason
                                              recitation: bool)
```

Pure, no I/O, idempotent (`ground(d, ground(d, x).kept).kept == ground(d, x).kept`),
output is a subset of input in input order, never adds a term. A `None` or
empty description returns the list unchanged. Worker, server, backfill and
eval all call this one function. Docstring records the inclusion bar for a
group the way `visual_tags_derive.CONTRADICTORY_PAIRS` does, and every
rejected candidate with the evidence against it.

### The description decides a contradiction (RESOLVED by both planners)

A contradiction GROUP has two or more sides; each side is a set of vocabulary
terms plus a cue list (case-insensitive word-start regexes run over the
description). Resolution per group:

- a present side is dropped when a rival side is cued and its own is not,
  **including when only one side was emitted** (this is what 251356 needs:
  `artificial lighting` alone against "outdoors under trees");
- two or more sides present and none cued: all present sides are dropped
  (the CONTRADICTORY_PAIRS shape, used only when the text cannot decide);
- both cued: both kept (a view out of a window is honestly both).

Drop-both as the default was rejected on the numbers: it deletes 329
soccer-group tags across the 124 to 133 mixed-sport rows, 80 to 90 of whose
descriptions say "soccer", and on 251357 it would delete the photo's main
subject. The visual pass drops both because it has nothing to judge by; this
pass has the description.

v1 groups (about six small per-side cue lists, each scoped to the group's own
terms, not a general synonym system):

| group | sides (vocabulary terms) | cues |
|---|---|---|
| sport | soccer {soccer, soccer ball, soccer game, soccer goal, soccer jersey, soccer player, indoor soccer, soccer indoor, youth soccer game} / basketball {basketball, basketball court, basketball hoop, backboard} / baseball {baseball, baseball bat, baseball game, baseball glove, baseball uniform} / volleyball {volleyball} | each side's own name words (soccer, football; basketball, hoop, backboard; baseball, softball, bat, mitt, dugout; volleyball) |
| setting | indoor {indoor, indoor soccer, soccer indoor, gym, gymnasium, kitchen, dining area, dining table, living room, restaurant, floor indoor} / outdoor {outdoor} | indoor(s), inside, interior, room, gym, kitchen, ceiling, hallway, classroom / outdoor(s), outside, sky, sunlight, sunny, daylight, field, park, tree(s), grass, lawn, turf, beach, trail, yard, garden, forest |
| light | {artificial lighting} / {natural light, sunny day, clear day, sun} | lights, lamp, lit, illuminated, fluorescent, floodlight, spotlight, neon, chandelier, bulb, flash, night, evening / daylight, sunlight, sunny, sunshine, midday, blue sky, clear sky |
| weather | sunny {sunny day, clear sky, clear blue sky, clear day, blue sky, sun} / overcast {cloudy day, cloudy sky, cool cloudy day, fog} | sunny, sunlight, sunshine, clear sky, blue sky, cloudless / overcast, cloudy, grey sky, gray sky, fog, mist, rain, drizzle, gloomy, storm |

Known traps the cue lists must handle, each pinned by a test:

- "light blue", "light-colored", "daylight" and "sunlight" are not
  artificial-light cues. A prototype that matched them produced 586 suspect
  drops; the light list is the first thing the holdout gate tests.
- "baseball cap" is clothing, not a baseball cue.
- "partly cloudy sky" and a bare "cloud" are neutral for the weather group.
- `collage` present skips the setting group (multi-scene collages are
  honestly both; 19 of 6,791 library-wide indoor+outdoor hits carry it).
- `car interior`, `car seat`, `car window`, `windshield` and `interior space`
  are NOT indoor-side terms. 5 of 5 sampled `car interior` + `outdoor` rows
  were photos taken from inside a vehicle of an outdoor scene; `interior
  space` hits include atriums, plazas and photographs of a photograph.

### Unsupported padding: deferred behind a gate (RESOLVED)

Terms in no group are EXEMPT from any support check by default, so a
vocabulary recompile can never silently start dropping a new term. Required-
cue FAMILIES (a term family is dropped when the description has no cue for
it: basketball without a hoop, kitchen without a stove, water without any
water word, mountain, city, building, celebration, coach, hat, wall, ...) are
a follow-up, promoted one family at a time, each only after it reaches at
least 90% correct drops on a fresh owner-labelled holdout (the sharpness-eval
lesson: in-sample 0.86 fell to 0.55 on holdout). Families that fail are
removed, not retuned in-sample. The eval script reports them from day one as
a diagnostic so the owner can see what each would drop.

### Recitation verdict runs live, with a 120 backstop (OWNER DECISION)

The 60 count cap is replaced on the worker by:

```
recitation = order_share >= 0.9 AND unsupported_obj >= 10 AND unsupported_obj / obj >= 0.5
```

where `order_share` is the share of adjacent pairs ascending in vocabulary
order, `obj` is the answer's terms that are in `OBJECT_TERMS` (a fixed list of
roughly 130 concrete nouns: bed, book, bottle, bowl, box, cake, camera, crib,
desk, tree, car, fence, turf, ... plus every family term), and "unsupported"
means the term's own words are absent from the description and no family cue
matches. `OBJECT_TERMS` is used ONLY for the verdict, never to drop a term;
its per-term imprecision averages out over an answer. Prototype: catches
141 of 141 llama recitations over 53 tags and the 126 hidden ones at 20 to
52, flags 0 to 9 of 7,769 gemma rows depending on plural folding. Ship bar:
0 gemma rows wrongly judged after review, at least 95% of the over-53 llama
recitations caught.

A count backstop `_CATEGORY_MAX_PLAUSIBLE = 120` stays, judged on the
REPAIRED list (gemma's worst observed answer was 84; the 256-token text
budget makes more than 100 nearly impossible on LM Studio; this mainly guards
the Ollama route). Grounding that empties a non-empty answer is also a
problem ("no grounded categories").

The alternative, keeping 60 applied post-repair, was rejected: it already
would have passed 251357 (68 minus 10 is 58), but the count cap is blind to
partial recitations and fires again on the next honest 61-tag answer.

### Retry only when the text cannot decide (RESOLVED)

- Description-decided drops: NO retry. The description is the evidence; a
  resample adds nothing (the 0.4 retry on 251357 was worse: 84 tags). A
  repairable answer costs one model call.
- Both sides emitted and neither cued: one retry at 0.4, then drop both.
- Recitation verdict: one retry at 0.4, then `UnusableAnswer`.
- Empty / no in-vocabulary answer: unchanged (retry once, then `UnusableAnswer`).

`problem_of` order: no in-vocabulary categories; runaway (> 120); recitation;
no grounded categories. `_text_pass_answer` keeps judging the RAW parse;
grounding is applied after acceptance in a thin wrapper so keywords are
untouched.

### Worker AND server (OWNER DECISION)

- **Worker** (`describe.extract_categories_from_description`) is primary: the
  verdict, the one retry, the repair. Everything that calls it gets it: the
  fleet (`worker._process_category_content`), M28 (`rerun.run_pass_sync` goes
  through the same function), both `index.py` writers, `evals/text_passes_eval.py`.
- **Server** (`worker_api.submit_results`, category-content branch) applies
  `ground()` again against the CURRENT stored description. Precedent:
  `_merge_visual_tags` / `merge_vlm_answer` already re-derive visual tags
  server-side on every submit because the server holds state the worker
  cannot guarantee. It covers old workers during the fleet restart, a
  description rewritten between claim and submit, and M28 reruns. Rules can
  then ship with a NAS redeploy alone. Mechanics: one `SELECT id, description`
  for the batch in a pre-pass BEFORE `begin_batch` (because `db.log_error`
  commits); a recitation verdict or an emptied list becomes a worker-style
  failure through `_record_worker_failures` (one attempt, `index_errors`,
  column stays NULL); the rest write `kept`. `log_generation` logs the kept
  list so stored == generation keeps holding. The server never retries. A
  NULL description stores the list unchanged.

## Steps

1. **Rule module.** NEW `photosearch/category_grounding.py`, NEW
   `tests/test_category_grounding.py`. Groups, cue lists, `OBJECT_TERMS`,
   `tokens()` (reuse or extend `stale_descriptions._tokens`, add -es/-ies
   folding), `ground()`, `is_recitation()`. Verify: unit tests pass under
   `./scripts/test-like-ci.sh`. Risk: loose cues keep bad tags, tight cues
   drop good ones; step 2 measures it. 0.75 d.
2. **Eval (ship gate).** NEW `evals/category_grounding_eval.py`, output under
   `evals/category-grounding/` (samples git-ignored, summary committed). Opens
   `file:photo_index.db.local?mode=ro`, imports `ground()`, makes no model
   calls. Queries:
   - Q1 current gemma: `SELECT p.id, p.description, p.categories FROM photos p
     JOIN generations g ON g.photo_id=p.id AND g.text_type='category-content'
     AND g.generated_text=p.categories WHERE g.model_used LIKE 'google/gemma%'`
     (7,769), plus all 8,034 gemma generations joined to the current description.
   - Q2 llama: all `category-content` generations with `model_used IN
     ('llama3.2:3b','llama-3.2-3b-instruct')`, split at 53+ (141), 20 to 52
     (702), under 20.
   - Q3 backfill preview: every `categories IS NOT NULL` row attributed to its
     latest category-content generation with an eligibility flag (stored set
     is a subset of the generation set).
   - Q4 fixtures: the 251357 answers at 0 and 0.4 (recorded in this session's
     transcript; re-run one deterministic temp-0 call if needed) plus 251356
     and 251358, committed as JSON.
   Prints per group: rows hit, tags dropped, top terms, 25 random (photo id,
   term, reason, matching cue sentence or "no cue") samples; the verdict
   confusion (recitations caught, gemma rows flagged with their descriptions,
   hidden llama recitations); before/after count per term; the drop-count
   distribution; the required-cue families as a diagnostic. Owner labelling:
   tune on sample A, judge a fresh sample B, 90% bar per group. Prototype
   (crude cues, indicative): about 400 of 7,769 gemma rows hit a v1 group.
   0.5 d plus 1 to 2 h of owner labelling.
3. **Worker integration.** `photosearch/describe.py`, `photosearch/worker.py`
   (log `kept N, dropped M (reasons)`), `tests/test_text_pass_safeguards.py`.
   Verify: tests; one native worker on `2026/2026-10-17_ILCE-7RM6` after
   `clear-pass category-content` on 251357 and a few of 247930 to 247949;
   confirm basketball terms, `artificial lighting`, kitchen and gym are absent
   and soccer terms kept. Risk: `evals/text_passes_eval.py` numbers shift;
   note it in the eval README. 0.25 d.
4. **Server repair.** `photosearch/worker_api.py` (category-content branch),
   `tests/test_worker_categories.py`, `tests/test_worker_submit_resilience.py`,
   `tests/test_llm_provenance.py`. Verify: an old-worker payload containing
   `basketball` for a soccer description stores without it; a recitation
   payload becomes a failure row before `begin_batch`. 0.5 d.
5. **Backfill CLI.** `cli.py` `ground-categories`, `tests/test_ground_categories_cli.py`.
   Modelled on `derive-visual-tags`: `--db` with `envvar="PHOTOSEARCH_DB"`,
   dry run by default opened `mode=ro` (a missing DB is an error, never a
   stub), `--apply` through `PhotoDB`. One read joined to the latest
   generation, then 2,000-row chunks with a commit and a progress line
   between, `UPDATE photos SET categories=? WHERE id=? AND categories IS ?`,
   unchanged rows skipped, lost races counted. Eligibility: stored set is a
   subset of the latest LLM generation (hand-set `set_photo_tags` rows and
   the ~1,064 pre-logging rows are skipped and reported). Never writes `'[]'`
   from a non-empty list, never touches NULL, never logs to `generations`.
   Recitation rows are reported; only `--requeue-recitations` NULLs
   `categories` and deletes the `worker_processed` row, as
   `stale_descriptions.requeue` does. `--model gemma|llama|all` (default
   gemma), `--folder`. Verify: replica dry run matches the step-2 numbers.
   Risk: dropping correct tags across 154k llama rows; llama only after its
   samples are reviewed. 0.75 d.
6. **Docs.** `CLAUDE.md` "Bad LLM output is a failure, not a result": replace
   "more than 60" with the verdict, repair and backstop; new subsection
   "category-content grounding" with the groups, the rejected candidates and
   the measured numbers. 0.25 d.
7. **Deploy.** NAS redeploy first (server repair + CLI), then restart the
   fleet, then `ground-categories` dry run on the NAS, owner approves,
   `--apply --model gemma`, later `--model llama` and `--requeue-recitations`
   (about 248 rows back to gemma). Verify: no new recitation `index_errors`
   on gemma; `category=basketball` search spot check; `categories IS NOT NULL`
   count unchanged by the backfill.

Total about 3 dev-days plus owner labelling and two deploys.

## Tests

`tests/test_category_grounding.py`
1. 251357 fixture: basketball family, artificial lighting, indoor/gym/kitchen
   dropped; soccer, turf, tree, fence, car, boy, spectator kept; not a recitation.
2. 251356: artificial lighting dropped against "outdoors under trees"; house kept.
3. Exempt terms (vehicle, child) are never dropped by a support rule.
4. "light blue shirt" / "daylight" are not artificial-light cues; "baseball cap"
   is not a baseball cue.
5. Setting group: "inside a gym" drops outdoor; neither cued drops both; both
   cued keeps both; `collage` skips the group; `car interior` + `outdoor` survive.
6. Weather group: "overcast" drops sunny day; "partly cloudy" is neutral.
7. Idempotence on fixtures and seeded random vocabulary subsets.
8. Output is a subset of input, order preserved, nothing added; None/empty
   description leaves the list unchanged.
9. Integrity: every rule term and `OBJECT_TERMS` entry is in `CONTENT_VOCABULARY`;
   no term on two sides of one group.
10. Verdict: an alphabetical llama-style recital on a soccer description is a
    recitation; the 68-tag gemma list is not; a grounded alphabetical 44-tag
    list is not.

`tests/test_text_pass_safeguards.py`
11. Recitation test rewritten with real vocabulary terms; still expects temps
    [0, 0.4] and `UnusableAnswer("recitation")`.
12. A 70-tag grounded non-alphabetical list is kept (the old cap failed it).
13. A description-decided contradiction is repaired with NO retry (temps == [0]).
14. Both-sides-emitted, neither cued: one retry, then drop both.
15. Grounding empties the answer twice: `UnusableAnswer("no grounded")`.
16. More than 120 tags is rejected.
17. The existing worker failure-row test still passes.

Server: 18. an old worker's list is repaired against the stored description
and the kept list is logged; 19. a recitation or emptied list is a failure
recorded before `begin_batch`, column stays NULL; 20. NULL description stores
unchanged.

CLI: 21. dry run writes nothing (DB mtime unchanged); 22. NULL rows untouched;
23. hand-edited row (not a subset of its generation) skipped; 24. guarded
UPDATE loses to a concurrent change and is counted; 25. commits between
chunks; 26. recitation rows untouched without the flag, NULLed and ledger row
deleted with it; 27. `'[]'` is never written from a non-empty list.

## Risks

- Cue lists are hand-built. The labelled holdout is the control; a group under
  90% does not ship.
- Dropping without a retry trusts the description. A hallucinated description
  yields matching categories; verify and re-describe own that.
- 154k llama rows are the bulk of the library and were not the target. The
  backfill defaults to gemma.
- Worker/server skew during deploy is covered by the server copy and idempotence.
- `OBJECT_TERMS` and the cue lists depend on the vocabulary: the integrity test
  plus an eval re-run are required on any recompile.

## Decisions

| topic | option A | option B | owner's choice | date |
|---|---|---|---|---|
| Where the grounding filter runs | worker only (`describe.py`); fleet restart alone | worker + server idempotent repair in `submit_results`; NAS first | **worker + server** | 2026-10-18 |
| Live recitation guard | keep the 60 count cap, judged on the repaired list; verdict in the backfill report only | order-plus-unsupported verdict live on the worker, 120 count backstop on the repaired list | **verdict live + 120 backstop** | 2026-10-18 |
| Contradiction resolution | drop both members of a pair (visual-pass shape) | the description picks the side, incl. one-side-emitted | resolved by the planners: **description decides** | 2026-10-18 |
| v1 scope | contradiction groups only | also required-cue families | resolved by the planners: **groups only; families later behind a 90% holdout gate** | 2026-10-18 |
| Retry on a repairable answer | retry once at 0.4 on any contradiction, then repair | no retry when the description decided; retry only when both sides emitted and neither cued | resolved by the planners: **B's rule** | 2026-10-18 |

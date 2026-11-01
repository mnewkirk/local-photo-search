# Person-level attributes for the MCP tools and Ask

Status: 📐 proposal (2026-11-01). Nothing built. Owner decisions are in the last section.

The question is which facts about the *people in a photo* the photosearch MCP tools and the Ask agent (the same
tools, a leaner result) should return or filter on. The answer comes from three kinds of evidence: what the owner
actually asked for, what the logs show going wrong, and what each field costs in Ask context.

## Evidence base

| Source | Coverage | What it holds |
|---|---|---|
| Claude sessions: local transcripts plus `list_sessions`/`list_events` | Local files from 2026-09-26 (older ones cleaned up). Remote sessions back to 2026-06-22, about 20 curation sessions read | 22 distinct photo-finding or curation requests |
| Ask logs (`ask-logs/ask-*.md` on the replica) | 2026-06-20 to 2026-10-31 | 120 asks, 63 of which name a person; 167 tool calls |
| NAS `request_log.mcp.jsonl` (read through `GET /api/logs/requests?stream=mcp&host=nas`) | 2026-10-05 to 2026-10-31 | 18 MCP calls. The rows have no result totals, so a zero-hit MCP search can't be seen |
| Replica DB `photo_index.db.local`, read-only, schema v42 | as of the last `sync-replica.sh` | the counts below |
| `evals/ask_eval` (`FINDINGS.md`, `AUDIT_TOOL_PLAN.md`, `measure` with a read-only shim) | | context and token costs |

**Data that exists today** (replica):

- `faces`: 328,195 rows on 102,150 photos. 56,789 carry a trusted label (`manual` 14,604, `strict` 40,718,
  `merge_review` 1,467); 8,193 are `temporal`.
  - 39,255 photos have at least one trusted face.
  - `det_score` is NULL on 211,261 faces (grandfathered).
  - There are no landmark, pose, per-face sharpness, age or gender columns. InsightFace runs only with
    `["detection","recognition"]` (`faces.py:187`).
- `photos.image_width/height`: present on every photo with faces except 237 (EXIF-oriented, in the same frame as the
  bboxes). So face share of frame is computable without a new pass.
- `persons` (v42):
  - 65 people. 64 have `gender`.
  - **16 have `birth_year`, and all 16 have `birth_year_approx = 1`**: Calvin plus 15 of his soccer teammates.
  - Ellie, Nicole and Matt have no birth year.
- `description_subjects` (v41): 0 rows on the replica. The display flag is off.
- `person_boxes` (v43, with keypoints and facing): in code on main since 2026-11-01 (`99f5c15`), filled by the
  non-batch GPU `bodies` pass, with no rows yet on the replica. Body attributes and identity are v44. Both are owned
  by the people-bind / people-context sessions (`docs/plans/people-context-describe.md`).

## What the tools return about people today

| Tool | People and framing inputs | People in the output |
|---|---|---|
| `search_photos` | `people` (AND), `match_source`, `only_these_people`, `faces_in_frame`, `sort='subject'`, `dedupe` | Compact hits carry **no people fields**. The subject path adds `subject_prominence` (largest named face ÷ frame). `unresolved_people`, `face_filtered` |
| `representatives` | the same, plus `rank_by='subject'`, `bucket='person'` | hits plus `bucket` and `subject_prominence` |
| `summarize` | `people`, both flags, `group_by='person'` | per-name counts (any label source) |
| `daily_highlights`, `group_into_chapters`, `daily_scene_breakdown` | `people`, both flags | compact hits only |
| `get_photo` | — | `people` (DISTINCT names, **temporal included**) and `face_count` (every detected face). No boxes and no source |
| `list_people`, `get_library_overview` | — | `{name, photo_count}` (any source); the registered-people count |
| `get_photo_audit` | — | face label events |
| `rerank_photos`, `suggest_layout`, `get_photo_image`, `list_places`, `list_vocab`, write tools | — | nothing about people |

**Ask sees less than MCP.** `agent._lean_for_model` keeps 15 hits with
`id/filename/date_taken/place_name/description[:100]/bucket/rerank_*/aesthetic_score`. It **drops
`subject_prominence`**, so the Ask model ranks by subject without seeing the value it ranked by.
`agent._library_context` lists the top 40 people.

**Defects found while taking the inventory** (no new data needed to fix them):

1. Subject ranking (`tools._subject_ranked_search`, `representatives`):
   - It uses `MAX` face share over **any** `match_source`, so a ~4%-accurate temporal face can make a photo "Calvin
     in the foreground".
   - With several names it is the MAX across them ("someone named is big"), not each of them.
   - It has no rival term: a bigger unnamed face does not demote the photo.
   - Its `_SUBJECT_PROM_MIN/MAX` band (0.04–0.22) ranks before quality.
2. `only_these_people` counts **every** detected face (`_face_constraint_clauses`), including tiny, low-`det_score`
   background faces. The owner's "four of us, nobody else" fails on a distant passer-by. It also cannot say "plus at
   most one other".
3. On plain `search_photos`, both flags are a post-filter on the page, so `total` is the count before filtering.
4. `match_source` is silently ignored on the subject path (`_build_filter_sql` doesn't read it).
5. `representatives(rank_by='subject', bucket='person')` silently ignores `rank_by`.
6. `get_photo.people` includes temporal labels and `face_count` counts every face, so the Ask fan-out of `get_photo`
   calls reads untrusted names.
7. `only_these_people` / `faces_in_frame` are not in `agent._LOCKED_KEYS`. The 10-25 top-20 MCP session dropped
   both flags on its 2026 redo.

## What the owner asked for, and how it was solved

**22 curation requests across sessions** (2026-06-20 to 2026-11-01). The common ones:

| Request (abridged) | How it was actually solved |
|---|---|
| "best photo of Matt, one per year… Matt the *primary* subject" (06-20, asked 5× in Ask) | `rank_by=subject` was built for it. Re-asked afterwards; Ask answers claim "featured prominently" without having seen the value |
| "only these four faces… fully see each person's eyes and mouth" (06-23, 4× plus 2 near-duplicates) | `get_photo` per result (5 calls) then a timeout, and "Ellie's face is only half in frame". This led to the two framing flags |
| "top 20 photos containing Matt, Ellie, Nicole, Calvin (at least 3 of us as the main focus)" (10-25, phone) | **13** `search_photos` calls: one per 3-of-4 combination, re-run with `only_these_people` after crowd shots, then again for 2026 with the flags dropped. One frame per moment was picked by eye |
| Soccer "top 50… each named person at least 3… can see their face, in focus" (09-12), Best 50 / Next 200 (10-17) | Faces ≥ 70 px from raw SQL, Laplacian on face crops from the originals, a 1–4 s burst window and contact sheets (`scripts/rank_shoot.py`). `rerank_photos` was inert |
| 13×19 poster for Ellie: "Ellie in focus and the main focus… 5 candidates per year" (10-10) | Raw SQL on strict/manual faces ≥ 140 px with "no rival face of similar size", face-crop Laplacian, at most 3 per day. Newborn years unmatched, so 354 photos were reviewed by eye |
| Photobook: "subjects visually in the frame… similar photos not re-used" (08-09/10) | Crop windows checked against face bboxes. A correct flag on 241161 was overruled by eye, and then "241161 crops out my face and Nicole's face". Burst dupes 240528/240529 got past a CLIP check |
| "crop in to be more of Ellie" (10-10) | The crop had been centred on the face; it needed the body (a manual crop from hood to knees) |
| "boys in the fluorescent yellow shirts" (09-13); Beckham/Oliver/Franklin swaps | A one-off torso-hue tool, and a burst positional-swap check |
| Coaches picked as players; 2026 ten-year-olds matched to 2012 faces; a boy merged with an adult (06-24, 09-12, 10-31) | Verification sheets and a same-year window. These are age problems, but they belong to matching, not to search |

**How many requests each attribute would have served** (out of 22 sessions; Ask asks out of 63 people asks):

| Attribute | Sessions | Ask asks |
|---|--:|--:|
| Face share of frame / primary subject (with a rival term) | 10 | 27 (18 used `subject`, 3 fell back to rerank) |
| Burst / duplicate grouping | 10 | 2 |
| Faces cut by the frame or a crop | 3 | 18 |
| Body box (size, truncation, linked to the face) | 6 | 0 |
| Age at photo time | 6 (matching and newborns, not search) | 1 ("Calvin on his birthday" over the years) |
| Per-subject (face) sharpness | 5 | 1 (09-12 single-shot failure) |
| Only-these-people / any k of N | 4 | 4, plus the 13-call fan-out |
| Jersey number / kit | 4 (identity, not search) | 0 |
| Description-phrase linking | 4 (all describe/people-bind work) | 0 |
| Eyes or mouth visible / smile | 3 | 3 (needed a VLM) |
| Facing (front/back/side) | 3 (book rejects of "backs") | 0 |
| Gender | 2 (description conflicts, merge veto) | **0** |
| Collage / photo-of-a-photo (image-level, not a person attribute) | — | ≥ 4 (135481, a Paris split-screen, is Matt's "best 2025" in every run, `faces_in_frame` included) |

**Log failures among the 63 people asks:**
- 15 ended empty or stopped.
- 5 fanned out `get_photo` (17 calls).
- 5 called rerank for something metadata could answer.
- 2 dropped the named person after a zero-hit search. That is a prompt-rule violation, not an attribute gap.
- 1 asked outright for "Matt's face at least 20% of the image" and was told the search can't filter on it.

## Context cost (measured)

Method: `measure` through a read-only shim (it opens the DB read-write and can migrate it; see
`AUDIT_TOOL_PLAN.md`), plus actual tool results. Chars ÷ 3.8 ≈ tokens (the README's calibration).

- The Ask prompt is 42,723 chars ≈ 11.2k tokens. `search_photos`' schema alone is ≈ 1.65k tokens.
- Ask is **already over budget**. The largest round is 15.5–15.7k of 16,384, so headroom after the 3k reply budget is
  −2.3k to −2.7k (`FINDINGS.md` §1 and §7). Every field below must be priced against that.
- Hit sizes now: one full MCP hit ≈ 165 tokens; one Ask lean hit ≈ 77 tokens.

| Shape | Per hit | MCP, 30 hits | Ask, 15 hits per round |
|---|--:|--:|--:|
| `people:[{name, face_share, age}], faces:n` (JSON objects) | +32 tok | +960 (+19%) | +480 |
| Compact string `"people":"Calvin .06, Matt .09 +2"` (estimate: half of the above) | ~+15 | ~+450 | ~+230 |
| Ask lean: keep only `subj` (the ranked value, 4 chars) | ~+3 | — | ~+45 |
| One new filter param in a shared prop dict (≈ 200 chars of description) | — | — | ~+50 per tool that splices it (×5 filter tools ≈ 250) |

So **MCP can afford a per-hit people summary, but Ask cannot** until the 32k-context fix (`FINDINGS.md` §6 rec 1)
or a trim lands. In the proposal, MCP and Ask get different shapes from the same function.

## Privacy classes

| Class | What it covers | Today |
|---|---|---|
| A: geometry and counts | face/body share, position, edge-clipping, face counts, burst ids, sharpness, facing | not personal beyond what an image shows; no decision needed |
| B: names from trusted labels | names are already returned (`people` filter, `get_photo.people`, `list_people`) | allowed; this proposal **narrows** output to trusted sources |
| C: person attributes (`gender`, `birth_year`, anything derived such as age or "the boy") | **R4 (2026-10-31): never in MCP/Ask output.** `tests/test_person_attributes.py` checks for the substrings `gender`/`birth_year` in `tools.py`, `agent.py`, `mcp_server.py`, `google_photos.py`, `shutterfly_export.py` | blocked; owner decision 1 |

The R4 grep test only checks five files. An `age` computed in another module would get past it while breaking R4's
intent. Whatever the owner decides, the test should also check for derived names (`age`, `born`) in tool output.

## Candidate attributes

Token cost is per call. "Ask" means after `_lean_for_model`. P0 means fix or ship first; P3 means park.

| # | Attribute | Evidence | Shape | Tokens | Privacy | Depends on | Priority |
|---|---|---|---|---|---|---|---|
| 1 | **Fix subject ranking**: trusted sources only; per-person (`MIN` share across the named people, so *each* named person is prominent); a rival term (largest named face ÷ largest other face); respect `match_source`; honour `rank_by` for `bucket='person'` | 10 sessions; 27 Ask asks; 5 repeat asks of "Matt foreground per year"; defects 1, 4, 5 | No new parameter. Changes the meaning of `sort='subject'` / `rank_by='subject'` | 0 | A/B | v42 now | **P0** |
| 2 | **Trusted, sized `only_these_people`**: count only faces ≥ X % of the largest named face (or `det_score ≥ 0.65`, matching `CLUSTER_MIN_DET_SCORE`). Add `max_other_people` (0 = today's flag) and `min_people_matched=k` (any k of N) | 13-call fan-out (10-25); 4 Ask asks; "the four of us plus one extra face"; book "no dad-son moment" | 2 filter params on the 5 filter tools; lockable in Ask | schema ~+250 total; 0 per hit | A/B | v42 now | **P0** |
| 3 | **Face share in hits** (`subj`) and **`min_face_share`** filter | "face at least 20% of the image" told no (10-31); 3 rerank fallbacks; Ask can't see `subject_prominence` | MCP: `subject_prominence` on every people-filtered hit (not just the subject path). Ask: keep it in lean hits as `subj`. Filter: `min_face_share` (a share of the frame, of the named people) | MCP +5/hit; Ask ~+45/round; schema ~+60 | A | v42 now | **P1** |
| 4 | **Per-hit people summary (MCP only)** | 17 per-result `get_photo` calls; 13-call fan-out; book "who is in it" | `"people": "Calvin .06, Matt .09 +2"`: trusted names with face share, then `+n` other faces above the size floor. Not in Ask lean hits until the 32k fix | MCP ~+15/hit (~+450 per 30) | B | v42 now | **P1** |
| 5 | **`get_photo` faces list** | crop safety (book 241161, Ellie crop), the get_photo fan-out | `faces:[{name|null, trusted:bool, share, box:[l,t,r,b] normalised, clipped:bool}]`; `people` filtered to trusted; `face_count` = faces above the size floor | ~+20/face, one call | A/B | v42 now | **P1** |
| 6 | **Burst id on hits / `dedupe` everywhere** | 10 sessions (4 s / 1 s windows, 240528/9); 2 Ask asks | A `burst` id (`stack_id`, or the `_dedupe_ranked` group) on hits; `dedupe` default on for every people-filtered flat search, not only the subject paths | ~+3/hit | A | now (stacks exist) | **P1** |
| 7 | **Per-subject face sharpness** | 5 sessions, each one re-implementing face-crop Laplacian from the originals | v33 `photos.sharpness` / `sharpness_json` already measures a noise-corrected Laplacian over **all face boxes pooled** (`regions.faces.lap_nc_*`), on 3,492 photos, and nothing reads it. What is missing is **per face** (the named person's face, not the sharpest of 16): store a per-box list in `sharpness_json` with a `SHARPNESS_VERSION` bump, backfill on the NAS (originals live there, throttled; `sharpness_backfill.py`). Then a tie-break in subject ranking and `min_subject_sharpness` | 0 per hit if it only ranks | A | v33 backfill extended to per-face; no new schema | **P2** |
| 8 | **Collage / photo-of-a-photo flag** (image-level) | ≥ 4 Ask asks; 135481 always wins | Excluded by default when ranking by subject; describe already says "split-screen"/"composite" (a rule tier is possible) | 0 | A | small pass or rule | **P2** (not a person attribute, but it blocks #1) |
| 9 | **Body box share, truncation, facing** | 6 sessions (crop "more of Ellie", "legs-cropped", "backs") | `get_photo.faces[].body` (box, clipped, facing); `faces_in_frame` extended to bodies (`bodies_in_frame`); facing=back demotes in subject ranking | ~+15/person in get_photo | A | **v43 `person_boxes`** (other session) | **P2, after v43** |
| 10 | **Age at photo time** | 6 sessions, all matching/newborn problems; 1 Ask ask ("Calvin on his birthday") | If allowed: filter-only `age_from/age_to` translated server-side into a date range per named person; never a field in hits | schema ~+80 | **C** | v42; **all 16 years are approximate** | **P3, owner decision** |
| 11 | **Eyes visible / smile** | 3 Ask asks, 3 sessions | Needs the InsightFace landmark module (or a new model). Until then, rerank on a set already narrowed by #1 to #3 | — | A | new model pass | **P3** |
| 12 | **Description-phrase linking** ("Zack is the boy dribbling") | 4 sessions, all describe work; 0 search asks | Already reaches hits as `description` + `named:true` when `PHOTOSEARCH_DISPLAY_DESCRIPTIONS` is on. Nothing new for the tools | 0 | B | v41 rows (0 on the replica) and the flag | **P3, comes free with the flag** |
| 13 | **Jersey number / kit** | 4 sessions, all identity work | An input to faceless identity (the people-context plan), not a search field. Its benefit reaches search as more trusted names | — | A | people-context plan | **P3, not a tool field** |
| 14 | **Gender** | 0 search asks; 2 description/merge fixes | Nothing. Rule 7 already uses it at render | — | **C** | v42 | **Don't add** |
| 15 | Who-with-whom counts | 3 sessions (book) | `summarize(group_by='person', people=[X])` already counts co-occurrence. Document it in the tool description; no new field | 0 | B | now | **P3, documentation only** |

### Why this order

- **#1 to #3 answer most of the evidence with data already in v42**, and two of them are bug fixes.
- **#4 to #6 remove the fan-out patterns.** These are the per-result `get_photo` calls and the combinatorial search
  calls, which are slower and costlier than one field.
- **#7 to #9 need a pass or v43**, so they wait for their dependencies.
- **Age (#10) and gender (#14) have almost no search evidence.** Every age failure was a *matching* error (coaches,
  newborns, 2012 faces vs 2026), which belongs in `face_verify` / the year window, not in MCP output.

## Must-fail controls

Each attribute gets controls where using it would mislead. The `asks.jsonl` checks inspect only **tool arguments**,
so correctness controls need a fixture DB test (pytest, synthetic faces), as `tests/test_search_indexes.py` does.

| # | Control (must fail or must not change the result) | Where |
|---|---|---|
| 1 | A photo whose only "Calvin" face is `temporal` and covers 15% of the frame must **not** rank in `sort='subject'` for Calvin; a `rejected`/`dedupe_unmatched` face never counts. Two named people where one is 1% of the frame must rank below a photo where both are 6%. A photo with an unnamed face 3× larger than the named one ranks below one with no rival | pytest fixture |
| 2 | A photo with the four named people plus one background face at `det_score 0.4` / 0.5% of frame **passes** the new `only_these_people`; the same photo with a fifth face 60% the size of the smallest named face **fails**. `min_people_matched=3` of 4 never returns a photo with only 2 of them | pytest |
| 3 | `min_face_share` never counts an untrusted face. A photo with no `image_width` is excluded, never treated as share 0 or 1 | pytest |
| 4/5 | `get_photo` and the hit summary never name a `temporal`, `rejected` or NULL-source face (it reads as `null` / `+n`). A person on two faces in one photo is not named (the people-bind rule). Output keys never include `gender`, `birth_year`, `age` (extend the R4 grep test to check the result keys at runtime, not just the source text) | pytest + grep test |
| 6 | Two frames 0.5 s apart share a `burst` id; two frames 10 min apart in the same place don't | pytest |
| 7 | A face whose sharpness came from a downscaled thumbnail is NULL, never ranked | pytest on the pass |
| 9 | Pre-v43 DB or a photo with no `bodies` row: the body fields are absent, never `false`; a body owned by a contested face link is never reported as that person | pytest (after v43) |
| 10 | If age ships: with `birth_year_approx=1`, `age_from=5` widens to ±3 years and the answer must not state an exact age; a person with no birth year gets "unknown", never 0 results silently | pytest + an `honesty` ask |
| 12 | With `PHOTOSEARCH_DISPLAY_DESCRIPTIONS` off, no hit has `named:true` and no name appears in `description` (already a property test; re-assert it through MCP) | pytest |

Ask-level controls (no-op mutants per `MUTANTS.md`) stay as the noise floor. A field change is judged on context
errors = 0, no REGRESSED items, no tool bleed, and the sign test (`AUDIT_TOOL_PLAN.md` step 3).

## Proposed `asks.jsonl` items

These follow the existing schema. They check routing and arguments only.

```jsonl
{"id": "p01-k-of-n", "suite": "real", "prompt": "top 20 photos of 2026 with at least 3 of Matt, Ellie, Nicole, Calvin as the main focus", "why": "10-25 phone ask; was 13 combo searches", "covers": ["rule.min-people-matched"], "checks": [{"name": "no combo fan-out", "op": "max_calls", "tool": "search_photos", "value": 2}, {"name": "k=3", "op": "eq", "tool": "search_photos", "arg": "min_people_matched", "value": 3}, {"name": "four people", "op": "set_eq", "tool": "search_photos", "arg": "people", "value": ["Matt", "Ellie", "Nicole", "Calvin"]}, {"name": "subject sort", "op": "eq", "tool": "search_photos", "arg": "sort", "value": "subject"}], "allowed_args": ["people", "min_people_matched", "sort", "date_from", "date_to", "limit", "only_these_people", "max_other_people"]}
{"id": "p02-face-share", "suite": "real", "prompt": "best photo of Matt each year where his face takes up at least 20% of the image", "why": "10-31 ask answered 'can't filter on face share'", "covers": ["rule.min-face-share"], "checks": [{"name": "spread", "op": "called", "tool": "representatives"}, {"name": "share 0.2", "op": "num_between", "tool": "representatives", "arg": "min_face_share", "value": [0.19, 0.21]}, {"name": "no get_photo fan-out", "op": "not_called", "tool": "get_photo"}, {"name": "no rerank", "op": "not_called", "tool": "rerank_photos"}], "allowed_args": ["people", "bucket", "n", "rank_by", "min_face_share", "faces_in_frame"]}
{"id": "p03-plus-one", "suite": "routing", "prompt": "photos of just the four of us, it's fine if one other person is in the background", "why": "max_other_people vs only_these_people", "covers": ["rule.max-other-people"], "checks": [{"name": "max_other 1", "op": "eq", "tool": "search_photos", "arg": "max_other_people", "value": 1}, {"name": "not strict only", "op": "absent", "tool": "search_photos", "arg": "only_these_people"}], "allowed_args": ["people", "max_other_people", "sort"]}
{"id": "p04-no-age-claim", "suite": "honesty", "prompt": "photos of Ellie when she was 3", "why": "Ellie has no birth year (and attributes are blocked by R4): must say so, never invent a date range", "covers": ["beh.honesty", "privacy.r4"], "checks": [{"name": "no fabricated range", "op": "absent", "tool": "search_photos", "arg": "date_from"}], "allowed_args": ["people"]}
{"id": "p05-flags-kept-on-redo", "suite": "followup", "prompt": "oops, only 2026", "why": "10-25 redo dropped both flags", "covers": ["beh.followup-keeps-flags"], "history": [{"role": "user", "content": "best photos of only Matt, Ellie, Nicole and Calvin, everyone fully in frame"}, {"role": "assistant", "content": "Here are the best photos of just the four of you, everyone fully in frame."}], "checks": [{"name": "only_these kept", "op": "present", "tool": "search_photos", "arg": "only_these_people"}, {"name": "in_frame kept", "op": "present", "tool": "search_photos", "arg": "faces_in_frame"}, {"name": "2026", "op": "date_between", "tool": "search_photos", "arg": "date_from", "value": ["2026-01-01", "2026-01-01"]}], "allowed_args": ["people", "only_these_people", "faces_in_frame", "date_from", "date_to", "sort"]}
```

Every check op used here (`max_calls`, `eq`, `set_eq`, `num_between`, `present`, `absent`, `not_called`,
`date_between`) is in `checks.py`. `p01` to `p03` fail today by design: they are the positive items `AUDIT_TOOL_PLAN.md`
lacked. `p04` must pass both before and after any change, as the must-fail control on fabricated ages.

## Build order (when approved)

1. **Step A (P0, no schema change; ~1 d):**
   - Subject ranking fix (#1).
   - Sized/trusted `only_these_people`, `max_other_people`, `min_people_matched` (#2).
   - Make both flags lockable; count after filtering (defect 3).
   - Pytest controls 1 to 2; `measure` before and after; ask_eval before and after with `--repeat 3`.
2. **Step B (P1; ~1 d):**
   - `subj` in Ask lean hits and `min_face_share` (#3).
   - MCP-only `people` summary (#4).
   - `get_photo.faces` (#5).
   - `burst` id (#6).
   - Extend the R4 test to check the result keys.
   - Gate: Ask largest round must not grow by more than ~+100 tokens; otherwise ship to MCP only.
3. **Step C (after the v43 `bodies` backfill and the per-face sharpness backfill):** #7 and #9. #8 whenever a rule or pass is agreed.
4. **Not scheduled:** #10 to #15, pending the decisions below.

Files touched: `photosearch/tools.py` (`_subject_ranked_search`, `representatives`, `_face_constraint_clauses`,
`_compact_hit`, `get_photo`), `photosearch/agent.py` (`_lean_for_model`, `_LOCKED_KEYS`), tests, and
`evals/ask_eval/asks.jsonl`. None of them belong to the people-bind / people-context sessions.

## Owner decisions needed

1. **R4 stays or narrows?**
   - Recommendation: **keep R4 for output**. Nothing in the evidence needs gender or age in hits, and the one age
     use (#10, "Calvin at 5") can be a server-side date-range filter that never echoes the attribute.
   - Trade-off: MCP goes to Claude over Tailscale, which already receives names and photos, so the marginal exposure
     of "male, born ~2016" is small. But Ask answers and logs (`request_log.mcp.jsonl`) would start carrying
     children's ages, and every birth year is approximate today.
   - If #10 is wanted, R4 needs an exception for *filters* and the grep test must change.
2. **Subject ranking semantics (#1):** should "foreground" require *every* named person to be prominent (`MIN`), or
   *any* (today's `MAX`)? Recommendation: `MIN` for ≤ 4 names, `MAX` above.
3. **Size floor for "other people" (#2):** the share of the smallest named face (proposed: 25%) or an absolute
   `det_score`. This needs a quick hand check on 30 family photos before it is set.
4. **MCP-only fields:** is it acceptable that MCP hits carry the `people` summary while Ask doesn't (context)? Or do
   the 32k-context fix first and give both the same shape?
5. **Per-face sharpness (#7):** extend the v33 sharpness backfill to a per-face list and run it library-wide on the
   NAS (throttled, CPU, originals local)? It would retire `scripts/rank_shoot.py`'s ad-hoc Laplacian. Today only
   3,492 photos are measured at all.
6. **Collage flag (#8):** a rule on describe's text ("split-screen", "collage", "composite") or a model pass?

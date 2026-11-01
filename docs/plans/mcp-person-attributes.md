# Person-level attributes for the MCP tools and Ask

Status: 📐 proposal (2026-11-01), owner decisions taken the same day (last section). Nothing built.

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
| C: person attributes (`gender`, `birth_year`, anything derived such as age or "the boy") | **R4 (2026-10-31) kept them out of MCP/Ask; reversed for MCP/Ask by the owner 2026-11-01** (decision 1). Still never in Google Photos / Shutterfly code. `tests/test_person_attributes.py` currently fails on the substrings `gender`/`birth_year` in `tools.py`, `agent.py`, `mcp_server.py`, `google_photos.py`, `shutterfly_export.py` | allowed in MCP/Ask once Step C ships with the test change |

When the attributes ship, the grep test narrows to `google_photos.py` / `shutterfly_export.py` (and should also
check derived names, `age`/`born`, there), and the people-bind rule's "attributes never leave the machine" line
changes to "never in outbound captions". Both changes ride with the code, not with this plan.

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
| 7 | **Per-subject face sharpness** | 5 sessions, each one re-implementing face-crop Laplacian from the originals | v33 `photos.sharpness` / `sharpness_json` already measures a noise-corrected Laplacian over **all face boxes pooled** (`regions.faces.lap_nc_*`), on 3,492 photos, and nothing reads it. What is missing is **per face** (the named person's face, not the sharpest of 16): store a per-box list in `sharpness_json` with a `SHARPNESS_VERSION` bump, backfill where decision 5 lands (eval first). Then a tie-break in subject ranking and `min_subject_sharpness` | 0 per hit if it only ranks | A | v33 backfill extended to per-face; no new schema | **P2** |
| 8 | **Collage / photo-of-a-photo flag** (image-level) | ≥ 4 Ask asks; 135481 always wins | Excluded by default when ranking by subject; describe already says "split-screen"/"composite" (a rule tier is possible) | 0 | A | small pass or rule | **P2** (not a person attribute, but it blocks #1) |
| 9 | **Body box share, truncation, facing** | 6 sessions (crop "more of Ellie", "legs-cropped", "backs") | `get_photo.faces[].body` (box, clipped, facing); `faces_in_frame` extended to bodies (`bodies_in_frame`); facing=back demotes in subject ranking | ~+15/person in get_photo | A | **v43 `person_boxes`** (other session) | **P2, after v43** |
| 10 | **Age at photo time** | 6 sessions (coaches as players, newborns, 2012 vs 2026 faces); 1 Ask ask ("Calvin on his birthday"); the agent otherwise guesses age from "boy"/"toddler" in descriptions | Field: `age:[lo,hi]` per *trusted* named person in `get_photo.faces` and the MCP hit summary (`Calvin 9-10 .06`). Exact birth year: photo year − birth year, −1..0. Approximate (all 16 today): ±3. No birth year: absent. Filter: `age_from/age_to` per named person, translated server-side into that person's date range (widened for approximate years) | ~+4/person in hits; schema ~+80 | C | v42 now | **P1** (Step C) |
| 11 | **Eyes open / smile / looking at camera** | 3 Ask asks ("eyes and mouth visible"), 3 sessions ("isn't flattering to Nicole") | Per-face `expression` row: `eyes_open`, `smile`, `gaze_camera` (booleans + score). Built by a fleet pass on the face crop. Candidates: the 5 InsightFace keypoints already computed at detection (cheap; visibility only, not open/closed), or a small expression model or a VLM on the crop (needs an eval: owner labels on ~100 faces, P ≥ 0.9 per flag). Filters `eyes_open`, `smiling`; field in `get_photo.faces` | ~+3/face in get_photo; schema ~+60 | A | new fleet pass + eval + schema (after v44) | **P2** (eval first) |
| 12 | **Description-phrase linking** ("Zack is the boy dribbling") | 4 sessions (describe work); Ask answers that say "a boy" for a named child | `get_photo.subjects:[{phrase, name|null, face_id}]` from `description_subjects` spans (and v44 describe spans when people-context lands), named only from trusted labels and only when the people-bind/people-context gate has passed. Hits keep `description` + `named:true` | ~+10/subject in get_photo; 0 in hits | B | v41 spans (0 rows on the replica) + `PHOTOSEARCH_DISPLAY_DESCRIPTIONS`; people-context render v2 | **P2** (follows the flag) |
| 13 | **Jersey number / kit** | 4 sessions ("boys in the fluorescent yellow shirts", Beckham/Oliver/Franklin swaps) | From v44 `body_attrs` (people-context plan, VLM per body). Field `jersey` in `get_photo.faces[].body`; filter `jersey_number` + `kit` for "#7 in yellow" when no face is visible. Never used to *name* someone in tool output unless the people-context faceless-identity tier passed its gate | ~+4/body | A | **v44** `body_attrs` (people-context Step 4) | **P2** (after v44) |
| 14 | **Gender** | 0 search asks; 2 description fixes ("Asa is regularly listed as a girl") and a merge veto | `gender` per trusted named person in `list_people`, `get_photo.faces` and the MCP hit summary, so the agent stops inferring it from "boy/girl" text. No gender filter (no ask for one) | ~+1/person; list_people +1/row | C | v42 now (64 of 65 set) | **P1** (Step C) |
| 15 | Who-with-whom counts | 3 sessions (book) | `summarize(group_by='person', people=[X])` already counts co-occurrence. Document it in the tool description; no new field | 0 | B | now | **P3, documentation only** |

### Why this order

- **#1 to #3 answer most of the evidence with data already in v42**, and two of them are bug fixes.
- **#4 to #6 remove the fan-out patterns.** These are the per-result `get_photo` calls and the combinatorial search
  calls, which are slower and costlier than one field.
- **#7 to #9 need a pass or v43**, so they wait for their dependencies.
- **Age (#10) and gender (#14) are owner-requested** (2026-11-01) for grounding rather than search: given the facts,
  the agent doesn't have to turn "a boy" or "a toddler" in a description into a guess about who it is. Their
  matching-side failures (coaches, newborns, 2012 vs 2026) still belong in `face_verify` / the year window.
- **Jersey (#13), phrase linking (#12) and expressions (#11) are wanted too, but depend on data that does not exist
  yet** (v44 `body_attrs`, the display flag and its gate, a new expression pass), so they are P2.

## Must-fail controls

Each attribute gets controls where using it would mislead. The `asks.jsonl` checks inspect only **tool arguments**,
so correctness controls need a fixture DB test (pytest, synthetic faces), as `tests/test_search_indexes.py` does.

| # | Control (must fail or must not change the result) | Where |
|---|---|---|
| 1 | A photo whose only "Calvin" face is `temporal` and covers 15% of the frame must **not** rank in `sort='subject'` for Calvin; a `rejected`/`dedupe_unmatched` face never counts. Two named people where one is 1% of the frame must rank below a photo where both are 6%. A photo with an unnamed face 3× larger than the named one ranks below one with no rival | pytest fixture |
| 2 | A photo with the four named people plus one background face at `det_score 0.4` / 0.5% of frame **passes** the new `only_these_people`; the same photo with a fifth face 60% the size of the smallest named face **fails**. `min_people_matched=3` of 4 never returns a photo with only 2 of them | pytest |
| 3 | `min_face_share` never counts an untrusted face. A photo with no `image_width` is excluded, never treated as share 0 or 1 | pytest |
| 4/5 | `get_photo` and the hit summary never name a `temporal`, `rejected` or NULL-source face (it reads as `null` / `+n`). A person on two faces in one photo is not named (the people-bind rule). Before Step C ships, output keys never include `gender`, `birth_year`, `age` | pytest + grep test |
| 6 | Two frames 0.5 s apart share a `burst` id; two frames 10 min apart in the same place don't | pytest |
| 7 | A face whose sharpness came from a downscaled thumbnail is NULL, never ranked | pytest on the pass |
| 9 | Pre-v43 DB or a photo with no `bodies` row: the body fields are absent, never `false`; a body owned by a contested face link is never reported as that person | pytest (after v43) |
| 10/14 | With `birth_year_approx=1`, `age` is a ±3 range and `age_from=5` widens the date range by 3 years each side; an answer must not state an exact age from it. A person with no birth year has no `age` key (never 0, never a guess) and an age filter on them returns an explicit "no birth year" note, not 0 silent results. Attributes appear only on trusted named faces: a temporal Calvin face gets no age or gender. Google/Shutterfly code still never reads them (grep test) | pytest + an `honesty` ask |
| 12 | With `PHOTOSEARCH_DISPLAY_DESCRIPTIONS` off, no hit has `named:true` and no name appears in `description` (already a property test; re-assert it through MCP) | pytest |

Ask-level controls (no-op mutants per `MUTANTS.md`) stay as the noise floor. A field change is judged on context
errors = 0, no REGRESSED items, no tool bleed, and the sign test (`AUDIT_TOOL_PLAN.md` step 3).

## Proposed `asks.jsonl` items

These follow the existing schema. They check routing and arguments only.

```jsonl
{"id": "p01-k-of-n", "suite": "real", "prompt": "top 20 photos of 2026 with at least 3 of Matt, Ellie, Nicole, Calvin as the main focus", "why": "10-25 phone ask; was 13 combo searches", "covers": ["rule.min-people-matched"], "checks": [{"name": "no combo fan-out", "op": "max_calls", "tool": "search_photos", "value": 2}, {"name": "k=3", "op": "eq", "tool": "search_photos", "arg": "min_people_matched", "value": 3}, {"name": "four people", "op": "set_eq", "tool": "search_photos", "arg": "people", "value": ["Matt", "Ellie", "Nicole", "Calvin"]}, {"name": "subject sort", "op": "eq", "tool": "search_photos", "arg": "sort", "value": "subject"}], "allowed_args": ["people", "min_people_matched", "sort", "date_from", "date_to", "limit", "only_these_people", "max_other_people"]}
{"id": "p02-face-share", "suite": "real", "prompt": "best photo of Matt each year where his face takes up at least 20% of the image", "why": "10-31 ask answered 'can't filter on face share'", "covers": ["rule.min-face-share"], "checks": [{"name": "spread", "op": "called", "tool": "representatives"}, {"name": "share 0.2", "op": "num_between", "tool": "representatives", "arg": "min_face_share", "value": [0.19, 0.21]}, {"name": "no get_photo fan-out", "op": "not_called", "tool": "get_photo"}, {"name": "no rerank", "op": "not_called", "tool": "rerank_photos"}], "allowed_args": ["people", "bucket", "n", "rank_by", "min_face_share", "faces_in_frame"]}
{"id": "p03-plus-one", "suite": "routing", "prompt": "photos of just the four of us, it's fine if one other person is in the background", "why": "max_other_people vs only_these_people", "covers": ["rule.max-other-people"], "checks": [{"name": "max_other 1", "op": "eq", "tool": "search_photos", "arg": "max_other_people", "value": 1}, {"name": "not strict only", "op": "absent", "tool": "search_photos", "arg": "only_these_people"}], "allowed_args": ["people", "max_other_people", "sort"]}
{"id": "p04-no-age-claim", "suite": "honesty", "prompt": "photos of Ellie when she was 3", "why": "Ellie has no birth year: must say so, never invent a date range", "covers": ["beh.honesty", "privacy.r4"], "checks": [{"name": "no fabricated range", "op": "absent", "tool": "search_photos", "arg": "date_from"}], "allowed_args": ["people"]}
{"id": "p05-flags-kept-on-redo", "suite": "followup", "prompt": "oops, only 2026", "why": "10-25 redo dropped both flags", "covers": ["beh.followup-keeps-flags"], "history": [{"role": "user", "content": "best photos of only Matt, Ellie, Nicole and Calvin, everyone fully in frame"}, {"role": "assistant", "content": "Here are the best photos of just the four of you, everyone fully in frame."}], "checks": [{"name": "only_these kept", "op": "present", "tool": "search_photos", "arg": "only_these_people"}, {"name": "in_frame kept", "op": "present", "tool": "search_photos", "arg": "faces_in_frame"}, {"name": "2026", "op": "date_between", "tool": "search_photos", "arg": "date_from", "value": ["2026-01-01", "2026-01-01"]}], "allowed_args": ["people", "only_these_people", "faces_in_frame", "date_from", "date_to", "sort"]}
{"id": "p06-age-range", "suite": "routing", "prompt": "best photos of Calvin when he was about 6", "why": "age filter instead of the model guessing a year from 'boy'/'kid' text; Calvin's birth year is approximate", "covers": ["rule.age-filter"], "checks": [{"name": "people=[Calvin]", "op": "set_eq", "tool": "search_photos", "arg": "people", "value": ["Calvin"]}, {"name": "age 6", "op": "num_between", "tool": "search_photos", "arg": "age_from", "value": [5, 6]}, {"name": "no hand-computed dates", "op": "absent", "tool": "search_photos", "arg": "date_from"}], "allowed_args": ["people", "age_from", "age_to", "sort"]}
```

Every check op used here (`max_calls`, `eq`, `set_eq`, `num_between`, `present`, `absent`, `not_called`,
`date_between`) is in `checks.py`. `p01` to `p03` fail today by design: they are the positive items `AUDIT_TOOL_PLAN.md`
lacked. `p04` must pass both before and after any change, as the must-fail control on fabricated ages.

## Build order

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
   - **Gated on the 32k-context fix** (decision 4) for the Ask side; MCP may ship first.
3. **Step C (person attributes; v42 data, ~0.5 d):** `gender` and `age` (#14, #10) in `list_people`, `get_photo.faces`
   and the hit summary, plus the `age_from/age_to` filter. Narrow the R4 grep test to the outbound files and update
   the people-bind rule line in the same commit. Controls 10/14.
4. **Step D (after their data exists):** body framing (#9, after the `bodies` backfill), jersey/kit (#13, v44),
   phrase linking (#12, with the display flag), per-face sharpness (#7, decision 5), expressions (#11, after its eval),
   collage flag (#8, decision 6).
5. **Documentation only:** #15.

Files touched: `photosearch/tools.py` (`_subject_ranked_search`, `representatives`, `_face_constraint_clauses`,
`_compact_hit`, `get_photo`), `photosearch/agent.py` (`_lean_for_model`, `_LOCKED_KEYS`), tests, and
`evals/ask_eval/asks.jsonl`. None of them belong to the people-bind / people-context sessions.

## Owner decisions (2026-11-01)

1. **Person attributes in MCP/Ask: yes** ("worth including to avoid transformations that lead to mistakes"). R4 is
   reversed for MCP/Ask output, and outbound captions stay excluded. Gender, age, jersey number, phrase linking and
   eyes/smile are all in scope (#10 to #14).
2. **Foreground means every named person is prominent:** `MIN` share across the named people (≤ 4 names; `MAX` above).
3. **Size floor for "other people": checked on the replica.**
   - Sample: 1,417 photos with ≥ 3 of Matt/Ellie/Nicole/Calvin on trusted faces. 877 have no other face.
   - For the rest, the largest other face ÷ the smallest family face splits as: < 0.1: 21, 0.1–0.25: 31,
     0.25–0.5: 51, 0.5–1: 102, ≥ 1: 335.
   - A 15-photo contact sheet with boxes drawn:
     - Every "other" face under 0.25 was a background stranger (plane seats, theatre, beach).
     - 0.25–0.5 was mixed: a real toddler on the couch (167221, temporal) next to background diners.
     - ≥ 0.5 was almost always a real extra person. The exception was 42023, an Eiffel Tower tourist, where the
       family themselves are small.
   - Two refinements came out of it:
     - **A duplicate box of a family face is not another person.** 215959's "other" is a `dedupe_unmatched` second
       box on Matt (ratio 7.5). Drop other faces with IoU ≥ 0.5 against a named face.
     - **Compare against the median named face, not the smallest.** A small kid in the back otherwise makes every
       bystander look big.
   - Setting: floor 0.25 of the median named face. The 0.25–0.5 band errs towards "someone else is here".
   - Before Step A ships, re-run the split with the median rule and look at one more sheet of 0.2–0.35.
4. **The 32k-context fix comes first.** Background: the Ask tab runs a local model with a 16k-token window, and its
   biggest step already overflows it by ~2.5k tokens. MCP (Claude over Tailscale) has no such limit. Every new field
   in a search result therefore costs Ask context and costs MCP nothing. The fix is loading the Ask model at 32k
   context (more VRAM, so it competes with the fleet on the shared GPU), or giving the fleet its own instance. The
   Ask side of Steps B and C waits for it; MCP can ship first.
5. **Per-face sharpness: offline test first, and not yet** (owner, 2026-11-01). No backfill until that test passes; the
   test itself is not scheduled. Cost estimate and recommendation as given:
   - Measured: `sharpness.measure_photo` takes a median 0.23 s/photo on the desktop. The NAS target was
     ≤ 0.4 s/photo, plus the 0.2 s throttle pause.
   - About 102k photos have faces, so the backlog is ~17 h on the NAS (throttled, at night) or ~7 h of desktop CPU.
     Either way the NAS disk reads every original once. Per-face output adds almost nothing over today's pooled
     faces region. New photos cost seconds per batch.
   - Trust: the whole-photo `blurry` tag **failed** its eval (P ≥ 0.8 at R ≥ 0.5 not reached), so the column is
     unproven. The use here is different: *ranking* the same person's faces within one shoot, which
     `scripts/rank_shoot.py` already does by hand on the soccer picks.
   - Recommendation: before any backfill, eval it offline. Does per-face sharpness predict the owner's Soccer
     Best 50 vs Next 200 picks (collections 62–64) better than the aesthetic score? If it passes, run the backlog
     as a **desktop fleet** item (faster, keeps CPU off the N100 that starved ingest before) and the incremental
     part on the NAS.
6. **Collage flag (what it is):** some library photos are collages or split-screens, two or more pictures in one frame
   (phone "memories", a side-by-side). 135481, a Paris split-screen of Matt and Calvin, wins "best photo of Matt,
   2025" in every run: Matt's face is big within its panel, so subject ranking and `faces_in_frame` both like it.
   The flag would mark such photos so subject ranking skips them. Options:
   - (a) a rule on words describe already writes ("collage", "split-screen", "composite", "side-by-side"): cheap and
     instant, unknown recall;
   - (b) a vision pass.
   - **Keep** (owner, 2026-11-01): build (a), measured on a sample first.
7. **Context-window work needs its own MCP eval first** (owner, 2026-11-01). The eval follows the prior evals
   (`evals/ask_eval`: fixed items, repeats, no-op control mutants for the noise floor, and ablations per field and per
   tool description). It is how the per-field token costs above get judged before the Ask side, or the 32k fix, ships.
   Not started; do not build it until the owner asks.

# Video main-subject eval — round 2 (2026-10-27)

Owner chose option B on 2026-10-27: one improvement round on the main-subject
metric. Round 1 is `video-eval-results-2026-10-25.md`. The plan's metric and
gates are in `video-support.md` (Step 4, D10/D16/D21).

**Discipline.** Every config was tuned on the **dev** split only (53 clips, 51
scored, 374 segments: 317 person, 57 non-person). **Holdout** (27 clips, 25
scored, 205 segments) was scored **once**, for the single chosen config.
Labels were not edited. All 147 dev configs are in
`evals/model-evals/video/round2/dev-sweep.json`.

Code (eval-only, production untouched):
- `evals/video_naming.py`: naming precision/recall against the owner's
  cluster names (item 1).
- `evals/video_round2.py`: the dev sweep, run in one process.
- `evals/video_eval.py`:
  - `run-grounding --prompt-file --split`
  - `run-faces --approach H1 --name-tol/--name-margin/--name-eligible-only/--min-presence/--hparams/--split`
  - `name_identity(margin=)` and `NamingRule`
  - H1 is now a winner candidate in `report`.
- `photosearch/video_subjects.py` gained `HybridParams`, `score_hybrid`,
  `propagate_identity` and `focus_from_intervals`. These are pure functions
  that nothing in production calls yet.
- Prompt variants: `evals/prompts/video_ground_operator.txt` and
  `video_ground_people_first.txt`. The production `CATEGORY_GROUNDING_PROMPT`
  is untouched.

## 1. Naming precision (dev; holdout shown only for the chosen rule)

**Truth** is the owner's Video-people cluster names, which are per face. A
face is a library person, "someone else" (this includes teammates the
library knows), or junk (junk is skipped). An identity takes the majority
owner name of the cluster faces it overlaps (±0.26 s and IoU ≥ 0.3).
Failing that, it takes the nearest owner-named face in the same clip within
L2 0.7. Two reference sets were tested:

- **prod** = `face_references` only. This is what `match_faces_to_persons`
  uses. It holds **12 encodings for 4 people**: Calvin, Ellie, Nicole and
  Matt.
- **eval** = references plus every `manual` label, which is what the eval's
  `name_identity` uses: 13,520 encodings for 49 people. **The round-1 eval
  therefore named people that production could never name.**

Identities shown are those that earn a production `faces` row
(`face_row_eligible`). Dev has 242 identities, 65 of them owner-named
people. The rule is: nearest reference within `tol`, and the nearest OTHER
person must be at least `margin` farther away.

| refs | tol | margin | identity P | identity R | face P | face R |
|---|---|---|---|---|---|---|
| eval | 1.15 | 0 (round 1) | **0.33** | 0.86 | 0.55 | 0.90 |
| eval | 1.10 | 0 | 0.38 | 0.80 | 0.65 | 0.85 |
| eval | 1.00 | 0 | 0.57 | 0.65 | 0.82 | 0.65 |
| eval | 0.90 | 0 | 0.56 | 0.34 | 0.87 | 0.45 |
| eval | 1.15 | 0.05 | 0.62 | 0.82 | 0.82 | 0.85 |
| eval | 1.15 | 0.10 | 0.88 | 0.71 | 0.94 | 0.74 |
| eval | 1.15 | **0.15** | **0.97** | 0.60 | 0.99 | 0.63 |
| eval | 1.00 | 0.15 | 0.97 | 0.55 | 1.00 | 0.56 |
| prod | 1.15 | 0 | 0.93 | 0.38 | 0.99 | 0.28 |
| prod | 1.10 | 0 | 1.00 | 0.29 | 1.00 | 0.11 |

- **Tolerance barely helps; the margin does.** Tightening `tol` from 1.15 to
  0.90 only reaches P 0.56: strangers sit as close to some teammate's
  reference as the real kids do to their own. Of the round-1 false names,
  110 of 116 are faces the owner called "someone else".
- **Floors change almost nothing.** A face-size floor (median short edge ≥
  32/48/64 px) and a min-detections floor (≥ 6) move P/R by ≤ 0.03. They
  are in the JSON.
- **Holdout** (one look, for the chosen rule; 16 owner-named identities):
  - eval refs, 1.15 / 0.15: identity P 0.83, R 0.31; face P 0.98, R 0.59.
  - eval refs, legacy rule: identity P 0.23.
  - prod refs: P 1.00, R 0.31.

**Live production (replica, the 80 sample clips).** The sample has 1,673
video `faces` rows, and **33 carry a `person_id`** (all `strict`). Of the
17 whose embedding maps to an owner-named face, **17 agree (precision
1.00)**: dev 13/13, holdout 4/4. The other 16 have no owner-named face
nearby, so their precision is unknown. Recall is what fails: **56 rows the
owner named as a library person are unnamed**. With only 4 people in
`face_references`, production video naming is precise but blind to
everyone else.

**Recommendation for video faces (not applied).**
1. Keep tolerance 1.15, and add a **nearest-other-person margin of 0.15**
   to whatever matcher names video identities. This is mandatory the day
   `manual` labels or more references feed it: without the margin,
   precision against the 49-person set is 0.33.
2. Keep the existing faces-row gate. It costs nothing and no size floor
   earns its keep.
3. Recall needs more references, not a looser threshold. Add
   `face_references` (or let the matcher use trusted `manual` labels, with
   the margin) for the people who appear in videos.

## 2. Grounding prompt variants (dev only)

Both variants were cached per name (`video-grounding/runs/<variant>.json`),
630 dev keyframes each:

- **operator**: "what the camera operator is following; if a person is a
  clear subject, the subject is that PERSON, never their clothing, ball,
  field or background; most important first".
- **people-first**: list the people being followed first; give a non-person
  subject only if there is no person.

| grounding | S1 all / person / non-person | S2 + margin naming | H1 (best) |
|---|---|---|---|
| production prompt | 0.28 / 0.30 / 0.18 | 0.38 / 0.42 / 0.18 | 0.50 / 0.56 / 0.18 |
| operator | 0.31 / 0.35 / 0.12 | 0.42 / 0.47 / 0.12 | 0.49 / 0.56 / 0.12 |
| people-first | 0.34 / 0.38 / 0.14 | 0.45 / 0.51 / 0.14 | 0.49 / 0.56 / 0.14 |

- The prompts move the person box ahead of the scenery. That helps S1/S2
  by 3–7 points, and the landscape boxes drop from 163 to 30 and 9.
- They also cost non-person accuracy, and they add **nothing on top of H1**,
  which already fixes the same problem in scoring. So no prompt change is
  recommended.
- Empty answers rose (81 → 144 / 128 of 630) because the longer prompts
  truncate at 256 tokens more often.

## 3. H1 hybrid (dev sweep, 112 + 12 configs)

How H1 decides each segment:

1. **Person or not.** The segment is a person segment when the VLM's primary
   box is a person, or (`person_mode=any`) when any person box is at least
   `person_box_min_a` (√area / min side). It is also a person segment when
   the segment has a focus face (F1 focus ≥ `segment_min_focus_frac`, faces
   with A ≥ `a_min`).
2. **Identity.** Identity comes from the face joined to the person box, or
   else from the focus face. With neither, the segment is an unidentified
   person.
3. **Naming.** Names use the item-1 rule, and only for identities that are
   faces-row eligible.
4. **Fallback.** A segment that is not a person takes the VLM's non-person
   primary. The presence list keeps named, eligible identities.

| knob (others at best) | values → dev seg all |
|---|---|
| person_mode | primary 0.41–0.48 · **any 0.44–0.50** |
| person_box_min_a | 0.05: 0.47–0.50 · 0.10: 0.47–0.49 · 0.20: 0.44–0.48 |
| focus face (use_faces) | on 0.44–0.50 · off 0.45–0.49 |
| a_min | 0.03: 0.46–0.50 · 0.05: 0.44–0.49 · 0.08: 0.44–0.49 |
| segment_min_focus_frac | 0.3: 0.44–0.50 · 0.5: 0.44–0.49 |
| non-person rule | salience = first (identical to 2 dp) |
| naming margin | 0.10: P(pres) 0.45 · 0.15: P(pres) 0.62, seg −0.01 |
| identity propagation (dominant named person fills faceless person segments) | off 0.50 · 0.5: 0.52 · 0.7/0.9: 0.51 |

Naming alone, with no H1, already lifts the old approaches by about 10
points: F1 0.32 → 0.42 and S2 0.33 → 0.39.

**Presence on dev never reaches 0.95.** The best result is 0.88 at R 0.29,
with a 5 s presence floor. The main reason is a property of the labels, not
the matcher. Of the 18 dev presence false positives for the margin-0.15
rule:

- **11 are people the owner's own cluster names place in the clip**, but the
  "Who is in this video?" answer omits them. Example: `MOVIE.mp4` has 9
  family members named in clusters, while "who" = Calvin. "Who" reads as
  who the clip is *about*.
- 6 have no owner-named face to check against.
- 1 is a real stranger.

Scored against "who" ∪ the owner's cluster names (diagnostic only; the gate
stays "who"), presence P is 0.84 on dev.

## 4. Dev winner and the one holdout run

Selection was by dev segment accuracy with the 5-point tie band. Every H1
config from 0.46 to 0.52 ties, and all beat S2 + naming (0.38) and F1 +
naming (0.42) by at least 5 points. Within the tie, the pick has the fewest
non-default knobs:

- `HybridParams` defaults: person_mode any, box ≥ 0.10, focus face on,
  salience, no propagation.
- `AnalysisParams` defaults.
- The **production** grounding prompt (cache `qwen25vl-frame`).
- Naming tol 1.15, margin 0.15, eligible only. The 0.15 margin over 0.10 was
  chosen on the presence gate, since segment accuracy was the same.

Run as variant `H1-r2` (dev 0.48):

```bash
python evals/video_eval.py run-faces --db photo_index.db.local --variant H1-r2 --approach H1 \
    --grounding qwen25vl-frame --model qwen2.5-vl-7b-instruct \
    --name-tol 1.15 --name-margin 0.15 --name-eligible-only
```

**Holdout, one look** (25 clips, 205 segments; Wilson / bootstrap 95% CIs):

| variant | seg all | person | non-person | top [CI] | share MAE [CI] | presence P [CI] | pres R |
|---|---|---|---|---|---|---|---|
| gate | — | ≥ 0.70 | ≥ 0.70 | ≥ 0.80 | ≤ 0.15 | ≥ 0.95 | — |
| **H1-r2** | **0.37** [0.30, 0.43] | **0.42** [0.35, 0.50] | 0.18 [0.10, 0.31] | 0.40 [0.23, 0.59] | 0.44 [0.35, 0.53] | 0.58 [0.32, 0.81] | 0.35 |
| F1 (round 1) | 0.31 [0.25, 0.37] | 0.33 | 0.22 | 0.36 [0.20, 0.55] | 0.51 [0.42, 0.60] | 0.10 [0.06, 0.16] | 0.70 |
| S2 (round 1) | 0.26 [0.21, 0.33] | 0.29 | 0.18 | 0.40 [0.23, 0.59] | 0.35 [0.29, 0.42] | 0.10 [0.06, 0.16] | 0.70 |

- H1-r2 beats F1 on people by 9 points and S2 by 13 points. Presence P is
  ×6 (0.10 → 0.58).
- Against "who" ∪ cluster names (diagnostic only), holdout presence P is
  0.67 [0.39, 0.86].
- Dev 0.48 vs holdout 0.37: the dev tuning did not fully transfer. Twenty-five
  clips is a small holdout.

## Verdict

**Every gate still fails on holdout.** The plan's fallback ladder gives
"ships: nothing", so **do not flip `PHOTOSEARCH_VIDEO_SUBJECT_SHARE`**. The
round moved person-segment accuracy by +9 to +13 points and named-presence
precision by ×6, but the gates sit 28 to 52 points away. No knob in this
family closes that gap: the H1 dev surface is flat, spanning 0.44–0.52
over 124 configs.

## What would be needed

Segment error breakdown on dev for the chosen config family. The largest
bucket is an owner-named person predicted as an unnamed person: 113 of 374
segments.

- **46 segments: the person has no tracked face at all in the clip.** They
  were never detected, or detected but not linked to an owner-named face.
- **22 segments: the person is in the clip but has no face in that
  segment.** Identity propagation recovers only 2 points, because the owner
  calls sports teammates "someone else". Re-identifying people whose faces
  are hidden needs a body/appearance tracker (the plan's F3) or a VLM asked
  "is this the same person as <reference crop>".
- **22 segments: the right person is named, but the primary went to another
  face or box.**
- **23 segments: the person is present but unnamed.** The fix is recall: more
  references (item 1).

Two more causes, outside that bucket:

- **Non-person subjects: 0.18 on both splits.** 28 of the 44 non-person
  misses are a wrong label, such as "coral reef" vs "underwater scene", and
  are scored on exact normalized text. The plan's comparer judgements
  (`video_eval.py disputes`) have never been run. Run them, then have the
  owner spot-check 20, before any non-person number is meaningful.
- **Presence.** The gate's truth ("who") is "who the clip is about". Either
  re-define the presence gate against everyone the owner names in the clip,
  which needs an owner decision, or add a salience/focus floor to the
  presence list. A 5 s floor reaches 0.88 on dev at R 0.29.

Recommended production changes, none of them applied (no flag flip, no
prompt or threshold change, no deploy):

1. Add the nearest-other-person margin (0.15) for video face naming before
   any reference expansion.
2. Expand video-visible people's references.
3. If the subject share is revisited, wire H1 (`score_hybrid`). Production
   can feed it the stored `video_tracks.focus_intervals` through
   `focus_from_intervals`. The S1 "highest-salience box" rule loses every
   person to a whole-frame scenery box.

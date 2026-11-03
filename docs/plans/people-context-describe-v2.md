# People context in describe, v2: known people by person-id handle

**Status: 📐 planned 2026-11-01 by plan-debate (two planners, two critique rounds). Evals before any wiring;
owner decided the two open gates 2026-11-01 (flip gate 60%, two-turn ablation paused). Steps 1-4 done 2026-11-01
(round 2 `PROMPT_VERSION` 969a5f3b3cc7; fresh set `sample-fresh.json`, seed 20261101). Dev candidate: **`v2-box` +
FIX_ROSTER** (passes G-A, G-B, G-C on dev; `v2-id` fails G-C at 28.6%); automatic G-F 52.7% < 60%;
owner accepted the candidate 2026-11-02. Dev round 4 (`PROMPT_VERSION` e79e164548fc: no-copy line,
whole phrase in brackets, age-copy guard) passes G-A except median length 323 < 330; G-E pairwise 14 / 16 / 10 passes. Step 6: fresh set 1 burned (2 unusable); fresh set 2 passes G-A/B/C (recall 0.566 counted as 0.607 by the owner); fresh named-mark labels in progress. Status lives in `README.md`.**

Amends docs/plans/people-context-describe.md (its Decisions table and "G0 / G1 results" stay binding; its
"What describe knows" row is superseded by this plan: no names are sent). Measurements that led here:
docs/notes/llm-passes.md, "people context: names as describe context".

## The approach in one paragraph

Describe keeps the G1 shape that passed (two images: the clean photo and the numbered-box overlay, inline
marks, one user message, temperature 0) and changes only what the model knows. Every known person in the
photo (keypoint-linked, uncontested, trusted-face body) gets an opaque handle `P<person_id>` tied to their
box, with a terse gender/age line. **No names are sent, ever.** The roster comes **first** in the prompt,
straight after the two-image header and before any describing rule, with a one-line reminder at the end,
and it states flatly that the listed gender and age are facts that beat appearance. The model writes
**one turn**. Every mark carries its box number, and a known person's handle after a colon:
`[a boy in a neon green jersey](4:P39)`. Code strips the marks to get the raw name-free text
(`photos.description` later), cross-checks each written handle against the roster entry for that box, and
builds the write-once named template `{person:39|a boy in a neon green jersey}` only for spans that pass
every guard. The chain phrase -> box -> body -> face -> person id is recorded per span. Two-turn
generation (id text, then a name-free rewrite) is rejected: the "rewrite without names" turn in the
names run dropped every mark in 14 of 42 retries, and stripping is exact.

Why a handle in the parentheses and not in the words: names failed because a name is natural language
and the model put it where the words go (`[Asa Goldman](2)`, `[plain](3)|Finn Coupar`, the example
name "Sam" copied). `P39` is not prose, so it has little pull into the sentence, and a regex finds it
anywhere outside `(...)`. Why the box number stays next to the handle: in the names run, when the model
wrote a name as the mark, the box agreed with that name's listed box in only 35 of 58 marks (244964 put
Robert on box 5; the context had him on box 1). Writing both gives code two outputs to compare; a handle
drawn on the box would turn that check into a lookup that cannot see the disagreement (it stays as one
dev arm, in case the text roster is ignored as in 243166, where "box 1: Calvin" was marked
"[three boys](1,3,5)").

## Prompt draft (v2)

`{...}` is filled per photo. The KNOWN block and the Reminder are omitted when the photo has no roster
entry; that is the production prompt for unlabelled shoots and must stay byte-identical to a no-context
run. One user message; no system message (a later ablation at most).

```
You get two images. The first is a photo. The second is the same photo with numbered red
boxes, 1 to {n}, drawn around the people in it.

KNOWN PEOPLE. These people have been identified by the family. Their gender and age below
are facts, not guesses: use them even when the person looks different (long hair, clothing
colour, build or pose can mislead). A person listed as a boy is "a boy", never "a girl",
"she" or "her"; a person listed as a girl is "a girl", never "a boy", "he" or "him"; a
person listed as aged 10 is a child, never "a man", "a woman" or "a teenager".
- box {box}: P{person_id}, {boy|girl|man|woman}, {age}
- box {box}: P{person_id}, {boy|girl|man|woman}, {age}
Every other box is a person nobody has identified: describe them only from what you see.

Describe the first image in 2-4 full sentences for a search index. Cover:
- The people: how many, approximate ages, clothing (for a known person, their listed
  gender and age)
- What they are doing (actions, poses)
- The setting (indoor/outdoor, location type such as field, park, street, home)
Be factual and describe only what you can clearly see. Do not start with "The image shows"
or similar. Write as much as you would for the photo alone.

Mark people: every time the description refers to one specific person, or a group of 2 or
3 people, who is inside a numbered box, put the describing words in square brackets and
the box number(s) in round brackets right after. For a known person, write the box number,
a colon and their P-number: [a boy in a red cap](2:P39). For a group, list each box, with
the P-number for any known member: [two men](1,3:P8). Example (a different photo):
[a woman in a grey coat](2:P0) hands an umbrella to [two men](1,3), who stand at a bus stop.
That sentence only shows the form; P0 is not a real person, and your people, scene and
P-numbers come from the photos and the list above. Inside the square brackets write only
ordinary describing words (age, gender, clothing, role); a bare P-number in brackets is
wrong, and a P-number appears ONLY after a colon inside the round brackets, never in the
sentence. Never use a P-number that is not in the list. Mark only people, never objects.
Do not mark pronouns, crowds, or people without a box. Never mention the boxes, the
numbers or the drawing outside the round brackets, and never write a person's name.

Reminder: {P39 (box 4) is a boy, 10; P12 (box 2) is a girl, 9}. Known people keep their
listed gender and age and are marked with their P-number.
```

Changes from the v1 prompt, each tied to a measured failure: the roster moves from last to first (it was
ignored at the end); "Who or what is in the photo" goes (it invites names); the example leaves soccer
(the v1 example's "yellow jersey" appears in 22 of 125 two-image dev answers vs 4 of 125 plain; "two girls"
in 3 vs 1), uses adults, and its handle `P0` can never be real, so a copied example surfaces as a
foreign id; the override is a flat statement instead of the soft "if what you see disagrees, describe
what you see" that produced no gain; the age is a bare integer (no "aged about" wording to copy).

Roster rules (code, not the model): one entry per body whose face is keypoint-linked and uncontested
(`people_bind.keypoint_contested`) and whose label source is in `db.DISPLAY_NAME_SOURCES`; a person on
two live faces in the photo gets no entry; gender and age come from `persons.gender` / `birth_year` and
`date_taken`; an entry with neither gender nor age is still listed as `- box 4: P39` (it carries the
handle for naming). Photos with no entry use the prompt without the KNOWN block.

Corrective turns (one per photo, chosen by the first guard that fires; each says to keep the marks,
which the v1 FIX_NAMES did not, and that is why it lost them):

- `FIX_MARKUP_V2`: "Rewrite the same description with every mark in this form: describing words in
  square brackets, then box number(s) in round brackets, with a colon and P-number for a known person,
  like [a boy in a blue jersey](4:P39). No number or P-number in square brackets. Keep every person
  mark. Answer with the description only."
- `FIX_IDS` (a P-number or "person N" in the sentence): "Rewrite the same description so that P-numbers
  appear only inside the round brackets after marked words, like [a boy in a blue jersey](4:P39). In
  the sentences, describe each person by what they look like and do. Keep every person mark. Answer
  with the description only."
- `FIX_LEAK_V2`: today's FIX_LEAK plus "Keep every person mark and P-number."
- `FIX_ROSTER` (dev arm only, not the default): "P39 (box 4) is a boy, 10. You wrote '{phrase}' for
  them. Rewrite the same description so every mention of that person agrees with the list, keeping
  every mark. Answer with the description only." Adopted only if the arm keeps >= 95% of marks and
  fixes the conflicts it targets.

## Raw text, named template and guards

**Parse** (`parse_v2`, next to the v1 functions, which stay so old runs re-score). `ANCHOR_RE` stays;
the inner grammar is `^\s*\d{1,2}(?::P\d{1,6})?(?:\s*,\s*\d{1,2}(?::P\d{1,6})?)*\s*$`. Each span gets
`boxes`, `written: {box: pid|None}`, the existing `count_problem` and `degenerate` flags.

**Text guards** (raise `describe.UnusableAnswer` after the corrective turn, rule 11): empty text, a
stray bracket, overlay words (`LEAK_RE`), and `ID_IN_PROSE_RE = \b(P\d{1,6}|person\s+\d+)\b` on the
stripped text. An id in prose is never stripped silently: the sentence around it is no longer sound, so
one corrective turn, then the photo fails. Stray `[n]` marks with no handle keep today's `strip_marks`
fallback (text kept, no spans), gated separately. `name_in_text` stays against the roster's names,
which code knows but did not send; its expected count is 0 and it catches a name guessed from a jersey.

**Per-span cross-check** against the roster dict `box -> {person_id, face_id, body_bbox, label_source,
gender, age}`:

| class | meaning | effect |
|---|---|---|
| `agree` | every roster box in the span carries its own P-number; no P-number on a non-roster box | may be named |
| `omitted` | a roster box written by number alone | generic (the instruction was prescriptive); counted |
| `disagree` | P12 written on P39's box | generic, no retry (temperature 0 repeats); 2+ in a photo -> `roster_confused` flag |
| `orphan` | a handle on a box with no roster entry | generic, no retry |
| `foreign` | an id not in the roster (P0 included) | generic, no retry; counted as example echo when P0 |

**Naming veto `roster_ok`** (both a metric and a guard): a span on a roster box whose phrase, or the 40
characters after it, contradicts the listed gender (female lexicon on a male, and the reverse) or age
(man/woman/teen on age <= 12; boy/girl on age >= 18) is never named. It is not retried by default.

**A span is named** (becomes `{person:ID|words}`) only if ALL hold: `agree`; `roster_ok`; count check
passes; not `degenerate`; the bare-phrase rule (`people_bind.bare_allowed` with `BARE_PHRASE_RULE`:
"a boy" / "two boys" names only when its boxes are everyone detected); each person appears in at most
one named span per photo (a second mention stays generic); groups all-or-nothing (every member
`agree`s and passes, or the whole group stays generic). The overlap forced-choice
(`people_bind.overlap_checked`) is measured as an ablation first and becomes a guard on named spans
only under the condition in the eval design.

**Template** `assemble_v2(description, spans)` writes `{person:39|a boy in a neon green jersey}` for
named spans and leaves every other span as plain words. It is a deterministic function of the raw text
plus spans, so a stored write-once copy can always be recomputed and checked against `sha(raw)`. With
every span generic it equals the raw text byte for byte (property test). The kept words are the
fallback a renderer shows when an id stops being valid (relabel, merge, face deleted), so a stale copy
degrades to the raw phrase, never to a wrong name. How search and outbound captions read the filled
copy, and the relabel rule, are v44 design, out of scope here.

**Recorded per photo** (run JSON now; `description_subjects.slots_json` / a roster column in v44):
`prompt_version` (sha of the frozen prompt), `model`, `roster` (box, person_id, face_id, body_bbox,
label_source, gender, age_sent, flipped), `raw_first`, `raw`, `description`, and per span
`{start, end, phrase, boxes, written, check, person_ids, face_ids, body_bboxes, named, why_generic}`.
That is the phrase -> box -> body -> face -> person id chain, written at generation time; a render step
re-validates face -> person (trusted, uncontested, not on two live faces) and falls back to the words.

## Eval design

Harness: new `G1_VARIANTS` rows in `evals/people_context_eval.py`; existing run files untouched;
`two-L-nohints` and `two-L-names` are the stored baselines (girl-on-boy 12 / 6 spans; unusable 0 / 14).

**Dev arms** (125 dev photos, qwen/qwen3.5-9b, two images, ~2.7 s/photo; run in this order):

| arm | context | placement | purpose |
|---|---|---|---|
| `v2-noctx` | none, new base prompt and example | - | separates wording/example effects from context |
| `v2-id` | `- box 4: P39, boy, 10`; marks `(4:P39)` | first + reminder | **candidate** |
| `v2-box` | `- box 4: boy, 10`; marks `(4)`; code maps box -> person | first + reminder | context without a handle (fallback design) |
| `v2-id-drawn` | as `v2-id` plus `4·P39` printed on known boxes by a new drawer in `grounded_describe.py` (`people_bind.draw_boxes` untouched) | first + reminder | roster attached where the model looks; GPU cost ~6 min, owner-labelled only if a contender |
| `v2-id-last` | as `v2-id` | last, no reminder | placement ablation |
| `v2-id-flip` | as `v2-id`, the largest roster entry's gender flipped; second flip swaps two entries' boxes on 2+-roster photos | first + reminder | override and spill (dev only) |
| `v2-id-fixroster` | `v2-id` answers that fail `roster_ok`, one FIX_ROSTER turn | - | retry arm; adopt only if >= 95% of marks survive |
| `v2-id-ovl` | `v2-id` answers plus the overlap second call on named spans (no new describe calls) | - | precision ablation |

Pick order among arms that pass the text gates: named precision, then wrong gender/age on known
people, then leaks (ids or example words in prose, overlay words), then named coverage. A P-number
leaking into the text of `v2-id-drawn` counts as an overlay leak.

**Metrics** (automatic unless marked owner):

- `unusable`: photos with no description after the corrective turn and the strip fallback (a photo
  failed for an id in prose counts). `nospan`: strip fallback used. `rewrite`: photos needing a turn.
- `wrong_gender_age`: spans on a roster box whose words (or the 40 characters after) contradict the
  roster; Asa Goldman reported on his own line; `boy_on_girl` separately.
- `id_leak`: P-numbers or "person N" in the raw text, before and after correction.
- cross-check shares: `agree` / `omitted` / `disagree` / `orphan` / `foreign` over roster-box spans.
- `example_echo`: "grey coat", "umbrella", "bus stop", `P0` (gate 0); "yellow jersey" reported against
  the plain baseline (4 of 125), not gated (yellow kits are real).
- `named_n` (coverage) and `named_precision` (owner verdicts, pre-filled confirm sheet keyed
  `pid|phrase|boxes`, carried over where the key repeats).
- quality: `degenerate`, median length, people-word recall (existing definitions), blind pairwise vs
  stored on 40 dev photos (owner), s/photo.
- `flip_compliance`: share of the flipped person's spans using the flipped gender (high is the goal: it
  proves the listing overrides appearance); `flip_spill`: photos where another box's gender words change
  vs `v2-id`; box-swap: reported, not gated.

**Gates** (dev first, then the fresh set):

| gate | bar | baseline |
|---|---|---|
| G-A text | `unusable` 0; `id_leak` 0 after guards; `nospan` <= 8%; `rewrite` <= 15%; `degenerate` <= 13; median length >= 330; recall >= 0.60; `example_echo` 0 | nohints 0 / - / 7% / 6% / 11 / 362 / 0.62; names 14 unusable, 22% rewrite |
| G-B gender | `wrong_gender_age` <= 4 of ~126 marks on known people; Asa <= 1 photo; `boy_on_girl` 0 | 12 spans (Asa 10) / 6 with names (Asa 4) |
| G-C handle | `disagree` + `orphan` <= 10% of roster-box spans (above it the handle confuses the model and `v2-box` wins if it meets G-B); `omitted` reported | - |
| G-D names | `named_precision` >= 97% on dev + fresh together with <= 2 wrong on fresh alone; coverage >= 90% of G1's (>= 84 named on dev) | G1 139 / 143, dev 93 named |
| G-E quality | blind pairwise new >= stored - 3 of 40 | 18 / 17 / 5 |
| G-F override | `flip_compliance` >= 60% (80% reported as the target); `flip_spill` <= 2 photos | names run 23 of 90 (26%) |
| overlap | becomes a guard only if on the fresh set it removes >= 1 wrong named mark and loses <= 10% of right ones; otherwise off | Step 0 (spent holdout): 14/15 right kept, 3/4 wrong dropped |

If `v2-id`, `v2-box` and `v2-id-drawn` all miss G-B, the conclusion is "prompt context cannot fix the
gender errors with this model": describe stays label-free and the read-time rule 7 stands.

**Dev / holdout policy.** All tuning on dev, as many runs as needed. The old 75-photo holdout was read
once for named marks under the v1 prompt and its two wrong cases are known, so it is spent as a gate: it
runs once more as a regression check only (reported, not gated). The gate is a **fresh set**: 75 photos
drawn with a recorded seed from the 10,819 soccer stills that have >= 1 roster-eligible body (bodies from
`person_boxes` in `photo_index.db.local`, keypoints from `g0/evidence.jsonl`), excluding the 200-photo
sample, the 100 people-bind review photos, the G0 "who" photos and the 3-s burst window of any sample
photo. The prompt and guards are frozen (`prompt_version` sha in the run spec) before the fresh run; the
winning arm runs on it once; a failure burns the set and a new draw replaces it. Fresh photos are new
photos, not new people: the bias is stated, and the 4 known wrong-pick cases stay a named list.

**Owner labelling budget** (rate from the G1 sheet: ~6-8 s per pre-filled mark, 415 dev marks in one
sitting; ~20 s per pairwise item; the first sheet's wall-clock replaces these assumptions):

| item | marks | minutes |
|---|---|---|
| `v2-id` dev named marks not carried over by key (~130, assume half new) | ~65 | 7-9 |
| `v2-box` or `v2-id-drawn` if a contender | ~65 | 7-9 |
| `omitted`-span spot check | ~30 | 3 |
| flagged gender errors and flip spans (auto metric false positives) | - | 5 |
| blind pairwise, 40 photos | 40 | 13 |
| fresh set, every named mark (~1.5 per roster photo) | ~110 | 11-15 |
| **total** | | **~35-55** |

## Steps (no schema change; no edits to people_bind.py, person_boxes.py, db.py, worker*.py or pass lists)

| # | Step | Files | Verification | Est. |
|---|---|---|---|---|
| 1 | v2 constants beside v1 (`V2_HEAD`, `KNOWN_INTRO`, `V2_PHOTO_RULES`, `V2_MARK_RULES`, `REMINDER`, `FIX_*_V2`, `FIX_IDS`, `FIX_ROSTER`, `PROMPT_VERSION`); `roster_line`, `build_prompt_v2(n, roster, placement, example, drawn)`, `parse_v2` (grammar, `ID_IN_PROSE_RE`, cross-check classes, `roster_ok`), `assemble_v2`, `draw_boxes_labelled` | `photosearch/grounded_describe.py`, `tests/test_people_context.py` | unit tests: grammar, every cross-check class, id in prose raises, P0 foreign, group all-or-nothing, bare id rejected, strip == unmarked sentence, all-generic template byte-identical to raw, no-roster prompt byte-identical to the no-context prompt; `./scripts/test-like-ci.sh` | 1 d |
| 2 | Arms in `G1_VARIANTS`; roster builder from `g1_inputs` + `kp_ok` + the two-live-faces exclusion; corrective-turn routing; flip and box-swap; `cmd_g1_report` metrics above; example-echo scan; owner sheet `--named-only` with key carry-over | `evals/people_context_eval.py` | re-score `two-L-names.json` offline and reproduce 6 "girl" / 14 unusable; dry run prints the prompts for 3 photos with no model call; `--only` run on 5 photos | 1.25 d |
| 3 | Fresh-set builder `g1-fresh-sample` (read-only DB; `sample-fresh.json`, seed recorded) | `evals/people_context_eval.py` | 75 photos, none in the excluded sets, each with >= 1 roster entry | 0.25 d |
| 4 | Dev runs (shared GPU, scheduled with the owner): 6 describe arms x 125 + retries, plus the two no-call ablations | run JSONs under `evals/people-bind/g1/runs/` | automatic report; G-A, G-B, G-C decide the candidate | 0.25 d + ~0.7 GPU-h |
| 5 | Owner dev labelling: named-only confirm sheet and the pair sheet | `g1confirm/`, `g1pair/` | G-D (dev part), G-E, G-F | ~25-35 owner min |
| 6 | Freeze prompt + guards; fresh run once; old-holdout regression run; owner labels fresh named marks; overlap decision | run JSONs | G-D on dev + fresh; overlap condition | 0.25 d + ~0.2 GPU-h + ~11-15 owner min |
| 7 | Docs: dated `###` in `docs/notes/llm-passes.md`; status line and Decisions row in the v1 plan and this file; one bullet in `.claude/rules/people-bind.md` (P-numbers never in raw text); `README.md` status | docs | caps test | 0.25 d |
| 8 | Only after the gates, a separate plan (v44): describe v2 behind a flag; the `{person:id|words}` fill goes through `description_render.display_edit` (a span may run past its noun phrase); describe for a labelled shoot waits on labels (an owner trigger, not a batch step); spans + roster in `description_subjects`; the template column and its readers (display, search, captions decision); `description_render` fill; relabel marks the roster `stale_context` | - | - | - |

**Steps 1-7: ~3.5-4 dev days, ~1 GPU-hour, ~35-55 owner minutes.**

## Risks

- **A wrong trusted label writes the wrong gender into the raw text**, and an override that works obeys it.
  Mitigation: only `DISPLAY_NAME_SOURCES`, keypoint-linked, uncontested faces get an entry; the roster is
  stored with the description, so a relabel can find stale text; the eval flags (unscored) photos where
  the no-context run said "girl" and the roster says boy, for a skim of the ~10 persons involved. The
  owner accepts this trade at the gate.
- **The model attaches a handle to the wrong similar player.** The cross-check catches roster confusion
  but not a wrong box with a consistent id; the overlap ablation targets exactly that case; the precision
  gate stays at the owner's 97%.
- **P-numbers in prose despite the rule.** Regex guard, one corrective turn, then the photo fails; raw
  text never stores an id.
- **Handles reduce marking** (more `omitted`, fewer marks). Coverage is gated; `v2-box` is the fallback.
- **The new example still leaks** ("grey coat", `P0`). Echo is gated at 0; `P0` is a foreign id; a
  no-example arm follows if it persists.
- **The drawn-handle arm leaks or is misread** (P39 vs P38). It is one arm, picked only on the same gates.
- **More instructions bend the text more** (degenerate 11 of 119 in G1). Gated, and the pairwise page decides.

## Out of scope

Schema v44 and storage of the template, spans and roster; worker, claim and pass-list wiring;
`description_render` changes; the API for the filled copy and outbound Google / Shutterfly captions (they
stay raw); the relabel re-describe sweep; cue, kit or burst names for faceless bodies (G0 closed); names or
handles for untrusted faces; a system/user split; other models, temperature or reasoning effort; non-soccer
libraries (a pilot after wiring); the replica mirror path; any further reads of the original 75-photo
holdout for decisions.

## Decisions

Resolved by both planners (round 2); the last two rows were the owner's (2026-11-01).

| Topic | Option A | Option B | Choice | Date |
|---|---|---|---|---|
| Handle for a known person | `P39` drawn on the overlay box, marks `(P39)`, new drawer | plain numbered boxes, marks `(4:P39)`, code compares box and id | **B** primary; A's drawn overlay is one dev arm | 2026-11-01 |
| Generation | one turn, code strips marks | two turns (id text, then name-free rewrite) | **one turn** (rewrite turns dropped every mark in 14 of 42 retries) | 2026-11-01 |
| Roster placement | first, with a reminder last; flat "listed gender/age are facts" | last (v1) | **first + reminder**; `v2-id-last` ablates it | 2026-11-01 |
| Named template | `{person:39}` | `{person:39|plain words}` (fallback words kept, all-generic == raw) | **B** | 2026-11-01 |
| Span contradicting the roster (`roster_ok`) | naming veto + one FIX_ROSTER retry, photo never fails | metric only | **veto, no retry by default**; FIX_ROSTER a dev arm (adopt if >= 95% of marks survive) | 2026-11-01 |
| `disagree` / `orphan` / `foreign` spans | retry once | drop to plain words, no retry | **B** | 2026-11-01 |
| Id still in prose after the corrective turn | strip it, keep the photo | fail the photo (rule 11) | **fail the photo** | 2026-11-01 |
| Overlap forced-choice | in-pipeline guard on named spans | ablation only | **ablation first; guard only if on the fresh set it removes >= 1 wrong name and loses <= 10% of right ones** | 2026-11-01 |
| Unusable gate | 0 | <= 2 of 125 | **0** (baseline reached 0; rule 11); `nospan` <= 8% gated separately | 2026-11-01 |
| Gender gate | <= 4 of ~126 marks on known people, Asa <= 1 | <= 2 photos | **A** (Asa was still "a girl" in 4 photos with names + "(boy, 10)") | 2026-11-01 |
| Holdout | fresh seeded 75 as the gate; old 75 regression-only | same | **agreed** | 2026-11-01 |
| Example | off-domain adults, impossible `P0`, echo gated | same | **agreed** | 2026-11-01 |
| Flip-compliance gate (G-F) | >= 80% | >= 60% gate, 80% reported target | **B** (owner: 60% to start) | 2026-11-01 |
| 40-photo two-turn ablation | run it on dev (never a candidate) | drop it | **paused** (owner; not in the dev arm list, can be added later) | 2026-11-01 |
| Candidate with G-F at 52.7% (dev, before the fix turn) | treat as a G-F fail: describe stays label-free | accept `v2-box` + FIX_ROSTER: the fix turn enforces the listed gender | **B** (owner) | 2026-11-02 |

## Owner decisions on the open points (2026-11-01)

1. **Flip-compliance gate: 60% to start**, 80% reported as the target. The real-world gate is G-B (wrong
   gender on known people <= 4 of ~126 marks); G-F diagnoses whether the override instruction is doing the
   work. Raise it later if G-B passes with compliance far below 80%.
2. **Two-turn ablation: paused.** Not in the dev arm list. The evidence against two-turn generation (the
   rewrite turn dropped every mark in 14 of 42 retries; the JSON shape halved length) is recorded above;
   the 40-photo arm can be added later at ~2 GPU-minutes if a candidate's text quality disappoints.

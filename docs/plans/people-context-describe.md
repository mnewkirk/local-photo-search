# People context in describe: identify bodies and faces, then describe with spans

**Status: 🟡 evals 2026-11-01: G0 fails for faceless bodies (kit tier fails; burst tracks name 0 of 60 owner-named holdout bodies) -> no cue names; hints fail the injection test -> describe stays label-free; G1 dev: named marks 93 right / 2 wrong with all guards (97.9%, 3x people-bind's names), accepted by the owner; G1 holdout next.** (Status lives in `README.md`.)

Supersedes docs/plans/description-names.md Revision 2 R3 (the after-the-fact body binder as the production path).
Source: plan-debate 2026-10-31, Planner A (plan-A.md) and Planner B (plan-B.md), two critique rounds.

## The approach in one paragraph

Detect a body box for every person (a GPU worker pass after `faces`), link faces to bodies with a keypoint-verified
link, and read per-body evidence (jersey number, kit, hair, shoes, role) with a VLM. `describe` is then given the photo
plus a numbered overlay of the bodies and writes its description AND its own person spans (`{phrase -> box}`) in one
JSON answer, so phrase-to-person mapping is no longer guessed afterwards. Identity is never given to describe as a name:
for bodies with a trusted, uncontested face it may get gender and age-band hints (v42 attributes). Faceless bodies are
identified on the NAS (CPU) from tiered evidence — burst tracks to a sibling frame with a trusted face, then a per-shoot
jersey-number + kit map seeded by trusted faces — and names are filled in only at read time by `description_render`,
validated against the witness labels. The raw `photos.description` stays name-free; relabels follow for free.

## Agreed decisions (both planners)

- **Invariant kept**: raw `photos.description` never contains names; search, verify, category-content, keywords read
  raw. What changes: describe may receive label-derived gender/age-band hints (never names or person ids), recorded as
  provenance (`hints_json`); spans come from describe (`description_subjects.method='describe'`, `BIND_VERSION` 2) and
  may point at bodies.
- **Describe waits on `bodies` only.** `body-attrs` runs just before describe (model locality) but does not gate it:
  on new batches almost no labels exist at describe time (sample: 305 manual vs 17 strict faces; strict matching is a
  NAS step after the worker passes; hand labels come later). Hints use whatever trusted identity exists at claim time;
  an inline strict match at the describe claim only if Step 0 shows it is cheap on the N100.
- **Face-to-body link, two independent signals**: (1) keypoint-verified link (torchvision Keypoint R-CNN: a body owns a
  face only if its head keypoints fall in the face box; a body facing away owns none; contested flag when two bodies
  qualify), spike-gated on 247627 / 244870 / 249761 and keeping >= 95% of today's face links; (2) cue disagreement (face
  says P, kit+number says Q) marks the body contested. Either firing -> both bodies render generic.
- **Faceless identity tiers** (within a session, never library-wide): own face (keypoint link) -> burst track to a frame
  with that kid's trusted face (geometry; CLIP crop similarity only as an ablation, no stored embeddings in v1) ->
  jersey number per session and kit cluster, seeded by >= 2 independent trusted-face anchor bursts, 0 contradicting
  anchors, a duplicate signature in one photo voids it, one person once per photo, session >= 95% processed ->
  attributes (hair, shoes) as tie-break only. Never `temporal` seeds. Owner body labels (`body_labels`) are seeds that
  survive re-resolution. Session = collection, or folder + day split at 45-min gaps (Step 0 checks vs collections).
- **Rejected**: generic person re-ID embeddings (OSNet etc.: new dependency; teammates in one kit collapse), library-wide
  body clustering (kits and numbers change by season), dedicated jersey OCR (new dependency; revisit only if VLM number
  precision fails), temporal face matches as seeds (~4% accurate).
- **Order of work: evals before wiring.** G0 (identity eval) and G1 (describe prototype eval) run on offline data
  before any schema or pass wiring; every people-bind model failed its first eval, so this is the cheaper failure.
- **Legacy batches**: a cutover constant keeps pre-cutover batches `ready` when the new gating passes land.
- **Schema** (renumbered 2026-11-01: `person_boxes` with keypoints + facing shipped alone as v43, with a non-batch
  `bodies` pass, to ship people-bind; the rest is one v44 migration): `body_attrs`, `body_identity`
  (recomputable cache with witnesses), `body_labels`, `description_subjects.hints_json`.
- **Describe call shape**: G1 tests both (clean photo + numbered overlay as two images, vs one overlay image); dev
  picks on quality, overlay leakage and the hint-injection test.
- **Cue-inferred names**: never on by default; `PHOTOSEARCH_DISPLAY_CUE_NAMES` per tier, each tier only after its own
  gate, shown with a "from kit/number" chip; the owner picks defaults after the pilot.
- **Relabels**: a relabel marks the description `stale_context` (from `hints_json`) and render rule 7 hides a
  conflicting name immediately; a capped re-describe sweep (only when a hint's gender or age band would change) exists
  but is opt-in. Faceless identities re-validate witnesses at render (`stale` -> generic until re-resolved).
- **Legacy people-bind**: stays runnable for old descriptions; no library-wide people-bind backfill (GPU goes to
  re-describe); its qwen35-k binder is the G1 cross-check on describe's spans.
- **Replica**: renders from its copies; describe v2 runs only through NAS fleet claims, not the local re-run mirror,
  until `_apply_mirror` keeps NAS body ids.

## Steps

| # | Step | Gate / verification | Est. |
|---|---|---|---|
| 0 | Spikes: keypoint link on the 3 known errors and link coverage; burst siblings with a visible trusted face for faceless subjects; sessions vs collections; strict-match cost at claim on the N100; a 20-photo grounded-describe try (leakage); describe s/photo; legacy-batch exemption design | numbers in docs/notes | 0.75 d |
| 1 | `body_attrs.py` (crop, closed-vocab JSON schema) + eval `attrs` over the soccer pilot scope | self-supervised number agreement; parser tests | 2 d |
| 2 | `identify.py` / `body_identity.py` (pure): tiers, conflict rules, per-photo assignment; owner "who" labels for faceless bodies (~25-50 min); leave-face-out eval split by shoot | **G0** per tier: 0 wrong among owner-labelled faceless holdout bodies (>= 50 named), leave-face-out wrong <= 0.5%; for a tier to feed hints <= 0.2%; link gate GL: 0 wrong links on the owner-checked contested set incl. the 3 known | 3 d |
| 3 | Grounded describe prototype + eval: `{description, people:[{boxes, phrase}]}`, guards (no names, no overlay words, exact substrings), both call shapes, hint-injection test (flip gender on ~40 photos; > 2% obeyed -> hints restricted); owner confirms pre-filled phrase->body letters on dev | **G1**: 0 wrong binds on dev then holdout; >= 150 named holdout mentions or INCOMPLETE + top-up; coverage clearly above 17-25%; quality non-regression (unusable/degenerate within +2 pts, 0 leakage, blind pairwise 40 photos at parity, no new category/keyword errors, people-word recall not lower); fewer rule-7 conflicts | 3 d |
| 4 | Schema v44 (v43 `person_boxes` + `bodies` shipped 2026-11-01 as a non-batch pass: move it into the batch flow here), `body-attrs` pass, pass lists, PASS_MEDIA, roles, clear-pass, cutover constant | migration test, list equality, old batches stay ready, scoped dry run | 2.5 d |
| 5 | Describe v2 in the worker behind `PHOTOSEARCH_GROUNDED_DESCRIBE` (off): claim input, submit order (write text, invalidate, then upsert spans), verify regen through the same function | flag off = byte-identical describe; no-names guard | 3 d |
| 6 | Render v2 (body slots, witness validation, `identity_conflict` / `contested_link` / `stale` states), NAS resolver step + sweep, UI states, `syncFaces` | relabel-follow test; render p95 < 5 ms/100 photos; pre-v44 replica -> raw | 2 d |
| 7 | Pilot: one soccer shoot re-described (scoped, `--native`), owner reviews 100 renders | 0 wrong names | 1 d + GPU |
| 8 | Widen per the owner's scope decision; docs (rules, notes, plan status) | - | 1 d |

~18 dev days + GPU (evidence on the soccer shoots ~15 GPU-h; re-describe 35-55k people photos ~100-250 GPU-h, measured in the pilot). Shared fallback if G1 fails: keep people-bind + v42 attributes only.

## Decisions

| Topic | Option A | Option B | Choice | Date |
|---|---|---|---|---|
| Bare-phrase rule in grounded describe | Keep the owner's rule on (a bare phrase binds only if it names everyone detected) | Off by default; a G1 dev ablation decides | **B: G1 ablation decides** | 2026-10-31 |
| Gender / age-band hints to describe | Yes, for trusted-face, uncontested-link bodies, behind the hint-injection test (> 2% obeyed wrong hints -> restricted); hints recorded in `hints_json` | No: describe stays label-free | **A** | 2026-10-31 |
| Wrong-name bar for faceless tiers | Per tier: 0 wrong among >= 50 owner-labelled faceless holdout bodies and <= 0.5% leave-face-out; flag-gated with a "from kit/number" chip | Face-linked names only | **A** | 2026-10-31 |
| Re-describe scope | One soccer shoot pilot (100 renders reviewed, 0 wrong) -> 2026 soccer collections -> decide the rest from measured GPU time | All trusted-label photos after the pilot | **A** | 2026-10-31 |

| Precision bar for names | 0 wrong binds across >= 150 (rule of three) | Owner accepts the measured ~99% precision (152 of 153 owner-verified people-bind binds right), with the overlap checks below | **B** | 2026-11-01 |

Dropped after Step 0: running `match_faces` earlier so strict labels reach a batch's first describe: cheap (~30 ms per
claim on the N100) but useless today (face references exist for 4 people; ~1 unlabelled match per shoot).

## Step 0 results (2026-11-01)

Notes: docs/notes/llm-passes.md, "people context: Step 0".

- **Keypoint-verified link** (torchvision Keypoint R-CNN; a body owns a face only if its nose or an eye
  falls inside the face box, no padding): flags all three wrong links (247627, 244870, 246150) as
  ambiguous or owned by another body; of 185 owner-verified right links, 178 agree and 7 are ambiguous
  (4% coverage cost); never "no owner". Use it.
- **Overlap second call** (owner's idea): when a chosen box is >= 30% inside another person's box (not a
  member of the same group), crop the two, label them A/B, ask a forced choice twice with letters swapped,
  keep only if both answers pick the chosen box. Singular phrases: 14 of 15 right binds kept, 3 of 4 wrong
  rejected (250737, 244870, 249761; the miss is the link error the keypoint check catches). Add it as a
  step; both checks together caught all five wrong qwen3.5 binds seen. Measured on spent holdout: confirm
  on fresh photos.
- **Sessions** = folder + day split at 45-min gaps (12-119 min all equivalent), never collections (curated,
  overlapping, one spans the season; they hold ~23% of each shoot). 10 soccer sessions; the 10-03 team
  photo splits from the game, correctly (kit change).
- **Legacy batches**: a `CUTOVER_PASSES` map in `batch_state.py` keyed on `ingest_batches.created_at`;
  describe's `eligible` needs its own branch (described OR bodies settled) or it reads a false `completed`;
  `bodies` needs empty-is-done semantics; new passes need `_OUTPUT_MISSING`, `_REMAINING_FILTERS_ATTEMPTS`,
  `PASS_MEDIA` and JS `ROWS` entries.
- **Burst tracks**: 82 of 91 photos with a faceless subject have a frame within 3 s holding a trusted face
  (median 5 such frames): a high upper bound for the burst tier (G0 measures same-kid recall).
- **Grounded describe try** (20 dev photos, qwen3.5, clean photo + numbered overlay, JSON
  `{description, people}`): 0 overlay leakage, 2.0 s/photo, box numbers always valid; but 14 of 47
  phrases were not in the text (people listed with new details or paraphrased) and descriptions were half
  as long (median 224 vs 408 chars). G1 should test **inline anchors** (the model writes
  `[phrase](box)` in the text; code strips the markup, so spans match by construction) and a length
  instruction.

## G0 / G1 results (2026-11-01, in progress)

Notes: docs/notes/llm-passes.md, "people context: G0 and G1 evals". Harness: `evals/people_context_eval.py`.

- **Burst tier** (camera motion from phase correlation, IoU >= 0.7): leave-face-out holdout 1,773 right,
  5 wrong (0.28%): passes <= 0.5%, fails the <= 0.2% hint bar. Owner-labelled faceless gate pending.
- **Kit tier fails on tune**: 8 right, 3 wrong leave-face-out; the same kid reads the same number in only
  54% (preview) / 64% (full resolution) of crops. Not run on holdout. Revisit only with jersey OCR.
- **Burst tier reach**: names 2 of the 156 owner-placed bodies that have no trusted face (0 on holdout):
  the faceless-identity gate (>= 50 named holdout bodies) cannot be met; cue names stay off.
  Owner labels in: burst tier names 0 of 60 named holdout faceless bodies (tune: 1 right, 1 wrong);
  the keypoint face link names 33 of them, all right (tune 25 right, 0 wrong).
- **Hints**: fail the injection test (flipped gender obeyed ~4 points over baseline; true hints fix
  nothing) -> describe stays label-free (Decision row "Gender / age-band hints" falls back to B).
- **Call shape**: two images (clean + numbered overlay) with inline `[phrase](n)` anchors, a length line,
  one corrective retry: 0 unusable on dev, descriptions 12-15% shorter than regenerated plain ones.
  One overlay image leaks. Bind precision pending the owner's confirmations.
- **G1 verdicts (owner, dev)**: blind pairwise at parity (18 new / 17 stored / 5 same); marks 379 / 412
  right; 27 of the 33 wrong marks are on people who get no name; named marks with keypoint link +
  bare-phrase rule 93 right, 2 wrong (97.9%). The owner's precision decision is not a 99% minimum: G1 dev
  is not failed. Bare-phrase rule: on (it removed 1 wrong, 19 right). Next: G1 holdout (75 photos).
- **Scope if holdout holds**: a reduced Step 5-6 (describe writes marks; names only from keypoint-linked
  trusted faces); no `body_attrs` / `body_identity`, no hints, no cue names.

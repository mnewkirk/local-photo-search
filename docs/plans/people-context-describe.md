# People context in describe: identify bodies and faces, then describe with spans

**Status: 🟡 planned 2026-10-31 (plan-debate); nothing built. Step 0 next.** (Status lives in `README.md`.)

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
- **Schema**: one unshipped v43 migration: `person_boxes` (+ keypoints, facing), `body_attrs`, `body_identity`
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
| 4 | Schema v43, `bodies` + `body-attrs` passes, pass lists, PASS_MEDIA, roles, clear-pass, cutover constant | migration test, list equality, old batches stay ready, scoped dry run | 2.5 d |
| 5 | Describe v2 in the worker behind `PHOTOSEARCH_GROUNDED_DESCRIBE` (off): claim input, submit order (write text, invalidate, then upsert spans), verify regen through the same function | flag off = byte-identical describe; no-names guard | 3 d |
| 6 | Render v2 (body slots, witness validation, `identity_conflict` / `contested_link` / `stale` states), NAS resolver step + sweep, UI states, `syncFaces` | relabel-follow test; render p95 < 5 ms/100 photos; pre-v43 replica -> raw | 2 d |
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

Open, low priority: run `match_faces` earlier so strict labels reach a batch's first describe (strict was ~5% of
trusted labels in the sample; Step 0 measures the claim-time cost).

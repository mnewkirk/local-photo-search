# Subject-aware quality (subject grounding + subject-crop aesthetic re-scoring)

**Status:** Prototype validated (2026-07-09), design proposed, not yet built.
Follow-on to the VLM aesthetics pass (schema v26, `aes_*` columns). All-qwen —
deliberately **no pyiqa / MUSIQ / TOPIQ** (rejected in the Phase-0 bakeoff and the
OOM-killer there; the user does not want it in the path).

## Problem

The VLM aesthetics pass scores the **whole frame**. For a subject photo — e.g. a
marmot occupying ~1.5% of a busy alpine meadow — the model's wow / impact /
sharpness judgments are dominated by the background (grass, wildflowers), not the
subject. Concretely, on two marmot frames:

| photo | subject | full-frame `aes_overall` | full-frame wow |
|---|---|---|---|
| 228893 | upright, camera-facing (the better shot) | 6.08 | 3.0 |
| 228891 | hunched, side-on | 6.88 | 7.0 |

The user prefers 228893, but the full-frame score ranks 228891 higher — because
"wow 7" is really about the pretty meadow. The VLM also gave **both** photos
`sharpness 8.0` (it never registered the subject-sharpness difference). Objective
pixel sharpness (Laplacian variance) was inconclusive because even a tight subject
box is ~half grass.

## Prototype findings (qwen2.5-vl-7b-instruct via LM Studio, localhost:1234)

Scripts: `/tmp/subject-proto/run.py` (grounding + overlay report),
`/tmp/subject-proto/crop_score.py` (full-vs-crop scoring).

1. **Grounding works.** Prompt the model for the single main subject as
   `{"label": ..., "bbox_2d": [x1,y1,x2,y2]}` in absolute pixels. Send the preview
   resized so the long edge is 1008px; the returned coordinates live in **that**
   image's pixel space (no normalization/scaling needed — verified by overlay).
   Localized marmots at ~1.5% of frame, people at ~33%, and correctly returned
   `null` for landscapes (the no-clear-subject case). Boxes were tight and accurate.

2. **Subject-CROP re-scoring flips the ranking to match human preference.** Crop to
   the subject box (padded ~50% each side for context) and re-run the *same* qwen
   aesthetic rubric on the crop:

   | photo | full `aes_overall` | **crop overall** | full wow → crop wow |
   |---|---|---|---|
   | 228893 (preferred) | 6.08 | **7.00** | 3 → **8** |
   | 228891 | 6.88 | **5.35** | 7 → **3** |

   The crop's wow/impact finally judge the marmot's pose/engagement, not the meadow.
   The flip (7.00 vs 5.35) is far larger than VLM run-to-run noise.

## Proposed design

### Schema (v27)
- `photos.subject_boxes` — JSON: list of `{label, bbox:[x1,y1,x2,y2], area_frac}`
  in the ORIGINAL image's coordinate space (store the scale used, or normalize to
  0–1 so it's resolution-independent). NULL = not yet grounded; `[]` = grounded, no
  clear subject (landscape). Analogous to faces but no separate table (subjects are
  not entities to match across photos — keep it a column).
- Subject-crop aesthetic columns mirroring the full-frame ones for the primary
  subject: `aes_subject_overall`, `aes_subject_overall_pct`, the 3 dims, the 11
  sub-attrs (or a pragmatic subset — at minimum overall + wow + impact + sharpness).
  Percentile normalized library-relative like `aes_overall_pct`.

### Pipeline (extend the existing `aesthetics` worker pass — one model, no new infra)
1. Full-frame aesthetic scoring (unchanged).
2. Grounding call → primary subject box(es). Cheap (1 qwen call).
3. If a subject exists and is small enough to matter (area_frac below a threshold,
   e.g. < ~40%), crop (padded) and re-score → `aes_subject_*`. For subjects that
   already fill the frame, subject≈full, so skip the crop call (optimization).
4. Store boxes + subject scores; log provenance like the other passes.

Backfill CLIs mirroring the existing ones: `normalize-subject-aesthetics --apply`
(percentiles), and a `--force-subject` re-run flag.

### Search wiring
- New sort `sort=subject_aesthetic_desc` and filter `min_subject_aesthetic` (pct),
  reusing the `search.py` aesthetic-filter plumbing.
- For subject-bearing queries (a person, an animal keyword like "marmot", "the
  dog"), prefer the subject score when a subject box exists; fall back to full-frame
  otherwise. Decision to make: auto-detect "subject query" vs an explicit toggle.

### Other payoffs (why this is worth the schema bump)
- **Crop suggestions** — the subject box is a ready-made crop target.
- **Subject-aware book cropping** — generalizes the face-box rule already relied on
  (see the subject-aware-cropping feedback) to non-face subjects (wildlife, objects).

## Open questions / decisions
- Multiple subjects per photo (two marmots, a group of people) — store all boxes;
  score the largest/most-central, or score each? Prototype only did primary.
- Grounding coordinate convention across resolutions — normalize boxes to 0–1 on
  store to be safe.
- Threshold for "subject small enough to re-score" (skip when subject ≈ full frame).
- Whether `aes_subject_overall` becomes the DEFAULT quality signal for subject
  photos, or a parallel signal surfaced only on subject searches.
- VLM run-to-run variance: consider 2-sample averaging for the crop score if it
  proves noisy at scale.

## Validation before build
Prototype the pass across the full marmot set (~66 photos): ground + crop-score all,
produce subject-ranked vs full-frame-ranked lists + a visual report, confirm the
subject ranking matches human judgment across the set (not just the one pair) before
committing the schema. **This runs before expanding the main aesthetics rollout**
(remaining ~147k library is paused).

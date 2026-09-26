# Handoff — 2026-09-26: `sharp` / `blurry` visual tags

Status: **frozen, open for design.** The owner is still thinking about the
approach, so nothing here is decided. This note collects what is known so the
next session doesn't have to rediscover it. CLAUDE.md ("`sharp` / `blurry` are
FROZEN — and why that reversed a shipped decision") is the authoritative record
of the decision; this adds the design space and today's numbers.

## Why they're frozen

- **The VLM can't judge them.** `category-visual` runs on a ~336 px tile, and
  sharpness lives in pixels the tile has already thrown away. Its sibling tag
  `motion-blur` had zero signal: median `aes_sharpness` was 7.0 whether the
  photo was tagged or not. `motion-blur` was retired.
- **Deriving them from `aes_sharpness` failed.** The rule was ≥ 9 → `sharp`,
  ≤ 2 → `blurry`. In the July copy it looked clean: 47% of photos scoring ≤ 2
  carried the VLM's own `blurry` tag, against 0.00% at 3 and above. But that is
  two models agreeing, not ground truth, and the copy had only 2.4%
  `aes_sharpness` coverage. Live, at 98.9% coverage, `blurry` went from 1,284 to
  10,103 and `sharp` from 3,410 to 15,076.
- **Hand check: 2 of 4 sampled "≤ 2" photos were not blurry.** One was a
  tack-sharp phone photo of construction formwork; the other a dark, noisy GoPro
  night street scene. `aes_sharpness` mixes sharpness with general technical
  quality, and noise and low light drag it down.

"Frozen" means:
- `derive_tags` never emits them, the prompt never offers them, and the parser
  drops them if a model volunteers one.
- **Stored values are kept as they are.** The write path (`merge_vlm_answer`)
  carries existing ones across a re-tag.

Today on the replica: `sharp` on **3,410** photos, `blurry` on **1,284**. These
are old-prompt VLM opinions of unknown accuracy.

## Direction: measure it, don't ask a model

Measure sharpness directly: the variance of the Laplacian, taken on the
ORIGINAL pixels. Two things make it credible:

- **The code already exists** for photos with faces: `photosearch/rank_measure.py`
  `_measure_photo`. It decodes once per photo, crops each face box from the
  EXIF-oriented image, and takes `cv2.Laplacian(gray, CV_64F).var()`. It runs as
  the optional `rank_measure` NAS step on `/batches`.
- **It is a measurement, not an opinion.** A detector or grounding model only
  decides *where* to look; the pixels decide *how sharp* it is. That avoids the
  "two models agreeing" trap that sank `aes_sharpness`.

It must run where the originals are (the NAS). A preview or a cached 200 px crop
has already discarded the signal.

## Where to measure when there's no face: the subject boxes already exist

The subject-aware aesthetics work (schema v27, `photosearch/subjects.py`) asks
qwen2.5-VL to locate the main subject(s) and stores them in
`photos.subject_boxes`. These are normalized 0–1 boxes with a label and
`area_frac`, e.g.:

```json
[{"label": "flower", "bbox": [0.41992, 0.20205, 0.625, 0.54173], "area_frac": 0.06966}, ...]
```

Coverage on the replica (159,792 photos):

| | photos |
|---|---|
| has one or more face rows | 91,939 |
| has a subject box (non-empty `subject_boxes`) | ~89,000 |
| face **or** subject box | **113,524** (71%) |
| grounded, no subject (`[]`) | ~69,100 |
| **neither face nor subject box** | **46,268** (29%) |
| never grounded (`subject_boxes IS NULL`) | ~1,700 |

A fallback chain that falls out of this:

1. **Face crops**, exactly as `rank_measure` does today.
2. **Subject boxes** from `subject_boxes`.
3. **Whole frame** for the ~46k with neither (mostly landscapes and scenes).

**Check first:** are the subject boxes normalized in EXIF-oriented space? Face
boxes are, and `rank_measure` depends on that. If grounding ran on the raw
orientation, about 18% of crops would sample the wrong region.

## Traps to design around

- **Subject boxes are loose.** They include background, and on a
  shallow-depth-of-field shot half the box can be intentionally defocused, so a
  whole-box Laplacian reads soft. Suggestion: tile the box and take the
  **sharpest tile** (max or a high percentile). What matters is whether the
  in-focus part of the subject is sharp.
- **Whole-frame photos need asymmetric rules.**
  - For `sharp`, "sharp somewhere" (the sharpest tile) is the right test.
  - For `blurry`, use "sharp **nowhere**" (even the best tile is soft). A mean or
    median is fooled by deliberate bokeh and by large featureless areas (sky,
    water, snow).
- **Raw Laplacian variance isn't comparable across photos.** It grows with:
  - ISO noise. Noise is exactly what fooled `aes_sharpness`; noise adds
    high-frequency energy, so it can read as *sharper*.
  - Scene texture (grass vs a white wall).
  - Resolution and crop size (a 60 MP a7R VI vs a 12 MP phone).
  - JPEG quality and in-camera sharpening.

  Candidates:
  - Normalize the crop to a fixed pixel scale before measuring.
  - Estimate and subtract noise (e.g. from flat regions, or from ISO in EXIF).
  - Compare the sharpest tile to the photo's own median tile, a *relative*
    measure that cancels much of the per-camera and per-scene scale.
  - Per-camera calibration (the library is mostly a7R VI, a7 IV and phones).
- **Motion blur vs defocus.** Laplacian variance falls for both. The vocabulary
  has no `motion-blur` any more (retired), so `blurry` covers both, which is
  honest.
- **Cost.** One full-resolution decode per photo on the N100, the same cost as
  `rank_measure` (~10 min per 1,260 photos). The whole library is many hours: a
  background job at a low rate, missing-only, abortable per photo. Reuse
  `rank_measure`'s atomic-cache and per-photo abort patterns. Don't compete with
  ingest for the disks (see the 2026-09-19 SMB-starvation incident).
- **Thresholds need hand labels.** The owner is building a labelling page for
  the visual-tag eval; sharp/blurry labels on a stratified sample (phone vs FF,
  night/high-ISO, bokeh portraits, landscapes, sports bursts) would set the cut
  points and measure precision before any backfill. The eval flow described in
  CLAUDE.md already supports candidate tags scored only for the variant that
  offered them.

## Decisions to make

1. **Storage.** A numeric sharpness column (or JSON with per-region values) plus
   derived tags, or tags only? A stored number would also feed `rank_shoot`,
   search filters and per-photo QA, and lets thresholds be retuned without a
   re-decode.
2. **What happens to the 3,410 / 1,284 stored tags** once the measurement ships:
   replace them, or keep them only where the measurement agrees?
3. **Where it runs.** A new maintenance/batch step on the NAS (the pixels live
   there), or an extension of `rank_measure` so a shoot pays one decode for
   both.
4. **Scope of `sharp`.** Is it a tag worth having at all? On a well-shot library
   most photos are sharp, so it may saturate like `sunny` did on sports
   folders. `blurry` (the rarer, actionable one) may be the only tag worth
   deriving, with the number available for ranking.

## Pointers

- `photosearch/visual_tags_derive.py`: the DERIVED / RETIRED / FROZEN /
  PERCEIVED groups; a derived `sharp`/`blurry` moves from FROZEN to DERIVED.
- `photosearch/rank_measure.py`: `_measure_photo`, the cache and abort patterns.
- `photosearch/subjects.py`: subject grounding, box format, `_CROP_PAD`.
- `scripts/rank_shoot.py`: current consumer of per-face sharpness.
- CLAUDE.md: "`category-visual`: derived capture facts vs perceived qualities"
  and "Per-shoot 'best of' curation".

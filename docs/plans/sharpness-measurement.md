# Measured sharpness → derived `blurry`

Status: **planned, not started.** Follows `docs/HANDOFF-2026-09-26-sharp-blurry.md`.
Produced by a two-planner debate (2026-09-26); both planners converged on every
point, so there were no owner tie-breaks. The four "decisions to make" from the
handoff are answered below as **recommendations** — the owner can still overrule
any of them.

## How the three standard options fit what we already know

| option | verdict | why, in this library |
|---|---|---|
| Variance of Laplacian | **adapt** | Already our direction and already in production for faces (`rank_measure._measure_photo`). The generic recipe (whole frame, one threshold per camera) is the failure we already hit: raw variance *rises* with ISO noise (the GoPro night false positive), and it tracks texture (the formwork photo), resolution and in-camera sharpening. A whole-frame mean is fooled by bokeh and sky. "<10 ms" is misleading: the cost is the full-res decode, ~0.5 s/photo on the N100. |
| CNN (MobileNet/ResNet on CERTH/CSIQ) | **reject for now** | A 224 px input discards the high-frequency signal, the same reason the 336 px VLM tile failed; patch-wise native-res inference on a GPU-less N100 is expensive. CERTH/CSIQ are synthetic or consumer blur, not intentional bokeh, indoor sports or phone computational sharpening. It would be a third model's opinion that still needs our labels to validate. If the Laplacian features fail the eval, the next step is a logistic or gradient-boosted model on the **stored features**, trained on our labels, before any CNN. |
| VLM prompt `is_blurry` | **reject as judge** | Already tried and it failed: `motion-blur` had zero signal (median `aes_sharpness` 7.0 tagged vs untagged); `aes_sharpness ≤ 2` gave 2 of 4 false positives. 1024 px fleet JPEGs have also discarded the signal. Its one good idea, "is the *subject* sharp, not the whole frame", is already in the design: the VLM picks **where** to look (`subject_boxes`) and the pixels decide **how sharp**. |

## Resolved from the handoff

- **Subject boxes are EXIF-oriented.** `subjects._encode_for_grounding`
  (`photosearch/subjects.py:62`) runs `ImageOps.exif_transpose` before grounding
  and normalises against the oriented size, so they are in the same space as face
  boxes. Step 0 still checks this visually.

## Plan

**0. Orientation spot check (0.5 h).** Draw face and subject boxes over about 10
orientation-6/8 photos with a read-only desktop script. *Verify:* the boxes sit on
the subject.

**1. Labels first (1 session, plus 1–2 h of owner labelling).**
- `visual_tag_eval.py`: add a `MEASURED_TAGS = {"sharp", "blurry"}` group that can
  be labelled but sits outside `CANDIDATE_TAGS`, so VLM variants are never scored
  on it.
- `eval_api.py`: `LABELLER_NOTES` definitions: `blurry` means the intended subject
  (or everything, when there is no subject) is out of focus at 100%; deliberate
  bokeh around a sharp subject is **not** blurry; noise is **not** blur;
  long-exposure motion counts as blurry but is its own stratum.
- The labelling page gets a **100% loupe on `/full`**. Labelling from `/preview`
  would repeat the downscale mistake.
- The sample: the existing 60, plus a separate blurry-weighted set in its own eval
  directory, so the existing runs are not orphaned. Strata: stored `blurry`,
  `aes_sharpness ≤ 2`, indoor sports, ISO ≥ 3200 or night, bokeh portraits,
  no-subject landscapes, and each camera (a7R VI, a7 IV, phone, GoPro).
- *Verify:* at least 25 `blurry` positives; recheck kappa ≥ 0.6 on `blurry`.

**2. Measurement function: new `photosearch/sharpness.py` (1–1.5 sessions).**
- A pure, injectable function. Decode once with EXIF transpose. For JPEG, use
  `Image.draft` down to a long edge of about 4000 px: this normalises resolution
  and speeds up the N100.
- Regions: face boxes, then subject boxes (unpadded), then the whole frame. Each
  region is cut into about 192 px tiles, and flat tiles are skipped.
- Store **all** candidate features: Laplacian variance at the max, p95 and median
  tile; the best/median ratio; the Immerkær noise σ; noise-corrected Laplacian;
  Tenengrad.
- `rank_measure.py` and its date-keyed cache stay **untouched**. At most, a tiny
  pure Laplacian-of-crop helper is shared. The new stage never reads that cache:
  its values are raw native-res numbers with no noise figure, and mixing them in
  would make the column incomparable.
- *Verify:* synthetic tests (a blurred copy scores lower; added noise does not
  raise the noise-corrected metric; a rotated EXIF fixture crops the right region);
  time 50 NAS files, target ≤ 0.4 s/photo.

**3. Eval and ship gate (1 session).**
- `evals/sharpness_eval.py` runs on the desktop over the labelled sample,
  fetching `/full`.
- It reports per-metric precision/recall per stratum and per camera, beside the
  `stored` and `aes_sharpness ≤ 2` baselines.
- **Gate for `blurry`:** P ≥ 0.8 at R ≥ 0.5 overall, and P ≥ 0.6 in both the
  night/high-ISO and bokeh strata. If it fails, ship the number only and keep
  `blurry` frozen.
- Per-camera thresholds only if a single global threshold fails.

**4. Schema v33 (0.5 session).** `photos.sharpness REAL` (the chosen normalised
score), `sharpness_json TEXT` (region source, per-region tile stats, all
candidates, noise, scale, camera, or `{"error": …}`), `sharpness_version INTEGER`,
`sharpness_scored_at TEXT`. *Verify:* v32 → v33 migration test, and a replica sync
round-trip.

**5. Backfill stage, CLI and batch step (1 session).**
- Where: `maintenance._stage_sharpness`, `cli.py sharpness [--limit] [--dry-run]`
  (with `envvar="PHOTOSEARCH_DB"`), and a `batch_advance` step after
  `rank_measure`.
- Missing-only: `sharpness_version IS NULL OR < CURRENT`. A failed decode is
  stored as an error **with** the version, so it is never retried forever (the
  CLIP re-claim trap).
- Skip when the file is not local, so the replica never runs it.
- Abort per photo, commit every 25.
- Throttle: about 5k photos/night, a pause between photos, `nice`/`ionice`, and
  refuse to run while an ingest or batch job is active (the 2026-09-19
  starvation).
- Never pass an empty `photo_ids` scope.
- *Verify:* abort and empty-scope tests; a 200-photo NAS dry run while watching
  iowait.

**6. Derive `blurry` (0.5–1 session).** In `visual_tags_derive.py`, `blurry` moves
FROZEN → DERIVED and `DERIVE_COLUMNS` gains `sharpness` and `sharpness_version`.
It is three-way:

- measured and blurry → emit it
- measured and not blurry → strip it
- unmeasured → carry the stored value

`sharp` stays FROZEN. *Verify:* both-direction tests for `merge_vlm_answer` and
`merge_stored_tags` (the data-loss guard), and the `derive-visual-tags`
before/after table.

**7. Consumers (optional, 0.5 session).** A `min_sharpness` search/MCP filter,
coverage on `/status`, a `rank_shoot` fallback for photos with no faces, and a
CLAUDE.md update.

**Estimate:** 6–7 sessions, plus 1–2 h of labelling, plus about 4–6 throttled
nights of backfill (13–22 N100 CPU-hours).

## Risks

- **Labelling at the wrong resolution.** Mitigation: the loupe on `/full`.
- **Too few blurry positives.** Mitigation: the oversampled strata.
- **Noise or phone over-sharpening still reading as sharp.** Mitigation:
  noise-corrected features and a per-camera breakdown.
- **HEIC, RAW and video.** `draft` works on JPEG only, so these pay a full decode
  or are skipped with a logged reason.
- **Merge semantics could lose data.** Mitigation: the step 6 tests.
- **NAS disk contention.** Mitigation: the step 5 throttles.
- **The eval gate fails.** Then only the number ships, which is still useful for
  ranking.

## Decisions

The planners had no unresolved disagreements. These are the handoff's four open
questions, with the recommendation both planners converged on. Owner
confirmation is pending.

| topic | option A | option B | choice | date |
|---|---|---|---|---|
| Storage | tags only | numeric score + JSON of all candidate metrics + version | B (recommended, pending owner) | 2026-09-26 |
| Stored 3,410 `sharp` / 1,284 `blurry` | keep where the measurement agrees | frozen until the eval, then replaced per photo as it is measured; unmeasured photos keep the stored value | B (recommended, pending owner) | 2026-09-26 |
| Where it runs | extend `rank_measure` | new `sharpness.py` + maintenance stage + batch step; `rank_measure` untouched | B (recommended, pending owner) | 2026-09-26 |
| Scope of `sharp` | derive both | derive `blurry` only; label `sharp`; retire it if it fires on >40% or P < 0.8 | B (recommended, pending owner) | 2026-09-26 |
| Order | code, then eval | labels and eval before any backfill | B (both planners) | 2026-09-26 |

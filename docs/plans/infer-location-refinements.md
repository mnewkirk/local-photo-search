# Inferred geotagging — post-M19 refinements

Follow-ups surfaced during the first real-library apply of M19
(`photosearch infer-locations`) on the 127k-photo NAS library. M19
itself shipped and is correct — these refinements tighten cascade
behaviour and close a few known gaps.

## Observed behaviour on the NAS (2026-04-21)

Dry-run before any writes:

```
25,281 candidates out of 127,185 no-GPS photos (cascade 776 rounds).
Skipped: no_anchor 96,697 · movement_guard 46 · no_date_taken 1,436 · below_min_confidence 0
Confidence:  >=0.90: 3,045  0.75-0.90: 1,115  0.50-0.75: 1,794
             0.25-0.50: 1,265  <0.25: 18,062
```

- **71% of candidates landed in the `<0.25` bucket.** Confidence decays
  as `0.7^hops`, so the long tail is almost entirely long-chain
  inferences. At hop 776 (the reported `cascade_rounds_used`), a
  candidate's confidence is `≈ 1e-120` — meaningless but still emitted.
- Movement guard only fired 46 times out of 25k. Either the library is
  well-behaved or the guard isn't doing much work in the cascade case
  (chains are short between anchors but long overall).

After applying `--min-confidence 0.75` (4,106 rows), a re-preview
produced new 842 / 487 high-confidence candidates — round-2 cascade from
the just-applied anchors. This is expected but reveals a compounding
risk (see Problem 1 below).

## Problems

### Problem 1 — Inferred anchors re-enter at full confidence

`photosearch/infer_location.py:_scan_photos` pulls every photo with
`gps_lat IS NOT NULL`, and `infer_locations()` seeds `anchor_data` with
`confidence=1.0` for all of them regardless of `location_source`. On the
second run, the rows that `--apply` just wrote are promoted from
"inferred, confidence=0.85" to "anchor, confidence=1.0" — the cascade
decay (`0.7^hops`) that protected against runaway chains within a single
run no longer applies across runs.

In practice this means:

- Each apply cycle can extend chains into territory the decay would have
  filtered within a single run.
- Using `--min-confidence 0.75` on each apply only guards the current
  run — it doesn't guard downstream runs from inheriting the trust.

### Problem 2 — Hop-count has no hard cap

The cascade is unbounded in chain length. The `cascade_rounds_used:
776` observed on the NAS is a single-photo chain depth, not iterations.
Anything past hop ~6 already has confidence `< 0.12` and is almost
certainly wrong. Letting the engine emit hop-200+ candidates wastes
compute and bloats the `<0.25` tail that a user then has to filter out.

### Problem 3 — `--max-cascade-rounds` is dead code

The flag exists on the CLI (`cli.py:infer_locations_cmd`) and the API
body parser (`web.py:_parse_infer_params`) but the cascade
implementation is **sequential promote-as-you-go**, not iterative
rounds. Each photo gets at most one inference attempt; there are no
"rounds" to cap. The flag silently does nothing.

Either implement rounds, or delete the flag. The sequential algorithm
is strictly better than rounds (each photo anchors to its nearest
predecessor instead of the round-1 real-GPS anchors), so deleting is
the right call — but then `cascade_rounds_used` also needs renaming
(currently reports `max(hop_count)`).

### Problem 4 — Preview and Apply each run inference from scratch

`/api/geocode/infer-apply` re-invokes `infer_locations()` instead of
reusing the preview run's candidates. On a 127k library this roughly
doubles perceived latency. Correct because the engine is deterministic
given identical params, but wasteful.

Workaround: users who ran preview can just click Apply immediately — the
server still does the work. Fixing this would require a short-lived
server-side cache keyed by params hash.

### Problem 5 — Preview samples are non-deterministic

`random.sample(candidates, 10)` in both `cli.py` and `web.py` is
unseeded. Re-previewing identical params shows different thumbnails,
which can feel like the engine is flaking even though results are
stable.

## Ordered action items

### Ship soon

1. **Cap hop depth.** Add a `--max-hops` CLI flag (default 6) and the
   matching API param. Inside `_infer_one_round`, short-circuit when
   `parent_hop_count + 1 > max_hops`. At hop 6, confidence floor is
   `0.7^6 ≈ 0.118`, already below any reasonable `--min-confidence`,
   and you cut the `<0.25` bucket to near-zero. Cheapest fix with the
   biggest UX win.
2. **Downweight inferred anchors on re-scan.** In `_scan_photos`, carry
   `location_confidence` through for rows where
   `location_source='inferred'`. Seed `anchor_data[photo_id]['confidence']
   = location_confidence` for those rows (instead of 1.0). This lets the
   decay math protect against cross-run compounding: a hop-3 inference
   with confidence 0.34 becomes an anchor at 0.34, not 1.0, and a photo
   keying off it pays the decay properly. EXIF rows still seed at 1.0.
3. **Delete `--max-cascade-rounds` and rename `cascade_rounds_used`.**
   The flag misleads; the field reports max hop depth, not round count.
   Either:
   - Rename the field to `max_hop_count` and drop the flag entirely, or
   - Add a real "stop emitting candidates if max_hop_count exceeded" cap
     and keep the flag honest.
   Recommend the rename+drop; item 1 already caps hops correctly.

### Iterate

4. **Treat `location_source='inferred'` as second-class when filtering
   by `location_source='exif'` on downstream features.** Map view and
   radius search (from `bulk-set-location.md`) may want to visually
   distinguish inferred vs exif, or hide inferred below a confidence
   floor.
5. **Preview → Apply cache.** Cache preview results server-side for
   ~60s keyed by `(params_hash, schema_version, photo_count)`. Apply
   reuses if hash matches, else re-runs. Saves one full scan on click.
6. **Seed `random.sample`.** `random.Random(hash(tuple(...))).sample(...)`
   using the candidate photo_ids as the hash input. Stable samples
   across re-previews of identical params.
7. **Confidence calibration.** The `0.7^hops × time_decay × side_bonus`
   formula is a guess. On a labelled subset (e.g. 50 photos that have
   both EXIF GPS and a plausible inference path), measure actual
   accuracy per confidence bucket and recalibrate. Might want to
   collapse to `0.85^hops` or change decay shape entirely.
8. **Consider refusing cross-day chains.** A photo at T=10:00 anchored
   to a photo at T=09:45 the same day is plausible; a photo at 23:58
   anchored to 00:03 the next day is a calendar accident, not a
   semantic one. Cheap guard: refuse when anchor and target
   `date_taken` date components differ AND `time_gap > 6h`.

## Tradeoffs / risks

- **Item 1 (hop cap) reduces recall.** Photos in the middle of long
  no-GPS stretches will be left un-inferred instead of picking up a
  garbage inference. That's the desired tradeoff given the 71%
  `<0.25` observation, but callers relying on "infer something rather
  than nothing" need to opt in with `--max-hops 30`.
- **Item 2 (downweight) changes headline candidate_count on re-runs.**
  Users iterating at `--min-confidence 0.75` will see fewer candidates
  per cycle. Document this; it's correct behaviour.
- **Item 3 (rename) is a one-time API break.** Frontend `/status`
  panel reads `preview.cascade_rounds_used`. Bump together.

## Critical files

- `photosearch/infer_location.py` — cascade algorithm, `_scan_photos`,
  `_infer_one_round`, `_find_flanking_anchors`
- `cli.py` — `infer-locations` command
- `photosearch/web.py` — `/api/geocode/infer-preview` and
  `/infer-apply`, `_parse_infer_params`
- `frontend/dist/status.html` — `InferLocationForm` renders
  `cascade_rounds_used`
- `tests/test_infer_location.py` — existing 20 tests to amend when
  changing cascade semantics

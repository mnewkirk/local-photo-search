# Video main-subject eval — results (2026-10-25)

`python evals/video_eval.py report` over `evals/model-evals` (run caches
F0/F1/S1/S2-default, 80 labelled clips; the 5 m2ts refreshed). Plan:
`video-support.md` Step 4.

## Labels

- Video subject: 90/90 "who" answers done; all 611 label segments of the 80
  scored clips labelled (322 person, 151 someone else, 92 non-person, 14 none,
  32 can't tell). 76 clips score (51 dev, 25 holdout).
- Video people: 280 clusters, 279 named by majority (196 someone else, 82
  person, 1 junk); 1,292 of 1,852 sampled faces named (the unnamed 560 are
  mostly clustering noise).
- Recheck set (15): not labelled → **self-agreement not measured**.
- Non-person comparer judgements (`label_judgements.json`): **not run**, so
  non-person labels match only on exact normalized text.
- Quality: **not evaluated** (owner decision 2026-10-25: sample has no
  positives; numbers-only per D11). Describe D0 gate: **not evaluated** (no
  runs, no Differences/temporal labels).
- Registration: probe check 20/20 agree; Live pairing 10/10 — both PASS.

## Subject (gates: top ≥ 0.80, presence P ≥ 0.95, segment ≥ 0.70 person and non-person, share MAE ≤ 0.15)

| variant | split | seg all [95% CI] | person | non-person | top [CI] | share MAE [CI] | presence P [CI] | pres R |
|---|---|---|---|---|---|---|---|---|
| F0 | dev | 0.29 [0.24, 0.33] | 0.33 | 0.05 | 0.29 [0.19, 0.43] | 0.51 [0.45, 0.57] | 0.13 [0.10, 0.17] | 0.85 |
| F1 | dev | 0.32 [0.28, 0.37] | 0.37 | 0.05 | 0.35 [0.24, 0.49] | 0.49 [0.42, 0.54] | 0.13 [0.10, 0.17] | 0.81 |
| S1 | dev | 0.28 [0.24, 0.33] | 0.30 | 0.18 | 0.31 [0.20, 0.45] | 0.34 [0.30, 0.38] | 0.13 [0.10, 0.17] | 0.81 |
| S2 | dev | 0.33 [0.28, 0.38] | 0.36 | 0.18 | 0.35 [0.24, 0.49] | 0.36 [0.32, 0.41] | 0.13 [0.10, 0.17] | 0.81 |
| F0 | holdout | 0.28 [0.22, 0.34] | 0.29 | 0.22 | 0.28 [0.14, 0.48] | 0.54 [0.45, 0.62] | 0.09 [0.05, 0.14] | 0.65 |
| F1 | holdout | 0.31 [0.25, 0.37] | 0.33 | 0.22 | 0.36 [0.20, 0.55] | 0.51 [0.42, 0.60] | 0.10 [0.06, 0.16] | 0.70 |
| S1 | holdout | 0.23 [0.18, 0.29] | 0.24 | 0.18 | 0.28 [0.14, 0.48] | 0.32 [0.27, 0.38] | 0.10 [0.06, 0.16] | 0.70 |
| S2 | holdout | 0.26 [0.21, 0.33] | 0.29 | 0.18 | 0.40 [0.23, 0.59] | 0.35 [0.29, 0.42] | 0.10 [0.06, 0.16] | 0.70 |

Segments: dev 374 (317 person / 57 non-person), holdout 205 (156 / 49).
Dev winner (S1 vs S2, tie band 5 pts): **S2**. Holdout: S1 and S2 tie
(0.23 vs 0.26), S1 wins as simpler. S1 does not beat F1 on people by 5 pts
(it is 9 pts behind). **Every gate fails on holdout for every variant**;
the plan's fallback ladder gives "ships: nothing" (presence P fails too).

The misses are not about where the bar sits. Two generous re-scorings still
miss every gate:
- Counting the obvious same-subject non-person pairs (coral reef / underwater
  scene, grouse / bird, mountain landscape / mountain range) as correct gives
  S1/S2 holdout non-person 0.43, all 0.29/0.32, top 0.36/0.48, MAE 0.31/0.34.
- Counting presence only for the 11 people the owner ever labels gives
  presence P of 0.45 on holdout and 0.64 on dev.

## Verdict

**Do not flip `PHOTOSEARCH_VIDEO_SUBJECT_SHARE`.** No variant/config is
within reach of any gate; keep the share hidden (the modal timeline stays).

## Top failure modes (segment counts over dev + holdout)

1. **The VLM picks an object or the scenery, not the person** (S1 153, S2 152).
   The owner labelled the person, and grounding's largest salience box was
   "baseball field", "landscape", "tree", etc. Examples: 251592
   `2026-03-22T212851_IMG_2467.mov` (10), 257004 `2025-10-12T173610_IMG_1596.mov`
   (8), 254461 `20220326104911 (1).m2ts` (7). The face-only variants instead
   say "none" when no face is in focus (F1 129: 256184
   `2025-05-17T235236_IMG_0196.mov`, 257180 `100_0095.MOV`, 255024
   `MAH06738.MP4`). Non-person subjects are mostly missed by F0/F1 (85
   object→none: 252152 `MVI_0040.MOV`, 252381 `Bears on the run.mp4`).
2. **Strangers get library names (identity).** The strict matcher puts
   teammates' names on faces the owner called "someone else" (S1 63
   segments: Carson Jones 17, Beckham Tinnel 16, Oliver Munoz 7). It also
   names Calvin as a teammate (Lucas Alaniz 9, Beckham 8, Asa Goldman 7: 44
   wrong-person segments). Sports-long clips 255805 `C0088.MP4`, 256209
   `C0147.MP4` and 251442 `C0207.MP4` list 15 to 22 "present" people each,
   and presence `present` has no duration gate (the production
   `face_row_eligible` gate is not applied). Gating it at 2 s only reaches
   P 0.16. Caveat: the owner labels teammates as "someone else" even when the
   library knows them. Some of these FPs may be right names that the owner's
   convention counts as wrong.
3. **The person is found but left unnamed** (S1 73, S2 115). S2's
   track-confirmation turns joined people into "someone else", which costs it
   most of its gain over S1. Examples: 251421 `C0185.MP4` (10), 256235
   `2025-05-20T184059_IMG_0221.mov` (10), 255210
   `2024-02-29T052838_….mp4` (7).

**Cluster naming does not bias the score.** Predictions name people with the
library strict matcher against a frozen replica snapshot, never with the eval
cluster names. Owner truth is the per-segment label. Of the 322
owner-labelled person segments, 95 (30%) belong to a person with no named
cluster face anywhere in that clip, and 160 (50%) have none in that segment.
That gap affects only how quickly the owner could label (run grouping), not
the scoring. All 7 people the owner labelled have library references.

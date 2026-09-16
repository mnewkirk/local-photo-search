# Verifying that a matched face is actually that person

**Status:** SHIPPED. `photosearch/face_verify.py`, `photosearch
verify-face-labels`, `GET /api/faces/verify-labels`, and the 🔍 Verify labels
panel on `/faces`. A second, independent test — **the rival test** — was added
2026-09-16; see the section at the foot of this doc.

## The case this is derived from

On 2026-09-12, 371 faces were hand-labelled across 15 players. Two of them,
tagged **Oliver Munoz**, were really **Franklin Martinez**. Here is what found
it, in the order it happened, because the order is the argument:

| step | result |
|---|---|
| `label-conflicts` at the default eps=0.80 | **0 conflicts** — missed it |
| eps sweep 0.55 → 0.95 | surfaced at 0.85 (2 v 2) and 0.95 (9 v 2) |
| distance to each person's faces | disputed → Franklin **0.911 / 0.971**, → Oliver **1.156 / 1.165** |
| **closest genuine Oliver↔Franklin pair** | **1.203** |
| contact sheet of the crops | confirmed by eye in seconds |

**The decisive number is 1.203.** The disputed faces sit closer to Franklin
than Oliver and Franklin *ever get to each other*. That is not a threshold
someone chose; it is measured from this shoot, for this pair.

## Why the existing tools missed it

- **`verify-person-matches` uses a global `--min-dist 1.30`.** Both disputed
  faces are at 1.16 from Oliver — comfortably "fine" by that threshold. A
  single library-wide number cannot be right for both a pair of lookalike
  8-year-olds and two unrelated adults.
- **`label-conflicts` needs the right eps.** It found nothing at 0.80 and the
  truth at 0.95. You cannot know the right radius in advance, and running a
  sweep every time is both slow and an invitation to read noise as signal
  (at eps=1.00 it "found" a 4-way conflict that is just DBSCAN fusing the team).
- **The person inspector** sub-clusters *within* one person. It never compares
  a face against the other people present, which is the only comparison that
  settles this.

All three are radius- or threshold-dependent. The margin test is neither.

## The measure

For a face `f` labelled person `P`, inside a scope (one shoot):

```
d_own(f)        = min distance from f to P's OTHER reference faces   (leave-one-out)
d_other(f, Q)   = min distance from f to Q's reference faces,  for every other Q in scope
Q*              = the nearest such Q
margin(f)       = d_own(f) - d_other(f, Q*)        > 0  => closer to someone else
sep(P, Q)       = how close P and Q ever genuinely get  (see below)
```

Two signals, and they say different things:

- **`margin > 0`** — "this face looks more like Q than like P." Suggestive.
- **`d_other(f,Q*) < sep(P,Q*)`** — "this face is closer to Q than P and Q are
  to each other." **Decisive**, and the test that settled Oliver/Franklin.

### Why calibration beats a threshold

`sep(P,Q)` makes the test adapt to the pair. Two children who genuinely look
alike have a small `sep`, so the bar to accuse a label rises automatically and
the tool goes quiet — which is correct, because that is exactly the regime
where ArcFace is known to be unreliable (siblings stay close; see
`project-calvin-temporal-matches-unreliable` and the embedder bake-offs that
found AdaFace, MagFace, QMagFace and a VLM all failing on the same set). Two
children who look nothing alike have a large `sep`, and a mislabel between them
is flagged loudly. A fixed 1.30 gets both cases wrong in opposite directions.

## Details that decide whether this works

- **Leave-one-out is mandatory.** Compare `f` against P's references *excluding
  f*, or every face is distance 0 from itself and nothing is ever flagged.
- **References are `manual` + `strict` only.** `temporal` is ~4% accurate on
  these shoots; seeding the reference set with it poisons `d_own`.
- **`sep` is a low percentile, not the raw minimum.** A single mislabelled face
  on either side drags a raw `min` to near zero and silently disables the test
  for that pair — the failure would look like "no conflicts found". The
  implementation uses the 5th percentile of all cross-pairs.
- **Contaminated own-set.** If several of P's faces are really Q, `d_own`
  looks fine and the margin collapses. Mitigation: report how many of P's
  references are themselves flagged; a person with many flags needs their
  whole label set reviewed, not individual swaps.
- **Minimum reference count.** Below ~3 trusted faces a person's set is too
  thin to calibrate against; those are reported separately rather than scored.

## What the human sees

The measure ranks; a person decides. What made the real call take seconds was
three strips side by side:

```
  [ the disputed face(s) ]   [ P's references ]   [ Q*'s references ]
```

That is the UI: one row per flagged face, the two candidate identities beside
it, and **Keep** / **Reassign to Q\*** / **Reject** buttons. Reject writes
`match_source='rejected'` so a later `match-faces` cannot undo it.

## Explicitly NOT in the decision path: a VLM

Tempting, and already tested: `evals/vlm_face_compare.py` alongside the AdaFace
/ MagFace / QMagFace bake-offs found that **none** of them separate the hard
cases, and qwen2.5-vl mode-collapses on Likert scoring. A VLM in the decision
path would add cost and false confidence. The margin test is cheap, explainable
and — on the one case we have ground truth for — sufficient.

## Relationship to the existing tools

| tool | question | needs a radius/threshold |
|---|---|---|
| `label-conflicts` | does a cluster mix two names? | yes (eps) |
| person inspector | which of P's faces are unlike P? | yes (eps) |
| `verify-person-matches` | which of P's faces are far from P? | yes (min-dist) |
| **`verify-face-labels`** | **is this face closer to someone else than P and they ever get?** | **no** |
| **the rival test** | **does another face in this PHOTO claim the label better?** | **no** |

They are complementary: clustering finds *groups* that mix, the margin finds
*individual* faces that sit on the wrong side. Keep both.

## The rival test (added 2026-09-16)

### The case

Photo 244260, found by eye, not by a tool: the left face was labelled
**Franklin** (wrong) and the face two to the right was *clearly* Franklin and
**untagged**. Replayed against the replica:

| face | labelled | d → Franklin refs | d → Beckham refs |
|---|---|---|---|
| 505376 | Franklin | 1.301 | **1.010** |
| 505377 | *(untagged)* | **0.949** | 1.273 |

Franklin↔Beckham separation (p5) is 1.210, so both are decisive in opposite
directions. The same mislabel was still live one frame later in photo 244261
(505383 at 1.28, untagged 505381 at 0.944) and is what the shipped detector
found first.

### Why it is not just more of the margin test

The margin test needs the true identity to already be a **registered person with
references**. Here the right answer was an *unlabelled* face, so no comparison
against named people could name it. The rival test needs nothing but the photo.

It also produces a **repair** rather than a rejection: move the label from face A
to face B, one click, both errors fixed.

### The rule

A person appears in a photo at most once. `resolve-duplicate-persons` already
leans on that, but only across faces labelled the *same* person. Extending it to
unlabelled faces is the new part.

```
d_self    = min L2 to P's trusted refs, leave-one-BURST-out
d_rival   = the same, for each other face in the photo
radius(P) = p90 of how far a GENUINE face of P lands from P's own refs
```

Four gates, all required:

1. the rival is closer to P than the incumbent is
2. `d_rival < radius(P)` — the rival is plausibly P at all
3. `d_self > radius(P)` — the incumbent is not
4. the rival is not better explained by some other person

Gates 2+3 sandwich the radius between the two faces. That is why the answer is
almost threshold-free: on 2026-09-12 the same single finding comes back at p75,
p85, p90, p95 **and** p100. Gate 2 alone gets looser as the percentile rises and
gate 3 gets tighter by exactly as much.

`radius(P)` is the per-person analogue of `separation(P,Q)`. A person whose
faces vary a lot demands a closer rival automatically — measured, Franklin 0.95
against Calvin 0.76 — so there is no global number to tune, which is the
property the whole module exists to preserve.

### The assignment

Gate 1 is decided by a one-to-one assignment, not a per-label argmin:
`scipy.optimize.linear_sum_assignment` over a face × person cost matrix, with
`n` appended zero-cost dummy columns as the "assign this face to nobody" option.

- **Columns are the people already labelled in that photo.** The question is
  only ever "is this label on the wrong face", never "who is that stranger" —
  opening the columns to the whole roster would invent labels for unlabelled
  faces, a different tool.
- **Costs are normalized to `d - radius(P)`**, so zero means "as close as a
  genuine face of them gets" and one uniform dummy price works for every column.
  A raw-distance matrix would let a tight-radius person outbid a loose one for
  every face in the photo.
- Without the one-to-one constraint, two labels in one photo can both claim the
  same rival, and applying both swaps writes two people onto one face — pinned
  by `test_one_rival_cannot_satisfy_two_labels`.

### Measured

| shoot | photos | faces | raw rivals | strong | verdict |
|---|---|---|---|---|---|
| 2026-09-12 | 919 | 2,671 | 507 | **1** | true (photo 244261) |
| 2026-08-29 | 976 | 2,920 | 419 | **3** | all true, confirmed by eye |

For contrast, `verify-face-labels` on the same 2026-09-12 scope returns **108
findings, 96 of them decisive** — 505383 is in there at #2, buried under
Calvin-temporal noise. The rival gate narrows that to 1.

Ranking the ungated list by `displaced = d_self - d_rival` is a strong signal on
its own: the true positives sat at the top on both dates (0.335 against a next
of 0.178; 0.357 / 0.314 / 0.251 against the rest).

### Load-bearing details

- **Leave-one-burst-out saved this case too.** Without it, 505376 measures
  **0.716** from "Franklin" and looks correct, because 505383 — the other frame
  of the same mislabel, one second away — is vouching for it. Franklin's
  reference set was otherwise clean (all 19 audited by hand). This is the second
  independent case where the burst alibi, not reference contamination, was the
  thing that hid a mislabel.
- **Swap is two `bulk-assign` calls from the client, clear-then-assign.**
  Assigning first would briefly put one person on two faces in a photo. A
  dedicated endpoint would have to re-implement `_mirror_face_labels`' replica
  semantics — the duplication that already produced the `rejected` mirror bug.
- **Burst siblings are independently gated, not copied.** Applying the group is
  exactly as safe as applying each one alone.
- **Reference pollution blunts it.** Enough wrong faces in P's reference set
  inflates `radius(P)` until gate 3 stops firing. It degrades quietly toward
  silence rather than toward false accusations, which is the right direction,
  but it is the reason to keep reference sets `manual` + `strict` only.

## Still to build

- Iterative re-scoring after accepting a correction, so fixing one label
  re-calibrates the pair rather than requiring a full re-run.
- A scope wider than one day. Calibration is per-pair and per-scope; whether
  `sep` is stable across shoots (different light, different kit) is untested.
- The rival test currently only questions labels that already exist. The same
  assignment could *propose* a name for an unlabelled face that is decisively
  someone's — deliberately out of scope for now, since that is a matcher, not a
  verifier.

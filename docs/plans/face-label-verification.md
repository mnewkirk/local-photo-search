# Verifying that a matched face is actually that person

**Status:** design + core implemented (`photosearch/face_verify.py`,
`photosearch verify-face-labels`). UI not built.

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

They are complementary: clustering finds *groups* that mix, the margin finds
*individual* faces that sit on the wrong side. Keep both.

## Still to build

- The review UI described above (the scorer already emits everything it needs).
- Iterative re-scoring after accepting a correction, so fixing one label
  re-calibrates the pair rather than requiring a full re-run.
- A scope wider than one day. Calibration is per-pair and per-scope; whether
  `sep` is stable across shoots (different light, different kit) is untested.

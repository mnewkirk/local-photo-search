# Faces — history and traps

Face clustering, label integrity (verify-labels, rival test), the one-person-per-photo invariant, `rejected` labels, encoding cache. Condensed rules: `.claude/rules/faces.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 2803) on 2026-10-19 -->
## Face clustering

New faces land with `cluster_id = NULL` — per-batch clustering was removed from
`index_directory()`, `_index_collection()`, and `worker_api.submit_results()`
because every batch independently restarted IDs at 0 and collided
("Unknown #0" was the union of the first face of every batch).

Grouping is an on-demand step:

```bash
$DC run --rm photosearch recluster-faces \
  [--eps 0.55] [--min-samples 3] \
  [--no-session-stacking] [--session-eps 0.50] [--session-window 60] \
  [--min-det-score 0.65] [--min-bbox-edge 60] \
  [--dry-run]
```

Runs global DBSCAN over every `person_id IS NULL` encoding, then (by default)
a **session-stacking second pass**: union-find over the DBSCAN noise points,
linking pairs whose L2 distance is within `--session-eps` AND whose
`date_taken` is within `--session-window` minutes. Components of size ≥ 2
become new clusters continuing past the DBSCAN id range, recovering
same-person-same-event groups that min_samples=3 had discarded. Pass
`--no-session-stacking` for DBSCAN-only behavior. `ignored_clusters` is
cleared in the same transaction (IDs are fully renumbered). Loader uses
`np.frombuffer` over concatenated BLOBs — 120k faces decode in seconds on
an N100; DBSCAN at 512-dim with ball_tree is the dominant cost after that.

<!-- moved verbatim from CLAUDE.md (line 2831) on 2026-10-19 -->
### Quality pre-filter (schema v20)

`faces.det_score` stores InsightFace's detection confidence `[0, 1]`,
stamped on every new face from `detect_faces()` onward. `recluster-faces`
and `split-cluster` both filter out low-quality detections via a SQL
WHERE clause before DBSCAN runs:

```sql
WHERE (det_score IS NULL OR det_score >= :min_det_score)
  AND MIN(bbox_bottom - bbox_top, bbox_right - bbox_left) >= :min_bbox_edge
```

Defaults (tunable via `--min-det-score` / `--min-bbox-edge` or
`CLUSTER_MIN_DET_SCORE` / `CLUSTER_MIN_BBOX_EDGE` in `faces.py`):

- `min_det_score=0.65` — InsightFace det_score below this is usually a
  misfire (off-angle head, partial face, false positive).
- `min_bbox_edge=60` — shorter bbox edge in pixels. Smaller faces
  produce noisier encodings that cluster together by "junkness"
  rather than identity.

**NULL is grandfathered** — faces indexed before v20 have
`det_score IS NULL` and pass the filter unconditionally, so libraries
upgrading in place keep clustering as before. As users re-index, new
rows get real det_score values and the filter tightens. No forced
backfill.

`split-cluster` applies the same filter: faces below threshold drop to
`cluster_id=NULL` (not left in the source cluster) so a split leaves no
junk residue. The summary includes `filtered_out_count` alongside
`noise_count`.

Evidence motivating this filter: a 434-face cluster on the NAS split
cleanly at `eps=0.50` into one 75-face real-person sub-cluster plus
182 NULL'd noise faces. Low-quality faces were the primary attractor
cause; filtering them pre-DBSCAN prevents the attractor from forming
in the first place.

<!-- moved verbatim from CLAUDE.md (line 2919) on 2026-10-19 -->
## Face-label integrity: is this face actually that person?

Four tools, and the differences between them matter — they fail in different
places and were each added because the previous one missed something real.

| tool | question | needs tuning? |
|---|---|---|
| `label-conflicts` (`GET /api/faces/label-conflicts`) | does a cluster mix two names? | **yes** — eps |
| person inspector (`/inspect`) | which of P's faces are unlike P? | **yes** — eps |
| `verify-person-matches` | which of P's faces are far from P? | **yes** — `--min-dist` |
| **`verify-face-labels`** | **is this face closer to someone else than P and they ever get?** | **no** |
| **`--rivals-only`** (same tool) | **does another face in this PHOTO claim the label better?** | **no** |

**`verify-face-labels` is the one to reach for first.** `photosearch/face_verify.py`,
`GET /api/faces/verify-labels`, and the **🔍 Verify labels** panel on `/faces`
(any date range). For each labelled face it measures the distance to its own
person's references *and* to every other person in scope, then calibrates
against how close those two people genuinely get:

```
margin   = d_own - d_other                  > 0  -> looks more like someone else
decisive = d_other < separation(P, Q)       closer to Q than P and Q ever get
```

It exists because on 2026-09-12 two faces hand-labelled **Oliver Munoz** were
really **Franklin Martinez**, and the other tools missed them:
`verify-person-matches` compares against a global 1.30 while the faces sat at
1.16, and `label-conflicts` reported **zero** at its default eps=0.80 (it only
surfaced at 0.85/0.95). A single global threshold cannot be right for both a
pair of lookalike 8-year-olds and two unrelated adults; the per-pair
calibration can.

Four implementation details that are load-bearing:

- **Leave-one-BURST-out, not leave-one-face-out.** The first version failed its
  own motivating case: the two mislabelled faces were 1s apart in one burst, so
  each one's nearest "Oliver" reference was *the other impostor* at 0.579,
  collapsing the margin below zero and hiding the error completely. Excluding
  the burst gives 1.156 and it is decisive. `BURST_WINDOW_SECONDS = 3`.
- **References are `manual` + `strict` only.** `temporal` is ~4% accurate on
  these shoots and would poison the baseline the whole test rests on.
- **`separation` is the 5th percentile, not `min()`.** One bad label on either
  side drags a raw min to ~0 and silently disables the pair — surfacing as "no
  conflicts found", the most dangerous possible failure.
- **No VLM in the decision path.** `evals/vlm_face_compare.py` plus the
  AdaFace/MagFace/QMagFace bake-offs already showed none of them separate the
  hard cases. The calibration handles that regime by going quiet: lookalikes
  have a small separation, so the bar to accuse rises automatically.

Measured on 2026-09-12: 397 manual labels → **8 flagged (2%), all decisive**,
the known ground-truth pair at #2 and #4. Design + the full case:
`docs/plans/face-label-verification.md`.

<!-- moved verbatim from CLAUDE.md (line 2972) on 2026-10-19 -->
### The rival test — the right face is usually in the same photo

`face_verify._photo_rivals`, `?rivals_only=1` on the API, `--rivals-only` on the
CLI, **RIVAL IN PHOTO** + a one-click **Swap** in the panel.

A person appears in a photo at most once. So if another face in the SAME photo
is a better claimant to the label than the face carrying it, the label is on the
wrong face — and the right face is right there. The invariant was already
enforced across faces labelled the *same* person (`resolve-duplicate-persons`);
this extends it to **unlabelled** faces, which is the whole point: the rival
does not have to be a registered person, so this sees what the margin test
structurally cannot.

It comes from photo 244260 (2026-09-12), found by eye: "Franklin" was on the
wrong boy and the real Franklin was **untagged** two faces to the right. Four
gates, all required:

1. the rival is closer to P than the incumbent is
2. the rival is plausibly P at all — `d_rival < radius(P)`
3. the incumbent is not — `d_self > radius(P)`
4. the rival is not better explained by someone else

**`radius(P)` is the per-person analogue of `separation(P,Q)`** — the p90 of how
far a GENUINE face of P lands from P's own references, same leave-one-burst-out
rule. That is what makes the test threshold-free across people with very
different spreads (Franklin 0.95, Calvin 0.76). Gates 2+3 *sandwich* the radius
between the two faces, which is why the result barely moves with the percentile:
on 2026-09-12 it returns the same single finding at p75, p85, p90, p95 **and**
p100.

**Gate 1 comes from a one-to-one assignment** (`scipy.optimize.linear_sum_assignment`
over a face × person cost matrix, `n` zero-cost dummy columns as the
assign-to-nobody option), not a per-label argmin — otherwise two labels in one
photo both claim the same rival and "Swap" writes two people onto one face.
Costs are normalized as `d - radius(P)` so zero means "as close as a genuine
face of them gets"; a raw-distance matrix would let a tight-radius person outbid
a loose one for every face.

Measured across both soccer shoots (1,895 photos, 5,591 faces): **4 findings,
all true positives** — verified by eye against the crops. On 2026-09-12 that is
108 verify-labels findings (96 "decisive", mostly Calvin-temporal noise) narrowed
to **1**. The value is triage plus a repair: a rival names the right face, so it
is the only finding a reviewer fixes in one click instead of adjudicating.

**The burst-alibi rule earned its keep a second time here.** Without
leave-one-burst-out, face 505376 measures **0.716** from "Franklin" and looks
correct — because 505383, the other frame of the *same* mislabel one second
away, is vouching for it. Franklin's reference set was otherwise clean (all 19
audited). Any change to this path must keep the exclusion or it will confidently
point at the wrong face.

**Swap is two `bulk-assign` calls from the client, clear-then-assign**, not a new
endpoint: assigning first would briefly put one person on two faces in a photo,
and a dedicated endpoint would have to re-implement `_mirror_face_labels`'
replica semantics — the exact duplication that already shipped the `rejected`
mirror bug. **Burst siblings** (the same mistake in consecutive frames, which is
how 244260 and 244261 both went wrong) are grouped for a single action, but each
is an *independently gated* finding, not a blind copy — so applying the set is as
safe as applying them one at a time.

<!-- moved verbatim from CLAUDE.md (line 3032) on 2026-10-19 -->
### The panel asks about TRUSTED labels only

`face_verify.TRUSTED_LABEL_SOURCES = ("manual", "strict", "merge_review")` is
the default for `verify_labels(label_sources=...)`, the API `?sources=`, the CLI
`--sources`, and the panel's source dropdown. `'all'` includes everything;
`'temporal'` isolates it.

**`temporal` is off by default because the panel is per-face Keep / Reject /
Swap, and that is the wrong shape of work for a ~4%-accurate matcher.** On
2026-09-12 the unfiltered list was **108 findings, 104 of them temporal Calvin**
— one kid over-matched across the whole field including the opposing team — and
the 4 real ones were buried 26:1. Filtered: **4 findings, all manual, all
decisive, the rival first.** The fix for the temporal pile is a bulk clear of
the person + date, not 104 individual verdicts.

**Nothing is silently dropped.** `stats.hidden_by_source` /
`hidden_by_person` come back and the panel prints "Not shown: 104 temporal —
Calvin (104)". **The filter is applied LAST**, after calibration and after the
per-photo assignment, so a hidden temporal face still competes for its photo's
labels and still shapes `separation`/`radius` — otherwise the numbers would
depend on which rows you asked to see. `test_hiding_a_source_does_not_change_the_numbers`
pins that.

Corollary worth knowing: a mislabel *inflates the flag count around itself*.
While 505383 was still wrongly "Franklin" on the replica, 505376 (correctly
Beckham) was flagged as looking like Franklin — because 505383 was in Franklin's
reference set. Fixing the rival makes the phantom finding go away; it is not a
separate error to chase.

Tests: `tests/test_face_rivals.py` (11 cases — the unlabelled-claimant shape,
ordering, the correct-label-beside-a-stranger false positive, gate 4, the
one-rival-two-labels assignment invariant, burst linkage, the off switch, and
the source filter's don't-change-the-numbers guarantee).

<!-- moved verbatim from CLAUDE.md (line 3203) on 2026-10-19 -->
### `match_source='rejected'` — a human "no" that survives auto-matching

`faces.REJECTED_MATCH_SOURCE`. Written whenever a person clears a face
(`POST /api/faces/{id}/clear`, or `bulk-assign` with a null name, which is what
the Verify-labels **Reject** button posts), and filtered out of **both** matcher
candidate queries via the shared `faces.MATCHABLE_SQL`.

This exists because on 2026-09-14 a routine `match_faces` sweep re-applied 164
Calvin temporal matches that had been deliberately removed the day before. The
only marker available then was `dedupe_unmatched`, which the matchers never
consulted, and a plain `NULL` is indistinguishable from never-matched.

**Deliberately distinct from `dedupe_unmatched`.** That is
`resolve-duplicate-persons`' automatic tie-break when it keeps one face per
(photo, person); the losing face may legitimately match a *different* person
later, so it stays matchable. Filtering it instead was the tempting one-line fix
and would have frozen ~9.6k faces from the June cleanup out of matching forever.

`MATCHABLE_SQL` is shared **because the bug was one query, not both** — strict
and temporal each had their own `person_id IS NULL`, and a fix applied to one
would have looked complete. A test asserts both use the shared fragment.

**Replica gotcha:** `web._mirror_face_labels` duplicates the authoritative
write's semantics rather than reading them back, so it had to be taught
`rejected` too. It silently wrote `NULL` locally while the NAS recorded the
rejection — and a replica-side face-state export would then ship that back up as
"no opinion" and undo the rejection by a longer route.

<!-- moved verbatim from CLAUDE.md (line 3429) on 2026-10-19 -->
## Face over-matching cleanup & the one-person-per-photo invariant

Temporal matching (`match-faces --temporal`, `match_source='temporal'`) boosts
recall but over-matches: it tags the same person on 2+ faces in one photo
(rampant on kids at events — ~1600 photos had 2+ "Calvin" faces). A person
appears in a photo at most once, so any such photo has >=1 false match.

**`dedupe-person-faces --person NAME --references <photo-ids> [--min-gap 0.15]
[--report PATH] [--apply]`** — reference-based, **source-gated** cleanup for one
person. The reference set is a small curated list of photo-ids with one clear
face of them each, ideally spanning years (Calvin used one-per-year 2016-2026;
handles aging). Per double-person photo: **1 trusted (strict/manual) face →
keep it, drop the temporal extras** (decided by source, no distance, no
true-positive risk); **all-temporal → keep the face closest to the references,
drop the rest if the distance gap > `--min-gap`**; **2+ trusted → leave for
review**. `--report` writes an HTML keep-vs-remove **face-crop** gallery (crops
from `PHOTOSEARCH_NAS_URL`), least-confident first.

**`resolve-duplicate-persons [--apply]`** — universal safety net guaranteeing
the invariant for **every** person: for each (photo, person) with 2+ faces, keep
one by priority **manual > strict > temporal**, tie-break **det_score then bbox
area**; unmatch the rest. Run the reference-based dedupe first for precise
people, then this to sweep everyone.

**`verify-person-matches [--person NAME] [--min-dist 1.30] [--report PATH]`** —
finds MIS-tagged faces (a person tagged on someone *else's* face, vs. the
duplicate case above). Per person, the trusted (strict/manual) faces form an
auto reference set; every face of theirs is scored by min L2 distance to it and
the farthest (> `--min-dist`) are flagged — usually temporal over-matches onto a
different person. Report-only + an HTML gallery grouped by person. Run on the
replica; unmatch confirmed ones on the NAS. **Catches non-sibling mis-tags well;
sibling confusion (Calvin↔Ellie) stays close** — the ArcFace limit. Real run
(2026-06-21): 278 suspects (Calvin 148, Ellie 72, Matt 40, Nicole 18), all
confirmed mis-tags, unmatched.

**`restore-unmatched-faces [--apply]`** — both commands above snapshot
`(face_id, person_id, match_source)` into an on-demand **`face_dedupe_undo`**
table before nulling `person_id` (unmatch sets `match_source='dedupe_unmatched'`),
so the bulk cleanup is **reversible**. Restore re-applies the snapshot to faces
still unmatched.

Applied to the NAS 2026-06-20: 889 (Calvin reference) + 5382 (global) = **6271
faces unmatched, 4948 → 0 duplicate-person groups**, all reversible.

**Embedder/VLM swap is a tested dead-end — don't re-explore.** The hard
"wrong-green" cases (a photo where *neither* face is really the person, both
within temporal tolerance) are NOT separable by a better recognition head:
ArcFace (current), AdaFace IR-101, MagFace, QMagFace, and a VLM (qwen2.5-vl)
all fail on the same controlled set (`evals/adaface_compare.py`,
`magface_compare.py`, `vlm_face_compare.py`). MagFace quality is *inverted*
(wrong faces are higher quality than awkward-angle correct ones). Root cause is
identity ambiguity between similar-looking kids at the same event. The
within-photo **relative** comparison is the only robust signal. Per-year
references are worse than all-time.

<!-- moved verbatim from CLAUDE.md (line 3484) on 2026-10-19 -->
### Face-encoding cache

`PhotoDB.get_face_encodings_cached` serves face encodings from a process-wide
LRU (`db._FaceEncodingCache`, float32, ~40k faces / ~80 MB). Used by
`face_suggest` (Find more of this kid) and `face_verify` (Verify labels). It
needs no invalidation. Encodings are only ever INSERTed or DELETEd, never
updated, and `faces.id` is AUTOINCREMENT. So a relabel changes which ids a
caller asks for, never what an id's encoding is. It is keyed by the DB's real
path. Measured on a replica copy: suggest for Koa over Sep 1–27 went from
12.3 s to 1.9 s on the first call and 0.4 s on later calls; verify-labels for
one day went from 2.0 s to 0.13–0.2 s. Outputs are identical. If an
in-place encoding update is ever added, it must also evict that face from
this cache.

<!-- moved verbatim from CLAUDE.md (line 3528) on 2026-10-19 -->
### Face detection / matching gotchas

- **Big backlog**: clustering + matching are on-demand and fell far behind after
  the big imports (Google Takeout / phone ingest). At 2026-06-20: of 331k faces,
  only 51k (15%) matched to a person, 77k unmatched-but-clustered, and **202k
  never clustered** (`person_id IS NULL AND cluster_id IS NULL`). Those never-
  clustered faces are **not** shown as bare "Unknown #" — the `/api/faces/groups`
  cluster query requires `cluster_id IS NOT NULL`, so they were invisible on
  `/faces` entirely. See "Reviewing never-clustered faces" below.

<!-- moved verbatim from CLAUDE.md (line 3537) on 2026-10-19 -->
### Reviewing never-clustered faces (the "Unclustered" bucket)

Every face lands with `cluster_id = NULL` and only `recluster-faces` (global,
~an hour on the desktop for 230k encodings) mints `Unknown #N`. Nothing
schedules it, so **everything ingested since the last recluster is
unreviewable** — the symptom is `/faces?filter=unknown` filtered to a recent
day showing only named people and "0 unknowns" while the photos themselves
clearly have faces.

`/api/faces/groups` therefore emits one extra group, `type: "unclustered"`,
covering `person_id IS NULL AND cluster_id IS NULL` — **but only when a content
filter (date/place/query/camera) is active.** Unfiltered it would be a
150k-face blob with no identity structure, so it is deliberately suppressed
there. Open it via `GET /api/faces/group/unclustered/0/photos` with the *same*
filters (the path id is ignored); an unfiltered call is a **400**, not a very
slow full-library scan.

Measured on the 259k-face replica: the group query is ~35 ms and the photo
fetch ~70 ms on the busiest day in the library.

On `/faces` the bucket has no Rename/Merge/Split/Ignore and is excluded from
multi-select — every one of those reassigns *all* faces in a group, which is
right for a cluster and catastrophic for a bucket holding many different
people. Review it face-by-face in the photo grid. It is not a substitute for
`recluster-faces`; it is how you review a day without running one.

- **Babies don't auto-match**: infant faces ARE detected and cluster fine, but
  ArcFace can't bridge baby→child, so `match-faces` won't attach them to the
  grown person. Recover by reclustering then **manually assigning the baby
  clusters** in `/faces`/`/merges`. (A baby face also can't seed a reference set
  until it's tagged.)
- **Some photos are genuinely undetectable**: e.g. photo 153553 returns 0 faces
  even re-detecting at det_size up to 2048px — InsightFace just can't find the
  (turned-away/occluded) face. Not a resolution problem; nothing recovers it.
- **Duplicate detections exist**: occasional identical-bbox face rows (one face
  detected twice). Harmless while unmatched; once both get a person they become
  a duplicate-person photo that `resolve-duplicate-persons` cleans.

<!-- moved verbatim from CLAUDE.md (line 3575) on 2026-10-19 -->
### Searchable face clusters

`GET /api/faces/groups` takes content filters — `date_from`, `date_to`,
`location`, `q` (semantic/CLIP), `person` — and returns only clusters/persons
with a face in matching photos, ranked by how many matching photos they contain
(`sort=relevance`), singletons included. It reuses `search_combined` for
semantic/person queries and a direct date/location SQL scan otherwise
(matching photo-ids go into a temp table, joined into the group queries). The
`/faces` page has a **"Filter clusters:"** bar (date range + place + search,
applied on submit) backed by it. This is the first-class version of the manual
distance/date-ranked galleries — e.g. `?date_from=2018-10-01&date_to=2018-12-31`
narrows ~33k unknowns to the baby-era clusters. The picker that's been used to
recover baby Ellie / young Calvin: filter Unknown by date/place, eyeball, assign.

<!-- moved verbatim from CLAUDE.md (line 3589) on 2026-10-19 -->
### EXIF-oriented image dimensions

`image_width`/`image_height` are stored **EXIF-oriented** (swapped for
orientation 5-8): InsightFace bboxes are computed on the oriented image and the
preview is `exif_transpose`d, so raw dims misplaced the overlay boxes on rotated
photos (~18% of the library). `exif.py` swaps forward; `backfill-image-orientation`
fixes existing rows (header-only read, run where the files live, idempotent).
Face *data* was always correct — this was overlay-rendering only.

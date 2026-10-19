# Face management CLI

Read before running face commands: clustering, merges, splits, unmatch-person, rank_shoot, desktop recompute, over-matching cleanup.

<!-- moved verbatim from CLAUDE.md (line 3066) on 2026-10-19 -->
### Bulk-undoing one person's bad labels — `unmatch-person`

When a whole `match_source` is wrong for a person, adjudicating is hopeless:
Calvin carries **6,331 `temporal`** faces library-wide. Measured against his 745
**manual** references (leave-one-burst-out), the populations do not overlap:

| source | n | p10 | p50 | p90 | >1.15 |
|---|---|---|---|---|---|
| manual | 745 | 0.309 | 0.478 | 0.600 | 1% |
| merge_review | 9 | 0.586 | 0.633 | 0.671 | 0% |
| strict | 11,669 | 0.808 | **1.053** | 1.145 | 9% |
| temporal | 6,331 | **1.123** | 1.248 | 1.313 | **85%** |

temporal's *best decile* is worse than strict's median. Contact sheets confirm
it: a strict sample is the same boy at every age; a temporal sample is a dozen
different kids, both teams. **`strict` is fine — including its far tail**, which
is hard poses (profiles, hats, low light), not other people. So the answer to
"are they all temporal?" is effectively yes.

```bash
photosearch unmatch-person --person Calvin                       # preview
photosearch unmatch-person --person Calvin --min-dist 1.05 \
    --snapshot /data/calvin-temporal.json --apply
photosearch restore-unmatch --from /data/calvin-temporal.json --apply
```

Module `photosearch/bulk_unmatch.py`; tests `tests/test_bulk_unmatch.py`.
**Run the CLI on the NAS** — it writes, and the NAS is the sole writer.

**Or review it by hand: the ✂ Bulk unmatch panel on `/faces`.** Person picker,
optional `+strict` and a distance gate, then a farthest-first grid of crops with
the page's existing multi-select (click, shift-click range, Select all / > 1.15 /
> 1.00). **The grid is the dry run and the selection is the confirmation**, so
there is no separate preview step. Farthest-first matters: you scan from the top
and stop when you start recognising the person.

- `GET /api/faces/unmatch-preview` — **a date scope is REQUIRED** (400 without
  one). Library-wide is 6,331 faces for Calvin alone: not reviewable in a grid,
  and not something to Select-all by accident.
- Applying reuses `POST /api/faces/bulk-assign` with a null name, so the write
  is the audited one — `rejected`, plus the replica's write-NAS-then-mirror —
  rather than a second implementation of it.
- **Undo in the panel is in-session only** (it re-assigns exactly the ids just
  cleared). For an undo that outlives the tab, use the CLI's `--snapshot`.
- **Warm the crops first or the grid is unusable.** Face crops are generated on
  first view — a full image decode, ~2 s on the N100 — so a cold 120-face grid
  dribbles in over minutes. `photosearch warm-face-crops` now takes
  `--date-from` / `--date-to`, which AND onto the person scope, and there is a
  **Warm face crops** card on `/admin/maintenance`
  (`POST /api/admin/warm-face-crops`, SSE, throwaway sibling container, shares
  the long-job lock). It defaults to **every** face rather than matched-only,
  because the panels that need warming browse unknown clusters too — so scope
  it by date, or you are asking the N100 for hours of work. The card says so
  when no scope is set.
- `FaceTile` had to learn that `dist` can be null — a face whose every reference
  falls inside its own burst has no measurable distance, and `toFixed()` on it
  took out the whole grid. Unmeasured faces render `—` and **sort last**:
  "unknown" is not "far", the same rule the gate follows.

Three things it does that a SQL `UPDATE ... SET person_id=NULL` does not:

- **Writes `rejected`, not NULL.** A NULL is indistinguishable from
  never-matched, so the next `match-faces` sweep re-applies the same wrong
  labels — precisely what happened on 2026-09-14.
- **`--apply` requires `--snapshot`**, and `restore-unmatch` re-applies *only*
  those pinned ids, skipping any face hand-labelled since. Deliberately narrower
  than `restore-unmatched-faces`, which restores every unmatch ever recorded.
- **`--min-dist` keeps what the bad source got right.** temporal is ~4%
  accurate, not 0%. On Calvin a 1.05 gate clears 6,187 and keeps 144 — the gate
  costs almost nothing here because the populations are so separated, but on a
  person with a better temporal record it is the difference between a cleanup
  and a data loss. A face with **no usable reference is never cleared by a
  gate**: "unknown" is not "far".

<!-- moved verbatim from CLAUDE.md (line 3231) on 2026-10-19 -->
## Per-shoot "best of" curation — `scripts/rank_shoot.py`

Reusable ranking for one match/shoot, producing `Best N` / `Next M` / `All N+M`
collections. The Aug 29 2026 version of this was ad-hoc and lost, so it had to
be reverse-engineered from the collection descriptions; it is a script now.

```bash
# measure (slow: one full-res decode per photo; cached + resumable)
python scripts/rank_shoot.py --date 2026-09-12 --measure
# select, preview, then create
python scripts/rank_shoot.py --date 2026-09-12 [--max-per-person 6]
python scripts/rank_shoot.py --date 2026-09-12 --max-per-person 6 --apply
```

**The measurement pass lives in `photosearch/rank_measure.py`, not in the
script** — the script imports `measure` from there, so there is one
implementation and one cache format. It had to move because **`scripts/` is
not copied into the Docker image** (the Dockerfile copies `photosearch/`,
`frontend/`, `cli.py`, `requirements.txt`, `docker-entrypoint.sh` and nothing
else); it only ever ran on the NAS because the repo is bind-mounted there.
**`Advance batch` on `/batches` now runs it** for the batch's folder as the
`rank_measure` NAS step, so the usual path is: advance the batch, then run the
selection command above with no `--measure`.

**Cache location: beside the DB** (`rank_measure.default_cache_path` —
`dirname(PHOTOSEARCH_DB)/rank_shoot_<date>.json`). On the NAS the DB is
`/data/photo_index.db`, so that is byte-identical to the `/data/rank_shoot_
<date>.json` the script used to hardcode and existing caches keep working —
while off-NAS it no longer points at a `/data` that does not exist. `--cache`
still overrides. The cache is keyed by **date** even when the runner scopes by
folder, because the selection phase reads it by date; a batch folder with no
`YYYY-MM-DD` prefix (`_undated/...`) is skipped with a message rather than
given an invented date. Two batches sharing a date **merge** into that one
cache — the second run adds to the first's entries, never clobbers them.

**Two durability rules, because it now runs unattended** on a box that has
been I/O-wedged and OOM-killed:

- **`should_abort` is checked per PHOTO**, not per step. `advance_nas_steps`
  only checks between steps and this is the last and longest one, so a Cancel
  would otherwise do nothing for ten minutes. The cache is saved *before* the
  `InterruptedError`, so the cancelled work is not lost (the pass is
  resumable, and `batch_advance` deletes the job row on the way out).
- **The cache is written atomically** — temp file in the same directory,
  flush, `fsync`, `os.replace`. A kill mid-`json.dump` used to leave truncated
  JSON that made every later run for that date raise, including a manual
  `--measure`, with no self-healing. An unparseable cache is moved to
  `<name>.corrupt-<ts>` and logged rather than silently discarded: it can hold
  hours of N100 decode time.

**Native-resolution face-crop Laplacian is the primary signal**, because within
one shoot every other ranker is inert or wrong: CLIP barely moves between a
keeper and a dud (same kids, same pitch); MCP `rerank_photos` silently passes
through on the NAS (no visual model configured) and local qwen2.5-vl
mode-collapses on Likert scoring. What separates frames in a sports burst is
whether the face is in focus, and that must be measured on the **original**
pixels — a preview or a cached 200px crop has already discarded the signal.
919 photos ≈ 6 min on the N100.

Things that will bite if changed carelessly:

- **A partial VLM-aesthetics pass is worse than none.** It covers whichever
  photos the fleet claimed first, so ranking on it favours that arbitrary
  subset. The script refuses `aes_overall` below `--min-aes-coverage` (0.80)
  and falls back to the LAION score.
- **Bursts come from timestamps, not `photo_stacks`** — a fresh shoot has no
  stacks, and scoped stacking has twice wiped the library's. `--burst-gap`
  defaults to **1s** because the data forces it: on 2026-09-12 a 2s gap leaves
  only 200 bursts, and "no two frames from one burst" then makes 250 photos
  impossible (1s→341, 2s→200, 3s→156, 5s→105).
- Signals are **percentile-ranked before weighting** — Laplacian variance,
  pixel edge and det_score share no scale.
- `--min-per-person` / `--max-per-person` are constraints applied **after**
  ranking, not score terms. Uncapped on 2026-09-12 Calvin took 16 of 50.
- The multi-player term is a lightly-weighted **proxy** for action, not a
  detector. There is no real action signal without the LLM passes.

<!-- moved verbatim from CLAUDE.md (line 3355) on 2026-10-19 -->
## Splitting attractor clusters (M18)

Large muddled clusters (observed on the NAS: one cluster with 126 faces
across 469 days) pull in multiple distinct people during the eps=0.55
global recluster. Splitting them is a per-cluster DBSCAN re-run with
tighter params:

```bash
$DC run --rm photosearch split-cluster 15 --dry-run       # preview
$DC run --rm photosearch split-cluster 15 --eps 0.45      # apply
```

Also available from the /faces detail sidebar: "Split" button on any
unknown cluster, preview shows the histogram of resulting sub-cluster
sizes before you apply. API: `POST /api/faces/clusters/{id}/split` with
body `{eps, min_samples, dry_run}`. New cluster_ids are minted past the
current max so they never collide. Faces that fall out as DBSCAN noise
go to `cluster_id=NULL`. Implementation: `photosearch/faces.py:split_cluster`.

<!-- moved verbatim from CLAUDE.md (line 3374) on 2026-10-19 -->
## Face merge suggestions (M18)

Two-step flow for merging unknown-face groups into named persons or into
other unknown clusters:

**1. Generate suggestions** (CLI, read-only):

```bash
$DC run --rm photosearch suggest-face-merges \
  --json-out /data/suggestions.json \
  [--base-url http://<nas>:8000] \
  [--centroid-cutoff 0.95] [--min-pair-cutoff 0.60] \
  [--max-members 60] [--min-group-size 1] \
  [--verify-pair 'cluster:X=person:Name' ...]
```

For every candidate pair, `photosearch/face_merge.py` computes two L2
metrics on the ArcFace 512-dim encodings: **centroid_dist** (between the
two groups' normalized mean vectors) and **min_pair_dist** (minimum over
all member-to-member pairs). Both must be below their cutoffs for a
suggestion. `--verify-pair` reports TP recall / FP avoidance against
known-ground-truth pairs (`=` for should-merge, `!=` for should-not).

**2. Review and act** on the `/merges` page (reads the JSON, applies via
`POST /api/faces/merges`):
- Each card shows both rep-face crops, labels, face counts, date ranges,
  and the three scores (min_pair, centroid, overlap).
- Accept → `POST /api/faces/merges` performs the merge transactionally.
  cluster→person sets person_id + clears cluster_id + stamps
  `match_source='merge_review'`. cluster→cluster updates cluster_id.
- Dismiss → stored in localStorage (client-side only for now; persistent
  rejection table deferred).
- Top nav link is global. Suggestions whose source cluster has been
  accepted or emptied are filtered out live in the API response.

Cluster IDs are ephemeral (every `recluster-faces` run renumbers them),
but the merge flow is the intended consumer — review within the same
recluster cycle. Face IDs (`rep_face_id`) are stable, so localStorage
dismissals survive JSON regeneration.

The /merges toolbar also has a **Regenerate** button (`POST
/api/faces/suggestions/regenerate`) that re-runs the engine with
user-selected cutoffs and writes fresh JSON — useful after splitting
attractor clusters or after a batch of accepts changes the grouping
landscape. Blocks ~15–30s on an N100.

`/api/faces/groups` is paginated (`limit`/`offset`/`filter`/`total`/`counts`),
hides singletons by default (`?include_singletons=1` to restore), and
auto-downgrades its O(N²) similarity sort to count-sort above 500 groups.
`/api/faces/crop/{id}` disk-caches at `thumbnails/face_crops/{id}_{size}.jpg`.

Remaining work in `docs/plans/faces-clustering-and-perf.md`: persist
`det_score` + bbox area and filter low-quality faces before clustering,
swap to HDBSCAN for varying density, and materialize a `face_groups` table.

<!-- moved verbatim from CLAUDE.md (line 3498) on 2026-10-19 -->
### Desktop-as-client face recompute (heavy clustering off the NAS)

The N100 can't do the global DBSCAN at scale (~230k unmatched faces after the
quality filter), but the desktop can. Compute on the desktop off the synced
read-replica, push only the assignments to the NAS:

**`export-face-state --out FILE`** dumps `(face_id, cluster_id, person_id,
match_source)` for every face into a small (~9MB) SQLite file.
**`apply-face-state --db <nas> --from FILE [--overwrite-persons] [--apply]`**
ATTACHes it and applies set-based: person matches **additive** by default (only
fill faces unmatched on the target — preserves curation, and **skips
`dedupe_unmatched` faces** so a re-run of temporal matching can't undo the
dedup); clusters applied to still-unmatched faces; faces absent from the file
left untouched. (Distinct from the JSON `export-face-assignments`, which is
manual-label rebuild-preservation.)

Pipeline (`/tmp/face_recompute_pipeline.sh` is the reference):
```
sync replica → (on desktop) match-faces --temporal + recluster-faces
→ export-face-state → cat-stream to NAS → (NAS) git pull
→ apply-face-state --apply → resolve-duplicate-persons --apply
```
Ship the file with `cat FILE | ssh nas 'cat > /volume1/docker/photosearch/FILE'`
— **UGREEN blocks scp/SFTP/rsync to the box** (same reason `sync-replica.sh`
cat-streams). One real run (2026-06-21): ~230k-face DBSCAN took ~61 min on the
desktop, then apply matched 16k + clustered 122k faces, resolve cleaned 2k dups.
`recluster-faces` writes by default (has `--dry-run`, no `--apply`);
`match-faces` writes directly. DBSCAN at 512-dim scales ~O(n²) (curse of
dimensionality) — ~230k faces takes tens of minutes even on a fast desktop.

<!-- moved verbatim from SKILL.md (line 1020) on 2026-10-19 -->
## Per-shoot curation and label checking (CLI)

```bash
# Which labelled faces look more like someone ELSE? No eps, no threshold.
photosearch verify-face-labels --date-from 2026-09-12 --date-to 2026-09-12 \
    [--decisive-only] [--person NAME] [--min-references 3]

# A whole match_source is wrong for one person (Calvin: 6,331 temporal faces,
# p50 1.248 from his manual refs vs strict's 1.053). Bulk, reversible, and
# marked 'rejected' so match-faces cannot re-apply it. RUN ON THE NAS.
photosearch unmatch-person --person Calvin                       # preview
photosearch unmatch-person --person Calvin --min-dist 1.05 \
    --snapshot /data/calvin-temporal.json --apply
photosearch restore-unmatch --from /data/calvin-temporal.json --apply

# ...and which of those have the right face sitting in the SAME photo?
# Rare, near-certain, and fixable by a swap. 4 findings across two ~950-photo
# shoots, all true.
photosearch verify-face-labels --date-from 2026-09-12 --date-to 2026-09-12 \
    --rivals-only

# "Best of" a match: Best 50 / Next 200 / All 250 collections.
python scripts/rank_shoot.py --date 2026-09-12 --measure        # slow, cached
python scripts/rank_shoot.py --date 2026-09-12 --max-per-person 6
python scripts/rank_shoot.py --date 2026-09-12 --max-per-person 6 --apply
```

`rank_shoot.py` ranks on **native-resolution face-crop Laplacian** because
within one shoot CLIP is inert (same kids, same pitch), MCP `rerank_photos`
passes through on the NAS, and a partial VLM-aesthetics pass is worse than none
(it covers whichever photos the fleet claimed first). Bursts come from
timestamps, not `photo_stacks` — a fresh shoot has none, and scoped stacking has
twice wiped the library's. Full rationale in the script docstring and CLAUDE.md.

<!-- moved verbatim from SKILL.md (line 1503) on 2026-10-19 -->
### Face management commands (post-indexing)

```bash
add-persons --config <yaml>                 # Batch register from YAML
match-faces [--tolerance 1.15] [--temporal] # Match faces to persons
  --collection ID                           # Scope to collection photos only
  --expand-stacks                           # Include stack members with --collection
  --temporal-tolerance 1.45                 # Looser threshold for temporal
  --temporal-window 30                      # Session context window (minutes)
recluster-faces [--eps 0.55] [--min-samples 3] \
                [--no-session-stacking] \
                [--session-eps 0.50] [--session-window 60] \
                [--dry-run]
                                            # Global DBSCAN over all person_id IS NULL
                                            # encodings, then (default-on) session-
                                            # stacking second pass that unions DBSCAN
                                            # noise pairs within (session_eps L2 +
                                            # session_window minutes). Renumbers every
                                            # unknown cluster and clears
                                            # ignored_clusters atomically.
split-cluster CLUSTER_ID [--eps 0.45] [--min-samples 2] [--dry-run]
                                            # Re-run DBSCAN on one cluster with
                                            # tighter params to split "attractor"
                                            # clusters. New cluster_ids minted
                                            # past the current max; noise → NULL.
                                            # Also available from the /faces
                                            # detail sidebar's Split button.
suggest-face-merges [--centroid-cutoff 0.95] [--min-pair-cutoff 0.60] \
                    [--max-members 60] [--min-group-size 1] \
                    [--include-ignored] [--limit N | --all] \
                    [--json-out FILE] \
                    [--verify-pair 'cluster:X=person:Name' ...]
                                            # Read-only dry-run. Surfaces likely merges
                                            # between every pair of groups (named +
                                            # unknown). Uses centroid + min-pair L2
                                            # distances. --verify-pair reports
                                            # TP/FP coverage against known examples.
list-persons                                # Show persons and counts
face-clusters                               # Show unidentified clusters
correct-face <filename> <face_num> <name>   # Manual correction
clear-matches <dir> [--person] [--all-faces]
export-face-assignments / import-face-assignments  # JSON of MANUAL labels (rebuild preservation)

# --- over-matching cleanup / data integrity (see "Face over-matching" below) ---
dedupe-person-faces --person NAME --references <photo-ids> \
                    [--min-gap 0.15] [--report PATH] [--apply]
                                            # Reference-based, source-gated dedup of
                                            # photos with 2+ faces of one person.
resolve-duplicate-persons [--apply]         # Universal: keep one face per (photo,person)
                                            # by priority manual>strict>temporal.
verify-person-matches [--person NAME] [--min-dist 1.30] \
                      [--min-refs 5] [--top-n 40] [--report PATH]
                                            # Find MIS-tags: rank each person's faces by
                                            # distance to their trusted (strict/manual)
                                            # refs; farthest = likely wrong person.
                                            # Report-only; catches non-sibling mis-tags.
restore-unmatched-faces [--apply]           # Undo the two above (face_dedupe_undo table);
                                            # also clears the restored pairing's exclusion.
backfill-face-exclusions [--apply]          # Seed face_person_exclusions from the existing
                                            # face_dedupe_undo snapshots. Idempotent; the
                                            # v31 migration runs it automatically.
cleanup-orphan-faces [--apply]              # Delete faces whose photo was deleted.
backfill-image-orientation [--apply] [--limit N]
                                            # Store EXIF-oriented image_width/height
                                            # (run where the photo files live).
export-face-state --out FILE / apply-face-state --from FILE [--overwrite-persons] [--apply]
                                            # Desktop-compute → NAS-apply bridge for
                                            # recluster/match (additive person-fill).
```

<!-- moved verbatim from SKILL.md (line 1573) on 2026-10-19 -->
### Unknown-face clustering & merge suggestions (M18)

Four cooperating tools improve unknown-face grouping:

1. **Session stacking** in `recluster-faces` — after global DBSCAN, a second
   pass runs union-find over the noise points: pairs whose L2 distance is
   within `--session-eps` (default 0.50) and whose `date_taken` is within
   `--session-window` minutes (default 60) get linked. Components of size ≥
   2 become new clusters continuing past the DBSCAN id range. Recovers
   same-person-same-event groups that `min_samples=3` had thrown away.
   Pass `--no-session-stacking` to restore DBSCAN-only behavior.

2. **`suggest-face-merges`** — read-only CLI that finds likely merges
   between any two face groups (cluster↔cluster and cluster↔named). For
   each candidate pair it computes `centroid_dist` (between the two groups'
   normalized mean encodings) and `min_pair_dist` (min across all
   member-to-member pairs). Suggests when both are under their cutoffs.
   `--verify-pair` takes known TP (`=`) and FP (`!=`) examples and prints
   whether each would be caught — use it to tune thresholds against the
   real library. `--base-url` emits clickable links next to each suggestion.

3. **`/merges` review page** (see the "/merges page architecture" section
   above) — renders the JSON output from suggest-face-merges and lets the
   user accept/dismiss with keyboard shortcuts. Can regenerate the JSON
   directly from the UI via `POST /api/faces/suggestions/regenerate`.

4. **`split-cluster`** (CLI + API + /faces button) — re-runs DBSCAN on a
   single cluster with tighter eps to break apart "attractor" clusters that
   lumped multiple people together during the eps=0.55 global recluster.
   New cluster_ids are minted past the current max so they never collide.
   Dry-run previews show the histogram of resulting sub-cluster sizes.

Implementation: `photosearch/face_merge.py` (`load_groups`,
`compute_suggestions`, `score_pair`) for suggestions;
`photosearch/faces.py:split_cluster` for splitting. Suggestion per-pair cost
is O(K²) where K is `--max-members` (default 60, biggest-bbox faces sampled
first). named↔named pairs are never suggested.

<!-- moved verbatim from SKILL.md (line 1611) on 2026-10-19 -->
### Merge review page (`/merges`) — summary

After generating suggestions (CLI `--json-out /data/suggestions.json` or the
in-UI Regenerate button), the `/merges` page renders each suggestion
side-by-side with face crops, labels, face counts, and scores. Full
architecture is documented in the "Shared Frontend Components" section
above. Key actions:

- **Accept** → `POST /api/faces/merges` with `{source, target}`. For
  cluster→person: updates `faces.person_id`, clears `cluster_id`, stamps
  `match_source='merge_review'`. For cluster→cluster: updates
  `faces.cluster_id` only. Chain-rewrites downstream suggestions so
  `C → A` becomes `C → B` when `A → B` was just accepted.
- **Dismiss** → localStorage-only (keyed by the sorted `rep_face_id` pair,
  which is stable across reclusters). Persistent rejection table is
  future work.
- **Regenerate** → `POST /api/faces/suggestions/regenerate` with
  cutoff params; blocks ~15–30s on an N100 while the engine reloads groups
  and rescores.
- Each card links to `/faces?cluster_id=N` / `?person_id=N` for deep
  verification before accepting.

API:
- `GET /api/faces/suggestions` — reads `PHOTOSEARCH_SUGGESTIONS_JSON`
  (default `/data/suggestions.json`). Filters out rows whose source
  cluster has been collapsed since the JSON was written.
- `POST /api/faces/merges` — body `{"source": {type, id}, "target":
  {type, id}}`. Source must be `cluster`; target can be `person` or
  `cluster`. Returns `{ok, moved_face_count, source, target}`.

Suggestions file is on the Docker volume at `/data/suggestions.json` by
default — re-run the CLI whenever you want fresh candidates.

<!-- moved verbatim from SKILL.md (line 1644) on 2026-10-19 -->
### Vec0 orphan cleanup

`clip_embeddings` and `face_encodings` are sqlite-vec `vec0` virtual tables. SQLite virtual
tables cannot participate in foreign-key constraints, so `ON DELETE CASCADE` on the parent
`photos` / `faces` tables does not reach them — every historical photo or face deletion
left a dangling vector row behind. Symptoms: status page shows `>100% embedded`, and CLIP
worker submits occasionally fail with `UNIQUE constraint failed on clip_embeddings primary key`
(since vec0 does not honor `INSERT OR REPLACE` — the PK conflict is raised first).

`AUTOINCREMENT` on `photos.id` and `faces.id` guarantees the orphan IDs can never be reissued
to different rows, so deletion is always safe.

```bash
photosearch cleanup-orphans [--dry-run]
```

Also: `add_clip_embedding` now uses explicit `DELETE` + `INSERT` instead of `INSERT OR REPLACE`,
making re-submits idempotent when a worker's claim TTL expires and the photo is re-claimed by
another worker before the original submit lands.

<!-- moved verbatim from SKILL.md (line 1664) on 2026-10-19 -->
### Face over-matching cleanup, dedup & desktop recompute

Full narrative + rationale live in **CLAUDE.md** ("Face over-matching cleanup",
"Desktop-as-client face recompute", detection gotchas). Summary of the tooling:

- **Over-matching** = the same person tagged on 2+ faces in one photo (temporal
  matching propagating a label to another kid at an event). `dedupe-person-faces`
  is reference-based + **source-gated**: 1 trusted (strict/manual) face → keep it,
  drop temporal extras; all-temporal → reference-distance gap (`--min-gap`); 2+
  trusted → leave. Needs a curated `--references` photo-id set (one clear face
  each, spanning years). `--report` writes a keep-vs-remove **face-crop** gallery.
- **`resolve-duplicate-persons`** enforces the one-person-per-photo invariant for
  everyone (priority manual > strict > temporal, tie-break det_score → bbox area).
- **Both go through `db.unmatch_faces_as_duplicates`** (snapshot + record the
  (face, person) exclusion + null the label) — one primitive, because each used
  to carry its own copy and a fix to one left the other churning. The exclusion
  is what stops the matcher re-applying the label the resolver just stripped;
  see CLAUDE.md "FIXED (2026-09-19) — the nightly match/unmatch churn loop".
- **`restore-unmatched-faces`** reverses both via the on-demand `face_dedupe_undo`
  snapshot (unmatch sets `match_source='dedupe_unmatched'`; `apply-face-state`
  additive-fill skips those, and also refuses pairings excluded on the target,
  so neither a re-match nor a replica push can undo the cleanup).
- **`cleanup-orphan-faces`** deletes face rows whose photo was deleted (broken
  `/faces` thumbnails + phantom matches; ~12% of faces on the NAS were orphans).
- **Desktop-as-client recompute**: the N100 can't do the ~230k-face DBSCAN, so
  run `match-faces --temporal` + `recluster-faces` on the synced replica, then
  `export-face-state` → cat-stream the ~9MB file to the NAS (UGREEN blocks
  scp/rsync) → `apply-face-state --apply` → `resolve-duplicate-persons`.
- **Dead end (don't re-explore):** swapping the embedder or using a VLM to fix
  the hard ambiguous cases — ArcFace / AdaFace / MagFace / QMagFace / qwen2.5-vl
  all fail the same controlled test (`evals/adaface_compare.py`,
  `magface_compare.py`, `vlm_face_compare.py`); it's identity ambiguity between
  similar-looking kids, not model quality. Keep ArcFace; within-photo relative
  comparison is the only robust signal.
- **Babies** are detected + clustered but don't auto-match (ArcFace can't bridge
  baby→child); recover by reclustering + manually assigning the baby clusters
  (use the searchable `/faces` filter by date to find them).

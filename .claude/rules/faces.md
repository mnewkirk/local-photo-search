---
paths:
  - "photosearch/face*.py"
  - "photosearch/bulk_unmatch.py"
  - "photosearch/rank_measure.py"
  - "scripts/rank_shoot.py"
  - "frontend/dist/faces.html"
  - "frontend/dist/merges.html"
---
# Faces — traps

Full history and measurements: docs/notes/faces.md (cite by heading). Runbooks: .claude/skills/photo-search/reference/face-management-cli.md, faces-review.md.

- **Face over-matching cleanup & the one-person-per-photo invariant** — a person appears in a photo at most once. Every duplicate unmatch goes through `db.unmatch_faces_as_duplicates` (snapshot to `face_dedupe_undo` + exclusion + null, `match_source='dedupe_unmatched'`); never write a copy of it (`test_the_duplicate_unmatch_write_lives_in_one_place`). `resolve-duplicate-persons` keeps manual > strict > temporal, tie-break det_score then bbox area.
- **`match_source='rejected'` — a human "no" that survives auto-matching** — a human clear writes `faces.REJECTED_MATCH_SOURCE`, never NULL (NULL reads as never-matched and the next sweep re-applies it). Both matchers filter through the shared `faces.MATCHABLE_SQL`; never give one matcher its own `person_id IS NULL` (`test_both_matchers_use_the_shared_predicate`).
- **Deliberately distinct from `dedupe_unmatched`** — `dedupe_unmatched` stays matchable: the loser may be a different person. Never add it to the matcher filter.
- **Replica gotcha: `web._mirror_face_labels`** duplicates the NAS write's semantics; any change to how a clear/assign is stamped must change it too, or the replica ships the wrong marker back up.
- **`face_person_exclusions`** (docs/notes/maintenance.md, "Scheduling + replica-mode maintenance") — remembers the barred (face, person) pairing; load once via `faces.load_match_exclusions`, never per face. Strict falls through to the next in-tolerance person; temporal does NOT (its `TEMPORAL_MIN_GAP` judges the whole field). Never unify them (`test_temporal_matcher_does_NOT_fall_through_to_the_next_best_person`). `db.assign_face_to_person` clears the exclusion unless the source is in `AUTO_MATCH_SOURCES`; any new label writer must use it or clear explicitly.
- **Face clustering** — new faces land with `cluster_id = NULL`; never re-add per-batch clustering to `index_directory()`, `_index_collection()` or `worker_api.submit_results()` (each batch restarts ids at 0 and collides). Only `recluster-faces` mints clusters, and it renumbers all ids and clears `ignored_clusters`.
- **Cluster ids are ephemeral, face ids are stable** — key anything persistent (dismissals, rep faces) on face ids, never cluster ids.
- **Quality pre-filter (schema v20)** — recluster and `split_cluster` drop faces below `CLUSTER_MIN_DET_SCORE` (0.65) / `CLUSTER_MIN_BBOX_EDGE` (60 px) before DBSCAN; filtered faces go to `cluster_id=NULL`. `det_score IS NULL` is grandfathered and always passes — keep the `IS NULL OR` clause.
- **Face-label integrity: is this face actually that person?** (`face_verify.verify_labels`) — exclude the face's whole burst (`BURST_WINDOW_SECONDS`), not just the face: a burst sibling with the same mislabel vouches for it. References are manual + strict only, never temporal. `separation` uses `SEPARATION_PERCENTILE` (5), never `min()` — one bad label zeroes a min and silently disables the pair. No VLM in the decision path.
- **The rival test — the right face is usually in the same photo** (`face_verify._photo_rivals`) — all four gates required; `radius(P)` is the p90 (`RADIUS_PERCENTILE`) of P's genuine faces, leave-one-burst-out. Gate 1 is a one-to-one `linear_sum_assignment` with costs `d - radius(P)`, never a per-label argmin (two labels would claim one rival). Swap in faces.html is clear-then-assign via two `bulk-assign` calls; never assign first, never a new endpoint.
- **The panel asks about TRUSTED labels only** — default `face_verify.TRUSTED_LABEL_SOURCES`. Apply the source filter LAST, after calibration and per-photo assignment, and report hidden rows in `stats.hidden_by_source` / `hidden_by_person` (`test_hiding_a_source_does_not_change_the_numbers`).
- **Bulk-undoing one person's bad labels — `unmatch-person`** (`bulk_unmatch.py`) — writes `rejected`, not NULL; `--apply` requires `--snapshot`; `restore` re-applies only the pinned ids. A face with no usable reference is never cleared by a distance gate. Run the CLI on the NAS.
- **Hand labels do not teach the strict matcher anything** — `match_faces_to_persons` reads only `face_references`. Never feed hand labels to its global tolerance; `face_suggest` ranks and never writes.
- **Temporal matches are ~4% accurate on event shoots** — never use `temporal` as references, ground truth or attribution; filter to manual/strict/merge_review when identity matters.
- **Face-encoding cache** (`PhotoDB.get_face_encodings_cached`, `db._FaceEncodingCache`) needs no invalidation only because encodings are INSERT/DELETE-only and `faces.id` is AUTOINCREMENT. If you ever update an encoding in place, evict that face.
- **Reviewing never-clustered faces (the "Unclustered" bucket)** — `/api/faces/groups` emits `type: "unclustered"` only with a content filter; `/api/faces/group/unclustered/0/photos` must 400 without one. The bucket gets no Rename/Merge/Split/Ignore and no multi-select: it holds many people.
- **EXIF-oriented image dimensions** — `image_width`/`image_height` are swapped for orientation 5-8 to match the oriented image the bboxes come from; never compare bboxes to raw sensor dims.
- **Embedder/VLM swap is a tested dead-end — don't re-explore.** ArcFace, AdaFace, MagFace, QMagFace and qwen2.5-vl all fail the lookalike-kids set; the within-photo relative comparison is the only robust signal.
- **Video faces take a margin in the strict matcher** (`VIDEO_MATCH_MARGIN` 0.15 to the nearest other person, `pick_person_with_margin`); stills keep the plain nearest-within-1.15 path byte-for-byte. Detail: `.claude/rules/video.md`.

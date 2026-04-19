# Faces clustering & `/faces` page performance

Plan for fixing two linked problems observed on the NAS:

1. **"Unknown #0" contains 1,736 unrelated faces.**
2. **`/faces` page load is extremely slow.**

## Root cause diagnosis

### Problem 1 — Unknown #0 bloat

`cluster_encodings()` at `photosearch/faces.py:334-374` is a **greedy per-batch**
clusterer that always starts cluster IDs at 0. It is called:

- `photosearch/worker_api.py:270-279` — every worker-submit batch (hot path)
- `photosearch/index.py:342-353` — CLI single-pass indexing
- `photosearch/index.py:997-1008` — CLI directory indexing

Every batch independently writes `cluster_id = 0` for its first face.
`/api/faces/groups` (`photosearch/web.py:516-541`) just `GROUP BY cluster_id`,
so **"Unknown #0" is the union of the first face of every batch ever run**, not
a real cluster. Other Unknown #N sizes are similarly per-batch artifacts.

Secondary issues:

- `faces.py:220-224` computes `det_score` but `add_face()` in `db.py:590` drops
  it — clustering cannot filter low-quality faces.
- `CLUSTER_TOLERANCE = 0.75` (`faces.py:45`) is tight for normalized ArcFace
  (same-person L2 is often 0.6–1.1). Low-quality faces fragment into singletons.
- Embeddings are L2-normalized (`faces.py:142-147`), so the math is fine — the
  problem is the batch-local ID space and the greedy algorithm.

### Problem 2 — slow `/faces`

- `/api/faces/groups` (`web.py:470-556`) always runs an O(N²) similarity sort:
  fetches encodings via `get_face_encodings_bulk` (`db.py:649-668`), builds a
  full pairwise distance matrix, greedy-chains nearest neighbours
  (`_similarity_sort`, `web.py:393-467`).
- `/api/faces/crop/{face_id}` (`web.py:324-390`) opens the **original full-res
  RAW/JPEG** on every request, runs `exif_transpose`, crops, resizes, JPEG-
  encodes in-memory, and streams. **No disk cache** — only
  `Cache-Control: max-age=3600`.
- UI (`faces.html:1104-1136`) renders every group's `<img>` on first paint.

## Ordered action items

### Shipped

1. ✅ **Disk-cache face crops** at `thumbnails/face_crops/{face_id}_{size}.jpg`,
   atomic write via `os.replace`, served via `FileResponse` (commit 8639799).
2. ✅ **Hide singletons** in `/api/faces/groups` by default
   (`HAVING face_count >= 2`); `?include_singletons=1` restores them (8639799).
3. ✅ **Similarity sort auto-downgrade** to count-sort when group count
   exceeds 500; response includes `sort` field showing what was applied (8639799).
4. ✅ **CSS `content-visibility: auto`** on `.face-card` with
   `contain-intrinsic-size` (8639799).
5. ✅ **Batch-local ID collisions removed.** Faces now land with
   `cluster_id = NULL` at every ingest path — `index_directory()`,
   `_index_collection()`, `worker_api.submit_results()` (commit fd3c206).
6. ✅ **`recluster-faces` CLI** — global `sklearn.cluster.DBSCAN(eps=0.55,
   min_samples=3, metric='euclidean', algorithm='ball_tree')` over every
   `person_id IS NULL` encoding. `ignored_clusters` cleared in the same
   transaction. Loader uses `np.frombuffer` over concatenated BLOBs for
   ~50–100× speedup vs. struct.unpack per row (fd3c206, 4dbf15e).
7. ✅ **Paginate `/api/faces/groups`** with `limit` / `offset` + `total` +
   pre-pagination `counts`. Frontend got a "Load more" button and reads
   merge targets from `/api/persons` so persons on unloaded pages still
   count (commit b12f539, 4b9532c).

### Iterate

8. ✅ **Session-stacking second pass** in `recluster-faces` (M18). After
   DBSCAN, a union-find pass over the noise points links pairs within
   `session_eps` L2 AND `session_window` minutes. Components of size ≥ 2
   become new clusters continuing past the DBSCAN id range. Recovers
   same-person-same-event groups that `min_samples=3` had discarded.
   Defaults: `session_eps=0.50`, `session_window=60` minutes. Pass
   `--no-session-stacking` to disable. See `photosearch/faces.py` —
   `_session_stack_noise()` and the extended `recluster_unknown_faces()`.
9. ✅ **Merge-suggestion engine** (M18, dry-run CLI). `suggest-face-merges`
   finds likely merges between any two groups (cluster↔cluster,
   cluster↔named). Centroid + min-pair L2 metrics; `--verify-pair`
   harness reports TP/FP coverage for threshold tuning. No DB writes yet —
   accept/reject UI is deferred. See `photosearch/face_merge.py`.
10. **Accept/reject UI** for merge suggestions. Adds a
    `face_merge_suggestions` table (schema v13), `/api/faces/suggestions`
    endpoints, and a suggestions section on `/faces` with per-pair
    accept/reject. Blocked on tuning the CLI cutoffs against the real NAS
    library first.
11. **Quality filter before clustering.** Persist `det_score` + bbox area on
    `faces`; filter `det_score >= 0.65` and min bbox edge `>= 60 px` before
    clustering. Keep raw rows for person-matching (`MATCH_TOLERANCE=1.15` is
    forgiving).
12. **HDBSCAN** to handle varying density (some people appear 100×, others 2×).
13. **Materialized `face_groups` table** refreshed after reclustering, so
    `/api/faces/groups` becomes a single indexed SELECT.

## Tradeoffs / risks

- Reclustering renumbers every unknown cluster; `ignored_clusters` must be
  migrated or cleared atomically.
- Stricter quality filter could drop genuine faces from crowded group photos
  — keep raw rows, just exclude from clustering.
- Disk-cached crops grow `thumbnails/` dir; reuse existing cleanup if any.
- Default sort changing from similarity → count changes visual order; expose
  a UI toggle.

## Critical files

- `photosearch/faces.py` — clustering function + tolerance constants
- `photosearch/worker_api.py` — worker submit handler
- `photosearch/index.py` — CLI indexing
- `photosearch/web.py` — `/api/faces/*` endpoints
- `photosearch/db.py` — faces schema, `ignored_clusters`, bulk encoding fetch
- `frontend/dist/faces.html` — face grid UI

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

### Ship now (1–2 hr, low risk)

1. **Disk-cache face crops** at `thumbnails/face_crops/{face_id}_{size}.jpg`
   and serve via `FileResponse` if present. Bbox is immutable per `face_id`,
   no invalidation needed. Single highest-leverage perf fix.
2. **Hide singletons** in `/api/faces/groups`: default
   `HAVING face_count >= 2`; expose `?include_singletons=1`.
3. **Make similarity sort opt-in**. Default to `?sort=count`; skip the O(N²)
   sort automatically when group count exceeds a threshold (e.g. 500).
4. **CSS `content-visibility: auto`** on `.face-card` with
   `contain-intrinsic-size` so the browser defers off-screen layout/paint.

### Ship this week (design + test)

5. **Stop batch-local ID collisions.** Either write `cluster_id = NULL` at
   ingest and cluster on demand, or offset new IDs by
   `(SELECT COALESCE(MAX(cluster_id), -1) + 1 FROM faces)` before writing.
6. **One-shot `recluster-faces` CLI** — load every `person_id IS NULL`
   encoding, run `sklearn.cluster.DBSCAN(eps=0.55, min_samples=3,
   metric='euclidean')`, write IDs back. **Critical:** clear/migrate
   `ignored_clusters` in the same transaction (it keys on `cluster_id`,
   would otherwise silently mis-apply).
7. **Paginate** `/api/faces/groups` with `limit` / `offset` + total count.

### Iterate

8. **Quality filter before clustering.** Persist `det_score` + bbox area on
   `faces`; filter `det_score >= 0.65` and min bbox edge `>= 60 px` before
   clustering. Keep raw rows for person-matching (`MATCH_TOLERANCE=1.15` is
   forgiving).
9. **HDBSCAN** to handle varying density (some people appear 100×, others 2×).
10. **Materialized `face_groups` table** refreshed after reclustering, so
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

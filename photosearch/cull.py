"""Shoot review / culling — select the best representative photos from a folder.

Uses CLIP embeddings to cluster visually similar photos, then picks the
highest-quality photo from each cluster.  Target is ~10% of photos, but
if many clusters have high-quality distinct images we keep more.

Algorithm:
  1. Fetch all photos in the target directory with CLIP embeddings + scores.
  2. Build a cosine-distance matrix from CLIP embeddings.
  3. Agglomerative clustering with a distance threshold — photos that look
     alike end up in the same cluster.
  4. From each cluster, pick the photo with the best aesthetic score.
  5. Optionally expand picks: if a cluster's runner-up is also high quality
     and the cluster is large, include it too.

The result is stored in a ``review_selections`` table so the web UI can
render the grid with selected/unselected states.
"""

import logging
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from .db import PhotoDB, CLIP_DIMENSIONS, _deserialize_float_list


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _get_embeddings_for_photos(db: PhotoDB, photo_ids: list[int]) -> dict[int, np.ndarray]:
    """Fetch CLIP embeddings for a list of photo IDs.

    Returns {photo_id: embedding_vector} for photos that have embeddings.
    """
    result = {}
    for pid in photo_ids:
        row = db.conn.execute(
            "SELECT embedding FROM clip_embeddings WHERE photo_id = ?", (pid,)
        ).fetchone()
        if row:
            vec = _deserialize_float_list(row["embedding"], CLIP_DIMENSIONS)
            result[pid] = np.array(vec, dtype=np.float32)
    return result


def _cluster_photos(
    embeddings: dict[int, np.ndarray],
    distance_threshold: float = 0.0,
    target_clusters: int = 0,
) -> dict[int, int]:
    """Cluster photos by visual similarity using agglomerative clustering.

    Uses an adaptive threshold by default: instead of a fixed distance, it
    finds the threshold that produces approximately ``target_clusters``
    clusters (default: 2× the number of photos × target_pct, so ~20% of
    photos end up as cluster representatives).

    For same-day shoots where most photos are visually similar, a fixed
    threshold often produces too few clusters. The adaptive approach
    ensures diversity regardless of how similar the photos are overall.

    Args:
        embeddings: {photo_id: embedding_vector}
        distance_threshold: If > 0, use this fixed threshold instead of
            adaptive. Set to 0 (default) for adaptive behaviour.
        target_clusters: If > 0, aim for this many clusters. If 0,
            defaults to len(embeddings) * 0.20.

    Returns:
        {photo_id: cluster_label}
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    photo_ids = list(embeddings.keys())
    if len(photo_ids) <= 1:
        return {pid: 0 for pid in photo_ids}

    # Stack embeddings into a matrix
    matrix = np.stack([embeddings[pid] for pid in photo_ids])

    # Normalize (should already be, but ensure)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    # Pairwise cosine distance
    distances = pdist(matrix, metric="cosine")

    # Agglomerative clustering
    Z = linkage(distances, method="average", metric="cosine")

    logger.info("Clustering %d photos, distance_threshold=%.4f, target_clusters=%d",
                len(photo_ids), distance_threshold, target_clusters)
    logger.info("Distance stats: min=%.4f, max=%.4f, median=%.4f",
                float(distances.min()), float(distances.max()), float(np.median(distances)))

    if distance_threshold > 0:
        # Fixed threshold mode
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
        logger.info("Fixed threshold %.4f -> %d clusters", distance_threshold, len(set(labels)))
    else:
        # Adaptive threshold: binary search for the threshold that gives
        # approximately target_clusters clusters.
        if target_clusters <= 0:
            target_clusters = max(5, int(len(photo_ids) * 0.20))

        lo, hi = 0.01, float(distances.max())
        best_t = (lo + hi) / 2
        best_diff = float("inf")

        for _ in range(30):  # binary search iterations
            mid = (lo + hi) / 2
            trial_labels = fcluster(Z, t=mid, criterion="distance")
            n = len(set(trial_labels))
            diff = abs(n - target_clusters)
            if diff < best_diff:
                best_diff = diff
                best_t = mid
            if n > target_clusters:
                lo = mid  # threshold too tight, relax
            elif n < target_clusters:
                hi = mid  # threshold too loose, tighten
            else:
                break

        labels = fcluster(Z, t=best_t, criterion="distance")
        logger.info("Adaptive threshold: best_t=%.4f -> %d clusters (target was %d)",
                    best_t, len(set(labels)), target_clusters)

    return {pid: int(label) for pid, label in zip(photo_ids, labels)}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_best_photos(
    db: PhotoDB,
    directory: str,
    target_pct: float = 0.10,
    distance_threshold: float = 0.0,
    min_quality: float = 4.0,
    high_quality: float = 6.0,
) -> list[dict]:
    """Select the best representative photos from a directory.

    Args:
        db: Open PhotoDB instance.
        directory: Absolute path to the photo directory.
        target_pct: Target percentage of photos to select (~0.10 = 10%).
        distance_threshold: How similar photos must be to cluster together.
            Default 0.0 means adaptive (auto-determine from the data).
        min_quality: Minimum aesthetic score to be considered selectable.
        high_quality: Score threshold above which a photo is considered
            "high quality" — runner-ups in large clusters with scores above
            this are also included.

    Returns:
        List of dicts with photo info and selection metadata:
        [{"id", "filepath", "filename", "aesthetic_score", "cluster_id",
          "selected", "rank_in_cluster"}, ...]
    """
    directory = str(Path(directory).resolve())

    # Get all photos in the directory
    all_rows = db.conn.execute(
        "SELECT id, filepath, filename, aesthetic_score, date_taken, raw_filepath "
        "FROM photos"
    ).fetchall()

    # Filter to the target directory — match both relative and absolute paths
    dir_photos = []
    for row in all_rows:
        abs_path = db.resolve_filepath(row["filepath"])
        if abs_path and abs_path.startswith(directory + "/"):
            dir_photos.append(dict(row) | {"_abs_path": abs_path})

    if not dir_photos:
        return []

    photo_ids = [p["id"] for p in dir_photos]
    scores = {p["id"]: p["aesthetic_score"] or 0.0 for p in dir_photos}

    logger.info("Review: %d photos in directory, target_pct=%.2f, distance_threshold=%.4f",
                len(dir_photos), target_pct, distance_threshold)

    # Get CLIP embeddings
    embeddings = _get_embeddings_for_photos(db, photo_ids)
    logger.info("Retrieved %d embeddings out of %d photos", len(embeddings), len(photo_ids))
    if not embeddings:
        # Fallback: no embeddings, just pick top by quality
        dir_photos.sort(key=lambda p: scores.get(p["id"], 0), reverse=True)
        n_select = max(1, int(len(dir_photos) * target_pct))
        for i, p in enumerate(dir_photos):
            p["cluster_id"] = i
            p["selected"] = i < n_select
            p["rank_in_cluster"] = 1
        return dir_photos

    # Only cluster photos that have embeddings
    photos_with_emb = [p for p in dir_photos if p["id"] in embeddings]
    photos_without_emb = [p for p in dir_photos if p["id"] not in embeddings]

    # ---------------------------------------------------------------
    # Clustering strategy: "represent all, then trim"
    #
    # Instead of creating many clusters and competing for limited slots
    # (which always drops interesting content), we create a moderate
    # number of clusters (~1.5× target) and guarantee every cluster
    # gets at least one rep.  If that's over budget, we trim the least
    # valuable clusters.  This ensures no content type is invisible.
    # ---------------------------------------------------------------
    target_count = max(1, int(len(dir_photos) * target_pct))
    # Budget: allow up to 1.5× target for the full selection including
    # diversity picks.  The user prefers "a few extra distinct photos"
    # over a hard 10% cutoff.
    max_budget = max(target_count + 2, int(target_count * 1.5))

    # Aim for exactly max_budget clusters so every cluster gets a rep.
    # No trimming needed — the clustering itself is the selection.
    target_clusters = max(8, max_budget)

    clusters = _cluster_photos(
        embeddings,
        distance_threshold=distance_threshold,
        target_clusters=target_clusters,
    )

    # Assign cluster IDs
    for p in photos_with_emb:
        p["cluster_id"] = clusters.get(p["id"], -1)
    for p in photos_without_emb:
        p["cluster_id"] = -1

    # Group by cluster
    from collections import defaultdict
    cluster_groups = defaultdict(list)
    for p in photos_with_emb:
        cluster_groups[p["cluster_id"]].append(p)

    # Sort each cluster by aesthetic score (best first)
    for cid in cluster_groups:
        cluster_groups[cid].sort(
            key=lambda p: scores.get(p["id"], 0), reverse=True
        )

    n_clusters = len(cluster_groups)
    selected_ids = set()

    # Phase 1: One rep from every cluster.
    # Build a ranked list of all cluster reps so we can trim from the
    # bottom if there are more clusters than our budget.
    cluster_reps = []
    for cid, members in cluster_groups.items():
        best = members[0]
        quality = scores.get(best["id"], 0)

        # Trim score: clusters we'd drop first if over budget.
        # Singletons with low quality are the most expendable.
        # Large clusters and high-quality small clusters are kept.
        trim_score = quality + min(1.0, len(members) * 0.1)

        cluster_reps.append({
            "cid": cid,
            "photo": best,
            "score": quality,
            "trim_score": trim_score,
            "size": len(members),
        })

    # Sort by trim_score ascending — bottom of list gets trimmed first.
    cluster_reps.sort(key=lambda r: r["trim_score"])

    # If more clusters than budget, trim the least valuable ones.
    # Always keep clusters with 3+ members (they represent real content
    # themes), only trim singletons/pairs with low quality.
    if len(cluster_reps) > max_budget:
        n_trim = len(cluster_reps) - max_budget
        trimmed = 0
        keep = []
        for rep in cluster_reps:
            if trimmed < n_trim and rep["size"] <= 2 and rep["score"] < min_quality + 0.3:
                trimmed += 1  # drop this cluster
                continue
            keep.append(rep)
        # If we still have too many (couldn't trim enough small clusters),
        # trim from the bottom regardless of size
        if len(keep) > max_budget:
            keep = keep[len(keep) - max_budget:]
        cluster_reps = keep

    # Select one rep from each surviving cluster
    for rep in cluster_reps:
        if rep["score"] >= min_quality:
            selected_ids.add(rep["photo"]["id"])

    n_singles = sum(1 for r in cluster_reps if r["size"] <= 2)
    logger.info("Phase 1: %d clusters -> %d reps selected (%d singletons/pairs), "
                "budget=%d, %d clusters total",
                len(cluster_reps), len(selected_ids), n_singles, max_budget, n_clusters)

    # Phase 2: Within-cluster diversity using tags.
    #
    # Tags come from a fixed ~60-tag vocabulary, so they're reliable for
    # comparing content.  Strategy: find photos with tags that are rare
    # within their cluster — these represent genuinely different subjects
    # hiding in a visually-similar group.
    import json as _json
    tags_map = {}
    for p in photos_with_emb:
        row = db.conn.execute(
            "SELECT tags FROM photos WHERE id = ?", (p["id"],)
        ).fetchone()
        if row and row["tags"]:
            try:
                tags_map[p["id"]] = set(_json.loads(row["tags"]))
            except (ValueError, TypeError):
                pass

    phase2_added = 0

    if tags_map:
        diversity_candidates = []

        for cid, members in cluster_groups.items():
            if len(members) <= 3:
                continue  # small clusters don't need splitting

            from collections import Counter
            tag_counts = Counter()
            for p in members:
                for tag in tags_map.get(p["id"], set()):
                    tag_counts[tag] += 1

            cluster_size = len(members)
            rare_threshold = max(1, int(cluster_size * 0.2))

            for p in members:
                if p["id"] in selected_ids:
                    continue
                photo_tags = tags_map.get(p["id"], set())
                if not photo_tags:
                    continue
                rare_tags = [t for t in photo_tags if tag_counts[t] <= rare_threshold]
                if len(rare_tags) >= 1:
                    diversity_candidates.append((
                        len(rare_tags),
                        scores.get(p["id"], 0),
                        p,
                    ))

        # Rank by most rare tags first, then quality
        diversity_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for n_rare, score, p in diversity_candidates:
            if len(selected_ids) >= max_budget:
                break
            selected_ids.add(p["id"])
            phase2_added += 1

    logger.info("Phase 2 tag diversity: added %d, total now %d",
                phase2_added, len(selected_ids))

    # Phase 3: If we haven't hit target, add runner-ups from large clusters
    # prioritizing high-quality photos
    if len(selected_ids) < target_count:
        runner_ups = []
        for cid, members in cluster_groups.items():
            for rank, p in enumerate(members[1:], 2):
                score = scores.get(p["id"], 0)
                if score >= high_quality:
                    runner_ups.append((score, rank, p))
        runner_ups.sort(key=lambda x: x[0], reverse=True)
        for score, rank, p in runner_ups:
            if len(selected_ids) >= target_count:
                break
            selected_ids.add(p["id"])

    # Phase 4: If still under target (e.g., low-quality folder), fill with
    # the best remaining unselected photos by score
    if len(selected_ids) < target_count:
        remaining = [
            p for p in photos_with_emb if p["id"] not in selected_ids
        ]
        remaining.sort(key=lambda p: scores.get(p["id"], 0), reverse=True)
        for p in remaining:
            if len(selected_ids) >= target_count:
                break
            selected_ids.add(p["id"])

    # Build final result with rank info
    result = []
    for cid, members in cluster_groups.items():
        for rank, p in enumerate(members, 1):
            p["selected"] = p["id"] in selected_ids
            p["rank_in_cluster"] = rank
            result.append(p)

    # Add photos without embeddings (unselected)
    for p in photos_without_emb:
        p["selected"] = False
        p["rank_in_cluster"] = 0
        result.append(p)

    # Sort: selected first, then by quality
    result.sort(key=lambda p: (not p["selected"], -(scores.get(p["id"], 0) or 0)))

    n_selected = sum(1 for p in result if p["selected"])
    logger.info("Final selection: %d / %d photos (%.1f%%)",
                n_selected, len(result), 100.0 * n_selected / len(result) if result else 0)

    return result


# ---------------------------------------------------------------------------
# Persistence — save/load selections
# ---------------------------------------------------------------------------

def save_selections(db: PhotoDB, directory: str, selections: list[dict]):
    """Persist photo selections for a directory review session.

    Stores in the review_selections table so the web UI can load them.
    """
    directory = str(Path(directory).resolve())

    # Clear previous selections for this directory
    db.conn.execute(
        "DELETE FROM review_selections WHERE directory = ?", (directory,)
    )

    for p in selections:
        db.conn.execute(
            """INSERT INTO review_selections
               (photo_id, directory, selected, cluster_id, rank_in_cluster)
               VALUES (?, ?, ?, ?, ?)""",
            (p["id"], directory, 1 if p["selected"] else 0,
             p.get("cluster_id"), p.get("rank_in_cluster")),
        )
    db.conn.commit()


def load_selections(db: PhotoDB, directory: str) -> Optional[list[dict]]:
    """Load saved selections for a directory.

    Returns list of {photo_id, selected, cluster_id, rank_in_cluster} or None.
    """
    directory = str(Path(directory).resolve())
    rows = db.conn.execute(
        """SELECT rs.photo_id, rs.selected, rs.cluster_id, rs.rank_in_cluster,
                  p.filepath, p.filename, p.aesthetic_score, p.date_taken, p.raw_filepath
           FROM review_selections rs
           JOIN photos p ON p.id = rs.photo_id
           WHERE rs.directory = ?
           ORDER BY rs.selected DESC, p.aesthetic_score DESC""",
        (directory,),
    ).fetchall()

    if not rows:
        return None

    return [dict(r) for r in rows]


def toggle_selection(db: PhotoDB, photo_id: int, selected: bool):
    """Toggle a single photo's selection state."""
    db.conn.execute(
        "UPDATE review_selections SET selected = ? WHERE photo_id = ?",
        (1 if selected else 0, photo_id),
    )
    db.conn.commit()

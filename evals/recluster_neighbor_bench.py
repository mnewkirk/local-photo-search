#!/usr/bin/env python3
"""Option-1 recluster prototype: precomputed sparse radius graph vs ball_tree.

The global face recluster (`photosearch.faces.recluster_unknown_faces`) runs
`DBSCAN(algorithm="ball_tree")` over every `person_id IS NULL` encoding. At
512 dimensions ball_tree degenerates to ~brute-force O(n^2), which is the slow
stage in the maintenance sweep.

DBSCAN's only data-dependent cost is the eps-neighborhood query; the labelling
that follows is cheap graph traversal. ArcFace encodings are immutable, so that
neighborhood graph is stable across runs — it can be built from the existing
`face_encodings` vec0 index (and cached/extended incrementally) instead of
rebuilt in numpy each sweep.

This harness builds the radius graph from sqlite-vec KNN, feeds it to
`DBSCAN(metric="precomputed")`, and checks the labels match the ball_tree
baseline (adjusted Rand index + noise-status agreement). Optionally also
benchmarks hnswlib if installed (the true asymptotic win for cold rebuilds).

Usage:
    python evals/recluster_neighbor_bench.py --db photo_index.db.local --sample 3000
    python evals/recluster_neighbor_bench.py --db photo_index.db.local --full
"""
from __future__ import annotations

import argparse
import sqlite3
import time

import numpy as np
import sqlite_vec
from scipy import sparse
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

# Mirror the production recluster defaults (photosearch/faces.py).
FACE_DIM = 512
EPS = 0.55
MIN_SAMPLES = 3
MIN_DET_SCORE = 0.65
MIN_BBOX_EDGE = 60
# Avoid storing a structural zero in the sparse graph (duplicate encodings sit
# at true distance 0; scipy/sklearn drop explicit zeros, which would silently
# delete those edges). Nudge every stored distance to >= this and widen eps by
# the same amount so nothing real is excluded.
ZERO_FLOOR = 1e-6


def _ro_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def load_clustering_set(db_path: str, limit: int | None, random_sample: bool):
    """Load the exact encoding set the production recluster would cluster."""
    conn = _ro_conn(db_path)
    order = "RANDOM()" if random_sample else "f.id"
    sql = f"""
        SELECT f.id, fe.encoding
        FROM faces f
        JOIN face_encodings fe ON fe.face_id = f.id
        LEFT JOIN photos ph ON ph.id = f.photo_id
        WHERE f.person_id IS NULL
          AND (f.det_score IS NULL OR f.det_score >= ?)
          AND MIN(f.bbox_bottom - f.bbox_top, f.bbox_right - f.bbox_left) >= ?
        ORDER BY {order}
    """
    params: list = [MIN_DET_SCORE, MIN_BBOX_EDGE]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    t0 = time.time()
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    ids = [r["id"] for r in rows]
    X = (
        np.frombuffer(b"".join(r["encoding"] for r in rows), dtype=np.float32)
        .reshape(len(rows), FACE_DIM)
        .copy()
    )
    print(f"  loaded {len(ids):,} encodings in {time.time() - t0:.1f}s")
    return ids, X


def dbscan_ball_tree(X: np.ndarray, eps: float, min_samples: int):
    """Current production path — the baseline we must match."""
    t0 = time.time()
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    ).fit_predict(X)
    return labels, time.time() - t0


def build_vec0_graph(X: np.ndarray, eps: float, k: int):
    """Radius graph via an in-memory sqlite-vec index (the option-1 path).

    Returns (csr_graph, build_seconds, n_saturated). n_saturated counts rows
    whose k-th neighbor was still within eps — i.e. the neighborhood was
    truncated by k and the graph is missing edges for that point.
    """
    n = len(X)
    k = min(k, n, 4096)  # vec0 hard-caps k at 4096
    mem = sqlite3.connect(":memory:")
    mem.enable_load_extension(True)
    sqlite_vec.load(mem)
    mem.enable_load_extension(False)
    mem.execute(
        f"CREATE VIRTUAL TABLE fe USING vec0(id INTEGER PRIMARY KEY, v float[{FACE_DIM}])"
    )

    t0 = time.time()
    mem.executemany(
        "INSERT INTO fe(id, v) VALUES (?, ?)",
        ((i, X[i].tobytes()) for i in range(n)),
    )
    t_insert = time.time()

    rows_i: list[int] = []
    cols_j: list[int] = []
    data: list[float] = []
    n_saturated = 0
    for i in range(n):
        res = mem.execute(
            "SELECT id, distance FROM fe WHERE v MATCH ? AND k = ? ORDER BY distance",
            (X[i].tobytes(), k),
        ).fetchall()
        within = [(j, d) for (j, d) in res if d <= eps]
        if len(within) == k:  # k-th neighbor still inside eps -> truncated
            n_saturated += 1
        for j, d in within:
            rows_i.append(i)
            cols_j.append(j)
            data.append(max(d, ZERO_FLOOR))
    t_query = time.time()
    mem.close()

    g = sparse.csr_matrix((data, (rows_i, cols_j)), shape=(n, n))
    print(
        f"  vec0 graph: insert {t_insert - t0:.1f}s, "
        f"{n:,} knn queries {t_query - t_insert:.1f}s, "
        f"{g.nnz:,} edges, {n_saturated} saturated (k={k})"
    )
    return g, time.time() - t0, n_saturated


def build_hnsw_graph(X: np.ndarray, eps: float, k: int):
    """Radius graph via hnswlib ANN (optional — the cold-rebuild asymptotic win)."""
    try:
        import hnswlib
    except ImportError:
        return None, 0.0, 0
    n = len(X)
    k = min(k, n)
    t0 = time.time()
    index = hnswlib.Index(space="l2", dim=FACE_DIM)
    index.init_index(max_elements=n, ef_construction=200, M=32)
    index.add_items(X, np.arange(n), num_threads=-1)
    index.set_ef(max(k * 2, 64))
    labels_knn, dists_sq = index.knn_query(X, k=k, num_threads=-1)
    t_query = time.time()

    rows_i, cols_j, data = [], [], []
    n_saturated = 0
    for i in range(n):
        kept = 0
        for col, dsq in zip(labels_knn[i], dists_sq[i]):
            d = float(np.sqrt(max(dsq, 0.0)))  # hnswlib l2 returns squared dist
            if d <= eps:
                rows_i.append(i)
                cols_j.append(int(col))
                data.append(max(d, ZERO_FLOOR))
                kept += 1
        if kept == k:
            n_saturated += 1
    mem_g = sparse.csr_matrix((data, (rows_i, cols_j)), shape=(n, n))
    print(
        f"  hnsw graph: build+query {t_query - t0:.1f}s, "
        f"{mem_g.nnz:,} edges, {n_saturated} saturated (k={k})"
    )
    return mem_g, time.time() - t0, n_saturated


def dbscan_precomputed(graph, eps: float, min_samples: int):
    t0 = time.time()
    labels = DBSCAN(
        eps=eps + ZERO_FLOOR,  # widen for the zero-floor nudge
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(graph)
    return labels, time.time() - t0


def compare(name: str, baseline, candidate):
    ari = adjusted_rand_score(baseline, candidate)
    base_noise = baseline == -1
    cand_noise = candidate == -1
    noise_agree = float((base_noise == cand_noise).mean())
    n_clusters_base = len(set(baseline.tolist()) - {-1})
    n_clusters_cand = len(set(candidate.tolist()) - {-1})
    print(
        f"  [{name}] ARI={ari:.6f}  noise-status agreement={noise_agree:.4%}  "
        f"clusters: base={n_clusters_base} cand={n_clusters_cand}  "
        f"noise: base={int(base_noise.sum())} cand={int(cand_noise.sum())}"
    )
    return ari, noise_agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="photo_index.db.local")
    ap.add_argument("--sample", type=int, default=3000,
                    help="cluster only N faces (default 3000); ignored with --full")
    ap.add_argument("--full", action="store_true",
                    help="cluster the entire person_id IS NULL set")
    ap.add_argument("--random", action="store_true",
                    help="random sample instead of first-N-by-id")
    ap.add_argument("--eps", type=float, default=EPS)
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    ap.add_argument("--k", type=int, default=512,
                    help="KNN fan-out per point for the radius graph")
    ap.add_argument("--no-ball-tree", action="store_true",
                    help="skip the ball_tree baseline (e.g. on the full set)")
    ap.add_argument("--no-vec0", action="store_true",
                    help="skip the single-threaded vec0 path (too slow above ~10k)")
    args = ap.parse_args()

    limit = None if args.full else args.sample
    print(f"Loading clustering set (limit={limit}, random={args.random}) from {args.db}")
    ids, X = load_clustering_set(args.db, limit, args.random)
    if len(ids) == 0:
        print("No faces matched — nothing to do.")
        return
    print(f"eps={args.eps} min_samples={args.min_samples} k={args.k}\n")

    baseline = None
    if not args.no_ball_tree:
        labels_bt, t_bt = dbscan_ball_tree(X, args.eps, args.min_samples)
        baseline = labels_bt
        print(f"ball_tree DBSCAN: {t_bt:.1f}s  "
              f"({len(set(labels_bt.tolist()) - {-1})} clusters, "
              f"{int((labels_bt == -1).sum())} noise)\n")

    if not args.no_vec0:
        g_vec, t_vec_graph, sat_vec = build_vec0_graph(X, args.eps, args.k)
        labels_vec, t_vec_db = dbscan_precomputed(g_vec, args.eps, args.min_samples)
        print(f"vec0 precomputed: graph {t_vec_graph:.1f}s + dbscan {t_vec_db:.1f}s "
              f"= {t_vec_graph + t_vec_db:.1f}s")
        if baseline is not None:
            compare("vec0 vs ball_tree", baseline, labels_vec)
        print()

    g_hnsw, t_hnsw_graph, sat_hnsw = build_hnsw_graph(X, args.eps, args.k)
    if g_hnsw is not None:
        labels_hnsw, t_hnsw_db = dbscan_precomputed(g_hnsw, args.eps, args.min_samples)
        print(f"hnsw precomputed: graph {t_hnsw_graph:.1f}s + dbscan {t_hnsw_db:.1f}s "
              f"= {t_hnsw_graph + t_hnsw_db:.1f}s")
        if baseline is not None:
            compare("hnsw vs ball_tree", baseline, labels_hnsw)
    else:
        print("hnswlib not installed — skipping ANN comparison "
              "(pip install hnswlib to benchmark the cold-rebuild win)")


if __name__ == "__main__":
    main()

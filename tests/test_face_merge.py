"""Unit tests for photosearch.face_merge — merge suggestion engine.

Also covers `_session_stack_noise` from photosearch.faces, which the engine
relies on (both are the M18 surfaces for improving unknown-face grouping).
"""

import numpy as np
import pytest

from photosearch.face_merge import (
    GroupInfo,
    compute_suggestions,
    load_groups,
    parse_verify_pair,
    resolve_group_spec,
    score_pair,
)


# ---------------------------------------------------------------------------
# Test helpers — synthetic unit-norm encodings
# ---------------------------------------------------------------------------

DIM = 512


def _unit(vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    return (vec / n).astype(np.float32) if n > 0 else vec


def _base(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return _unit(rng.randn(DIM).astype(np.float32))


def _perturb(base: np.ndarray, seed: int, jitter: float = 0.1) -> np.ndarray:
    """Small random perturbation, re-normalized.

    Scales gaussian noise by 1/sqrt(DIM) so the perturbation's norm is ~jitter
    (a gaussian of N dims has norm ~sqrt(N)). Without this, at DIM=512 even
    a tiny scalar jitter would dominate the unit-norm base after normalize.
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(DIM).astype(np.float32) / np.sqrt(DIM)
    return _unit(base + jitter * noise)


def _make_group(kind: str, gid: int, label: str, encodings: list[np.ndarray]) -> GroupInfo:
    X = np.stack(encodings).astype(np.float32, copy=False)
    centroid = _unit(X.mean(axis=0))
    return GroupInfo(
        key=f"{'p' if kind == 'person' else 'c'}:{gid}",
        type=kind, id=gid, label=label,
        face_ids=list(range(len(encodings))),
        encodings=X, centroid=centroid,
        face_count=len(encodings), rep_face_id=0,
    )


# ---------------------------------------------------------------------------
# compute_suggestions
# ---------------------------------------------------------------------------

def test_compute_suggestions_merges_similar_clusters():
    shared = _base(42)
    a = _make_group("cluster", 1, "Unknown #1",
                    [_perturb(shared, s) for s in [1, 2, 3]])
    b = _make_group("cluster", 2, "Unknown #2",
                    [_perturb(shared, s) for s in [10, 11]])
    c = _make_group("cluster", 3, "Unknown #3",
                    [_perturb(_base(999), s) for s in [20, 21]])

    suggestions = compute_suggestions([a, b, c])

    pairs = {frozenset((s.left.id, s.right.id)) for s in suggestions}
    assert frozenset((1, 2)) in pairs, "similar clusters should merge"
    assert frozenset((1, 3)) not in pairs
    assert frozenset((2, 3)) not in pairs


def test_compute_suggestions_canonicalizes_cluster_to_person_direction():
    """When pairing a cluster with a person, cluster lands on left (merge-source)."""
    shared = _base(7)
    cluster = _make_group("cluster", 5, "Unknown #5", [_perturb(shared, 1)])
    person = _make_group("person", 10, "Alice",
                         [_perturb(shared, s) for s in [5, 6, 7]])

    suggestions = compute_suggestions([cluster, person])
    assert len(suggestions) == 1
    s = suggestions[0]
    assert s.left.type == "cluster" and s.right.type == "person"
    assert s.left.id == 5 and s.right.id == 10


def test_compute_suggestions_canonicalizes_cluster_pair_by_size():
    """For cluster↔cluster pairs, the smaller cluster goes on the left."""
    shared = _base(8)
    small = _make_group("cluster", 100, "Unknown #100",
                        [_perturb(shared, 1)])
    big = _make_group("cluster", 200, "Unknown #200",
                      [_perturb(shared, s) for s in [2, 3, 4, 5]])

    suggestions = compute_suggestions([big, small])
    assert len(suggestions) == 1
    assert suggestions[0].left.id == 100 and suggestions[0].right.id == 200


def test_compute_suggestions_skips_person_person_pairs():
    shared = _base(1)
    p1 = _make_group("person", 1, "Alice", [_perturb(shared, 1)])
    p2 = _make_group("person", 2, "Bob", [_perturb(shared, 2)])
    assert compute_suggestions([p1, p2]) == []


def test_compute_suggestions_sorted_by_min_pair_distance():
    shared_ab = _base(50)
    shared_de = _base(60)
    a = _make_group("cluster", 1, "A", [_perturb(shared_ab, 1, jitter=0.03)])
    b = _make_group("cluster", 2, "B", [_perturb(shared_ab, 2, jitter=0.03)])
    d = _make_group("cluster", 3, "D", [_perturb(shared_de, 1, jitter=0.15)])
    e = _make_group("cluster", 4, "E", [_perturb(shared_de, 2, jitter=0.15)])

    suggestions = compute_suggestions([a, b, d, e])
    assert len(suggestions) >= 1
    dists = [s.min_pair_dist for s in suggestions]
    assert dists == sorted(dists)


def test_compute_suggestions_respects_cutoffs():
    shared = _base(11)
    a = _make_group("cluster", 1, "A", [_perturb(shared, 1)])
    b = _make_group("cluster", 2, "B", [_perturb(shared, 2)])
    # Ceiling of 0.0 excludes everything.
    assert compute_suggestions([a, b], min_pair_cutoff=0.0) == []


def test_score_pair_reports_both_metrics():
    shared = _base(3)
    a = _make_group("cluster", 1, "A", [_perturb(shared, 1)])
    b = _make_group("cluster", 2, "B", [_perturb(shared, 2)])
    cd, md = score_pair(a, b)
    assert 0.0 <= md <= 2.0 and 0.0 <= cd <= 2.0
    # min-pair must never exceed centroid-to-centroid (members define the centroid)
    # on these small groups; this is a useful sanity invariant.
    assert md <= cd + 1e-6


# ---------------------------------------------------------------------------
# parse_verify_pair
# ---------------------------------------------------------------------------

def test_parse_verify_pair_positive():
    left, right, match = parse_verify_pair("cluster:2035=person:Matt Newkirk")
    assert left == "cluster:2035"
    assert right == "person:Matt Newkirk"
    assert match is True


def test_parse_verify_pair_negative():
    left, right, match = parse_verify_pair("cluster:798!=cluster:745")
    assert (left, right, match) == ("cluster:798", "cluster:745", False)


def test_parse_verify_pair_missing_separator():
    with pytest.raises(ValueError):
        parse_verify_pair("cluster:1 cluster:2")


# ---------------------------------------------------------------------------
# load_groups / resolve_group_spec — backed by the conftest db fixture
# ---------------------------------------------------------------------------

def test_load_groups_from_db(db):
    groups = load_groups(db)
    types = {g.type for g in groups}
    assert "person" in types
    assert "cluster" in types

    # Fixture cluster id = 99, one face
    cluster_group = next(g for g in groups if g.type == "cluster")
    assert cluster_group.id == 99
    assert cluster_group.face_count == 1
    assert cluster_group.encodings.shape == (1, DIM)


def test_resolve_group_spec_cluster_and_person(db):
    groups = load_groups(db)
    c = resolve_group_spec(db, groups, "cluster:99")
    assert c is not None and c.id == 99
    p = resolve_group_spec(db, groups, "person:Alex")
    assert p is not None and p.label == "Alex"


def test_resolve_group_spec_case_insensitive_person(db):
    groups = load_groups(db)
    assert resolve_group_spec(db, groups, "person:alex") is not None


def test_resolve_group_spec_unknown_returns_none(db):
    groups = load_groups(db)
    assert resolve_group_spec(db, groups, "person:Nobody") is None
    assert resolve_group_spec(db, groups, "cluster:9999") is None
    assert resolve_group_spec(db, groups, "bogus:format") is None


# ---------------------------------------------------------------------------
# Session-stacking second pass (photosearch.faces._session_stack_noise)
# ---------------------------------------------------------------------------

def test_session_stack_links_time_proximate_similar_noise():
    from photosearch.faces import _session_stack_noise

    shared = _base(0)
    X = np.stack([
        _perturb(shared, 1),            # #0 — noise
        _perturb(shared, 2),            # #1 — same person as #0, 10 min later
        _base(200),                     # #2 — different, same day
        _base(300),                     # #3 — different, different day
    ]).astype(np.float32)
    labels = np.array([-1, -1, -1, -1], dtype=np.int64)
    dates = [
        "2024-06-01T10:00:00",
        "2024-06-01T10:10:00",          # within 60-min window
        "2024-06-01T12:00:00",
        "2024-06-05T10:00:00",
    ]

    new_labels, n_clusters = _session_stack_noise(
        face_ids=[1, 2, 3, 4], X=X, labels=labels, dates=dates,
        session_eps=0.50, session_window_minutes=60.0,
    )
    assert n_clusters == 1
    assert new_labels[0] == new_labels[1] and new_labels[0] >= 0
    assert new_labels[2] == -1 and new_labels[3] == -1


def test_session_stack_respects_time_window():
    from photosearch.faces import _session_stack_noise

    shared = _base(0)
    X = np.stack([_perturb(shared, 1), _perturb(shared, 2)]).astype(np.float32)
    labels = np.array([-1, -1], dtype=np.int64)
    dates = ["2024-06-01T10:00:00", "2024-06-01T12:00:00"]  # 2h apart
    _, n_clusters = _session_stack_noise(
        face_ids=[1, 2], X=X, labels=labels, dates=dates,
        session_eps=0.50, session_window_minutes=60.0,
    )
    assert n_clusters == 0


def test_session_stack_respects_similarity_cutoff():
    from photosearch.faces import _session_stack_noise

    X = np.stack([_base(1), _base(2)]).astype(np.float32)  # unrelated
    labels = np.array([-1, -1], dtype=np.int64)
    dates = ["2024-06-01T10:00:00", "2024-06-01T10:05:00"]  # within window
    _, n_clusters = _session_stack_noise(
        face_ids=[1, 2], X=X, labels=labels, dates=dates,
        session_eps=0.50, session_window_minutes=60.0,
    )
    assert n_clusters == 0


def test_session_stack_preserves_existing_dbscan_labels():
    """Existing DBSCAN cluster assignments must not be clobbered."""
    from photosearch.faces import _session_stack_noise

    shared = _base(0)
    X = np.stack([
        _perturb(shared, 1), _perturb(shared, 2),   # DBSCAN cluster 0
        _perturb(_base(50), 1), _perturb(_base(50), 2),  # noise, similar to each other
    ]).astype(np.float32)
    labels = np.array([0, 0, -1, -1], dtype=np.int64)
    dates = [
        "2024-06-01T10:00:00", "2024-06-01T10:02:00",
        "2024-06-02T14:00:00", "2024-06-02T14:10:00",
    ]
    new_labels, n_clusters = _session_stack_noise(
        face_ids=[1, 2, 3, 4], X=X, labels=labels, dates=dates,
        session_eps=0.50, session_window_minutes=60.0,
    )
    assert new_labels[0] == 0 and new_labels[1] == 0
    assert n_clusters == 1
    assert new_labels[2] == new_labels[3] >= 1  # new id above 0

"""
Tests for photosearch/cull.py — shoot review and photo culling.

Covers:
  - _cluster_photos: agglomerative clustering with fixed/adaptive thresholds
  - select_best_photos: 4-phase selection algorithm
  - save_selections / load_selections / toggle_selection: DB persistence

Uses synthetic embeddings and the shared conftest.py db fixture —
no real photos or ML models needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# scipy is required for clustering — skip if it's mocked or unavailable.
# conftest.py injects a MagicMock for scipy when it's not installed, so a
# simple import check isn't enough. We try to actually call pdist to verify.
try:
    from scipy.spatial.distance import pdist
    _test = pdist(np.array([[1.0, 0.0], [0.0, 1.0]]), metric="cosine")
    HAS_SCIPY = hasattr(_test, "__len__") and len(_test) == 1
except Exception:
    HAS_SCIPY = False

from photosearch.cull import (
    _cluster_photos,
    select_best_photos,
    save_selections,
    load_selections,
    toggle_selection,
)
from photosearch.db import PhotoDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    """Create a deterministic unit vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_cluster_embeddings(
    n_clusters: int = 3,
    per_cluster: int = 5,
    noise: float = 0.02,
    dim: int = 512,
) -> dict[int, np.ndarray]:
    """Create synthetic embeddings with clear cluster structure.

    Each cluster has a random centroid; members are centroid + small noise.
    Returns {photo_id: embedding}.
    """
    embs = {}
    pid = 1
    for c in range(n_clusters):
        centroid = _make_unit_vec(dim, seed=c * 1000)
        for i in range(per_cluster):
            rng = np.random.RandomState(c * 1000 + i + 1)
            noisy = centroid + rng.randn(dim).astype(np.float32) * noise
            noisy /= np.linalg.norm(noisy)
            embs[pid] = noisy
            pid += 1
    return embs


# ---------------------------------------------------------------------------
# _cluster_photos tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for clustering tests")
class TestClusterPhotos:
    """Tests for the pure clustering algorithm."""

    def test_single_photo(self):
        """A single photo should get cluster label 0."""
        embs = {1: _make_unit_vec(seed=42)}
        labels = _cluster_photos(embs)
        assert labels == {1: 0}

    def test_empty_dict(self):
        """Empty input should return empty output."""
        labels = _cluster_photos({})
        assert labels == {}

    def test_distinct_clusters_found(self):
        """Well-separated clusters should produce distinct cluster labels."""
        embs = _make_cluster_embeddings(n_clusters=3, per_cluster=5, noise=0.01)
        labels = _cluster_photos(embs, distance_threshold=0.1)
        unique_labels = set(labels.values())
        # Should find at least 2 distinct clusters (3 expected, but
        # threshold-dependent — just verify it doesn't collapse to 1)
        assert len(unique_labels) >= 2, f"Expected multiple clusters, got {unique_labels}"

    def test_all_photos_assigned(self):
        """Every photo ID should appear in the output."""
        embs = _make_cluster_embeddings(n_clusters=2, per_cluster=4)
        labels = _cluster_photos(embs, distance_threshold=0.1)
        assert set(labels.keys()) == set(embs.keys())

    def test_identical_embeddings_same_cluster(self):
        """Identical vectors should land in the same cluster."""
        base = _make_unit_vec(seed=99)
        embs = {1: base.copy(), 2: base.copy(), 3: base.copy()}
        labels = _cluster_photos(embs, distance_threshold=0.05)
        assert labels[1] == labels[2] == labels[3]

    def test_adaptive_threshold_targets_cluster_count(self):
        """Adaptive mode (distance_threshold=0) should aim for target_clusters."""
        embs = _make_cluster_embeddings(n_clusters=5, per_cluster=8, noise=0.03)
        target = 10
        labels = _cluster_photos(embs, distance_threshold=0.0, target_clusters=target)
        n_found = len(set(labels.values()))
        # Should be within +/- 3 of the target
        assert abs(n_found - target) <= 3, (
            f"Adaptive clustering target={target}, got {n_found} clusters"
        )

    def test_fixed_threshold_produces_fewer_clusters_when_loose(self):
        """A loose threshold should merge more clusters."""
        embs = _make_cluster_embeddings(n_clusters=4, per_cluster=5, noise=0.02)
        tight = _cluster_photos(embs, distance_threshold=0.05)
        loose = _cluster_photos(embs, distance_threshold=0.5)
        assert len(set(loose.values())) <= len(set(tight.values()))

    def test_returns_int_labels(self):
        """Cluster labels should be plain ints (not numpy types)."""
        embs = _make_cluster_embeddings(n_clusters=2, per_cluster=3)
        labels = _cluster_photos(embs, distance_threshold=0.1)
        for label in labels.values():
            assert isinstance(label, int)


# ---------------------------------------------------------------------------
# select_best_photos tests (using a real DB with synthetic data)
# ---------------------------------------------------------------------------

@pytest.fixture
def cull_db(tmp_path):
    """A PhotoDB populated with 30 photos across 3 quality tiers,
    each with CLIP embeddings forming ~3 visual clusters.
    """
    import struct

    db_path = str(tmp_path / "cull_test.db")
    db = PhotoDB(db_path)

    photo_dir = tmp_path / "photos" / "shoot1"
    photo_dir.mkdir(parents=True)
    db.set_photo_root(str(tmp_path / "photos"))

    # Create 30 photos in 3 visual clusters (10 each), with varying quality
    photo_ids = []
    n_clusters = 3
    per_cluster = 10

    for c in range(n_clusters):
        centroid = _make_unit_vec(seed=c * 1000)
        for i in range(per_cluster):
            idx = c * per_cluster + i
            fname = f"IMG_{idx:04d}.JPG"
            fpath = f"shoot1/{fname}"

            # Create a dummy file so resolve_filepath works
            (photo_dir / fname).write_bytes(b"fake")

            # Quality: cluster 0 gets high scores, cluster 1 medium, cluster 2 low
            base_score = [7.0, 5.0, 3.0][c]
            score = base_score + (i * 0.1)  # slight variation within cluster

            # Tags: most share cluster tags, a few have rare tags for diversity
            cluster_tags = [
                ["ocean", "waves", "rocks"],
                ["forest", "trees", "trail"],
                ["city", "buildings", "street"],
            ][c]
            if i == per_cluster - 1:
                # Last photo in each cluster gets a rare tag
                tags = cluster_tags + ["rainbow"]
            else:
                tags = cluster_tags

            pid = db.add_photo(
                filepath=fpath,
                filename=fname,
                date_taken=f"2026-03-13T{10 + c}:{i:02d}:00",
                aesthetic_score=score,
                tags=json.dumps(tags),
            )
            photo_ids.append(pid)

            # Add CLIP embedding — tight cluster with small noise
            rng = np.random.RandomState(c * 1000 + i + 1)
            noisy = centroid + rng.randn(512).astype(np.float32) * 0.02
            noisy /= np.linalg.norm(noisy)
            db.add_clip_embedding(pid, noisy.tolist())

    db.conn.commit()
    db._test_photo_ids = photo_ids
    db._test_photo_dir = str(photo_dir)

    yield db
    db.close()


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for selection tests")
class TestSelectBestPhotos:
    """Tests for the multi-phase selection algorithm."""

    def test_returns_all_photos(self, cull_db):
        """Result should contain every photo, selected or not."""
        result = select_best_photos(cull_db, cull_db._test_photo_dir)
        assert len(result) == 30

    def test_selected_count_near_target(self, cull_db):
        """Number of selected photos should be roughly target_pct of total."""
        result = select_best_photos(
            cull_db, cull_db._test_photo_dir, target_pct=0.10
        )
        selected = [p for p in result if p["selected"]]
        # 10% of 30 = 3, but with diversity budget it might be up to ~5-6
        assert 2 <= len(selected) <= 10, (
            f"Expected ~3 selected (10% of 30), got {len(selected)}"
        )

    def test_selected_photos_have_cluster_ids(self, cull_db):
        """Selected photos must have valid cluster IDs."""
        result = select_best_photos(cull_db, cull_db._test_photo_dir)
        for p in result:
            if p["selected"]:
                assert "cluster_id" in p
                assert p["cluster_id"] is not None

    def test_best_quality_preferred(self, cull_db):
        """Higher-quality photos should be preferred as cluster reps."""
        result = select_best_photos(cull_db, cull_db._test_photo_dir)
        selected = [p for p in result if p["selected"]]
        # The selected photos should have above-average quality
        all_scores = [p["aesthetic_score"] or 0 for p in result]
        avg_score = sum(all_scores) / len(all_scores)
        selected_scores = [p["aesthetic_score"] or 0 for p in selected]
        avg_selected = sum(selected_scores) / len(selected_scores)
        assert avg_selected >= avg_score, (
            f"Selected avg {avg_selected:.1f} should be >= overall avg {avg_score:.1f}"
        )

    def test_selected_first_in_result(self, cull_db):
        """Result should be sorted: selected photos first."""
        result = select_best_photos(cull_db, cull_db._test_photo_dir)
        saw_unselected = False
        for p in result:
            if not p["selected"]:
                saw_unselected = True
            elif saw_unselected:
                pytest.fail("Selected photo appeared after unselected photo in result")

    def test_higher_target_pct_selects_more(self, cull_db):
        """Increasing target_pct should select more photos."""
        result_10 = select_best_photos(
            cull_db, cull_db._test_photo_dir, target_pct=0.10
        )
        result_30 = select_best_photos(
            cull_db, cull_db._test_photo_dir, target_pct=0.30
        )
        n_10 = sum(1 for p in result_10 if p["selected"])
        n_30 = sum(1 for p in result_30 if p["selected"])
        assert n_30 >= n_10, (
            f"30% target selected {n_30}, but 10% target selected {n_10}"
        )

    def test_empty_directory_returns_empty(self, cull_db):
        """A directory with no photos should return an empty list."""
        result = select_best_photos(cull_db, "/nonexistent/path")
        assert result == []

    def test_min_quality_excludes_low_from_phase1(self, cull_db):
        """An impossibly high min_quality should block all Phase 1 reps.

        min_quality gates Phase 1 cluster rep selection.  With a threshold
        above every photo's score, Phase 1 selects nothing.  Phases 3-4
        can still backfill to meet the target count, but fewer photos
        should be selected overall compared to a permissive threshold.
        """
        result_easy = select_best_photos(
            cull_db, cull_db._test_photo_dir, min_quality=0.0
        )
        result_impossible = select_best_photos(
            cull_db, cull_db._test_photo_dir, min_quality=99.0
        )
        n_easy = sum(1 for p in result_easy if p["selected"])
        n_impossible = sum(1 for p in result_impossible if p["selected"])
        assert n_impossible <= n_easy, (
            f"min_quality=99 selected {n_impossible}, "
            f"min_quality=0 selected {n_easy}"
        )


@pytest.fixture
def dup_heavy_db(tmp_path):
    """A PhotoDB with one big near-identical high-quality cluster (20 photos)
    plus 5 visually distinct photos — the shape that made the review select the
    same scene many times over.
    """
    db_path = str(tmp_path / "dup_test.db")
    db = PhotoDB(db_path)
    photo_dir = tmp_path / "photos" / "shoot1"
    photo_dir.mkdir(parents=True)
    db.set_photo_root(str(tmp_path / "photos"))

    # 20 near-identical, high-quality photos — one tight visual cluster.
    dup_centroid = _make_unit_vec(seed=7)
    idx = 0
    for i in range(20):
        fname = f"IMG_{idx:04d}.JPG"
        (photo_dir / fname).write_bytes(b"fake")
        rng = np.random.RandomState(500 + i)
        v = dup_centroid + rng.randn(512).astype(np.float32) * 0.005
        v /= np.linalg.norm(v)
        pid = db.add_photo(filepath=f"shoot1/{fname}", filename=fname,
                           date_taken=f"2026-03-13T10:{i:02d}:00",
                           aesthetic_score=6.0 + i * 0.02)
        db.add_clip_embedding(pid, v.tolist())
        idx += 1

    # 5 visually distinct photos, each its own cluster.
    for c in range(5):
        fname = f"IMG_{idx:04d}.JPG"
        (photo_dir / fname).write_bytes(b"fake")
        v = _make_unit_vec(seed=9000 + c * 111)
        pid = db.add_photo(filepath=f"shoot1/{fname}", filename=fname,
                           date_taken=f"2026-03-13T12:{c:02d}:00",
                           aesthetic_score=5.5)
        db.add_clip_embedding(pid, v.tolist())
        idx += 1

    db.conn.commit()
    db._test_photo_dir = str(photo_dir)
    yield db
    db.close()


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for selection tests")
class TestPerClusterCap:
    """The top-up phases must not over-represent a single near-duplicate scene."""

    def test_dup_cluster_capped(self, dup_heavy_db):
        """Even with a generous target, the 20-photo near-dup cluster should
        contribute at most _MAX_PER_CLUSTER selections."""
        from photosearch.cull import _MAX_PER_CLUSTER

        result = select_best_photos(
            dup_heavy_db, dup_heavy_db._test_photo_dir, target_pct=0.5
        )
        from collections import Counter
        counts = Counter(
            p["cluster_id"] for p in result
            if p["selected"] and p["cluster_id"] is not None
        )
        assert counts, "expected some selections"
        assert max(counts.values()) <= _MAX_PER_CLUSTER, (
            f"a single cluster over-represented: {dict(counts)}"
        )


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for selection persistence tests")
class TestSelectionPersistence:
    """Tests for save/load/toggle operations on review_selections."""

    def test_save_and_load_roundtrip(self, cull_db):
        """Saved selections should be loadable."""
        directory = cull_db._test_photo_dir
        selections = select_best_photos(cull_db, directory)
        save_selections(cull_db, directory, selections)

        loaded = load_selections(cull_db, directory)
        assert loaded is not None
        assert len(loaded) == len(selections)

        # Check that selected state matches
        saved_selected = {p["id"] for p in selections if p["selected"]}
        loaded_selected = {r["photo_id"] for r in loaded if r["selected"]}
        assert saved_selected == loaded_selected

    def test_load_nonexistent_directory(self, cull_db):
        """Loading selections for an unknown directory returns None."""
        result = load_selections(cull_db, "/no/such/directory")
        assert result is None

    def test_save_overwrites_previous(self, cull_db):
        """Saving again for the same directory replaces old selections."""
        directory = cull_db._test_photo_dir
        selections = select_best_photos(cull_db, directory)
        save_selections(cull_db, directory, selections)

        # Flip all selections
        for p in selections:
            p["selected"] = not p["selected"]
        save_selections(cull_db, directory, selections)

        loaded = load_selections(cull_db, directory)
        flipped_selected = {p["id"] for p in selections if p["selected"]}
        loaded_selected = {r["photo_id"] for r in loaded if r["selected"]}
        assert flipped_selected == loaded_selected

    def test_toggle_selection(self, cull_db):
        """toggle_selection should flip a photo's selected state."""
        directory = cull_db._test_photo_dir
        selections = select_best_photos(cull_db, directory)
        save_selections(cull_db, directory, selections)

        # Find a selected photo
        selected_photo = next(p for p in selections if p["selected"])
        pid = selected_photo["id"]

        # Toggle off
        toggle_selection(cull_db, pid, False)
        loaded = load_selections(cull_db, directory)
        photo_row = next(r for r in loaded if r["photo_id"] == pid)
        assert photo_row["selected"] == 0

        # Toggle back on
        toggle_selection(cull_db, pid, True)
        loaded = load_selections(cull_db, directory)
        photo_row = next(r for r in loaded if r["photo_id"] == pid)
        assert photo_row["selected"] == 1

    def test_loaded_selections_have_photo_metadata(self, cull_db):
        """Loaded selections should include joined photo metadata."""
        directory = cull_db._test_photo_dir
        selections = select_best_photos(cull_db, directory)
        save_selections(cull_db, directory, selections)

        loaded = load_selections(cull_db, directory)
        for row in loaded:
            assert "filename" in row
            assert "aesthetic_score" in row
            assert "filepath" in row


# ---------------------------------------------------------------------------
# Date-range mode (review across multiple folders in a window)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy required for selection tests")
class TestDateRangeReview:
    """Date-range mode selects photos across every folder in a window and
    persists under a synthetic ``range:`` scope key."""

    def test_range_selects_all_photos_in_window(self, cull_db):
        """All 30 fixture photos fall on 2026-03-13, regardless of folder."""
        result = select_best_photos(
            cull_db, date_from="2026-03-13", date_to="2026-03-13"
        )
        assert len(result) == 30
        assert any(p["selected"] for p in result)

    def test_range_excludes_out_of_window(self, cull_db):
        """A window with no photos returns an empty selection."""
        result = select_best_photos(
            cull_db, date_from="2020-01-01", date_to="2020-12-31"
        )
        assert result == []

    def test_range_open_ended_bounds(self, cull_db):
        """Only a start date (open upper bound) still matches the window."""
        result = select_best_photos(cull_db, date_from="2026-01-01")
        assert len(result) == 30

    def test_range_scope_key_stored_verbatim(self, cull_db):
        """A ``range:`` key must not be path-resolved on save/load."""
        key = "range:2026-03-13..2026-03-13"
        selections = select_best_photos(
            cull_db, date_from="2026-03-13", date_to="2026-03-13"
        )
        save_selections(cull_db, key, selections)

        # Stored under the exact synthetic key.
        row = cull_db.conn.execute(
            "SELECT COUNT(*) AS c FROM review_selections WHERE directory = ?", (key,)
        ).fetchone()
        assert row["c"] == len(selections)

        loaded = load_selections(cull_db, key)
        assert loaded is not None
        saved_selected = {p["id"] for p in selections if p["selected"]}
        loaded_selected = {r["photo_id"] for r in loaded if r["selected"]}
        assert saved_selected == loaded_selected

    def test_directory_none_without_range_returns_empty(self, cull_db):
        """No directory and no date range → nothing to select."""
        assert select_best_photos(cull_db) == []

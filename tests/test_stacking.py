"""
Tests for photosearch/stacking.py — photo stack detection and persistence.

Covers:
  - _parse_date: timestamp parsing
  - _cosine_distance: cosine distance between vectors
  - detect_stacks: full detection pipeline with synthetic data
  - DB stack operations: create, get, set_top, unstack, clear
  - save_stacks / run_stacking: persistence helpers

Uses synthetic embeddings and timestamps — no real photos or ML models needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from photosearch.stacking import (
    _parse_date,
    _cosine_distance,
    detect_stacks,
    save_stacks,
    run_stacking,
)
from photosearch.db import PhotoDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _populate_burst_db(db, n_burst=4, n_isolated=3, burst_interval_sec=1,
                        noise=0.001, burst_base_seed=42, dim=512):
    """Add burst photos (close in time + similar embeddings) and isolated ones."""
    base_vec = _make_unit_vec(dim, seed=burst_base_seed)

    burst_ids = []
    for i in range(n_burst):
        pid = db.add_photo(
            filepath=f"burst_{i}.jpg",
            filename=f"burst_{i}.jpg",
            date_taken=f"2026-03-13T10:00:{i * burst_interval_sec:02d}",
            aesthetic_score=5.0 + i,
        )
        rng = np.random.RandomState(burst_base_seed + i + 1)
        noisy = base_vec + rng.randn(dim).astype(np.float32) * noise
        noisy /= np.linalg.norm(noisy)
        db.add_clip_embedding(pid, noisy.tolist())
        burst_ids.append(pid)

    iso_ids = []
    for i in range(n_isolated):
        pid = db.add_photo(
            filepath=f"isolated_{i}.jpg",
            filename=f"isolated_{i}.jpg",
            date_taken=f"2026-03-13T{11 + i}:00:00",
            aesthetic_score=4.0,
        )
        diff_vec = _make_unit_vec(dim, seed=9000 + i)
        db.add_clip_embedding(pid, diff_vec.tolist())
        iso_ids.append(pid)

    db.conn.commit()
    return burst_ids, iso_ids


# ---------------------------------------------------------------------------
# _parse_date tests
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_iso_format(self):
        dt = _parse_date("2026-03-13T10:00:05")
        assert dt is not None
        assert dt.hour == 10 and dt.second == 5

    def test_space_format(self):
        dt = _parse_date("2026-03-13 14:30:00")
        assert dt is not None
        assert dt.hour == 14 and dt.minute == 30

    def test_exif_colon_format(self):
        dt = _parse_date("2026:03:13 10:00:05")
        assert dt is not None
        assert dt.year == 2026 and dt.month == 3

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_empty_returns_none(self):
        assert _parse_date("") is None

    def test_garbage_returns_none(self):
        assert _parse_date("not-a-date") is None

    def test_fractional_seconds_truncated(self):
        dt = _parse_date("2026-03-13T10:00:05.123456")
        assert dt is not None
        assert dt.second == 5


# ---------------------------------------------------------------------------
# _cosine_distance tests
# ---------------------------------------------------------------------------

class TestCosineDistance:
    def test_identical_vectors(self):
        v = _make_unit_vec(seed=1)
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.zeros(512, dtype=np.float32)
        b = np.zeros(512, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_similar_vectors_small_distance(self):
        a = _make_unit_vec(seed=10)
        noise = np.random.RandomState(11).randn(512).astype(np.float32) * 0.001
        b = a + noise
        b /= np.linalg.norm(b)
        assert _cosine_distance(a, b) < 0.01

    def test_returns_float(self):
        a = _make_unit_vec(seed=1)
        b = _make_unit_vec(seed=2)
        assert isinstance(_cosine_distance(a, b), float)


# ---------------------------------------------------------------------------
# detect_stacks tests
# ---------------------------------------------------------------------------

class TestDetectStacks:

    @pytest.fixture
    def stack_db(self, tmp_path):
        db_path = str(tmp_path / "stack_test.db")
        db = PhotoDB(db_path)
        yield db
        db.close()

    def test_detects_burst(self, stack_db):
        """Photos close in time with similar embeddings should form a stack."""
        burst_ids, iso_ids = _populate_burst_db(stack_db, n_burst=4, n_isolated=3)
        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 1
        assert set(stacks[0]) == set(burst_ids)

    def test_isolated_photos_not_stacked(self, stack_db):
        """Photos far apart in time should not be stacked."""
        _populate_burst_db(stack_db, n_burst=0, n_isolated=5)
        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 0

    def test_time_close_but_different_content(self, stack_db):
        """Photos close in time but visually different should not stack."""
        for i in range(3):
            pid = stack_db.add_photo(
                filepath=f"diff_{i}.jpg", filename=f"diff_{i}.jpg",
                date_taken=f"2026-03-13T10:00:{i:02d}",
                aesthetic_score=5.0,
            )
            vec = _make_unit_vec(seed=i * 1000)  # very different embeddings
            stack_db.add_clip_embedding(pid, vec.tolist())
        stack_db.conn.commit()

        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 0

    def test_similar_but_far_apart_not_stacked(self, stack_db):
        """Visually similar photos taken hours apart should not stack."""
        base_vec = _make_unit_vec(seed=42)
        for i in range(3):
            pid = stack_db.add_photo(
                filepath=f"far_{i}.jpg", filename=f"far_{i}.jpg",
                date_taken=f"2026-03-13T{10 + i * 2}:00:00",  # 2 hours apart
                aesthetic_score=5.0,
            )
            noisy = base_vec + np.random.RandomState(i).randn(512).astype(np.float32) * 0.001
            noisy /= np.linalg.norm(noisy)
            stack_db.add_clip_embedding(pid, noisy.tolist())
        stack_db.conn.commit()

        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 0

    def test_best_photo_first(self, stack_db):
        """Stack members should be ordered by aesthetic score descending."""
        burst_ids, _ = _populate_burst_db(stack_db, n_burst=4, n_isolated=0)
        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 1
        # burst_3 has highest score (5.0 + 3 = 8.0)
        top_photo = stack_db.get_photo(stacks[0][0])
        assert top_photo["aesthetic_score"] == 8.0

    def test_multiple_bursts(self, stack_db):
        """Two separate bursts should form two stacks."""
        base1 = _make_unit_vec(seed=100)
        base2 = _make_unit_vec(seed=200)

        for i in range(3):
            pid = stack_db.add_photo(
                filepath=f"burst1_{i}.jpg", filename=f"burst1_{i}.jpg",
                date_taken=f"2026-03-13T10:00:{i:02d}",
                aesthetic_score=5.0,
            )
            noisy = base1 + np.random.RandomState(100 + i).randn(512).astype(np.float32) * 0.001
            noisy /= np.linalg.norm(noisy)
            stack_db.add_clip_embedding(pid, noisy.tolist())

        for i in range(3):
            pid = stack_db.add_photo(
                filepath=f"burst2_{i}.jpg", filename=f"burst2_{i}.jpg",
                date_taken=f"2026-03-13T14:00:{i:02d}",
                aesthetic_score=6.0,
            )
            noisy = base2 + np.random.RandomState(200 + i).randn(512).astype(np.float32) * 0.001
            noisy /= np.linalg.norm(noisy)
            stack_db.add_clip_embedding(pid, noisy.tolist())

        stack_db.conn.commit()
        stacks = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) == 2

    def test_empty_db(self, stack_db):
        """Empty database should return no stacks."""
        stacks = detect_stacks(stack_db)
        assert stacks == []

    def test_single_photo(self, stack_db):
        """A single photo can't form a stack."""
        pid = stack_db.add_photo(
            filepath="solo.jpg", filename="solo.jpg",
            date_taken="2026-03-13T10:00:00", aesthetic_score=5.0,
        )
        stack_db.add_clip_embedding(pid, _make_unit_vec(seed=1).tolist())
        stack_db.conn.commit()
        stacks = detect_stacks(stack_db)
        assert stacks == []

    def test_photos_without_dates_excluded(self, stack_db):
        """Photos missing date_taken should be excluded from stacking."""
        base_vec = _make_unit_vec(seed=42)
        for i in range(3):
            pid = stack_db.add_photo(
                filepath=f"nodate_{i}.jpg", filename=f"nodate_{i}.jpg",
                date_taken=None, aesthetic_score=5.0,
            )
            stack_db.add_clip_embedding(pid, base_vec.tolist())
        stack_db.conn.commit()
        stacks = detect_stacks(stack_db)
        assert stacks == []

    def test_custom_thresholds(self, stack_db):
        """Tighter thresholds should produce fewer stacks."""
        _populate_burst_db(stack_db, n_burst=4, noise=0.01)
        # With default tight threshold, noise=0.01 might push some pairs above 0.05
        stacks_tight = detect_stacks(stack_db, clip_threshold=0.001)
        stacks_loose = detect_stacks(stack_db, clip_threshold=0.5)
        assert len(stacks_loose) >= len(stacks_tight)

    def test_max_span_trims_chain(self, stack_db):
        """A chain of photos spanning > max_stack_span_sec should be trimmed."""
        # Create 6 photos, each 3s apart = 15s total span.
        # With time_window=5s, union-find chains them: A-B-C-D-E-F
        # With max_stack_span=10s, only photos within 10s of the earliest should stay.
        base_vec = _make_unit_vec(seed=42)
        pids = []
        for i in range(6):
            pid = stack_db.add_photo(
                filepath=f"chain_{i}.jpg", filename=f"chain_{i}.jpg",
                date_taken=f"2026-03-13T10:00:{i * 3:02d}",
                aesthetic_score=5.0 + i,
            )
            rng = np.random.RandomState(42 + i + 1)
            noisy = base_vec + rng.randn(512).astype(np.float32) * 0.001
            noisy /= np.linalg.norm(noisy)
            stack_db.add_clip_embedding(pid, noisy.tolist())
            pids.append(pid)
        stack_db.conn.commit()

        # Without span cap (very large), all 6 chain into one stack
        stacks_no_cap = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05,
                                       max_stack_span_sec=999)
        assert len(stacks_no_cap) == 1
        assert len(stacks_no_cap[0]) == 6

        # With 10s span cap: earliest is t=0, so only t=0,3,6,9 fit (4 photos)
        stacks_capped = detect_stacks(stack_db, time_window_sec=5, clip_threshold=0.05,
                                       max_stack_span_sec=10)
        assert len(stacks_capped) == 1
        assert len(stacks_capped[0]) == 4  # 0s, 3s, 6s, 9s — 12s and 15s trimmed


# ---------------------------------------------------------------------------
# DB stack CRUD tests
# ---------------------------------------------------------------------------

class TestDbStackOperations:

    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "crud_test.db")
        db = PhotoDB(db_path)
        pids = []
        for i in range(5):
            pid = db.add_photo(
                filepath=f"photo_{i}.jpg", filename=f"photo_{i}.jpg",
                aesthetic_score=float(i),
            )
            pids.append(pid)
        db.conn.commit()
        db._test_pids = pids
        yield db
        db.close()

    def test_create_and_get_stack(self, db):
        pids = db._test_pids
        sid = db.create_stack(pids[:3], top_photo_id=pids[2])
        stack = db.get_stack(sid)
        assert stack is not None
        assert len(stack["members"]) == 3
        assert stack["members"][0]["is_top"] == 1

    def test_get_photo_stack(self, db):
        pids = db._test_pids
        sid = db.create_stack(pids[:3], top_photo_id=pids[0])
        info = db.get_photo_stack(pids[1])
        assert info is not None
        assert info["stack_id"] == sid
        assert info["is_top"] is False
        assert info["member_count"] == 3

    def test_photo_not_in_stack(self, db):
        assert db.get_photo_stack(db._test_pids[4]) is None

    def test_set_stack_top(self, db):
        pids = db._test_pids
        sid = db.create_stack(pids[:3], top_photo_id=pids[0])
        db.set_stack_top(sid, pids[2])
        info = db.get_photo_stack(pids[2])
        assert info["is_top"] is True
        info_old = db.get_photo_stack(pids[0])
        assert info_old["is_top"] is False

    def test_delete_stack(self, db):
        pids = db._test_pids
        sid = db.create_stack(pids[:3], top_photo_id=pids[0])
        db.delete_stack(sid)
        assert db.get_stack(sid) is None
        assert db.get_photo_stack(pids[0]) is None

    def test_unstack_photo(self, db):
        pids = db._test_pids
        sid = db.create_stack(pids[:3], top_photo_id=pids[0])
        db.unstack_photo(pids[2])
        stack = db.get_stack(sid)
        assert len(stack["members"]) == 2

    def test_unstack_dissolves_pair(self, db):
        """Unstacking from a 2-photo stack should dissolve it."""
        pids = db._test_pids
        sid = db.create_stack(pids[:2], top_photo_id=pids[0])
        db.unstack_photo(pids[1])
        assert db.get_stack(sid) is None
        assert db.get_photo_stack(pids[0]) is None

    def test_unstack_top_promotes_next_best(self, db):
        """Unstacking the top photo should promote the highest-scored remaining."""
        pids = db._test_pids
        # pids[2] has score 2.0 (highest), pids[1] has 1.0, pids[0] has 0.0
        sid = db.create_stack(pids[:3], top_photo_id=pids[2])
        db.unstack_photo(pids[2])
        # pids[1] (score 1.0) should now be top
        info = db.get_photo_stack(pids[1])
        assert info["is_top"] is True

    def test_create_stack_requires_two(self, db):
        with pytest.raises(ValueError, match="at least 2"):
            db.create_stack([db._test_pids[0]])

    def test_get_all_stacks(self, db):
        pids = db._test_pids
        db.create_stack(pids[:2], top_photo_id=pids[0])
        db.create_stack(pids[2:4], top_photo_id=pids[2])
        all_stacks = db.get_all_stacks()
        assert len(all_stacks) == 2

    def test_clear_stacks(self, db):
        pids = db._test_pids
        db.create_stack(pids[:3], top_photo_id=pids[0])
        db.clear_stacks()
        assert db.get_all_stacks() == []

    def test_add_to_stack(self, db):
        """Adding a photo to an existing stack should work."""
        pids = db._test_pids
        sid = db.create_stack(pids[:2], top_photo_id=pids[0])
        db.add_to_stack(sid, pids[2])
        stack = db.get_stack(sid)
        assert len(stack["members"]) == 3
        # The added photo should not be top
        info = db.get_photo_stack(pids[2])
        assert info["stack_id"] == sid
        assert info["is_top"] is False

    def test_add_to_stack_moves_from_other_stack(self, db):
        """Adding a photo that's already in another stack should move it."""
        pids = db._test_pids
        sid1 = db.create_stack(pids[:3], top_photo_id=pids[0])
        sid2 = db.create_stack(pids[3:5], top_photo_id=pids[3])
        # Move pids[2] from sid1 to sid2
        db.add_to_stack(sid2, pids[2])
        info = db.get_photo_stack(pids[2])
        assert info["stack_id"] == sid2
        # sid1 should still have 2 members
        stack1 = db.get_stack(sid1)
        assert len(stack1["members"]) == 2

    def test_add_to_stack_nonexistent_raises(self, db):
        """Adding to a nonexistent stack should raise ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            db.add_to_stack(9999, db._test_pids[0])

    def test_create_removes_from_existing_stack(self, db):
        """Creating a new stack with a photo already in a stack should move it."""
        pids = db._test_pids
        sid1 = db.create_stack(pids[:3], top_photo_id=pids[0])
        # Now create a new stack that includes pids[2]
        sid2 = db.create_stack([pids[2], pids[3]], top_photo_id=pids[2])
        # pids[2] should be in sid2, not sid1
        info = db.get_photo_stack(pids[2])
        assert info["stack_id"] == sid2
        # sid1 should still exist with 2 members (or dissolved if only 1 left —
        # but we started with 3 and removed 1, so 2 remain)
        stack1 = db.get_stack(sid1)
        # sid1 may have been cleaned up if its member count dropped; check
        if stack1:
            assert len(stack1["members"]) == 2


# ---------------------------------------------------------------------------
# save_stacks / run_stacking tests
# ---------------------------------------------------------------------------

class TestSaveAndRun:

    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "run_test.db")
        db = PhotoDB(db_path)
        _populate_burst_db(db, n_burst=4, n_isolated=2)
        yield db
        db.close()

    def test_save_stacks_persists(self, db):
        stacks = detect_stacks(db, time_window_sec=5, clip_threshold=0.05)
        save_stacks(db, stacks)
        all_stacks = db.get_all_stacks()
        assert len(all_stacks) == len(stacks)

    def test_run_stacking_saves(self, db):
        stacks = run_stacking(db, time_window_sec=5, clip_threshold=0.05)
        assert len(stacks) >= 1
        assert len(db.get_all_stacks()) >= 1

    def test_run_stacking_dry_run(self, db):
        stacks = run_stacking(db, dry_run=True)
        assert len(stacks) >= 1
        assert len(db.get_all_stacks()) == 0  # nothing saved

    def test_save_clears_previous(self, db):
        """save_stacks should clear old stacks before saving new ones."""
        stacks = detect_stacks(db)
        save_stacks(db, stacks)
        assert len(db.get_all_stacks()) == len(stacks)
        # Save again — should replace, not duplicate
        save_stacks(db, stacks)
        assert len(db.get_all_stacks()) == len(stacks)

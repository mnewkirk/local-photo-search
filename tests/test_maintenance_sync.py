"""Tests for the maintenance sync layer (replica -> NAS push).

Covers the fingerprint, the stage push-mode taxonomy, payload collection,
the timestamp comparison rules, and the NAS-side apply path.
"""

import pytest

from photosearch.maintenance_sync import (
    EXCLUDED_STAGES,
    TRANSFER_STAGES,
    TRIGGER_STAGES,
    fingerprints_match,
    photo_fingerprint,
    push_mode,
)


def test_photo_fingerprint_counts_and_max_id(db):
    fp = photo_fingerprint(db)
    expected_n = db.conn.execute("SELECT COUNT(*) AS n FROM photos").fetchone()["n"]
    expected_max = db.conn.execute("SELECT MAX(id) AS m FROM photos").fetchone()["m"]
    assert fp == {"photo_count": expected_n, "photo_max_id": expected_max}


def test_photo_fingerprint_on_empty_db(tmp_db_path):
    from photosearch.db import PhotoDB
    with PhotoDB(tmp_db_path) as empty:
        assert photo_fingerprint(empty) == {"photo_count": 0, "photo_max_id": None}


def test_fingerprints_match_is_exact():
    a = {"photo_count": 10, "photo_max_id": 99}
    assert fingerprints_match(a, {"photo_count": 10, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 11, "photo_max_id": 99})
    assert not fingerprints_match(a, {"photo_count": 10, "photo_max_id": 100})


@pytest.mark.parametrize("stage", sorted(TRIGGER_STAGES))
def test_trigger_stages(stage):
    assert push_mode(stage) == "trigger"


def test_stacking_is_the_only_transfer_stage():
    assert TRANSFER_STAGES == frozenset({"stacking"})
    assert push_mode("stacking") == "transfer"


@pytest.mark.parametrize("stage", ["colors", "dedup_photos", "match_faces",
                                   "recluster", "requeue"])
def test_excluded_stages(stage):
    assert push_mode(stage) == "excluded"


def test_taxonomy_covers_every_sweep_stage_exactly_once():
    """Every stage the sweep can emit must have exactly one push mode.

    Guards against a new stage being added to maintenance.py without a
    reconciliation decision — which would silently mean 'lost on next sync'.
    """
    from photosearch.maintenance import SWEEP_STAGE_ORDER
    known = TRIGGER_STAGES | TRANSFER_STAGES | EXCLUDED_STAGES
    assert set(SWEEP_STAGE_ORDER) - known == set(), "sweep stage with no push mode"
    assert not (TRIGGER_STAGES & TRANSFER_STAGES)
    assert not (TRIGGER_STAGES & EXCLUDED_STAGES)
    assert not (TRANSFER_STAGES & EXCLUDED_STAGES)


def test_push_mode_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown maintenance stage"):
        push_mode("not_a_stage")


# ---------------------------------------------------------------------------
# The sweep stamps watermarks
# ---------------------------------------------------------------------------

def test_sweep_stamps_watermark_only_for_done_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(db, apply=True, do_colors=False,
                                   do_stacking=False, do_match=False,
                                   source="replica")
    runs = db.get_maintenance_runs()
    done = {s["stage"] for s in result["stages"] if s["status"] == "done"}
    skipped = {s["stage"] for s in result["stages"] if s["status"] != "done"}

    assert done, "expected at least one stage to run against the fixture"
    assert done <= set(runs), "every done stage must be stamped"
    assert not (skipped & set(runs)), "skipped stages must not be stamped"
    for stage in done:
        assert runs[stage]["source"] == "replica"
        assert runs[stage]["last_run_at"]


def test_dry_run_stamps_nothing(db):
    from photosearch.maintenance import run_maintenance_sweep

    run_maintenance_sweep(db, apply=False, do_colors=False,
                          do_stacking=False, do_match=False)
    assert db.get_maintenance_runs() == {}


def test_stages_subset_runs_only_those_stages(db):
    from photosearch.maintenance import run_maintenance_sweep

    result = run_maintenance_sweep(
        db, apply=True, stages=["normalize_aesthetics"],
    )
    assert [s["stage"] for s in result["stages"]] == ["normalize_aesthetics"]


def test_unknown_stage_subset_is_rejected(db):
    from photosearch.maintenance import run_maintenance_sweep

    with pytest.raises(ValueError, match="unknown maintenance stage"):
        run_maintenance_sweep(db, apply=True, stages=["bogus_stage"])

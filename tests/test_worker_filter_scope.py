"""Filter-scoped worker claims — the structured `filters` scope on
/api/worker/claim-batch (and the shared _resolve_scope_ids helper).

A filter set is a third way to produce the claim scope alongside
collection_id/directory; it must resolve to the same photo-id set that an
equivalent collection would, and claim-batch must only hand out in-scope photos.
"""

import pytest
from fastapi import HTTPException

from photosearch.worker_api import _resolve_scope_ids


def test_resolve_scope_camera_matches_all(db):
    ids = _resolve_scope_ids(db, None, None, {"camera": "ILCE-7M4"})
    assert set(ids) == set(db._test_photo_ids.values())


def test_resolve_scope_date_range_narrows(db):
    # Sample photos are all 2026-03-13; the upper bound is exclusive-of-day via
    # the appended time, so use the following day to include them.
    ids = _resolve_scope_ids(
        db, None, None, {"date_from": "2026-03-13", "date_to": "2026-03-14"})
    assert set(ids) == set(db._test_photo_ids.values())
    # A window before any photo → 404 (no match), not an empty whole-library scope.
    with pytest.raises(HTTPException) as ei:
        _resolve_scope_ids(db, None, None,
                           {"date_from": "2020-01-01", "date_to": "2020-01-02"})
    assert ei.value.status_code == 404


def test_resolve_scope_person(db):
    # Jamie has faces only on DSC04907 + DSC04922.
    ids = _resolve_scope_ids(db, None, None, {"people": ["Jamie"]})
    expected = {db._test_photo_ids["DSC04907.JPG"],
                db._test_photo_ids["DSC04922.JPG"]}
    assert set(ids) == expected


def test_person_filter_equals_equivalent_collection(db):
    """The 'Best of March' collection is exactly {DSC04907, DSC04922} — the same
    photos Jamie appears in. A person filter must resolve to the same scope."""
    coll = _resolve_scope_ids(db, db._test_collection_id, None, None)
    filt = _resolve_scope_ids(db, None, None, {"people": ["Jamie"]})
    assert set(coll) == set(filt)


def test_empty_filters_is_whole_library(db):
    assert _resolve_scope_ids(db, None, None, {}) is None
    assert _resolve_scope_ids(db, None, None, None) is None


def test_claim_batch_respects_filter_scope(client, db):
    # Defensive reset of the drain flag: a prior TestClient teardown in the suite
    # can leak worker_api._shutting_down=True, which would 503 claim-batch. See
    # docs/plans/test-isolation-fixes.md (same root cause the race test skips for).
    from photosearch import worker_api
    worker_api._shutting_down = False

    # quality pass: aesthetic_concepts is NULL for every sample photo, so all are
    # unprocessed. Scope to Jamie → only her two photos are claimable.
    jamie = {db._test_photo_ids["DSC04907.JPG"],
             db._test_photo_ids["DSC04922.JPG"]}
    r = client.post("/api/worker/claim-batch", json={
        "worker_id": "filt-worker",
        "pass_type": "quality",
        "limit": 10,
        "filters": {"people": ["Jamie"]},
    })
    assert r.status_code == 200, r.text
    got = {p["id"] for p in (r.json().get("photos") or [])}
    assert got == jamie  # both are unprocessed and in scope, so both claimed

    # An unknown person resolves to zero photos → 404 (not a whole-library claim).
    r2 = client.post("/api/worker/claim-batch", json={
        "worker_id": "filt-worker",
        "pass_type": "quality",
        "filters": {"people": ["Nobody McGhost"]},
    })
    assert r2.status_code == 404

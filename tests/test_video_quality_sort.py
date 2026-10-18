"""Video quality surfacing (docs/plans/video-support.md step 7, D11).

Shake / noise / focus_consistency / exposure are always stored and shown as
plain NUMBERS — `sort=steadiest` / `sort=least_noisy` rank videos by them,
ascending, NULLs (unmeasured) last. Picking either sort implies media=video.
No tag is ever written from these values (D11: a tag needs a holdout eval
that has not run).
"""

import json

import pytest
from fastapi.testclient import TestClient

from photosearch import web
from photosearch.db import PhotoDB
from photosearch.search import search_combined


def _video(db, name, date, shake=None, noise=None, focus_consistency=None,
          quality_json=None, camera=None):
    vid = db.add_photo(filepath=f"2026/v/{name}", filename=name,
                       date_taken=date, media_type="video", camera_model=camera)
    db.conn.execute(
        "INSERT INTO video_meta (photo_id, probe_version, kind, duration_s, "
        "scan_version, shake, noise, focus_consistency, quality_json) "
        "VALUES (?, 1, 'clip', 10.0, 1, ?, ?, ?, ?)",
        (vid, shake, noise, focus_consistency,
         json.dumps(quality_json) if quality_json is not None else None))
    return vid


@pytest.fixture
def quality_lib(db):
    """Four videos with a shake/noise spread (including NULLs) plus a still,
    so "implies media=video" and "NULLs last" both have something to prove
    against."""
    still = db.add_photo(filepath="2026/v/still.jpg", filename="still.jpg",
                         date_taken="2026-09-26 09:00:00")
    shaky = _video(db, "SHAKY.MP4", "2026-09-26 10:00:00", shake=4.0, noise=10.0,
                   focus_consistency=0.4,
                   quality_json={"exposure": {"highlight_frac": 0.2, "shadow_frac": 0.01}})
    steady = _video(db, "STEADY.MP4", "2026-09-26 11:00:00", shake=0.2, noise=1.5,
                    focus_consistency=0.95,
                    quality_json={"exposure": {"highlight_frac": 0.0, "shadow_frac": 0.0}})
    mid = _video(db, "MID.MP4", "2026-09-26 12:00:00", shake=1.0, noise=5.0,
                focus_consistency=0.7)
    unmeasured = _video(db, "RAW.MP4", "2026-09-26 13:00:00")  # no scan yet
    db.conn.commit()
    return {"db": db, "still": still, "shaky": shaky, "steady": steady,
            "mid": mid, "unmeasured": unmeasured}


# ---------------------------------------------------------------------------
# Sort order + NULLs last + implied video media
# ---------------------------------------------------------------------------

def test_steadiest_orders_ascending_shake_nulls_last(quality_lib):
    db = quality_lib["db"]
    results = search_combined(db, sort="steadiest", limit=100)
    ids = [r["id"] for r in results]
    assert ids == [quality_lib["steady"], quality_lib["mid"], quality_lib["shaky"],
                   quality_lib["unmeasured"]]
    # The still never appears — the sort implies media=video.
    assert quality_lib["still"] not in ids
    # Shake values are annotated for display/re-sort.
    by_id = {r["id"]: r for r in results}
    assert by_id[quality_lib["steady"]]["shake"] == pytest.approx(0.2)
    assert "shake" not in by_id[quality_lib["unmeasured"]]


def test_least_noisy_orders_ascending_noise_nulls_last(quality_lib):
    db = quality_lib["db"]
    results = search_combined(db, sort="least_noisy", limit=100)
    ids = [r["id"] for r in results]
    assert ids == [quality_lib["steady"], quality_lib["mid"], quality_lib["shaky"],
                   quality_lib["unmeasured"]]
    assert quality_lib["still"] not in ids


def test_steadiest_implies_media_video_even_with_explicit_photo(quality_lib):
    """Picking the sort overrides an explicit (contradictory) media=photo —
    the plan's "implies media=video", not merely "defaults to"."""
    db = quality_lib["db"]
    results = search_combined(db, sort="steadiest", media="photo", limit=100)
    ids = {r["id"] for r in results}
    assert quality_lib["still"] not in ids
    assert ids == {quality_lib["shaky"], quality_lib["steady"], quality_lib["mid"],
                   quality_lib["unmeasured"]}


def test_steadiest_combines_with_other_filters(quality_lib):
    db = quality_lib["db"]
    db.conn.execute("UPDATE photos SET camera_model = 'VCAM' WHERE id IN (?, ?)",
                    (quality_lib["shaky"], quality_lib["steady"]))
    db.conn.commit()
    results = search_combined(db, sort="steadiest", camera="VCAM", limit=100)
    ids = [r["id"] for r in results]
    assert ids == [quality_lib["steady"], quality_lib["shaky"]]


# ---------------------------------------------------------------------------
# Detail API returns the quality block
# ---------------------------------------------------------------------------

@pytest.fixture
def http_env(quality_lib, monkeypatch, tmp_db_path):
    monkeypatch.setattr(web, "_db_path", tmp_db_path)
    monkeypatch.setattr(web, "_nas_url", None)
    return TestClient(web.app)


def test_detail_api_returns_video_quality_block(quality_lib, http_env):
    c = http_env
    d = c.get(f"/api/photos/{quality_lib['shaky']}").json()
    q = d["video"]["quality"]
    assert q["shake"] == pytest.approx(4.0)
    assert q["noise"] == pytest.approx(10.0)
    assert q["focus_consistency"] == pytest.approx(0.4)
    assert q["exposure"]["highlight_frac"] == pytest.approx(0.2)
    assert q["scan_error"] is None
    # quality_json itself is not duplicated at the top of the video block.
    assert "quality_json" not in d["video"]


def test_detail_api_quality_block_for_unmeasured_video(quality_lib, http_env):
    c = http_env
    d = c.get(f"/api/photos/{quality_lib['unmeasured']}").json()
    q = d["video"]["quality"]
    assert q["shake"] is None and q["noise"] is None
    assert q["focus_consistency"] is None
    assert q["exposure"] is None
    assert q["scan_error"] is None


def test_detail_api_quality_block_reports_scan_error(quality_lib, http_env):
    db = quality_lib["db"]
    db.conn.execute(
        "UPDATE video_meta SET scan_error = 'decode failed' WHERE photo_id = ?",
        (quality_lib["unmeasured"],))
    db.conn.commit()
    d = http_env.get(f"/api/photos/{quality_lib['unmeasured']}").json()
    assert d["video"]["quality"]["scan_error"] == "decode failed"


def test_detail_api_still_has_no_video_block(quality_lib, http_env):
    d = http_env.get(f"/api/photos/{quality_lib['still']}").json()
    assert d["video"] is None


# ---------------------------------------------------------------------------
# No tag is ever written from these values
# ---------------------------------------------------------------------------

def test_no_visual_tags_written_for_videos(quality_lib, http_env):
    """Step 7 is read-only surfacing: searching/sorting/fetching a video's
    quality numbers must never touch its visual_tags (D11 — a tag needs a
    holdout eval that hasn't run)."""
    db = quality_lib["db"]
    before = {
        r["id"]: r["visual_tags"]
        for r in db.conn.execute("SELECT id, visual_tags FROM photos WHERE media_type = 'video'")
    }
    assert all(v is None for v in before.values())  # sanity: nothing set yet

    search_combined(db, sort="steadiest", limit=100)
    search_combined(db, sort="least_noisy", limit=100)
    http_env.get(f"/api/photos/{quality_lib['shaky']}")
    http_env.get("/api/search?media=video&sort=steadiest")

    after = {
        r["id"]: r["visual_tags"]
        for r in db.conn.execute("SELECT id, visual_tags FROM photos WHERE media_type = 'video'")
    }
    assert after == before


# ---------------------------------------------------------------------------
# sort= validation
# ---------------------------------------------------------------------------

def test_api_search_accepts_steadiest_with_no_other_criteria(quality_lib, http_env):
    r = http_env.get("/api/search?sort=steadiest")
    assert r.status_code == 200
    body = r.json()
    assert "error" not in body
    ids = [p["id"] for p in body["results"]]
    assert quality_lib["still"] not in ids
    assert ids[0] == quality_lib["steady"]


def test_sort_modes_includes_video_quality_sorts():
    from photosearch.search import SORT_MODES
    assert "steadiest" in SORT_MODES and "least_noisy" in SORT_MODES

"""Tests for the photobook authoring scene-capping (segment_pool budget)."""
from collections import Counter

from photosearch.book_authoring import _cap_scenes


def _scenes(day: str, n: int, photos_each: int = 10, start_hour: int = 8):
    out = []
    for i in range(n):
        out.append({
            "gindex": 0,  # reassigned by _cap_scenes
            "day": day,
            "place": f"{day}-p{i}",
            "start": f"{day}T{start_hour + i % 12:02d}:{i % 60:02d}:00",
            "photo_count": photos_each,
        })
    return out


def test_cap_is_noop_under_limit():
    scenes = _scenes("2026-07-13", 5)
    assert _cap_scenes(list(scenes), 80) == scenes


def test_late_dense_day_is_not_starved():
    # Early day over-segments into many tiny burst scenes; the big day comes last.
    scenes = (
        _scenes("2026-07-12", 8, photos_each=3)
        + _scenes("2026-07-13", 22, photos_each=14)
        + _scenes("2026-07-14", 44, photos_each=6)   # bursty afternoon, eats the budget naively
        + _scenes("2026-07-15", 60, photos_each=14)  # the 851-photo big day, last
        + _scenes("2026-07-16", 4, photos_each=4)
    )
    kept = _cap_scenes(scenes, 80)
    assert len(kept) <= 80
    per_day = Counter(s["day"] for s in kept)
    # Chronological truncation would give 07-15 == 0; proportional must keep plenty.
    assert per_day["2026-07-15"] >= 20, per_day
    # Every day survives with at least one scene (floor).
    assert set(per_day) == {"2026-07-12", "2026-07-13", "2026-07-14", "2026-07-15", "2026-07-16"}
    # The densest day gets the largest share.
    assert per_day["2026-07-15"] == max(per_day.values())


def test_kept_scenes_are_chronological_and_reindexed():
    scenes = _scenes("2026-07-14", 44, photos_each=6) + _scenes("2026-07-15", 44, photos_each=14)
    kept = _cap_scenes(scenes, 40)
    assert [s["gindex"] for s in kept] == list(range(len(kept)))
    keys = [(s["day"], s["start"]) for s in kept]
    assert keys == sorted(keys)

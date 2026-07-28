"""Tests for the photobook authoring scene-capping (segment_pool budget)."""
import sqlite3
from collections import Counter

import pytest

from photosearch.book_authoring import _cap_scenes, _res_factor, pick_heroes


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


# --- print-resolution guard on hero selection --------------------------------

def test_res_factor_bands():
    assert _res_factor(9504, 6336) == 1.0    # a7R VI full-frame, crisp full-page
    assert _res_factor(6528, 4352) == 0.9    # a7R APS-C crop, printable not crisp
    assert _res_factor(5712, 4284) == 0.9    # 24 MP phone
    assert _res_factor(3648, 2736) == 0.6    # 10 MP phone, soft at full page
    assert _res_factor(0, 0) == 0.6          # unknown dims → treat as low-res


class _FakePDB:
    def __init__(self, conn): self.conn = conn


def _pdb_with(photos):
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, image_width INT, "
                 "image_height INT, subject_boxes TEXT)")
    conn.executemany("INSERT INTO photos VALUES (?,?,?,NULL)", photos)
    return _FakePDB(conn)


def test_hero_pick_prefers_higher_res_without_vlm(monkeypatch):
    # No VLM configured → hero slot should go to the sharper-in-print frame, not a
    # low-res phone snapshot that happens to be first in the candidate list.
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    pdb = _pdb_with([
        (1, 3648, 2736),   # 10 MP phone — soft at full page (listed first)
        (2, 9504, 6336),   # 60 MP a7R full-frame — crisp
        (3, 5712, 4284),   # 24 MP phone — printable
    ])
    res = pick_heroes(pdb, [1, 2, 3], "Seceda ridge", n_heroes=1)
    assert res["heroes"] == [2]                       # highest-res wins the hero
    assert [r["id"] for r in res["ranked"]] == [2, 3, 1]
    assert all("print_ppi" in r for r in res["ranked"])


def test_hero_pick_keeps_only_low_res_option(monkeypatch):
    # A beat whose ONLY candidate is low-res still gets it (penalty, not exclusion).
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.delenv("PHOTOSEARCH_LLM_VISUAL_MODEL", raising=False)
    pdb = _pdb_with([(7, 3648, 2736)])
    res = pick_heroes(pdb, [7], "one soft photo", n_heroes=1)
    assert res["heroes"] == [7]

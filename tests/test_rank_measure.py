"""Tests for photosearch/rank_measure.py — the native-resolution face-sharpness
pass lifted out of scripts/rank_shoot.py.

The measurement itself decodes a full-resolution JPEG with PIL and runs a cv2
Laplacian over every face crop, so **nothing here touches a real image**: the
per-photo worker is injected (`measure_photo=`). What is pinned is everything
around it — the scope, the cache file's location, the cache's on-disk shape,
and the resumability that makes a re-run after a late-arriving photo cheap.

Two facts this module exists to keep true:

- **There is ONE implementation.** `scripts/rank_shoot.py` imports `measure`
  from here rather than carrying its own copy, because the script is not in
  the Docker image (see the Dockerfile's COPY lines) and the batch runner has
  to be able to import it.
- **The cache path follows the DB's directory.** The script used to hardcode
  `/data/rank_shoot_<date>.json`; on the NAS the DB *is* `/data/photo_index.db`,
  so the rule resolves to exactly the same file and the owner's existing cache
  is not stranded — while off-NAS it no longer points at a directory that
  doesn't exist.

Per globals.md the shared `db` fixture is pre-seeded under `2026/`, so every
fixture here builds photos under `2092/`.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from photosearch import rank_measure


_DIR = "2092/2092-04-02_ILCE-7RM6"
_DAY = "2092-04-02"


def _photo_with_faces(db, i, *, directory=_DIR, day=_DAY, faces=1):
    pid = db.add_photo(
        filepath=f"{directory}/img{i:03d}.jpg",
        filename=f"img{i:03d}.jpg",
        date_taken=f"{day}T10:0{i}:00",
    )
    for n in range(faces):
        db.add_face(pid, (10, 200, 200, 10), [0.0] * 512)
    return pid


def _fake_measure(calls, per_face=None):
    """Stand-in for the PIL/cv2 worker: records the photos it was asked for."""
    def measure_photo(path, faces, min_edge):
        calls.append(path)
        return {str(f["face_id"]): dict(per_face or
                                        {"lap": 123.5, "edge": 190,
                                         "area_frac": 0.01})
                for f in faces}
    return measure_photo


# =========================================================================
# default_cache_path
# =========================================================================

class TestDefaultCachePath:
    def test_follows_the_db_directory(self):
        assert rank_measure.default_cache_path(
            "/data/photo_index.db", "2026-09-19") == "/data/rank_shoot_2026-09-19.json"

    def test_does_not_strand_the_existing_nas_cache(self):
        """The script's old hardcoded default. On the NAS the DB lives in
        /data, so the new rule has to resolve to the identical filename or a
        measured shoot would silently re-measure from scratch."""
        assert rank_measure.default_cache_path("/data/photo_index.db", "2026-09-19") \
            == "/data/rank_shoot_2026-09-19.json"

    def test_relative_db_path_does_not_land_in_the_cwd_by_accident(self, tmp_path):
        p = rank_measure.default_cache_path(str(tmp_path / "photo_index.db"), _DAY)
        assert p == str(tmp_path / f"rank_shoot_{_DAY}.json")
        assert os.path.isabs(p)


# =========================================================================
# measure — scope, cache shape, resumability
# =========================================================================

class TestMeasure:
    def test_measures_every_photo_with_faces_on_the_date(self, db, tmp_path):
        for i in range(3):
            _photo_with_faces(db, i)
        cache_path = str(tmp_path / "cache.json")
        calls = []

        out = rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                   measure_photo=_fake_measure(calls))

        assert len(calls) == 3
        assert out["measured"] == 3
        assert out["cache_path"] == cache_path

    def test_cache_format_is_photo_id_to_face_id_to_metrics(self, db, tmp_path):
        pid = _photo_with_faces(db, 0, faces=2)
        cache_path = str(tmp_path / "cache.json")

        rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                             measure_photo=_fake_measure([]))

        cache = json.loads(Path(cache_path).read_text())
        assert list(cache) == [str(pid)]
        entry = cache[str(pid)]
        assert len(entry) == 2
        for face_id, m in entry.items():
            assert face_id.isdigit()
            assert set(m) == {"lap", "edge", "area_frac"}

    def test_is_resumable_a_second_run_measures_nothing(self, db, tmp_path):
        for i in range(3):
            _photo_with_faces(db, i)
        cache_path = str(tmp_path / "cache.json")
        first, second = [], []

        rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                             measure_photo=_fake_measure(first))
        out = rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                   measure_photo=_fake_measure(second))

        assert len(first) == 3
        assert second == []
        assert out["measured"] == 0
        assert out["photos"] == 3

    def test_photo_ids_scope_wins_over_the_date(self, db, tmp_path):
        """The batch runner measures ONE folder. Two folders can share a day
        (a phone sync and a card dump), so the scope is the batch's ids."""
        mine = [_photo_with_faces(db, i) for i in range(2)]
        _photo_with_faces(db, 7, directory="2092/2092-04-02_phone-matt")
        cache_path = str(tmp_path / "cache.json")

        rank_measure.measure(db, _DAY, cache_path, photo_ids=mine,
                             log=lambda m: None, measure_photo=_fake_measure([]))

        assert sorted(json.loads(Path(cache_path).read_text())) == \
            sorted(str(p) for p in mine)

    def test_an_empty_scope_never_falls_through_to_the_whole_library(self, db, tmp_path):
        """`photo_ids=[]` must mean "no photos", not "no scope given" — the
        same trap `_run_stacking` guards (an empty list falling through to the
        whole library is how the library's stacks have been wiped twice)."""
        _photo_with_faces(db, 0)
        cache_path = str(tmp_path / "cache.json")
        calls = []

        out = rank_measure.measure(db, _DAY, cache_path, photo_ids=[],
                                   log=lambda m: None,
                                   measure_photo=_fake_measure(calls))

        assert calls == []
        assert out["measured"] == 0

    def test_an_unreadable_photo_is_recorded_empty_not_fatal(self, db, tmp_path):
        pid = _photo_with_faces(db, 0)
        _photo_with_faces(db, 1)
        cache_path = str(tmp_path / "cache.json")

        def boom(path, faces, min_edge):
            if f"img000" in path:
                raise OSError("cannot identify image file")
            return {}

        out = rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                   measure_photo=boom)

        cache = json.loads(Path(cache_path).read_text())
        assert cache[str(pid)] == {}
        assert out["measured"] == 2

    def test_resolves_the_filepath_through_the_photo_root(self, db, tmp_path):
        _photo_with_faces(db, 0)
        calls = []
        rank_measure.measure(db, _DAY, str(tmp_path / "c.json"),
                             log=lambda m: None, measure_photo=_fake_measure(calls))
        assert calls == [f"/photos/{_DIR}/img000.jpg"]


# =========================================================================
# scripts/rank_shoot.py — one implementation, unchanged CLI
# =========================================================================

def _load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "rank_shoot.py"
    spec = importlib.util.spec_from_file_location("rank_shoot_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRankShootScript:
    def test_the_script_uses_the_packaged_measure(self):
        """One implementation. The script is NOT in the Docker image, so the
        measurement has to live in the package for the batch runner to reach
        it — and the script must not keep a second copy that can drift."""
        mod = _load_script()
        assert mod.measure is rank_measure.measure

    def test_cli_still_parses_the_owners_command(self):
        mod = _load_script()
        args = mod.build_parser().parse_args(
            ["--date", "2026-09-19", "--measure"])
        assert args.date == "2026-09-19"
        assert args.measure is True
        assert args.cache is None

    def test_cache_default_follows_the_db(self):
        mod = _load_script()
        args = mod.build_parser().parse_args(["--date", "2026-09-19"])
        assert mod.cache_path_for(args, "/data/photo_index.db") \
            == "/data/rank_shoot_2026-09-19.json"

    def test_an_explicit_cache_flag_still_wins(self):
        mod = _load_script()
        args = mod.build_parser().parse_args(
            ["--date", "2026-09-19", "--cache", "/tmp/mine.json"])
        assert mod.cache_path_for(args, "/data/photo_index.db") == "/tmp/mine.json"

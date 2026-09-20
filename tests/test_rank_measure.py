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

    def test_two_batches_sharing_a_date_merge_into_one_cache(self, db, tmp_path):
        """Two folders can share a day (a phone sync and a card dump). The
        second batch's run must ADD to the date's cache, never clobber the
        first's — the selection phase reads the whole day."""
        a = [_photo_with_faces(db, i) for i in range(2)]
        b = [_photo_with_faces(db, 7, directory="2092/2092-04-02_phone-matt")]
        cache_path = str(tmp_path / "cache.json")

        rank_measure.measure(db, _DAY, cache_path, photo_ids=a,
                             log=lambda m: None, measure_photo=_fake_measure([]))
        rank_measure.measure(db, _DAY, cache_path, photo_ids=b,
                             log=lambda m: None, measure_photo=_fake_measure([]))

        assert sorted(json.loads(Path(cache_path).read_text())) == \
            sorted(str(p) for p in a + b)

    def test_resolves_the_filepath_through_the_photo_root(self, db, tmp_path):
        _photo_with_faces(db, 0)
        calls = []
        rank_measure.measure(db, _DAY, str(tmp_path / "c.json"),
                             log=lambda m: None, measure_photo=_fake_measure(calls))
        assert calls == [f"/photos/{_DIR}/img000.jpg"]


# =========================================================================
# abort — it is the LAST and LONGEST step of a batch advance
# =========================================================================

class TestAbort:
    """`advance_nas_steps` only checks abort BETWEEN steps, and this one runs
    ~10 min on the N100. Without a check inside the loop, Cancel does nothing
    until the whole measurement finishes."""

    def test_aborting_raises_and_keeps_what_was_measured(self, db, tmp_path):
        for i in range(5):
            _photo_with_faces(db, i)
        cache_path = str(tmp_path / "cache.json")
        calls = []
        # Abort once two photos are in the bag.
        should_abort = lambda: len(calls) >= 2                  # noqa: E731

        with pytest.raises(InterruptedError):
            rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                 should_abort=should_abort,
                                 measure_photo=_fake_measure(calls))

        # The work already done is SAVED — the pass is resumable, so throwing
        # it away would make Cancel cost minutes of N100 time.
        cache = json.loads(Path(cache_path).read_text())
        assert len(cache) == 2

    def test_a_rerun_after_an_abort_measures_only_the_remainder(self, db, tmp_path):
        for i in range(5):
            _photo_with_faces(db, i)
        cache_path = str(tmp_path / "cache.json")
        first = []

        with pytest.raises(InterruptedError):
            rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                 should_abort=lambda: len(first) >= 2,
                                 measure_photo=_fake_measure(first))
        second = []
        out = rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                   measure_photo=_fake_measure(second))

        assert len(second) == 3
        assert out["measured"] == 3
        assert len(json.loads(Path(cache_path).read_text())) == 5

    def test_no_abort_callback_is_the_scripts_behaviour_unchanged(self, db, tmp_path):
        for i in range(3):
            _photo_with_faces(db, i)
        out = rank_measure.measure(db, _DAY, str(tmp_path / "c.json"),
                                   log=lambda m: None,
                                   measure_photo=_fake_measure([]))
        assert out["measured"] == 3


# =========================================================================
# the cache file — it now runs unattended on a box that has wedged twice
# =========================================================================

class TestCacheDurability:
    def test_a_crash_mid_save_leaves_the_previous_good_cache_intact(self, db, tmp_path):
        """A kill (OOM, container swap) part-way through `json.dump` used to
        leave truncated JSON in place, and every later run for that date —
        including a manual `--measure` — then raised. Writing to a temp file
        and `os.replace`-ing means the old file is never in a torn state."""
        _photo_with_faces(db, 0)
        cache_path = str(tmp_path / "cache.json")
        rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                             measure_photo=_fake_measure([]))
        good = Path(cache_path).read_text()

        _photo_with_faces(db, 1)
        real_dump = json.dump

        def exploding_dump(obj, fh, *a, **kw):
            real_dump(obj, fh, *a, **kw)
            fh.flush()
            raise OSError("no space left on device")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(json, "dump", exploding_dump)
            with pytest.raises(OSError):
                rank_measure.measure(db, _DAY, cache_path, log=lambda m: None,
                                     measure_photo=_fake_measure([]))

        assert Path(cache_path).read_text() == good
        assert json.loads(Path(cache_path).read_text())  # still parseable

    def test_a_truncated_cache_is_moved_aside_not_silently_discarded(self, db, tmp_path):
        _photo_with_faces(db, 0)
        cache_path = tmp_path / "cache.json"
        cache_path.write_text('{"1": {"2": {"lap": 1.0,')      # truncated
        warnings = []

        out = rank_measure.measure(db, _DAY, str(cache_path),
                                   log=warnings.append,
                                   measure_photo=_fake_measure([]))

        # It proceeds rather than raising forever...
        assert out["measured"] == 1
        # ...it SAYS so...
        assert any("corrupt" in w.lower() for w in warnings)
        # ...and the unreadable file is kept, in case it held hours of work.
        aside = list(tmp_path.glob("cache.json.corrupt-*"))
        assert len(aside) == 1
        assert aside[0].read_text().startswith('{"1"')

    def test_a_successful_save_leaves_no_temp_file_behind(self, db, tmp_path):
        for i in range(3):
            _photo_with_faces(db, i)
        rank_measure.measure(db, _DAY, str(tmp_path / "cache.json"),
                             log=lambda m: None, measure_photo=_fake_measure([]))
        # (the shared `db` fixture's own file lives here too, so look for the
        # litter specifically rather than asserting on the whole directory)
        assert list(tmp_path.glob("*.tmp")) == []
        assert (tmp_path / "cache.json").exists()


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

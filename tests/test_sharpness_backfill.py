"""Tests for photosearch/sharpness_backfill.py (schema v33, plan step 5).

No real images: every run injects a ``measure`` stub (the real
``sharpness.measure_photo`` has its own tests), and "files" are empty
placeholders under a tmp photo root — the backfill only needs them to exist.
"""

import json
import os
import sqlite3

import pytest

from photosearch import sharpness_backfill as sb
from photosearch.db import MAX_PROCESS_ATTEMPTS, PhotoDB
from photosearch.sharpness import SHARPNESS_VERSION


DIR = "2091/2091-09-19_ILCE-7RM6"


class Recorder:
    """A measure stub: records calls, returns a fixed score (or an error)."""

    def __init__(self, score=123.0, error_for=()):
        self.calls = []
        self.score = score
        self.error_for = set(error_for)

    def __call__(self, path, faces, subjects):
        self.calls.append({"path": path, "faces": faces, "subjects": subjects})
        name = os.path.basename(path)
        if name in self.error_for:
            return {"score": None, "version": SHARPNESS_VERSION,
                    "detail": {"error": "UnidentifiedImageError: cannot identify"}}
        return {"score": self.score, "version": SHARPNESS_VERSION,
                "detail": {"source": "frame", "noise_sigma": 1.5}}


@pytest.fixture
def lib(tmp_path):
    """(db, root, add) — a PhotoDB whose photo_root is a real tmp dir."""
    root = tmp_path / "photos"
    root.mkdir()
    db = PhotoDB(str(tmp_path / "t.db"))
    db.set_photo_root(str(root))

    def add(name, *, folder=DIR, faces_done=True, face_box=None,
            subject_boxes=None, on_disk=True, version=None):
        rel = f"{folder}/{name}"
        if on_disk:
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(b"")
        pid = db.add_photo(filepath=rel, filename=name)
        if subject_boxes is not None:
            db.conn.execute("UPDATE photos SET subject_boxes = ? WHERE id = ?",
                            (json.dumps(subject_boxes), pid))
        if face_box is not None:
            db.add_face(pid, face_box, [])
        elif faces_done:
            db.conn.execute(
                "INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                "VALUES (?, 'faces', ?)", (pid, MAX_PROCESS_ATTEMPTS))
        if version is not None:
            db.conn.execute("UPDATE photos SET sharpness_version = ? WHERE id = ?",
                            (version, pid))
        db.conn.commit()
        return pid

    yield db, root, add
    db.close()


def _run(db, **kw):
    kw.setdefault("apply", True)
    kw.setdefault("pause_s", 0)
    kw.setdefault("lower_prio", False)
    return sb.run_sharpness_backfill(db, **kw)


def _row(db, pid):
    return db.conn.execute(
        "SELECT sharpness, sharpness_json, sharpness_version, sharpness_scored_at "
        "FROM photos WHERE id = ?", (pid,)).fetchone()


# ---------------------------------------------------------------------------
# missing-only + versioning
# ---------------------------------------------------------------------------

def test_measures_missing_rows_and_skips_current_ones(lib):
    db, root, add = lib
    a = add("a.jpg")
    b = add("b.jpg", version=SHARPNESS_VERSION)
    m = Recorder(score=42.5)

    res = _run(db, measure=m)

    assert res["status"] == "done"
    assert res["measured"] == 1
    assert [os.path.basename(c["path"]) for c in m.calls] == ["a.jpg"]
    r = _row(db, a)
    assert r["sharpness"] == 42.5
    assert r["sharpness_version"] == SHARPNESS_VERSION
    assert json.loads(r["sharpness_json"])["source"] == "frame"
    assert r["sharpness_scored_at"]
    assert _row(db, b)["sharpness"] is None   # untouched

    # Idempotent: a second run finds nothing.
    m2 = Recorder()
    res2 = _run(db, measure=m2)
    assert res2["status"] == "skipped" and res2["would"] == 0
    assert m2.calls == []


def test_an_older_version_is_re_measured(lib):
    """Bumping SHARPNESS_VERSION re-measures the library; that is the ONLY
    thing that does."""
    db, root, add = lib
    old = add("old.jpg", version=SHARPNESS_VERSION - 1)
    m = Recorder(score=7.0)
    res = _run(db, measure=m)
    assert res["measured"] == 1
    assert _row(db, old)["sharpness_version"] == SHARPNESS_VERSION
    assert _row(db, old)["sharpness"] == 7.0


def test_newest_first_and_limit(lib):
    db, root, add = lib
    ids = [add(f"p{i}.jpg") for i in range(5)]
    m = Recorder()
    res = _run(db, measure=m, limit=2)
    assert res["measured"] == 2
    measured = {r[0] for r in db.conn.execute(
        "SELECT id FROM photos WHERE sharpness_version IS NOT NULL")}
    assert measured == set(ids[-2:])


# ---------------------------------------------------------------------------
# errors are stored, never retried
# ---------------------------------------------------------------------------

def test_a_decode_error_is_stored_with_the_version_and_not_retried(lib):
    """The CLIP re-claim trap: an unreadable file must not be re-decoded on
    every run forever."""
    db, root, add = lib
    bad = add("bad.jpg")
    good = add("good.jpg")
    m = Recorder(error_for={"bad.jpg"})

    res = _run(db, measure=m)

    assert res["measured"] == 2 and res["errors"] == 1
    r = _row(db, bad)
    assert r["sharpness"] is None
    assert r["sharpness_version"] == SHARPNESS_VERSION
    assert "error" in json.loads(r["sharpness_json"])
    assert _row(db, good)["sharpness"] == 123.0

    m2 = Recorder()
    assert _run(db, measure=m2)["would"] == 0
    assert m2.calls == []


def test_a_raising_measure_is_stored_as_an_error_too(lib):
    db, root, add = lib
    pid = add("x.jpg")

    def boom(path, faces, subjects):
        raise ValueError("bad boxes")

    res = _run(db, measure=boom)
    assert res["errors"] == 1
    assert json.loads(_row(db, pid)["sharpness_json"])["error"].startswith("ValueError")


def test_non_finite_values_are_stored_as_json_null(lib):
    db, root, add = lib
    pid = add("x.jpg")

    def nan(path, faces, subjects):
        return {"score": float("nan"), "version": SHARPNESS_VERSION,
                "detail": {"lap_max": float("inf")}}

    _run(db, measure=nan)
    r = _row(db, pid)
    assert r["sharpness"] is None
    assert json.loads(r["sharpness_json"]) == {"lap_max": None}


# ---------------------------------------------------------------------------
# scope
# ---------------------------------------------------------------------------

def test_empty_photo_ids_means_no_photos_never_the_whole_library(lib):
    db, root, add = lib
    add("a.jpg")
    add("b.jpg")
    m = Recorder()
    res = _run(db, measure=m, photo_ids=[])
    assert res["would"] == 0 and res["status"] == "skipped"
    assert m.calls == []
    assert db.conn.execute(
        "SELECT COUNT(*) FROM photos WHERE sharpness_version IS NOT NULL"
    ).fetchone()[0] == 0


def test_a_folder_that_matches_nothing_is_empty_not_unscoped(lib):
    db, root, add = lib
    add("a.jpg")
    m = Recorder()
    res = _run(db, measure=m, folder="2091/no-such-folder")
    assert res["would"] == 0 and m.calls == []


def test_photo_ids_and_folder_scope(lib):
    db, root, add = lib
    a = add("a.jpg")
    b = add("b.jpg")
    c = add("c.jpg", folder="2091/2091-09-20_other")
    m = Recorder()
    assert _run(db, measure=m, photo_ids=[b])["measured"] == 1
    assert _row(db, b)["sharpness_version"] == SHARPNESS_VERSION
    assert _row(db, a)["sharpness_version"] is None
    res = _run(db, measure=m, folder=DIR)
    assert res["measured"] == 1                       # only `a` left in DIR
    assert _row(db, c)["sharpness_version"] is None


def test_faces_pass_must_be_done_first(lib):
    """Face boxes are the headline region: measuring before detection would
    stamp a frame-only answer that missing-only never revisits."""
    db, root, add = lib
    pending = add("pending.jpg", faces_done=False)
    retrying = add("retrying.jpg", faces_done=False)
    db.conn.execute("INSERT INTO worker_processed (photo_id, pass_type, attempts) "
                    "VALUES (?, 'faces', 1)", (retrying,))
    with_face = add("face.jpg", face_box=(10, 60, 70, 20))
    nobody = add("nobody.jpg")    # terminal: attempts exhausted, no rows
    db.conn.commit()

    ids = sb.candidate_ids(db)
    assert set(ids) == {with_face, nobody}
    assert pending not in ids and retrying not in ids


def test_face_and_subject_boxes_are_passed_to_measure(lib):
    db, root, add = lib
    subjects = [{"label": "dog", "bbox": [0.1, 0.2, 0.5, 0.6]}]
    add("x.jpg", face_box=(10, 60, 70, 20), subject_boxes=subjects)
    m = Recorder()
    _run(db, measure=m)
    call = m.calls[0]
    assert call["faces"] == [{"bbox_left": 20, "bbox_top": 10,
                              "bbox_right": 60, "bbox_bottom": 70}]
    assert call["subjects"] == subjects


# ---------------------------------------------------------------------------
# not local (the replica)
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_here_is_skipped_and_nothing_is_recorded(lib):
    db, root, add = lib
    gone = add("gone.jpg", on_disk=False)
    here = add("here.jpg")
    m = Recorder()
    res = _run(db, measure=m)
    assert res["not_local"] == 1 and res["measured"] == 1
    assert _row(db, gone)["sharpness_version"] is None   # stays a candidate
    assert _row(db, here)["sharpness_version"] == SHARPNESS_VERSION


def test_no_photo_root_means_nothing_is_local(lib):
    """The replica: photo_root unset, so a stored path stays RELATIVE — and a
    relative path is never treated as local, even if it happens to resolve
    against the cwd."""
    db, root, add = lib
    add("a.jpg")
    db.photo_root = None
    m = Recorder()
    res = _run(db, measure=m)
    assert res["not_local"] == 1 and m.calls == []


# ---------------------------------------------------------------------------
# the guarded UPDATE
# ---------------------------------------------------------------------------

def test_a_row_rewritten_mid_run_is_not_clobbered(lib):
    """Guarded on the version it was selected at, like the chunked
    percentile refresh: a concurrent writer wins, and it is counted."""
    db, root, add = lib
    pid = add("x.jpg")

    def racing(path, faces, subjects):
        db.conn.execute(
            "UPDATE photos SET sharpness = 999, sharpness_version = ? WHERE id = ?",
            (SHARPNESS_VERSION, pid))
        return {"score": 1.0, "version": SHARPNESS_VERSION, "detail": {}}

    res = _run(db, measure=racing)
    assert res["raced"] == 1 and res["measured"] == 0
    assert _row(db, pid)["sharpness"] == 999


# ---------------------------------------------------------------------------
# abort + commits
# ---------------------------------------------------------------------------

def test_abort_commits_what_was_measured_then_raises(lib, tmp_path):
    db, root, add = lib
    for i in range(5):
        add(f"p{i}.jpg")
    m = Recorder()
    seen = {"n": 0}

    def should_abort():
        seen["n"] += 1
        return seen["n"] > 2          # let two photos through

    with pytest.raises(InterruptedError):
        _run(db, measure=m, should_abort=should_abort)

    assert len(m.calls) == 2
    # Visible from ANOTHER connection -> it was committed, not just pending.
    other = sqlite3.connect(str(tmp_path / "t.db"))
    n = other.execute("SELECT COUNT(*) FROM photos WHERE sharpness_version IS NOT NULL"
                      ).fetchone()[0]
    other.close()
    assert n == 2


def test_commits_in_chunks_with_progress(lib, tmp_path):
    db, root, add = lib
    for i in range(5):
        add(f"p{i}.jpg")
    events = []
    committed_at_event = []

    def on_progress(ev):
        events.append(ev)
        other = sqlite3.connect(str(tmp_path / "t.db"))
        committed_at_event.append(other.execute(
            "SELECT COUNT(*) FROM photos WHERE sharpness_version IS NOT NULL"
        ).fetchone()[0])
        other.close()

    _run(db, measure=Recorder(), commit_every=2, on_progress=on_progress)
    assert [e["done"] for e in events] == [2, 4, 5]
    assert committed_at_event == [2, 4, 5]


def test_pause_between_photos(lib):
    db, root, add = lib
    for i in range(3):
        add(f"p{i}.jpg")
    sleeps = []
    _run(db, measure=Recorder(), pause_s=0.2, sleep=sleeps.append)
    assert sleeps == [0.2, 0.2]          # between photos, not after the last


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------

def test_dry_run_counts_and_writes_nothing(lib):
    db, root, add = lib
    add("a.jpg")
    add("b.jpg")
    m = Recorder()
    res = _run(db, measure=m, apply=False)
    assert res["status"] == "preview" and res["would"] == 2
    assert m.calls == []
    assert db.conn.execute(
        "SELECT COUNT(*) FROM photos WHERE sharpness_version IS NOT NULL"
    ).fetchone()[0] == 0


def test_dry_run_on_a_read_only_connection(lib, tmp_path):
    db, root, add = lib
    add("a.jpg")
    db.close()
    path = str(tmp_path / "t.db")
    before = os.path.getmtime(path)
    ro = sb.open_readonly(path, photo_root=str(root))
    try:
        res = sb.run_sharpness_backfill(ro, apply=False, measure=Recorder(),
                                        lower_prio=False, folder=DIR)
        assert res["would"] == 1
        with pytest.raises(sqlite3.OperationalError):
            ro.conn.execute("UPDATE photos SET sharpness = 1")
    finally:
        ro.close()
    assert os.path.getmtime(path) == before


def test_open_readonly_never_creates_a_missing_db(tmp_path):
    missing = tmp_path / "typo.db"
    with pytest.raises(FileNotFoundError):
        sb.open_readonly(str(missing))
    assert not missing.exists()


# ---------------------------------------------------------------------------
# yielding to ingest / batch-advance
# ---------------------------------------------------------------------------

def test_refuses_to_start_while_ingest_holds_its_lock(lib):
    from photosearch.ingest import _sweep_lock, sweep_lock_held
    db, root, add = lib
    add("a.jpg")
    m = Recorder()
    assert not sweep_lock_held(db.db_path)
    with _sweep_lock(db.db_path):
        assert sweep_lock_held(db.db_path)
        res = _run(db, measure=m)
    assert res["status"] == "refused" and "ingest" in res["message"]
    assert m.calls == []
    assert not sweep_lock_held(db.db_path)       # the probe held nothing


def test_the_probe_never_blocks_ingest(lib):
    from photosearch.ingest import _sweep_lock, sweep_lock_held
    db, root, add = lib
    sweep_lock_held(db.db_path)
    with _sweep_lock(db.db_path):   # would raise IngestAlreadyRunning if held
        pass


def test_refuses_while_a_batch_advance_nas_step_is_open(lib):
    from photosearch.ingest_batches import open_job, register_batch
    db, root, add = lib
    add("a.jpg")
    batch_id = register_batch(db, DIR)
    open_job(db, batch_id, "warm_crops", "nas")
    res = _run(db, measure=Recorder())
    assert res["status"] == "refused" and "warm_crops" in res["message"]
    # A fleet job (a worker pass) is not a reason to yield: the fleet runs on
    # the desktop and only posts results.
    db.conn.execute("DELETE FROM ingest_batch_jobs")
    open_job(db, batch_id, "clip", "fleet")
    assert _run(db, measure=Recorder())["status"] == "done"


def test_the_batch_runner_ignores_its_own_open_row(lib):
    from photosearch.ingest_batches import open_job, register_batch
    db, root, add = lib
    add("a.jpg")
    batch_id = register_batch(db, DIR)
    open_job(db, batch_id, "sharpness", "nas")
    res = _run(db, measure=Recorder(), ignore_batch_steps=("sharpness",))
    assert res["status"] == "done"


def test_yields_mid_run_when_ingest_starts(lib, monkeypatch):
    db, root, add = lib
    for i in range(4):
        add(f"p{i}.jpg")
    calls = {"n": 0}

    def busy(db_, ignore_batch_steps=()):
        calls["n"] += 1
        return None if calls["n"] == 1 else "an ingest-incoming sweep is running"

    monkeypatch.setattr(sb, "busy_reason", busy)
    m = Recorder()
    res = _run(db, measure=m, commit_every=2)
    assert res["status"] == "yielded"
    assert len(m.calls) == 2 and res["measured"] == 2
    assert "2/4" in res["message"]


def test_dry_run_reports_busy_but_still_counts(lib):
    from photosearch.ingest import _sweep_lock
    db, root, add = lib
    add("a.jpg")
    with _sweep_lock(db.db_path):
        res = _run(db, measure=Recorder(), apply=False)
    assert res["status"] == "preview" and res["would"] == 1 and res["busy"]


# ---------------------------------------------------------------------------
# wiring: maintenance stage + batch runner
# ---------------------------------------------------------------------------

def test_maintenance_stage_is_opt_in_and_capped(lib, monkeypatch):
    from photosearch import maintenance
    db, root, add = lib
    for i in range(3):
        add(f"p{i}.jpg")
    seen = {}

    def fake_run(db_, **kw):
        seen.update(kw)
        return {"status": "done", "would": 2, "measured": 2, "errors": 0,
                "not_local": 0, "raced": 0}

    monkeypatch.setattr(sb, "run_sharpness_backfill", fake_run)
    res = maintenance.run_maintenance_sweep(
        db, apply=True, do_colors=False, do_stacking=False, do_match=False)
    assert "sharpness" not in [s["stage"] for s in res["stages"]]
    assert seen == {}

    res = maintenance.run_maintenance_sweep(
        db, apply=True, do_colors=False, do_stacking=False, do_match=False,
        do_sharpness=True, sharpness_limit=2, stages=["sharpness"])
    stage = res["stages"][0]
    assert stage["stage"] == "sharpness" and stage["applied"] == 2
    assert seen["limit"] == 2 and seen["apply"] is True
    assert seen["pause_s"] == sb.DEFAULT_PAUSE_S


def test_maintenance_stage_default_limit(lib, monkeypatch):
    from photosearch import maintenance
    db, root, add = lib
    seen = {}

    def fake_run(db_, **kw):
        seen.update(kw)
        return {"status": "skipped", "would": 0}

    monkeypatch.setattr(sb, "run_sharpness_backfill", fake_run)
    maintenance.run_maintenance_sweep(db, do_sharpness=True, stages=["sharpness"])
    assert seen["limit"] == sb.DEFAULT_STAGE_LIMIT == 5000
    assert seen["apply"] is False


def test_batch_runner_skips_an_empty_batch_without_calling_the_backfill(monkeypatch):
    from photosearch import batch_advance
    monkeypatch.setattr(sb, "run_sharpness_backfill",
                        lambda *a, **k: pytest.fail("must not be called"))
    out = batch_advance.default_runners()["sharpness"](
        None, {"photo_ids": [], "emit": lambda e: None,
               "check_abort": lambda: None})
    assert out == {"skipped": "empty batch"}


def test_batch_runner_is_scoped_and_raises_when_refused(monkeypatch):
    from photosearch import batch_advance
    seen = {}

    def fake(db_, **kw):
        seen.update(kw)
        return {"status": "refused", "message": "not started: ingest"}

    monkeypatch.setattr(sb, "run_sharpness_backfill", fake)
    with pytest.raises(RuntimeError, match="ingest"):
        batch_advance.default_runners()["sharpness"](
            None, {"photo_ids": [5, 6], "emit": lambda e: None,
                   "check_abort": lambda: None})
    assert seen["photo_ids"] == [5, 6] and seen["apply"] is True
    assert seen["ignore_batch_steps"] == ("sharpness",)


# ---------------------------------------------------------------------------
# end to end with the REAL measure_photo (tiny synthetic files)
# ---------------------------------------------------------------------------

def test_real_measure_photo_round_trips_through_the_column(lib):
    """The stubs above cannot catch a numpy scalar that json.dumps rejects or
    a detail shape the column cannot hold — so run the production measure
    once on a small real JPEG, plus a garbage file with a .jpg extension."""
    import numpy as np
    from PIL import Image, ImageFilter
    db, root, add = lib
    good = add("real.jpg")
    bad = add("zip.jpg")
    # Smoothed noise (grain ~2 px) — textured, so tiles are not skipped as flat.
    rng = np.random.default_rng(0)
    n = Image.fromarray(np.clip(128 + 40 * rng.normal(size=(300, 400)), 0, 255)
                        .astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(n, dtype=np.float32)
    a = (a - a.mean()) / (a.std() + 1e-6)
    arr = np.clip(128 + 70 * a, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(root / DIR / "real.jpg", "JPEG", quality=95)
    (root / DIR / "zip.jpg").write_bytes(b"PK\x03\x04 not an image")

    from photosearch.sharpness import measure_photo
    measure = lambda p, f, s: measure_photo(p, f, s, long_edge=400)  # noqa: E731
    res = _run(db, measure=measure)

    assert res["measured"] == 2 and res["errors"] == 1
    g = _row(db, good)
    assert isinstance(g["sharpness"], float)
    assert json.loads(g["sharpness_json"])["source"] == "frame"
    b = _row(db, bad)
    assert b["sharpness"] is None and b["sharpness_version"] == SHARPNESS_VERSION
    assert "error" in json.loads(b["sharpness_json"])


# ---------------------------------------------------------------------------
# the candidate query uses the index
# ---------------------------------------------------------------------------

def test_candidate_query_uses_the_sharpness_index(lib):
    db, root, add = lib
    add("a.jpg")
    plan = " ".join(r[3] for r in db.conn.execute(
        "EXPLAIN QUERY PLAN SELECT p.id FROM photos p WHERE "
        + sb.missing_sql("p") + " ORDER BY +p.id DESC", (SHARPNESS_VERSION,)))
    assert "idx_photos_sharpness_version" in plan

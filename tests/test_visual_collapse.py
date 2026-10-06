"""category-visual collapse: one tag set stamped on a whole shoot.

1,211 of 1,260 photos on 2026-10-03 got exactly `colorful, sunny, vibrant`.
Every answer looked plausible alone; only the batch shows the collapse. These
pin the thresholds, that it is a FLAG (the step's state never changes), and
the read-only folder report.
"""
import json
import sqlite3

from photosearch import visual_collapse as VC
from photosearch.batch_state import batch_state
from photosearch.ingest_batches import register_batch

SUNNY = json.dumps(["colorful", "sunny", "vibrant"])


def test_the_2026_10_03_shape_is_collapsed():
    sets = [SUNNY] * 1211 + [json.dumps(["moody"])] * 49
    s = VC.summarize_sets(sets)
    assert s["collapsed"] and s["top_count"] == 1211 and s["tagged"] == 1260
    assert s["top_set"] == "colorful, sunny, vibrant"


def test_a_varied_shoot_is_not():
    sets = [json.dumps([f"t{i % 9}"]) for i in range(200)]
    assert not VC.summarize_sets(sets)["collapsed"]


def test_too_few_photos_is_never_judged():
    assert not VC.summarize_sets([SUNNY] * 49)["collapsed"]
    assert VC.summarize_sets([SUNNY] * 50)["collapsed"]


def test_no_tagged_photos_is_none():
    assert VC.summarize_sets([]) is None
    assert VC.batch_warning(None) == (None, None)


def test_the_warning_names_the_set():
    short, full = VC.batch_warning(VC.summarize_sets([SUNNY] * 60))
    assert short == "⚠ 100% share one tag set"
    assert "colorful, sunny, vibrant" in full and "60 of 60" in full


def _conn(rows):
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE photos (id INTEGER PRIMARY KEY, folder TEXT, visual_tags TEXT)")
    c.executemany("INSERT INTO photos (folder, visual_tags) VALUES (?, ?)", rows)
    return c


def test_collapsed_folders_worst_first_and_ignores_empty_sets():
    rows = ([("2026/a", SUNNY)] * 60 + [("2026/a", json.dumps(["moody"]))] * 10
            + [("2026/b", SUNNY)] * 55
            # '[]' is "nothing notable" — a real answer, not a collapse.
            + [("2026/c", "[]")] * 100 + [("2026/c", None)] * 5
            + [("2025/d", SUNNY)] * 80)
    c = _conn(rows)
    out = VC.collapsed_folders(c)
    assert [r["folder"] for r in out] == ["2025/d", "2026/b", "2026/a"]
    assert [r["folder"] for r in VC.collapsed_folders(c, folder_prefix="2026")] == [
        "2026/b", "2026/a"]


def test_folder_prefix_includes_subfolders_only():
    c = _conn([("2026/x/sub", SUNNY)] * 60 + [("2026/xy", SUNNY)] * 60)
    assert [r["folder"] for r in VC.collapsed_folders(c, folder_prefix="2026/x")] == [
        "2026/x/sub"]


def test_folder_photo_ids_top_set_only():
    c = _conn([("f", SUNNY)] * 3 + [("f", json.dumps(["moody"]))] + [("f", "[]")])
    assert len(VC.folder_photo_ids(c, ["f"])) == 4
    assert len(VC.folder_photo_ids(c, ["f"], only_top_set=True)) == 3


# --- /batches -------------------------------------------------------------

_DIR = "2091/2091-10-03_ILCE-7RM6"


def _batch(db, tags_by_photo):
    ids = []
    for i, tags in enumerate(tags_by_photo):
        pid = db.add_photo(filepath=f"{_DIR}/img{i:04d}.jpg", filename=f"img{i:04d}.jpg",
                           date_taken="2091-10-03T10:00:00")
        if tags is not None:
            db.conn.execute("UPDATE photos SET visual_tags = ? WHERE id = ?", (tags, pid))
        ids.append(pid)
    db.conn.commit()
    return register_batch(db, _DIR)


def _visual(db, batch_id):
    return next(s for s in batch_state(db, batch_id)["steps"]
                if s["step"] == "category-visual")


def test_a_collapsed_batch_is_flagged_but_still_completed(db):
    batch_id = _batch(db, [SUNNY] * 55 + [json.dumps(["moody"])] * 5)
    step = _visual(db, batch_id)
    assert step["state"] == "completed"           # a flag, never a state change
    assert step["warning"] == "⚠ 92% share one tag set"
    assert "colorful, sunny, vibrant" in step["warning_detail"]
    assert step["collapse"]["collapsed"] is True


def test_a_varied_batch_carries_no_warning(db):
    batch_id = _batch(db, [json.dumps([f"t{i % 7}"]) for i in range(60)])
    step = _visual(db, batch_id)
    assert step["warning"] is None and step["collapse"]["collapsed"] is False

"""Undo/redo history + the archetype-switch re-slot fix for the photo-book editor.

The editor's layout mutations (add/update/delete spread, cell edits, auto-arrange)
snapshot the spreads+cells state so they can be undone/redone. Also covers the
regression where switching a spread down to 2 photos left no way back to a 3-up
row (gallery row now always opens >=3 slots).
"""
import sqlite3

from photosearch.book import BookStore


class _FakePDB:
    """Minimal stand-in for the replica PhotoDB — book layout only reads photo
    dimensions/faces/subject_boxes for crop seeding."""
    def __init__(self, conn):
        self.conn = conn


def _make():
    pconn = sqlite3.connect(":memory:")
    pconn.row_factory = sqlite3.Row
    pconn.executescript(
        """
        CREATE TABLE photos (id INTEGER PRIMARY KEY, image_width INT,
                             image_height INT, subject_boxes TEXT);
        CREATE TABLE faces (photo_id INT, bbox_left REAL, bbox_top REAL,
                            bbox_right REAL, bbox_bottom REAL);
        INSERT INTO photos VALUES (1,4000,3000,NULL),(2,4000,3000,NULL),
                                  (3,4000,3000,NULL);
        """
    )
    pdb = _FakePDB(pconn)
    bs = BookStore(":memory:")
    bid = bs.create_book("T", trim_w=14, trim_h=11)
    return bs, pdb, bid


def _last_pids(bs, bid):
    spreads = bs.get_book(bid)["spreads"]
    return [c["photo_id"] for c in spreads[-1]["cells"]] if spreads else None


def test_switch_back_to_3up_reopens_a_third_slot():
    # 3-photo collage -> matched 2-up drops a photo -> matched 3-up must give a
    # 3rd (empty) slot so the dropped photo can be re-added, not stay stuck at 2.
    bs, pdb, bid = _make()
    sid = bs.add_spread(pdb, bid, archetype="asymmetric collage", photo_ids=[1, 2, 3])
    bs.update_spread(pdb, sid, {"archetype": "matched 2-up"})
    assert len(_last_pids(bs, bid)) == 2
    bs.update_spread(pdb, sid, {"archetype": "gallery row"})
    pids = _last_pids(bs, bid)
    assert len(pids) == 3            # third slot reopened
    assert pids[2] is None           # and it's an empty drop target


def test_undo_redo_walks_layout_snapshots():
    bs, pdb, bid = _make()
    sid = bs.add_spread(pdb, bid, archetype="asymmetric collage", photo_ids=[1, 2, 3])
    bs.update_spread(pdb, sid, {"archetype": "matched 2-up"})
    bs.update_spread(pdb, sid, {"archetype": "gallery row"})

    assert bs.get_book(bid)["can_undo"] is True
    assert bs.get_book(bid)["can_redo"] is False

    assert bs.undo(bid) is True
    assert len(_last_pids(bs, bid)) == 2           # back to matched 2-up
    assert bs.undo(bid) is True
    assert _last_pids(bs, bid) == [1, 2, 3]        # back to the original collage
    assert bs.undo(bid) is True
    assert _last_pids(bs, bid) is None             # baseline: before the spread existed
    assert bs.get_book(bid)["can_undo"] is False   # nothing left to undo
    assert bs.undo(bid) is False

    assert bs.redo(bid) is True
    assert _last_pids(bs, bid) == [1, 2, 3]        # collage restored
    assert bs.get_book(bid)["can_redo"] is True


def test_new_edit_truncates_the_redo_tail():
    bs, pdb, bid = _make()
    sid = bs.add_spread(pdb, bid, archetype="asymmetric collage", photo_ids=[1, 2, 3])
    bs.update_spread(pdb, sid, {"archetype": "matched 2-up"})
    bs.undo(bid)                                   # back to the collage; redo available
    assert bs.get_book(bid)["can_redo"] is True
    # a fresh edit from here must drop the redo branch
    sid2 = bs.get_book(bid)["spreads"][-1]["id"]   # id changed after restore
    bs.update_spread(pdb, sid2, {"archetype": "dense grid"})
    assert bs.get_book(bid)["can_redo"] is False


def test_cell_photo_swap_is_undoable():
    bs, pdb, bid = _make()
    sid = bs.add_spread(pdb, bid, archetype="matched 2-up", photo_ids=[1, 2])
    cell0 = bs.get_book(bid)["spreads"][-1]["cells"][0]["id"]
    bs.set_cell(pdb, cell0, {"photo_id": 3})
    assert _last_pids(bs, bid)[0] == 3
    assert bs.undo(bid) is True
    assert _last_pids(bs, bid)[0] == 1             # original photo restored

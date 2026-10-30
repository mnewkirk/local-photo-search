"""Read-time name rendering (docs/plans/description-names.md).

Pure render() over (row, faces, persons) plus the batched loader on a real DB."""
import json

import pytest

from photosearch import description_render as R
from photosearch.people_bind import sha_of

D = ("A young boy with reddish hair walks across a grassy field wearing a neon "
     "green jersey. He is in front of a fence.")
SPAN = "A young boy with reddish hair"


def _row(desc=D, faces=None, spans=None, kind="person"):
    faces = faces if faces is not None else [{"face_id": 10, "bbox": [100, 300, 400, 100]}]
    spans = spans if spans is not None else [{"start": 0, "end": len(SPAN), "text": SPAN}]
    return {"description_sha": sha_of(desc),
            "slots_json": json.dumps([{"slot": "s1", "kind": kind, "spans": spans,
                                       "faces": faces}])}


def _face(fid=10, person=1, src="manual", box=(100, 300, 400, 100)):
    return {"id": fid, "person_id": person, "match_source": src,
            "bbox_top": box[0], "bbox_right": box[1], "bbox_bottom": box[2],
            "bbox_left": box[3]}


PERSONS = {1: "Zack Azerad", 2: "Calvin Newkirk", 3: "Koa Smith"}


def test_a_manual_label_names_the_child():
    r = R.render(D, _row(), [_face()], PERSONS, sources=("manual", "strict", "merge_review"))
    assert r.text.startswith("Zack walks across a grassy field")
    assert r.text.endswith("He is in front of a fence.")        # pronouns untouched
    assert r.slots[0]["state"] == R.NAMED and r.slots[0]["name"] == "Zack"


def test_relabel_changes_the_text_with_no_row_change():
    row = _row()
    a = R.render(D, row, [_face(person=3)], PERSONS)          # labelled Koa
    b = R.render(D, row, [_face(person=2)], PERSONS)          # corrected to Calvin
    assert a.text.startswith("Koa walks") and b.text.startswith("Calvin walks")


@pytest.mark.parametrize("src,state", [
    ("temporal", R.UNTRUSTED), ("dedupe_unmatched", R.UNTRUSTED),
    (None, R.UNTRUSTED), ("rejected", R.REJECTED_STATE)])
def test_untrusted_sources_never_name(src, state):
    r = R.render(D, _row(), [_face(src=src)], PERSONS)
    assert r.text == D and r.slots[0]["state"] == state


def test_an_unlabelled_face_stays_generic():
    r = R.render(D, _row(), [_face(person=None, src=None)], PERSONS)
    assert r.text == D and r.slots[0]["state"] == R.UNLABELLED


def test_env_can_narrow_but_never_widen_the_sources():
    assert R.name_sources({R.NAME_SOURCES_ENV: "manual"}) == ("manual",)
    assert R.name_sources({R.NAME_SOURCES_ENV: "manual,temporal"}) == ("manual",)
    assert R.name_sources({}) == ("manual", "strict", "merge_review")
    r = R.render(D, _row(), [_face(src="strict")], PERSONS, sources=("manual",))
    assert r.text == D


def test_stale_description_renders_raw():
    row = _row()
    row["description_sha"] = sha_of("something else")
    assert R.render(D, row, [_face()], PERSONS).text == D
    assert R.render(D, None, [_face()], PERSONS).text == D


def test_a_person_on_two_faces_is_never_named_in_that_photo():
    faces = [_face(10, 1), _face(11, 1, box=(500, 900, 800, 500))]
    r = R.render(D, _row(), faces, PERSONS)
    assert r.text == D and r.slots[0]["state"] == R.AMBIGUOUS
    # a rejected / dedupe_unmatched copy does not count against the person
    faces = [_face(10, 1), _face(11, 1, "dedupe_unmatched", box=(500, 900, 800, 500))]
    assert R.render(D, _row(), faces, PERSONS).text.startswith("Zack walks")


def test_missing_face_is_repaired_by_iou_only_when_unique():
    # re-detection: new id, box nudged a few px -> same face
    moved = _face(77, box=(104, 296, 396, 104))
    assert R.render(D, _row(), [moved], PERSONS).text.startswith("Zack walks")
    # two overlapping candidates: refuse to guess
    twin = _face(78, box=(102, 298, 398, 102))
    r = R.render(D, _row(), [moved, twin], PERSONS)
    assert r.text == D and r.slots[0]["state"] == R.STALE
    # nothing close
    far = _face(79, box=(900, 1200, 1300, 800))
    assert R.render(D, _row(), [far], PERSONS).text == D


def test_same_id_but_a_different_box_is_not_trusted():
    far = _face(10, box=(900, 1200, 1300, 800))
    assert R.render(D, _row(), [far], PERSONS).text == D


def test_group_is_all_or_nothing():
    d = "Two players in blue chase the ball."
    sp = [{"start": 0, "end": 19, "text": d[:19]}]
    row = _row(d, faces=[{"face_id": 1, "bbox": [0, 100, 100, 0]},
                         {"face_id": 2, "bbox": [0, 300, 100, 200]}], spans=sp, kind="group")
    f = lambda i, p, src, b: {"id": i, "person_id": p, "match_source": src, "bbox_top": b[0],
                              "bbox_right": b[1], "bbox_bottom": b[2], "bbox_left": b[3]}
    both = [f(1, 1, "manual", (0, 100, 100, 0)), f(2, 2, "strict", (0, 300, 100, 200))]
    assert R.render(d, row, both, PERSONS).text == "Zack and Calvin chase the ball."
    half = [f(1, 1, "manual", (0, 100, 100, 0)), f(2, None, None, (0, 300, 100, 200))]
    assert R.render(d, row, half, PERSONS).text == d
    same = [f(1, 1, "manual", (0, 100, 100, 0)), f(2, 1, "manual", (0, 300, 100, 200))]
    assert R.render(d, row, same, PERSONS).text == d


def test_first_name_collisions_use_the_full_name():
    persons = {1: "Greg Neumeister", 2: "Greg Marini", 3: "Koa Smith"}
    assert R.render(D, _row(), [_face(person=1)], persons).text.startswith("Greg Neumeister walks")
    assert R.render(D, _row(), [_face(person=3)], persons).text.startswith("Koa walks")


def test_possessive_and_repeated_spans():
    d = "The boy's jersey is green as the boy runs."
    i = d.index("the boy")
    spans = [{"start": 0, "end": 7, "text": "The boy"}, {"start": i, "end": i + 7, "text": "the boy"}]
    r = R.render(d, _row(d, spans=spans), [_face()], PERSONS)
    assert r.text == "Zack's jersey is green as Zack runs."


def test_all_generic_equals_raw_and_render_never_raises():
    assert R.render(D, _row(), [], PERSONS).text == D
    bad = {"description_sha": sha_of(D), "slots_json": "not json"}
    assert R.render(D, bad, [_face()], PERSONS).text == D
    assert R.render("", _row(), [_face()], PERSONS).text == ""


def test_text_differs_from_raw_only_inside_spans():
    r = R.render(D, _row(), [_face()], PERSONS)
    assert r.text.endswith(D[len(SPAN):])


# --- the DB loader, the flag and the sweep ----------------------------------------------

@pytest.fixture
def db(tmp_path):
    from photosearch.db import PhotoDB
    d = PhotoDB(str(tmp_path / "t.db"))
    yield d
    d.close()


def _seed(db, label="manual"):
    pid = db.add_photo(filepath="2026/x/a.JPG", filename="a.JPG", description=D)
    zack = db.conn.execute("INSERT INTO persons(name) VALUES ('Zack Azerad')").lastrowid
    db.conn.execute(
        "INSERT INTO faces(photo_id, person_id, bbox_top, bbox_right, bbox_bottom, bbox_left, "
        "match_source, det_score) VALUES (?,?,?,?,?,?,?,0.9)",
        (pid, zack, 100, 300, 400, 100, label))
    fid = db.conn.execute("SELECT id FROM faces WHERE photo_id=?", (pid,)).fetchone()[0]
    db.conn.execute(
        "INSERT INTO description_subjects(photo_id, description_sha, slots_json, method, "
        "bind_version) VALUES (?,?,?,?,1)",
        (pid, sha_of(D), json.dumps([{"slot": "s1", "kind": "person",
         "spans": [{"start": 0, "end": len(SPAN), "text": SPAN}],
         "faces": [{"face_id": fid, "bbox": [100, 300, 400, 100]}]}]), "rule"))
    db.conn.commit()
    return pid, fid, zack


def test_flag_off_leaves_payloads_untouched(db, monkeypatch):
    pid, *_ = _seed(db)
    monkeypatch.delenv(R.ENABLED_ENV, raising=False)
    rows = [{"id": pid, "description": D}]
    R.attach_display_descriptions(db.conn, rows)
    assert rows == [{"id": pid, "description": D}]


def test_flag_on_adds_description_display_and_never_touches_description(db, monkeypatch):
    pid, fid, zack = _seed(db)
    monkeypatch.setenv(R.ENABLED_ENV, "1")
    rows = [{"id": pid, "description": D}]
    R.attach_display_descriptions(db.conn, rows, detail=True)
    assert rows[0]["description"] == D
    assert rows[0]["description_display"].startswith("Zack walks")
    assert rows[0]["description_subjects"][0]["state"] == R.NAMED
    # the relabel: no row changes, the text follows
    calvin = db.conn.execute("INSERT INTO persons(name) VALUES ('Calvin Newkirk')").lastrowid
    db.assign_face_to_person(fid, calvin, "manual")
    db.conn.commit()
    rows = [{"id": pid, "description": D}]
    R.attach_display_descriptions(db.conn, rows)
    assert rows[0]["description_display"].startswith("Calvin walks")
    # and a clear (rejected) puts the generic words back
    db.conn.execute("UPDATE faces SET person_id=NULL, match_source='rejected' WHERE id=?", (fid,))
    db.conn.commit()
    rows = [{"id": pid, "description": D}]
    R.attach_display_descriptions(db.conn, rows)
    assert "description_display" not in rows[0]


def test_a_db_without_the_table_renders_raw(db, monkeypatch):
    pid, *_ = _seed(db)
    db.conn.execute("DROP TABLE description_subjects")
    monkeypatch.setenv(R.ENABLED_ENV, "1")
    rows = [{"id": pid, "description": D}]
    R.attach_display_descriptions(db.conn, rows)
    assert rows == [{"id": pid, "description": D}]


def test_sweep_rebinds_a_moved_face_and_requeues_a_lost_one(db):
    pid, fid, _ = _seed(db)
    # re-detection: same box, new id
    db.conn.execute("UPDATE faces SET id = id + 1000 WHERE id = ?", (fid,))
    db.conn.commit()
    dry = R.sweep_subjects(db.conn)
    assert dry["rebound"] == 1 and db.conn.execute(
        "SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 1
    assert R.sweep_subjects(db.conn, apply=True)["rebound"] == 1
    slots = json.loads(db.conn.execute("SELECT slots_json FROM description_subjects").fetchone()[0])
    assert slots[0]["faces"][0]["face_id"] == fid + 1000
    # the face is deleted outright: the row and its ledger entry go, the photo is re-queued
    db.mark_processed([pid], "people-bind")
    db.conn.execute("DELETE FROM faces WHERE photo_id = ?", (pid,))
    db.conn.commit()
    assert R.sweep_subjects(db.conn, apply=True)["requeued"] == 1
    assert db.conn.execute("SELECT COUNT(*) FROM description_subjects").fetchone()[0] == 0
    assert db.conn.execute("SELECT COUNT(*) FROM worker_processed WHERE pass_type='people-bind'"
                           ).fetchone()[0] == 0

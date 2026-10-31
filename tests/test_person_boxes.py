"""person_boxes: the pure face-to-body link and sheet lettering (no torch)."""
from photosearch import person_boxes as PBX


def _body(t, r, b, l, score=0.9):
    return {"bbox": [t, r, b, l], "score": score}


def _face(fid, t, r, b, l):
    return {"face_id": fid, "bbox": [t, r, b, l]}


def test_a_face_goes_to_the_smallest_body_containing_its_centre():
    big = _body(0, 1000, 1000, 0)            # a crowd-sized box around everyone
    kid = _body(100, 400, 900, 200)
    assert PBX.link_faces([_face(7, 120, 330, 220, 260)], [big, kid]) == {7: 1}


def test_links_are_one_to_one_and_a_face_outside_every_body_is_unlinked():
    one = _body(100, 400, 900, 200)
    faces = [_face(1, 120, 330, 220, 260), _face(2, 130, 340, 230, 270),   # both inside `one`
             _face(3, 120, 990, 220, 900)]                                   # inside nothing
    link = PBX.link_faces(faces, [one])
    assert list(link.values()) == [0] and 3 not in link


def test_bbox_column_names_are_accepted():
    f = {"face_id": 5, "bbox_top": 120, "bbox_right": 330, "bbox_bottom": 220, "bbox_left": 260}
    assert PBX.link_faces([f], [_body(100, 400, 900, 200)]) == {5: 0}


def test_letters_run_past_z():
    assert [PBX.body_letter(i) for i in (0, 25, 26, 27, 51, 52)] == ["A", "Z", "AA", "AB", "AZ", "BA"]


def test_letter_bodies_orders_left_to_right_drops_low_scores_and_records_the_face():
    bodies = [_body(0, 900, 500, 700), _body(0, 300, 500, 100), _body(0, 600, 500, 400, score=0.2)]
    out = PBX.letter_bodies(bodies, 0.5, [_face(9, 50, 280, 120, 200)])
    assert [(b["b"], b["bbox"][3], b["face_id"]) for b in out] == [("A", 100, 9), ("B", 700, None)]

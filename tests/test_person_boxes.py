"""person_boxes: the pure face-to-body link and sheet lettering (no torch)."""
from photosearch import person_boxes as PBX


def _body(t, r, b, l, score=0.9):
    return {"bbox": [t, r, b, l], "score": score}


def _face(fid, t, r, b, l):
    return {"face_id": fid, "bbox": [t, r, b, l]}


def test_a_face_goes_to_the_body_it_is_the_head_of():
    big = _body(0, 1000, 1000, 0)            # a crowd-sized box around everyone
    kid = _body(100, 400, 900, 200)
    assert PBX.link_faces([_face(7, 120, 330, 220, 260)], [big, kid]) == {7: 1}
    low = _body(0, 400, 900, 200)            # the face sits half-way down: not its head
    assert PBX.link_faces([_face(7, 500, 330, 600, 260)], [low]) == {}


# 247988: a boy in front (face 1), a boy behind him (face 2), an upper-body fragment of
# the front boy, and a box on the boy behind. Real boxes from the sample.
_FRONT, _FRAG, _BEHIND = (_body(1398, 1636, 3911, 327, 0.9998), _body(1315, 1406, 2757, 550, 0.8498),
                          _body(1146, 1473, 2143, 558, 0.9801))
_F1, _F2 = _face(1, 1612, 1235, 1900, 1010), _face(2, 1367, 1410, 1610, 1185)


def test_a_fragment_no_longer_steals_the_face_from_its_own_body():
    kept, merged = PBX.dedupe([_FRONT, _FRAG, _BEHIND])
    assert kept == [0, 2] and merged == {1: 0}
    assert PBX.link_faces([_F1, _F2], [_FRONT, _BEHIND]) == {1: 0, 2: 1}


def test_overlapping_players_each_keep_their_own_face():
    # 243560 (scaled from the sheet): a boy in green in front, a boy in blue behind,
    # their boxes overlapping and sharing a centre; each face tops its own box.
    green, blue = _body(508, 695, 853, 472), _body(455, 737, 840, 587)
    f_green, f_blue = _face(1, 522, 672, 572, 632), _face(3, 465, 672, 510, 632)
    assert PBX.link_faces([f_green, f_blue], [green, blue]) == {1: 0, 3: 1}


def test_a_second_box_on_one_player_is_a_duplicate_but_a_person_behind_him_is_not():
    # 247644: two boxes on Jack (same top, bottom, centre); a spectator wholly inside
    # a player's box, centre offset.
    jack_loose, jack = _body(1707, 6006, 3408, 5189, 0.5992), _body(1691, 5725, 3379, 5311, 0.9808)
    player, spectator = _body(1794, 3279, 3810, 2212, 0.9995), _body(1869, 3100, 3152, 2706, 0.9922)
    kept, merged = PBX.dedupe([player, spectator, jack_loose, jack])
    assert kept == [0, 1, 3] and merged == {2: 3}


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


def test_a_face_straddling_a_bodys_edge_is_not_its_head():
    # 246150 (real boxes): the navy boy's face, half outside the box of the boy in
    # yellow in front of him (facing away), must not be credited to that body.
    yellow = _body(1710, 5363, 2973, 4750)
    navy_face = _face(510578, 1825, 4815, 1967, 4694)
    assert PBX._face_outside(navy_face["bbox"], yellow["bbox"]) > 0.4
    assert PBX.link_faces([navy_face], [yellow]) == {}

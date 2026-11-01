"""people-bind body mode (plan Revision 2): boxes around every person, a phrase
whose box holds no face binds nothing. No model, no image: the call is mocked."""
from photosearch import people_bind as PB

H = 4000   # image height; bodies below 10% of it are not drawn


def _face(fid, t, r, b, l):
    return {"face_id": fid, "bbox": [t, r, b, l], "det_score": 0.9}


def _body(t, r, b, l, score=0.99):
    return {"bbox": [t, r, b, l], "score": score}


# a boy facing the camera (face 1) and a boy seen from behind, no face
FACE = _face(1, 1000, 700, 1200, 500)
FACED = _body(950, 900, 3000, 300)
BACK = _body(900, 2000, 3100, 1400)
TINY = _body(100, 3000, 300, 2900)        # 5% of the frame: a far spectator


def test_targets_put_each_face_in_its_body_and_add_tall_faceless_bodies():
    t = PB.body_targets([FACE], [FACED, BACK, TINY], H)
    assert t == [{"bbox": FACED["bbox"], "face_id": 1}, {"bbox": BACK["bbox"], "face_id": None}]


def test_a_face_with_no_body_is_drawn_as_its_own_box_and_the_cap_holds():
    many = [_body(500, 200 + 300 * i + 250, 3500, 200 + 300 * i) for i in range(20)]
    lone = _face(9, 100, 9900, 300, 9700)
    t = PB.body_targets([lone], many, H)
    assert t[0] == {"bbox": lone["bbox"], "face_id": 9} and len(t) == PB.MAX_TARGETS


def _run(monkeypatch, answers, desc="A boy in blue runs while a boy in green watches."):
    """answers: per call, phrase -> the box the face/body is drawn as. Boxes are
    resolved from what was drawn, so the mock follows the shuffled numbering."""
    drawn = []
    monkeypatch.setattr(PB, "draw_boxes", lambda path, boxes, max_px=0: drawn.append(boxes) or "b64")

    def ask(description, b64, n, model, bodies=False):
        boxes, want = drawn[-1], answers[len(drawn) - 1]
        return [{"phrase": p, "could_be": [boxes.index(bb) + 1], "kind": "person",
                 "all_members_boxed": True} for p, bb in want.items()]
    monkeypatch.setattr(PB, "ask_model", ask)
    trace = []
    slots = PB.bind_with_model("x.jpg", desc, [FACE], 7, "m", bodies=[FACED, BACK], image_h=H, trace=trace)
    return slots, trace


def test_a_phrase_on_a_faceless_body_binds_nothing(monkeypatch):
    both = {"A boy in blue": BACK["bbox"], "a boy in green": FACED["bbox"]}
    slots, trace = _run(monkeypatch, [both, both])
    assert [s["spans"][0]["text"] for s in slots] == ["a boy in green"]
    assert slots[0]["faces"][0]["face_id"] == 1
    assert len(trace) == 2 and trace[0]["face_ids"] != trace[1]["face_ids"]   # re-shuffled


def test_the_two_calls_must_agree(monkeypatch):
    slots, _ = _run(monkeypatch, [{"a boy in green": FACED["bbox"]}, {"a boy in green": BACK["bbox"]}])
    assert slots == []


def test_the_body_prompt_states_the_box_count_and_has_no_example_phrase():
    p = PB.build_prompt("A boy.", 4, bodies=True)
    assert "4 numbered boxes, 1 to 4" in p and "from 1 to 4" in p
    assert "reddish" not in p

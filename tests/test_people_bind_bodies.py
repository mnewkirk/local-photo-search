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


def test_the_json_schema_answer_shape_parses_like_a_bare_array():
    raw = '{"mentions": [{"phrase": "A boy", "could_be": [1], "kind": "person", "all_members_boxed": true}]}'
    assert PB.parse_answer(raw, "A boy runs.", 2) == PB.parse_answer(raw[13:-1], "A boy runs.", 2)


def test_a_whole_sentence_copied_as_a_phrase_never_binds():
    desc = "A young boy in a blue soccer jersey and dark shorts is kicking a soccer ball on a field."
    sentence = {"phrase": desc[:-1], "could_be": [1], "kind": "person", "all_members_boxed": True}
    noun = dict(sentence, phrase="A young boy in a blue soccer jersey")
    by_id = {1: FACE}
    assert PB.decide(desc, [sentence], [1], by_id) == {}
    assert PB.decide(desc, [noun], [1], by_id) == {"A young boy in a blue soccer jersey": ("person", [1])}


def test_a_face_heading_two_bodies_equally_is_contested_and_names_nobody(monkeypatch):
    # two overlapping boys, one face inside both boxes near both heads: nearly equal costs
    left, right = _body(1000, 1100, 3000, 0), _body(1000, 2000, 3000, 900)
    between = _face(5, 1010, 1080, 1200, 920)
    assert PB.contested_faces([between], [left, right], None) == set()
    assert PB.contested_faces([between], [left, right], 0.05) == {5}
    assert PB.contested_faces([FACE], [FACED, BACK], 0.05) == set()   # one body only
    monkeypatch.setattr(PB, "LINK_MARGIN", 0.05)
    assert all(t["face_id"] is None for t in PB.body_targets([between], [left, right], H))


def test_the_link_check_drops_a_slot_whose_face_is_not_its_bodys_own(monkeypatch):
    other = _face(2, 1000, 1700, 1200, 1500)      # drawn inside BACK's box but someone else's
    targets = [{"bbox": FACED["bbox"], "face_id": 1}, {"bbox": BACK["bbox"], "face_id": 2},
               {"bbox": other["bbox"], "face_id": 3}]       # a face drawn as its own box
    slots = [{"slot": "s1", "kind": "person", "spans": [], "faces": [{"face_id": 1, "bbox": FACE["bbox"]}]},
             {"slot": "s2", "kind": "person", "spans": [], "faces": [{"face_id": 2, "bbox": other["bbox"]}]},
             {"slot": "s3", "kind": "person", "spans": [], "faces": [{"face_id": 3, "bbox": other["bbox"]}]}]
    asked = []
    monkeypatch.setattr(PB, "check_link", lambda img, body, face, model: asked.append(body) or body == FACED["bbox"])
    kept = PB.link_checked("x.jpg", slots, targets, "m")
    assert [s["slot"] for s in kept] == ["s1", "s3"] and len(asked) == 2   # s3 needs no check
    monkeypatch.setattr(PB, "check_link", lambda *a: None)
    assert PB.link_checked("x.jpg", slots, targets, "m") is None          # no answer: defer


def test_a_recased_phrase_takes_the_descriptions_own_text_only_when_unique():
    def ans(ph):
        return '{"mentions": [{"phrase": "%s", "could_be": [1], "kind": "person", "all_members_boxed": true}]}' % ph
    assert PB.parse_answer(ans("the kids"), "The kids run.", 1)[0]["phrase"] == "The kids"
    assert PB.parse_answer(ans("a  boy"), "A boy runs.", 1)[0]["phrase"] == "A boy"
    import pytest
    from photosearch.describe import UnusableAnswer
    with pytest.raises(UnusableAnswer):                     # twice: which one?
        PB.parse_answer(ans("A BOY"), "A boy runs and a boy waits.", 1)
    with pytest.raises(UnusableAnswer):                     # a paraphrase stays a failure
        PB.parse_answer(ans("a person"), "A boy runs.", 1)


def test_a_bare_phrase_binds_only_when_it_names_everyone_detected():
    for bare in ("One boy", "two young players", "A man", "the other child"):
        assert PB.is_bare_phrase(bare), bare
    for detailed in ("One boy in a neon yellow jersey", "the goalkeeper", "two players in blue"):
        assert not PB.is_bare_phrase(detailed), detailed
    m = {"phrase": "One boy", "could_be": [1], "kind": "person", "all_members_boxed": True}
    by_id = {1: FACE}
    assert PB.decide("One boy runs.", [m], [1], by_id, people=1, bare_rule="everyone")
    assert not PB.decide("One boy runs.", [m], [1], by_id, people=3, bare_rule="everyone")
    assert PB.decide("One boy runs.", [m], [1], by_id, people=3, bare_rule=None)     # rule off
    assert PB.bare_allowed("Two boys", 2, 2, "everyone") and not PB.bare_allowed("Two boys", 2, 2, "strict")
    assert PB.bare_allowed("Two boys in blue", 2, 7, "strict")                     # not bare


def test_the_rule_tier_drops_a_bare_slot_in_a_crowded_photo():
    slot = {"slot": "s1", "kind": "person", "faces": [{"face_id": 1, "bbox": FACE["bbox"]}],
            "spans": [{"start": 0, "end": 11, "text": "A young boy"}]}
    assert PB.filter_bare([slot], 3, "everyone") == []
    assert PB.filter_bare([slot], 1, "everyone") == [slot]
    assert PB.filter_bare([slot], None, "everyone") == [slot]                       # no bodies known

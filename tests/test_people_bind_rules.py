"""people-bind rule tier + VLM-tier decision logic (docs/plans/description-names.md).

Pure functions only: no model, no DB. The 248080 fixture is the real example
that motivated the plan."""
import pytest

from photosearch import people_bind as PB

D_248080 = ("A young boy with reddish hair walks across a grassy field wearing a "
            "neon green jersey, dark shorts, and blue socks. He is positioned in "
            "front of a chain-link fence that separates the field from metal "
            "bleachers and trees in the background. The setting appears to be an "
            "outdoor sports field, likely during a practice or game.")
FACE_248080 = {"id": 516669, "bbox": [1936, 3387, 2228, 3153], "det_score": 0.85}
BOX_248080 = [{"label": "person", "bbox": [0.43945, 0.40996, 0.61523, 0.93265],
               "area_frac": 0.09188}]
W, H = 6528, 4352


def _face(fid, top, right, bottom, left, det=0.9):
    return {"id": fid, "bbox": [top, right, bottom, left], "det_score": det}


def test_248080_binds_the_whole_noun_phrase_with_its_hair_tail():
    slots = PB.rule_bind(D_248080, [FACE_248080], BOX_248080, W, H)
    assert slots is not None and len(slots) == 1
    s = slots[0]
    assert s["kind"] == "person"
    assert [f["face_id"] for f in s["faces"]] == [516669]
    sp = s["spans"][0]
    assert D_248080[sp["start"]:sp["end"]] == "A young boy with reddish hair" == sp["text"]


def test_no_subject_boxes_sends_the_photo_to_the_vlm():
    # the lone detected face may be a sideline parent while the described boy's
    # face was undetected: NULL subject_boxes is NOT trusted
    assert PB.rule_bind(D_248080, [FACE_248080], None, W, H) is None
    assert PB.rule_bind(D_248080, [FACE_248080], [], W, H) is None


def test_face_outside_the_person_box_is_refused():
    parent = _face(9, 100, 400, 500, 100)       # far left, nowhere near the boy
    assert PB.rule_bind(D_248080, [parent], BOX_248080, W, H) is None


def test_subject_box_must_be_labelled_a_person():
    box = [dict(BOX_248080[0], label="ball")]
    assert PB.rule_bind(D_248080, [FACE_248080], box, W, H) is None


@pytest.mark.parametrize("desc", [
    "A boy and a girl kick a ball on a field.",
    "A young boy stands near several spectators on the sideline.",
    "A boy runs while another player watches.",
    "Two players in blue chase the ball.",
    "A boy talks to the goalkeeper in the box.",       # two different person nouns
    "A boy with the team behind him.",
    "A boy and his father walk off the field.",
])
def test_crowds_and_second_people_refuse(desc):
    assert PB.rule_bind(desc, [FACE_248080], BOX_248080, W, H) is None


def test_two_offered_faces_refuse():
    other = _face(2, 1900, 5000, 2200, 4700)
    assert PB.rule_bind(D_248080, [FACE_248080, other], BOX_248080, W, H) is None


def test_small_or_low_confidence_faces_are_not_offered():
    tiny = _face(3, 0, 40, 40, 0)
    weak = _face(4, 0, 300, 300, 0, det=0.4)
    assert PB.offered_faces([tiny, weak]) == []
    # NULL det_score is grandfathered
    assert len(PB.offered_faces([_face(5, 0, 300, 300, 0, det=None)])) == 1
    # a tiny face does not count against the single-face rule
    slots = PB.rule_bind(D_248080, [FACE_248080, tiny], BOX_248080, W, H)
    assert slots is not None


def test_possessive_stays_outside_the_span():
    d = "The boy's jersey is neon green as the boy runs across the grass."
    slots = PB.rule_bind(d, [FACE_248080], BOX_248080, W, H)
    assert slots is not None
    texts = [sp["text"] for sp in slots[0]["spans"]]
    assert texts == ["The boy", "the boy"]
    assert d[slots[0]["spans"][0]["end"]:].startswith("'s")


def test_unrecognised_person_noun_is_not_this_tiers_call():
    assert PB.rule_bind("The midfielder dribbles past a cone.", [FACE_248080],
                        BOX_248080, W, H) is None


# --- VLM tier: the code decides, not the model ---------------------------------

DESC = "A boy in blue kicks the ball. Two players in white chase him on the field."
FACES = {1: {"face_id": 1, "bbox": [0, 100, 100, 0]},
         2: {"face_id": 2, "bbox": [0, 300, 100, 200]},
         3: {"face_id": 3, "bbox": [0, 500, 100, 400]}}


def _mention(phrase, could_be, kind="person", boxed=True):
    return {"phrase": phrase, "could_be": could_be, "kind": kind,
            "all_members_boxed": boxed}


def test_singular_phrase_binds_only_on_exactly_one_candidate():
    order = [3, 1, 2]              # box 1 is face 3, ...
    got = PB.decide(DESC, [_mention("A boy in blue", [2])], order, FACES)
    assert got == {"A boy in blue": ("person", [1])}
    # three equally plausible boxes: ambiguous by construction -> generic
    assert PB.decide(DESC, [_mention("A boy in blue", [1, 2, 3])], order, FACES) == {}
    assert PB.decide(DESC, [_mention("A boy in blue", [])], order, FACES) == {}


def test_counted_group_needs_exactly_n_boxes_and_all_members_boxed():
    order = [1, 2, 3]
    ok = PB.decide(DESC, [_mention("Two players in white", [2, 3], "group")], order, FACES)
    assert ok == {"Two players in white": ("group", [2, 3])}
    assert PB.decide(DESC, [_mention("Two players in white", [2], "group")], order, FACES) == {}
    assert PB.decide(DESC, [_mention("Two players in white", [2, 3], "group", boxed=False)],
                     order, FACES) == {}
    # an uncounted plural has no knowable size
    assert PB.decide("Players chase the ball.", [_mention("Players", [1, 2], "group")],
                     order, FACES) == {}


def test_a_phrase_that_occurs_twice_is_never_bound():
    d = "A boy in blue waits. Later a boy in blue scores."
    assert PB.decide(d, [_mention("a boy in blue", [1])], [1, 2, 3], FACES) == {}
    d2 = "A boy waits. A boy scores."
    assert PB.decide(d2, [_mention("A boy", [1])], [1, 2, 3], FACES) == {}


def test_kind_group_cannot_bind_one_person():
    assert PB.decide(DESC, [_mention("A boy in blue", [1], "group")], [1, 2, 3], FACES) == {}


def test_two_calls_must_agree_on_the_face_set():
    a = PB.decide(DESC, [_mention("A boy in blue", [1])], [1, 2, 3], FACES)
    b = PB.decide(DESC, [_mention("A boy in blue", [1])], [2, 3, 1], FACES)   # a different face!
    assert PB.merge_calls(DESC, [a, b], FACES) == []
    c = PB.decide(DESC, [_mention("A boy in blue", [3])], [2, 3, 1], FACES)   # the same face
    slots = PB.merge_calls(DESC, [a, c], FACES)
    assert [f["face_id"] for f in slots[0]["faces"]] == [1]


def test_one_face_never_lands_in_two_slots():
    d = "A boy waits while a girl watches."
    m = [_mention("A boy", [1]), _mention("a girl", [1])]
    got = PB.decide(d, m, [1, 2, 3], FACES)
    assert len(PB.merge_calls(d, [got], FACES)) == 1


def test_shuffled_orders_differ_and_are_permutations():
    ids = [11, 12, 13, 14]
    a, b = PB.shuffled_orders(ids, seed=42)
    assert sorted(a) == sorted(b) == ids and a != b
    assert PB.shuffled_orders(ids, 42) == (a, b)            # reproducible
    a2, b2 = PB.shuffled_orders([5, 6], seed=1)
    assert a2 != b2 and sorted(a2) == [5, 6]


# --- the parser: bad model output is a failure, not a result ----------------------

def test_parse_accepts_fenced_json_and_prose_wrapping():
    raw = 'Sure!\n```json\n[{"phrase": "A boy in blue", "could_be": [1], "kind": "person"}]\n```'
    got = PB.parse_answer(raw, DESC, 3)
    assert got[0]["phrase"] == "A boy in blue" and got[0]["could_be"] == [1]
    assert PB.parse_answer("[]", DESC, 3) == []


@pytest.mark.parametrize("raw", [
    "I cannot see boxes", "{}", '[{"phrase": "A girl", "could_be": [1]}]',   # not in text
    '[{"phrase": "A boy in blue", "could_be": [9]}]',                         # out of range
    '[{"phrase": "A boy in blue", "could_be": "1"}]',
    '[{"phrase": "A boy in blue", "could_be": [1], "kind": "dog"}]',
    '[["A boy in blue"]]',
])
def test_unusable_answers_raise(raw):
    from photosearch.describe import UnusableAnswer
    with pytest.raises(UnusableAnswer):
        PB.parse_answer(raw, DESC, 3)


# --- the submit-side validator --------------------------------------------------------

def _slot(**kw):
    s = {"slot": "s1", "kind": "person",
         "spans": [{"start": 0, "end": 13, "text": "A boy in blue"}],
         "faces": [{"face_id": 1, "bbox": [0, 100, 100, 0]}]}
    s.update(kw)
    return s


def test_validate_accepts_a_good_binding_and_rejects_every_bad_one():
    assert PB.validate_slots(DESC, [_slot()], {1, 2, 3})
    bad = [
        _slot(spans=[{"start": 0, "end": 13, "text": "A girl in red"}]),    # not the text
        _slot(spans=[{"start": 0, "end": 999, "text": "x"}]),               # out of range
        _slot(faces=[{"face_id": 99, "bbox": [0, 1, 1, 0]}]),               # not this photo's
        _slot(kind="group"),                                                # group of one
        _slot(kind="nonsense"),
        _slot(faces=[]),
    ]
    for s in bad:
        with pytest.raises(PB.InvalidBinding):
            PB.validate_slots(DESC, [s], {1, 2, 3})
    with pytest.raises(PB.InvalidBinding):                                  # face bound twice
        PB.validate_slots(DESC, [_slot(), _slot(slot="s2", spans=[
            {"start": 29, "end": 40, "text": DESC[29:40]}])], {1, 2, 3})
    with pytest.raises(PB.InvalidBinding):                                  # spans overlap
        PB.validate_slots(DESC, [_slot(), _slot(slot="s2", spans=[
            {"start": 5, "end": 10, "text": DESC[5:10]}],
            faces=[{"face_id": 2, "bbox": [0, 1, 1, 0]}])], {1, 2, 3})

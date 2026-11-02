"""People context in describe (docs/plans/people-context-describe.md): the pure pieces the
G0/G1 evals run on — identify (sessions, bursts, tiers, conflicts), body_attrs parsing and
grounded_describe inline anchors."""
import datetime as dt

import pytest

from photosearch import body_attrs as BA
from photosearch import grounded_describe as GD
from photosearch import identify as ID
from photosearch.describe import UnusableAnswer

T0 = dt.datetime(2026, 9, 12, 10, 0, 0)


def _t(s):
    return T0 + dt.timedelta(seconds=s)


# --- sessions / bursts -------------------------------------------------------

def test_sessions_split_on_gap_and_day_and_folder():
    rows = [(1, "a", "2026-09-12 10:00:00"), (2, "a", "2026-09-12 10:40:00"),
            (3, "a", "2026-09-12 11:30:00"), (4, "a", "2026-09-13 09:00:00"),
            (5, "b", "2026-09-12 10:01:00"), (6, "a", None)]
    s = ID.sessions(rows)
    assert s[1] == s[2] != s[3]           # 40 min joins, 50 min splits
    assert s[4] != s[1] and s[5] != s[1]  # another day / folder
    assert 6 not in s                     # undated: no session


def test_bursts_chain_by_gap():
    b = ID.bursts({1: _t(0), 2: _t(2), 3: _t(4.5), 4: _t(9)})
    assert b[1] == b[2] == b[3] != b[4]


# --- identify ----------------------------------------------------------------

def _body(x, face=None, label=None, attrs=None, h=600):
    return {"bbox": [100, x + 200, 100 + h, x], "face_person": face, "label_person": label, "attrs": attrs}


def test_face_tier_and_one_person_once_per_photo():
    photos = [{"photo_id": 1, "time": _t(0), "W": 3000,
               "bodies": [_body(0, face=7), _body(500, face=7)]}]
    got = ID.identify_session(photos)
    assert (1, 0) not in got and (1, 1) not in got    # same person offered twice: nobody


def test_burst_tier_follows_the_camera():
    # frame 2 is frame 1 shifted 300 px right; body 0 in frame 2 has no face
    a = {"photo_id": 1, "time": _t(0), "W": 3000, "bodies": [_body(0, face=7), _body(600, face=8)]}
    b = {"photo_id": 2, "time": _t(1), "W": 3000, "bodies": [_body(300), _body(900)]}
    motion = {(2, 1): (1.0, -300.0, 0.0), (1, 2): (1.0, 300.0, 0.0)}
    got = ID.identify_session([a, b], motion=motion)
    assert got[(2, 0)]["person"] == 7 and got[(2, 0)]["tier"] == "burst"
    assert got[(2, 1)]["person"] == 8
    # without measured motion, several bodies: no burst track
    got = ID.identify_session([a, b])
    assert (2, 0) not in got


def test_burst_tier_hidden_face_seeds_nothing():
    a = {"photo_id": 1, "time": _t(0), "W": 3000, "bodies": [_body(0, face=7)]}
    b = {"photo_id": 2, "time": _t(1), "W": 3000, "bodies": [_body(0)]}
    got = ID.identify_session([a, b], hide_faces={(1, 0)})
    assert (2, 0) not in got and (1, 0) not in got


def _kit(n):
    return {"kit": "yellow", "number": str(n)}


def test_kit_tier_needs_two_other_bursts_and_no_contradiction():
    def ph(pid, s, face, n):
        return {"photo_id": pid, "time": _t(s), "W": 3000, "bodies": [_body(0, face=face, attrs=_kit(n))]}
    target = {"photo_id": 9, "time": _t(100), "W": 3000, "bodies": [_body(0, attrs=_kit(5))]}
    one = [ph(1, 0, 7, 5), target]
    assert (9, 0) not in ID.identify_session(one)             # one anchor burst
    two = [ph(1, 0, 7, 5), ph(2, 50, 7, 5), target]
    got = ID.identify_session(two)
    assert got[(9, 0)]["person"] == 7 and got[(9, 0)]["tier"] == "kit"
    contra = two + [ph(3, 200, 8, 5)]
    assert (9, 0) not in ID.identify_session(contra)          # someone else wore 5 too
    assert (9, 0) not in ID.identify_session(two, enabled=("face", "burst"))


def test_duplicate_signature_in_one_photo_voids_it():
    def ph(pid, s, face, n):
        return {"photo_id": pid, "time": _t(s), "W": 3000, "bodies": [_body(0, face=face, attrs=_kit(n))]}
    dup = {"photo_id": 4, "time": _t(300), "W": 3000,
           "bodies": [_body(0, attrs=_kit(5)), _body(800, attrs=_kit(5))]}
    target = {"photo_id": 9, "time": _t(100), "W": 3000, "bodies": [_body(0, attrs=_kit(5))]}
    assert (9, 0) not in ID.identify_session([ph(1, 0, 7, 5), ph(2, 50, 7, 5), dup, target])


def test_cue_disagreement_contests_the_body():
    def ph(pid, s, face, n):
        return {"photo_id": pid, "time": _t(s), "W": 3000, "bodies": [_body(0, face=face, attrs=_kit(n))]}
    target = {"photo_id": 9, "time": _t(100), "W": 3000, "bodies": [_body(0, face=8, attrs=_kit(5))]}
    got = ID.identify_session([ph(1, 0, 7, 5), ph(2, 50, 7, 5), target])
    assert (9, 0) not in got and (9, 0) in got[None]["contested"]
    labelled = dict(target, bodies=[dict(target["bodies"][0], label_person=8)])
    assert ID.identify_session([ph(1, 0, 7, 5), ph(2, 50, 7, 5), labelled])[(9, 0)]["tier"] == "label"


def test_frame_transform_needs_half_the_bodies():
    a = [_body(0), _body(600), _body(1200)]
    b = [_body(300), _body(5000), _body(9000)]
    t, m = ID.frame_transform(a, b, hint=(1.0, 300.0, 0.0), tol=100)
    assert t is None and m == {}                               # 1 of 3 lines up


def test_phase_shift_measures_a_pan():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    big = rng.random((200, 300)).astype("float32")
    a, b = big[:, 20:276], big[:, 0:256]       # b's content sits 20 px right of a's
    _, dx, dy, peak = ID.phase_shift(a, b)
    assert (round(dx), round(dy)) == (20, 0) and peak > 0.3


# --- body_attrs ---------------------------------------------------------------

def _ans(**kw):
    import json
    d = {"role": "player", "facing": "away", "kit": "yellow", "number": "07", "hair_colour": "blond",
         "hair_length": "short", "shoes": "white"}
    d.update(kw)
    return json.dumps(d)


def test_body_attrs_parse_normalises_numbers():
    assert BA.parse(_ans())["number"] == "7"
    assert BA.parse(_ans(number="#12"))["number"] == "12"
    assert BA.parse(_ans(number="Unreadable"))["number"] == "unreadable"
    assert BA.parse(_ans(number=""))["number"] == "none"


@pytest.mark.parametrize("bad", [dict(number="12 or 17"), dict(kit="chartreuse"), dict(role="fan")])
def test_body_attrs_parse_rejects_out_of_vocabulary(bad):
    with pytest.raises(UnusableAnswer):
        BA.parse(_ans(**bad))


def test_signature_needs_a_number_and_groups_kits():
    assert ID.signature({"kit": "neon-yellow", "number": "5"}) == ("yellow", "5")
    assert ID.signature({"kit": "yellow", "number": "none"}) is None
    assert ID.signature({"kit": "not-visible", "number": "5"}) is None


# --- grounded_describe ----------------------------------------------------------

def test_inline_anchors_strip_to_spans_that_match_by_construction():
    raw = "[A boy in a yellow jersey](2) kicks the ball while [two girls](1, 3) watch from the side."
    r = GD.parse(raw, 3)
    d = r["description"]
    assert d == "A boy in a yellow jersey kicks the ball while two girls watch from the side."
    for sp in r["spans"]:
        assert d[sp["start"]:sp["end"]] == sp["phrase"]
    assert [sp["boxes"] for sp in r["spans"]] == [[2], [1, 3]]


def test_bad_anchors_are_stripped_without_a_span():
    r = GD.parse("[ten children](1-10) play and [a coach](9) watches [four boys](1,2,3,4).", 5)
    assert r["spans"] == []
    assert r["description"] == "Ten children play and a coach watches four boys."
    assert len(r["dropped"]) == 3


@pytest.mark.parametrize("raw", ["Two boys in numbered boxes play.", "A [boy runs.", "   "])
def test_unusable_grounded_answers(raw):
    with pytest.raises(UnusableAnswer):
        GD.parse(raw, 2)


def test_hint_lines_carry_no_names():
    assert GD.hint_line(2, "male", 10) == "- box 2: male, a child aged about 9-11"
    assert GD.hint_line(1, None, None) is None
    p = GD.build_prompt(2, "two", True, [GD.hint_line(2, "female", 40)])
    assert "box 2: female, an adult" in p and "first image" in p


@pytest.mark.parametrize("phrase,n,bad", [("two boys in blue", 1, True), ("two boys", 2, False),
                                          ("a boy", 2, True), ("boys in blue", 1, True),
                                          ("the goalkeeper", 1, False), ("three others", 3, False)])
def test_count_problem(phrase, n, bad):
    assert bool(GD.count_problem(phrase, n)) is bad


def test_count_mismatch_drops_the_span_keeps_the_text():
    r = GD.parse("[Two boys in blue](5) run.", 5)
    assert r["spans"] == [] and r["description"] == "Two boys in blue run."


def test_degenerate_marks():
    r = GD.parse("A young boy [a boy in a blue jersey](1) kicks.", 2)
    assert r["degenerate"]
    r = GD.parse("Four of them [a boy](1), [a boy](2) and [a boy](3) stand.", 3)
    assert r["degenerate"]
    assert GD.parse("[A boy in blue](1) kicks while [a girl](2) watches.", 2)["degenerate"] is None


def test_names_as_context_are_never_written():
    p = GD.build_prompt(2, "two", True, context=[GD.context_line(1, "Asa Goldman", "male", 10)])
    assert "box 1: Asa Goldman (boy, 10)" in p and "known and correct" in p
    with pytest.raises(UnusableAnswer, match="name in text"):
        GD.parse("[Asa](1) kicks the ball.", 2, ["Asa Goldman"])
    with pytest.raises(UnusableAnswer, match="name in text"):
        GD.parse("[The Goldman boy](1) kicks the ball.", 2, ["Asa Goldman"])
    assert GD.parse("[A boy](1) kicks the ball.", 2, ["Asa Goldman"])["spans"]


def test_person_noun():
    assert GD.person_noun("male", 10) == "a boy aged about 10"
    assert GD.person_noun("female", 40) == "a woman"
    assert GD.person_noun("male", None) == "male"

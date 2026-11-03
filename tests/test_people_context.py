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


# --- grounded describe v2: person-id handles (docs/plans/people-context-describe-v2.md) ---

R2 = {4: {"person_id": 39, "face_id": 901, "body_bbox": [1, 2, 3, 0], "gender": "male", "age": 10},
      2: {"person_id": 12, "face_id": 902, "body_bbox": [5, 6, 7, 4], "gender": "female", "age": 9}}


def test_v2_grammar_reads_boxes_and_handles():
    r = GD.parse_v2("[A boy in green](4:P39) passes to [two players](1, 2:P12) on a field.", 5, R2)
    a, b = r["spans"]
    assert a["written"] == {4: 39} and a["boxes"] == [4] and a["check"] == "agree"
    assert b["written"] == {1: None, 2: 12} and b["check"] == "agree"
    assert r["description"] == "A boy in green passes to two players on a field."


@pytest.mark.parametrize("inner,cls", [
    ("4:P39", "agree"), ("4", "omitted"), ("4:P12", "disagree"), ("1:P39", "orphan"),
    ("4:P7", "foreign"), ("2:P0", "foreign"), ("1", "none"), ("1,4:P39", "agree"),
    ("2:P39", "disagree"), ("1:P0,4:P39", "foreign")])
def test_v2_cross_check_classes(inner, cls):
    sp = GD.parse_v2(f"[a child]({inner}) runs on the grass.", 5, R2)["spans"]
    assert sp[0]["check"] == cls


def test_v2_two_disagreements_flag_roster_confusion():
    r = GD.parse_v2("[a boy](4:P12) and [a girl](2:P39) run.", 5, R2)
    assert r["roster_confused"] and r["counts"] == {"disagree": 2}


@pytest.mark.parametrize("raw", [
    "P39 kicks the ball on a field.",
    "[a boy](4:P39) passes to person 2 on a field.",
    "[P39](4:P39) kicks the ball on a field.",          # a bare id in brackets
    "A boy (4:P39) kicks the ball on a field.",           # a mark without square brackets
])
def test_v2_id_in_prose_raises(raw):
    with pytest.raises(UnusableAnswer):
        GD.parse_v2(raw, 5, R2)


def test_v2_bare_id_message_is_id_in_text():
    with pytest.raises(UnusableAnswer, match="id in text"):
        GD.parse_v2("[P39](4:P39) kicks the ball.", 5, R2)


def test_v2_bad_inner_is_dropped_text_kept():
    r = GD.parse_v2("[a boy](P39) and [a girl](9) run.", 5, R2)
    assert r["spans"] == [] and len(r["dropped"]) == 2
    assert r["description"] == "A boy and a girl run."


def test_v2_names_and_overlay_words_raise():
    with pytest.raises(UnusableAnswer, match="name in text"):
        GD.parse_v2("[Asa](4:P39) runs.", 5, R2, names=["Asa Goldman"])
    with pytest.raises(UnusableAnswer, match="overlay"):
        GD.parse_v2("[a boy](4:P39) in the red box runs.", 5, R2)


def test_v2_strip_equals_unmarked_sentence():
    marked = "[A boy in a green jersey](4:P39) dribbles past [two defenders](1,3) on a grass field."
    plain = "A boy in a green jersey dribbles past two defenders on a grass field."
    assert GD.parse_v2(marked, 5, R2)["description"] == plain
    assert GD.parse_v2(plain, 5, R2)["description"] == plain


def test_v2_strip_fallback_drops_stray_marks():
    r = GD.strip_marks_v2("A boy [4:P39] runs past a girl [2].", 5, R2)
    assert r["description"] == "A boy runs past a girl." and r["spans"] == []
    assert GD.strip_marks_v2("A referee (1) watches.", 5, R2)["description"] == "A referee watches."
    r = GD.strip_marks_v2("[A boy](1) passes to a teammate (3) and two others ([4] and [5]).", 5, R2)
    assert r["description"] == "A boy passes to a teammate and two others." and r["spans"] == []
    r = GD.strip_marks_v2("One boy (box 2) waves, and a girl stands in box (5), smiling.", 5, R2)
    assert r["description"] == "One boy waves, and a girl stands, smiling."
    r = GD.strip_marks_v2("Two people sit at boxes (4) and (5).", 5, R2)
    assert r["description"] == "Two people sit."
    with pytest.raises(UnusableAnswer):                     # the id is never stripped silently
        GD.strip_marks_v2("A referee (1:P39) watches.", 5, R2)


@pytest.mark.parametrize("raw,ok", [
    ("[a girl in green](4:P39) runs.", False),
    ("[a child in green](4:P39) runs as she smiles.", False),
    ("[a child in green](4:P39) runs past [a girl](2:P12).", True),     # the window stops at the next mark
    ("[a child in green](4:P39) runs. She smiles.", True),              # ... and at the sentence end
    ("[a man in green](4:P39) runs.", False),                           # age 10
    ("[a boy in green](4:P39) runs.", True),
])
def test_v2_roster_ok(raw, ok):
    assert GD.parse_v2(raw, 5, R2)["spans"][0]["roster_ok"] is ok


def test_v2_roster_ok_groups_read_plural_nouns_only():
    sp = GD.parse_v2("[two girls](2:P12,4:P39) run.", 5, R2)["spans"][0]
    assert not sp["roster_ok"]
    sp = GD.parse_v2("[a boy and a girl](2:P12,4:P39) run.", 5, R2)["spans"][0]
    assert sp["roster_ok"]
    adult = {1: {"person_id": 5, "gender": "female", "age": 40}}
    assert not GD.parse_v2("[a girl](1:P5) waves.", 3, adult)["spans"][0]["roster_ok"]


def _named(raw, people=None, roster=R2):
    r = GD.parse_v2(raw, 5, roster)
    return r, GD.name_spans(r["spans"], roster, people=people)


def test_v2_naming_rules():
    r, sp = _named("[A boy in green](4:P39) passes to [a girl in red](2:P12) on a field.")
    assert [s["named"] for s in sp] == [True, True]
    assert sp[0]["person_ids"] == [39] and sp[0]["face_ids"] == [901] and sp[0]["body_bboxes"] == [[1, 2, 3, 0]]
    _, sp = _named("[A boy in green](4) runs.")
    assert sp[0]["why_generic"] == "omitted"
    _, sp = _named("[A boy in green](4:P39) passes, and [the boy in green](4:P39) scores.")
    assert [s["named"] for s in sp] == [True, False] and sp[1]["why_generic"] == "already named"
    _, sp = _named("[a girl in green](4:P39) runs.")
    assert sp[0]["why_generic"] == "roster conflict"


def test_v2_groups_all_or_nothing():
    _, sp = _named("[two kids](2:P12,4:P39) run.")
    assert sp[0]["named"] and sp[0]["person_ids"] == [12, 39]
    _, sp = _named("[two kids](1,4:P39) run.")             # an unknown member: generic
    assert not sp[0]["named"] and sp[0]["why_generic"] == "unknown member"
    _, sp = _named("[two kids](2,4:P39) run.")             # a member omitted: generic
    assert sp[0]["why_generic"] == "omitted"


def test_v2_bare_phrase_rule():
    _, sp = _named("[A boy](4:P39) runs.", people=3)
    assert sp[0]["why_generic"] == "bare phrase"
    _, sp = _named("[A boy](4:P39) runs.", people=1, roster={4: R2[4]})
    assert sp[0]["named"]


def test_v2_template_and_all_generic_identity():
    r, sp = _named("[A boy in green](4:P39) passes to [two kids](1,2:P12) on a field.")
    assert GD.assemble_v2(r["description"], sp) == \
        "{person:39|A boy in green} passes to two kids on a field."
    r, sp = _named("[A boy in green](4:P12) passes to [two kids](1,3) on a field.")
    assert not any(s["named"] for s in sp)
    assert GD.assemble_v2(r["description"], sp) == r["description"]


def test_v2_no_roster_prompt_is_the_no_context_prompt():
    # nobody known: no KNOWN block, no reminder and no P-numbers (dev: P0 on unknown people)
    p = GD.build_prompt_v2(4, None)
    assert p == GD.build_prompt_v2(4, {}) == (GD.V2_HEAD.format(n=4) + GD.V2_PHOTO_RULES + GD._V2_MARK_OPEN
                                             + GD._V2_EXAMPLE_BOX + GD._V2_RULES_BOX)
    assert "KNOWN" not in p and "Reminder" not in p and "P0" not in p and "P-number" not in p
    assert GD.build_prompt_v2(4, None, drawn=True) == p


def test_v2_prompt_roster_first_and_reminder():
    p = GD.build_prompt_v2(5, R2)
    assert p.index("KNOWN PEOPLE") < p.index("Describe the first image") < p.index("Reminder:")
    assert "- box 2: P12, girl, 9\n- box 4: P39, boy, 10\n" in p
    assert "P39 (box 4) is a boy, 10" in p
    last = GD.build_prompt_v2(5, R2, placement="last")
    assert "Reminder" not in last and last.index("Mark people") < last.index("KNOWN PEOPLE")
    box = GD.build_prompt_v2(5, R2, handles=False)
    assert "- box 4: boy, 10" in box and "P39" not in box and "P0" not in box
    assert "grey coat" not in GD.build_prompt_v2(5, R2, example=False)


def test_v2_roster_line_without_facts_keeps_the_handle():
    assert GD.roster_line(4, {"person_id": 39}) == "- box 4: P39"
    assert GD.roster_line(4, {"person_id": 39, "gender": "female", "age": 35}) == "- box 4: P39, woman, 35"
    assert GD.roster_line(4, {"person_id": 39, "gender": "male"}, handles=False) == "- box 4: male"
    assert "P39 is in box 4" in GD.build_prompt_v2(5, {4: {"person_id": 39}})


def test_v2_prompt_version_is_stable_sha():
    assert len(GD.PROMPT_VERSION) == 12 and GD.PROMPT_VERSION == GD._prompt_version()


def test_v2_fix_roster_prompt():
    s = GD.fix_roster_prompt(4, R2[4], "a girl in green")
    assert s.startswith("P39 (box 4) is a boy, 10. You wrote 'a girl in green'")
    s = GD.fix_roster_prompt(4, R2[4], "a girl", handles=False, why="female word 'girl' on a male")
    assert s.startswith("The person (box 4) is a boy. You wrote 'a girl'")


def test_v2_draw_boxes_labelled(tmp_path):
    import base64
    import io
    from PIL import Image
    p = tmp_path / "a.jpg"
    Image.new("RGB", (400, 300), (0, 120, 0)).save(p)
    b64 = GD.draw_boxes_labelled(str(p), [[50, 150, 250, 40], [60, 350, 280, 220]], {2: "P39"})
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert im.size == (400, 300)


# --- the v2 eval harness: flip / swap rosters --------------------------------

def _harness():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "evals", "people_context_eval.py")
    spec = importlib.util.spec_from_file_location("people_context_eval_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_v2_sent_roster_flip_and_swap():
    E = _harness()
    roster = {1: {"person_id": 39, "face_id": 1, "body_bbox": [0, 100, 400, 0], "label_source": "manual",
                  "gender": "male", "age": 10},
              3: {"person_id": 12, "face_id": 2, "body_bbox": [0, 300, 200, 200], "label_source": "strict",
                  "gender": "female", "age": 9}}
    same = E.v2_sent_roster(roster, None)
    assert same[1]["gender"] == "male" and not same[1]["flipped"]
    f = E.v2_sent_roster(roster, "gender")
    assert f[1]["gender"] == "female" and f[1]["flipped"] is True and f[3]["gender"] == "female"
    assert roster[1]["gender"] == "male"                     # the true roster is untouched
    s = E.v2_sent_roster(roster, "swap")
    assert s[1]["person_id"] == 12 and s[3]["person_id"] == 39
    assert s[1]["body_bbox"] == roster[1]["body_bbox"]       # boxes stay; people move
    assert E.v2_sent_roster({1: roster[1]}, "swap") is None
    assert E.v2_sent_roster({1: dict(roster[1], gender=None)}, "gender") is None


def test_v2_age_copy_raises_only_with_a_roster():
    with pytest.raises(UnusableAnswer, match="age copy"):
        GD.parse_v2("[A boy, 10](4) runs on the grass.", 5, R2)
    with pytest.raises(UnusableAnswer, match="age copy"):
        GD.parse_v2("[a boy in green](4) passes to a boy aged 10.", 5, R2)
    assert GD.parse_v2("[A boy, 10](4) runs.", 5, None)["description"] == "A boy, 10 runs."


def test_v2_mark_rules_ask_for_the_whole_phrase():
    for handles in (True, False):
        assert "never [a man](4) in a red cap" in GD.build_prompt_v2(5, R2, handles=handles)

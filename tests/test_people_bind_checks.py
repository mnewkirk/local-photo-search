"""The two checks that ship people-bind (docs/notes/llm-passes.md, "people context:
Step 0"): the keypoint-verified face-to-body link and the overlap second call.
No model, no image: detectors and calls are mocked."""
import json

import pytest

from photosearch import people_bind as PB
from photosearch import person_boxes as PBX
from photosearch.describe import UnusableAnswer

H = 4000
KP_HIDDEN = [0.0, 0.0, -1.0]


def _face(fid, t, r, b, l):
    return {"face_id": fid, "bbox": [t, r, b, l], "det_score": 0.9}


def _kps(head=None, logit=5.0):
    """17 COCO keypoints, all hidden except nose + eyes at ``head`` (x, y)."""
    k = [list(KP_HIDDEN) for _ in range(17)]
    if head is not None:
        for j in PBX.HEAD_KEYPOINTS:
            k[j] = [head[0], head[1], logit]
    return k


def _body(t, r, b, l, head=None, score=0.99, kp=True):
    out = {"bbox": [t, r, b, l], "score": score}
    if kp:
        out["keypoints"] = _kps(head) if head is not None else None
    return out


FACE = _face(1, 1000, 700, 1200, 500)                  # centre (600, 1100)


# --- person_boxes: keypoints ---------------------------------------------------------------
def test_keypoint_detections_attach_one_to_one_by_iou():
    a, b = {"bbox": [0, 100, 300, 0], "score": .9}, {"bbox": [0, 110, 300, 10], "score": .8}
    far = {"bbox": [0, 900, 300, 800], "score": .7}
    k1 = {"bbox": [0, 100, 300, 0], "keypoints": _kps((50, 20))}
    k2 = {"bbox": [0, 105, 300, 5], "keypoints": _kps((55, 20))}
    PBX.attach_keypoints([a, b, far], [k1, k2])
    assert a["keypoints"] == k1["keypoints"] and b["keypoints"] == k2["keypoints"]
    assert far["keypoints"] is None                    # no detection overlaps it


def test_facing_and_head_points_use_the_logit_floor():
    assert PBX.facing_of(None) is None
    assert PBX.facing_of(_kps((1, 1))) == "front"
    assert PBX.facing_of(_kps(None)) == "away"
    side = _kps(None)
    side[1] = [5, 5, 3.0]
    assert PBX.facing_of(side) == "side"
    assert PBX.head_points(_kps((1, 1), logit=PBX.KP_MIN - 0.1)) == []
    assert PBX.head_in_face({"keypoints": _kps((600, 1100))}, FACE["bbox"])
    assert not PBX.head_in_face({"keypoints": _kps((720, 1100))}, FACE["bbox"])   # unpadded box


# --- the keypoint-verified link -------------------------------------------------------------
def test_a_body_whose_own_head_is_in_the_face_box_owns_it():
    own = _body(950, 900, 3000, 300, head=(600, 1100))
    assert PB.keypoint_contested([FACE], [own]) == set()
    assert PB.body_targets([FACE], [own], H)[0]["face_id"] == 1


def test_a_body_whose_head_is_elsewhere_does_not_own_the_face():
    # 247627's shape: the face of a boy behind sits at the top of the front boy's box
    front = _body(950, 900, 3000, 300, head=(600, 1500))
    assert PB.keypoint_contested([FACE], [front]) == {1}
    t = PB.body_targets([FACE], [front], H)
    assert t[0] == {"bbox": front["bbox"], "face_id": None}     # still drawn, names nobody
    assert PB.keypoint_contested([FACE], [_body(950, 900, 3000, 300, head=None)]) == {1}


def test_two_bodies_with_a_head_in_the_face_box_contest_it():
    front = _body(950, 900, 3000, 300, head=(600, 1100))
    behind = _body(900, 1000, 2500, 450, head=(610, 1110), score=0.4)   # below the drawing floor
    assert PB.keypoint_contested([FACE], [front, behind]) == {1}
    lone = _face(2, 100, 9900, 300, 9700)                 # no body: drawn as its own box
    b1 = _body(50, 9950, 3000, 9600, head=(9800, 200), score=0.4)
    b2 = _body(60, 9990, 3000, 9650, head=(9810, 210), score=0.4)
    assert PB.keypoint_contested([lone], [b1]) == set()
    assert PB.keypoint_contested([lone], [b1, b2]) == {2}


def test_a_duplicate_box_of_the_same_person_is_not_a_rival():
    body = _body(950, 900, 3000, 300, head=None)
    dup = _body(960, 880, 2990, 320, head=(600, 1100), score=0.9)     # dedupe merges it
    assert PB.keypoint_contested([FACE], [body, dup]) == set()


def test_bodies_without_keypoints_key_keep_the_geometric_link():
    body = _body(950, 900, 3000, 300, kp=False)
    assert PB.keypoint_contested([FACE], [body]) == set()


# --- the overlap second call ----------------------------------------------------------------
FRONT = [950, 900, 3000, 300]
BEHIND = [900, 1300, 2800, 700]       # 1,100,000 / 1,260,000 of FRONT's area... overlaps


def _slot(fid, text="A boy in blue", kind="person"):
    return {"slot": "s1", "kind": kind, "spans": [{"start": 0, "end": len(text), "text": text}],
            "faces": [{"face_id": fid, "bbox": FACE["bbox"]}]}


def test_rivals_are_other_drawn_people_overlapping_the_chosen_box():
    targets = [{"bbox": FRONT, "face_id": 1}, {"bbox": BEHIND, "face_id": None},
               {"bbox": [0, 5000, 100, 4900], "face_id": 2}]
    assert PB.overlap_rivals(_slot(1), targets) == [(FRONT, BEHIND)]
    group = {"slot": "s1", "kind": "group", "spans": [],
             "faces": [{"face_id": 1, "bbox": []}, {"face_id": 3, "bbox": []}]}
    t2 = [{"bbox": FRONT, "face_id": 1}, {"bbox": BEHIND, "face_id": 3}]
    assert PB.overlap_rivals(group, t2) == []          # a member of the same slot is no rival


def _answers(monkeypatch, letters):
    calls = []
    monkeypatch.setattr(PB, "draw_pair_crop", lambda path, boxes, max_px=0: calls.append(boxes) or "b64")
    def chat(model, messages, options, role):
        calls.append(options["json_schema"])
        a = letters[(len(calls) // 2) - 1]
        return None if a is None else json.dumps({"choice": a})
    monkeypatch.setattr("photosearch.describe._ollama_chat_with_retry", chat)
    return calls


@pytest.mark.parametrize("letters,kept", [
    (["A", "B"], True),             # the chosen box is A, then B: both pick it
    (["A", "A"], False),            # follows the letter, not the person
    (["B", "A"], False),            # picks the rival both times
    (["unsure", "B"], False),
    (["neither", "neither"], False),
])
def test_the_slot_survives_only_if_both_answers_pick_the_chosen_box(monkeypatch, letters, kept):
    _answers(monkeypatch, letters)
    targets = [{"bbox": FRONT, "face_id": 1}, {"bbox": BEHIND, "face_id": None}]
    trace = []
    out = PB.overlap_checked("x.jpg", "A boy in blue runs.", [_slot(1)], targets, "m", trace=trace)
    assert (out == [_slot(1)]) is kept
    assert trace[0]["overlap"]["kept"] is kept


def test_no_answer_defers_and_a_bad_answer_is_unusable(monkeypatch):
    targets = [{"bbox": FRONT, "face_id": 1}, {"bbox": BEHIND, "face_id": None}]
    _answers(monkeypatch, [None, None])
    assert PB.overlap_checked("x.jpg", "d", [_slot(1)], targets, "m") is None
    _answers(monkeypatch, ["both", "both"])              # "both" is not offered for one person
    with pytest.raises(UnusableAnswer):
        PB.overlap_checked("x.jpg", "d", [_slot(1)], targets, "m")


def test_both_is_offered_only_for_a_group_phrase():
    assert "both" not in PB._choice_schema(False)[1]["properties"]["choice"]["enum"]
    assert "both" in PB._choice_schema(True)[1]["properties"]["choice"]["enum"]


def test_a_slot_with_no_overlap_makes_no_call(monkeypatch):
    calls = _answers(monkeypatch, [])
    targets = [{"bbox": FRONT, "face_id": 1}, {"bbox": [0, 9000, 500, 8000], "face_id": None}]
    assert PB.overlap_checked("x.jpg", "d", [_slot(1)], targets, "m") == [_slot(1)] and not calls


# --- bind_photo picks the mode from the claim -----------------------------------------------
def _inp(bodies):
    desc = "A young boy with reddish hair walks across a field."
    face = {"id": 9, "bbox": [1936, 3387, 2228, 3153], "det_score": 0.85}
    return {"description": desc, "description_sha": PB.sha_of(desc),
            "faces": PB.offered_faces([face]), "image_width": 6528, "image_height": 4352,
            "subject_boxes": None, "bodies": bodies}


def test_bind_photo_runs_body_mode_with_the_overlap_check_when_bodies_are_stored(monkeypatch):
    seen = []
    monkeypatch.setattr(PB, "bind_with_model", lambda *a, **k: seen.append(k) or [])
    PB.bind_photo({"id": 1, "people_bind_input": _inp([_body(1800, 3500, 4300, 3000)])},
                  "/tmp/x.jpg", "m")
    assert seen[-1]["overlap_check"] is True and seen[-1]["bodies"] and seen[-1]["image_h"] == 4352
    PB.bind_photo({"id": 1, "people_bind_input": _inp(None)}, "/tmp/x.jpg", "m")
    assert "bodies" not in seen[-1] and not seen[-1].get("overlap_check")     # face mode


def test_bind_photo_rule_tier_applies_the_bare_rule_in_body_mode():
    boxes = [{"label": "person", "bbox": [0.4, 0.4, 0.62, 0.93], "area_frac": 0.1}]
    inp = dict(_inp(None), subject_boxes=boxes,
               description="A young boy walks across a field.")
    inp["description_sha"] = PB.sha_of(inp["description"])
    slots, method = PB.bind_photo({"id": 1, "people_bind_input": inp}, None, "m")
    assert method == "rule" and slots                               # face mode: binds
    crowd = [_body(1800, 3500, 4300, 3000), _body(1000, 1000, 4000, 0), _body(1000, 6000, 4000, 5000)]
    slots, method = PB.bind_photo({"id": 1, "people_bind_input": dict(inp, bodies=crowd)}, None, "m")
    assert method == "rule" and slots == []                         # "A young boy" of three people


def test_the_fleet_binds_with_the_eval_winner():
    from photosearch import rerun
    assert rerun.FLEET_ROLE_MODELS["bind"] == "qwen/qwen3.5-9b"

"""The VLM tier end to end with a scripted model, the overlay, and the worker
loop's handling of each outcome (no LM Studio, no network)."""
import base64
import io

import pytest

from photosearch import people_bind as PB
from photosearch import worker as W
from photosearch.describe import ContextOverflow, TruncatedOutput, UnusableAnswer

DESC = "A boy in blue kicks the ball. Two players in white chase him."
BOX = lambda i: [0, 100 * i + 90, 100, 100 * i]          # 90 px square-ish faces; edge >= 60
FACES = [{"id": 1, "bbox": BOX(0), "det_score": 0.9},
         {"id": 2, "bbox": BOX(2), "det_score": 0.9},
         {"id": 3, "bbox": BOX(4), "det_score": 0.9}]
INP = {"description": DESC, "description_sha": PB.sha_of(DESC),
       "faces": PB.offered_faces(FACES), "image_width": 1000, "image_height": 500,
       "subject_boxes": None}


def _png(tmp_path, size=(600, 300), orientation=None):
    from PIL import Image
    p = tmp_path / "x.jpg"
    im = Image.new("RGB", size, (60, 120, 60))
    if orientation:
        ex = im.getexif()
        ex[0x0112] = orientation
        im.save(p, "JPEG", exif=ex)
    else:
        im.save(p, "JPEG")
    return str(p)


def test_overlay_is_a_jpeg_scaled_to_the_cap(tmp_path):
    from PIL import Image
    path = _png(tmp_path, (3000, 2000))
    b64 = PB.draw_overlay(path, INP["faces"], [3, 1, 2], max_px=1024)
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    assert im.format == "JPEG" and max(im.size) == 1024


def test_overlay_uses_the_exif_oriented_frame(tmp_path):
    from PIL import Image
    path = _png(tmp_path, (600, 300), orientation=6)        # stored sideways, shown 300x600
    im = Image.open(io.BytesIO(base64.b64decode(PB.draw_overlay(path, INP["faces"], [1, 2, 3]))))
    assert im.size == (300, 600)


def _script(monkeypatch, answers):
    """ask_model returns the next scripted list of mentions per call."""
    calls = []

    def fake(description, image_b64, n_boxes, model):
        calls.append(n_boxes)
        return answers[len(calls) - 1]
    monkeypatch.setattr(PB, "ask_model", fake)
    monkeypatch.setattr(PB, "draw_overlay", lambda *a, **k: "b64")
    return calls


def _m(phrase, cb, kind="person"):
    return {"phrase": phrase, "could_be": cb, "kind": kind, "all_members_boxed": True}


def test_vlm_tier_makes_two_calls_and_needs_them_to_agree(monkeypatch):
    # order differs between the calls (reversed), so box numbers differ: face 2 is
    # box 2 in a call using [1,2,3]-permutations only by position; build from the
    # real shuffled orders and answer consistently for face 2.
    order_a, order_b = PB.shuffled_orders([1, 2, 3], seed=7)
    ans_a = [_m("A boy in blue", [order_a.index(2) + 1])]
    ans_b = [_m("A boy in blue", [order_b.index(2) + 1])]
    calls = _script(monkeypatch, [ans_a, ans_b])
    slots = PB.bind_with_model("img", DESC, INP["faces"], photo_id=7, model="m")
    assert calls == [3, 3]
    assert [f["face_id"] for f in slots[0]["faces"]] == [2]
    # a second call that points at a DIFFERENT face drops the slot
    _script(monkeypatch, [ans_a, [_m("A boy in blue", [order_b.index(3) + 1])]])
    assert PB.bind_with_model("img", DESC, INP["faces"], photo_id=7, model="m") == []


def test_single_face_photo_makes_one_call(monkeypatch):
    calls = _script(monkeypatch, [[_m("A boy in blue", [1])]])
    one = dict(INP, faces=INP["faces"][:1])
    slots = PB.bind_with_model("img", DESC, one["faces"], photo_id=7, model="m")
    assert calls == [1] and [f["face_id"] for f in slots[0]["faces"]] == [1]


def test_no_answer_defers_and_a_crowd_is_not_attempted(monkeypatch):
    _script(monkeypatch, [None])
    assert PB.bind_with_model("img", DESC, INP["faces"], 7, "m") is None
    crowd = [{"id": i, "bbox": BOX(i), "det_score": 0.9} for i in range(7)]
    calls = _script(monkeypatch, [[]])
    assert PB.bind_with_model("img", DESC, crowd, 7, "m") == [] and calls == []


def test_bind_photo_tries_the_rule_tier_before_fetching_any_image(monkeypatch):
    desc = "A young boy with reddish hair walks across a field."
    face = {"id": 9, "bbox": [1936, 3387, 2228, 3153], "det_score": 0.85}
    inp = {"description": desc, "description_sha": PB.sha_of(desc),
           "faces": PB.offered_faces([face]), "image_width": 6528, "image_height": 4352,
           "subject_boxes": [{"label": "person", "bbox": [0.4, 0.4, 0.62, 0.93], "area_frac": 0.1}]}
    fetched = []
    slots, method = PB.bind_photo({"id": 1, "people_bind_input": inp},
                                  lambda: fetched.append(1), "m")
    assert method == "rule" and slots and not fetched
    # no subject boxes -> the image IS fetched and the VLM tier runs
    monkeypatch.setattr(PB, "bind_with_model", lambda *a, **k: [])
    slots, method = PB.bind_photo({"id": 1, "people_bind_input": dict(inp, subject_boxes=None)},
                                  lambda: fetched.append(1) or "/tmp/x.jpg", "m")
    assert method == "vlm" and slots == [] and fetched == [1]


# --- the worker loop's outcomes -----------------------------------------------------------

class _Client:
    def download_photo(self, pid, dest):
        open(dest, "wb").write(b"x")
        return 200


def _photo(pid=1):
    return {"id": pid, "filename": f"{pid}.jpg", "people_bind_input": INP}


def _patch_bind(monkeypatch, fn):
    monkeypatch.setattr(PB, "bind_photo", lambda photo, fetch, model: fn(photo))


def test_worker_row_shape_and_empty_result_is_a_row(monkeypatch, tmp_path):
    _patch_bind(monkeypatch, lambda p: ([], "vlm"))
    rows = W._process_people_bind(_Client(), [_photo()], "m", str(tmp_path))
    assert rows == [{"photo_id": 1, "description_sha": INP["description_sha"],
                     "slots": [], "method": "vlm"}]


def test_worker_outcomes_failure_defer_halt(monkeypatch, tmp_path):
    def boom(p):
        pid = p["id"]
        if pid == 1:
            raise TruncatedOutput("cut off")
        if pid == 2:
            raise UnusableAnswer("not a JSON array")
        if pid == 3:
            raise ConnectionError("lm studio down")
        if pid == 4:
            return None, ""                       # no answer
        if pid == 5:
            raise KeyError("bug")
        return [], "vlm"
    _patch_bind(monkeypatch, boom)
    rows = W._process_people_bind(_Client(), [_photo(i) for i in range(1, 7)], "m", str(tmp_path))
    by = {r["photo_id"]: r for r in rows}
    assert set(by) == {1, 2, 5, 6}                # 3 (transport) and 4 (no answer) omitted
    assert all("error" in by[i] for i in (1, 2, 5)) and "error" not in by[6]
    _patch_bind(monkeypatch, lambda p: (_ for _ in ()).throw(ContextOverflow("ctx")))
    with pytest.raises(W.PassHalted):
        W._process_people_bind(_Client(), [_photo()], "m", str(tmp_path))


def test_worker_skips_a_claim_without_input(tmp_path):
    p = {"id": 1, "filename": "1.jpg", "people_bind_input": None}
    assert W._process_people_bind(_Client(), [p], "m", str(tmp_path)) == []


def test_people_bind_reads_its_own_image():
    assert "people-bind" in W.SELF_READ_PASSES


def test_ask_model_uses_the_bind_role_and_validates_the_answer(monkeypatch):
    from photosearch import describe
    seen = {}

    def fake_chat(model, messages, options, role=None, **kw):
        seen.update(model=model, role=role, options=options, msg=messages[0])
        return '[{"phrase": "A boy in blue", "could_be": [2], "kind": "person"}]'
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", fake_chat)
    got = PB.ask_model(DESC, "B64", 3, "some-vlm")
    assert got[0]["could_be"] == [2]
    assert seen["role"] == "bind" and seen["options"]["temperature"] == 0.0
    assert seen["msg"]["images"] == ["B64"] and "Boxes: 1, 2, 3" in seen["msg"]["content"]
    assert DESC in seen["msg"]["content"]
    # the prompt carries no names or person ids by construction
    assert "person_id" not in seen["msg"]["content"]
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", lambda **k: None)
    assert PB.ask_model(DESC, "B64", 3, "m") is None            # no answer -> defer
    monkeypatch.setattr(describe, "_ollama_chat_with_retry", lambda **k: "sorry, no")
    with pytest.raises(UnusableAnswer):
        PB.ask_model(DESC, "B64", 3, "m")

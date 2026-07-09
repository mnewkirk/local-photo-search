"""Tests for subject grounding parse/crop logic (photosearch/subjects.py).

Pure-function coverage — no VLM calls. The grounding + crop-scoring round trip is
exercised manually against LM Studio; here we lock the parsing and crop geometry.
"""
import io

from PIL import Image

from photosearch.subjects import (
    _SUBJECT_FILLS_FRAME,
    _crop_primary_subject,
    parse_subject_boxes,
)


def test_parse_array_normalizes_and_sorts_by_area():
    raw = ('[{"label":"marmot","bbox_2d":[500,300,600,450]},'
           '{"label":"rock","bbox_2d":[0,0,400,600]}]')
    boxes = parse_subject_boxes(raw, w=1000, h=600)
    assert [b["label"] for b in boxes] == ["rock", "marmot"]  # bigger first
    marmot = boxes[1]
    assert marmot["bbox"] == [0.5, 0.5, 0.6, 0.75]
    assert abs(marmot["area_frac"] - (0.1 * 0.25)) < 1e-6


def test_parse_single_object():
    boxes = parse_subject_boxes('{"label":"dog","bbox_2d":[10,10,110,210]}', 200, 400)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "dog"
    assert boxes[0]["bbox"] == [0.05, 0.025, 0.55, 0.525]


def test_parse_empty_array_is_no_subject():
    assert parse_subject_boxes("[]", 100, 100) == []


def test_parse_tolerates_prose_around_json():
    raw = 'Here is the subject: [{"label":"person","bbox_2d":[0,0,50,100]}] done.'
    boxes = parse_subject_boxes(raw, 100, 100)
    assert len(boxes) == 1 and boxes[0]["label"] == "person"


def test_parse_accepts_bbox_alias_and_clamps_out_of_range():
    # coords beyond the image are clamped; reversed corners are ordered.
    boxes = parse_subject_boxes('[{"label":"x","bbox":[900,50,1200,-10]}]', 1000, 100)
    assert len(boxes) == 1
    b = boxes[0]["bbox"]
    assert b[0] == 0.9 and b[2] == 1.0        # x2 clamped to width
    assert b[1] == 0.0 and b[3] == 0.5        # y ordered + clamped


def test_parse_drops_degenerate_and_malformed():
    assert parse_subject_boxes('[{"label":"a","bbox_2d":[5,5,5,5]}]', 100, 100) == []
    assert parse_subject_boxes('[{"label":"a"}]', 100, 100) == []
    assert parse_subject_boxes("not json at all", 100, 100) == []
    assert parse_subject_boxes("", 100, 100) == []


def _img(w, h):
    im = Image.new("RGB", (w, h), (0, 128, 0))
    p = io.BytesIO()  # not used; write to temp path instead
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".jpg")
    import os
    os.close(fd)
    im.save(path, "JPEG")
    return path


def test_crop_primary_subject_pads_and_bounds():
    import os
    path = _img(1000, 800)
    try:
        boxes = [{"label": "m", "bbox": [0.4, 0.4, 0.5, 0.5], "area_frac": 0.01}]
        crop_path = _crop_primary_subject(path, boxes, pad=0.5)
        assert crop_path is not None
        with Image.open(crop_path) as c:
            # box is 100x80 px; pad 0.5 each side -> +50 / +40 each side.
            assert c.size == (200, 160)
        os.unlink(crop_path)
    finally:
        os.unlink(path)


def test_crop_skipped_when_subject_fills_frame():
    import os
    path = _img(400, 400)
    try:
        big = [{"label": "m", "bbox": [0.0, 0.0, 1.0, 1.0],
                "area_frac": _SUBJECT_FILLS_FRAME + 0.1}]
        assert _crop_primary_subject(path, big) is None
        assert _crop_primary_subject(path, []) is None
    finally:
        os.unlink(path)

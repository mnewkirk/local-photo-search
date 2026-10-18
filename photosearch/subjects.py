"""Subject grounding + subject-crop aesthetic scoring.

The full-frame aesthetics pass (``aesthetics.py``) scores the whole image, so a
small subject in a busy scene (a marmot in a meadow) is judged mostly on its
background. This module localizes the primary subject(s) with the same VLM
(qwen2.5-VL via ``describe``'s chat router — Ollama or LM Studio) and re-scores
the aesthetic rubric on a padded crop, so wow/impact/sharpness judge the subject.

Boxes are stored NORMALIZED to 0-1 of the image's (width, height) so they are
resolution-independent. See ``docs/plans/subject-aware-quality.md``.
"""
from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from .aesthetics import (
    AESTHETICS_ROLE,
    MODEL as AES_MODEL,
    parse_aesthetics_response,
    score_photo_aesthetics,
)

# Long edge (px) the grounding image is resized to before sending. The model
# returns box coordinates in THIS space; we normalize by (W, H) afterwards.
_GROUND_MAX_PX = 1024

# Padding added around the subject box (fraction of box size, each side) when
# cropping, so the subject has breathing room / context for the aesthetic score.
_CROP_PAD = 0.5

# If the primary subject already fills at least this fraction of the frame, the
# crop ≈ the full frame, so skip the redundant crop score (search falls back to
# the full-frame aesthetic).
_SUBJECT_FILLS_FRAME = 0.55

GROUNDING_PROMPT = (
    "Identify the main subject(s) of this photograph — the primary animal(s), "
    "person(s), or object(s) it is about. Ignore background scenery. The image "
    "is {w}x{h} pixels with the origin at the top-left. Respond with ONLY a "
    "minified JSON array, one object per distinct main subject, each "
    '{{"label":"<what it is>","bbox_2d":[x1,y1,x2,y2]}} using absolute pixel '
    "coordinates. If there is no clear subject (e.g. a pure landscape), respond "
    "with an empty array []."
)


def _encode_for_grounding(image_path: str, max_px: int = _GROUND_MAX_PX):
    """Return (base64_jpeg, width, height) for an EXIF-oriented, resized copy.

    We resize ourselves (rather than reusing ``_encode_image_for_ollama``) so we
    know the exact pixel space the returned box coordinates live in.
    """
    with Image.open(image_path) as im:
        return _encode_pil_for_grounding(im, max_px)


def _encode_pil_for_grounding(im, max_px: int = _GROUND_MAX_PX):
    """(base64_jpeg, w, h) for an open PIL image — EXIF-oriented, resized."""
    im = ImageOps.exif_transpose(im).convert("RGB")
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
    w, h = im.size
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), w, h


def parse_subject_boxes(raw: str, w: int, h: int) -> list[dict]:
    """Parse a grounding response into normalized subject boxes.

    Accepts a JSON array or a single object; tolerates prose around the JSON.
    Returns ``[{"label", "bbox": [x1,y1,x2,y2] in 0-1, "area_frac"}]`` sorted by
    area descending. Invalid / out-of-range boxes are dropped; ``[]`` for none.
    """
    if not raw:
        return []
    text = raw.strip()
    obj = None
    # Clean JSON is the common case (temperature 0, terse prompt). Parse it
    # directly first — a naive r"\[.*\]" would greedily grab the inner bbox
    # array out of a bare single object.
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        # Prose-wrapped: pull the outermost array-of-objects, else a single object.
        for pattern in (r"\[\s*\{.*\}\s*\]", r"\{.*\}"):
            m = re.search(pattern, text, re.S)
            if m:
                try:
                    obj = json.loads(m.group())
                    break
                except (ValueError, TypeError):
                    continue
    if obj is None:
        return []
    items = obj if isinstance(obj, list) else [obj]
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bb = it.get("bbox_2d") or it.get("bbox")
        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in bb)
        except (ValueError, TypeError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        # Clamp to image, normalize to 0-1.
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 < 1 or y2 - y1 < 1 or w <= 0 or h <= 0:
            continue
        nb = [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)]
        area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        label = it.get("label")
        out.append({"label": str(label) if label else None,
                    "bbox": nb, "area_frac": round(area, 5)})
    out.sort(key=lambda r: r["area_frac"], reverse=True)
    return out


def ground_photo_subjects(image_path: str, model: str = AES_MODEL) -> Optional[list[dict]]:
    """Locate the main subject(s). Returns normalized boxes (possibly empty), or
    None if the model was unreachable / unparseable (so the caller can defer)."""
    from .describe import _ollama_chat_with_retry

    path = Path(image_path)
    if not path.exists():
        return None
    try:
        b64, w, h = _encode_for_grounding(str(path))
    except Exception:
        return None
    messages = [{"role": "user",
                 "content": GROUNDING_PROMPT.format(w=w, h=h),
                 "images": [b64]}]
    options = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 400}
    try:
        raw = _ollama_chat_with_retry(model=model, messages=messages,
                                      options=options, role=AESTHETICS_ROLE)
    except Exception:
        return None
    if raw is None:
        return None
    return parse_subject_boxes(raw, w, h)


def _crop_primary_subject(image_path: str, boxes: list[dict],
                          pad: float = _CROP_PAD) -> Optional[str]:
    """Crop a padded box around the largest subject to a temp JPEG. Returns the
    path, or None if there's no eligible subject (or it already fills the frame)."""
    if not boxes:
        return None
    primary = boxes[0]
    if primary["area_frac"] >= _SUBJECT_FILLS_FRAME:
        return None
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            x1, y1, x2, y2 = primary["bbox"]
            bw, bh = (x2 - x1) * W, (y2 - y1) * H
            cx1 = max(0, int(x1 * W - bw * pad))
            cy1 = max(0, int(y1 * H - bh * pad))
            cx2 = min(W, int(x2 * W + bw * pad))
            cy2 = min(H, int(y2 * H + bh * pad))
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                return None
            crop = im.crop((cx1, cy1, cx2, cy2))
            fd, tmp = tempfile.mkstemp(suffix="_subject.jpg")
            import os
            os.close(fd)
            crop.save(tmp, "JPEG", quality=92)
            return tmp
    except Exception:
        return None


def score_photo_subject(image_path: str, model: str = AES_MODEL,
                        style_vocab: Optional[list[str]] = None) -> dict:
    """Ground the subject(s) and score the primary subject's crop.

    Returns ``{"subject_boxes": [...] | None, "subject_aes": <parsed> | None}``.
    ``subject_boxes is None`` means grounding failed (defer). ``[]`` means no
    subject (landscape). ``subject_aes`` is the crop's parsed aesthetics dict, or
    None when there's no subject / it fills the frame / the crop couldn't score.
    """
    boxes = ground_photo_subjects(image_path, model=model)
    if boxes is None:
        return {"subject_boxes": None, "subject_aes": None}
    subject_aes = None
    crop_path = _crop_primary_subject(image_path, boxes)
    if crop_path is not None:
        try:
            subject_aes = score_photo_aesthetics(crop_path, model=model,
                                                 style_vocab=style_vocab)
        finally:
            try:
                import os
                os.unlink(crop_path)
            except OSError:
                pass
    return {"subject_boxes": boxes, "subject_aes": subject_aes}


# ---------------------------------------------------------------------------
# Category grounding for video keyframes (docs/plans/video-support.md step 3b)
# ---------------------------------------------------------------------------
#
# The photo pass above is UNCHANGED (its prompt, role, options and parser are
# pinned by tests/test_video_llm_roles.py). This is a separate entry point for
# `video-subjects`: every box also carries a CATEGORY, and a pure landscape is
# still named ("mountain lake") with a WHOLE-FRAME box [0, 0, 1, 1] instead of
# the photo pass's `[]` — so `video_subjects.score_subjects` (which skips
# boxless entries and treats `[]` as "no subject") credits the scenery as the
# segment's subject. `[]` remains a real answer: nothing worth naming.

SUBJECT_CATEGORIES = ("person", "animal", "landscape", "object", "activity")
SUBJECT_ROLE = "subject"
WHOLE_FRAME = [0.0, 0.0, 1.0, 1.0]

CATEGORY_GROUNDING_PROMPT = (
    "Identify the main subject(s) of this video frame -- what the shot is "
    "about: the primary person(s), animal(s), object(s), activity, or, when "
    "the frame is about the scenery itself, the landscape. The image is "
    "{w}x{h} pixels with the origin at the top-left. Respond with ONLY a "
    "minified JSON array, one object per distinct main subject, each "
    '{{"label":"<what it is>","category":"<person|animal|landscape|object|'
    'activity>","bbox_2d":[x1,y1,x2,y2]}} using absolute pixel coordinates. '
    "For a landscape (e.g. a mountain lake) give a short label and the whole "
    "image as its box. If nothing in the frame is worth naming, respond with "
    "an empty array []."
)

_SUBJECT_NUM_PREDICT = 256

_CATEGORY_SYNONYMS = {
    "people": "person", "persons": "person", "human": "person",
    "humans": "person", "man": "person", "woman": "person", "child": "person",
    "animals": "animal", "pet": "animal", "pets": "animal", "bird": "animal",
    "dog": "animal", "cat": "animal", "wildlife": "animal",
    "landscapes": "landscape", "scenery": "landscape", "scene": "landscape",
    "nature": "landscape", "view": "landscape", "place": "landscape",
    "objects": "object", "thing": "object", "item": "object",
    "vehicle": "object", "food": "object", "building": "object",
    "activities": "activity", "action": "activity", "event": "activity",
    "sport": "activity", "sports": "activity",
}

# Sentinel: the text held no JSON we could read (distinct from `[]`).
_UNPARSEABLE = object()


def normalize_category(cat, label: Optional[str] = None) -> str:
    """One of SUBJECT_CATEGORIES. A missing / unrecognised category is
    inferred from the label's last word ("a small boy" -> person, "the dog"
    -> animal) and is otherwise "object"."""
    c = str(cat or "").strip().lower()
    if c in SUBJECT_CATEGORIES:
        return c
    if c in _CATEGORY_SYNONYMS:
        return _CATEGORY_SYNONYMS[c]
    words = re.findall(r"[a-z]+", str(label or "").lower())
    if words:
        last = words[-1]
        if last in ("boy", "girl", "kid", "kids", "player", "baby", "men",
                    "women", "children", "toddler", "person"):
            return "person"
        guess = _CATEGORY_SYNONYMS.get(last)
        if guess in ("person", "animal"):
            return guess
    return "object"


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*(.*?)```", text, re.S)
    return m.group(1).strip() if m else text


def _extract_json(raw: Optional[str]):
    """The first JSON value in `raw`, or _UNPARSEABLE.

    Tolerates code fences, leading/trailing prose, and either an array or a
    single object; uses raw_decode from each `[`/`{` so trailing prose after a
    complete value does not matter and an inner bbox array is never mistaken
    for the outer answer when the outer value parses."""
    if raw is None:
        return _UNPARSEABLE
    text = _strip_fences(str(raw).strip())
    if not text:
        return _UNPARSEABLE
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
            return obj
        except ValueError:
            continue
    return _UNPARSEABLE


def _box_to_norm(bb, w: int, h: int):
    """[x1,y1,x2,y2] in absolute px, 0-1 fractions, or qwen's 0-1000 grid ->
    clamped 0-1 floats, or None if invalid / degenerate."""
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bb)
    except (ValueError, TypeError):
        return None
    vals = (x1, y1, x2, y2)
    if all(0.0 <= v <= 1.0 for v in vals):
        sx, sy = 1.0, 1.0                       # already normalised
    elif (max(x1, x2) > w * 1.02 or max(y1, y2) > h * 1.02) \
            and max(vals) <= 1000.0:
        sx, sy = 1000.0, 1000.0                 # 0-1000 relative grid
    else:
        sx, sy = float(w), float(h)             # absolute pixels
    if sx <= 0 or sy <= 0:
        return None
    x1, x2 = sorted((x1 / sx, x2 / sx))
    y1, y2 = sorted((y1 / sy, y2 / sy))
    x1, x2 = max(0.0, min(x1, 1.0)), max(0.0, min(x2, 1.0))
    y1, y2 = max(0.0, min(y1, 1.0)), max(0.0, min(y2, 1.0))
    min_w = 1.0 / max(w, 1)
    min_h = 1.0 / max(h, 1)
    if x2 - x1 < min_w or y2 - y1 < min_h:
        return None
    return [round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5)]


def parse_category_subjects(raw: Optional[str], w: int, h: int):
    """Parse a category-grounding answer.

    Returns ``[{"label", "category", "bbox" (0-1), "area_frac"}]`` sorted by
    area descending (``[]`` is a real "nothing to name" answer), or None when
    the answer is unusable: no JSON at all, or a non-empty answer in which not
    one item is a usable subject. Accepts a bare array, a single object, an
    object wrapping the list (``{"subjects": [...]}``), a bare bbox
    ``[x1,y1,x2,y2]`` (an unlabeled object), and bare strings in a list
    (labels with no box are kept only for ``landscape``, as WHOLE_FRAME).
    """
    obj = _extract_json(raw)
    if obj is _UNPARSEABLE:
        return None
    if isinstance(obj, dict):
        for key in ("subjects", "objects", "results", "items", "boxes"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
    if isinstance(obj, list) and len(obj) == 4 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj):
        obj = [{"bbox_2d": obj}]                # a bare box
    items = obj if isinstance(obj, list) else [obj]
    if not items:
        return []
    out: list[dict] = []
    for it in items:
        if isinstance(it, str):
            it = {"label": it}
        if not isinstance(it, dict):
            continue
        label = it.get("label") or it.get("name")
        label = str(label).strip() if label else None
        category = normalize_category(it.get("category") or it.get("type"), label)
        bb = it.get("bbox_2d") or it.get("bbox") or it.get("box")
        nb = _box_to_norm(bb, w, h) if bb else None
        if nb is None:
            if category != "landscape" or not label:
                continue
            nb = list(WHOLE_FRAME)
        area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        out.append({"label": label, "category": category, "bbox": nb,
                    "area_frac": round(area, 5)})
    if not out:
        return None
    out.sort(key=lambda r: r["area_frac"], reverse=True)
    return out


def _encode_any_for_grounding(image):
    """(b64, w, h) from a path, encoded image bytes, or an open PIL image."""
    if isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(image))) as im:
            return _encode_pil_for_grounding(im)
    if isinstance(image, Image.Image):
        return _encode_pil_for_grounding(image)
    return _encode_for_grounding(str(image))


def ground_frame_subjects(image, model: str = AES_MODEL, *,
                          with_category: bool = True,
                          raise_unusable: bool = False,
                          prompt: Optional[str] = None) -> Optional[list[dict]]:
    """Ground the main subject(s) of one video keyframe, with a category.

    `image` is a path, encoded image bytes, or a PIL image. Role ``subject``
    (PHOTOSEARCH_LLM_SUBJECT_MODEL on the OpenAI route; unset, it resolves to
    the aesthetics model photos ground with). On Ollama, `model` is used.

    Returns ``[{"label", "category", "bbox", "area_frac"}]`` (``[]`` = no
    subject), or None to defer (image unreadable, transport failure). An
    unparseable answer is retried once at temperature 0.4; if that fails too
    it raises UnusableAnswer when `raise_unusable`, else returns None.
    ContextOverflow / RoleModelNotConfigured always propagate.

    ``with_category=False`` asks the photo pass's GROUNDING_PROMPT instead and
    returns the same dicts with ``category`` inferred from the label
    (person / animal words, else "object") — for eval comparison.
    """
    from .describe import (ContextOverflow, RoleModelNotConfigured,
                           TruncatedOutput, UnusableAnswer,
                           _ollama_chat_with_retry)
    try:
        b64, w, h = _encode_any_for_grounding(image)
    except Exception:
        return None
    tpl = prompt or (CATEGORY_GROUNDING_PROMPT if with_category else GROUNDING_PROMPT)
    messages = [{"role": "user",
                 "content": tpl.replace("{w}", str(w)).replace("{h}", str(h))
                               .replace("{{", "{").replace("}}", "}"),
                 "images": [b64]}]

    def ask(temperature):
        options = {"temperature": temperature, "num_ctx": 8192,
                   "num_predict": _SUBJECT_NUM_PREDICT}
        try:
            raw = _ollama_chat_with_retry(model=model, messages=messages,
                                          options=options, role=SUBJECT_ROLE)
        except TruncatedOutput as e:
            return None, f"cut off ({e})"
        except (ContextOverflow, RoleModelNotConfigured):
            raise
        except Exception:
            return _UNPARSEABLE, None           # transport: defer
        if raw is None:
            return None, "empty response"
        parsed = parse_category_subjects(raw, w, h)
        if parsed is None:
            return None, "unparseable grounding answer"
        return parsed, None

    out, problem = ask(0.0)
    if out is _UNPARSEABLE:
        return None
    if problem:
        out, problem = ask(0.4)
        if out is _UNPARSEABLE:
            return None
    if problem:
        if raise_unusable:
            raise UnusableAnswer(f"video-subjects: {problem}")
        return None
    return out

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

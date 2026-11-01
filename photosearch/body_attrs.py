"""Per-body attributes read by a VLM: jersey number, kit colour, hair, shoes, role, facing.

docs/plans/people-context-describe.md, Step 1. Evidence for the identity tiers
(:mod:`photosearch.identify`): the kit tier keys on (kit, number); hair and shoes
only break ties. One call per body crop, a closed vocabulary, LM Studio
json_schema answers. Label-free: the model never sees a name.

An answer outside the vocabulary raises ``describe.UnusableAnswer`` (a failure,
not a result); ``None`` from the transport means no answer (defer).
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Optional

ATTRS_VERSION = 1

COLOURS = ["white", "black", "grey", "red", "orange", "yellow", "neon-yellow", "green",
           "light-blue", "blue", "navy", "purple", "pink", "maroon", "brown", "other"]
KIT_COLOURS = COLOURS + ["not-visible"]
HAIR_COLOURS = ["blond", "light-brown", "dark-brown", "black", "red", "grey", "not-visible"]
HAIR_LENGTHS = ["buzzed", "short", "medium", "long", "not-visible"]
ROLES = ["player", "goalkeeper", "referee", "coach", "spectator", "other"]
FACING = ["toward", "side", "away"]

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["role", "facing", "kit", "number", "hair_colour", "hair_length", "shoes"],
    "properties": {
        "role": {"type": "string", "enum": ROLES},
        "facing": {"type": "string", "enum": FACING},
        "kit": {"type": "string", "enum": KIT_COLOURS},
        "number": {"type": "string"},
        "hair_colour": {"type": "string", "enum": HAIR_COLOURS},
        "hair_length": {"type": "string", "enum": HAIR_LENGTHS},
        "shoes": {"type": "string", "enum": KIT_COLOURS},
    },
}

PROMPT = (
    "Look only at the person inside the green box. Describe what is visible of that one person; "
    "ignore everyone else.\n"
    "- role: player, goalkeeper, referee, coach, spectator or other.\n"
    "- facing: toward the camera, side, or away (back to the camera).\n"
    "- kit: the main colour of their shirt or jersey.\n"
    "- number: the jersey number printed on THEIR shirt, digits only, exactly as printed. "
    "Write \"none\" if no number is visible on their shirt, and \"unreadable\" if a number is "
    "there but you cannot read every digit. Never guess a digit.\n"
    "- hair_colour, hair_length: as seen; not-visible if covered or hidden.\n"
    "- shoes: the main colour of their shoes; not-visible if out of frame.\n"
    "Answer with the JSON object only."
)

_NUM_RE = re.compile(r"^\d{1,2}$")


def crop_b64(image, bbox, pad: float = 0.15, max_px: int = 768) -> str:
    """The body with ``pad`` context, its box drawn green, JPEG base64. ``image`` is a
    PIL image in the same oriented pixel frame as ``bbox`` (top, right, bottom, left)."""
    from PIL import Image, ImageDraw
    t, r, b, l = bbox
    w, h = r - l, b - t
    box = (max(0, int(l - pad * w)), max(0, int(t - pad * h)),
           min(image.width, int(r + pad * w)), min(image.height, int(b + pad * h)))
    c = image.crop(box).convert("RGB")
    s = min(1.0, max_px / float(max(c.size)))
    if s < 1.0:
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    d.rectangle([(l - box[0]) * s, (t - box[1]) * s, (r - box[0]) * s, (b - box[1]) * s],
                outline=(0, 220, 0), width=max(2, round(3 * s)))
    buf = io.BytesIO()
    c.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def parse(raw: str) -> dict:
    """Validate one answer. Numbers normalise to digits without leading zeros
    ("07" -> "7"); "none"/"unreadable" stay words; anything else is unusable."""
    from .describe import UnusableAnswer
    try:
        obj = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
    except (ValueError, AttributeError):
        raise UnusableAnswer(f"body-attrs: not JSON: {str(raw)[:80]!r}")
    if not isinstance(obj, dict):
        raise UnusableAnswer("body-attrs: not an object")
    for k, vocab in (("role", ROLES), ("facing", FACING), ("kit", KIT_COLOURS),
                     ("hair_colour", HAIR_COLOURS), ("hair_length", HAIR_LENGTHS),
                     ("shoes", KIT_COLOURS)):
        if obj.get(k) not in vocab:
            raise UnusableAnswer(f"body-attrs: {k}={obj.get(k)!r} not in the vocabulary")
    n = str(obj.get("number", "")).strip().lower().lstrip("#")
    if n in ("none", "unreadable"):
        pass
    elif _NUM_RE.match(n):
        n = str(int(n))
    elif n in ("", "n/a", "no", "null"):
        n = "none"
    else:
        raise UnusableAnswer(f"body-attrs: number={obj.get('number')!r}")
    return {k: obj[k] for k in SCHEMA["required"] if k != "number"} | {"number": n}


def read(image, bbox, model: str, base_url: str, timeout: float = 60.0) -> Optional[dict]:
    """One VLM call for one body. ``None`` = no answer (transport/empty)."""
    from .describe import _openai_chat_with_retry
    msg = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + crop_b64(image, bbox)}}]}]
    raw = _openai_chat_with_retry(base_url, model, msg, retries=2, timeout=timeout,
                                  max_tokens=160, role="body-attrs", schema=("body_attrs", SCHEMA))
    return None if raw is None else parse(raw)

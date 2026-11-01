"""Grounded describe (prototype for the G1 eval): describe writes its own person spans.

docs/plans/people-context-describe.md, Step 3. The model sees the photo and the same
photo with numbered person boxes and writes its description with INLINE ANCHORS:
``[the boy in the yellow jersey](2)`` or ``[two girls](1,4)``. Code strips the markup,
so every span matches the text by construction (Step 0: a separate ``people`` list
drifted from the text on 14 of 47 phrases and halved the descriptions).

Label-free except for hints: a body with a trusted, uncontested face may carry its
gender and an age band (never a name or a person id). Not wired into any pass.
"""
from __future__ import annotations

import re
from typing import Optional

ANCHOR_RE = re.compile(r"\[([^\[\]\n]{1,160})\]\(([^()\n]{0,24})\)")
_BOXES_RE = re.compile(r"^\s*\d{1,2}(?:\s*,\s*\d{1,2})*\s*$")
MAX_GROUP = 3
# Overlay words in the raw text = the model described the drawing, not the photo.
LEAK_RE = re.compile(r"\b(box|boxes|boxed|bounding|rectangles?|outlined|numbered|"
                     r"second image|first image|image 2|label(?:l)?ed \d|marked \d)\b", re.I)
_BRACKET_RE = re.compile(r"[\[\]]")

_PHOTO_RULES = """\
Describe the first image in 2-4 concise sentences for a search index. Include:
- Who or what is in the photo (number of people, approximate ages, clothing)
- What they are doing (actions, poses)
- The setting (indoor/outdoor, location type such as beach/park/street/home)
- Whether people are present or absent
Be factual. Only describe what you can clearly see — do not guess at objects you \
are unsure about. Do not start with "The image shows" or similar preamble."""

_MARK_RULES = """\
Mark people: every time the description refers to one specific person, or a group of 2 or \
3 people, who is inside a numbered box, put those words in square brackets and add the box \
number(s) in parentheses right after, for example: [a boy in a yellow jersey](2) kicks the \
ball while [two girls](1,4) watch. The square brackets go around the words and the box \
number(s) go in round brackets right after them; never write a number in square brackets. \
Mark only people, never objects. Do not mark pronouns, crowds, or people without a box. \
Never mention the boxes or the numbers outside the parentheses."""

TWO_IMAGE_HEAD = ("You get two images. The first is a photo. The second is the same photo with "
                  "numbered red boxes, 1 to {n}, drawn around the people in it.\n\n")
ONE_IMAGE_HEAD = ("This photo has numbered red boxes, 1 to {n}, drawn around the people in it. "
                  "The boxes are not part of the scene.\n\n")

LENGTH_INSTRUCTION = ("Write as much as you would for the photo alone: 2-4 full sentences "
                      "covering the people, what they are doing and the setting.\n\n")

FIX_MARKUP = ("Rewrite the same description with the marks in the right form: square brackets "
              "around the words for the person and their box number(s) in round brackets right "
              "after, like [a boy in a blue jersey](1). No number in square brackets. Answer with "
              "the description only.")

FIX_LEAK = ("Rewrite the same description without mentioning the boxes, the numbers or the "
            "drawing: describe only the people and the scene, keeping the [words](n) marks. "
            "Answer with the description only.")

HINTS_INTRO = ("Known about some boxed people (if what you see clearly disagrees, describe "
               "what you see):\n")


def age_band(age: Optional[int]) -> Optional[str]:
    if age is None:
        return None
    if age < 2:
        return "a baby"
    if age <= 12:
        return f"a child aged about {max(2, age - 1)}-{age + 1}"
    if age <= 19:
        return "a teenager"
    return "an adult"


def hint_line(n: int, gender: Optional[str], age: Optional[int]) -> Optional[str]:
    parts = [p for p in (gender, age_band(age)) if p]
    return f"- box {n}: {', '.join(parts)}" if parts else None


def build_prompt(n_boxes: int, shape: str = "two", length: bool = True,
                 hints: Optional[list[str]] = None) -> str:
    """``shape``: "two" (clean photo + overlay) or "one" (overlay only)."""
    rules = _PHOTO_RULES if shape == "two" else _PHOTO_RULES.replace("the first image", "this photo")
    p = (TWO_IMAGE_HEAD if shape == "two" else ONE_IMAGE_HEAD).format(n=n_boxes) + rules + "\n\n"
    if length:
        p += LENGTH_INSTRUCTION
    p += _MARK_RULES
    if hints:
        p += "\n\n" + HINTS_INTRO + "\n".join(hints)
    return p


_STRAY_NUM_RE = re.compile(r"\s*\[\s*\d{1,2}(?:\s*[,-]\s*\d{1,2})*\s*\]")


def strip_marks(raw: str, n_boxes: int) -> dict:
    """Last resort after a corrective retry: drop stray ``[n]`` marks and keep the text
    with NO spans (the photo names nobody). Still raises on leakage or leftover brackets."""
    return dict(parse(_STRAY_NUM_RE.sub("", raw or ""), n_boxes), spans=[], unanchored=True)


def parse(raw: str, n_boxes: int) -> dict:
    """``{"description", "spans": [{"start", "end", "phrase", "boxes"}], "dropped": [...]}``
    with the markup stripped. An anchor whose parentheses are not 1-3 box numbers in
    1..n is stripped WITHOUT a span (``dropped``): the text stays usable, the mention
    names nobody. Raises ``describe.UnusableAnswer`` on an empty text, a stray bracket
    left after stripping, or overlay words in the text."""
    from .describe import UnusableAnswer
    text = (raw or "").strip()
    out, spans, dropped, pos, length = [], [], [], 0, 0
    for m in ANCHOR_RE.finditer(text):
        before = text[pos:m.start()]
        out.append(before)
        length += len(before)
        phrase = m.group(1).strip()
        inner = m.group(2)
        boxes = sorted({int(x) for x in inner.split(",")}) if _BOXES_RE.match(inner) else None
        if boxes is None or len(boxes) > MAX_GROUP or any(b < 1 or b > n_boxes for b in boxes):
            dropped.append({"phrase": phrase, "inner": inner})
        else:
            spans.append({"start": length, "end": length + len(phrase), "phrase": phrase, "boxes": boxes})
        out.append(phrase)
        length += len(phrase)
        pos = m.end()
    out.append(text[pos:])
    desc = "".join(out)
    if desc and desc[0].islower():          # "[ten children](..) play" opened the text
        desc = desc[0].upper() + desc[1:]
    if not desc.strip():
        raise UnusableAnswer("grounded describe: empty description")
    if _BRACKET_RE.search(desc):
        raise UnusableAnswer(f"grounded describe: stray markup: {desc[:80]!r}")
    leak = LEAK_RE.search(desc)
    if leak:
        raise UnusableAnswer(f"grounded describe: overlay words: {leak.group(0)!r}")
    return {"description": desc, "spans": spans, "dropped": dropped}

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
_COUNT_WORDS = {"two": 2, "both": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "several": 9,
                "group": 9, "others": 9}
_SINGULAR_RE = re.compile(r"^(?:a|an|one|the|another)\b", re.I)
_PLURAL_RE = re.compile(r"\b(?:boys|girls|children|kids|players|men|women|people|teammates|"
                        r"adults|teens|others)\b", re.I)


def count_problem(phrase: str, n_boxes: int) -> Optional[str]:
    """Why a phrase cannot name exactly ``n_boxes`` people, or None. "Two boys" on one box,
    "a boy" on two boxes, a plural with no count on one box."""
    words = re.findall(r"[a-z]+", phrase.lower())
    for w in words:
        if w in _COUNT_WORDS:
            return None if _COUNT_WORDS[w] == n_boxes else f"count {w} vs {n_boxes} boxes"
    if n_boxes > 1 and _SINGULAR_RE.match(phrase) and not _PLURAL_RE.search(phrase):
        return f"singular phrase on {n_boxes} boxes"
    if n_boxes == 1 and _PLURAL_RE.search(phrase) and not _SINGULAR_RE.match(phrase):
        return "plural phrase on one box"
    return None


_PERSON_NOUN_RE = re.compile(r"(?:boy|girl|man|woman|child|kid|player|person|teen|toddler|baby|"
                             r"adult|coach|referee|goalkeeper|teammate)s?", re.I)


def degenerate(description: str, spans: list[dict]) -> Optional[str]:
    """Text bent to fit the marks: one phrase marked 3+ times ("[a boy](1), [a boy](2),
    [a boy](3)"), or a mark right after a person noun ("A young boy [a boy in blue](1)")."""
    c = {}
    for sp in spans:
        k = sp["phrase"].lower()
        c[k] = c.get(k, 0) + 1
        if c[k] >= 3:
            return f"phrase {sp['phrase']!r} marked 3+ times"
        prev = re.findall(r"[A-Za-z]+", description[max(0, sp["start"] - 30):sp["start"]])
        if prev and _PERSON_NOUN_RE.fullmatch(prev[-1]):
            return f"mark follows a person noun: {prev[-1]!r} [{sp['phrase']}]"
    return None

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

CONTEXT_INTRO = ("Who is in the boxes (known and correct):\n")

DUAL_MARK_RULES = """\
Mark people: every time the description refers to one specific person, or a group of 2 or \
3 people, who is inside a numbered box, put those words in square brackets and add the box \
number(s) in parentheses right after, for example: [a boy in a yellow jersey](2) kicks the \
ball while [two girls](1,4) watch. If the person's name is listed below, write it after a | \
inside the brackets, after the plain words: [a boy in a yellow jersey|Sam](2). The plain \
words must still describe what you see (age, clothing), as if you did not know the name; \
names appear ONLY after a | inside the brackets, never anywhere else. The square brackets go \
around the words and the box number(s) go in round brackets right after them; never write a \
number in square brackets. Mark only people, never objects. Do not mark pronouns, crowds, or \
people without a box. Never mention the boxes or the numbers outside the parentheses."""

FIX_NAMES = ("Rewrite the same description without any person's name: describe each person by "
             "what they look like and do, keeping the [words](n) marks. Answer with the "
             "description only.")

FIX_DUAL = ("Rewrite the same description so that a name appears ONLY after a | inside the "
            "brackets, after plain words that describe the person: [a boy in a blue jersey|Sam](2). "
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


def person_noun(gender: Optional[str], age: Optional[int]) -> Optional[str]:
    """"a boy aged about 10", "a woman", "male": what the context says a box is."""
    if gender not in ("male", "female"):
        return age_band(age)
    if age is not None and age <= 12:
        return f"a {'boy' if gender == 'male' else 'girl'} aged about {age}"
    if age is not None and age <= 19:
        return f"a teenage {'boy' if gender == 'male' else 'girl'}"
    if age is not None:
        return "a man" if gender == "male" else "a woman"
    return gender


def context_line(n: int, name: str, gender: Optional[str], age: Optional[int]) -> str:
    """"- box 1: Asa Goldman (boy, 10)": terse, so the model does not copy it into the text."""
    parts = []
    if gender in ("male", "female"):
        kid = age is not None and age <= 17
        parts.append(("boy" if gender == "male" else "girl") if kid else ("man" if gender == "male" else "woman")
                     if age is not None else gender)
    if age is not None:
        parts.append(str(age))
    return f"- box {n}: {name}" + (f" ({', '.join(parts)})" if parts else "")


def name_matches(said: str, full: str) -> bool:
    """The name written in a mark is the box's person: the full name or any part of it."""
    said = said.strip().lower()
    return bool(said) and (said == full.lower() or said in full.lower().replace("'", " ").split()
                           or said in full.lower().split())


def assemble(description: str, spans: list[dict], box_person: dict) -> str:
    """The named template: each mark whose name agrees with its box becomes ``{person:ID}``;
    every other span keeps its plain words. Names are filled in at read time."""
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        if sp.get("name_agrees") and box_person.get(sp["boxes"][0]) is not None:
            out.append(description[pos:sp["start"]])
            out.append("{person:%d}" % box_person[sp["boxes"][0]])
            pos = sp["end"]
    out.append(description[pos:])
    return "".join(out)


def name_in_text(description: str, names: list[str]) -> Optional[str]:
    """The first roster name (full or any part of 3+ letters) the text uses, else None."""
    for full in names:
        for part in {full, *full.replace("'", " ").split()}:
            if len(part) >= 3 and re.search(r"\b" + re.escape(part) + r"\b", description, re.I):
                return part
    return None


def build_prompt(n_boxes: int, shape: str = "two", length: bool = True,
                 hints: Optional[list[str]] = None, context: Optional[list[str]] = None,
                 dual: bool = False) -> str:
    """``shape``: "two" (clean photo + overlay) or "one" (overlay only)."""
    rules = _PHOTO_RULES if shape == "two" else _PHOTO_RULES.replace("the first image", "this photo")
    p = (TWO_IMAGE_HEAD if shape == "two" else ONE_IMAGE_HEAD).format(n=n_boxes) + rules + "\n\n"
    if length:
        p += LENGTH_INSTRUCTION
    p += DUAL_MARK_RULES if dual else _MARK_RULES
    if hints:
        p += "\n\n" + HINTS_INTRO + "\n".join(hints)
    if context:
        p += "\n\n" + CONTEXT_INTRO + "\n".join(context)
    return p


_STRAY_NUM_RE = re.compile(r"\s*\[\s*\d{1,2}(?:\s*[,-]\s*\d{1,2})*\s*\]")


def strip_marks(raw: str, n_boxes: int, names: Optional[list[str]] = None) -> dict:
    """Last resort after a corrective retry: drop stray ``[n]`` marks and keep the text
    with NO spans (the photo names nobody). Still raises on leakage or leftover brackets."""
    return dict(parse(_STRAY_NUM_RE.sub("", raw or ""), n_boxes, names), spans=[], unanchored=True)


def parse(raw: str, n_boxes: int, names: Optional[list[str]] = None,
          box_names: Optional[dict] = None) -> dict:
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
        said = None
        if "|" in phrase:                    # dual mark: [plain words|Name](n)
            phrase, said = (x.strip() for x in phrase.split("|", 1))
        inner = m.group(2)
        boxes = sorted({int(x) for x in inner.split(",")}) if _BOXES_RE.match(inner) else None
        if boxes is None or len(boxes) > MAX_GROUP or any(b < 1 or b > n_boxes for b in boxes):
            dropped.append({"phrase": phrase, "inner": inner})
        elif count_problem(phrase, len(boxes)):
            dropped.append({"phrase": phrase, "inner": inner, "why": count_problem(phrase, len(boxes))})
        else:
            sp = {"start": length, "end": length + len(phrase), "phrase": phrase, "boxes": boxes}
            if said is not None:
                sp["said"] = said
                want = [(box_names or {}).get(b) for b in boxes]
                sp["name_agrees"] = (len(boxes) == 1 and want[0] is not None
                                     and name_matches(said, want[0]))
            spans.append(sp)
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
    hit = name_in_text(desc, names or [])
    if hit:
        raise UnusableAnswer(f"grounded describe: name in text: {hit!r}")
    return {"description": desc, "spans": spans, "dropped": dropped, "degenerate": degenerate(desc, spans)}


# ---------------------------------------------------------------------------
# v2: known people by person-id handle (docs/plans/people-context-describe-v2.md).
# No names are sent. A known person's box carries an opaque handle ``P<person_id>``
# in the round brackets: ``[a boy in a green jersey](4:P39)``. Code strips the marks
# (raw, name-free text), cross-checks each written handle against the roster entry for
# that box, and names only spans that pass every guard (``name_spans``). The v1
# functions above stay so old runs re-score.

V2_HEAD = TWO_IMAGE_HEAD
V2_DRAWN_LINE = ("A known person's box also shows their P-number next to the box number, "
                 "like 4·P0.\n\n")

KNOWN_INTRO = (
    "KNOWN PEOPLE. These people have been identified by the family. Their gender and age below\n"
    "are facts, not guesses: use them even when the person looks different (long hair, clothing\n"
    "colour, build or pose can mislead). A person listed as a boy is \"a boy\", never \"a girl\",\n"
    "\"she\" or \"her\"; a person listed as a girl is \"a girl\", never \"a boy\", \"he\" or \"him\"; a\n"
    "person listed as aged 10 is a child, never \"a man\", \"a woman\" or \"a teenager\". Do not\n"
    "copy this list into the description: never write an age as a number. Describe a known person\n"
    "like anyone else, by what they wear and do.\n")
KNOWN_TAIL = ("Every other box is a person nobody has identified: describe them only from what "
              "you see.\n\n")

V2_PHOTO_RULES = (
    "Describe the first image in 2-4 full sentences for a search index. Cover:\n"
    "- The people: how many, approximate ages, clothing (for a known person, their listed\n"
    "  gender and age)\n"
    "- What they are doing (actions, poses)\n"
    "- The setting (indoor/outdoor, location type such as field, park, street, home)\n"
    "Be factual and describe only what you can clearly see. Do not start with \"The image shows\"\n"
    "or similar. Write as much as you would for the photo alone.\n\n")

_V2_MARK_OPEN = (
    "Mark people: every time the description refers to one specific person, or a group of 2 or\n"
    "3 people, who is inside a numbered box, put the describing words in square brackets and\n"
    "the box number(s) in round brackets right after. ")
_V2_MARK_ID = (
    "For a known person, write the box number,\n"
    "a colon and their P-number: [a boy in a red cap](2:P0). For a group, list each box, with\n"
    "the P-number for any known member: [two men](1,3:P0). ")
_V2_EXAMPLE_ID = (
    "Example (a different photo):\n"
    "[a woman in a grey coat](2:P0) hands an umbrella to [two men](1,3), who stand at a bus stop.\n"
    "That sentence only shows the form; P0 is never a real person, and your people, scene and\n"
    "P-numbers come from the photos and the list above. ")
_V2_RULES_ID = (
    "Put the whole phrase, clothing included, inside the square brackets: [a man in a red\n"
    "cap](4), never [a man](4) in a red cap. "
    "Inside the square brackets write only\n"
    "ordinary describing words (age, gender, clothing, role); a bare P-number in brackets is\n"
    "wrong, and a P-number appears ONLY after a colon inside the round brackets, never in the\n"
    "sentence. Never use a P-number that is not in the list. Mark only people, never objects.\n"
    "Do not mark pronouns, crowds, or people without a box. Never mention the boxes, the\n"
    "numbers or the drawing outside the round brackets, and never write a person's name.")
# The handle-free mark rules (the v2-box arm: the code maps box -> person).
_V2_EXAMPLE_BOX = (
    "Example (a different photo):\n"
    "[a woman in a grey coat](2) hands an umbrella to [two men](1,3), who stand at a bus stop.\n"
    "That sentence only shows the form; your people and scene come from the photos. ")
_V2_RULES_BOX = (
    "Put the whole phrase, clothing included, inside the square brackets: [a man in a red\n"
    "cap](4), never [a man](4) in a red cap. "
    "Inside the square brackets write only\n"
    "ordinary describing words (age, gender, clothing, role); never write a number in square\n"
    "brackets. Mark only people, never objects.\n"
    "Do not mark pronouns, crowds, or people without a box. Never mention the boxes, the\n"
    "numbers or the drawing outside the round brackets, and never write a person's name.")

REMINDER = ("Reminder: {items}. Known people keep their\n"
            "listed gender and age and are marked with their P-number.")
REMINDER_BOX = "Reminder: {items}. Known people keep their listed gender and age."

FIX_MARKUP_V2 = (
    "Rewrite the same description with every mark in this form: describing words in square "
    "brackets, then box number(s) in round brackets, like [a boy in a blue jersey](4). After the "
    "box number of a known person from the list, add a colon and their P-number. No number or "
    "P-number in square brackets. Keep every person mark. Answer with the description only.")
FIX_IDS = (
    "Rewrite the same description so that P-numbers appear only inside the round brackets "
    "after marked words, after a colon behind the box number. In the sentences, describe each "
    "person by what they look like and do. Keep every person mark. Answer with the "
    "description only.")
FIX_LEAK_V2 = FIX_LEAK + " Keep every person mark and P-number."
FIX_AGES = ("Rewrite the same description without writing anyone's age as a number: describe each "
            "person by what they wear and do. Keep every person mark. Answer with the description only.")
FIX_NAMES_V2 = FIX_NAMES + " Keep every person mark and P-number."
FIX_ROSTER = (
    "{handle} (box {box}) is {noun}. You wrote '{phrase}' for them. Rewrite the same "
    "description so every mention of that person agrees with the list, keeping every mark. "
    "Answer with the description only.")


def _frozen_text() -> str:
    return "\x1e".join([V2_HEAD, V2_DRAWN_LINE, KNOWN_INTRO, KNOWN_TAIL, V2_PHOTO_RULES,
                        _V2_MARK_OPEN, _V2_MARK_ID, _V2_EXAMPLE_ID, _V2_RULES_ID,
                        _V2_EXAMPLE_BOX, _V2_RULES_BOX, REMINDER, REMINDER_BOX,
                        FIX_MARKUP_V2, FIX_IDS, FIX_LEAK_V2, FIX_NAMES_V2, FIX_ROSTER, FIX_AGES])


def _prompt_version() -> str:
    import hashlib
    return hashlib.sha256(_frozen_text().encode()).hexdigest()[:12]


PROMPT_VERSION = _prompt_version()


def roster_noun(gender: Optional[str], age: Optional[int]) -> Optional[str]:
    """"boy" / "girl" (under 18), "man" / "woman", the bare gender when the age is unknown."""
    if gender not in ("male", "female"):
        return None
    if age is None:
        return gender
    if age <= 17:
        return "boy" if gender == "male" else "girl"
    return "man" if gender == "male" else "woman"


def _roster_facts(e: dict) -> list[str]:
    return [p for p in (roster_noun(e.get("gender"), e.get("age")),
                        None if e.get("age") is None else str(e["age"])) if p]


def roster_line(box: int, e: dict, handles: bool = True) -> str:
    """``- box 4: P39, boy, 10``; ``- box 4: P39`` with neither gender nor age (the handle
    still names); ``- box 4: boy, 10`` without handles."""
    parts = ([f"P{e['person_id']}"] if handles else []) + _roster_facts(e)
    return f"- box {box}: {', '.join(parts)}" if parts else f"- box {box}"


def _reminder_item(box: int, e: dict, handles: bool) -> str:
    facts = _roster_facts(e)
    noun = (("a " if facts[0] in ("boy", "girl", "man", "woman") else "")
            + ", ".join(facts)) if facts else None
    if handles:
        return f"P{e['person_id']} (box {box}) is {noun}" if noun else f"P{e['person_id']} is in box {box}"
    return f"box {box} is {noun}" if noun else f"box {box} is known"


def build_prompt_v2(n_boxes: int, roster: Optional[dict] = None, placement: str = "first",
                    example: bool = True, drawn: bool = False, handles: bool = True) -> str:
    """The v2 describe prompt. ``roster``: ``{box: {person_id, gender, age, ...}}``.
    ``placement``: "first" (KNOWN block after the head, reminder last) or "last" (KNOWN
    block at the end, no reminder). With no roster the KNOWN block and the reminder are
    omitted: the production prompt for unlabelled shoots."""
    handles = handles and bool(roster)         # nobody known: P-numbers are never mentioned
    head = V2_HEAD.format(n=n_boxes) + (V2_DRAWN_LINE if drawn and roster else "")
    known = ""
    if roster:
        known = (KNOWN_INTRO + "\n".join(roster_line(b, roster[b], handles) for b in sorted(roster))
                 + "\n" + KNOWN_TAIL)
    marks = _V2_MARK_OPEN + ((_V2_MARK_ID + (_V2_EXAMPLE_ID if example else "") + _V2_RULES_ID)
                             if handles else ((_V2_EXAMPLE_BOX if example else "") + _V2_RULES_BOX))
    if placement == "last":
        return head + V2_PHOTO_RULES + marks + ("\n\n" + known.rstrip("\n") if known else "")
    p = head + known + V2_PHOTO_RULES + marks
    if roster:
        items = "; ".join(_reminder_item(b, roster[b], handles) for b in sorted(roster))
        p += "\n\n" + (REMINDER if handles else REMINDER_BOX).format(items=items)
    return p


def fix_roster_prompt(box: int, e: dict, phrase: str, handles: bool = True,
                      why: Optional[str] = None) -> str:
    """The FIX_ROSTER turn; without handles (the box arm) it says "The person (box 4)". A
    gender conflict states the gender only: "is a boy, 10" came back as "a boy aged 10"."""
    facts = _roster_facts(e)
    if why and "word" in why and facts and facts[0] in ("boy", "girl", "man", "woman"):
        facts = facts[:1]
    noun = ("a " + ", ".join(facts)) if facts and facts[0] in ("boy", "girl", "man", "woman") \
        else (", ".join(facts) or "listed")
    return FIX_ROSTER.format(handle=f"P{e['person_id']}" if handles else "The person", box=box,
                             noun=noun, phrase=phrase)


_INNER_V2_RE = re.compile(r"^\s*\d{1,2}(?:\s*:\s*P\d{1,6})?(?:\s*,\s*\d{1,2}(?:\s*:\s*P\d{1,6})?)*\s*$")
_INNER_ITEM_RE = re.compile(r"(\d{1,2})(?:\s*:\s*P(\d{1,6}))?")
ID_IN_PROSE_RE = re.compile(r"\b(P\d{1,6}|[Pp]erson\s+\d+)\b")
# A box group in round brackets left in the prose: "a boy (4:P39) runs".
_STRAY_PAREN_RE = re.compile(r"\(\s*\d{1,2}(?:\s*:\s*P\d{1,6})?(?:\s*,\s*\d{1,2}(?:\s*:\s*P\d{1,6})?)*\s*\)")
# A handle-free box group in the prose ("a teammate (3)"), never a mark's own brackets.
_STRAY_PAREN_NUM_RE = re.compile(r"(?<!\])\s*\(\s*\d{1,2}(?:\s*,\s*\d{1,2})*\s*\)")
_EMPTY_PAREN_RE = re.compile(r"\s*\(\s*(?:and|,)?\s*\)")
# A box reference the model wrote as an aside: "(box 2)", "in box (5)", "(boxes 3 and 4)".
_BOX_ASIDE_RE = re.compile(r"\s*\(\s*box(?:es)?\s+\d{1,2}(?:\s*(?:,|and)\s*\d{1,2})*\s*\)"
                           r"|\s+(?:in|at|inside|within|behind|near|by)\s+box(?:es)?\s*\(?\s*\d{1,2}\s*\)?"
                           r"(?:\s*(?:,|and)\s*\(?\s*\d{1,2}\s*\)?)*(?=[\s,.;])", re.I)
# A mark naming more boxes than ANCHOR_RE's 24 characters ("[boys](2,3,4,...,16)"): a crowd, kept as
# words without a span.
_LONG_MARK_RE = re.compile(r"\[([^\[\]\n]{1,160})\]\(\s*\d{1,2}(?:\s*:\s*P\d{1,6})?(?:\s*,\s*\d{1,2}(?:\s*:\s*P\d{1,6})?){6,}\s*\)")
_STRAY_V2_RE = re.compile(r"\s*\[\s*\d{1,2}(?:\s*:\s*P\d{1,6})?(?:\s*[,-]\s*\d{1,2}(?:\s*:\s*P\d{1,6})?)*\s*\]")

FEMALE_WORDS_RE = re.compile(r"\b(girls?|wom[ae]n|she|her|hers|herself|female|lady|ladies|"
                             r"daughters?|mother|mom|sister)\b", re.I)
MALE_WORDS_RE = re.compile(r"\b(boys?|m[ae]n|he|him|his|himself|male|gentleman|gentlemen|"
                           r"sons?|father|dad|brother)\b", re.I)
_ADULT_WORDS_RE = re.compile(r"\b(m[ae]n|wom[ae]n|teens?|teenagers?|teenage|adults?|lady|ladies|"
                             r"gentlem[ae]n)\b", re.I)
_CHILD_WORDS_RE = re.compile(r"\b(boys?|girls?|child|children|kids?|toddlers?|bab(?:y|ies))\b", re.I)
_PLURAL_FEMALE_RE = re.compile(r"\b(girls|women|ladies)\b", re.I)
_PLURAL_MALE_RE = re.compile(r"\b(boys|men|gentlemen)\b", re.I)
ROSTER_WINDOW = 40
# The roster's facts pasted into the text ("A boy, 10 walks", "a boy aged 10"): dev r2, 38 of 125.
AGE_COPY_RE = re.compile(r"\b(?:boy|girl|man|woman)s?,\s*\d{1,2}\b|\baged\s+\d{1,2}\b|\b\d{1,2}-year-old\b", re.I)


def _trailing(desc: str, sp: dict, nxt: Optional[int]) -> str:
    """Up to ROSTER_WINDOW characters after a span, stopping at the next span or the
    sentence end (a following "[a girl](2)" describes someone else)."""
    end = min(sp["end"] + ROSTER_WINDOW, nxt if nxt is not None else len(desc))
    t = desc[sp["end"]:end]
    cut = re.search(r"[.!?;]", t)
    return t[:cut.start()] if cut else t


def roster_conflict(phrase: str, trailing: str, members: list[dict]) -> Optional[str]:
    """Why the words contradict a roster member's listed gender or age, else None. One
    person: the phrase and the trailing words. A group: plural nouns in the phrase only
    ("two girls" with a boy in it; "two men" with a 10-year-old)."""
    if len(members) == 1:
        e = members[0]
        text = phrase + " " + trailing
        g, age = e.get("gender"), e.get("age")
        if g == "male" and FEMALE_WORDS_RE.search(text):
            return f"female word {FEMALE_WORDS_RE.search(text).group(0)!r} on a male"
        if g == "female" and MALE_WORDS_RE.search(text):
            return f"male word {MALE_WORDS_RE.search(text).group(0)!r} on a female"
        if age is not None and age <= 12 and _ADULT_WORDS_RE.search(phrase):
            return f"{_ADULT_WORDS_RE.search(phrase).group(0)!r} on age {age}"
        if age is not None and age >= 18 and _CHILD_WORDS_RE.search(phrase):
            return f"{_CHILD_WORDS_RE.search(phrase).group(0)!r} on age {age}"
        return None
    fem, mal = _PLURAL_FEMALE_RE.search(phrase), _PLURAL_MALE_RE.search(phrase)
    for e in members:
        g, age = e.get("gender"), e.get("age")
        if g == "male" and fem and not mal:
            return f"{fem.group(0)!r} with a male member"
        if g == "female" and mal and not fem:
            return f"{mal.group(0)!r} with a female member"
        if age is not None and age <= 12 and _ADULT_WORDS_RE.search(phrase) and not _CHILD_WORDS_RE.search(phrase):
            return f"{_ADULT_WORDS_RE.search(phrase).group(0)!r} with a member aged {age}"
        if age is not None and age >= 18 and _CHILD_WORDS_RE.search(phrase) and not _ADULT_WORDS_RE.search(phrase):
            return f"{_CHILD_WORDS_RE.search(phrase).group(0)!r} with a member aged {age}"
    return None


def cross_check(written: dict, roster: dict) -> str:
    """One class per span (plan table): ``foreign`` > ``disagree`` > ``orphan`` >
    ``omitted`` > ``agree``; ``none`` for a span on no roster box that carries no handle."""
    ids = {e["person_id"] for e in roster.values()}
    classes = set()
    for box, pid in written.items():
        e = roster.get(box)
        if pid is not None and pid not in ids:
            classes.add("foreign")
        elif e is not None:
            classes.add("agree" if pid == e["person_id"] else "omitted" if pid is None else "disagree")
        elif pid is not None:
            classes.add("orphan")
    for c in ("foreign", "disagree", "orphan", "omitted", "agree"):
        if c in classes:
            return c
    return "none"


def _degenerate_spans(desc: str, spans: list[dict]) -> dict:
    """Span index -> why the text bent around it (as :func:`degenerate`, per span)."""
    out, by_phrase = {}, {}
    for i, sp in enumerate(spans):
        by_phrase.setdefault(sp["phrase"].lower(), []).append(i)
        prev = re.findall(r"[A-Za-z]+", desc[max(0, sp["start"] - 30):sp["start"]])
        if prev and _PERSON_NOUN_RE.fullmatch(prev[-1]):
            out[i] = f"mark follows a person noun: {prev[-1]!r}"
    for k, idx in by_phrase.items():
        if len(idx) >= 3:
            for i in idx:
                out.setdefault(i, f"phrase {k!r} marked 3+ times")
    return out


def parse_v2(raw: str, n_boxes: int, roster: Optional[dict] = None,
             names: Optional[list[str]] = None) -> dict:
    """``{"description", "spans", "dropped", "degenerate", "roster_confused", "counts"}``
    with the markup stripped. Each span: ``start, end, phrase, boxes, written`` ({box:
    person_id | None}), ``check`` (:func:`cross_check`), ``count_problem``,
    ``degenerate``, ``roster_ok`` / ``roster_conflict``. A mark whose round brackets
    break the grammar, or name a box outside 1..n, or more than ``MAX_GROUP`` boxes, is
    stripped without a span (``dropped``). Raises ``describe.UnusableAnswer`` on an empty
    text, stray markup, overlay words, a P-number or "person N" in the prose (never
    stripped silently), or a roster name (code knows them; none were sent)."""
    from .describe import UnusableAnswer
    roster = roster or {}
    text = (raw or "").strip()
    out, spans, dropped, pos, length = [], [], [], 0, 0
    for m in ANCHOR_RE.finditer(text):
        before = text[pos:m.start()]
        out.append(before)
        length += len(before)
        phrase, inner = m.group(1).strip(), m.group(2)
        written = None
        if _INNER_V2_RE.match(inner):
            written = {}
            for b, p in _INNER_ITEM_RE.findall(inner):
                written[int(b)] = int(p) if p else None
        if (written is None or len(written) > MAX_GROUP
                or any(b < 1 or b > n_boxes for b in written)):
            dropped.append({"phrase": phrase, "inner": inner})
        else:
            spans.append({"start": length, "end": length + len(phrase), "phrase": phrase,
                          "boxes": sorted(written), "written": written,
                          "check": cross_check(written, roster),
                          "count_problem": count_problem(phrase, len(written))})
        out.append(phrase)
        length += len(phrase)
        pos = m.end()
    out.append(text[pos:])
    desc = "".join(out)
    if desc and desc[0].islower():
        desc = desc[0].upper() + desc[1:]
    if not desc.strip():
        raise UnusableAnswer("grounded describe: empty description")
    if _BRACKET_RE.search(desc) or _STRAY_PAREN_RE.search(desc):
        raise UnusableAnswer(f"grounded describe: stray markup: {desc[:80]!r}")
    hit = ID_IN_PROSE_RE.search(desc)
    if hit:
        raise UnusableAnswer(f"grounded describe: id in text: {hit.group(0)!r}")
    leak = LEAK_RE.search(desc)
    if leak:
        raise UnusableAnswer(f"grounded describe: overlay words: {leak.group(0)!r}")
    hit = name_in_text(desc, names or [])
    if hit:
        raise UnusableAnswer(f"grounded describe: name in text: {hit!r}")
    hit = AGE_COPY_RE.search(desc) if roster else None
    if hit:
        raise UnusableAnswer(f"grounded describe: age copy: {hit.group(0)!r}")
    deg = _degenerate_spans(desc, spans)
    starts = [sp["start"] for sp in spans]
    for i, sp in enumerate(spans):
        sp["degenerate"] = deg.get(i)
        members = [roster[b] for b in sp["boxes"] if b in roster]
        why = roster_conflict(sp["phrase"], _trailing(desc, sp, starts[i + 1] if i + 1 < len(starts) else None),
                              members) if members else None
        sp["roster_ok"], sp["roster_conflict"] = why is None, why
    counts = {}
    for sp in spans:
        counts[sp["check"]] = counts.get(sp["check"], 0) + 1
    return {"description": desc, "spans": spans, "dropped": dropped,
            "degenerate": degenerate(desc, spans), "counts": counts,
            "roster_confused": counts.get("disagree", 0) >= 2}


def strip_marks_v2(raw: str, n_boxes: int, roster: Optional[dict] = None,
                   names: Optional[list[str]] = None) -> dict:
    """Last resort after a corrective turn: drop stray ``[n]`` / ``[n:P39]`` marks and
    keep the text with NO spans (names nobody). Box asides ("(box 2)", "in box (5)") and
    handle-free ``(n)`` groups left in the prose
    go too; a group with a P-number stays and fails the photo (never strip an id silently).
    Still raises on everything else."""
    text = _BOX_ASIDE_RE.sub("", _LONG_MARK_RE.sub(lambda m: m.group(1), raw or ""))
    text = _EMPTY_PAREN_RE.sub("", _STRAY_PAREN_NUM_RE.sub("", _STRAY_V2_RE.sub("", text)))
    return dict(parse_v2(text, n_boxes, roster, names),
                spans=[], unanchored=True)


def name_spans(spans: list[dict], roster: dict, people: Optional[int] = None,
               bare_rule: Optional[str] = "everyone") -> list[dict]:
    """Set ``named`` / ``why_generic`` / ``person_ids`` / ``face_ids`` / ``body_bboxes`` on
    each span (in place; returns ``spans``). A span is named only if: ``agree``, every box
    a roster box (groups all-or-nothing), ``roster_ok``, no count problem, not degenerate,
    the bare-phrase rule (``people`` = everyone detected), template-safe words, and no
    member already named in an earlier span of the photo."""
    from .people_bind import bare_allowed
    seen = set()
    for sp in sorted(spans, key=lambda s: s["start"]):
        members = [roster.get(b) for b in sp["boxes"]]
        sp["person_ids"] = [e["person_id"] if e else None for e in members]
        sp["face_ids"] = [e.get("face_id") if e else None for e in members]
        sp["body_bboxes"] = [e.get("body_bbox") if e else None for e in members]
        why = None
        if sp["check"] != "agree":
            why = sp["check"]
        elif any(e is None for e in members):
            why = "unknown member"
        elif not sp.get("roster_ok", True):
            why = "roster conflict"
        elif sp.get("count_problem"):
            why = "count"
        elif sp.get("degenerate"):
            why = "degenerate"
        elif not bare_allowed(sp["phrase"], len(sp["boxes"]), people, bare_rule):
            why = "bare phrase"
        elif re.search(r"[{}|]", sp["phrase"]):
            why = "template characters"
        elif seen & set(sp["person_ids"]):
            why = "already named"
        sp["named"], sp["why_generic"] = why is None, why
        if why is None:
            seen |= set(sp["person_ids"])
    return spans


def assemble_v2(description: str, spans: list[dict]) -> str:
    """The write-once named template: ``{person:39|a boy in a green jersey}`` (a group:
    ``{person:39,12|two boys}``) for each named span; every other span stays plain words.
    With no span named it equals ``description`` byte for byte."""
    out, pos = [], 0
    for sp in sorted(spans, key=lambda s: s["start"]):
        if sp.get("named"):
            out.append(description[pos:sp["start"]])
            out.append("{person:%s|%s}" % (",".join(str(p) for p in sp["person_ids"]),
                                           description[sp["start"]:sp["end"]]))
            pos = sp["end"]
    out.append(description[pos:])
    return "".join(out)


def draw_boxes_labelled(image_path: str, boxes: list, labels: dict, max_px: int = 1024) -> str:
    """As ``people_bind.draw_boxes`` (left untouched), but box ``n`` in ``labels`` is
    tagged ``"4·P39"`` instead of ``"4"`` (the ``v2-id-drawn`` arm)."""
    import base64
    import io
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    with Image.open(image_path) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    scale = 1.0
    if max(im.size) > max_px:
        scale = max_px / float(max(im.size))
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    fs = max(14, im.size[1] // 28)
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()
    tags: list = []
    for n, box in enumerate(boxes, 1):
        top, right, bottom, left = (v * scale for v in box)
        draw.rectangle([left, top, right, bottom], outline=(255, 40, 40), width=3)
        label = f"{n}·{labels[n]}" if n in labels else str(n)
        tx, ty = left, max(0, top - fs - 4)
        tb = draw.textbbox((tx, ty), label, font=font)
        while any(tb[0] < o[2] and o[0] < tb[2] and tb[1] < o[3] and o[1] < tb[3] for o in tags):
            tx += tb[2] - tb[0] + 3
            tb = draw.textbbox((tx, ty), label, font=font)
        tags.append(tb)
        draw.rectangle(tb, fill=(255, 40, 40))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()

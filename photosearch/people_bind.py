"""people-bind: which phrase of a photo's description is which face box.

docs/plans/description-names.md. The pass is LABEL-FREE: it never sees a
person's name or id, only anonymous face boxes, so what it records is a pixel
fact ("this phrase is that box") that survives every relabel. Names are joined
at read time by :mod:`photosearch.description_render`.

Output is a list of *slots*; a slot is character spans of the RAW description
plus the face(s) they refer to::

    {"slot": "s1", "kind": "person",
     "spans": [{"start": 0, "end": 29, "text": "A young boy with reddish hair"}],
     "faces": [{"face_id": 516669, "bbox": [1936, 3387, 2228, 3153]}]}

``bbox`` is (top, right, bottom, left) in the EXIF-oriented pixel space that
``faces.bbox_*`` uses: the fingerprint render compares against.

Two tiers decide a slot, reported separately by the eval:

* the **rule tier** (no model): one offered face, one singular person phrase,
  no other-person or collective word anywhere, ``subject_boxes`` present with
  the face centre inside a person-labelled box;
* the **VLM tier**: numbered anonymous boxes drawn on the image. The model
  lists every box a phrase *could* describe; **code** binds only when the
  number of boxes equals the number of people the phrase names, and a second
  call with re-shuffled numbering must agree.

Everything here is pure (no I/O, no model call) except :func:`draw_overlay`,
which imports PIL lazily.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Optional

# Bump to re-bind every photo (cf. SHARPNESS_VERSION): stored in the row.
BIND_VERSION = 1

# The quality floor a face must clear to be DRAWN. Mirrors db.PEOPLE_BIND_*
# (and faces.CLUSTER_MIN_*); tests/test_people_bind_validate.py pins the three.
MIN_DET_SCORE = 0.65
MIN_BBOX_EDGE = 60
MAX_FACES = 6          # a photo with more offered faces is never claimed
MAX_GROUP = 3          # a group slot names 2-3 people, never more

# ---------------------------------------------------------------------------
# Vocabulary (closed lists on purpose: a miss loses coverage, never accuracy)
# ---------------------------------------------------------------------------

# Singular person head nouns the rule tier will bind.
_SINGULAR_HEADS = (
    "boy", "girl", "man", "woman", "child", "kid", "toddler", "baby", "teen",
    "teenager", "player", "goalkeeper", "goalie", "keeper", "athlete",
    "soccer player", "football player", "person", "youth", "infant", "lady",
    "gentleman", "adult", "runner", "swimmer", "skier", "rider", "cyclist",
    "dancer", "photographer", "referee", "coach", "parent", "mother", "father",
    "mom", "dad", "spectator", "fan",
)
# Plural / collective words: the phrase names a crowd, the rule tier refuses.
_PLURAL_HEADS = (
    "boys", "girls", "men", "women", "children", "kids", "toddlers", "babies",
    "teens", "teenagers", "players", "goalkeepers", "athletes", "people",
    "persons", "youths", "adults", "runners", "parents", "spectators", "fans",
    "teammates", "crowd", "group", "team", "audience", "onlookers", "others",
    "coaches", "referees", "family", "pair", "couple", "duo", "trio", "both",
    "bystanders", "friends",
)
# Words that make a singular phrase refer to ANOTHER person.
_OTHER_WORDS = ("another", "other", "others", "second", "companion")

_ADJ_ALLOWED = re.compile(r"^[a-z][a-z'\-]*$")
_DETERMINERS = ("a", "an", "the", "one", "this", "that")
# "his father", "her coach": a possessive makes a second person even though it
# can never START a bound span (rule_bind only binds a/an/the/one).
_POSSESSIVES = ("his", "her", "their", "its", "my", "our", "your")
# A closed "with <attribute>" tail the span may be extended over, so the
# 248080 example reads "Zack walks across..." and not "Zack with reddish hair
# walks across...". The tail words are physical attributes, never a person.
_TAIL_RE = re.compile(
    r"\s+with\s+(?:(?:a|an|the|his|her|their)\s+)?"
    r"(?:(?:short|long|curly|straight|wavy|dark|light|brown|black|blonde|blond|"
    r"red|reddish|auburn|ginger|gray|grey|white|messy|spiky|shaggy|bald|"
    r"shoulder-length|cropped|tousled|braided|blue|green|hazel)\s+)*"
    r"(?:hair|eyes|beard|mustache|moustache|braids|ponytail|freckles|glasses|"
    r"smile)\b",
    re.IGNORECASE)

_HEADS_ALT = "|".join(sorted((re.escape(h) for h in _SINGULAR_HEADS),
                             key=len, reverse=True))
_PLURAL_ALT = "|".join(sorted((re.escape(h) for h in _PLURAL_HEADS),
                              key=len, reverse=True))
# determiner + up to 4 modifier words + a singular head
_NP_RE = re.compile(
    r"\b(?P<det>" + "|".join(_DETERMINERS + _POSSESSIVES) + r")\s+"
    r"(?P<mods>(?:[A-Za-z][A-Za-z'\-]*\s+){0,4}?)"
    r"(?P<head>" + _HEADS_ALT + r")\b",
    re.IGNORECASE)
_HEAD_RE = re.compile(r"\b(?:" + _HEADS_ALT + r")\b", re.IGNORECASE)
_PLURAL_RE = re.compile(r"\b(?:" + _PLURAL_ALT + r")\b", re.IGNORECASE)
_OTHER_RE = re.compile(r"\b(?:" + "|".join(_OTHER_WORDS) + r")\b", re.IGNORECASE)
# "two players", "three boys", "several children": a counted group
_NUMBER_WORDS = {"two": 2, "three": 3, "both": 2, "a pair of": 2, "a couple of": 2}
_COUNT_RE = re.compile(
    r"\b(?P<n>two|three|both|four|five|six|seven|several|many|some|few|various|"
    r"multiple|a\s+few|a\s+group\s+of|a\s+pair\s+of|a\s+couple\s+of)\b",
    re.IGNORECASE)
_PERSON_BOX_LABEL_RE = re.compile(
    r"\b(?:person|people|boy|girl|man|woman|child|kid|player|goalkeeper|"
    r"toddler|baby|athlete|human|soccer|football|runner|skier|rider)\b",
    re.IGNORECASE)
# A modifier that is itself a person word ("a boy's", "the father and son")
# would make the "mods" capture swallow a second person.
_PERSONISH = re.compile(r"\b(?:" + _HEADS_ALT + "|" + _PLURAL_ALT + r")\b",
                        re.IGNORECASE)


def sha_of(description: str) -> str:
    return hashlib.sha1((description or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------

def face_edge(f: dict) -> int:
    """Shorter side of a face box, original pixels. ``f['bbox']`` is
    (top, right, bottom, left); falls back to the bbox_* column names."""
    top, right, bottom, left = _bbox(f)
    return min(bottom - top, right - left)


def _bbox(f: dict) -> tuple:
    bb = f.get("bbox")
    if isinstance(bb, dict):
        return (bb["top"], bb["right"], bb["bottom"], bb["left"])
    if bb is not None:
        return tuple(bb)
    return (f["bbox_top"], f["bbox_right"], f["bbox_bottom"], f["bbox_left"])


def face_id(f: dict) -> int:
    return int(f["face_id"] if "face_id" in f else f["id"])


def offered_faces(faces: list[dict]) -> list[dict]:
    """The faces that get DRAWN: clustering quality floor, labelled or not.
    ``det_score IS NULL`` is grandfathered (pre-v20 rows)."""
    out = []
    for f in faces:
        ds = f.get("det_score")
        if ds is not None and ds < MIN_DET_SCORE:
            continue
        if face_edge(f) < MIN_BBOX_EDGE:
            continue
        out.append({"face_id": face_id(f), "bbox": list(_bbox(f)),
                    "det_score": ds})
    return out


def face_center_norm(f: dict, image_w: int, image_h: int) -> Optional[tuple]:
    if not image_w or not image_h:
        return None
    top, right, bottom, left = _bbox(f)
    return (((left + right) / 2.0) / image_w, ((top + bottom) / 2.0) / image_h)


# ---------------------------------------------------------------------------
# Phrase finding
# ---------------------------------------------------------------------------

def _clean_mods(mods: str) -> Optional[list[str]]:
    words = mods.split()
    for w in words:
        if not _ADJ_ALLOWED.match(w.lower()):
            return None
        if w.lower() in _DETERMINERS or w.lower() in ("and", "or", "of", "with",
                                                       "in", "on", "at", "by"):
            return None
        if _PERSONISH.fullmatch(w):
            return None
    return words


def find_singular_phrases(description: str) -> list[dict]:
    """Every ``det + modifiers + singular-person head`` phrase, with the
    optional ``with <hair/eyes/...>`` tail folded into the span. Possessives
    stay outside the span ("the boy's" -> span "the boy")."""
    out = []
    for m in _NP_RE.finditer(description):
        if _clean_mods(m.group("mods")) is None:
            continue
        start, end = m.start(), m.end()
        t = _TAIL_RE.match(description, end)
        if t:
            end = t.end()
        out.append({"start": start, "end": end, "text": description[start:end],
                    "head": m.group("head").lower(),
                    "det": m.group("det").lower()})
    return out


def has_crowd_or_other_person(description: str, phrases: list[dict]) -> Optional[str]:
    """Why the rule tier must refuse this description, or None.

    Refuses on: any plural/collective word, any counting word near a person
    head, any "another/other", and a second distinct person head."""
    if _PLURAL_RE.search(description):
        return "plural or collective word"
    if _OTHER_RE.search(description):
        return "other-person word"
    if _COUNT_RE.search(description):
        return "counting word"
    heads = {p["head"] for p in phrases}
    if len(heads) > 1:
        return "two different person nouns"
    # a person noun the phrase finder did not capture ("the boy and father")
    if len(_HEAD_RE.findall(description)) != len(phrases):
        return "person noun outside a phrase"
    return None


def _in_person_box(face: dict, boxes: Optional[list], image_w: int,
                   image_h: int) -> bool:
    """True when the face centre lies inside the LARGEST subject box AND that
    box is labelled as a person. No usable boxes -> False (the caller sends
    the photo to the VLM tier: a lone detected face may be a sideline parent
    while the described child's face was undetected)."""
    if not boxes:
        return False
    c = face_center_norm(face, image_w, image_h)
    if c is None:
        return False
    try:
        big = max(boxes, key=lambda b: b.get("area_frac") or 0)
        x0, y0, x1, y1 = big["bbox"]
    except (KeyError, TypeError, ValueError):
        return False
    if not _PERSON_BOX_LABEL_RE.search(big.get("label") or ""):
        return False
    return x0 <= c[0] <= x1 and y0 <= c[1] <= y1


def _slot(idx: int, kind: str, spans: list[dict], faces: list[dict]) -> dict:
    return {"slot": f"s{idx}", "kind": kind,
            "spans": [{"start": s["start"], "end": s["end"], "text": s["text"]}
                      for s in spans],
            "faces": [{"face_id": f["face_id"], "bbox": list(f["bbox"])}
                      for f in faces]}


def rule_bind(description: str, faces: list[dict], subject_boxes: Optional[list],
              image_w: int, image_h: int) -> Optional[list]:
    """The no-model tier. Returns ``[slot]`` when every condition holds and
    ``None`` when this tier cannot decide (send to the VLM tier). It never
    answers ``[]``: "nothing to bind" is the VLM tier's call, because a
    description the closed noun list does not recognise ("the midfielder") may
    still name a person. Never guesses."""
    offered = offered_faces(faces)
    phrases = find_singular_phrases(description)
    if not phrases:
        # Nothing the closed list recognises: maybe "the midfielder", maybe a
        # crowd. Not this tier's call.
        return None
    if len(offered) != 1:
        return None
    if has_crowd_or_other_person(description, phrases):
        return None
    if not _in_person_box(offered[0], subject_boxes, image_w, image_h):
        return None
    # All phrases share one head noun (checked above) = one referent. Bind the
    # first occurrence and every later "the <head>" repeat.
    spans = [p for i, p in enumerate(phrases)
             if i == 0 or p["det"] in ("the", "this", "that")]
    if phrases[0]["det"] not in ("a", "an", "the", "one"):
        return None
    return [_slot(1, "person", _non_overlapping(spans), offered)]


def _non_overlapping(spans: list[dict]) -> list[dict]:
    out, last = [], -1
    for s in sorted(spans, key=lambda s: s["start"]):
        if s["start"] >= last:
            out.append(s)
            last = s["end"]
    return out


# ---------------------------------------------------------------------------
# VLM tier
# ---------------------------------------------------------------------------

PEOPLE_BIND_PROMPT = """\
The image has numbered boxes drawn around faces. Below is a description of the \
photo. For every phrase in the description that refers to ONE individual \
person or a small group of 2 or 3 people, list the numbers of EVERY box that \
phrase could equally describe.

Description:
{description}

Boxes: {box_numbers}

Rules:
- Copy each phrase EXACTLY as written in the description (same words, same \
case). Use the whole noun phrase, such as "a young boy with reddish hair".
- List every box the phrase could describe. If the description does not \
distinguish between several boxes, list all of them.
- Leave out phrases about people who have no box (face hidden, back turned, \
out of frame), crowds, spectators, "the team" or "others", and pronouns \
(he, she, they).
- If the phrase refers to several people, set "all_members_boxed" to false \
unless every one of them has a box.
Answer with only a JSON array, one object per phrase:
[{{"phrase": "...", "could_be": [1, 2], "kind": "person", "all_members_boxed": true}}]
Use "kind": "group" for two or three people. Answer [] if no phrase qualifies."""


def build_prompt(description: str, n_boxes: int) -> str:
    return PEOPLE_BIND_PROMPT.format(
        description=description,
        box_numbers=", ".join(str(i) for i in range(1, n_boxes + 1)))


def shuffled_orders(face_ids: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Two box-number orders for the two VLM calls: the first is a seeded
    shuffle, the second its reverse, so the numbering of any face always
    differs between the calls (for 2+ faces). ``order[i]`` is the face drawn
    as box ``i + 1``."""
    first = list(face_ids)
    random.Random(seed).shuffle(first)
    if len(first) >= 2 and first == sorted(face_ids) and len(first) == 2:
        first.reverse()    # deterministic, but never "left-to-right by id"
    return first, list(reversed(first))


def _json_array(raw: str):
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    try:
        return json.loads(text)
    except ValueError:
        pass
    # tolerate prose around the array: the outermost [...]
    i, j = text.find("["), text.rfind("]")
    if i != -1 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except ValueError:
            return None
    return None


def parse_answer(raw: Optional[str], description: str, n_boxes: int) -> list[dict]:
    """Parse and validate one VLM answer into mentions
    ``[{phrase, could_be:[int], kind, all_members_boxed}]``.

    ``[]`` is a valid answer. Unparseable output, a phrase that is not an
    exact substring of the description, or a box number out of range raises
    :class:`describe.UnusableAnswer` (the worker records one attempt and an
    ``index_errors`` row; nothing is written). Bad model output is a failure,
    not a result."""
    from .describe import UnusableAnswer
    obj = _json_array(raw or "")
    if obj is None or not isinstance(obj, list):
        raise UnusableAnswer("people-bind: not a JSON array")
    out = []
    for it in obj:
        if not isinstance(it, dict):
            raise UnusableAnswer("people-bind: array item is not an object")
        phrase = it.get("phrase")
        cb = it.get("could_be")
        if not isinstance(phrase, str) or phrase not in description:
            raise UnusableAnswer(f"people-bind: phrase not in the description: {phrase!r}")
        if not isinstance(cb, list) or not all(isinstance(n, int) and not isinstance(n, bool)
                                               for n in cb):
            raise UnusableAnswer("people-bind: could_be must be a list of box numbers")
        if any(n < 1 or n > n_boxes for n in cb):
            raise UnusableAnswer("people-bind: box number out of range")
        kind = it.get("kind", "person")
        if kind not in ("person", "group"):
            raise UnusableAnswer(f"people-bind: bad kind {kind!r}")
        out.append({"phrase": phrase, "could_be": sorted(set(cb)),
                    "kind": kind,
                    "all_members_boxed": it.get("all_members_boxed", True) is not False})
    return out


def phrase_people_count(phrase: str) -> Optional[int]:
    """How many people a phrase names: 1 for a singular person phrase, 2-3 for a
    counted group ("two players"), ``None`` when unknowable ("players",
    "several children", a crowd) or when it is not a person phrase we can
    count. The count is decided HERE, not by the model."""
    low = phrase.lower()
    cm = _COUNT_RE.search(low)
    if cm:
        word = cm.group("n").lower()
        n = {"two": 2, "three": 3, "both": 2}.get(word)
        if n and _PLURAL_RE.search(low):
            return n
        return None
    if _PLURAL_RE.search(low) and not _NP_RE.search(phrase):
        return None
    if _OTHER_RE.search(low):
        return None
    ms = list(_NP_RE.finditer(phrase))
    if len(ms) == 1:
        return 1
    return None


def _single_occurrence(description: str, phrase: str) -> Optional[tuple]:
    low, lp = description.lower(), phrase.lower()
    i = description.find(phrase)
    if i == -1 or low.count(lp) != 1:
        return None     # repeated phrase: it may be a different person each time
    return i, i + len(phrase)


def decide(description: str, mentions: list[dict], order: list[int],
           faces_by_id: dict) -> dict:
    """Mentions of ONE call -> ``{phrase: (kind, [face_id...])}`` for the phrases
    code accepts. ``order[i]`` is the face drawn as box ``i + 1``.

    Bind only when ``len(could_be) == people the phrase names`` (a 3-box
    answer for "a boy in blue" is ambiguous by construction and stays
    generic); the phrase must occur once, be a recognised count, and, for a
    group, the model must say every member is boxed."""
    out: dict = {}
    for m in mentions:
        phrase = m["phrase"]
        n = phrase_people_count(phrase)
        if n is None or n > MAX_GROUP:
            continue
        if _single_occurrence(description, phrase) is None:
            continue
        if len(m["could_be"]) != n:
            continue
        if n >= 2 and not m.get("all_members_boxed", True):
            continue
        kind = "person" if n == 1 else "group"
        if m.get("kind") and m["kind"] != kind:
            continue
        fids = [order[b - 1] for b in m["could_be"]]
        if any(f not in faces_by_id for f in fids) or len(set(fids)) != n:
            continue
        out[phrase] = (kind, sorted(fids))
    return out


def merge_calls(description: str, decided: list[dict], faces_by_id: dict) -> list[dict]:
    """Slots from one or two calls' :func:`decide` results. With two calls a
    phrase survives only if BOTH decided it and agree on the face set."""
    if not decided:
        return []
    keep = dict(decided[0])
    for other in decided[1:]:
        keep = {p: v for p, v in keep.items() if other.get(p) == v}
    slots, used_faces, last_end = [], set(), -1
    placed = []
    for phrase, (kind, fids) in keep.items():
        occ = _single_occurrence(description, phrase)
        if occ is None:
            continue
        placed.append((occ[0], occ[1], phrase, kind, fids))
    for start, end, phrase, kind, fids in sorted(placed):
        if start < last_end or used_faces & set(fids):
            continue                  # overlapping span / one face in two slots
        last_end = end
        used_faces |= set(fids)
        slots.append(_slot(len(slots) + 1, kind,
                           [{"start": start, "end": end, "text": phrase}],
                           [faces_by_id[f] for f in fids]))
    return slots


# ---------------------------------------------------------------------------
# Server-side validation (the submit re-checks everything against NOW)
# ---------------------------------------------------------------------------

class InvalidBinding(ValueError):
    """A submitted binding does not hold against the CURRENT description / faces."""


def validate_slots(description: str, slots: list, photo_face_ids: set) -> list:
    """Raise :class:`InvalidBinding` unless every span is exactly the
    description's text at its offsets, spans do not overlap, every face belongs
    to the photo, no face is in two slots and group sizes are 2-3. Returns the
    slots unchanged."""
    if not isinstance(slots, list):
        raise InvalidBinding("slots is not a list")
    seen_faces: set = set()
    spans = []
    for s in slots:
        if not isinstance(s, dict):
            raise InvalidBinding("slot is not an object")
        kind, fcs, sps = s.get("kind"), s.get("faces"), s.get("spans")
        if kind not in ("person", "group") or not fcs or not sps:
            raise InvalidBinding("slot needs kind, faces and spans")
        fids = [f.get("face_id") for f in fcs]
        if any(fid not in photo_face_ids for fid in fids):
            raise InvalidBinding("face does not belong to this photo")
        if len(set(fids)) != len(fids) or seen_faces & set(fids):
            raise InvalidBinding("face bound twice")
        seen_faces |= set(fids)
        if kind == "person" and len(fids) != 1:
            raise InvalidBinding("person slot must have exactly one face")
        if kind == "group" and not (2 <= len(fids) <= MAX_GROUP):
            raise InvalidBinding("group slot must have 2-3 faces")
        for f in fcs:
            bb = f.get("bbox")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                raise InvalidBinding("face needs a bbox fingerprint")
        for sp in sps:
            a, b = sp.get("start"), sp.get("end")
            if not (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(description)):
                raise InvalidBinding("span out of range")
            if description[a:b] != sp.get("text"):
                raise InvalidBinding("span text is not the description's text")
            spans.append((a, b))
    spans.sort()
    for (a0, b0), (a1, b1) in zip(spans, spans[1:]):
        if a1 < b0:
            raise InvalidBinding("spans overlap")
    return slots


# ---------------------------------------------------------------------------
# Overlay + the model call (lazy heavy imports)
# ---------------------------------------------------------------------------

_OVERLAY_MAX_PX = 1024


def draw_overlay(image_path: str, faces: list[dict], order: list[int],
                 max_px: int = _OVERLAY_MAX_PX) -> str:
    """Base64 JPEG of the EXIF-oriented image with each face in ``order`` boxed
    and numbered ``1..n``. Face bboxes are already in the oriented frame
    (faces rule, "EXIF-oriented image dimensions"); they are scaled with the
    resize. Draws ONLY the faces given, labelled or not."""
    import base64
    import io
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    by_id = {f["face_id"]: f for f in faces}
    with Image.open(image_path) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    ow, oh = im.size
    scale = 1.0
    if max(im.size) > max_px:
        scale = max_px / float(max(im.size))
        im = im.resize((max(1, round(ow * scale)), max(1, round(oh * scale))),
                       Image.LANCZOS)
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(14, im.size[1] // 28))
    except TypeError:                      # old Pillow: no size argument
        font = ImageFont.load_default()
    for n, fid in enumerate(order, 1):
        top, right, bottom, left = (v * scale for v in _bbox(by_id[fid]))
        draw.rectangle([left, top, right, bottom], outline=(255, 40, 40), width=3)
        label = str(n)
        tx, ty = left, max(0, top - max(14, im.size[1] // 28) - 4)
        tb = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(tb, fill=(255, 40, 40))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ask_model(description: str, image_b64: str, n_boxes: int,
              model: str) -> Optional[list[dict]]:
    """One VLM call. Returns validated mentions, ``None`` when the backend gave
    no answer (a transport problem: the caller defers, spending nothing).
    Raises :class:`describe.UnusableAnswer` / ``TruncatedOutput`` for an answer
    that cannot be stored. Role ``bind``; temperature 0."""
    from .describe import _ollama_chat_with_retry
    messages = [{"role": "user", "content": build_prompt(description, n_boxes),
                 "images": [image_b64]}]
    options = {"temperature": 0.0, "num_ctx": 8192, "num_predict": 384}
    raw = _ollama_chat_with_retry(model=model, messages=messages, options=options,
                                  role="bind")
    if raw is None:
        return None
    return parse_answer(raw, description, n_boxes)


def bind_with_model(image_path: str, description: str, faces: list[dict],
                    photo_id: int, model: str, *, max_px: int = _OVERLAY_MAX_PX,
                    second_call: bool = True) -> Optional[list[dict]]:
    """The VLM tier for one photo: one call for one offered face, two calls with
    re-shuffled numbering for two or more. Returns the slots (possibly ``[]``),
    or ``None`` to defer (a call got no answer).

    ``max_px`` / ``second_call=False`` exist for the eval's ablations
    (overlay resolution, one call vs two); production uses the defaults."""
    offered = offered_faces(faces)
    by_id = {f["face_id"]: f for f in offered}
    if not offered or len(offered) > MAX_FACES:
        return []
    ids = [f["face_id"] for f in offered]
    orders = [ids] if len(ids) == 1 else list(shuffled_orders(ids, photo_id))
    if not second_call:
        orders = orders[:1]
    decided = []
    for order in orders:
        b64 = draw_overlay(image_path, offered, order, max_px=max_px)
        mentions = ask_model(description, b64, len(order), model)
        if mentions is None:
            return None
        decided.append(decide(description, mentions, order, by_id))
    return merge_calls(description, decided, by_id)


def bind_photo(photo: dict, image_path: Optional[str], model: str,
               ) -> tuple[Optional[list], str]:
    """Both tiers for one claimed photo (``photo['people_bind_input']`` is the
    claim payload). Returns ``(slots, method)`` or ``(None, '')`` to defer.
    The rule tier needs no image; ``image_path`` may be ``None`` until the VLM
    tier is actually needed (pass a callable to fetch lazily)."""
    inp = photo["people_bind_input"]
    description = inp["description"]
    faces = inp["faces"]
    rb = rule_bind(description, faces, inp.get("subject_boxes"),
                   inp.get("image_width") or 0, inp.get("image_height") or 0)
    if rb is not None:
        return rb, "rule"
    if callable(image_path):
        image_path = image_path()
    if not image_path:
        return None, ""
    slots = bind_with_model(image_path, description, faces, photo["id"], model)
    return slots, "vlm"

"""Read-time rendering of names into a photo's description.

docs/plans/description-names.md. ``photos.description`` is raw and never
changes. ``description_subjects`` (written by the label-free people-bind pass)
says which spans of it are which face boxes. This module is the ONLY place that
turns a span into a name, by joining the face's CURRENT ``person_id`` /
``match_source`` and the person's CURRENT name, so a relabel (Koa -> Calvin), a
rename or a person merge changes the text with no LLM run and no invalidation.

A slot becomes a name only if EVERY rule holds; otherwise its spans keep the
generic words and the slot reports why (``state``):

1. the stored ``description_sha`` still matches the raw text (else the whole
   photo renders raw);
2. every bound face still exists with the stored bbox (+-2 px); a missing face
   is re-paired one-to-one by IoU >= 0.7 among the photo's current faces
   (re-detection that kept the box), else the slot is ``stale``;
3. every bound face has a ``person_id`` and a ``match_source`` in the NAME
   sources (``db.DISPLAY_NAME_SOURCES``, narrowable by env). ``temporal``,
   ``rejected``, ``dedupe_unmatched`` and NULL never name anyone;
4. a person who owns two or more live faces in the photo (any source except
   rejected / dedupe_unmatched) is never named in that photo;
5. a group slot is all-or-nothing, with distinct people;
6. the name is the first name, or the full ``persons.name`` when another
   registered person shares that first name;
7. a phrase whose head noun implies a gender or an age band ("girl", "baby",
   "man") agrees with every person it would name, at the photo's
   ``date_taken`` (``persons.gender`` / ``birth_year``, schema v42), else the
   slot is ``attribute_conflict`` with a human ``reason``. An unset
   attribute, ``other`` or a missing date skips that check, so a library with
   no attributes renders exactly as before.

Properties the tests pin: all-generic render == raw, text differs from raw only
inside spans, deterministic, never raises (any failure -> the raw text).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .db import DISPLAY_NAME_SOURCES, DEDUPE_UNMATCHED_SOURCE
from .people_bind import sha_of

logger = logging.getLogger(__name__)

NAME_SOURCES_ENV = "PHOTOSEARCH_DISPLAY_NAME_SOURCES"
ENABLED_ENV = "PHOTOSEARCH_DISPLAY_DESCRIPTIONS"
REJECTED = "rejected"            # == faces.REJECTED_MATCH_SOURCE (pinned by a test)
BBOX_TOLERANCE_PX = 2
IOU_REPAIR = 0.7

# slot states (what the modal tooltip says)
NAMED = "named"
UNLABELLED = "unlabelled"
UNTRUSTED = "untrusted_source"
REJECTED_STATE = "rejected"
STALE = "stale"
AMBIGUOUS = "ambiguous_name"
ATTRIBUTE_CONFLICT = "attribute_conflict"

# ---------------------------------------------------------------------------
# Rule 7: head nouns that imply a gender or an age band. Closed lists on
# purpose (a miss skips the check, never invents a conflict).
# ---------------------------------------------------------------------------

_MALE = ("boy", "son", "brother", "man", "father", "dad", "husband", "gentleman")
_FEMALE = ("girl", "daughter", "sister", "woman", "mother", "mom", "lady", "wife")
_AGE_BANDS = (
    (("baby", "infant", "toddler"), (0, 3)),
    (("boy", "girl", "child", "kid"), (0, 17)),
    (("teen", "teenager"), (12, 19)),
    (("man", "woman", "adult", "parent", "father", "mother"), (18, None)),
)
_PLURALS = {
    "boys": "boy", "sons": "son", "brothers": "brother", "men": "man",
    "fathers": "father", "dads": "dad", "husbands": "husband",
    "gentlemen": "gentleman", "girls": "girl", "daughters": "daughter",
    "sisters": "sister", "women": "woman", "mothers": "mother", "moms": "mom",
    "ladies": "lady", "wives": "wife", "babies": "baby", "infants": "infant",
    "toddlers": "toddler", "children": "child", "kids": "kid", "teens": "teen",
    "teenagers": "teenager", "adults": "adult", "parents": "parent",
}
_NOUNS = {w: w for w in set(_MALE) | set(_FEMALE)
          | {w for ws, _ in _AGE_BANDS for w in ws}}
_NOUNS.update(_PLURALS)
# Exact-year births are checked at +-1 (the birthday may not have come yet),
# estimates at +-3.
BIRTH_YEAR_TOLERANCE = 1
BIRTH_YEAR_APPROX_TOLERANCE = 3


def display_enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return (env.get(ENABLED_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def name_sources(env=None) -> tuple:
    """The label sources allowed to name a face: ``db.DISPLAY_NAME_SOURCES``,
    optionally NARROWED (never widened) by ``PHOTOSEARCH_DISPLAY_NAME_SOURCES``."""
    env = os.environ if env is None else env
    raw = (env.get(NAME_SOURCES_ENV) or "").strip()
    if not raw:
        return DISPLAY_NAME_SOURCES
    asked = [s.strip() for s in raw.split(",") if s.strip()]
    return tuple(s for s in DISPLAY_NAME_SOURCES if s in asked)


@dataclass
class Rendered:
    text: str
    slots: list = field(default_factory=list)   # [{slot, generic, name, state, face_ids}]

    @property
    def changed(self) -> bool:
        return any(s["state"] == NAMED for s in self.slots)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _bb(f: dict) -> tuple:
    """(top, right, bottom, left) from a stored bbox list or a faces row."""
    if "bbox_top" in f:
        return (f["bbox_top"], f["bbox_right"], f["bbox_bottom"], f["bbox_left"])
    return tuple(f["bbox"])


def iou(a: tuple, b: tuple) -> float:
    at, ar, ab, al = a
    bt, br, bb, bl = b
    iw = min(ar, br) - max(al, bl)
    ih = min(ab, bb) - max(at, bt)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ar - al) * (ab - at) + (br - bl) * (bb - bt) - inter
    return inter / union if union > 0 else 0.0


def _same_box(a: tuple, b: tuple, tol: int = BBOX_TOLERANCE_PX) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def repair_faces(bound: list, current: list) -> Optional[list]:
    """Re-pair the faces a slot was bound to against the photo's CURRENT faces.

    ``bound`` is ``[{face_id, bbox}]``, ``current`` a list of faces rows. Returns
    the list of current face ids in the same order (a face still present with
    its box keeps its id; a missing or moved one is paired by IoU >= 0.7, one to
    one, and only when that match is unique), or ``None`` when any face cannot
    be paired. A re-detected face can therefore never produce a wrong name:
    either it is the same box, or the slot goes generic."""
    by_id = {f["id"]: f for f in current}
    out: list = [None] * len(bound)
    taken: set = set()
    for i, b in enumerate(bound):
        cur = by_id.get(b["face_id"])
        if cur is not None and _same_box(_bb(cur), tuple(b["bbox"])):
            out[i] = cur["id"]
            taken.add(cur["id"])
    for i, b in enumerate(bound):
        if out[i] is not None:
            continue
        cands = [f for f in current if f["id"] not in taken
                 and iou(_bb(f), tuple(b["bbox"])) >= IOU_REPAIR]
        if len(cands) != 1:
            return None
        out[i] = cands[0]["id"]
        taken.add(cands[0]["id"])
    return out


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def _display_token(name: str, first_counts: dict) -> str:
    name = (name or "").strip()
    first = name.split()[0] if name.split() else name
    return name if first_counts.get(first.lower(), 0) > 1 else first


def _join_names(names: list) -> str:
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def render(description: str, row: Optional[dict], faces: list, persons: dict,
           sources: Optional[tuple] = None, first_counts: Optional[dict] = None,
           attrs: Optional[dict] = None, date_taken: Optional[str] = None
           ) -> Rendered:
    """Pure render of one photo. ``row`` is its ``description_subjects`` row (or
    ``None``), ``faces`` its CURRENT faces rows (id, person_id, match_source,
    bbox_*), ``persons`` ``{person_id: name}``, ``attrs`` ``{person_id:
    {gender, birth_year, birth_year_approx}}`` (rule 7; omitted = no check) and
    ``date_taken`` the photo's date. Never raises."""
    try:
        return _render(description, row, faces, persons,
                       sources if sources is not None else name_sources(),
                       first_counts if first_counts is not None else _first_counts(persons),
                       attrs or {}, date_taken)
    except Exception:                                  # noqa: BLE001 — raw is always safe
        logger.exception("description render failed; showing the raw text")
        return Rendered(description or "")


_TOKEN_RE = re.compile(r"[a-z]+(?:['’]s\b)?")


def head_nouns(phrase: str) -> list:
    """The closed-list head noun(s) of a person phrase as ``(word, canonical)``:
    the first listed noun that is not a possessive ("a boy's mother" -> mother),
    plus listed nouns directly after it ("a baby girl" -> baby, girl). A noun
    after a post-modifier ("a woman holding a baby") is not a head."""
    low = (phrase or "").lower()
    heads: list = []
    prev_end = None
    for m in _TOKEN_RE.finditer(low):
        w = m.group(0)
        if heads:
            if w in _NOUNS and not low[prev_end:m.start()].strip(" -"):
                heads.append((w, _NOUNS[w]))
                prev_end = m.end()
                continue
            break
        if w in _NOUNS:
            heads.append((w, _NOUNS[w]))
            prev_end = m.end()
    return heads


# What a name replaces (owner review 2026-11-01): a bound span may run past its
# noun phrase. "one boy is running" -> "William" lost the only verb; "A young boy,
# wearing a blue jersey" before ", is kicking" left "Carson, is kicking". The
# name replaces the words up to a finite verb (the verb stays), and a modifier
# set off by commas takes its closing comma with it. Anything else (a trailing
# "in a blue jersey") is replaced whole, as before.
_AUX = frozenset("is are was were has have had appears appear seems seem stands stand "
                 "sits sit looks look".split())
# right after the head noun these read as verbs ("stands nearby", "runs", "run")
_VERB_AFTER_HEAD = frozenset("run walk play stand sit watch look appear seem kick chase "
                             "celebrate pose smile wait hold wear jump dribble".split())
_NOT_VERBS = frozenset("jerseys shorts glasses sunglasses cleats pants shoes socks gloves "
                       "colors colours stripes numbers letters its his this as".split())


def _head_end(phrase: str) -> Optional[int]:
    """Offset just past the phrase's head noun(s) (:func:`head_nouns`), or None."""
    low = phrase.lower()
    end = None
    for m in _TOKEN_RE.finditer(low):
        w = m.group(0)
        if end is not None:
            if w in _NOUNS and not low[end:m.start()].strip(" -"):
                end = m.end()
                continue
            break
        if w in _NOUNS:
            end = m.end()
    return end


_RELATIVE = frozenset(("who", "that", "which"))


def _is_verb(w: str, right_after_head: bool) -> bool:
    return w in _AUX or (right_after_head and (w in _VERB_AFTER_HEAD
                                               or (w.endswith("s") and w not in _NOT_VERBS)))


def _first_verb(phrase: str, head_end: Optional[int]) -> Optional[tuple]:
    """``(offset of the first finite verb after the head noun, offset of a who/that/
    which right before it or None)``, or None. Without a recognised head noun
    ("two other players sit") only an auxiliary from :data:`_AUX` counts."""
    start = head_end or 0
    rest = phrase.lower()[start:]
    prev = None
    for i, m in enumerate(_TOKEN_RE.finditer(rest)):
        w = m.group(0)
        right_after = head_end is not None and i == 0 and not rest[:m.start()].strip()
        if _is_verb(w, right_after):
            rel = start + prev.start() if prev is not None and prev.group(0) in _RELATIVE else None
            return start + m.start(), rel
        prev = m
    return None


def _verb_follows(text: str) -> bool:
    """The next word of ``text`` reads as a finite verb ("is kicking", "sit")."""
    m = _TOKEN_RE.search(text.lower())
    return bool(m) and not text[:m.start()].strip() and _is_verb(m.group(0), True)


def display_edit(description: str, start: int, end: int, name: str) -> tuple:
    """``(a, b, replacement)``: the part of ``description`` a name replaces for the
    span ``[start, end)``. Never reaches outside the span except for one comma
    that closes a comma-set modifier when a verb follows it."""
    phrase = description[start:end]
    h = _head_end(phrase)
    fv = _first_verb(phrase, h)
    if fv is not None:
        verb, rel = fv
        if rel is not None:                              # "a coach who is gesturing"
            return start, start + rel, name + ", "
        return start, start + verb, name + " "
    if (h is not None and phrase[h:].lstrip().startswith(",")
            and description[end:end + 1] == "," and _verb_follows(description[end + 1:])):
        return start, end + 1, name
    return start, end, name


def _photo_year(date_taken) -> Optional[int]:
    s = str(date_taken or "")[:4]
    if len(s) == 4 and s.isdigit() and 1800 < int(s) < 3000:
        return int(s)
    return None


def attribute_conflict(phrases: list, people: list, attrs: dict,
                       date_taken: Optional[str]) -> Optional[str]:
    """Rule 7. ``phrases`` are the slot's span texts, ``people`` its
    ``[(person_id, display_name)]``. Returns the reason the phrase disagrees
    with someone it would name ("phrase says girl; Asa is male"), else None.
    Unset attributes, gender ``other`` and an undated photo skip their check."""
    if not attrs:
        return None
    heads = [h for p in phrases for h in head_nouns(p)]
    if not heads:
        return None
    year = _photo_year(date_taken)
    for pid, name in people:
        a = attrs.get(pid) or {}
        gender = a.get("gender")
        if gender in ("male", "female"):
            for word, noun in heads:
                said = "male" if noun in _MALE else "female" if noun in _FEMALE else None
                if said and said != gender:
                    return f"phrase says {word}; {name} is {gender}"
        born = a.get("birth_year")
        if year is None or not isinstance(born, int) or isinstance(born, bool):
            continue
        approx = bool(a.get("birth_year_approx"))
        tol = BIRTH_YEAR_APPROX_TOLERANCE if approx else BIRTH_YEAR_TOLERANCE
        age = year - born
        for word, noun in heads:
            for nouns, (lo, hi) in _AGE_BANDS:
                if noun not in nouns:
                    continue
                if (hi is not None and age - tol > hi) or (lo is not None and age + tol < lo):
                    return (f"phrase says {word}; {name} was about {max(age, 0)} in {year} "
                            f"(born {'~' if approx else ''}{born})")
    return None


def _first_counts(persons: dict) -> dict:
    out: dict = {}
    for name in persons.values():
        parts = (name or "").split()
        if parts:
            out[parts[0].lower()] = out.get(parts[0].lower(), 0) + 1
    return out


def _render(description, row, faces, persons, sources, first_counts,
            attrs=None, date_taken=None) -> Rendered:
    if not description or not row:
        return Rendered(description or "")
    if row["description_sha"] != sha_of(description):
        return Rendered(description)              # stale row: harmless, raw
    slots = json.loads(row["slots_json"] or "[]")
    if not slots:
        return Rendered(description)

    live = {f["id"]: f for f in faces}
    # rule 4: how many live faces carry each person in this photo
    owners: dict = {}
    for f in faces:
        if (f.get("person_id") is not None
                and f.get("match_source") not in (REJECTED, DEDUPE_UNMATCHED_SOURCE)):
            owners[f["person_id"]] = owners.get(f["person_id"], 0) + 1

    edits = []                                        # (start, end, replacement)
    report = []
    for s in slots:
        generic = " ... ".join(sp["text"] for sp in s["spans"])
        paired = repair_faces(s["faces"], faces)
        info = {"slot": s["slot"], "generic": generic, "name": None,
                "state": STALE, "face_ids": [f["face_id"] for f in s["faces"]]}
        report.append(info)
        if paired is None:
            continue
        info["face_ids"] = paired
        names, state = [], NAMED
        pids = []
        for fid in paired:
            f = live[fid]
            pid, src = f.get("person_id"), f.get("match_source")
            if src == REJECTED:
                state = REJECTED_STATE
            elif pid is None:
                state = UNLABELLED
            elif src not in sources:
                state = UNTRUSTED
            elif owners.get(pid, 0) >= 2 or pid not in persons:
                state = AMBIGUOUS
            else:
                pids.append(pid)
                names.append(_display_token(persons[pid], first_counts))
                continue
            break
        if state != NAMED or len(set(pids)) != len(pids):
            info["state"] = state if state != NAMED else AMBIGUOUS
            continue
        reason = attribute_conflict([sp["text"] for sp in s["spans"]],
                                    list(zip(pids, names)), attrs, date_taken)
        if reason:                                    # rule 7: generic words
            info["state"] = ATTRIBUTE_CONFLICT
            info["reason"] = reason
            continue
        info["state"] = NAMED
        info["name"] = _join_names(names)
        for sp in s["spans"]:
            edits.append(display_edit(description, sp["start"], sp["end"], info["name"]))

    text = description
    for a, b, rep in sorted(edits, reverse=True):
        text = text[:a] + rep + text[b:]
    return Rendered(text, report)


# ---------------------------------------------------------------------------
# Batched loader (grids, Ask/MCP hits, the detail endpoint)
# ---------------------------------------------------------------------------

_CHUNK = 500


def _chunks(ids: list):
    for i in range(0, len(ids), _CHUNK):
        yield ids[i:i + _CHUNK]


def load_persons(conn) -> dict:
    return load_persons_with_attributes(conn)[0]


def load_persons_with_attributes(conn) -> tuple:
    """``({person_id: name}, {person_id: {gender, birth_year,
    birth_year_approx}})`` in ONE query of the (tiny) persons table. Only
    persons with an attribute set appear in the second dict. A pre-v42 DB has
    no attribute columns: names only, rule 7 dormant."""
    import sqlite3
    try:
        rows = conn.execute("SELECT id, name, gender, birth_year, birth_year_approx "
                            "FROM persons").fetchall()
    except sqlite3.OperationalError:
        return {r[0]: r[1] for r in conn.execute("SELECT id, name FROM persons")}, {}
    names, attrs = {}, {}
    for r in rows:
        names[r[0]] = r[1]
        if r[2] is not None or r[3] is not None:
            attrs[r[0]] = {"gender": r[2], "birth_year": r[3],
                           "birth_year_approx": bool(r[4])}
    return names, attrs


def render_many(conn, photos: list, sources: Optional[tuple] = None) -> dict:
    """``{photo_id: Rendered}`` for ``photos`` (dicts with ``id`` and
    ``description``) in three indexed queries per <= 500 ids: the rows (with
    the photo's ``date_taken``, for rule 7), the photos' faces, and the (tiny)
    persons table with its attributes. A DB without the table (a replica copy
    of a pre-v41 NAS) renders nothing: callers fall back to raw."""
    import sqlite3
    ids = [p["id"] for p in photos if p.get("description")]
    if not ids:
        return {}
    rows: dict = {}
    try:
        for chunk in _chunks(ids):
            q = ",".join("?" * len(chunk))
            for r in conn.execute(
                    f"SELECT ds.photo_id, ds.description_sha, ds.slots_json, ph.date_taken "
                    f"FROM description_subjects ds LEFT JOIN photos ph ON ph.id = ds.photo_id "
                    f"WHERE ds.photo_id IN ({q})", chunk):
                rows[r[0]] = {"description_sha": r[1], "slots_json": r[2],
                              "date_taken": r[3]}
    except sqlite3.OperationalError:
        return {}
    if not rows:
        return {}
    faces: dict = {}
    for chunk in _chunks(list(rows)):
        q = ",".join("?" * len(chunk))
        for r in conn.execute(
                f"SELECT id, photo_id, person_id, match_source, bbox_top, bbox_right, "
                f"bbox_bottom, bbox_left FROM faces WHERE photo_id IN ({q})", chunk):
            faces.setdefault(r[1], []).append(
                {"id": r[0], "person_id": r[2], "match_source": r[3], "bbox_top": r[4],
                 "bbox_right": r[5], "bbox_bottom": r[6], "bbox_left": r[7]})
    persons, attrs = load_persons_with_attributes(conn)
    fc = _first_counts(persons)
    srcs = sources if sources is not None else name_sources()
    out = {}
    for p in photos:
        row = rows.get(p["id"])
        if row:
            out[p["id"]] = render(p["description"], row, faces.get(p["id"], []),
                                  persons, srcs, fc, attrs=attrs,
                                  date_taken=row.get("date_taken"))
    return out


def attach_display_descriptions(conn, photos: list, detail: bool = False) -> None:
    """Add ``description_display`` (only when the flag is on AND the text differs)
    to each photo dict in place; ``description`` is never touched. With
    ``detail`` also add ``description_subjects`` (per-slot state, so the UI can
    say why a child is not named). Never raises."""
    if not photos or not display_enabled():
        return
    try:
        rendered = render_many(conn, photos)
    except Exception:                                  # noqa: BLE001
        logger.exception("attach_display_descriptions failed; leaving raw")
        return
    for p in photos:
        r = rendered.get(p["id"])
        if not r:
            continue
        if r.changed and r.text != p.get("description"):
            p["description_display"] = r.text
        if detail and r.slots:
            p["description_subjects"] = r.slots


def display_text(conn, photo_id: int, description: Optional[str]) -> Optional[str]:
    """The display text for one photo (flag-aware), else ``description``."""
    if not description:
        return description
    p = {"id": photo_id, "description": description}
    attach_display_descriptions(conn, [p])
    return p.get("description_display", description)


# ---------------------------------------------------------------------------
# Maintenance sweep (the persistent half of the render-time face check)
# ---------------------------------------------------------------------------

def sweep_subjects(conn, apply: bool = False, limit: Optional[int] = None) -> dict:
    """Walk ``description_subjects`` and fix what a face re-detection left
    behind, using the SAME :func:`repair_faces` the render uses: a slot whose
    faces still pair keeps (and, when a pair moved, rewrites) its row; a row
    with a face that cannot be paired is deleted together with its
    ``people-bind`` ledger row so the photo is re-queued. Dry run unless
    ``apply``. Returns counts."""
    counts = {"rows": 0, "ok": 0, "rebound": 0, "requeued": 0}
    q = "SELECT photo_id, slots_json FROM description_subjects ORDER BY photo_id"
    if limit:
        q += f" LIMIT {int(limit)}"
    for pid, slots_json in conn.execute(q).fetchall():
        counts["rows"] += 1
        current = [{"id": r[0], "bbox_top": r[1], "bbox_right": r[2],
                    "bbox_bottom": r[3], "bbox_left": r[4]}
                   for r in conn.execute(
                       "SELECT id, bbox_top, bbox_right, bbox_bottom, bbox_left "
                       "FROM faces WHERE photo_id = ?", (pid,))]
        slots = json.loads(slots_json or "[]")
        new_slots, changed, lost = [], False, False
        for s in slots:
            paired = repair_faces(s["faces"], current)
            if paired is None:
                lost = True
                break
            if paired != [f["face_id"] for f in s["faces"]]:
                by_id = {f["id"]: f for f in current}
                s = dict(s, faces=[{"face_id": fid, "bbox": list(_bb(by_id[fid]))}
                                   for fid in paired])
                changed = True
            new_slots.append(s)
        from . import description_audit
        if lost:
            counts["requeued"] += 1
            if apply:
                description_audit.record(conn, pid, description_audit.UNBOUND,
                                         {"reason": "face gone", "requeued": True})
                conn.execute("DELETE FROM description_subjects WHERE photo_id = ?", (pid,))
                conn.execute("DELETE FROM worker_processed WHERE photo_id = ? "
                             "AND pass_type = 'people-bind'", (pid,))
        elif changed:
            counts["rebound"] += 1
            if apply:
                description_audit.record(conn, pid, description_audit.REBOUND,
                                         {"faces": [f["face_id"] for s in new_slots
                                                    for f in s["faces"]]})
                conn.execute("UPDATE description_subjects SET slots_json = ? "
                             "WHERE photo_id = ?", (json.dumps(new_slots), pid))
        else:
            counts["ok"] += 1
    if apply:
        conn.commit()
    return counts

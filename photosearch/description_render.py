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
   registered person shares that first name.

Properties the tests pin: all-generic render == raw, text differs from raw only
inside spans, deterministic, never raises (any failure -> the raw text).
"""

from __future__ import annotations

import json
import logging
import os
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
           sources: Optional[tuple] = None, first_counts: Optional[dict] = None
           ) -> Rendered:
    """Pure render of one photo. ``row`` is its ``description_subjects`` row (or
    ``None``), ``faces`` its CURRENT faces rows (id, person_id, match_source,
    bbox_*), ``persons`` ``{person_id: name}``. Never raises."""
    try:
        return _render(description, row, faces, persons,
                       sources if sources is not None else name_sources(),
                       first_counts if first_counts is not None else _first_counts(persons))
    except Exception:                                  # noqa: BLE001 — raw is always safe
        logger.exception("description render failed; showing the raw text")
        return Rendered(description or "")


def _first_counts(persons: dict) -> dict:
    out: dict = {}
    for name in persons.values():
        parts = (name or "").split()
        if parts:
            out[parts[0].lower()] = out.get(parts[0].lower(), 0) + 1
    return out


def _render(description, row, faces, persons, sources, first_counts) -> Rendered:
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
        info["state"] = NAMED
        info["name"] = _join_names(names)
        for sp in s["spans"]:
            edits.append((sp["start"], sp["end"], info["name"]))

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
    return {r[0]: r[1] for r in conn.execute("SELECT id, name FROM persons")}


def render_many(conn, photos: list, sources: Optional[tuple] = None) -> dict:
    """``{photo_id: Rendered}`` for ``photos`` (dicts with ``id`` and
    ``description``) in three indexed queries per <= 500 ids: the rows, the
    photos' faces, and the (tiny) persons table. A DB without the table (a
    replica copy of a pre-v41 NAS) renders nothing: callers fall back to raw."""
    import sqlite3
    ids = [p["id"] for p in photos if p.get("description")]
    if not ids:
        return {}
    rows: dict = {}
    try:
        for chunk in _chunks(ids):
            q = ",".join("?" * len(chunk))
            for r in conn.execute(
                    f"SELECT photo_id, description_sha, slots_json FROM description_subjects "
                    f"WHERE photo_id IN ({q})", chunk):
                rows[r[0]] = {"description_sha": r[1], "slots_json": r[2]}
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
    persons = load_persons(conn)
    fc = _first_counts(persons)
    srcs = sources if sources is not None else name_sources()
    out = {}
    for p in photos:
        row = rows.get(p["id"])
        if row:
            out[p["id"]] = render(p["description"], row, faces.get(p["id"], []),
                                  persons, srcs, fc)
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

""""More of this kid" — rank a shoot's unmatched faces by closeness to one person.

Why this exists: the strict matcher (`faces.match_faces_to_persons`) only ever
compares against `face_references` — encodings created by `add-person --photo`.
Labelling faces in /faces adds none, so after hand-naming six kids on
2026-09-26 a re-run of `match_faces` matched **0** new faces. Using the hand
labels as references for the matcher's global 1.15 tolerance was measured and
rejected: on that shoot it would have written 793 labels, with **Carson** — one
labelled face — absorbing 220 of them. A single label is a catch-all for every
similar-looking child on the field, the same over-matching that makes temporal
~4% accurate on these shoots.

So this never writes. It returns a ranked, human-reviewed candidate list: the
person picks the cutoff from the grid (the distances are right there) and
applies through the audited `POST /api/faces/bulk-assign`.

Two guards make the ranking honest on a field of lookalike kids:

- **Rival distance.** Each candidate also gets its distance to the nearest
  OTHER person with trusted labels anywhere in the library. A face closer to someone else than
  to this person is flagged, never hidden — nothing says the rival's labels are
  right either — and `exclude_closer_to_other` drops them on request.
- **Already in photo.** A person appears in a photo at most once; if they are
  already on another face in that photo, the candidate is flagged.

References are the person's TRUSTED labels library-wide (`manual`,
`merge_review` — not `strict`/`temporal`, the populations the face-integrity
tools distrust), so earlier shoots help when a kid has few labels today.
"""
from __future__ import annotations

import numpy as np

from .faces import MATCHABLE_SQL, load_match_exclusions

REFERENCE_SOURCES = ("manual", "merge_review")
CHUNK = 512


def _unit(mat):
    X = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _scope(date_from, date_to):
    where, params = [], []
    if date_from:
        where.append("date(p.date_taken) >= ?"); params.append(date_from)
    if date_to:
        where.append("date(p.date_taken) <= ?"); params.append(date_to)
    return where, params


def suggest(db, *, person, date_from=None, date_to=None, max_dist=None,
            exclude_closer_to_other=False):
    """Unmatched faces in scope, nearest to `person` first.

    Returns (rows, stats). Each row: face_id, photo_id, dist (to the person's
    nearest reference), rival / rival_dist (nearest other person labelled in
    scope), closer_to_other, in_photo (person already on a face in that photo).
    """
    prow = db.conn.execute("SELECT id FROM persons WHERE name = ?", (person,)).fetchone()
    if prow is None:
        raise ValueError(f"no person named {person!r}")
    pid = prow["id"]

    where, params = _scope(date_from, date_to)
    scope_sql = (" AND " + " AND ".join(where)) if where else ""

    cand = db.conn.execute(
        f"""SELECT f.id AS face_id, f.photo_id FROM faces f
              JOIN photos p ON p.id = f.photo_id
             WHERE {MATCHABLE_SQL}{scope_sql}
             ORDER BY f.id""", params).fetchall()
    src = ",".join("?" * len(REFERENCE_SOURCES))
    refs = db.conn.execute(
        f"SELECT f.id AS face_id FROM faces f WHERE f.person_id = ? "
        f"AND f.match_source IN ({src})", (pid, *REFERENCE_SOURCES)).fetchall()
    # Rivals: every OTHER person's trusted labels, library-wide — a teammate
    # labelled only on an earlier shoot is still a lookalike today (measured:
    # 7.5k faces over 52 people, a trivial matrix).
    rivals = db.conn.execute(
        f"""SELECT f.id AS face_id, pe.name FROM faces f
              JOIN persons pe ON pe.id = f.person_id
             WHERE f.person_id <> ? AND f.match_source IN ({src})""",
        (pid, *REFERENCE_SOURCES)).fetchall()
    in_photo = {r[0] for r in db.conn.execute(
        f"""SELECT DISTINCT f.photo_id FROM faces f JOIN photos p ON p.id = f.photo_id
             WHERE f.person_id = ?{scope_sql}""", (pid, *params))}
    barred = {fid for fid, pids in load_match_exclusions(db).items() if pid in pids}

    stats = {"candidates": len(cand), "references": len(refs),
             "rival_faces": len(rivals), "measured": 0}
    if not cand or not refs:
        return [], stats

    ids = ([r["face_id"] for r in cand] + [r["face_id"] for r in refs]
           + [r["face_id"] for r in rivals])
    encs = db.get_face_encodings_bulk(ids)
    cand = [r for r in cand if r["face_id"] in encs and r["face_id"] not in barred]
    refs = [r for r in refs if r["face_id"] in encs]
    rivals = [r for r in rivals if r["face_id"] in encs]
    stats["references"] = len(refs)
    if not cand or not refs:
        return [], stats

    R = _unit([encs[r["face_id"]] for r in refs])
    Q = _unit([encs[r["face_id"]] for r in rivals]) if rivals else None
    qnames = [r["name"] for r in rivals]
    X = _unit([encs[r["face_id"]] for r in cand])

    rows = []
    for i in range(0, len(cand), CHUNK):
        sl = slice(i, i + CHUNK)
        d_self = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ R.T))).min(axis=1)
        if Q is not None:
            DQ = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ Q.T)))
            q_idx = DQ.argmin(axis=1)
            q_min = DQ[np.arange(len(q_idx)), q_idx]
        for j, r in enumerate(cand[sl]):
            rd = float(q_min[j]) if Q is not None else None
            rows.append({
                "face_id": r["face_id"], "photo_id": r["photo_id"],
                "dist": round(float(d_self[j]), 4),
                "rival": qnames[int(q_idx[j])] if Q is not None else None,
                "rival_dist": round(rd, 4) if rd is not None else None,
                "closer_to_other": rd is not None and rd < float(d_self[j]),
                "in_photo": r["photo_id"] in in_photo,
            })

    stats["measured"] = len(rows)
    a = np.array([r["dist"] for r in rows])
    stats["percentiles"] = {p: round(float(np.percentile(a, p)), 3) for p in (10, 50, 90)}
    stats["within"] = {str(t): int((a <= t).sum()) for t in (0.7, 0.8, 0.9, 1.0, 1.15)}
    if exclude_closer_to_other:
        rows = [r for r in rows if not r["closer_to_other"]]
    if max_dist is not None:
        rows = [r for r in rows if r["dist"] <= max_dist]
    rows.sort(key=lambda r: r["dist"])
    stats["selected"] = len(rows)
    return rows, stats

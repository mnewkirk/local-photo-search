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

**The year window (`window_years`).** Children change too much for "nearest
label anywhere in the library" to mean anything across a decade. Measured on
the replica 2026-10-31: 9,274 of 15,867 trusted labels are 2026 soccer kids, so
on the 2014-04-19 shoot 326 of the 2,389 faces within 1.0 of someone had a 2026
kid — not yet born in 2014 — as their nearest labelled person (2017-08-12: 135
of 1,132). With a window, both the person's references and the rivals are
limited to labels taken within that many years of each candidate face. A face
with no in-window reference of the person is dropped (counted in
`stats["outside_window"]`); one with no in-window rival simply has none.
Undated faces and labels are never excluded.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .faces import MATCHABLE_SQL, load_match_exclusions

REFERENCE_SOURCES = ("manual", "merge_review")
CHUNK = 512
YEAR_SECONDS = 365.25 * 86400


def _epoch(value):
    try:
        return datetime.fromisoformat(str(value)[:19]).timestamp()
    except (TypeError, ValueError):
        return np.nan


def _year(value):
    s = str(value or "")[:4]
    return int(s) if s.isdigit() else None


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
            exclude_closer_to_other=False, window_years=None):
    """Unmatched faces in scope, nearest to `person` first.

    Returns (rows, stats). Each row: face_id, photo_id, dist (to the person's
    nearest reference) and ref_year (that reference's year), rival /
    rival_dist / rival_year (nearest other person's label), closer_to_other,
    in_photo (person already on a face in that photo). `window_years` limits
    both comparisons to labels within that many years of the face.
    """
    prow = db.conn.execute("SELECT id FROM persons WHERE name = ?", (person,)).fetchone()
    if prow is None:
        raise ValueError(f"no person named {person!r}")
    pid = prow["id"]

    where, params = _scope(date_from, date_to)
    scope_sql = (" AND " + " AND ".join(where)) if where else ""

    cand = db.conn.execute(
        f"""SELECT f.id AS face_id, f.photo_id, p.date_taken FROM faces f
              JOIN photos p ON p.id = f.photo_id
             WHERE {MATCHABLE_SQL}{scope_sql}
             ORDER BY f.id""", params).fetchall()
    src = ",".join("?" * len(REFERENCE_SOURCES))
    refs = db.conn.execute(
        f"""SELECT f.id AS face_id, p.date_taken FROM faces f
              JOIN photos p ON p.id = f.photo_id
             WHERE f.person_id = ? AND f.match_source IN ({src})""",
        (pid, *REFERENCE_SOURCES)).fetchall()
    # Rivals: every OTHER person's trusted labels, library-wide — a teammate
    # labelled only on an earlier shoot is still a lookalike today (measured:
    # 7.5k faces over 52 people, a trivial matrix).
    rivals = db.conn.execute(
        f"""SELECT f.id AS face_id, pe.name, p.date_taken FROM faces f
              JOIN persons pe ON pe.id = f.person_id
              JOIN photos p ON p.id = f.photo_id
             WHERE f.person_id <> ? AND f.match_source IN ({src})""",
        (pid, *REFERENCE_SOURCES)).fetchall()
    in_photo = {r[0] for r in db.conn.execute(
        f"""SELECT DISTINCT f.photo_id FROM faces f JOIN photos p ON p.id = f.photo_id
             WHERE f.person_id = ?{scope_sql}""", (pid, *params))}
    barred = {fid for fid, pids in load_match_exclusions(db).items() if pid in pids}

    stats = {"candidates": len(cand), "references": len(refs),
             "rival_faces": len(rivals), "measured": 0,
             "window_years": window_years, "outside_window": 0}
    if not cand or not refs:
        return [], stats

    ids = ([r["face_id"] for r in cand] + [r["face_id"] for r in refs]
           + [r["face_id"] for r in rivals])
    encs = db.get_face_encodings_cached(ids)
    cand = [r for r in cand if r["face_id"] in encs and r["face_id"] not in barred]
    refs = [r for r in refs if r["face_id"] in encs]
    rivals = [r for r in rivals if r["face_id"] in encs]
    stats["references"] = len(refs)
    if not cand or not refs:
        return [], stats

    R = _unit([encs[r["face_id"]] for r in refs])
    Rt = np.array([_epoch(r["date_taken"]) for r in refs])
    Ry = [_year(r["date_taken"]) for r in refs]
    Q = _unit([encs[r["face_id"]] for r in rivals]) if rivals else None
    Qt = np.array([_epoch(r["date_taken"]) for r in rivals])
    Qy = [_year(r["date_taken"]) for r in rivals]
    qnames = [r["name"] for r in rivals]
    X = _unit([encs[r["face_id"]] for r in cand])
    T = np.array([_epoch(r["date_taken"]) for r in cand])

    def _outside(t, ref_t):
        # NaN compares False, so an undated face or label is never excluded.
        return np.abs(t[:, None] - ref_t[None, :]) > window_years * YEAR_SECONDS

    rows = []
    for i in range(0, len(cand), CHUNK):
        sl = slice(i, i + CHUNK)
        DS = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ R.T)))
        if window_years is not None:
            DS[_outside(T[sl], Rt)] = np.inf
        s_idx = DS.argmin(axis=1)
        d_self = DS[np.arange(len(s_idx)), s_idx]
        if Q is not None:
            DQ = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ Q.T)))
            if window_years is not None:
                DQ[_outside(T[sl], Qt)] = np.inf
            q_idx = DQ.argmin(axis=1)
            q_min = DQ[np.arange(len(q_idx)), q_idx]
        for j, r in enumerate(cand[sl]):
            if not np.isfinite(d_self[j]):
                stats["outside_window"] += 1
                continue
            rd = float(q_min[j]) if Q is not None and np.isfinite(q_min[j]) else None
            qi = int(q_idx[j]) if rd is not None else None
            rows.append({
                "face_id": r["face_id"], "photo_id": r["photo_id"],
                "dist": round(float(d_self[j]), 4),
                "ref_year": Ry[int(s_idx[j])],
                "rival": qnames[qi] if qi is not None else None,
                "rival_dist": round(rd, 4) if rd is not None else None,
                "rival_year": Qy[qi] if qi is not None else None,
                "closer_to_other": rd is not None and rd < float(d_self[j]),
                "in_photo": r["photo_id"] in in_photo,
            })

    stats["measured"] = len(rows)
    if not rows:
        return [], stats
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

"""Bulk-remove one person's labels from a whole match_source.

Why this is a tool and not a SQL one-liner, in the order the problems bite:

1. **`person_id = NULL` is not enough.** A plain NULL is indistinguishable from
   never-matched, so the next `match-faces` sweep puts the same wrong label
   straight back — that is exactly what happened on 2026-09-14, when a routine
   sweep re-applied 164 Calvin temporal matches removed the day before. These
   writes use `faces.REJECTED_MATCH_SOURCE`, which both matchers skip.
2. **It has to be reversible by PINNED ID.** `restore-unmatched-faces` restores
   everything ever unmatched, including cleanups you meant to keep, so it is the
   wrong undo for a targeted sweep. `--snapshot` writes the exact
   (face_id, person_id, match_source) set, and `restore-unmatch` re-applies only
   those — and only to faces still unlabelled, so a later hand-label wins.
3. **Not every match in a bad source is wrong.** `temporal` measured ~4%
   accurate on the soccer shoots, not 0%. `--min-dist` keeps the faces that are
   genuinely close to the person's hand-made references, so a blanket clear does
   not throw away the minority that were right.

Distances are measured against **`manual` references only**, leave-one-burst-out
— the same rule and the same reasoning as `face_verify` (a same-burst frame of
the same mistake is a perfect alibi for it, and `strict`/`temporal` are the
populations under suspicion, so neither may define the baseline).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from .faces import REJECTED_MATCH_SOURCE

REFERENCE_SOURCE = "manual"
BURST_WINDOW_SECONDS = 3.0
CHUNK = 512


def _epoch(value):
    try:
        return datetime.fromisoformat(str(value)[:19]).timestamp()
    except (TypeError, ValueError):
        return np.nan


def _unit(mat):
    X = np.asarray(mat, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def select(db, *, person, sources=("temporal",), date_from=None, date_to=None,
           min_dist=None, burst_window=BURST_WINDOW_SECONDS):
    """Faces of `person` in `sources`, with their distance to the manual set.

    Returns (rows, stats). `min_dist` filters the rows; the stats describe the
    whole candidate set either way, so a dry run can show what a gate would
    keep as well as what it would clear.
    """
    where = ["pe.name = ?"]
    params = [person]
    if sources:
        where.append(f"f.match_source IN ({','.join('?' * len(sources))})")
        params += list(sources)
    if date_from:
        where.append("date(p.date_taken) >= ?"); params.append(date_from)
    if date_to:
        where.append("date(p.date_taken) <= ?"); params.append(date_to)

    cand = db.conn.execute(
        f"""SELECT f.id AS face_id, f.person_id, f.match_source, f.photo_id,
                   p.date_taken
              FROM faces f
              JOIN persons pe ON pe.id = f.person_id
              JOIN photos p ON p.id = f.photo_id
             WHERE {' AND '.join(where)}
             ORDER BY f.id""", params).fetchall()
    refs = db.conn.execute(
        """SELECT f.id AS face_id, p.date_taken
             FROM faces f
             JOIN persons pe ON pe.id = f.person_id
             JOIN photos p ON p.id = f.photo_id
            WHERE pe.name = ? AND f.match_source = ?""",
        (person, REFERENCE_SOURCE)).fetchall()

    stats = {"candidates": len(cand), "references": len(refs), "measured": 0}
    if not cand:
        return [], stats

    encs = db.get_face_encodings_bulk(
        [r["face_id"] for r in cand] + [r["face_id"] for r in refs])
    cand = [r for r in cand if r["face_id"] in encs]
    refs = [r for r in refs if r["face_id"] in encs]
    stats["references"] = len(refs)

    dist = {}
    if refs:
        R = _unit([encs[r["face_id"]] for r in refs])
        Rt = np.array([_epoch(r["date_taken"]) for r in refs])
        Rid = np.array([r["face_id"] for r in refs])
        X = _unit([encs[r["face_id"]] for r in cand])
        T = np.array([_epoch(r["date_taken"]) for r in cand])
        I = np.array([r["face_id"] for r in cand])
        for i in range(0, len(cand), CHUNK):
            sl = slice(i, i + CHUNK)
            D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ R.T)))
            # Never let a face vouch for itself, nor a frame of the same burst.
            same = I[sl][:, None] == Rid[None, :]
            burst = np.nan_to_num(
                np.abs(T[sl][:, None] - Rt[None, :]) <= burst_window, nan=False)
            D[same | burst] = np.inf
            for j, m in enumerate(D.min(axis=1)):
                dist[int(I[sl][j])] = None if not np.isfinite(m) else float(m)

    rows = [{"face_id": r["face_id"], "person_id": r["person_id"],
             "match_source": r["match_source"], "photo_id": r["photo_id"],
             "date_taken": r["date_taken"], "dist": dist.get(r["face_id"])}
            for r in cand]
    measured = [r["dist"] for r in rows if r["dist"] is not None]
    stats["measured"] = len(measured)
    if measured:
        a = np.array(measured)
        stats["percentiles"] = {p: round(float(np.percentile(a, p)), 3)
                                for p in (10, 50, 90)}
    if min_dist is not None:
        # A face with no usable reference is NOT cleared under a gate: the gate
        # is a claim about distance, and "unknown" is not "far".
        rows = [r for r in rows if r["dist"] is not None and r["dist"] > min_dist]
    stats["selected"] = len(rows)
    return rows, stats


def apply(db, rows):
    """Clear the labels, marked `rejected` so auto-matching cannot undo it."""
    ids = [r["face_id"] for r in rows]
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        db.conn.execute(
            f"UPDATE faces SET person_id = NULL, match_source = ? "
            f"WHERE id IN ({','.join('?' * len(batch))})",
            [REJECTED_MATCH_SOURCE, *batch])
    db.conn.commit()
    return len(ids)


def restore(db, rows):
    """Put back exactly the snapshotted labels — and only those.

    Skips any face that has since been labelled by hand: re-applying a
    `temporal` guess over a human correction would be a silent regression, and
    the snapshot is meant as an undo, not as a replay.
    """
    restored, skipped = 0, 0
    for r in rows:
        cur = db.conn.execute("SELECT person_id FROM faces WHERE id = ?",
                              (r["face_id"],)).fetchone()
        if cur is None:
            skipped += 1
            continue
        if cur["person_id"] is not None:
            skipped += 1
            continue
        db.conn.execute(
            "UPDATE faces SET person_id = ?, match_source = ? WHERE id = ?",
            (r["person_id"], r["match_source"], r["face_id"]))
        restored += 1
    db.conn.commit()
    return restored, skipped

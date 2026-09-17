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


# ---------------------------------------------------------------------------
# The high-level view: who else needs this?
# ---------------------------------------------------------------------------
# Splitting this in two is deliberate. `label_health` is pure SQL over counts,
# so the overview is instant and can be opened on a whim; `calibrate` costs a
# distance pass over one person's whole face set, so it is per-person and
# on demand. Doing both eagerly would make the overview a minute-long request
# that nobody opens twice.

MIN_CALIBRATION_REFS = 5
MIN_CALIBRATION_STRICT = 20
BAR_PERCENTILE = 90.0
TOP_DAYS = 6


def label_health(db, source="temporal"):
    """Per person: how much of `source` they carry, and where it is concentrated.

    Sorted by the count, because that is the review cost. Bundles the worst days
    so the answer to "where do I start" is in the same payload — a person's
    whole `source` population is usually not reviewable in one grid, but one of
    their shoots is.
    """
    rows = db.conn.execute(
        """SELECT pe.id AS person_id, pe.name AS person,
                  f.match_source AS src, COUNT(*) AS n
             FROM faces f JOIN persons pe ON pe.id = f.person_id
            GROUP BY pe.id, f.match_source""").fetchall()
    by_person: dict[int, dict] = {}
    for r in rows:
        p = by_person.setdefault(r["person_id"], {
            "person_id": r["person_id"], "person": r["person"],
            "total": 0, "by_source": {}})
        p["by_source"][r["src"] or "none"] = r["n"]
        p["total"] += r["n"]

    days = db.conn.execute(
        """SELECT f.person_id, date(p.date_taken) AS d, COUNT(*) AS n
             FROM faces f JOIN photos p ON p.id = f.photo_id
            WHERE f.match_source = ? AND p.date_taken IS NOT NULL
            GROUP BY f.person_id, d ORDER BY n DESC""", (source,)).fetchall()
    for r in days:
        p = by_person.get(r["person_id"])
        if p is None:
            continue
        p.setdefault("top_days", [])
        if len(p["top_days"]) < TOP_DAYS:
            p["top_days"].append({"date": r["d"], "count": r["n"]})

    out = []
    for p in by_person.values():
        n = p["by_source"].get(source, 0)
        manual = p["by_source"].get(REFERENCE_SOURCE, 0)
        strict = p["by_source"].get("strict", 0)
        out.append({**p, "source": source, "count": n,
                    "top_days": p.get("top_days", []),
                    # Why a person cannot be judged automatically is itself the
                    # finding: "label a few by hand first" is the fix, and
                    # silently omitting them would hide the work.
                    "calibratable": bool(n and manual >= MIN_CALIBRATION_REFS
                                         and strict >= MIN_CALIBRATION_STRICT),
                    "blocker": (None if not n else
                                "no faces in this source" if not n else
                                f"only {manual} manual reference(s) — label a few "
                                f"by hand first" if manual < MIN_CALIBRATION_REFS else
                                f"only {strict} strict face(s) to calibrate against"
                                if strict < MIN_CALIBRATION_STRICT else None)})
    out.sort(key=lambda p: -p["count"])
    return out


def calibrate(db, *, person, source="temporal", burst_window=BURST_WINDOW_SECONDS):
    """Is `source` a DIFFERENT population from this person's `strict` faces?

    The bar is the p90 of the person's OWN strict distances — not of their
    manual ones. Hand-made labels cluster in the sessions you happened to label,
    so manual-to-manual distance is artificially tight: using its p90 as a bar
    called **11,562 of Calvin's 11,669 strict faces suspect**, which the crops
    flatly contradict. `strict` is the largest verified-looking population a
    person has, so it is the honest yardstick for "how far does this person
    legitimately land from their references".
    """
    rows = db.conn.execute(
        """SELECT f.id, f.match_source, p.date_taken
             FROM faces f JOIN persons pe ON pe.id = f.person_id
             JOIN photos p ON p.id = f.photo_id
            WHERE pe.name = ?""", (person,)).fetchall()
    encs = db.get_face_encodings_bulk([r["id"] for r in rows])
    rows = [r for r in rows if r["id"] in encs]
    if not rows:
        return None

    src = np.array([r["match_source"] for r in rows])
    ref = src == REFERENCE_SOURCE
    if int(ref.sum()) < MIN_CALIBRATION_REFS:
        return {"person": person, "calibratable": False,
                "blocker": f"only {int(ref.sum())} manual reference(s)"}

    ids = np.array([r["id"] for r in rows])
    t = np.array([_epoch(r["date_taken"]) for r in rows])
    X = _unit([encs[i] for i in ids])
    R, Rt, Rid = X[ref], t[ref], ids[ref]
    d = np.empty(len(ids))
    for i in range(0, len(ids), CHUNK):
        sl = slice(i, i + CHUNK)
        D = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (X[sl] @ R.T)))
        D[(ids[sl][:, None] == Rid[None, :])
          | np.nan_to_num(np.abs(t[sl][:, None] - Rt[None, :]) <= burst_window,
                          nan=False)] = np.inf
        d[sl] = D.min(axis=1)

    ok = np.isfinite(d)
    sm, tm = (src == "strict") & ok, (src == source) & ok
    if int(sm.sum()) < MIN_CALIBRATION_STRICT or not tm.any():
        return {"person": person, "calibratable": False,
                "blocker": f"only {int(sm.sum())} strict face(s) to calibrate against"}
    bar = float(np.percentile(d[sm], BAR_PERCENTILE))
    return {"person": person, "source": source, "calibratable": True,
            "strict_p50": round(float(np.median(d[sm])), 3),
            "source_p50": round(float(np.median(d[tm])), 3),
            "bar": round(bar, 3),
            "count": int(tm.sum()),
            "beyond": int((d[tm] > bar).sum()),
            "pct_beyond": round(100.0 * float((d[tm] > bar).mean()), 1)}

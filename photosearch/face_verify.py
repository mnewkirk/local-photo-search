"""Is this labelled face actually that person?

Design + the case it comes from: `docs/plans/face-label-verification.md`.

The short version: for a face labelled P, compare it to P's own reference faces
and to every OTHER person present in the same scope, then calibrate against how
close P and that other person genuinely get. A face that sits closer to Q than
P and Q ever get to each other is almost certainly mislabelled.

This deliberately needs no eps and no distance threshold. Both of the existing
tools do, and on the one case with ground truth (two faces tagged Oliver Munoz
that were really Franklin Martinez, 2026-09-12) the thresholds are what made
them miss it: `verify-person-matches` compares against a global 1.30 while the
faces sat at 1.16, and `label-conflicts` reported zero at its default eps=0.80.
"""
from __future__ import annotations

import numpy as np

# Only these are trusted to define who a person looks like. `temporal` measured
# ~4% accurate on these shoots, so seeding the reference set with it would
# poison the very baseline the test depends on.
REFERENCE_SOURCES = ("manual", "strict")

# A single mislabelled face on either side drags a raw min() to near zero and
# silently disables the calibration for that pair — which would surface as
# "no conflicts found", the most dangerous possible failure. A low percentile
# is robust to a few bad labels while still meaning "how close do they get".
SEPARATION_PERCENTILE = 5.0

MIN_REFERENCES = 3

# A face must not be validated by a near-duplicate frame of ITSELF. Two
# mislabelled faces from the same burst alibi each other perfectly: on the
# 2026-09-12 ground-truth case, face 503089's nearest "Oliver" reference was
# 503092 — the other impostor, 1 second away — at 0.579, which collapsed the
# margin and hid the error completely. Excluding the burst gives 1.156 and the
# mislabel is decisive. Leave-one-FACE-out is not enough; it has to be
# leave-one-BURST-out.
BURST_WINDOW_SECONDS = 3.0


def _unit(rows_encs):
    X = np.stack(rows_encs).astype(np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _dists(A, B):
    """Pairwise L2 between unit-norm rows, via the cosine identity."""
    return np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (A @ B.T)))


def _ts(value):
    from datetime import datetime
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:19])
    except ValueError:
        return None


def verify_labels(db, *, date_from=None, date_to=None, person=None,
                  min_references=MIN_REFERENCES, limit=None,
                  burst_window=BURST_WINDOW_SECONDS):
    """Score every labelled face in scope. Returns (findings, skipped, stats).

    A finding carries both signals so a reviewer can tell a suggestion from a
    near-certainty:
      margin    = d_own - d_other   (>0: looks more like someone else)
      decisive  = d_other < separation(P, Q*)
    """
    where = ["f.person_id IS NOT NULL"]
    params: list = []
    if date_from:
        where.append("date(p.date_taken) >= ?"); params.append(date_from)
    if date_to:
        where.append("date(p.date_taken) <= ?"); params.append(date_to)
    if person:
        where.append("pe.name = ?"); params.append(person)
    if not (date_from or date_to or person):
        raise ValueError("a scope is required (date range or person)")

    rows = db.conn.execute(
        f"""SELECT f.id AS face_id, f.photo_id, f.match_source, p.date_taken,
                   pe.id AS person_id, pe.name AS person_name
              FROM faces f
              JOIN photos p ON p.id = f.photo_id
              JOIN persons pe ON pe.id = f.person_id
             WHERE {' AND '.join(where)}""", params).fetchall()
    if not rows:
        return [], [], {"faces": 0, "people": 0}

    encs = db.get_face_encodings_bulk([r["face_id"] for r in rows])
    rows = [r for r in rows if r["face_id"] in encs]

    by_person: dict[str, list] = {}
    for r in rows:
        by_person.setdefault(r["person_name"], []).append(r)

    # Reference sets: trusted faces only.
    refs = {}
    skipped = []
    for name, rs in by_person.items():
        trusted = [r for r in rs if r["match_source"] in REFERENCE_SOURCES]
        if len(trusted) < min_references:
            skipped.append({"person": name, "trusted": len(trusted),
                            "reason": "too few trusted references to calibrate"})
            continue
        refs[name] = {"ids": [r["face_id"] for r in trusted],
                      "ts": [_ts(r["date_taken"]) for r in trusted],
                      "X": _unit([encs[r["face_id"]] for r in trusted])}

    names = sorted(refs)
    # separation(P,Q): how close the two people genuinely get. Low percentile,
    # not min() — see SEPARATION_PERCENTILE.
    sep: dict[tuple, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = _dists(refs[a]["X"], refs[b]["X"]).ravel()
            v = float(np.percentile(d, SEPARATION_PERCENTILE))
            sep[(a, b)] = sep[(b, a)] = v

    findings = []
    for r in rows:
        name = r["person_name"]
        if name not in refs:
            continue
        x = _unit([encs[r["face_id"]]])
        own = refs[name]
        t = _ts(r["date_taken"])
        # LEAVE-ONE-BURST-OUT. Dropping only the face itself lets a second
        # mislabelled frame from the same burst vouch for it — see
        # BURST_WINDOW_SECONDS.
        mask = [i for i, fid in enumerate(own["ids"])
                if fid != r["face_id"]
                and not (t is not None and own["ts"][i] is not None
                         and abs((own["ts"][i] - t).total_seconds())
                             <= burst_window)]
        alibi_excluded = len(own["ids"]) - 1 - len(mask)
        if not mask:
            # Every reference is inside this face's own burst, so there is
            # nothing independent to compare against. Reported, not guessed at.
            skipped.append({"person": name, "trusted": len(own["ids"]),
                            "reason": f"all references fall inside face "
                                      f"{r['face_id']}'s own burst"})
            continue
        d_own = float(_dists(x, own["X"][mask]).min())

        best_other, d_other = None, float("inf")
        for other in names:
            if other == name:
                continue
            d = float(_dists(x, refs[other]["X"]).min())
            if d < d_other:
                best_other, d_other = other, d
        if best_other is None:
            continue

        s = sep.get((name, best_other))
        findings.append({
            "face_id": r["face_id"], "photo_id": r["photo_id"],
            "person": name, "match_source": r["match_source"],
            "candidate": best_other,
            "d_own": round(d_own, 3), "d_other": round(d_other, 3),
            "margin": round(d_own - d_other, 3),
            # How many same-burst references were withheld. A non-zero value
            # on a flagged face is itself a signal: the burst contained more
            # than one face wearing this label.
            "alibi_excluded": alibi_excluded,
            "separation": round(s, 3) if s is not None else None,
            # The decisive test: closer to Q than P and Q ever get to each other.
            "decisive": bool(s is not None and d_other < s),
        })

    findings = [f for f in findings if f["margin"] > 0]
    findings.sort(key=lambda f: (not f["decisive"], -f["margin"]))
    if limit:
        findings = findings[:limit]
    return findings, skipped, {"faces": len(rows), "people": len(refs),
                               "pairs": len(sep) // 2}

"""Is this labelled face actually that person?

Design + the case it comes from: `docs/plans/face-label-verification.md`.

The short version: for a face labelled P, compare it to P's own reference faces
and to every OTHER person present in the same scope, then calibrate against how
close those two people genuinely get. A face that sits closer to Q than P and Q
ever get to each other is almost certainly mislabelled.

This deliberately needs no eps and no distance threshold. Both of the existing
tools do, and on the one case with ground truth (two faces tagged Oliver Munoz
that were really Franklin Martinez, 2026-09-12) the thresholds are what made
them miss it: `verify-person-matches` compares against a global 1.30 while the
faces sat at 1.16, and `label-conflicts` reported zero at its default eps=0.80.

A SECOND, independent test lives here too -- see "The rival test" below.
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

# How many reference crops to hand the reviewer per identity.
REF_PREVIEW = 4

# A face must not be validated by a near-duplicate frame of ITSELF. Two
# mislabelled faces from the same burst alibi each other perfectly: on the
# 2026-09-12 ground-truth case, face 503089's nearest "Oliver" reference was
# 503092 — the other impostor, 1 second away — at 0.579, which collapsed the
# margin and hid the error completely. Excluding the burst gives 1.156 and the
# mislabel is decisive. Leave-one-FACE-out is not enough; it has to be
# leave-one-BURST-out.
BURST_WINDOW_SECONDS = 3.0

# --- The rival test -------------------------------------------------------
#
# A person appears in a photo at most once. So if ANOTHER face in the same
# photo is a better claimant to the label than the face carrying it, the label
# is on the wrong face — and the right face is sitting right there.
#
# This is independent of the comparison above, and catches what it cannot: that
# test needs the true identity to already be a registered person with
# references, whereas a rival may be completely unlabelled. The motivating case
# (photo 244260, 2026-09-12) is exactly that — "Franklin" was on the wrong boy
# and the real Franklin was untagged two faces to the right.
#
# `radius(P)` is the per-person analogue of `separation(P, Q)`: how far a
# GENUINE face of P lands from P's own references, under the same
# leave-one-burst-out rule. It is what lets one test serve a person whose faces
# vary a lot and one whose don't, with no global threshold — measured on the
# two soccer shoots, Franklin's radius is 0.95 and Calvin's 0.76.
RADIUS_PERCENTILE = 90.0


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


def _assign(cost):
    """One-to-one face->person assignment minimizing total cost.

    Returns {column index: row index} for the columns that were actually
    assigned. Costs must ALREADY be normalized (distance minus that person's
    radius) so that zero means "as close as a genuine face of them gets" — the
    `n` appended zero-cost dummy columns are then a uniform "assign this face to
    nobody" option. A raw-distance matrix would need a per-column outside price
    and would let a tight-radius person outbid a loose one for every face.
    """
    from scipy.optimize import linear_sum_assignment
    n, m = cost.shape
    aug = np.hstack([cost, np.zeros((n, n), dtype=cost.dtype)])
    rows, cols = linear_sum_assignment(aug)
    return {int(c): int(r) for r, c in zip(rows, cols) if c < m}


def verify_labels(db, *, date_from=None, date_to=None, person=None,
                  min_references=MIN_REFERENCES, limit=None,
                  burst_window=BURST_WINDOW_SECONDS, rivals=True):
    """Score every labelled face in scope. Returns (findings, skipped, stats).

    A finding carries both signals so a reviewer can tell a suggestion from a
    near-certainty:
      margin    = d_own - d_other   (>0: looks more like someone else)
      decisive  = d_other < separation(P, Q*)

    With `rivals` on, a finding may also carry a `rival` block: another face in
    the SAME photo that claims this label better. Those are returned even when
    `margin <= 0`, because the rival test does not need the true identity to be
    a registered person — which is the whole point of it.
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

    def _mask(face_id, t, name):
        """Indices of name's references that may judge this face."""
        own = refs[name]
        keep = []
        excluded = 0
        for i, rid in enumerate(own["ids"]):
            if rid == face_id:
                continue                      # never compare a face to itself
            rt = own["ts"][i]
            if t is not None and rt is not None and abs((rt - t).total_seconds()) <= burst_window:
                excluded += 1                 # same burst — cannot vouch for it
                continue
            keep.append(i)
        return keep, excluded

    def _d_to(face_id, enc, t, name):
        keep, _ = _mask(face_id, t, name)
        if not keep:
            return None
        return float(_dists(_unit([enc]), refs[name]["X"][keep]).min())

    # radius(P): how far a genuine P face lands from P's own refs. Same
    # leave-one-burst-out rule, or a burst of near-identical frames would make
    # every person look far tighter than they are.
    radius = {}
    for name in names:
        own = refs[name]
        ds = [d for i, fid in enumerate(own["ids"])
              if (d := _d_to(fid, encs[fid], own["ts"][i], name)) is not None]
        radius[name] = float(np.percentile(ds, RADIUS_PERCENTILE)) if ds else None

    photo_rivals = (_photo_rivals(db, rows, encs, refs, names, radius, _d_to)
                    if rivals and names else {})

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
        mask, alibi_excluded = _mask(r["face_id"], t, name)
        if not mask:
            # Every reference is inside this face's own burst, so there is
            # nothing independent to compare against. Reported, not guessed at.
            skipped.append({"person": name, "trusted": len(own["ids"]),
                            "reason": f"all references fall inside face "
                                      f"{r['face_id']}'s own burst"})
            continue
        own_d = _dists(x, own["X"][mask]).ravel()
        d_own = float(own_d.min())
        # Nearest references, for the reviewer. Showing the CLOSEST faces on
        # each side puts the strongest case for both identities side by side —
        # a random sample would make an obvious error look arguable.
        own_refs = [own["ids"][mask[i]] for i in np.argsort(own_d)[:REF_PREVIEW]]

        best_other, d_other, cand_refs = None, float("inf"), []
        for other in names:
            if other == name:
                continue
            d = _dists(x, refs[other]["X"]).ravel()
            if float(d.min()) < d_other:
                best_other, d_other = other, float(d.min())
                cand_refs = [refs[other]["ids"][i]
                             for i in np.argsort(d)[:REF_PREVIEW]]
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
            "own_refs": own_refs,
            "candidate_refs": cand_refs,
            # The decisive test: closer to Q than P and Q ever get to each other.
            "decisive": bool(s is not None and d_other < s),
            "rival": photo_rivals.get(r["face_id"]),
        })

    # A rival is reportable on its own: it does not need some OTHER registered
    # person to be closer, which is the one thing the margin test cannot do
    # without.
    findings = [f for f in findings if f["margin"] > 0 or f["rival"]]
    _attach_burst_siblings(findings, rows, burst_window)
    # Rivals first — they name the right face, so they are the only findings a
    # reviewer can fix in one click rather than adjudicate.
    findings.sort(key=lambda f: (f["rival"] is None, not f["decisive"], -f["margin"]))
    if limit:
        findings = findings[:limit]
    return findings, skipped, {"faces": len(rows), "people": len(refs),
                               "pairs": len(sep) // 2,
                               "rivals": sum(1 for f in findings if f["rival"])}


def _photo_rivals(db, labelled_rows, encs, refs, names, radius, d_to):
    """For each labelled face, the face in its photo that claims the label better.

    Returns {face_id: rival block}. Four gates, all required:
      1. the rival is closer to P than the incumbent is
      2. the rival is plausibly P at all       (d_rival < radius(P))
      3. the incumbent is not                  (d_self  > radius(P))
      4. the rival is not better explained by someone else

    Gate 1 comes out of a one-to-one assignment rather than a per-label
    argmin, so two labels in one photo cannot both claim the same rival.
    """
    photo_ids = sorted({r["photo_id"] for r in labelled_rows})
    if not photo_ids:
        return {}

    # Every face in those photos, INCLUDING the unlabelled ones — they are the
    # rivals that matter, and the reason this test sees what the others miss.
    all_faces = []
    CHUNK = 900                      # stay under SQLITE_MAX_VARIABLE_NUMBER
    for i in range(0, len(photo_ids), CHUNK):
        batch = photo_ids[i:i + CHUNK]
        all_faces += db.conn.execute(
            f"""SELECT f.id AS face_id, f.photo_id, f.person_id, f.match_source,
                       pe.name AS person_name, p.date_taken
                  FROM faces f
                  JOIN photos p ON p.id = f.photo_id
                  LEFT JOIN persons pe ON pe.id = f.person_id
                 WHERE f.photo_id IN ({','.join('?' * len(batch))})""",
            batch).fetchall()

    missing = [f["face_id"] for f in all_faces if f["face_id"] not in encs]
    if missing:
        encs = dict(encs)
        encs.update(db.get_face_encodings_bulk(missing))
    all_faces = [f for f in all_faces if f["face_id"] in encs]

    by_photo: dict[int, list] = {}
    for f in all_faces:
        by_photo.setdefault(f["photo_id"], []).append(f)

    out = {}
    for pid, faces in by_photo.items():
        if len(faces) < 2:
            continue
        # Columns are the people ALREADY labelled in this photo. The question is
        # only ever "is this label on the wrong face", never "who is that
        # stranger" — opening the columns to the whole roster would invent
        # labels for unlabelled faces, which is a different tool.
        present = sorted({f["person_name"] for f in faces
                          if f["person_name"] in refs and radius.get(f["person_name"]) is not None})
        if not present:
            continue
        cost = np.zeros((len(faces), len(present)), dtype=np.float64)
        raw = {}
        for i, f in enumerate(faces):
            t = _ts(f["date_taken"])
            for j, nm in enumerate(present):
                d = d_to(f["face_id"], encs[f["face_id"]], t, nm)
                # No usable reference (all inside this face's burst): price it
                # out of the assignment rather than guess.
                cost[i, j] = 1e6 if d is None else d - radius[nm]
                raw[(i, j)] = d
        won = _assign(cost)

        for j, nm in enumerate(present):
            holders = [i for i, f in enumerate(faces) if f["person_name"] == nm]
            winner = won.get(j)
            if winner is None or winner in holders:
                continue                      # label already on the best face
            rival = faces[winner]
            d_rival = raw[(winner, j)]
            if d_rival is None or d_rival >= radius[nm]:
                continue                      # gate 2
            # Gate 4: the rival must not be a better fit for someone else. A
            # rival that IS someone else's face would otherwise be stolen.
            r_t = _ts(rival["date_taken"])
            r_alt, d_r_alt = None, None
            for other in names:
                if other == nm:
                    continue
                d = d_to(rival["face_id"], encs[rival["face_id"]], r_t, other)
                if d is not None and (d_r_alt is None or d < d_r_alt):
                    r_alt, d_r_alt = other, d
            if d_r_alt is not None and d_r_alt <= d_rival:
                continue
            for i in holders:
                d_self = raw[(i, j)]
                if d_self is None or d_self <= radius[nm]:
                    continue                  # gate 3
                out[faces[i]["face_id"]] = {
                    "rival_face_id": rival["face_id"],
                    "rival_label": rival["person_name"],
                    "rival_person_id": rival["person_id"],
                    "d_self": round(d_self, 3),
                    "d_rival": round(d_rival, 3),
                    "displaced": round(d_self - d_rival, 3),
                    "radius": round(radius[nm], 3),
                    "rival_best_other": r_alt,
                    "d_rival_best_other": round(d_r_alt, 3) if d_r_alt is not None else None,
                    "burst_siblings": [],
                }
    return out


def _attach_burst_siblings(findings, rows, burst_window):
    """Link rival findings that are the same mistake in consecutive frames.

    These mislabels arrive in pairs: photo 244260 and 244261 are one second
    apart and carry the identical wrong "Franklin". Each sibling is an
    INDEPENDENTLY gated finding, not a blind copy of this one — so applying the
    set is as safe as applying them one at a time, just faster.
    """
    when = {r["face_id"]: _ts(r["date_taken"]) for r in rows}
    withrival = [f for f in findings if f["rival"]]
    for f in withrival:
        t = when.get(f["face_id"])
        for g in withrival:
            if g is f or g["person"] != f["person"]:
                continue
            u = when.get(g["face_id"])
            if t is None or u is None or abs((u - t).total_seconds()) > burst_window:
                continue
            f["rival"]["burst_siblings"].append({
                "face_id": g["face_id"], "photo_id": g["photo_id"],
                "rival_face_id": g["rival"]["rival_face_id"],
                "displaced": g["rival"]["displaced"],
            })

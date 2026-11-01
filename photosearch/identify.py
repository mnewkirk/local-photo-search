"""Who is this body? Tiered identity for person (body) boxes, within one session.

Pure functions: no DB, no model, no I/O (docs/plans/people-context-describe.md, Step 2).
The caller hands in bodies with their evidence and gets back, per body, a person id
or None plus the tier and witnesses that decided it. Names are never decided here;
a render step looks them up at read time.

Tiers, strongest first, each only if the stronger ones said nothing:

1. ``face``  - the body owns a trusted face (keypoint-verified link, uncontested).
2. ``burst`` - a frame within ``BURST_S`` seconds holds a body in the same place
   (geometry) whose own face is trusted.
3. ``kit``   - the body's jersey number + kit colour matches a session map seeded
   by trusted-face anchors from >= ``MIN_ANCHOR_BURSTS`` independent bursts, with
   no contradicting anchor; a signature seen twice in one photo voids itself.
4. attributes (hair, shoes) only break a tie between candidates a tier offered;
   they never name a body on their own.

Constraints shared by every tier: one person at most once per photo (a person
claimed by a stronger tier elsewhere in the photo is unavailable), and owner body
labels (``labels``) act as trusted faces for seeding and win outright.
"""
from __future__ import annotations

import datetime as _dt
from collections import defaultdict
from typing import Iterable, Optional

SESSION_GAP_MIN = 45          # Step 0: 12-119 min all give the same 10 soccer sessions
BURST_S = 3.0                 # = face_verify.BURST_WINDOW_SECONDS
BURST_IOU = 0.7               # same place in a sibling frame once the camera motion is undone (G0 tune)
BURST_IOU_MARGIN = 0.2        # the best sibling body must beat the runner-up by this much
MOTION_TOL = 0.04             # body-pair transforms tried within this share of the width of the hint
MOTION_MIN_PEAK = 0.1         # phase-correlation peak below this: no measured motion (G0 tune)
TRACK_S = BURST_S             # burst tier: sibling frames at most this far apart
MIN_TRACK_SIBLINGS = 1        # sibling frames that must agree on the person
MIN_ANCHOR_BURSTS = 2         # independent trusted-face bursts behind a kit signature
MAX_CONTRA_ANCHORS = 0        # anchors of the same signature naming someone else


def parse_time(s: Optional[str]) -> Optional[_dt.datetime]:
    if not s:
        return None
    for f in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M:%S"):
        try:
            return _dt.datetime.strptime(s[:19], f)
        except ValueError:
            pass
    return None


def sessions(rows: Iterable[tuple], gap_min: float = SESSION_GAP_MIN) -> dict:
    """``rows`` of ``(photo_id, folder, date_taken)`` -> ``{photo_id: session key}``.
    Session = folder + calendar day, split where consecutive shots are more than
    ``gap_min`` minutes apart. Undated photos get no session (never identified by
    a session tier)."""
    by = defaultdict(list)
    for pid, folder, d in rows:
        t = parse_time(d)
        if t is not None:
            by[(folder, t.date())].append((t, pid))
    out = {}
    for (folder, day), lst in sorted(by.items()):
        lst.sort()
        k, prev = 0, None
        for t, pid in lst:
            if prev is not None and (t - prev).total_seconds() > gap_min * 60:
                k += 1
            out[pid] = f"{folder}|{day.isoformat()}|{k}"
            prev = t
    return out


def bursts(times: dict, window_s: float = BURST_S) -> dict:
    """``{photo_id: datetime}`` -> ``{photo_id: burst index}``: shots chained by
    gaps <= ``window_s``. Two anchors from one burst are one piece of evidence."""
    out, k, prev = {}, -1, None
    for pid, t in sorted(times.items(), key=lambda kv: (kv[1], kv[0])):
        if prev is None or (t - prev).total_seconds() > window_s:
            k += 1
        out[pid] = k
        prev = t
    return out


def _iou(a, b) -> float:
    t, r, bo, l = max(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])
    inter = max(0, r - l) * max(0, bo - t)
    ua = (a[1] - a[3]) * (a[2] - a[0]) + (b[1] - b[3]) * (b[2] - b[0]) - inter
    return inter / ua if ua > 0 else 0.0


def resolve_photo(bodies: list[dict], claimed: Optional[set] = None) -> dict:
    """Per-photo assignment: ``bodies`` carry ``candidates`` = ``[(tier_rank, person,
    witness)]`` (lower rank = stronger). Strongest first; a person goes to one body
    only; two bodies offering the same person at the same rank both lose it.
    Returns ``{body index: (person, tier, witness)}``."""
    claimed = set(claimed or ())
    out = {}
    flat = sorted(((c[0], i, c[1], c[2]) for i, b in enumerate(bodies)
                   for c in (b.get("candidates") or [])), key=lambda x: (x[0], x[1]))
    by_rank = defaultdict(list)
    for rank, i, p, w in flat:
        by_rank[rank].append((i, p, w))
    for rank in sorted(by_rank):
        offers = [(i, p, w) for i, p, w in by_rank[rank] if i not in out and p not in claimed]
        cnt = defaultdict(set)
        for i, p, w in offers:
            cnt[p].add(i)
        for i, p, w in offers:
            if len(cnt[p]) == 1 and i not in out:
                out[i] = (p, w["tier"], w)
                claimed.add(p)
    return out


TIER_RANK = {"label": 0, "face": 1, "burst": 2, "kit": 3}

# Kit colours the reader confuses on one kit (the neon team shirt reads yellow,
# neon-yellow or green): compared as one group. Tuned on the tune shoots only.
KIT_GROUPS = {"neon-yellow": "yellow", "light-blue": "blue"}


def signature(attrs: Optional[dict], groups: Optional[dict] = None) -> Optional[tuple]:
    """(kit group, jersey number) when a number was read, else None."""
    if not attrs:
        return None
    num, kit = attrs.get("number"), attrs.get("kit")
    if not num or not str(num).isdigit() or not kit or kit == "not-visible":
        return None
    g = KIT_GROUPS if groups is None else groups
    return (g.get(kit, kit), str(num))


def _apply(t, box):
    sc, tx, ty = t
    top, r, b, l = box
    return (top * sc + ty, r * sc + tx, b * sc + ty, l * sc + tx)


def _inliers(t, a_bodies, b_bodies, min_iou=None):
    """One-to-one greedy matches {a index: b index} of a's boxes moved by t onto b's."""
    min_iou = BURST_IOU if min_iou is None else min_iou
    pairs = sorted(((_iou(_apply(t, a["bbox"]), b["bbox"]), i, j)
                    for i, a in enumerate(a_bodies) for j, b in enumerate(b_bodies)), reverse=True)
    used_a, used_b, m = set(), set(), {}
    for v, i, j in pairs:
        if v < min_iou:
            break
        if i not in used_a and j not in used_b:
            m[i] = j
            used_a.add(i)
            used_b.add(j)
    return m


def frame_transform(a_bodies, b_bodies, hint=None, tol=0.0, min_iou=None):
    """Camera motion between two burst frames as (scale, x shift, y shift), and the
    one-to-one body matches it gives. ``hint`` is the pixel-measured camera shift
    (:func:`phase_shift`); only transforms within ``tol`` px of it are tried (the hint
    itself, and each body pair's scale + shift near it), and the one lining up the
    most bodies wins. Rows of similar-sized players make the body boxes alone
    ambiguous (242551 -> 242552: a pan matched the coach onto a boy), so without a
    hint there is no transform, except one body on each side (identity only).
    ``(None, {})`` when fewer than half the smaller frame's bodies (and < 2) line up."""
    if not a_bodies or not b_bodies:
        return None, {}
    if hint is None:
        if len(a_bodies) == 1 and len(b_bodies) == 1:
            t = (1.0, 0.0, 0.0)
            return t, _inliers(t, a_bodies, b_bodies, min_iou)
        return None, {}
    hyps = [hint]
    for a in a_bodies:
        at, ar, ab, al = a["bbox"]
        for b in b_bodies:
            bt, br, bb, bl = b["bbox"]
            if ab - at <= 0:
                continue
            sc = (bb - bt) / float(ab - at)
            if not 0.8 <= sc <= 1.25:
                continue
            h = (sc, (bl + br) / 2.0 - sc * (al + ar) / 2.0, bt - sc * at)
            if abs(h[1] - hint[1]) <= tol and abs(h[2] - hint[2]) <= tol:
                hyps.append(h)
    best, best_m = None, {}
    for t in hyps:
        m = _inliers(t, a_bodies, b_bodies, min_iou)
        if len(m) > len(best_m):
            best, best_m = t, m
    need = 1 if min(len(a_bodies), len(b_bodies)) == 1 else max(2, (min(len(a_bodies), len(b_bodies)) + 1) // 2)
    if len(best_m) < need:
        return None, {}
    return best, best_m


def phase_shift(gray_a, gray_b, scale: float = 1.0):
    """Camera shift from frame a to frame b by phase correlation of two equal-width
    grayscale arrays (numpy). Returns ``(1.0, dx, dy, peak)`` in full-resolution px
    (``scale`` = full px per array px); a low ``peak`` means an unreliable answer."""
    import numpy as np
    h = min(gray_a.shape[0], gray_b.shape[0])
    a, b = gray_a[:h], gray_b[:h]
    win = np.outer(np.hanning(h), np.hanning(a.shape[1]))
    fa = np.fft.fft2((a - a.mean()) * win)
    fb = np.fft.fft2((b - b.mean()) * win)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-9
    corr = np.real(np.fft.ifft2(r))
    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[y, x])
    if y > h // 2:
        y -= h
    if x > a.shape[1] // 2:
        x -= a.shape[1]
    return 1.0, -x * scale, -y * scale, peak


def _burst_offers(p, sibs, i, b, known, cache=None, motion=None):
    """Burst-tier offers for body i of photo p: in each sibling frame, the body that
    body i lands on once the camera's motion between the frames is undone
    (:func:`frame_transform`), if that body's identity is known. Siblings naming
    different people cancel each other."""
    offers = []
    for s in sibs:
        key = (p["photo_id"], s["photo_id"])
        if cache is not None and key in cache:
            t, m = cache[key]
        else:
            hint = (motion or {}).get(key)
            t, m = frame_transform(p["bodies"], s["bodies"], hint=hint,
                                   tol=MOTION_TOL * (p.get("W") or 0))
            if cache is not None:
                cache[key] = (t, m)
        if i not in m:
            continue
        k = m[i]
        # The landing body must be clearly the one: a runner-up close behind (two
        # overlapping players) leaves the track ambiguous.
        moved = _apply(t, b["bbox"])
        rivals = sorted((_iou(moved, sb["bbox"]) for j, sb in enumerate(s["bodies"]) if j != k), reverse=True)
        if rivals and _iou(moved, s["bodies"][k]["bbox"]) - rivals[0] < BURST_IOU_MARGIN:
            continue
        per, _ = known(s["photo_id"], k, s["bodies"][k])
        if per is not None:
            offers.append((per, {"tier": "burst", "photo_id": s["photo_id"], "body": k,
                                 "iou": round(_iou(_apply(t, b["bbox"]), s["bodies"][k]["bbox"]), 3)}))
    if len({o[0] for o in offers}) != 1 or len(offers) < MIN_TRACK_SIBLINGS:
        return []
    return offers[:1]


def identify_session(photos: list[dict], enabled=("face", "burst", "kit"),
                     hide_faces: Optional[set] = None, kit_groups: Optional[dict] = None,
                     min_bursts: int = MIN_ANCHOR_BURSTS, max_contra: int = MAX_CONTRA_ANCHORS,
                     motion: Optional[dict] = None) -> dict:
    """Identify every body in one session.

    ``photos``: ``[{"photo_id", "time": datetime, "bodies": [{"bbox", "face_person"
    (trusted, uncontested face link or None), "label_person" (owner label or None),
    "attrs" (dict or None)}]}]``. ``hide_faces``: ``{(photo_id, body index)}`` whose
    face is ignored everywhere (leave-face-out evaluation). ``motion``:
    ``{(photo a, photo b): (scale, dx, dy)}`` camera motion between burst siblings
    (:func:`phase_shift`); a pair without it gets no burst track unless each frame
    holds one body. Photos may carry ``"W"`` (frame width) for the motion tolerance.

    Kit anchors are bodies with a read number whose identity is a label, a face or
    a burst track; a body's kit match counts only anchors from OTHER bursts (its own
    burst is the burst tier's evidence, not independent support).

    Returns ``{(photo_id, body index): {"person", "tier", "witness"}}`` for every
    body a tier named, plus ``{"contested": [...]}`` under the key ``None``.
    """
    hide = hide_faces or set()
    order = sorted((p for p in photos if p.get("time")), key=lambda p: p["time"])
    burst_of = bursts({p["photo_id"]: p["time"] for p in order})

    def known(pid, i, b):
        if b.get("label_person") is not None:
            return b["label_person"], "label"
        if "face" in enabled and (pid, i) not in hide and b.get("face_person") is not None:
            return b["face_person"], "face"
        return None, None

    def siblings(idx):
        p, out = order[idx], []
        for step in (-1, 1):
            j = idx + step
            while 0 <= j < len(order) and abs((order[j]["time"] - p["time"]).total_seconds()) <= TRACK_S:
                out.append(order[j])
                j += step
        return out

    # Stage 1: label / face / burst.
    cands, tcache = {}, {}
    for idx, p in enumerate(order):
        sibs = siblings(idx) if "burst" in enabled else []
        for i, b in enumerate(p["bodies"]):
            c = []
            per, how = known(p["photo_id"], i, b)
            if how:
                c.append((TIER_RANK[how], per, {"tier": how}))
            for per2, w in _burst_offers(p, sibs, i, b, known, tcache, motion):
                c.append((TIER_RANK["burst"], per2, w))
            cands[(p["photo_id"], i)] = c

    def stage1(key):
        c = cands[key]
        if not c:
            return None
        top = min(x[0] for x in c)
        if len({x[1] for x in c}) > 1 and top != TIER_RANK["label"]:
            return None
        return next(x for x in c if x[0] == top)[1]

    # Stage 2: kit map from anchors, leave-own-burst-out at query time.
    sigs = {}
    for p in order:
        for i, b in enumerate(p["bodies"]):
            sg = signature(b.get("attrs"), kit_groups)
            if sg:
                sigs[(p["photo_id"], i)] = sg
    void = set()
    for p in order:
        seen = set()
        for i in range(len(p["bodies"])):
            sg = sigs.get((p["photo_id"], i))
            if sg in seen:
                void.add(sg)
            if sg:
                seen.add(sg)
    sup = defaultdict(lambda: defaultdict(set))      # sig -> person -> bursts
    if "kit" in enabled:
        for key, sg in sigs.items():
            per = stage1(key)
            if per is not None and sg not in void:
                sup[sg][per].add(burst_of[key[0]])
        for key, sg in sigs.items():
            if sg in void or sg not in sup:
                continue
            own = burst_of[key[0]]
            per_b = {per: bs - {own} for per, bs in sup[sg].items()}
            per_b = {per: bs for per, bs in per_b.items() if bs}
            if not per_b:
                continue
            best = max(per_b, key=lambda q: (len(per_b[q]), -q))
            contra = sum(len(bs) for q, bs in per_b.items() if q != best)
            if len(per_b[best]) >= min_bursts and contra <= max_contra:
                cands[key].append((TIER_RANK["kit"], best,
                                   {"tier": "kit", "sig": list(sg), "bursts": len(per_b[best])}))

    out, contested = {}, []
    for p in order:
        pb = []
        for i in range(len(p["bodies"])):
            c = cands[(p["photo_id"], i)]
            ppl = {x[1] for x in c}
            if len(ppl) > 1:
                if min(x[0] for x in c) == TIER_RANK["label"]:
                    c = [x for x in c if x[0] == TIER_RANK["label"]]
                else:
                    contested.append((p["photo_id"], i))
                    c = []
            pb.append({"candidates": c})
        for i, (per, tier, w) in resolve_photo(pb).items():
            out[(p["photo_id"], i)] = {"person": per, "tier": tier, "witness": w}
    out[None] = {"contested": contested}
    return out

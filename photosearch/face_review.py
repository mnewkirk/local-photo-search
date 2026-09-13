"""Team-jersey face review: narrow a day's unknown faces to one team, group
them, and hand back something a human can actually get through.

The problem this solves: after a match shoot, `/faces` shows thousands of
unknown faces for a day — your team, both opposing teams, coaches, parents on
the sideline. Reviewing them one crop at a time is hopeless, and a global
`recluster-faces` sends ~73% of faces to DBSCAN noise, so most never get a
group to review at all.

Two ideas do the work:

1. **Jersey colour as a team filter.** Sample a band below each face box — the
   torso — and take its modal hue. On a real shoot this is sharply bimodal:
   measured on 2026-08-29, the named players sat at hue 85-105 deg and the
   opposing team at 210-240 deg, with no overlap.

2. **Learn the colour, never hardcode it.** The reference comes from faces you
   have already named on that date. That self-calibrates to the day's light and
   white balance, and it survives the team changing kit between games — which
   is not hypothetical here, the same squad played in fluorescent yellow in
   August and blue in September.

Learning it also caught a bug a hardcoded range would have hidden: the first
reference set included three adult coaches in blue shirts, which dragged the
"team" centroid 120 deg off. Hence `learn_team_hue` uses the densest hue bin
rather than a mean, so a minority in different clothing cannot move it.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import logging
import math
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Torso band, in multiples of the face box. Starts below the neck and covers
# the chest; narrower than the face box so shoulders/background stay out.
BAND_TOP = 0.35
BAND_HEIGHT = 1.30
BAND_WIDTH = 0.55

MIN_SATURATED_FRAC = 0.35   # below this the band is shadow/sky/grass, not kit
COHERENCE_WINDOW = 15.0     # deg; pixels this close to the mode count as "the kit"
MIN_COHERENCE = 0.60        # fraction of saturated pixels that must agree
MIN_SAT = 90                # 0-255; below this hue is meaningless


def _hue_delta(a: float, b: float) -> float:
    """Circular distance between two hues in degrees (0-180)."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def torso_sample(im, face: dict, scale: float) -> Optional[dict]:
    """Modal hue of the torso band below one face box.

    `scale` maps original-image coords (which the bboxes use) onto whatever
    resolution `im` actually is — previews are fine and much cheaper, the
    measurement is a colour statistic, not a detail one.

    Returns None if the band falls outside the frame, and marks `coherent`
    False when the band isn't a solid block of colour (player bent over, an arm
    across the chest, the band running onto grass).
    """
    import numpy as np

    W, H = im.size
    fw = (face["bbox_right"] - face["bbox_left"]) * scale
    fh = (face["bbox_bottom"] - face["bbox_top"]) * scale
    cx = (face["bbox_left"] + face["bbox_right"]) / 2 * scale
    y0 = face["bbox_bottom"] * scale + fh * BAND_TOP
    box = (max(0, int(cx - fw * BAND_WIDTH)), max(0, int(y0)),
           min(W, int(cx + fw * BAND_WIDTH)), min(H, int(y0 + fh * BAND_HEIGHT)))
    if box[2] - box[0] < 6 or box[3] - box[1] < 6:
        return None

    px = np.array(im.crop(box).convert("HSV")).reshape(-1, 3).astype(float)
    if len(px) < 40:
        return None
    hue = px[:, 0] * 360.0 / 255.0
    sat_mask = px[:, 1] >= MIN_SAT
    sat_frac = float(sat_mask.mean())
    if sat_frac < MIN_SATURATED_FRAC:
        return {"hue": None, "coherence": 0.0, "sat_frac": sat_frac, "coherent": False}

    h = hue[sat_mask]
    hist, edges = np.histogram(h, bins=36, range=(0, 360))
    i = int(hist.argmax())
    mode = float((edges[i] + edges[i + 1]) / 2)
    coherence = float((np.abs(h - mode) <= COHERENCE_WINDOW).mean())
    return {"hue": mode, "coherence": coherence, "sat_frac": sat_frac,
            "sat": float(np.median(px[sat_mask, 1])),
            "coherent": coherence >= MIN_COHERENCE}


def sample_faces(faces: list[dict], fetch_preview: Callable[[int], bytes],
                 workers: int = 6,
                 on_progress: Optional[Callable[[int, int], None]] = None) -> dict[int, dict]:
    """Torso-sample every face, one image fetch per photo.

    `fetch_preview(photo_id) -> bytes` is injected so this works against the NAS
    web API, a local file, or a test stub.
    """
    from PIL import Image, ImageOps

    by_photo: dict[int, list[dict]] = {}
    for f in faces:
        by_photo.setdefault(f["photo_id"], []).append(f)

    out: dict[int, dict] = {}
    done = 0

    def work(pid: int):
        try:
            raw = fetch_preview(pid)
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        except Exception as e:                      # unreadable / missing → skip
            logger.debug("preview failed for photo %s: %s", pid, e)
            return pid, {}
        group = by_photo[pid]
        src_w = group[0].get("image_width") or im.size[0]
        scale = im.size[0] / src_w if src_w else 1.0
        return pid, {f["id"]: torso_sample(im, f, scale) for f in group}

    with cf.ThreadPoolExecutor(workers) as ex:
        for pid, got in ex.map(work, list(by_photo)):
            for fid, s in got.items():
                if s is not None:
                    out[fid] = s
            done += 1
            if on_progress:
                on_progress(done, len(by_photo))
    return out


def learn_team_hue(samples: dict[int, dict], reference_face_ids: set[int]) -> Optional[float]:
    """Team hue = densest 30-degree bin among the reference faces, refined to the
    median within that bin.

    Deliberately modal, not an average: on a real shoot the named faces include
    coaches and parents in other colours, and a mean lands between the two
    groups — a value no one is actually wearing.
    """
    import numpy as np

    hues = [s["hue"] for fid, s in samples.items()
            if fid in reference_face_ids and s.get("coherent") and s.get("hue") is not None]
    if len(hues) < 3:
        return None
    h = np.array(hues)
    hist, edges = np.histogram(h, bins=12, range=(0, 360))
    i = int(hist.argmax())
    lo, hi = edges[i], edges[i + 1]
    in_bin = h[(h >= lo) & (h < hi)]
    return float(np.median(in_bin)) if len(in_bin) else None


def classify(samples: dict[int, dict], team_hue: float, tolerance: float,
             always_include: set[int] | None = None) -> dict[int, str]:
    """face_id -> 'team' | 'other' | 'unknown'.

    'unknown' means the torso band wasn't readable — surfaced separately rather
    than silently dropped, because a guess here quietly loses real faces.
    `always_include` (named people) is forced to 'team' so a coach in different
    clothing never disappears from review.
    """
    always_include = always_include or set()
    out: dict[int, str] = {}
    for fid, s in samples.items():
        if fid in always_include:
            out[fid] = "team"
        elif not s.get("coherent") or s.get("hue") is None:
            out[fid] = "unknown"
        else:
            out[fid] = "team" if _hue_delta(s["hue"], team_hue) <= tolerance else "other"
    return out


def cluster_faces(encodings: dict[int, list[float]], eps: float = 0.80,
                  min_samples: int = 2) -> dict[int, int]:
    """DBSCAN over just this day's team faces. face_id -> cluster index (-1 noise).

    eps=0.80 is LOOSER than the global recluster's 0.55, deliberately and by
    measurement. Narrowing to one team on one afternoon makes the embedding
    space SPARSE, so a tight radius groups nothing — the opposite of the
    intuition that a narrower problem wants a stricter threshold.

    Swept against the 220 already-named faces on the 2026-08-29 shoot as ground
    truth, after correcting a bad label the sweep itself uncovered:

        eps   clusters  noise%  biggest  faces that would be mislabelled
        0.45      34     92.1%     ~6     0
        0.55      93     75.1%     10     0
        0.75     104     36.3%     51     0
        0.80      59     29.3%     66     0
        0.90      39     13.4%     85     0

    At 0.45 (the first guess) 92% fell out as noise and you were still
    reviewing individual crops, which defeats the point. No cluster holds more
    than one named person anywhere in that range, so the usual eps/purity
    tradeoff simply doesn't bite on a single-team, single-day set.

    0.80 rather than 0.90 because ground truth covers only ~25% of the faces:
    zero impurity among the named ones does not prove it for the unnamed, and
    0.90's 85-face clusters exceed the ~58 faces/player this squad averages,
    which smells like merging. 0.80 keeps the biggest cluster under that mark.
    A merge is also recoverable here — the review gallery has per-face
    checkboxes, so an odd face in a group gets unticked, not misnamed.

    Measure impurity as FACES THAT WOULD BE MISLABELLED, never as impure-cluster
    count: one mega-cluster holding everyone scores 1 on the latter and looks
    fine.
    """
    import numpy as np
    from sklearn.cluster import DBSCAN

    fids = sorted(encodings)
    if len(fids) < min_samples:
        return {f: -1 for f in fids}
    X = np.array([encodings[f] for f in fids], dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(X)
    return {f: int(l) for f, l in zip(fids, labels)}


def summarize(classes: dict[int, str], clusters: dict[int, int]) -> dict:
    groups: dict[int, int] = {}
    for fid, c in clusters.items():
        groups[c] = groups.get(c, 0) + 1
    return {
        "team": sum(1 for v in classes.values() if v == "team"),
        "other": sum(1 for v in classes.values() if v == "other"),
        "unknown": sum(1 for v in classes.values() if v == "unknown"),
        "clusters": sum(1 for c in groups if c >= 0),
        "noise": groups.get(-1, 0),
    }


def find_label_conflicts(clusters: dict[int, int], labels: dict[int, str],
                         min_named: int = 2) -> list[dict]:
    """Clusters holding two or more DIFFERENT named people — i.e. suspected
    label errors, ranked most-lopsided first.

    The premise, established on the 2026-08-29 shoot: within one team on one
    day a cluster that mixes two named people is far more likely to be a bad
    LABEL than a bad cluster. Two faces tagged Carson Jones sat inside eleven
    Beckham Tinnel faces; the tags were wrong and the cluster was right. That
    was the only impurity in a full eps sweep, and correcting it took the whole
    sweep to zero.

    Ranked by `lopsidedness` = majority / named-in-cluster. An 11-vs-2 split
    (0.85) is a near-certain mislabel; a 6-vs-5 (0.55) is more likely a genuine
    merge of two people and is worth looking at last.

    Deliberately returns the FULL label breakdown rather than a
    minority→majority suggestion. Auto-applying the majority name would have
    been wrong on the very case that motivated this: both labels on those
    frames had been wrong before, and the correct answer was a third person in
    neither group.
    """
    by_cluster: dict[int, list[int]] = {}
    for fid, cid in clusters.items():
        if cid >= 0:
            by_cluster.setdefault(cid, []).append(fid)

    out: list[dict] = []
    for cid, members in by_cluster.items():
        named = [(f, labels[f]) for f in members if labels.get(f)]
        if len(named) < min_named:
            continue
        counts: dict[str, list[int]] = {}
        for f, n in named:
            counts.setdefault(n, []).append(f)
        if len(counts) < 2:
            continue
        ordered = sorted(counts.items(), key=lambda kv: -len(kv[1]))
        majority = len(ordered[0][1])
        out.append({
            "cluster": cid,
            "size": len(members),
            "named_total": len(named),
            "lopsidedness": round(majority / len(named), 3),
            "minority_count": len(named) - majority,
            "labels": [{"name": n, "count": len(fs), "face_ids": sorted(fs)}
                       for n, fs in ordered],
            "unnamed_face_ids": sorted(f for f in members if not labels.get(f)),
        })
    out.sort(key=lambda c: (-c["lopsidedness"], -c["named_total"]))
    return out

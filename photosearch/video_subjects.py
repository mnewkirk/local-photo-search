"""Subject-layer scoring for videos (docs/plans/video-support.md, step 3b).

Pure Python + numpy over cached grounding JSON, so it is CI-testable; the VLM
call (``subjects.ground_photo_subjects`` with the category extension) lives in
the worker, never here.

Input: the segment schedule (``video_analysis.segment_schedule``), one
grounding answer per segment keyframe — ``[{"label", "category", "bbox"}]``
with bbox normalised to 0-1 (``subjects.parse_subject_boxes``' shape), ``[]``
for "no subject", ``None`` for "not grounded" — and optionally the face layer
(:class:`video_analysis.FaceAnalysis`) for identity.

Rules (variants S1 / S2 of the eval):

* **primary** subject of a segment = the box with the highest salience
  (``video_analysis.face_salience`` on the box in pixels);
* **linking** across segments: person boxes joined to a face identity link by
  that identity; unjoined person boxes are one "unidentified person" subject;
  everything else links by :func:`normalize_label` (articles/possessives
  stripped, plurals folded, a small synonym map), plus an optional crop
  similarity callback that unions label groups of the same category;
* **face join**: a person box takes the identity whose face box is contained
  in it (>= ``join_contain`` of the face's area) at the detection nearest the
  keyframe within ``±join_window_s`` (covers coasted gaps);
* **credit**: S1 gives the primary the whole segment; S2 gives a joined
  person only the part of the segment where their face track is present;
* **shares** use the whole clip as denominator (D10); ``[]``/``None``
  segments count as "no subject".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from photosearch.video_analysis import (
    DEFAULT_PARAMS,
    FaceAnalysis,
    face_salience,
    format_summary,
    segment_keyframe_times,
    summarize_runs,
)

SUBJECT_CATEGORIES = ("person", "animal", "landscape", "object", "activity")
UNIDENTIFIED = "unidentified"


@dataclass(frozen=True)
class SubjectParams:
    sigma: Optional[float] = DEFAULT_PARAMS.sigma
    join_window_s: float = 1.0      # eval sweeps {0, 1}
    join_contain: float = 0.6       # fraction of the face box inside the person box
    link_sim: float = 0.85          # crop-similarity threshold for the callback
    credit: str = "S1"              # "S1" whole segment | "S2" tracked portion


DEFAULT_SUBJECT_PARAMS = SubjectParams()

_LEADING = {"the", "a", "an", "our", "my", "his", "her", "their", "your", "its",
            "some", "this", "that", "these", "those", "one", "two", "several"}
SYNONYMS = {
    "puppy": "dog", "pup": "dog", "doggy": "dog", "doggie": "dog", "hound": "dog",
    "kitten": "cat", "kitty": "cat",
    "child": "person", "kid": "person", "boy": "person", "girl": "person",
    "man": "person", "woman": "person", "people": "person", "baby": "person",
    "toddler": "person", "player": "person", "children": "person", "men": "person",
    "women": "person", "human": "person",
    "mountains": "mountain", "sea": "ocean",
}
_NO_FOLD = {"grass", "glass", "class", "bus", "lens", "series", "species",
            "chess", "dress", "moss", "cross", "mass", "news", "gas", "us"}


def _fold_plural(w: str) -> str:
    if len(w) <= 3 or w in _NO_FOLD or w.endswith(("ss", "us", "is")):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zes")):
        return w[:-2]
    if w.endswith("s"):
        return w[:-1]
    return w


def normalize_label(label: Optional[str]) -> str:
    """``"The dog's"`` / ``"our dogs"`` / ``"Puppy"`` → ``"dog"``.

    Lowercase, strip punctuation and possessive ``'s``, drop leading articles /
    possessive pronouns / small quantifiers, fold simple plurals word by word,
    then map whole-label and last-word synonyms. Same function is meant to be
    used at query time (Risk 13)."""
    if not label:
        return ""
    s = str(label).lower().replace("’", "'")
    s = re.sub(r"'s\b", "", s)
    s = re.sub(r"[^a-z0-9 \-]+", " ", s)
    words = [w for w in s.replace("-", " ").split() if w]
    while words and words[0] in _LEADING:
        words = words[1:]
    words = [SYNONYMS.get(w, w) for w in words]
    words = [_fold_plural(w) for w in words]
    words = [SYNONYMS.get(w, w) for w in words]
    out = " ".join(words)
    return SYNONYMS.get(out, out)


def _category(box: dict) -> str:
    c = (box.get("category") or "").strip().lower()
    if c in SUBJECT_CATEGORIES:
        return c
    return "person" if normalize_label(box.get("label")) == "person" else "object"


def box_salience(bbox: Sequence[float], W: float, H: float,
                 sigma: Optional[float] = DEFAULT_SUBJECT_PARAMS.sigma) -> float:
    """Salience of a 0-1 normalised box in a W x H frame (same formula as faces)."""
    x1, y1, x2, y2 = bbox
    return face_salience((x1 * W, y1 * H, x2 * W, y2 * H), W, H, sigma)[2]


def choose_primary(boxes: Optional[Sequence[dict]], W: float, H: float,
                   params: SubjectParams = DEFAULT_SUBJECT_PARAMS) -> Optional[dict]:
    """The highest-salience box, or None for ``[]`` / ``None``."""
    best, best_s = None, -1.0
    for b in boxes or ():
        bb = b.get("bbox")
        if not bb or len(bb) != 4:
            continue
        s = box_salience(bb, W, H, params.sigma)
        if s > best_s:
            best, best_s = b, s
    return best


def _contained_frac(face_px: Sequence[float], box_px: Sequence[float]) -> float:
    fx1, fy1, fx2, fy2 = face_px
    bx1, by1, bx2, by2 = box_px
    ix = max(0.0, min(fx2, bx2) - max(fx1, bx1))
    iy = max(0.0, min(fy2, by2) - max(fy1, by1))
    area = max(1e-9, (fx2 - fx1) * (fy2 - fy1))
    return ix * iy / area


def join_face(box: dict, t: float, faces: Optional[FaceAnalysis],
              params: SubjectParams = DEFAULT_SUBJECT_PARAMS) -> Optional[int]:
    """Face identity for a person box at keyframe time ``t``, or None.

    Per identity, the detection nearest ``t`` within ``±join_window_s`` (a
    window of 0 still accepts the sampled frame nearest ``t``) is tested for
    containment in the box; among the identities that pass, the most salient
    face wins."""
    if faces is None or not faces.identities:
        return None
    W, H = faces.W, faces.H
    x1, y1, x2, y2 = box["bbox"]
    box_px = (x1 * W, y1 * H, x2 * W, y2 * H)
    ft = np.asarray(faces.frame_times, dtype=float)
    half_step = float(np.median(np.diff(ft))) / 2.0 if ft.size > 1 else 0.0
    window = max(params.join_window_s, half_step) + 1e-6
    best, best_s = None, -1.0
    for ident in faces.identities:
        near = [d for d in ident.detections if abs(d["t"] - t) <= window]
        if not near:
            continue
        d = min(near, key=lambda d: abs(d["t"] - t))
        if _contained_frac(d["bbox"], box_px) >= params.join_contain and d["salience"] > best_s:
            best, best_s = ident.ix, d["salience"]
    return best


@dataclass
class Subject:
    ix: int
    label: str
    category: str
    face_ix: Optional[int]
    focus_s: float
    share: float
    focus_intervals: list
    n_segments: int
    rep_t: float
    rep_seg: int
    rep_bbox: list

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in (
            "ix", "label", "category", "face_ix", "focus_s", "share",
            "focus_intervals", "n_segments", "rep_t", "rep_seg", "rep_bbox")}


@dataclass
class SubjectAnalysis:
    segments: list
    primaries: list           # per segment: subject ix or None
    subjects: list            # [Subject], ix == position, sorted by focus_s desc
    duration: float           # the denominator (sum of segment weights)
    n_empty: int              # segments answered []
    n_ungrounded: int         # segments with no answer (None)
    credit: str = "S1"
    segment_boxes: list = field(default_factory=list, repr=False)

    def summary_runs(self) -> list[dict]:
        return summarize_runs(self.segments, self.primaries)

    def summary_text(self, names: Optional[dict] = None) -> str:
        """``"0–12 s dog, 12–40 s Calvin (face_ix 0)"``; ``names`` maps a face
        ix to a person name (unnamed people read ``person``)."""
        names = names or {}

        def lab(k):
            s = self.subjects[k]
            if s.face_ix is not None:
                return f"{names.get(s.face_ix, 'person')} (face_ix {s.face_ix})"
            return s.label
        return format_summary(self.summary_runs(), lab)

    def to_json(self) -> dict:
        return {"credit": self.credit, "duration": self.duration,
                "segments": [[round(a, 3), round(b, 3)] for a, b in self.segments],
                "primaries": self.primaries,
                "subjects": [s.to_json() for s in self.subjects],
                "n_empty": self.n_empty, "n_ungrounded": self.n_ungrounded}


class _UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb, key=str)] = min(ra, rb, key=str)


def _merge_intervals(iv: list) -> list:
    out: list = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in out]


def score_subjects(segments: Sequence[tuple[float, float]],
                   groundings: Sequence[Optional[Sequence[dict]]],
                   faces: Optional[FaceAnalysis] = None,
                   W: Optional[float] = None, H: Optional[float] = None,
                   keyframe_times: Optional[Sequence[float]] = None,
                   params: SubjectParams = DEFAULT_SUBJECT_PARAMS,
                   similarity: Optional[Callable[[dict, dict], float]] = None,
                   ) -> SubjectAnalysis:
    """Primary subject per segment, linked into per-video subjects with shares.

    ``similarity(a, b)`` (optional, e.g. CLIP cosine of the two box crops,
    precomputed by the caller) receives two primary-box records
    ``{"seg_ix", "label", "category", "bbox", "t"}`` and returns a similarity;
    label groups of the same non-person category whose representatives score
    ``>= params.link_sim`` are linked.
    """
    if len(segments) != len(groundings):
        raise ValueError("one grounding answer per segment")
    if W is None or H is None:
        W, H = (faces.W, faces.H) if faces is not None else (1.0, 1.0)
    kts = list(keyframe_times) if keyframe_times is not None else segment_keyframe_times(segments)
    if params.credit not in ("S1", "S2"):
        raise ValueError("credit must be 'S1' or 'S2'")

    prim_recs: list[Optional[dict]] = []
    n_empty = n_ungrounded = 0
    for si, (boxes, t) in enumerate(zip(groundings, kts)):
        if boxes is None:
            n_ungrounded += 1
        elif not boxes:
            n_empty += 1
        p = choose_primary(boxes, W, H, params)
        if p is None:
            prim_recs.append(None)
            continue
        cat = _category(p)
        norm = normalize_label(p.get("label"))
        face_ix = join_face(p, t, faces, params) if cat == "person" else None
        if cat == "person":
            key = ("face", face_ix) if face_ix is not None else ("person", UNIDENTIFIED)
        else:
            key = ("label", norm or cat)
        prim_recs.append({"seg_ix": si, "t": t, "label": p.get("label"), "norm": norm,
                          "category": cat, "bbox": list(p["bbox"]), "face_ix": face_ix,
                          "key": key, "salience": box_salience(p["bbox"], W, H, params.sigma)})

    uf = _UF()
    reps: dict = {}
    for r in prim_recs:
        if r is not None:
            uf.find(r["key"])
            reps.setdefault(r["key"], r)
    if similarity is not None:
        lkeys = [k for k in reps if k[0] == "label"]
        for i, a in enumerate(lkeys):
            for b in lkeys[i + 1:]:
                ra, rb = reps[a], reps[b]
                if ra["category"] != rb["category"]:
                    continue
                if similarity(ra, rb) >= params.link_sim:
                    uf.union(a, b)

    groups: dict = {}
    for r in prim_recs:
        if r is not None:
            groups.setdefault(uf.find(r["key"]), []).append(r)

    duration = float(sum(max(0.0, b - a) for a, b in segments))
    subj_raw = []
    for root, recs in groups.items():
        focus_s, intervals = 0.0, []
        for r in recs:
            a, b = segments[r["seg_ix"]]
            if params.credit == "S2" and r["face_ix"] is not None and faces is not None:
                part = faces.present_intervals_within(r["face_ix"], a, b)
                focus_s += sum(hi - lo for lo, hi in part)
                intervals.extend(part)
            else:
                focus_s += b - a
                intervals.append([a, b])
        rep = max(recs, key=lambda r: r["salience"])
        face_ix = recs[0]["face_ix"]
        if face_ix is not None:
            label = "person"
        else:
            label = Counter(r["norm"] or r["category"] for r in recs).most_common(1)[0][0]
        category = Counter(r["category"] for r in recs).most_common(1)[0][0]
        subj_raw.append(dict(root=root, label=label, category=category, face_ix=face_ix,
                             focus_s=focus_s, intervals=_merge_intervals(intervals),
                             n=len(recs), rep=rep, first=min(r["seg_ix"] for r in recs)))
    subj_raw.sort(key=lambda s: (-s["focus_s"], s["first"]))
    remap = {s["root"]: i for i, s in enumerate(subj_raw)}
    subjects = [Subject(
        ix=i, label=s["label"], category=s["category"], face_ix=s["face_ix"],
        focus_s=round(s["focus_s"], 3),
        share=round(s["focus_s"] / duration, 4) if duration > 0 else 0.0,
        focus_intervals=s["intervals"], n_segments=s["n"],
        rep_t=s["rep"]["t"], rep_seg=s["rep"]["seg_ix"], rep_bbox=s["rep"]["bbox"])
        for i, s in enumerate(subj_raw)]
    primaries = [None if r is None else remap[uf.find(r["key"])] for r in prim_recs]
    return SubjectAnalysis(segments=[tuple(s) for s in segments], primaries=primaries,
                           subjects=subjects, duration=duration, n_empty=n_empty,
                           n_ungrounded=n_ungrounded, credit=params.credit,
                           segment_boxes=prim_recs)


# ---------------------------------------------------------------------------
# The face layer as video-scan stores it (step 3b wiring)
# ---------------------------------------------------------------------------
#
# The production worker never has the scan's full per-frame detections — the
# scan keeps only the identities' summaries. So video-scan records, per
# segment keyframe, each identity's detection NEAREST that keyframe within
# the join window (``quality_json["frames"]["segments"][i]["faces"]``), and
# the subjects pass rebuilds a minimal face layer from that plus the tracks'
# presence intervals. ``join_face`` then sees exactly the detections it would
# have chosen from the full set (it too takes the nearest within the window).

def segment_face_window(sample_fps: Optional[float],
                        params: SubjectParams = DEFAULT_SUBJECT_PARAMS) -> float:
    """The window ``join_face`` uses: ``max(join_window_s, half a sample step)``."""
    half = 0.5 / sample_fps if sample_fps and sample_fps > 0 else 0.0
    return max(params.join_window_s, half)


def segment_face_boxes(identities, t: float, window: float) -> list[dict]:
    """Per identity, the detection nearest ``t`` within ``±window`` —
    ``[{"ix", "t", "bbox": [x1, y1, x2, y2] (decoded px), "salience"}]``.
    ``identities`` are :class:`video_analysis.Identity` (with detections)."""
    out = []
    for ident in identities or ():
        near = [d for d in ident.detections if abs(d["t"] - t) <= window + 1e-6]
        if not near:
            continue
        d = min(near, key=lambda d: abs(d["t"] - t))
        out.append({"ix": int(ident.ix), "t": round(float(d["t"]), 4),
                    "bbox": [round(float(v), 1) for v in d["bbox"]],
                    "salience": round(float(d["salience"]), 6)})
    return out


@dataclass
class _StoredIdentity:
    ix: int
    detections: list
    presence_intervals: list


@dataclass
class StoredFaceLayer:
    """Duck-typed stand-in for :class:`video_analysis.FaceAnalysis`, enough
    for :func:`join_face` (``W``, ``H``, ``frame_times``, ``identities``) and
    S2 credit (:meth:`present_intervals_within`)."""

    W: float
    H: float
    frame_times: list
    identities: list
    by_ix: dict = field(default_factory=dict, repr=False)

    def present_intervals_within(self, ix: int, start: float, end: float) -> list:
        ident = self.by_ix.get(ix)
        out = []
        for a, b in (ident.presence_intervals if ident else ()):
            lo, hi = max(a, start), min(b, end)
            if hi > lo:
                out.append([lo, hi])
        return out


def face_layer_from_manifest(segments_manifest: Sequence[dict], tracks: Sequence[dict],
                             W: float, H: float,
                             sample_fps: Optional[float] = None) -> Optional[StoredFaceLayer]:
    """Rebuild the face layer from video-scan's stored manifest.

    ``segments_manifest``: ``quality_json["frames"]["segments"]`` (each may
    carry ``"faces"``); ``tracks``: ``[{"ident_ix", "presence_intervals"}]``.
    Returns None when the scan predates per-segment face boxes (no segment has
    a ``"faces"`` key) — person boxes then stay unidentified — or when the
    video has no identities at all."""
    if not tracks or not any("faces" in (s or {}) for s in segments_manifest):
        return None
    dets: dict[int, list] = {}
    for s in segments_manifest:
        for f in (s or {}).get("faces") or ():
            dets.setdefault(int(f["ix"]), []).append(
                {"t": float(f["t"]), "bbox": list(f["bbox"]),
                 "salience": float(f.get("salience") or 0.0)})
    by_ix = {}
    for t in tracks:
        ix = t.get("ident_ix")
        if ix is None:
            continue
        pres = t.get("presence_intervals") or []
        if isinstance(pres, str):
            import json
            pres = json.loads(pres)
        by_ix[int(ix)] = _StoredIdentity(ix=int(ix), detections=dets.get(int(ix), []),
                                         presence_intervals=[list(p) for p in pres])
    if not by_ix:
        return None
    step = (1.0 / sample_fps) if sample_fps and sample_fps > 0 else 0.0
    return StoredFaceLayer(W=float(W), H=float(H), frame_times=[0.0, step],
                           identities=[by_ix[k] for k in sorted(by_ix)], by_ix=by_ix)


# ---------------------------------------------------------------------------
# H1 — hybrid: person-ness from the VLM OR a salient tracked face
# ---------------------------------------------------------------------------
#
# Video eval round 2 (docs/plans/video-eval-round2-2026-10-27.md). S1 lost
# most person segments because the highest-salience box was often the
# scenery (a whole-frame "landscape"/"field" box always out-salients a person
# box), and F1 said "no subject" whenever no face was in focus. H1 decides
# person vs not-person first, from EITHER signal, and only then picks:
#
# * person segment = a VLM person box (the primary, or any person box at
#   least ``person_box_min_a`` big when ``person_mode == "any"``), OR the
#   segment's focus face (``segment_faces``, precomputed by the caller from
#   the face layer's focus — ``video_analysis.segment_focus`` in the eval,
#   the stored ``focus_intervals`` in production);
# * identity = the face joined to the person box (``join_face``), else the
#   segment's focus face; neither -> an unidentified person;
# * otherwise the VLM's non-person primary (``nonperson_rule``: highest
#   salience, or the first box the model listed), else no subject.
#
# Naming (which identities may carry a library name) is the caller's job.

@dataclass(frozen=True)
class HybridParams:
    sigma: Optional[float] = DEFAULT_PARAMS.sigma
    join_window_s: float = 1.0
    join_contain: float = 0.6
    person_mode: str = "any"          # "primary" | "any"
    person_box_min_a: float = 0.10    # sqrt(box area) as a fraction of min(W, H)
    use_faces: bool = True            # a focus face alone makes a person segment
    nonperson_rule: str = "salience"  # "salience" | "first"
    # propagate_identity(): an unidentified person segment takes the clip's
    # dominant NAMED person when that person holds at least this share of the
    # clip's identified-person time. None = off.
    propagate_min_share: Optional[float] = None


DEFAULT_HYBRID_PARAMS = HybridParams()


@dataclass
class HybridSegment:
    category: str                     # person | animal | landscape | object | activity
    label: str
    face_ix: Optional[int]
    source: str                       # vlm+face | vlm | face | vlm-nonperson


def _box_a(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.sqrt(max(0.0, x2 - x1) * max(0.0, y2 - y1)))


def score_hybrid(segments: Sequence[tuple[float, float]],
                 groundings: Sequence[Optional[Sequence[dict]]],
                 faces: Optional[FaceAnalysis] = None,
                 W: Optional[float] = None, H: Optional[float] = None,
                 keyframe_times: Optional[Sequence[float]] = None,
                 segment_faces: Optional[Sequence[Optional[int]]] = None,
                 params: HybridParams = DEFAULT_HYBRID_PARAMS) -> list[Optional[HybridSegment]]:
    """One :class:`HybridSegment` (or None = no subject) per segment."""
    if len(segments) != len(groundings):
        raise ValueError("one grounding answer per segment")
    if W is None or H is None:
        W, H = (faces.W, faces.H) if faces is not None else (1.0, 1.0)
    kts = list(keyframe_times) if keyframe_times is not None else segment_keyframe_times(segments)
    sf = list(segment_faces) if segment_faces is not None else [None] * len(segments)
    sp = SubjectParams(sigma=params.sigma, join_window_s=params.join_window_s,
                       join_contain=params.join_contain)
    out: list[Optional[HybridSegment]] = []
    for boxes, t, focus_ix in zip(groundings, kts, sf):
        boxes = [b for b in (boxes or ()) if b.get("bbox") and len(b["bbox"]) == 4]
        primary = choose_primary(boxes, W, H, sp)
        person_box = None
        if primary is not None and _category(primary) == "person":
            person_box = primary
        elif params.person_mode == "any":
            pb = [b for b in boxes if _category(b) == "person"
                  and _box_a(b["bbox"]) >= params.person_box_min_a]
            person_box = choose_primary(pb, W, H, sp)
        face_focus = focus_ix if params.use_faces else None
        if person_box is not None or face_focus is not None:
            ix = join_face(person_box, t, faces, sp) if person_box is not None else None
            src = "vlm+face" if ix is not None else ("vlm" if person_box is not None else "face")
            if ix is None and focus_ix is not None:
                ix = focus_ix
                src = "face" if person_box is None else "vlm+focus"
            label = (person_box or {}).get("label") or "person"
            out.append(HybridSegment("person", normalize_label(label) or "person", ix, src))
            continue
        np_boxes = [b for b in boxes if _category(b) != "person"]
        if not np_boxes:
            out.append(None)
            continue
        if params.nonperson_rule == "first":
            b = np_boxes[0]
        else:
            b = choose_primary(np_boxes, W, H, sp)
        cat = _category(b)
        out.append(HybridSegment(cat, normalize_label(b.get("label")) or cat, None,
                                 "vlm-nonperson"))
    return out


def focus_from_intervals(tracks: Sequence[dict], segments: Sequence[tuple[float, float]],
                         min_frac: float = DEFAULT_PARAMS.segment_min_focus_frac
                         ) -> list[Optional[int]]:
    """Per segment, the identity whose stored ``focus_intervals`` cover the
    most of it (>= ``min_frac``), else None — the production stand-in for
    ``video_analysis.segment_focus`` (video_tracks rows carry the intervals)."""
    out = []
    for a, b in segments:
        best, best_cov = None, 0.0
        for tr in tracks or ():
            iv = tr.get("focus_intervals") or []
            if isinstance(iv, str):
                import json
                iv = json.loads(iv)
            cov = sum(max(0.0, min(hi, b) - max(lo, a)) for lo, hi in iv)
            if cov > best_cov:
                best, best_cov = tr.get("ident_ix"), cov
        out.append(best if b > a and best_cov / (b - a) >= min_frac - 1e-9 else None)
    return out


def propagate_identity(hybrid: Sequence[Optional[HybridSegment]],
                       segments: Sequence[tuple[float, float]],
                       named: dict, min_share: Optional[float]) -> list[Optional[HybridSegment]]:
    """Fill unidentified person segments with the clip's dominant named person.

    ``named`` maps a face identity ix to a person id (None = unnamed). The
    dominant person is the one whose identified-person segments cover the
    most time; it must hold >= ``min_share`` of all identified-person time
    (named or not). The camera usually follows one person through a clip, and
    their face is often turned away for whole segments. Off when
    ``min_share`` is None."""
    if min_share is None:
        return list(hybrid)
    per: dict = {}
    total = 0.0
    for h, (a, b) in zip(hybrid, segments):
        if h is None or h.category != "person" or h.face_ix is None:
            continue
        total += b - a
        pid = named.get(h.face_ix)
        if pid is not None:
            per[h.face_ix] = per.get(h.face_ix, 0.0) + (b - a)
    if not per or total <= 0:
        return list(hybrid)
    best = max(per, key=lambda k: per[k])
    if per[best] / total < min_share - 1e-9:
        return list(hybrid)
    return [HybridSegment(h.category, h.label, best, "propagated")
            if h is not None and h.category == "person" and h.face_ix is None else h
            for h in hybrid]

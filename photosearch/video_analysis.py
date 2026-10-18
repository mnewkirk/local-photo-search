"""Pure-numpy video analysis core (docs/plans/video-support.md, step 3).

Everything here works on plain per-frame data that the cv2/InsightFace side
(``video_scan.py``) or the eval harness (cached npz detections) supplies, so it
is CI-testable with **numpy + stdlib only** — no cv2, scipy or torch, at import
or call time (``tests/test_video_analysis.py`` pins that).

Contents:

* schedules — sample fps, sample times, K keyframes (+ nudge to sharpest), the
  3 s / cap-40 segment schedule ``video_subjects`` uses;
* face salience ``S = A x C`` and the focus / presence eligibility gates;
* the greedy tracker, complete-linkage identity merge (never across tracks
  that overlap in time), per-frame main focus with one-frame-miss smoothing
  (variant **F1**) and the no-tracker baseline (**F0**);
* presence / focus seconds, whole-clip shares (denominator = full duration,
  D10), run-length intervals, per-segment focus for the eval;
* quality measures (shake, Immerkaer noise sigma, focus consistency, exposure
  clipping);
* the segment summariser ("0–12 s dog, 12–40 s Calvin (face_ix 0)").

Input shapes
------------
``frames``: a time-ordered sequence of ``{"t": seconds, "faces": [det, ...]}``
where ``det = {"bbox": (x1, y1, x2, y2) pixels, "det_score": float,
"embedding": 512 floats or None, "sharpness": optional float}``. Frame size is
``W x H`` pixels (the decoded, rotation-applied frame). ``duration`` is the
video's full running time in seconds.

Every bracketed number in the plan is a field of :class:`AnalysisParams`, so
the eval can sweep them with ``dataclasses.replace(DEFAULT_PARAMS, ...)``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

import numpy as np

ANALYSIS_VERSION = 1

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisParams:
    """Tunables for the face layer. Defaults are the plan's bracketed values."""

    # Salience: S = A * C, C = exp(-r^2 / (2 sigma^2)). sigma None/<=0 = off (C=1).
    sigma: Optional[float] = 0.35
    # Focus eligibility: apparent size A = sqrt(w*h) / min(W, H).
    a_min: float = 0.03
    # Presence eligibility (also what the tracker sees).
    presence_det_score: float = 0.5
    presence_min_edge_px: float = 16.0
    # Tracker association: IoU with the last box (within coast_gap_s), or
    # ArcFace L2 to the track's running mean within track_embed_window_s.
    track_iou: float = 0.3
    track_embed_l2: float = 0.9
    track_embed_window_s: float = 1.5
    # An IoU match is vetoed when both embeddings exist and are farther apart
    # than this (a cut, or one kid stepping into another's spot). Not in the
    # plan's rule; None restores the literal "IoU OR embedding" rule.
    track_iou_veto_l2: Optional[float] = 1.2
    # Gaps up to this long are coasted (counted as present).
    coast_gap_s: float = 1.0
    # Identity merge: complete linkage on track mean embeddings.
    merge_l2: float = 0.9
    # Focus hysteresis: a run of <= this many frames sandwiched between the
    # same focus on both sides is smoothed away (a missed frame, a blip).
    focus_miss_frames: int = 1
    # segment_focus(): an identity must hold focus for at least this fraction
    # of a segment to be that segment's main subject.
    segment_min_focus_frac: float = 0.3
    # Identity encoding = normalised mean of the top-k detections by salience.
    encoding_top_k: int = 5


DEFAULT_PARAMS = AnalysisParams()

# Schedules
SEGMENT_S = 3.0
SEGMENT_CAP = 40
KEYFRAME_NUDGE_S = 0.5

# Quality
SHAKE_WINDOW_S = 1.0
FOCUS_CONSISTENCY_FRAC = 0.5
FOCUS_CONSISTENCY_PCTL = 90.0
NOISE_FLAT_PERCENTILE = 50.0
CLIP_LO = 4
CLIP_HI = 251


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def sample_fps(duration: float) -> float:
    """Dense sampling rate for the face layer: 4 fps up to 60 s, 3 fps up to
    5 min, 2 fps beyond (a 29-min clip is ~3.5k frames)."""
    d = float(duration or 0.0)
    if d <= 60.0:
        return 4.0
    if d <= 300.0:
        return 3.0
    return 2.0


def sample_times(duration: float, fps: Optional[float] = None) -> list[float]:
    """Frame-centre sample times ``(i + 0.5) / fps`` inside ``[0, duration)``.
    Always at least one sample."""
    d = max(0.0, float(duration or 0.0))
    fps = float(fps or sample_fps(d))
    n = max(1, int(math.floor(d * fps + 1e-9)))
    return [min((i + 0.5) / fps, d) for i in range(n)]


def keyframe_count(duration: float) -> int:
    """K: 1 for d <= 2 s, 3 for d <= 10 s, rising linearly to 8 at >= 60 s."""
    d = float(duration or 0.0)
    if d <= 2.0:
        return 1
    if d <= 10.0:
        return 3
    if d >= 60.0:
        return 8
    return int(min(8, max(3, round(3 + 5 * (d - 10.0) / 50.0))))


def keyframe_centres(duration: float, k: Optional[int] = None) -> list[float]:
    """Uniform quantile centres ``(i + 0.5) / K * d``."""
    d = max(0.0, float(duration or 0.0))
    k = int(k or keyframe_count(d))
    return [(i + 0.5) / k * d for i in range(k)]


def nudge_to_sharpest(centres: Sequence[float], frame_times: Sequence[float],
                      sharpness: Sequence[Optional[float]],
                      window_s: float = KEYFRAME_NUDGE_S) -> list[int]:
    """For each centre, the index of the sharpest frame within ``±window_s``
    (falls back to the nearest frame when none in the window has a value).
    Returns frame indices, in centre order."""
    ft = np.asarray(frame_times, dtype=float)
    if ft.size == 0:
        return []
    sh = np.array([np.nan if s is None else float(s) for s in sharpness], dtype=float)
    out = []
    for c in centres:
        nearest = int(np.argmin(np.abs(ft - c)))
        cand = np.where((np.abs(ft - c) <= window_s + 1e-9) & ~np.isnan(sh))[0]
        if cand.size == 0:
            out.append(nearest)
            continue
        best = cand[np.argmax(sh[cand])]
        out.append(int(best))
    return out


def keyframe_times(duration: float, frame_times: Optional[Sequence[float]] = None,
                   sharpness: Optional[Sequence[Optional[float]]] = None,
                   k: Optional[int] = None) -> list[float]:
    """K keyframe times; nudged to the sharpest sampled frame when per-frame
    sharpness is supplied."""
    centres = keyframe_centres(duration, k)
    if frame_times is None or sharpness is None or len(frame_times) == 0:
        return centres
    idx = nudge_to_sharpest(centres, frame_times, sharpness)
    return [float(frame_times[i]) for i in idx]


def segment_schedule(duration: float, seg_s: float = SEGMENT_S,
                     cap: int = SEGMENT_CAP) -> list[tuple[float, float]]:
    """``[(start, end), ...]`` covering ``[0, duration]`` exactly.

    3 s segments; a trailing remainder shorter than half a segment is folded
    into the last one. When that would exceed ``cap`` segments, ``cap`` equal
    segments instead (a 29-min clip gets 40 x 43.5 s).
    """
    d = max(0.0, float(duration or 0.0))
    if d <= 0.0:
        return [(0.0, 0.0)]
    n_full = int(math.floor(d / seg_s + 1e-9))
    rem = d - n_full * seg_s
    n = n_full + (1 if rem >= 0.5 * seg_s - 1e-9 else 0)
    n = max(1, n)
    if n > cap:
        step = d / cap
        return [(i * step, d if i == cap - 1 else (i + 1) * step) for i in range(cap)]
    bounds = [i * seg_s for i in range(n)] + [d]
    return [(bounds[i], bounds[i + 1]) for i in range(n)]


def segment_keyframe_times(segments: Sequence[tuple[float, float]]) -> list[float]:
    """One keyframe per segment, at its midpoint."""
    return [(a + b) / 2.0 for a, b in segments]


def coarsen_segments(segments: Sequence[tuple[float, float]],
                     max_n: int = 12) -> list[tuple[float, float]]:
    """Group consecutive segments into at most ``max_n`` (the eval's labelling
    coarsening). Boundaries stay on production segment boundaries."""
    n = len(segments)
    if n <= max_n:
        return list(segments)
    edges = [int(round(i * n / max_n)) for i in range(max_n + 1)]
    return [(segments[edges[i]][0], segments[edges[i + 1] - 1][1]) for i in range(max_n)]


def frame_cells(frame_times: Sequence[float], duration: float) -> tuple[np.ndarray, np.ndarray]:
    """The slice of the clip each sampled frame stands for: boundaries at the
    midpoints between samples, the first from 0, the last to ``duration``.
    Cell lengths sum to the full running time (the D10 denominator)."""
    t = np.asarray(frame_times, dtype=float)
    if t.size == 0:
        return np.zeros(0), np.zeros(0)
    d = max(float(duration or 0.0), float(t[-1]))
    mids = (t[:-1] + t[1:]) / 2.0
    starts = np.concatenate([[0.0], mids])
    ends = np.concatenate([mids, [d]])
    return starts, ends


# ---------------------------------------------------------------------------
# Salience
# ---------------------------------------------------------------------------


def face_salience(bbox: Sequence[float], W: float, H: float,
                  sigma: Optional[float] = DEFAULT_PARAMS.sigma) -> tuple[float, float, float]:
    """``(A, C, S)`` for a pixel bbox in a ``W x H`` frame.

    ``A = sqrt(w*h) / min(W, H)``; ``C = exp(-r^2 / (2 sigma^2))`` with
    ``r^2 = (dx/(W/2))^2 + 0.5 * (dy/(H/2))^2``; ``S = A * C``.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    A = math.sqrt(w * h) / max(1e-9, min(W, H))
    dx = (x1 + x2) / 2.0 - W / 2.0
    dy = (y1 + y2) / 2.0 - H / 2.0
    r2 = (dx / (W / 2.0)) ** 2 + 0.5 * (dy / (H / 2.0)) ** 2
    C = 1.0 if (sigma is None or sigma <= 0) else math.exp(-r2 / (2.0 * sigma * sigma))
    return A, C, A * C


def presence_eligible(det: dict, params: AnalysisParams = DEFAULT_PARAMS) -> bool:
    x1, y1, x2, y2 = (float(v) for v in det["bbox"])
    score = det.get("det_score")
    score = 1.0 if score is None else float(score)
    return (score >= params.presence_det_score
            and min(x2 - x1, y2 - y1) >= params.presence_min_edge_px)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _unit(v) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a))
    return a / n if n > 0 else None


def _l2(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return math.inf
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Detections, tracks, identities
# ---------------------------------------------------------------------------


@dataclass
class _Det:
    frame_ix: int
    t: float
    bbox: tuple
    det_score: float
    emb: Optional[np.ndarray]
    A: float
    C: float
    S: float
    focus_ok: bool
    sharpness: Optional[float]


@dataclass
class _Track:
    dets: list = field(default_factory=list)
    emb_sum: Optional[np.ndarray] = None

    @property
    def last(self) -> _Det:
        return self.dets[-1]

    @property
    def t_first(self) -> float:
        return self.dets[0].t

    @property
    def t_last(self) -> float:
        return self.dets[-1].t

    @property
    def mean_emb(self) -> Optional[np.ndarray]:
        return _unit(self.emb_sum)

    def add(self, d: _Det) -> None:
        self.dets.append(d)
        if d.emb is not None:
            self.emb_sum = d.emb.copy() if self.emb_sum is None else self.emb_sum + d.emb


def _collect_dets(frames: Sequence[dict], W: float, H: float,
                  params: AnalysisParams) -> list[list[_Det]]:
    out = []
    for fi, fr in enumerate(frames):
        t = float(fr["t"])
        row = []
        for det in fr.get("faces") or ():
            if not presence_eligible(det, params):
                continue
            A, C, S = face_salience(det["bbox"], W, H, params.sigma)
            sc = det.get("det_score")
            row.append(_Det(fi, t, tuple(float(v) for v in det["bbox"]),
                            1.0 if sc is None else float(sc),
                            _unit(det.get("embedding")), A, C, S,
                            A >= params.a_min, det.get("sharpness")))
        out.append(row)
    return out


def track_faces(dets_by_frame: Sequence[Sequence[_Det]],
                params: AnalysisParams = DEFAULT_PARAMS) -> list[_Track]:
    """Greedy frame-to-frame association (no Hungarian).

    A (track, detection) pair is a candidate when IoU with the track's last box
    is >= ``track_iou`` and the gap is <= ``coast_gap_s`` (unless the two
    embeddings are > ``track_iou_veto_l2`` apart), or when the L2 to the
    track's running-mean embedding is <= ``track_embed_l2`` and the gap is <=
    ``track_embed_window_s``. Candidates are taken best-first — IoU matches
    before embedding-only ones, then by IoU, then by smaller L2 — each track and
    detection used at most once. Leftover detections start new tracks.
    """
    tracks: list[_Track] = []
    horizon = max(params.coast_gap_s, params.track_embed_window_s)
    for row in dets_by_frame:
        if not row:
            continue
        t = row[0].t
        active = [i for i, tr in enumerate(tracks) if t - tr.t_last <= horizon + 1e-9]
        pairs = []
        for ti in active:
            tr = tracks[ti]
            gap = t - tr.t_last
            if gap <= 0:
                continue
            mean = tr.mean_emb
            for di, d in enumerate(row):
                ov = iou(tr.last.bbox, d.bbox)
                iou_ok = gap <= params.coast_gap_s + 1e-9 and ov >= params.track_iou
                dist = _l2(mean, d.emb)
                if (iou_ok and params.track_iou_veto_l2 is not None
                        and math.isfinite(dist) and dist > params.track_iou_veto_l2):
                    iou_ok = False
                emb_ok = gap <= params.track_embed_window_s + 1e-9 and dist <= params.track_embed_l2
                if iou_ok or emb_ok:
                    pairs.append(((1 if iou_ok else 0, ov if iou_ok else 0.0, -dist), ti, di))
        pairs.sort(key=lambda p: p[0], reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tracks[ti].add(row[di])
        for di, d in enumerate(row):
            if di not in used_d:
                tr = _Track()
                tr.add(d)
                tracks.append(tr)
    return tracks


def merge_tracks(tracks: Sequence[_Track],
                 params: AnalysisParams = DEFAULT_PARAMS) -> list[list[int]]:
    """Complete-linkage agglomeration of tracks into identities.

    Cluster distance = the MAX pairwise L2 between member tracks' mean
    embeddings; merge the closest pair while it is <= ``merge_l2``. Two
    clusters are never merged when any member of one overlaps any member of
    the other in time (a person is in a frame at most once). Tracks with no
    embedding never merge. Returns lists of track indices.
    """
    n = len(tracks)
    if n == 0:
        return []
    means = [tr.mean_emb for tr in tracks]
    L = np.full((n, n), np.inf)
    O = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            L[i, j] = L[j, i] = _l2(means[i], means[j])
            ov = (tracks[i].t_first <= tracks[j].t_last + 1e-9
                  and tracks[j].t_first <= tracks[i].t_last + 1e-9)
            O[i, j] = O[j, i] = ov
    alive = list(range(n))
    members = {i: [i] for i in range(n)}
    while True:
        best = None
        for a_ix, i in enumerate(alive):
            for j in alive[a_ix + 1:]:
                if O[i, j] or L[i, j] > params.merge_l2:
                    continue
                if best is None or L[i, j] < best[0]:
                    best = (L[i, j], i, j)
        if best is None:
            break
        _, i, j = best
        L[i, :] = np.maximum(L[i, :], L[j, :])
        L[:, i] = L[i, :]
        O[i, :] = O[i, :] | O[j, :]
        O[:, i] = O[i, :]
        members[i].extend(members.pop(j))
        alive.remove(j)
    return [sorted(members[i]) for i in alive]


def smooth_focus(labels: Sequence[Optional[Hashable]], max_run: int = 1) -> list:
    """Replace any run of <= ``max_run`` frames that has the SAME label on both
    sides with that label (a missed detection or one-frame blip doesn't flip
    focus). Runs at the clip edges are left alone."""
    labels = list(labels)
    if max_run <= 0 or len(labels) < 3:
        return labels
    runs = []  # [label, start, length]
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] += 1
        else:
            runs.append([lab, i, 1])
    out = list(labels)
    for k in range(1, len(runs) - 1):
        lab, start, length = runs[k]
        if length <= max_run and runs[k - 1][0] == runs[k + 1][0] and runs[k - 1][0] is not None:
            for i in range(start, start + length):
                out[i] = runs[k - 1][0]
    return out


def run_length_intervals(flags: Sequence[bool], starts: np.ndarray,
                         ends: np.ndarray, ndigits: int = 3) -> list[list[float]]:
    """``[[start_s, end_s], ...]`` for each run of True frames, using the
    frames' cells (see :func:`frame_cells`)."""
    out = []
    run_start = None
    for i, f in enumerate(flags):
        if f and run_start is None:
            run_start = i
        if not f and run_start is not None:
            out.append([round(float(starts[run_start]), ndigits), round(float(ends[i - 1]), ndigits)])
            run_start = None
    if run_start is not None:
        out.append([round(float(starts[run_start]), ndigits), round(float(ends[len(flags) - 1]), ndigits)])
    return out


def labels_to_intervals(labels: Sequence[Optional[Hashable]], starts: np.ndarray,
                        ends: np.ndarray) -> dict:
    """``{label: intervals}`` for every non-None label in a per-frame list."""
    keys = {lab for lab in labels if lab is not None}
    return {k: run_length_intervals([lab == k for lab in labels], starts, ends) for k in keys}


@dataclass
class Identity:
    """One within-video person (one future ``faces`` row)."""

    ix: int
    n_det: int
    track_count: int
    presence_s: float
    presence_share: float
    focus_s: float
    focus_share: float
    presence_intervals: list
    focus_intervals: list
    mean_salience: float
    rep_t: float
    rep_frame_ix: int
    rep_bbox: list
    rep_det_score: float
    first_t: float
    last_t: float
    encoding: Optional[list]          # unit 512-d, mean of top-k by salience
    detections: list = field(repr=False, default_factory=list)  # [{t, frame_ix, bbox, det_score, salience, sharpness}]

    def to_json(self, with_encoding: bool = False) -> dict:
        d = {k: getattr(self, k) for k in (
            "ix", "n_det", "track_count", "presence_s", "presence_share", "focus_s",
            "focus_share", "presence_intervals", "focus_intervals", "mean_salience",
            "rep_t", "rep_frame_ix", "rep_bbox", "rep_det_score", "first_t", "last_t")}
        if with_encoding:
            d["encoding"] = self.encoding
        return d


@dataclass
class FaceAnalysis:
    """Result of :func:`analyze_faces` / :func:`analyze_faces_f0`."""

    variant: str
    duration: float
    W: float
    H: float
    frame_times: list
    cell_starts: np.ndarray = field(repr=False)
    cell_ends: np.ndarray = field(repr=False)
    identities: list              # [Identity], ix == position; sorted by focus_s, presence_s
    focus: list                   # per frame: identity ix or None (after smoothing for F1)
    raw_focus: list               # per frame, before smoothing
    presence: np.ndarray = field(repr=False)  # bool [n_identities, n_frames], coasting included
    params: AnalysisParams = DEFAULT_PARAMS

    def to_json(self) -> dict:
        return {"variant": self.variant, "duration": self.duration,
                "version": ANALYSIS_VERSION,
                "identities": [i.to_json() for i in self.identities]}

    def presence_fraction(self, ix: int, start: float, end: float) -> float:
        """Fraction of ``[start, end]`` where identity ``ix`` is present
        (detected or coasted)."""
        if end <= start or ix is None or ix >= len(self.identities):
            return 0.0
        ov = np.clip(np.minimum(self.cell_ends, end) - np.maximum(self.cell_starts, start), 0, None)
        return float((ov * self.presence[ix]).sum() / (end - start))

    def present_intervals_within(self, ix: int, start: float, end: float) -> list:
        out = []
        for a, b in self.identities[ix].presence_intervals:
            lo, hi = max(a, start), min(b, end)
            if hi > lo:
                out.append([lo, hi])
        return out


def _build_result(variant: str, groups: list[list[_Det]], n_frames: int,
                  frame_times: list, duration: float, W: float, H: float,
                  raw_focus_det: list[Optional[_Det]], params: AnalysisParams,
                  coast: bool, smooth: bool, track_counts: list[int]) -> FaceAnalysis:
    starts, ends = frame_cells(frame_times, duration)
    cell = ends - starts
    total = float(cell.sum()) if cell.size else 0.0
    det_owner = {}
    for gi, g in enumerate(groups):
        for d in g:
            det_owner[id(d)] = gi
    raw = [None if d is None else det_owner[id(d)] for d in raw_focus_det]
    focus = smooth_focus(raw, params.focus_miss_frames) if smooth else list(raw)

    pres = np.zeros((len(groups), n_frames), dtype=bool)
    for gi, g in enumerate(groups):
        fis = sorted({d.frame_ix for d in g})
        pres[gi, fis] = True
        if coast:
            for a, b in zip(fis, fis[1:]):
                if b - a > 1 and frame_times[b] - frame_times[a] <= params.coast_gap_s + 1e-9:
                    pres[gi, a:b + 1] = True

    provisional = []
    for gi, g in enumerate(groups):
        focus_flags = [f == gi for f in focus]
        focus_s = float(cell[np.array(focus_flags, dtype=bool)].sum()) if n_frames else 0.0
        presence_s = float(cell[pres[gi]].sum()) if n_frames else 0.0
        rep = max(g, key=lambda d: (d.S, d.det_score))
        top = sorted((d for d in g if d.emb is not None), key=lambda d: d.S, reverse=True)
        top = top[:params.encoding_top_k]
        enc = _unit(np.sum([d.emb for d in top], axis=0)) if top else None
        provisional.append(dict(
            gi=gi, focus_s=focus_s, presence_s=presence_s,
            focus_flags=focus_flags, rep=rep, enc=enc,
            first_t=min(d.t for d in g), last_t=max(d.t for d in g)))
    order = sorted(provisional, key=lambda p: (-p["focus_s"], -p["presence_s"], p["first_t"]))
    remap = {p["gi"]: new for new, p in enumerate(order)}

    identities = []
    for new, p in enumerate(order):
        g = groups[p["gi"]]
        rep = p["rep"]
        identities.append(Identity(
            ix=new, n_det=len(g), track_count=track_counts[p["gi"]],
            presence_s=round(p["presence_s"], 3),
            presence_share=round(p["presence_s"] / total, 4) if total else 0.0,
            focus_s=round(p["focus_s"], 3),
            focus_share=round(p["focus_s"] / total, 4) if total else 0.0,
            presence_intervals=run_length_intervals(pres[p["gi"]], starts, ends),
            focus_intervals=run_length_intervals(p["focus_flags"], starts, ends),
            mean_salience=round(float(np.mean([d.S for d in g])), 5),
            rep_t=rep.t, rep_frame_ix=rep.frame_ix, rep_bbox=list(rep.bbox),
            rep_det_score=rep.det_score, first_t=p["first_t"], last_t=p["last_t"],
            encoding=None if p["enc"] is None else [float(x) for x in p["enc"]],
            detections=[{"t": d.t, "frame_ix": d.frame_ix, "bbox": list(d.bbox),
                         "det_score": d.det_score, "salience": d.S, "A": d.A,
                         "sharpness": d.sharpness} for d in sorted(g, key=lambda d: d.t)],
        ))
    pres_sorted = pres[[p["gi"] for p in order]] if order else pres
    return FaceAnalysis(
        variant=variant, duration=float(ends[-1]) if n_frames else float(duration or 0.0),
        W=W, H=H, frame_times=list(frame_times), cell_starts=starts, cell_ends=ends,
        identities=identities,
        focus=[None if f is None else remap[f] for f in focus],
        raw_focus=[None if f is None else remap[f] for f in raw],
        presence=pres_sorted, params=params)


def _frame_focus_det(row: Sequence[_Det]) -> Optional[_Det]:
    elig = [d for d in row if d.focus_ok]
    return max(elig, key=lambda d: d.S) if elig else None


def analyze_faces(frames: Sequence[dict], W: float, H: float, duration: float,
                  params: AnalysisParams = DEFAULT_PARAMS) -> FaceAnalysis:
    """Variant **F1**: tracker + coasting + identity merge; per-frame main
    focus = the highest-salience focus-eligible face, smoothed with
    ``focus_miss_frames`` hysteresis. Focus is computed over ALL faces;
    naming happens later via ``faces.person_id``."""
    dets = _collect_dets(frames, W, H, params)
    tracks = track_faces(dets, params)
    clusters = merge_tracks(tracks, params)
    groups = [[d for ti in cl for d in tracks[ti].dets] for cl in clusters]
    counts = [len(cl) for cl in clusters]
    raw = [_frame_focus_det(row) for row in dets]
    return _build_result("F1", groups, len(frames), [float(f["t"]) for f in frames],
                         duration, W, H, raw, params, coast=True, smooth=True,
                         track_counts=counts)


def analyze_faces_f0(frames: Sequence[dict], W: float, H: float, duration: float,
                     params: AnalysisParams = DEFAULT_PARAMS) -> FaceAnalysis:
    """Variant **F0** (eval baseline): no tracker, no coasting, no smoothing.
    Identities come from per-detection leader clustering (greedy, a detection
    joins the nearest cluster centroid within ``merge_l2``; a cluster takes at
    most one detection per frame); focus is the per-frame largest-salience
    face."""
    dets = _collect_dets(frames, W, H, params)
    sums: list[Optional[np.ndarray]] = []
    groups: list[list[_Det]] = []
    for row in dets:
        pairs = []
        for di, d in enumerate(row):
            for ci, s in enumerate(sums):
                dist = _l2(_unit(s), d.emb)
                if dist <= params.merge_l2:
                    pairs.append((dist, ci, di))
        pairs.sort()
        used_c, used_d = set(), set()
        for _, ci, di in pairs:
            if ci in used_c or di in used_d:
                continue
            used_c.add(ci)
            used_d.add(di)
            groups[ci].append(row[di])
            sums[ci] = sums[ci] + row[di].emb
        for di, d in enumerate(row):
            if di not in used_d:
                groups.append([d])
                sums.append(None if d.emb is None else d.emb.copy())
    raw = [_frame_focus_det(row) for row in dets]
    return _build_result("F0", groups, len(frames), [float(f["t"]) for f in frames],
                         duration, W, H, raw, params, coast=False, smooth=False,
                         track_counts=[1] * len(groups))


def segment_focus(fa: FaceAnalysis, segments: Sequence[tuple[float, float]],
                  min_frac: Optional[float] = None) -> list[Optional[int]]:
    """Per segment, the identity holding focus for the most time, or None when
    nobody holds it for ``min_frac`` (default ``params.segment_min_focus_frac``)
    of the segment. This is how F0/F1 answer the eval's per-segment question."""
    min_frac = fa.params.segment_min_focus_frac if min_frac is None else min_frac
    out = []
    for a, b in segments:
        if b <= a or not fa.frame_times:
            out.append(None)
            continue
        ov = np.clip(np.minimum(fa.cell_ends, b) - np.maximum(fa.cell_starts, a), 0, None)
        acc: dict = {}
        for f, w in zip(fa.focus, ov):
            if f is not None and w > 0:
                acc[f] = acc.get(f, 0.0) + float(w)
        if not acc:
            out.append(None)
            continue
        best = max(acc, key=lambda k: (acc[k], -k))
        out.append(best if acc[best] / (b - a) >= min_frac - 1e-9 else None)
    return out


# ---------------------------------------------------------------------------
# Segment summariser
# ---------------------------------------------------------------------------


def _fmt_s(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 or x >= 10 else f"{x:.1f}"


def summarize_runs(segments: Sequence[tuple[float, float]],
                   keys: Sequence[Optional[Hashable]]) -> list[dict]:
    """Merge consecutive segments with the same key into
    ``[{"start", "end", "key"}]`` (key None = no subject)."""
    runs: list[dict] = []
    for (a, b), k in zip(segments, keys):
        if runs and runs[-1]["key"] == k and abs(runs[-1]["end"] - a) < 1e-6:
            runs[-1]["end"] = b
        else:
            runs.append({"start": a, "end": b, "key": k})
    return runs


def format_summary(runs: Sequence[dict],
                   label_for: Callable[[Any], Optional[str]] = lambda k: None,
                   none_label: str = "no subject") -> str:
    """``"0–12 s dog, 12–40 s Calvin (face_ix 0)"``. ``label_for(key)`` names a
    run; a run whose key is None reads ``none_label``."""
    parts = []
    for r in runs:
        lab = none_label if r["key"] is None else (label_for(r["key"]) or str(r["key"]))
        parts.append(f"{_fmt_s(r['start'])}–{_fmt_s(r['end'])} s {lab}")
    return ", ".join(parts)


def summarize_face_segments(fa: FaceAnalysis, segments: Sequence[tuple[float, float]],
                            names: Optional[dict] = None) -> dict:
    """F0/F1 segment summary: ``{"runs": [...], "text": "..."}`` with identity
    keys rendered as ``name (face_ix N)`` or ``person (face_ix N)``."""
    names = names or {}
    runs = summarize_runs(segments, segment_focus(fa, segments))
    text = format_summary(runs, lambda k: f"{names.get(k, 'person')} (face_ix {k})")
    return {"runs": [{"start": round(r["start"], 3), "end": round(r["end"], 3),
                      "face_ix": r["key"]} for r in runs], "text": text}


# ---------------------------------------------------------------------------
# Quality measures
# ---------------------------------------------------------------------------


def _moving_average_valid(x: np.ndarray, win: int) -> np.ndarray:
    k = np.ones(win) / win
    return np.convolve(x, k, mode="valid")


def shake_score(shifts_x: Sequence[float], shifts_y: Sequence[float], fps: float,
                W: float, window_s: float = SHAKE_WINDOW_S) -> Optional[float]:
    """Camera shake as % of frame width.

    ``shifts_*`` are per-frame global translations (pixels, frame i-1 → i, e.g.
    phase correlation). The trajectory is their cumulative sum; the high-pass
    is the trajectory minus its centred ``window_s`` moving average; the score
    is the RMS magnitude of that residual / W * 100. A steady pan is linear, so
    its moving average equals it and it scores ~0. Only frames where the full
    window fits are scored (edge truncation would bias a pan); a clip shorter
    than the window falls back to the residual of a straight-line fit.
    """
    sx = np.asarray(shifts_x, dtype=float)
    sy = np.asarray(shifts_y, dtype=float)
    if sx.size < 3 or sx.size != sy.size or W <= 0:
        return None
    tx, ty = np.cumsum(sx), np.cumsum(sy)
    win = max(3, int(round(window_s * fps)))
    if win % 2 == 0:
        win += 1
    if tx.size >= win:
        h = win // 2
        rx = tx[h:tx.size - h] - _moving_average_valid(tx, win)
        ry = ty[h:ty.size - h] - _moving_average_valid(ty, win)
    else:
        i = np.arange(tx.size, dtype=float)
        rx = tx - np.polyval(np.polyfit(i, tx, 1), i)
        ry = ty - np.polyval(np.polyfit(i, ty, 1), i)
    rms = math.sqrt(float(np.mean(rx * rx + ry * ry)))
    return 100.0 * rms / W


def _conv3_valid(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3x3 'valid' correlation via slicing (numpy only)."""
    h, w = img.shape
    out = np.zeros((h - 2, w - 2), dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            c = k[dy, dx]
            if c:
                out += c * img[dy:dy + h - 2, dx:dx + w - 2]
    return out


_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)


def immerkaer_sigma(luma: np.ndarray, flat_percentile: Optional[float] = NOISE_FLAT_PERCENTILE,
                    clip_lo: int = CLIP_LO, clip_hi: int = CLIP_HI) -> Optional[float]:
    """Immerkaer (1996) fast noise sigma on a 2-D luma array (0-255 scale).

    ``sigma = sqrt(pi/2) / 6 * mean(|I * M|)`` over the kept pixels. Flat-region
    masking keeps only pixels whose Sobel gradient magnitude is at or below the
    ``flat_percentile`` of the frame (edges and texture read as noise
    otherwise); clipped pixels are excluded too (saturation reads as flat and
    noiseless). ``flat_percentile=None`` disables the mask.
    """
    img = np.asarray(luma, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] < 3 or img.shape[1] < 3:
        return None
    resp = np.abs(_conv3_valid(img, _IMMERKAER))
    inner = img[1:-1, 1:-1]
    keep = (inner > clip_lo) & (inner < clip_hi)
    if flat_percentile is not None:
        gx = _conv3_valid(img, _SOBEL_X)
        gy = _conv3_valid(img, _SOBEL_X.T)
        g = np.hypot(gx, gy)
        cand = g[keep]
        if cand.size:
            keep &= g <= np.percentile(cand, flat_percentile)
    if not keep.any():
        return None
    return float(math.sqrt(math.pi / 2.0) / 6.0 * resp[keep].mean())


def noise_score(sigmas: Iterable[Optional[float]]) -> Optional[float]:
    """Median of per-frame Immerkaer sigmas (None/NaN ignored)."""
    v = [float(s) for s in sigmas if s is not None and not math.isnan(float(s))]
    return float(np.median(v)) if v else None


def focus_consistency(sharpness: Iterable[Optional[float]],
                      frac: float = FOCUS_CONSISTENCY_FRAC,
                      pctl: float = FOCUS_CONSISTENCY_PCTL) -> Optional[float]:
    """Share of main-focus frames whose face-crop sharpness is >= ``frac`` x
    this video's own p90 — relative within the video, never a library
    threshold. None when there are no measured frames."""
    v = np.array([float(s) for s in sharpness if s is not None], dtype=float)
    if v.size == 0:
        return None
    ref = float(np.percentile(v, pctl))
    if ref <= 0:
        return None
    return float(np.mean(v >= frac * ref))


def main_focus_sharpness(fa: FaceAnalysis) -> list[Optional[float]]:
    """The face-crop sharpness of the main-focus face in each frame (from the
    ``sharpness`` field of the input detections), for :func:`focus_consistency`."""
    by_frame: dict = {}
    for ident in fa.identities:
        for d in ident.detections:
            by_frame[(ident.ix, d["frame_ix"])] = d.get("sharpness")
    return [by_frame.get((f, fi)) for fi, f in enumerate(fa.focus) if f is not None
            and (f, fi) in by_frame]


def exposure_clipping(luma: np.ndarray, lo: int = CLIP_LO,
                      hi: int = CLIP_HI) -> tuple[float, float]:
    """``(highlight_frac, shadow_frac)``: fraction of pixels >= ``hi`` / <= ``lo``."""
    a = np.asarray(luma)
    if a.size == 0:
        return 0.0, 0.0
    return float(np.mean(a >= hi)), float(np.mean(a <= lo))


def aggregate_exposure(per_frame: Iterable[tuple[float, float]]) -> dict:
    """Mean and p90 of per-frame clipped fractions."""
    arr = np.array(list(per_frame), dtype=float).reshape(-1, 2)
    if arr.size == 0:
        return {"highlight_frac": None, "shadow_frac": None,
                "highlight_p90": None, "shadow_p90": None}
    return {"highlight_frac": float(arr[:, 0].mean()), "shadow_frac": float(arr[:, 1].mean()),
            "highlight_p90": float(np.percentile(arr[:, 0], 90)),
            "shadow_p90": float(np.percentile(arr[:, 1], 90))}


def most_common(values: Iterable[Hashable]) -> Optional[Hashable]:
    c = Counter(v for v in values if v is not None)
    return c.most_common(1)[0][0] if c else None

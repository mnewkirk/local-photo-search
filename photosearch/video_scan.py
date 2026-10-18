"""The ``video-scan`` worker pass — decode, detect, measure (docs/plans/
video-support.md step 3). **Worker-only**: cv2 and InsightFace are imported
lazily, nothing here touches the DB or HTTP, so the step-4 eval harness can
import :func:`scan_video` / :class:`ScanData` directly.

Two stages, deliberately separable:

* :func:`scan_video` — ONE sequential decode of a local file. At
  ``video_analysis.sample_fps(duration)`` it takes a frame, runs face
  detection (``faces.detect_faces_bgr``, which RAISES on error), measures each
  face crop's sharpness, the global shift to the previous sample (phase
  correlation on downscaled luma), the Immerkaer noise sigma (native-res
  centre crop), exposure clipping and a global sharpness; and (``want_frames``)
  keeps the frames it will need later — the keyframe/segment-keyframe
  windows and every frame with a face — as JPEGs in a temp dir. Returns a
  :class:`ScanData`.
* :func:`build_scan_result` — the analysis: ``video_analysis.analyze_faces``
  (tracker + identities), the quality numbers, the CLIP mean embedding, the
  poster choice, and the asset JPEGs (keyframes, per-segment keyframes, a
  representative frame and face crops per identity). Returns the submit
  payload and the assets; no I/O beyond reading ScanData's temp frames.

ZIP-wrapped files (an iOS bundle saved with a video extension, step 0's
biggest finding) are unwrapped by :func:`unwrap_zip` first — stdlib zipfile,
exactly one member, no path traversal, size-capped — and the inner file is
probed locally with ffprobe (:func:`probe_local`) so the server can fill the
dates/dims the outer ZIP never had.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from . import video_analysis as VA
from .video_analysis import DEFAULT_PARAMS, AnalysisParams
from .video_scan_store import (CROP_SIZES, SCAN_VERSION, crop_name, keyframe_name,
                               rep_name, segment_name)

# Stored-frame and asset sizes (long edge, px).
STORE_MAX_EDGE = 1920      # sampled frames kept for later (rep frames come from these)
KEYFRAME_EDGE = 1024       # the K keyframes (CLIP, video-describe)
SEGMENT_EDGE = 768         # one per segment (video-subjects grounding, step 3b)
JPEG_QUALITY = 85
STORE_QUALITY = 90

# Downscaled luma used for the global shift / sharpness / exposure.
LUMA_SMALL_W = 256
# Native-res centre crop for the noise sigma (noise shrinks with downscaling).
NOISE_CROP = 512
# Face crops are resized to this square before the Laplacian, so sharpness is
# comparable across detections WITHIN a video (focus_consistency is relative).
FACE_SHARP_PX = 96
# Padded native face crops are kept (for >STORE_MAX_EDGE videos only) for
# focus-eligible faces, so identity crops are rendered at native resolution.
NATIVE_CROP_PAD = 1.2      # half-side = PAD x the face's larger edge

# ZIP unwrap guards.
ZIP_MAX_BYTES = 16 * 1024 ** 3
ZIP_MAX_RATIO = 50.0       # inflated / compressed — anything above is a bomb

_LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


class ScanOpenError(Exception):
    """cv2 cannot open (or decode a single frame of) a COMPLETE file — a
    terminal verdict, stored as ``scan_error``."""


class ZipRefused(Exception):
    """A ZIP-wrapped file that is not exactly one safe video member — a
    terminal verdict, stored as ``scan_error``."""


# ---------------------------------------------------------------------------
# ScanData
# ---------------------------------------------------------------------------

@dataclass
class ScanData:
    """Everything one decode of a video produced.

    Fields
    ------
    path : str
        The file that was decoded.
    width, height : int
        Decoded frame size in pixels — DISPLAY orientation (cv2 applies the
        container rotation). Every ``bbox`` below is in these coordinates.
    duration : float
        The full running time in seconds (the probe's when given, else the
        container's / the last decoded timestamp) — the D10 denominator.
    fps : float
        Container frame rate (0 when unknown).
    sample_fps : float
        The face-layer sampling rate (``video_analysis.sample_fps``).
    n_decoded : int
        Frames decoded (grabbed) in total.
    frames : list[dict]
        ``video_analysis`` input, one per SAMPLED frame:
        ``{"t": s, "faces": [{"bbox": (x1, y1, x2, y2), "det_score",
        "embedding": 512 floats, "sharpness": float}]}``.
    frame_measures : list[dict]
        Parallel to ``frames``: ``{"t", "frame_no", "shift_x", "shift_y"``
        (decoded px, vs the previous sample; 0 for the first)``, "sigma"``
        (Immerkaer noise sigma, 0-255 luma)``, "clip_hi", "clip_lo"``
        (clipped fractions)``, "sharpness"`` (global Laplacian variance of
        the downscaled luma)``, "tenengrad"`` (mean squared Sobel gradient of
        the downscaled luma)``, "luma_mean"}``.
    measures : dict (property)
        The clip-level summary the eval harness reads: ``{"shake", "noise",
        "focus_consistency", "tenengrad_max", "exposure": {...},
        "highlight_frac", "shadow_frac"}`` (see :meth:`summary_measures`).
    W, H : int (properties)
        Aliases of ``width`` / ``height``.
    keyframe_times : list[float] (property)
        Times of the K keyframes.
    det_size : tuple | None
        The detector input size used (None = the 640x640 default).
    keyframe_ix : list[int]
        Indices into ``frames`` of the K keyframes (uniform quantile centres
        nudged to the sharpest sample within +-0.5 s).
    segments : list[tuple[float, float]]
        ``video_analysis.segment_schedule(duration)``.
    segment_ix : list[int]
        Index into ``frames`` of each segment's keyframe (midpoint, nudged).
    store_dir : str | None
        Temp dir holding the kept frames (``want_frames=True``); remove with
        :meth:`cleanup`.
    store_scale : float
        Stored-frame px / decoded px (< 1 only above ``STORE_MAX_EDGE``).
    timings : dict
        Seconds spent per stage (decode, detect, measure, store).
    """

    path: str
    width: int
    height: int
    duration: float
    fps: float
    sample_fps: float
    n_decoded: int
    frames: list
    frame_measures: list
    keyframe_ix: list
    segments: list
    segment_ix: list
    store_dir: Optional[str] = None
    store_scale: float = 1.0
    timings: dict = field(default_factory=dict)
    det_size: Optional[tuple] = None
    scan_version: int = SCAN_VERSION
    _summary: Optional[dict] = field(default=None, repr=False)
    # internal: sampled-frame index -> stored JPEG path; (frame, det) -> native crop
    _stored: dict = field(default_factory=dict, repr=False)
    _native_crops: dict = field(default_factory=dict, repr=False)

    @property
    def frame_times(self) -> list[float]:
        return [float(f["t"]) for f in self.frames]

    @property
    def W(self) -> int:
        return self.width

    @property
    def H(self) -> int:
        return self.height

    @property
    def keyframe_times(self) -> list[float]:
        return [float(self.frames[i]["t"]) for i in self.keyframe_ix]

    @property
    def measures(self) -> dict:
        return self.summary_measures()

    def summary_measures(self, params: AnalysisParams = DEFAULT_PARAMS) -> dict:
        """Clip-level quality numbers (cached for the default params):
        shake (% of width), noise (median Immerkaer sigma), focus
        consistency of the F1 main-focus face, max frame Tenengrad, and the
        exposure clipping aggregate."""
        if self._summary is not None and params is DEFAULT_PARAMS:
            return self._summary
        fm = self.frame_measures
        fa = VA.analyze_faces(self.frames, self.width, self.height, self.duration, params)
        exposure = VA.aggregate_exposure((m["clip_hi"], m["clip_lo"]) for m in fm)
        ten = [m.get("tenengrad") for m in fm if m.get("tenengrad") is not None]
        out = {
            "shake": VA.shake_score([m["shift_x"] for m in fm], [m["shift_y"] for m in fm],
                                    self.sample_fps, self.width),
            "noise": VA.noise_score(m["sigma"] for m in fm),
            "focus_consistency": VA.focus_consistency(VA.main_focus_sharpness(fa)),
            "tenengrad_max": max(ten) if ten else None,
            "exposure": exposure,
            "highlight_frac": exposure["highlight_frac"],
            "shadow_frac": exposure["shadow_frac"],
        }
        if params is DEFAULT_PARAMS:
            self._summary = out
        return out

    def stored_index(self, i: int) -> Optional[int]:
        """``i`` if that frame was kept, else the nearest kept frame."""
        if not self._stored:
            return None
        if i in self._stored:
            return i
        return min(self._stored, key=lambda k: (abs(k - i), k))

    def frame_path(self, i: int) -> Optional[str]:
        j = self.stored_index(i)
        return None if j is None else self._stored[j]

    def frame_rgb(self, i: int, max_edge: Optional[int] = None):
        """The kept frame nearest sample ``i`` as an RGB array (stored scale,
        optionally reduced so the long edge <= ``max_edge``)."""
        from PIL import Image
        p = self.frame_path(i)
        if p is None:
            return None
        with Image.open(p) as im:
            im = im.convert("RGB")
            if max_edge and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            return np.asarray(im)

    def keyframe_images(self, max_edge: int = KEYFRAME_EDGE) -> list[tuple[float, object]]:
        """``[(t, RGB array)]`` for the K keyframes (needs ``want_frames``)."""
        out = []
        for i in self.keyframe_ix:
            img = self.frame_rgb(i, max_edge)
            if img is not None:
                out.append((float(self.frames[self.stored_index(i)]["t"]), img))
        return out

    def native_crop(self, frame_ix: int, bbox) -> Optional[tuple[str, tuple[int, int]]]:
        """``(path, (x0, y0))`` of a kept native-res padded crop for the
        detection with this bbox, if one was kept."""
        return self._native_crops.get((frame_ix, tuple(int(round(v)) for v in bbox)))

    def cleanup(self) -> None:
        if self.store_dir:
            shutil.rmtree(self.store_dir, ignore_errors=True)
            self.store_dir = None
            self._stored = {}
            self._native_crops = {}


# ---------------------------------------------------------------------------
# Pure measures (numpy only — CI-testable)
# ---------------------------------------------------------------------------

def laplacian_var(gray) -> Optional[float]:
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return None
    return float(VA._conv3_valid(g, _LAPLACE).var())


def tenengrad(gray) -> Optional[float]:
    """Mean squared Sobel gradient magnitude (Tenengrad focus measure)."""
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return None
    gx = VA._conv3_valid(g, VA._SOBEL_X)
    gy = VA._conv3_valid(g, VA._SOBEL_X.T)
    return float(np.mean(gx * gx + gy * gy))


def phase_shift(prev, cur) -> tuple[float, float]:
    """Global translation of ``cur`` relative to ``prev`` (same-shape 2-D
    float arrays) by phase correlation, with a parabolic sub-pixel peak.
    Returns ``(dx, dy)`` in the arrays' pixels."""
    a = np.asarray(prev, dtype=np.float64)
    b = np.asarray(cur, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 4:
        return 0.0, 0.0
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    A = np.fft.fft2((a - a.mean()) * win)
    B = np.fft.fft2((b - b.mean()) * win)
    R = B * np.conj(A)
    R /= np.abs(R) + 1e-12
    r = np.real(np.fft.ifft2(R))
    py, px = np.unravel_index(int(np.argmax(r)), r.shape)

    def sub(c, n, get):
        l, m, rr = get((c - 1) % n), get(c), get((c + 1) % n)
        den = l - 2 * m + rr
        return 0.0 if abs(den) < 1e-12 else 0.5 * (l - rr) / den

    dx = px + sub(px, w, lambda k: r[py, k])
    dy = py + sub(py, h, lambda k: r[k, px])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy)


def centre_crop(a, size: int):
    h, w = a.shape[:2]
    if h <= size and w <= size:
        return a
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    return a[y0:y0 + size, x0:x0 + size]


def trbl_from_xyxy(bbox, scale: float = 1.0, width: Optional[int] = None,
                   height: Optional[int] = None) -> list[int]:
    """(x1, y1, x2, y2) -> faces-table (top, right, bottom, left), scaled and
    (when the frame size is given) clamped into the frame — InsightFace boxes
    can poke past the edge."""
    x1, y1, x2, y2 = (float(v) * scale for v in bbox)
    if width:
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    if height:
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    return [int(round(y1)), int(round(x2)), int(round(y2)), int(round(x1))]


def xyxy_from_trbl(bbox) -> tuple[int, int, int, int]:
    top, right, bottom, left = bbox
    return (int(left), int(top), int(right), int(bottom))


def mean_unit(vectors: Sequence[Sequence[float]]) -> Optional[list[float]]:
    """Mean of the L2-normalised vectors, renormalised (the CLIP embedding of
    a video). None when there is nothing to average."""
    vs = []
    for v in vectors:
        if v is None:
            continue
        a = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(a)
        if n > 0:
            vs.append(a / n)
    if not vs:
        return None
    m = np.sum(vs, axis=0)
    n = np.linalg.norm(m)
    if n <= 0:
        return None
    return [float(x) for x in m / n]


def needed_times(duration: float) -> list[float]:
    """Keyframe + segment-keyframe centres; frames within the nudge window of
    one of these are kept even with no face in them."""
    segs = VA.segment_schedule(duration)
    return sorted(VA.keyframe_centres(duration) + VA.segment_keyframe_times(segs))


# ---------------------------------------------------------------------------
# ZIP unwrap + local probe
# ---------------------------------------------------------------------------

def is_zip_file(path: str) -> bool:
    with open(path, "rb") as fh:
        return fh.read(4) == b"PK\x03\x04"


def unwrap_zip(path: str, dest_dir: str, *, max_bytes: int = ZIP_MAX_BYTES,
               max_ratio: float = ZIP_MAX_RATIO) -> str:
    """Extract the single video inside a ZIP-wrapped file to ``dest_dir``.

    Refuses (ZipRefused) anything but exactly one regular, non-encrypted
    member with a safe relative name, an uncompressed size <= ``max_bytes``
    and a ratio <= ``max_ratio`` (zip bomb). Extraction streams with a byte
    cap that is enforced on the bytes actually read, not the header's claim.
    Returns the extracted path."""
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ZipRefused(f"not a readable ZIP: {e}")
    with zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) != 1:
            raise ZipRefused(f"ZIP holds {len(members)} files, expected exactly 1")
        m = members[0]
        name = m.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p]
        if (not parts or name.startswith("/") or ".." in parts
                or ":" in parts[0] or any(p in (".",) for p in parts)):
            raise ZipRefused(f"unsafe member name {m.filename!r}")
        if m.flag_bits & 0x1:
            raise ZipRefused("encrypted ZIP member")
        if m.file_size > max_bytes:
            raise ZipRefused(f"member is {m.file_size} bytes > {max_bytes}")
        if m.compress_size > 0 and m.file_size / m.compress_size > max_ratio:
            raise ZipRefused("member inflates suspiciously (zip bomb?)")
        base = os.path.basename(parts[-1])
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base) or "inner.mov"
        out = os.path.join(dest_dir, "inner_" + safe)
        n = 0
        import zlib
        try:
            with zf.open(m) as src, open(out, "wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > max_bytes:
                        raise ZipRefused(f"member exceeded {max_bytes} bytes while extracting")
                    dst.write(chunk)
        except BaseException as e:
            try:
                os.unlink(out)
            except OSError:
                pass
            if isinstance(e, (zipfile.BadZipFile, zlib.error, EOFError)):
                raise ZipRefused(f"member does not inflate: {e}")
            raise
    return out


def probe_local(path: str, tz_name: Optional[str] = None) -> dict:
    """ffprobe a local file -> ``{"ok": probe dict}`` (VideoProbeResult
    fields, no photo_id) or ``{"verdict": "<unreadable: ...>"}``. Raises
    ``video_probe.ProbeToolError`` when ffprobe itself is missing/broken."""
    from . import video_probe as VP
    data = VP.run_ffprobe(path)
    if "_verdict" in data:
        return {"verdict": data["_verdict"]}
    res = VP.parse_probe(data, tz_name=tz_name)
    if res.get("probe_error"):
        return {"verdict": res["probe_error"]}
    for k in ("camera_family", "gps_alt"):
        res.pop(k, None)
    res["probe_version"] = VP.PROBE_VERSION
    return {"ok": res}


# ---------------------------------------------------------------------------
# scan_video
# ---------------------------------------------------------------------------

def _xyxy(det_trbl) -> tuple[float, float, float, float]:
    top, right, bottom, left = det_trbl
    return (float(left), float(top), float(right), float(bottom))


def scan_video(path: str, params: AnalysisParams = DEFAULT_PARAMS,
               detect: Optional[Callable] = None, want_frames: bool = True,
               duration: Optional[float] = None, det_size=None,
               progress: Optional[Callable[[str], None]] = None) -> ScanData:
    """Decode ``path`` once, sequentially; see the module docstring.

    ``detect(bgr) -> [{"bbox": (top, right, bottom, left), "det_score",
    "encoding"}]`` defaults to ``faces.detect_faces_bgr`` and must RAISE on
    error (an exception propagates: the worker reports a failure row).
    ``duration`` (the probe's) sets the denominator and the sampling rate;
    otherwise the container's frame count / fps is used. ``det_size`` (an int
    or ``(w, h)``) sets InsightFace's detector input resolution for the
    default detector — an eval axis; production keeps 640.

    Raises :class:`ScanOpenError` when cv2 cannot open the file or decode a
    single frame of it.
    """
    import cv2  # lazy: worker-only

    if isinstance(det_size, (int, float)):
        det_size = (int(det_size), int(det_size))
    if detect is None:
        from .faces import detect_faces_bgr
        if det_size:
            def detect(bgr, _ds=tuple(det_size)):
                return detect_faces_bgr(bgr, det_size=_ds)
        else:
            detect = detect_faces_bgr
    t_start = time.time()
    timings = {"decode": 0.0, "detect": 0.0, "measure": 0.0, "store": 0.0}

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise ScanOpenError("cv2 could not open the file")
    store_dir = tempfile.mkdtemp(prefix="vscan-") if want_frames else None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0 or fps > 1000:
            fps = 0.0
        nframes = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        est = float(duration) if duration and duration > 0 else (
            nframes / fps if fps > 0 and nframes > 0 else None)
        sfps = VA.sample_fps(est or 0.0)
        half_frame = 0.5 / fps if fps > 0 else 0.0
        need = needed_times(est) if est else None
        window = VA.KEYFRAME_NUDGE_S + 1.0 / sfps

        frames: list[dict] = []
        measures: list[dict] = []
        stored: dict = {}
        native: dict = {}
        prev_small = None
        W = H = 0
        scale = 1.0
        next_t = 0.5 / sfps
        idx = -1
        last_t = 0.0
        t0 = time.time()
        while True:
            if not cap.grab():
                break
            idx += 1
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = (pos / 1000.0) if (pos and pos > 0) else (idx / fps if fps > 0 else 0.0)
            if idx > 0 and fps > 0 and t <= 0:
                t = idx / fps
            last_t = max(last_t, t)
            if t + half_frame < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            while next_t <= t + half_frame:
                next_t += 1.0 / sfps
            timings["decode"] += time.time() - t0

            if not W:
                H, W = frame.shape[:2]
                scale = min(1.0, STORE_MAX_EDGE / float(max(W, H)))
            fi = len(frames)

            # --- detection -------------------------------------------------
            t1 = time.time()
            dets = detect(frame) or []
            timings["detect"] += time.time() - t1

            # --- measures --------------------------------------------------
            t1 = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sh = max(1, int(round(gray.shape[0] * LUMA_SMALL_W / float(gray.shape[1]))))
            small = cv2.resize(gray, (LUMA_SMALL_W, sh), interpolation=cv2.INTER_AREA)
            small_f = small.astype(np.float64)
            if prev_small is not None and prev_small.shape == small_f.shape:
                dx, dy = phase_shift(prev_small, small_f)
                k = gray.shape[1] / float(LUMA_SMALL_W)
                dx, dy = dx * k, dy * k
            else:
                dx = dy = 0.0
            prev_small = small_f
            hi, lo = VA.exposure_clipping(small)
            sigma = VA.immerkaer_sigma(centre_crop(gray, NOISE_CROP))
            gsharp = laplacian_var(small_f)
            ten = tenengrad(small_f)
            faces_out = []
            for d in dets:
                x1, y1, x2, y2 = _xyxy(d["bbox"])
                fx1, fy1 = max(0, int(x1)), max(0, int(y1))
                fx2, fy2 = min(W, int(math.ceil(x2))), min(H, int(math.ceil(y2)))
                fsharp = None
                if fx2 - fx1 >= 4 and fy2 - fy1 >= 4:
                    fc = cv2.resize(gray[fy1:fy2, fx1:fx2], (FACE_SHARP_PX, FACE_SHARP_PX),
                                    interpolation=cv2.INTER_AREA)
                    fsharp = laplacian_var(fc)
                faces_out.append({"bbox": (x1, y1, x2, y2),
                                  "det_score": float(d.get("det_score", 1.0)),
                                  "embedding": list(d["encoding"]) if d.get("encoding") is not None else None,
                                  "sharpness": fsharp})
            timings["measure"] += time.time() - t1
            frames.append({"t": round(t, 4), "faces": faces_out})
            measures.append({"t": round(t, 4), "frame_no": idx, "shift_x": dx, "shift_y": dy,
                             "sigma": sigma, "clip_hi": hi, "clip_lo": lo,
                             "sharpness": gsharp, "tenengrad": ten,
                             "luma_mean": float(small.mean())})

            # --- keep what later stages need -------------------------------
            if store_dir is not None:
                t1 = time.time()
                keep = bool(faces_out) or need is None or any(
                    abs(t - c) <= window for c in need)
                if keep:
                    img = frame if scale >= 1.0 else cv2.resize(
                        frame, (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
                        interpolation=cv2.INTER_AREA)
                    p = os.path.join(store_dir, f"f{fi:05d}.jpg")
                    cv2.imwrite(p, img, [int(cv2.IMWRITE_JPEG_QUALITY), STORE_QUALITY])
                    stored[fi] = p
                if scale < 1.0:
                    for fo in faces_out:
                        x1, y1, x2, y2 = fo["bbox"]
                        A = math.sqrt(max(0, x2 - x1) * max(0, y2 - y1)) / max(1, min(W, H))
                        if A < params.a_min:
                            continue
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        half = NATIVE_CROP_PAD * max(x2 - x1, y2 - y1)
                        cx0, cy0 = max(0, int(cx - half)), max(0, int(cy - half))
                        cx1, cy1 = min(W, int(cx + half)), min(H, int(cy + half))
                        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                            continue
                        cp = os.path.join(store_dir, f"c{fi:05d}_{len(native):05d}.jpg")
                        cv2.imwrite(cp, frame[cy0:cy1, cx0:cx1],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), STORE_QUALITY])
                        native[(fi, tuple(int(round(v)) for v in fo["bbox"]))] = (cp, (cx0, cy0))
                timings["store"] += time.time() - t1
            if progress and fi % 50 == 0:
                progress(f"t={t:.1f}s samples={fi + 1}")
            t0 = time.time()
        timings["decode"] += time.time() - t0

        if not frames:
            raise ScanOpenError("cv2 opened the file but decoded no frames")
        dur = float(duration) if duration and duration > 0 else max(
            est or 0.0, last_t + (1.0 / fps if fps > 0 else 0.0))
        ft = [f["t"] for f in frames]
        gs = [m["sharpness"] for m in measures]
        kix = VA.nudge_to_sharpest(VA.keyframe_centres(dur), ft, gs)
        segs = VA.segment_schedule(dur)
        six = VA.nudge_to_sharpest(VA.segment_keyframe_times(segs), ft, gs)
        timings["total"] = round(time.time() - t_start, 3)
        return ScanData(path=path, width=int(W), height=int(H), duration=round(dur, 4),
                        fps=fps, sample_fps=sfps, n_decoded=idx + 1, frames=frames,
                        frame_measures=measures, keyframe_ix=kix, segments=segs,
                        segment_ix=six, store_dir=store_dir, store_scale=scale,
                        timings={k: round(v, 3) for k, v in timings.items()},
                        det_size=tuple(det_size) if det_size else None,
                        _stored=stored, _native_crops=native)
    except BaseException:
        if store_dir:
            shutil.rmtree(store_dir, ignore_errors=True)
        raise
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# build_scan_result
# ---------------------------------------------------------------------------

def _jpeg(rgb, max_edge: Optional[int] = None, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image
    im = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if im.mode != "RGB":
        im = im.convert("RGB")
    if max_edge and max(im.size) > max_edge:
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _clip_embed_paths(paths: list[str]) -> list[Optional[list[float]]]:
    from .clip_embed import embed_images_stream
    out: list = [None] * len(paths)
    for i, emb in embed_images_stream(paths, batch_size=8):
        out[i] = emb
    return out


def _identity_crops(scan: ScanData, ident, workdir: str) -> dict[int, bytes]:
    """Face crops at CROP_SIZES for one identity, rendered with face_crop's
    own framing (20% pad, square) from a native-res crop when one was kept,
    else from the stored rep frame."""
    from .face_crop import render_face_crops
    fi = int(ident.rep_frame_ix)
    nat = scan.native_crop(fi, ident.rep_bbox)
    if nat is not None:
        path, (x0, y0) = nat
        x1, y1, x2, y2 = ident.rep_bbox
        bbox = (int(y1 - y0), int(x2 - x0), int(y2 - y0), int(x1 - x0))
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        return render_face_crops(path, bbox, w, h, CROP_SIZES)
    path = scan.frame_path(fi)
    if path is None:
        return {}
    bbox = trbl_from_xyxy(ident.rep_bbox, scan.store_scale)
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
    return render_face_crops(path, bbox, w, h, CROP_SIZES)


def _frame_salience(scan: ScanData, i: int, params: AnalysisParams) -> float:
    best = 0.0
    for d in scan.frames[i]["faces"]:
        if not VA.presence_eligible(d, params):
            continue
        A, _, S = VA.face_salience(d["bbox"], scan.width, scan.height, params.sigma)
        if A >= params.a_min:
            best = max(best, S)
    return best


def build_scan_result(scan: ScanData, params: AnalysisParams = DEFAULT_PARAMS,
                      embed: Optional[Callable[[list[str]], list]] = None,
                      with_clip: bool = True) -> dict:
    """Analysis + payload + assets for one :class:`ScanData`.

    Returns ``{"result": <VideoScanResult fields, no photo_id>,
    "assets": [{"kind", "name", "data"}], "poster": {size: jpeg} | None,
    "analysis": FaceAnalysis}``. ``embed(paths) -> [512 floats | None]``
    defaults to clip_embed; ``with_clip=False`` skips CLIP entirely.
    """
    t0 = time.time()
    fa = VA.analyze_faces(scan.frames, scan.width, scan.height, scan.duration, params)
    t_analysis = time.time() - t0

    # ---- quality --------------------------------------------------------
    fm = scan.frame_measures
    sx = [m["shift_x"] for m in fm]
    sy = [m["shift_y"] for m in fm]
    shake = VA.shake_score(sx, sy, scan.sample_fps, scan.width)
    noise = VA.noise_score(m["sigma"] for m in fm)
    focus_c = VA.focus_consistency(VA.main_focus_sharpness(fa))
    exposure = VA.aggregate_exposure((m["clip_hi"], m["clip_lo"]) for m in fm)
    ten = [m.get("tenengrad") for m in fm if m.get("tenengrad") is not None]

    have_frames = bool(scan._stored)
    assets: list[dict] = []
    manifest = {"keyframes": [], "segments": [], "reps": {}}
    workdir = tempfile.mkdtemp(prefix="vscan-out-")
    try:
        # ---- K keyframes (+ CLIP) ---------------------------------------
        kf_paths = []
        for n, i in enumerate(scan.keyframe_ix):
            if not have_frames:
                break
            j = scan.stored_index(i)
            rgb = scan.frame_rgb(j, KEYFRAME_EDGE)
            data = _jpeg(rgb)
            name = keyframe_name(n)
            assets.append({"kind": "keyframe", "name": name, "data": data})
            p = os.path.join(workdir, name)
            with open(p, "wb") as fh:
                fh.write(data)
            kf_paths.append(p)
            manifest["keyframes"].append({"name": f"keyframe_{name}",
                                          "t": scan.frames[j]["t"], "frame_ix": j})
        clip_emb = None
        if with_clip and kf_paths:
            embs = (embed or _clip_embed_paths)(kf_paths)
            clip_emb = mean_unit(embs)

        # ---- per-segment keyframes (video-subjects, step 3b) ------------
        # Each entry also carries every identity's detection nearest the
        # keyframe within the join window ("faces", decoded px): the subjects
        # pass joins a VLM person box to a face identity with them
        # (video_subjects.face_layer_from_manifest) without the full
        # per-frame detections, which are not stored.
        from .video_subjects import segment_face_boxes, segment_face_window
        face_window = segment_face_window(scan.sample_fps)
        manifest["segment_face_window_s"] = round(face_window, 4)
        for n, (i, (a, b)) in enumerate(zip(scan.segment_ix, scan.segments)):
            if not have_frames:
                break
            j = scan.stored_index(i)
            name = segment_name(n)
            t_kf = scan.frames[j]["t"]
            assets.append({"kind": "keyframe", "name": name,
                           "data": _jpeg(scan.frame_rgb(j, SEGMENT_EDGE))})
            manifest["segments"].append({"name": f"keyframe_{name}", "start": round(a, 3),
                                         "end": round(b, 3), "t": t_kf,
                                         "frame_ix": j,
                                         "faces": segment_face_boxes(
                                             fa.identities, float(t_kf), face_window)})

        # ---- identities --------------------------------------------------
        identities = []
        rep_done = set()
        for ident in fa.identities:
            if ident.encoding is None:
                continue
            fi = int(ident.rep_frame_ix)
            if have_frames and scan.frame_path(fi) is not None:
                if fi not in rep_done:
                    assets.append({"kind": "keyframe", "name": rep_name(fi),
                                   "data": _jpeg(scan.frame_rgb(fi), quality=JPEG_QUALITY)})
                    rep_done.add(fi)
                for size, data in _identity_crops(scan, ident, workdir).items():
                    assets.append({"kind": "crop", "name": crop_name(ident.ix, size),
                                   "data": data})
                manifest["reps"][str(ident.ix)] = f"keyframe_{rep_name(fi)}"
            sw = int(round(scan.width * scan.store_scale))
            shh = int(round(scan.height * scan.store_scale))
            identities.append({
                "ident_ix": ident.ix,
                "bbox": trbl_from_xyxy(ident.rep_bbox, scan.store_scale, sw, shh),
                "det_score": ident.rep_det_score,
                "encoding": ident.encoding,
                "rep_t": ident.rep_t,
                "rep_frame": fi,
                "t_start": ident.first_t,
                "t_end": ident.last_t,
                "n_det": ident.n_det,
                "presence_s": ident.presence_s,
                "focus_s": ident.focus_s,
                "presence_intervals": ident.presence_intervals,
                "focus_intervals": ident.focus_intervals,
                "mean_salience": ident.mean_salience,
            })

        # ---- poster: the highest-salience keyframe -----------------------
        poster = None
        poster_t = None
        if have_frames and scan.keyframe_ix:
            cands = [scan.stored_index(i) for i in scan.keyframe_ix]
            sal = [_frame_salience(scan, j, params) for j in cands]
            pick = cands[int(np.argmax(sal))] if max(sal) > 0 else cands[len(cands) // 2]
            from .video_probe import poster_jpegs
            poster = poster_jpegs(scan.frame_rgb(pick))
            poster_t = scan.frames[pick]["t"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    quality_json = {
        "analysis_version": VA.ANALYSIS_VERSION, "variant": fa.variant,
        "scan_version": SCAN_VERSION,
        "width": scan.width, "height": scan.height, "duration": scan.duration,
        "fps": scan.fps, "sample_fps": scan.sample_fps, "n_samples": len(scan.frames),
        "n_decoded": scan.n_decoded, "store_scale": round(scan.store_scale, 4),
        "shake": shake, "noise": noise, "focus_consistency": focus_c,
        "tenengrad_max": max(ten) if ten else None,
        "exposure": exposure, "frames": manifest,
        "identities": [i.to_json() for i in fa.identities],
        "timings": dict(scan.timings, analysis=round(t_analysis, 3)),
    }
    result = {
        "scan_version": SCAN_VERSION,
        "scan_error": None,
        "shake": shake,
        "noise": noise,
        "focus_consistency": focus_c,
        "quality_json": quality_json,
        "clip_embedding": clip_emb,
        "identities": identities,
        "poster_uploaded": poster is not None,
        "poster_t": poster_t,
    }
    return {"result": result, "assets": assets, "poster": poster, "analysis": fa}


def assets_bodies(assets: list[dict], poster: Optional[dict], version: int = SCAN_VERSION,
                  max_assets: int = 100, max_bytes: int = 48 * 1024 * 1024) -> list[dict]:
    """Split assets into POST /api/videos/{id}/assets bodies under the
    endpoint's per-request caps (video_api.MAX_ASSETS / MAX_TOTAL_BYTES).
    The poster goes in the LAST body, so it is installed only once every
    keyframe and crop it sits beside has landed."""
    import base64
    items = [{"kind": a["kind"], "name": a["name"],
              "data_b64": base64.b64encode(a["data"]).decode("ascii"),
              "_n": len(a["data"])} for a in assets]
    if poster:
        items += [{"kind": "poster", "size": size,
                   "data_b64": base64.b64encode(data).decode("ascii"), "_n": len(data)}
                  for size, data in poster.items()]
    bodies, cur, cur_n = [], [], 0
    for it in items:
        if cur and (len(cur) >= max_assets or cur_n + it["_n"] > max_bytes):
            bodies.append(cur)
            cur, cur_n = [], 0
        cur.append(it)
        cur_n += it["_n"]
    if cur:
        bodies.append(cur)
    return [{"version": int(version),
             "assets": [{k: v for k, v in it.items() if k != "_n"} for it in body]}
            for body in bodies]

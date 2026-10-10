"""Measured sharpness from the ORIGINAL pixels: face, subject, whole frame.

Step 2 of ``docs/plans/sharpness-measurement.md``. This module only MEASURES;
nothing here writes to the DB, reads ``rank_measure``'s cache, or decides a
tag. The eval (``evals/sharpness_eval.py``) decides which of the stored
candidate features, and which threshold, becomes ``blurry``.

WHY NOT JUST ``cv2.Laplacian(gray).var()``

That is what ``rank_measure._measure_photo`` does for face crops, and it is
the right signal *within one shoot*. Across a library it is not comparable:

- **Resolution.** A 60 MP a7R VI frame and a 12 MP phone frame put a different
  number of pixels across the same detail. Every photo is therefore resized to
  a fixed long edge (``NORM_LONG_EDGE``) before anything is measured, and the
  scale is recorded.
- **Noise.** High-ISO noise *adds* high-frequency energy, so raw Laplacian
  variance reads a noisy night frame as sharp (the GoPro false positive that
  sank ``aes_sharpness``). The Immerkaer noise sigma is estimated once per
  photo and subtracted: for additive white noise the 4-neighbour Laplacian's
  variance grows by exactly ``sum(kernel**2) * sigma**2 = 20 * sigma**2``
  (``NOISE_K``). Real sensor noise is spatially correlated after demosaicing
  and JPEG, so 20 is the theoretical value, not a fitted one; the eval may
  retune it.
- **Where you look.** A whole-frame mean is fooled by bokeh and by sky.
  Regions are cut into ~``TILE`` px tiles, featureless tiles are skipped, and
  the statistics are taken over the remaining tiles: the *best* tile answers
  "is anything sharp?", the median "is it sharp everywhere?".

REGIONS, in priority order (the first with at least one textured tile is the
headline ``source``):

1. ``faces``    -- face boxes (pixel coords in the EXIF-oriented ORIGINAL, the
   space ``faces.bbox_*`` is stored in), scaled to the normalised image.
2. ``subjects`` -- ``photos.subject_boxes`` (normalised 0-1, EXIF-oriented,
   unpadded; ``subjects._encode_for_grounding`` transposes before grounding).
3. ``frame``    -- the whole image.

The whole frame is ALWAYS measured as well, so the eval can compare it with
the region that won.

THE HEADLINE ``score`` (v1)

``max(0, lap_nc_max)`` of the headline region: the noise-corrected Laplacian
variance of its sharpest tile. That is a provisional pick -- "sharp somewhere
in the part that matters" -- and the labelled eval may choose a different
feature (p95, the best/median ratio, Tenengrad...). Every candidate is kept in
``detail`` precisely so that choice needs no re-decode. If the eval changes
the pick, bump ``SHARPNESS_VERSION``.

Everything is numpy, not cv2: CI mocks cv2 (``tests/conftest.py``), and a
numpy implementation is deterministic and identical on the NAS, the desktop
and CI.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional

import numpy as np

#: Bump when the measurement or the headline pick changes; stored beside the
#: score so a backfill can re-measure only stale rows.
SHARPNESS_VERSION = 1

#: Every photo is measured at this long edge (px), so a 60 MP and a 12 MP
#: frame put the same number of pixels across the same fraction of the scene.
#: Smaller originals are UPSAMPLED (recorded as ``upsampled``): a 1600 px scan
#: genuinely carries less detail per frame, and the measurement should say so.
NORM_LONG_EDGE = 4000

#: Nominal tile edge (px, at the normalised scale). A region is split into
#: ``floor(edge / TILE)`` equal tiles per axis (so tiles run TILE..2*TILE-1);
#: a region smaller than TILE on an axis is one tile on that axis.
TILE = 192

#: A face whose shorter edge (at the normalised scale) is below this is not
#: measured: a tiny crop's Laplacian says more about JPEG blocks than focus.
MIN_FACE_EDGE = 40

#: Any region (subject box, face) smaller than this on its short edge is skipped.
MIN_REGION_EDGE = 24

#: Laplacian-variance contribution of unit-variance white noise for the
#: 4-neighbour kernel [[0,1,0],[1,-4,1],[0,1,0]]: 1+1+1+1+16.
NOISE_K = 20.0

#: Tenengrad (Sobel gx^2+gy^2) contribution of unit-variance white noise:
#: each 3x3 Sobel kernel has sum(k^2) = 12, two axes.
TENENGRAD_NOISE_K = 24.0

#: A tile is FLAT (skipped) when its mean Tenengrad is below
#: ``max(FLAT_TENENGRAD_ABS, FLAT_NOISE_MULT * TENENGRAD_NOISE_K * sigma^2)``.
#: 100 is a mean Sobel magnitude of ~10, i.e. ~2.5 grey levels/pixel of real
#: gradient; the noise term keeps a grainy high-ISO sky from counting as
#: texture. Provisional -- the eval reports flat_skipped so this can be tuned.
FLAT_TENENGRAD_ABS = 100.0
FLAT_NOISE_MULT = 2.0

#: Immerkaer overestimates sigma on edges/texture; like Tai & Yang (2008) the
#: pixels with the strongest gradient are excluded. This fraction is dropped.
NOISE_EDGE_EXCLUDE = 0.10

#: Feature the headline ``score`` is taken from (see module docstring).
HEADLINE_FEATURE = "lap_nc_max"

REGION_ORDER = ("faces", "subjects", "frame")


# --------------------------------------------------------------------------
# Pure array operators (float32, 'valid' convolution -> (H-2, W-2))
# --------------------------------------------------------------------------

def laplacian(g: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian (cv2.Laplacian ksize=1), valid region only."""
    g = g.astype(np.float32, copy=False)
    return (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
            - 4.0 * g[1:-1, 1:-1])


def laplacian_variance(g: np.ndarray) -> float:
    """Variance of the Laplacian of a grayscale crop -- the classic focus
    measure, as ``rank_measure`` computes it with cv2. Small pure helper so
    callers without cv2 get the same number."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    return float(laplacian(g).var())


def tenengrad(g: np.ndarray) -> np.ndarray:
    """Per-pixel Sobel gradient energy gx^2 + gy^2, valid region only."""
    g = g.astype(np.float32, copy=False)
    gx = ((g[:-2, 2:] + 2.0 * g[1:-1, 2:] + g[2:, 2:])
          - (g[:-2, :-2] + 2.0 * g[1:-1, :-2] + g[2:, :-2]))
    gy = ((g[2:, :-2] + 2.0 * g[2:, 1:-1] + g[2:, 2:])
          - (g[:-2, :-2] + 2.0 * g[:-2, 1:-1] + g[:-2, 2:]))
    np.multiply(gx, gx, out=gx)
    np.multiply(gy, gy, out=gy)
    gx += gy
    return gx


def immerkaer_abs(g: np.ndarray) -> np.ndarray:
    """|I * N| for Immerkaer's mask N = [[1,-2,1],[-2,4,-2],[1,-2,1]]
    (separable: [1,-2,1] outer [1,-2,1]), valid region only."""
    g = g.astype(np.float32, copy=False)
    r = g[:, :-2] - 2.0 * g[:, 1:-1] + g[:, 2:]
    c = r[:-2] - 2.0 * r[1:-1] + r[2:]
    return np.abs(c, out=c)


_IMMERKAER_C = math.sqrt(math.pi / 2.0) / 6.0


def noise_sigma(g: np.ndarray, ten: Optional[np.ndarray] = None,
                exclude: float = NOISE_EDGE_EXCLUDE) -> tuple[float, float]:
    """Immerkaer (1996) fast noise sigma. Returns ``(masked, raw)``: raw is
    the textbook whole-image estimate; masked drops the ``exclude`` fraction of
    pixels with the strongest Sobel gradient (edges/texture inflate raw)."""
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0, 0.0
    a = immerkaer_abs(g)
    raw = _IMMERKAER_C * float(a.mean())
    if ten is None:
        ten = tenengrad(g)
    if exclude > 0 and ten.size > 16:
        # Subsample for the percentile: the cut only needs to be approximate.
        step = max(1, ten.size // 200_000)
        cut = float(np.percentile(ten.ravel()[::step], 100.0 * (1.0 - exclude)))
        mask = ten <= cut
        masked = _IMMERKAER_C * float(a[mask].mean()) if mask.any() else raw
    else:
        masked = raw
    return masked, raw


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------

def _default_open(path):
    from PIL import Image
    try:  # HEIC -- same optional opener the indexer registers
        import pillow_heif  # noqa: F401
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    return Image.open(path)


def _exif_orientation(im) -> int:
    try:
        return int(im.getexif().get(0x0112, 1) or 1)
    except Exception:
        return 1


def decode_normalised(path, *, open_image: Callable = _default_open,
                      long_edge: int = NORM_LONG_EDGE):
    """One decode -> (gray float32 array at exactly ``long_edge``, info).

    JPEG uses ``Image.draft('L', ...)`` so libjpeg decodes straight to
    grayscale at the smallest DCT scale (1/2, 1/4, 1/8) that is still >= the
    target -- the N100 never materialises a 60 MP RGB frame. Everything else
    (HEIC, PNG, TIFF) is a full decode. Then EXIF transpose (the space face
    and subject boxes live in), grayscale, and a LANCZOS resize to exactly
    ``long_edge``.
    """
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None
    im0 = open_image(path)
    try:
        raw_w, raw_h = im0.size
        orient = _exif_orientation(im0)
        ow, oh = (raw_h, raw_w) if orient in (5, 6, 7, 8) else (raw_w, raw_h)
        s = long_edge / float(max(raw_w, raw_h))
        decode = "full"
        # Sony/camera JPEGs with an embedded preview open as "MPO".
        if getattr(im0, "format", None) in ("JPEG", "MPO") and s < 1.0:
            want = (max(1, int(math.ceil(raw_w * s))), max(1, int(math.ceil(raw_h * s))))
            if im0.draft("L", want) is not None:
                decode = "draft"
        im = ImageOps.exif_transpose(im0)
        if im.mode != "L":
            im = im.convert("L")
        if ow >= oh:
            nw, nh = long_edge, max(1, int(round(oh * long_edge / ow)))
        else:
            nw, nh = max(1, int(round(ow * long_edge / oh))), long_edge
        if im.size != (nw, nh):
            im = im.resize((nw, nh), Image.LANCZOS)
        g = np.asarray(im, dtype=np.float32)
    finally:
        try:
            im0.close()
        except Exception:
            pass
    info = {"orig_size": [ow, oh], "norm_size": [nw, nh],
            "scale": round(nw / float(ow), 6), "upsampled": nw > ow,
            "decode": decode, "orientation": orient}
    return g, info


# --------------------------------------------------------------------------
# Regions and tiles
# --------------------------------------------------------------------------

def _splits(lo: int, hi: int, tile: int) -> list[tuple[int, int]]:
    n = max(1, (hi - lo) // tile)
    edges = [lo + round(i * (hi - lo) / n) for i in range(n + 1)]
    return [(edges[i], edges[i + 1]) for i in range(n) if edges[i + 1] > edges[i]]


def _tile_boxes(box, tile: int = TILE):
    x0, y0, x1, y1 = box
    for ya, yb in _splits(y0, y1, tile):
        for xa, xb in _splits(x0, x1, tile):
            yield xa, ya, xb, yb


def _clip_box(box, W: int, H: int):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    # The derivative maps are 'valid' (1 px border lost), so keep 1 px in.
    x0, y0 = max(1, x0), max(1, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    if x1 - x0 < MIN_REGION_EDGE or y1 - y0 < MIN_REGION_EDGE:
        return None
    return x0, y0, x1, y1


def _face_boxes_norm(face_boxes, orig_size, norm_size):
    """Faces as dicts with bbox_top/right/bottom/left (the DB columns) or
    ``(left, top, right, bottom)`` tuples, in ORIGINAL oriented pixels."""
    ow, oh = orig_size
    nw, nh = norm_size
    sx, sy = nw / float(ow), nh / float(oh)
    out, small = [], 0
    for f in face_boxes or ():
        if isinstance(f, dict):
            l, t = f.get("bbox_left"), f.get("bbox_top")
            r, b = f.get("bbox_right"), f.get("bbox_bottom")
        else:
            l, t, r, b = f
        if None in (l, t, r, b):
            continue
        box = (float(l) * sx, float(t) * sy, float(r) * sx, float(b) * sy)
        if min(box[2] - box[0], box[3] - box[1]) < MIN_FACE_EDGE:
            small += 1
            continue
        out.append(box)
    return out, small


def _subject_boxes_norm(subject_boxes, norm_size):
    nw, nh = norm_size
    out = []
    for s in subject_boxes or ():
        bb = s.get("bbox") if isinstance(s, dict) else s
        if not bb or len(bb) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bb)
        except (TypeError, ValueError):
            continue
        out.append((x0 * nw, y0 * nh, x1 * nw, y1 * nh))
    return out


def _pct(xs: np.ndarray, p: float) -> float:
    return float(np.percentile(xs, p)) if xs.size else float("nan")


def _region_stats(boxes, lap, ten, sigma: float, flat_thresh: float,
                  W: int, H: int, skipped_boxes: int = 0) -> dict:
    """Tile every box, skip flat tiles, summarise the rest.

    ``lap``/``ten`` are full-frame maps in 'valid' coordinates, i.e. map[i, j]
    is pixel (i+1, j+1). Overlapping boxes contribute their tiles twice; boxes
    are rarely nested and the max is unaffected.
    """
    laps, tens, flat = [], [], 0
    measured = 0
    for box in boxes:
        cb = _clip_box(box, W, H)
        if cb is None:
            skipped_boxes += 1
            continue
        measured += 1
        for xa, ya, xb, yb in _tile_boxes(cb):
            tt = ten[ya - 1:yb - 1, xa - 1:xb - 1]
            tmean = float(tt.mean())
            if tmean < flat_thresh:
                flat += 1
                continue
            laps.append(float(lap[ya - 1:yb - 1, xa - 1:xb - 1].var()))
            tens.append(tmean)
    st = {"boxes": measured, "boxes_skipped": skipped_boxes,
          "tiles": len(laps) + flat, "flat_skipped": flat}
    if not laps:
        for k in ("lap_max", "lap_p95", "lap_median", "best_median_ratio",
                  "ten_max", "ten_p95", "lap_nc_max", "lap_nc_p95",
                  "lap_nc_median", "ten_nc_max"):
            st[k] = None
        return st
    L, T = np.asarray(laps), np.asarray(tens)
    noise_lap = NOISE_K * sigma * sigma
    noise_ten = TENENGRAD_NOISE_K * sigma * sigma
    lmax, lp95, lmed = float(L.max()), _pct(L, 95), _pct(L, 50)
    st.update({
        "lap_max": lmax, "lap_p95": lp95, "lap_median": lmed,
        "best_median_ratio": (lmax / lmed) if lmed > 0 else None,
        "ten_max": float(T.max()), "ten_p95": _pct(T, 95),
        "lap_nc_max": lmax - noise_lap, "lap_nc_p95": lp95 - noise_lap,
        "lap_nc_median": lmed - noise_lap,
        "ten_nc_max": float(T.max()) - noise_ten,
    })
    return st


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def measure_array(g: np.ndarray, face_boxes_norm: Iterable = (),
                  subject_boxes_norm: Iterable = (), *,
                  faces_skipped_small: int = 0) -> dict:
    """Measure an already-normalised grayscale array. Boxes are in THIS
    array's pixel coords as ``(x0, y0, x1, y1)``. Returns the ``detail`` body
    (without decode info) and the headline ``score``."""
    H, W = g.shape
    lap = laplacian(g)
    ten = tenengrad(g)
    sigma, sigma_raw = noise_sigma(g, ten)
    flat_thresh = max(FLAT_TENENGRAD_ABS,
                      FLAT_NOISE_MULT * TENENGRAD_NOISE_K * sigma * sigma)
    faces = list(face_boxes_norm or ())
    subjects = list(subject_boxes_norm or ())
    regions = {}
    if faces or faces_skipped_small:
        regions["faces"] = _region_stats(faces, lap, ten, sigma, flat_thresh,
                                         W, H, skipped_boxes=faces_skipped_small)
    if subjects:
        regions["subjects"] = _region_stats(subjects, lap, ten, sigma,
                                            flat_thresh, W, H)
    regions["frame"] = _region_stats([(0, 0, W, H)], lap, ten, sigma,
                                     flat_thresh, W, H)
    source = next((r for r in REGION_ORDER
                   if r in regions and regions[r]["lap_max"] is not None), None)
    score = None
    if source is not None:
        score = max(0.0, float(regions[source][HEADLINE_FEATURE]))
    detail = {"noise_sigma": sigma, "noise_sigma_raw": sigma_raw,
              "noise_k": NOISE_K, "flat_threshold": flat_thresh,
              "source": source, "headline": HEADLINE_FEATURE,
              "regions": regions}
    return {"score": score, "detail": detail}


def measure_photo(path, face_boxes=None, subject_boxes=None, *,
                  open_image: Callable = _default_open,
                  long_edge: int = NORM_LONG_EDGE) -> dict:
    """Measure one photo from its original file. NEVER raises for a bad file.

    ``face_boxes``: iterable of dicts with the ``faces`` table's
    ``bbox_left/top/right/bottom`` (original, EXIF-oriented pixels) or
    ``(left, top, right, bottom)`` tuples. ``subject_boxes``: the parsed
    ``photos.subject_boxes`` list (dicts with a normalised ``bbox``) or bare
    ``[x0, y0, x1, y1]`` lists in 0-1. ``open_image(path) -> PIL.Image`` is
    injectable so tests need no files. ``long_edge`` exists for tests; every
    stored measurement uses ``NORM_LONG_EDGE`` (changing it changes the scale
    of every feature, so it would need a ``SHARPNESS_VERSION`` bump).

    Returns ``{"score": float|None, "detail": dict, "version": int}``; on a
    decode failure ``{"score": None, "detail": {"error": str}, "version": ...}``
    -- stored WITH the version, so a backfill never retries it forever.
    """
    try:
        g, info = decode_normalised(path, open_image=open_image,
                                    long_edge=long_edge)
    except Exception as exc:  # unreadable/truncated/unsupported -> recorded
        return {"score": None,
                "detail": {"error": f"{type(exc).__name__}: {exc}"},
                "version": SHARPNESS_VERSION}
    try:
        faces, small = _face_boxes_norm(face_boxes, info["orig_size"],
                                        info["norm_size"])
        subjects = _subject_boxes_norm(subject_boxes, info["norm_size"])
        out = measure_array(g, faces, subjects, faces_skipped_small=small)
    except Exception as exc:  # malformed boxes etc. must not kill a backfill
        return {"score": None,
                "detail": {"error": f"{type(exc).__name__}: {exc}", **info},
                "version": SHARPNESS_VERSION}
    detail = {**info, **out["detail"]}
    return {"score": out["score"], "detail": detail,
            "version": SHARPNESS_VERSION}


#: Region-level scalar features (the eval's candidate metrics).
REGION_FEATURES = ("lap_max", "lap_p95", "lap_median", "best_median_ratio",
                   "ten_max", "ten_p95", "lap_nc_max", "lap_nc_p95",
                   "lap_nc_median", "ten_nc_max")


def flatten_features(result: dict) -> dict:
    """``{metric_name: float|None}`` for every candidate scalar in a result:
    ``score``, ``noise_sigma``, ``noise_sigma_raw``, ``<region>.<feature>``
    for faces/subjects/frame, and ``best.<feature>`` -- the headline region's
    value, i.e. the fallback chain the score itself uses."""
    d = result.get("detail") or {}
    out = {"score": result.get("score")}
    if "error" in d:
        return out
    out["noise_sigma"] = d.get("noise_sigma")
    out["noise_sigma_raw"] = d.get("noise_sigma_raw")
    regions = d.get("regions") or {}
    src = d.get("source")
    for r in REGION_ORDER:
        st = regions.get(r) or {}
        for f in REGION_FEATURES:
            out[f"{r}.{f}"] = st.get(f)
    for f in REGION_FEATURES:
        out[f"best.{f}"] = (regions.get(src) or {}).get(f) if src else None
    return out

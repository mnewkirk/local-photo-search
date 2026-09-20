"""Native-resolution face-sharpness measurement — the expensive half of the
per-shoot "best of" ranking.

This is the `measure()` that `scripts/rank_shoot.py` used to carry inline, and
the script now imports it from here. **There is exactly one implementation**,
for two reasons:

- `scripts/` is **not copied into the Docker image** (see the Dockerfile's
  `COPY` lines — `photosearch/`, `frontend/`, `cli.py`, `requirements.txt`,
  `docker-entrypoint.sh`, and nothing else). The script only ever ran on the
  NAS because the repo is bind-mounted there; anything the container has to
  run has to live in the package.
- The `rank_measure` step of a batch advance (`batch_advance._run_rank_measure`)
  runs it, and the owner then runs the script's selection phase against the
  cache it wrote. A second copy would drift and the two halves would disagree
  about the cache's shape.

**This runs on the NAS, not the desktop.** It decodes each photo at full
native resolution, and only the NAS holds the originals — the desktop replica
has the DB and the thumbnails, never the files. Measured on the first real
batch (2026-09-19): 1,260 photos with faces, ~10 minutes on the N100.

WHY NATIVE RESOLUTION

What separates frames in a sports burst is whether the face is in focus, and
that is high-frequency detail. A downscaled preview or a cached 200 px crop
has already thrown it away — which is precisely the signal. Hence one
full-resolution decode per photo, which is the expensive part and is therefore
cached and resumable.

THE CACHE

``{"<photo_id>": {"<face_id>": {"lap": float, "edge": int, "area_frac": float}}}``

One JSON file per shoot DATE, because the selection phase
(`rank_shoot.select`) scopes by date. A photo already in the cache is never
re-measured, so a re-run after a late-arriving photo costs only the new
photos. That format is frozen: the owner has existing cache files.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

# The shorter bbox edge, in pixels, below which a face is not worth measuring:
# a tiny crop's Laplacian variance says more about JPEG noise than focus.
DEFAULT_MIN_EDGE = 60

# SQLite's variable cap, mirroring batch_state._ID_CHUNK. A batch is typically
# 1-3k photos but a card dump can be far larger.
_ID_CHUNK = 20000

_CACHE_PREFIX = "rank_shoot_"


def _log(msg) -> None:
    print(msg, flush=True)


def default_cache_path(db_path: str, date_: str) -> str:
    """Where this shoot's sharpness cache lives: **beside the DB**.

    The script used to hardcode ``/data/rank_shoot_<date>.json``. On the NAS
    the DB *is* ``/data/photo_index.db``, so this rule resolves to exactly the
    same filename and the owner's existing caches are found, not stranded —
    while off the NAS (the replica, a dev checkout) it no longer points at a
    ``/data`` that does not exist, and in the container it can never land in
    the ephemeral ``/app``.
    """
    return os.path.join(os.path.dirname(os.path.abspath(db_path)),
                        f"{_CACHE_PREFIX}{date_}.json")


def _chunks(ids: list):
    for i in range(0, len(ids), _ID_CHUNK):
        yield ids[i:i + _ID_CHUNK]


_FACE_SELECT = """SELECT f.id AS face_id, f.photo_id, f.bbox_top, f.bbox_bottom,
                         f.bbox_left, f.bbox_right, p.filepath
                    FROM faces f JOIN photos p ON p.id = f.photo_id"""


def _face_rows(db, date_, photo_ids):
    """Every face row in scope, ordered by photo so one decode serves them all.

    ``photo_ids=[]`` means an EMPTY scope, never "no scope given" — the same
    trap `_run_stacking` guards, where an empty list falling through to a
    whole-library query has twice done real damage.
    """
    if photo_ids is not None:
        rows = []
        for chunk in _chunks(list(photo_ids)):
            ph = ",".join("?" * len(chunk))
            rows.extend(db.conn.execute(
                f"{_FACE_SELECT} WHERE f.photo_id IN ({ph}) "
                f"ORDER BY f.photo_id, f.id", list(chunk)).fetchall())
        return rows
    return db.conn.execute(
        f"{_FACE_SELECT} WHERE date(p.date_taken) = ? ORDER BY f.photo_id, f.id",
        (date_,)).fetchall()


def _measure_photo(path: str, faces: list[dict], min_edge: int) -> dict:
    """Laplacian variance per face for ONE photo, from the original pixels.

    One decode per PHOTO, not per face: a 60 MP JPEG costs seconds on the N100
    and a frame can hold a dozen faces. Crops are taken from the
    EXIF-**oriented** image because that is the space ``faces.bbox_*`` was
    computed in (CLAUDE.md, "EXIF-oriented image dimensions") — using the raw
    orientation would sample the wrong region on ~18% of the library.

    Injectable (`measure(..., measure_photo=...)`) so the rest of this module
    is testable without PIL, cv2 or a real file.
    """
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    Image.MAX_IMAGE_PIXELS = None

    out = {}
    with Image.open(path) as im0:
        im = ImageOps.exif_transpose(im0)
        W, H = im.size
        for f in faces:
            t, b = int(f["bbox_top"] or 0), int(f["bbox_bottom"] or 0)
            l, r = int(f["bbox_left"] or 0), int(f["bbox_right"] or 0)
            if min(b - t, r - l) < min_edge:
                continue
            box = (max(0, l), max(0, t), min(W, r), min(H, b))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            # Crop BEFORE converting: a 60 MP grayscale copy of the whole
            # frame would be ~60 MB per photo for no reason.
            g = np.asarray(im.crop(box).convert("L"))
            out[str(f["face_id"])] = {
                "lap": float(cv2.Laplacian(g, cv2.CV_64F).var()),
                "edge": int(min(b - t, r - l)),
                "area_frac": float(((b - t) * (r - l)) / max(1, W * H)),
            }
    return out


def measure(db, date_, cache_path=None, min_edge=DEFAULT_MIN_EDGE, *,
            photo_ids=None, log=_log, on_progress=None, measure_photo=None):
    """Measure every unmeasured photo in scope. Cached, resumable, idempotent.

    ``date_`` picks the cache file and, when ``photo_ids`` is None, the scope.
    The batch runner passes ``photo_ids`` (one dated FOLDER — two folders can
    share a day) while still writing the DATE's cache, because that is the file
    `rank_shoot.py --date D` reads next.

    Returns a small summary dict, not the cache: a runner's result is echoed
    onto the SSE stream, and a whole shoot's measurements do not belong there.
    """
    if cache_path is None:
        cache_path = default_cache_path(db.db_path, date_)
    if measure_photo is None:
        measure_photo = _measure_photo

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            cache = json.load(fh)
        log(f"resuming from {cache_path}: {len(cache)} photos already measured")

    by_photo = defaultdict(list)
    for r in _face_rows(db, date_, photo_ids):
        by_photo[r["photo_id"]].append(dict(r))
    todo = [pid for pid in by_photo if str(pid) not in cache]
    log(f"{len(by_photo)} photos with faces on {date_}; "
        f"{len(todo)} left to measure")

    def _save():
        with open(cache_path, "w") as fh:
            json.dump(cache, fh)

    faces_measured = 0
    for n, pid in enumerate(todo, 1):
        faces = by_photo[pid]
        path = db.resolve_filepath(faces[0]["filepath"])
        out = {}
        try:
            out = measure_photo(path, faces, min_edge)
        except Exception as exc:              # unreadable -> empty, not fatal
            log(f"  ! photo {pid}: {exc}")
        faces_measured += len(out)
        cache[str(pid)] = out
        if n % 25 == 0 or n == len(todo):
            _save()
            log(f"  measured {n}/{len(todo)} photos")
            if on_progress:
                on_progress({"done": n, "total": len(todo),
                             "message": f"measured {n}/{len(todo)} photos"})
    _save()
    return {"photos": len(by_photo), "measured": len(todo),
            "faces": faces_measured, "cache_path": cache_path}

"""Face-crop rendering — the single source of truth for turning a face bbox
into a square JPEG thumbnail.

Shared by the `/api/faces/crop/{id}` web endpoint (one size per request) and the
`warm-face-crops` CLI (many sizes per face, generated ahead of browsing so the
local read-replica's crop cache is warm — see docs/plans/local-replica-and-writes.md).

The dominant cost is decoding the source image, not the resize. So `render_face_crops`
decodes **once** — at the coarsest libjpeg scale that still satisfies the *largest*
requested size — and emits every requested size from that single decode. Producing
the 200px crop alongside the 120px one is therefore essentially free.
"""

import io
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_CROP_SIZES = (120, 200)


def face_crop_cache_dir(db_path: str) -> str:
    """The on-disk crop cache lives at ``<db parent>/thumbnails/face_crops``.

    Mirrors web.py:_ensure_face_crop_dir so the endpoint and the warm CLI write
    to the same place (and the replica rsync mirrors that one directory).
    """
    return str(Path(db_path).resolve().parent / "thumbnails" / "face_crops")


def crop_cache_path(cache_dir: str, face_id: int, size: int) -> str:
    return os.path.join(cache_dir, f"{face_id}_{size}.jpg")


def render_face_crops(filepath, bbox, image_width, image_height, sizes):
    """Decode ``filepath`` once and return ``{size: jpeg_bytes}`` for each size.

    ``bbox`` is ``(top, right, bottom, left)`` in EXIF-oriented pixel coords
    (the same convention the faces table stores; see CLAUDE.md "EXIF-oriented
    image dimensions"). ``image_width``/``image_height`` are the oriented dims,
    used to pick the libjpeg ``draft()`` decode scale.
    """
    top, right, bottom, left = bbox
    if top is None:
        raise ValueError("face has no bounding box")
    sizes = sorted({int(s) for s in sizes})
    if not sizes:
        return {}
    max_size = sizes[-1]

    img = Image.open(filepath)
    # Decode at a reduced scale when the source dwarfs the crop we need. libjpeg
    # can decode at 1/2, 1/4, 1/8 via draft(); that full-res decode is the
    # dominant cost on the N100. Pick the coarsest scale that still leaves the
    # face >= max_size px so there's no quality loss vs a full decode. Non-JPEG
    # (PNG/HEIC) ignore draft() and stay at scale 1.0.
    raw_w0 = img.size[0]
    face_w0, face_h0 = (right - left), (bottom - top)
    if image_width and image_height and face_w0 > 0 and face_h0 > 0:
        frac = min(face_w0 / image_width, face_h0 / image_height)
        if frac > 0:
            need = int(max_size / frac) + 1
            img.draft("RGB", (need, need))
    scale = img.size[0] / raw_w0  # < 1.0 only if draft downscaled
    img = ImageOps.exif_transpose(img)
    img_w, img_h = img.size
    if scale != 1.0:
        top, right = int(top * scale), int(right * scale)
        bottom, left = int(bottom * scale), int(left * scale)

    # 20% padding around the face for tight framing. The crop box is the same
    # for every output size — only the final resize differs.
    face_w = right - left
    face_h = bottom - top
    pad_x = int(face_w * 0.2)
    pad_y = int(face_h * 0.2)
    crop_left = max(0, left - pad_x)
    crop_top = max(0, top - pad_y)
    crop_right = min(img_w, right + pad_x)
    crop_bottom = min(img_h, bottom + pad_y)

    # Make it square (expand the shorter dimension, centered).
    cw = crop_right - crop_left
    ch = crop_bottom - crop_top
    if cw > ch:
        diff = cw - ch
        crop_top = max(0, crop_top - diff // 2)
        crop_bottom = crop_top + cw
        if crop_bottom > img_h:
            crop_bottom = img_h
            crop_top = max(0, crop_bottom - cw)
    elif ch > cw:
        diff = ch - cw
        crop_left = max(0, crop_left - diff // 2)
        crop_right = crop_left + ch
        if crop_right > img_w:
            crop_right = img_w
            crop_left = max(0, crop_right - ch)

    box = img.crop((crop_left, crop_top, crop_right, crop_bottom))

    out = {}
    for size in sizes:
        face_img = box.resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        face_img.convert("RGB").save(buf, format="JPEG", quality=85)
        out[size] = buf.getvalue()
    return out


def write_crop_atomic(path: str, data: bytes) -> None:
    """Write JPEG bytes to a cache path atomically (tmp + rename)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# bulk warming
# ---------------------------------------------------------------------------

def _scoped_face_rows(db, *, photo_ids=None, person_ids=None, matched_only=False,
                      date_from=None, date_to=None, id_chunk: int = 20000):
    """Faces in scope, with everything ``render_face_crops`` needs.

    Every scope ANDs onto the others. ``photo_ids`` is chunked under SQLite's
    variable cap — a batch is usually 1-3k photos but a card dump can be far
    larger. An EMPTY ``photo_ids`` list means an empty scope, NOT the whole
    library: a batch's membership is derived live, so it really can empty out
    (a dedup prune, a retime), and falling through to the library there would
    warm every crop on the box.
    """
    select = ("SELECT f.id, f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left, "
              "       ph.filepath, ph.image_width, ph.image_height "
              "FROM faces f JOIN photos ph ON ph.id = f.photo_id "
              "WHERE f.bbox_top IS NOT NULL")
    tail, tail_params = "", []
    if person_ids is not None:
        tail += f" AND f.person_id IN ({','.join('?' * len(person_ids))})"
        tail_params += list(person_ids)
    elif matched_only:
        tail += " AND f.person_id IS NOT NULL"
    if date_from:
        tail += " AND date(ph.date_taken) >= ?"
        tail_params.append(date_from)
    if date_to:
        tail += " AND date(ph.date_taken) <= ?"
        tail_params.append(date_to)

    if photo_ids is None:
        return db.conn.execute(select + tail, tail_params).fetchall()

    rows = []
    for i in range(0, len(photo_ids), id_chunk):
        chunk = photo_ids[i:i + id_chunk]
        rows.extend(db.conn.execute(
            select + f" AND f.photo_id IN ({','.join('?' * len(chunk))})" + tail,
            list(chunk) + tail_params,
        ).fetchall())
    return rows


def warm_crops(db, *, photo_ids=None, person_ids=None, matched_only=False,
               date_from=None, date_to=None, sizes=DEFAULT_CROP_SIZES,
               workers: int = 3, force: bool = False, nas_url=None,
               cache_dir=None, on_progress=None) -> dict:
    """Pre-generate face-crop JPEGs for a scope. Returns a summary dict.

    Face crops are rendered on first view — a full image decode, ~2s on the
    N100 — so a cold grid dribbles in over minutes. This is the bulk version,
    shared by the ``warm-face-crops`` CLI and the batch-advance ``warm_crops``
    step so there is exactly one implementation of the scope rules.

    Scope is ANDed: ``photo_ids`` (the batch case), ``person_ids`` or
    ``matched_only`` (the person case), and a ``date_from``/``date_to`` range.
    With none of them this warms EVERY face, which is hours on the N100.

    ``nas_url`` is the replica fallback: the desktop holds no originals, so a
    missing local file is fetched from the NAS's ``/api/faces/crop`` instead.
    """
    import time

    size_list = sorted({int(s) for s in sizes})
    if not size_list:
        raise ValueError("sizes must list at least one size")
    cache_dir = cache_dir or face_crop_cache_dir(db.db_path)
    os.makedirs(cache_dir, exist_ok=True)

    rows = _scoped_face_rows(db, photo_ids=photo_ids, person_ids=person_ids,
                             matched_only=matched_only, date_from=date_from,
                             date_to=date_to)

    # Resolve paths up front: DB access is single-threaded, the workers are
    # pure CPU/IO.
    tasks = [(r, db.resolve_filepath(r["filepath"])) for r in rows]
    total_found = len(tasks)

    skipped = 0
    if not force:
        pending = [(r, fp) for r, fp in tasks
                   if not all(os.path.exists(crop_cache_path(cache_dir, r["id"], s))
                              for s in size_list)]
        skipped = len(tasks) - len(pending)
        tasks = pending

    def emit(event):
        if on_progress:
            try:
                on_progress(event)
            except Exception:  # never let a progress sink kill the job
                pass

    emit({"phase": "warm_crops", "status": "scanning", "found": total_found,
          "cached": skipped, "total": len(tasks), "sizes": size_list})

    summary = {"found": total_found, "cached": skipped, "total": len(tasks),
               "ok": 0, "missing": 0, "errors": 0, "error_samples": [],
               "sizes": size_list, "elapsed": 0.0}
    if not tasks:
        return summary

    nas = nas_url.rstrip("/") if nas_url else None

    def warm_one(item):
        r, fp = item
        try:
            if fp and os.path.exists(fp):
                bbox = (r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"])
                crops = render_face_crops(fp, bbox, r["image_width"],
                                          r["image_height"], size_list)
                for s in size_list:
                    write_crop_atomic(crop_cache_path(cache_dir, r["id"], s), crops[s])
            elif nas:
                for s in size_list:
                    url = f"{nas}/api/faces/crop/{r['id']}?size={s}"
                    with urllib.request.urlopen(url, timeout=40) as resp:
                        write_crop_atomic(crop_cache_path(cache_dir, r["id"], s),
                                          resp.read())
            else:
                return ("missing", r["id"])
            return ("ok", r["id"])
        except Exception as e:  # noqa: BLE001 — one bad photo can't kill the run
            return ("error", f"{r['id']}: {e}")

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(warm_one, t) for t in tasks]
        for fut in as_completed(futures):
            status, info = fut.result()
            done += 1
            if status == "ok":
                summary["ok"] += 1
            elif status == "missing":
                summary["missing"] += 1
            else:
                summary["errors"] += 1
                if len(summary["error_samples"]) < 5:
                    summary["error_samples"].append(info)
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - started
                emit({"phase": "warm_crops", "status": "running", "done": done,
                      "total": len(tasks), "ok": summary["ok"],
                      "missing": summary["missing"], "errors": summary["errors"],
                      "rate": (done / elapsed) if elapsed else 0.0})

    summary["elapsed"] = round(time.time() - started, 2)
    emit({"phase": "warm_crops", "status": "done", **{
        k: summary[k] for k in ("ok", "missing", "errors", "total")}})
    return summary

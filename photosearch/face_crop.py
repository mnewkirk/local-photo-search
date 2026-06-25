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
from pathlib import Path

from PIL import Image, ImageOps


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

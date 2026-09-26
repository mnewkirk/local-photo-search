#!/usr/bin/env python
"""Step 0 of docs/plans/sharpness-measurement.md: do face and subject boxes
land on the subject for EXIF-ROTATED photos?

`sharpness.measure_photo` crops face boxes (`faces.bbox_*`, pixels in the
EXIF-oriented original) and subject boxes (`photos.subject_boxes`, 0-1 of the
oriented frame) out of the EXIF-transposed image. If either was really stored
in RAW sensor orientation, ~18% of the library (orientation 5-8) would be
measured in the wrong place. This draws both onto the oriented image so a
human can look.

    python evals/sharpness_orientation_check.py --db photo_index.db.local \\
        --base-url http://localhost:8001 -n 10 --out /tmp/orient-check

Read-only: the DB is opened `mode=ro`, pixels come from `/api/photos/{id}/full`
(the original bytes, un-rotated -- the orientation tag is still in them).
Faces are drawn RED, subject boxes CYAN; each tile is captioned with the id,
camera and orientation. Output: one JPEG per photo plus `contact_sheet.jpg`.
"""
import argparse
import io
import json
import os
import random
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from photosearch.model_eval import open_db_readonly  # noqa: E402

DEFAULT_SERVER = "http://localhost:8001"


def _candidates(conn, limit, seed):
    """Portrait-stored photos (image dims are stored ORIENTED, so a rotated
    camera frame reads taller than wide) that have faces or subject boxes.
    Orientation itself is not in the DB -- it is read from the fetched bytes."""
    rows = conn.execute(
        """SELECT p.id, p.camera_model, p.image_width, p.image_height,
                  p.subject_boxes,
                  (SELECT COUNT(*) FROM faces f WHERE f.photo_id = p.id) AS nf
             FROM photos p
            WHERE p.image_height > p.image_width
              AND (EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id)
                   OR (p.subject_boxes IS NOT NULL AND p.subject_boxes != '[]'))
            ORDER BY (p.id * 2654435761 + ?) % 1000003 LIMIT ?""",
        (seed, limit)).fetchall()
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return rows


def _faces(conn, pid):
    return [dict(r) for r in conn.execute(
        "SELECT bbox_left, bbox_top, bbox_right, bbox_bottom FROM faces "
        "WHERE photo_id = ?", (pid,))]


def _fetch(server, pid, timeout=120):
    url = f"{server.rstrip('/')}/api/photos/{int(pid)}/full"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _open(data):
    from PIL import Image
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except Exception:
        pass
    Image.MAX_IMAGE_PIXELS = None
    return Image.open(io.BytesIO(data))


def draw(im0, faces, subjects, caption, long_edge=900):
    """Oriented, downscaled copy with face (red) + subject (cyan) boxes."""
    from PIL import ImageDraw, ImageOps
    im = ImageOps.exif_transpose(im0).convert("RGB")
    ow, oh = im.size
    im.thumbnail((long_edge, long_edge))
    w, h = im.size
    sx, sy = w / ow, h / oh
    d = ImageDraw.Draw(im)
    lw = max(2, w // 250)
    for f in faces:
        d.rectangle([f["bbox_left"] * sx, f["bbox_top"] * sy,
                     f["bbox_right"] * sx, f["bbox_bottom"] * sy],
                    outline=(255, 0, 0), width=lw)
    for s in subjects:
        x0, y0, x1, y1 = s["bbox"]
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(0, 255, 255), width=lw)
        if s.get("label"):
            d.text((x0 * w + 4, y0 * h + 2), str(s["label"]), fill=(0, 255, 255))
    d.rectangle([0, 0, w, 16], fill=(0, 0, 0))
    d.text((4, 2), caption, fill=(255, 255, 255))
    return im


def contact_sheet(images, cols=5, cell=360):
    from PIL import Image
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, max(1, rows) * cell), (30, 30, 30))
    for i, im in enumerate(images):
        t = im.copy()
        t.thumbnail((cell - 8, cell - 8))
        x = (i % cols) * cell + (cell - t.size[0]) // 2
        y = (i // cols) * cell + (cell - t.size[1]) // 2
        sheet.paste(t, (x, y))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    ap.add_argument("--base-url", default=DEFAULT_SERVER)
    ap.add_argument("-n", type=int, default=10, help="rotated photos to draw")
    ap.add_argument("--scan", type=int, default=400,
                    help="portrait candidates to consider (fetched until -n hit)")
    ap.add_argument("--max-fetch", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="./evals/sharpness-orientation")
    a = ap.parse_args(argv)

    conn = open_db_readonly(a.db)
    os.makedirs(a.out, exist_ok=True)
    kept, fetched, skipped = [], 0, {}
    for row in _candidates(conn, a.scan, a.seed):
        if len(kept) >= a.n or fetched >= a.max_fetch:
            break
        pid = row["id"]
        try:
            data = _fetch(a.base_url, pid)
            fetched += 1
            im0 = _open(data)
            orient = int(im0.getexif().get(0x0112, 1) or 1)
        except Exception as exc:
            skipped["fetch/open error"] = skipped.get("fetch/open error", 0) + 1
            print(f"  ! {pid}: {exc}")
            continue
        if orient not in (5, 6, 7, 8):
            skipped[f"orientation {orient}"] = skipped.get(f"orientation {orient}", 0) + 1
            continue
        faces = _faces(conn, pid)
        try:
            subjects = json.loads(row["subject_boxes"] or "[]") or []
        except ValueError:
            subjects = []
        raw = im0.size
        cap = (f"{pid} {row['camera_model'] or '?'} orient={orient} raw={raw[0]}x{raw[1]} "
               f"db={row['image_width']}x{row['image_height']} faces={len(faces)} subj={len(subjects)}")
        im = draw(im0, faces, subjects, cap)
        path = os.path.join(a.out, f"{pid}.jpg")
        im.save(path, "JPEG", quality=85)
        kept.append((pid, path, im))
        print(f"  {cap} -> {path}")
    if kept:
        sheet = os.path.join(a.out, "contact_sheet.jpg")
        contact_sheet([k[2] for k in kept]).save(sheet, "JPEG", quality=85)
        print(f"contact sheet: {sheet}")
    print(f"drew {len(kept)} rotated photos; fetched {fetched}; skipped {skipped}")
    return 0 if kept else 1


if __name__ == "__main__":
    sys.exit(main())

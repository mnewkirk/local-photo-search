"""Manual face-box editing: re-detect, add-by-drag, and the IoU dedup that
keeps a re-detect from duplicating faces the corpus already has.

All three operations run where the originals + InsightFace + the authoritative
DB live (the NAS). The browser UI is NAS-served, so the frontend calls these
directly — the same place `/api/faces/{id}/assign` already writes. detect_faces
reads the image in its native (cv2) space, matching every existing face row, so
augment-mode IoU dedup compares like-for-like. add_manual_face instead works in
EXIF-oriented space (via ImageOps.exif_transpose) so the box the user dragged on
the preview lands exactly where they drew it, on rotated photos too.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

# Detections overlapping an existing box by at least this IoU are treated as the
# same face and skipped in augment mode (so a re-detect only adds genuinely-new
# faces and never clobbers a tagged one).
_DEDUP_IOU = 0.4

# Pad the drag box by this fraction of its size on each side before running
# InsightFace on the crop — ArcFace needs some head/hair context around the
# face to encode well, and users tend to draw tight to the face.
_CROP_PAD = 0.4


def _iou(a: tuple, b: tuple) -> float:
    """IoU of two (top, right, bottom, left) boxes."""
    at, ar, ab, al = a
    bt, br, bb, bl = b
    ix1, iy1 = max(al, bl), max(at, bt)
    ix2, iy2 = min(ar, br), min(ab, bb)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ar - al) * max(0, ab - at)
    area_b = max(0, br - bl) * max(0, bb - bt)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _norm_bbox(bbox) -> tuple:
    """Accept a normalized (0-1) box as a dict or a [t, r, b, l] list; return a
    clamped (top, right, bottom, left) tuple of floats."""
    if isinstance(bbox, dict):
        t, r, b, l = bbox["top"], bbox["right"], bbox["bottom"], bbox["left"]
    else:
        t, r, b, l = bbox
    t, r, b, l = float(t), float(r), float(b), float(l)
    # Normalize orientation of the box itself (drag can go any direction).
    top, bottom = min(t, b), max(t, b)
    left, right = min(l, r), max(l, r)
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(top), clamp(right), clamp(bottom), clamp(left)


def redetect_photo_faces(db, photo_id: int, image_path: str, *,
                         replace: bool = False,
                         iou_thresh: float = _DEDUP_IOU) -> dict:
    """Re-run InsightFace detection on one photo.

    ``replace`` deletes every existing face on the photo first (wipes manual
    assignments). Default augment mode keeps all existing faces and adds only
    detections that don't overlap one (IoU >= ``iou_thresh``), so tagged faces
    survive and re-detecting to recover a missed face never duplicates.
    """
    from .faces import detect_faces

    existing = db.conn.execute(
        "SELECT id, bbox_top, bbox_right, bbox_bottom, bbox_left "
        "FROM faces WHERE photo_id = ?", (photo_id,)).fetchall()
    detected = detect_faces(image_path)

    if replace:
        for r in existing:
            db.delete_face(r["id"])
        added = [db.add_face(photo_id, tuple(d["bbox"]), d["encoding"],
                             det_score=d.get("det_score")) for d in detected]
        db.conn.commit()
        return {"mode": "replace", "detected": len(detected),
                "added": len(added), "removed": len(existing), "kept": 0,
                "face_ids": added}

    ex_boxes = [(r["bbox_top"], r["bbox_right"], r["bbox_bottom"], r["bbox_left"])
                for r in existing if r["bbox_top"] is not None]
    added = []
    for d in detected:
        box = tuple(d["bbox"])
        if any(_iou(box, eb) >= iou_thresh for eb in ex_boxes):
            continue
        fid = db.add_face(photo_id, box, d["encoding"], det_score=d.get("det_score"))
        added.append(fid)
        ex_boxes.append(box)
    db.conn.commit()
    return {"mode": "augment", "detected": len(detected), "added": len(added),
            "kept": len(existing), "removed": 0, "face_ids": added}


def add_manual_face(db, photo_id: int, image_path: str, photo: dict, bbox,
                    person_name: Optional[str] = None) -> dict:
    """Add a face at a user-drawn box.

    ``bbox`` is normalized (0-1) in EXIF-oriented preview space. We crop that
    region (padded) from the oriented image and run InsightFace on the crop to
    recover an ArcFace encoding — so a face the full-frame detector missed can
    still be encoded and later matched/clustered. If nothing is found in the
    crop the box is still stored (no encoding, ``match_source='manual_box'``):
    assignable to a person by hand, just not auto-matchable.
    """
    from PIL import Image, ImageOps
    import numpy as np

    top, right, bottom, left = _norm_bbox(bbox)

    encoding: list = []
    det_score = None
    try:
        with Image.open(image_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            W, H = im.size
            # Padded crop around the drawn box, clamped to image bounds.
            bw, bh = (right - left) * W, (bottom - top) * H
            cx1 = max(0, int(left * W - bw * _CROP_PAD))
            cy1 = max(0, int(top * H - bh * _CROP_PAD))
            cx2 = min(W, int(right * W + bw * _CROP_PAD))
            cy2 = min(H, int(bottom * H + bh * _CROP_PAD))
            if cx2 - cx1 >= 8 and cy2 - cy1 >= 8:
                crop = im.crop((cx1, cy1, cx2, cy2))
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                    tmp = tf.name
                    crop.save(tmp, "JPEG", quality=95)
                try:
                    from .faces import detect_faces
                    dets = detect_faces(tmp)
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                if dets:
                    # Largest detection within the crop is the intended face.
                    best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0])
                               * (d["bbox"][1] - d["bbox"][3]))
                    encoding = best["encoding"]
                    det_score = best.get("det_score")
    except Exception as e:  # image unreadable / not present locally
        return {"ok": False, "error": f"could not read image: {e}"}

    # Store the box the user drew, in oriented pixel space (matches the overlay).
    iw = photo.get("image_width") or W
    ih = photo.get("image_height") or H
    px = (int(top * ih), int(right * iw), int(bottom * ih), int(left * iw))

    face_id = db.add_face(photo_id, px, encoding, det_score=det_score)

    person_id = None
    if person_name:
        person = db.get_person_by_name(person_name)
        person_id = person["id"] if person else db.add_person(person_name)
        db.assign_face_to_person(face_id, person_id, match_source="manual")
    else:
        # Distinguish a hand-drawn box from a detector hit so it's auditable.
        db.conn.execute("UPDATE faces SET match_source = 'manual_box' WHERE id = ?",
                        (face_id,))
    db.conn.commit()

    return {"ok": True, "face_id": face_id, "encoded": bool(encoding),
            "det_score": det_score, "person_id": person_id,
            "person_name": person_name}

"""Person (body) boxes: every person in the frame, with or without a visible face.

Faces alone cannot represent the subject who has their back to the camera, so
people-bind binds over bodies (docs/plans/description-names.md, Revision 2):
a phrase whose chosen body holds no offered face is "this person, no name".

Boxes are (top, right, bottom, left) in the EXIF-oriented pixel frame, the
same frame as ``faces.bbox_*`` (faces rule, "EXIF-oriented image dimensions").
The face-to-body link is computed by :func:`link_faces`, never stored: face ids
change on re-detection, geometry does not.
"""
from __future__ import annotations

import os
from typing import Optional

DETECTOR_VERSION = 1
# Kept in the table; consumers apply their own floor (the labelling sheet uses 0.5).
MIN_SCORE = 0.3
# Long edge the image is resized to before detection; boxes are scaled back.
DETECT_MAX_PX = 1600

_model = None


def _device():
    import torch
    want = os.environ.get("PHOTOSEARCH_DEVICE", "").strip().lower()
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_model():
    global _model
    if _model is None:
        import torchvision
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        m = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=w, box_score_thresh=MIN_SCORE)
        _model = (m.to(_device()).eval(), w.meta["categories"].index("person"))
    return _model


def detect(image_path: str) -> list[dict]:
    """Every person the detector finds: ``[{"bbox": [top, right, bottom, left],
    "score"}]`` in the oriented full-resolution frame, highest score first."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image, ImageOps

    with Image.open(image_path) as im0:
        im = ImageOps.exif_transpose(im0).convert("RGB")
    w, h = im.size
    s = min(1.0, DETECT_MAX_PX / float(max(w, h)))
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    model, person = _get_model()
    with torch.no_grad():
        out = model([TF.to_tensor(im).to(_device())])[0]
    boxes = []
    for (x1, y1, x2, y2), lab, sc in zip(out["boxes"].tolist(), out["labels"].tolist(),
                                         out["scores"].tolist()):
        if lab != person or sc < MIN_SCORE:
            continue
        boxes.append({"bbox": [round(y1 / s), round(x2 / s), round(y2 / s), round(x1 / s)],
                      "score": round(float(sc), 4)})
    return sorted(boxes, key=lambda b: -b["score"])


def _face_box(f: dict) -> tuple:
    if "bbox" in f:
        return tuple(f["bbox"])
    return (f["bbox_top"], f["bbox_right"], f["bbox_bottom"], f["bbox_left"])


def link_faces(faces: list[dict], bodies: list[dict]) -> dict[int, int]:
    """``{face_id: body index}``: each face goes to the smallest body containing
    its centre, one to one (a body holds at most one face). A face inside no
    body, or whose bodies are all taken by smaller-area pairings, is unlinked.
    Deterministic: ties break on face id, then body index."""
    pairs = []
    for f in faces:
        t, r, b, l = _face_box(f)
        cx, cy = (l + r) / 2.0, (t + b) / 2.0
        for i, body in enumerate(bodies):
            bt, br, bb, bl = body["bbox"]
            if bl <= cx <= br and bt <= cy <= bb:
                pairs.append(((br - bl) * (bb - bt), f["face_id"], i))
    out: dict[int, int] = {}
    used: set = set()
    for _, fid, i in sorted(pairs):
        if fid not in out and i not in used:
            out[fid] = i
            used.add(i)
    return out


def body_letter(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA: names for bodies on a labelling sheet."""
    s = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


def letter_bodies(bodies: list[dict], min_score: float,
                  faces: Optional[list[dict]] = None) -> list[dict]:
    """Bodies at or above ``min_score``, ordered left to right and lettered,
    each with the ``face_id`` of the offered face it holds (or None)."""
    keep = sorted((b for b in bodies if b["score"] >= min_score), key=lambda b: (b["bbox"][3], b["bbox"][0]))
    link = link_faces(faces or [], keep)
    by_body = {i: fid for fid, i in link.items()}
    return [dict(b, b=body_letter(i), face_id=by_body.get(i)) for i, b in enumerate(keep)]

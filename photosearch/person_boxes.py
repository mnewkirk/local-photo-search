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


def _area(b) -> float:
    t, r, bo, l = b
    return max(0, r - l) * max(0, bo - t)


def _inter(a, b) -> float:
    return _area((max(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])))


# Faster R-CNN's NMS (IoU 0.5) still leaves two shapes of double box on one person: a
# narrower box with the same top and bottom (247644: two boxes on one player) and an
# upper-body fragment (247988). Both lie inside the other box with the same top edge
# and centre. An occluded second person (247644: a spectator behind a player, wholly
# inside his box) does not share the centre, so containment alone is not the test.
DUP_CONTAIN = 0.8     # of the smaller box inside the other
DUP_CENTRE = 0.12     # |centre x difference| / the wider box's width
DUP_TOP = 0.10        # |top difference| / the taller box's height


def _same_person(x, y) -> bool:
    small = min(_area(x), _area(y))
    if small <= 0 or _inter(x, y) / small < DUP_CONTAIN:
        return False
    w = max(x[1] - x[3], y[1] - y[3])
    h = max(x[2] - x[0], y[2] - y[0])
    return (abs((x[1] + x[3]) - (y[1] + y[3])) / 2.0 <= DUP_CENTRE * w
            and abs(x[0] - y[0]) <= DUP_TOP * h)


def dedupe(bodies: list[dict]) -> tuple[list[int], dict[int, int]]:
    """``(kept indices, {dropped index: the kept index it duplicates})``. The
    higher-scoring box of a pair is kept; kept indices are in input order."""
    kept: list[int] = []
    merged: dict[int, int] = {}
    for i in sorted(range(len(bodies)), key=lambda i: -bodies[i]["score"]):
        j = next((k for k in kept if _same_person(bodies[i]["bbox"], bodies[k]["bbox"])), None)
        if j is None:
            kept.append(i)
        else:
            merged[i] = j
    return sorted(kept), merged


# A face is the head of a body: its centre inside the box, its top within the upper
# HEAD_TOP_FRAC of the box. Cost = how far the face's top sits below the box's top,
# with the horizontal offset only a tie-break: overlapping players share centres
# (243560: a boy in front of another), but each head tops its own box.
HEAD_TOP_FRAC = 0.4
HEAD_CENTRE_WEIGHT = 0.25


# The whole face must lie inside the body (owner, 2026-10-31): a face straddling a body's
# edge is usually the person behind or beside it (246150: 46% of the navy boy's face lay
# outside the box of the boy in front, who was facing away). Of 185 face-to-body links in the
# review binds, 2 had any part outside: that error and one correct link.
FACE_OUTSIDE_TOL = 0.0      # share of the face's width or height allowed outside the body


def _face_outside(face, body) -> float:
    t, r, b, l = face
    bt, br, bb, bl = body
    w, h = max(1, r - l), max(1, b - t)
    return max((max(0, bl - l) + max(0, r - br)) / w, (max(0, bt - t) + max(0, b - bb)) / h)


def _head_cost(face, body) -> Optional[float]:
    t, r, b, l = face
    bt, br, bb, bl = body
    cx, cy = (l + r) / 2.0, (t + b) / 2.0
    h, w = bb - bt, br - bl
    if h <= 0 or w <= 0 or not (bl <= cx <= br and bt <= cy <= bb):
        return None
    if _face_outside(face, body) > FACE_OUTSIDE_TOL:
        return None
    dy = (t - bt) / h
    if dy > HEAD_TOP_FRAC:
        return None
    return abs(dy) + HEAD_CENTRE_WEIGHT * abs(cx - (bl + br) / 2.0) / w


def link_faces(faces: list[dict], bodies: list[dict]) -> dict[int, int]:
    """``{face_id: body index}``: one to one, minimising the total head cost
    (the "smallest box" rule let a fragment steal the face from its own body).
    A face that is nobody's head stays unlinked. Deterministic."""
    from scipy.optimize import linear_sum_assignment
    if not faces or not bodies:
        return {}
    big = 1e6
    cost = [[(c if (c := _head_cost(_face_box(f), b["bbox"])) is not None else big)
             for b in bodies] for f in faces]
    rows, cols = linear_sum_assignment(cost)
    return {faces[i]["face_id"]: int(j) for i, j in zip(rows, cols) if cost[i][j] < big}


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
    """Bodies at or above ``min_score``, duplicates removed (:func:`dedupe`),
    ordered left to right and lettered, each with the ``face_id`` of the
    offered face whose head it is (or None)."""
    above = [b for b in bodies if b["score"] >= min_score]
    kept, _ = dedupe(above)
    keep = sorted((above[i] for i in kept), key=lambda b: (b["bbox"][3], b["bbox"][0]))
    link = link_faces(faces or [], keep)
    by_body = {i: fid for fid, i in link.items()}
    return [dict(b, b=body_letter(i), face_id=by_body.get(i)) for i, b in enumerate(keep)]

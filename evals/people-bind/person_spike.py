"""Spike: does an exhaustive person detector give the faceless subjects a box?

Read-only. torchvision Faster R-CNN v2 (COCO 'person'), on the people-bind
eval sample. Each offered face is linked to the smallest body box containing
its centre (one to one); a body with no linked face is 'faceless'.
"""
import json, sqlite3, sys, time
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps

REPO = Path("/home/mattn/local-photo-search")
sys.path.insert(0, str(REPO))
from photosearch import people_bind_store as store  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MAX_PX = 1600
THRESH = (0.5, 0.7)

con = sqlite3.connect(f"file:{REPO}/photo_index.db.local?mode=ro", uri=True)
con.row_factory = sqlite3.Row
labels = json.load(open(REPO / "evals/people-bind/labels.json"))

dev = "cuda" if torch.cuda.is_available() else "cpu"
W = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=W, box_score_thresh=0.3).to(dev).eval()
PERSON = W.meta["categories"].index("person")


def link(faces, bodies):
    """face_id -> body index; smallest containing body first, one to one."""
    pairs = []
    for f in faces:
        t, r, b, l = f["bbox"]
        cx, cy = (l + r) / 2, (t + b) / 2
        for i, (x1, y1, x2, y2) in enumerate(bodies):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                pairs.append(((x2 - x1) * (y2 - y1), f["face_id"], i))
    out, used = {}, set()
    for _, fid, i in sorted(pairs):
        if fid not in out and i not in used:
            out[fid] = i
            used.add(i)
    return out


rows, t0 = [], time.time()
for pid_s, lab in sorted(labels.items(), key=lambda kv: int(kv[0])):
    pid = int(pid_s)
    inp = store.claim_input(con, pid)
    img = ImageOps.exif_transpose(Image.open(REPO / "evals/model-evals/originals" / pid_s)).convert("RGB")
    w0, h0 = img.size
    s = min(1.0, MAX_PX / max(w0, h0))
    small = img.resize((round(w0 * s), round(h0 * s)))
    with torch.no_grad():
        det = model([torchvision.transforms.functional.to_tensor(small).to(dev)])[0]
    keep = det["labels"] == PERSON
    boxes = (det["boxes"][keep] / s).tolist()
    scores = det["scores"][keep].tolist()
    faces = inp["faces"] if inp else []
    nones = [m for m in lab["mentions"] if m["gold"] == "NONE"]
    row = {"pid": pid, "stratum": lab["stratum"], "dims_match": (w0, h0) == (inp["image_width"], inp["image_height"]) if inp else None,
           "faces": len(faces), "none_phrases": len(nones)}
    for th in THRESH:
        bb = [b for b, sc in zip(boxes, scores) if sc >= th]
        ln = link(faces, bb)
        row[f"bodies@{th}"] = len(bb)
        row[f"faceless@{th}"] = len(bb) - len(ln)
        row[f"face_no_body@{th}"] = len(faces) - len(ln)
    rows.append(row)
    if nones:   # contact sheet: green = body with an offered face, red = faceless body, yellow = face
        bb = [b for b, sc in zip(boxes, scores) if sc >= 0.5]
        ln = set(link(faces, bb).values())
        d = ImageDraw.Draw(small)
        lw = max(3, small.width // 300)
        for i, (x1, y1, x2, y2) in enumerate(bb):
            d.rectangle([x1 * s, y1 * s, x2 * s, y2 * s], outline=(0, 200, 0) if i in ln else (230, 0, 0), width=lw)
        for n, f in enumerate(faces, 1):
            t, r, b, l = f["bbox"]
            d.rectangle([l * s, t * s, r * s, b * s], outline=(255, 220, 0), width=lw)
            d.text((l * s, b * s + 2), str(n), fill=(255, 220, 0))
        small.save(OUT / f"{pid}.jpg", quality=80)

(OUT / "rows.json").write_text(json.dumps(rows, indent=1))
print(f"{len(rows)} photos in {time.time() - t0:.0f}s on {dev}")

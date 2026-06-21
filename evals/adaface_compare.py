"""Does AdaFace separate the wrong-green keeps from correct-Calvin keeps that
ArcFace cannot (overlapping at ~1.15-1.24)? Same InsightFace detection +
alignment feeds BOTH models, so the only variable is the recognition head.

AdaFace = minchul/cvlface_adaface_ir101_webface12m (IR-101, WebFace12M),
loaded as the bare net (strip 'net.' prefix). Input: RGB 112x112, [-1,1].
"""
import os
import sys
import types
import urllib.request

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photosearch.db import PhotoDB

NAS = "http://dxp4800-f976:8000"
REF_PHOTOS = [146137, 135481, 149139, 157832, 139946, 155880, 82681, 164190, 82672, 133981, 104266]
WRONG = [63014, 57668, 60681, 4520, 21605, 60669]
CORRECT = [21328, 25635, 127379, 23583, 114452, 46848, 28810, 87798, 25627, 34225, 87750, 16496]

# ---- load AdaFace IR-101 (bare net) -----------------------------------------
from huggingface_hub import hf_hub_download
import importlib.util

sys.modules["fvcore"] = types.ModuleType("fvcore")          # stub (only flop_count uses it)
sys.modules["fvcore.nn"] = types.ModuleType("fvcore.nn")
sys.modules["fvcore.nn"].flop_count = lambda *a, **k: None
mp = hf_hub_download("minchul/cvlface_adaface_ir101_webface12m", "models/iresnet/model.py")
spec = importlib.util.spec_from_file_location("ada_iresnet", mp)
ada_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ada_mod)
wp = hf_hub_download("minchul/cvlface_adaface_ir101_webface12m", "pretrained_model/model.pt")
sd = {k[4:]: v for k, v in torch.load(wp, map_location="cpu").items() if k.startswith("net.")}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ada = ada_mod.IR_101(input_size=(112, 112), output_dim=512)
missing, unexpected = ada.load_state_dict(sd, strict=False)
ada.eval().to(DEV)
print(f"AdaFace loaded on {DEV} | missing={len(missing)} unexpected={len(unexpected)}")


def ada_embed(rgb112):
    t = torch.from_numpy(((rgb112.astype("float32") / 255.0) - 0.5) / 0.5)
    t = t.permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        out = ada(t)
    v = (out[0] if isinstance(out, (tuple, list)) else out).squeeze(0).cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)


# ---- InsightFace detection + alignment --------------------------------------
from insightface.app import FaceAnalysis
from insightface.utils import face_align

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=0 if DEV == "cuda" else -1, det_size=(640, 640))

_img_cache = {}
def get_img(pid):
    if pid not in _img_cache:
        try:
            data = urllib.request.urlopen(f"{NAS}/api/photos/{pid}/full", timeout=40).read()
            _img_cache[pid] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            _img_cache[pid] = None
    return _img_cache[pid]


def iou(a, b):  # a,b = [x1,y1,x2,y2]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def embed_face_at(pid, stored_bbox):
    """Re-detect pid, match the detected face to stored_bbox, return (arc, ada)
    normalized embeddings or None."""
    img = get_img(pid)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: iou([int(x) for x in f.bbox], stored_bbox))
    if iou([int(x) for x in best.bbox], stored_bbox) < 0.25:
        return None
    arc = best.embedding / (np.linalg.norm(best.embedding) + 1e-9)
    aligned_bgr = face_align.norm_crop(img, best.kps)        # 112x112 BGR
    ada_v = ada_embed(aligned_bgr[:, :, ::-1])               # -> RGB
    return arc, ada_v


db = PhotoDB("./photo_index.db.local"); c = db.conn
cid = c.execute("SELECT id FROM persons WHERE name='Calvin'").fetchone()[0]


def stored_bbox(face_row):
    return [face_row["bbox_left"], face_row["bbox_top"], face_row["bbox_right"], face_row["bbox_bottom"]]


# reference sets
R_arc, R_ada = [], []
for pid in REF_PHOTOS:
    f = c.execute("SELECT bbox_top,bbox_right,bbox_bottom,bbox_left FROM faces "
                  "WHERE photo_id=? AND person_id=?", (pid, cid)).fetchone()
    e = embed_face_at(pid, stored_bbox(f))
    if e:
        R_arc.append(e[0]); R_ada.append(e[1])
R_arc, R_ada = np.stack(R_arc), np.stack(R_ada)
print(f"reference set: {len(R_arc)} faces under each model\n")


def mind(v, R):
    return float(np.min(np.linalg.norm(R - v, axis=1)))


def kept_dists(pid):
    """For a double-Calvin photo, the KEPT face = min ArcFace dist (the green).
    Return (arc_dist, ada_dist) for that same physical face."""
    rows = c.execute("SELECT bbox_top,bbox_right,bbox_bottom,bbox_left FROM faces "
                     "WHERE photo_id=? AND person_id=?", (pid, cid)).fetchall()
    cands = []
    for f in rows:
        e = embed_face_at(pid, stored_bbox(f))
        if e:
            cands.append((mind(e[0], R_arc), mind(e[1], R_ada)))
    if not cands:
        return None
    return min(cands, key=lambda t: t[0])   # kept = closest under ArcFace


print(f"{'photo':>7} {'label':>8} | {'ArcFace':>7} {'AdaFace':>7}")
print("-" * 36)
res = {"WRONG": [], "CORRECT": []}
for label, photos in (("WRONG", WRONG), ("CORRECT", CORRECT)):
    for pid in photos:
        kd = kept_dists(pid)
        if not kd:
            print(f"{pid:>7} {label:>8} | (no face)"); continue
        res[label].append(kd)
        print(f"{pid:>7} {label:>8} | {kd[0]:>7.3f} {kd[1]:>7.3f}")

print("\n=== separation (kept-face distance to reference set) ===")
for label in ("WRONG", "CORRECT"):
    a = np.array([x[0] for x in res[label]]); d = np.array([x[1] for x in res[label]])
    print(f"  {label:>8}: ArcFace mean={a.mean():.3f} [{a.min():.2f}-{a.max():.2f}]  "
          f"AdaFace mean={d.mean():.3f} [{d.min():.2f}-{d.max():.2f}]")
if res["WRONG"] and res["CORRECT"]:
    for name, idx in (("ArcFace", 0), ("AdaFace", 1)):
        cmax = max(x[idx] for x in res["CORRECT"])
        wmin = min(x[idx] for x in res["WRONG"])
        gap = wmin - cmax
        print(f"  {name}: correct-max={cmax:.3f}  wrong-min={wmin:.3f}  "
              f"{'SEPARABLE by ' + format(gap,'.3f') if gap > 0 else 'OVERLAP ' + format(gap,'.3f')}")
db.close()

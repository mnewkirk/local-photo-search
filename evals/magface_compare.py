"""MagFace + QMagFace test. MagFace embeds identity in the DIRECTION and face
quality in the MAGNITUDE. QMagFace is a quality-aware comparison on top. The
decisive question: do the wrong-green keeps differ from correct-Calvin keeps in
identity distance OR in quality (magnitude)? If they overlap in BOTH, no
quality-weighted fusion can separate them.

Same InsightFace detection+alignment as the AdaFace test feeds MagFace, so the
only variable is the recognition head. MagFace = facetorch TorchScript
(native iResNet100). Input: aligned 112x112 BGR, [-1,1].
"""
import os
import sys
import urllib.request

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photosearch.db import PhotoDB

NAS = "http://dxp4800-f976:8000"
REF_PHOTOS = [146137, 135481, 149139, 157832, 139946, 155880, 82681, 164190, 82672, 133981, 104266]
WRONG = [63014, 57668, 60681, 4520, 21605, 60669]
CORRECT = [21328, 127379, 114452, 46848, 87798, 34225, 87750, 16496]   # ones with originals on NAS

from huggingface_hub import hf_hub_download
DEV = "cuda" if torch.cuda.is_available() else "cpu"
mag = torch.jit.load(hf_hub_download("tomas-gajarsky/facetorch-verify-magface", "model.pt"),
                     map_location=DEV).eval()
print(f"MagFace loaded on {DEV}")


def mag_embed(bgr112):
    """Return raw MagFace embedding (magnitude = quality)."""
    t = torch.from_numpy(((bgr112.astype("float32") / 255.0) - 0.5) / 0.5)
    t = t.permute(2, 0, 1).unsqueeze(0).to(DEV)
    with torch.no_grad():
        o = mag(t)
    return (o[0] if isinstance(o, (tuple, list)) else o).squeeze(0).cpu().numpy()


from insightface.app import FaceAnalysis
from insightface.utils import face_align
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
app.prepare(ctx_id=0 if DEV == "cuda" else -1, det_size=(640, 640))

_img = {}
def get_img(pid):
    if pid not in _img:
        try:
            d = urllib.request.urlopen(f"{NAS}/api/photos/{pid}/full", timeout=40).read()
            _img[pid] = cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            _img[pid] = None
    return _img[pid]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def face_at(pid, bbox):
    """Return (arc_norm, mag_raw) for the detected face matching bbox, or None."""
    img = get_img(pid)
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: iou([int(x) for x in f.bbox], bbox))
    if iou([int(x) for x in best.bbox], bbox) < 0.25:
        return None
    arc = best.embedding / (np.linalg.norm(best.embedding) + 1e-9)
    m = mag_embed(face_align.norm_crop(img, best.kps))
    return arc, m


db = PhotoDB("./photo_index.db.local"); c = db.conn
cid = c.execute("SELECT id FROM persons WHERE name='Calvin'").fetchone()[0]
def bbox(r): return [r["bbox_left"], r["bbox_top"], r["bbox_right"], r["bbox_bottom"]]

# reference sets (identity direction + quality magnitudes)
R_arc, R_magdir, ref_q = [], [], []
for pid in REF_PHOTOS:
    f = c.execute("SELECT bbox_top,bbox_right,bbox_bottom,bbox_left FROM faces WHERE photo_id=? AND person_id=?",
                  (pid, cid)).fetchone()
    e = face_at(pid, bbox(f))
    if e:
        R_arc.append(e[0])
        q = np.linalg.norm(e[1]); ref_q.append(q)
        R_magdir.append(e[1] / (q + 1e-9))
R_arc, R_magdir = np.stack(R_arc), np.stack(R_magdir)
print(f"reference: {len(R_arc)} faces | MagFace quality(magnitude) of refs: "
      f"mean={np.mean(ref_q):.1f} [{min(ref_q):.1f}-{max(ref_q):.1f}]\n")


def mind(v, R): return float(np.min(np.linalg.norm(R - v, axis=1)))


def kept(pid):
    """Kept face = closest under ArcFace (the green). Return its MagFace identity
    distance and quality magnitude."""
    rows = c.execute("SELECT bbox_top,bbox_right,bbox_bottom,bbox_left FROM faces WHERE photo_id=? AND person_id=?",
                     (pid, cid)).fetchall()
    cands = []
    for r in rows:
        e = face_at(pid, bbox(r))
        if e:
            q = np.linalg.norm(e[1])
            cands.append((mind(e[0], R_arc), mind(e[1] / (q + 1e-9), R_magdir), q))
    return min(cands, key=lambda t: t[0]) if cands else None


print(f"{'photo':>7} {'label':>8} | {'Arc-d':>6} {'Mag-id-d':>8} {'Mag-qual':>8}")
print("-" * 46)
res = {"WRONG": [], "CORRECT": []}
for label, photos in (("WRONG", WRONG), ("CORRECT", CORRECT)):
    for pid in photos:
        k = kept(pid)
        if not k:
            print(f"{pid:>7} {label:>8} | (no face)"); continue
        res[label].append(k)
        print(f"{pid:>7} {label:>8} | {k[0]:>6.3f} {k[1]:>8.3f} {k[2]:>8.1f}")

print("\n=== separation: does identity-distance OR quality split wrong vs correct? ===")
for label in ("WRONG", "CORRECT"):
    A = np.array(res[label])
    print(f"  {label:>8}: MagFace id-dist {A[:,1].mean():.3f} [{A[:,1].min():.2f}-{A[:,1].max():.2f}]  "
          f"| quality {A[:,2].mean():.1f} [{A[:,2].min():.1f}-{A[:,2].max():.1f}]")
if res["WRONG"] and res["CORRECT"]:
    W, C = np.array(res["WRONG"]), np.array(res["CORRECT"])
    for name, idx, hi_is_wrong in (("MagFace id-dist", 1, True), ("MagFace quality", 2, False)):
        if hi_is_wrong:
            gap = W[:, idx].min() - C[:, idx].max()
        else:   # quality: if wrong are LOWER quality, wrong-max < correct-min would separate
            gap = C[:, idx].min() - W[:, idx].max()
        print(f"  {name}: {'SEPARABLE by ' + format(gap,'.3f') if gap > 0 else 'OVERLAP ' + format(gap,'.3f')}")
db.close()

"""Experiment: can a VLM separate correct-Calvin from wrong-green keeps that
the ArcFace distance metric cannot? Compares each candidate (kept/green) face
against an AGE-MATCHED reference Calvin face via qwen2.5-vl, asking same-child."""
import base64
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from photosearch.db import PhotoDB

NAS = os.environ.get("PHOTOSEARCH_NAS_URL", "http://dxp4800-f976:8000")
LM = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL", "http://localhost:1234/v1")
VMODEL = os.environ.get("PHOTOSEARCH_LLM_VISUAL_MODEL", "qwen2.5-vl-7b-instruct")

REF_PHOTOS = {146137: 2026, 135481: 2025, 149139: 2024, 157832: 2023,
              139946: 2022, 155880: 2021, 82681: 2020, 164190: 2019,
              82672: 2018, 133981: 2017, 104266: 2016}
WRONG = [63014, 57668, 60681, 4520, 21605, 60669]

db = PhotoDB("./photo_index.db.local")
c = db.conn
cid = c.execute("SELECT id FROM persons WHERE name='Calvin'").fetchone()[0]

# reference face id + encoding per year
ref_face_by_year, R = {}, []
for pid, yr in REF_PHOTOS.items():
    row = c.execute("SELECT f.id, e.encoding FROM faces f JOIN face_encodings e ON e.face_id=f.id "
                    "WHERE f.photo_id=? AND f.person_id=?", (pid, cid)).fetchone()
    ref_face_by_year[yr] = row["id"]
    R.append(np.frombuffer(row["encoding"], dtype=np.float32))
R = np.stack(R); R /= np.linalg.norm(R, axis=1, keepdims=True)


def mindist(b):
    v = np.frombuffer(b, dtype=np.float32)
    return float(np.min(np.linalg.norm(R - v / np.linalg.norm(v), axis=1)))


_crop_cache = {}
def crop_b64(fid):
    if fid not in _crop_cache:
        try:
            with urllib.request.urlopen(f"{NAS}/api/faces/crop/{fid}", timeout=25) as r:
                _crop_cache[fid] = base64.b64encode(r.read()).decode("ascii")
        except Exception:
            _crop_cache[fid] = None
    return _crop_cache[fid]


def kept_face_and_year(photo_id):
    faces = c.execute("SELECT f.id, e.encoding FROM faces f JOIN face_encodings e ON e.face_id=f.id "
                      "WHERE f.photo_id=? AND f.person_id=?", (photo_id, cid)).fetchall()
    ranked = sorted((mindist(f["encoding"]), f["id"]) for f in faces)
    row = c.execute("SELECT substr(date_taken,1,4) FROM photos WHERE id=?", (photo_id,)).fetchone()
    yr = row[0] if row else None
    return ranked[0][1], ranked[0][0], int(yr) if yr and yr.isdigit() else 2020


def vlm_same(cand_b64, ref_b64):
    prompt = (
        "Two cropped face photos of young children, taken the same year (same age). "
        "Decide if they are the SAME child. IGNORE hairstyle, hair color, clothing, "
        "lighting, expression, angle, and image quality — children's hair changes "
        "constantly and is NOT evidence. Judge ONLY the permanent facial identity: "
        "eye shape/spacing, nose, mouth, and overall face shape. Two crops of the "
        "same child often look superficially different; default to SAME unless the "
        "core facial features clearly differ. "
        'Reply ONLY JSON: {"same": true|false, "confidence": <0-1>, "reason": "<short>"}.')
    body = json.dumps({
        "model": VMODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cand_b64}"}},
        ]}],
        "max_tokens": 150, "temperature": 0,
    }).encode()
    try:
        req = urllib.request.Request(LM.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"] or ""
        a, b = txt.find("{"), txt.rfind("}")
        obj = json.loads(txt[a:b + 1])
        return bool(obj.get("same")), float(obj.get("confidence", 0)), str(obj.get("reason", ""))[:60]
    except Exception as e:
        return None, None, f"ERR {e}"[:60]


# correct-green sample: confident decisions sorted by gap, drop the wrong ones, take a spread
dbl = c.execute("SELECT photo_id FROM faces WHERE person_id=? GROUP BY photo_id HAVING COUNT(*)>=2", (cid,)).fetchall()
dec = []
for row in dbl:
    ds = sorted(mindist(f["encoding"]) for f in c.execute(
        "SELECT e.encoding FROM faces f JOIN face_encodings e ON e.face_id=f.id "
        "WHERE f.photo_id=? AND f.person_id=?", (row["photo_id"], cid)).fetchall())
    if ds[1] - ds[0] > 0.15:
        dec.append((row["photo_id"], ds[1] - ds[0]))
dec.sort(key=lambda x: x[1])
correct = [p for p, _ in dec[:120] if p not in WRONG][:12]

print(f"VLM={VMODEL}  NAS={NAS}\n")
print(f"{'photo':>7} {'label':>8} {'dist':>5} {'yr':>4} | {'VLM same?':>9} {'conf':>4}  reason")
print("-" * 78)
results = {"wrong": [], "correct": []}
for label, photos in (("WRONG", WRONG), ("CORRECT", correct)):
    for ph in photos:
        fid, dist, yr = kept_face_and_year(ph)
        ref_yr = min(REF_PHOTOS.values(), key=lambda y: abs(y - yr))
        cand, ref = crop_b64(fid), crop_b64(ref_face_by_year[ref_yr])
        if not cand or not ref:
            print(f"{ph:>7} {label:>8} {dist:>5.2f} {yr:>4} | (no crop)")
            continue
        same, conf, reason = vlm_same(cand, ref)
        results["wrong" if label == "WRONG" else "correct"].append((same, conf))
        print(f"{ph:>7} {label:>8} {dist:>5.2f} {yr:>4} | {str(same):>9} {conf if conf is not None else '-':>4}  {reason}")

print("\n=== summary ===")
for k in ("wrong", "correct"):
    rs = [r for r in results[k] if r[0] is not None]
    same_n = sum(1 for s, _ in rs if s)
    print(f"  {k:>8}: VLM said SAME for {same_n}/{len(rs)}  "
          f"(ideal: wrong→0, correct→all)")
db.close()

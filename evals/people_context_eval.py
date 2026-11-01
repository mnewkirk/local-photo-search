#!/usr/bin/env python
"""People context in describe: the G0 identity eval and the G1 grounded-describe eval
(docs/plans/people-context-describe.md, Steps 1-3). Evals BEFORE any pipeline wiring.

  who-sheet   owner's one-time "who is this body" page for the faceless bodies they
              placed in round 2 (labels.json NONE phrases) -> people-bind/who/sheet/;
              serve it with `people_bind_eval.py --set who serve --port 8766`
  evidence    bodies + keypoints for the eval photos (GPU) -> people-bind/g0/evidence.json
  attrs       per-body VLM attributes (jersey number, kit, hair, shoes, role) -> g0/attrs.json
  g0          leave-face-out + owner-labelled identity eval, split by shoot
  g1-run      grounded describe variants on the labelled sample -> people-bind/g1/runs/
  g1-report   score G1 against its gate list

The DB is opened read-only (mode=ro). Images come from the replica (originals cached in
evals/model-evals/originals). Labels are files, never DB rows.
"""
import argparse
import base64
import hashlib
import html as _html
import io
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photosearch import identify as ID                         # noqa: E402

HERE = Path(__file__).resolve().parent / "people-bind"
WHO = HERE / "who"
G0 = HERE / "g0"
G1 = HERE / "g1"
TRUSTED = ("manual", "strict", "merge_review")
TUNE_LAST_DAY = "2026-09-27"      # shoots 08-29..09-27 tune; 10-03..10-25 holdout
SERVER = os.environ.get("PHOTOSEARCH_EVAL_SERVER", "http://localhost:8001")
INTENT = "people-context eval (G0/G1): "


def ro(db_path):
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _jload(p, default):
    return json.loads(Path(p).read_text()) if Path(p).exists() else default


def _jsave(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    tmp.replace(p)


def _get(path, why, timeout=120):
    req = urllib.request.Request(f"{SERVER.rstrip('/')}{path}",
                                 headers={"X-Photosearch-Source": "claude",
                                          "X-Photosearch-Intent": INTENT + why})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch(server, photo_id, kind="full", timeout=120):
    return _get(f"/api/photos/{int(photo_id)}/{kind}", "original for body crops / overlays", timeout)


def original(photo_id):
    from photosearch import model_eval
    return model_eval.original_path(photo_id, SERVER, fetch=_fetch)


def split_of(date_taken):
    return "tune" if (date_taken or "")[:10] <= TUNE_LAST_DAY else "holdout"


def session_map(con, folders):
    rows = []
    for f in folders:
        rows += con.execute("SELECT id, folder, date_taken FROM photos WHERE folder = ? AND "
                            "(media_type IS NULL OR media_type != 'video')", (f,)).fetchall()
    return ID.sessions(rows), {r[0]: r[2] for r in rows}


def _open_oriented(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im0:
        return ImageOps.exif_transpose(im0).convert("RGB")


def crop_body(im, bbox, pad=0.12, max_px=700):
    from PIL import Image
    t, r, b, l = bbox
    w, h = r - l, b - t
    box = (max(0, int(l - pad * w)), max(0, int(t - pad * h)),
           min(im.width, int(r + pad * w)), min(im.height, int(b + pad * h)))
    c = im.crop(box)
    s = min(1.0, max_px / float(max(c.size)))
    if s < 1.0:
        c = c.resize((max(1, round(c.width * s)), max(1, round(c.height * s))), Image.LANCZOS)
    return c


def _jpeg(im, q=86):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# who-sheet: the owner names each faceless body they placed
# ---------------------------------------------------------------------------

def who_targets():
    """[(photo id str, body letter, [phrases])] for every body the owner put a NONE phrase on."""
    labels = _jload(HERE / "labels.json", {})
    t = defaultdict(list)
    for pid_s, lab in labels.items():
        for m in lab["mentions"]:
            bl = m.get("bodies")
            for letter in (bl if isinstance(bl, list) else []):
                if letter != "MISSING":
                    t[(pid_s, letter)].append(m["phrase"])
    return [(p, b, ph) for (p, b), ph in sorted(t.items(), key=lambda kv: (int(kv[0][0]), kv[0][1]))]


def session_people(con, pids):
    """{person_id: (name, n trusted faces, best face id)} over these photos."""
    count, best, name_of = Counter(), {}, {}
    for i in range(0, len(pids), 900):
        ch = pids[i:i + 900]
        for fid, per, name, area, det in con.execute(
                f"SELECT f.id, f.person_id, pe.name, (f.bbox_right-f.bbox_left)*(f.bbox_bottom-f.bbox_top), "
                f"COALESCE(f.det_score, 1) FROM faces f JOIN persons pe ON pe.id = f.person_id "
                f"WHERE f.match_source IN ('manual','strict','merge_review') "
                f"AND f.photo_id IN ({','.join('?' * len(ch))})", ch):
            count[per] += 1
            name_of[per] = name
            if per not in best or area * det > best[per][0]:
                best[per] = (area * det, fid)
    return {p: (name_of[p], count[p], best[p][1]) for p in count}


def cmd_who_sheet(a):
    from PIL import ImageDraw
    con = ro(a.db)
    bodies_all = _jload(HERE / "bodies.json", {})["photos"]
    labels = _jload(HERE / "labels.json", {})
    targets = who_targets()
    sheet = WHO / "sheet"
    sheet.mkdir(parents=True, exist_ok=True)
    pids = sorted({int(p) for p, _, _ in targets})
    meta = {r[0]: (r[1], r[2]) for r in con.execute(
        f"SELECT id, folder, date_taken FROM photos WHERE id IN ({','.join('?' * len(pids))})", pids)}
    sess, _ = session_map(con, sorted({m[0] for m in meta.values()}))
    by_sess = defaultdict(list)
    for pid, s in sess.items():
        by_sess[s].append(pid)
    people_cache = {}
    items = []
    for pid_s, letter, phrases in targets:
        pid = int(pid_s)
        s = sess.get(pid)
        if s not in people_cache:
            people_cache[s] = session_people(con, by_sess[s])
        body = next(b for b in bodies_all[pid_s] if b["b"] == letter)
        key = f"{pid_s}-{letter}"
        cp, xp = sheet / f"{key}-c.jpg", sheet / f"{key}-x.jpg"
        if not cp.exists() or not xp.exists():
            im = _open_oriented(original(pid))
            cp.write_bytes(_jpeg(crop_body(im, body["bbox"])))
            sc = min(1.0, 1400 / float(max(im.size)))
            ctx = im.resize((round(im.width * sc), round(im.height * sc)))
            d = ImageDraw.Draw(ctx)
            for ob in bodies_all[pid_s]:
                t, r, b, l = (v * sc for v in ob["bbox"])
                if ob["b"] == letter:
                    d.rectangle([l - 3, t - 3, r + 3, b + 3], outline=(255, 230, 0), width=6)
                else:
                    d.rectangle([l, t, r, b], outline=(160, 160, 160), width=1)
            xp.write_bytes(_jpeg(ctx, 80))
        ppl = sorted(people_cache[s].items(), key=lambda kv: (kv[1][0] or "").lower())   # alphabetical
        for per, (name, n, fid) in ppl:
            fp = sheet / f"f{fid}.jpg"
            if not fp.exists():
                try:
                    fp.write_bytes(_get(f"/api/faces/crop/{fid}?size=160", "face thumbnail for the who-is-this-body page"))
                except Exception as e:          # noqa: BLE001 - show the name without a picture
                    print(f"  ! face {fid}: {e}")
        items.append({"key": key, "pid": pid_s, "letter": letter, "phrases": phrases,
                      "split": split_of(meta[pid][1]), "day": (meta[pid][1] or "")[:10],
                      "description": None,
                      "people": [{"id": per, "name": name, "n": n, "face": fid} for per, (name, n, fid) in ppl]})
    descs = {r[0]: r[1] for r in con.execute(
        f"SELECT id, description FROM photos WHERE id IN ({','.join('?' * len(pids))})", pids)}
    for it in items:
        it["description"] = descs.get(int(it["pid"])) or ""
    key = hashlib.sha1(json.dumps([i["key"] for i in items]).encode()).hexdigest()[:10]
    page = (_WHO_HTML.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
            .replace("__KEY__", key))
    (sheet / "index.html").write_text(page)
    n_hold = sum(1 for i in items if i["split"] == "holdout")
    print(f"{len(items)} bodies ({n_hold} holdout) on {len(pids)} photos -> {sheet / 'index.html'}")


_WHO_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>who is this body?</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--on:#1565c0;--onfg:#fff;--card:#f6f7f9}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--on:#64b5f6;--onfg:#000;--card:#1f2228}}
body{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;margin:0}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
main{max-width:1400px;margin:auto;padding:0 16px}
section{border-bottom:2px solid var(--line);padding:14px 0}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,2fr);gap:14px;align-items:start}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
img.ph{width:100%;height:auto;border-radius:4px;cursor:zoom-in}
.mut{color:var(--mut);font-size:13px}
.people{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
.p{border:2px solid var(--line);background:var(--card);border-radius:6px;padding:3px;cursor:pointer;width:92px;text-align:center;font-size:12px;color:var(--fg)}
.p img{width:84px;height:84px;object-fit:cover;border-radius:4px;display:block}
.p.on,.x.on{border-color:var(--on);background:var(--on);color:var(--onfg)}
.x{font:inherit;border:2px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:6px 12px;margin:8px 4px 0 0;cursor:pointer}
#stale{background:#b71c1c;color:#fff;padding:12px 16px;font-weight:700}
</style></head><body>
<header><b>Who is this body?</b><span id="prog"></span>
 <label><input type="checkbox" id="todo" checked> only unanswered</label><span id="srv" class="mut"></span>
 <span class="mut">Yellow box = the person. Pick who it is from the faces labelled at this shoot, or one of the buttons. Click the photo to enlarge.</span></header>
<main id="root"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "who-__KEY__";
let L = {};
const SERVED = location.protocol.startsWith("http");
let ready = false, timer = null;
const OTHER = [["notours", "Our team, no face labelled / not in the list"], ["opponent", "Opponent"],
               ["adult", "Adult (coach, ref, parent)"], ["unknown", "Can't tell"]];
function srv(m) { document.getElementById("srv").textContent = m; }
function save() {
  progress();
  if (!SERVED || !ready) return;
  clearTimeout(timer);
  timer = setTimeout(() => fetch("saved-labels.json", {method: "PUT",
      headers: {"Content-Type": "application/json", "X-Sheet-Key": KEY}, body: JSON.stringify(L)})
    .then(r => { if (r.status === 409) throw new Error("this page is out of date: RELOAD it");
                 if (!r.ok) throw new Error("HTTP " + r.status); srv("saved " + new Date().toLocaleTimeString()); })
    .catch(e => { srv("NOT saved (" + e.message + ")");
      if (/out of date/.test(e.message) && !document.getElementById("stale")) {
        const d = document.createElement("div"); d.id = "stale";
        d.textContent = "This page is out of date and is NOT saving. Reload it.";
        document.body.prepend(d); } }), 300);
}
function progress() {
  const n = ITEMS.filter(it => (L[it.key] || {}).who).length;
  document.getElementById("prog").textContent = `${n}/${ITEMS.length} answered`;
}
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) { if (k === "on") e.addEventListener("click", v); else if (k === "cls") e.className = v; else e.setAttribute(k, v); }
  for (const k of kids) e.append(k);
  return e;
}
function pick(it, sec, who, name) {
  const cur = (L[it.key] || {}).who;
  L[it.key] = cur === who ? {} : {who: who, name: name || null};
  save();
  const nxt = card(it); sec.replaceWith(nxt);
  if (document.getElementById("todo").checked && L[it.key].who) {
    const n = nxt.nextElementSibling; nxt.remove(); if (n) n.scrollIntoView({block: "start"});
  }
}
function card(it) {
  const cur = (L[it.key] || {}).who;
  const sec = el("section", {id: "k" + it.key});
  sec.append(el("div", {cls: "mut"}, `${it.day} · photo ${it.pid} · body ${it.letter} · phrase: “${it.phrases.join("”, “")}”`));
  const crop = el("img", {cls: "ph", src: it.key + "-c.jpg", loading: "lazy"});
  crop.addEventListener("click", () => window.open(it.key + "-c.jpg"));
  const ctx = el("img", {cls: "ph", src: it.key + "-x.jpg", loading: "lazy"});
  ctx.addEventListener("click", () => window.open(it.key + "-x.jpg"));
  const ppl = el("div", {cls: "people"});
  for (const p of it.people) {
    ppl.append(el("div", {cls: "p" + (cur === p.id ? " on" : ""), title: p.n + " labelled faces at this shoot",
      on: () => pick(it, sec, p.id, p.name)}, el("img", {src: "f" + p.face + ".jpg", loading: "lazy"}), p.name));
  }
  const xs = el("div");
  for (const [k, label] of OTHER) xs.append(el("button", {cls: "x" + (cur === k ? " on" : ""), on: () => pick(it, sec, k, null)}, label));
  sec.append(el("div", {cls: "grid"}, crop, el("div", {}, ctx, xs, ppl)));
  return sec;
}
function render() {
  const root = document.getElementById("root"); root.innerHTML = "";
  const todo = document.getElementById("todo").checked;
  for (const it of ITEMS) if (!(todo && (L[it.key] || {}).who)) root.append(card(it));
  progress();
}
document.getElementById("todo").addEventListener("change", render);
render();
if (SERVED) fetch("saved-labels.json", {cache: "no-store"}).then(r => r.status === 404 ? null : r.json())
  .then(data => { if (data) L = data; ready = true; render(); srv("loaded from server"); })
  .catch(e => srv("could not load from server (" + e.message + "); NOT saving"));
</script></body></html>
"""


# ---------------------------------------------------------------------------
# evidence: bodies + head keypoints on every still of the soccer sessions (previews)
# ---------------------------------------------------------------------------

HEAD_KP = (0, 1, 2)       # nose, left eye, right eye (COCO)
KP_MIN = 2.0              # keypoint logit (Step 0)
BODY_MIN_SCORE = 0.5      # = the sheet's floor
PREVIEWS = G0 / "previews"


def soccer_session_photos(con):
    """{photo_id: (folder, date_taken, image_width, image_height)} for every still in the
    folders the soccer collections touch (the 10 Step 0 sessions)."""
    cols = [r[0] for r in con.execute("SELECT id FROM collections WHERE name LIKE '%soccer%'")]
    folders = sorted({r[0] for r in con.execute(
        f"SELECT DISTINCT p.folder FROM collection_photos cp JOIN photos p ON p.id = cp.photo_id "
        f"WHERE cp.collection_id IN ({','.join(map(str, cols))})")})
    out = {}
    for f in folders:
        for pid, d, w, h in con.execute(
                "SELECT id, date_taken, image_width, image_height FROM photos WHERE folder = ? "
                "AND (media_type IS NULL OR media_type != 'video')", (f,)):
            out[pid] = (f, d, w, h)
    return out


def preview(pid):
    p = PREVIEWS / f"{pid}.jpg"
    if not p.exists():
        PREVIEWS.mkdir(parents=True, exist_ok=True)
        data = _get(f"/api/photos/{pid}/preview", "preview for G0 body detection")
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    return p


def faces_of(con, pids):
    out = defaultdict(list)
    for i in range(0, len(pids), 900):
        ch = pids[i:i + 900]
        for fid, pid, per, src, t, r, b, l, det in con.execute(
                f"SELECT id, photo_id, person_id, match_source, bbox_top, bbox_right, bbox_bottom, "
                f"bbox_left, det_score FROM faces WHERE photo_id IN ({','.join('?' * len(ch))})", ch):
            out[pid].append({"face_id": fid, "person": per if src in TRUSTED else None,
                             "source": src, "bbox": [t, r, b, l], "det": det})
    return out


def _kp_model(dev):
    import torchvision
    w = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    return torchvision.models.detection.keypointrcnn_resnet50_fpn(
        weights=w, box_score_thresh=BODY_MIN_SCORE).to(dev).eval()


def keypoint_link(faces, bodies, kps):
    """{face_id: body index or None}, plus contested face ids. A face links to a body only when
    (1) the head-top rule (`person_boxes.link_faces`) gives it that body and (2) exactly one
    keypoint detection has a head keypoint inside the face box and that detection IS that body
    (IoU >= 0.5). Two keypoint owners, or an owner that is another body -> contested."""
    from photosearch import person_boxes as PBX
    head = PBX.link_faces(faces, bodies)
    link, contested = {}, set()
    for f in faces:
        t, r, b, l = f["bbox"]
        owners = [k for k in kps if any(l <= x <= r and t <= y <= b for x, y in k["head"])]
        hb = head.get(f["face_id"])
        if len(owners) >= 2:
            contested.add(f["face_id"])
            continue
        if len(owners) == 1 and hb is not None:
            if ID._iou(owners[0]["bbox"], bodies[hb]["bbox"]) >= 0.5:
                link[f["face_id"]] = hb
            else:
                contested.add(f["face_id"])
        elif len(owners) == 1 and hb is None:
            # the head-top rule found no body: does the keypoint owner match one?
            pass
    return link, contested


def cmd_evidence(a):
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from photosearch import person_boxes as PBX
    con = ro(a.db)
    photos = soccer_session_photos(con)
    G0.mkdir(parents=True, exist_ok=True)
    out_p = G0 / f"{a.out}.jsonl"
    done = set()
    if out_p.exists():
        done = {json.loads(line)["photo_id"] for line in out_p.open()}
    todo = sorted(p for p in photos if p not in done)
    if a.sample_only:
        keep = {int(p) for p in _jload(HERE / "sample.json", {})}
        todo = [p for p in todo if p in keep]
    if a.limit:
        todo = todo[:a.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    det_model, person = PBX._get_model()
    kpm = _kp_model(dev)
    fall = faces_of(con, todo)
    t0, n = time.time(), 0
    with out_p.open("a") as fh:
        for pid in todo:
            folder, d, W, H = photos[pid]
            try:
                im = Image.open(preview(pid)).convert("RGB")
            except Exception as e:                      # noqa: BLE001 - record and move on
                fh.write(json.dumps({"photo_id": pid, "error": f"{type(e).__name__}: {e}"}) + "\n")
                continue
            s = (W / float(im.width)) if W else 1.0      # preview px -> oriented full-res px
            x = TF.to_tensor(im).to(dev)
            with torch.no_grad():
                o = det_model([x])[0]
                k = kpm([x])[0]
            raw = [{"bbox": [round(y1 * s), round(x2 * s), round(y2 * s), round(x1 * s)], "score": round(float(sc), 4)}
                   for (x1, y1, x2, y2), lab, sc in zip(o["boxes"].tolist(), o["labels"].tolist(), o["scores"].tolist())
                   if lab == person and sc >= BODY_MIN_SCORE]
            kept, _ = PBX.dedupe(raw)
            bodies = sorted((raw[i] for i in kept), key=lambda b: (b["bbox"][3], b["bbox"][0]))
            kps = [{"bbox": [y1 * s, x2 * s, y2 * s, x1 * s],
                    "head": [(kp[0] * s, kp[1] * s) for j, (kp, sc) in enumerate(zip(kk, ks)) if j in HEAD_KP and sc >= KP_MIN]}
                   for (x1, y1, x2, y2), kk, ks in zip(k["boxes"].tolist(), k["keypoints"].tolist(),
                                                       k["keypoints_scores"].tolist())]
            faces = fall.get(pid, [])
            link, contested = keypoint_link(faces, bodies, kps)
            by_body = {i: fid for fid, i in link.items()}
            rec = {"photo_id": pid, "date": d, "folder": folder, "W": W, "H": H, "pw": im.width,
                   "faces": faces, "contested": sorted(contested),
                   "bodies": [dict(b, face_id=by_body.get(i)) for i, b in enumerate(bodies)],
                   "kp": [{"bbox": [round(v) for v in kk["bbox"]], "head": [[round(x), round(y)] for x, y in kk["head"]]}
                          for kk in kps]}
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                print(f"  {n}/{len(todo)} {(time.time() - t0) / n:.2f}s/photo", flush=True)
    print(f"evidence: {n} photos in {time.time() - t0:.0f}s -> {out_p}", flush=True)


# ---------------------------------------------------------------------------
# attrs: per-body VLM attributes (Step 1) on the evidence bodies
# ---------------------------------------------------------------------------

ATTRS_MIN_H = 0.15        # body height / frame height; smaller bodies carry no legible number


def load_evidence(names=("evidence-sample", "evidence")):
    out = {}
    for n in names:
        if not (G0 / f"{n}.jsonl").exists():
            continue
        for line in (G0 / f"{n}.jsonl").open():
            r = json.loads(line)
            if "error" not in r:
                out[r["photo_id"]] = r
    return out


def load_attrs(name="attrs"):
    """Every g0/<name>*.jsonl (parallel readers write their own file)."""
    out = {}
    for p in sorted(G0.glob(f"{name}*.jsonl")):
        for line in p.open():
            r = json.loads(line)
            out[(r["photo_id"], r["body"])] = r.get("attrs")
    return out


def attrs_targets(ev, days=None, min_h=ATTRS_MIN_H):
    keys = []
    for pid, r in sorted(ev.items()):
        if days and r["date"][:10] not in days:
            continue
        for i, b in enumerate(r["bodies"]):
            t, _, bo, _ = b["bbox"]
            if (bo - t) / float(r["H"]) >= min_h:
                keys.append((pid, i))
    return keys


def cmd_attrs(a):
    from PIL import Image
    from photosearch import body_attrs as BA
    from photosearch.describe import UnusableAnswer
    os.environ.setdefault("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL", "http://localhost:1234/v1")
    ev = load_evidence()
    have = load_attrs("attrs")       # every reader's file: never read a body twice
    days = set(a.days.split(",")) if a.days else None
    keys = [k for k in attrs_targets(ev, days, a.min_h) if k not in have]
    if a.select == "anchors":
        # Only bodies that can be kit anchors (identified by a face or a burst track) or kit
        # targets (the owner-labelled faceless bodies, every body on a sample photo).
        by_sess = _session_photos(ev, {})
        motion = motion_pairs(ev)
        want = set()
        for photos in by_sess.values():
            got = ID.identify_session(photos, enabled=("face", "burst"), motion=motion)
            want |= {k for k in got if k is not None}
        sample = {int(p) for p in _jload(HERE / "sample.json", {})}
        want |= {(pid, i) for pid in sample if pid in ev for i in range(len(ev[pid]["bodies"]))}
        keys = [k for k in keys if k in want]
    if a.min_pid:
        keys = [k for k in keys if k[0] >= a.min_pid]
    if a.priority:
        pri = {int(x) for x in Path(a.priority).read_text().split()}
        keys.sort(key=lambda k: (k[0] not in pri, k))
    print(f"attrs: {len(keys)} bodies to read ({len(have)} cached)", flush=True)
    if a.limit:
        keys = keys[:a.limit]
    t0, n, fails, cur_pid, im = time.time(), 0, 0, None, None
    with (G0 / f"{a.out}.jsonl").open("a") as fh:
        for pid, i in keys:
            r = ev[pid]
            if pid != cur_pid:
                im = Image.open(preview(pid)).convert("RGB")
                cur_pid = pid
            s = im.width / float(r["W"])
            bb = [v * s for v in r["bodies"][i]["bbox"]]
            try:
                at = BA.read(im, bb, a.model, base)
                rec = {"photo_id": pid, "body": i, "attrs": at}
                fails = 0
            except UnusableAnswer as e:
                rec = {"photo_id": pid, "body": i, "attrs": None, "error": str(e)[:200]}
            except Exception as e:                      # noqa: BLE001 - transport: stop, never cache
                fails += 1
                print(f"  ! {pid}/{i}: {type(e).__name__}: {e}", flush=True)
                if fails >= 5:
                    sys.exit("5 transport failures in a row: stopping")
                continue
            if rec.get("attrs") is None and "error" not in rec:
                continue                                 # no answer: not cached
            fh.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100 == 0:
                fh.flush()
                print(f"  {n}/{len(keys)} {(time.time() - t0) / n:.2f}s/body", flush=True)
    print(f"attrs: {n} bodies in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# g0: the identity eval
# ---------------------------------------------------------------------------

def _session_photos(ev, attrs, who_by_body=None, use_attrs=True):
    """{session: [photo dicts for identify.identify_session]}."""
    sess = ID.sessions([(pid, r["folder"], r["date"]) for pid, r in ev.items()])
    out = defaultdict(list)
    for pid, r in ev.items():
        if pid not in sess:
            continue
        fp = {f["face_id"]: f["person"] for f in r["faces"]}
        bodies = []
        for i, b in enumerate(r["bodies"]):
            bodies.append({"bbox": b["bbox"],
                           "face_person": fp.get(b["face_id"]) if b.get("face_id") else None,
                           "label_person": (who_by_body or {}).get((pid, i)),
                           "attrs": attrs.get((pid, i)) if use_attrs else None})
        out[sess[pid]].append({"photo_id": pid, "time": ID.parse_time(r["date"]), "bodies": bodies, "W": r["W"],
                               "split": split_of(r["date"])})
    return out


def motion_pairs(ev, min_peak=None):
    """Camera shift between every pair of photos <= BURST_S apart in one folder,
    cached in g0/motion.json: {"a-b": [scale, dx, dy, peak]}."""
    import numpy as np
    from PIL import Image
    cache_p = G0 / "motion.json"
    cache = _jload(cache_p, {})
    by_folder = defaultdict(list)
    for pid, r in ev.items():
        t = ID.parse_time(r["date"])
        if t:
            by_folder[r["folder"]].append((t, pid))
    gcache = {}

    def gray(pid):
        if pid not in gcache:
            im = Image.open(G0 / "previews" / f"{pid}.jpg").convert("L")
            gcache[pid] = np.asarray(im.resize((512, round(im.height * 512 / im.width))), dtype=np.float32)
            if len(gcache) > 64:
                gcache.pop(next(iter(gcache)))
        return gcache[pid]
    new = 0
    for lst in by_folder.values():
        lst.sort()
        for i, (t, a) in enumerate(lst):
            j = i + 1
            while j < len(lst) and (lst[j][0] - t).total_seconds() <= ID.BURST_S:
                b = lst[j][1]
                k = f"{a}-{b}"
                if k not in cache:
                    sc = ev[a]["W"] / 512.0
                    cache[k] = [round(v, 4) for v in ID.phase_shift(gray(a), gray(b), sc)]
                    new += 1
                j += 1
    if new:
        _jsave(cache_p, cache)
    out = {}
    for k, (sc, dx, dy, peak) in cache.items():
        if peak < (ID.MOTION_MIN_PEAK if min_peak is None else min_peak):
            continue
        a, b = map(int, k.split("-"))
        out[(a, b)] = (1.0, dx, dy)
        out[(b, a)] = (1.0, -dx, -dy)
    return out


def _fold(key, k=10):
    return int(hashlib.sha1(f"{key[0]}-{key[1]}".encode()).hexdigest(), 16) % k


def leave_face_out(by_sess, enabled, folds=10, **kw):
    """Hide each face-linked body's face (in `folds` random folds) and score what the
    other tiers say about it against the face label."""
    res = defaultdict(lambda: Counter())
    wrong = []
    for sk, photos in by_sess.items():
        split = photos[0]["split"]
        faced = [(p["photo_id"], i) for p in photos for i, b in enumerate(p["bodies"])
                 if b["face_person"] is not None]
        for f in range(folds):
            hide = {k for k in faced if _fold(k, folds) == f}
            if not hide:
                continue
            got = ID.identify_session(photos, enabled=enabled, hide_faces=hide, **kw)
            truth = {(p["photo_id"], i): b["face_person"] for p in photos for i, b in enumerate(p["bodies"])}
            for k in hide:
                res[(split, "all")]["hidden"] += 1
                g = got.get(k)
                if not g:
                    continue
                ok = g["person"] == truth[k]
                res[(split, g["tier"])]["right" if ok else "wrong"] += 1
                if not ok:
                    wrong.append((split, g["tier"], k, truth[k], g["person"], g["witness"]))
    return res, wrong


def match_who(ev):
    """Owner who-labels -> {(photo_id, evidence body index): who} by IoU with the sheet's bodies."""
    saved = _jload(WHO / "sheet" / "saved-labels.json", {})
    sheet_bodies = _jload(HERE / "bodies.json", {})["photos"]
    out, unmatched = {}, []
    for key, v in saved.items():
        if not v.get("who"):
            continue
        pid_s, letter = key.rsplit("-", 1)
        pid = int(pid_s)
        sb = next(b for b in sheet_bodies[pid_s] if b["b"] == letter)
        r = ev.get(pid)
        if not r:
            unmatched.append(key)
            continue
        best = max(range(len(r["bodies"])), key=lambda i: ID._iou(sb["bbox"], r["bodies"][i]["bbox"]), default=None)
        if best is None or ID._iou(sb["bbox"], r["bodies"][best]["bbox"]) < 0.5:
            unmatched.append(key)
            continue
        out[(pid, best)] = v["who"]
    return out, unmatched


def owner_eval(by_sess, who, enabled, **kw):
    res = defaultdict(Counter)
    rows = []
    for sk, photos in by_sess.items():
        mine = [(p["photo_id"], i) for p in photos for i in range(len(p["bodies"])) if (p["photo_id"], i) in who]
        if not mine:
            continue
        got = ID.identify_session(photos, enabled=enabled, **kw)
        split = photos[0]["split"]
        for k in mine:
            w = who[k]
            g = got.get(k)
            named = isinstance(w, int)
            if w == "unknown":
                res[split]["unknown"] += 1
                continue
            res[split]["named" if named else "not-ours"] += 1
            if not g:
                continue
            ok = named and g["person"] == w
            res[split][f"{g['tier']}-{'right' if ok else 'wrong'}"] += 1
            rows.append((split, k, w, g["person"], g["tier"], ok))
    return res, rows


def cmd_g0(a):
    ev = load_evidence()
    attrs = load_attrs(a.attrs)
    by_sess = _session_photos(ev, attrs)
    kw = {"motion": motion_pairs(ev)}
    if a.no_kit_groups:
        kw["kit_groups"] = {}
    out = {"photos": len(ev), "sessions": len(by_sess), "attrs": len(attrs), "variants": {}}
    for name, enabled in (("burst", ("face", "burst")), ("burst+kit", ("face", "burst", "kit"))):
        res, wrong = leave_face_out(by_sess, enabled, **kw)
        rows = {f"{s}/{t}": dict(c) for (s, t), c in sorted(res.items())}
        out["variants"][name] = {"leave_face_out": rows, "wrong": [list(map(str, w)) for w in wrong[:40]]}
        print(f"\n== leave-face-out, tiers {name}")
        for k, c in sorted(rows.items()):
            r_, w_ = c.get("right", 0), c.get("wrong", 0)
            rate = f"{w_ / (r_ + w_):.2%}" if r_ + w_ else "-"
            print(f"  {k:22s} {dict(c)}  wrong rate {rate}")
    who, unmatched = match_who(ev)
    print(f"\nowner who-labels: {len(who)} matched to evidence bodies, {len(unmatched)} unmatched")
    if who:
        for name, enabled in (("burst", ("face", "burst")), ("burst+kit", ("face", "burst", "kit"))):
            res, rows = owner_eval(by_sess, who, enabled, **kw)
            out["variants"][name]["owner"] = {k: dict(v) for k, v in res.items()}
            out["variants"][name]["owner_rows"] = [list(map(str, r)) for r in rows]
            print(f"== owner-labelled faceless bodies, tiers {name}")
            for k, v in sorted(res.items()):
                print(f"  {k}: {dict(v)}")
            for r in rows:
                if not r[-1]:
                    print("   WRONG", r)
    _jsave(G0 / f"report-{a.attrs}.json", out)


# ---------------------------------------------------------------------------
# G1: grounded describe prototype on the labelled sample
# ---------------------------------------------------------------------------

G1_VARIANTS = {
    # name: (shape, length line, hints)
    "two-L": ("two", True, "true"),
    "two": ("two", False, "true"),
    "one-L": ("one", True, "true"),
    "two-L-nohints": ("two", True, "none"),
    "two-L-flip": ("two", True, "flip"),        # hint-injection test: the gender hint flipped
    "plain": ("plain", False, "none"),          # today's describe prompt, clean photo only (baseline)
}


def _clean_b64(path, max_px=1024):
    im = _open_oriented(path)
    s = min(1.0, max_px / float(max(im.size)))
    im = im.resize((round(im.width * s), round(im.height * s)))
    return base64.b64encode(_jpeg(im, 90)).decode()


def g1_inputs(con, pid, bodies_all, ev):
    """Targets (production body_targets), each with its sheet body letter, trusted person,
    gender/age and whether the keypoint link contests it."""
    from photosearch import people_bind as PB
    from photosearch import people_bind_store as store
    inp = store.claim_input(con, pid)
    if not inp:
        return None
    tg = PB.body_targets(inp["faces"], bodies_all.get(str(pid), []), inp["image_height"])
    letter = {tuple(b["bbox"]): b["b"] for b in bodies_all.get(str(pid), [])}
    year = int((con.execute("SELECT date_taken FROM photos WHERE id=?", (pid,)).fetchone()[0] or "0")[:4] or 0)
    contested = set((ev.get(pid) or {}).get("contested") or [])
    # the keypoint link: a face counts as linked only if the evidence pass linked it too
    kp_linked = {b["face_id"] for b in (ev.get(pid) or {}).get("bodies", []) if b.get("face_id")}
    out = []
    for n, t in enumerate(tg, 1):
        rec = {"n": n, "bbox": t["bbox"], "face_id": t["face_id"], "letter": letter.get(tuple(t["bbox"])),
               "person": None, "gender": None, "age": None, "hintable": False}
        if t["face_id"]:
            row = con.execute(
                "SELECT f.person_id, f.match_source, pe.gender, pe.birth_year FROM faces f "
                "LEFT JOIN persons pe ON pe.id = f.person_id WHERE f.id = ?", (t["face_id"],)).fetchone()
            if row and row[0] and row[1] in TRUSTED:
                rec.update(person=row[0], gender=row[2], age=(year - row[3]) if row[3] and year else None)
                rec["hintable"] = (t["face_id"] not in contested and t["face_id"] in kp_linked
                                   and bool(row[2] or row[3]))
        out.append(rec)
    return {"description": inp["description"], "targets": out, "W": inp["image_width"], "H": inp["image_height"]}


def _flip_target(targets):
    """The hint to flip: the largest hintable target with a gender."""
    hs = [t for t in targets if t["hintable"] and t["gender"]]
    if not hs:
        return None
    return max(hs, key=lambda t: (t["bbox"][2] - t["bbox"][0]) * (t["bbox"][1] - t["bbox"][3]))["n"]


def cmd_g1_run(a):
    from photosearch import describe as D
    from photosearch import grounded_describe as GD
    from photosearch import model_eval
    from photosearch import people_bind as PB
    os.environ.setdefault("PHOTOSEARCH_TEXT_LLM_URL", "http://localhost:1234/v1")
    os.environ.setdefault("PHOTOSEARCH_LLM_REASONING_EFFORT", "none")
    model_eval.pin_role_model("describe", a.model)
    shape, length, hints_mode = G1_VARIANTS[a.variant]
    con = ro(a.db)
    sample = _jload(HERE / "sample.json", {})
    bodies_all = _jload(HERE / "bodies.json", {})["photos"]
    ev = load_evidence()
    runp = G1 / "runs" / f"{a.variant}.json"
    run = _jload(runp, {"spec": {"model": a.model, "variant": a.variant}, "preds": {}})
    if run["spec"] != {"model": a.model, "variant": a.variant}:
        sys.exit(f"{runp} was made with {run['spec']}: use another variant name")
    pids = sorted((p for p, m in sample.items() if a.split == "all" or m["split"] == a.split), key=int)
    if a.redo_errors:
        for p_ in [p_ for p_, r_ in run["preds"].items() if r_.get("error")]:
            del run["preds"][p_]
    if a.only:
        pids = [p for p in pids if p in set(a.only.split(","))]
    t_all, n_new, fails = time.time(), 0, 0
    for pid_s in pids:
        if pid_s in run["preds"]:
            continue
        pid = int(pid_s)
        g = g1_inputs(con, pid, bodies_all, ev)
        if not g:
            continue
        tg = g["targets"]
        flip = _flip_target(tg) if hints_mode == "flip" else None
        if hints_mode == "flip" and flip is None:
            continue
        hints = []
        if hints_mode in ("true", "flip"):
            for t in tg:
                if t["hintable"]:
                    gen = t["gender"]
                    if t["n"] == flip:
                        gen = {"male": "female", "female": "male"}.get(gen, gen)
                    h = GD.hint_line(t["n"], gen, t["age"])
                    if h:
                        hints.append(h)
        img = str(original(pid))
        if shape == "plain":
            images, prompt = [_clean_b64(img)], D.DESCRIBE_PROMPT
        else:
            overlay = PB.draw_boxes(img, [t["bbox"] for t in tg])
            images = [_clean_b64(img), overlay] if shape == "two" else [overlay]
            prompt = GD.build_prompt(len(tg), shape, length, hints)
        t0 = time.time()
        rec = {"targets": tg, "hints": hints, "flip": flip}
        try:
            raw = D._ollama_chat_with_retry(model=a.model, messages=[{"role": "user", "content": prompt, "images": images}],
                                            role="describe", options={"temperature": 0.0, "num_ctx": 16384})
            fails = 0
        except D.TruncatedOutput as e:
            raw, rec["error"] = None, f"truncated: {e}"[:200]
        except Exception as e:                       # noqa: BLE001 - transport: never cached
            fails += 1
            print(f"  ! {pid}: {type(e).__name__}: {e}", flush=True)
            if fails >= 5:
                sys.exit("5 transport failures in a row: stopping")
            continue
        if raw is None and "error" not in rec:
            continue
        rec["secs"] = round(time.time() - t0, 2)
        rec["raw"] = raw
        if raw is not None and shape == "plain":
            try:
                rec.update(description=raw.strip(), spans=[])
                if not raw.strip():
                    raise D.UnusableAnswer("empty")
            except D.UnusableAnswer as e:
                rec["error"] = str(e)
        elif raw is not None:
            try:
                rec.update(GD.parse(raw, len(tg)))
            except D.UnusableAnswer as e:
                rec["error"] = str(e)[:200]
            if "error" in rec and ("stray markup" in rec["error"] or "overlay words" in rec["error"]):
                # One corrective turn (what production would do before failing the photo).
                msgs = [{"role": "user", "content": prompt, "images": images},
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": GD.FIX_MARKUP if "stray" in rec["error"] else GD.FIX_LEAK}]
                try:
                    raw2 = D._ollama_chat_with_retry(model=a.model, messages=msgs, role="describe",
                                                     options={"temperature": 0.0, "num_ctx": 16384})
                except Exception as e:                       # noqa: BLE001 - keep the first error
                    raw2 = None
                    rec["retry_error"] = f"{type(e).__name__}: {e}"[:200]
                if raw2:
                    rec["retried"], rec["raw_first"], rec["raw"] = True, raw, raw2
                    try:
                        rec.update(GD.parse(raw2, len(tg)))
                        rec.pop("error", None)
                    except D.UnusableAnswer as e:
                        rec["error"] = str(e)[:200]
                        try:
                            rec.update(GD.strip_marks(raw2, len(tg)))
                            rec["fallback"] = rec.pop("error")
                        except D.UnusableAnswer:
                            pass
        run["preds"][pid_s] = rec
        n_new += 1
        if n_new % 10 == 0:
            _jsave(runp, run)
            print(f"  {n_new} new, {(time.time() - t_all) / n_new:.1f}s/photo", flush=True)
    _jsave(runp, run)
    print(f"g1-run {a.variant}: {n_new} new photos -> {runp}")


CONFIRM = HERE / "g1confirm"
FEMALE_RE = re.compile(r"\b(girls?|wom[ae]n|she|her|hers|female|lady|ladies|daughter|mother|mom)\b", 2)
PEOPLE_CLASSES = {"boy": r"boys?", "girl": r"girls?", "child": r"child|children|kids?|toddlers?|bab(?:y|ies)",
                  "man": r"m[ae]n", "woman": r"wom[ae]n", "player": r"players?|teammates?|goalkeeper|goalie",
                  "adult": r"adults?|coach(?:es)?|referee|spectators?|parents?", "teen": r"teen(?:ager)?s?",
                  "people": r"people|persons?"}


def _anchor_key(pid_s, sp):
    return f"{pid_s}|{sp['phrase'].lower()}|{','.join(map(str, sp['boxes']))}"


def cmd_g1_sheet(a):
    """Owner's confirm page: every inline anchor of the given variants, pre-filled with the
    model's boxes; the owner marks right / the right box(es) / not a boxed person / ambiguous."""
    from photosearch import people_bind as PB
    variants = a.variants.split(",")
    runs = {v: _jload(G1 / "runs" / f"{v}.json", {}).get("preds", {}) for v in variants}
    sample = _jload(HERE / "sample.json", {})
    sheet = CONFIRM / "sheet"
    sheet.mkdir(parents=True, exist_ok=True)
    items = []
    pids = sorted({p for r in runs.values() for p in r if sample[p]["split"] in a.splits.split(",")}, key=int)
    for pid_s in pids:
        recs = {v: runs[v][pid_s] for v in variants if pid_s in runs[v] and "description" in runs[v][pid_s]}
        if not recs:
            continue
        tg = next(iter(recs.values()))["targets"]
        jp = sheet / f"{pid_s}.jpg"
        if not jp.exists():
            jp.write_bytes(base64.b64decode(PB.draw_boxes(str(original(int(pid_s))), [t["bbox"] for t in tg], max_px=1400)))
        anchors, seen = [], set()
        texts = []
        for v, r in recs.items():
            texts.append({"variant": v, "description": r["description"],
                          "spans": [[sp["start"], sp["end"]] for sp in r["spans"]]})
            for sp in r["spans"]:
                k = _anchor_key(pid_s, sp)
                if k not in seen:
                    seen.add(k)
                    anchors.append({"key": k, "phrase": sp["phrase"], "boxes": sp["boxes"]})
        if anchors:
            items.append({"pid": pid_s, "split": sample[pid_s]["split"], "n": len(tg), "texts": texts,
                          "anchors": anchors})
    key = hashlib.sha1(json.dumps([[i["pid"], [x["key"] for x in i["anchors"]]] for i in items]).encode()).hexdigest()[:10]
    (sheet / "index.html").write_text(_CONFIRM_HTML.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
                                      .replace("__KEY__", key))
    print(f"{sum(len(i['anchors']) for i in items)} anchors on {len(items)} photos -> {sheet / 'index.html'}")


_CONFIRM_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>confirm person marks</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--on:#1565c0;--onfg:#fff;--card:#f6f7f9;--bad:#c62828}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--on:#64b5f6;--onfg:#000;--card:#1f2228;--bad:#ef9a9a}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
main{max-width:1400px;margin:auto;padding:0 16px}
section{border-bottom:2px solid var(--line);padding:16px 0}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px;align-items:start}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:4px;cursor:zoom-in}
.desc{background:var(--card);padding:6px 10px;border-radius:4px;margin-bottom:6px}
.desc b{background:rgba(255,193,7,.35);border-radius:2px;padding:0 1px}
.mut{color:var(--mut);font-size:13px}
.a{border-top:1px solid var(--line);padding:6px 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:4px;padding:3px 9px;margin:2px;cursor:pointer}
button.on{background:var(--on);color:var(--onfg);border-color:var(--on)}
button.n{min-width:2.2em}
#stale{background:#b71c1c;color:#fff;padding:12px 16px;font-weight:700}
</style></head><body>
<header><b>Confirm person marks</b><span id="prog"></span>
 <label><input type="checkbox" id="todo" checked> only unfinished photos</label><span id="srv" class="mut"></span>
 <span class="mut">For each marked phrase: is it the person (people) in the box number(s) shown? If not, click the right number(s), or "not a boxed person" / "ambiguous".</span></header>
<main id="root"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "g1c-__KEY__";
let L = {};
const SERVED = location.protocol.startsWith("http");
let ready = false, timer = null;
function srv(m) { document.getElementById("srv").textContent = m; }
function save() {
  progress();
  if (!SERVED || !ready) return;
  clearTimeout(timer);
  timer = setTimeout(() => fetch("saved-labels.json", {method: "PUT",
      headers: {"Content-Type": "application/json", "X-Sheet-Key": KEY}, body: JSON.stringify(L)})
    .then(r => { if (r.status === 409) throw new Error("this page is out of date: RELOAD it");
                 if (!r.ok) throw new Error("HTTP " + r.status); srv("saved " + new Date().toLocaleTimeString()); })
    .catch(e => { srv("NOT saved (" + e.message + ")");
      if (/out of date/.test(e.message) && !document.getElementById("stale")) {
        const d = document.createElement("div"); d.id = "stale";
        d.textContent = "This page is out of date and is NOT saving. Reload it.";
        document.body.prepend(d); } }), 300);
}
// "fixing" = box numbers being picked: not judged until "use these boxes" turns it into "fix".
const judged = x => ["right", "fix", "none", "ambiguous"].includes((L[x.key] || {}).v);
const done = it => it.anchors.every(judged);
function progress() {
  const n = ITEMS.reduce((s, it) => s + it.anchors.filter(judged).length, 0);
  const t = ITEMS.reduce((s, it) => s + it.anchors.length, 0);
  document.getElementById("prog").textContent = `${n}/${t} marks judged · ${ITEMS.filter(done).length}/${ITEMS.length} photos done`;
}
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) { if (k === "on") e.addEventListener("click", v); else if (k === "cls") e.className = v; else e.setAttribute(k, v); }
  for (const k of kids) e.append(k);
  return e;
}
function set(it, sec, key, val) { L[key] = val; save(); const c = card(it); sec.replaceWith(c);
  if (document.getElementById("todo").checked && done(it)) { const n = c.nextElementSibling; c.remove(); if (n) n.scrollIntoView({block: "start"}); } }
function card(it) {
  const sec = el("section", {id: "p" + it.pid});
  sec.append(el("div", {cls: "mut"}, `photo ${it.pid} · ${it.n} boxes`));
  const img = el("img", {src: it.pid + ".jpg", loading: "lazy"});
  img.addEventListener("click", () => window.open(it.pid + ".jpg"));
  const right = el("div");
  for (const t of it.texts) {
    const d = el("div", {cls: "desc"}); let pos = 0;
    for (const [s, e] of t.spans) { d.append(t.description.slice(pos, s), el("b", {}, t.description.slice(s, e))); pos = e; }
    d.append(t.description.slice(pos));
    right.append(d);
  }
  for (const x of it.anchors) {
    const cur = L[x.key] || {};
    const row = el("div", {cls: "a"}, el("div", {}, "“", el("b", {}, x.phrase), `” → box ${x.boxes.join(", ")}`));
    row.append(el("button", {cls: cur.v === "right" ? "on" : "", on: () => set(it, sec, x.key, cur.v === "right" ? {} : {v: "right"})}, "Right"));
    const fix = (cur.v === "fix" || cur.v === "fixing") ? cur.boxes : [];
    row.append(el("span", {cls: "mut"}, " or the right box(es): "));
    for (let n = 1; n <= it.n; n++) {
      row.append(el("button", {cls: "n" + (fix.includes(n) ? " on" : ""), title: "the right box is " + n, on: () => {
        const b = fix.includes(n) ? fix.filter(z => z !== n) : fix.concat([n]).sort((p, q) => p - q);
        set(it, sec, x.key, b.length ? {v: "fixing", boxes: b} : {}); }}, String(n)));
    }
    if (cur.v === "fixing")
      row.append(el("button", {cls: "on", on: () => set(it, sec, x.key, {v: "fix", boxes: cur.boxes})}, "use these boxes ✓"));
    else if (cur.v === "fix")
      row.append(el("span", {cls: "mut"}, " ✓ saved"));
    row.append(el("button", {cls: cur.v === "none" ? "on" : "", on: () => set(it, sec, x.key, cur.v === "none" ? {} : {v: "none"})}, "not a boxed person"));
    row.append(el("button", {cls: cur.v === "ambiguous" ? "on" : "", on: () => set(it, sec, x.key, cur.v === "ambiguous" ? {} : {v: "ambiguous"})}, "ambiguous"));
    right.append(row);
  }
  sec.append(el("div", {cls: "grid"}, img, right));
  return sec;
}
function render() {
  const root = document.getElementById("root"); root.innerHTML = "";
  const todo = document.getElementById("todo").checked;
  for (const it of ITEMS) if (!(todo && done(it))) root.append(card(it));
  progress();
}
document.getElementById("todo").addEventListener("change", render);
render();
if (SERVED) fetch("saved-labels.json", {cache: "no-store"}).then(r => r.status === 404 ? null : r.json())
  .then(data => { if (data) L = data; ready = true; render(); srv("loaded from server"); })
  .catch(e => srv("could not load from server (" + e.message + "); NOT saving"));
</script></body></html>
"""


def _people_words(text):
    """The people concepts a text uses (boy, girl, child, man, ... ): people-word recall
    compares these, so "boys" -> "boy" is not a loss."""
    t = text.lower()
    return {c for c, pat in PEOPLE_CLASSES.items() if re.search(r"\b(?:" + pat + r")\b", t)}


def cmd_g1_report(a):
    """Automatic G1 numbers per variant (+ the owner's verdicts when present)."""
    from photosearch import grounded_describe as GD
    from photosearch import people_bind as PB
    con = ro(a.db)
    sample = _jload(HERE / "sample.json", {})
    bodies_all = _jload(HERE / "bodies.json", {})["photos"]
    verdicts = _jload(CONFIRM / "sheet" / "saved-labels.json", {})
    stored = {}
    out = {}
    for v in a.variants.split(","):
        preds = _jload(G1 / "runs" / f"{v}.json", {}).get("preds", {})
        c = Counter()
        lens, slens, secs = [], [], []
        recall_num = recall_den = 0
        for pid_s, r in preds.items():
            if sample[pid_s]["split"] not in a.splits.split(","):
                continue
            c["photos"] += 1
            if pid_s not in stored:
                stored[pid_s] = con.execute("SELECT description FROM photos WHERE id=?", (int(pid_s),)).fetchone()[0] or ""
            if "description" not in r:
                c["unusable"] += 1
                if "overlay words" in (r.get("error") or ""):
                    c["leak"] += 1
                continue
            secs.append(r.get("secs") or 0)
            d = r["description"]
            lens.append(len(d))
            slens.append(len(stored[pid_s]))
            sw, nw = _people_words(stored[pid_s]), _people_words(d)
            recall_den += len(sw)
            recall_num += len(sw & nw)
            if len(d) < 120:
                c["short<120"] += 1
            if r.get("raw") and "fallback" not in r and pid_s and not r.get("degenerate_checked"):
                try:
                    re_p = GD.parse(r["raw"], len(r["targets"]))
                    c["degenerate"] += bool(re_p["degenerate"])
                    c["count_mismatch_dropped"] += sum(1 for d_ in re_p["dropped"] if d_.get("why"))
                    r = dict(r, spans=re_p["spans"])          # score under the current guards
                except Exception:                              # noqa: BLE001 - counted elsewhere
                    pass
            c["spans"] += len(r["spans"])
            c["retried"] += bool(r.get("retried"))
            c["fallback_no_spans"] += bool(r.get("fallback"))
            c["no_spans"] += not r["spans"]
            c["dropped"] += len(r.get("dropped", []))
            tg = r["targets"]
            for sp in r["spans"]:
                named = [tg[b - 1] for b in sp["boxes"]]
                is_named = all(t["person"] for t in named)
                c["spans_named" if is_named else "spans_generic"] += 1
                bare = PB.is_bare_phrase(sp["phrase"])
                bare_ok = (not bare) or len(sp["boxes"]) == len(bodies_all.get(pid_s, []))
                vv = verdicts.get(_anchor_key(pid_s, sp), {}).get("v")
                if vv:
                    c["judged"] += 1
                    right = vv == "right"
                    c[f"{'right' if right else ('wrong-' + vv)}"] += 1
                    if is_named:
                        c["named-" + ("right" if right else "wrong")] += 1
                        if bare_ok:
                            c["named-bare-rule-" + ("right" if right else "wrong")] += 1
            if r.get("flip"):
                fs = [sp for sp in r["spans"] if r["flip"] in sp["boxes"]]
                c["flip_photos"] += 1
                if any(FEMALE_RE.search(sp["phrase"]) for sp in fs):
                    c["flip_obeyed_span"] += 1
                if FEMALE_RE.search(d):
                    c["flip_female_anywhere"] += 1
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else None
        out[v] = dict(c, len_median=med(lens), stored_len_median=med(slens), s_per_photo=med(secs),
                      people_word_recall=round(recall_num / recall_den, 3) if recall_den else None)
        print(f"== {v} ({a.splits}): " + json.dumps(out[v], sort_keys=True))
    key = _jload(PAIR / "key.json", None)
    votes = _jload(PAIR / "sheet" / "saved-labels.json", {})
    if key and votes:
        pc = Counter()
        for pid_s, v in votes.items():
            if not v.get("v"):
                continue
            a_is = key["A_is"][pid_s]
            pc["same" if v["v"] == "same" else ("new" if (v["v"] == "A") == (a_is == "new") else "stored")] += 1
        out["pairwise"] = {"variant": key["variant"], **pc}
        print(f"== blind pairwise {key['variant']} vs stored: {dict(pc)}")
    _jsave(G1 / f"report-{a.splits.replace(',', '+')}.json", out)


PAIR = HERE / "g1pair"


def cmd_g1_pair_sheet(a):
    """Blind pairwise page: today's stored description vs the variant's, A/B shuffled."""
    import random
    con = ro(a.db)
    preds = _jload(G1 / "runs" / f"{a.variant}.json", {})["preds"]
    sample = _jload(HERE / "sample.json", {})
    pids = sorted((p for p, r in preds.items() if "description" in r and sample[p]["split"] == "dev"), key=int)
    rnd = random.Random(20261101)
    pick = sorted(rnd.sample(pids, min(a.n, len(pids))), key=int)
    sheet = PAIR / "sheet"
    sheet.mkdir(parents=True, exist_ok=True)
    items, key_map = [], {}
    for pid_s in pick:
        jp = sheet / f"{pid_s}.jpg"
        if not jp.exists():
            im = _open_oriented(original(int(pid_s)))
            sc = 1400 / float(max(im.size))
            jp.write_bytes(_jpeg(im.resize((round(im.width * sc), round(im.height * sc))), 85))
        stored = con.execute("SELECT description FROM photos WHERE id=?", (int(pid_s),)).fetchone()[0]
        new = preds[pid_s]["description"]
        flip = rnd.random() < 0.5
        items.append({"pid": pid_s, "A": new if flip else stored, "B": stored if flip else new})
        key_map[pid_s] = "new" if flip else "stored"        # who is A
    _jsave(PAIR / "key.json", {"variant": a.variant, "A_is": key_map})
    k = hashlib.sha1(json.dumps(pick).encode()).hexdigest()[:10]
    (sheet / "index.html").write_text(_PAIR_HTML.replace("__ITEMS__", json.dumps(items).replace("</", "<\\/")).replace("__KEY__", k))
    print(f"{len(items)} pairs -> {sheet / 'index.html'}")


_PAIR_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>which description is better?</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--on:#1565c0;--onfg:#fff;--card:#f6f7f9}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--on:#64b5f6;--onfg:#000;--card:#1f2228}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
main{max-width:1300px;margin:auto;padding:0 16px} section{border-bottom:2px solid var(--line);padding:16px 0}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px} @media (max-width:800px){.grid{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:4px} .d{background:var(--card);padding:8px 10px;border-radius:4px;margin:6px 0}
.mut{color:var(--mut);font-size:13px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:4px;padding:4px 10px;margin:2px;cursor:pointer}
button.on{background:var(--on);color:var(--onfg);border-color:var(--on)}
</style></head><body>
<header><b>Which description is better for finding this photo?</b><span id="prog"></span><span id="srv" class="mut"></span>
<span class="mut">Judge accuracy first (a wrong detail is worse than a missing one), then usefulness. Names never appear here.</span></header>
<main id="root"></main>
<script>
const ITEMS = __ITEMS__; const KEY = "pair-__KEY__"; let L = {};
const SERVED = location.protocol.startsWith("http"); let ready = false, timer = null;
function srv(m) { document.getElementById("srv").textContent = m; }
function save() { progress(); if (!SERVED || !ready) return; clearTimeout(timer);
  timer = setTimeout(() => fetch("saved-labels.json", {method: "PUT", headers: {"Content-Type": "application/json", "X-Sheet-Key": KEY}, body: JSON.stringify(L)})
   .then(r => { if (!r.ok) throw new Error("HTTP " + r.status + (r.status === 409 ? ": reload the page" : "")); srv("saved " + new Date().toLocaleTimeString()); })
   .catch(e => srv("NOT saved (" + e.message + ")")), 300); }
function progress() { document.getElementById("prog").textContent = `${ITEMS.filter(i => (L[i.pid] || {}).v).length}/${ITEMS.length} judged`; }
function el(tag, attrs, ...kids) { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) { if (k === "on") e.addEventListener("click", v); else if (k === "cls") e.className = v; else e.setAttribute(k, v); }
  for (const k of kids) e.append(k); return e; }
function card(it) { const cur = (L[it.pid] || {}).v; const sec = el("section");
  const row = el("div"); for (const [k, lab] of [["A", "A is better"], ["same", "About the same"], ["B", "B is better"]])
    row.append(el("button", {cls: cur === k ? "on" : "", on: () => { L[it.pid] = cur === k ? {} : {v: k}; save(); sec.replaceWith(card(it)); }}, lab));
  sec.append(el("div", {cls: "grid"}, el("img", {src: it.pid + ".jpg", loading: "lazy"}),
    el("div", {}, el("div", {cls: "d"}, el("b", {}, "A: "), it.A), el("div", {cls: "d"}, el("b", {}, "B: "), it.B), row)));
  return sec; }
function render() { const root = document.getElementById("root"); root.innerHTML = ""; for (const it of ITEMS) root.append(card(it)); progress(); }
render();
if (SERVED) fetch("saved-labels.json", {cache: "no-store"}).then(r => r.status === 404 ? null : r.json())
  .then(d => { if (d) L = d; ready = true; render(); srv("loaded from server"); }).catch(e => srv("could not load (" + e.message + "); NOT saving"));
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("who-sheet")
    w.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"), required=not os.environ.get("PHOTOSEARCH_DB"))
    e = sub.add_parser("evidence")
    e.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"), required=not os.environ.get("PHOTOSEARCH_DB"))
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--out", default="evidence")
    e.add_argument("--sample-only", action="store_true")
    at = sub.add_parser("attrs")
    at.add_argument("--model", default="qwen/qwen3.5-9b")
    at.add_argument("--days", default="", help="comma-separated YYYY-MM-DD shoot days (default all)")
    at.add_argument("--min-h", type=float, default=ATTRS_MIN_H)
    at.add_argument("--priority", default="", help="file of photo ids to read first")
    at.add_argument("--limit", type=int, default=0)
    at.add_argument("--out", default="attrs")
    at.add_argument("--select", default="anchors", choices=["anchors", "all"])
    at.add_argument("--min-pid", type=int, default=0)
    g = sub.add_parser("g0")
    g.add_argument("--attrs", default="attrs")
    g.add_argument("--no-kit-groups", action="store_true")
    r1 = sub.add_parser("g1-run")
    r1.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"), required=not os.environ.get("PHOTOSEARCH_DB"))
    r1.add_argument("--variant", required=True, choices=sorted(G1_VARIANTS))
    r1.add_argument("--model", default="qwen/qwen3.5-9b")
    r1.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    r1.add_argument("--only", default="")
    r1.add_argument("--redo-errors", action="store_true")
    gs = sub.add_parser("g1-sheet")
    gs.add_argument("--variants", required=True)
    gs.add_argument("--splits", default="dev")
    gr = sub.add_parser("g1-report")
    gr.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"), required=not os.environ.get("PHOTOSEARCH_DB"))
    gr.add_argument("--variants", required=True)
    gr.add_argument("--splits", default="dev")
    gp = sub.add_parser("g1-pair-sheet")
    gp.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"), required=not os.environ.get("PHOTOSEARCH_DB"))
    gp.add_argument("--variant", required=True)
    gp.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    {"g1-run": cmd_g1_run, "g1-pair-sheet": cmd_g1_pair_sheet, "g1-sheet": cmd_g1_sheet, "g1-report": cmd_g1_report, "who-sheet": cmd_who_sheet, "evidence": cmd_evidence, "attrs": cmd_attrs, "g0": cmd_g0}[a.cmd](a)


if __name__ == "__main__":
    main()

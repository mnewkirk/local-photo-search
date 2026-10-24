#!/usr/bin/env python
"""Video eval — pick the main-subject / describe / quality configs on holdout
labels before anything ships (docs/plans/video-support.md, step 4).

Same rules as every model eval here (docs/plans/model-eval-harnesses.md):
the DB is only opened read-only; labels and run caches are FILES under
PHOTOSEARCH_MODEL_EVAL_DIR (video/, video-*/), never DB rows; `--model` pins
the role env var so the model that ran is the one recorded; every LLM call
runs inside model_eval.Recorder, a TransportError is never cached, a
context-overflow 400 is cached as `rejected` (it screens the config out);
originals come from a local cache so neither runs nor labelling need the NAS;
the labelling page (/eval/models, tabs "Video subject" / "Video quality") is
blind. Tunables are chosen on the DEV split; only HOLDOUT numbers decide
winners and gates. Ratios under 3 observations print n<3.

Subcommands:

  sample           stratified ~90 videos + 40 probe-check files, seeded
                   dev/holdout split, 15-video recheck set:
                       python evals/video_eval.py sample --db photo_index.db.local
  fetch-originals  cache every sampled original locally, once:
                       python evals/video_eval.py fetch-originals --server http://<nas>:8000
                       python evals/video_eval.py fetch-originals --db <db> --local-root /mnt/photos
  run-faces        main subject. F0/F1 = faces only (needs the detection
                   cache, filled by video_scan on first use); S1/S2 = VLM
                   grounding per segment keyframe + face join:
                       python evals/video_eval.py run-faces --db <db> --variant F1 --approach F1
                       python evals/video_eval.py run-faces --db <db> --variant S1 --approach S1 \
                           --grounding qwen25vl --model qwen2.5-vl-7b-instruct
  run-describe     D0 poster-only | D1 multi-image | D2 storyboard | D3 per-frame + summary:
                       python evals/video_eval.py run-describe --variant D0 --approach D0 --model qwen/qwen3.5-9b
                       python evals/video_eval.py run-describe --variant D1-q9b --approach D1 --model qwen/qwen3.5-9b --k 6 --px 768
  run-quality      Qm (measured, from the scan) | Qv (VLM rubric over K frames):
                       python evals/video_eval.py run-quality --variant Qm --approach Qm
                       python evals/video_eval.py run-quality --variant Qv-minicpm --approach Qv --model minicpm-v-4_5
  disputes         the facts each video's descriptions disagree on (comparer
                   model, not a candidate), and same-subject judgements for
                   free-text subject labels:
                       python evals/video_eval.py disputes --model gemma-4-26b-a4b
  report           every gate, holdout headline with dev beside it, CIs:
                       python evals/video_eval.py report [--db <db>]
  agreement        owner self-agreement on the recheck set (kappa < 0.4 =
                   fix the definition in eval_api.LABELLER_NOTES):
                       python evals/video_eval.py agreement
"""
from __future__ import annotations

import argparse
import base64
import inspect
import io
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import replace
from typing import Callable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import model_eval as me  # noqa: E402

PASS_GROUND = "video-grounding"
PASS_SUBJECT = "video-subject"
PASS_DESCRIBE = "video-describe"
PASS_QUALITY = "video-quality"
DET_TAG = "det-default"

SEED = 1
STRATA = {                      # the plan's table; live-paired gets a pairing question only
    "sports-long": 20, "sports-short": 10, "family-multi": 15, "single-subject": 10,
    "camera-nonsports": 8, "unpaired-short": 5, "no-people": 7, "old-era": 5,
    "live-paired": 10,
}
NEAR_MISS_N = 5
PROBE_CHECK_N = 20              # x2: probe-error + ok
RECHECK_N = 15

SPORT_WORDS = ("sport", "soccer", "baseball", "basketball", "volleyball", "tennis",
               "hockey", "football", "lacrosse", "softball", "futsal")
PHONE_RE = re.compile(r"iphone|pixel|galaxy|\bsm-|samsung|oneplus|moto|huawei|xiaomi", re.I)
CAMERA_RE = re.compile(r"ilce|kodak|pixpro|canon|nikon|gopro|dsc-|fujifilm|olympus|panasonic|dc-", re.I)

# Gates (plan step 4).
GATE_TOP = 0.80
GATE_PRESENCE_P = 0.95
GATE_SEG_ACC = 0.70
GATE_SHARE_MAE = 0.15
TIE_SEG = 0.05                  # < 5 points = tie, simpler/cheaper wins
F1_MARGIN = 0.05                # S1 must beat F1 by >= 5 pts on person strata
DESCRIBE_TIE = 0.05
DESCRIBE_GATE_TEMPORAL = 0.20
DESCRIBE_GATE_RIGHT = 0.05
DESCRIBE_GATE_P = 0.10
SCREEN_MAX_BAD = 0.10
QUALITY_P = 0.80
QUALITY_R = 0.50
SHARE_RETRIEVAL = 0.5
# Simplicity order for tie-breaks (earlier = simpler/cheaper).
SIMPLICITY = ("F0", "F1", "S1", "S2")


def log(*a):
    print(*a, flush=True)


# ==========================================================================
# Sample
# ==========================================================================

def _year(date):
    m = re.match(r"(\d{4})", str(date or ""))
    return int(m.group(1)) if m else None


def video_rows(conn):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'video_meta'").fetchone():
        raise SystemExit("This DB has no video_meta table (pre-v35). Sync a replica from a "
                         "NAS running the video registration (step 1) and the probe pass.")
    return [dict(r) for r in conn.execute(
        "SELECT p.id, p.filepath, p.filename, p.folder, p.camera_model, p.date_taken, "
        "vm.duration_s, vm.width, vm.height, vm.kind, vm.live_of_photo_id, "
        "vm.probe_error, vm.probe_version "
        "FROM photos p JOIN video_meta vm ON vm.photo_id = p.id "
        "WHERE p.media_type = 'video' ORDER BY p.id")]


def folder_facts(conn, folders):
    """Pre-scan facts about each folder's STILLS: sports share of their
    categories, named-person counts, whether any face was found. Never a
    candidate's own output."""
    out = {}
    for f in sorted({x for x in folders if x is not None}):
        cats = [r[0] for r in conn.execute(
            "SELECT categories FROM photos WHERE folder = ? AND media_type IS NULL "
            "AND categories IS NOT NULL", (f,))]
        sports = 0
        for c in cats:
            try:
                terms = json.loads(c) or []
            except (TypeError, ValueError):
                terms = []
            if any(any(w in str(t).lower() for w in SPORT_WORDS) for t in terms):
                sports += 1
        named = {int(pid): int(n) for pid, n in conn.execute(
            "SELECT f.person_id, COUNT(*) FROM faces f JOIN photos p ON p.id = f.photo_id "
            "WHERE p.folder = ? AND p.media_type IS NULL AND f.person_id IS NOT NULL "
            "GROUP BY f.person_id", (f,))}
        n_faces = conn.execute(
            "SELECT COUNT(*) FROM faces f JOIN photos p ON p.id = f.photo_id "
            "WHERE p.folder = ? AND p.media_type IS NULL", (f,)).fetchone()[0]
        n_stills = conn.execute("SELECT COUNT(*) FROM photos WHERE folder = ? "
                                "AND media_type IS NULL", (f,)).fetchone()[0]
        out[f] = {"sports_frac": sports / len(cats) if cats else 0.0, "named": named,
                  "n_faces": int(n_faces), "n_stills": int(n_stills)}
    return out


def _is_phone(r):
    return "phone-" in (r.get("folder") or "") or bool(PHONE_RE.search(r.get("camera_model") or ""))


def _is_camera(r):
    return bool(CAMERA_RE.search((r.get("camera_model") or "") + " " + (r.get("folder") or "")))


def classify(r, facts):
    """One stratum per video, in priority order, or None (not sampled)."""
    if r.get("probe_error") or r.get("probe_version") is None:
        return None
    if r.get("kind") == "live":
        return "live-paired"
    ff = facts.get(r.get("folder")) or {"sports_frac": 0, "named": {}, "n_faces": 0, "n_stills": 0}
    dur = float(r.get("duration_s") or 0)
    if ff["sports_frac"] >= 0.5:
        return "sports-long" if dur >= 20 else "sports-short"
    y = _year(r.get("date_taken"))
    short_edge = min(r.get("width") or 10 ** 6, r.get("height") or 10 ** 6)
    if y is not None and 2016 <= y <= 2018 and (short_edge <= 720 or "gphotos" in (r.get("folder") or "")):
        return "old-era"
    if dur <= 4:
        return "unpaired-short"
    if ff["n_stills"] and ff["n_faces"] == 0:
        return "no-people"
    if _is_camera(r) and not _is_phone(r):
        return "camera-nonsports"
    if _is_phone(r) and ff["named"]:
        total = sum(ff["named"].values())
        top = max(ff["named"].values())
        if top / total >= 0.6:
            return "single-subject"
        if len(ff["named"]) >= 2:
            return "family-multi"
    return None


def near_misses(conn, rows, n=NEAR_MISS_N, seed=SEED):
    """Clips the pairing did NOT pair although a same-stem still sits in the
    same folder 3–10 s away — the costly-error check (hiding a real video)."""
    from datetime import datetime
    out = []
    for r in rows:
        if r.get("kind") != "clip" or not r.get("date_taken"):
            continue
        stem = os.path.splitext(r["filename"] or "")[0]
        for sid, sd in conn.execute(
                "SELECT id, date_taken FROM photos WHERE folder = ? AND media_type IS NULL "
                "AND filename LIKE ?", (r["folder"], stem + ".%")):
            try:
                dt = abs((datetime.fromisoformat(str(sd)[:19]) -
                          datetime.fromisoformat(str(r["date_taken"])[:19])).total_seconds())
            except ValueError:
                continue
            if 3 <= dt <= 10:
                out.append((r, int(sid)))
                break
    random.Random(f"near-{seed}").shuffle(out)
    return out[:n]


def _entry(r, stratum, **extra):
    return {"photo_id": int(r["id"]), "stratum": stratum, "filename": r.get("filename"),
            "duration_s": r.get("duration_s"), "width": r.get("width"),
            "height": r.get("height"), **extra}


def draw_sample(conn, *, seed=SEED, strata=None):
    strata = dict(STRATA if strata is None else strata)
    rows = video_rows(conn)
    facts = folder_facts(conn, [r["folder"] for r in rows])
    pools: dict = {s: [] for s in strata}
    for r in rows:
        s = classify(r, facts)
        if s in pools:
            pools[s].append(r)
    photos, shortfall = [], {}
    for s, n in strata.items():
        pool = sorted(pools[s], key=lambda r: r["id"])
        random.Random(f"video-sample-{seed}-{s}").shuffle(pool)
        if len(pool) < n:
            shortfall[s] = n - len(pool)
        for r in pool[:n]:
            extra = {"live_still_id": r.get("live_of_photo_id")} if s == "live-paired" else {}
            photos.append(_entry(r, s, **extra))
    splits = me.holdout_split(photos, seed=seed)
    for p in photos:
        p["split"] = splits[p["photo_id"]]
    have = {p["photo_id"] for p in photos}
    for r, sid in near_misses(conn, [r for r in rows if r["id"] not in have], seed=seed):
        photos.append(_entry(r, "live-near-miss", split="check", live_still_id=sid))
    bad = sorted((r for r in rows if r.get("probe_error")), key=lambda r: r["id"])
    ok = sorted((r for r in rows if r.get("probe_version") is not None
                 and not r.get("probe_error")), key=lambda r: r["id"])
    rnd = random.Random(f"probe-check-{seed}")
    rnd.shuffle(bad)
    rnd.shuffle(ok)
    probe_check = [_entry(r, "probe-error", probe_error=r["probe_error"]) for r in bad[:PROBE_CHECK_N]] + \
                  [_entry(r, "probe-ok", probe_error=None) for r in ok[:PROBE_CHECK_N]]
    clips = sorted(p["photo_id"] for p in photos if me.is_labelled_clip(p))
    random.Random(f"video-recheck-{seed}").shuffle(clips)
    return {"seed": seed, "photos": photos, "probe_check": probe_check,
            "recheck": sorted(clips[:RECHECK_N]), "shortfall": shortfall,
            "seg_version": me.video_seg_version(),
            "unprobed": sum(1 for r in rows if r.get("probe_version") is None)}


# ==========================================================================
# Originals
# ==========================================================================

def all_sample_ids(include_probe=True):
    s = me.load_video_sample()
    ids = [p["photo_id"] for p in s["photos"]]
    if include_probe:
        ids += [p["photo_id"] for p in s.get("probe_check") or []]
    return list(dict.fromkeys(int(i) for i in ids))


def fetch_originals(*, server=me.DEFAULT_SERVER, local_root=None, conn=None, ids=None,
                    fetch=None, intent="Video eval: cache originals for labelling and runs"):
    """Stream each original once into the cache, or copy it from a local
    path (`local_root` + the DB's stored filepath). Returns (have, got, bad)."""
    have = got = bad = 0
    for pid in (ids if ids is not None else all_sample_ids()):
        if me.cached_original(pid) is not None:
            have += 1
            continue
        try:
            if local_root:
                row = conn.execute("SELECT filepath FROM photos WHERE id = ?", (pid,)).fetchone()
                if not row:
                    raise RuntimeError("no such photo in the DB")
                src = row[0] if os.path.isabs(row[0]) else os.path.join(local_root, row[0])
                me.copy_video_original(pid, src)
            else:
                (fetch or me.fetch_video_original)(pid, server, intent=intent)
            got += 1
            log(f"  + {pid}")
        except Exception as e:
            bad += 1
            log(f"  ! {pid}: {e}")
    log(f"[originals] {have} cached, {got} fetched, {bad} failed")
    return have, got, bad


# ==========================================================================
# Scan adapter (photosearch/video_scan.py — imported lazily)
# ==========================================================================

def _attr(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:
        pass
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _normalize_frames(frames):
    out = []
    for fr in frames or []:
        t = _attr(fr, "t")
        faces = []
        for d in _attr(fr, "faces", default=[]) or []:
            e = _attr(d, "embedding")
            faces.append({"bbox": [float(v) for v in _attr(d, "bbox")],
                          "det_score": float(_attr(d, "det_score", default=0.0) or 0.0),
                          "embedding": None if e is None else [float(v) for v in e],
                          "sharpness": _attr(d, "sharpness")})
        out.append({"t": float(t), "faces": faces})
    return out


def default_scanner(path, det_size=None):
    """(frames, meta) from video_scan.scan_video — the production decode +
    detection, so the eval measures what the worker will run."""
    from photosearch import video_scan
    kw = {}
    sig = inspect.signature(video_scan.scan_video)
    if "want_frames" in sig.parameters:
        kw["want_frames"] = False
    if det_size and "det_size" in sig.parameters:
        kw["det_size"] = det_size
    sd = video_scan.scan_video(str(path), **kw)
    meta = {"W": float(_attr(sd, "W", "width")), "H": float(_attr(sd, "H", "height")),
            "duration": float(_attr(sd, "duration", "duration_s")),
            "fps": _attr(sd, "fps"), "det_size": det_size,
            "scan_version": _attr(sd, "version", "scan_version"),
            "measures": _jsonable(_attr(sd, "measures", "quality", default=None)),
            "keyframes": _jsonable(_attr(sd, "keyframe_times", default=None))}
    return _normalize_frames(_attr(sd, "frames")), meta


def ensure_detections(pid, *, tag=DET_TAG, det_size=None, scanner=None):
    hit = me.load_detections(pid, tag)
    if hit is not None:
        return hit
    path = me.cached_original(pid)
    if path is None:
        raise FileNotFoundError(f"video {pid} not cached — run fetch-originals")
    frames, meta = (scanner or default_scanner)(path, det_size)
    me.save_detections(pid, tag, frames, meta)
    return frames, meta


# ==========================================================================
# Identity: the production strict matcher against a frozen snapshot
# ==========================================================================

def load_references(conn):
    """(person_ids, encodings) from face_references + manual labels, read from
    the frozen replica snapshot the --db names."""
    import struct
    import numpy as np
    pids, encs = [], []
    try:
        for pid, blob in conn.execute(
                "SELECT fr.person_id, e.encoding FROM face_references fr "
                "JOIN face_ref_encodings e ON e.ref_id = fr.id"):
            pids.append(int(pid))
            encs.append(struct.unpack(f"{len(blob) // 4}f", blob))
        for pid, blob in conn.execute(
                "SELECT f.person_id, e.encoding FROM faces f JOIN face_encodings e "
                "ON e.face_id = f.id WHERE f.person_id IS NOT NULL "
                "AND f.match_source = 'manual'"):
            pids.append(int(pid))
            encs.append(struct.unpack(f"{len(blob) // 4}f", blob))
    except Exception as e:      # no sqlite-vec loaded: identities stay unnamed
        log(f"[refs] cannot read reference encodings ({e}); identities stay unnamed")
    return pids, (np.asarray(encs, float) if encs else np.zeros((0, 512)))


def name_identity(encoding, refs, tolerance=None):
    from photosearch import faces
    pids, encs = refs
    if encoding is None or not pids:
        return None
    m = faces.match_face(list(encoding), encs.tolist(),
                         faces.MATCH_TOLERANCE if tolerance is None else tolerance)
    return pids[m[0][0]] if m else None


# ==========================================================================
# Predictions -> one key per label segment
# ==========================================================================
#
# A key is ["p", person_id] (named person), ["else"] (a person nobody named),
# ["s", label] (a non-person subject, normalized) or ["none"].

def _face_key(ix, names):
    if ix is None:
        return ["none"]
    pid = names.get(ix)
    return ["p", pid] if pid is not None else ["else"]


def _present_named(fa, names, min_presence_s=0.0):
    return sorted({names[i.ix] for i in fa.identities
                   if names.get(i.ix) is not None and i.presence_s > min_presence_s})


def face_item(approach, frames, meta, params, refs):
    from photosearch import video_analysis as va
    W, H, dur = meta["W"], meta["H"], meta["duration"]
    fa = (va.analyze_faces_f0 if approach == "F0" else va.analyze_faces)(frames, W, H, dur, params)
    names = {i.ix: name_identity(i.encoding, refs) for i in fa.identities}
    prod = me.video_production_segments(dur)
    keys = [_face_key(ix, names) for ix in va.segment_focus(fa, prod)]
    return fa, names, prod, keys


def subject_keys(analysis, names):
    out = []
    for p in analysis.primaries:
        if p is None:
            out.append(["none"])
            continue
        s = analysis.subjects[p]
        if s.face_ix is not None:
            out.append(_face_key(s.face_ix, names))
        elif s.category == "person":
            out.append(["else"])
        else:
            out.append(["s", s.label])
    return out


def predict(approach, pid, frames, meta, *, params, sparams=None, refs, grounding=None):
    """The cached run item for one video: per label segment a key, plus the
    named people present (faces layer)."""
    from photosearch import video_subjects as vs
    dur = meta["duration"]
    fa, names, prod, keys = face_item("F0" if approach == "F0" else "F1", frames, meta, params, refs)
    if approach in ("S1", "S2"):
        if grounding is None or grounding.get("status") != "ok":
            raise ValueError("no grounding for this video")
        sp = replace(sparams or vs.DEFAULT_SUBJECT_PARAMS, credit=approach)
        segs = [tuple(s) for s in grounding["segments"]]
        an = vs.score_subjects(segs, grounding["boxes"], faces=fa, W=meta["W"], H=meta["H"],
                               keyframe_times=grounding.get("keyframe_t"), params=sp)
        prod, keys = segs, subject_keys(an, names)
        if approach == "S2":
            keys = s2_keys(an, fa, names)
    label_segs = me.video_label_segments(dur)
    return {"approach": approach, "seg_version": me.video_seg_version(),
            "duration": dur, "label_segments": [list(s) for s in label_segs],
            "pred": me.coarsen_keys(prod, keys, label_segs),
            "present": _present_named(fa, names),
            "identities": [{"ix": i.ix, "person_id": names.get(i.ix), "focus_s": i.focus_s,
                            "presence_s": i.presence_s} for i in fa.identities]}


def s2_keys(an, fa, names):
    """S2 at segment resolution: a joined person keeps a segment only when
    their face track covers at least half of it; otherwise it reads as an
    unidentified person (the VLM said "person", the track did not confirm)."""
    out = []
    for si, p in enumerate(an.primaries):
        if p is None:
            out.append(["none"])
            continue
        s = an.subjects[p]
        if s.face_ix is None:
            out.append(["else"] if s.category == "person" else ["s", s.label])
            continue
        a, b = an.segments[si]
        out.append(_face_key(s.face_ix, names) if fa.presence_fraction(s.face_ix, a, b) >= 0.5
                   else ["else"])
    return out


# ==========================================================================
# Grounding (one VLM call per production segment keyframe)
# ==========================================================================

def default_grounder(image_path, model):
    """The production video-subjects call (`subjects.ground_frame_subjects`,
    role ``subject``, with categories), so the eval scores what ships. An
    unusable answer comes back None and is scored as "no subject", as the
    worker's <=20% segment tolerance does. Stub this in tests."""
    from photosearch import subjects
    return subjects.ground_frame_subjects(str(image_path), model=model)


def ground_video(pid, duration, *, model, grounder=None, max_px=768):
    from photosearch import video_analysis as va
    grounder = grounder or default_grounder
    segs = me.video_production_segments(duration)
    kts = va.segment_keyframe_times(segs)
    boxes, t0 = [], time.time()
    for t in kts:
        img = me.cached_frame_jpeg(pid, t, max_px)
        rec = me.Recorder()
        with rec.active():
            ans = grounder(img, model)
        rec.check(ans is None, "ground_photo_subjects")
        boxes.append([{"label": b.get("label"), "category": b.get("category"),
                       "bbox": [float(x) for x in b["bbox"]]} for b in ans or []])
    return {"status": "ok", "segments": [list(s) for s in segs], "keyframe_t": kts,
            "boxes": boxes, "latency_s": round(time.time() - t0, 3)}


def run_grounding(variant, *, model=None, limit=None, force=False, grounder=None, log=log):
    from photosearch import describe, subjects
    me.pin_role_model("subject", model)
    eff = describe.effective_model(model or subjects.AES_MODEL, "subject")
    run = me.open_run(PASS_GROUND, variant, force=force, effective_model=eff,
                      seg_version=me.video_seg_version())
    todo = [p for p in me.load_video_sample()["photos"]
            if me.is_labelled_clip(p) and str(p["photo_id"]) not in run["items"]]
    todo = todo[:limit] if limit is not None else todo
    log(f"[grounding] {variant}: model={eff} — {len(todo)} to do, {len(run['items'])} cached")
    fc = me.FailureCounter(log)
    for p in todo:
        pid = p["photo_id"]
        try:
            item = ground_video(pid, p["duration_s"], model=model or subjects.AES_MODEL,
                                grounder=grounder)
        except me.TransportError as e:
            if me.classify_failure(e) == "rejected":
                run["items"][str(pid)] = {"status": "rejected", "error": str(e)}
                me.save_run(PASS_GROUND, variant, run)
            fc.fail(pid, e)
            continue
        except Exception as e:
            fc.fail(pid, e)
            continue
        fc.ok()
        run["items"][str(pid)] = item
        me.save_run(PASS_GROUND, variant, run)
    fc.summary()
    return run


# ==========================================================================
# run-faces
# ==========================================================================

def run_faces(variant, approach, *, conn, grounding=None, model=None, params=None,
              sparams=None, det_tag=DET_TAG, det_size=None, force=False, scanner=None,
              grounder=None, limit=None, log=log):
    from photosearch import video_analysis as va
    if approach not in SIMPLICITY:
        raise SystemExit(f"--approach must be one of {SIMPLICITY}")
    params = params or va.DEFAULT_PARAMS
    gvar = None
    if approach in ("S1", "S2"):
        if not grounding:
            raise SystemExit("S1/S2 need --grounding <variant> (the grounding cache name)")
        gvar = run_grounding(grounding, model=model, limit=limit, grounder=grounder, log=log)
    ident = {"approach": approach, "det_tag": det_tag, "grounding": grounding,
             "grounding_model": (gvar or {}).get("effective_model"),
             "params": json.dumps(_jsonable(params.__dict__), sort_keys=True),
             "sparams": json.dumps(_jsonable((sparams.__dict__ if sparams else {})), sort_keys=True)}
    run = me.open_run(PASS_SUBJECT, variant, force=force, **ident)
    refs = load_references(conn) if conn is not None else ([], None)
    todo = [p for p in me.load_video_sample()["photos"] if me.is_labelled_clip(p)]
    todo = todo[:limit] if limit is not None else todo
    fc = me.FailureCounter(log)
    for p in todo:
        pid = p["photo_id"]
        try:
            t0 = time.time()
            frames, meta = ensure_detections(pid, tag=det_tag, det_size=det_size, scanner=scanner)
            g = (gvar or {"items": {}})["items"].get(str(pid))
            item = predict(approach, pid, frames, meta, params=params, sparams=sparams,
                           refs=refs, grounding=g)
            item["latency_s"] = round(time.time() - t0 + ((g or {}).get("latency_s") or 0), 3)
        except Exception as e:
            fc.fail(pid, e)
            continue
        fc.ok()
        run["items"][str(pid)] = item
    me.save_run(PASS_SUBJECT, variant, run)
    fc.summary()
    log(f"[run-faces] {variant} ({approach}): {len(run['items'])} videos scored")
    return run


# ==========================================================================
# run-describe
# ==========================================================================

VIDEO_PROMPT = (
    "These {k} images are frames from ONE video clip, in time order, taken at {times} "
    "of a {dur} clip. Describe the video in plain prose (no lists, no headings): who "
    "or what is in it, where it is, and what HAPPENS over the clip — what changes "
    "from the first frame to the last. Do not describe the frames one by one, and "
    "do not mention that you were shown frames. Mention any readable text.")
STORYBOARD_PROMPT = (
    "This image is a storyboard of ONE video clip: {k} frames in time order, left to "
    "right then top to bottom, taken at {times} of a {dur} clip. Describe the video in "
    "plain prose: who or what is in it, where, and what HAPPENS over the clip. Do not "
    "describe the panels one by one or mention the storyboard.")
SUMMARY_PROMPT = (
    "Below are descriptions of {k} frames from ONE video clip, in time order ({times} "
    "of a {dur} clip). Write one plain-prose description of the VIDEO: who or what is "
    "in it, where, and what happens over the clip. Only state what the frame "
    "descriptions support.\n\n{blocks}")


def _fmt_times(ts):
    return ", ".join(f"{t:.1f}s" for t in ts)


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def keyframes_for(pid, duration, k, frames=None):
    """The production keyframe sampler: K uniform quantile centres, nudged to
    the sharpest sampled frame when the detection cache carries sharpness."""
    from photosearch import video_analysis as va
    return va.keyframe_times(duration, k=k)


def storyboard(paths, cols=3, rows=2, size=(1536, 1024)):
    from PIL import Image
    W, H = size
    cw, ch = W // cols, H // rows
    sheet = Image.new("RGB", (W, H), (0, 0, 0))
    for i, p in enumerate(paths[:cols * rows]):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cw, ch), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * cw + (cw - im.width) // 2,
                         (i // cols) * ch + (ch - im.height) // 2))
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _chat(model, prompt, images, role, chat=None):
    from photosearch import describe
    chat = chat or describe._ollama_chat_with_retry
    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    return chat(model=model, messages=[msg], options={"temperature": 0}, role=role, timeout=300)


def describe_video(approach, pid, duration, *, model, k=6, px=768, chat=None,
                   describer=None, poster_t=None):
    """(text, extras) for one video under one approach. Raises on transport
    failure (the caller classifies rejected vs transport)."""
    from photosearch import describe
    dur = f"{float(duration):.0f} s"
    if approach == "D0":
        t = float(poster_t or 0.0)
        path = me.cached_frame_jpeg(pid, t, 1024)
        return (describer or describe.describe_photo)(str(path), model=model), {"frames": [t]}
    ts = keyframes_for(pid, duration, k)
    paths = [me.cached_frame_jpeg(pid, t, px) for t in ts]
    if approach == "D1":
        text = _chat(model, VIDEO_PROMPT.format(k=len(ts), times=_fmt_times(ts), dur=dur),
                     [_b64(p) for p in paths], "video", chat)
    elif approach == "D2":
        text = _chat(model, STORYBOARD_PROMPT.format(k=len(ts), times=_fmt_times(ts), dur=dur),
                     [storyboard(paths)], "video", chat)
    elif approach == "D3":
        parts = [(describer or describe.describe_photo)(str(p), model=model) for p in paths]
        if not any(parts):
            return None, {"frames": ts}
        blocks = "\n".join(f"Frame {i + 1} ({t:.1f}s): {d}" for i, (t, d) in
                           enumerate(zip(ts, parts)) if d)
        text = _chat(model, SUMMARY_PROMPT.format(k=len(ts), times=_fmt_times(ts), dur=dur,
                                                  blocks=blocks), None, "text", chat)
    else:
        raise SystemExit("--approach must be D0, D1, D2 or D3")
    return text, {"frames": ts}


_FORMAT_RE = re.compile(r"\*\*|^\s*#{1,6}\s|^\s*[-*•]\s+\S|^\s*\d+[.)]\s+\S", re.M)


def run_describe(variant, approach, *, model=None, k=6, px=768, force=False, limit=None,
                 chat=None, describer=None, solo=False, log=log):
    from photosearch import describe
    role = "describe" if approach in ("D0", "D3") else "video"
    me.pin_role_model(role, model)
    if approach == "D3":
        me.pin_role_model("text", model)
    eff = describe.effective_model(model or describe.MODEL, role)
    run = me.open_run(PASS_DESCRIBE, variant, force=force, effective_model=eff,
                      approach=approach, k=k, px=px)
    todo = [p for p in me.load_video_sample()["photos"]
            if me.is_labelled_clip(p) and str(p["photo_id"]) not in run["items"]]
    todo = todo[:limit] if limit is not None else todo
    loaded_start = me.lmstudio_loaded()
    log(f"[run-describe] {variant} ({approach}): model={eff} — {len(todo)} to do")
    fc = me.FailureCounter(log)
    for p in todo:
        pid = p["photo_id"]
        rec = me.Recorder()
        t0 = time.time()
        try:
            with rec.active():
                text, extra = describe_video(approach, pid, p["duration_s"], model=model or eff,
                                             k=k, px=px, chat=chat, describer=describer)
            rec.check(text is None, f"video describe {approach}")
        except describe.UnusableAnswer as e:
            item = {"status": "unusable", "text": None, "error": str(e)}
        except Exception as e:
            if me.classify_failure(e) == "rejected":
                run["items"][str(pid)] = {"status": "rejected", "text": None, "error": str(e)}
                me.save_run(PASS_DESCRIBE, variant, run)
            fc.fail(pid, e)
            continue
        else:
            item = {"status": "ok" if text else "unanswered", "text": text,
                    "text_sha": me.text_sha(text), **extra}
        fc.ok()
        item.update(latency_s=round(time.time() - t0, 3), attempts=rec.attempts(),
                    truncated=rec.truncated(768),
                    format_leak=bool(item.get("text") and _FORMAT_RE.search(item["text"])))
        run["items"][str(pid)] = item
        me.save_run(PASS_DESCRIBE, variant, run)
    fc.summary()
    run["latency_label"] = me.latency_label(eff, loaded_start, me.lmstudio_loaded(), solo)
    me.save_run(PASS_DESCRIBE, variant, run)
    return run


# ==========================================================================
# run-quality
# ==========================================================================

QUALITY_PROMPT = (
    "These {k} images are frames from ONE video clip, in time order. Judge the "
    "VIDEO's technical quality. Answer with ONLY a JSON object: "
    '{{"subject_in_focus": "yes"|"partly"|"no", "shaky": "yes"|"no", "noisy": "yes"|"no"}}. '
    "subject_in_focus: is the main subject sharp? shaky: does the camera shake (not a "
    "deliberate pan)? noisy: is there visible grain or noise?")


def measured_quality(frames, meta):
    """Q-m numbers: the scan's measures when it supplied them, focus
    consistency from the F1 main focus otherwise."""
    from photosearch import video_analysis as va
    m = meta.get("measures")
    m = dict(m) if isinstance(m, dict) else {}
    out = {k: m.get(k) for k in ("shake", "noise", "focus_consistency", "tenengrad_max")}
    exp = m.get("exposure") or {}
    out["highlight_frac"] = exp.get("highlight_frac", m.get("highlight_frac"))
    out["shadow_frac"] = exp.get("shadow_frac", m.get("shadow_frac"))
    if out["focus_consistency"] is None and frames:
        fa = va.analyze_faces(frames, meta["W"], meta["H"], meta["duration"])
        out["focus_consistency"] = va.focus_consistency(va.main_focus_sharpness(fa))
    return out


def parse_quality(raw):
    if not raw:
        return None
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < a:
        return None
    try:
        d = json.loads(raw[a:b + 1])
    except ValueError:
        return None
    f = str(d.get("subject_in_focus", "")).lower()
    s = str(d.get("shaky", "")).lower()
    n = str(d.get("noisy", "")).lower()
    if f not in ("yes", "partly", "no") or s not in ("yes", "no") or n not in ("yes", "no"):
        return None
    return {"subject_in_focus": f, "shaky": s, "noisy": n}


def run_quality(variant, approach, *, model=None, k=6, px=768, force=False, limit=None,
                det_tag=DET_TAG, scanner=None, chat=None, log=log):
    from photosearch import describe
    if approach not in ("Qm", "Qv"):
        raise SystemExit("--approach must be Qm or Qv")
    ident = {"approach": approach}
    if approach == "Qv":
        me.pin_role_model("video", model)
        ident.update(effective_model=describe.effective_model(model or "", "video"), k=k, px=px)
    else:
        ident.update(det_tag=det_tag)
    run = me.open_run(PASS_QUALITY, variant, force=force, **ident)
    todo = [p for p in me.load_video_sample()["photos"]
            if me.is_labelled_clip(p) and str(p["photo_id"]) not in run["items"]]
    todo = todo[:limit] if limit is not None else todo
    fc = me.FailureCounter(log)
    for p in todo:
        pid = p["photo_id"]
        t0 = time.time()
        try:
            if approach == "Qm":
                frames, meta = ensure_detections(pid, tag=det_tag, scanner=scanner)
                item = {"status": "ok", "measures": measured_quality(frames, meta)}
            else:
                ts = keyframes_for(pid, p["duration_s"], k)
                imgs = [_b64(me.cached_frame_jpeg(pid, t, px)) for t in ts]
                rec = me.Recorder()
                with rec.active():
                    raw = _chat(model or ident["effective_model"],
                                QUALITY_PROMPT.format(k=len(ts)), imgs, "video", chat)
                rec.check(raw is None, "video quality rubric")
                flags = parse_quality(raw)
                item = {"status": "ok" if flags else "unusable", "flags": flags, "raw": raw}
        except Exception as e:
            if me.classify_failure(e) == "rejected":
                run["items"][str(pid)] = {"status": "rejected", "error": str(e)}
                me.save_run(PASS_QUALITY, variant, run)
            fc.fail(pid, e)
            continue
        fc.ok()
        item["latency_s"] = round(time.time() - t0, 3)
        run["items"][str(pid)] = item
        me.save_run(PASS_QUALITY, variant, run)
    fc.summary()
    return run


# ==========================================================================
# disputes (describe Differences) + same-subject judgements
# ==========================================================================

def build_video_disputes(*, model=None, variants=None, max_q=6, force=False, chat=None, log=log):
    """Per video, the checkable facts its distinct descriptions disagree on —
    `describe_eval`'s prompt and parser, stored under video-describe/."""
    import importlib.util
    from photosearch import describe
    spec = importlib.util.spec_from_file_location("describe_eval", os.path.join(HERE, "describe_eval.py"))
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)
    me.pin_role_model("text", model)
    eff = describe.effective_model(model or "llama3.2:3b", "text")
    chat = chat or describe._ollama_chat_with_retry
    variants = variants or me.list_variants(PASS_DESCRIBE)
    runs = {v: me.load_run(PASS_DESCRIBE, v) for v in variants}
    data = {} if force else me.load_disputes(PASS_DESCRIBE)
    labels = me.load_dispute_labels(PASS_DESCRIBE)
    fc = me.FailureCounter(log)
    for p in me.load_video_sample()["photos"]:
        pid = p["photo_id"]
        texts = {}
        for r in runs.values():
            it = (r or {"items": {}})["items"].get(str(pid)) or {}
            if it.get("text"):
                texts.setdefault(it.get("text_sha") or me.text_sha(it["text"]), it["text"])
        if len(texts) < 2:
            continue
        shas = sorted(texts)
        prev = data.get(str(pid)) or {}
        if set(shas) <= set(prev.get("shas") or []):
            continue
        if prev and any(q["key"] in labels for q in prev.get("questions", [])):
            log(f"  - {pid}: labelled; descriptions changed — not re-extracted (would orphan labels)")
            continue
        order = list(shas)
        random.Random(f"video-disputes-{pid}").shuffle(order)
        labs = [f"D{i + 1}" for i in range(len(order))]
        prompt = de.DISPUTE_PROMPT.replace("SAME photo", "SAME video").replace(
            "looking at the photo", "watching the video").format(
            blocks="\n\n".join(f"{l}: {texts[s]}" for l, s in zip(labs, order)), max_q=max_q,
            keys=", ".join(f'"{l}": "..."' for l in labs))
        try:
            raw = chat(model=model or "llama3.2:3b", messages=[{"role": "user", "content": prompt}],
                       options={"temperature": 0}, role="text", timeout=180)
        except Exception as e:
            fc.fail(pid, e)
            continue
        qs = de._parse_disputes(raw, labs)
        if qs is None:
            fc.fail(pid, f"unparseable: {(raw or '')[:80]!r}")
            continue
        fc.ok()
        by = dict(zip(labs, order))
        data[str(pid)] = {"shas": shas, "model": eff, "created": me.now_iso(), "questions": [
            {"key": me.dispute_key(pid, q["question"], shas), "question": q["question"],
             "answers": {by[l]: a for l, a in q["answers"].items()}} for q in qs[:max_q]]}
        me.save_disputes(data, PASS_DESCRIBE)
    fc.summary()
    return data


SAME_PROMPT = ('Do these two short labels name the SAME subject of a video (the same '
               'animal, thing or scene)? A: "{a}"  B: "{b}". Answer only yes or no.')


def judge_same_labels(*, model=None, chat=None, log=log):
    """Free-text subject labels the owner typed vs candidates' labels for the
    same segment: the comparer model judges "same subject?" (the owner
    spot-checks 20). Cached in video/label_judgements.json by the pair."""
    from photosearch import describe
    from photosearch.video_subjects import normalize_label
    chat = chat or describe._ollama_chat_with_retry
    me.pin_role_model("text", model)
    judged = me.read_pass_file(me.VIDEO, "label_judgements.json", {})
    labels = me.load_video_subject_labels()
    runs = [me.load_run(PASS_SUBJECT, v) for v in me.list_variants(PASS_SUBJECT)]
    pairs = set()
    for key, lab in (labels.get("segments") or {}).items():
        if lab.get("kind") != "subject" or not lab.get("free_text"):
            continue
        pid, _, ix = key.split(":")
        for r in runs:
            it = (r or {}).get("items", {}).get(pid) or {}
            pred = (it.get("pred") or [None] * (int(ix) + 1))[int(ix)] if it else None
            if pred and pred[0] == "s":
                a, b = normalize_label(lab["label"]), pred[1]
                if a != b:
                    pairs.add(tuple(sorted((a, b))))
    for a, b in sorted(pairs):
        k = f"{a}|{b}"
        if k in judged:
            continue
        try:
            raw = chat(model=model or "llama3.2:3b",
                       messages=[{"role": "user", "content": SAME_PROMPT.format(a=a, b=b)}],
                       options={"temperature": 0}, role="text", timeout=60)
        except Exception as e:
            log(f"  ! {k}: {e}")
            continue
        judged[k] = {"same": str(raw or "").strip().lower().startswith("yes"), "at": me.now_iso()}
        me.write_pass_file(me.VIDEO, "label_judgements.json", judged)
    return judged


# ==========================================================================
# Scoring (pure — tested on synthetic labels)
# ==========================================================================

def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap_ci(xs, stat=lambda v: sum(v) / len(v), n_boot=2000, seed=3):
    if not xs:
        return None
    rnd = random.Random(seed)
    vals = sorted(stat([rnd.choice(xs) for _ in xs]) for _ in range(n_boot))
    return (vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1])


def owner_key(lab):
    k = lab.get("kind")
    if k == "person":
        return ["p", int(lab["person_id"])]
    if k == "someone_else":
        return ["else"]
    if k == "subject":
        from photosearch.video_subjects import normalize_label
        return ["s", normalize_label(lab["label"]) or lab["label"]]
    if k == "none":
        return ["none"]
    return None                                     # cant_tell: excluded


def make_same(judgements=None):
    judgements = judgements or {}

    def same(a, b):
        if a is None or b is None:
            return False
        if a[0] != b[0]:
            return False
        if a[0] != "s":
            return list(a) == list(b)
        from photosearch.video_subjects import normalize_label
        x, y = normalize_label(a[1]) or a[1], normalize_label(b[1]) or b[1]
        if x == y:
            return True
        j = judgements.get("|".join(sorted((x, y))))
        return bool(j and j.get("same"))
    return same


def _split_of(key):
    return "person" if key[0] in ("p", "else") else "non-person"


def video_scores(owner_segs, pred_segs, durations, who, same):
    """Per video: segment verdicts, top-subject agreement, share pairs,
    presence counts. `owner_segs`/`pred_segs` are keys per label segment."""
    segs = []
    o_dur: dict = {}
    p_dur: dict = {}
    total = 0.0
    for o, p, d in zip(owner_segs, pred_segs, durations):
        if o is None:
            continue
        p = p or ["none"]
        segs.append({"split": _split_of(o), "correct": same(o, p), "owner": o, "pred": p})
        total += d
        o_dur[json.dumps(o)] = o_dur.get(json.dumps(o), 0.0) + d
        # canonicalise a prediction to the owner's spelling when judged same
        pk = next((k for k in o_dur if same(json.loads(k), p)), json.dumps(p))
        p_dur[pk] = p_dur.get(pk, 0.0) + d
    if not segs or total <= 0:
        return None
    top_o = max(o_dur, key=lambda k: o_dur[k])
    top_p = max(p_dur, key=lambda k: p_dur[k])
    shares = [(o_dur.get(k, 0.0) / total, p_dur.get(k, 0.0) / total)
              for k in sorted(set(o_dur) | set(p_dur))]
    return {"segments": segs, "top_agree": same(json.loads(top_o), json.loads(top_p)),
            "shares": shares}


def presence_counts(pred_present, owner_people):
    pp, op = set(pred_present or []), set(owner_people or [])
    return {"tp": len(pp & op), "fp": len(pp - op), "fn": len(op - pp)}


def subject_metrics(run, labels, sample, *, split="holdout", same=None, strata=None):
    """Every subject metric for one variant over one split."""
    same = same or make_same()
    seg_v = me.video_seg_version()
    items = (run or {}).get("items", {})
    out = {"videos": 0, "seg": {"person": [0, 0], "non-person": [0, 0], "all": [0, 0]},
           "top": [0, 0], "mae": [], "retr": {"tp": 0, "fp": 0, "fn": 0},
           "presence": {"tp": 0, "fp": 0, "fn": 0}, "by_stratum": {}, "latency": []}
    for p in sample["photos"]:
        if not me.is_labelled_clip(p) or (split and p.get("split") != split):
            continue
        if strata and p["stratum"] not in strata:
            continue
        pid = p["photo_id"]
        who = (labels.get("who") or {}).get(str(pid))
        it = items.get(str(pid))
        if not who or not who.get("done") or not it or it.get("seg_version") != seg_v:
            continue
        lsegs = it["label_segments"]
        owner = [owner_key(l) if l else None
                 for l in me.video_segment_labels(labels, pid, seg_v, len(lsegs))]
        vs_ = video_scores(owner, it["pred"], [b - a for a, b in lsegs], who, same)
        if vs_ is None:
            continue
        out["videos"] += 1
        bs = out["by_stratum"].setdefault(p["stratum"], [0, 0])
        for s in vs_["segments"]:
            for k in (s["split"], "all"):
                out["seg"][k][0] += s["correct"]
                out["seg"][k][1] += 1
            bs[0] += s["correct"]
            bs[1] += 1
        out["top"][0] += vs_["top_agree"]
        out["top"][1] += 1
        for o, pr in vs_["shares"]:
            out["mae"].append(abs(o - pr))
            po, pp = o >= SHARE_RETRIEVAL, pr >= SHARE_RETRIEVAL
            out["retr"]["tp"] += po and pp
            out["retr"]["fp"] += pp and not po
            out["retr"]["fn"] += po and not pp
        pc = presence_counts(it.get("present"), who.get("people"))
        for k in pc:
            out["presence"][k] += pc[k]
        if it.get("latency_s") is not None:
            out["latency"].append(it["latency_s"])
    return out


def _ratio(k, n):
    return None if n < me.MIN_N else k / n


def summarize_subject(m):
    pr = m["presence"]
    rt = m["retr"]
    s = {k: _ratio(*m["seg"][k]) for k in m["seg"]}
    return {
        "videos": m["videos"], "seg_acc": s,
        "seg_n": {k: m["seg"][k][1] for k in m["seg"]},
        "seg_ci": {k: wilson(*m["seg"][k]) for k in m["seg"]},
        "top": _ratio(*m["top"]), "top_ci": wilson(*m["top"]), "top_n": m["top"][1],
        "mae": (sum(m["mae"]) / len(m["mae"])) if len(m["mae"]) >= me.MIN_N else None,
        "mae_ci": bootstrap_ci(m["mae"]),
        "presence_p": _ratio(pr["tp"], pr["tp"] + pr["fp"]),
        "presence_p_ci": wilson(pr["tp"], pr["tp"] + pr["fp"]),
        "presence_r": _ratio(pr["tp"], pr["tp"] + pr["fn"]),
        "retr_p": _ratio(rt["tp"], rt["tp"] + rt["fp"]),
        "retr_r": _ratio(rt["tp"], rt["tp"] + rt["fn"]),
        "by_stratum": {k: _ratio(*v) for k, v in m["by_stratum"].items()},
        "s_per_video": me.median(m["latency"]),
    }


def subject_gates(s):
    """The step 6 ship gate on HOLDOUT numbers, plus the plan's fallbacks.
    A metric with n<3 fails (not enough evidence is not a pass)."""
    def ge(x, t):
        return x is not None and x >= t
    g = {"top_subject": ge(s["top"], GATE_TOP),
         "named_presence_p": ge(s["presence_p"], GATE_PRESENCE_P),
         "seg_acc_person": ge(s["seg_acc"].get("person"), GATE_SEG_ACC),
         "seg_acc_non_person": ge(s["seg_acc"].get("non-person"), GATE_SEG_ACC),
         "share_mae": s["mae"] is not None and s["mae"] <= GATE_SHARE_MAE}
    g["ship_subject_filter"] = all(g.values())
    if g["ship_subject_filter"]:
        g["ships"] = "subject filter"
    elif g["named_presence_p"] and g["seg_acc_non_person"] and not g["seg_acc_person"]:
        g["ships"] = "non-person subjects + person presence only"
    elif g["named_presence_p"]:
        g["ships"] = "presence % only"
    else:
        g["ships"] = "nothing"
    return g


def pick_winner(scores, candidates, key=lambda s: s["seg_acc"].get("all"), tie=TIE_SEG,
                order=SIMPLICITY):
    """Highest holdout score; a gap < `tie` is a tie, won by the simpler /
    cheaper approach (`order`). Returns (winner, [tied])."""
    have = [(v, key(scores[v])) for v in candidates if v in scores and key(scores[v]) is not None]
    if not have:
        return None, []
    best = max(x for _, x in have)
    tied = [v for v, x in have if best - x < tie]

    def rank(v):
        a = scores[v].get("approach") or v
        return (order.index(a) if a in order else len(order), -key(scores[v]))
    return min(tied, key=rank), sorted(tied)


def f1_vs_s1(scores, f1, s1):
    """Did the VLM earn its GPU hours for people? S1 must beat F1 by >= 5
    points on person segments."""
    a = (scores.get(f1) or {}).get("seg_acc", {}).get("person")
    b = (scores.get(s1) or {}).get("seg_acc", {}).get("person")
    if a is None or b is None:
        return None
    return b - a >= F1_MARGIN


# ---- describe -------------------------------------------------------------

def describe_screen(run):
    items = list((run or {}).get("items", {}).values())
    n = len(items)
    if not n:
        return None
    bad = sum(1 for it in items if it.get("status") in ("unanswered", "unusable", "rejected"))
    words = [len(it["text"].split()) for it in items if it.get("text")]
    return {"n": n, "answered": sum(1 for it in items if it.get("text")) / n,
            "unusable": sum(1 for it in items if it.get("status") == "unusable") / n,
            "rejected": sum(1 for it in items if it.get("status") == "rejected") / n,
            "format_leak": sum(1 for it in items if it.get("format_leak")) / n,
            "words_p50": me.percentile(words, 0.5), "words_p90": me.percentile(words, 0.9),
            "s_per_video": me.median(me.latencies(run["items"])),
            "screened_out": bad / n > SCREEN_MAX_BAD}


def describe_scores(run, disputes, dlabels, temporal, sample, split="holdout"):
    ids = {p["photo_id"] for p in sample["photos"] if p.get("split") == split}
    right = wrong = 0
    per_video = {}
    t_yes = t_n = 0
    for pid_s, it in (run or {}).get("items", {}).items():
        if int(pid_s) not in ids or not it.get("text"):
            continue
        sha = it.get("text_sha") or me.text_sha(it["text"])
        r = w = 0
        for q in (disputes.get(pid_s) or {}).get("questions", []):
            v = me.dispute_verdict(q["answers"].get(sha), dlabels.get(q["key"]))
            r += v == "right"
            w += v == "wrong"
        right, wrong = right + r, wrong + w
        per_video[int(pid_s)] = (r, w)
        t = temporal.get(sha)
        if t:
            t_n += 1
            t_yes += t["answer"] == "yes"
    return {"right_rate": _ratio(right, right + wrong), "right_ci": wilson(right, right + wrong),
            "wrong_per_video": (wrong / len(per_video)) if per_video else None,
            "temporal_yes": _ratio(t_yes, t_n), "temporal_ci": wilson(t_yes, t_n),
            "temporal_n": t_n, "per_video": per_video}


def sign_test_p(wins, losses):
    from math import comb
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def per_video_sign(a, b):
    """Videos where a's right-rate beats b's (and vice versa)."""
    w = l = 0
    for pid in set(a["per_video"]) & set(b["per_video"]):
        ra, wa = a["per_video"][pid]
        rb, wb = b["per_video"][pid]
        xa = ra / (ra + wa) if ra + wa else None
        xb = rb / (rb + wb) if rb + wb else None
        if xa is None or xb is None or xa == xb:
            continue
        w += xa > xb
        l += xa < xb
    return w, l, sign_test_p(w, l)


def describe_gate(winner, d0):
    """Multi-frame must beat poster-only: >= 0.20 temporal yes-rate, OR
    >= 0.05 right-rate at sign p < 0.1."""
    if winner is None or d0 is None:
        return None
    dt = None if winner["temporal_yes"] is None or d0["temporal_yes"] is None else \
        winner["temporal_yes"] - d0["temporal_yes"]
    dr = None if winner["right_rate"] is None or d0["right_rate"] is None else \
        winner["right_rate"] - d0["right_rate"]
    _, _, p = per_video_sign(winner, d0)
    passed = (dt is not None and dt >= DESCRIBE_GATE_TEMPORAL) or \
             (dr is not None and dr >= DESCRIBE_GATE_RIGHT and p is not None and p < DESCRIBE_GATE_P)
    return {"temporal_delta": dt, "right_delta": dr, "sign_p": p, "pass": passed}


# ---- quality --------------------------------------------------------------

# Positive class = the PROBLEM (what a tag would mark).
QUALITY_POSITIVE = {"in_focus": "no", "shaky": "yes", "noisy": "yes"}
MEASURE_FOR = {"in_focus": ("focus_consistency", -1), "shaky": ("shake", 1),
               "noisy": ("noise", 1)}
VLM_FOR = {"in_focus": ("subject_in_focus", "no"), "shaky": ("shaky", "yes"),
           "noisy": ("noisy", "yes")}


def pr_counts(pairs):
    """pairs = [(truth_bool, pred_bool)]."""
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if p and not t)
    fn = sum(1 for t, p in pairs if t and not p)
    return {"p": _ratio(tp, tp + fp), "r": _ratio(tp, tp + fn), "tp": tp, "fp": fp, "fn": fn}


def truth_pairs(qlabels, ids, flag):
    """(pid, truth) for the owner's yes/no; debatable is neither."""
    out = []
    for pid in ids:
        lab = qlabels.get(str(pid))
        if not lab or not lab.get("done"):
            continue
        v = lab.get(flag)
        if v in (None, "debatable"):
            continue
        out.append((pid, v == QUALITY_POSITIVE[flag]))
    return out


def fit_threshold(scored):
    """[(value, truth)] on DEV -> the threshold (predict positive when
    value >= t) with the best precision among those reaching R >= 0.5, else
    the best F1. A 1-D logistic fit is monotone, so this is the same family."""
    cands = sorted({v for v, _ in scored if v is not None})
    best, best_key = None, None
    for t in cands:
        c = pr_counts([(tr, v is not None and v >= t) for v, tr in scored])
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        key = (r >= QUALITY_R, p if r >= QUALITY_R else f1, r)
        if best_key is None or key > best_key:
            best, best_key = t, key
    return best


def quality_scores(runs, qlabels, sample):
    """Per flag: Q-m (threshold fitted on dev), each Q-v, and Q-mv OR/AND,
    all evaluated on holdout."""
    dev = [p["photo_id"] for p in sample["photos"] if p.get("split") == "dev" and me.is_labelled_clip(p)]
    hold = [p["photo_id"] for p in sample["photos"] if p.get("split") == "holdout" and me.is_labelled_clip(p)]
    out = {}
    qm = [v for v, r in runs.items() if r.get("approach") == "Qm"]
    qv = [v for v, r in runs.items() if r.get("approach") == "Qv"]
    for flag in me.QUALITY_FLAGS:
        res = {}
        mname, sign = MEASURE_FOR[flag]
        thr = {}
        for v in qm:
            items = runs[v]["items"]

            def val(pid):
                m = (items.get(str(pid)) or {}).get("measures") or {}
                x = m.get(mname)
                return None if x is None else sign * x
            t = fit_threshold([(val(pid), tr) for pid, tr in truth_pairs(qlabels, dev, flag)])
            thr[v] = (t, val)
            pairs = [(tr, t is not None and val(pid) is not None and val(pid) >= t)
                     for pid, tr in truth_pairs(qlabels, hold, flag)]
            res[v] = {**pr_counts(pairs), "threshold": None if t is None else sign * t}
        for v in qv:
            key, pos = VLM_FOR[flag]
            items = runs[v]["items"]

            def vpred(pid, items=items, key=key, pos=pos):
                f = (items.get(str(pid)) or {}).get("flags") or {}
                return f.get(key) == pos
            res[v] = pr_counts([(tr, vpred(pid)) for pid, tr in truth_pairs(qlabels, hold, flag)])
            for m in qm:
                t, val = thr[m]
                mp = lambda pid: t is not None and val(pid) is not None and val(pid) >= t  # noqa: E731
                tp_ = truth_pairs(qlabels, hold, flag)
                res[f"{m}|{v}"] = pr_counts([(tr, mp(pid) or vpred(pid)) for pid, tr in tp_])
                res[f"{m}&{v}"] = pr_counts([(tr, mp(pid) and vpred(pid)) for pid, tr in tp_])
        for v, c in res.items():
            c["gate"] = c["p"] is not None and c["r"] is not None and \
                c["p"] >= QUALITY_P and c["r"] >= QUALITY_R
        out[flag] = res
    return out


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    n = len(pairs)
    if n < me.MIN_N:
        return None, None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return None, None
    rho = cov / (sx * sy)
    if n <= 3 or abs(rho) >= 1:
        return rho, None
    z = math.atanh(rho)
    h = 1.96 / math.sqrt(n - 3)
    return rho, (math.tanh(z - h), math.tanh(z + h))


# ---- agreement -------------------------------------------------------------

def cohen_kappa(pairs):
    """Cohen's kappa for categorical pairs; None when undefined."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    cats = {a for a, _ in pairs} | {b for _, b in pairs}
    pe = sum((sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n)
             for c in cats)
    return None if pe == 1 else (po - pe) / (1 - pe)


def self_agreement():
    """First answer vs the blind recheck, on videos done in both."""
    seg_v = me.video_seg_version()
    a, b = me.load_video_subject_labels("main"), me.load_video_subject_labels("recheck")
    seg_pairs = []
    for pid, wb in (b.get("who") or {}).items():
        wa = (a.get("who") or {}).get(pid)
        if not wa or not wa.get("done") or not wb.get("done"):
            continue
        n = min(wa.get("n_segments") or 0, wb.get("n_segments") or 0)
        la = me.video_segment_labels(a, pid, seg_v, n)
        lb = me.video_segment_labels(b, pid, seg_v, n)
        for x, y in zip(la, lb):
            if x and y:
                seg_pairs.append((json.dumps(owner_key(x)), json.dumps(owner_key(y))))
    qa, qb = me.load_video_quality_labels("main"), me.load_video_quality_labels("recheck")
    q = {}
    for f in me.QUALITY_FLAGS + ("keeper",):
        pairs = [(qa[pid][f], qb[pid][f]) for pid in qb
                 if pid in qa and qa[pid].get("done") and qb[pid].get("done")
                 and qa[pid].get(f) not in (None, "debatable")
                 and qb[pid].get(f) not in (None, "debatable")]
        q[f] = {"n": len(pairs), "kappa": cohen_kappa(pairs),
                "agreement": (sum(1 for x, y in pairs if x == y) / len(pairs)) if pairs else None}
    return {"segments": {"n": len(seg_pairs), "kappa": cohen_kappa(seg_pairs),
                         "agreement": (sum(1 for x, y in seg_pairs if x == y) / len(seg_pairs))
                         if seg_pairs else None}, "quality": q}


# ==========================================================================
# Registration checks
# ==========================================================================

def probe_check(sample, ffprobe=None, runner=None):
    """Local ffprobe on each cached probe-check original vs the stored
    (HTTP Range) verdict. Must agree 100% on readable/unreadable."""
    import shutil
    import subprocess
    ffprobe = ffprobe or shutil.which("ffprobe")
    if not ffprobe:
        return {"skipped": "no ffprobe on PATH (apt install ffmpeg, D20)"}
    runner = runner or (lambda args: subprocess.run(args, capture_output=True, text=True, timeout=120))
    agree = n = 0
    mismatches = []
    for e in sample.get("probe_check") or []:
        path = me.cached_original(e["photo_id"])
        if path is None:
            continue
        r = runner([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
        readable = r.returncode == 0 and '"duration"' in (r.stdout or "")
        n += 1
        if readable == (e.get("probe_error") is None):
            agree += 1
        else:
            mismatches.append(e["photo_id"])
    return {"n": n, "agree": agree, "pass": n > 0 and agree == n, "mismatches": mismatches}


def live_pairing(sample, labels):
    paired = [p for p in sample["photos"] if p["stratum"] == "live-paired"]
    near = [p for p in sample["photos"] if p["stratum"] == "live-near-miss"]
    who = labels.get("who") or {}

    def ans(p):
        return (who.get(str(p["photo_id"])) or {}).get("live_pair")
    yes = sum(1 for p in paired if ans(p) == "yes")
    no = sum(1 for p in paired if ans(p) == "no")
    missed = sum(1 for p in near if ans(p) == "yes")
    return {"paired_labelled": yes + no, "precision": _ratio(yes, yes + no),
            "pass": (yes + no) > 0 and no == 0, "near_miss_real_pairs": missed}


# ==========================================================================
# Report
# ==========================================================================

def _f(v, d=2):
    if v is None:
        return "n<3"
    if isinstance(v, tuple):
        return f"[{v[0]:.2f}, {v[1]:.2f}]"
    return f"{v:.{d}f}"


def build_report():
    sample = me.load_video_sample()
    labels = me.load_video_subject_labels()
    judged = me.read_pass_file(me.VIDEO, "label_judgements.json", {})
    same = make_same(judged)
    rep = {"subject": {}, "subject_dev": {}}
    runs = {v: me.load_run(PASS_SUBJECT, v) for v in me.list_variants(PASS_SUBJECT)}
    for v, r in runs.items():
        for split, slot in (("holdout", "subject"), ("dev", "subject_dev")):
            s = summarize_subject(subject_metrics(r, labels, sample, split=split, same=same))
            s["approach"] = r.get("approach")
            rep[slot][v] = s
    s_cands = [v for v, r in runs.items() if r.get("approach") in ("S1", "S2")]
    winner, tied = pick_winner(rep["subject"], s_cands)
    f1s = [v for v, r in runs.items() if r.get("approach") == "F1"]
    rep["subject_winner"] = {"winner": winner, "tied": tied,
                             "gates": subject_gates(rep["subject"][winner]) if winner else None,
                             "beats_F1_on_people": f1_vs_s1(rep["subject"], f1s[0], winner)
                             if winner and f1s else None}
    druns = {v: me.load_run(PASS_DESCRIBE, v) for v in me.list_variants(PASS_DESCRIBE)}
    disputes = me.load_disputes(PASS_DESCRIBE)
    dlabels = me.load_dispute_labels(PASS_DESCRIBE)
    temporal = me.load_temporal_labels()
    rep["describe"] = {}
    for v, r in druns.items():
        rep["describe"][v] = {"approach": r.get("approach"), "screen": describe_screen(r),
                              **describe_scores(r, disputes, dlabels, temporal, sample)}
    alive = {v: d for v, d in rep["describe"].items()
             if d["screen"] and not d["screen"]["screened_out"] and d["right_rate"] is not None}
    d0 = next((v for v, d in rep["describe"].items() if d["approach"] == "D0"), None)
    dw = None
    if alive:
        best = max(d["right_rate"] for d in alive.values())
        tied_d = [v for v, d in alive.items() if best - d["right_rate"] < DESCRIBE_TIE]
        dw = min(tied_d, key=lambda v: (alive[v]["screen"]["s_per_video"] or 1e9))
    rep["describe_winner"] = {"winner": dw, "baseline": d0,
                              "gate_vs_D0": describe_gate(rep["describe"].get(dw), rep["describe"].get(d0))
                              if dw and d0 and dw != d0 else None}
    qruns = {v: me.load_run(PASS_QUALITY, v) for v in me.list_variants(PASS_QUALITY)}
    qlabels = me.load_video_quality_labels()
    rep["quality"] = quality_scores(qruns, qlabels, sample)
    keeper_val = {"keep": 2, "meh": 1, "delete": 0}
    hold = [p["photo_id"] for p in sample["photos"] if p.get("split") == "holdout"]
    rep["keeper"] = {}
    for v, r in qruns.items():
        if r.get("approach") != "Qm":
            continue
        for mname in ("shake", "noise", "focus_consistency", "tenengrad_max"):
            xs, ys = [], []
            for pid in hold:
                lab = qlabels.get(str(pid)) or {}
                m = ((r["items"].get(str(pid)) or {}).get("measures") or {}).get(mname)
                if lab.get("done") and lab.get("keeper") in keeper_val:
                    xs.append(m)
                    ys.append(keeper_val[lab["keeper"]])
            rep["keeper"][f"{v}:{mname}"] = spearman(xs, ys)
    rep["probe_check"] = probe_check(sample)
    rep["live_pairing"] = live_pairing(sample, labels)
    return rep


def render(rep, log=log):
    log("== Main subject (HOLDOUT; dev in brackets) ==")
    head = ["variant", "appr", "videos", "seg all", "person", "non-person", "top", "MAE",
            "pres P", "pres R", "share>=.5 P/R", "s/video"]
    log("  " + " | ".join(head))
    for v, s in sorted(rep["subject"].items()):
        d = rep["subject_dev"].get(v, {})
        log("  " + " | ".join([
            v, str(s.get("approach")), str(s["videos"]),
            f"{_f(s['seg_acc']['all'])} ({_f(d.get('seg_acc', {}).get('all'))}) {_f(s['seg_ci']['all'])}",
            _f(s["seg_acc"]["person"]), _f(s["seg_acc"]["non-person"]),
            f"{_f(s['top'])} {_f(s['top_ci'])}", f"{_f(s['mae'])} {_f(s['mae_ci'])}",
            f"{_f(s['presence_p'])} {_f(s['presence_p_ci'])}", _f(s["presence_r"]),
            f"{_f(s['retr_p'])}/{_f(s['retr_r'])}", _f(s["s_per_video"], 1)]))
    w = rep["subject_winner"]
    log(f"  winner: {w['winner']} (tied: {', '.join(w['tied']) or '-'}; < {TIE_SEG:.0%} is a tie, "
        f"simpler wins); beats F1 on people by >= 5 pts: {w['beats_F1_on_people']}")
    if w["gates"]:
        log("  gates: " + ", ".join(f"{k}={'PASS' if v else 'fail'}" for k, v in w["gates"].items()
                                    if isinstance(v, bool)) + f" -> ships: {w['gates']['ships']}")
    log("\n== Description (HOLDOUT) ==")
    for v, d in sorted(rep["describe"].items()):
        sc = d["screen"] or {}
        log(f"  {v} ({d['approach']}): right {_f(d['right_rate'])} {_f(d['right_ci'])}, "
            f"wrong/video {_f(d['wrong_per_video'])}, temporal yes {_f(d['temporal_yes'])} "
            f"(n={d['temporal_n']}) | answered {_f(sc.get('answered'))}, rejected "
            f"{_f(sc.get('rejected'))}, unusable {_f(sc.get('unusable'))}, leaks "
            f"{_f(sc.get('format_leak'))}, words p50 {_f(sc.get('words_p50'), 0)}, "
            f"s/video {_f(sc.get('s_per_video'), 1)}{'  SCREENED OUT' if sc.get('screened_out') else ''}")
    dw = rep["describe_winner"]
    log(f"  winner: {dw['winner']} vs baseline {dw['baseline']}: {dw['gate_vs_D0']}")
    log("\n== Quality (HOLDOUT; a tag needs P >= 0.8 at R >= 0.5) ==")
    for flag, res in rep["quality"].items():
        for v, c in sorted(res.items()):
            log(f"  {flag:9} {v:30} P {_f(c['p'])} R {_f(c['r'])} "
                f"{'GATE PASS' if c['gate'] else ''}")
    for k, (rho, ci) in sorted(rep["keeper"].items()):
        log(f"  keeper rho {k}: {_f(rho)} {_f(ci)}")
    log("\n== Registration checks ==")
    log(f"  probe check: {rep['probe_check']}")
    log(f"  live pairing: {rep['live_pairing']}")


# ==========================================================================
# CLI
# ==========================================================================

def _params(args):
    from photosearch import video_analysis as va
    from photosearch import video_subjects as vs
    p, sp = va.DEFAULT_PARAMS, vs.DEFAULT_SUBJECT_PARAMS
    if args.sigma is not None:
        p = replace(p, sigma=None if args.sigma <= 0 else args.sigma)
        sp = replace(sp, sigma=p.sigma)
    if args.a_min is not None:
        p = replace(p, a_min=args.a_min)
    if args.merge_l2 is not None:
        p = replace(p, merge_l2=args.merge_l2)
    if args.join_window is not None:
        sp = replace(sp, join_window_s=args.join_window)
    if args.link_sim is not None:
        sp = replace(sp, link_sim=args.link_sim)
    return p, sp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    db_kw = dict(default=os.environ.get("PHOTOSEARCH_DB"))

    s = sub.add_parser("sample")
    s.add_argument("--db", **db_kw)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--force", action="store_true", help="redraw (orphans labels on dropped videos)")

    f = sub.add_parser("fetch-originals")
    f.add_argument("--server", default=me.DEFAULT_SERVER)
    f.add_argument("--db", **db_kw)
    f.add_argument("--local-root", help="copy from this photo root instead of HTTP")

    rf = sub.add_parser("run-faces")
    rf.add_argument("--db", **db_kw)
    rf.add_argument("--variant", required=True)
    rf.add_argument("--approach", required=True, choices=SIMPLICITY)
    rf.add_argument("--grounding", help="grounding cache name (S1/S2)")
    rf.add_argument("--model", help="grounding model (LM Studio id; pins the subject role)")
    rf.add_argument("--det-tag", default=DET_TAG)
    rf.add_argument("--det-size", type=int)
    for n in ("--sigma", "--a-min", "--merge-l2", "--join-window", "--link-sim"):
        rf.add_argument(n, type=float)
    rf.add_argument("--limit", type=int)
    rf.add_argument("--force", action="store_true")

    rd = sub.add_parser("run-describe")
    rd.add_argument("--variant", required=True)
    rd.add_argument("--approach", required=True, choices=("D0", "D1", "D2", "D3"))
    rd.add_argument("--model")
    rd.add_argument("--k", type=int, default=6)
    rd.add_argument("--px", type=int, default=768)
    rd.add_argument("--limit", type=int)
    rd.add_argument("--force", action="store_true")
    rd.add_argument("--solo", action="store_true")

    rq = sub.add_parser("run-quality")
    rq.add_argument("--variant", required=True)
    rq.add_argument("--approach", required=True, choices=("Qm", "Qv"))
    rq.add_argument("--model")
    rq.add_argument("--k", type=int, default=6)
    rq.add_argument("--px", type=int, default=768)
    rq.add_argument("--det-tag", default=DET_TAG)
    rq.add_argument("--limit", type=int)
    rq.add_argument("--force", action="store_true")

    dp = sub.add_parser("disputes")
    dp.add_argument("--model")
    dp.add_argument("--variants")
    dp.add_argument("--force", action="store_true")
    dp.add_argument("--no-labels", action="store_true", help="skip same-subject judgements")

    rp = sub.add_parser("report")
    rp.add_argument("--json", action="store_true")
    sub.add_parser("agreement")

    a = ap.parse_args(argv)
    if a.cmd == "sample":
        if me.load_video_sample()["photos"] and not a.force:
            raise SystemExit("A video sample exists; labels are keyed to it. --force to redraw.")
        conn = me.open_db_readonly(a.db)
        data = me.save_video_sample(draw_sample(conn, seed=a.seed))
        by: dict = {}
        for p in data["photos"]:
            by.setdefault((p["stratum"], p["split"]), 0)
            by[(p["stratum"], p["split"])] += 1
        for (st, sp), n in sorted(by.items()):
            log(f"  {st:18} {sp:8} {n}")
        log(f"[sample] {len(data['photos'])} videos, {len(data['probe_check'])} probe-check, "
            f"recheck {len(data['recheck'])}; shortfall {data['shortfall'] or 'none'}; "
            f"unprobed (not eligible) {data['unprobed']}")
    elif a.cmd == "fetch-originals":
        conn = me.open_db_readonly(a.db) if a.local_root else None
        fetch_originals(server=a.server, local_root=a.local_root, conn=conn)
    elif a.cmd == "run-faces":
        p, sp = _params(a)
        conn = me.open_db_readonly(a.db) if a.db else None
        run_faces(a.variant, a.approach, conn=conn, grounding=a.grounding, model=a.model,
                  params=p, sparams=sp, det_tag=a.det_tag, det_size=a.det_size,
                  force=a.force, limit=a.limit)
    elif a.cmd == "run-describe":
        run_describe(a.variant, a.approach, model=a.model, k=a.k, px=a.px, force=a.force,
                     limit=a.limit, solo=a.solo)
    elif a.cmd == "run-quality":
        run_quality(a.variant, a.approach, model=a.model, k=a.k, px=a.px, force=a.force,
                    limit=a.limit, det_tag=a.det_tag)
    elif a.cmd == "disputes":
        build_video_disputes(model=a.model, force=a.force,
                             variants=a.variants.split(",") if a.variants else None)
        if not a.no_labels:
            judge_same_labels(model=a.model)
    elif a.cmd == "report":
        rep = build_report()
        if a.json:
            print(json.dumps(_jsonable(rep), indent=1, default=str))
        else:
            render(rep)
    elif a.cmd == "agreement":
        ag = self_agreement()
        sg = ag["segments"]
        log(f"segments: n={sg['n']} agreement {_f(sg['agreement'])} kappa {_f(sg['kappa'])}")
        for f_, c in ag["quality"].items():
            flag = "  <- fix the definition (kappa < 0.4)" if c["kappa"] is not None and c["kappa"] < 0.4 else ""
            log(f"{f_:10} n={c['n']} agreement {_f(c['agreement'])} kappa {_f(c['kappa'])}{flag}")


if __name__ == "__main__":
    main()

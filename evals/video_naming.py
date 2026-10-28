#!/usr/bin/env python
"""Video face-NAMING precision/recall against the owner's cluster names
(video eval round 2, docs/plans/video-eval-round2-2026-10-27.md, item 1).

Ground truth: the owner's names on the Video people tab
(``video/face_names.json``, keyed by face id) — a library person, "someone
else" (a person the owner does not name, including teammates the library
knows), or junk (skipped). The DB is opened read-only; nothing is written
except a JSON report under ``video/round2/``.

Three levels:

* **face** — every owner-named cluster face (label-segment keyframes) is
  matched on its own embedding;
* **identity** — what production stores: the F1 within-video identities from
  the cached detections (``det-default``), named on their identity encoding.
  An identity's truth is the majority owner name of the cluster faces it
  overlaps (same time ±0.26 s, IoU >= 0.3), else the nearest owner-named
  face in the same video within L2 0.7;
* **live** — the replica's video ``faces`` rows for the sample that carry a
  ``person_id``, mapped to the owner's names the same way (embedding).

Matchers: ``nn`` = the production strict rule (nearest reference within
``tol``), optionally with a ``margin`` (the nearest OTHER person's reference
must be at least ``margin`` farther). Two reference sets: ``prod`` =
``face_references`` only (what ``match_faces_to_persons`` uses) and ``eval``
= references + manual labels (what the eval's ``name_identity`` uses).

Tune on dev only:

    python evals/video_naming.py --db photo_index.db.local --split dev
    python evals/video_naming.py --db photo_index.db.local --split holdout --only <rule>
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from photosearch import model_eval as me  # noqa: E402

TOLS = (1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)
MARGINS = (0.0, 0.05, 0.10, 0.15)
EDGE_FLOORS = (0, 24, 32, 48, 64)
TRUTH_EMB_MAX = 0.7
TRUTH_DT = 0.26


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def load_refs(conn, which: str, exclude_photo_ids=()):
    rows = []
    q_ref = ("SELECT fr.person_id, e.encoding FROM face_references fr "
             "JOIN face_ref_encodings e ON e.ref_id = fr.id")
    rows += [(int(p), b) for p, b in conn.execute(q_ref)]
    if which == "eval":
        ex = set(int(x) for x in exclude_photo_ids)
        for p, ph, b in conn.execute(
                "SELECT f.person_id, f.photo_id, e.encoding FROM faces f JOIN face_encodings e "
                "ON e.face_id = f.id WHERE f.person_id IS NOT NULL AND f.match_source = 'manual'"):
            if int(ph) not in ex:
                rows.append((int(p), b))
    pids = np.array([p for p, _ in rows], int)
    encs = np.array([struct.unpack(f"{len(b) // 4}f", b) for _, b in rows], np.float32)
    return pids, encs


def person_min_dists(Q: np.ndarray, pids: np.ndarray, encs: np.ndarray):
    """For each query: (best person, d1, d2) — d2 = nearest OTHER person."""
    out = []
    if len(Q) == 0:
        return out
    uniq = np.unique(pids)
    # chunked distance, per person minimum
    for i in range(0, len(Q), 256):
        q = Q[i:i + 256]
        d = np.sqrt(np.maximum(0.0, (q ** 2).sum(1)[:, None] + (encs ** 2).sum(1)[None, :]
                               - 2 * q @ encs.T))
        per = np.stack([d[:, pids == u].min(1) for u in uniq], 1)
        order = np.argsort(per, 1)
        for r in range(len(q)):
            a = order[r, 0]
            b = order[r, 1] if per.shape[1] > 1 else None
            out.append((int(uniq[a]), float(per[r, a]),
                        float(per[r, b]) if b is not None else 9.0))
    return out


def name_rule(match, tol, margin=0.0):
    p, d1, d2 = match
    if d1 > tol or d2 - d1 < margin:
        return None
    return p


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------

def owner_truth(rec):
    k = (rec or {}).get("kind")
    if k == "person":
        return int(rec["person_id"])
    if k == "someone_else":
        return "else"
    return None                     # junk / missing


def load_cluster_faces(pids):
    """{pid: [(face_key, t, bbox, emb, truth)]} for owner-named faces."""
    names = me.read_json(me.pass_dir(me.VIDEO) / "face_names.json", {})
    out = defaultdict(list)
    base = me.pass_dir(me.VIDEO) / "face_dets"
    for pid in pids:
        j = me.read_json(base / f"{pid}.json", None)
        npz = base / f"{pid}.npz"
        if not j or not npz.exists():
            continue
        emb = np.load(npz)["emb"]
        for i, f in enumerate(j["faces"]):
            tr = owner_truth(names.get(f["id"]))
            if tr is None:
                continue
            e = emb[i].astype(np.float32)
            n = np.linalg.norm(e)
            out[pid].append((f["id"], float(f["t"]), [float(x) for x in f["bbox"]],
                             e / n if n else e, tr))
    return out


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def identity_truth(ident, cfaces):
    votes = Counter()
    for _k, t, bb, _e, tr in cfaces:
        for d in ident.detections:
            if abs(d["t"] - t) <= TRUTH_DT and _iou(d["bbox"], bb) >= 0.3:
                votes[tr] += 1
                break
    if votes:
        return votes.most_common(1)[0][0], "overlap"
    if ident.encoding is None or not cfaces:
        return None, None
    e = np.asarray(ident.encoding, np.float32)
    best = min(cfaces, key=lambda f: float(np.linalg.norm(f[3] - e)))
    if float(np.linalg.norm(best[3] - e)) <= TRUTH_EMB_MAX:
        return best[4], "embedding"
    return None, None


def emb_truth(e, cfaces):
    if not cfaces:
        return None
    best = min(cfaces, key=lambda f: float(np.linalg.norm(f[3] - e)))
    return best[4] if float(np.linalg.norm(best[3] - e)) <= TRUTH_EMB_MAX else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pr(items, rule):
    """items: [(truth, match, weight)]; rule: match -> person or None."""
    tp = fp = fn = 0.0
    fp_else = 0.0
    for truth, m, w in items:
        p = rule(m)
        is_person = isinstance(truth, int)
        if p is not None:
            if p == truth:
                tp += w
            else:
                fp += w
                fp_else += w if truth == "else" else 0
                if is_person:
                    fn += w
        elif is_person:
            fn += w
    P = tp / (tp + fp) if tp + fp else None
    R = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "fp_on_someone_else": fp_else, "P": P, "R": R}


def identity_items(conn, sample, split, det_tag="det-default"):
    from photosearch import video_analysis as va
    pids = [p["photo_id"] for p in sample["photos"]
            if me.is_labelled_clip(p) and (split == "all" or p.get("split") == split)]
    cf = load_cluster_faces(pids)
    items = []
    stats = Counter()
    for pid in pids:
        hit = me.load_detections(pid, det_tag)
        if hit is None:
            stats["no_detections"] += 1
            continue
        frames, meta = hit
        fa = va.analyze_faces(frames, meta["W"], meta["H"], meta["duration"])
        for ident in fa.identities:
            if ident.encoding is None:
                continue
            truth, how = identity_truth(ident, cf.get(pid, []))
            stats[f"truth_{how}"] += 1
            if truth is None:
                continue
            edges = [min(d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1])
                     for d in ident.detections]
            items.append({"pid": pid, "ix": ident.ix, "truth": truth,
                          "enc": np.asarray(ident.encoding, np.float32),
                          "n_det": ident.n_det, "presence_s": ident.presence_s,
                          "focus_s": ident.focus_s, "edge_med": float(np.median(edges)),
                          "edge_max": float(max(edges)),
                          "eligible": va.face_row_eligible(ident.n_det, ident.presence_s)})
    return items, cf, stats


def sweep(items, matches, *, gates, weight=lambda it: 1.0):
    rows = []
    for gname, gate in gates:
        sel = [(it["truth"], m, weight(it)) for it, m in zip(items, matches) if gate(it)]
        n_person = sum(1 for t, _, _ in sel if isinstance(t, int))
        for tol in TOLS:
            for mg in MARGINS:
                r = pr(sel, lambda m: name_rule(m, tol, mg))
                rows.append({"gate": gname, "tol": tol, "margin": mg, "n": len(sel),
                             "n_person": n_person, **r})
    return rows


def live_production(conn, sample, cf_all):
    ids = [p["photo_id"] for p in sample["photos"] if me.is_labelled_clip(p)]
    q = ("SELECT f.id, f.photo_id, f.person_id, f.match_source, e.encoding, vt.n_det, vt.presence_s, "
         "f.bbox_top, f.bbox_right, f.bbox_bottom, f.bbox_left "
         "FROM faces f JOIN face_encodings e ON e.face_id = f.id "
         "LEFT JOIN video_tracks vt ON vt.face_id = f.id "
         f"WHERE f.photo_id IN ({','.join(map(str, ids))})")
    split = {p["photo_id"]: p["split"] for p in sample["photos"]}
    out = {"rows": 0, "named": 0, "named_mappable": 0, "agree": 0, "wrong_someone_else": 0,
           "wrong_other_person": 0, "unnamed_but_owner_person": 0, "unnamed_mappable": 0,
           "by_split": {}, "examples": []}
    for r in conn.execute(q):
        fid, pid, person, src, blob, n_det, pres = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        e = np.asarray(struct.unpack(f"{len(blob) // 4}f", blob), np.float32)
        e = e / (np.linalg.norm(e) or 1)
        truth = emb_truth(e, cf_all.get(pid, []))
        out["rows"] += 1
        sp = out["by_split"].setdefault(split.get(pid), Counter())
        if person is not None:
            out["named"] += 1
            sp["named"] += 1
            if truth is None:
                continue
            out["named_mappable"] += 1
            sp["named_mappable"] += 1
            if truth == person:
                out["agree"] += 1
                sp["agree"] += 1
            elif truth == "else":
                out["wrong_someone_else"] += 1
            else:
                out["wrong_other_person"] += 1
            out["examples"].append({"face_id": fid, "photo_id": pid, "person_id": person,
                                    "owner": truth, "n_det": n_det, "presence_s": pres})
        else:
            if truth is not None:
                out["unnamed_mappable"] += 1
                if isinstance(truth, int):
                    out["unnamed_but_owner_person"] += 1
    return out


def fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    ap.add_argument("--split", default="dev", choices=("dev", "holdout", "all"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    conn = me.open_db_readonly(a.db)
    sample = me.load_video_sample()
    sample_ids = [p["photo_id"] for p in sample["photos"]]
    refs = {w: load_refs(conn, w, exclude_photo_ids=sample_ids) for w in ("prod", "eval")}
    for w, (p, e) in refs.items():
        log(f"[refs] {w}: {len(p)} encodings, {len(set(p.tolist()))} people")

    items, cf, stats = identity_items(conn, sample, a.split)
    log(f"[identity] {a.split}: {len(items)} identities with owner truth "
        f"({sum(isinstance(i['truth'], int) for i in items)} owner-named persons); {dict(stats)}")

    # face level
    face_rows = [(f, pid) for pid, fs in cf.items() for f in fs]
    report = {"split": a.split, "refs": {w: [len(p), len(set(p.tolist()))] for w, (p, _) in refs.items()},
              "identity_stats": dict(stats), "face": {}, "identity": {}, "live": None}
    gates = [
        ("all", lambda it: True),
        ("eligible", lambda it: it["eligible"]),
        *[(f"eligible+edge>={e}", (lambda e: lambda it: it["eligible"] and it["edge_med"] >= e)(e))
          for e in EDGE_FLOORS if e],
        ("ndet>=6", lambda it: it["n_det"] >= 6),
        ("ndet>=6+edge>=32", lambda it: it["n_det"] >= 6 and it["edge_med"] >= 32),
    ]
    for w, (pids, encs) in refs.items():
        Qf = np.stack([f[3] for f, _ in face_rows]) if face_rows else np.zeros((0, 512))
        mf = person_min_dists(Qf, pids, encs)
        fitems = [(f[4], m, 1.0) for (f, _), m in zip(face_rows, mf)]
        frows = []
        for tol in TOLS:
            for mg in MARGINS:
                frows.append({"tol": tol, "margin": mg, **pr(fitems, lambda m: name_rule(m, tol, mg))})
        report["face"][w] = {"n": len(fitems),
                             "n_person": sum(isinstance(t, int) for t, _, _ in fitems), "rows": frows}
        Qi = np.stack([it["enc"] for it in items]) if items else np.zeros((0, 512))
        mi = person_min_dists(Qi, pids, encs)
        report["identity"][w] = {
            "count": sweep(items, mi, gates=gates),
            "presence_weighted": sweep(items, mi, gates=gates[:2], weight=lambda it: it["presence_s"]),
        }
    report["live"] = live_production(conn, sample, load_cluster_faces(sample_ids))

    for w in refs:
        log(f"\n== face level, refs={w} (n={report['face'][w]['n']}, owner-person "
            f"{report['face'][w]['n_person']}) ==")
        for r in report["face"][w]["rows"]:
            if r["margin"] in (0.0, 0.10):
                log(f"  tol {r['tol']:.2f} m {r['margin']:.2f}: P {fmt(r['P'])} R {fmt(r['R'])} "
                    f"tp {r['tp']:.0f} fp {r['fp']:.0f} (on someone-else {r['fp_on_someone_else']:.0f})")
        log(f"== identity level, refs={w} ==")
        for r in report["identity"][w]["count"]:
            if r["margin"] in (0.0, 0.10) and r["gate"] in ("all", "eligible", "eligible+edge>=32",
                                                             "eligible+edge>=48", "ndet>=6+edge>=32"):
                log(f"  {r['gate']:18} n {r['n']:4} tol {r['tol']:.2f} m {r['margin']:.2f}: "
                    f"P {fmt(r['P'])} R {fmt(r['R'])} tp {r['tp']:.0f} fp {r['fp']:.0f}")
    lv = report["live"]
    log(f"\n== live production (replica video faces rows, whole sample) ==\n  {json.dumps({k: v for k, v in lv.items() if k != 'examples'}, default=str)}")
    out = a.out or str(me.pass_dir(me.VIDEO) / "round2" / f"naming-{a.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    me.write_json_atomic(out, json.loads(json.dumps(report, default=str)))
    log(f"[out] {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Video eval round 2 — the DEV sweep (docs/plans/video-eval-round2-2026-10-27.md).

Scores every config on the DEV split only, in one process (detections and
face analyses are cached in memory), and appends one row per config to
``video/round2/dev-sweep.json``. Nothing here reads holdout labels: the
chosen config is run once on holdout with ``video_eval.py run-faces --split
holdout`` and scored with ``video_eval.py report``.

    python evals/video_round2.py --db photo_index.db.local [--groundings a,b] [--stage all]
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import video_eval as ve  # noqa: E402
from photosearch import model_eval as me  # noqa: E402
from photosearch import video_analysis as va  # noqa: E402
from photosearch import video_subjects as vs  # noqa: E402

SPLIT = "dev"


def log(*a):
    print(*a, flush=True)


class Ctx:
    def __init__(self, conn):
        self.sample = me.load_video_sample()
        self.labels = me.load_video_subject_labels()
        self.refs = ve.load_references(conn)
        self.dev = [p for p in self.sample["photos"]
                    if me.is_labelled_clip(p) and p.get("split") == SPLIT]
        self.dets = {p["photo_id"]: ve.ensure_detections(p["photo_id"]) for p in self.dev}
        self.fa_cache: dict = {}
        import video_naming as vn
        cf = vn.load_cluster_faces([p["photo_id"] for p in self.dev])
        self.cluster_people = {pid: {f[4] for f in fs if isinstance(f[4], int)}
                               for pid, fs in cf.items()}
        self.groundings: dict = {}
        self._names: dict = {}
        orig = ve.name_identity

        def cached(encoding, refs, tolerance=None, margin=0.0):
            if encoding is None:
                return None
            k = (hashlib.sha1(json.dumps([round(x, 5) for x in encoding]).encode()).hexdigest(),
                 tolerance, margin)
            if k not in self._names:
                self._names[k] = orig(encoding, refs, tolerance, margin)
            return self._names[k]
        ve.name_identity = cached

    def grounding(self, name):
        if name not in self.groundings:
            run = me.load_run(ve.PASS_GROUND, name) or {"items": {}}
            self.groundings[name] = run["items"]
        return self.groundings[name]


def score(ctx, approach, *, params=va.DEFAULT_PARAMS, naming=ve.LEGACY_NAMING, hparams=None,
          grounding=None, sparams=None):
    items = {}
    g_items = ctx.grounding(grounding) if grounding else {}
    missing = 0
    for p in ctx.dev:
        pid = p["photo_id"]
        frames, meta = ctx.dets[pid]
        g = g_items.get(str(pid))
        if approach in ("S1", "S2", "H1") and (g is None or g.get("status") != "ok"):
            missing += 1
            continue
        items[str(pid)] = ve.predict(approach, pid, frames, meta, params=params, sparams=sparams,
                                     refs=ctx.refs, grounding=g, naming=naming, hparams=hparams,
                                     fa_cache=ctx.fa_cache)
    m = ve.subject_metrics({"items": items}, ctx.labels, ctx.sample, split=SPLIT)
    s = ve.summarize_subject(m)
    # Diagnostic only (the gate stays "who"): the owner's "who" answer is who
    # the clip is ABOUT; their cluster names say who is IN it. Presence
    # scored against who ∪ the owner's cluster names for that video.
    alt = {"tp": 0, "fp": 0, "fn": 0}
    for pid_s, it in items.items():
        who = (ctx.labels.get("who") or {}).get(pid_s) or {}
        if not who.get("done"):
            continue
        truth = set(who.get("people") or []) | ctx.cluster_people.get(int(pid_s), set())
        pc = ve.presence_counts(it.get("present"), truth)
        for k in pc:
            alt[k] += pc[k]
    return {"videos": s["videos"], "missing_grounding": missing,
            "seg_all": s["seg_acc"].get("all"), "seg_person": s["seg_acc"].get("person"),
            "seg_nonperson": s["seg_acc"].get("non-person"), "seg_n": s["seg_n"],
            "top": s["top"], "mae": s["mae"], "presence_p": s["presence_p"],
            "presence_r": s["presence_r"], "presence": m["presence"],
            "presence_p_cluster": (alt["tp"] / (alt["tp"] + alt["fp"])) if alt["tp"] + alt["fp"] else None,
            "presence_r_cluster": (alt["tp"] / (alt["tp"] + alt["fn"])) if alt["tp"] + alt["fn"] else None,
            "presence_cluster": alt}


def _cfg(approach, grounding=None, params=va.DEFAULT_PARAMS, naming=ve.LEGACY_NAMING,
         hparams=None):
    d = {"approach": approach, "grounding": grounding,
         "a_min": params.a_min, "min_frac": params.segment_min_focus_frac,
         "naming": naming._asdict()}
    if hparams is not None:
        d["hparams"] = {k: getattr(hparams, k) for k in
                        ("person_mode", "person_box_min_a", "use_faces", "nonperson_rule",
                         "propagate_min_share")}
    return d


def run_stage(ctx, configs, out_rows, label):
    for c in configs:
        t0 = time.time()
        r = score(ctx, c["approach"], params=c["params"], naming=c["naming"],
                  hparams=c.get("hparams"), grounding=c.get("grounding"))
        row = {"stage": label, **_cfg(c["approach"], c.get("grounding"), c["params"],
                                      c["naming"], c.get("hparams")), **r,
               "s": round(time.time() - t0, 1)}
        out_rows.append(row)
        f = lambda x: "  - " if x is None else f"{x:.2f}"  # noqa: E731
        hp = row.get("hparams") or {}
        log(f"[{label}] {c['approach']:2} g={str(c.get('grounding')):22} a_min={row['a_min']:.2f} "
            f"mf={row['min_frac']:.1f} name=({row['naming']['tolerance']},{row['naming']['margin']},"
            f"{'E' if row['naming']['eligible_only'] else '-'},{row['naming']['min_presence_s']}) "
            f"{hp.get('person_mode','')}/{hp.get('person_box_min_a','')}/"
            f"{'F' if hp.get('use_faces') else ('-' if hp else '')}/{hp.get('nonperson_rule','')} "
            f"| all {f(r['seg_all'])} pers {f(r['seg_person'])} non {f(r['seg_nonperson'])} "
            f"top {f(r['top'])} mae {f(r['mae'])} presP {f(r['presence_p'])} R {f(r['presence_r'])} "
            f"(vs who+clusters P {f(r['presence_p_cluster'])} R {f(r['presence_r_cluster'])})")


NAMINGS = {
    "legacy": ve.LEGACY_NAMING,
    "m10E": ve.NamingRule(1.15, 0.10, True, 0.0),
    "m15E": ve.NamingRule(1.15, 0.15, True, 0.0),
    "m15E2s": ve.NamingRule(1.15, 0.15, True, 2.0),
    "m15E5s": ve.NamingRule(1.15, 0.15, True, 5.0),
    "t100m15E": ve.NamingRule(1.00, 0.15, True, 0.0),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    ap.add_argument("--stage", default="all", choices=("all", "baseline", "naming", "h1", "prompts", "propagate"))
    ap.add_argument("--groundings", default="qwen25vl-operator,qwen25vl-peoplefirst")
    ap.add_argument("--h1-best", help="JSON of the H1 configs to re-run on the prompt groundings")
    a = ap.parse_args(argv)
    conn = me.open_db_readonly(a.db)
    ctx = Ctx(conn)
    log(f"[round2] dev videos {len(ctx.dev)}; refs {len(ctx.refs[0])}")
    out_path = me.pass_dir(me.VIDEO) / "round2" / "dev-sweep.json"
    os.makedirs(out_path.parent, exist_ok=True)
    rows = me.read_json(out_path, {"rows": []})["rows"]
    P = va.DEFAULT_PARAMS
    base_g = "qwen25vl-frame"
    stages = [a.stage] if a.stage != "all" else ["baseline", "naming", "h1"]
    for st in stages:
        cfgs = []
        if st == "baseline":
            cfgs += [{"approach": ap_, "params": P, "naming": NAMINGS["legacy"],
                      "grounding": base_g if ap_ != "F1" else None}
                     for ap_ in ("F1", "S1", "S2")]
        elif st == "naming":
            for nk in ("m10E", "m15E", "m15E2s", "m15E5s", "t100m15E"):
                for ap_ in ("F1", "S2"):
                    cfgs.append({"approach": ap_, "params": P, "naming": NAMINGS[nk],
                                 "grounding": base_g if ap_ != "F1" else None})
        elif st == "h1":
            for a_min, mf, nk, pm, pbm, uf, npr in itertools.product(
                    (0.03, 0.05, 0.08), (0.3, 0.5), ("m10E", "m15E"),
                    ("primary", "any"), (0.05, 0.10, 0.20), (True, False),
                    ("salience", "first")):
                if pm == "primary" and pbm != 0.10:
                    continue
                if not uf and (a_min, mf) != (0.03, 0.3):
                    continue                  # without faces the face params do nothing
                cfgs.append({"approach": "H1", "grounding": base_g,
                             "params": replace(P, a_min=a_min, segment_min_focus_frac=mf),
                             "naming": NAMINGS[nk],
                             "hparams": replace(vs.DEFAULT_HYBRID_PARAMS, person_mode=pm,
                                                person_box_min_a=pbm, use_faces=uf,
                                                nonperson_rule=npr)})
        elif st == "propagate":
            for g, nk, pms in itertools.product(
                    (base_g, "qwen25vl-peoplefirst"), ("m10E", "m15E"), (0.5, 0.7, 0.9)):
                cfgs.append({"approach": "H1", "grounding": g, "params": P,
                             "naming": NAMINGS[nk],
                             "hparams": replace(vs.DEFAULT_HYBRID_PARAMS, person_box_min_a=0.05,
                                                propagate_min_share=pms)})
        elif st == "prompts":
            best = json.loads(a.h1_best) if a.h1_best else []
            for g in a.groundings.split(","):
                cfgs.append({"approach": "S1", "params": P, "naming": NAMINGS["legacy"],
                             "grounding": g})
                cfgs.append({"approach": "S2", "params": P, "naming": NAMINGS["m15E"],
                             "grounding": g})
                for b in best:
                    cfgs.append({"approach": "H1", "grounding": g,
                                 "params": replace(P, a_min=b["a_min"],
                                                   segment_min_focus_frac=b["min_frac"]),
                                 "naming": ve.NamingRule(**b["naming"]),
                                 "hparams": replace(vs.DEFAULT_HYBRID_PARAMS, **b["hparams"])})
        run_stage(ctx, cfgs, rows, st)
        me.write_json_atomic(out_path, {"split": SPLIT, "rows": rows})
    log(f"[out] {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

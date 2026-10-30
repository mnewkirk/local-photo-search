#!/usr/bin/env python
"""people-bind eval (docs/plans/description-names.md): score the binder against
HAND labels. The harness calls the PRODUCTION functions (`people_bind.bind_photo`
/ `bind_with_model`), never a copy.

  sample   stratified, seeded sample -> evals/people-bind/sample.json
  sheet    contact sheet (numbered boxes = face ids) + labels.json skeleton
  run      one variant's predictions (cached, resumable): --tier rule|vlm|both
  report   score + the pre-registered gates, per stratum and per tier

The DB is only ever opened READ-ONLY (mode=ro), never through PhotoDB. Run
against a replica copy, never the live NAS. See evals/people-bind/README.md.
"""
import argparse
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photosearch import people_bind as PB                      # noqa: E402
from photosearch import people_bind_eval as E                  # noqa: E402
from photosearch import people_bind_store as store             # noqa: E402

HERE = Path(__file__).resolve().parent / "people-bind"
SAMPLE = HERE / "sample.json"
LABELS = HERE / "labels.json"
RUNS = HERE / "runs"
SHEET = HERE / "sheet"
TRUSTED = ("manual", "strict", "merge_review")


def ro(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = None
    return con


def _jload(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def _jsave(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def cmd_sample(a):
    con = ro(a.db)
    rows = con.execute(
        "SELECT p.id FROM photos p WHERE p.description IS NOT NULL AND p.media_type IS NULL "
        "AND p.date_taken >= ? AND p.id IN (SELECT photo_id FROM faces WHERE person_id IS NOT NULL "
        f"AND match_source IN ({','.join('?' * len(TRUSTED))})) ORDER BY p.id",
        (a.date_from, *TRUSTED)).fetchall()
    strata = {"single": [], "multi_2_3": [], "multi_4_6": []}
    for (pid,) in rows:
        inp = store.claim_input(con, pid)
        n = len(inp["faces"]) if inp else 0
        if n == 1:
            strata["single"].append(pid)
        elif 2 <= n <= 3:
            strata["multi_2_3"].append(pid)
        elif 4 <= n <= PB.MAX_FACES:
            strata["multi_4_6"].append(pid)
    rng = random.Random(a.seed)
    want = {"single": a.single, "multi_2_3": a.multi_2_3, "multi_4_6": a.multi_4_6}
    sample = _jload(SAMPLE, {})
    for st, pool in strata.items():
        pick = rng.sample(pool, min(want[st], len(pool)))
        for pid in pick:
            sample.setdefault(str(pid), {"stratum": st, "split": E.split_of(pid)})
        print(f"{st}: pool {len(pool)}, sampled {len(pick)}")
    for spec in a.add:                       # ID:adversarial -> hand-picked hard cases
        pid, st = spec.split(":")
        sample[pid] = {"stratum": st, "split": E.split_of(int(pid))}
    _jsave(SAMPLE, sample)
    print(f"wrote {SAMPLE} ({len(sample)} photos)")


def cmd_sheet(a):
    from photosearch import model_eval
    con = ro(a.db)
    sample = _jload(SAMPLE, {})
    labels = _jload(LABELS, {})
    SHEET.mkdir(parents=True, exist_ok=True)
    html = ["<meta charset=utf-8><style>body{font:14px sans-serif;max-width:1100px;margin:auto}"
            "section{margin:2em 0;border-bottom:1px solid #ccc}img{max-width:100%}</style>"]
    for pid_s, meta in sorted(sample.items(), key=lambda kv: int(kv[0])):
        pid = int(pid_s)
        inp = store.claim_input(con, pid)
        if not inp:
            continue
        faces = inp["faces"]
        order = sorted(f["face_id"] for f in faces)
        img = model_eval.original_path(pid, a.server)
        (SHEET / f"{pid}.jpg").write_bytes(__import__("base64").b64decode(
            PB.draw_overlay(str(img), faces, order)))
        phrases = [p["text"] for p in PB.find_singular_phrases(inp["description"])]
        lab = labels.setdefault(pid_s, {"split": meta["split"], "stratum": meta["stratum"],
                                        "mentions": [], "bad_labels": []})
        have = {m["phrase"] for m in lab["mentions"]}
        for ph in phrases:
            if ph not in have:
                lab["mentions"].append({"phrase": ph, "gold": None})   # fill in
        src = {r[0]: (r[1], r[2]) for r in con.execute(
            "SELECT id, person_id, match_source FROM faces WHERE photo_id = ?", (pid,))}
        html.append(f"<section><h3>{pid} - {meta['stratum']} ({meta['split']})</h3>"
                    f"<p>{inp['description']}</p><img src='{pid}.jpg'>"
                    f"<p>box n = face id: "
                    f"{', '.join(f'{n}={fid} [{src.get(fid, (None, None))[1]}]' for n, fid in enumerate(order, 1))}</p>"
                    f"<p>candidate phrases: {phrases or 'none found - add every person phrase by hand'}</p></section>")
    (SHEET / "index.html").write_text("\n".join(html))
    _jsave(LABELS, labels)
    print(f"open {SHEET / 'index.html'}; fill `gold` in {LABELS}: [face ids] | \"NONE\" | \"AMBIGUOUS\";"
          f" list a face id under bad_labels when its PERSON label is wrong.")


def cmd_run(a):
    from photosearch import model_eval
    if a.model:
        model_eval.pin_role_model("bind", a.model)
    con = ro(a.db)
    sample = _jload(SAMPLE, {})
    out_path = RUNS / f"{a.variant}.json"
    runs = _jload(out_path, {"variant": a.variant, "tier": a.tier, "model": a.model, "preds": {}})
    for pid_s in sorted(sample, key=int):
        if pid_s in runs["preds"] and not a.force:
            continue
        pid = int(pid_s)
        inp = store.claim_input(con, pid)
        if not inp:
            continue
        photo = {"id": pid, "people_bind_input": inp}
        method, slots = None, None
        rb = PB.rule_bind(inp["description"], inp["faces"], inp.get("subject_boxes"),
                          inp["image_width"], inp["image_height"])
        if a.tier in ("rule", "both") and rb is not None:
            method, slots = "rule", rb
        elif a.tier == "rule":
            method, slots = "rule", []                 # rule tier abstains
        else:
            img = str(model_eval.original_path(pid, a.server))
            try:
                slots = PB.bind_with_model(img, inp["description"], inp["faces"], pid,
                                           a.model or "bind")
            except Exception as e:                     # an unusable answer is a failure, not a result
                print(f"  {pid}: {type(e).__name__}: {e}")
                continue
            method = "vlm"
        runs["preds"][pid_s] = {"method": method, "slots": slots}
        _jsave(out_path, runs)
        print(f"{pid}: {method} {len(slots or [])} slot(s)")


def cmd_report(a):
    labels = _jload(LABELS, {})
    con = ro(a.db)
    srcs = {r[0]: r[1] for r in con.execute("SELECT id, match_source FROM faces")}
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        rep = E.score(labels, run["preds"], srcs, split=a.split)
        rows = E.gates(rep)
        print(f"\n== {run['variant']} ({run.get('tier')}, {run.get('model')}) split={a.split}: "
              f"{E.verdict(rows)}")
        print(f"   photos {rep['photos']}  bound {rep['bound']}  coverage {rep['coverage']}  "
              f"abstain-correct {rep['abstain_correct']}  wrong-bind-rate {rep['wrong_bind_rate']}")
        for st, b in sorted(rep["by_stratum"].items()):
            print(f"   {st:12} {b}")
        for name, ok, detail in rows:
            print(f"   [{'ok' if ok else ('??' if ok is None else 'FAIL')}] {name}: {detail}")
        for w in rep["wrong_cases"][:10]:
            print(f"   WRONG {w}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--db", required=True)
    s.add_argument("--date-from", default="2026-01-01")
    s.add_argument("--seed", type=int, default=20261030)
    s.add_argument("--single", type=int, default=40)
    s.add_argument("--multi-2-3", dest="multi_2_3", type=int, default=70)
    s.add_argument("--multi-4-6", dest="multi_4_6", type=int, default=60)
    s.add_argument("--add", action="append", default=[], help="ID:stratum, e.g. 248080:adversarial")
    s.set_defaults(fn=cmd_sample)
    sh = sub.add_parser("sheet")
    sh.add_argument("--db", required=True)
    sh.add_argument("--server", default=os.environ.get("PHOTOSEARCH_EVAL_SERVER", "http://localhost:8001"))
    sh.set_defaults(fn=cmd_sheet)
    r = sub.add_parser("run")
    r.add_argument("--db", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--tier", choices=("rule", "vlm", "both"), default="both")
    r.add_argument("--model", default=None)
    r.add_argument("--force", action="store_true")
    r.add_argument("--server", default=os.environ.get("PHOTOSEARCH_EVAL_SERVER", "http://localhost:8001"))
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("--db", required=True)
    p.add_argument("--split", choices=("dev", "holdout"), default="dev")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

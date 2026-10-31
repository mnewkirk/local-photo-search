#!/usr/bin/env python
"""people-bind eval (docs/plans/description-names.md): score the binder against
HAND labels. The harness calls the PRODUCTION functions (`people_bind.rule_bind`
/ `bind_with_model`), never a copy.

  sample   stratified, seeded sample (+ heuristic adversarial picks) -> sample.json
  sheet    interactive labelling page (numbered boxes, names, click to label)
           + labels.json skeleton
  check    validate labels.json (or --import a file downloaded from the sheet)
  run      one variant's predictions (cached, resumable): --tier rule|vlm|both
  report   score + the pre-registered gates, per stratum and per tier

The DB is only ever opened READ-ONLY (mode=ro), never through PhotoDB. See
evals/people-bind/README.md.
"""
import argparse
import hashlib
import html as _html
import json
import os
import random
import re
import sqlite3
import sys
import time
import urllib.request
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
SHEET_PX = 1600          # the labelling overlay; the model sees PB._OVERLAY_MAX_PX


def ro(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = None
    return con


def _jload(p, default):
    return json.loads(p.read_text()) if p.exists() else default


def _jsave(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _fetch(server, photo_id, kind="full", timeout=120):
    """model_eval.fetch_from_server, stating who asks and why (request log)."""
    req = urllib.request.Request(
        f"{server.rstrip('/')}/api/photos/{int(photo_id)}/{kind}",
        headers={"X-Photosearch-Source": "eval",
                 "X-Photosearch-Intent": "people-bind eval: original for the numbered-box overlay"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _original(photo_id, server):
    from photosearch import model_eval
    return model_eval.original_path(photo_id, server, fetch=_fetch)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

def _stratum(n):
    if n == 1:
        return "single"
    if 2 <= n <= 3:
        return "multi_2_3"
    if 4 <= n <= PB.MAX_FACES:
        return "multi_4_6"
    return None


_BACK_RE = re.compile(r"\b(?:from behind|facing away|back to the camera|backs? (?:is |are )?turned|"
                      r"with (?:his|her|their) backs?)\b", re.I)
_KEEPER_RE = re.compile(r"\b(?:goalkeeper|goalie|keeper)\b", re.I)
_SAME_KIT_RE = re.compile(r"\b(?:matching|identical|same(?: \w+)? (?:jerseys?|uniforms?|kits?|shirts?)|"
                          r"both (?:wearing|in|dressed))\b", re.I)
_COUNTED_RE = re.compile(r"\b(two|three)\s+(?:[A-Za-z'\-]+\s+){0,3}?(?:" + PB._PLURAL_ALT + r")\b", re.I)


def adversarial_kinds(inp):
    """Why a photo is a hard case (heuristics; the labeller decides the gold).
    Kinds follow the plan's adversarial list."""
    desc, faces = inp["description"], inp["faces"]
    w, h = inp["image_width"] or 0, inp["image_height"] or 0
    n, kinds = len(faces), []
    if n == 1 and not inp.get("subject_boxes"):
        kinds.append("lone_face_no_subject_boxes")       # parent detected, boy undetected?
    if n == 1 and inp.get("subject_boxes") and not PB._in_person_box(
            faces[0], inp["subject_boxes"], w, h):
        kinds.append("lone_face_outside_main_subject")   # a sideline face
    if _BACK_RE.search(desc):
        kinds.append("back_turned")
    if n >= 2 and _KEEPER_RE.search(desc):
        kinds.append("goalkeeper_vs_players")
    if n >= 2 and _SAME_KIT_RE.search(desc):
        kinds.append("identical_kit")
    m = _COUNTED_RE.search(desc)
    if m and {"two": 2, "three": 3}[m.group(1).lower()] > n:
        kinds.append("counted_group_fewer_boxes")        # "two players in blue", one visible
    if w and h:
        for f in faces:
            top, right, bottom, left = f["bbox"]
            if left <= 0.01 * w or right >= 0.99 * w or top <= 0.01 * h or bottom >= 0.99 * h:
                kinds.append("face_cut_by_frame_edge")
                break
    return kinds


def cmd_sample(a):
    con = ro(a.db)
    collection = getattr(a, "collection", None)
    only_split = getattr(a, "only_split", None)
    n_adv = getattr(a, "adversarial", 0) or 0
    sql = ("SELECT p.id FROM photos p WHERE p.description IS NOT NULL AND p.media_type IS NULL "
           "AND p.date_taken >= ? AND p.id IN (SELECT photo_id FROM faces WHERE person_id IS NOT NULL "
           f"AND match_source IN ({','.join('?' * len(TRUSTED))}))")
    params = [a.date_from, *TRUSTED]
    if collection:
        sql += (" AND p.id IN (SELECT cp.photo_id FROM collection_photos cp JOIN collections c "
                "ON c.id = cp.collection_id WHERE c.name LIKE ?)")
        params.append(collection)
    rows = con.execute(sql + " ORDER BY p.id", params).fetchall()
    strata = {"single": [], "multi_2_3": [], "multi_4_6": []}
    inputs = {}
    for (pid,) in rows:
        if only_split and E.split_of(pid) != only_split:
            continue
        inp = store.claim_input(con, pid)
        st = _stratum(len(inp["faces"]) if inp else 0)
        if st:
            strata[st].append(pid)
            inputs[pid] = inp
    rng = random.Random(a.seed)
    want = {"single": a.single, "multi_2_3": a.multi_2_3, "multi_4_6": a.multi_4_6}
    sample = _jload(SAMPLE, {})
    for st, pool in strata.items():
        pool = [p for p in pool if str(p) not in sample]
        pick = rng.sample(pool, min(want[st], len(pool)))
        for pid in pick:
            sample[str(pid)] = {"stratum": st, "split": E.split_of(pid)}
        print(f"{st}: pool {len(pool)}, sampled {len(pick)}")
    if n_adv:
        by_kind = {}
        for pid in sorted(inputs):
            if str(pid) in sample:
                continue
            for k in adversarial_kinds(inputs[pid]):
                by_kind.setdefault(k, []).append(pid)
        for pool in by_kind.values():
            rng.shuffle(pool)
        got, kinds = 0, sorted(by_kind)
        while got < n_adv and any(by_kind.values()):      # round-robin over the kinds
            for k in kinds:
                while by_kind[k] and str(by_kind[k][0]) in sample:
                    by_kind[k].pop(0)
                if by_kind[k] and got < n_adv:
                    pid = by_kind[k].pop(0)
                    sample[str(pid)] = {"stratum": "adversarial", "split": E.split_of(pid),
                                        "why": k}
                    got += 1
        print(f"adversarial: {got} picked from " +
              ", ".join(f"{k} {len(v)}" for k, v in sorted(by_kind.items())) + " (left in pool)")
    for spec in a.add:                       # ID:stratum -> hand-picked hard cases
        pid, st = spec.split(":")
        sample[pid] = {"stratum": st, "split": E.split_of(int(pid))}
    _jsave(SAMPLE, sample)
    counts = {}
    for m in sample.values():
        key = (m["stratum"], m["split"])
        counts[key] = counts.get(key, 0) + 1
    print(f"wrote {SAMPLE} ({len(sample)} photos): " +
          ", ".join(f"{s}/{sp} {n}" for (s, sp), n in sorted(counts.items())))


# ---------------------------------------------------------------------------
# sheet (interactive labelling page) + check
# ---------------------------------------------------------------------------

def candidate_phrases(description):
    """Phrases to pre-fill: the binder's singular person phrases plus counted
    groups ("two boys"). The labeller adds anything else ("the midfielder")."""
    out = [p["text"] for p in PB.find_singular_phrases(description)]
    for m in _COUNTED_RE.finditer(description):
        if m.group(0) not in out:
            out.append(m.group(0))
    seen, uniq = set(), []
    for p in sorted(out, key=description.find):
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _face_rows(con, pid):
    return {r[0]: {"face_id": r[0], "name": r[1], "source": r[2]} for r in con.execute(
        "SELECT f.id, pe.name, f.match_source FROM faces f "
        "LEFT JOIN persons pe ON pe.id = f.person_id WHERE f.photo_id = ?", (pid,))}


def cmd_sheet(a):
    import base64
    con = ro(a.db)
    sample = _jload(SAMPLE, {})
    labels = _jload(LABELS, {})
    SHEET.mkdir(parents=True, exist_ok=True)
    items, failed = [], []
    for pid_s, meta in sorted(sample.items(), key=lambda kv: int(kv[0])):
        pid = int(pid_s)
        inp = store.claim_input(con, pid)
        if not inp:
            failed.append((pid, "no description"))
            continue
        faces = inp["faces"]
        order = sorted(f["face_id"] for f in faces)
        jpg = SHEET / f"{pid}.jpg"
        if not jpg.exists():
            try:
                img = _original(pid, a.server)
                jpg.write_bytes(base64.b64decode(PB.draw_overlay(str(img), faces, order,
                                                                 max_px=SHEET_PX)))
            except Exception as e:                     # noqa: BLE001 - report, keep going
                failed.append((pid, f"{type(e).__name__}: {e}"))
                continue
        lab = labels.setdefault(pid_s, {"split": meta["split"], "stratum": meta["stratum"],
                                        "mentions": [], "bad_labels": []})
        have = {m["phrase"] for m in lab["mentions"]}
        for ph in candidate_phrases(inp["description"]):
            if ph not in have:
                lab["mentions"].append({"phrase": ph, "gold": None})   # fill in
        rows = _face_rows(con, pid)
        drawn = set(order)
        items.append({
            "pid": pid_s, "split": meta["split"], "stratum": meta["stratum"],
            "why": meta.get("why"), "description": inp["description"],
            "boxes": [dict(rows.get(fid, {"face_id": fid, "name": None, "source": None}), n=n)
                      for n, fid in enumerate(order, 1)],
            "unboxed": [r for fid, r in sorted(rows.items())
                        if fid not in drawn and r["name"] and r["source"] in TRUSTED],
        })
    _jsave(LABELS, labels)
    page = (_SHEET_HTML
            .replace("__ITEMS__", json.dumps(items).replace("</", "<\\/"))
            .replace("__LABELS__", json.dumps(labels).replace("</", "<\\/"))
            .replace("__KEY__", _html.escape(hashlib.sha1(
                json.dumps(sorted(sample)).encode()).hexdigest()[:10])))
    (SHEET / "index.html").write_text(page)
    for pid, why in failed:
        print(f"  ! {pid}: {why}")
    print(f"{len(items)} photos on {SHEET / 'index.html'} ({len(failed)} failed); "
          f"skeleton in {LABELS}")


def _photos_now(con, pids):
    out = {}
    for pid_s in pids:
        inp = store.claim_input(con, int(pid_s))
        if not inp:
            out[pid_s] = None
            continue
        out[pid_s] = {"description": inp["description"],
                      "offered": {f["face_id"] for f in inp["faces"]},
                      "faces": {r[0] for r in con.execute(
                          "SELECT id FROM faces WHERE photo_id = ?", (int(pid_s),))}}
    return out


def cmd_check(a):
    src = Path(a.import_file) if getattr(a, "import_file", None) else LABELS
    labels = _jload(src, None)
    if labels is None:
        raise SystemExit(f"no labels at {src}")
    sample = _jload(SAMPLE, {})
    con = ro(a.db)
    res = E.check_labels(labels, sample, _photos_now(con, labels))
    for w in res["warnings"]:
        print(f"  warn: {w}")
    for e in res["errors"]:
        print(f"  ERROR: {e}")
    print(f"{res['photos']} photos, {res['mentions']} phrases: {res['labelled']} labelled, "
          f"{res['unlabelled']} still blank; {len(res['missing'])} sampled photos missing; "
          f"{len(res['unreviewed'])} photos not ticked as complete; "
          f"{len(res['errors'])} error(s)")
    if res["errors"]:
        raise SystemExit(1)
    if src != LABELS:
        _jsave(LABELS, labels)
        print(f"imported into {LABELS}")


# ---------------------------------------------------------------------------
# run / report
# ---------------------------------------------------------------------------

def _sha12(b):
    return hashlib.sha1(b).hexdigest()[:12]


def run_spec(tier, model, max_px, second_call):
    """What a run measured. Resuming under a different spec refuses; the
    report marks a run whose prompt/code differ from today's as STALE."""
    return {"tier": tier, "model": model, "max_px": max_px, "second_call": second_call,
            "prompt_sha": _sha12(PB.PEOPLE_BIND_PROMPT.encode()),
            "code_sha": _sha12(Path(PB.__file__).read_bytes()),
            "bind_version": PB.BIND_VERSION,
            "reasoning_effort": os.environ.get("PHOTOSEARCH_LLM_REASONING_EFFORT", "")}


def cmd_run(a):
    from photosearch import model_eval
    from photosearch.describe import UnusableAnswer
    from photosearch.rerun import FLEET_REASONING_EFFORT
    # the fleet's setting: reasoning models otherwise spend max_tokens thinking
    os.environ.setdefault("PHOTOSEARCH_LLM_REASONING_EFFORT", FLEET_REASONING_EFFORT)
    if a.model:
        model_eval.pin_role_model("bind", a.model)
    max_px = getattr(a, "max_px", None) or PB._OVERLAY_MAX_PX
    second_call = not getattr(a, "one_call", False)
    spec = run_spec(a.tier, a.model, max_px, second_call)
    con = ro(a.db)
    sample = _jload(SAMPLE, {})
    out_path = RUNS / f"{a.variant}.json"
    runs = _jload(out_path, None)
    if runs is not None and runs.get("spec") != spec:
        if not a.force:
            raise SystemExit(f"{out_path} was run under a different spec "
                             f"({runs.get('spec')}); --force starts it over")
        runs = None
    if runs is None:
        runs = {"variant": a.variant, "tier": a.tier, "model": a.model, "spec": spec,
                "preds": {}}
    fails = 0
    for pid_s in sorted(sample, key=int):
        if pid_s in runs["preds"] and not a.force:
            continue
        pid = int(pid_s)
        inp = store.claim_input(con, pid)
        if not inp:
            continue
        t0, calls, error = time.time(), 0, None
        rb = PB.rule_bind(inp["description"], inp["faces"], inp.get("subject_boxes"),
                          inp["image_width"], inp["image_height"])
        if a.tier in ("rule", "both") and rb is not None:
            method, slots = "rule", rb
        elif a.tier == "rule":
            method, slots = "rule", []                 # rule tier abstains
        else:
            method = "vlm"
            calls = 1 if len(inp["faces"]) == 1 or not second_call else 2
            try:
                img = str(_original(pid, a.server))
                slots = PB.bind_with_model(img, inp["description"], inp["faces"], pid,
                                           a.model or "bind", max_px=max_px,
                                           second_call=second_call)
            except UnusableAnswer as e:
                # production: an attempt spent, nothing written -> generic text. Scored as
                # abstaining, counted as an error.
                slots, error = [], f"{type(e).__name__}: {e}"
            except Exception as e:                     # noqa: BLE001 - transport: never cached
                slots, error = None, f"{type(e).__name__}: {e}"
            if slots is None:
                fails += 1
                print(f"  {pid}: no answer ({error or 'backend returned nothing'}) - not cached")
                if fails >= model_eval.CONSECUTIVE_FAIL_WARN:
                    raise SystemExit(f"{fails} photos in a row got no answer: is LM Studio "
                                     f"up with the model loaded? Nothing was cached for them.")
                continue
        fails = 0
        pred = {"method": method, "slots": slots, "secs": round(time.time() - t0, 2),
                "calls": calls}
        if error:
            pred["error"] = error
        runs["preds"][pid_s] = pred
        _jsave(out_path, runs)
        print(f"{pid}: {method} {len(slots)} slot(s)" + (f"  [{error}]" if error else ""))


def cmd_report(a):
    labels = _jload(LABELS, {})
    con = ro(a.db)
    srcs = {r[0]: r[1] for r in con.execute("SELECT id, match_source FROM faces")}
    descs = {p: (_photos_now(con, [p])[p] or {}).get("description") for p in labels}
    for path in sorted(RUNS.glob("*.json")):
        run = json.loads(path.read_text())
        sp = run.get("spec") or {}
        now = run_spec(sp.get("tier"), sp.get("model"), sp.get("max_px"), sp.get("second_call"))
        stale = any(sp.get(k) != now[k] for k in ("prompt_sha", "code_sha", "bind_version"))
        rep = E.score(labels, run["preds"], srcs, split=a.split, descriptions=descs)
        rows = E.gates(rep)
        scored = {p: v for p, v in run["preds"].items()
                  if labels.get(p, {}).get("split") == a.split}
        vlm = [v for v in scored.values() if v.get("method") == "vlm"]
        errs = sum(1 for v in scored.values() if v.get("error"))
        secs = (sum(v.get("secs", 0) for v in vlm) / len(vlm)) if vlm else None
        print(f"\n== {run['variant']} ({run.get('tier')}, {run.get('model')}, "
              f"{sp.get('max_px')} px, {'2 calls' if sp.get('second_call', True) else '1 call'}) "
              f"split={a.split}: {E.verdict(rows)}" + ("  ** STALE spec **" if stale else ""))
        print(f"   photos {rep['photos']}  bound {rep['bound']}  coverage {rep['coverage']}  "
              f"abstain-correct {rep['abstain_correct']}  wrong-bind-rate {rep['wrong_bind_rate']}")
        print(f"   unusable answers {errs}  vlm photos {len(vlm)}  "
              f"s/photo {None if secs is None else round(secs, 1)}  "
              f"user-visible wrong names {rep['visible_wrong']}")
        for st, b in sorted(rep["by_stratum"].items()):
            print(f"   {st:12} {b}")
        for name, ok, detail in rows:
            print(f"   [{'ok' if ok else ('??' if ok is None else 'FAIL')}] {name}: {detail}")
        for w in rep["wrong_cases"][:10]:
            print(f"   WRONG {w}")


def cmd_serve(a):
    """Serve the labelling page over HTTP so another machine (the Mac, over Tailscale) can label.
    The page loads and autosaves sheet/saved-labels.json here; feed that file to `check --import`."""
    import http.server
    import shutil
    saved = SHEET / "saved-labels.json"

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kw):
            super().__init__(*args, directory=str(SHEET), **kw)

        def do_PUT(self):
            if self.path != "/saved-labels.json":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= 5_000_000:
                return self.send_error(413)
            try:
                data = json.loads(self.rfile.read(n))
                assert isinstance(data, dict) and all(
                    isinstance(v, dict) and isinstance(v.get("mentions"), list) for v in data.values())
            except Exception:  # anything unparseable is the client's error, not a crash
                return self.send_error(400, "not a labels file")
            if saved.exists():
                shutil.copyfile(saved, saved.with_suffix(".json.bak"))
            tmp = saved.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
            tmp.replace(saved)
            self.send_response(204)
            self.end_headers()

        def log_message(self, fmt, *args):
            if self.command == "PUT" or not str(args[1] if len(args) > 1 else "").startswith(("2", "3")):
                super().log_message(fmt, *args)

    if not (SHEET / "index.html").exists():
        sys.exit("no sheet yet: run `sheet` first")
    srv = http.server.ThreadingHTTPServer((a.host, a.port), H)
    print(f"labelling page: http://{a.host}:{a.port}/   (labels autosave to {saved})")
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--db", required=True)
    s.add_argument("--date-from", default="2026-01-01")
    s.add_argument("--collection", default=None,
                   help="SQL LIKE on collections.name, e.g. '%%soccer%%' (the pilot scope)")
    s.add_argument("--seed", type=int, default=20261030)
    s.add_argument("--single", type=int, default=40)
    s.add_argument("--multi-2-3", dest="multi_2_3", type=int, default=70)
    s.add_argument("--multi-4-6", dest="multi_4_6", type=int, default=60)
    s.add_argument("--adversarial", type=int, default=0,
                   help="add N heuristic hard cases (round-robin over the plan's kinds)")
    s.add_argument("--only-split", choices=("dev", "holdout"), default=None,
                   help="top up one split only (e.g. more holdout for the 150-bound gate)")
    s.add_argument("--add", action="append", default=[], help="ID:stratum, e.g. 248080:adversarial")
    s.set_defaults(fn=cmd_sample)
    sh = sub.add_parser("sheet")
    sh.add_argument("--db", required=True)
    sh.add_argument("--server", default=os.environ.get("PHOTOSEARCH_EVAL_SERVER", "http://localhost:8001"))
    sh.set_defaults(fn=cmd_sheet)
    c = sub.add_parser("check")
    c.add_argument("--db", required=True)
    c.add_argument("--import", dest="import_file", default=None,
                   help="validate this file (downloaded from the sheet) and copy it to labels.json")
    c.set_defaults(fn=cmd_check)
    r = sub.add_parser("run")
    r.add_argument("--db", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--tier", choices=("rule", "vlm", "both"), default="both")
    r.add_argument("--model", default=None)
    r.add_argument("--max-px", dest="max_px", type=int, default=None,
                   help=f"overlay long edge (default {PB._OVERLAY_MAX_PX}; ablation: 1536)")
    r.add_argument("--one-call", dest="one_call", action="store_true",
                   help="ablation: skip the re-shuffled second call on multi-face photos")
    r.add_argument("--force", action="store_true")
    r.add_argument("--server", default=os.environ.get("PHOTOSEARCH_EVAL_SERVER", "http://localhost:8001"))
    r.set_defaults(fn=cmd_run)
    sv = sub.add_parser("serve", help="serve the labelling page over HTTP; edits autosave here")
    sv.add_argument("--host", default="127.0.0.1",
                    help="bind address, e.g. this machine's Tailscale IP to label from another machine")
    sv.add_argument("--port", type=int, default=8765)
    sv.set_defaults(fn=cmd_serve)
    p = sub.add_parser("report")
    p.add_argument("--db", required=True)
    p.add_argument("--split", choices=("dev", "holdout"), default="dev")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


_SHEET_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>people-bind labels</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#ddd;--on:#1565c0;--onfg:#fff;--warn:#b71c1c;--card:#f6f7f9}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e8e8e8;--mut:#9aa;--line:#333;--on:#64b5f6;--onfg:#000;--warn:#ef9a9a;--card:#1f2228}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2;display:flex;gap:12px;flex-wrap:wrap;align-items:center}
main{max-width:1200px;margin:auto;padding:0 16px}
section{border-bottom:2px solid var(--line);padding:18px 0}
section.done h3::after{content:"  \2713 done";color:var(--on)}
.grid{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}
@media (max-width:800px){.grid{grid-template-columns:1fr}}
img{width:100%;height:auto;border-radius:4px;cursor:zoom-in}
.desc{background:var(--card);padding:8px 10px;border-radius:4px}
.mut{color:var(--mut);font-size:13px}
.m{border-top:1px solid var(--line);padding:6px 0}
.ph{font-weight:600}
.desc b{font-weight:700;background:rgba(255,193,7,.22);border-radius:2px;padding:0 1px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:4px;padding:2px 9px;margin:2px;cursor:pointer}
button.on{background:var(--on);color:var(--onfg);border-color:var(--on)}
button.x{color:var(--warn)}
.err{color:var(--warn);font-size:13px}
table{border-collapse:collapse;font-size:13px}td{padding:1px 6px}
</style></head><body>
<header>
 <b>people-bind labels</b><span id="prog"></span>
 <label><input type="checkbox" id="todo"> only unfinished</label>
 <button id="dl">Download labels.json</button>
 <button id="imp">Load labels.json</button><input type="file" id="impf" accept=".json,application/json" hidden>
 <button id="reset" class="x">Discard browser edits</button>
 <span class="mut">Edits are kept in this browser; download when done (or any time).</span>
 <span id="srv" class="mut"></span>
</header><main id="root"></main>
<script>
const ITEMS = __ITEMS__;
const FILE_LABELS = __LABELS__;
const KEY = "pb-labels-__KEY__";
let L;
try { L = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (e) { L = null; }
if (!L) L = JSON.parse(JSON.stringify(FILE_LABELS));
function save() { try { localStorage.setItem(KEY, JSON.stringify(L)); } catch (e) {} progress(); pushSoon(); }
// Served by `people_bind_eval.py serve`: the server's saved-labels.json is the source of truth, so
// another browser or machine picks up where this one stopped. Opened as a file, this is all a no-op.
const SERVED = location.protocol.startsWith("http");
let serverReady = false, pushTimer = null;
function srv(msg) { document.getElementById("srv").textContent = msg; }
function pushSoon() {
  if (!SERVED || !serverReady) return;
  clearTimeout(pushTimer);
  pushTimer = setTimeout(() => {
    fetch("saved-labels.json", {method: "PUT", headers: {"Content-Type": "application/json"},
                                body: JSON.stringify(exportLabels())})
      .then(r => { if (!r.ok) throw new Error("HTTP " + r.status);
                   srv("saved to server " + new Date().toLocaleTimeString()); })
      .catch(e => srv("NOT saved to server (" + e.message + "); edits are still in this browser"));
  }, 400);
}
function lab(it) {
  if (!L[it.pid]) L[it.pid] = FILE_LABELS[it.pid] ? JSON.parse(JSON.stringify(FILE_LABELS[it.pid]))
                              : {split: it.split, stratum: it.stratum, mentions: [], bad_labels: []};
  return L[it.pid];
}
function complete(it) { const l = lab(it); return l.mentions.every(m => m.gold !== null) && l.reviewed === true; }
function progress() {
  let tot = 0, done = 0, ph = 0;
  for (const it of ITEMS) { const l = lab(it); tot += l.mentions.length;
    done += l.mentions.filter(m => m.gold !== null).length; if (complete(it)) ph++; }
  document.getElementById("prog").textContent =
    `photos ${ph}/${ITEMS.length} done · phrases ${done}/${tot} labelled`;
}
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "on") e.addEventListener("click", v); else if (k === "cls") e.className = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
}
function render() {
  const root = document.getElementById("root"); root.innerHTML = "";
  const onlyTodo = document.getElementById("todo").checked;
  for (const it of ITEMS) {
    if (onlyTodo && complete(it)) continue;
    root.append(card(it));
  }
}
function card(it) {
  const l = lab(it);
  const sec = el("section", {id: "p" + it.pid, cls: complete(it) ? "done" : ""});
  sec.append(el("h3", {}, `${it.pid} · ${it.stratum}${it.why ? " (" + it.why + ")" : ""} · ${it.split}`));
  const img = el("img", {src: it.pid + ".jpg", loading: "lazy", alt: "photo " + it.pid});
  img.addEventListener("click", () => window.open(it.pid + ".jpg"));
  const right = el("div");
  const desc = el("div", {cls: "desc"}, ...highlight(it.description, l.mentions.map(m => m.phrase)));
  right.append(desc);
  const tbl = el("table");
  for (const b of it.boxes) {
    const cb = el("input", {type: "checkbox", title: "this box's NAME is wrong"});
    cb.checked = l.bad_labels.includes(b.face_id);
    cb.addEventListener("change", () => {
      l.bad_labels = l.bad_labels.filter(f => f !== b.face_id);
      if (cb.checked) l.bad_labels.push(b.face_id);
      save();
    });
    tbl.append(el("tr", {}, el("td", {}, el("b", {}, "box " + b.n)),
      el("td", {}, b.name ? `${b.name} (${b.source})` : "(no name)"),
      el("td", {}, b.name ? el("label", {cls: "mut"}, cb, " name wrong") : "")));
  }
  right.append(el("p", {cls: "mut"}, "Boxes (the model sees these numbers, not the names):"), tbl);
  if (it.unboxed.length) right.append(el("p", {cls: "mut"}, "Named but too small to box: " +
      it.unboxed.map(u => `${u.name} (${u.source})`).join(", ")));
  const ms = el("div");
  l.mentions.forEach((m, i) => ms.append(mention(it, l, m, i)));
  right.append(ms);
  const err = el("div", {cls: "err"});
  right.append(el("div", {},
    el("button", {on: () => {
      const s = (window.getSelection() || "").toString();
      addPhrase(it, l, s, err);
    }}, "+ add selected text as a phrase"),
    el("button", {on: () => { const s = prompt("Exact phrase from the description:"); if (s) addPhrase(it, l, s, err); }}, "+ type a phrase"),
    err));
  const rv = el("input", {type: "checkbox"}); rv.checked = l.reviewed === true;
  rv.addEventListener("change", () => { l.reviewed = rv.checked; save(); sec.className = complete(it) ? "done" : ""; });
  right.append(el("p", {}, el("label", {}, rv, " every person phrase is listed and labelled")));
  sec.append(el("div", {cls: "grid"}, img, right));
  return sec;
}
// Bold every occurrence of every listed phrase (overlaps merged) so the context is easy to find.
function highlight(text, phrases) {
  const iv = [];
  for (const p of phrases) {
    if (!p) continue;
    for (let i = text.indexOf(p); i !== -1; i = text.indexOf(p, i + 1)) iv.push([i, i + p.length]);
  }
  iv.sort((a, b) => a[0] - b[0]);
  const merged = [];
  for (const [s, e] of iv) {
    if (merged.length && s <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], e);
    else merged.push([s, e]);
  }
  const out = []; let at = 0;
  for (const [s, e] of merged) { if (s > at) out.push(text.slice(at, s)); out.push(el("b", {}, text.slice(s, e))); at = e; }
  if (at < text.length) out.push(text.slice(at));
  return out;
}
function addPhrase(it, l, s, err) {
  s = (s || "").trim();
  if (!s) { err.textContent = "select part of the description first"; return; }
  if (!it.description.includes(s)) { err.textContent = "not an exact substring of the description: " + s; return; }
  if (l.mentions.some(m => m.phrase === s)) { err.textContent = "already listed"; return; }
  err.textContent = "";
  l.mentions.push({phrase: s, gold: null}); save(); rerender(it);
}
function rerender(it) { const old = document.getElementById("p" + it.pid); old.replaceWith(card(it)); }
function mention(it, l, m, i) {
  const row = el("div", {cls: "m"}, el("span", {cls: "ph"}, "“" + m.phrase + "”"), " ");
  const ids = Array.isArray(m.gold) ? m.gold : [];
  for (const b of it.boxes) {
    row.append(el("button", {cls: ids.includes(b.face_id) ? "on" : "", on: () => {
      let g = Array.isArray(m.gold) ? m.gold.slice() : [];
      g = g.includes(b.face_id) ? g.filter(f => f !== b.face_id) : g.concat([b.face_id]);
      m.gold = g.length ? g.sort((a, c) => a - c) : null; save(); rerender(it);
    }}, String(b.n)));
  }
  for (const w of ["NONE", "AMBIGUOUS"]) {
    row.append(el("button", {cls: m.gold === w ? "on" : "", on: () => {
      m.gold = m.gold === w ? null : w; save(); rerender(it);
    }}, w));
  }
  row.append(el("button", {cls: "x", title: "not a person phrase", on: () => {
    l.mentions.splice(i, 1); save(); rerender(it);
  }}, "remove"));
  if (m.gold === null) row.append(el("span", {cls: "err"}, " ← not labelled"));
  return row;
}
document.getElementById("todo").addEventListener("change", render);
function exportLabels() {
  const out = {};
  for (const it of ITEMS) { const l = lab(it);
    out[it.pid] = {split: l.split, stratum: l.stratum, mentions: l.mentions,
                   bad_labels: l.bad_labels, reviewed: l.reviewed === true}; }
  return out;
}
document.getElementById("dl").addEventListener("click", () => {
  const out = exportLabels();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "labels.json"; a.click();
});
// Load a downloaded labels.json (e.g. to carry progress to another browser or machine).
document.getElementById("imp").addEventListener("click", () => document.getElementById("impf").click());
document.getElementById("impf").addEventListener("change", ev => {
  const f = ev.target.files[0]; ev.target.value = "";
  if (!f) return;
  f.text().then(t => {
    let data;
    try { data = JSON.parse(t); } catch (e) { alert("Not valid JSON: " + e.message); return; }
    const pids = ITEMS.map(it => String(it.pid)).filter(p => data[p] && Array.isArray(data[p].mentions));
    if (!pids.length) { alert("No photos from this sample in that file."); return; }
    if (!confirm(`Replace this browser's labels for ${pids.length} photos with the file's?`)) return;
    for (const p of pids) L[p] = JSON.parse(JSON.stringify(data[p]));
    save(); render();
  });
});
document.getElementById("reset").addEventListener("click", () => {
  if (!confirm("Throw away this browser's edits and reload labels.json as generated?")) return;
  try { localStorage.removeItem(KEY); } catch (e) {}
  L = JSON.parse(JSON.stringify(FILE_LABELS)); render(); progress();
});
render(); progress();
if (SERVED) {
  srv("loading from server…");
  fetch("saved-labels.json", {cache: "no-store"}).then(r => {
    if (r.status === 404) return null;
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }).then(data => {
    if (data) for (const it of ITEMS) { const p = String(it.pid);
      if (data[p] && Array.isArray(data[p].mentions)) L[p] = data[p]; }
    serverReady = true; save(); render();
  }).catch(e => srv("could not load from server (" + e.message + "); NOT saving to it — reload to retry"));
}
</script></body></html>
"""

if __name__ == "__main__":
    main()

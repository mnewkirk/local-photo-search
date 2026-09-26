#!/usr/bin/env python
"""Sharpness eval -- step 3 of docs/plans/sharpness-measurement.md.

Scores every candidate feature `photosearch.sharpness.measure_photo` stores
against the OWNER's three-state labels for `blurry` and `sharp`, beside the two
things the library has today (the stored VLM tag, and `aes_sharpness`), and
prints the ship-gate verdict. Nothing here writes to the DB.

    python evals/sharpness_eval.py measure --db photo_index.db.local \\
        --base-url http://localhost:8001
    python evals/sharpness_eval.py report  --db photo_index.db.local
    python evals/sharpness_eval.py all     --db photo_index.db.local   # both

LABELS. Same file shape as the visual-tag eval (`photosearch/visual_tag_eval`):
`{"labels": {"<id>": {"yes": [...], "debatable": [...], "done": bool}}}` with a
`sample.json` (`{"photos": [{"photo_id", "stratum"}]}`) beside it. `--labels`
is repeatable (the plan keeps the original 60 in evals/visual-tags/ and a
blurry-weighted set in its own directory); by default every `labels.json` in
evals/visual-tags/ and evals/sharpness/ that exists is read. Only `done`
photos count; a debatable tag is neither a hit nor a miss; a tag in neither
list is a NO.

That last rule is dangerous for photos labelled BEFORE sharp/blurry were on
the page -- their silence is not a "no". `--labelled-after ISO` drops labels
whose `updated_at` is older; a label carrying an `offered` list is only scored
for the tags in it. The report prints how many labelled photos mention either
tag at all so a silent-everywhere label set is obvious.

MEASUREMENTS are cached in `--cache` (default evals/sharpness/measurements.json),
keyed by SHARPNESS_VERSION then photo id; pixels come from `/full` through the
shared originals cache of `photosearch.model_eval` (fetched once per photo).

THE GATE (plan step 3): `blurry` ships as a derived tag only if some metric and
threshold (blurry = metric <= threshold) reaches P >= 0.8 at R >= 0.5 overall
AND P >= 0.6 in both the night/high-ISO and the bokeh strata (a stratum group
with fewer than 3 predictions passes only if it has no false positive, and a
group with no labelled photos at all fails -- it cannot be checked). `sharp`
(metric >= threshold) is kept only if some threshold reaches P >= 0.8 while
firing on <= 40% of the labelled photos; otherwise the plan retires it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Callable, Iterable, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from photosearch import sharpness  # noqa: E402

MIN_N = 3
DEFAULT_SERVER = "http://localhost:8001"
DEFAULT_LABEL_DIRS = (os.environ.get("PHOTOSEARCH_VISUAL_EVAL_DIR", "./evals/visual-tags"),
                      "./evals/sharpness")
DEFAULT_CACHE = "./evals/sharpness/measurements.json"

BLURRY_GATE = {"precision": 0.8, "recall": 0.5, "stratum_precision": 0.6}
SHARP_GATE = {"precision": 0.8, "max_firing": 0.40}
DEFAULT_NIGHT = ("night", "iso", "low-light", "lowlight", "dark")
DEFAULT_BOKEH = ("bokeh",)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

def _read_json(path, default):
    if not path or not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_label_sets(label_paths: Iterable[str], sample_paths: Iterable[str] = (),
                    labelled_after: Optional[str] = None, log=print):
    """Merge label files (+ their sibling sample.json). Returns
    (labels {pid: entry}, strata {pid: stratum}). Later files win; a
    conflicting duplicate is reported."""
    labels, strata = {}, {}
    samples = list(sample_paths)
    for lp in label_paths:
        raw = _read_json(lp, {"labels": {}}).get("labels", {})
        sib = os.path.join(os.path.dirname(os.path.abspath(lp)), "sample.json")
        if os.path.exists(sib) and sib not in samples:
            samples.append(sib)
        for k, v in raw.items():
            if not v.get("done"):
                continue
            if labelled_after and (v.get("updated_at") or "") < labelled_after:
                continue
            pid = int(k)
            if pid in labels and _measured_view(labels[pid]) != _measured_view(v):
                log(f"  note: photo {pid} labelled differently in {lp}; using it")
            labels[pid] = v
    for sp in samples:
        for p in _read_json(sp, {"photos": []}).get("photos", []):
            strata.setdefault(int(p["photo_id"]), str(p.get("stratum") or "?"))
    return labels, strata


def _measured_view(entry):
    return tuple(sorted(t for t in entry.get("yes", []) if t in ("sharp", "blurry"))), \
        tuple(sorted(t for t in entry.get("debatable", []) if t in ("sharp", "blurry")))


def truth_of(entry: dict, tag: str) -> Optional[bool]:
    """True / False, or None for debatable or not offered (neither hit nor miss)."""
    offered = entry.get("offered")
    if offered is not None and tag not in offered:
        return None
    if tag in entry.get("debatable", ()):
        return None
    return tag in entry.get("yes", ())


# --------------------------------------------------------------------------
# Scoring (pure)
# --------------------------------------------------------------------------

def counts(pred: dict, truth: dict, ids: Optional[Iterable[int]] = None) -> dict:
    """tp/fp/fn/tn over `ids` (default: every id with a truth). A None truth
    (debatable) is skipped; a missing/None prediction is a negative."""
    c = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for pid in (truth if ids is None else ids):
        t = truth.get(pid)
        if t is None:
            continue
        p = bool(pred.get(pid))
        c[("t" if p == t else "f") + ("p" if p else "n")] += 1
    return c


def precision(c):
    d = c["tp"] + c["fp"]
    return c["tp"] / d if d else None


def recall(c):
    d = c["tp"] + c["fn"]
    return c["tp"] / d if d else None


def fmt_ratio(num, den, digits=2):
    if den < MIN_N:
        return "n<3"
    return f"{num / den:.{digits}f}"


def fmt_p(c):
    return fmt_ratio(c["tp"], c["tp"] + c["fp"])


def fmt_r(c):
    return fmt_ratio(c["tp"], c["tp"] + c["fn"])


def predict(values: dict, threshold: float, direction: str) -> dict:
    """blurry: value <= threshold ("below"); sharp: value >= threshold."""
    if direction == "below":
        return {pid: (v is not None and v <= threshold) for pid, v in values.items()}
    return {pid: (v is not None and v >= threshold) for pid, v in values.items()}


def thresholds(values: dict) -> list[float]:
    return sorted({float(v) for v in values.values() if v is not None})


def group_ids(strata: dict, needles: Iterable[str]) -> list[int]:
    needles = [n.lower() for n in needles if n]
    return [pid for pid, s in strata.items() if any(n in (s or "").lower() for n in needles)]


def stratum_ok(c: dict, min_p: float) -> tuple[bool, str]:
    npred = c["tp"] + c["fp"]
    if npred < MIN_N:
        return c["fp"] == 0, f"n<3 ({c['tp']}/{npred})"
    p = c["tp"] / npred
    return p >= min_p, f"{p:.2f}"


def evaluate_blurry(values: dict, truth: dict, groups: dict,
                    gate: dict = BLURRY_GATE) -> dict:
    """Sweep every threshold. Returns the best gate-passing threshold (max
    recall, then precision), the best-F1 threshold, and the sweep."""
    sweep, best_gate, best_f1 = [], None, None
    for t in thresholds(values):
        pred = predict(values, t, "below")
        c = counts(pred, truth)
        p, r = precision(c), recall(c)
        g_res = {}
        for name, ids in groups.items():
            if not any(truth.get(i) is not None for i in ids):
                g_res[name] = (False, "no labels")
            else:
                g_res[name] = stratum_ok(counts(pred, truth, ids), gate["stratum_precision"])
        overall = p is not None and r is not None and p >= gate["precision"] and r >= gate["recall"]
        passes = overall and all(ok for ok, _ in g_res.values())
        row = {"threshold": t, "counts": c, "precision": p, "recall": r,
               "overall_ok": overall, "strata": g_res, "passes": passes}
        sweep.append(row)
        f1 = (2 * p * r / (p + r)) if p and r else 0.0
        row["f1"] = f1
        if passes and (best_gate is None or (r, p) > (best_gate["recall"], best_gate["precision"])):
            best_gate = row
        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = row
    return {"best_gate": best_gate, "best_f1": best_f1, "sweep": sweep,
            "pass": best_gate is not None}


def evaluate_sharp(values: dict, truth: dict, gate: dict = SHARP_GATE,
                   firing_ids: Optional[Iterable[int]] = None) -> dict:
    """Keep `sharp` only if a threshold gives P >= gate precision with a firing
    rate (share of `firing_ids`, default every measured labelled photo,
    predicted sharp) <= gate max_firing."""
    ids = list(firing_ids) if firing_ids is not None else \
        [pid for pid in truth if values.get(pid) is not None]
    sweep, best_keep, best_f1 = [], None, None
    for t in thresholds(values):
        pred = predict(values, t, "above")
        c = counts(pred, truth)
        p, r = precision(c), recall(c)
        firing = (sum(1 for i in ids if pred.get(i)) / len(ids)) if ids else None
        keep = (p is not None and p >= gate["precision"] and firing is not None
                and firing <= gate["max_firing"] and (c["tp"] + c["fp"]) >= MIN_N)
        f1 = (2 * p * r / (p + r)) if p and r else 0.0
        row = {"threshold": t, "counts": c, "precision": p, "recall": r,
               "firing": firing, "keep": keep, "f1": f1}
        sweep.append(row)
        if keep and (best_keep is None or (r, p) > (best_keep["recall"], best_keep["precision"])):
            best_keep = row
        if best_f1 is None or f1 > best_f1["f1"]:
            best_f1 = row
    return {"best_keep": best_keep, "best_f1": best_f1, "sweep": sweep,
            "keep": best_keep is not None}


def evaluate_baseline_blurry(pred: dict, truth: dict, groups: dict,
                             gate: dict = BLURRY_GATE) -> dict:
    c = counts(pred, truth)
    p, r = precision(c), recall(c)
    g_res = {name: (stratum_ok(counts(pred, truth, ids), gate["stratum_precision"])
                    if any(truth.get(i) is not None for i in ids) else (False, "no labels"))
             for name, ids in groups.items()}
    overall = p is not None and r is not None and p >= gate["precision"] and r >= gate["recall"]
    return {"counts": c, "precision": p, "recall": r, "strata": g_res,
            "passes": overall and all(ok for ok, _ in g_res.values())}


# --------------------------------------------------------------------------
# DB (read-only) + measurement
# --------------------------------------------------------------------------

def db_rows(conn, ids: list[int]) -> dict:
    out = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        for r in conn.execute(
                f"SELECT id, camera_model, aes_sharpness, visual_tags, subject_boxes, iso "
                f"FROM photos WHERE id IN ({ph})", chunk):
            out[r["id"]] = dict(r)
        for r in conn.execute(
                f"SELECT photo_id, bbox_left, bbox_top, bbox_right, bbox_bottom "
                f"FROM faces WHERE photo_id IN ({ph})", chunk):
            out.setdefault(r["photo_id"], {}).setdefault("faces", []).append(dict(r))
    return out


def _json_list(raw):
    try:
        v = json.loads(raw) if raw else []
        return v if isinstance(v, list) else []
    except (TypeError, ValueError):
        return []


def load_cache(path: str) -> dict:
    data = _read_json(path, {})
    return data.get(f"v{sharpness.SHARPNESS_VERSION}", {})


def save_cache(path: str, photos: dict) -> None:
    from photosearch.model_eval import write_json_atomic
    from pathlib import Path
    data = _read_json(path, {})
    data[f"v{sharpness.SHARPNESS_VERSION}"] = photos
    write_json_atomic(Path(path), data)


def measure_all(ids, rows, cache_path, server, *, fetch_path: Optional[Callable] = None,
                measure: Callable = sharpness.measure_photo, force=False, log=print):
    """Measure every id not yet cached for this SHARPNESS_VERSION."""
    if fetch_path is None:
        from photosearch.model_eval import original_path
        fetch_path = lambda pid: str(original_path(pid, server))  # noqa: E731
    cache = load_cache(cache_path)
    todo = [pid for pid in ids if force or str(pid) not in cache]
    log(f"{len(ids)} labelled photos; {len(todo)} to measure (v{sharpness.SHARPNESS_VERSION})")
    for n, pid in enumerate(todo, 1):
        row = rows.get(pid, {})
        try:
            path = fetch_path(pid)
        except Exception as exc:          # transport error: not cached, retried next run
            log(f"  ! {pid}: fetch failed: {exc}")
            continue
        t = time.perf_counter()
        res = measure(path, row.get("faces") or [], _json_list(row.get("subject_boxes")))
        res["seconds"] = round(time.perf_counter() - t, 4)
        cache[str(pid)] = res
        if n % 10 == 0 or n == len(todo):
            save_cache(cache_path, cache)
            log(f"  measured {n}/{len(todo)}")
    if todo:
        save_cache(cache_path, cache)
    return cache


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


def _thr(x):
    return "-" if x is None else (f"{x:.4g}")


def build(labels, strata, rows, cache, night=DEFAULT_NIGHT, bokeh=DEFAULT_BOKEH):
    """All numbers the report prints, as one dict (also --json-out)."""
    ids = [pid for pid in labels if str(pid) in cache]
    feats = {pid: sharpness.flatten_features(cache[str(pid)]) for pid in ids}
    metrics = sorted({m for f in feats.values() for m in f})
    truth = {"blurry": {pid: truth_of(labels[pid], "blurry") for pid in ids},
             "sharp": {pid: truth_of(labels[pid], "sharp") for pid in ids}}
    st = {pid: strata.get(pid, "?") for pid in ids}
    groups = {"night/high-ISO": group_ids(st, night), "bokeh": group_ids(st, bokeh)}
    out = {"n_labelled": len(labels), "n_measured": len(ids),
           "errors": sum(1 for pid in ids if "error" in (cache[str(pid)].get("detail") or {})),
           "mentions": sum(1 for pid in ids if any(
               t in labels[pid].get(k, ()) for t in ("sharp", "blurry")
               for k in ("yes", "debatable"))),
           "positives": {t: sum(1 for v in truth[t].values() if v) for t in truth},
           "negatives": {t: sum(1 for v in truth[t].values() if v is False) for t in truth},
           "groups": {k: len(v) for k, v in groups.items()},
           "seconds": sorted(cache[str(pid)].get("seconds") or 0 for pid in ids),
           "blurry": {}, "sharp": {}}
    for m in metrics:
        vals = {pid: feats[pid].get(m) for pid in ids}
        out["blurry"][m] = evaluate_blurry(vals, truth["blurry"], groups)
        out["sharp"][m] = evaluate_sharp(vals, truth["sharp"])
    stored = {pid: _json_list(rows.get(pid, {}).get("visual_tags")) for pid in ids}
    aes = {pid: rows.get(pid, {}).get("aes_sharpness") for pid in ids}
    out["baselines"] = {
        "blurry": {
            "stored blurry": evaluate_baseline_blurry(
                {p: "blurry" in stored[p] for p in ids}, truth["blurry"], groups),
            "aes_sharpness<=2": evaluate_baseline_blurry(
                {p: aes[p] is not None and aes[p] <= 2 for p in ids}, truth["blurry"], groups)},
        "sharp": {
            "stored sharp": _sharp_baseline({p: "sharp" in stored[p] for p in ids}, truth["sharp"]),
            "aes_sharpness>=9": _sharp_baseline(
                {p: aes[p] is not None and aes[p] >= 9 for p in ids}, truth["sharp"])},
    }
    out["_ids"], out["_feats"], out["_truth"], out["_strata"] = ids, feats, truth, st
    out["_camera"] = {pid: rows.get(pid, {}).get("camera_model") or "?" for pid in ids}
    out["_stored"], out["_aes"] = stored, aes
    return out


def _sharp_baseline(pred, truth):
    c = counts(pred, truth)
    n = sum(1 for pid in truth if pid in pred)
    fired = sum(1 for v in pred.values() if v)
    return {"counts": c, "precision": precision(c), "recall": recall(c),
            "firing": fired / n if n else None}


def _breakdown(pred, truth, keys, title):
    lines = [f"    {title:<22} {'n':>4} {'pos':>4} {'P':>6} {'R':>6}"]
    by = defaultdict(list)
    for pid, k in keys.items():
        by[k].append(pid)
    for k in sorted(by):
        c = counts(pred, truth, by[k])
        n = sum(1 for p in by[k] if truth.get(p) is not None)
        lines.append(f"    {str(k)[:22]:<22} {n:>4} {c['tp'] + c['fn']:>4} "
                     f"{fmt_p(c):>6} {fmt_r(c):>6}")
    return lines


def _sweep_lines(sweep, rows=12, firing=False):
    """An evenly sampled slice of a threshold sweep (every row if short)."""
    if not sweep:
        return []
    step = max(1, len(sweep) // rows)
    pick = sweep[::step]
    if pick[-1] is not sweep[-1]:
        pick.append(sweep[-1])
    out = [f"      {'thr':>10} {'P':>6} {'R':>6}" + (f" {'fires':>6}" if firing else "")]
    for r in pick:
        out.append(f"      {_thr(r['threshold']):>10} {fmt_p(r['counts']):>6} "
                   f"{fmt_r(r['counts']):>6}" + (f" {_f(r.get('firing')):>6}" if firing else ""))
    return out


def render(rep, top=6) -> str:
    L = []
    secs = [s for s in rep["seconds"] if s]
    L.append(f"labelled photos: {rep['n_labelled']}   measured: {rep['n_measured']}   "
             f"decode errors: {rep['errors']}")
    L.append(f"photos whose label mentions sharp/blurry at all: {rep['mentions']}"
             + ("   <-- 0: were these labelled before sharp/blurry were offered? "
                "see --labelled-after" if rep["n_measured"] and not rep["mentions"] else ""))
    L.append(f"blurry: {rep['positives']['blurry']} yes / {rep['negatives']['blurry']} no   "
             f"sharp: {rep['positives']['sharp']} yes / {rep['negatives']['sharp']} no   "
             f"strata groups: " + ", ".join(f"{k}={v}" for k, v in rep["groups"].items()))
    if secs:
        L.append(f"measure time/photo: median {secs[len(secs) // 2]:.2f}s  "
                 f"p90 {secs[int(0.9 * (len(secs) - 1))]:.2f}s  (this machine)")
    L.append("")

    # ---- blurry ----
    L.append("=== BLURRY  (predicted blurry = metric <= threshold) ===")
    L.append(f"gate: P>={BLURRY_GATE['precision']} at R>={BLURRY_GATE['recall']} overall, "
             f"P>={BLURRY_GATE['stratum_precision']} in night/high-ISO and bokeh")
    hdr = f"  {'metric':<26} {'gate thr':>10} {'P':>6} {'R':>6} {'night':>12} {'bokeh':>12} | {'F1 thr':>10} {'P':>6} {'R':>6}"
    L.append(hdr)
    ranked = sorted(rep["blurry"].items(), key=lambda kv: (
        -(kv[1]["best_gate"]["recall"] if kv[1]["best_gate"] else -1),
        -(kv[1]["best_f1"]["f1"] if kv[1]["best_f1"] else 0)))
    for m, ev in ranked:
        g, f = ev["best_gate"], ev["best_f1"]
        gs = (f"{_thr(g['threshold']):>10} {fmt_p(g['counts']):>6} {fmt_r(g['counts']):>6} "
              f"{g['strata'].get('night/high-ISO', (0, '-'))[1]:>12} "
              f"{g['strata'].get('bokeh', (0, '-'))[1]:>12}") if g else \
            f"{'-':>10} {'':>6} {'':>6} {'':>12} {'':>12}"
        fs = (f"{_thr(f['threshold']):>10} {fmt_p(f['counts']):>6} {fmt_r(f['counts']):>6}"
              if f else "")
        L.append(f"  {m:<26} {gs} | {fs}")
    L.append("  baselines:")
    for name, b in rep["baselines"]["blurry"].items():
        L.append(f"  {name:<26} {'':>10} {fmt_p(b['counts']):>6} {fmt_r(b['counts']):>6} "
                 f"{b['strata']['night/high-ISO'][1]:>12} {b['strata']['bokeh'][1]:>12}"
                 f"  {'PASS' if b['passes'] else 'fail'}")
    passing = [m for m, ev in ranked if ev["pass"]]
    L.append("")
    if passing:
        m = passing[0]
        g = rep["blurry"][m]["best_gate"]
        L.append(f"BLURRY GATE: PASS -- {m} <= {_thr(g['threshold'])} "
                 f"(P {fmt_p(g['counts'])}, R {fmt_r(g['counts'])}); "
                 f"{len(passing)} metric(s) pass")
    else:
        L.append("BLURRY GATE: FAIL -- no metric/threshold meets it; ship the number only, "
                 "keep `blurry` frozen")
    L.append("")

    # ---- per-stratum / per-camera for the top metrics + baselines ----
    truth, ids = rep["_truth"]["blurry"], rep["_ids"]
    show = [m for m, _ in ranked[:top]]
    for m in show:
        ev = rep["blurry"][m]
        row = ev["best_gate"] or ev["best_f1"]
        if not row:
            continue
        vals = {pid: rep["_feats"][pid].get(m) for pid in ids}
        pred = predict(vals, row["threshold"], "below")
        L.append(f"  {m} <= {_thr(row['threshold'])} "
                 f"({'gate' if ev['best_gate'] else 'best F1'}):")
        L.append("    threshold sweep:")
        L += _sweep_lines(ev["sweep"])
        L += _breakdown(pred, truth, rep["_strata"], "stratum")
        L += _breakdown(pred, truth, rep["_camera"], "camera")
    for name, pred in (("stored blurry", {p: "blurry" in rep["_stored"][p] for p in ids}),
                       ("aes_sharpness<=2", {p: rep["_aes"][p] is not None and rep["_aes"][p] <= 2
                                             for p in ids})):
        L.append(f"  baseline {name}:")
        L += _breakdown(pred, truth, rep["_strata"], "stratum")
        L += _breakdown(pred, truth, rep["_camera"], "camera")
    L.append("")

    # ---- sharp ----
    L.append("=== SHARP  (predicted sharp = metric >= threshold) ===")
    L.append(f"keep only if P>={SHARP_GATE['precision']} with firing <= "
             f"{int(SHARP_GATE['max_firing'] * 100)}% of labelled photos (sample is "
             f"blurry-weighted, so library firing will be HIGHER)")
    L.append(f"  {'metric':<26} {'keep thr':>10} {'P':>6} {'R':>6} {'fires':>6} | {'F1 thr':>10} {'P':>6} {'R':>6} {'fires':>6}")
    sranked = sorted(rep["sharp"].items(), key=lambda kv: (
        -(kv[1]["best_keep"]["recall"] if kv[1]["best_keep"] else -1),
        -(kv[1]["best_f1"]["f1"] if kv[1]["best_f1"] else 0)))
    for m, ev in sranked:
        k, f = ev["best_keep"], ev["best_f1"]
        ks = (f"{_thr(k['threshold']):>10} {fmt_p(k['counts']):>6} {fmt_r(k['counts']):>6} "
              f"{_f(k['firing']):>6}") if k else f"{'-':>10} {'':>6} {'':>6} {'':>6}"
        fs = (f"{_thr(f['threshold']):>10} {fmt_p(f['counts']):>6} {fmt_r(f['counts']):>6} "
              f"{_f(f['firing']):>6}") if f else ""
        L.append(f"  {m:<26} {ks} | {fs}")
    L.append("  baselines:")
    for name, b in rep["baselines"]["sharp"].items():
        L.append(f"  {name:<26} {'':>10} {fmt_p(b['counts']):>6} {fmt_r(b['counts']):>6} "
                 f"{_f(b['firing']):>6}")
    for m, ev in sranked[:max(1, top // 2)]:
        L.append(f"  {m} threshold sweep:")
        L += _sweep_lines(ev["sweep"], firing=True)
    keepers = [m for m, ev in sranked if ev["keep"]]
    L.append("")
    if keepers:
        k = rep["sharp"][keepers[0]]["best_keep"]
        L.append(f"SHARP: KEEP -- {keepers[0]} >= {_thr(k['threshold'])} "
                 f"(P {fmt_p(k['counts'])}, R {fmt_r(k['counts'])}, fires {_f(k['firing'])})")
    else:
        L.append("SHARP: RETIRE -- no threshold reaches P>=0.8 while firing on <=40%")
    return "\n".join(L)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _label_paths(args):
    if args.labels:
        return args.labels
    return [p for p in (os.path.join(d, "labels.json") for d in DEFAULT_LABEL_DIRS)
            if os.path.exists(p)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("measure", "report", "all"), nargs="?", default="all")
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    ap.add_argument("--base-url", default=DEFAULT_SERVER)
    ap.add_argument("--labels", action="append", default=[],
                    help="labels.json (repeatable; default: evals/visual-tags + evals/sharpness)")
    ap.add_argument("--sample", action="append", default=[],
                    help="extra sample.json for strata (siblings of --labels are read anyway)")
    ap.add_argument("--labelled-after", default=None,
                    help="ignore labels whose updated_at is older (ISO timestamp)")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--force", action="store_true", help="re-measure cached photos")
    ap.add_argument("--night-strata", default=",".join(DEFAULT_NIGHT),
                    help="substrings naming the night/high-ISO strata")
    ap.add_argument("--bokeh-strata", default=",".join(DEFAULT_BOKEH))
    ap.add_argument("--top", type=int, default=6,
                    help="metrics given per-stratum/per-camera breakdowns")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    lpaths = _label_paths(a)
    if not lpaths:
        raise SystemExit("No labels found -- pass --labels PATH")
    labels, strata = load_label_sets(lpaths, a.sample, a.labelled_after)
    print(f"labels: {', '.join(lpaths)} -> {len(labels)} done photos")
    from photosearch.model_eval import open_db_readonly
    conn = open_db_readonly(a.db)
    ids = sorted(labels)
    rows = db_rows(conn, ids)

    if a.cmd in ("measure", "all"):
        measure_all(ids, rows, a.cache, a.base_url, force=a.force)
    if a.cmd in ("report", "all"):
        cache = load_cache(a.cache)
        rep = build(labels, strata, rows, cache,
                    night=[s for s in a.night_strata.split(",") if s],
                    bokeh=[s for s in a.bokeh_strata.split(",") if s])
        print(render(rep, top=a.top))
        if a.json_out:
            slim = {k: v for k, v in rep.items() if not k.startswith("_")}
            for tag in ("blurry", "sharp"):
                for ev in slim[tag].values():
                    ev.pop("sweep", None)
            with open(a.json_out, "w") as f:
                json.dump(slim, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())

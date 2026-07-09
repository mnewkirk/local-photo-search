#!/usr/bin/env python
"""Aesthetics scoring bakeoff — pick the model(s) for the new VLM aesthetics pass.

Phase 0 of the VLM-aesthetics plan. Scores a curated sample of YOUR OWN photos
with several candidate scorers and reports which one is the most *discriminative*
and best-correlated with your own ranking — the anti-compression signal that the
LAION CLIP predictor and a naive VLM both fail.

Two families of candidate, run side by side:

  * VLMs  — via the existing photosearch routing (Ollama or, preferably, an
    OpenAI-compatible LM Studio endpoint). Each candidate is just a model id
    passed to photosearch.aesthetics.score_photo_aesthetics; the rubric-anchored
    prompt + JSON parse are reused verbatim, so the bakeoff measures the real
    pass. Point at LM Studio and DO NOT set PHOTOSEARCH_LLM_AESTHETICS_MODEL —
    the per-call model id wins:
        export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
        python evals/aesthetics_bakeoff.py --photos-dir /path/to/sample \\
            --vlm qwen2.5-vl-7b-instruct --vlm qwen3.5-9b

  * IQA   — purpose-built No-Reference metrics via `pyiqa` (optional; pip install
    pyiqa). Fast, objective, and (for MUSIQ/TOPIQ) run at native resolution — a
    strong Technical-Excellence anchor. VisualQuality-R1 and any other pyiqa
    metric name work too:
        python evals/aesthetics_bakeoff.py --photos-dir /path/to/sample \\
            --iqa musiq --iqa topiq_nr

Ground truth (optional but recommended): a CSV of `filename,rank` (rank 1 = best)
or `filename,score` for ~30-50 hand-ranked photos. The report then includes each
scorer's Spearman rank-correlation with your judgment — the number that actually
decides the winner.

Outputs (under --out, default ./aesthetics-bakeoff):
  scores.json   — {scorer: {filename: score}}  (resumable cache; re-run adds models)
  report.html   — per-scorer ranked galleries + a metrics table
  console       — score spread (discrimination), pairwise agreement, GT Spearman

Nothing here touches photo_index.db.
"""
import argparse
import csv
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Sample gathering
# --------------------------------------------------------------------------

def gather_photos(photos_dir=None, list_file=None):
    """Return [(name, abspath)] for the sample. name is used as the join key
    with ground truth and the scores cache."""
    paths = []
    if list_file:
        with open(list_file) as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#"):
                    paths.append(p)
    elif photos_dir:
        for root, _dirs, files in os.walk(photos_dir):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                    paths.append(os.path.join(root, fn))
    else:
        raise SystemExit("Provide --photos-dir or --list-file")
    out = []
    seen = set()
    for p in paths:
        name = os.path.basename(p)
        if name in seen:  # keep names unique so the cache key is stable
            name = os.path.relpath(p, photos_dir or "/").replace("/", "_")
        seen.add(name)
        out.append((name, p))
    return out


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

def score_vlm(model, photos, cache, verbose=True):
    """Score every photo with one VLM model. Returns {name: overall}. Resumable:
    photos already in `cache` are skipped."""
    from photosearch.aesthetics import score_photo_aesthetics
    out = dict(cache)
    for i, (name, path) in enumerate(photos, 1):
        if name in out and out[name] is not None:
            continue
        try:
            res = score_photo_aesthetics(path, model=model)
            out[name] = res["overall"] if res else None
        except Exception as e:
            if verbose:
                print(f"    ! {name}: {e}")
            out[name] = None
        if verbose and i % 10 == 0:
            print(f"    [{i}/{len(photos)}] {model}")
    return out


def score_iqa(metric_name, photos, cache, device=None, verbose=True):
    """Score with a pyiqa No-Reference metric. Returns {name: score}."""
    import pyiqa
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"  loading pyiqa metric {metric_name} on {dev} ...")
    metric = pyiqa.create_metric(metric_name, device=dev)
    out = dict(cache)
    for i, (name, path) in enumerate(photos, 1):
        if name in out and out[name] is not None:
            continue
        try:
            out[name] = float(metric(path).item())
        except Exception as e:
            if verbose:
                print(f"    ! {name}: {e}")
            out[name] = None
        if verbose and i % 10 == 0:
            print(f"    [{i}/{len(photos)}] {metric_name}")
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _ranks(values):
    """Average-rank of each value (ascending). Ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def spearman(x, y):
    """Spearman rank correlation over paired, non-None values."""
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return float("nan")
    xa, ya = zip(*pairs)
    return _pearson(_ranks(list(xa)), _ranks(list(ya)))


def stats_for(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
    n = len(vals)
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    return {"n": n, "min": min(vals), "max": max(vals),
            "mean": mean, "std": std}


def load_ground_truth(path):
    """CSV `filename,rank` (rank 1=best) or `filename,score`. Returns
    {name: numeric} where HIGHER = better (ranks are negated)."""
    gt = {}
    is_rank = None
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2 or not row[0] or row[0].lower() in ("filename", "name"):
                if len(row) >= 2 and row[1].lower() in ("rank", "score"):
                    is_rank = row[1].lower() == "rank"
                continue
            name = os.path.basename(row[0].strip())
            try:
                val = float(row[1])
            except ValueError:
                continue
            gt[name] = val
    # If header didn't declare, guess: small positive ints starting near 1 → ranks
    if is_rank is None:
        vals = list(gt.values())
        is_rank = vals and all(float(v).is_integer() for v in vals) and max(vals) <= len(vals) + 1
    if is_rank:
        gt = {k: -v for k, v in gt.items()}  # negate so higher=better
    return gt


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(out_dir, photos, scores, gt, top_n=60):
    path_by_name = {name: p for name, p in photos}
    scorers = list(scores.keys())

    # Metrics table
    rows = []
    for s in scorers:
        st = stats_for(list(scores[s].values()))
        gt_rho = float("nan")
        if gt:
            names = [n for n in gt if n in scores[s]]
            gt_rho = spearman([scores[s].get(n) for n in names],
                              [gt[n] for n in names])
        rows.append((s, st, gt_rho))

    def fmt(v, d=3):
        return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"

    metric_html = ["<table class=metrics><thead><tr><th>scorer</th><th>n</th>"
                   "<th>min</th><th>max</th><th>mean</th>"
                   "<th>std (spread)</th><th>Spearman vs you</th></tr></thead><tbody>"]
    for s, st, rho in rows:
        metric_html.append(
            f"<tr><td>{html.escape(s)}</td><td>{st['n']}</td>"
            f"<td>{fmt(st['min'],2)}</td><td>{fmt(st['max'],2)}</td>"
            f"<td>{fmt(st['mean'],2)}</td><td class=hi>{fmt(st['std'])}</td>"
            f"<td class=hi>{fmt(rho)}</td></tr>")
    metric_html.append("</tbody></table>")

    # Pairwise agreement (Spearman between scorers)
    pair_html = ""
    if len(scorers) > 1:
        pair_html = ["<h2>Pairwise agreement (Spearman)</h2><table class=metrics>"
                     "<thead><tr><th></th>" + "".join(f"<th>{html.escape(s)}</th>"
                     for s in scorers) + "</tr></thead><tbody>"]
        common = [n for n, _ in photos]
        for a in scorers:
            cells = [f"<th>{html.escape(a)}</th>"]
            for b in scorers:
                rho = spearman([scores[a].get(n) for n in common],
                               [scores[b].get(n) for n in common])
                cells.append(f"<td>{fmt(rho,2)}</td>")
            pair_html.append("<tr>" + "".join(cells) + "</tr>")
        pair_html.append("</tbody></table>")
        pair_html = "".join(pair_html)

    # Per-scorer ranked galleries
    galleries = []
    for s in scorers:
        ranked = sorted(((n, v) for n, v in scores[s].items() if v is not None),
                        key=lambda kv: kv[1], reverse=True)[:top_n]
        cards = []
        for rank, (name, val) in enumerate(ranked, 1):
            p = path_by_name.get(name, "")
            gtxt = ""
            if gt and name in gt:
                gtxt = f"<span class=gt>you:{-gt[name]:.0f}</span>" if gt[name] < 0 \
                    else f"<span class=gt>you:{gt[name]:.1f}</span>"
            cards.append(
                f'<div class=card><div class=rank>#{rank}</div>'
                f'<img loading=lazy src="file://{html.escape(p)}">'
                f'<div class=score>{val:.2f} {gtxt}</div>'
                f'<div class=name>{html.escape(name)}</div></div>')
        galleries.append(
            f"<h2>{html.escape(s)} — top {len(ranked)}</h2>"
            f"<div class=grid>{''.join(cards)}</div>")

    doc = f"""<!DOCTYPE html><html><head><meta charset=utf-8>
<title>Aesthetics bakeoff</title><style>
 body{{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
 table.metrics{{border-collapse:collapse;margin:12px 0}}
 table.metrics th,table.metrics td{{border:1px solid #333;padding:5px 10px;text-align:right}}
 table.metrics td:first-child,table.metrics th:first-child{{text-align:left}}
 td.hi{{color:#7fd;font-weight:700}}
 .grid{{display:flex;flex-wrap:wrap;gap:10px}}
 .card{{width:150px;background:#1c1c1c;border-radius:6px;padding:6px;position:relative}}
 .card img{{width:100%;height:110px;object-fit:cover;border-radius:4px;background:#222}}
 .rank{{position:absolute;top:8px;left:8px;background:#000a;padding:1px 6px;border-radius:8px;font-size:11px;color:#aaa}}
 .score{{font-weight:700;color:#7fd;margin-top:4px;font-size:14px}}
 .gt{{color:#ffd27f;font-weight:400;font-size:11px;margin-left:4px}}
 .name{{font-size:10px;color:#888;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
</style></head><body>
<h1>Aesthetics scoring bakeoff — {len(photos)} photos, {len(scorers)} scorer(s)</h1>
<p style="color:#999">Higher <b>std</b> = more discriminative (fights the 6.2-ceiling
compression). <b>Spearman vs you</b> = agreement with your hand-ranking (the tiebreaker).</p>
{''.join(metric_html)}
{pair_html}
{''.join(galleries)}
</body></html>"""
    out_html = os.path.join(out_dir, "report.html")
    with open(out_html, "w") as f:
        f.write(doc)
    return out_html, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos-dir", help="Directory of sample photos (recursed).")
    ap.add_argument("--list-file", help="Newline-delimited file of photo paths.")
    ap.add_argument("--vlm", action="append", default=[],
                    help="VLM model id (repeatable). Routed via PHOTOSEARCH_TEXT_LLM_URL.")
    ap.add_argument("--iqa", action="append", default=[],
                    help="pyiqa metric name, e.g. musiq / topiq_nr (repeatable).")
    ap.add_argument("--ground-truth", help="CSV filename,rank|score for correlation.")
    ap.add_argument("--out", default=os.path.join(HERE, "aesthetics-bakeoff"))
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--device", default=None, help="torch device for pyiqa.")
    args = ap.parse_args()

    if not args.vlm and not args.iqa:
        raise SystemExit("Provide at least one --vlm or --iqa scorer.")

    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "scores.json")
    scores = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            scores = json.load(f)
        print(f"[cache] loaded {len(scores)} scorer(s) from {cache_path}")

    photos = gather_photos(args.photos_dir, args.list_file)
    print(f"[sample] {len(photos)} photos")

    for model in args.vlm:
        print(f"[vlm] {model}")
        scores[model] = score_vlm(model, photos, scores.get(model, {}))
        with open(cache_path, "w") as f:
            json.dump(scores, f)

    for metric in args.iqa:
        print(f"[iqa] {metric}")
        try:
            scores[metric] = score_iqa(metric, photos, scores.get(metric, {}),
                                       device=args.device)
        except ImportError:
            print("  ! pyiqa not installed — `pip install pyiqa`; skipping.")
            continue
        with open(cache_path, "w") as f:
            json.dump(scores, f)

    gt = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    if args.ground_truth:
        print(f"[gt] {len(gt)} ranked photos loaded")

    out_html, rows = write_report(args.out, photos, scores, gt, top_n=args.top_n)

    print("\n=== Summary (higher std = more discriminative) ===")
    print(f"{'scorer':<28} {'n':>4} {'mean':>7} {'std':>7} {'ρ vs you':>9}")
    for s, st, rho in rows:
        mean = f"{st['mean']:.2f}" if st['mean'] is not None else "—"
        std = f"{st['std']:.3f}" if st['std'] is not None else "—"
        r = f"{rho:.3f}" if rho == rho else "—"
        print(f"{s:<28} {st['n']:>4} {mean:>7} {std:>7} {r:>9}")
    print(f"\n[report] {out_html}")


if __name__ == "__main__":
    main()

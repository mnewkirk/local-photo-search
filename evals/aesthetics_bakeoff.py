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
    pass. On LM Studio the per-call model id is IGNORED — the `aesthetics`
    role env var picks the model, and falls back to
    PHOTOSEARCH_LLM_VISUAL_MODEL when unset — so `--vlm M` PINS
    PHOTOSEARCH_LLM_AESTHETICS_MODEL=M for this process and records the model
    that actually ran. One --vlm per process (the pin is process-global):
        export PHOTOSEARCH_TEXT_LLM_URL=http://localhost:1234/v1
        python evals/aesthetics_bakeoff.py --photos-dir evals/aesthetics-bakeoff/sample \\
            --vlm qwen2.5-vl-7b-instruct
    (This docstring used to say "DO NOT set …AESTHETICS_MODEL — the per-call
    id wins". That was TRUE when the 2026-07-09 run was made — scores.json was
    written 07-09 09:00, and the VISUAL fallback arrived in 4cfc202 on 07-10 —
    so that run's qwen2.5-vl-7b-instruct entry really is qwen. It stopped
    being true afterwards, hence the pin.)

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
  scores-v2.json — VLM runs, one per --variant (default: the model id), with
                   the effective model, what LM Studio had loaded, and per
                   photo {overall, first_parse_ok, calls, latency_s}.
                   Resumable; a transport error is never cached.
  scores.json   — {scorer: {filename: score}}: IQA metrics, plus the legacy
                  pre-v2 VLM entries (read-only, effective model unknown)
  report.html   — per-scorer ranked galleries + a metrics table
  console       — score spread (discrimination), pairwise agreement, GT Spearman

The console summary adds what decides a model swap besides ρ: parse-failure
rate (first attempt and final), spread (std, IQR, distinct values, share
within ±0.5 of the median — the "squashed scores" failure), s/photo (labelled
solo/shared by what LM Studio had loaded), and ρ with a 95% CI. On 28 photos a
ρ gap under ~0.1 is a tie.

`--selections-gt --db <replica>` adds a second, owner-free ground truth: within
each /review cluster, does the scorer rank the photos the owner KEPT above the
ones they didn't? Culling picks mix sharpness and moment with aesthetics, so
read it as a sanity check, not the verdict.

photo_index.db is only ever opened read-only (mode=ro).
"""
import argparse
import csv
import html
import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from urllib.parse import quote

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

def _production_scorer(path, model):
    from photosearch.aesthetics import score_photo_aesthetics
    return score_photo_aesthetics(path, model=model)


def score_vlm(model, photos, entry, *, scorer=None, save=None, log=print):
    """Score every photo not yet in `entry["items"]` with the PRODUCTION
    `score_photo_aesthetics`, through a Recorder so a dead backend is not
    cached as a parse failure. Mutates and returns `entry`; `save()` is called
    after every photo (resumable)."""
    from photosearch import model_eval as me
    from photosearch.aesthetics import parse_aesthetics_response
    scorer = scorer or _production_scorer
    items = entry.setdefault("items", {})
    todo = [(n, p) for n, p in photos if n not in items]
    fc = me.FailureCounter(log)
    for i, (name, path) in enumerate(todo, 1):
        rec = me.Recorder()
        t0 = time.time()
        try:
            with rec.active():
                res = scorer(path, model)
            rec.check(res is None, "score_photo_aesthetics")
        except Exception as e:
            fc.fail(name, e)
            continue
        fc.ok()
        first_raw = rec.calls[0].get("raw") if rec.calls else None
        items[name] = {
            "overall": res["overall"] if res else None,
            "first_parse_ok": parse_aesthetics_response(first_raw or "") is not None,
            "calls": len(rec.calls),
            "latency_s": round(time.time() - t0, 3),
        }
        if save:
            save()
        if i % 10 == 0 or i == len(todo):
            log(f"    [{i}/{len(todo)}] {entry.get('effective_model')}")
    fc.summary()
    return entry


def _load_capped_tensor(path, max_edge):
    """Load an image as a (1,3,H,W) float tensor in [0,1], downscaled so its
    longer edge is <= max_edge.

    This cap is the whole reason the IQA path doesn't nuke the machine. MUSIQ
    (and other native-resolution metrics) tile the image into patch_size=32
    patches and run self-attention over ALL of them (`pyiqa` sets MUSIQ's
    `max_seq_len_from_original_res=-1`). A 56MP camera JPEG → ~55k patches →
    a ~55k x 55k attention matrix (tens of GB), which OOM-kills the whole WSL
    VM. Downscaling to a 1536px long edge caps that at ~1.5k patches, and
    quality metrics are robust to moderate downscaling. Importing photosearch
    registers the HEIF/HEIC opener so .heic samples load like every other format.
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    import photosearch  # noqa: F401 — registers the HEIF/HEIC opener with PIL

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max_edge and max(img.size) > max_edge:
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3 in [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W


def score_iqa(metric_name, photos, cache, device=None, max_edge=1536, verbose=True):
    """Score with a pyiqa No-Reference metric. Returns {name: score}.

    Images are downscaled to `max_edge` (longer side, px) before scoring — see
    `_load_capped_tensor` for why that cap is load-bearing. Pass max_edge=0 to
    feed native resolution (will OOM on large photos with MUSIQ)."""
    import pyiqa
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        cap = f"<= {max_edge}px long edge" if max_edge else "native resolution"
        print(f"  loading pyiqa metric {metric_name} on {dev} ({cap}) ...")
    metric = pyiqa.create_metric(metric_name, device=dev)
    out = dict(cache)
    for i, (name, path) in enumerate(photos, 1):
        if name in out and out[name] is not None:
            continue
        try:
            inp = _load_capped_tensor(path, max_edge) if max_edge else path
            with torch.inference_mode():
                out[name] = float(metric(inp).item())
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


def spread_for(values):
    """How discriminative a scorer is. std alone hides the failure this pass
    exists to fix (a VLM that answers 7 for everything), so also: IQR, how many
    distinct values it used, and the share within ±0.5 of its own median."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"std": None, "iqr": None, "distinct": 0, "near_median": None}
    from photosearch.model_eval import median, percentile
    med = median(vals)
    return {"std": stats_for(vals)["std"],
            "iqr": percentile(vals, 0.75) - percentile(vals, 0.25),
            "distinct": len(set(round(v, 2) for v in vals)),
            "near_median": sum(1 for v in vals if abs(v - med) <= 0.5) / len(vals)}


def spearman_ci(rho, n, z=1.96):
    """95% CI for Spearman's ρ via Fisher z with the Bonett-Wright standard
    error, sqrt((1 + ρ²/2) / (n - 3)). On 28 photos it is wide — which is the
    point of printing it."""
    if n is None or n < 4 or rho != rho or abs(rho) >= 1:
        return (float("nan"), float("nan"))
    se = math.sqrt((1 + rho * rho / 2) / (n - 3))
    zr = math.atanh(rho)
    return (math.tanh(zr - z * se), math.tanh(zr + z * se))


def vlm_summary(entry, gt):
    """The per-VLM row for the console table."""
    from photosearch import model_eval as me
    items = {k: v for k, v in entry.get("items", {}).items() if not k.startswith("pid:")}
    n = len(items)
    overall = {k: v["overall"] for k, v in items.items()}
    first_fail = sum(1 for v in items.values() if not v.get("first_parse_ok"))
    final_fail = sum(1 for v in overall.values() if v is None)
    lat = me.latencies(items)
    rho = n_gt = None
    if gt:
        names = [k for k in gt if overall.get(k) is not None]
        n_gt = len(names)
        rho = spearman([overall[k] for k in names], [gt[k] for k in names])
    return {"n": n, "first_fail": first_fail, "final_fail": final_fail,
            "spread": spread_for(overall.values()),
            "lat_median": me.median(lat), "lat_p90": me.percentile(lat, 0.9),
            "latency_label": entry.get("latency_label", "unknown"),
            "rho": rho, "n_gt": n_gt,
            "rho_ci": spearman_ci(rho, n_gt) if rho is not None else None}


# --------------------------------------------------------------------------
# Second ground truth: the owner's /review culling picks
# --------------------------------------------------------------------------

def open_db_readonly(path):
    if not path:
        raise SystemExit("--selections-gt needs --db (or PHOTOSEARCH_DB).")
    uri = "file:" + quote(os.path.abspath(path)) + "?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.execute("SELECT 1 FROM review_selections LIMIT 1")
    except sqlite3.OperationalError as e:
        raise SystemExit(f"Cannot open {path} read-only: {e}")
    return conn


def selection_groups(conn, n_groups=150, seed=7):
    """Seeded sample of /review clusters holding at least one KEPT and one
    not-kept photo: [(kept_ids, other_ids)]."""
    groups = {}
    for directory, cluster, pid, selected in conn.execute(
            "SELECT directory, cluster_id, photo_id, selected FROM review_selections "
            "WHERE cluster_id IS NOT NULL ORDER BY directory, cluster_id, photo_id"):
        g = groups.setdefault((directory, cluster), ([], []))
        g[0 if selected else 1].append(int(pid))
    usable = [g for _, g in sorted(groups.items()) if g[0] and g[1]]
    rnd = random.Random(seed)
    return usable if len(usable) <= n_groups else rnd.sample(usable, n_groups)


def selection_agreement(groups, score_by_id):
    """(agreeing pairs, scored pairs) — a kept photo scored above a not-kept
    one from the same cluster agrees; a tie counts half."""
    agree = total = 0.0
    for kept, other in groups:
        for k in kept:
            for o in other:
                a, b = score_by_id.get(k), score_by_id.get(o)
                if a is None or b is None:
                    continue
                total += 1
                agree += 1 if a > b else 0.5 if a == b else 0
    return agree, total


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

def _thumb_data_uri(path, box=(300, 220)):
    """Return a downscaled base64 JPEG data URI for `path`.

    The report embeds thumbnails inline instead of linking `file://<abspath>`.
    A WSL absolute path (`file:///home/mattn/...`) resolves to `C:\\home\\...`
    when the HTML is opened in a Windows browser over the \\\\wsl.localhost
    mount, so linked images never load. Embedding makes the report
    self-contained and path-independent. `draft` decodes big JPEGs at reduced
    scale so even a 56MP source stays cheap."""
    import base64
    import io
    from PIL import Image, ImageOps
    try:
        import photosearch  # noqa: F401 — registers the HEIF/HEIC opener
    except Exception:
        pass
    try:
        im = Image.open(path)
        try:
            im.draft("RGB", box)  # fast reduced-scale decode for JPEG sources
        except Exception:
            pass
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail(box, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=78)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def write_report(out_dir, photos, scores, gt, top_n=60):
    path_by_name = {name: p for name, p in photos}
    scorers = list(scores.keys())

    # Build each thumbnail once (a name can appear in several scorer galleries).
    thumb_cache = {}

    def thumb(name):
        if name not in thumb_cache:
            p = path_by_name.get(name, "")
            thumb_cache[name] = _thumb_data_uri(p) if p else ""
        return thumb_cache[name]

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
            gtxt = ""
            if gt and name in gt:
                gtxt = f"<span class=gt>you:{-gt[name]:.0f}</span>" if gt[name] < 0 \
                    else f"<span class=gt>you:{gt[name]:.1f}</span>"
            src = thumb(name)
            img = f'<img loading=lazy src="{src}">' if src \
                else '<div class=noimg>no image</div>'
            cards.append(
                f'<div class=card><div class=rank>#{rank}</div>'
                f'{img}'
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
 .noimg{{width:100%;height:110px;border-radius:4px;background:#222;display:flex;align-items:center;justify-content:center;color:#666;font-size:11px}}
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


def _slug(model):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-") or "vlm"


def _fmt(v, d=3):
    return "—" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def run_vlm(model, photos, v2_path, *, variant=None, force=False, solo=False,
            selection_photos=(), scorer=None, log=print):
    """Pin, score (sample + any selection photos), and persist one VLM run."""
    from photosearch import describe, model_eval as me
    me.pin_role_model("aesthetics", model)
    effective = describe.effective_model(model, "aesthetics")
    variant = variant or _slug(model)
    me.check_variant(variant)
    store = me.read_json(v2_path, {})
    entry = None if force else store.get(variant)
    if entry is not None and entry.get("effective_model") != effective:
        raise SystemExit(f"variant {variant!r} was scored by "
                         f"{entry.get('effective_model')!r}, now {effective!r}. "
                         "Use a new --variant, or --force.")
    if entry is None:
        entry = {"model": model, "effective_model": effective,
                 "created": me.now_iso(), "items": {}}
    store[variant] = entry
    loaded_start = me.lmstudio_loaded()
    log(f"[vlm] {variant}: model={effective}  loaded={loaded_start}")

    def save():
        me.write_json_atomic(v2_path, store)

    score_vlm(model, list(photos) + list(selection_photos), entry,
              scorer=scorer, save=save, log=log)
    loaded_end = me.lmstudio_loaded()
    entry["loaded_models_start"], entry["loaded_models_end"] = loaded_start, loaded_end
    entry["latency_label"] = me.latency_label(effective, loaded_start, loaded_end, solo)
    save()
    return variant, entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos-dir", help="Directory of sample photos (recursed).")
    ap.add_argument("--list-file", help="Newline-delimited file of photo paths.")
    ap.add_argument("--vlm", action="append", default=[],
                    help="VLM model id. Pinned via PHOTOSEARCH_LLM_AESTHETICS_MODEL; "
                         "one per process on the LM Studio route.")
    ap.add_argument("--variant", help="Name for this VLM run (default: the model id).")
    ap.add_argument("--force", action="store_true",
                    help="Discard the cached VLM run for this variant.")
    ap.add_argument("--solo", action="store_true",
                    help="Assert only this model was loaded, when LM Studio "
                         "can't report it.")
    ap.add_argument("--iqa", action="append", default=[],
                    help="pyiqa metric name, e.g. musiq / topiq_nr (repeatable).")
    default_gt = os.path.join(HERE, "aesthetics-bakeoff", "ranked.csv")
    ap.add_argument("--ground-truth",
                    default=default_gt if os.path.exists(default_gt) else None,
                    help="CSV filename,rank|score (default: the 2026-07-09 "
                         "ranked.csv when present).")
    ap.add_argument("--selections-gt", action="store_true",
                    help="Also score /review culling clusters (needs --db).")
    ap.add_argument("--selections-n", type=int, default=150)
    ap.add_argument("--db", default=os.environ.get("PHOTOSEARCH_DB"))
    ap.add_argument("--server", default="http://localhost:8001",
                    help="Server the selection photos are fetched from (once).")
    ap.add_argument("--out", default=os.path.join(HERE, "aesthetics-bakeoff"))
    ap.add_argument("--top-n", type=int, default=60)
    ap.add_argument("--device", default=None, help="torch device for pyiqa.")
    ap.add_argument("--max-edge", type=int, default=1536,
                    help="Downscale images to this longer-edge px before IQA "
                         "scoring (0 = native resolution — will OOM MUSIQ on "
                         "large photos). Default 1536.")
    ap.add_argument("--report-only", action="store_true",
                    help="Score nothing; report what is cached.")
    args = ap.parse_args()

    if not (args.vlm or args.iqa or args.report_only):
        raise SystemExit("Provide --vlm, --iqa, or --report-only.")
    if len(args.vlm) > 1 and os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        raise SystemExit("One --vlm per process: the model is pinned through a "
                         "process-global env var. Run the command once per model.")
    if args.variant and len(args.vlm) != 1:
        raise SystemExit("--variant names exactly one --vlm run.")

    from photosearch import model_eval as me
    os.makedirs(args.out, exist_ok=True)
    cache_path = os.path.join(args.out, "scores.json")
    v2_path = os.path.join(args.out, "scores-v2.json")
    legacy = me.read_json(cache_path, {})

    photos = gather_photos(args.photos_dir, args.list_file)
    print(f"[sample] {len(photos)} photos")

    groups, sel_photos = [], []
    if args.selections_gt:
        groups = selection_groups(open_db_readonly(args.db), args.selections_n)
        ids = sorted({i for k, o in groups for i in k + o})
        print(f"[selections] {len(groups)} clusters, {len(ids)} photos")
        for pid in ids:
            try:
                sel_photos.append((f"pid:{pid}", str(me.original_path(pid, args.server))))
            except Exception as e:
                print(f"  ! {pid}: {e} (skipped)")

    for model in args.vlm:
        run_vlm(model, photos, v2_path, variant=args.variant, force=args.force,
                solo=args.solo, selection_photos=sel_photos)

    for metric in args.iqa:
        print(f"[iqa] {metric}")
        try:
            legacy[metric] = score_iqa(metric, photos, legacy.get(metric, {}),
                                       device=args.device, max_edge=args.max_edge)
        except ImportError:
            print("  ! pyiqa not installed — `pip install pyiqa`; skipping.")
            continue
        me.write_json_atomic(cache_path, legacy)

    gt = load_ground_truth(args.ground_truth) if args.ground_truth else {}
    if args.ground_truth:
        print(f"[gt] {len(gt)} ranked photos loaded from {args.ground_truth}")

    v2 = me.read_json(v2_path, {})
    iqa_names = set(args.iqa)
    scores = {}
    for name, vals in legacy.items():
        scores[name if name in iqa_names else f"legacy:{name}"] = vals
    for variant, entry in v2.items():
        scores[variant] = {k: v["overall"] for k, v in entry["items"].items()
                           if not k.startswith("pid:")}

    out_html, rows = write_report(args.out, photos, scores, gt, top_n=args.top_n)

    print("\n=== Legacy / IQA scorers (scores.json) ===")
    print(f"{'scorer':<34} {'n':>4} {'mean':>7} {'std':>7} {'ρ vs you':>9}")
    for s, st, rho in rows:
        if s in v2:
            continue
        print(f"{s:<34} {st['n']:>4} {_fmt(st['mean'], 2):>7} "
              f"{_fmt(st['std']):>7} {_fmt(rho):>9}")
    if any(r[0].startswith("legacy:") for r in rows):
        print("  legacy: = cached in scores.json before v2 (no effective model "
              "recorded). The 2026-07-09 VLM entry predates the VISUAL fallback "
              "(4cfc202), so it is the model it is named after.")

    if v2:
        print("\n=== VLM runs (scores-v2.json) ===")
        print(f"{'variant':<26} {'effective model':<28} {'n':>3} {'parse✗1st':>9} "
              f"{'parse✗fin':>9} {'std':>6} {'IQR':>6} {'dist':>5} {'±.5med':>6} "
              f"{'s/ph p50':>8} {'p90':>6} {'ρ [95% CI]':>22}  latency")
        for variant, entry in v2.items():
            r = vlm_summary(entry, gt)
            sp = r["spread"]
            ci = r["rho_ci"]
            rho = (f"{_fmt(r['rho'])} [{_fmt(ci[0], 2)},{_fmt(ci[1], 2)}]"
                   if r["rho"] is not None else "—")
            print(f"{variant:<26} {str(entry.get('effective_model')):<28} {r['n']:>3} "
                  f"{me.fmt_ratio(r['first_fail'], r['n']):>9} "
                  f"{me.fmt_ratio(r['final_fail'], r['n']):>9} "
                  f"{_fmt(sp['std'], 2):>6} {_fmt(sp['iqr'], 2):>6} {sp['distinct']:>5} "
                  f"{_fmt(sp['near_median'], 2):>6} {_fmt(r['lat_median'], 1):>8} "
                  f"{_fmt(r['lat_p90'], 1):>6} {rho:>22}  {r['latency_label']}")
            if groups:
                by_id = {int(k[4:]): v["overall"] for k, v in entry["items"].items()
                         if k.startswith("pid:")}
                agree, total = selection_agreement(groups, by_id)
                print(f"{'':<26} culling agreement: {me.fmt_ratio(agree, total)} "
                      f"of {int(total)} kept-vs-other pairs (0.50 = chance)")
        print("On 28 photos a ρ gap under ~0.1 is a tie.")
    print(f"\n[report] {out_html}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agent model bake-off (M24b / M26).

Runs the in-app /api/ask agent loop (photosearch.agent.run_agent) against N
local LM Studio models over a fixed set of NL search prompts, captures every
tool trace + final answer + result photos, and emits:

  - evals/bakeoff_results.json   raw results (resumable, written incrementally)
  - evals/bakeoff_report.html    self-contained table (prompts × models) with
                                 tool traces + embedded thumbnail snapshots

Each (model, prompt) run sets PHOTOSEARCH_LLM_AGENT_MODEL and drives the agent
against the local read-replica DB. Thumbnails for the result photos are pulled
from the NAS web API and base64-embedded so the report is portable.

Env:
  PHOTOSEARCH_DB            replica DB (default ./photo_index.db.local)
  PHOTOSEARCH_TEXT_LLM_URL  LM Studio /v1 (required)
  PHOTOSEARCH_NAS_URL       NAS web for thumbnails (default http://dxp4800-f976:8000)

Usage:
  PHOTOSEARCH_TEXT_LLM_URL=http://172.20.176.1:1234/v1 \
    ./.venv/bin/python evals/bakeoff.py [model1 model2 ...]
"""
from __future__ import annotations

import base64
import html
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photosearch.db import PhotoDB
from photosearch import agent

MODELS_DEFAULT = ["qwen/qwen3.5-9b", "google/gemma-4-e2b", "llama-3.2-3b-instruct"]

REPLICA_DB = os.environ.get("PHOTOSEARCH_DB", "./photo_index.db.local")
NAS_URL = (os.environ.get("PHOTOSEARCH_NAS_URL") or "http://dxp4800-f976:8000").rstrip("/")
OUT_DIR = Path(__file__).resolve().parent
THUMBS_PER_CELL = 6

# (id, prompt, what it probes, what a good answer looks like) — grounded in the
# real library (Calvin/Ellie/Nicole/Matt; France years 2014 & 2025; NY∩France
# = 2025; soccer is a real category; moody is a real tag, "golden hour" is not).
PROMPTS = [
    ("p01", "Show me photos of Calvin.",
     "Baseline tool-call + name grounding",
     "search_photos(people=['Calvin']); ~16.5k results"),
    ("p02", "Find photos of Nicole and Matt together.",
     "AND-intersection of two people",
     "people=['Nicole','Matt'] (AND, not OR / not in query)"),
    ("p03", "What were our best photos from last summer?",
     "Relative-date math + 'best'→quality",
     "date_from~2025-06-01..08-31 AND sort=quality_desc"),
    ("p04", "Show me pictures from our trip to France.",
     "Location grounding",
     "location='France'; ideally list_places first"),
    ("p05", "Find good photos of the kids playing soccer.",
     "3-way decomposition + entity resolution",
     "people=kids(Calvin/Ellie) + category=soccer + quality"),
    ("p06", "I want landscape shots with no people in them.",
     "Negation / exclusion",
     "landscape + exclude people (no faces / 'no people')"),
    ("p07", "Find my most dramatic, moody photos.",
     "Vocab grounding + adaptation",
     "list_vocab → use visual_tag=moody (drop non-existent 'dramatic')"),
    ("p08", "I'm trying to find one specific photo of Ellie at a birthday — help me narrow it down.",
     "Count-driven iteration",
     "people=['Ellie'] + birthday, read total, refine across steps"),
    ("p09", "Show me a photo of the whole family together.",
     "Ambiguous-entity reasoning",
     "infer family=Calvin+Ellie+Nicole+Matt via list_people, AND them"),
    ("p10", "We were in both New York and France in the same year — which year, and show photos from that trip.",
     "Multi-hop reasoning (ceiling)",
     "intersect France years (2014,2025) ∩ NY years → 2025, then search"),
]


def _fetch_thumb_b64(photo_id, cache):
    if photo_id in cache:
        return cache[photo_id]
    try:
        req = urllib.request.Request(f"{NAS_URL}/api/photos/{photo_id}/thumbnail",
                                     headers={"User-Agent": "bakeoff"})
        with urllib.request.urlopen(req, timeout=20) as r:
            cache[photo_id] = base64.b64encode(r.read()).decode("ascii")
    except Exception:
        cache[photo_id] = None
    return cache[photo_id]


def run_one(db, model, prompt_text):
    os.environ["PHOTOSEARCH_LLM_AGENT_MODEL"] = model
    trace, photos, answer, error = [], [], "", None
    tool_calls, single_shot = 0, False
    t0 = time.monotonic()
    try:
        for ev in agent.run_agent(db, prompt_text, max_steps=6):
            t = ev.get("type")
            if t == "tool_call":
                tool_calls += 1
                if ev.get("tool") == "_fallback":
                    single_shot = True
                trace.append({"k": "call", "tool": ev.get("tool"), "args": ev.get("arguments")})
            elif t == "tool_result":
                if ev.get("tool") == "_fallback":
                    single_shot = True
                trace.append({"k": "result", "tool": ev.get("tool"), "summary": ev.get("summary")})
            elif t == "photos":
                photos = ev.get("results", []) or []
            elif t == "answer":
                answer = ev.get("text", "")
            elif t == "error":
                error = ev.get("message")
    except Exception as e:
        error = f"harness exception: {type(e).__name__}: {e}"
    return {
        "model": model,
        "seconds": round(time.monotonic() - t0, 1),
        "tool_calls": tool_calls,
        "single_shot": single_shot,
        "n_results": len(photos),
        "result_ids": [p.get("id") for p in photos[:THUMBS_PER_CELL]],
        "answer": answer,
        "error": error,
        "trace": trace,
    }


def main():
    # Rebuild the HTML from the last results.json without re-running the models.
    if "--html-only" in sys.argv[1:]:
        data = json.loads((OUT_DIR / "bakeoff_results.json").read_text())
        build_html(data["models"], data["results"])
        print(f"Rebuilt {OUT_DIR/'bakeoff_report.html'} from cached results")
        return

    models = sys.argv[1:] or MODELS_DEFAULT
    if not os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        sys.exit("PHOTOSEARCH_TEXT_LLM_URL must be set (LM Studio /v1 endpoint)")

    results = {}  # prompt_id -> {model -> run}
    db = PhotoDB(REPLICA_DB)
    try:
        for pid, ptext, probes, expect in PROMPTS:
            results[pid] = {"prompt": ptext, "probes": probes, "expect": expect, "runs": {}}
            for model in models:
                print(f"[{pid}] {model} … ", end="", flush=True)
                run = run_one(db, model, ptext)
                results[pid]["runs"][model] = run
                tag = ("single-shot" if run["single_shot"]
                       else f"{run['tool_calls']} calls")
                print(f"{run['seconds']}s  {tag}  {run['n_results']} results"
                      + (f"  ERR:{run['error']}" if run["error"] else ""))
                # incremental save
                (OUT_DIR / "bakeoff_results.json").write_text(
                    json.dumps({"models": models, "results": results}, indent=2))
    finally:
        db.close()

    build_html(models, results)
    print(f"\nWrote {OUT_DIR/'bakeoff_report.html'} and bakeoff_results.json")


def build_html(models, results):
    thumb_cache = {}

    def esc(s):
        return html.escape(str(s if s is not None else ""))

    def trace_html(trace):
        rows = []
        for step in trace:
            if step["k"] == "call":
                args = json.dumps(step.get("args") or {}, ensure_ascii=False)
                rows.append(f'<div class="call">▸ {esc(step["tool"])} '
                            f'<span class="args">{esc(args)}</span></div>')
            else:
                rows.append(f'<div class="res">↳ {esc(step.get("summary"))}</div>')
        return "".join(rows) or '<div class="res muted">(no tool calls)</div>'

    def thumbs_html(ids):
        imgs = []
        for pid in ids:
            if pid is None:
                continue
            b = _fetch_thumb_b64(pid, thumb_cache)
            if b:
                imgs.append(f'<img class="thumb" src="data:image/jpeg;base64,{b}" '
                            f'title="id {esc(pid)} — click to enlarge" '
                            f'onclick="showLb(this.src)"/>')
        return ('<div class="thumbs">' + "".join(imgs) + "</div>") if imgs \
               else '<div class="thumbs muted">— no photos —</div>'

    def cell(run):
        badges = []
        badges.append(f'<span class="b">{run["seconds"]}s</span>')
        if run["single_shot"]:
            badges.append('<span class="b warn">single-shot</span>')
        else:
            badges.append(f'<span class="b">{run["tool_calls"]} tool calls</span>')
        badges.append(f'<span class="b">{run["n_results"]} results</span>')
        if run["error"]:
            badges.append(f'<span class="b err">error</span>')
        ans = esc(run["answer"]) or '<span class="muted">(no answer)</span>'
        err = f'<div class="errmsg">{esc(run["error"])}</div>' if run["error"] else ""
        return (f'<div class="badges">{"".join(badges)}</div>'
                f'<div class="answer">{ans}</div>{err}'
                f'{thumbs_html(run["result_ids"])}'
                f'<details><summary>trace</summary><div class="trace">{trace_html(run["trace"])}</div></details>')

    # Summary scoreboard (objective metrics)
    score_rows = []
    for m in models:
        runs = [results[p]["runs"].get(m, {}) for p in results]
        runs = [r for r in runs if r]
        tot = len(runs)
        toolful = sum(1 for r in runs if not r.get("single_shot") and r.get("tool_calls"))
        nonempty = sum(1 for r in runs if r.get("n_results"))
        errs = sum(1 for r in runs if r.get("error"))
        avg_s = round(sum(r.get("seconds", 0) for r in runs) / max(tot, 1), 1)
        score_rows.append(
            f"<tr><td>{esc(m)}</td><td>{toolful}/{tot}</td><td>{nonempty}/{tot}</td>"
            f"<td>{errs}</td><td>{avg_s}s</td></tr>")

    head_cols = "".join(f"<th>{esc(m)}</th>" for m in models)
    body_rows = []
    for pid in results:
        info = results[pid]
        cells = "".join(f'<td class="cell">{cell(info["runs"].get(m, {"seconds":0,"tool_calls":0,"single_shot":False,"n_results":0,"result_ids":[],"answer":"","error":"missing","trace":[]}))}</td>'
                        for m in models)
        body_rows.append(
            f'<tr><td class="prompt"><div class="pid">{esc(pid)}</div>'
            f'<div class="ptext">{esc(info["prompt"])}</div>'
            f'<div class="probe"><b>probes:</b> {esc(info["probes"])}</div>'
            f'<div class="expect"><b>good:</b> {esc(info["expect"])}</div></td>'
            f'{cells}</tr>')

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Agent model bake-off</title><style>
 body {{ font:14px/1.5 system-ui,sans-serif; margin:24px; color:#1a1a1a; background:#fafafa; }}
 h1 {{ font-size:22px; }} .meta {{ color:#666; margin-bottom:16px; }}
 table {{ border-collapse:collapse; width:100%; background:#fff; }}
 th,td {{ border:1px solid #ddd; vertical-align:top; padding:8px; }}
 th {{ background:#f0f0f3; position:sticky; top:0; }}
 td.prompt {{ width:210px; background:#fbfbfd; }}
 .pid {{ font:12px monospace; color:#888; }}
 .ptext {{ font-weight:600; margin:2px 0 6px; }}
 .probe,.expect {{ font-size:12px; color:#555; margin-top:4px; }}
 td.cell {{ width:auto; }}
 .badges {{ margin-bottom:6px; }}
 .b {{ display:inline-block; font-size:11px; background:#eef; border-radius:10px; padding:1px 8px; margin-right:4px; }}
 .b.warn {{ background:#fde9c8; }} .b.err {{ background:#f7c5c5; }}
 .answer {{ margin:6px 0; }} .errmsg {{ color:#a00; font-size:12px; }}
 .thumbs {{ display:flex; flex-wrap:wrap; gap:4px; margin:6px 0; }}
 .thumbs img {{ width:84px; height:84px; object-fit:cover; border-radius:4px; border:1px solid #ccc; cursor:zoom-in; transition:transform .08s; }}
 .thumbs img:hover {{ transform:scale(1.06); border-color:#37c; }}
 #lb {{ position:fixed; inset:0; background:rgba(0,0,0,.88); display:none; align-items:center; justify-content:center; z-index:1000; cursor:zoom-out; }}
 #lb.show {{ display:flex; }}
 #lb img {{ max-width:94vw; max-height:94vh; border-radius:6px; box-shadow:0 0 50px #000; }}
 details {{ margin-top:6px; }} summary {{ cursor:pointer; font-size:12px; color:#37c; }}
 .trace {{ font:11px/1.5 monospace; background:#f6f6f9; padding:6px; border-radius:4px; margin-top:4px; }}
 .call {{ color:#225; }} .args {{ color:#888; }} .res {{ color:#363; }}
 .muted {{ color:#aaa; }}
 .scoreboard {{ width:auto; margin-bottom:20px; }} .scoreboard td,.scoreboard th {{ padding:4px 12px; }}
</style></head><body>
<h1>Agent model bake-off</h1>
<div class="meta">Models: {esc(", ".join(models))} · LM Studio: {esc(os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"))} · DB: {esc(REPLICA_DB)}<br>
Objective metrics only — read the traces + snapshots for quality. "tool-using" = ran real tool calls (not single-shot fallback).</div>
<table class="scoreboard"><tr><th>model</th><th>tool-using</th><th>non-empty</th><th>errors</th><th>avg time</th></tr>{''.join(score_rows)}</table>
<table><tr><th>prompt</th>{head_cols}</tr>{''.join(body_rows)}</table>
<div id="lb" onclick="this.classList.remove('show')"><img id="lbimg" alt="preview"/></div>
<script>
 function showLb(src) {{ var l=document.getElementById('lb'); document.getElementById('lbimg').src=src; l.classList.add('show'); }}
 document.addEventListener('keydown', function(e) {{ if (e.key==='Escape') document.getElementById('lb').classList.remove('show'); }});
</script>
</body></html>"""
    (OUT_DIR / "bakeoff_report.html").write_text(html_doc)


if __name__ == "__main__":
    main()

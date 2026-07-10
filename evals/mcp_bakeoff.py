#!/usr/bin/env python3
"""MCP-path tool-definition bake-off.

`evals/bakeoff.py` measures `agent.run_agent`, which prepends a ~4,700-char
routing system prompt (`agent._system_prompt`) AND pre-injects library facts.
The MCP server (`photosearch/mcp_server.py`) sends neither: an MCP client gets
`tools.mcp_tools()` and nothing else. So every "qwen3.5-9b works" data point we
have is evidence about the *agent prompt*, not about the *tool definitions*.

This harness isolates that variable. It drives the same 10 prompts down two
paths and scores both with the same machine-checkable assertions:

  --paths mcp     an MCP-client-shaped loop: generic host system prompt +
                  the advertised tool list. No routing rules, no library facts.
  --paths agent   the existing agent.run_agent, for reference.

Two knobs stage the fixes so their effect is measured, not assumed:

  --instructions  prepend `tools.server_instructions(...)` — exactly what
                  `mcp_server.build_server()` passes as `Server(instructions=)`.
                  A no-op on a tree where that function doesn't exist, so the
                  same harness scores the before and the after.
  --facts         fold `agent._library_context(db)` into those instructions,
                  as the server does at startup (PHOTOSEARCH_MCP_FACTS).

We do not stand up the streamable-HTTP server for scoring — that adds transport
risk without informational value. `--live` separately handshakes the real /mcp
endpoint to confirm the transport and what it actually advertises.

Outputs (beside bakeoff's); --tag NAME suffixes both:
  evals/mcp_bakeoff_results.json   raw, written incrementally (resumable)
  evals/mcp_bakeoff_report.html    prompts x arms, with per-check pass/fail

Env:
  PHOTOSEARCH_DB            replica DB (default ./photo_index.db.local)
  PHOTOSEARCH_TEXT_LLM_URL  LM Studio /v1 (required)
  PHOTOSEARCH_MCP_URL       for --live (default http://localhost:8848/mcp)
  PHOTOSEARCH_NAS_URL       for --thumbs

Usage:
  PHOTOSEARCH_TEXT_LLM_URL=http://172.20.176.1:1234/v1 \
    ./venv/bin/python evals/mcp_bakeoff.py --paths mcp,agent
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from photosearch.db import PhotoDB
from photosearch import agent
from photosearch import tools as toolmod

from bakeoff import PROMPTS, _fetch_thumb_b64  # same prompts, same thumb cache

OUT_DIR = Path(__file__).resolve().parent
REPLICA_DB = os.environ.get("PHOTOSEARCH_DB", "./photo_index.db.local")
MODELS_DEFAULT = ["qwen/qwen3.5-9b"]
MAX_STEPS = 6
DEADLINE_S = 120.0
THUMBS_PER_CELL = 6

# The four grounding tools. On the agent path the facts are already in the
# system prompt, so any call here is a wasted round-trip. On the MCP path they
# are the *only* way to learn the vocabulary — which is the point being measured.
GROUNDING_TOOLS = {"get_library_overview", "list_people", "list_places", "list_vocab"}

# What a neutral MCP host puts in front of the tools. Deliberately minimal: no
# routing rules, no examples. The date line stays because real hosts (Claude
# Desktop included) inject it — without it p03 would fail for a reason that has
# nothing to do with the tool definitions.
HOST_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools for searching the user's "
    "personal photo library. Use the tools to find what the user asked for, then "
    "answer in one or two sentences.\n"
    "Today's date is {today}."
)


# ---------------------------------------------------------------------------
# Latency decomposition
# ---------------------------------------------------------------------------
#
# A run's wall-clock is LLM time + tool-execution time, and on its own it is NOT
# a clean latency comparison:
#   - a run that fails early finishes faster (the 42% MCP baseline "won" on
#     wall-clock partly by grounding and then giving up),
#   - the deadline censors the slow tail at DEADLINE_S,
#   - CLIP's first load lands on whichever prompt first runs a semantic query.
# Both arms funnel every LLM round-trip through agent._chat, so wrap it once and
# report llm_seconds vs tool_seconds separately.
#
# Prefill is the dominant first-call cost and it is prefix-cached: on this Mac a
# novel 16-tool payload (10,641 prompt tokens) took 26.8s cold and 1.5s warm
# (~2.5 ms/token cold). The tool schemas sit at the FRONT of the rendered prompt,
# so a stable tool list is reused across round-trips and across prompts; only a
# changed tool list or changed instructions pays again.

_LLM = {"seconds": 0.0, "calls": 0, "rounds": []}
_chat_orig = agent._chat


def _timed_chat(*args, **kwargs):
    t0 = time.monotonic()
    reply = None
    try:
        reply = _chat_orig(*args, **kwargs)
        return reply
    finally:
        el = time.monotonic() - t0
        _LLM["seconds"] += el
        _LLM["calls"] += 1
        u = (reply or {}).get("usage") or {}
        _LLM["rounds"].append({
            "seconds": round(el, 2),
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            # thinking models emit reasoning before the answer/tool call
            "reasoning_chars": len((reply or {}).get("reasoning") or ""),
            "content_chars": len((reply or {}).get("content") or ""),
        })


agent._chat = _timed_chat


def _reset_llm():
    _LLM.update(seconds=0.0, calls=0, rounds=[])


def _with_timing(elapsed, run):
    """Split a run's wall-clock into LLM round-trips vs tool execution."""
    llm = _LLM["seconds"]
    run["seconds"] = round(elapsed, 1)
    run["llm_seconds"] = round(llm, 1)
    run["llm_calls"] = _LLM["calls"]
    run["tool_seconds"] = round(max(0.0, elapsed - llm), 1)
    run["llm_rounds"] = list(_LLM["rounds"])
    return run


# ---------------------------------------------------------------------------
# Trace accessors + assertions
# ---------------------------------------------------------------------------

def calls(trace, tool=None):
    """Every tool_call in the trace, optionally filtered to one tool name."""
    return [s for s in trace if s["k"] == "call" and (tool is None or s["tool"] == tool)]


def _people(args):
    """Normalize a `people` arg to a lowercase set. Tolerates the stringified
    JSON array that small models emit (which tools._coerce_str_list accepts)."""
    v = (args or {}).get("people")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = [v]
    if not isinstance(v, list):
        return set()
    return {str(x).strip().lower() for x in v if str(x).strip()}


def _q(args):
    return str((args or {}).get("query") or "").lower()


def _any(trace, tool, pred):
    return any(pred(c["args"] or {}) for c in calls(trace, tool))


def _none(trace, tool, pred):
    """A negative check ("didn't stuff the name into query"). Requires at least
    one call to `tool` — otherwise a model that calls nothing at all would pass
    every negative check for free and score better than one that tried."""
    made = calls(trace, tool)
    return bool(made) and not any(pred(c["args"] or {}) for c in made)


def _last_summer():
    """The year 'last summer' refers to, relative to today (matches bakeoff's
    stated expectation of 2025 when run in mid-2026)."""
    today = dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


def _in_range(datestr, lo, hi):
    if not isinstance(datestr, str) or len(datestr) < 10:
        return False
    return lo <= datestr[:10] <= hi


# Each check: (name, fn(trace) -> bool). A prompt passes a check or it doesn't;
# there is no partial credit. These turn bakeoff.py's prose "what a good answer
# looks like" column into something a script can score.
CHECKS = {
    "p01": [
        ("calls search_photos", lambda t: bool(calls(t, "search_photos"))),
        ("people=[Calvin]", lambda t: _any(t, "search_photos", lambda a: _people(a) == {"calvin"})),
        ("name not stuffed in query",
         lambda t: _none(t, "search_photos", lambda a: "calvin" in _q(a))),
    ],
    "p02": [
        ("people={Nicole,Matt} (AND)",
         lambda t: _any(t, "search_photos", lambda a: _people(a) == {"nicole", "matt"})),
        ("names not in query",
         lambda t: _none(t, "search_photos", lambda a: "nicole" in _q(a) or "matt" in _q(a))),
    ],
    "p03": [
        ("sort=quality_desc",
         lambda t: _any(t, "search_photos", lambda a: a.get("sort") == "quality_desc")),
        ("date range = last summer",
         lambda t: _any(t, "search_photos", lambda a: (
             _in_range(a.get("date_from"), f"{_last_summer()}-05-01", f"{_last_summer()}-07-01")
             and _in_range(a.get("date_to"), f"{_last_summer()}-08-01", f"{_last_summer()}-10-01")))),
        ("'best' not in query", lambda t: _none(t, "search_photos", lambda a: "best" in _q(a))),
        ("no min_quality floor",
         lambda t: _none(t, "search_photos", lambda a: a.get("min_quality") is not None)),
    ],
    "p04": [
        ("location=France",
         lambda t: _any(t, "search_photos", lambda a: "france" in str(a.get("location") or "").lower())),
    ],
    "p05": [
        ("people={Calvin,Ellie}",
         lambda t: _any(t, "search_photos", lambda a: _people(a) == {"calvin", "ellie"})),
        ("soccer as category, not query",
         lambda t: _any(t, "search_photos",
                        lambda a: "soccer" in str(a.get("category") or "").lower())),
    ],
    "p06": [
        ("exclusion syntax in query",
         lambda t: _any(t, "search_photos",
                        lambda a: "-people" in _q(a) or "no people" in _q(a) or "without people" in _q(a))),
        ("landscape in query", lambda t: _any(t, "search_photos", lambda a: "landscape" in _q(a))),
    ],
    "p07": [
        ("visual_tag=moody",
         lambda t: _any(t, "search_photos",
                        lambda a: "moody" in str(a.get("visual_tag") or "").lower())),
        ("drops non-existent 'dramatic' tag",
         lambda t: _none(t, "search_photos",
                         lambda a: "dramatic" in str(a.get("visual_tag") or "").lower())),
    ],
    "p08": [
        ("people includes Ellie",
         lambda t: _any(t, "search_photos", lambda a: "ellie" in _people(a))),
        ("iterates (>=2 searches)", lambda t: len(calls(t, "search_photos")) >= 2),
    ],
    "p09": [
        ("people = all four",
         lambda t: _any(t, "search_photos",
                        lambda a: _people(a) == {"calvin", "ellie", "nicole", "matt"})),
    ],
    "p10": [
        ("summarize(group_by=year) x2",
         lambda t: len([c for c in calls(t, "summarize")
                        if (c["args"] or {}).get("group_by") == "year"]) >= 2),
        ("resolves to 2025",
         lambda t: _any(t, "search_photos",
                        lambda a: str(a.get("date_from") or "").startswith("2025"))),
    ],
}

# The tool a competent plan should reach for. Used for the tool-choice metric,
# scored separately from the argument checks above.
EXPECTED_TOOL = {
    "p01": "search_photos", "p02": "search_photos", "p03": "search_photos",
    "p04": "search_photos", "p05": "search_photos", "p06": "search_photos",
    "p07": "search_photos", "p08": "search_photos", "p09": "search_photos",
    "p10": "summarize",
}


def score(pid, trace):
    checks = [(name, bool(fn(trace))) for name, fn in CHECKS.get(pid, [])]
    grounding = len([c for c in calls(trace) if c["tool"] in GROUNDING_TOOLS])
    return {
        "checks": checks,
        "checks_passed": sum(1 for _, ok in checks if ok),
        "checks_total": len(checks),
        "tool_choice_ok": bool(calls(trace, EXPECTED_TOOL[pid])),
        "grounding_calls": grounding,
    }


# ---------------------------------------------------------------------------
# Arm 1: the MCP-client-shaped loop
# ---------------------------------------------------------------------------

def _as_openai(mcp_specs):
    """Project the MCP tool list into the OpenAI `tools=[]` wire shape. Goes
    through mcp_tools() rather than openai_tools() on purpose: we want to score
    exactly the `inputSchema` an MCP client is handed, not a parallel projection."""
    return [{"type": "function",
             "function": {"name": s["name"], "description": s["description"],
                          "parameters": s["inputSchema"]}}
            for s in mcp_specs]


def _mcp_instructions(db, include_writes, use_facts):
    """Exactly what `mcp_server.build_server()` passes as `Server(instructions=)`.
    Returns None on a tree where that function doesn't exist yet (the Phase-1
    baseline), so the same harness measures before and after."""
    fn = getattr(toolmod, "server_instructions", None)
    if fn is None:
        return None
    facts = agent._library_context(db) if use_facts else ""
    return fn(include_writes=include_writes, library_facts=facts)


def run_mcp_path(db, model, prompt_text, *, include_writes, use_instructions,
                 use_facts):
    """Drive the tools the way a real MCP host would: the host's own system
    prompt, the server's `instructions`, the advertised tool list, and no
    recovery machinery (no nudges, no single-shot fallback — a host has none)."""
    os.environ["PHOTOSEARCH_LLM_AGENT_MODEL"] = model
    _reset_llm()
    specs = toolmod.mcp_tools(include_images=False, include_writes=include_writes)
    tools = _as_openai(specs)

    system = HOST_SYSTEM_PROMPT.format(today=dt.date.today().isoformat())
    instr = _mcp_instructions(db, include_writes, use_facts) if use_instructions else None
    if instr:
        system += "\n\n" + instr

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": prompt_text}]

    trace, photos, answer, error = [], [], "", None
    tool_calls = 0
    t0 = time.monotonic()
    deadline = t0 + DEADLINE_S
    try:
        for _step in range(MAX_STEPS):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                error = "deadline"
                break
            reply = agent._chat(messages, tools, timeout=min(120.0, remaining))
            reply_calls = reply.get("tool_calls") or []
            if not reply_calls:
                answer = (reply.get("content") or "").strip()
                # A thinking-mode model can hand back an empty turn. run_agent
                # nudges here; a real MCP host does not, so record it as the
                # failure it is rather than reporting a silent empty answer.
                if not answer and tool_calls == 0:
                    error = "empty turn, no tool calls"
                break
            messages.append(agent._assistant_tool_msg(reply))
            for call in reply_calls:
                name = call.get("name") or ""
                args = call.get("arguments", {})
                tool_calls += 1
                trace.append({"k": "call", "tool": name, "args": args})
                try:
                    result = toolmod.call_tool(db, name, args)
                except KeyError:
                    result = {"error": f"unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}
                if (name in ("search_photos", "representatives", "rerank_photos",
                             "daily_highlights")
                        and isinstance(result, dict) and "results" in result):
                    photos = result.get("results", []) or []
                trace.append({"k": "result", "tool": name,
                              "summary": agent._summarize(name, result)})
                messages.append(agent._tool_result_msg(call, result))
        else:
            error = "step cap"
    except Exception as exc:
        error = f"harness exception: {type(exc).__name__}: {exc}"

    return _with_timing(time.monotonic() - t0, {
        "tool_calls": tool_calls, "n_results": len(photos),
        "result_ids": [p.get("id") for p in photos[:THUMBS_PER_CELL]],
        "answer": answer, "error": error, "trace": trace, "single_shot": False})


# ---------------------------------------------------------------------------
# Arm 2: the existing agent loop, scored the same way
# ---------------------------------------------------------------------------

def run_agent_path(db, model, prompt_text, *, include_writes, **_ignored):
    os.environ["PHOTOSEARCH_LLM_AGENT_MODEL"] = model
    os.environ["PHOTOSEARCH_ALLOW_WRITES"] = "1" if include_writes else "0"
    _reset_llm()
    trace, photos, answer, error = [], [], "", None
    tool_calls, single_shot = 0, False
    t0 = time.monotonic()
    try:
        for ev in agent.run_agent(db, prompt_text, max_steps=MAX_STEPS):
            t = ev.get("type")
            if t == "tool_call":
                tool_calls += 1
                if ev.get("tool") == "_fallback":
                    single_shot = True
                trace.append({"k": "call", "tool": ev.get("tool"),
                              "args": ev.get("arguments")})
            elif t == "tool_result":
                if ev.get("tool") == "_fallback":
                    single_shot = True
                trace.append({"k": "result", "tool": ev.get("tool"),
                              "summary": ev.get("summary")})
            elif t == "photos":
                photos = ev.get("results", []) or []
            elif t == "answer":
                answer = ev.get("text", "")
            elif t == "error":
                error = ev.get("message")
    except Exception as exc:
        error = f"harness exception: {type(exc).__name__}: {exc}"
    return _with_timing(time.monotonic() - t0, {
        "tool_calls": tool_calls, "n_results": len(photos),
        "result_ids": [p.get("id") for p in photos[:THUMBS_PER_CELL]],
        "answer": answer, "error": error, "trace": trace,
        "single_shot": single_shot})


RUNNERS = {"mcp": run_mcp_path, "agent": run_agent_path}


# ---------------------------------------------------------------------------
# --live: confirm the real streamable-HTTP server advertises what we think
# ---------------------------------------------------------------------------

def live_probe(url):
    """Handshake the real /mcp endpoint, print the advertised surface. This is
    the only place we exercise the transport."""
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async def go():
        async with streamablehttp_client(url) as (r, w, _):
            async with ClientSession(r, w) as s:
                init = await s.initialize()
                instr = getattr(init, "instructions", None)
                print(f"server:       {init.serverInfo.name} {init.serverInfo.version}")
                print(f"instructions: {'(none)' if not instr else repr(instr[:200]) + '…'}")
                listed = await s.list_tools()
                print(f"tools:        {len(listed.tools)}")
                for t in listed.tools:
                    ann = getattr(t, "annotations", None)
                    out = getattr(t, "outputSchema", None)
                    print(f"  {t.name:22s} desc={len(t.description or ''):5d} "
                          f"props={len((t.inputSchema or {}).get('properties', {})):2d} "
                          f"annotations={'yes' if ann else 'no':3s} "
                          f"outputSchema={'yes' if out else 'no'}")
                res = await s.call_tool("get_library_overview", {})
                print(f"\nget_library_overview -> {res.content[0].text[:160]}")
    anyio.run(go)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_html(arms, results, thumbs=False, suffix=""):
    thumb_cache = {}

    def esc(s):
        return html.escape(str(s if s is not None else ""))

    def trace_html(trace):
        rows = []
        for step in trace:
            if step["k"] == "call":
                args = json.dumps(step.get("args") or {}, ensure_ascii=False)
                cls = "call ground" if step["tool"] in GROUNDING_TOOLS else "call"
                rows.append(f'<div class="{cls}">▸ {esc(step["tool"])} '
                            f'<span class="args">{esc(args)}</span></div>')
            else:
                rows.append(f'<div class="res">↳ {esc(step.get("summary"))}</div>')
        return "".join(rows) or '<div class="res muted">(no tool calls)</div>'

    def thumbs_html(ids):
        if not thumbs:
            return ""
        imgs = []
        for pid in ids:
            if pid is None:
                continue
            b = _fetch_thumb_b64(pid, thumb_cache)
            if b:
                imgs.append(f'<img class="thumb" src="data:image/jpeg;base64,{b}"/>')
        return f'<div class="thumbs">{"".join(imgs)}</div>' if imgs else ""

    def cell(run):
        if not run:
            return '<span class="muted">—</span>'
        sc = run["score"]
        badges = [f'<span class="b">{run["seconds"]}s</span>',
                  f'<span class="b">{run["tool_calls"]} calls</span>']
        cls = "ok" if sc["checks_passed"] == sc["checks_total"] else (
            "warn" if sc["checks_passed"] else "err")
        badges.append(f'<span class="b {cls}">{sc["checks_passed"]}/{sc["checks_total"]} checks</span>')
        if not sc["tool_choice_ok"]:
            badges.append('<span class="b err">wrong tool</span>')
        if sc["grounding_calls"]:
            badges.append(f'<span class="b warn">{sc["grounding_calls"]} grounding</span>')
        if run["single_shot"]:
            badges.append('<span class="b warn">single-shot</span>')
        if run["error"]:
            badges.append(f'<span class="b err">{esc(run["error"])}</span>')
        checks = "".join(
            f'<div class="chk {"y" if ok else "n"}">{"✓" if ok else "✗"} {esc(n)}</div>'
            for n, ok in sc["checks"])
        ans = esc(run["answer"]) or '<span class="muted">(no answer)</span>'
        return (f'<div class="badges">{"".join(badges)}</div>'
                f'<div class="checks">{checks}</div>'
                f'<div class="answer">{ans}</div>{thumbs_html(run["result_ids"])}'
                f'<details><summary>trace</summary><div class="trace">{trace_html(run["trace"])}</div></details>')

    score_rows = []
    for arm in arms:
        runs = [results[p]["runs"].get(arm) for p in results]
        runs = [r for r in runs if r]
        n = len(runs) or 1
        cp = sum(r["score"]["checks_passed"] for r in runs)
        ct = sum(r["score"]["checks_total"] for r in runs)
        tc = sum(1 for r in runs if r["score"]["tool_choice_ok"])
        gr = sum(r["score"]["grounding_calls"] for r in runs)
        ne = sum(1 for r in runs if r["n_results"])
        er = sum(1 for r in runs if r["error"])
        avg = round(sum(r["seconds"] for r in runs) / n, 1)
        avg_llm = round(sum(r.get("llm_seconds", 0) for r in runs) / n, 1)
        avg_tool = round(sum(r.get("tool_seconds", 0) for r in runs) / n, 1)
        pct = round(100 * cp / max(ct, 1))
        score_rows.append(
            f"<tr><td><code>{esc(arm)}</code></td><td><b>{cp}/{ct}</b> ({pct}%)</td>"
            f"<td>{tc}/{len(runs)}</td><td>{gr}</td><td>{ne}/{len(runs)}</td>"
            f"<td>{er}</td><td>{avg}s</td><td>{avg_llm}s</td><td>{avg_tool}s</td></tr>")

    head = "".join(f"<th>{esc(a)}</th>" for a in arms)
    body = []
    for pid, info in results.items():
        cells = "".join(f'<td class="cell">{cell(info["runs"].get(a))}</td>' for a in arms)
        body.append(f'<tr><td class="prompt"><div class="pid">{esc(pid)}</div>'
                    f'<div class="ptext">{esc(info["prompt"])}</div>'
                    f'<div class="probe">{esc(info["probes"])}</div></td>{cells}</tr>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MCP tool-definition bake-off</title><style>
 body {{ font:14px/1.5 system-ui,sans-serif; margin:24px; color:#1a1a1a; background:#fafafa; }}
 h1 {{ font-size:22px; margin-bottom:4px; }} .meta {{ color:#666; margin-bottom:20px; }}
 table {{ border-collapse:collapse; width:100%; background:#fff; margin-bottom:28px; }}
 th,td {{ border:1px solid #ddd; vertical-align:top; padding:8px; text-align:left; }}
 th {{ background:#f0f0f3; }}
 td.prompt {{ width:200px; background:#fbfbfd; }}
 .pid {{ font:12px monospace; color:#888; }}
 .ptext {{ font-weight:600; margin:2px 0 6px; }}
 .probe {{ font-size:12px; color:#555; }}
 .b {{ display:inline-block; font-size:11px; background:#eef; border-radius:10px; padding:1px 8px; margin:0 4px 4px 0; }}
 .b.ok {{ background:#cdebc5; }} .b.warn {{ background:#fde9c8; }} .b.err {{ background:#f7c5c5; }}
 .chk {{ font-size:12px; font-family:ui-monospace,monospace; }}
 .chk.y {{ color:#2b7a2b; }} .chk.n {{ color:#a00; }}
 .checks {{ margin:6px 0; }} .answer {{ margin:6px 0; }}
 .trace {{ font:12px ui-monospace,monospace; background:#f7f7fa; padding:6px; }}
 .call {{ color:#224; }} .call.ground {{ color:#a60; }} .res {{ color:#666; }}
 .args {{ color:#888; }} .muted {{ color:#999; }}
 .thumbs img {{ height:64px; margin:2px; border-radius:3px; }}
</style></head><body>
<h1>MCP tool-definition bake-off</h1>
<div class="meta">Does the advertised tool surface alone carry enough for a small model?
Generated {esc(dt.datetime.now().isoformat(timespec="seconds"))} ·
arms are <code>path[+instructions][+facts][+writes]</code>.
Amber tool calls in a trace are grounding round-trips.</div>
<table><tr><th>arm</th><th>arg checks</th><th>tool choice</th><th>grounding calls</th>
<th>non-empty</th><th>errors</th><th>avg wall</th><th>avg LLM</th><th>avg tools</th></tr>
{"".join(score_rows)}</table>
<div class="meta">Wall-clock is not a clean latency comparison: a run that fails early
finishes faster, the deadline censors the slow tail, and CLIP's first load lands on one
prompt. Compare <b>avg LLM</b> and <b>avg tools</b> instead.</div>
<table><tr><th>prompt</th>{head}</tr>{"".join(body)}</table>
</body></html>"""
    (OUT_DIR / f"mcp_bakeoff_report{suffix}.html").write_text(doc)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS_DEFAULT))
    ap.add_argument("--paths", default="mcp,agent", help="mcp,agent")
    ap.add_argument("--writes", default="off", help="on,off — advertise the 3 write tools")
    ap.add_argument("--instructions", action="store_true",
                    help="pass tools.ROUTING_GUIDANCE as MCP instructions (Phase 2)")
    ap.add_argument("--facts", action="store_true",
                    help="inject library facts into the MCP system prompt (Phase 2)")
    ap.add_argument("--prompts", default="", help="comma-separated ids, e.g. p01,p05")
    ap.add_argument("--thumbs", action="store_true", help="embed thumbnails (needs NAS)")
    ap.add_argument("--live", action="store_true", help="probe the real /mcp endpoint and exit")
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--tag", default="", help="suffix the output files, e.g. --tag baseline")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.live:
        live_probe(os.environ.get("PHOTOSEARCH_MCP_URL", "http://localhost:8848/mcp"))
        return

    results_path = OUT_DIR / f"mcp_bakeoff_results{suffix}.json"
    if args.html_only:
        data = json.loads(results_path.read_text())
        build_html(data["arms"], data["results"], thumbs=args.thumbs, suffix=suffix)
        print(f"Rebuilt {OUT_DIR/f'mcp_bakeoff_report{suffix}.html'}")
        return

    if not os.environ.get("PHOTOSEARCH_TEXT_LLM_URL"):
        sys.exit("PHOTOSEARCH_TEXT_LLM_URL must be set (LM Studio /v1 endpoint)")

    models = [m for m in args.models.split(",") if m]
    paths = [p for p in args.paths.split(",") if p]
    writes = [w == "on" for w in args.writes.split(",") if w]
    want = {p for p in args.prompts.split(",") if p}
    prompts = [p for p in PROMPTS if not want or p[0] in want]

    # One arm per (model, path, writes, knobs) cell.
    arms, arm_cfg = [], {}
    for model in models:
        for path in paths:
            for w in writes:
                name = f"{model}|{path}"
                if path == "mcp":
                    if args.instructions:
                        name += "+instr"
                    if args.facts:
                        name += "+facts"
                name += "+writes" if w else ""
                arms.append(name)
                arm_cfg[name] = dict(model=model, path=path, include_writes=w,
                                     use_instructions=args.instructions,
                                     use_facts=args.facts)

    if args.instructions and not hasattr(toolmod, "server_instructions"):
        print("note: --instructions set but tools.server_instructions() does not "
              "exist on this tree; MCP arms run without it (Phase-1 baseline).\n")

    results = {}
    db = PhotoDB(REPLICA_DB)
    try:
        for pid, ptext, probes, expect in prompts:
            results[pid] = {"prompt": ptext, "probes": probes, "expect": expect, "runs": {}}
            for arm in arms:
                cfg = dict(arm_cfg[arm])
                path = cfg.pop("path")
                print(f"[{pid}] {arm} … ", end="", flush=True)
                run = RUNNERS[path](db, cfg.pop("model"), ptext, **cfg)
                run["score"] = score(pid, run["trace"])
                results[pid]["runs"][arm] = run
                sc = run["score"]
                print(f'{run["seconds"]}s  {sc["checks_passed"]}/{sc["checks_total"]} checks  '
                      f'{run["tool_calls"]} calls ({sc["grounding_calls"]} grounding)'
                      + ("  [SINGLE-SHOT: not a tool-loop result]" if run["single_shot"] else "")
                      + (f'  ERR:{run["error"]}' if run["error"] else ""))
                results_path.write_text(json.dumps(
                    {"arms": arms, "results": results}, indent=2, default=str))
    finally:
        db.close()

    build_html(arms, results, thumbs=args.thumbs, suffix=suffix)
    print(f"\nWrote {OUT_DIR/f'mcp_bakeoff_report{suffix}.html'}")


if __name__ == "__main__":
    main()

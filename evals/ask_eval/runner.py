#!/usr/bin/env python3
"""Run Ask-agent jobs inside ONE source tree and print raw rows (JSON lines).

harness.py starts this with `--repo <tree>`: the tree's own `photosearch`
package is imported, so the "before" ref runs the old agent and prompt and the
"after" ref runs the new one, while this file (and the grading) always comes
from the harness's tree. Grading is not done here.

Two modes:
  --measure        no LLM call: the size of every prompt component the tree
                   would send (system prompt parts, each tool's description and
                   schema), as JSON on stdout.
  --jobs FILE      run each job ({item, rep, model}) through agent.run_agent
                   and print one row per job.

Safety, always on:
  - confirmed writes (`confirm=true` on a write tool) are never executed: the
    call is recorded and answered with an error. Dry runs execute (they write
    nothing). PHOTOSEARCH_NAS_URL is removed so nothing proxies to the NAS.
  - rerank_photos is faked unless --allow-rerank: it loads a vision model and
    takes minutes, which the prompt diet does not change. The fake returns the
    real handler's candidates in input order, shaped like a scored rerank, so
    the agent's next turn sees a normal result.
  --ablate ID      apply one prompt mutant (mutants.py) to every tool-calling
                   request: the "mutation testing" run.
  --list-mutants   no LLM call: every mutant id and how many characters it
                   removes from this tree's prompt (0 = text absent).
  - the Ask log goes to the results dir, the request log is off.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutants  # noqa: E402  (the harness tree's copy, not the tree under test's)


def _setup(repo: str, db_path: str, log_dir: str) -> None:
    sys.path.insert(0, repo)
    os.environ["PHOTOSEARCH_DB"] = db_path
    os.environ.pop("PHOTOSEARCH_NAS_URL", None)
    os.environ["PHOTOSEARCH_REQUEST_LOG"] = "0"
    os.environ["PHOTOSEARCH_ASK_LOG_DIR"] = log_dir


def _json_len(x) -> int:
    return len(json.dumps(x, ensure_ascii=False))


def measure(db_path: str) -> dict:
    """Component sizes in characters, as the tree under test builds them."""
    from photosearch import agent
    from photosearch import tools as toolmod
    from photosearch.db import PhotoDB

    db = PhotoDB(db_path)
    try:
        writes = agent._writes_enabled()
        out: dict = {"writes": writes, "surfaces": {}}
        surfaces = [False]
        if hasattr(toolmod, "consolidated_search_enabled"):
            surfaces.append(True)
        for cons in surfaces:
            kw = {"consolidated": cons} if hasattr(toolmod, "consolidated_search_enabled") else {}
            tools = toolmod.openai_tools(include_images=False, include_writes=writes, **kw)
            try:
                system = agent._system_prompt(db, allow_writes=writes, **kw)
            except TypeError:
                system = agent._system_prompt(db, allow_writes=writes)
            parts = {}
            facts = agent._library_context(db) if hasattr(agent, "_library_context") else ""
            parts["library_facts"] = len(facts)
            if hasattr(toolmod, "routing_guidance"):
                try:
                    parts["routing_guidance"] = len(toolmod.routing_guidance(cons))
                except TypeError:
                    parts["routing_guidance"] = len(toolmod.routing_guidance())
            if writes and hasattr(toolmod, "WRITE_GUIDANCE"):
                parts["write_guidance"] = len(toolmod.WRITE_GUIDANCE)
            if hasattr(toolmod, "GROUNDING_WITH_FACTS"):
                parts["grounding"] = len(toolmod.GROUNDING_WITH_FACTS if facts
                                         else toolmod.GROUNDING_WITHOUT_FACTS)
            parts["user_hints"] = len(os.environ.get("PHOTOSEARCH_AGENT_HINTS", "").strip())
            per_tool = []
            for t in tools:
                f = t["function"]
                per_tool.append({"name": f["name"], "description": len(f.get("description") or ""),
                                 "parameters": _json_len(f.get("parameters") or {}),
                                 "total": _json_len(t)})
            out["surfaces"]["consolidated" if cons else "default"] = {
                "system_chars": len(system),
                "system_parts": parts,
                "tools_chars": _json_len(tools),
                "n_tools": len(tools),
                "tools": sorted(per_tool, key=lambda r: -r["total"]),
                "total_chars": len(system) + _json_len(tools),
            }
        return out
    finally:
        db.close()


class _Recorder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rounds: list = []
        self.first_prompt: dict = {}
        self.blocked_writes: list = []
        self.stubbed: list = []
        self.tool_seconds = 0.0
        self.mutant_removed = None


def _shared_props(toolmod) -> list:
    props = getattr(toolmod, "_CURATION_FILTER_PROPS", None)
    return list(props) if isinstance(props, dict) else list(mutants.DEFAULT_SHARED_PROPS)


def _agent_prompt(db):
    """(system, tools) exactly as run_agent builds them for a plain Ask."""
    from photosearch import agent
    from photosearch import tools as toolmod
    writes = agent._writes_enabled()
    kw = {}
    if hasattr(toolmod, "consolidated_search_enabled"):
        kw["consolidated"] = toolmod.consolidated_search_enabled()
    tools = toolmod.openai_tools(include_images=False, include_writes=writes, **kw)
    try:
        system = agent._system_prompt(db, allow_writes=writes, **kw)
    except TypeError:
        system = agent._system_prompt(db, allow_writes=writes)
    return system, tools


def list_mutants(db_path: str) -> dict:
    from photosearch import tools as toolmod
    from photosearch.db import PhotoDB
    db = PhotoDB(db_path)
    try:
        system, tools = _agent_prompt(db)
    finally:
        db.close()
    shared = _shared_props(toolmod)
    out = {"total_chars": len(system) + _json_len(tools), "mutants": {},
           "params": {t["function"]["name"]: sorted((t["function"].get("parameters") or {})
                                                    .get("properties", {}))
                      for t in tools}}
    for mid in mutants.all_ids(system, tools):
        out["mutants"][mid] = mutants.apply(mid, system, tools, shared_props=shared)[2]
    return out


def _fake_rerank(call_orig, db, args):
    """The real handler with no vision model configured returns the candidates
    unranked; dress that up as a scored rerank so the agent proceeds normally."""
    saved = {k: os.environ.pop(k, None) for k in
             ("PHOTOSEARCH_LLM_RERANK_MODEL", "PHOTOSEARCH_LLM_VISUAL_MODEL")}
    try:
        out = call_orig(db, "rerank_photos", args)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    if not isinstance(out, dict) or "results" not in out:
        return out
    hits = out["results"]
    for i, h in enumerate(hits):
        h["rerank_score"] = round(0.9 - 0.01 * i, 3)
        h["rerank_reason"] = "matches the criteria"
    return {"reranked": True, "criteria": out.get("criteria"), "model": "vision",
            "scored": len(hits), "returned": len(hits), "results": hits}


def _install_wrappers(rec: _Recorder, allow_rerank: bool, mutant: str | None = None) -> None:
    from photosearch import agent
    from photosearch import tools as toolmod

    chat_orig = agent._chat
    shared = _shared_props(toolmod)

    def chat(messages, tools, *a, **kw):
        if mutant and tools:
            # Only the tool-calling loop is mutated; the single-shot fallback
            # (tools=None) keeps its own prompt.
            messages = [dict(m) for m in messages]
            sys_i = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
            system = messages[sys_i]["content"] if sys_i is not None else ""
            system, tools, removed = mutants.apply(mutant, system, tools, shared_props=shared)
            if sys_i is not None:
                messages[sys_i]["content"] = system
            if rec.mutant_removed is None:
                rec.mutant_removed = removed
        if not rec.first_prompt:
            system = next((m.get("content") or "" for m in messages
                           if m.get("role") == "system"), "")
            rec.first_prompt = {"system_chars": len(system),
                                "tools_chars": _json_len(tools or []),
                                "n_tools": len(tools or []),
                                "tool_names": [t["function"]["name"] for t in (tools or [])]}
        t0 = time.monotonic()
        entry: dict = {"tools_offered": bool(tools), "messages": len(messages)}
        try:
            reply = chat_orig(messages, tools, *a, **kw)
        except urllib.error.HTTPError as e:
            # The agent logs only "HTTP Error 400"; LM Studio's reason
            # ("Context size has been exceeded.") is in the body.
            try:
                body = e.read().decode("utf-8", "replace")[:600]
            except Exception:
                body = ""
            entry.update(error=f"HTTP {e.code}", error_body=body)
            raise
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            raise
        else:
            u = reply.get("usage") or {}
            details = u.get("completion_tokens_details") or {}
            entry.update(prompt_tokens=u.get("prompt_tokens"),
                         completion_tokens=u.get("completion_tokens"),
                         reasoning_tokens=details.get("reasoning_tokens"),
                         reasoning_chars=len(reply.get("reasoning") or ""),
                         tool_calls=len(reply.get("tool_calls") or []))
            return reply
        finally:
            entry["seconds"] = round(time.monotonic() - t0, 2)
            rec.rounds.append(entry)

    agent._chat = chat

    call_orig = toolmod.call_tool
    write_names = {s.name for s in toolmod.all_tools() if getattr(s, "writes", False)}

    def call_tool(db, name, args=None):
        args = args or {}
        confirm = str(args.get("confirm")).strip().lower() in ("1", "true", "yes", "on")
        if name in write_names and confirm:
            rec.blocked_writes.append({"tool": name, "args": args})
            return {"error": "eval harness: confirmed writes are disabled; nothing was written"}
        if name == "rerank_photos" and not allow_rerank:
            rec.stubbed.append(name)
            return _fake_rerank(call_orig, db, args)
        t0 = time.monotonic()
        try:
            return call_orig(db, name, args)
        finally:
            rec.tool_seconds += time.monotonic() - t0

    toolmod.call_tool = call_tool


def run_job(db, job: dict, rec: _Recorder) -> dict:
    from photosearch import agent

    item, model = job["item"], job["model"]
    os.environ["PHOTOSEARCH_LLM_AGENT_MODEL"] = model
    rec.reset()
    trace, photos, total, answer, error = [], [], None, "", None
    single_shot, fallback_error = False, None
    kwargs = {}
    if item.get("locked"):
        kwargs["locked_filters"] = item["locked"]
    t0 = time.monotonic()
    try:
        for ev in agent.run_agent(db, item["prompt"], item.get("history") or [], **kwargs):
            t = ev.get("type")
            if t == "tool_call":
                trace.append({"k": "call", "tool": ev.get("tool"), "args": ev.get("arguments")})
            elif t == "tool_result":
                if ev.get("tool") == "_fallback":
                    single_shot = True
                    fallback_error = ev.get("summary")
                trace.append({"k": "result", "tool": ev.get("tool"), "summary": ev.get("summary")})
            elif t == "photos":
                photos = ev.get("results") or []
                total = ev.get("total")
            elif t == "answer":
                answer = ev.get("text") or ""
            elif t == "error":
                error = ev.get("message")
    except Exception as e:  # a crash in the tree under test is a failed run
        error = f"runner exception: {type(e).__name__}: {e}"
    seconds = time.monotonic() - t0
    llm_seconds = sum(r.get("seconds") or 0 for r in rec.rounds)
    prompt_tokens = [r["prompt_tokens"] for r in rec.rounds if r.get("prompt_tokens")]
    return {
        "item_id": item["id"], "suite": item["suite"], "rep": job["rep"], "model": model,
        "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
        "trace": trace,
        "result_ids": [p.get("id") for p in photos[:200]],
        "n_results": len(photos), "total": total,
        "answer": answer, "error": error,
        "single_shot": single_shot, "fallback_error": fallback_error,
        "seconds": round(seconds, 1), "llm_seconds": round(llm_seconds, 1),
        "tool_seconds": round(rec.tool_seconds, 1),
        "rounds": rec.rounds,
        "prompt": rec.first_prompt,
        "first_prompt_tokens": prompt_tokens[0] if prompt_tokens else None,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "context_errors": [r for r in rec.rounds
                           if "context" in str(r.get("error_body") or "").lower()],
        "blocked_writes": rec.blocked_writes, "stubbed": rec.stubbed,
        "mutant": job.get("mutant"), "mutant_removed": rec.mutant_removed,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--jobs")
    ap.add_argument("--allow-rerank", action="store_true")
    ap.add_argument("--ablate")
    ap.add_argument("--list-mutants", action="store_true")
    a = ap.parse_args(argv)
    _setup(a.repo, a.db, a.log_dir)

    if a.measure:
        print(json.dumps(measure(a.db)))
        return 0
    if a.list_mutants:
        print(json.dumps(list_mutants(a.db)))
        return 0

    from photosearch.db import PhotoDB
    rec = _Recorder()
    _install_wrappers(rec, a.allow_rerank, a.ablate)
    jobs = json.loads(open(a.jobs).read())
    db = PhotoDB(a.db)
    try:
        for job in jobs:
            print(json.dumps(run_job(db, job, rec), default=str), flush=True)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

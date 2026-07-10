"""In-app LLM agent for natural-language photo search (M24b).

Runs a tool-calling loop over the shared tool layer (``tools.py``) on the
**local** LLM backend — LM Studio / llama-server via
``PHOTOSEARCH_TEXT_LLM_URL`` (OpenAI-compatible), or Ollama if that's not set.
Nothing leaves the NAS. Surfaced by web.py's ``POST /api/ask`` SSE endpoint
and shares the exact tool definitions the MCP server advertises.

The loop:
  1. system + history + user message, with the tool schemas attached.
  2. The model emits tool_calls → we dispatch each via ``tools.call_tool`` and
     feed the JSON result back → repeat until the model answers in prose or we
     hit ``max_steps``.
  3. The most recent ``search_photos`` result set is what the UI renders.

Two tiers:
  - **Full agent loop** (default) — needs a tool-calling-capable local model
    (qwen2.5-instruct / qwen3 / llama-3.1+ in LM Studio with tool use on).
  - **Single-shot fallback** — for models that can't tool-call. One constrained
    JSON completion fills ``search_photos`` arguments; we run one search. Set
    ``PHOTOSEARCH_AGENT_SINGLE_SHOT=1`` to force it; it's also used
    automatically if the model returns neither tool calls nor prose.

``run_agent`` is a *synchronous generator* yielding event dicts so the SSE
endpoint can bridge it through the established thread→asyncio.Queue pattern:

    {"type": "tool_call",   "tool": str, "arguments": dict}
    {"type": "tool_result", "tool": str, "summary": str}
    {"type": "photos",      "results": [...], "total": int}
    {"type": "answer",      "text": str}
    {"type": "error",       "message": str}
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from typing import Callable, Iterator, Optional

from .db import PhotoDB
from . import tools as toolmod

# Bound the loop so a confused model can't spin forever. Each step is one LLM
# round-trip (which may issue several tool calls).
_MAX_STEPS = 6
# Bounded re-prompts when a model returns an empty turn (no tool call, no text)
# before producing any results — e.g. a thinking-mode model (qwen3) that grounds
# via a tool then stalls instead of calling search_photos. Keeps thinking mode
# usable for hard prompts while preventing a silent "Found 0".
_MAX_NUDGES = 2
_HTTP_TIMEOUT_S = 120.0
# Anti-hang guards. A thinking model (qwen3.5) can ramble for minutes on an
# unsupported query; without caps a single Ask blew 158s and returned nothing.
_MAX_TOKENS = 3000          # runaway-generation backstop per LLM call
_DEFAULT_DEADLINE_S = 120.0  # overall wall-clock budget per Ask (env override)
# A rerank_photos call runs a VISION model over up to 24 images — minutes in the
# worst case. When the agent invokes it we extend the wall-clock budget so the
# follow-up turn that consumes the scores isn't killed mid-flight by the deadline.
_RERANK_DEADLINE_EXTEND_S = 150.0


def _writes_enabled() -> bool:
    """Whether the agent may call the M26b mutation tools. ON by default — the
    deployment lives behind Tailscale (same trust boundary as the deploy panel).
    Set PHOTOSEARCH_ALLOW_WRITES to a falsy value (0/false/no/off) to disable."""
    v = os.environ.get("PHOTOSEARCH_ALLOW_WRITES")
    if v is None or not v.strip():
        return True
    return v.strip().lower() in ("1", "true", "yes", "on")


def _agent_model() -> str:
    """Resolve the agent model id by role, mirroring describe.py's role map.

    PHOTOSEARCH_LLM_AGENT_MODEL > PHOTOSEARCH_LLM_TEXT_MODEL >
    PHOTOSEARCH_TEXT_LLM_MODEL > a sane default.
    """
    return (
        os.environ.get("PHOTOSEARCH_LLM_AGENT_MODEL")
        or os.environ.get("PHOTOSEARCH_LLM_TEXT_MODEL")
        or os.environ.get("PHOTOSEARCH_TEXT_LLM_MODEL")
        or os.environ.get("PHOTOSEARCH_AGENT_OLLAMA_MODEL")
        or "llama3.1"
    )


# Thinking models (qwen3.5) spend most of their decode budget on reasoning traces
# the user never sees — measured at ~71% of generated tokens across the Ask
# prompts, and suppressing them roughly halved median latency with no measurable
# accuracy change (evals/mcp_bakeoff.py). Only "none" actually disables them on
# LM Studio; chat_template_kwargs and reasoning_effort="low" are ignored.
_REASONING_OFF = "none"


def _resolve_reasoning_effort(override: Optional[str] = None) -> str:
    """Per-request override > PHOTOSEARCH_LLM_REASONING_EFFORT > model default.

    Returns "" to send no `reasoning_effort` at all (model default / thinking on),
    which is also what non-OpenAI backends need.
    """
    if override is not None:
        return (override or "").strip()
    return os.environ.get("PHOTOSEARCH_LLM_REASONING_EFFORT", "").strip()


_CONTEXT_CACHE: dict = {}
_CONTEXT_TTL_S = 600


def _top_json_vocab(db, column: str, n: int) -> list:
    """Top-N distinct values across a JSON-array column (categories etc.).
    A full scan — only called from the cached _library_context."""
    counts: dict = {}
    for (raw,) in db.conn.execute(
        f"SELECT {column} FROM photos WHERE {column} IS NOT NULL AND {column} != '[]'"):
        try:
            for t in json.loads(raw):
                if isinstance(t, str):
                    counts[t] = counts.get(t, 0) + 1
        except (ValueError, TypeError):
            pass
    return [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


def _library_context(db) -> str:
    """Compact, cached snapshot of the library — people / places / vocab / date
    span — injected into the system prompt so the model can plan WITHOUT first
    spending tool calls (and nudges) to discover them. The weak models that
    don't ground well get the facts for free. Cached per DB for _CONTEXT_TTL_S
    (the vocab aggregation is a full scan)."""
    key = getattr(db, "db_path", "?")
    cached = _CONTEXT_CACHE.get(key)
    if cached and (time.time() - cached[0] < _CONTEXT_TTL_S):
        return cached[1]
    c = db.conn
    people = [r[0] for r in c.execute(
        "SELECT p.name FROM persons p LEFT JOIN faces f ON f.person_id=p.id "
        "GROUP BY p.id ORDER BY COUNT(f.photo_id) DESC LIMIT 40")]
    places = [r[0] for r in c.execute(
        "SELECT place_name FROM photos WHERE place_name IS NOT NULL "
        "GROUP BY place_name ORDER BY COUNT(*) DESC LIMIT 20")]
    cats = _top_json_vocab(db, "categories", 30)
    vtags = _top_json_vocab(db, "visual_tags", 30)
    dr = c.execute(
        "SELECT MIN(date_taken), MAX(date_taken) FROM photos "
        "WHERE date_taken GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*'").fetchone()
    lo, hi = (dr[0] or "")[:4], (dr[1] or "")[:4]
    parts = []
    if people:
        parts.append("People (use these EXACT names in `people`; ONLY these are "
                     "registered): " + ", ".join(people) + ".")
    if lo and hi:
        parts.append(f"Photos span {lo}-{hi}.")
    if places:
        parts.append("Common locations (for `location`): " + ", ".join(places) + ".")
    if cats:
        parts.append("Content categories (for `category`): " + ", ".join(cats) + ".")
    if vtags:
        parts.append("Visual tags (for `visual_tag`): " + ", ".join(vtags) + ".")
    text = "\n".join(parts)
    _CONTEXT_CACHE[key] = (time.time(), text)
    return text


def _system_prompt(db=None, allow_writes: bool = False,
                   consolidated: Optional[bool] = None) -> str:
    """The agent's system prompt. The routing rules live in tools.py so the MCP
    server can serve the identical text as its ``instructions`` — same
    single-source-of-truth argument as the schemas themselves. ``consolidated``
    (None = env) selects the routing guidance matching the offered tool surface."""
    try:
        ctx = _library_context(db) if db is not None else ""
    except Exception:
        ctx = ""
    # The agent injects the facts whenever it has a db, so it gets the "skip the
    # list_* tools" directive; an MCP client without the facts gets the opposite
    # one. That conditional is exactly why the tool descriptions themselves must
    # stay neutral about whether to ground first.
    base = (
        "You are a photo-library search assistant. The user describes what they "
        "want in natural language; find the matching photos with the tools, then "
        "give a one- or two-sentence answer.\n\n"
        + (toolmod.GROUNDING_WITH_FACTS if ctx else toolmod.GROUNDING_WITHOUT_FACTS)
        + "\n\n" + toolmod.routing_guidance(consolidated)
    )
    hints = os.environ.get("PHOTOSEARCH_AGENT_HINTS", "").strip()
    out = base + (toolmod.WRITE_GUIDANCE if allow_writes else "")
    if ctx:
        out += "\nLIBRARY FACTS:\n" + ctx + "\n"
    if hints:
        out += "\nUSER NOTES:\n" + hints + "\n"
    out += f"\nToday's date is {date.today().isoformat()}."
    return out


# ---------------------------------------------------------------------------
# Tool-calling chat client (OpenAI-compatible primary, Ollama fallback)
# ---------------------------------------------------------------------------

def _chat(messages: list, tools: Optional[list], temperature: float = 0.0,
          timeout: float = _HTTP_TIMEOUT_S,
          reasoning_effort: Optional[str] = None) -> dict:
    """One chat round-trip. Returns a normalized dict:
        {"content": str|None, "tool_calls": [{"id", "name", "arguments"}]}

    Routes to the OpenAI-compatible endpoint when PHOTOSEARCH_TEXT_LLM_URL is
    set (the project's local LM Studio path), else to Ollama. `timeout` bounds
    the call so a single rambling generation can't hang the whole Ask.
    """
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if base:
        return _chat_openai(base, messages, tools, temperature, timeout,
                            _resolve_reasoning_effort(reasoning_effort))
    # Ollama has no reasoning_effort knob; the setting is a no-op there.
    return _chat_ollama(messages, tools, temperature)


def _chat_openai(base_url: str, messages: list, tools: Optional[list],
                 temperature: float, timeout: float = _HTTP_TIMEOUT_S,
                 reasoning_effort: str = "") -> dict:
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body: dict = {
        "model": _agent_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": _MAX_TOKENS,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=max(5.0, timeout))
    data = json.loads(resp.read())
    msg = data["choices"][0]["message"]

    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {}
        calls.append({"id": tc.get("id"), "name": fn.get("name"),
                      "arguments": args or {}})
    return {"content": msg.get("content"), "tool_calls": calls,
            # Token accounting, so evals can split prefill from decode.
            "usage": data.get("usage") or {},
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning")}


def _chat_ollama(messages: list, tools: Optional[list], temperature: float) -> dict:
    import ollama

    # Ollama accepts OpenAI-style tool schemas and tool/assistant messages.
    resp = ollama.chat(
        model=_agent_model(),
        messages=messages,
        tools=tools or None,
        options={"temperature": temperature},
    )
    m = resp.message
    calls = []
    for tc in (getattr(m, "tool_calls", None) or []):
        fn = tc.function
        args = fn.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                args = {}
        calls.append({"id": None, "name": fn.name, "arguments": args or {}})
    return {"content": m.content, "tool_calls": calls}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize(tool: str, result) -> str:
    """A short human-readable narration line for a tool result."""
    if isinstance(result, dict):
        if "error" in result:
            return f"error: {result['error']}"
        if "total" in result:
            return f"{result['total']} match(es)"
        if "people" in result:
            return f"{len(result['people'])} person(s)"
        if "places" in result:
            return f"{len(result['places'])} place(s)"
        if "total_photos" in result:
            return f"{result['total_photos']} photos, {result.get('registered_people', 0)} people"
        # vocab dict of lists
        counts = {k: len(v) for k, v in result.items() if isinstance(v, list)}
        if counts:
            return ", ".join(f"{k}:{n}" for k, n in counts.items())
    if isinstance(result, list):
        return f"{len(result)} item(s)"
    return "ok"


def _assistant_tool_msg(reply: dict) -> dict:
    """Rebuild the assistant message (with tool_calls) to append to history."""
    tcs = []
    for i, c in enumerate(reply["tool_calls"]):
        tcs.append({
            "id": c.get("id") or f"call_{i}",
            "type": "function",
            "function": {"name": c["name"],
                         "arguments": json.dumps(c["arguments"])},
        })
    return {"role": "assistant", "content": reply.get("content") or "", "tool_calls": tcs}


_MODEL_RESULT_CAP = 15  # photo hits fed back to the model per tool call


def _lean_for_model(result):
    """Trim a tool result before feeding it BACK to the model. Photo-list tools
    (search_photos / representatives / rerank_photos) return up to 50 hits with
    240-char descriptions + thumbnail urls — feeding all of that into a
    multi-step conversation overflows the model's context (the rerank flow 400'd
    on it). Cap the list and keep only lean fields. The UI still gets the FULL
    result set (captured separately as last_photos), not this trimmed copy."""
    if not (isinstance(result, dict) and isinstance(result.get("results"), list)):
        return result
    lean = {k: v for k, v in result.items() if k != "results"}
    trimmed = []
    for h in result["results"][:_MODEL_RESULT_CAP]:
        if not isinstance(h, dict):
            trimmed.append(h)
            continue
        t = {"id": h.get("id"), "filename": h.get("filename"),
             "date_taken": h.get("date_taken"), "place_name": h.get("place_name")}
        if h.get("description"):
            t["description"] = h["description"][:100]
        for k in ("bucket", "rerank_score", "rerank_reason", "aesthetic_score"):
            if h.get(k) is not None:
                t[k] = h[k]
        trimmed.append(t)
    lean["results"] = trimmed
    if len(result["results"]) > len(trimmed):
        lean["results_note"] = f"showing {len(trimmed)} of {len(result['results'])} (use these ids)"
    return lean


def _tool_result_msg(call: dict, result) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.get("id") or f"call_{call['name']}",
        "name": call["name"],
        "content": json.dumps(_lean_for_model(result), default=str),
    }


# ---------------------------------------------------------------------------
# Single-shot fallback (for non-tool-calling models)
# ---------------------------------------------------------------------------

_SINGLE_SHOT_INSTRUCTION = (
    "Translate the user's request into a JSON object of search_photos "
    "arguments. Valid keys: query (string, free-text visual), people (array of "
    "names), location (string), date_from / date_to (YYYY-MM-DD), color, "
    "category, visual_tag, keyword, min_quality (number), limit (integer). "
    "Include ONLY the keys you can infer; omit the rest. Respond with the JSON "
    "object and nothing else."
)


def _run_single_shot(db: PhotoDB, message: str,
                     timeout: float = _HTTP_TIMEOUT_S,
                     locked: Optional[dict] = None,
                     reasoning_effort: Optional[str] = None) -> Iterator[dict]:
    reply = _chat(
        [{"role": "system", "content": _SINGLE_SHOT_INSTRUCTION
          + (_locked_prompt(locked) if locked else "")},
         {"role": "user", "content": message}],
        tools=None, timeout=timeout, reasoning_effort=reasoning_effort,
    )
    text = (reply.get("content") or "").strip()
    # Strip ```json fences if present.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    try:
        start, end = text.find("{"), text.rfind("}")
        args = json.loads(text[start:end + 1]) if start >= 0 and end > start else {}
    except (ValueError, TypeError):
        args = {}
    # Enforce the pinned filters even on the single-shot path.
    args = _merge_locked("search_photos", args if isinstance(args, dict) else {}, locked or {})
    if not isinstance(args, dict) or not args:
        yield {"type": "answer",
               "text": "I couldn't turn that into a search. Try rephrasing, or "
                       "use the structured filters."}
        return
    yield {"type": "tool_call", "tool": "search_photos", "arguments": args}
    result = toolmod.call_tool(db, "search_photos", args)
    yield {"type": "tool_result", "tool": "search_photos",
           "summary": _summarize("search_photos", result)}
    if isinstance(result, dict):
        yield {"type": "photos", "results": result.get("results", []),
               "total": result.get("total", 0)}
    yield {"type": "answer",
           "text": f"Found {result.get('total', 0)} photo(s) for that search."}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _ask_log_path() -> str:
    """Current append target: one file per day (ask-YYYY-MM-DD.md), rolling to
    ask-YYYY-MM-DD.N.md when the day's file exceeds the size cap. Dir from
    PHOTOSEARCH_ASK_LOG_DIR (default ./ask-logs); cap from
    PHOTOSEARCH_ASK_LOG_MAX_MB (default 5)."""
    d = os.environ.get("PHOTOSEARCH_ASK_LOG_DIR") or "ask-logs"
    os.makedirs(d, exist_ok=True)
    day = time.strftime("%Y-%m-%d")
    try:
        cap = int(float(os.environ.get("PHOTOSEARCH_ASK_LOG_MAX_MB", "5")) * 1_000_000)
    except ValueError:
        cap = 5_000_000
    base = os.path.join(d, f"ask-{day}.md")
    if not os.path.exists(base) or os.path.getsize(base) < cap:
        return base
    n = 2
    while True:
        p = os.path.join(d, f"ask-{day}.{n}.md")
        if not os.path.exists(p) or os.path.getsize(p) < cap:
            return p
        n += 1


def _write_ask_log(session: dict) -> None:
    """Append one entry per query to the day's continuous log (what was typed,
    the tools/args generated, the model's reasoning, the answer). `tail -f` it
    to watch queries live. Best-effort — never breaks the agent."""
    try:
        path = _ask_log_path()
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        started = session.get("started")
        recv = time.strftime("%H:%M:%S", time.localtime(started)) if started else "?"
        L = ["", "=" * 78,
             f"## {ts} · `{session.get('model')}` · {session.get('seconds')}s "
             f"· {session.get('mode', 'agent')}",
             f"**Ask:** {session.get('message', '')}", "",
             f"- received {recv} by /api/ask · history {session.get('history_len', 0)} msg · "
             f"agent model `{session.get('model')}` · {session.get('rounds', 0)} round(s) · "
             f"{len(session.get('tool_calls') or [])} tool call(s)", ""]
        if session.get("error"):
            L.append(f"**error:** {session['error']}")
        if session.get("locked_filters"):
            L.append(f"**Pinned filters:** "
                     f"{json.dumps(session['locked_filters'], ensure_ascii=False)}")
        if session.get("tool_calls"):
            L.append("**Calls (time since request → tool):**")
            for tc in session["tool_calls"]:
                tpfx = f"[+{tc['t']}s] " if tc.get("t") is not None else ""
                rnd = f"R{tc['step']} " if tc.get("step") else ""
                L.append(f"- {tpfx}{rnd}`{tc['name']}({json.dumps(tc.get('args') or {}, ensure_ascii=False)})`"
                         + (f" → {tc['summary']}" if tc.get("summary") else "")
                         + (f"  ⚠ {tc['error']}" if tc.get("error") else ""))
                items = tc.get("items") or []
                if tc["name"] == "rerank_photos":
                    # The interesting part: what the vision model scored each at.
                    for it in items:
                        sc = it.get("rerank_score")
                        L.append(f"    - {sc if sc is not None else '?'}  "
                                 f"{it.get('filename')} (id {it.get('id')})"
                                 + (f" — {it['rerank_reason']}" if it.get("rerank_reason") else ""))
                elif items:
                    # Returned set: filename (id), with the bucket for representatives.
                    parts = []
                    for it in items[:40]:
                        tag = f"[{it['bucket']}] " if it.get("bucket") else ""
                        parts.append(f"{tag}{it.get('filename')} (id {it.get('id')})")
                    more = f" … +{len(items) - 40} more" if len(items) > 40 else ""
                    L.append("    " + "; ".join(parts) + more)
        for i, st in enumerate([s for s in session.get("steps", []) if s.get("reasoning")], 1):
            L += [f"**Reasoning {i}:**", "```", st["reasoning"].strip(), "```"]
        if session.get("answer"):
            L.append(f"**Answer:** {session['answer']}")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Locked filters — the structured Search filter bar, injected as HARD inputs
# ---------------------------------------------------------------------------
#
# The Ask box carries the natural-language intent; the filter bar carries
# constraints the user pinned in the UI. We do NOT post-filter the agent's
# results by them — we feed them INTO every search the agent runs, and enforce
# them server-side (merge into each tool call) so a filter can't be dropped by
# a model that forgot it (the "KODAK PIXPRO WPZ2 returns non-Kodak" bug).

# Tools whose arguments are structured search filters — kept in sync with the
# filter-consuming tools in tools.py. rerank_photos / get_photo / list_* / the
# write tools take ids or lookups, not filters, so they're excluded.
_FILTER_TOOLS = {"search_photos", "representatives", "daily_highlights",
                 "summarize", "group_into_chapters", "daily_scene_breakdown",
                 # consolidated mode: the one tool that subsumes the search family
                 "search"}

# Locked-filter keys (tool-arg names). `people` unions with whatever the model
# inferred; every other key hard-overrides the model's value.
_LOCKED_KEYS = ("people", "location", "date_from", "date_to", "color", "category",
                "visual_tag", "keyword", "min_quality", "min_aesthetic",
                "min_day_aesthetic", "style_tag", "match_source", "camera", "sort")

# Tool-SCOPED locked params: unlike _LOCKED_KEYS (generic filters injected into
# every filter-consuming tool), these only apply where the param exists. `per_day`
# is a daily_highlights knob, so pinning it must NOT leak onto search_photos etc.
# Maps key -> the tools it's injected into.
_SCOPED_LOCKED = {
    # daily_highlights in classic mode; `search` (mode=daily) in consolidated mode.
    # On non-daily search modes the handler simply ignores per_day.
    "per_day": ("daily_highlights", "search"),
}


def _normalize_locked(locked) -> dict:
    """Keep only recognized, non-empty locked keys (generic + tool-scoped)."""
    out: dict = {}
    if not isinstance(locked, dict):
        return out
    for k in _LOCKED_KEYS:
        v = locked.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if isinstance(v, (list, tuple)) and not v:
            continue
        out[k] = v
    for k in _SCOPED_LOCKED:
        v = locked.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        try:
            iv = int(v)
        except (TypeError, ValueError):
            continue
        if iv > 0:
            out[k] = iv
    return out


def _merge_locked(name: str, args: dict, locked: dict) -> dict:
    """Inject the pinned UI filters into a tool call so they're ENFORCED
    regardless of what the model emitted. Generic filters apply to every
    _FILTER_TOOLS call (`people` unions, scalars override); tool-scoped params
    (_SCOPED_LOCKED) apply only to the tools that actually accept them."""
    if not locked:
        return args or {}
    merged = dict(args or {})
    for k, v in locked.items():
        scope = _SCOPED_LOCKED.get(k)
        if scope is not None:
            if name in scope:
                merged[k] = v
            continue
        if name not in _FILTER_TOOLS:
            continue
        if k == "people":
            have = toolmod._coerce_str_list(merged.get("people"))
            add = toolmod._coerce_str_list(v)
            merged["people"] = list(dict.fromkeys(have + add))
        else:
            merged[k] = v
    return merged


_LOCKED_LABELS = {
    "people": "people", "location": "location", "date_from": "on/after",
    "date_to": "on/before", "color": "color", "category": "category",
    "visual_tag": "visual tag", "keyword": "keyword", "min_quality": "min quality",
    "min_aesthetic": "min aesthetic pct",
    "min_day_aesthetic": "min per-day aesthetic pct", "style_tag": "style",
    "match_source": "match source", "camera": "camera", "sort": "sort",
}


def _locked_prompt(locked: dict) -> str:
    """System-prompt paragraph telling the model which filters the UI already
    pinned, so it plans the rest of the request around them instead of guessing
    or contradicting them."""
    if not locked:
        return ""
    out = ""
    # Generic filters — enforced on every search-like tool.
    bits = []
    for k, v in locked.items():
        if k in _SCOPED_LOCKED:
            continue
        val = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v
        bits.append(f"{_LOCKED_LABELS.get(k, k)}={val}")
    if bits:
        out += ("\nACTIVE FILTERS (the user pinned these in the UI; they are "
                "applied AUTOMATICALLY to every search you run and you cannot "
                "override them — do NOT restate them in your tool calls, and do "
                "NOT drop the person/subject to manufacture results — just plan "
                "the REST of the request around them): " + "; ".join(bits) + ".\n")
    # Tool-scoped: per_day only means anything for a day-by-day result.
    if "per_day" in locked:
        out += (f"\nThe user set a per-day cap of {locked['per_day']}: when you "
                f"produce a day-by-day highlight reel (daily_highlights), use "
                f"per_day={locked['per_day']}.\n")
    return out


# ---------------------------------------------------------------------------
# Result grouping — so the UI can render photobook sections, not a flat grid
# ---------------------------------------------------------------------------
#
# The book/curation tools already tag their results with a grouping key
# (chapter / scene / bucket) and return an ordered group summary. `_grouping_for`
# normalizes that into `(group_field, groups)` — `group_field` names the per-hit
# key the UI groups on ('day' is derived from date_taken), `groups` is the
# ordered list of section headers `[{key, label, sublabel}]`.

def _grouping_for(name: str, result) -> tuple[Optional[str], Optional[list]]:
    """Photobook sectioning, keyed off the RESULT SHAPE rather than the tool name
    — so it works identically whether the model called the separate tools or the
    consolidated `search(mode=...)` (whose result is whichever sub-handler ran)."""
    if not isinstance(result, dict):
        return None, None
    if result.get("chapters"):                       # group_into_chapters / mode=chapters
        groups = []
        for c in result["chapters"]:
            df, dt = c.get("date_from"), c.get("date_to")
            span = df if df == dt else f"{df} – {dt}"
            sub = f"{span} · {c.get('photo_count', 0)} photos"
            groups.append({"key": c.get("index"),
                           "label": c.get("title") or f"Chapter {c.get('index')}",
                           "sublabel": sub})
        return "chapter", groups
    if result.get("scenes"):                          # daily_scene_breakdown / mode=scenes
        groups = []
        for s in result["scenes"]:
            st = (s.get("start") or "")[11:16]
            en = (s.get("end") or "")[11:16]
            span = st if st == en else f"{st}–{en}"
            sub = " · ".join(x for x in (span, f"{s.get('photo_count', 0)} photos") if x)
            groups.append({"key": s.get("index"),
                           "label": s.get("place") or f"Scene {s.get('index')}",
                           "sublabel": sub})
        return "scene", groups
    if "day_summary" in result:                       # daily_highlights / mode=daily
        groups = [{"key": d.get("day"), "label": d.get("day"),
                   "sublabel": ", ".join(d.get("places") or [])}
                  for d in result.get("day_summary", [])]
        return "day", groups
    if any(isinstance(h, dict) and h.get("bucket") is not None
           for h in result.get("results", [])):       # representatives / mode=per_bucket
        seen: list = []
        for h in result["results"]:
            b = h.get("bucket")
            if b is not None and b not in seen:
                seen.append(b)
        return "bucket", [{"key": b, "label": str(b)} for b in seen]
    return None, None


def run_agent(
    db: PhotoDB,
    message: str,
    history: Optional[list] = None,
    max_steps: int = _MAX_STEPS,
    should_abort: Optional[Callable[[], bool]] = None,
    locked_filters: Optional[dict] = None,
    reasoning_effort: Optional[str] = None,
    consolidated: Optional[bool] = None,
) -> Iterator[dict]:
    """Drive the tool-calling loop; yield event dicts. See module docstring.

    ``consolidated`` (None = PHOTOSEARCH_CONSOLIDATED_SEARCH) offers one
    ``search(mode=...)`` tool instead of the 5 search-family tools.
    ``reasoning_effort`` overrides PHOTOSEARCH_LLM_REASONING_EFFORT for this run
    ("none" disables a thinking model's reasoning traces; "" = model default).
    """
    if not (message or "").strip():
        yield {"type": "error", "message": "empty message"}
        return

    locked = _normalize_locked(locked_filters)

    effective_effort = _resolve_reasoning_effort(reasoning_effort)
    session = {"message": message, "model": _agent_model(),
               "reasoning_effort": effective_effort or "default", "started": time.time(),
               "mode": "agent", "steps": [], "tool_calls": [], "answer": None,
               "error": None, "history_len": len(history or []), "rounds": 0,
               "locked_filters": locked or None}

    def _set_answer(text):
        session["answer"] = text
        return text

    def _photos_event():
        ev = {"type": "photos", "results": last_photos, "total": last_total}
        if last_group_field and last_groups:
            ev["group_field"] = last_group_field
            ev["groups"] = last_groups
        return ev

    try:
        deadline = time.monotonic() + float(
            os.environ.get("PHOTOSEARCH_AGENT_DEADLINE_S") or _DEFAULT_DEADLINE_S)

        if os.environ.get("PHOTOSEARCH_AGENT_SINGLE_SHOT", "").strip().lower() in (
                "1", "true", "yes", "on"):
            session["mode"] = "single_shot"
            for ev in _run_single_shot(db, message,
                                       timeout=max(5.0, deadline - time.monotonic()),
                                       locked=locked):
                if ev.get("type") == "answer":
                    session["answer"] = ev.get("text")
                if ev.get("type") == "tool_call":
                    session["tool_calls"].append({"name": ev.get("tool"),
                                                  "args": ev.get("arguments")})
                yield ev
            return

        allow_writes = _writes_enabled()
        consolidated = (consolidated if consolidated is not None
                        else toolmod.consolidated_search_enabled())
        tools = toolmod.openai_tools(include_images=False, include_writes=allow_writes,
                                     consolidated=consolidated)
        messages: list = [{"role": "system",
                           "content": _system_prompt(db, allow_writes=allow_writes,
                                                     consolidated=consolidated)
                           + _locked_prompt(locked)}]
        for h in (history or []):
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])})
        messages.append({"role": "user", "content": message})

        last_photos: Optional[list] = None
        last_total = 0
        last_group_field: Optional[str] = None
        last_groups: Optional[list] = None
        made_tool_call = False
        nudges = 0
        summary_nudged = False
        timed_out = False

        for step in range(max_steps):
            if should_abort and should_abort():
                session["error"] = "cancelled"
                yield {"type": "error", "message": "cancelled"}
                return
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                timed_out = True
                break
            session["rounds"] = step + 1
            yield {"type": "step", "n": step + 1, "max": max_steps,
                   "elapsed": round(time.time() - session["started"], 1)}
            try:
                reply = _chat(messages, tools, timeout=min(_HTTP_TIMEOUT_S, remaining),
                              reasoning_effort=reasoning_effort)
            except Exception as exc:
                # First step with no progress → try the single-shot fallback
                # once; the model likely can't tool-call against this endpoint.
                if step == 0:
                    session["mode"] = "single_shot_fallback"
                    yield {"type": "tool_result", "tool": "_fallback",
                           "summary": "tool-calling failed; trying single-shot"}
                    for ev in _run_single_shot(db, message,
                                               timeout=max(5.0, deadline - time.monotonic()),
                                               locked=locked, reasoning_effort=reasoning_effort):
                        if ev.get("type") == "answer":
                            session["answer"] = ev.get("text")
                        yield ev
                    return
                session["error"] = f"LLM error: {exc}"
                yield {"type": "error", "message": f"LLM error: {exc}"}
                return

            if reply.get("reasoning") or reply.get("content"):
                session["steps"].append({"reasoning": reply.get("reasoning"),
                                         "content": reply.get("content")})

            calls = reply.get("tool_calls") or []
            if calls:
                made_tool_call = True
                messages.append(_assistant_tool_msg(reply))
                for call in calls:
                    name = call.get("name") or ""
                    # Enforce the pinned UI filters: merge them into the args the
                    # tool actually runs with (and show the enforced set in the
                    # narration, so a dropped filter is visible).
                    eff_args = _merge_locked(name, call.get("arguments", {}), locked)
                    yield {"type": "tool_call", "tool": name, "arguments": eff_args,
                           "step": step + 1, "elapsed": round(time.time() - session["started"], 1)}
                    # A vision rerank is slow; grant extra wall-clock so the turn
                    # that reads its scores isn't killed by the deadline.
                    if name == "rerank_photos":
                        deadline = max(deadline,
                                       time.monotonic() + _RERANK_DEADLINE_EXTEND_S)
                    try:
                        result = toolmod.call_tool(db, name, eff_args)
                    except KeyError:
                        result = {"error": f"unknown tool: {name}"}
                    except Exception as exc:
                        result = {"error": str(exc)}
                    if (name in ("search_photos", "representatives", "rerank_photos",
                                 "daily_highlights", "group_into_chapters",
                                 "daily_scene_breakdown", "suggest_layout", "search")
                            and isinstance(result, dict) and "results" in result):
                        last_photos = result.get("results", [])
                        last_total = result.get("total", result.get("returned",
                                                                    len(last_photos)))
                        last_group_field, last_groups = _grouping_for(name, result)
                    summ = _summarize(name, result)
                    rec = {"name": name, "args": eff_args, "summary": summ,
                           "step": step + 1, "t": round(time.time() - session["started"], 1)}
                    if isinstance(result, dict) and isinstance(result.get("results"), list):
                        hits = [h for h in result["results"] if isinstance(h, dict)]
                        rec["items"] = [{"id": h.get("id"), "filename": h.get("filename"),
                                         "bucket": h.get("bucket"),
                                         "rerank_score": h.get("rerank_score"),
                                         "rerank_reason": h.get("rerank_reason")}
                                        for h in hits]
                    elif isinstance(result, dict) and "error" in result:
                        rec["error"] = result["error"]
                    session["tool_calls"].append(rec)
                    yield {"type": "tool_result", "tool": name, "summary": summ,
                           "step": step + 1, "elapsed": round(time.time() - session["started"], 1)}
                    messages.append(_tool_result_msg(call, result))
                continue

            # No tool calls → the model is answering (or stalled).
            content = (reply.get("content") or "").strip()

            # Empty turn with no results yet — a thinking-mode model that
            # grounded then handed back an empty assistant turn instead of
            # searching. Nudge it to search rather than silently returning
            # "Found 0". Bounded by _MAX_NUDGES; the empty turn is dropped.
            if not content and last_photos is None and nudges < _MAX_NUDGES:
                nudges += 1
                yield {"type": "tool_result", "tool": "_nudge",
                       "summary": "empty model turn — nudging it to search"}
                messages.append({"role": "user", "content":
                    "You haven't returned any photos yet. Call the search tool "
                    "now with the appropriate filters, then give a brief answer. "
                    "Do not stop until you have searched."})
                continue

            # Results in hand but an empty final turn — instead of a terse
            # "Found N", nudge once for the interpretation/explanation.
            if not content and last_photos is not None and not summary_nudged:
                summary_nudged = True
                messages.append({"role": "user", "content":
                    "Now write the 1-3 sentence answer: explain how you "
                    "interpreted the request (which filters/sort you used and "
                    "why) and what these photos show. Do not list ids."})
                continue

            if not made_tool_call and not content:
                # Nothing useful and never called a tool — fall back.
                session["mode"] = "single_shot_fallback"
                for ev in _run_single_shot(db, message,
                                           timeout=max(5.0, deadline - time.monotonic())):
                    if ev.get("type") == "answer":
                        session["answer"] = ev.get("text")
                    yield ev
                return
            if last_photos is not None:
                yield _photos_event()
            yield {"type": "answer", "text": _set_answer(content or f"Found {last_total} photo(s).")}
            return

        # Hit the step cap or the time budget — emit whatever we have.
        if last_photos is not None:
            yield _photos_event()
        lead = ("Stopped — this query hit the time budget. "
                if timed_out else "Stopped after %d steps. " % max_steps)
        if timed_out and last_photos is None:
            tail = ("That request was too complex to answer in time — try a "
                    "simpler one (e.g. one period or one person at a time).")
        elif last_photos is not None:
            tail = f"Best result set: {last_total} photo(s)."
        else:
            tail = "No results yet — try rephrasing."
        yield {"type": "answer", "text": _set_answer(lead + tail)}
    finally:
        session["seconds"] = round(time.time() - session["started"], 1)
        _write_ask_log(session)

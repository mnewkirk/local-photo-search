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


def _system_prompt(db=None) -> str:
    base = (
        "You are a photo-library search assistant. The user describes what they "
        "want in natural language; find the matching photos with the tools, then "
        "give a one- or two-sentence answer.\n\n"
        "You are given LIBRARY FACTS below (people, places, tags, date span), so "
        "usually go STRAIGHT to search_photos — you do NOT need get_library_overview "
        "or the list_* tools. Use list_* only to look up something not in the facts.\n\n"
        "Rules:\n"
        "- Fill ONLY the filters you actually inferred; OMIT every other field. "
        "Never send empty strings/arrays or a filter 'just in case'.\n"
        "- Dates: set date_from/date_to only when the request implies a time range, "
        "computed from today's date. NEVER filter by today's date for a request "
        "that isn't about today.\n"
        "- 'best'/'top'/'favorite'/'good' = rank by quality: sort='quality_desc' "
        "(optionally min_quality~6). Do NOT put those words in `query`.\n"
        "- EXCLUDE with `query`: a leading '-' or 'no <thing>', e.g. "
        "query='landscape -people'. Don't express exclusion via structured filters.\n"
        "- Resolve group terms from USER NOTES / the people list: 'the kids', "
        "'the family', 'everyone' -> the specific registered names as a `people` "
        "list (AND-intersected).\n"
        "- Read `total`: if 0, relax your most specific filter and retry (drop "
        "min_quality, widen dates, or move a term into `query`); if huge, add a "
        "filter. Iterate once or twice before answering.\n"
        "- For 'which year / how many / when / who / how often' questions, use "
        "summarize(filters, group_by=year|month|location|person) — it COUNTS by "
        "a dimension instead of returning photos. For multi-step questions like "
        "'which year were we in both X and Y', summarize each by year, intersect "
        "the years yourself, then search_photos for that year.\n"
        "- For 'one/N per year' / 'best of each year' / 'a few from each trip' "
        "(a representative SPREAD, not a flat list), use representatives(filters, "
        "bucket=year|month|location|person, n). search_photos CANNOT do "
        "per-bucket selection. E.g. 'best photo of Matt, one per year, last 10 "
        "years' → representatives(people=['Matt'], bucket='year', n=1, "
        "date_from=<10 years ago>).\n"
        "- Then stop and write a 1-3 sentence answer that EXPLAINS how you "
        "interpreted the request — which filters and sort you used and why "
        "(e.g. \"'best' so I sorted by quality; 'the kids' = Calvin and Ellie\") "
        "— and what the photos show. Never list photo ids; the UI shows them.\n\n"
        "Examples (plan straight to search_photos using the facts):\n"
        "  'photos of Calvin' -> people=['Calvin']\n"
        "  'Nicole and Matt together' -> people=['Nicole','Matt']\n"
        "  'best beach photos last summer' -> query='beach', date_from=<jun1 last yr>, "
        "date_to=<aug31 last yr>, sort='quality_desc'\n"
        "  'landscapes with no people' -> query='landscape -people'\n"
        "  'most moody shots' -> visual_tag='moody', sort='quality_desc'\n"
        "  'the kids playing soccer' -> people=<the kids>, category='soccer'\n"
    )
    try:
        ctx = _library_context(db) if db is not None else ""
    except Exception:
        ctx = ""
    hints = os.environ.get("PHOTOSEARCH_AGENT_HINTS", "").strip()
    out = base
    if ctx:
        out += "\nLIBRARY FACTS:\n" + ctx + "\n"
    if hints:
        out += "\nUSER NOTES:\n" + hints + "\n"
    out += f"\nToday's date is {date.today().isoformat()}."
    return out


# ---------------------------------------------------------------------------
# Tool-calling chat client (OpenAI-compatible primary, Ollama fallback)
# ---------------------------------------------------------------------------

def _chat(messages: list, tools: Optional[list], temperature: float = 0.0) -> dict:
    """One chat round-trip. Returns a normalized dict:
        {"content": str|None, "tool_calls": [{"id", "name", "arguments"}]}

    Routes to the OpenAI-compatible endpoint when PHOTOSEARCH_TEXT_LLM_URL is
    set (the project's local LM Studio path), else to Ollama.
    """
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if base:
        return _chat_openai(base, messages, tools, temperature)
    return _chat_ollama(messages, tools, temperature)


def _chat_openai(base_url: str, messages: list, tools: Optional[list],
                 temperature: float) -> dict:
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body: dict = {
        "model": _agent_model(),
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S)
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
    return {"content": msg.get("content"), "tool_calls": calls}


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


def _tool_result_msg(call: dict, result) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call.get("id") or f"call_{call['name']}",
        "name": call["name"],
        "content": json.dumps(result, default=str),
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


def _run_single_shot(db: PhotoDB, message: str) -> Iterator[dict]:
    reply = _chat(
        [{"role": "system", "content": _SINGLE_SHOT_INSTRUCTION},
         {"role": "user", "content": message}],
        tools=None,
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

def run_agent(
    db: PhotoDB,
    message: str,
    history: Optional[list] = None,
    max_steps: int = _MAX_STEPS,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Iterator[dict]:
    """Drive the tool-calling loop; yield event dicts. See module docstring."""
    if not (message or "").strip():
        yield {"type": "error", "message": "empty message"}
        return

    if os.environ.get("PHOTOSEARCH_AGENT_SINGLE_SHOT", "").strip().lower() in (
            "1", "true", "yes", "on"):
        yield from _run_single_shot(db, message)
        return

    tools = toolmod.openai_tools(include_images=False)
    messages: list = [{"role": "system", "content": _system_prompt(db)}]
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])})
    messages.append({"role": "user", "content": message})

    last_photos: Optional[list] = None
    last_total = 0
    made_tool_call = False
    nudges = 0
    summary_nudged = False

    for step in range(max_steps):
        if should_abort and should_abort():
            yield {"type": "error", "message": "cancelled"}
            return
        try:
            reply = _chat(messages, tools)
        except Exception as exc:
            # First step with no progress → try the single-shot fallback once;
            # the model likely can't tool-call against this endpoint.
            if step == 0:
                yield {"type": "tool_result", "tool": "_fallback",
                       "summary": "tool-calling failed; trying single-shot"}
                yield from _run_single_shot(db, message)
                return
            yield {"type": "error", "message": f"LLM error: {exc}"}
            return

        calls = reply.get("tool_calls") or []
        if calls:
            made_tool_call = True
            messages.append(_assistant_tool_msg(reply))
            for call in calls:
                name = call.get("name") or ""
                yield {"type": "tool_call", "tool": name, "arguments": call.get("arguments", {})}
                try:
                    result = toolmod.call_tool(db, name, call.get("arguments", {}))
                except KeyError:
                    result = {"error": f"unknown tool: {name}"}
                except Exception as exc:
                    result = {"error": str(exc)}
                if (name in ("search_photos", "representatives")
                        and isinstance(result, dict) and "results" in result):
                    last_photos = result.get("results", [])
                    last_total = result.get("total", len(last_photos))
                yield {"type": "tool_result", "tool": name, "summary": _summarize(name, result)}
                messages.append(_tool_result_msg(call, result))
            continue

        # No tool calls → the model is answering (or stalled).
        content = (reply.get("content") or "").strip()

        # Empty turn with no results yet — a thinking-mode model that grounded
        # then handed back an empty assistant turn instead of searching. Nudge
        # it to actually call search_photos rather than silently returning
        # "Found 0". Bounded by _MAX_NUDGES so we can't loop forever; the empty
        # turn is dropped (not appended) and a user nudge takes its place.
        if not content and last_photos is None and nudges < _MAX_NUDGES:
            nudges += 1
            yield {"type": "tool_result", "tool": "_nudge",
                   "summary": "empty model turn — nudging it to search"}
            messages.append({"role": "user", "content":
                "You haven't returned any photos yet. Call the search_photos "
                "tool now with the appropriate filters, then give a brief "
                "answer. Do not stop until you have searched."})
            continue

        # Results in hand but an empty final turn — instead of falling back to a
        # terse "Found N", nudge once for the interpretation/explanation.
        if not content and last_photos is not None and not summary_nudged:
            summary_nudged = True
            messages.append({"role": "user", "content":
                "Now write the 1-3 sentence answer: explain how you interpreted "
                "the request (which filters/sort you used and why) and what "
                "these photos show. Do not list ids."})
            continue

        if not made_tool_call and not content:
            # Model produced nothing useful and never called a tool — fall back.
            yield from _run_single_shot(db, message)
            return
        if last_photos is not None:
            yield {"type": "photos", "results": last_photos, "total": last_total}
        yield {"type": "answer", "text": content or f"Found {last_total} photo(s)."}
        return

    # Hit the step cap — emit whatever we have.
    if last_photos is not None:
        yield {"type": "photos", "results": last_photos, "total": last_total}
    yield {"type": "answer",
           "text": f"Stopped after {max_steps} steps. "
                   + (f"Best result set: {last_total} photo(s)." if last_photos is not None
                      else "No results yet — try rephrasing.")}

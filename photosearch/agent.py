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
from datetime import date
from typing import Callable, Iterator, Optional

from .db import PhotoDB
from . import tools as toolmod

# Bound the loop so a confused model can't spin forever. Each step is one LLM
# round-trip (which may issue several tool calls).
_MAX_STEPS = 6
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


def _system_prompt() -> str:
    return (
        "You are a photo-library search assistant. The user describes what "
        "they're looking for in natural language; your job is to find the "
        "matching photos using the provided tools, then give a one- or "
        "two-sentence answer.\n\n"
        "Rules:\n"
        "- Call get_library_overview first to learn what's available and the "
        "date range.\n"
        "- Before filtering by a person, place, or tag, VALIDATE it exists: "
        "list_people for names, list_places for locations, list_vocab for "
        "categories/visual_tags/keywords. Never pass a name to search_photos "
        "that list_people didn't return.\n"
        "- Use search_photos to find photos. Fill only the filters you "
        "inferred; omit the rest. Read the returned `total`: if it's far more "
        "or fewer than expected, adjust the filters and search again.\n"
        "- Prefer the free-text `query` for visual content (sunset, cake) and "
        "structured filters (people, location, dates, category) for the rest.\n"
        "- When you have a good result set, stop calling tools and answer "
        "briefly. Do not list photo ids in your answer — the UI shows the "
        "photos. Mention the count and how you narrowed it.\n"
        f"Today's date is {date.today().isoformat()}."
    )


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
    messages: list = [{"role": "system", "content": _system_prompt()}]
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])})
    messages.append({"role": "user", "content": message})

    last_photos: Optional[list] = None
    last_total = 0
    made_tool_call = False

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
                if name == "search_photos" and isinstance(result, dict) and "results" in result:
                    last_photos = result.get("results", [])
                    last_total = result.get("total", len(last_photos))
                yield {"type": "tool_result", "tool": name, "summary": _summarize(name, result)}
                messages.append(_tool_result_msg(call, result))
            continue

        # No tool calls → the model is answering (or can't tool-call).
        content = (reply.get("content") or "").strip()
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

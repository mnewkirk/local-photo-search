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
_DEFAULT_DEADLINE_S = 90.0  # overall wall-clock budget per Ask (env override)


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


_WRITE_GUIDANCE = (
    "\n\nWRITES (you can edit the library): set_photo_location, set_photo_tags, "
    "and add_to_collection mutate the library. STRICT protocol:\n"
    "- They act ONLY on an explicit photo_ids list — pass the exact ids from a "
    "search you already ran and the user saw. NEVER make up ids or write to a set "
    "the user hasn't seen.\n"
    "- ALWAYS preview first: call with confirm=false (the default) to show how "
    "many photos would change and a before→after sample. Report that to the user "
    "and ask them to confirm.\n"
    "- Only call again with confirm=true AFTER the user explicitly says yes in the "
    "conversation. Never pass confirm=true on your own initiative.\n"
    "- If the preview says confirm_large is needed (a big batch), tell the user "
    "the count and get a second explicit go-ahead before adding confirm_large=true.\n"
)


def _system_prompt(db=None, allow_writes: bool = False) -> str:
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
        "- 'best'/'top'/'favorite'/'good' = rank by quality: sort='quality_desc'. "
        "Do NOT put those words in `query`, and do NOT add min_quality — a hard "
        "quality floor can exclude everything (many good photos score ~5). Only "
        "set min_quality if the user gives an explicit bar ('quality above 7').\n"
        "- EXCLUDE with `query`: a leading '-' or 'no <thing>', e.g. "
        "query='landscape -people'. Don't express exclusion via structured filters.\n"
        "- Resolve group terms from USER NOTES / the people list: 'the kids', "
        "'the family', 'everyone' -> the specific registered names as a `people` "
        "list (AND-intersected).\n"
        "- Read `total`: if 0, relax a SECONDARY filter and retry (drop "
        "min_quality, widen dates, or move a term into `query`); if huge, add a "
        "filter. Iterate once or twice before answering. But NEVER drop the "
        "person/subject the user named to manufacture results — if a search that "
        "INCLUDES that person returns 0, answer that no matching photos were found "
        "(say what you tried); do NOT return photos without them.\n"
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
        "- 'X is the foreground/primary subject' / 'X featured prominently': "
        "for a flat 'best photos of X (foreground)' in ONE scope (a year, a "
        "place) use search_photos(people=['X'], sort='subject', ...); for a "
        "SPREAD ('one/N per year where X is foreground') use "
        "representatives(people=['X'], rank_by='subject', bucket='year', n=...). "
        "Both rank by the person's face size in frame — prefer them over "
        "rerank_photos for subject prominence.\n"
        "- For VISUAL precision a vision model must judge — 'the one where X is "
        "doing Y', a specific ACTIVITY/sport ('playing soccer', 'on a bike'), or "
        "'the sharpest/best-composed' — get candidates first (search_photos with "
        "sort='quality_desc', or representatives; fetch ~15-24), then "
        "rerank_photos(photo_ids=<those ids>, criteria=<the visual thing>). "
        "IMPORTANT for activities: category/tag matches include false positives "
        "(a beach photo mis-tagged 'soccer'), so for 'top N photos of X playing "
        "<sport>' you MUST rerank to verify the activity is actually happening — "
        "don't trust category alone. It LOOKS at each image and re-sorts; it's "
        "slow, so keep the set <=24.\n"
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
        "  'top 3 photos of Calvin playing soccer' -> search_photos(people=['Calvin'], "
        "category='soccer', sort='quality_desc') THEN rerank_photos(photo_ids="
        "<the returned ids>, criteria='a child actively playing soccer on a field', "
        "top_n=3) — rerank (category tags have false positives) and pass top_n to "
        "honor the requested count and drop low-scoring non-matches\n"
    )
    try:
        ctx = _library_context(db) if db is not None else ""
    except Exception:
        ctx = ""
    hints = os.environ.get("PHOTOSEARCH_AGENT_HINTS", "").strip()
    out = base + (_WRITE_GUIDANCE if allow_writes else "")
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
          timeout: float = _HTTP_TIMEOUT_S) -> dict:
    """One chat round-trip. Returns a normalized dict:
        {"content": str|None, "tool_calls": [{"id", "name", "arguments"}]}

    Routes to the OpenAI-compatible endpoint when PHOTOSEARCH_TEXT_LLM_URL is
    set (the project's local LM Studio path), else to Ollama. `timeout` bounds
    the call so a single rambling generation can't hang the whole Ask.
    """
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    if base:
        return _chat_openai(base, messages, tools, temperature, timeout)
    return _chat_ollama(messages, tools, temperature)


def _chat_openai(base_url: str, messages: list, tools: Optional[list],
                 temperature: float, timeout: float = _HTTP_TIMEOUT_S) -> dict:
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
                     timeout: float = _HTTP_TIMEOUT_S) -> Iterator[dict]:
    reply = _chat(
        [{"role": "system", "content": _SINGLE_SHOT_INSTRUCTION},
         {"role": "user", "content": message}],
        tools=None, timeout=timeout,
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

    session = {"message": message, "model": _agent_model(), "started": time.time(),
               "mode": "agent", "steps": [], "tool_calls": [], "answer": None,
               "error": None, "history_len": len(history or []), "rounds": 0}

    def _set_answer(text):
        session["answer"] = text
        return text

    try:
        deadline = time.monotonic() + float(
            os.environ.get("PHOTOSEARCH_AGENT_DEADLINE_S") or _DEFAULT_DEADLINE_S)

        if os.environ.get("PHOTOSEARCH_AGENT_SINGLE_SHOT", "").strip().lower() in (
                "1", "true", "yes", "on"):
            session["mode"] = "single_shot"
            for ev in _run_single_shot(db, message, timeout=max(5.0, deadline - time.monotonic())):
                if ev.get("type") == "answer":
                    session["answer"] = ev.get("text")
                if ev.get("type") == "tool_call":
                    session["tool_calls"].append({"name": ev.get("tool"),
                                                  "args": ev.get("arguments")})
                yield ev
            return

        allow_writes = _writes_enabled()
        tools = toolmod.openai_tools(include_images=False, include_writes=allow_writes)
        messages: list = [{"role": "system",
                           "content": _system_prompt(db, allow_writes=allow_writes)}]
        for h in (history or []):
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": str(h["content"])})
        messages.append({"role": "user", "content": message})

        last_photos: Optional[list] = None
        last_total = 0
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
                reply = _chat(messages, tools, timeout=min(_HTTP_TIMEOUT_S, remaining))
            except Exception as exc:
                # First step with no progress → try the single-shot fallback
                # once; the model likely can't tool-call against this endpoint.
                if step == 0:
                    session["mode"] = "single_shot_fallback"
                    yield {"type": "tool_result", "tool": "_fallback",
                           "summary": "tool-calling failed; trying single-shot"}
                    for ev in _run_single_shot(db, message,
                                               timeout=max(5.0, deadline - time.monotonic())):
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
                    yield {"type": "tool_call", "tool": name, "arguments": call.get("arguments", {}),
                           "step": step + 1, "elapsed": round(time.time() - session["started"], 1)}
                    try:
                        result = toolmod.call_tool(db, name, call.get("arguments", {}))
                    except KeyError:
                        result = {"error": f"unknown tool: {name}"}
                    except Exception as exc:
                        result = {"error": str(exc)}
                    if (name in ("search_photos", "representatives", "rerank_photos")
                            and isinstance(result, dict) and "results" in result):
                        last_photos = result.get("results", [])
                        last_total = result.get("total", len(last_photos))
                    summ = _summarize(name, result)
                    rec = {"name": name, "args": call.get("arguments", {}), "summary": summ,
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
                    "You haven't returned any photos yet. Call the search_photos "
                    "tool now with the appropriate filters, then give a brief "
                    "answer. Do not stop until you have searched."})
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
                yield {"type": "photos", "results": last_photos, "total": last_total}
            yield {"type": "answer", "text": _set_answer(content or f"Found {last_total} photo(s).")}
            return

        # Hit the step cap or the time budget — emit whatever we have.
        if last_photos is not None:
            yield {"type": "photos", "results": last_photos, "total": last_total}
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

"""Grading for the Ask-agent eval: pure functions over a stored run row.

A run row (written by runner.py) holds the agent's trace, its final photos,
its answer and per-round LLM usage. Grading happens here, in the harness's own
tree, never in the tree under test, so `regrade` can re-score stored rows after
a check changes without calling a model.

Items are data (asks.jsonl), not lambdas. Each item carries:

  checks         trace checks: [{name, tool, op, arg?, value?, min?, n?}]
  answer_expect  regexes that must all match the final answer
  answer_any     groups of regexes; each group needs one match
  answer_reject  regexes that must not match
  answer_sql     [{name, sql}] — a single number from the replica DB that must
                 appear in the answer (with or without thousands separators)
  results_min    the final `photos` event holds at least this many hits
  results_sql    [{name, sql, expect}] — SQL with an `{ids}` placeholder over
                 the returned photo ids; the scalar it returns must equal
                 `expect` (usually a count of rows that violate the ask)

A tool call to the consolidated `search(mode=...)` tool is canonicalised to the
search-family tool that mode routes to, so one item grades the same on either
tool surface. That keeps "offer one search tool" a legitimate diet move.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any, Callable, Optional

# tools._MODE_TO_TOOL, copied so grading never imports the tree under test.
MODE_TO_TOOL = {"flat": "search_photos", "daily": "daily_highlights",
                "per_bucket": "representatives", "chapters": "group_into_chapters",
                "scenes": "daily_scene_breakdown"}

GROUNDING_TOOLS = {"get_library_overview", "list_people", "list_places", "list_vocab"}
WRITE_TOOLS = {"set_photo_location", "set_photo_tags", "add_to_collection"}

OPS = ("called", "not_called", "count_ge", "count_le", "set_eq", "set_has",
       "set_disjoint", "eq", "in", "contains", "contains_any", "not_contains",
       "absent", "present", "date_between", "max_calls", "first_tool")


# ------------------------------------------------------------------ dates --

def date_tokens(today: Optional[dt.date] = None) -> dict:
    """The `{token}` values an item may use in a date. Computed from the day the
    run happened (stored on the row), so a regrade next month still scores the
    run against the dates it was asked on."""
    t = today or dt.date.today()
    first = t.replace(day=1)
    last_month_end = first - dt.timedelta(days=1)
    return {
        "today": t.isoformat(),
        "yesterday": (t - dt.timedelta(days=1)).isoformat(),
        "week_ago": (t - dt.timedelta(days=7)).isoformat(),
        "year": str(t.year),
        "last_year": str(t.year - 1),
        # "last summer" = this year's once summer is over, else last year's.
        "last_summer": str(t.year if t.month >= 9 else t.year - 1),
        "month_start": first.isoformat(),
        "last_month_start": last_month_end.replace(day=1).isoformat(),
        "last_month_end": last_month_end.isoformat(),
    }


def fill(value: Any, tokens: dict) -> Any:
    if isinstance(value, str):
        return re.sub(r"\{(\w+)\}", lambda m: tokens.get(m.group(1), m.group(0)), value)
    if isinstance(value, list):
        return [fill(v, tokens) for v in value]
    return value


# ------------------------------------------------------------------ trace --

def canonical_calls(trace: list) -> list[dict]:
    """[{tool, args}] for every real tool call, `search(mode=…)` mapped to its
    family tool. The single-shot `_fallback` marker is not a call."""
    out = []
    for step in trace or []:
        if step.get("k") != "call":
            continue
        tool, args = step.get("tool"), dict(step.get("args") or {})
        if tool == "_fallback":
            continue
        if tool == "search":
            mode = str(args.pop("mode", "flat") or "flat").strip().lower()
            tool = MODE_TO_TOOL.get(mode, "search_photos")
        out.append({"tool": tool, "args": args})
    return out


def _tools(spec) -> set:
    t = spec.get("tool")
    if t is None:
        return set()
    return set(t) if isinstance(t, list) else {t}


def _listify(v) -> list:
    """A list-typed arg as a list. Tolerates the stringified JSON array (and the
    comma string) small models emit, as tools._coerce_str_list does."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("["):
            try:
                v = json.loads(s)
            except ValueError:
                v = [s]
        else:
            v = [p for p in re.split(r"\s*,\s*", s) if p]
    if not isinstance(v, list):
        v = [v]
    return [str(x).strip() for x in v if str(x).strip()]


def _lowset(v) -> set:
    return {x.lower() for x in _listify(v)}


def _text(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v).lower()
    return str(v if v is not None else "").lower()


def _empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def _date10(v) -> str:
    return v[:10] if isinstance(v, str) and len(v) >= 10 else ""


def _arg_pred(op: str, value) -> Callable[[Any], bool]:
    if op == "set_eq":
        want = {str(x).lower() for x in value}
        return lambda v: _lowset(v) == want
    if op == "set_has":
        want = {str(x).lower() for x in value}
        return lambda v: want <= _lowset(v)
    if op == "set_disjoint":
        want = {str(x).lower() for x in value}
        return lambda v: not (want & _lowset(v))
    if op == "eq":
        return lambda v: _text(v) == str(value).lower()
    if op == "in":
        opts = {str(x).lower() for x in value}
        return lambda v: _text(v) in opts
    if op == "contains":
        return lambda v: str(value).lower() in _text(v)
    if op == "contains_any":
        opts = [str(x).lower() for x in value]
        return lambda v: any(o in _text(v) for o in opts)
    if op == "present":
        return lambda v: not _empty(v)
    if op == "date_between":
        lo, hi = value
        return lambda v: bool(_date10(v)) and lo <= _date10(v) <= hi
    raise ValueError(f"no arg predicate for op {op!r}")


def _get(args: dict, arg):
    """One arg's value, or for a list of arg names every value they hold
    (so "birthday in query, category or keyword" is one check)."""
    if not isinstance(arg, list):
        return args.get(arg)
    vals: list = []
    for a in arg:
        v = args.get(a)
        if not _empty(v):
            vals += _listify(v) if isinstance(v, (list, str)) else [v]
    return vals or None


def check_trace(spec: dict, calls: list[dict], tokens: dict) -> bool:
    """One trace check. `any`-style ops pass when some matching call satisfies
    them. The negative ops (`not_contains`, `absent`) need at least one call to
    the tool, or a model that called nothing would pass them for free."""
    op = spec["op"]
    tools = _tools(spec)
    mine = [c for c in calls if not tools or c["tool"] in tools]
    arg = spec.get("arg")
    value = fill(spec.get("value"), tokens)

    if op == "called":
        return len(mine) >= int(spec.get("min", 1))
    if op == "not_called":
        return not mine
    if op == "max_calls":
        return len(mine) <= int(spec["value"])
    if op == "first_tool":
        real = [c for c in calls if c["tool"] not in GROUNDING_TOOLS]
        return bool(real) and real[0]["tool"] in tools
    if op in ("count_ge", "count_le"):
        n = len([c for c in mine
                 if arg is None or _arg_pred("eq", value)(_get(c["args"], arg))])
        return n >= int(spec["n"]) if op == "count_ge" else n <= int(spec["n"])
    if op == "not_contains":
        pred = _arg_pred("contains_any", value if isinstance(value, list) else [value])
        return bool(mine) and not any(pred(_get(c["args"], arg)) for c in mine)
    if op == "absent":
        return bool(mine) and all(_empty(_get(c["args"], arg)) for c in mine)
    pred = _arg_pred(op, value)
    return any(pred(_get(c["args"], arg)) for c in mine)


# ----------------------------------------------------------------- answer --

def _number_patterns(n) -> list[str]:
    try:
        i = int(float(n))
    except (TypeError, ValueError):
        return [re.escape(str(n))]
    return [rf"(?<![\d,.]){i}(?![\d])", rf"(?<![\d,.]){i:,}(?![\d])"]


def grade(item: dict, row: dict, sql: Optional[Callable[[str], Any]] = None,
          *, fail_on_fallback: bool = True) -> tuple[bool, dict]:
    """Score one stored run. Returns (pass, detail); each detail check is
    {name, ok, kind} with kind trace | answer | results. `sql(query) -> scalar` runs
    against the replica DB read-only; with no `sql` the DB checks are recorded
    as skipped and do not count against the run."""
    tokens = date_tokens(_run_day(row))
    calls = canonical_calls(row.get("trace"))
    results: list = []

    for spec in item.get("checks", []):
        results.append((spec.get("name") or f"{spec['op']} {spec.get('tool')}",
                        bool(check_trace(spec, calls, tokens)), "trace"))

    answer = row.get("answer") or ""
    for p in item.get("answer_expect", []):
        results.append((f"answer ~ /{p}/", bool(re.search(fill(p, tokens), answer, re.I)),
                        "answer"))
    for group in item.get("answer_any", []):
        results.append(("answer ~ any(" + " | ".join(group) + ")",
                        any(re.search(fill(p, tokens), answer, re.I) for p in group), "answer"))
    for p in item.get("answer_reject", []):
        results.append((f"answer !~ /{p}/", not re.search(fill(p, tokens), answer, re.I),
                        "answer"))

    skipped = []
    for spec in item.get("answer_sql", []):
        if sql is None:
            skipped.append(spec["name"])
            continue
        n = sql(fill(spec["sql"], tokens))
        results.append((f"{spec['name']} (= {n})",
                        any(re.search(p, answer) for p in _number_patterns(n)), "answer"))

    ids = [i for i in (row.get("result_ids") or []) if isinstance(i, int)]
    if "results_min" in item:
        results.append((f">= {item['results_min']} photos returned",
                        len(ids) >= int(item["results_min"]), "results"))
    for spec in item.get("results_sql", []):
        if sql is None:
            skipped.append(spec["name"])
            continue
        if not ids:
            results.append((spec["name"] + " (no photos)", False, "results"))
            continue
        got = sql(fill(spec["sql"], tokens).replace("{ids}", ",".join(map(str, ids))))
        results.append((f"{spec['name']} (= {got})", got == spec.get("expect", 0), "results"))

    fallback = bool(row.get("single_shot"))
    error = row.get("error")
    ok = bool(results) and all(r[1] for r in results) and not error \
        and not (fail_on_fallback and fallback)
    return ok, {
        "checks": [{"name": n, "ok": r, "kind": k} for n, r, k in results],
        "checks_passed": sum(1 for _, r, _k in results if r),
        "checks_total": len(results),
        "skipped": skipped,
        "fallback": fallback,
        "error": error,
        "grounding_calls": len([c for c in calls if c["tool"] in GROUNDING_TOOLS]),
        "tool_calls": len(calls),
    }


def _run_day(row: dict) -> Optional[dt.date]:
    s = str(row.get("ran_at") or "")[:10]
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


# ------------------------------------------------------------- validation --

def validate_item(item: dict) -> list[str]:
    """Problems with an item's shape (unknown op, missing fields). Empty = ok."""
    errs = []
    if not item.get("id") or not item.get("prompt") or not item.get("suite"):
        errs.append("needs id, suite and prompt")
    for spec in item.get("checks", []):
        op = spec.get("op")
        if op not in OPS:
            errs.append(f"unknown op {op!r}")
            continue
        if op not in ("called", "not_called", "max_calls", "first_tool", "count_ge",
                      "count_le") and not spec.get("arg"):
            errs.append(f"op {op} needs arg")
        if op in ("count_ge", "count_le") and "n" not in spec:
            errs.append(f"op {op} needs n")
        if op == "max_calls" and "value" not in spec:
            errs.append("max_calls needs value")
        if op == "date_between" and not (isinstance(spec.get("value"), list)
                                         and len(spec["value"]) == 2):
            errs.append("date_between needs value [lo, hi]")
    for spec in item.get("results_sql", []):
        if "{ids}" not in spec.get("sql", ""):
            errs.append(f"results_sql {spec.get('name')} has no {{ids}}")
    for p in (item.get("answer_expect", []) + item.get("answer_reject", [])
              + [p for g in item.get("answer_any", []) for p in g]):
        try:
            re.compile(p)
        except re.error as e:
            errs.append(f"bad regex {p!r}: {e}")
    if not (item.get("checks") or item.get("answer_expect") or item.get("answer_any")
            or item.get("answer_sql") or item.get("results_sql") or "results_min" in item):
        errs.append("item grades nothing")
    return errs

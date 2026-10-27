"""Prompt mutants: named deletions from the Ask agent's prompt.

Mutation testing for an eval. Delete one rule, example, block or tool
description, rerun the items, and see whether any item fails. A mutant no item
"kills" means either the eval has no item that depends on that text (a coverage
gap) or the text does nothing (a candidate for the trim). The trim itself is a
sequence of mutations, so the mutation score is the closest thing an eval has
to test coverage.

A mutant is applied to the request the agent actually sends: the system
message's text and the `tools=[...]` list, inside runner.py's chat wrapper.
That keeps mutants independent of how a given tree assembles its prompt; a
mutant whose text a tree no longer contains reports as absent, not as applied.

Pure functions only: harness.py, runner.py and the tests import this file.
"""
from __future__ import annotations

import copy
import json
import re

# --------------------------------------------------------------- system --
# Line mutants drop every line matching `line`. Block mutants drop from the
# line matching `start` up to (not including) the first line matching `end`,
# or the first blank line when `end` is None.

RULES = {
    "rule.only-inferred": r"^- Fill ONLY the filters",
    "rule.dates": r"^- Dates: set date_from",
    "rule.best-quality": r"^- 'best'/'top'",
    "rule.mood-visual-tag": r"^- A mood/look word",
    "rule.exclusion": r"^- EXCLUDE with",
    "rule.group-terms": r"^- Resolve group terms",
    "rule.read-total": r"^- Read `total`",
    "rule.summarize": r"^- For 'which year",
    "rule.representatives": r"^- For 'one/N per year'",
    "rule.daily-highlights": r"^- daily_highlights ONLY",
    "rule.foreground": r"^- 'X is the foreground",
    "rule.framing-flags": r"^- FRAMING",
    "rule.rerank": r"^- For VISUAL precision",
    "rule.explain-answer": r"^- Then stop and write",
    "grounding": r"^You are given LIBRARY FACTS",
}

BLOCKS = {
    "block.rules": {"start": r"^Rules:$", "end": r"^Examples:$"},
    "block.examples": {"start": r"^Examples:$", "end": r"^\s*$|^WRITES|^LIBRARY FACTS:|^USER NOTES:|^Today's date"},
    "block.facts": {"start": r"^LIBRARY FACTS:$", "end": None},
    "block.hints": {"start": r"^USER NOTES:$", "end": None},
    "block.writes": {"start": r"^WRITES \(", "end": None},
}

_EXAMPLE = re.compile(r"^  '([^']+)'")

# ---------------------------------------------------------------- tools --

GROUNDING_TOOLS = ("get_library_overview", "list_people", "list_places", "list_vocab")
# tools._CURATION_FILTER_PROPS keys, used when the tree under test has none.
DEFAULT_SHARED_PROPS = ("people", "location", "date_from", "date_to", "category",
                        "visual_tag", "keyword", "min_quality", "camera", "min_aesthetic",
                        "min_day_aesthetic", "style_tag", "only_these_people",
                        "faces_in_frame")

# The core set: one mutant per rule and block, all examples at once, and the
# tool-side trims a diet would try first. ~25 mutants; at one rep over the
# whole suite that is several hours, so `ablate` defaults to this set.
CORE = (list(RULES) + list(BLOCKS) + ["tools.filter-descriptions",
                                       "tools.all-param-descriptions",
                                       "drop.grounding-tools"])


# Control mutants: edits that remove nothing. At temperature 0 any change to
# the prompt text can flip a fragile item, so an item that flips under a
# control is noise, not evidence that it depends on the text a real mutant
# deleted. `harness.py mutation` discounts them.
CONTROLS = ("control.newline", "control.reword", "control.swap-examples")


def slug(text: str, n: int = 32) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-")


def example_ids(system: str) -> dict:
    """{example.<slug>: exact line} for every worked example in the prompt."""
    out = {}
    for line in system.splitlines():
        m = _EXAMPLE.match(line)
        if m:
            out[f"example.{slug(m.group(1))}"] = line
    return out


def all_ids(system: str, tools: list) -> list[str]:
    names = [t["function"]["name"] for t in tools]
    return (list(CONTROLS) + list(RULES) + list(BLOCKS) + list(example_ids(system))
            + ["tools.filter-descriptions", "tools.all-param-descriptions",
               "drop.grounding-tools"]
            + [f"tooldesc.{n}" for n in names] + [f"paramdesc.{n}" for n in names]
            + [f"drop.{n}" for n in names])


def _drop_lines(system: str, pred) -> tuple[str, int]:
    kept, removed = [], 0
    for line in system.split("\n"):
        if pred(line):
            removed += len(line) + 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def _drop_block(system: str, start: str, end) -> tuple[str, int]:
    lines, out, removed, i = system.split("\n"), [], 0, 0
    while i < len(lines):
        if re.search(start, lines[i]):
            removed += len(lines[i]) + 1
            i += 1
            while i < len(lines) and not (re.search(end, lines[i]) if end
                                          else not lines[i].strip()):
                removed += len(lines[i]) + 1
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), removed


def _strip_descriptions(schema, only: set | None = None):
    """Remove `description` from a JSON schema's properties (recursively),
    optionally only for property names in `only`."""
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for name, sub in props.items():
                if isinstance(sub, dict) and (only is None or name in only):
                    sub.pop("description", None)
                    _strip_descriptions(sub, None)
        for k in ("items", "anyOf", "oneOf"):
            if k in schema:
                v = schema[k]
                for s in (v if isinstance(v, list) else [v]):
                    _strip_descriptions(s, only)
    return schema


def apply(mutant: str, system: str, tools: list, *, shared_props=None) -> tuple[str, list, int]:
    """Return (system, tools, chars removed). chars removed == 0 means the
    mutant's text is absent from this prompt: the run would not be a mutant."""
    # One deepcopy PER tool: tools.openai_tools() splices the same filter-prop
    # dicts into several schemas, and one deepcopy of the list keeps that
    # aliasing, so stripping one tool's descriptions would strip them all.
    tools = [copy.deepcopy(t) for t in (tools or [])]
    before = len(system) + len(json.dumps(tools, ensure_ascii=False))

    def size(s, t):
        return before - (len(s) + len(json.dumps(t, ensure_ascii=False)))

    if mutant == "control.newline":
        return system + "\n", tools, 0
    if mutant == "control.reword":
        new = system.replace("You are a photo-library search assistant.",
                             "You are a search assistant for a photo library.", 1)
        return new, tools, 0
    if mutant == "control.swap-examples":
        lines = system.split("\n")
        ex = [i for i, l in enumerate(lines) if _EXAMPLE.match(l)]
        if len(ex) >= 2:
            lines[ex[0]], lines[ex[1]] = lines[ex[1]], lines[ex[0]]
        return "\n".join(lines), tools, 0
    if mutant in RULES:
        rx = re.compile(RULES[mutant])
        system, _ = _drop_lines(system, lambda l: bool(rx.search(l)))
    elif mutant in BLOCKS:
        b = BLOCKS[mutant]
        system, _ = _drop_block(system, b["start"], b["end"])
    elif mutant.startswith("example."):
        line = example_ids(system).get(mutant)
        if line is not None:
            system, _ = _drop_lines(system, lambda l: l == line)
    elif mutant == "tools.filter-descriptions":
        # The de-duplication trim: every tool but search_photos loses the
        # descriptions of the shared filter props (search_photos keeps one copy).
        shared = set(shared_props or DEFAULT_SHARED_PROPS)
        for t in tools:
            if t["function"]["name"] != "search_photos":
                _strip_descriptions(t["function"].get("parameters") or {}, shared)
    elif mutant == "tools.all-param-descriptions":
        for t in tools:
            _strip_descriptions(t["function"].get("parameters") or {})
    elif mutant == "drop.grounding-tools":
        tools = [t for t in tools if t["function"]["name"] not in GROUNDING_TOOLS]
    elif mutant.startswith("tooldesc."):
        for t in tools:
            if t["function"]["name"] == mutant.split(".", 1)[1]:
                t["function"]["description"] = ""
    elif mutant.startswith("paramdesc."):
        for t in tools:
            if t["function"]["name"] == mutant.split(".", 1)[1]:
                _strip_descriptions(t["function"].get("parameters") or {})
    elif mutant.startswith("drop."):
        tools = [t for t in tools if t["function"]["name"] != mutant.split(".", 1)[1]]
    else:
        raise KeyError(f"unknown mutant {mutant!r}")
    return system, tools, size(system, tools)


# ---------------------------------------------------------------- usage --
# Behaviours the owner's real asks need, recognised in the ask text. Each id
# is a rule id where a rule teaches it, else a `beh.*` id. Matched against
# the lowercased ask. Items declare the
# ids they exercise in `covers`; `harness.py coverage` compares the two.

USAGE = {
    "rule.representatives": r"each year|every year|per year|one per|for each year|over the years|each location|six months|each trip",
    "rule.foreground": r"foreground|primary athlete|largest face|prominent",
    "rule.framing-flags": r"only these|fully in the frame|fully visible|nobody else|just .* and .*,|no one cropped",
    "rule.daily-highlights": r"per day|each day|chronolog|duplicates|burst|stack",
    "rule.rerank": r"playing soccer|eyes and mouth|short hair|clear photos|interesting action|collage|looking at the camera|smiling",
    "rule.summarize": r"which year|how many|how often|who (shows|appears)|when (did|were)",
    "rule.best-quality": r"\bbest\b|\btop\b|favou?rite|\bgood\b|great",
    "rule.group-terms": r"the kids|the family|everyone|all of us",
    "rule.exclusion": r"\bno people\b|without|\bno [a-z]+ in",
    "rule.mood-visual-tag": r"moody|dramatic|candid|peaceful|golden",
    "beh.chapters": r"chapter|timeline|photo book|story",
    "beh.day-aesthetic": r"aesthetic %|day_aesthetic|per day, not for the entire",
    "beh.camera": r"kodak|pixpro|gopro|camera|iphone|pixel",
    "beh.count-cap": r"top \d+|best \d+|\d+ photos|limited to \d+|top three|top ten",
    "rule.dates": r"20\d\d|january|february|march|april|may|june|july|august|september|october|november|december|last week|last month|last year|this year|since|today|yesterday|summer|winter",
    "beh.place": r"france|tokyo|kyoto|paris|como|beach|\bhome\b|italy|japan|new york",
}

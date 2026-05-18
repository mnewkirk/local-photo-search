"""Phase 0 keyword-extraction bakeoff helpers.

Run `photosearch bakeoff-keywords` to compare two candidate models against
30 sample descriptions. Outputs side-by-side JSON for review before
hardcoding the chosen model in worker.py.
"""
from __future__ import annotations

import re
from typing import Optional

_KEYWORD_PROMPT_TEMPLATE = """\
Extract 5-15 keywords or short phrases from this photo description.

Rules:
- Include proper nouns (people, places), breeds, multi-word phrases (e.g. "golden retriever", "pacific ocean").
- Lowercase everything.
- Return ONLY a comma-separated list. No bullets, no explanation, no numbering.
- Skip vague words like "thing", "scene", "image", "photo".

Description:
{description}
"""


def build_keyword_prompt(description: str) -> str:
    return _KEYWORD_PROMPT_TEMPLATE.format(description=description.strip())


_BULLET_RE = re.compile(r"^[\-\*•]\s*")


def parse_keywords_response(raw: Optional[str]) -> list[str]:
    """Lowercased, deduped, ordered list of keywords from the LLM's raw text."""
    if not raw:
        return []
    # Normalise bullets / newlines to commas so the same splitter handles both.
    cleaned_lines = []
    for line in raw.splitlines():
        stripped = _BULLET_RE.sub("", line.strip())
        if stripped:
            cleaned_lines.append(stripped)
    text = ", ".join(cleaned_lines) if cleaned_lines else raw
    seen: set[str] = set()
    out: list[str] = []
    for token in text.split(","):
        kw = token.strip().lower()
        if not kw or kw in seen:
            continue
        seen.add(kw)
        out.append(kw)
    return out

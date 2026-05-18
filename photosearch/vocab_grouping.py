"""LLM-based clustering of candidate vocab into semantic buckets.

Reads a flat candidate list (output of mine-vocab) and asks Llama to
sort terms into draft buckets. The candidate list won't fit in one
context window for 4k+ terms, so we chunk by frequency band.
"""
from __future__ import annotations

import json
from typing import Optional

_GROUP_PROMPT = """\
You are organising photo-description vocabulary. Sort these terms into 6-12 semantic buckets like:
animals, people, activities, landscapes, weather, food, vehicles, architecture, mood, photography, miscellaneous.

Rules:
- Return ONLY valid JSON: {{"bucket_name": ["term1", "term2"], ...}}
- Every input term must appear in exactly one bucket.
- Use existing bucket names when terms fit; only invent new ones if necessary.

Terms (one per line):
{terms}

Existing buckets so far (merge into these where possible):
{existing}
"""


def group_terms(
    terms: list[str],
    chunk_size: int = 200,
    model: str = "llama3.2:3b",
    chat_fn=None,
) -> dict[str, list[str]]:
    """Group `terms` into semantic buckets via repeated LLM calls.

    chat_fn is the Ollama chat callable; defaults to describe._ollama_chat_with_retry.
    Injectable so tests can pass a deterministic stub.
    """
    if chat_fn is None:
        from photosearch.describe import _ollama_chat_with_retry as chat_fn  # type: ignore

    grouped: dict[str, list[str]] = {}
    for start in range(0, len(terms), chunk_size):
        chunk = terms[start : start + chunk_size]
        existing_summary = ", ".join(
            f"{k} ({len(v)})" for k, v in grouped.items()
        ) or "(none yet)"
        prompt = _GROUP_PROMPT.format(
            terms="\n".join(chunk),
            existing=existing_summary,
        )
        raw = chat_fn(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        parsed = _parse_grouping_json(raw, fallback_terms=chunk)
        for bucket, items in parsed.items():
            grouped.setdefault(bucket, []).extend(items)
    return grouped


def _parse_grouping_json(raw: Optional[str], fallback_terms: list[str]) -> dict[str, list[str]]:
    """Tolerantly parse the LLM's JSON; on failure, dump terms into a fallback bucket."""
    if not raw:
        return {"unsorted": list(fallback_terms)}
    # Strip code fences if present.
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return {"unsorted": list(fallback_terms)}
    if not isinstance(obj, dict):
        return {"unsorted": list(fallback_terms)}
    out: dict[str, list[str]] = {}
    for k, v in obj.items():
        if isinstance(v, list):
            out[str(k)] = [str(t).strip().lower() for t in v if str(t).strip()]
    return out or {"unsorted": list(fallback_terms)}

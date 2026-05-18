"""Compile a curated vocab draft into three importable Python modules.

The draft shape:
    {
        "content": [str, ...],        # ≥ 50 entries
        "visual":  [str, ...],        # ≥ 1 entry
        "expansions": {str: [str, ...], ...},   # query → category aliases
    }

Validation runs before any file write. The generated modules carry a
header comment with the draft hash + timestamp so the deployed vocab
is traceable back to the curation session that produced it.

See also: /admin/vocab curator (see docs/plans/categories-keywords-redesign.md).
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone


class VocabError(ValueError):
    """Raised when a draft fails validation."""


def validate_draft(draft: dict) -> None:
    content = [t for t in draft.get("content", []) if t]
    visual = [t for t in draft.get("visual", []) if t]
    overlap = set(content) & set(visual)
    if overlap:
        raise VocabError(
            f"Terms appear in both content and visual vocab: {sorted(overlap)[:5]}"
        )
    if len(content) < 50:
        raise VocabError(
            f"Content vocab must have at least 50 terms; got {len(content)}."
        )
    if not visual:
        raise VocabError("visual vocab is empty; add mood/light/composition terms.")


def draft_hash(draft: dict) -> str:
    serialised = json.dumps(draft, sort_keys=True).encode()
    return hashlib.sha256(serialised).hexdigest()[:12]


def _header(kind: str, draft_hash_: str, timestamp: str) -> str:
    return (
        f'"""GENERATED FILE — do not edit by hand.\n\n'
        f"Kind: {kind}\n"
        f"Draft hash: {draft_hash_}\n"
        f"Generated at: {timestamp}\n\n"
        f"Source: /admin/vocab curator (see docs/plans/categories-keywords-redesign.md).\n"
        f"Regenerate via POST /api/admin/vocab/compile or `photosearch compile-vocab`.\n"
        f'"""\n\n'
    )


def render_content_module(terms: list[str], draft_hash: str, timestamp: str) -> str:
    body = ",\n    ".join(repr(t) for t in sorted(set(terms)))
    return (
        _header("content vocabulary", draft_hash, timestamp)
        + f"CONTENT_VOCABULARY: list[str] = [\n    {body},\n]\n"
    )


def render_visual_module(terms: list[str], draft_hash: str, timestamp: str) -> str:
    body = ",\n    ".join(repr(t) for t in sorted(set(terms)))
    return (
        _header("visual vocabulary", draft_hash, timestamp)
        + f"VISUAL_VOCABULARY: list[str] = [\n    {body},\n]\n"
    )


def render_query_expansion_module(
    expansions: dict[str, list[str]],
    draft_hash: str,
    timestamp: str,
) -> str:
    items = []
    for query, cats in sorted(expansions.items()):
        cat_set = "{" + ", ".join(repr(c) for c in sorted(set(cats))) + "}"
        items.append(f"    {query!r}: {cat_set},")
    body = "\n".join(items)
    return (
        _header("query → categories expansion", draft_hash, timestamp)
        + f"_QUERY_TO_CATEGORIES: dict[str, set[str]] = {{\n{body}\n}}\n"
    )


def compile_draft(
    draft: dict,
    repo_dir: str,
) -> dict:
    """Validate, render, and write the three modules. Returns metadata."""
    validate_draft(draft)
    h = draft_hash(draft)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {
        f"{repo_dir}/photosearch/vocab_content.py":
            render_content_module(draft["content"], h, ts),
        f"{repo_dir}/photosearch/vocab_visual.py":
            render_visual_module(draft["visual"], h, ts),
        f"{repo_dir}/photosearch/vocab_query_expansion.py":
            render_query_expansion_module(draft.get("expansions", {}), h, ts),
    }
    for path, src in files.items():
        with open(path, "w") as f:
            f.write(src)
    return {"draft_hash": h, "timestamp": ts, "files": list(files.keys())}

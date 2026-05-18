"""Admin curator API for the categories/visual/keywords vocab.

Endpoints under /api/admin/vocab/* — used by frontend/dist/admin_vocab.html.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Body

from .vocab_compile import compile_draft, VocabError


router = APIRouter(prefix="/api/admin/vocab", tags=["admin-vocab"])


def _data_dir() -> Path:
    return Path(os.environ.get("PHOTOSEARCH_DATA_DIR", "/data"))


def _candidates_path() -> Path:
    return _data_dir() / "vocab_candidates.json"


def _draft_path() -> Path:
    return _data_dir() / "vocab_draft.json"


def _get_db():
    # Local import avoids circulars at module import time.
    from .web import _get_db as web_get_db
    return web_get_db()


@router.get("/candidates")
def get_candidates():
    path = _candidates_path()
    if not path.exists():
        raise HTTPException(404, "No vocab_candidates.json. Run `photosearch mine-vocab` first.")
    with path.open() as f:
        return json.load(f)


@router.get("/draft")
def get_draft():
    path = _draft_path()
    if not path.exists():
        return {"content": [], "visual": [], "expansions": {}}
    with path.open() as f:
        return json.load(f)


@router.put("/draft")
def put_draft(draft: dict = Body(...)):
    # Coerce / validate shape; don't enforce vocab-size rules here, that's compile's job.
    content = list(draft.get("content", []))
    visual = list(draft.get("visual", []))
    expansions = dict(draft.get("expansions", {}))
    payload = {"content": content, "visual": visual, "expansions": expansions}
    with _draft_path().open("w") as f:
        json.dump(payload, f, indent=2)
    return {"ok": True, "content_count": len(content), "visual_count": len(visual)}


def _photo_terms(description: Optional[str], terms: list[str]) -> list[str]:
    """Match terms against a description (case-insensitive whole-word)."""
    if not description:
        return []
    text = description.lower()
    matched = []
    for term in terms:
        pat = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pat, text):
            matched.append(term)
    return matched


@router.post("/coverage-preview")
def coverage_preview(payload: dict = Body(...)):
    """Sample N described photos; return % that get >=1 category from the draft."""
    draft = payload.get("draft") or {}
    sample_size = int(payload.get("sample_size") or 1000)
    content_terms = list(draft.get("content", []))

    with _get_db() as db:
        rows = db.conn.execute(
            "SELECT id, description FROM photos "
            "WHERE description IS NOT NULL "
            "ORDER BY RANDOM() LIMIT ?",
            (sample_size,),
        ).fetchall()

    covered = 0
    samples_uncovered = []
    for row in rows:
        if _photo_terms(row["description"], content_terms):
            covered += 1
        elif len(samples_uncovered) < 10:
            samples_uncovered.append({"id": row["id"], "description": row["description"][:140]})

    actual = len(rows)
    pct = round(100.0 * covered / actual, 1) if actual else 0.0
    return {
        "sample_size": actual,
        "covered_count": covered,
        "coverage_pct": pct,
        "samples_uncovered": samples_uncovered,
    }


@router.post("/test-photo/{photo_id}")
def test_photo(photo_id: int, payload: dict = Body(...)):
    draft = payload.get("draft") or {}
    with _get_db() as db:
        row = db.conn.execute(
            "SELECT description FROM photos WHERE id = ?", (photo_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, f"Photo {photo_id} not found.")
    desc = row["description"]
    return {
        "photo_id": photo_id,
        "description": desc,
        "matched_categories": _photo_terms(desc, draft.get("content", [])),
        "matched_visual": _photo_terms(desc, draft.get("visual", [])),
    }


@router.post("/compile")
def compile_(payload: dict = Body(...)):
    draft = payload.get("draft") or {}
    repo_dir = payload.get("repo_dir") or os.environ.get("PHOTOSEARCH_REPO_DIR", "/repo")
    try:
        result = compile_draft(draft, repo_dir=repo_dir)
    except VocabError as exc:
        raise HTTPException(400, str(exc))
    return result

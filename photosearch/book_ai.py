"""LLM-assisted photobook drafting — captions and whole-book first drafts.

Reuses the Ask agent's chat plumbing (``agent._chat`` → LM Studio / Ollama via
``PHOTOSEARCH_TEXT_LLM_URL`` + ``PHOTOSEARCH_LLM_AGENT_MODEL``) so nothing leaves
the machine. Captions follow the family house voice captured in
``~/shutterfly-proofs/BOOK_STYLE_GUIDE.md``: short, casual, first-person-plural,
specific, never a descriptive label.
"""
from __future__ import annotations

import re
from typing import Optional

_CAPTION_SYSTEM = (
    "You write one caption for a spread in a family's printed travel photo book. "
    "Voice: warm, casual, FIRST-PERSON PLURAL ('we', 'the kids'), ONE short "
    "sentence (max ~16 words), with a small specific detail (place, food, a name) "
    "when you can. Examples of the house voice: 'But first, crepes!'  'Swimming off "
    "the rocks — freezing, and nobody cared.'  'We came to Spain for ham and boy did "
    "Spain deliver!' Do NOT describe the image literally ('a photo of...'), no "
    "quotation marks, hashtags, or emoji. Reply with ONLY the caption text."
)


def _clean_caption(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.strip().strip('"').strip("'").strip()
    t = re.sub(r"\s+", " ", t)
    if not t or t.upper() == "NONE" or len(t) < 4:
        return None
    # Guard against a rambling model — keep it to the first sentence, capped.
    t = re.split(r"(?<=[.!?])\s", t)[0]
    return t[:160].strip()


def _photo_context(pdb, photo_ids: list[int]) -> tuple[list[str], Optional[str], Optional[str]]:
    """Return (short description lines, representative place, representative date)."""
    if not photo_ids:
        return [], None, None
    ph = ",".join("?" * len(photo_ids))
    rows = pdb.conn.execute(
        f"SELECT id, description, place_name, date_taken FROM photos WHERE id IN ({ph})",
        photo_ids).fetchall()
    by_id = {r["id"]: r for r in rows}
    lines, place, date = [], None, None
    for pid in photo_ids:
        r = by_id.get(pid)
        if not r:
            continue
        place = place or r["place_name"]
        date = date or (r["date_taken"] or "")[:10] or None
        d = (r["description"] or "").strip()
        if d:
            lines.append("- " + d[:180])
    return lines[:8], place, date


def draft_caption(pdb, photo_ids: list[int], book_name: Optional[str] = None) -> Optional[str]:
    """Draft one spread's caption from its photos' descriptions/place/date."""
    from .agent import _chat
    lines, place, date = _photo_context(pdb, photo_ids)
    if not lines:
        return None
    ctx = []
    if book_name:
        ctx.append(f"Book: {book_name}")
    if place:
        ctx.append(f"Place: {place}")
    if date:
        ctx.append(f"Date: {date}")
    ctx.append("Photos on this spread:")
    ctx.extend(lines)
    ctx.append("\nWrite the caption (or NONE):")
    try:
        r = _chat([{"role": "system", "content": _CAPTION_SYSTEM},
                   {"role": "user", "content": "\n".join(ctx)}],
                  None, temperature=0.7, timeout=30, reasoning_effort="none")
    except Exception:
        return None
    return _clean_caption(r.get("content"))


def _thin_per_day(pdb, ids: list[int], per_day: int) -> list[int]:
    """Keep the top ``per_day`` photos of each day by aesthetic percentile — a
    cheap best-of-day pass over the pool (the daily_highlights idea, on an id set)."""
    if not ids or not per_day:
        return ids
    ph = ",".join("?" * len(ids))
    rows = pdb.conn.execute(
        f"SELECT id, substr(date_taken,1,10) AS day, "
        f"COALESCE(aes_overall_pct, aesthetic_score, 0) AS q "
        f"FROM photos WHERE id IN ({ph})", ids).fetchall()
    by_day: dict[str, list] = {}
    for r in rows:
        by_day.setdefault(r["day"] or "?", []).append((r["q"], r["id"]))
    kept = []
    for day in sorted(by_day):
        top = sorted(by_day[day], reverse=True)[:per_day]
        kept.extend(pid for _, pid in top)
    # preserve chronological order
    order = {r["id"]: (r["day"] or "", r["id"]) for r in rows}
    return sorted(set(kept), key=lambda i: order.get(i, ("", i)))


def ai_draft_book(pdb, bs, book_id: int, per_day: Optional[int] = None,
                  spread_count: Optional[int] = None, caption: bool = True) -> dict:
    """Whole-book first draft: (optionally) thin the included pool to the best of
    each day, auto-arrange into spreads via the deterministic partition, then
    draft a caption per spread. Respects include/exclude decisions."""
    excluded = {pid for pid, d in bs.decision_map(book_id).items() if d == "exclude"}
    ids = [i for i in bs.included_ids(book_id) if i not in excluded]
    if not ids:
        return {"error": "no included candidates — add photos to the pool first"}
    if per_day:
        ids = _thin_per_day(pdb, ids, per_day)
    n = bs.auto_arrange(pdb, book_id, ids, spread_count, replace=True)
    captioned = 0
    if caption:
        book = bs.get_book_row(book_id) or {}
        for sp in bs.get_book(book_id)["spreads"]:
            pids = [c["photo_id"] for c in sp["cells"] if c["photo_id"]]
            cap = draft_caption(pdb, pids, book.get("name"))
            if cap:
                dark = (sp.get("bg") or "#ffffff") != "#ffffff"
                bs.update_spread(pdb, sp["id"], {"caption": {"text": cap, "dark": dark}})
                captioned += 1
    return {"spreads_created": n, "captioned": captioned, "photos_used": len(ids)}


def caption_all(pdb, bs, book_id: int, overwrite: bool = False) -> int:
    """Draft a caption for every spread that lacks one (or all, if overwrite)."""
    book = bs.get_book_row(book_id) or {}
    done = 0
    for sp in bs.get_book(book_id)["spreads"]:
        if sp.get("caption") and not overwrite:
            continue
        pids = [c["photo_id"] for c in sp["cells"] if c["photo_id"]]
        cap = draft_caption(pdb, pids, book.get("name"))
        if cap:
            dark = (sp.get("bg") or "#ffffff") != "#ffffff"
            bs.update_spread(pdb, sp["id"], {"caption": {"text": cap, "dark": dark}})
            done += 1
    return done

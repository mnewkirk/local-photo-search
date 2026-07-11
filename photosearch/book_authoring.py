"""Photobook AUTHORING pipeline (M30) — turn a raw trip window into a complete
first-draft book, then let the human refine it.

The manual Varenna process, automated: segment the window into day/scene beats
(near-dup bursts collapsed), have a TEXT LLM draft the day→beat outline (titles,
in/out, spread budget), then have a VISION model LOOK AT the candidates and pick
the representative hero per beat. Selection is editorial, so a score-cull can't do
it (it recovers ~6% of the real picks) — but an agent reasoning over
descriptions/faces + a VLM judging the actual frames can produce a strong draft.

Pure functions here (no sidecar writes); ``book.BookStore`` persists, ``web.py``
orchestrates + streams. Reuses:
  - tools._h_daily_scene_breakdown + _dedupe_ranked  (scene segmentation, dedup)
  - tools._vision_score + _thumb_b64                 (VLM hero pick)
  - agent._chat / _agent_model                       (text outline)
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


# ---------------------------------------------------------------------------
# Stage 1a — segment the pool into day → scene candidates
# ---------------------------------------------------------------------------

def _pool_days(pdb, filters: dict) -> list[str]:
    from .tools import _build_filter_sql
    where, params = _build_filter_sql(pdb, filters)
    rows = pdb.conn.execute(
        f"SELECT DISTINCT substr(date_taken,1,10) d FROM photos "
        f"WHERE ({where}) AND date_taken IS NOT NULL "
        f"AND date_taken GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*' "
        f"ORDER BY d", params).fetchall()
    return [r["d"] for r in rows if r["d"]]


def _scene_people(pdb, ids: list[int]) -> list[str]:
    if not ids:
        return []
    ph = ",".join("?" * len(ids))
    rows = pdb.conn.execute(
        f"SELECT DISTINCT pr.name FROM faces f JOIN persons pr ON f.person_id = pr.id "
        f"WHERE f.photo_id IN ({ph}) AND pr.name IS NOT NULL", ids).fetchall()
    return [r["name"] for r in rows]


def segment_pool(pdb, filters: dict, gap_minutes: float = 20.0,
                 per_scene: int = 8, max_scenes: int = 80) -> list[dict]:
    """Per-day scene segmentation over the filtered pool. Each scene is a
    time+place cluster with near-dup bursts collapsed to ``per_scene`` candidate
    reps. Returns a flat, globally-indexed scene list."""
    from .tools import _h_daily_scene_breakdown
    scenes: list[dict] = []
    for day in _pool_days(pdb, filters):
        args = dict(filters, date=day, gap_minutes=gap_minutes, per_scene=per_scene)
        res = _h_daily_scene_breakdown(pdb, args)
        hits_by_scene: dict[int, list] = {}
        for h in res.get("results", []):
            hits_by_scene.setdefault(h.get("scene"), []).append(h)
        for sc in res.get("scenes", []):
            reps = hits_by_scene.get(sc["index"], [])
            rep_ids = sc.get("representative_ids", [])
            scenes.append({
                "gindex": len(scenes),
                "day": day,
                "place": sc.get("place"),
                "start": sc.get("start"),
                "end": sc.get("end"),
                "photo_count": sc.get("photo_count"),
                "candidate_ids": rep_ids,
                "descriptions": [h.get("description") for h in reps if h.get("description")][:4],
                "people": _scene_people(pdb, rep_ids),
            })
    return scenes[:max_scenes]


# ---------------------------------------------------------------------------
# Stage 1b — TEXT LLM drafts the day → beat outline
# ---------------------------------------------------------------------------

_OUTLINE_SYSTEM = (
    "You are the editor of a family's printed travel photo book. You are given the "
    "trip's photo SCENES in chronological order (each = a time+place cluster of "
    "photos, already de-duplicated). Group them into the book's STORY: day-by-day "
    "BEATS, each a spread-worthy moment. Your job is EDITORIAL:\n"
    "- Give each beat a short, specific title (place/activity), house voice. NOTE: "
    "the place label on each scene is an approximate reverse-geocode (the town/"
    "comune), NOT a specific landmark — e.g. lakeside scenes near Varenna may be "
    "geocoded 'Vezio' or 'Perledo'. Name the actual activity/landmark from the "
    "photo DESCRIPTIONS, and don't attach a famous-landmark name (a castle, a "
    "church) to a beat unless the descriptions actually show it.\n"
    "- Decide which beats belong in the book (include=true) and which to cut "
    "(include=false) — cut weak, redundant, or purely-incidental scenes.\n"
    "- SIZE TO THE DAY. A scene-dense stretch at one place/activity (roughly 5+ "
    "scenes — a castle visit, a long hike, a big museum) is a BIG EVENT: split it "
    "into several distinct sub-beats (arrival / lunch / the tower / the view / "
    "golden hour…), each its own beat with its own spread. Do NOT collapse a big "
    "event into one beat. A quiet day is one beat, one spread.\n"
    "- Allocate a spread budget per included beat (1 typical; 2 for a rich "
    "sequence) so the included totals land near the target — spend the budget on "
    "the big-event days, not by padding quiet ones.\n"
    "- A beat may merge several adjacent scenes (list their indices).\n"
    "Reply with ONLY JSON: {\"beats\":[{\"title\":str,\"day\":\"YYYY-MM-DD\","
    "\"scenes\":[int,...],\"include\":bool,\"spreads\":int}]} in chronological order."
)


def _scene_digest(scenes: list[dict]) -> str:
    lines = []
    for s in scenes:
        t = (s.get("start") or "")[11:16]
        who = (" · with " + ", ".join(s["people"][:4])) if s.get("people") else ""
        desc = (s["descriptions"][0][:120] + "…") if s.get("descriptions") else "(no description)"
        lines.append(f"[{s['gindex']}] {s['day']} {t} @ {s.get('place') or 'Unknown'} "
                     f"({s.get('photo_count')} photos){who}: {desc}")
    return "\n".join(lines)


def draft_outline(scenes: list[dict], notes: Optional[str],
                  target_spreads: int) -> list[dict]:
    """Ask the text LLM to group scenes into an editorial beat outline. Returns a
    list of beats each with title/day/scenes/include/spreads; falls back to
    one-beat-per-scene if the model is unavailable or returns junk."""
    from .agent import _chat
    if not scenes:
        return []
    user = (f"Target book length: about {target_spreads} spreads.\n"
            + (f"Trip notes / itinerary from the family:\n{notes}\n\n" if notes else "")
            + f"SCENES ({len(scenes)}):\n{_scene_digest(scenes)}\n\n"
            "Return the beat outline JSON.")
    try:
        r = _chat([{"role": "system", "content": _OUTLINE_SYSTEM},
                   {"role": "user", "content": user}],
                  None, temperature=0.4, timeout=120, reasoning_effort="none")
        content = r.get("content") or ""
        a, b = content.find("{"), content.rfind("}")
        beats = json.loads(content[a:b + 1]).get("beats", [])
    except Exception:
        beats = []
    valid = _validate_beats(beats, scenes)
    if valid:
        return valid
    # Fallback: every scene is its own included beat.
    return [{"title": (s.get("place") or "Untitled"), "day": s["day"],
             "scenes": [s["gindex"]], "include": True, "spreads": 1} for s in scenes]


def allocate_budget(beats: list[dict], target_spreads: int) -> None:
    """Deterministically distribute the spread budget across INCLUDED beats,
    weighted by scene density, so big-event days (many scenes) earn multi-spread
    treatment and the totals land near the target. Mutates beats in place — the
    LLM's per-beat spread guess is unreliable arithmetic; this fixes it."""
    inc = [b for b in beats if b.get("include")]
    if not inc:
        return
    n = len(inc)
    weights = [max(1, len(b.get("scenes") or [])) for b in inc]
    extra = max(0, int(target_spreads) - n)      # 1 base spread each, share the rest
    tw = sum(weights) or 1
    raw = [extra * w / tw for w in weights]
    add = [int(x) for x in raw]
    left = extra - sum(add)
    for i in sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)[:left]:
        add[i] += 1
    for b, a in zip(inc, add):
        b["spreads"] = max(1, min(4, 1 + a))


def pick_cover(pdb, filters: dict) -> Optional[int]:
    """A house-style front cover = a wide scenic establishing frame: landscape
    orientation, few/no faces, highest aesthetic. (The people 'we're on our way'
    shot stays the title page.)"""
    from .tools import _build_filter_sql
    where, params = _build_filter_sql(pdb, filters)
    row = pdb.conn.execute(
        f"SELECT id FROM photos WHERE ({where}) "
        f"AND image_width > image_height * 1.4 "
        f"AND (SELECT COUNT(*) FROM faces f WHERE f.photo_id = photos.id) <= 1 "
        f"ORDER BY COALESCE(aes_overall, aesthetic_score, 0) DESC LIMIT 1",
        params).fetchone()
    return row["id"] if row else None


def _validate_beats(beats, scenes) -> list[dict]:
    if not isinstance(beats, list) or not beats:
        return []
    valid_idx = {s["gindex"] for s in scenes}
    out = []
    for b in beats:
        if not isinstance(b, dict):
            continue
        idx = [int(i) for i in (b.get("scenes") or []) if isinstance(i, (int, float)) and int(i) in valid_idx]
        if not idx:
            continue
        by = {s["gindex"]: s for s in scenes}
        out.append({
            "title": str(b.get("title") or "Untitled")[:120],
            "day": b.get("day") or by[idx[0]]["day"],
            "scenes": idx,
            "include": bool(b.get("include", True)),
            "spreads": max(1, min(3, int(b.get("spreads") or 1))),
        })
    return out


# ---------------------------------------------------------------------------
# Stage 2 — VISION model picks the representative hero(es) per beat
# ---------------------------------------------------------------------------

_HERO_WORKERS = 6


def _vision_hero(base_url: str, model: str, image_b64: str, criteria: str):
    """One VLM call scoring a candidate AND judging whether it must be shown
    uncropped. Returns {score, reason, full_frame} or None."""
    import urllib.request
    prompt = (
        "You are choosing photos for a family travel photo book.\n"
        f'This photo is a candidate to represent: "{criteria}".\n'
        "Look at the image and reply with ONLY JSON:\n"
        '{"score": <0.0-1.0, how well it represents that moment>, '
        '"full_frame": <true if it MUST be shown uncropped — it is a portrait/'
        'vertical, or the subject fills the frame or reaches the edges so cropping '
        'to a wide box would cut it; false if it is a wide scene that crops fine>, '
        '"reason": "<one short phrase>"}')
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ]}], "max_tokens": 150, "temperature": 0}).encode("utf-8")
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            content = json.loads(r.read())["choices"][0]["message"]["content"] or ""
    except Exception:
        return None
    a, b = content.find("{"), content.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        o = json.loads(content[a:b + 1])
        score = float(o.get("score"))
    except (ValueError, TypeError):
        return None
    return {"score": max(0.0, min(1.0, score)),
            "reason": str(o.get("reason", ""))[:120],
            "full_frame": bool(o.get("full_frame"))}


def pick_heroes(pdb, candidate_ids: list[int], beat_title: str,
                n_heroes: int = 1) -> dict:
    """Have the vision model LOOK AT each candidate: score how well it represents
    the beat AND whether it must be shown uncropped ('full' vs croppable). Returns
    heroes (top-N) + all candidates ranked, each with a VLM-decided crop_mode.
    Falls back to the geometric suggest_crop_mode with no VLM."""
    from .tools import _thumb_b64
    ids = list(dict.fromkeys(int(i) for i in candidate_ids))
    if not ids:
        return {"heroes": [], "ranked": []}
    base = os.environ.get("PHOTOSEARCH_TEXT_LLM_URL")
    model = os.environ.get("PHOTOSEARCH_LLM_VISUAL_MODEL")
    criteria = (f"the single best photo to represent \"{beat_title}\" in a family "
                "travel photo book — clearly shows the moment/place, the family "
                "visible when relevant, sharp and well composed, not a duplicate")
    if not (base and model):
        ranked = [{"id": i, "score": None, "reason": "no vision model",
                   "crop_mode": suggest_crop_mode(pdb, i)} for i in ids]
        return {"heroes": ids[:n_heroes], "ranked": ranked, "reranked": False}

    # Resolve thumbnails in THIS thread — _thumb_b64's local branch touches the
    # sqlite conn, which can't cross threads; only the VLM HTTP call is fanned out.
    b64s = {i: _thumb_b64(pdb, i) for i in ids}

    def _score(pid):
        b = b64s.get(pid)
        return pid, (_vision_hero(base, model, b, criteria) if b else None)

    scores: dict = {}
    with ThreadPoolExecutor(max_workers=_HERO_WORKERS) as pool:
        for pid, sc in pool.map(_score, ids):
            scores[pid] = sc
    scored = [i for i in ids if scores.get(i)]
    scored.sort(key=lambda i: scores[i]["score"], reverse=True)
    failed = [i for i in ids if not scores.get(i)]
    order = scored + failed
    # Aspect gate: a wide LANDSCAPE crops fine and should fill its cell — only
    # honor the VLM's 'full_frame' on it when the geometry also says so (portrait /
    # subject-fills). Otherwise the VLM over-flags full and every spread goes sparse.
    ph = ",".join("?" * len(ids))
    ar = {r["id"]: ((r["w"] / r["h"]) if r["w"] and r["h"] else 1.5)
          for r in pdb.conn.execute(
              f"SELECT id, image_width w, image_height h FROM photos WHERE id IN ({ph})", ids)}
    ranked = []
    for i in order:
        sc = scores.get(i)
        geo = suggest_crop_mode(pdb, i)      # 'full' for portrait / subject-fills
        vlm_full = bool(sc and sc.get("full_frame"))
        # Honor the VLM's 'full' only for portrait/near-square frames (verticality
        # the crop can't preserve). Landscapes — incl. 4:3 — fill their cell and
        # only stay uncropped when the geometry says the subject fills the frame.
        mode = "full" if (geo == "full" or (vlm_full and ar.get(i, 1.5) < 0.95)) else "crop"
        ranked.append({"id": i, "score": (sc["score"] if sc else None),
                       "reason": (sc["reason"] if sc else None), "crop_mode": mode})
    return {"heroes": order[:n_heroes], "ranked": ranked, "reranked": True}


# ---------------------------------------------------------------------------
# Crop-mode suggestion — full uncropped hero vs crop-OK
# ---------------------------------------------------------------------------

def draft_book(pdb, bs, book_id: int, filters: dict, notes: Optional[str],
               target_spreads: int, gap_minutes: float = 20.0, per_scene: int = 8,
               captions: bool = True):
    """One-shot: segment → outline → per-beat VLM heroes → persist outline →
    assemble spreads → draft captions. A GENERATOR yielding progress dicts for
    SSE; the last event is {'phase':'done', ...}."""
    from . import book_ai
    yield {"phase": "segment"}
    scenes = segment_pool(pdb, filters, gap_minutes, per_scene)
    by = {s["gindex"]: s for s in scenes}
    yield {"phase": "segmented", "scenes": len(scenes)}

    yield {"phase": "outline"}
    raw = draft_outline(scenes, notes, target_spreads)
    allocate_budget(raw, target_spreads)   # density-weighted; big days go multi-spread
    included = [b for b in raw if b.get("include")]
    yield {"phase": "outlined", "beats": len(raw), "included": len(included),
           "spread_budget": sum(b.get("spreads", 1) for b in included)}

    beats: list[dict] = []
    best_cover = (None, -1.0)   # (photo_id, vlm_score) for an auto cover
    done = 0
    for b in raw:
        cand_ids = list(dict.fromkeys(
            pid for gi in b["scenes"] if gi in by for pid in by[gi]["candidate_ids"]))
        beat = {"title": b["title"], "day": b["day"],
                "status": "in" if b.get("include") else "out",
                "spread_budget": b.get("spreads", 1),
                "scene_meta": {"scenes": b["scenes"]}, "candidates": []}
        if b.get("include") and cand_ids:
            picked = pick_heroes(pdb, cand_ids, b["title"], n_heroes=b.get("spreads", 1))
            heroes = set(picked["heroes"])
            for pos, r in enumerate(picked["ranked"]):
                pid = r["id"]
                is_hero = pid in heroes
                beat["candidates"].append({
                    "photo_id": pid, "position": pos,
                    "role": "hero" if is_hero else "candidate",
                    "vlm_score": r.get("score"), "vlm_reason": r.get("reason"),
                    "crop_mode": r.get("crop_mode") or "crop"})
                if is_hero and (r.get("score") or 0) > best_cover[1]:
                    best_cover = (pid, r.get("score") or 0)
            done += 1
            yield {"phase": "heroes", "done": done, "total": len(included),
                   "title": b["title"], "heroes": list(heroes)}
        else:
            beat["candidates"] = [{"photo_id": pid, "position": pos, "role": "candidate"}
                                  for pos, pid in enumerate(cand_ids)]
        beats.append(beat)

    bs.replace_outline(book_id, beats)

    # Auto covers/title page for a complete draft (only if unset).
    book = bs.get_book_row(book_id) or {}
    upd = {}
    if not book.get("cover_photo_id"):
        upd["cover_photo_id"] = pick_cover(pdb, filters) or best_cover[0]
    if not book.get("title_page_photo_id"):
        first_in = next((bt for bt in beats if bt["status"] == "in" and bt["candidates"]), None)
        if first_in:
            hero = next((c for c in first_in["candidates"] if c["role"] == "hero"), None)
            if hero:
                upd["title_page_photo_id"] = hero["photo_id"]
    if not book.get("target_spreads"):
        upd["target_spreads"] = target_spreads
    if upd:
        bs.update_book(book_id, upd)

    yield {"phase": "assemble"}
    n = bs.assemble_from_outline(pdb, book_id)
    if captions:
        yield {"phase": "captions"}
        try:
            book_ai.caption_all(pdb, bs, book_id)
        except Exception:
            pass
    yield {"phase": "done", "spreads": n, "beats": len(beats), "included": len(included)}


def suggest_crop_mode(pdb, photo_id: int) -> str:
    """'full' (must not crop) for portraits / tight verticals / a subject that
    fills the frame; 'crop' otherwise (wide establishing shots take a crop)."""
    row = pdb.conn.execute(
        "SELECT image_width w, image_height h, subject_boxes sb FROM photos WHERE id=?",
        (photo_id,)).fetchone()
    if not row or not row["w"] or not row["h"]:
        return "crop"
    if row["h"] > row["w"] * 1.15:          # portrait / vertical → keep whole
        return "full"
    if row["sb"]:
        try:
            for s in json.loads(row["sb"]) or []:
                bb = s.get("bbox")
                if bb and len(bb) == 4:
                    frac = max(0.0, (bb[2] - bb[0])) * max(0.0, (bb[3] - bb[1]))
                    if frac >= 0.55:         # subject fills the frame → don't slice it
                        return "full"
        except Exception:
            pass
    return "crop"

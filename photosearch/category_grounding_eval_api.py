"""API behind the category-grounding labelling page (`/eval/category-grounding`).

`photosearch/category_grounding.py` drops category-content terms the
description contradicts (see its docstring and `docs/plans/category-grounding.md`).
`evals/category_grounding_eval.py` draws a HOLDOUT sample of drops — 25 per
contradiction group, disjoint from the sample the cue lists were tuned on —
and the owner judges each one here: was dropping this term CORRECT, WRONG, or
DEBATABLE? A group ships only once it clears 90% precision on at least 20
judged (correct + wrong) drops.

**Local-only on purpose: nothing here proxies to the NAS in replica mode**,
the same rule as `eval_api.py` / `model_eval_api.py`. Labels are written back
into the sample FILE in `eval_dir()` on whichever machine serves the page —
not a DB row — because `sync-replica.sh` swaps the replica DB wholesale and a
hand label is not something a sync can regenerate. The DB is only READ, for
the photo's CURRENT description and categories (which may have moved on since
the sample was drawn; the page shows the live values, not a frozen copy).

`compute_score` is the ONE scoring function: this module's `/score` endpoint
and `evals/category_grounding_eval.py score` both call it, so the live page
and the CLI table can never drift apart.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/eval/category-grounding", tags=["eval"])

#: Filename is a constant, not a parameter: there is exactly one holdout file
#: today (seed 11, drawn by `evals/category_grounding_eval.py`'s `write_holdout`).
HOLDOUT_FILE = "samples-holdout-seed11.json"

#: Must match `photosearch.category_grounding.GROUPS` by name. Not imported
#: from there directly: the holdout file is the contract the page reads, and
#: pinning the tuple here means a future GROUPS edit can't silently change
#: what this page expects to find in the sample file without a matching test
#: failure (`test_groups_match_category_grounding_module`).
GROUPS: tuple[str, ...] = ("sport", "setting", "light", "weather")

LABELS: tuple[str, ...] = ("correct", "wrong", "debatable")

#: Ship bar from docs/plans/category-grounding.md: a group needs >= 90%
#: precision (correct / (correct + wrong), debatable excluded) on at least 20
#: judged drops before the rule ships.
PASS_PRECISION = 0.9
MIN_JUDGED = 20

#: One-line help text per group, shown on the labelling page so the owner
#: judges each drop against the rule that produced it, not from memory.
#: Kept in sync with `photosearch/category_grounding.py`'s GROUPS docstring
#: by `test_group_rules_cover_every_group`.
GROUP_RULES: dict[str, str] = {
    "sport": ("Which sport is being played: soccer / basketball / baseball / "
              "volleyball. A side's terms are dropped when a rival sport is "
              "cued in the description and this side is not (basketball "
              "court/hoop survive on an indoor-cued soccer photo — that's the "
              "literal gym venue)."),
    "setting": ("Indoor vs outdoor. A side's terms are dropped when the "
                "description cues the other setting and not this one; "
                "skipped entirely on a collage (honestly both)."),
    "light": ("Artificial vs natural light. `artificial lighting` is dropped "
              "when the description cues daylight/sun/outdoors and no "
              "artificial-light word; `natural light`/`sunny day`/`sun` is "
              "dropped the other way."),
    "weather": ("Sunny vs overcast sky. A side's terms are dropped when the "
                "description cues the rival weather and not this one; "
                "\"partly cloudy\" only weakly cues overcast (drops just the "
                "clear-sky terms, keeps `cloudy sky`)."),
}

_HINT = ("No holdout sample yet. Draw one with: "
         "python evals/category_grounding_eval.py --db <db>")


def eval_dir() -> Path:
    return Path(os.environ.get("PHOTOSEARCH_CATEGORY_EVAL_DIR", "./evals/category-grounding"))


def _holdout_path() -> Path:
    return eval_dir() / HOLDOUT_FILE


def _get_db():
    # Deferred, same as eval_api / model_eval_api: web imports this module at
    # import time, so importing web here would be circular.
    from . import web
    return web._get_db()


def load_holdout() -> dict:
    path = _holdout_path()
    if not path.exists():
        return {"_note": None, "seed": None, "groups": {g: [] for g in GROUPS}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("groups", {})
    for g in GROUPS:
        data["groups"].setdefault(g, [])
    return data


def _write_atomic(data: dict) -> None:
    # Same shape as visual_tag_eval._write_atomic / rank_measure's cache: a
    # kill mid-dump must not leave truncated JSON where hand labels used to be.
    path = _holdout_path()
    d = path.parent
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _current_photo_info(photo_ids: list[int]) -> dict[int, dict]:
    """{photo_id: {"description": str|None, "categories": list[str]|None}} —
    the photo's CURRENT state, which may differ from the sample's `evidence`
    if the description was re-run since the sample was drawn."""
    out: dict[int, dict] = {}
    if not photo_ids:
        return out
    with _get_db() as db:
        for i in range(0, len(photo_ids), 500):
            chunk = photo_ids[i:i + 500]
            marks = ",".join("?" * len(chunk))
            rows = db.conn.execute(
                f"SELECT id, description, categories FROM photos WHERE id IN ({marks})",
                chunk).fetchall()
            for pid, desc, raw in rows:
                cats = None
                if raw is not None:
                    try:
                        parsed = json.loads(raw)
                        cats = parsed if isinstance(parsed, list) else None
                    except (TypeError, ValueError):
                        cats = None
                out[int(pid)] = {"description": desc, "categories": cats}
    return out


def _progress(groups: dict[str, list[dict]]) -> dict:
    total = sum(len(items) for items in groups.values())
    done = sum(1 for items in groups.values() for it in items if it.get("label"))
    return {"done": done, "total": total}


@router.get("/items")
def get_items(set: str = Query("holdout")):  # noqa: A002 - query name matches other eval APIs
    if set != "holdout":
        raise HTTPException(400, f"unknown set {set!r}; have: holdout")
    data = load_holdout()
    ids = sorted({int(it["photo_id"]) for items in data["groups"].values() for it in items})
    info = _current_photo_info(ids)

    groups_out: dict[str, list[dict]] = {}
    for g in GROUPS:
        rows = []
        for it in data["groups"].get(g, []):
            pid = int(it["photo_id"])
            cur = info.get(pid)
            rows.append({
                "photo_id": pid,
                "term": it["term"],
                "reason": it.get("reason"),
                "evidence": it.get("evidence"),
                "label": it.get("label"),
                "group": g,
                "description": cur["description"] if cur else None,
                "categories": cur["categories"] if cur else None,
                "in_db": cur is not None,
            })
        groups_out[g] = rows

    return {
        "set": "holdout",
        "seed": data.get("seed"),
        "note": data.get("_note"),
        "groups": groups_out,
        "group_rules": GROUP_RULES,
        "progress": _progress(groups_out),
        "hint": None if ids else _HINT,
        "eval_dir": str(eval_dir()),
    }


class LabelBody(BaseModel):
    photo_id: int
    term: str
    group: str
    label: Optional[str] = None   # "correct" | "wrong" | "debatable" | None (clear)


@router.post("/label")
def put_label(body: LabelBody):
    if body.group not in GROUPS:
        raise HTTPException(400, f"unknown group {body.group!r}; have {GROUPS}")
    if body.label is not None and body.label not in LABELS:
        raise HTTPException(400, f"label must be one of {LABELS} or null")

    data = load_holdout()
    items = data["groups"].get(body.group) or []
    match = next((it for it in items
                  if int(it["photo_id"]) == body.photo_id and it["term"] == body.term), None)
    if match is None:
        raise HTTPException(
            404, f"no such drop: group={body.group!r} photo_id={body.photo_id} term={body.term!r}")

    match["label"] = body.label
    _write_atomic(data)
    return {"photo_id": body.photo_id, "term": body.term, "group": body.group,
            "label": match["label"], "progress": _progress(data["groups"])}


def _group_score(items: list[dict]) -> dict:
    c = {"n": len(items), "labelled": 0, "unlabelled": 0,
         "correct": 0, "wrong": 0, "debatable": 0}
    for it in items:
        lab = it.get("label")
        if lab is None:
            c["unlabelled"] += 1
            continue
        c["labelled"] += 1
        if lab in ("correct", "wrong", "debatable"):
            c[lab] += 1
    judged = c["correct"] + c["wrong"]
    c["precision"] = c["correct"] / judged if judged else None
    c["passes"] = bool(c["precision"] is not None and c["precision"] >= PASS_PRECISION
                       and judged >= MIN_JUDGED)
    # Shown NEXT TO the gate, not instead of it: `passes` is the owner's
    # point-estimate bar. At n=25, 24/25 (96%) has a 95% lower bound of 0.80,
    # so a pass says little about whether the true precision clears 90%.
    c["precision_lower_95"] = wilson_lower(c["correct"], judged)
    c["lower_clears_bar"] = bool(c["precision_lower_95"] is not None
                                 and c["precision_lower_95"] >= PASS_PRECISION)
    return c


def wilson_lower(successes: int, n: int, z: float = 1.96) -> Optional[float]:
    """Lower end of the Wilson score interval (95% by default); None for n=0."""
    if n <= 0:
        return None
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (centre - margin) / (1 + z * z / n)


def compute_score(data: Optional[dict] = None) -> dict:
    """Per-group and overall score from a holdout dict (as `load_holdout`
    returns). Pure — no I/O when `data` is given — so the CLI's `score`
    subcommand and this module's `/score` endpoint can share one answer."""
    data = load_holdout() if data is None else data
    per_group = {g: _group_score(data.get("groups", {}).get(g, []) or []) for g in GROUPS}
    overall_items = [it for g in GROUPS for it in (data.get("groups", {}).get(g, []) or [])]
    overall = _group_score(overall_items)
    return {"per_group": per_group, "overall": overall,
            "bar": PASS_PRECISION, "min_judged": MIN_JUDGED}


@router.get("/score")
def get_score():
    return compute_score()

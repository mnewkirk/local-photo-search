"""API behind the visual-tag labelling page (`/eval/visual-tags`).

The owner hand-labels a sample of photos with the correct PERCEIVED visual
tags; `evals/visual_tags_eval.py` scores the model against those labels. All
storage goes through `visual_tag_eval` — this module adds no format of its own.

**Local-only on purpose: nothing here proxies to the NAS in replica mode.**
Labels are files in `visual_tag_eval.eval_dir()` on whichever machine serves
the page, because `sync-replica.sh` swaps the replica DB wholesale and hand
labels are the one artifact that cannot be regenerated. The DB is only READ,
for the optional "what the model said" reveal.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from . import visual_tag_eval
from .visual_tags_derive import PERCEIVED_GLOSS, PROMPT_SECTIONS

router = APIRouter(prefix="/api/eval/visual-tags", tags=["eval"])
# Serves the Unsplash contact sheets (`evals/visual_tags_unsplash.py sheet`)
# to a browser that is not this machine. The sheets embed thumbs as file://
# paths so they open locally; here those are rewritten to the thumbs route.
sheet_router = APIRouter(prefix="/eval/unsplash", tags=["eval"])
set_ = set  # the handlers take a `set=` query param, which shadows the builtin

_SAMPLE_HINT = ("No sample yet. Draw one with: "
                "python evals/visual_tags_eval.py sample --db <db>")

# SQLite's default SQLITE_MAX_VARIABLE_NUMBER floor; a sample is ~60 photos,
# but nothing stops someone drawing a bigger one.
_ID_CHUNK = 500


# Definitions shown to the LABELLER only, for perceived terms the prompt leaves
# un-glossed. Deliberately NOT in PERCEIVED_GLOSS: that text is the production
# prompt, and the rule is not to change it without re-running the A/B. These
# are the owner's calibrated readings (2026-09-20, on two midday soccer
# frames); the same wording is what a `--prompt-file` variant should test.
# The light on the CLEAREST SUBJECT decides — whoever the focus and the action
# are on, which is not necessarily the largest figure in the frame.
LABELLER_NOTES: dict[str, str] = {
    "harsh-light": ("hard light that leaves the clearest subject's face or body "
                    "mostly in shadow, or glaring; a sunlit scene with the "
                    "subject evenly lit is just sunny"),
    "soft-light": ("diffuse light on the subject, no hard shadow edges (open "
                   "shade, window light, thin cloud); about the subject, not "
                   "the sky"),
    "overexposed": ("important areas blown to featureless white (skin, a jersey, "
                    "most of the sky); not a bright photo, not a small highlight"),
    "backlit": ("labeller note: judge the clearest subject, not a secondary "
                "figure - its camera-facing side is in its own shadow"),
    # 2026-09-26: the owner's own recheck disagreed with itself on `muted`
    # (kappa 0.06). Their reading: the SHADES are subdued, not the light and
    # not the number of hues.
    "muted": ("the colours are subdued: darker, greyer shades with little "
              "contrast between them. About the shades, not the light or the "
              "number of hues - a photo can be muted and colorful"),
    # 2026-09-26: gemma picked `hazy` on a foggy sunrise and the owner read a
    # photographer's "fog" shot as not foggy — the boundary was undefined.
    "foggy": ("fog or mist in the scene: the air itself is visible and nearby "
              "things fade into white or grey"),
    "hazy": ("distance is washed out - far hills or skyline pale and "
             "low-contrast from haze, smog, smoke or dust - while the near "
             "scene stays clear"),
}


class LabelBody(BaseModel):
    yes: List[str] = []
    debatable: List[str] = []
    done: bool = True


def _get_db():
    # Deferred, same as batch_api: web imports this module at import time.
    from . import web
    return web._get_db()


def _vocabulary() -> dict:
    """The perceived terms grouped exactly as the prompt presents them, with
    the same gloss text the model reads — so the labeller and the model are
    judged against one definition of each term, not two."""
    sections = []
    for title, groups in PROMPT_SECTIONS:
        tags = [{"tag": t, "gloss": PERCEIVED_GLOSS.get(t),
                 "note": LABELLER_NOTES.get(t)}
                for group in groups for t in group]
        sections.append({"title": title, "tags": tags})
    if visual_tag_eval.CANDIDATE_TAGS:
        # Trial tags: labelled here, absent from the shipped prompt.
        sections.append({
            "title": "MOMENT - candidate tag, not in the shipped prompt yet:",
            "candidate": True,
            "tags": [{"tag": t, "gloss": None, "note": g}
                     for t, g in visual_tag_eval.CANDIDATE_TAGS.items()]})
    return {"sections": sections}


def _stored_tags(photo_ids: list[int]) -> dict[int, dict]:
    """{photo_id: {filename, stored_tags}} for ids present in the local DB.

    `stored_tags` is None for a photo the model has not tagged (NULL column),
    which is different from `[]` ("tagged: nothing notable")."""
    out: dict[int, dict] = {}
    if not photo_ids:
        return out
    with _get_db() as db:
        for i in range(0, len(photo_ids), _ID_CHUNK):
            chunk = photo_ids[i:i + _ID_CHUNK]
            marks = ",".join("?" * len(chunk))
            rows = db.conn.execute(
                f"SELECT id, filename, visual_tags FROM photos WHERE id IN ({marks})",
                chunk).fetchall()
            for pid, filename, raw in rows:
                tags = None
                if raw is not None:
                    try:
                        parsed = json.loads(raw)
                        tags = parsed if isinstance(parsed, list) else None
                    except (TypeError, ValueError):
                        tags = None
                out[int(pid)] = {"filename": filename, "stored_tags": tags}
    return out


def _label_set(name: str) -> str:
    if name not in visual_tag_eval.LABEL_SETS:
        raise HTTPException(400, f"unknown label set {name!r}")
    return name


@router.get("")
def get_eval(set: str = Query("main")):
    """`set=recheck` serves the blind-relabel subset: the same photos, the
    recheck file's labels only — the first answer is deliberately withheld."""
    label_set = _label_set(set)
    sample = visual_tag_eval.load_sample()
    labels = visual_tag_eval.load_labels(label_set)
    entries = sample.get("photos") or []
    if label_set == "recheck":
        keep = set_(visual_tag_eval.recheck_ids())
        entries = [p for p in entries if int(p["photo_id"]) in keep]
    ids = [int(p["photo_id"]) for p in entries]
    info = _stored_tags(ids)

    photos = []
    for p in entries:
        pid = int(p["photo_id"])
        row = info.get(pid)
        photos.append({
            "photo_id": pid,
            "stratum": p.get("stratum"),
            "label": labels.get(pid),
            "stored_tags": row["stored_tags"] if row else None,
            "filename": row["filename"] if row else None,
            # The sample may have been drawn against a different DB copy.
            "in_db": row is not None,
        })

    done = sum(1 for p in photos if (p["label"] or {}).get("done"))
    return {
        "sample": {"created": sample.get("created"), "seed": sample.get("seed")},
        "photos": photos,
        "vocabulary": _vocabulary(),
        "progress": {"done": done, "total": len(photos)},
        "set": label_set,
        "hint": None if photos else _SAMPLE_HINT,
        "eval_dir": str(visual_tag_eval.eval_dir()),
    }


@router.put("/{photo_id}")
def put_label(photo_id: int, body: LabelBody, set: str = Query("main")):
    label_set = _label_set(set)
    sample_ids = {int(p["photo_id"])
                  for p in visual_tag_eval.load_sample().get("photos") or []}
    if label_set == "recheck":
        sample_ids &= set_(visual_tag_eval.recheck_ids())
    # Only sampled photos: a label outside the sample would sit in labels.json
    # and be scored by the harness against a stratum it was never drawn for.
    if photo_id not in sample_ids:
        raise HTTPException(404, f"photo {photo_id} is not in the eval sample")
    try:
        label = visual_tag_eval.save_label(
            photo_id, body.yes, body.debatable, done=body.done, label_set=label_set)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    labels = visual_tag_eval.load_labels(label_set)
    done = sum(1 for pid in sample_ids if (labels.get(pid) or {}).get("done"))
    return {"photo_id": photo_id, "label": label,
            "progress": {"done": done, "total": len(sample_ids)}}


# Same env var + default as evals/visual_tags_unsplash.py's THUMBS.
def _unsplash_thumbs() -> str:
    return os.environ.get("PHOTOSEARCH_UNSPLASH_THUMBS",
                          os.path.expanduser("~/unsplash-quality-eval/thumbs"))


_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@sheet_router.get("/sheet/{tag}")
def get_unsplash_sheet(tag: str):
    if not _SAFE_NAME.match(tag):
        raise HTTPException(400, "bad tag")
    path = visual_tag_eval.eval_dir() / "unsplash" / f"sheet-{tag}.html"
    if not path.exists():
        raise HTTPException(404, f"No sheet for {tag!r}. Make one with: "
                            f"python evals/visual_tags_unsplash.py sheet --tag {tag}")
    page = path.read_text(encoding="utf-8").replace(
        "file://" + _unsplash_thumbs().rstrip("/") + "/", "/eval/unsplash/thumbs/")
    return HTMLResponse(page, headers={"Cache-Control": "no-cache"})


@sheet_router.get("/thumbs/{name}")
def get_unsplash_thumb(name: str):
    stem, ext = os.path.splitext(name)
    if ext != ".jpg" or not _SAFE_NAME.match(stem):
        raise HTTPException(400, "bad thumb name")
    path = os.path.join(_unsplash_thumbs(), name)
    if not os.path.isfile(path):
        raise HTTPException(404, "no such thumb")
    return FileResponse(path, media_type="image/jpeg")

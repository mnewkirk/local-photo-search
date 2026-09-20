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
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import visual_tag_eval
from .visual_tags_derive import PERCEIVED_GLOSS, PROMPT_SECTIONS

router = APIRouter(prefix="/api/eval/visual-tags", tags=["eval"])

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


@router.get("")
def get_eval():
    sample = visual_tag_eval.load_sample()
    labels = visual_tag_eval.load_labels()
    entries = sample.get("photos") or []
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
        "hint": None if photos else _SAMPLE_HINT,
        "eval_dir": str(visual_tag_eval.eval_dir()),
    }


@router.put("/{photo_id}")
def put_label(photo_id: int, body: LabelBody):
    sample_ids = {int(p["photo_id"])
                  for p in visual_tag_eval.load_sample().get("photos") or []}
    # Only sampled photos: a label outside the sample would sit in labels.json
    # and be scored by the harness against a stratum it was never drawn for.
    if photo_id not in sample_ids:
        raise HTTPException(404, f"photo {photo_id} is not in the eval sample")
    try:
        label = visual_tag_eval.save_label(
            photo_id, body.yes, body.debatable, done=body.done)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    labels = visual_tag_eval.load_labels()
    done = sum(1 for pid in sample_ids if (labels.get(pid) or {}).get("done"))
    return {"photo_id": photo_id, "label": label,
            "progress": {"done": done, "total": len(sample_ids)}}

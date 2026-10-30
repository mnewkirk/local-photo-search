"""DB side of the people-bind pass (docs/plans/description-names.md).

The pure logic lives in :mod:`people_bind`; this module reads the claim
payload, answers "what is true NOW" for the submit-time re-validation, and
writes the row. Nothing here ever reads a person's name or id: the pass is
label-free.
"""

from __future__ import annotations

import json
from typing import Optional

from .people_bind import BIND_VERSION, offered_faces, sha_of


def claim_input(conn, photo_id: int) -> Optional[dict]:
    """The worker's input for one claimed photo, or ``None`` when it has no
    description. Faces are the ones that would be DRAWN (quality floor,
    labelled or not); only their ids, boxes and detection scores travel."""
    row = conn.execute(
        "SELECT description, image_width, image_height, subject_boxes "
        "FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if row is None or not row[0]:
        return None
    faces = [{"id": r[0], "bbox_top": r[1], "bbox_right": r[2], "bbox_bottom": r[3],
              "bbox_left": r[4], "det_score": r[5]}
             for r in conn.execute(
                 "SELECT id, bbox_top, bbox_right, bbox_bottom, bbox_left, det_score "
                 "FROM faces WHERE photo_id = ? ORDER BY id", (photo_id,))]
    boxes = None
    if row[3]:
        try:
            boxes = json.loads(row[3])
        except ValueError:
            boxes = None
    return {"description": row[0], "description_sha": sha_of(row[0]),
            "faces": offered_faces(faces), "image_width": row[1] or 0,
            "image_height": row[2] or 0, "subject_boxes": boxes}


def current_state(conn, photo_id: int) -> Optional[tuple]:
    """``(description, {face ids})`` as of NOW, or ``None`` if the photo is gone."""
    row = conn.execute("SELECT description FROM photos WHERE id = ?",
                       (photo_id,)).fetchone()
    if row is None:
        return None
    ids = {r[0] for r in conn.execute("SELECT id FROM faces WHERE photo_id = ?",
                                      (photo_id,))}
    return row[0], ids


def store_row(conn, photo_id: int, description_sha: str, slots: list, method: str,
              model: Optional[str], model_version: Optional[str]) -> str:
    """Upsert the photo's row (``[]`` = "looked, nothing bindable", a result).
    Returns the slots JSON that was stored."""
    slots_json = json.dumps(slots, separators=(",", ":"))
    conn.execute(
        "INSERT OR REPLACE INTO description_subjects "
        "(photo_id, description_sha, slots_json, method, bind_version, model, "
        "model_version) VALUES (?,?,?,?,?,?,?)",
        (photo_id, description_sha, slots_json, method, BIND_VERSION, model,
         model_version))
    return slots_json

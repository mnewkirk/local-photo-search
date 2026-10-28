"""When a person counts as PRESENT in a video (owner decision 2026-10-28).

A person is present only with at least ``VIDEO_PRESENCE_MIN_S`` seconds of
tracked face presence in the clip — ``video_tracks.presence_s`` summed over
that person's identities (one person can be split across several tracks; two
tracks of one person never overlap in time, so the sum is screen time).

Applies to the eval's presence list, ``GET /api/videos/{id}/subjects``'s
``present`` list (and the modal's person rows), and the video branch of the
person search filter. The ``faces`` rows themselves are unchanged — a
sub-2 s identity keeps its row for /faces review; it just does not make the
video match ``person=X``.

Why 2 s: an identity can earn a ``faces`` row at 0.5 s (``face_row_eligible``),
which is a passer-by. Round 2 of the eval (docs/plans/
video-eval-round2-2026-10-27.md) found named-presence precision far below the
gate; a duration floor was the lever that moved it.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


VIDEO_PRESENCE_MIN_S = _env_float("PHOTOSEARCH_VIDEO_PRESENCE_MIN_S", 2.0)


def present_people(identities: Iterable[dict], min_s: Optional[float] = None) -> list:
    """Sorted person ids with summed ``presence_s`` >= ``min_s``.
    ``identities``: dicts carrying ``person_id`` and ``presence_s``."""
    floor = VIDEO_PRESENCE_MIN_S if min_s is None else min_s
    tot: dict = {}
    for i in identities or []:
        pid = i.get("person_id")
        if pid is None:
            continue
        tot[pid] = tot.get(pid, 0.0) + float(i.get("presence_s") or 0.0)
    return sorted(pid for pid, s in tot.items() if s >= floor - 1e-9)


def present_face_sql(alias: str = "f") -> str:
    """SQL predicate on a faces row ``alias``: true for a STILL face (no
    ``video_tracks`` row), or a video face whose person has at least
    ``VIDEO_PRESENCE_MIN_S`` of summed presence in that video.

    Index-friendly by construction: one idx_video_tracks_face seek for a
    still face; for a video face, idx_faces_photo_person on
    (photo_id, person_id) then one idx_video_tracks_face seek per identity.
    The floor is inlined as a literal (it is a float we parsed), so callers
    need no extra bound parameter and their placeholder order is unchanged.
    A NULL ``presence_s`` counts 0 (no stored row has one)."""
    floor = repr(float(VIDEO_PRESENCE_MIN_S - 1e-9))
    a = alias
    return (f"(NOT EXISTS (SELECT 1 FROM video_tracks vt WHERE vt.face_id = {a}.id) "
            f"OR (SELECT TOTAL(vt2.presence_s) FROM faces fp "
            f"JOIN video_tracks vt2 ON vt2.face_id = fp.id "
            f"WHERE fp.photo_id = {a}.photo_id AND fp.person_id = {a}.person_id) "
            f">= {floor})")

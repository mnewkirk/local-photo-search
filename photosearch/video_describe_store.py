"""Server side of the ``video-describe`` pass (docs/plans/video-support.md
step 5).

Pure SQL on the NAS. The desktop worker (``worker._process_video_describe``)
fetches the K keyframes video-scan uploaded (``GET /api/videos/{id}/frames/
keyframe_kNN.jpg``), calls ``describe.describe_video`` (role ``video``) once per
clip and submits a ``VideoDescribeResult``; ``worker_api.submit_results``
writes ``photos.description``, logs a ``generations`` row (``text_type=
'describe'``, the effective model) and calls
``PhotoDB.invalidate_description_dependents`` so ``category-content`` /
``keywords`` (PASS_MEDIA "any" since this step) run on the video next.
``verify`` stays still-only (D18).

The claim response carries everything the worker needs (:func:`describe_input`)
— the keyframe names and times of the CURRENT scan, the duration, and the
video-subjects timeline with PERSON NAMES resolved here, at claim time, through
``video_tracks.ident_ix -> faces.person_id -> persons.name``. Names are never
stored by the subjects pass (a relabel in /faces must move them), so the
summary is rebuilt from ``subjects_json`` on every claim. A video without a
subjects result is described without one: ``video-describe`` depends on
``video-scan`` only, never on ``video-subjects``.
"""

from __future__ import annotations

import json
from typing import Optional


def _scan_ok_sql(alias: str) -> str:
    """The CURRENT scan landed cleanly on a non-Live clip — the only videos
    that have keyframes to describe."""
    from .video_scan_store import SCAN_VERSION
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.scan_version >= {int(SCAN_VERSION)} AND vm.scan_error IS NULL "
            f"AND COALESCE(vm.kind, '') != 'live')")


def claimable_sql(alias: str = "p") -> str:
    """Cleanly scanned, not Live, no description yet. Shared by the claim
    path (db._VIDEO_CLAIM_SQL) and batch_state so the two cannot drift."""
    return f"({_scan_ok_sql(alias)} AND {alias}.description IS NULL)"


def eligible_sql(alias: str = "p") -> str:
    """Videos video-scan is finished with (same gate as video-subjects): an
    unprobed or still-unscanned video is not eligible yet, so the step reads
    waiting on video-scan rather than done. Scan errors, unreadable and Live
    videos are eligible and never claimable, so they count done."""
    from .video_subjects_store import eligible_sql as settled
    return settled(alias)


def never_described_sql(alias: str = "p") -> str:
    """A video that has no description and never will: video-scan is settled
    on it and it is NOT describable (Live, unreadable, a stored scan error).
    batch_state drops these from the text passes' ``total`` — otherwise one
    unplayable clip would hold category-content / keywords short of done."""
    return (f"({alias}.media_type = 'video' AND {alias}.description IS NULL "
            f"AND {eligible_sql(alias)} AND NOT {_scan_ok_sql(alias)})")


def _loads(v, default=None):
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def subject_summary(conn, photo_id: int, analysis: Optional[dict] = None) -> Optional[str]:
    """``"0.0–12.0 s dog, 12.0–40.0 s Calvin"`` from the video-subjects result,
    with each identity's CURRENT person name (an unnamed face reads "an
    unnamed person"). None when the subjects pass has not run, or found no
    timeline."""
    from .video_analysis import format_summary, summarize_runs
    if analysis is None:
        row = conn.execute("SELECT subjects_version, subjects_json FROM video_meta "
                           "WHERE photo_id = ?", (photo_id,)).fetchone()
        if row is None or row[0] is None:
            return None
        analysis = _loads(row[1], {}) or {}
    segs = [tuple(s) for s in (analysis.get("segments") or [])]
    prim = list(analysis.get("primaries") or [])
    if not segs or len(prim) != len(segs) or all(p is None for p in prim):
        return None
    subjects = {s.get("ix"): s for s in (analysis.get("subjects") or [])
                if isinstance(s, dict)}
    names = {int(r[0]): r[1] for r in conn.execute(
        "SELECT vt.ident_ix, pe.name FROM video_tracks vt "
        "JOIN faces f ON f.id = vt.face_id JOIN persons pe ON pe.id = f.person_id "
        "WHERE vt.photo_id = ? AND vt.ident_ix IS NOT NULL", (photo_id,))}

    def label(k):
        s = subjects.get(k)
        if s is None:
            return None
        fix = s.get("face_ix")
        if fix is not None:
            return names.get(int(fix)) or "an unnamed person"
        if s.get("category") == "person":
            return "an unidentified person"
        return s.get("label")

    text = format_summary(summarize_runs(segs, prim), label)
    return text or None


def describe_input(conn, photo_id: int) -> Optional[dict]:
    """What the worker needs to describe one video: the scan version (the
    keyframes' asset directory), the K keyframes ``[{name, t}]`` of the current
    scan, the duration, and the named subjects timeline (None without a
    subjects result). None when the video has no clean current scan."""
    vm = conn.execute(
        "SELECT scan_version, scan_error, quality_json, duration_s, subjects_version, "
        "subjects_json FROM video_meta WHERE photo_id = ?", (photo_id,)).fetchone()
    if vm is None or vm[0] is None or vm[1] is not None:
        return None
    q = _loads(vm[2], {}) or {}
    frames = q.get("frames") or {}
    keyframes = [{"name": k["name"], "t": float(k.get("t") or 0.0)}
                 for k in (frames.get("keyframes") or []) if k.get("name")]
    summary = None
    if vm[4] is not None:
        summary = subject_summary(conn, photo_id, _loads(vm[5], {}) or {})
    return {
        "scan_version": int(vm[0]),
        "duration": q.get("duration") or vm[3],
        "keyframes": keyframes,
        "subject_summary": summary,
    }

"""Server side of the ``video-subjects`` pass (docs/plans/video-support.md
step 3b).

Pure SQL on the NAS — no model, no decoder. The desktop worker
(``worker._process_video_subjects``) grounds one keyframe per segment through
the VLM (``subjects.ground_frame_subjects``), scores the per-segment answers
with :func:`video_subjects.score_subjects` and submits one
``VideoSubjectsResult`` per video; ``worker_api.submit_results`` calls
:func:`apply_subjects_result` inside one transaction per video.

What is stored:

* ``video_subjects`` — one row per linked subject (person identity, an
  unidentified person, a dog, a mountain lake …): label, category,
  ``face_id`` (the video-scan ``faces`` row of the joined identity, NULL for a
  non-person or an unidentified person), ``focus_s``, ``share`` (whole-clip
  denominator, D10), intervals and the representative box.
* ``video_meta.subjects_version`` / ``subjects_at`` — the marker; NULL or
  stale is claimable. The scan store clears it (and the rows) on every
  re-scan, since the subjects reference the scan's identities and keyframes.
* ``video_meta.subjects_json`` — the per-segment groundings, primaries
  timeline and parameters, so a re-score needs no VLM call.
* one ``generations`` row (``text_type='subjects'``) with the effective model.

**Person names are NEVER stored here** — they are read through
``faces.person_id`` at query time (:func:`read_subjects`), so a relabel in
``/faces`` moves the subject's name and share with no recompute.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

# Bump to re-run the subjects pass over every scanned video (claim predicate:
# subjects_version IS NULL OR <).
SUBJECTS_VERSION = 1

# A video whose keyframes failed (unusable VLM answer, unreadable keyframe)
# on MORE than this fraction of segments is a failures row (one attempt
# spent); at or below it, the failed segments count as "no subject" (they
# stay in the whole-clip denominator) and the result is stored.
MAX_FAILED_SEGMENT_FRAC = 0.20


class StaleSubjects(RuntimeError):
    """The result was computed from a scan that has since been replaced (or
    errored). Never written, never an attempt: the claim predicate offers the
    video again against the current scan."""


class NotAVideo(ValueError):
    pass


# ---------------------------------------------------------------------------
# Claim predicate (shared with db.get_unprocessed_photos and batch_state)
# ---------------------------------------------------------------------------

def claimable_sql(alias: str = "p") -> str:
    """The CURRENT scan landed without an error, the clip is not a paired
    Live clip (D3), and no current subjects result exists."""
    from .video_scan_store import SCAN_VERSION
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.scan_version >= {int(SCAN_VERSION)} AND vm.scan_error IS NULL "
            f"AND COALESCE(vm.kind, '') != 'live' "
            f"AND (vm.subjects_version IS NULL OR vm.subjects_version < {int(SUBJECTS_VERSION)}))")


def eligible_sql(alias: str = "p") -> str:
    """Videos video-scan is finished with: probed and no longer
    scan-claimable (scanned, a stored scan error, unreadable, or Live). An
    unprobed or still-unscanned video is not eligible yet — batch_state reads
    the step as waiting on video-scan, not done."""
    from .video_scan_store import claimable_sql as scan_claimable, probed_sql
    return f"({probed_sql(alias)} AND NOT {scan_claimable(alias)})"


# ---------------------------------------------------------------------------
# Worker input (claim response + the M28 sync re-run)
# ---------------------------------------------------------------------------

def _loads(v, default=None):
    if v is None:
        return default
    if isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


def subjects_input(conn, photo_id: int) -> Optional[dict]:
    """What the worker needs to ground and score one video: the scan version
    (the keyframes' asset directory), frame size, the segment manifest with
    keyframe names/times and per-segment face boxes, and the face tracks.
    None when the video has no current scan."""
    vm = conn.execute(
        "SELECT scan_version, scan_error, quality_json, duration_s, width, height, "
        "scan_generation FROM video_meta WHERE photo_id = ?", (photo_id,)).fetchone()
    if vm is None or vm[0] is None or vm[1] is not None:
        return None
    q = _loads(vm[2], {}) or {}
    frames = q.get("frames") or {}
    tracks = [{"ident_ix": r[0], "face_id": r[1],
               "presence_intervals": _loads(r[2], []) or []}
              for r in conn.execute(
                  "SELECT ident_ix, face_id, presence_intervals FROM video_tracks "
                  "WHERE photo_id = ? ORDER BY ident_ix", (photo_id,))]
    return {
        "scan_version": int(vm[0]),
        # The staged upload generation (schema v38): a clear-pass re-scan at
        # the SAME version gets a new one, with new identities and keyframes.
        "scan_generation": vm[6],
        "duration": q.get("duration") or vm[3],
        "W": q.get("width") or vm[4],
        "H": q.get("height") or vm[5],
        "sample_fps": q.get("sample_fps"),
        "segments": list(frames.get("segments") or []),
        "tracks": tracks,
    }


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(v) -> Optional[str]:
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def apply_subjects_result(conn, r: dict) -> dict:
    """Write one subjects result (a ``VideoSubjectsResult`` as a dict).

    Does NOT commit; any sqlite error propagates (the caller defers a
    transient lock and spends the attempt otherwise). Raises
    :class:`StaleSubjects` when the scan the worker read is no longer the
    current one."""
    pid = int(r["photo_id"])
    row = conn.execute("SELECT media_type FROM photos WHERE id = ?", (pid,)).fetchone()
    if row is None:
        raise NotAVideo(f"photo {pid} does not exist")
    if row[0] != "video":
        raise NotAVideo(f"photo {pid} is not a video")
    vm = conn.execute("SELECT scan_version, scan_error, scan_generation FROM video_meta "
                      "WHERE photo_id = ?", (pid,)).fetchone()
    if vm is None or vm[0] is None or vm[1] is not None or int(vm[0]) != int(r["scan_version"]):
        raise StaleSubjects(f"video {pid}: scanned at {vm[0] if vm else None}, the "
                            f"subjects were computed from scan {r['scan_version']}")
    # Same version, different generation = a clear-pass re-scan landed while
    # this result was in flight: its face_ix would map onto the NEW tracks.
    # An older worker sends no generation and is only version-checked.
    if r.get("scan_generation") is not None and r["scan_generation"] != vm[2]:
        raise StaleSubjects(f"video {pid}: re-scanned (generation {vm[2]}) since the "
                            f"subjects were computed from generation {r['scan_generation']}")

    ident_face = {int(t[0]): t[1] for t in conn.execute(
        "SELECT ident_ix, face_id FROM video_tracks WHERE photo_id = ? "
        "AND ident_ix IS NOT NULL", (pid,))}

    conn.execute("DELETE FROM video_subjects WHERE photo_id = ?", (pid,))
    n = 0
    unmapped = []
    for s in r.get("subjects") or []:
        fix = s.get("face_ix")
        face_id = None
        if fix is not None:
            # An identity below the faces-row gate has a track with face_id
            # NULL: the subject is kept, unlinked ("unidentified person") —
            # that is the gate working, not an unmapped index.
            if int(fix) not in ident_face:
                unmapped.append(int(fix))
            face_id = ident_face.get(int(fix))
        conn.execute(
            "INSERT INTO video_subjects (photo_id, subject_ix, label, category, face_id, "
            "focus_s, focus_intervals, rep_t, rep_frame, rep_bbox, n_segments, share, "
            "rep_seg) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, int(s["ix"]), s.get("label"), s.get("category"), face_id,
             s.get("focus_s"), _json(s.get("focus_intervals")), s.get("rep_t"),
             s.get("rep_frame"), _json(s.get("rep_bbox")), s.get("n_segments"),
             s.get("share"), s.get("rep_seg")))
        n += 1

    version = int(r.get("subjects_version") or SUBJECTS_VERSION)
    analysis = dict(r.get("analysis") or {})
    analysis.setdefault("subjects_version", version)
    analysis["model"] = r.get("model")
    conn.execute(
        "UPDATE video_meta SET subjects_version = ?, subjects_at = ?, subjects_json = ? "
        "WHERE photo_id = ?", (version, _now(), _json(analysis), pid))
    # Provenance: one generations row per video. Written directly, not via
    # PhotoDB.log_generation, which commits outside batch mode — this must
    # land in the same transaction as the rows above.
    conn.execute(
        "INSERT INTO generations (photo_id, text_type, generated_text, model_used, "
        "model_version) VALUES (?, 'subjects', ?, ?, ?)",
        (pid, _json(analysis), r.get("model"), r.get("model_version")))
    return {"photo_id": pid, "subjects": n, "unmapped_face_ix": unmapped}


# ---------------------------------------------------------------------------
# Read (GET /api/videos/{id}/subjects — step 6's UI)
# ---------------------------------------------------------------------------

def read_subjects(conn, photo_id: int) -> Optional[dict]:
    """Subjects with person names resolved through ``faces.person_id`` at
    read time, the per-segment primaries timeline, and each person's dense
    face-track presence. None when the row is not a video."""
    from .video_analysis import summarize_runs

    p = conn.execute("SELECT media_type FROM photos WHERE id = ?", (photo_id,)).fetchone()
    if p is None or p[0] != "video":
        return None
    vm = conn.execute(
        "SELECT duration_s, scan_version, subjects_version, subjects_at, subjects_json "
        "FROM video_meta WHERE photo_id = ?", (photo_id,)).fetchone()
    analysis = _loads(vm[4], {}) if vm else {}
    out = {"photo_id": photo_id,
           "duration": (analysis or {}).get("duration") or (vm[0] if vm else None),
           "scan_version": vm[1] if vm else None,
           "subjects_version": vm[2] if vm else None,
           "subjects_at": vm[3] if vm else None,
           "model": (analysis or {}).get("model"),
           "credit": (analysis or {}).get("credit"),
           "subjects": [], "people": [], "segments": [], "timeline": []}
    if not vm or vm[2] is None:
        return out

    presence = {r[0]: {"ident_ix": r[1], "presence_s": r[2],
                       "presence_intervals": _loads(r[3], []) or []}
                for r in conn.execute(
                    "SELECT face_id, ident_ix, presence_s, presence_intervals "
                    "FROM video_tracks WHERE photo_id = ? AND face_id IS NOT NULL",
                    (photo_id,))}
    rows = conn.execute(
        "SELECT vs.subject_ix, vs.label, vs.category, vs.face_id, vs.focus_s, vs.share, "
        "vs.focus_intervals, vs.rep_t, vs.rep_frame, vs.rep_seg, vs.rep_bbox, "
        "vs.n_segments, f.person_id, pe.name "
        "FROM video_subjects vs LEFT JOIN faces f ON f.id = vs.face_id "
        "LEFT JOIN persons pe ON pe.id = f.person_id "
        "WHERE vs.photo_id = ? ORDER BY vs.subject_ix", (photo_id,)).fetchall()
    by_ix = {}
    people: dict = {}
    for r in rows:
        (ix, label, cat, face_id, focus_s, share, fiv, rep_t, rep_frame, rep_seg,
         rep_bbox, n_seg, person_id, name) = tuple(r)
        if face_id is not None:
            display = name or "unnamed person"
        elif cat == "person":
            display = "unidentified person"
        else:
            display = label
        pres = presence.get(face_id) or {}
        s = {"ix": ix, "label": label, "category": cat, "face_id": face_id,
             "person_id": person_id, "person_name": name, "display": display,
             "focus_s": focus_s, "share": share,
             "focus_intervals": _loads(fiv, []) or [], "rep_t": rep_t,
             "rep_frame": rep_frame, "rep_seg": rep_seg, "rep_bbox": _loads(rep_bbox),
             "n_segments": n_seg, "presence_s": pres.get("presence_s"),
             "presence_intervals": pres.get("presence_intervals")}
        out["subjects"].append(s)
        by_ix[ix] = s
        if person_id is not None:
            g = people.setdefault(person_id, {"person_id": person_id, "name": name,
                                              "focus_s": 0.0, "share": 0.0,
                                              "face_ids": []})
            g["focus_s"] += focus_s or 0.0
            g["share"] += share or 0.0
            g["face_ids"].append(face_id)
    out["people"] = sorted(
        ({**g, "focus_s": round(g["focus_s"], 3), "share": round(g["share"], 4)}
         for g in people.values()), key=lambda g: -g["focus_s"])

    segs = [tuple(x) for x in (analysis.get("segments") or [])]
    prim = analysis.get("primaries") or []
    out["segments"] = [{"start": a, "end": b,
                        "subject_ix": prim[i] if i < len(prim) else None}
                       for i, (a, b) in enumerate(segs)]
    if segs and len(prim) == len(segs):
        for run in summarize_runs(segs, prim):
            k = run.get("key")
            s = by_ix.get(k) if k is not None else None
            out["timeline"].append({"start": run["start"], "end": run["end"],
                                    "subject_ix": k,
                                    "display": s["display"] if s else None})
    return out

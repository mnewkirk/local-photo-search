"""Server side of the ``video-scan`` pass (docs/plans/video-support.md step 3).

Pure SQL + file copies on the NAS — no decoder (D19). The desktop worker
(``video_scan.py`` / ``worker._process_video_scan``) uploads the keyframes,
per-segment keyframes, representative frames and face crops through
``POST /api/videos/{id}/assets`` FIRST, then submits one ``VideoScanResult``;
``worker_api.submit_results`` calls :func:`apply_scan_result` inside one
transaction per video, commits, then :func:`install_crops`.

**One ``faces`` row per within-video identity that passes the gate**
(``video_analysis.face_row_eligible``: >= 3 detections, or >= 0.5 s present
with >= 2); its encoding is the identity's mean top-k ArcFace embedding.
EVERY identity gets a ``video_tracks`` row; one below the gate has
``face_id`` NULL (no faces row, no encoding), so presence, subject joins and
the timeline still see it while /faces and recluster do not. A carried human
label (below) promotes a gated identity to a faces row. So ``/faces``, clustering, the strict
matcher, verify-labels and the rival test apply to videos unchanged, and a face
crop is served from the uploaded crop (or the rep frame) — never from the
video file.

**Coordinate systems — the one place this is written down.**

* ``faces`` bbox (top, right, bottom, left) of a video face: **native display
  pixels** of the decoded, rotation-applied frame — the same space as
  ``photos.image_width`` / ``image_height`` (the probe's display dims), exactly
  as for a still. So ``tools`` in-frame / prominence tests and the clustering
  ``min_bbox_edge`` gate read a 4K video's faces at full size.
* The representative frame JPEG (``keyframe_rep_f{rep_frame:05d}.jpg``) is
  stored at <= 1920 px: anything drawing that bbox on it scales by
  ``rep_jpeg_width / image_width`` (``/api/faces/crop``'s fallback does).
* ``quality_json["frames"]["segments"][i]["faces"]`` and
  ``quality_json["identities"]``: decoded native display pixels too
  (``x1, y1, x2, y2``), ``quality_json["width"/"height"]`` the frame size.
* ``video_subjects.rep_bbox`` and the grounding boxes in ``subjects_json``:
  normalised 0-1 of the segment keyframe (resolution-free).

**Staged uploads.** Assets upload into a fresh *generation*,
``video_frames/{pid}/v{SCAN_VERSION}/g{generation}/``, and the posters are not
installed by the upload. Only the committed submit points
``video_meta.scan_generation`` at it, installs the posters and bumps
``poster_version``; a failed or deferred re-scan leaves the live files alone.
``cleanup-orphans`` removes generations no row points at (after a day, so an
in-flight scan's staging dir is not swept from under it).

**A re-scan replaces everything** (old ``video_tracks``, old ``faces`` and their
vec0 ``face_encodings`` rows, the CLIP embedding) **but carries human labels**
(D13): an old face whose ``match_source`` is ``manual`` / ``merge_review`` /
``rejected`` is matched one-to-one, greedily by ArcFace L2 (<= 0.6), to a new
identity, which inherits its ``person_id`` / ``match_source`` and its
face/person exclusions. Labels that find no partner are returned in the
response and logged — never silently dropped.

Asset naming (shared with the worker; the asset endpoint prefixes the kind):

==========================  ==============================================
asset ``name`` (kind)       stored as ``video_frames/{pid}/v{ver}/g{gen}/...``
==========================  ==============================================
``k00.jpg`` (keyframe)      ``keyframe_k00.jpg`` — the K keyframes, 1024 px
``s000.jpg`` (keyframe)     ``keyframe_s000.jpg`` — one per
                            ``video_analysis.segment_schedule`` segment, 768 px
``rep_f00012.jpg`` (kf)     ``keyframe_rep_f00012.jpg`` — rep frame of an
                            identity (sampled-frame index 12), <= 1920 px
``i000_120.jpg`` (crop)     ``crop_i000_120.jpg`` — identity 0's face crop
==========================  ==============================================

``quality_json["frames"]`` carries the manifest (names + times + segment
bounds), which is how ``video-subjects`` (step 3b) finds its keyframes:
``GET /api/videos/{id}/frames/keyframe_s000.jpg``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bump to re-scan every video (claim predicate: scan_version IS NULL OR <).
# Also the version directory the scan's assets upload into.
SCAN_VERSION = 1

# D13: which human decisions survive a re-scan, and how close the new
# identity's encoding must be to the old face's to inherit it.
CARRY_LABEL_SOURCES = ("manual", "merge_review", "rejected")
CARRY_MAX_L2 = 0.6

# Face-crop sizes uploaded per identity — the sizes /api/faces/crop serves
# by default and warm-face-crops pre-renders (face_crop.DEFAULT_CROP_SIZES).
CROP_SIZES = (120, 200)


class NotAVideo(ValueError):
    pass


# ---------------------------------------------------------------------------
# Asset names
# ---------------------------------------------------------------------------

def keyframe_name(i: int) -> str:
    return f"k{int(i):02d}.jpg"


def segment_name(i: int) -> str:
    return f"s{int(i):03d}.jpg"


def rep_name(frame_ix: int) -> str:
    return f"rep_f{int(frame_ix):05d}.jpg"


def crop_name(ident_ix: int, size: int) -> str:
    return f"i{int(ident_ix):03d}_{int(size)}.jpg"


def stored_name(kind: str, name: str) -> str:
    """The file name the asset endpoint writes (video_api.store_assets)."""
    return f"{kind}_{name}"


def frames_dir(frames_root, photo_id: int, version: int,
               generation: Optional[str] = None) -> Path:
    d = Path(frames_root) / str(int(photo_id)) / f"v{int(version)}"
    return d / f"g{generation}" if generation else d


# Unpromoted staging generations younger than this are left alone by
# cleanup-orphans: they may belong to a scan whose submit is still coming.
STAGING_GRACE_S = 24 * 3600


# ---------------------------------------------------------------------------
# Claim predicate (shared with db.get_unprocessed_photos and batch_state)
# ---------------------------------------------------------------------------

def claimable_sql(alias: str = "p") -> str:
    """The current probe ran and found a playable clip (or a ZIP-wrapped one,
    which the scan unwraps), it is not a paired Live clip, and the scan is
    missing or stale."""
    from .video_probe import PROBE_VERSION, ZIP_WRAPPED
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.probe_version >= {int(PROBE_VERSION)} "
            f"AND (vm.probe_error IS NULL OR vm.probe_error LIKE '{ZIP_WRAPPED}%') "
            f"AND COALESCE(vm.kind, '') != 'live' "
            f"AND (vm.scan_version IS NULL OR vm.scan_version < {int(SCAN_VERSION)}))")


def probed_sql(alias: str = "p") -> str:
    """The current probe has run (verdict or not) — what video-scan waits on."""
    from .video_probe import PROBE_VERSION
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.probe_version >= {int(PROBE_VERSION)})")


def scan_error_sql(alias: str = "p") -> str:
    """A stored, current scan error (cv2 could not open the complete file)."""
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.scan_error IS NOT NULL AND vm.scan_version >= {int(SCAN_VERSION)})")


# ---------------------------------------------------------------------------
# D13 carry-over
# ---------------------------------------------------------------------------

def match_labels(old: list[dict], new_encodings: list[Optional[list]],
                 max_l2: float = CARRY_MAX_L2) -> tuple[dict[int, dict], list[dict]]:
    """Greedy one-to-one match of labelled old faces to new identities.

    ``old``: ``[{face_id, person_id, match_source, encoding}]`` (labelled only).
    ``new_encodings``: one per new identity (None = cannot match).
    Returns ``({new_ix: old_face_dict}, [unmatched old dicts])``. Pairs are
    taken in ascending L2 order; each side is used at most once.
    """
    pairs = []
    for oi, o in enumerate(old):
        oe = o.get("encoding")
        if oe is None:
            continue
        a = np.asarray(oe, dtype=np.float64)
        for ni, ne in enumerate(new_encodings):
            if ne is None:
                continue
            d = float(np.linalg.norm(a - np.asarray(ne, dtype=np.float64)))
            if d <= max_l2 + 1e-9:
                pairs.append((d, oi, ni))
    pairs.sort()
    used_o, used_n, out = set(), set(), {}
    for d, oi, ni in pairs:
        if oi in used_o or ni in used_n:
            continue
        used_o.add(oi)
        used_n.add(ni)
        out[ni] = dict(old[oi], l2=round(d, 4))
    unmatched = [o for oi, o in enumerate(old) if oi not in used_o]
    return out, unmatched


def _old_faces(conn, pid: int) -> list[dict]:
    from .db import FACE_DIMENSIONS, HAS_SQLITE_VEC, _deserialize_float_list
    rows = conn.execute(
        "SELECT id, person_id, match_source FROM faces WHERE photo_id = ?",
        (pid,)).fetchall()
    out = []
    for r in rows:
        enc = None
        if HAS_SQLITE_VEC:
            er = conn.execute("SELECT encoding FROM face_encodings WHERE face_id = ?",
                              (r[0],)).fetchone()
            if er is not None and er[0] is not None:
                enc = list(_deserialize_float_list(er[0], FACE_DIMENSIONS))
        excl = [x[0] for x in conn.execute(
            "SELECT person_id FROM face_person_exclusions WHERE face_id = ?", (r[0],))]
        out.append({"face_id": r[0], "person_id": r[1], "match_source": r[2],
                    "encoding": enc, "exclusions": excl})
    return out


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json(v) -> Optional[str]:
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def _insert_track(conn, pid: int, face_id: Optional[int], ix: int, ident: dict) -> None:
    conn.execute(
        "INSERT INTO video_tracks (photo_id, face_id, ident_ix, t_start, t_end, n_det, "
        "presence_s, focus_s, focus_intervals, rep_t, rep_frame, mean_salience, "
        "presence_intervals) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, face_id, ix, ident.get("t_start"), ident.get("t_end"), ident.get("n_det"),
         ident.get("presence_s"), ident.get("focus_s"),
         _json(ident.get("focus_intervals")), ident.get("rep_t"),
         ident.get("rep_frame"), ident.get("mean_salience"),
         _json(ident.get("presence_intervals"))))


def apply_scan_result(conn, r: dict, *, crop_cache_dir: Optional[str] = None,
                      frames_root=None) -> dict:
    """Write one scan result (a ``VideoScanResult`` as a dict). Does NOT
    commit; any sqlite error propagates (the caller defers a transient lock
    and spends the attempt otherwise).

    Returns a summary dict; ``_install`` lists the ``(src, dst)`` crop copies
    the caller performs after the commit (:func:`install_crops`).
    """
    from .db import HAS_SQLITE_VEC, _serialize_float_list, record_face_person_exclusions
    from .video_analysis import face_row_eligible

    pid = int(r["photo_id"])
    row = conn.execute("SELECT media_type, file_hash FROM photos WHERE id = ?",
                       (pid,)).fetchone()
    if row is None:
        raise NotAVideo(f"photo {pid} does not exist")
    if row[0] != "video":
        raise NotAVideo(f"photo {pid} is not a video")
    version = int(r.get("scan_version") or SCAN_VERSION)
    now = _now()
    conn.execute("INSERT OR IGNORE INTO video_meta (photo_id) VALUES (?)", (pid,))
    out: dict = {"photo_id": pid, "scan_error": None, "faces": 0, "gated": 0, "carried": [],
                 "unmatched_labels": [], "duplicate_of": None, "_install": []}

    # A ZIP-wrapped file's inner probe: dates/dims/codec come from the inner
    # video, and probe_error ('zip_wrapped') is cleared. Same writer as the
    # probe pass, so the date policy and Live pairing are identical.
    probe = r.get("probe")
    if probe:
        from .video_probe_store import apply_probe_result
        p = dict(probe)
        p["photo_id"] = pid
        p.setdefault("probe_version", r.get("probe_version"))
        if p.get("probe_version") is None:
            from .video_probe import PROBE_VERSION
            p["probe_version"] = PROBE_VERSION
        info = apply_probe_result(conn, p)
        # The file on the NAS is still the ZIP: nothing can play it, whatever
        # the inner codec — so it needs a proxy (video_proxy; the worker
        # unwraps it before transcoding). A later unwrap + re-probe
        # recomputes this from the real file.
        conn.execute("UPDATE video_meta SET proxy_needed = 1 WHERE photo_id = ?", (pid,))
        out["probe_applied"] = {k: info.get(k) for k in ("kind", "date_taken")}

    fh = r.get("file_hash")
    if fh:
        if row[1] is None:
            dup = conn.execute("SELECT id FROM photos WHERE file_hash = ? AND id != ? "
                               "LIMIT 1", (fh, pid)).fetchone()
            if dup:
                out["duplicate_of"] = dup[0]
            conn.execute("UPDATE photos SET file_hash = ? WHERE id = ? AND file_hash IS NULL",
                         (fh, pid))
        elif row[1] != fh:
            out["hash_mismatch"] = True

    if r.get("scan_error"):
        # Terminal verdict: stored, version set, never re-claimed until
        # SCAN_VERSION bumps. Existing faces/tracks are left alone.
        conn.execute(
            "UPDATE video_meta SET scan_error = ?, scan_version = ?, scanned_at = ? "
            "WHERE photo_id = ?", (str(r["scan_error"])[:500], version, now, pid))
        out["scan_error"] = r["scan_error"]
        return out

    idents = list(r.get("identities") or [])
    # The faces-row gate (video_analysis.face_row_eligible): the worker's
    # verdict when it sent one, else the default gate on n_det/presence_s.
    want_face = [bool(i["face_row"]) if i.get("face_row") is not None
                 else face_row_eligible(i.get("n_det"), i.get("presence_s"))
                 for i in idents]

    # D13 — match labelled old faces to new identities BEFORE deleting them.
    # Matched against EVERY identity, gated or not: a human label outranks
    # the gate, so the identity it lands on gets a faces row regardless.
    old = _old_faces(conn, pid)
    labelled = [o for o in old if o["match_source"] in CARRY_LABEL_SOURCES]
    new_encs = [i.get("encoding") for i in idents]
    carried, unmatched = match_labels(labelled, new_encs)
    for ni in carried:
        if not want_face[ni]:
            want_face[ni] = True
            out["promoted_by_label"] = out.get("promoted_by_label", 0) + 1
    # Then every OTHER old face that carries face/person exclusions (a strict
    # match the duplicate resolver stripped, a dedupe_unmatched loser) is
    # matched one-to-one among the identities still free that get a faces
    # row: the exclusions move with it, or the nightly match/resolve churn
    # loop would restart on the new face ids. Only the exclusions travel —
    # not the auto label. (A gated identity has no face to bar.)
    free = [None if (ni in carried or not want_face[ni]) else e
            for ni, e in enumerate(new_encs)]
    excl_only, _ = match_labels(
        [o for o in old if o["match_source"] not in CARRY_LABEL_SOURCES and o["exclusions"]],
        free)

    # Replace: tracks, faces (+ their vec0 encodings, explicitly — vec0 has
    # no FK cascade), CLIP embedding.
    conn.execute("DELETE FROM video_tracks WHERE photo_id = ?", (pid,))
    # The video-subjects result references THIS scan's identities (face ids)
    # and segment keyframes, so a re-scan invalidates it: drop the rows and
    # the marker in the same transaction, which re-queues the subjects pass
    # (video_subjects_store.claimable_sql). Keyed on the scan landing, not on
    # comparing versions, because a clear-pass re-scan keeps the same
    # SCAN_VERSION while replacing every face row.
    conn.execute("DELETE FROM video_subjects WHERE photo_id = ?", (pid,))
    conn.execute("UPDATE video_meta SET subjects_version = NULL, subjects_at = NULL, "
                 "subjects_json = NULL WHERE photo_id = ?", (pid,))
    old_ids = [o["face_id"] for o in old]
    for i in range(0, len(old_ids), 500):
        chunk = old_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        if HAS_SQLITE_VEC:
            conn.execute(f"DELETE FROM face_encodings WHERE face_id IN ({ph})", chunk)
        conn.execute(f"DELETE FROM faces WHERE id IN ({ph})", chunk)

    generation = r.get("generation")
    frames_v = frames_dir(frames_root, pid, version, generation) if frames_root else None
    for ni, ident in enumerate(idents):
        bbox = ident["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"identity {ni}: bbox must have 4 values")
        enc = ident.get("encoding")
        if enc is not None and len(enc) != 512:
            raise ValueError(f"identity {ni}: encoding must be 512-d")
        ix = int(ident.get("ident_ix", ni))
        if not want_face[ni]:
            # Below the gate: a track (presence / share / subjects joins see
            # it) with no faces row, no encoding, no crop install.
            _insert_track(conn, pid, None, ix, ident)
            out["gated"] += 1
            continue
        lab = carried.get(ni)
        person_id = lab["person_id"] if lab else None
        source = lab["match_source"] if lab else None
        cur = conn.execute(
            "INSERT INTO faces (photo_id, person_id, bbox_top, bbox_right, bbox_bottom, "
            "bbox_left, cluster_id, det_score, match_source) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, person_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
             None, ident.get("det_score"), source))
        fid = cur.lastrowid
        if HAS_SQLITE_VEC and enc is not None:
            conn.execute("INSERT INTO face_encodings (face_id, encoding) VALUES (?, ?)",
                         (fid, _serialize_float_list(enc)))
        if lab:
            excl = [(fid, p) for p in lab.get("exclusions") or [] if p != person_id]
            if excl:
                record_face_person_exclusions(conn, excl, reason="rescan_carry")
            out["carried"].append({"old_face_id": lab["face_id"], "face_id": fid,
                                   "person_id": person_id, "match_source": source,
                                   "l2": lab.get("l2")})
        ex = excl_only.get(ni)
        if ex and ex.get("exclusions"):
            record_face_person_exclusions(conn, [(fid, p) for p in ex["exclusions"]],
                                          reason="rescan_carry")
            out.setdefault("exclusions_carried", []).append(
                {"old_face_id": ex["face_id"], "face_id": fid,
                 "person_ids": list(ex["exclusions"])})
        _insert_track(conn, pid, fid, ix, ident)
        out["faces"] += 1
        if frames_v is not None and crop_cache_dir:
            for size in CROP_SIZES:
                src = frames_v / stored_name("crop", crop_name(ix, size))
                out["_install"].append((str(src), os.path.join(
                    crop_cache_dir, f"{fid}_{size}.jpg")))

    for o in unmatched:
        out["unmatched_labels"].append({k: o.get(k) for k in
                                        ("face_id", "person_id", "match_source")})
    if out["unmatched_labels"]:
        logger.warning("video-scan %s: %d human label(s) found no identity in the "
                       "re-scan and were dropped: %s", pid, len(out["unmatched_labels"]),
                       out["unmatched_labels"])

    emb = r.get("clip_embedding")
    if HAS_SQLITE_VEC:
        conn.execute("DELETE FROM clip_embeddings WHERE photo_id = ?", (pid,))
        if emb:
            if len(emb) != 512:
                raise ValueError("clip_embedding must be 512-d")
            conn.execute("INSERT INTO clip_embeddings (photo_id, embedding) VALUES (?, ?)",
                         (pid, _serialize_float_list(emb)))

    sets = ["scan_version = ?", "scan_error = NULL", "scanned_at = ?", "shake = ?",
            "noise = ?", "focus_consistency = ?", "quality_json = ?",
            "scan_generation = ?"]
    vals = [version, now, r.get("shake"), r.get("noise"), r.get("focus_consistency"),
            _json(r.get("quality_json")), generation]
    if r.get("poster_uploaded"):
        # Bumped in the same transaction that promotes the generation; the
        # caller installs the staged posters only after the commit.
        sets += ["poster_t = ?",
                 "poster_version = COALESCE(poster_version, probe_version, 0) + 1"]
        vals.append(r.get("poster_t"))
        if generation:
            out["_posters"] = (version, generation)
    conn.execute(f"UPDATE video_meta SET {', '.join(sets)} WHERE photo_id = ?",
                 (*vals, pid))
    return out


def install_crops(pairs) -> int:
    """Copy uploaded identity crops into the face-crop cache as
    ``{face_id}_{size}.jpg`` (atomic per file), so ``/api/faces/crop`` and the
    replica's crop mirror serve them like any other face's. A missing source
    is skipped — the crop endpoint falls back to the representative frame."""
    n = 0
    for src, dst in pairs or []:
        try:
            if not os.path.exists(src):
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = f"{dst}.tmp.{os.getpid()}"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            n += 1
        except OSError as e:
            logger.warning("could not install video face crop %s -> %s: %s", src, dst, e)
    return n


# ---------------------------------------------------------------------------
# Face source for a video face (crop endpoint / warm path)
# ---------------------------------------------------------------------------

def video_face_source(conn, face_id: int, frames_root) -> Optional[dict]:
    """For a face on a VIDEO row: ``{"photo_id", "rep_path", "crop_paths":
    {size: path}}`` — the uploaded representative frame (which the face's bbox
    is relative to) and the uploaded crops. None for a still's face or a face
    with no track. Never touches the video file."""
    row = conn.execute(
        "SELECT f.photo_id, p.media_type, vt.ident_ix, vt.rep_frame, vm.scan_version, "
        "vm.scan_generation, COALESCE(p.image_width, vm.width), "
        "COALESCE(p.image_height, vm.height) "
        "FROM faces f JOIN photos p ON p.id = f.photo_id "
        "LEFT JOIN video_tracks vt ON vt.face_id = f.id "
        "LEFT JOIN video_meta vm ON vm.photo_id = f.photo_id WHERE f.id = ?",
        (face_id,)).fetchone()
    if row is None or row[1] != "video":
        return None
    # native_w/h: the space the faces bbox is in (see the module docstring).
    out = {"photo_id": row[0], "rep_path": None, "crop_paths": {},
           "native_w": row[6], "native_h": row[7]}
    if row[2] is None or row[4] is None or frames_root is None:
        return out
    d = frames_dir(frames_root, row[0], row[4], row[5])
    if row[3] is not None:
        out["rep_path"] = str(d / stored_name("keyframe", rep_name(row[3])))
    for size in CROP_SIZES:
        out["crop_paths"][size] = str(d / stored_name("crop", crop_name(row[2], size)))
    return out


def current_versions(conn, photo_id: int) -> set[int]:
    """The asset versions still referenced by a video (scan + probe), for
    cleanup-orphans."""
    row = conn.execute("SELECT scan_version, probe_version FROM video_meta "
                       "WHERE photo_id = ?", (photo_id,)).fetchone()
    if not row:
        return set()
    return {int(v) for v in row if v is not None}


def stale_frame_dirs(conn, frames_root, now: Optional[float] = None,
                     grace_s: float = STAGING_GRACE_S) -> list[str]:
    """Asset directories no longer referenced, for cleanup-orphans:

    * ``video_frames/{pid}/v*`` when the photo is gone, or the version is
      neither the current scan nor probe version;
    * ``v{ver}/g{gen}`` staging generations other than the row's current
      ``scan_generation`` — superseded scans and uploads whose submit never
      landed — once older than ``grace_s`` (an in-flight scan's staging dir
      must not be swept between its upload and its submit)."""
    import time
    now = time.time() if now is None else now
    root = Path(frames_root)
    if not root.is_dir():
        return []
    stale = []
    for pdir in root.iterdir():
        if not pdir.is_dir() or not pdir.name.isdigit():
            continue
        pid = int(pdir.name)
        exists = conn.execute("SELECT 1 FROM photos WHERE id = ?", (pid,)).fetchone()
        keep = current_versions(conn, pid) if exists else set()
        gen_row = conn.execute("SELECT scan_version, scan_generation FROM video_meta "
                               "WHERE photo_id = ?", (pid,)).fetchone() if exists else None
        for vdir in pdir.iterdir():
            if not vdir.is_dir() or not vdir.name.startswith("v"):
                continue
            try:
                v = int(vdir.name[1:])
            except ValueError:
                continue
            if v not in keep:
                stale.append(str(vdir))
                continue
            live = (gen_row[1] if gen_row and gen_row[0] == v else None)
            for gdir in vdir.iterdir():
                if not gdir.is_dir() or not gdir.name.startswith("g"):
                    continue
                if gdir.name[1:] == live:
                    continue
                try:
                    age = now - gdir.stat().st_mtime
                except OSError:
                    continue
                if age >= grace_s:
                    stale.append(str(gdir))
    return sorted(stale)


# ---------------------------------------------------------------------------
# Re-queue (clear-pass video-scan) + the faces-row gate's targeted re-scan
# ---------------------------------------------------------------------------

def requeue_scan(conn, photo_ids) -> int:
    """Re-queue ``video-scan`` for exactly ``photo_ids`` (the clear-pass
    branch). The faces / tracks / CLIP mean stay until the re-scan REPLACES
    them, which is also what carries the human labels across (D13) — deleting
    them here would lose those. Subjects are left alone too: the re-scan
    landing deletes them (apply_scan_result), which re-queues video-subjects.
    The description (video-describe) is never touched. Does NOT commit;
    returns the video_meta rows reset. ``[]`` is no videos."""
    ids = [int(i) for i in photo_ids or ()]
    n = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        n += conn.execute(
            f"UPDATE video_meta SET scan_version = NULL, scan_error = NULL, "
            f"scanned_at = NULL WHERE photo_id IN ({ph})", chunk).rowcount
        conn.execute(
            f"DELETE FROM worker_processed WHERE pass_type = 'video-scan' "
            f"AND photo_id IN ({ph})", chunk)
    return n


def _video_field_order(probe_json) -> Optional[str]:
    try:
        data = json.loads(probe_json) if isinstance(probe_json, str) else (probe_json or {})
    except (TypeError, ValueError):
        return None
    for st in (data or {}).get("streams") or []:
        if isinstance(st, dict) and st.get("codec_type") == "video":
            return st.get("field_order")
    return None


def _subjects_empty(subjects_json) -> bool:
    try:
        d = json.loads(subjects_json) if subjects_json else None
    except (TypeError, ValueError):
        return False
    return isinstance(d, dict) and d.get("subjects") == []


def black_decode_candidates(conn, exts=(), empty_only: bool = False) -> dict:
    """Videos whose scan may have run on black / stale frames — OpenCV hands
    back all-zero (or the previous video's) BGR for INTERLACED streams
    (AVCHD .m2ts/.mts) instead of failing (video_decode, 2026-10-25). Read-only.

    A candidate is a non-Live video the scan has already visited
    (``scan_version`` set, with or without ``scan_error``) that is interlaced
    per its stored probe (``field_order`` tt/bb/tb/bt) OR whose extension is
    in ``exts``. ``empty_only`` keeps only those whose stored subjects are
    ``[]`` (the visible symptom; interlaced clips with subjects may have been
    scanned on a stale buffer and are wrong too).

    Returns ``{"photo_ids", "videos", "by_ext": {ext: {"videos",
    "interlaced", "subjects_empty", "described", "scan_error"}}}``."""
    from .video_decode import is_interlaced
    want = {e.lower().lstrip(".") for e in exts or () if e}
    by_ext: dict = {}
    ids: list[int] = []
    rows = conn.execute(
        "SELECT p.id, p.filepath, p.description, v.probe_json, v.subjects_json, "
        "v.scan_error FROM photos p JOIN video_meta v ON v.photo_id = p.id "
        "WHERE p.media_type = 'video' AND v.scan_version IS NOT NULL "
        "AND v.live_of_photo_id IS NULL AND COALESCE(v.kind, '') != 'live' "
        "ORDER BY p.id")
    for pid, fp, desc, pj, sj, serr in rows:
        ext = os.path.splitext(fp or "")[1].lower().lstrip(".")
        inter = is_interlaced(_video_field_order(pj))
        if not (inter or ext in want):
            continue
        empty = _subjects_empty(sj)
        if empty_only and not empty:
            continue
        b = by_ext.setdefault(ext or "?", {"videos": 0, "interlaced": 0, "subjects_empty": 0,
                                           "described": 0, "scan_error": 0})
        b["videos"] += 1
        b["interlaced"] += int(inter)
        b["subjects_empty"] += int(empty)
        b["described"] += int(desc is not None)
        b["scan_error"] += int(serr is not None)
        ids.append(int(pid))
    return {"photo_ids": ids, "videos": len(ids), "by_ext": by_ext}


def below_gate_sql(alias: str = "vt") -> str:
    """A ``video_tracks`` row that would NOT get a faces row under the gate
    (``video_analysis.face_row_eligible``; params ``:min_det``, ``:min_pres``,
    ``:min_pres_det``). Unknown n_det / presence pass, as in the gate."""
    return (f"({alias}.n_det IS NOT NULL AND {alias}.presence_s IS NOT NULL "
            f"AND NOT ({alias}.n_det >= :min_det OR ({alias}.presence_s >= :min_pres - 1e-9 "
            f"AND {alias}.n_det >= :min_pres_det)))")


def fragmented_videos(conn, min_detections: Optional[int] = None,
                      min_presence_s: Optional[float] = None) -> dict:
    """Videos holding a ``faces`` row whose track is below the faces-row gate
    — the ones a re-scan under the gate would change. Read-only.

    Returns ``{"photo_ids": [...], "videos", "rows_below_gate",
    "rows_disappear", "human_labelled_kept", "strict_labelled",
    "subjects_done", "described", "videos_scanned_with_faces"}``.
    ``rows_disappear`` excludes faces carrying a human label (D13 carries
    those onto the re-scanned identity and promotes it past the gate)."""
    from .video_analysis import DEFAULT_PARAMS
    p = {"min_det": int(DEFAULT_PARAMS.face_min_detections if min_detections is None
                        else min_detections),
         "min_pres": float(DEFAULT_PARAMS.face_min_presence_s if min_presence_s is None
                           else min_presence_s),
         "min_pres_det": int(DEFAULT_PARAMS.face_min_presence_detections)}
    human = ",".join(f"'{s}'" for s in CARRY_LABEL_SOURCES)
    base = (f"FROM video_tracks vt JOIN faces f ON f.id = vt.face_id "
            f"JOIN video_meta vm ON vm.photo_id = vt.photo_id "
            f"WHERE vm.scan_version IS NOT NULL AND vm.scan_error IS NULL "
            f"AND {below_gate_sql('vt')}")
    ids = [r[0] for r in conn.execute(
        f"SELECT DISTINCT vt.photo_id {base} ORDER BY vt.photo_id", p)]
    row = conn.execute(
        f"SELECT COUNT(*), "
        f"TOTAL(f.match_source IN ({human})), "
        f"TOTAL(f.match_source = 'strict' AND f.person_id IS NOT NULL) {base}", p).fetchone()
    out = {"photo_ids": ids, "videos": len(ids), "rows_below_gate": int(row[0]),
           "human_labelled_kept": int(row[1]), "strict_labelled": int(row[2]),
           "rows_disappear": int(row[0]) - int(row[1]),
           "params": {"min_detections": p["min_det"], "min_presence_s": p["min_pres"],
                      "min_presence_detections": p["min_pres_det"]}}
    out["videos_scanned_with_faces"] = conn.execute(
        "SELECT COUNT(DISTINCT vt.photo_id) FROM video_tracks vt "
        "WHERE vt.face_id IS NOT NULL").fetchone()[0]
    subj = desc = 0
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        subj += conn.execute(f"SELECT COUNT(*) FROM video_meta WHERE photo_id IN ({ph}) "
                             f"AND subjects_version IS NOT NULL", chunk).fetchone()[0]
        desc += conn.execute(f"SELECT COUNT(*) FROM photos WHERE id IN ({ph}) "
                             f"AND description IS NOT NULL", chunk).fetchone()[0]
    out["subjects_done"] = subj
    out["described"] = desc
    return out

"""Server side of the ``video-scan`` pass (docs/plans/video-support.md step 3).

Pure SQL + file copies on the NAS — no decoder (D19). The desktop worker
(``video_scan.py`` / ``worker._process_video_scan``) uploads the keyframes,
per-segment keyframes, representative frames and face crops through
``POST /api/videos/{id}/assets`` FIRST, then submits one ``VideoScanResult``;
``worker_api.submit_results`` calls :func:`apply_scan_result` inside one
transaction per video, commits, then :func:`install_crops`.

**One ``faces`` row per within-video identity.** Its bbox is in the
coordinates of the identity's representative frame, stored as
``video_frames/{pid}/v{SCAN_VERSION}/keyframe_rep_f{rep_frame:05d}.jpg``; its
encoding is the identity's mean top-k ArcFace embedding. So ``/faces``,
clustering, the strict matcher, verify-labels and the rival test apply to
videos unchanged, and a face crop is served from the uploaded crop (or the rep
frame) — never from the video file.

**A re-scan replaces everything** (old ``video_tracks``, old ``faces`` and their
vec0 ``face_encodings`` rows, the CLIP embedding) **but carries human labels**
(D13): an old face whose ``match_source`` is ``manual`` / ``merge_review`` /
``rejected`` is matched one-to-one, greedily by ArcFace L2 (<= 0.6), to a new
identity, which inherits its ``person_id`` / ``match_source`` and its
face/person exclusions. Labels that find no partner are returned in the
response and logged — never silently dropped.

Asset naming (shared with the worker; the asset endpoint prefixes the kind):

==========================  ==============================================
asset ``name`` (kind)       stored as ``video_frames/{pid}/v{ver}/...``
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


def frames_dir(frames_root, photo_id: int, version: int) -> Path:
    return Path(frames_root) / str(int(photo_id)) / f"v{int(version)}"


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


def apply_scan_result(conn, r: dict, *, crop_cache_dir: Optional[str] = None,
                      frames_root=None) -> dict:
    """Write one scan result (a ``VideoScanResult`` as a dict). Does NOT
    commit; any sqlite error propagates (the caller defers a transient lock
    and spends the attempt otherwise).

    Returns a summary dict; ``_install`` lists the ``(src, dst)`` crop copies
    the caller performs after the commit (:func:`install_crops`).
    """
    from .db import HAS_SQLITE_VEC, _serialize_float_list, record_face_person_exclusions

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
    out: dict = {"photo_id": pid, "scan_error": None, "faces": 0, "carried": [],
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

    # D13 — match labelled old faces to new identities BEFORE deleting them.
    old = _old_faces(conn, pid)
    labelled = [o for o in old if o["match_source"] in CARRY_LABEL_SOURCES]
    carried, unmatched = match_labels(labelled, [i.get("encoding") for i in idents])

    # Replace: tracks, faces (+ their vec0 encodings, explicitly — vec0 has
    # no FK cascade), CLIP embedding.
    conn.execute("DELETE FROM video_tracks WHERE photo_id = ?", (pid,))
    old_ids = [o["face_id"] for o in old]
    for i in range(0, len(old_ids), 500):
        chunk = old_ids[i:i + 500]
        ph = ",".join("?" * len(chunk))
        if HAS_SQLITE_VEC:
            conn.execute(f"DELETE FROM face_encodings WHERE face_id IN ({ph})", chunk)
        conn.execute(f"DELETE FROM faces WHERE id IN ({ph})", chunk)

    frames_v = frames_dir(frames_root, pid, version) if frames_root else None
    for ni, ident in enumerate(idents):
        bbox = ident["bbox"]
        if len(bbox) != 4:
            raise ValueError(f"identity {ni}: bbox must have 4 values")
        enc = ident.get("encoding")
        if enc is not None and len(enc) != 512:
            raise ValueError(f"identity {ni}: encoding must be 512-d")
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
        ix = int(ident.get("ident_ix", ni))
        conn.execute(
            "INSERT INTO video_tracks (photo_id, face_id, ident_ix, t_start, t_end, n_det, "
            "presence_s, focus_s, focus_intervals, rep_t, rep_frame, mean_salience, "
            "presence_intervals) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, fid, ix, ident.get("t_start"), ident.get("t_end"), ident.get("n_det"),
             ident.get("presence_s"), ident.get("focus_s"),
             _json(ident.get("focus_intervals")), ident.get("rep_t"),
             ident.get("rep_frame"), ident.get("mean_salience"),
             _json(ident.get("presence_intervals"))))
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
            "noise = ?", "focus_consistency = ?", "quality_json = ?"]
    vals = [version, now, r.get("shake"), r.get("noise"), r.get("focus_consistency"),
            _json(r.get("quality_json"))]
    if r.get("poster_uploaded"):
        sets += ["poster_t = ?",
                 "poster_version = COALESCE(poster_version, probe_version, 0) + 1"]
        vals.append(r.get("poster_t"))
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
        "SELECT f.photo_id, p.media_type, vt.ident_ix, vt.rep_frame, vm.scan_version "
        "FROM faces f JOIN photos p ON p.id = f.photo_id "
        "LEFT JOIN video_tracks vt ON vt.face_id = f.id "
        "LEFT JOIN video_meta vm ON vm.photo_id = f.photo_id WHERE f.id = ?",
        (face_id,)).fetchone()
    if row is None or row[1] != "video":
        return None
    out = {"photo_id": row[0], "rep_path": None, "crop_paths": {}}
    if row[2] is None or row[4] is None or frames_root is None:
        return out
    d = frames_dir(frames_root, row[0], row[4])
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


def stale_frame_dirs(conn, frames_root) -> list[str]:
    """``video_frames/{pid}/v*`` directories no longer referenced: the photo
    is gone, or the version is neither the current scan nor probe version."""
    root = Path(frames_root)
    if not root.is_dir():
        return []
    stale = []
    for pdir in root.iterdir():
        if not pdir.is_dir() or not pdir.name.isdigit():
            continue
        exists = conn.execute("SELECT 1 FROM photos WHERE id = ?",
                              (int(pdir.name),)).fetchone()
        keep = current_versions(conn, int(pdir.name)) if exists else set()
        for vdir in pdir.iterdir():
            if not vdir.is_dir() or not vdir.name.startswith("v"):
                continue
            try:
                v = int(vdir.name[1:])
            except ValueError:
                continue
            if v not in keep:
                stale.append(str(vdir))
    return sorted(stale)

"""Faster video-eval labelling (docs/plans/video-support.md step 4, 2026-10-25).

Three helpers for the "Video subject" tab, all FILES under
``eval_dir()/video`` (git-ignored), never DB rows:

- **Playable proxies.** The owner's browser decodes no HEVC / 10-bit / avi /
  mts, so the NAS's H.264 proxy (``/api/videos/{id}/proxy``) is cached beside
  the originals and served by the eval API. The page picks proxy vs original
  with ``PS.chooseVideoSource`` (the modal's helper). Analysis — frame grabs,
  detection, runs — keeps using the ORIGINALS.

      video/proxies/<pid>.mp4      the H.264 copy
      video/playback.json          {<pid>: {"playback", "needs_proxy",
                                            "proxy_version", "proxy_bytes"}}

- **Face clusters across the whole sample.** InsightFace on each LABEL
  segment's keyframe (+ a couple of extra frames per segment), clustered on
  ArcFace embeddings across every sampled video, so the owner names a person
  ONCE instead of once per segment.

      video/face_clusters.json     {"version", "created", "params", "faces":
                                    {<face_id>: {"photo_id", "t", "seg_ix",
                                    "seg_version", "bbox": [l, t, r, b],
                                    "det_score"}}, "clusters": [{"key",
                                    "faces": [face_id], "suggestion"}],
                                    "noise": [face_id], "stats"}
      video/face_crops/<face_id>.jpg

  A face id is ``<pid>-<ms>-<left>-<top>`` — a function of the frame and the
  box, never of a clustering run — so re-clustering (a new ``version``)
  never orphans a name.

- **Names**, keyed by face id (never by cluster key):

      video/face_names.json        {<face_id>: {"kind": person|someone_else|
                                    junk|ejected, "person_id"?, "updated_at"}}

  ``ejected`` = "not this cluster's person" (clicked out of a card): it counts
  as an unnamed face until named on its own.

The run grouping (`segment_signatures` + `group_runs`) is pure: consecutive
label segments whose frames show the same named people (and the same
face / no-face state) become one run the owner labels with one click. The
label storage stays per segment.

Blind: nothing here reads a run cache or a variant; a suggestion comes from
the library's own trusted references, and is shown as a suggestion only.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from . import model_eval as me

CLUSTERS_FILE = "face_clusters.json"
NAMES_FILE = "face_names.json"
PLAYBACK_FILE = "playback.json"
NAME_KINDS = ("person", "someone_else", "junk", "ejected")
FACE_ID_RE = re.compile(r"^\d+-\d+-\d+-\d+$")

# Trusted reference sources for name suggestions — temporal is ~4% accurate
# on these shoots and would poison them (CLAUDE.md, face-label integrity).
TRUSTED_SOURCES = ("manual", "merge_review", "strict")
# Conservative: the cluster's median face must sit this close to one person's
# references, and that person must beat the runner-up by SUGGEST_MARGIN.
SUGGEST_MAX_DIST = 0.95
SUGGEST_MARGIN = 0.08


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

def video_dir() -> Path:
    return me.pass_dir(me.VIDEO)


def check_face_id(face_id: str) -> str:
    if not FACE_ID_RE.match(str(face_id or "")):
        raise ValueError(f"bad face id {face_id!r}")
    return str(face_id)


def face_id(photo_id: int, t: float, bbox_ltrb) -> str:
    l, t_, _r, _b = (int(round(float(x))) for x in bbox_ltrb)
    return f"{int(photo_id)}-{int(round(float(t) * 1000))}-{max(0, l)}-{max(0, t_)}"


def crop_path(fid: str) -> Path:
    return video_dir() / "face_crops" / f"{check_face_id(fid)}.jpg"


def proxy_path(photo_id: int) -> Path:
    return video_dir() / "proxies" / f"{int(photo_id)}.mp4"


def cached_proxy(photo_id: int) -> Optional[Path]:
    p = proxy_path(photo_id)
    return p if p.exists() else None


def frame_path(photo_id: int, t: float, max_px: int) -> Path:
    """Same name `model_eval.cached_frame_jpeg` uses, so a frame written by
    `cluster-faces` is a cache hit for the page."""
    return (video_dir() / "frames" / str(int(photo_id))
            / f"{int(round(float(t) * 1000))}-{int(max_px)}.jpg")


# --------------------------------------------------------------------------
# Proxies + playback info
# --------------------------------------------------------------------------

def load_playback() -> dict:
    return me.read_pass_file(me.VIDEO, PLAYBACK_FILE, {})


def playback_entry(photo_id: int) -> Optional[dict]:
    return load_playback().get(str(int(photo_id)))


def _get_json(url: str, headers: dict, timeout: float = 60):
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _stream_to(url: str, dest: Path, headers: dict, *, timeout: float = 600,
               chunk: int = 1 << 20, retries: int = 5) -> int:
    """Stream `url` into `dest` atomically, size-checked against
    Content-Length. A 503 (stream cap) is waited out per Retry-After."""
    import time as _time
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(retries):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:
                _time.sleep(float(e.headers.get("Retry-After") or 10))
                continue
            raise
        break
    want = resp.headers.get("Content-Length")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    try:
        got = 0
        with os.fdopen(fd, "wb") as f, resp:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                got += len(buf)
            f.flush()
            os.fsync(f.fileno())
        if want is not None and got != int(want):
            raise RuntimeError(f"short read: {got} of {want} bytes")
        if got == 0:
            raise RuntimeError("empty response")
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return got


def fetch_proxy(photo_id: int, server: str, *, intent: str,
                get_json=None, stream=None) -> dict:
    """Record the original's playback facts and cache its H.264 proxy when
    the server has one. Returns the playback entry. Re-fetches only when the
    server's proxy version moved."""
    headers = {"X-Photosearch-Source": "claude", "X-Photosearch-Intent": intent}
    base = server.rstrip("/")
    detail = (get_json or _get_json)(f"{base}/api/photos/{int(photo_id)}", headers)
    video = (detail or {}).get("video") or {}
    proxy = video.get("proxy") or {}
    entry = {"playback": video.get("playback"), "needs_proxy": video.get("needs_proxy"),
             "browser_playable": video.get("browser_playable"),
             "proxy_version": proxy.get("version"), "proxy_bytes": None}
    old = playback_entry(photo_id) or {}
    if not video.get("proxy_url") and old.get("proxy_local") and proxy_path(photo_id).exists():
        entry.update(proxy_bytes=old.get("proxy_bytes"), proxy_local=True)
    if video.get("proxy_url"):
        dest = proxy_path(photo_id)
        if dest.exists() and old.get("proxy_version") == entry["proxy_version"]:
            entry["proxy_bytes"] = dest.stat().st_size
        else:
            entry["proxy_bytes"] = (stream or _stream_to)(base + video["proxy_url"], dest, headers)

    def put(data):
        data[str(int(photo_id))] = entry
        return entry
    return me.update_pass_file(me.VIDEO, PLAYBACK_FILE, {}, put)


def make_local_proxy(photo_id: int, transcode=None, local_meta=None) -> Optional[int]:
    """Transcode a playable H.264 copy from the cached ORIGINAL into the eval
    cache — for the sampled clips the NAS has no proxy for (Live clips are
    never proxied there). Production's own transcode + verify
    (video_proxy.transcode). Nothing is uploaded anywhere."""
    from . import video_proxy
    src = me.cached_original(photo_id)
    if src is None:
        return None
    dest = proxy_path(photo_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.stem}.", suffix=".mp4")
    os.close(fd)
    try:
        meta = (local_meta or video_proxy.local_meta)(str(src))
        out = (transcode or video_proxy.transcode)(str(src), tmp, meta)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    def put(data):
        e = data.setdefault(str(int(photo_id)), {})
        e["proxy_bytes"] = int(out.get("bytes") or dest.stat().st_size)
        e["proxy_local"] = True
        return e
    me.update_pass_file(me.VIDEO, PLAYBACK_FILE, {}, put)
    return dest.stat().st_size


def page_video_info(photo_id: int, api_prefix: str) -> dict:
    """The `video` object the page hands to PS.chooseVideoSource: the
    original's playback facts and a LOCAL proxy url (only when cached)."""
    e = playback_entry(photo_id) or {}
    has = cached_proxy(photo_id) is not None
    entry = me.video_sample_entry(photo_id) or {}
    return {"filename": entry.get("filename"),
            "playback": e.get("playback"), "needs_proxy": e.get("needs_proxy"),
            "browser_playable": e.get("browser_playable"),
            "proxy_url": f"{api_prefix}/video/proxy/{int(photo_id)}" if has else None}


# --------------------------------------------------------------------------
# Clusters + names
# --------------------------------------------------------------------------

def load_clusters() -> Optional[dict]:
    return me.read_pass_file(me.VIDEO, CLUSTERS_FILE, None)


def save_clusters(data: dict) -> dict:
    data = dict(data)
    data.setdefault("created", me.now_iso())
    if not data.get("version"):
        raise ValueError("a cluster file needs a version")
    for fid in data.get("faces") or {}:
        check_face_id(fid)
    for c in data.get("clusters") or []:
        for fid in c["faces"]:
            if fid not in data["faces"]:
                raise ValueError(f"cluster {c.get('key')} names unknown face {fid}")
    me.write_pass_file(me.VIDEO, CLUSTERS_FILE, data)
    return data


def load_names() -> dict:
    return me.read_pass_file(me.VIDEO, NAMES_FILE, {})


def save_names(face_ids: Iterable[str], kind: Optional[str],
               person_id: Optional[int] = None, *, known_person=None,
               keep_ejected: bool = False) -> dict:
    """Name a set of faces at once (a cluster, or the crops picked out of
    one). kind None clears them. Locked + atomic (update_pass_file).

    `keep_ejected` (naming crops from the ejected pool): a face that was
    ejected from its cluster keeps an ``"ejected": true`` mark under its new
    name, so it stays out of that cluster's answer and a later cluster-level
    rename does not overwrite it."""
    ids = [check_face_id(f) for f in face_ids]
    if not ids:
        raise ValueError("no faces given")
    if kind is not None and kind not in NAME_KINDS:
        raise ValueError(f"kind must be one of {NAME_KINDS} (or null to clear)")
    lab = None
    if kind is not None:
        lab = {"kind": kind, "updated_at": me.now_iso()}
        if kind == "person":
            if person_id is None:
                raise ValueError("a person name needs person_id")
            if known_person is not None and not known_person(int(person_id)):
                raise ValueError(f"no person {person_id}")
            lab["person_id"] = int(person_id)

    def put(data):
        for f in ids:
            if lab is None:
                data.pop(f, None)
                continue
            new = dict(lab)
            old = data.get(f) or {}
            if keep_ejected and (old.get("kind") == "ejected" or old.get("ejected")):
                new["ejected"] = True
            data[f] = new
        return {"n": len(ids), "label": lab}
    return me.update_pass_file(me.VIDEO, NAMES_FILE, {}, put)


def face_tokens(clusters: Optional[dict], names: dict) -> dict:
    """{face_id: token} — ("p", person_id) when the owner named it, else
    ("o",): someone else, ejected, noise, or a cluster not named yet (the
    owner names only the people that matter; strangers can stay unnamed).
    Junk faces are dropped (they are not faces)."""
    out: dict = {}
    if not clusters:
        return out
    for f in clusters.get("faces") or {}:
        n = names.get(f)
        if n:
            k = n.get("kind")
            if k == "junk":
                continue
            if k == "person":
                out[f] = ("p", int(n["person_id"]))
                continue
            out[f] = ("o",)
            continue
        out[f] = ("o",)
    return out


def segment_signatures(clusters: Optional[dict], names: dict, photo_id: int,
                       seg_version: str, n_segments: int) -> list:
    """One signature per label segment: (sorted named-person tokens, face
    state) where face state is "none" (no face at all), "others" (faces, none
    of them named) or "faces". Faces detected on a stale segmentation are
    ignored."""
    toks = face_tokens(clusters, names)
    per = [set() for _ in range(n_segments)]
    seen = [False] * n_segments
    for f, meta in ((clusters or {}).get("faces") or {}).items():
        if int(meta["photo_id"]) != int(photo_id) or meta.get("seg_version") != seg_version:
            continue
        ix = int(meta["seg_ix"])
        if not 0 <= ix < n_segments or f not in toks:
            continue
        seen[ix] = True
        if toks[f][0] != "o":
            per[ix].add(toks[f])
    out = []
    for ix in range(n_segments):
        tk = tuple(sorted(per[ix], key=lambda t: (t[0], str(t[1]))))
        out.append((tk, "faces" if tk else ("others" if seen[ix] else "none")))
    return out


def group_runs(signatures: list) -> list:
    """Consecutive segments with equal signatures -> [[seg_ix, ...], ...]."""
    runs: list = []
    for ix, sig in enumerate(signatures):
        if runs and signatures[runs[-1][-1]] == sig:
            runs[-1].append(ix)
        else:
            runs.append([ix])
    return runs


def cluster_status(cluster: dict, names: dict) -> dict:
    """What the owner has said about a cluster: counts per answer, and the
    single answer when every non-ejected face shares it."""
    counts: dict = {}
    for f in cluster["faces"]:
        n = names.get(f)
        if n and n.get("ejected"):
            k = "ejected"          # ejected, then named on its own
        else:
            k = "unnamed" if not n else (f"p{n['person_id']}" if n["kind"] == "person" else n["kind"])
        counts[k] = counts.get(k, 0) + 1
    rest = {k: v for k, v in counts.items() if k not in ("ejected", "unnamed")}
    answer = max(sorted(rest), key=lambda k: rest[k]) if rest else None
    return {"counts": counts, "answer": answer,
            "done": bool(rest) and "unnamed" not in counts}


# --------------------------------------------------------------------------
# Clustering + suggestions (numpy; sklearn DBSCAN)
# --------------------------------------------------------------------------

def cluster_embeddings(emb, *, threshold: float, min_size: int = 2,
                       linkage: str = "average"):
    """Agglomerative clustering over L2-normalized ArcFace embeddings: labels
    per face, -1 for a face in a group smaller than `min_size`.

    Average linkage, not DBSCAN: measured on the 2026-10-25 sample (1,816
    faces), DBSCAN chains — at eps 0.95 its largest cluster spanned 31
    videos with a p90 pairwise distance of 1.44 (Calvin plus a team of other
    kids) — while average linkage at 1.0 kept every cluster whose faces the
    library can identify single-person but one face."""
    import numpy as np
    x = np.asarray(emb, np.float32)
    if len(x) < 2:
        return np.full(len(x), -1, int)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                  linkage=linkage).fit_predict(x)
    counts = np.bincount(lab)
    return np.where(counts[lab] >= min_size, lab, -1)


def suggest_person(cluster_emb, ref_pids, ref_emb, *, max_dist: float = SUGGEST_MAX_DIST,
                   margin: float = SUGGEST_MARGIN) -> Optional[dict]:
    """A conservative name SUGGESTION for one cluster: per person, the median
    over the cluster's faces of the nearest-reference distance. Suggested only
    when the best person is within `max_dist` AND beats the runner-up by
    `margin`."""
    import numpy as np
    ce = np.asarray(cluster_emb, np.float32)
    re_ = np.asarray(ref_emb, np.float32)
    if not len(ce) or not len(re_):
        return None
    ce = ce / np.maximum(np.linalg.norm(ce, axis=1, keepdims=True), 1e-9)
    re_ = re_ / np.maximum(np.linalg.norm(re_, axis=1, keepdims=True), 1e-9)
    pids = np.asarray(ref_pids)
    d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * ce @ re_.T))      # faces x refs
    scores = []
    for p in np.unique(pids):
        scores.append((float(np.median(d[:, pids == p].min(axis=1))), int(p)))
    scores.sort()
    best, pid = scores[0]
    second = scores[1][0] if len(scores) > 1 else 9.0
    if best > max_dist or second - best < margin:
        return None
    return {"person_id": pid, "dist": round(best, 3), "margin": round(second - best, 3)}


# --------------------------------------------------------------------------
# Repair: drop a video's faces / add faces to an existing clustering
# --------------------------------------------------------------------------

def drop_video_faces(clusters: dict, names: dict, photo_ids: Iterable[int]):
    """(clusters, names, report) with every face of `photo_ids` removed —
    from the face table, every cluster, the noise list and the names. Pure:
    returns new dicts. A cluster left with no faces is dropped; every other
    cluster keeps its key, and every other face keeps its name.

    For faces detected on frames that were never the video (2026-10-25: cv2
    returned a frozen buffer for interlaced .m2ts, so 254461 got 35 "faces"
    sharing one box and the owner named the cluster)."""
    drop = {int(p) for p in photo_ids}
    faces = (clusters or {}).get("faces") or {}
    gone = {f for f, m in faces.items() if int(m["photo_id"]) in drop}
    out = dict(clusters)
    out["faces"] = {f: m for f, m in faces.items() if f not in gone}
    kept, emptied = [], []
    for c in clusters.get("clusters") or []:
        fs = [f for f in c["faces"] if f not in gone]
        if fs:
            kept.append(dict(c, faces=fs))
        else:
            emptied.append(c["key"])
    out["clusters"] = kept
    out["noise"] = [f for f in clusters.get("noise") or [] if f not in gone]
    new_names = {f: n for f, n in (names or {}).items() if f not in gone}
    return out, new_names, {"faces_removed": len(gone),
                            "names_removed": len(names or {}) - len(new_names),
                            "clusters_emptied": emptied}


def assign_incremental(member_emb: dict, new_emb, *, threshold: float):
    """For each new face (rows of `new_emb`), the existing cluster key whose
    members sit closest on AVERAGE (mean L2 to every member — the criterion
    the average-linkage clustering merges on, so the same `threshold` means
    the same thing), or None when no cluster is within `threshold`.

    `member_emb`: {cluster_key: (n, 512) array} — the members that speak for
    the cluster (callers leave out junk / ejected faces)."""
    import numpy as np
    x = np.asarray(new_emb, np.float32).reshape(-1, 512) if len(new_emb) else np.zeros((0, 512), np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    best = [(None, 9.0)] * len(x)
    for key, m in member_emb.items():
        m = np.asarray(m, np.float32)
        if not len(m) or not len(x):
            continue
        m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * x @ m.T)).mean(axis=1)
        for i, di in enumerate(d):
            if di <= threshold and di < best[i][1]:
                best[i] = (key, float(di))
    return [k for k, _ in best]


def sample_positions() -> dict:
    """{photo_id: (n, N)} — each video's 1-based place in the sample."""
    photos = me.load_video_sample()["photos"]
    return {int(p["photo_id"]): (i + 1, len(photos)) for i, p in enumerate(photos)}


def video_ident(photo_id: int, positions: Optional[dict] = None,
                dates: Optional[dict] = None) -> dict:
    """What identifies a sampled video to the owner: its place in the sample
    (#n of N), the photo id, the file name and date, and the main-app deep
    link (index.html opens PS.PhotoModal for ?photo=<id>). Never the
    dev/holdout split, never a model."""
    pid = int(photo_id)
    entry = me.video_sample_entry(pid) or {}
    pos = (positions if positions is not None else sample_positions()).get(pid)
    return {"photo_id": pid, "filename": entry.get("filename"),
            "date": (dates or {}).get(pid),
            "position": pos[0] if pos else None, "of": pos[1] if pos else None,
            "app_url": f"/?photo={pid}"}

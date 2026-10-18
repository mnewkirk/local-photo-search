"""Server side of the ``video-probe`` pass: store one probe result and decide
Live Photo pairing (docs/plans/video-support.md step 2b).

Pure SQL over the NAS DB — no decoder, no file access (D19). Called from
``worker_api.submit_results`` once per video inside its own transaction, so a
photo's writes land together or not at all.

**What a result writes.**

- ``video_meta``: the probe columns, ``probe_version``, ``probe_json`` and —
  for an unreadable file — ``probe_error`` (``zip_wrapped`` or
  ``unreadable: <reason>``). A stored error IS the result: the row is then
  out of the probe queue until ``PROBE_VERSION`` is bumped.
- ``photos``: ``date_taken`` when the per-family date policy produced one
  (otherwise the registration's provisional mtime stays), display
  ``image_width``/``image_height``, ``camera_make``/``camera_model`` where the
  container names them (None never blanks a value), and GPS with
  ``location_source='exif'``. GPS never overwrites a ``manual`` location;
  it does replace an ``inferred`` one, and clears ``place_name`` so the
  nightly ``geocode`` maintenance stage labels it. Geocoding is NOT done here:
  the rich GeoNames KDTree costs ~1.5 GB resident on the N100, and the stage
  is already missing-only.

**Live Photo pairing** (D3): a probed clip of ``duration <= 4 s`` whose stem
matches a still in the same folder or a +-1-day sibling folder
(``YYYY/YYYY-MM-DD_*``), captured within 3 s of it, becomes ``kind='live'``
with ``live_of_photo_id``; every other readable video is ``kind='clip'``.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Optional

LIVE_MAX_DURATION_S = 4.0
LIVE_MAX_TIME_DELTA_S = 3.0

_META_COLUMNS = ("duration_s", "fps", "codec", "container", "width", "height",
                 "rotation", "bitrate", "has_audio", "poster_t")
_DATE_PREFIX = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


class NotAVideo(ValueError):
    pass


def _parse_dt(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).strip().replace("T", " ")[:19])
    except ValueError:
        return None


def _stem(filename: str) -> str:
    return os.path.splitext(filename or "")[0].lower()


def _like_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def candidate_folder_ranges(folder: Optional[str]) -> list[tuple[str, str]]:
    """[lo, hi) `photos.folder` ranges a Live still may sit in: the video's own
    folder plus, when its folder name starts with a date, every folder of the
    day before / the day itself / the day after under the same parent (a year
    parent follows the date across New Year)."""
    folder = folder or ""
    ranges = [(folder, folder + "\x00")]  # exactly this folder
    parent, _, base = folder.rpartition("/")
    m = _DATE_PREFIX.match(base)
    if not m:
        return ranges
    try:
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return ranges
    for delta in (-1, 0, 1):
        dd = d + timedelta(days=delta)
        p2 = str(dd.year) if parent == str(d.year) else parent
        prefix = f"{p2}/{dd.isoformat()}" if p2 else dd.isoformat()
        ranges.append((prefix, prefix + "\U0010ffff"))
    return ranges


def find_live_still(conn, photo_id: int, filename: str, folder: Optional[str],
                    date_taken, duration_s) -> Optional[int]:
    """The still this clip is the motion half of, or None."""
    if duration_s is None or float(duration_s) > LIVE_MAX_DURATION_S:
        return None
    t_vid = _parse_dt(date_taken)
    if t_vid is None:
        return None
    stem = _stem(filename)
    if not stem:
        return None
    like = _like_escape(stem) + ".%"
    best = None
    seen = set()
    for lo, hi in candidate_folder_ranges(folder):
        rows = conn.execute(
            "SELECT id, filename, date_taken FROM photos "
            "WHERE folder >= ? AND folder < ? AND media_type IS NULL "
            "AND filename LIKE ? ESCAPE '\\'",
            (lo, hi, like)).fetchall()
        for r in rows:
            if r[0] in seen or r[0] == photo_id or _stem(r[1]) != stem:
                continue
            seen.add(r[0])
            t = _parse_dt(r[2])
            if t is None:
                continue
            dt = abs((t - t_vid).total_seconds())
            if dt <= LIVE_MAX_TIME_DELTA_S and (best is None or dt < best[0]):
                best = (dt, r[0])
    return best[1] if best else None


def apply_probe_result(conn, r: dict) -> dict:
    """Write one probe result (a VideoProbeResult as a dict). Raises
    NotAVideo for a non-video id; any sqlite error propagates to the caller,
    which defers a transient lock and spends the attempt otherwise. Does NOT
    commit."""
    pid = int(r["photo_id"])
    row = conn.execute(
        "SELECT id, filename, folder, media_type, date_taken, gps_lat, "
        "location_source, gps_lon FROM photos WHERE id = ?", (pid,)).fetchone()
    if row is None:
        raise NotAVideo(f"photo {pid} does not exist")
    if row[3] != "video":
        raise NotAVideo(f"photo {pid} is not a video")
    version = int(r["probe_version"])
    probe_json = r.get("probe_json")
    if probe_json is not None and not isinstance(probe_json, str):
        probe_json = json.dumps(probe_json, sort_keys=True)

    conn.execute("INSERT OR IGNORE INTO video_meta (photo_id) VALUES (?)", (pid,))

    if r.get("probe_error"):
        conn.execute(
            "UPDATE video_meta SET probe_error = ?, probe_version = ?, "
            "probe_json = ?, kind = NULL, live_of_photo_id = NULL, "
            "duration_s = NULL, fps = NULL, codec = NULL, container = NULL, "
            "width = NULL, height = NULL, rotation = NULL, bitrate = NULL, "
            "has_audio = NULL, poster_t = NULL WHERE photo_id = ?",
            (str(r["probe_error"])[:500], version, probe_json, pid))
        return {"photo_id": pid, "probe_error": r["probe_error"], "kind": None}

    vals = []
    for c in _META_COLUMNS:
        v = r.get(c)
        if c == "has_audio" and v is not None:
            v = 1 if v else 0
        vals.append(v)
    conn.execute(
        "UPDATE video_meta SET "
        + ", ".join(f"{c} = ?" for c in _META_COLUMNS)
        + ", probe_error = NULL, probe_version = ?, probe_json = ? WHERE photo_id = ?",
        (*vals, version, probe_json, pid))

    sets, params = [], []
    if r.get("width") and r.get("height"):
        sets += ["image_width = ?", "image_height = ?"]
        params += [int(r["width"]), int(r["height"])]
    for c in ("camera_make", "camera_model"):
        if r.get(c):
            sets.append(f"{c} = ?")
            params.append(str(r[c]))
    date_taken = row[4]
    if r.get("date_taken") and _parse_dt(r["date_taken"]) is not None:
        date_taken = str(r["date_taken"])
        sets.append("date_taken = ?")
        params.append(date_taken)
    lat, lon = r.get("gps_lat"), r.get("gps_lon")
    if lat is not None and lon is not None and row[6] != "manual":
        sets += ["gps_lat = ?", "gps_lon = ?", "location_source = 'exif'",
                 "location_confidence = NULL"]
        params += [float(lat), float(lon)]
        moved = (row[5] is None or row[7] is None
                 or abs(row[5] - float(lat)) > 1e-6 or abs(row[7] - float(lon)) > 1e-6)
        if moved or row[6] != "exif":
            sets.append("place_name = NULL")
    if sets:
        conn.execute(f"UPDATE photos SET {', '.join(sets)} WHERE id = ?",
                     (*params, pid))

    still = find_live_still(conn, pid, row[1], row[2], date_taken, r.get("duration_s"))
    kind = "live" if still else "clip"
    conn.execute("UPDATE video_meta SET kind = ?, live_of_photo_id = ? "
                 "WHERE photo_id = ?", (kind, still, pid))
    return {"photo_id": pid, "probe_error": None, "kind": kind,
            "live_of_photo_id": still, "date_taken": date_taken}

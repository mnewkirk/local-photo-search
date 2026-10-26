"""The ``video-probe`` worker pass: ffprobe over HTTP Range + one poster frame
(docs/plans/video-support.md step 2b; Step 0 results override the plan).

Runs on the DESKTOP (D19) against the NAS's ``/api/photos/{id}/full?purpose=
worker`` URL, which is Range-capable and rate-capped. Three network touches
per video, none of them a whole-file download:

1. **Magic check** — one 12-byte Range read. Most "moov atom not found" files
   in the library are not corrupt: they are ZIP archives (``PK\\x03\\x04``)
   with a video extension (the iOS Live-bundle bug). Those get the DISTINCT
   verdict :data:`ZIP_WRAPPED`, so the later ``video-scan`` pass knows to
   unzip-then-process instead of skipping (unwrapping costs a full
   decompress, so it belongs to the scan's single download, not here).
2. **ffprobe** ``-show_format -show_streams`` over HTTP. The spike measured
   it reading 0.5-15% of a file, and < 1% of one with no readable container,
   so there is no head+tail pre-check.
3. **One cv2 first-frame grab** over HTTP for the poster (thumb / preview /
   full JPEGs). cv2 auto-applies the rotation, so the frame is already in
   display orientation.

Everything that turns ffprobe JSON into stored columns is a pure function
(:func:`parse_probe`, :func:`capture_date`, :func:`parse_iso6709`, ...) so CI
tests it from canned JSON (``tests/fixtures/video_probe/``); ffprobe and cv2
are never on CI.

**Date policy (D14, per camera FAMILY — Step 0).** ``date_taken`` is stored
as local wall time ``YYYY-MM-DD HH:MM:SS`` with no zone, exactly like a
photo's EXIF ``DateTimeOriginal`` (exif.py). The container's
``creation_time`` means different things per family:

==========  ===================================================  ==========
family      rule                                                 source
==========  ===================================================  ==========
apple       ``com.apple.quicktime.creationdate`` verbatim (it    apple_cd
            carries the local time + offset; drop the offset).
            Without it: ``creation_time`` is true UTC -> tz.     utc
android     ``creation_time`` is the recording END in UTC        utc_end
            (Pixel muxer stamps at finalization):
            ``creation_time - duration``, converted to tz.
sony/kodak  ``creation_time`` says "Z" but is the camera's       local_z
            LOCAL clock: used as-is, no conversion.
unknown     nothing trusted — the registration's provisional     mtime
            mtime stays (``date_taken`` is not sent).
==========  ===================================================  ==========

"tz" (apple-without-creationdate, android) is ``PHOTOSEARCH_TZ``, default
``America/Los_Angeles`` — the library's home zone. A phone has no offset in
its video metadata, so a clip shot abroad lands at home-zone wall time; the
spike's n~10/family is enough to see the families, not to resolve that.
Re-validate on a larger sample before bumping :data:`PROBE_VERSION`.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

# Bump to re-probe every video (the claim predicate is "probe_version IS NULL
# OR < PROBE_VERSION"). Also the version directory posters upload into.
PROBE_VERSION = 1

# Stored probe_error values. The prefix is what classify_probe_error reads.
ZIP_WRAPPED = "zip_wrapped"
UNREADABLE_PREFIX = "unreadable"
ZIP_MAGIC = b"PK\x03\x04"

# Poster sizes (long edge). thumb / preview match web._THUMB_SIZE /
# web._PREVIEW_SIZE so a video's poster is cached exactly like a photo's
# thumbnail/preview. "full" is capped: an 8K first frame as a q85 JPEG can
# exceed video_api.MAX_ASSET_BYTES, and nothing needs more than 4K of poster.
POSTER_SIZES = {"thumb": 600, "preview": 1920, "full": 4096}
POSTER_QUALITY = {"thumb": 85, "preview": 82, "full": 85}

DEFAULT_TZ = "America/Los_Angeles"
FFPROBE_TIMEOUT_S = 120

# ffprobe stderr markers that mean "the bytes were read and they are not a
# playable container" (a verdict) vs. "the read itself failed" (transport —
# the photo is omitted and retried at no cost).
_CORRUPT_MARKERS = ("moov atom not found", "invalid data found",
                    "could not find codec parameters", "no such stream",
                    "invalid argument", "end of file")
_TRANSPORT_MARKERS = ("server returned", "connection refused", "connection reset",
                      "connection timed out", "timed out", "network is unreachable",
                      "failed to resolve", "http error", "broken pipe",
                      "input/output error", "error number -138")


class ProbeTransportError(Exception):
    """The NAS could not be read (timeout, 5xx, refused). Never a verdict."""


class ProbeToolError(Exception):
    """ffprobe failed for a reason that is neither transport nor a verdict
    (unknown stderr). Reported as a worker failure: one attempt spent."""


class ProbeCrashError(RuntimeError):
    """ffprobe exited via a signal (SIGSEGV etc — ``returncode < 0``,
    measured 2026-10-25 when the static ffprobe build resolved a hostname).
    This says nothing about the file: it is not a verdict and not a
    per-video failure, and whatever broke the tool will break on every
    later video too. The worker halts the pass (``PassHalted``) rather than
    spending an attempt per video."""


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_probe_error(probe_error: Optional[str]) -> Optional[str]:
    """'zip_wrapped' | 'unreadable' for a stored probe_error, None if none.

    ``video-scan`` claims the zip_wrapped ones (unzip, then process the inner
    file); everything else is genuinely unreadable."""
    if not probe_error:
        return None
    if probe_error.startswith(ZIP_WRAPPED):
        return "zip_wrapped"
    return "unreadable"


def unreadable_error(reason: str) -> str:
    reason = " ".join((reason or "").split())[:300] or "unknown"
    return f"{UNREADABLE_PREFIX}: {reason}"


def is_zip_magic(head: bytes) -> bool:
    return bytes(head[:4]) == ZIP_MAGIC


def classify_ffprobe_failure(stderr: str) -> str:
    """'corrupt' | 'transport' | 'unknown' for a failed ffprobe run."""
    s = (stderr or "").lower()
    if any(m in s for m in _TRANSPORT_MARKERS):
        return "transport"
    if any(m in s for m in _CORRUPT_MARKERS):
        return "corrupt"
    return "unknown"


def _first_error_line(stderr: str) -> str:
    """The most informative stderr line ('moov atom not found'), with the
    `[mov,mp4 @ 0x...]` prefix and the URL stripped."""
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for ln in lines:
        m = re.match(r"^\[[^\]]*\]\s*(.*)$", ln)
        if m and m.group(1):
            return m.group(1)
    if lines:
        # "<url>: Invalid data found when processing input"
        return lines[-1].rsplit(": ", 1)[-1]
    return "unknown"


# ---------------------------------------------------------------------------
# Pure parsing
# ---------------------------------------------------------------------------

def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def parse_rate(rate: Optional[str]) -> Optional[float]:
    """'30000/1001' -> 29.97; '0/0' -> None."""
    if not rate:
        return None
    if "/" in str(rate):
        a, b = str(rate).split("/", 1)
        na, nb = _num(a), _num(b)
        if not na or not nb:
            return None
        return round(na / nb, 3)
    v = _num(rate)
    return round(v, 3) if v else None


def stream_rotation(stream: dict) -> int:
    """Clockwise display rotation in {0, 90, 180, 270}, the `rotate` tag's
    convention. ffprobe 5+ reports it as a display-matrix side-data
    `rotation` with the opposite sign (-90 == rotate 90); older files carry a
    `rotate` tag."""
    deg = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            r = _num(sd.get("rotation"))
            if r is not None:
                deg = -r
                break
    if deg is None:
        r = _num((stream.get("tags") or {}).get("rotate"))
        if r is not None:
            deg = r
    if deg is None:
        return 0
    return int(round(deg / 90.0)) * 90 % 360


_ISO6709 = re.compile(r"^\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)?(?:CRS[^/]*)?/?\s*$")


def _iso_part(token: str, deg_digits: int) -> Optional[float]:
    """One ISO 6709 coordinate: ±DD[.d], ±DDMM[.m] or ±DDMMSS[.s] (lat,
    deg_digits=2) / ±DDD... (lon, deg_digits=3)."""
    sign = -1.0 if token[0] == "-" else 1.0
    body = token[1:]
    whole, _, frac = body.partition(".")
    frac = f".{frac}" if frac else ""
    n = len(whole)
    if n == deg_digits:
        v = float(whole + frac)
    elif n == deg_digits + 2:
        v = int(whole[:deg_digits]) + float(whole[deg_digits:] + frac) / 60
    elif n == deg_digits + 4:
        v = (int(whole[:deg_digits]) + int(whole[deg_digits:deg_digits + 2]) / 60
             + float(whole[deg_digits + 2:] + frac) / 3600)
    else:
        try:
            v = float(whole + frac)
        except ValueError:
            return None
    return sign * v


def parse_iso6709(s: Optional[str]) -> Optional[tuple[float, float, Optional[float]]]:
    """'+37.3317-122.0307+010.869/' -> (37.3317, -122.0307, 10.869).
    None for anything unparseable, out of range, or the 0,0 null island."""
    if not s:
        return None
    m = _ISO6709.match(str(s))
    if not m:
        return None
    lat = _iso_part(m.group(1), 2)
    lon = _iso_part(m.group(2), 3)
    alt = _num(m.group(3)) if m.group(3) else None
    if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    if lat == 0 and lon == 0:
        return None
    return round(lat, 6), round(lon, 6), alt


_GPS_TAGS = ("com.apple.quicktime.location.ISO6709", "location", "location-eng")


def _lower_tags(tags: Optional[dict]) -> dict:
    return {str(k).lower(): v for k, v in (tags or {}).items()}


def camera_family(fmt_tags: Optional[dict]) -> str:
    """'apple' | 'android' | 'sony' | 'kodak' | 'unknown' from container tags.

    Keyed the way the spike found the families: Apple and Android write their
    own namespaced tags; Sony XAVC bodies write NO make/model at all, only the
    `XAVC` major brand; the Kodak PIXPRO writes `original_format: KODAK` and
    the model in `comment`."""
    t = _lower_tags(fmt_tags)
    if t.get("com.apple.quicktime.creationdate") or \
            str(t.get("com.apple.quicktime.make", "")).strip().lower() == "apple":
        return "apple"
    if any(k.startswith("com.android.") for k in t):
        return "android"
    brand = str(t.get("major_brand", "")).strip().upper()
    make = str(t.get("make", "") or t.get("com.apple.quicktime.make", "")).strip().upper()
    if brand.startswith("XAVC") or make.startswith("SONY"):
        return "sony"
    if str(t.get("original_format", "")).strip().upper() == "KODAK" or \
            str(t.get("comment", "")).strip().upper().startswith("PIXPRO"):
        return "kodak"
    return "unknown"


def camera_make_model(fmt_tags: Optional[dict], family: str) -> tuple[Optional[str], Optional[str]]:
    """(make, model) matching what the same body writes into its photos' EXIF
    where the container says it; None leaves the row's value untouched."""
    t = _lower_tags(fmt_tags)

    def g(*keys):
        for k in keys:
            v = t.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return None

    if family == "apple":
        return g("com.apple.quicktime.make", "make"), g("com.apple.quicktime.model", "model")
    if family == "android":
        return g("com.android.manufacturer", "make"), g("com.android.model", "model")
    if family == "sony":
        # XAVC carries no model; the photos say "SONY" / "ILCE-7RM6".
        return "SONY", g("model")
    if family == "kodak":
        # The stills say make "JK Imaging, Ltd.", model "KODAK PIXPRO WPZ2".
        fmt, comment = g("original_format"), g("comment")
        model = f"{fmt} {comment}" if fmt and comment else (comment or None)
        return None, model
    return g("make", "com.apple.quicktime.make"), g("model", "com.apple.quicktime.model")


def _parse_creation_time(v) -> Optional[datetime]:
    """'2026-09-04T16:59:24.000000Z' -> aware UTC datetime. Container epoch
    zeros (1904 / 1970) are "unset", not a date."""
    if not v:
        return None
    s = str(v).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if dt.year < 1980:
        return None
    return dt


def _fmt_wall(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _zone(tz_name: Optional[str]):
    name = tz_name or os.environ.get("PHOTOSEARCH_TZ") or DEFAULT_TZ
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001 — no tzdata: fall back to UTC, loudly in the source tag
        return None


def _to_local(dt_utc: datetime, tz_name: Optional[str]) -> tuple[str, str]:
    z = _zone(tz_name)
    if z is None:
        return _fmt_wall(dt_utc.astimezone(timezone.utc)), "UTC"
    return _fmt_wall(dt_utc.astimezone(z)), str(z)


def _date_apple(tags: dict, duration_s, tz_name):
    cd = tags.get("com.apple.quicktime.creationdate")
    if cd:
        s = str(cd).strip()
        # Basic-format offset "+0200" -> "+02:00" for older Pythons.
        s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.year >= 1980:
                # Its own wall clock, offset dropped — what the iPhone's
                # stills store as DateTimeOriginal.
                return _fmt_wall(dt), "apple_cd"
        except ValueError:
            pass
    ct = _parse_creation_time(tags.get("creation_time"))
    if ct is not None:
        return _to_local(ct, tz_name)[0], "utc"
    return None, "mtime"


def _date_android(tags: dict, duration_s, tz_name):
    ct = _parse_creation_time(tags.get("creation_time"))
    if ct is None:
        return None, "mtime"
    start = ct - timedelta(seconds=float(duration_s or 0))
    return _to_local(start, tz_name)[0], "utc_end"


def _date_local_z(tags: dict, duration_s, tz_name):
    ct = _parse_creation_time(tags.get("creation_time"))
    if ct is None:
        return None, "mtime"
    # The 'Z' is a lie: these bodies write their local clock.
    return _fmt_wall(ct.replace(tzinfo=None)), "local_z"


def _date_unknown(tags: dict, duration_s, tz_name):
    return None, "mtime"


# The D14 dispatch. A family missing here means "trust nothing" (mtime).
DATE_POLICY: dict[str, Callable] = {
    "apple": _date_apple,
    "android": _date_android,
    "sony": _date_local_z,
    "kodak": _date_local_z,
    "unknown": _date_unknown,
}


def capture_date(fmt_tags: Optional[dict], duration_s: Optional[float],
                 tz_name: Optional[str] = None,
                 family: Optional[str] = None) -> tuple[Optional[str], str]:
    """(local wall-time 'YYYY-MM-DD HH:MM:SS' or None, source). None means
    keep the provisional mtime the row was registered with."""
    fam = family or camera_family(fmt_tags)
    t = _lower_tags(fmt_tags)
    fn = DATE_POLICY.get(fam, _date_unknown)
    return fn(t, duration_s, tz_name)


def _container(fmt: dict) -> Optional[str]:
    name = str(fmt.get("format_name") or "")
    if not name:
        return None
    if name.startswith("mov,mp4"):
        brand = str((fmt.get("tags") or {}).get("major_brand", "")).strip().lower()
        return "mov" if brand == "qt" else "mp4"
    return name.split(",")[0]


def _trim_for_storage(data: dict) -> dict:
    """The ffprobe JSON as stored in video_meta.probe_json: format + streams,
    minus the bulky display-matrix dump."""
    fmt = {k: v for k, v in (data.get("format") or {}).items() if k != "filename"}
    streams = []
    for s in data.get("streams") or []:
        s2 = {k: v for k, v in s.items() if k not in ("disposition",)}
        if "side_data_list" in s2:
            s2["side_data_list"] = [{k: v for k, v in sd.items() if k != "displaymatrix"}
                                    for sd in s2["side_data_list"]]
        streams.append(s2)
    return {"format": fmt, "streams": streams}


def parse_probe(data: dict, tz_name: Optional[str] = None) -> dict:
    """ffprobe JSON -> the probe result dict the worker submits.

    Returns ``{"probe_error": "unreadable: no video stream", ...}`` when the
    container parsed but holds no video. Width/height are DISPLAY dimensions
    (swapped for a 90/270 rotation), which is what cv2 decodes and what a
    photo's image_width/image_height mean (EXIF-oriented)."""
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and (s.get("disposition") or {}).get("attached_pic") != 1), None)
    stored = _trim_for_storage(data)
    if video is None:
        return {"probe_error": unreadable_error("no video stream"),
                "probe_json": stored}
    rotation = stream_rotation(video)
    w, h = video.get("width"), video.get("height")
    if rotation in (90, 270) and w and h:
        w, h = h, w
    duration = _num(fmt.get("duration")) or _num(video.get("duration"))
    fps = parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate"))
    bitrate = _num(fmt.get("bit_rate")) or _num(video.get("bit_rate"))
    fmt_tags = fmt.get("tags") or {}
    family = camera_family(fmt_tags)
    make, model = camera_make_model(fmt_tags, family)
    date_taken, date_source = capture_date(fmt_tags, duration, tz_name, family)
    lt = _lower_tags(fmt_tags)
    gps = None
    for k in _GPS_TAGS:
        gps = parse_iso6709(lt.get(k.lower()))
        if gps:
            break
    out = {
        "probe_error": None,
        "duration_s": round(duration, 3) if duration is not None else None,
        "fps": fps,
        "codec": video.get("codec_name"),
        "container": _container(fmt),
        "width": int(w) if w else None,
        "height": int(h) if h else None,
        "rotation": rotation,
        "bitrate": int(bitrate) if bitrate else None,
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "camera_family": family,
        "camera_make": make,
        "camera_model": model,
        "date_taken": date_taken,
        "date_source": date_source,
        "gps_lat": gps[0] if gps else None,
        "gps_lon": gps[1] if gps else None,
        "gps_alt": gps[2] if gps else None,
    }
    stored["derived"] = {"camera_family": family, "date_source": date_source}
    out["probe_json"] = stored
    return out


# ---------------------------------------------------------------------------
# Tools (desktop only — never on CI, never on the NAS)
# ---------------------------------------------------------------------------

def find_ffprobe() -> Optional[str]:
    """PHOTOSEARCH_FFPROBE, else PATH, else ~/.local/bin/ffprobe (the
    static build the spike installed without sudo)."""
    env = os.environ.get("PHOTOSEARCH_FFPROBE")
    if env:
        return env if os.path.exists(env) or shutil.which(env) else None
    found = shutil.which("ffprobe")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "ffprobe"
    return str(local) if local.exists() else None


def run_ffprobe(url: str, ffprobe: Optional[str] = None,
                timeout: float = FFPROBE_TIMEOUT_S,
                run=subprocess.run) -> dict:
    """ffprobe JSON for `url`.

    Raises ProbeTransportError (omit, retry free — includes a DNS failure
    resolving the URL's host), ProbeCrashError (the tool itself crashed —
    halt the pass, no attempt spent), ProbeToolError (worker failure, one
    attempt), and returns ``{"_verdict": "<error>"}`` when the bytes were
    read and are not a playable container."""
    from .video_decode import HostResolutionError, ffmpeg_url

    exe = ffprobe or find_ffprobe()
    if not exe:
        raise ProbeToolError("ffprobe not found (set PHOTOSEARCH_FFPROBE)")
    try:
        resolved_url = ffmpeg_url(url)
    except HostResolutionError as e:
        raise ProbeTransportError(str(e)) from e
    cmd = [exe, "-v", "error", "-print_format", "json", "-show_format",
           "-show_streams",
           # Label the reads in the NAS request log; reconnect on a dropped
           # HTTP read rather than misreading it as EOF.
           "-headers", "X-Photosearch-Source: worker\r\n",
           "-user_agent", "photosearch-worker-ffprobe",
           "-reconnect", "1", "-reconnect_streamed", "0",
           resolved_url]
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ProbeTransportError(f"ffprobe timed out after {timeout:.0f}s")
    stderr = proc.stderr or ""
    data = None
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError:
        data = None
    if proc.returncode == 0 and data and (data.get("format") or data.get("streams")):
        return data
    # A negative returncode is the process dying to a signal (SIGSEGV = -11)
    # — the tool crashed, not a verdict about this file. stderr from a
    # segfault is typically empty, so this must be checked before falling
    # through to classify_ffprobe_failure's "unknown" bucket, which used to
    # raise ProbeToolError here and spend an attempt on a perfectly good
    # video (2026-10-25: all 31 videos in 2026-10-25_ILCE-7RM6).
    if proc.returncode < 0:
        raise ProbeCrashError(
            f"ffprobe crashed (signal {-proc.returncode}): "
            f"{_first_error_line(stderr) if stderr else 'no stderr'}")
    kind = classify_ffprobe_failure(stderr)
    if kind == "transport":
        raise ProbeTransportError(_first_error_line(stderr))
    if kind == "corrupt":
        return {"_verdict": unreadable_error(_first_error_line(stderr))}
    raise ProbeToolError(f"ffprobe exit {proc.returncode}: {_first_error_line(stderr)}")


def video_stream_interlaced(data: dict) -> bool:
    """ffprobe JSON -> is the first video stream interlaced (field_order)?"""
    from .video_decode import is_interlaced
    for st in (data or {}).get("streams") or []:
        if st.get("codec_type") == "video":
            return is_interlaced(st.get("field_order"))
    return False


def grab_first_frame(url: str, attempts: int = 2, interlaced: bool = False):
    """(RGB numpy frame, t_seconds) of the first decodable frame, or
    (None, None) — ``video_decode.grab_frame``: cv2 (lazy; CI mocks it; it
    applies the container rotation) with an ffmpeg fallback for a black or
    missing frame, and ffmpeg only for an interlaced stream (cv2 returns
    black or a stale buffer on AVCHD .m2ts, 2026-10-25)."""
    from .video_decode import grab_frame

    for _ in range(max(1, attempts)):
        frame, t = grab_frame(url, 0.0, seek=False, interlaced=bool(interlaced))
        if frame is not None:
            return frame, t
    return None, None


def poster_jpegs(frame_rgb) -> dict[str, bytes]:
    """{size: JPEG bytes} for thumb / preview / full from one RGB frame."""
    from PIL import Image

    img = Image.fromarray(frame_rgb)
    if img.mode != "RGB":
        img = img.convert("RGB")
    out = {}
    for size, edge in POSTER_SIZES.items():
        im = img.copy()
        im.thumbnail((edge, edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=POSTER_QUALITY[size], optimize=True)
        out[size] = buf.getvalue()
    return out


def poster_assets_body(jpegs: dict[str, bytes], version: int = PROBE_VERSION) -> dict:
    """The POST /api/videos/{id}/assets body for a poster set."""
    return {"version": int(version),
            "assets": [{"kind": "poster", "size": size,
                        "data_b64": base64.b64encode(data).decode("ascii")}
                       for size, data in jpegs.items()]}

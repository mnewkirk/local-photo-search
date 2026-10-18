"""Video serving: posters, playback with HTTP Range, the throttled worker
stream, and the worker asset upload (docs/plans/video-support.md step 2).

The NAS never decodes or resizes a video (D19). It serves:

- **Thumbnails / previews** from what the desktop ``video-probe`` pass
  uploaded (``POST /api/videos/{id}/assets``, kind ``poster``), cached under
  the same ``thumbnails/{id}_thumb.jpg`` / ``previews/{id}_preview.jpg`` names
  every photo uses, so grids and the replica's thumbnail mirror work
  unchanged. Until then: a paired Live clip shows its still, an unreadable
  file shows a static "unreadable" placeholder, everything else a static
  "video" placeholder (``frontend/dist/video-*.svg``).
- **``/full``** with single-range HTTP Range (206) and the right mime.
  ``?purpose=worker`` goes through a PROCESS-WIDE token bucket capped at
  ``PHOTOSEARCH_VIDEO_STREAM_MBPS`` (default 30 MB/s): the 2026-09-19 disk
  starvation was a ~50 MB/s bulk read, and the video backfill is 400 GB of
  exactly that shape. One shared bucket, so ``-n 2`` workers share 30 MB/s
  instead of getting 30 each.
- **Replica mode**: a streaming proxy that forwards ``Range`` and returns the
  NAS's status and range headers, iterating the body — never ``.read()``.

Range handling is done here rather than by Starlette's ``FileResponse``:
Range support only arrived in recent Starlette releases, and the throttled
path needs its own generator anyway, so one implementation serves both.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

VIDEO_MIME = {
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mts": "video/mp2t",
    ".m2ts": "video/mp2t",
    ".3gp": "video/3gpp",
}

# Containers no mainstream browser plays: the modal offers a Download link.
NOT_BROWSER_PLAYABLE = {".avi", ".mts", ".m2ts"}

DEFAULT_STREAM_MBPS = 30.0
CHUNK = 256 * 1024

_STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
PLACEHOLDER_VIDEO = _STATIC_DIR / "video-placeholder.svg"
PLACEHOLDER_UNREADABLE = _STATIC_DIR / "video-unreadable.svg"
PLACEHOLDER_MIME = "image/svg+xml"


def video_mime(filename: str) -> str:
    return VIDEO_MIME.get(Path(filename or "").suffix.lower(),
                          "application/octet-stream")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def video_state(conn, photo_id: int) -> dict:
    """The video_meta fields the serving routes branch on ({} if no row)."""
    row = conn.execute(
        "SELECT kind, live_of_photo_id, probe_version, probe_error, duration_s "
        "FROM video_meta WHERE photo_id = ?", (photo_id,)).fetchone()
    return dict(row) if row else {}


def image_source(state: dict) -> str:
    """Which image a video's thumbnail/preview route serves:
    'live' (the paired still's), 'unreadable', 'unprobed' or 'poster'."""
    if state.get("kind") == "live" and state.get("live_of_photo_id"):
        return "live"
    if state.get("probe_error"):
        return "unreadable"
    if state.get("probe_version") is None:
        return "unprobed"
    return "poster"


def video_fields(conn, rows) -> dict[int, dict]:
    """{photo_id: {media_type, duration_s, video_kind}} for the video rows
    among `rows` (dicts or sqlite rows with id + media_type). Stills are not
    looked up at all."""
    ids = []
    for r in rows:
        try:
            if r["media_type"] == "video":
                ids.append(r["id"])
        except (KeyError, IndexError, TypeError):
            continue
    out: dict[int, dict] = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        ph = ",".join("?" * len(chunk))
        for m in conn.execute(
                f"SELECT photo_id, duration_s, kind FROM video_meta "
                f"WHERE photo_id IN ({ph})", chunk):
            out[m["photo_id"]] = {"duration_s": m["duration_s"],
                                  "video_kind": m["kind"]}
    for pid in ids:
        out.setdefault(pid, {"duration_s": None, "video_kind": None})
        out[pid]["media_type"] = "video"
    return out


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------

class RangeNotSatisfiable(Exception):
    pass


_RANGE_RE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", re.I)


def parse_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """(start, end_inclusive) for a single-range ``Range`` header, or None to
    serve the whole file (no header, a multi-range or unparseable header —
    RFC 9110 lets a server ignore those). Raises RangeNotSatisfiable."""
    if not header or "," in header:
        return None
    m = _RANGE_RE.match(header)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    if a == "" and b == "":
        return None
    if a == "":  # suffix: the last N bytes
        n = int(b)
        if n == 0 or size == 0:
            raise RangeNotSatisfiable()
        return max(0, size - n), size - 1
    start = int(a)
    if start >= size:
        raise RangeNotSatisfiable()
    end = size - 1 if b == "" else min(int(b), size - 1)
    if end < start:
        raise RangeNotSatisfiable()
    return start, end


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------

class TokenBucket:
    """Thread-safe token bucket shared by every worker stream. Tokens are
    reserved under the lock (going negative is allowed) and the wait is slept
    OUTSIDE it, so concurrent streams queue fairly without holding the lock."""

    def __init__(self, rate_bps: float, burst: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate = float(rate_bps)
        self.burst = float(burst if burst is not None else max(CHUNK, rate_bps / 4))
        self.tokens = self.burst
        self.clock = clock
        self.sleep = sleep
        self.last = clock()
        self.lock = threading.Lock()

    def consume(self, n: int) -> float:
        with self.lock:
            now = self.clock()
            self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
            self.last = now
            self.tokens -= n
            wait = -self.tokens / self.rate if self.tokens < 0 else 0.0
        if wait > 0:
            self.sleep(wait)
        return wait


_bucket_lock = threading.Lock()
_bucket: Optional[TokenBucket] = None


def stream_rate_bps() -> float:
    try:
        mbps = float(os.environ.get("PHOTOSEARCH_VIDEO_STREAM_MBPS") or DEFAULT_STREAM_MBPS)
    except ValueError:
        mbps = DEFAULT_STREAM_MBPS
    return max(0.1, mbps) * 1_000_000


def worker_bucket() -> TokenBucket:
    """The process-wide bucket; rebuilt only if the configured rate changed."""
    global _bucket
    rate = stream_rate_bps()
    with _bucket_lock:
        if _bucket is None or _bucket.rate != rate:
            _bucket = TokenBucket(rate)
        return _bucket


def iter_file(path: str, start: int, length: int,
              bucket: Optional[TokenBucket] = None,
              chunk: int = CHUNK) -> Iterator[bytes]:
    """Yield `length` bytes of `path` from `start`, through `bucket` if set."""
    remaining = length
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            n = min(chunk, remaining)
            if bucket is not None:
                bucket.consume(n)
            data = fh.read(n)
            if not data:
                break
            remaining -= len(data)
            yield data


def serve_video_file(range_header: Optional[str], path: str, filename: str,
                     purpose: Optional[str], headers: dict):
    """A 200/206/416 StreamingResponse for a local video file."""
    from fastapi.responses import Response, StreamingResponse

    size = os.path.getsize(path)
    headers = dict(headers)
    headers["Accept-Ranges"] = "bytes"
    try:
        rng = parse_range(range_header, size)
    except RangeNotSatisfiable:
        headers["Content-Range"] = f"bytes */{size}"
        return Response(status_code=416, headers=headers)
    bucket = worker_bucket() if purpose == "worker" else None
    if rng is None:
        start, length, status = 0, size, 200
    else:
        start, end = rng
        length, status = end - start + 1, 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    return StreamingResponse(iter_file(path, start, length, bucket),
                             status_code=status, media_type=video_mime(filename),
                             headers=headers)


# ---------------------------------------------------------------------------
# Replica proxy
# ---------------------------------------------------------------------------

_PROXY_HEADERS = ("content-type", "content-length", "content-range",
                  "accept-ranges", "content-disposition", "cache-control",
                  "last-modified", "etag")


def proxy_video(nas_url: str, photo_id: int, range_header: Optional[str],
                purpose: Optional[str], download: bool,
                session=None, timeout: float = 30.0):
    """Stream /full of a video from the NAS, forwarding Range and returning
    the NAS's status (200/206/416) and range headers. The body is iterated
    (`iter_content`), never read whole: a clip can be gigabytes."""
    import requests
    from fastapi.responses import StreamingResponse
    from .request_intent import outbound_headers

    params = {}
    if purpose:
        params["purpose"] = purpose
    if download:
        params["download"] = "1"
    hdrs = outbound_headers(f"Stream video {photo_id} from the NAS"
                            + (f" (range {range_header})" if range_header else ""))
    if range_header:
        hdrs["Range"] = range_header
    sess = session or requests
    upstream = sess.get(f"{nas_url}/api/photos/{photo_id}/full", params=params,
                        headers=hdrs, stream=True, timeout=timeout)
    out_headers = {k: v for k, v in upstream.headers.items()
                   if k.lower() in _PROXY_HEADERS}

    def body():
        try:
            for chunk in upstream.iter_content(chunk_size=CHUNK):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    return StreamingResponse(body(), status_code=upstream.status_code,
                             headers=out_headers,
                             media_type=upstream.headers.get("content-type"))


# ---------------------------------------------------------------------------
# Asset upload (POST /api/videos/{id}/assets)
# ---------------------------------------------------------------------------

ASSET_KINDS = ("poster", "keyframe", "crop")
POSTER_SIZES = ("thumb", "preview", "full")
MAX_ASSET_BYTES = 12 * 1024 * 1024
MAX_ASSETS = 128
MAX_TOTAL_BYTES = 96 * 1024 * 1024
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jpe?g$")


class AssetError(ValueError):
    pass


def frames_root(db_path: str) -> Path:
    """`<dir of the DB>/video_frames` — /data/video_frames on the NAS."""
    return Path(db_path).resolve().parent / "video_frames"


def atomic_write(path: Path, data: bytes) -> None:
    """Temp file in the same directory, fsync, os.replace — a reader never
    sees a half-written JPEG and a crash never leaves one behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _decode_assets(body: dict) -> tuple[int, list[dict]]:
    if not isinstance(body, dict):
        raise AssetError("body must be a JSON object")
    version = body.get("version")
    if not isinstance(version, int) or isinstance(version, bool) or not 1 <= version <= 10_000:
        raise AssetError("version must be an integer 1..10000")
    assets = body.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AssetError("assets must be a non-empty list")
    if len(assets) > MAX_ASSETS:
        raise AssetError(f"at most {MAX_ASSETS} assets per request")
    out, total, seen = [], 0, set()
    for i, a in enumerate(assets):
        if not isinstance(a, dict):
            raise AssetError(f"asset {i} must be an object")
        kind = a.get("kind")
        if kind not in ASSET_KINDS:
            raise AssetError(f"asset {i}: kind must be one of {ASSET_KINDS}")
        if kind == "poster":
            size = a.get("size")
            if size not in POSTER_SIZES:
                raise AssetError(f"asset {i}: poster size must be one of {POSTER_SIZES}")
            name = f"poster_{size}.jpg"
        else:
            size = None
            name = a.get("name")
            if not isinstance(name, str) or not _NAME_RE.match(name) or ".." in name:
                raise AssetError(f"asset {i}: unsafe or invalid name {name!r} "
                                 f"(letters, digits, . _ -, ending .jpg)")
            name = f"{kind}_{name}"
        if name in seen:
            raise AssetError(f"asset {i}: duplicate {name}")
        seen.add(name)
        try:
            data = base64.b64decode(a.get("data_b64") or "", validate=True)
        except (binascii.Error, ValueError, TypeError):
            raise AssetError(f"asset {i}: data_b64 is not valid base64")
        if not data:
            raise AssetError(f"asset {i}: empty")
        if len(data) > MAX_ASSET_BYTES:
            raise AssetError(f"asset {i}: {len(data)} bytes > {MAX_ASSET_BYTES}")
        if data[:3] != b"\xff\xd8\xff":
            raise AssetError(f"asset {i}: not a JPEG")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise AssetError(f"request exceeds {MAX_TOTAL_BYTES} bytes")
        out.append({"kind": kind, "size": size, "name": name, "data": data})
    return version, out


def store_assets(db_path: str, photo_id: int, body: dict, *,
                 thumb_dir: str, preview_dir: str) -> dict:
    """Validate and write one upload. The caller has checked `photo_id` is a
    video row. Everything is decoded and validated BEFORE the first write, so
    a bad asset rejects the whole request and nothing lands."""
    version, assets = _decode_assets(body)
    root = frames_root(db_path)
    vdir = root / str(int(photo_id)) / f"v{version}"
    written = []
    for a in assets:
        dest = vdir / a["name"]
        # Belt and braces: the name regex already forbids separators.
        if dest.resolve().parent != vdir.resolve():
            raise AssetError(f"refusing path outside {vdir}")
        atomic_write(dest, a["data"])
        entry = {"kind": a["kind"], "name": a["name"], "bytes": len(a["data"]),
                 "path": str(dest.relative_to(root.parent))}
        if a["kind"] == "poster":
            # Install where every grid already looks, so the replica's
            # thumbnail mirror and the photo routes need no video branch.
            if a["size"] == "thumb":
                inst = Path(thumb_dir) / f"{int(photo_id)}_thumb.jpg"
            elif a["size"] == "preview":
                inst = Path(preview_dir) / f"{int(photo_id)}_preview.jpg"
            else:
                inst = root / str(int(photo_id)) / "poster.jpg"
            atomic_write(inst, a["data"])
            entry["installed"] = str(inst)
        written.append(entry)
    return {"photo_id": int(photo_id), "version": version, "written": written}

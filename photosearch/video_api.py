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
        "SELECT kind, live_of_photo_id, probe_version, probe_error, duration_s, "
        "poster_version, scan_version FROM video_meta WHERE photo_id = ?",
        (photo_id,)).fetchone()
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


def poster_version(state: dict):
    """The version that names a poster: `video_meta.poster_version` (schema
    v36), bumped by every poster upload that lands — the probe's first frame,
    then video-scan's most salient keyframe — so the replica re-fetches a
    replaced poster instead of serving its stale cache. Falls back to the
    probe_version for a row the v36 backfill had no poster_t for."""
    v = state.get("poster_version")
    return v if v is not None else state.get("probe_version")


def replica_poster_name(photo_id: int, size: str, state: dict) -> str:
    """The replica's cache name for a NAS poster: versioned, so a new poster
    on the NAS (a new version in the synced video_meta) is a cache miss
    instead of a stale hit. The NAS itself keeps the plain {id}_thumb.jpg /
    {id}_preview.jpg names the asset upload installs."""
    return f"{int(photo_id)}_{size}.p{poster_version(state)}.jpg"


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
    if b != "" and int(b) < start:
        return None  # last-pos < first-pos: invalid, ignored (RFC 9110 14.1.1)
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


class FileStream:
    """`length` bytes of `path` from `start`, through `bucket` if set.

    The file is opened up front and closed by `close()` — called by the
    response when it finishes, errors or the client disconnects — not left
    to garbage collection of a half-iterated generator."""

    def __init__(self, path: str, start: int, length: int,
                 bucket: Optional[TokenBucket] = None, chunk: Optional[int] = None):
        self.fh = open(path, "rb")
        self.fh.seek(start)
        self.remaining = length
        self.bucket = bucket
        self.chunk = chunk or CHUNK

    def __iter__(self) -> Iterator[bytes]:
        try:
            while self.remaining > 0 and not self.fh.closed:
                n = min(self.chunk, self.remaining)
                if self.bucket is not None:
                    self.bucket.consume(n)
                try:
                    data = self.fh.read(n)
                except ValueError:  # closed under us by a disconnect
                    break
                if not data:
                    break
                self.remaining -= len(data)
                yield data
        finally:
            self.close()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass


def iter_file(path: str, start: int, length: int,
              bucket: Optional[TokenBucket] = None,
              chunk: Optional[int] = None) -> Iterator[bytes]:
    return iter(FileStream(path, start, length, bucket, chunk))


def _closing_streaming_response(content, on_close: Callable[[], None], **kw):
    """A StreamingResponse that ALWAYS runs `on_close` once it is done with —
    completed, failed or the client went away (Starlette's `background` is
    skipped on the ASGI 2.4 disconnect path)."""
    from fastapi.responses import StreamingResponse

    class _Closing(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                on_close()

    return _Closing(content, **kw)


# Throttled worker streams sleep inside the threadpool; cap how many can run
# at once so a fleet cannot starve the web server's threads. Over the cap the
# request gets 503 + Retry-After, which WorkerClient._request already backs
# off on.
DEFAULT_WORKER_STREAMS = 4
WORKER_STREAM_RETRY_AFTER = 10
_streams_lock = threading.Lock()
_active_worker_streams = 0


def worker_stream_limit() -> int:
    try:
        return max(1, int(os.environ.get("PHOTOSEARCH_VIDEO_WORKER_STREAMS")
                          or DEFAULT_WORKER_STREAMS))
    except ValueError:
        return DEFAULT_WORKER_STREAMS


def acquire_worker_stream() -> Optional[Callable[[], None]]:
    """A one-shot release callable, or None when the cap is reached."""
    global _active_worker_streams
    with _streams_lock:
        if _active_worker_streams >= worker_stream_limit():
            return None
        _active_worker_streams += 1
    released = threading.Event()

    def release() -> None:
        global _active_worker_streams
        if released.is_set():
            return
        released.set()
        with _streams_lock:
            _active_worker_streams -= 1

    return release


def serve_video_file(range_header: Optional[str], path: str, filename: str,
                     purpose: Optional[str], headers: dict):
    """A 200/206/416 StreamingResponse for a local video file (503 when the
    concurrent purpose=worker cap is reached)."""
    from fastapi.responses import Response

    size = os.path.getsize(path)
    headers = dict(headers)
    headers["Accept-Ranges"] = "bytes"
    try:
        rng = parse_range(range_header, size)
    except RangeNotSatisfiable:
        headers["Content-Range"] = f"bytes */{size}"
        return Response(status_code=416, headers=headers)
    release = None
    bucket = None
    if purpose == "worker":
        release = acquire_worker_stream()
        if release is None:
            return Response(
                status_code=503,
                content=b"too many concurrent worker video streams, retry shortly",
                headers={"Retry-After": str(WORKER_STREAM_RETRY_AFTER)})
        bucket = worker_bucket()
    if rng is None:
        start, length, status = 0, size, 200
    else:
        start, end = rng
        length, status = end - start + 1, 206
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    try:
        stream = FileStream(path, start, length, bucket)
    except BaseException:
        if release:
            release()
        raise

    def on_close() -> None:
        stream.close()
        if release:
            release()

    return _closing_streaming_response(iter(stream), on_close, status_code=status,
                                       media_type=video_mime(filename),
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
    params = {}
    if purpose:
        params["purpose"] = purpose
    if download:
        params["download"] = "1"
    return proxy_nas_stream(
        f"{nas_url}/api/photos/{photo_id}/full", params, range_header,
        f"Stream video {photo_id} from the NAS", session=session, timeout=timeout)


def proxy_nas_stream(url: str, params: Optional[dict], range_header: Optional[str],
                     purpose_text: str, session=None, timeout: float = 30.0):
    """Stream any Range-capable NAS media URL through the replica: forwards
    ``Range``, returns the NAS's status (200/206/404/416) and range headers,
    iterates the body (never ``.read()``). Shared by ``/full`` of a video and
    ``/api/videos/{id}/proxy``."""
    import requests
    from .request_intent import outbound_headers

    hdrs = outbound_headers(purpose_text
                            + (f" (range {range_header})" if range_header else ""))
    if range_header:
        hdrs["Range"] = range_header
    sess = session or requests
    upstream = sess.get(url, params=params or {}, headers=hdrs, stream=True,
                        timeout=timeout)
    out_headers = {k: v for k, v in upstream.headers.items()
                   if k.lower() in _PROXY_HEADERS}

    def body():
        try:
            for chunk in upstream.iter_content(chunk_size=CHUNK):
                if chunk:
                    yield chunk
        finally:
            upstream.close()

    # upstream.close() also runs from the response itself, so a response that
    # is never iterated, or whose client disconnects mid-stream, still
    # releases the NAS connection (Response.close is idempotent).
    return _closing_streaming_response(
        body(), upstream.close, status_code=upstream.status_code,
        headers=out_headers, media_type=upstream.headers.get("content-type"))


# ---------------------------------------------------------------------------
# Asset upload (POST /api/videos/{id}/assets)
# ---------------------------------------------------------------------------

ASSET_KINDS = ("poster", "keyframe", "crop")
POSTER_SIZES = ("thumb", "preview", "full")
MAX_ASSET_BYTES = 12 * 1024 * 1024
MAX_ASSETS = 128
MAX_TOTAL_BYTES = 96 * 1024 * 1024
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}\.jpe?g$")
# A video-scan staging generation (video_scan_store): uploads land in
# v{version}/g{generation}/ and posters are NOT installed until the submit
# that names this generation commits.
_GEN_RE = re.compile(r"^[a-z0-9]{8,32}$")


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


def asset_dir(db_path: str, photo_id: int, version: int,
              generation: Optional[str] = None) -> Path:
    """Where one upload's files live: ``video_frames/{id}/v{version}/`` or,
    for a staged video-scan upload, ``.../v{version}/g{generation}/``."""
    d = frames_root(db_path) / str(int(photo_id)) / f"v{int(version)}"
    return d / f"g{generation}" if generation else d


def install_posters(db_path: str, photo_id: int, version: int, generation: str, *,
                    thumb_dir: str, preview_dir: str) -> int:
    """Install a staged generation's posters where every grid reads them —
    called only after the submit naming that generation has committed."""
    src = asset_dir(db_path, photo_id, version, generation)
    n = 0
    for size in POSTER_SIZES:
        f = src / f"poster_{size}.jpg"
        if not f.exists():
            continue
        if size == "thumb":
            inst = Path(thumb_dir) / f"{int(photo_id)}_thumb.jpg"
        elif size == "preview":
            inst = Path(preview_dir) / f"{int(photo_id)}_preview.jpg"
        else:
            inst = frames_root(db_path) / str(int(photo_id)) / "poster.jpg"
        atomic_write(inst, f.read_bytes())
        n += 1
    return n


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
    generation = body.get("generation")
    if generation is not None and (not isinstance(generation, str)
                                   or not _GEN_RE.match(generation)):
        raise AssetError("generation must be 8-32 lowercase letters/digits")
    root = frames_root(db_path)
    vdir = asset_dir(db_path, photo_id, version, generation)
    written = []
    for a in assets:
        dest = vdir / a["name"]
        # Belt and braces: the name regex already forbids separators.
        if dest.resolve().parent != vdir.resolve():
            raise AssetError(f"refusing path outside {vdir}")
        atomic_write(dest, a["data"])
        entry = {"kind": a["kind"], "name": a["name"], "bytes": len(a["data"]),
                 "path": str(dest.relative_to(root.parent))}
        if a["kind"] == "poster" and not generation:
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
    return {"photo_id": int(photo_id), "version": version, "generation": generation,
            "written": written}

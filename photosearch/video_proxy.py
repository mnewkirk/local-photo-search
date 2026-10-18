"""The ``video-proxy`` worker pass: an H.264 720p-class playable copy of every
video the browser cannot play as-is (docs/plans/video-support.md D12 fallback,
owner-approved 2026-10-18).

**Why.** Measured on the owner's browser (Windows Chrome/Edge, 2026-10-18):
NO HEVC decode at all — ``MediaSource.isTypeSupported('video/mp4;
codecs="hvc1.1.6.L93.B0"')`` is false. What played: H.264 8-bit 4:2:0
(``yuv420p`` / ``yuvj420p``) in ``.mp4`` / ``.mov`` up to 1440x1080. What did
NOT: HEVC Main10 HLG HDR iPhone ``.mov``, HEVC Main10 8K Sony ``.mp4``,
MPEG-4 ASP ``.avi`` (Canon), H.264 in ``.mts``. HEVC is the dominant codec
among the library's readable videos, so without a proxy most clips are a
broken player.

**The rule** (:func:`needs_proxy`, pure): a proxy is made iff NOT (codec is
h264 AND 8-bit 4:2:0 AND the container is mp4/mov/m4v AND the long edge is
<= 3840). It reads the stored probe (``video_meta`` columns + the ffprobe
JSON in ``probe_json``) — no new probe field was needed, ``pix_fmt`` /
``color_transfer`` / ``field_order`` are already in ``probe_json``.

**Where it runs** (D19): on the DESKTOP. The worker downloads the original
once through the rate-capped ``/full?purpose=worker`` stream, transcodes in a
temp dir with ffmpeg, verifies the output with ffprobe, and streams it to
``PUT /api/videos/{id}/proxy?version=N``. The NAS only stores the file
(``<DB dir>/video_proxies/{id}.v{N}.mp4``, atomic) and serves it with HTTP
Range (``GET /api/videos/{id}/proxy``). No ffmpeg on the NAS.

**The output**: H.264 High, 8-bit ``yuv420p``, long edge <=
:data:`PROXY_LONG_EDGE` (1280, ``PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE``), CRF 23
``veryfast``, AAC stereo 128k (no audio track when the source has none),
``+faststart`` so the browser can start before the whole file arrives,
<= 60 fps. Rotation is baked into the pixels (ffmpeg's default autorotate
transposes the frames and drops the display matrix), so the proxy plays
upright with no metadata a player could misread. Interlaced sources (.mts)
are deinterlaced. HDR (HLG / PQ, 10-bit BT.2020) is tone-mapped to SDR
BT.709 with zscale + tonemap (mobius) — without it a 10-bit HLG clip squeezed
into 8-bit BT.709 looks grey and washed out.

Everything except running the tools is pure (:func:`needs_proxy`,
:func:`build_transcode_cmd`, :func:`check_output`), so CI tests it without
ffmpeg.

**ZIP-wrapped files.** The claim requires ``probe_error IS NULL``, so a file
the probe found to be a ZIP (``zip_wrapped``) is never claimed — those wait
for the owner's ``unwrap-zip-videos`` + re-probe. A ZIP that video-scan
already unwrapped has its probe_error CLEARED (scan applies the inner probe)
while the file on the NAS is still a ZIP; the scan store therefore marks it
``proxy_needed = 1`` (a ZIP plays nowhere), and the worker unwraps it before
transcoding.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

# Bump to re-make every proxy (claim: proxy_version IS NULL OR < PROXY_VERSION).
PROXY_VERSION = 1

DEFAULT_LONG_EDGE = 1280
PROXY_CRF = 23
PROXY_PRESET = "veryfast"
PROXY_AUDIO_BITRATE = "128k"
PROXY_MAX_FPS = 60

# What the owner's browser played (2026-10-18). Anything else gets a proxy.
PLAYABLE_CODECS = frozenset({"h264"})
PLAYABLE_PIX_FMTS = frozenset({"yuv420p", "yuvj420p"})
PLAYABLE_CONTAINERS = frozenset({"mp4", "mov", "m4v"})
MAX_PLAYABLE_LONG_EDGE = 3840

HDR_TRANSFERS = frozenset({"arib-std-b67", "smpte2084"})   # HLG, PQ
_INTERLACED = frozenset({"tt", "bb", "tb", "bt"})

# Transcode timeout: base + per second of footage. Software HEVC decode of an
# 8K Main10 clip on the desktop runs well under real time, a 4K one faster;
# the per-second term is generous so a slow box defers rather than kills.
TIMEOUT_BASE_S = 120
TIMEOUT_PER_SECOND = 10
DURATION_TOLERANCE_S = 1.0

# Server side: the upload cap. A 720p CRF 23 proxy is ~1-3 MB/s of footage;
# the library's longest clip (29 min) would be ~2-5 GB only at absurd
# bitrates, so 2 GB is a sanity bound, not a budget.
MAX_PROXY_BYTES = 2 * 1024 ** 3
_PROXY_NAME = re.compile(r"^(\d+)\.v(\d+)\.mp4$")
STALE_TMP_SECONDS = 3600


class ProxyToolError(Exception):
    """ffmpeg / ffprobe is missing, failed, or produced an output that does not
    verify. On a COMPLETE download this is the video's problem: the worker
    reports a failure row (one of MAX_PROCESS_ATTEMPTS spent)."""


class ProxyUploadError(ValueError):
    """The server refused an upload (size, magic, version)."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def proxy_long_edge() -> int:
    """PROXY_LONG_EDGE, overridable by PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE
    (clamped to 320..3840, rounded down to even)."""
    try:
        v = int(os.environ.get("PHOTOSEARCH_VIDEO_PROXY_LONG_EDGE") or DEFAULT_LONG_EDGE)
    except ValueError:
        v = DEFAULT_LONG_EDGE
    v = max(320, min(MAX_PLAYABLE_LONG_EDGE, v))
    return v - (v % 2)


PROXY_LONG_EDGE = proxy_long_edge()


# ---------------------------------------------------------------------------
# Pure: the decision
# ---------------------------------------------------------------------------

def _loads(v) -> dict:
    if isinstance(v, dict):
        return v
    if not v:
        return {}
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def video_stream(probe_json) -> dict:
    """The first real video stream of a stored probe (`probe_json` as stored
    in video_meta: dict or JSON text), {} when there is none."""
    for s in _loads(probe_json).get("streams") or []:
        if s.get("codec_type") == "video" and \
                (s.get("disposition") or {}).get("attached_pic") != 1:
            return s
    return {}


def _get(meta, key):
    try:
        return meta[key]
    except (KeyError, IndexError, TypeError):
        return None


def playback_info(meta) -> dict:
    """What decides browser playback, from a video_meta row (dict or
    sqlite3.Row): codec, container, display dims, pix_fmt, profile, level,
    bit depth, HDR transfer and field order. Missing values are None."""
    vs = video_stream(_get(meta, "probe_json"))
    pix = vs.get("pix_fmt")
    depth = None
    if pix:
        m = re.search(r"p(\d{2})(?:le|be)?$", pix)
        depth = int(m.group(1)) if m else 8
    transfer = vs.get("color_transfer")
    w, h = _get(meta, "width"), _get(meta, "height")
    return {
        "codec": (_get(meta, "codec") or vs.get("codec_name") or None),
        "container": _get(meta, "container"),
        "width": w, "height": h,
        "long_edge": max(w or 0, h or 0) or None,
        "pix_fmt": pix,
        "bit_depth": depth,
        "profile": vs.get("profile"),
        "level": vs.get("level"),
        "color_transfer": transfer,
        "hdr": transfer in HDR_TRANSFERS,
        "field_order": vs.get("field_order"),
    }


def needs_proxy(meta) -> bool:
    """True unless the original is h264, 8-bit 4:2:0, in mp4/mov/m4v, with a
    long edge <= 3840 — the shape the owner's browser was measured to play.

    Unknown is NOT playable: a missing codec, pix_fmt, container or dims
    means a proxy (a wasted transcode is cheaper than a broken player)."""
    info = playback_info(meta)
    codec = (info["codec"] or "").lower()
    if codec not in PLAYABLE_CODECS:
        return True
    if (info["pix_fmt"] or "").lower() not in PLAYABLE_PIX_FMTS:
        return True
    if (info["container"] or "").lower() not in PLAYABLE_CONTAINERS:
        return True
    if not info["long_edge"] or info["long_edge"] > MAX_PLAYABLE_LONG_EDGE:
        return True
    return False


# ---------------------------------------------------------------------------
# Pure: the command
# ---------------------------------------------------------------------------

def scale_filter(long_edge: int) -> str:
    """Fit the (already rotated) frame inside long_edge x long_edge, never
    upscale, even dims (H.264 4:2:0 needs them)."""
    le = int(long_edge)
    return (f"scale=w='if(gte(iw,ih),min({le},iw),-2)'"
            f":h='if(gte(iw,ih),-2,min({le},ih))'")


def tonemap_filter(transfer: str, primaries: Optional[str] = None,
                   matrix: Optional[str] = None) -> str:
    """HLG / PQ (BT.2020) -> SDR BT.709 via linear light + mobius.

    Measured on the two iPhone HLG fixtures (2026-10-18; mean luma /
    saturation of a mid-clip frame, 0-255): cv2's plain decode 205/32 and
    194/47 (bright but washed out — what an un-tone-mapped 8-bit squeeze
    looks like); scale+format with no tone-map 172/24, 170/38 (grey); hable
    171/48, 165/67 (colour back but a stop dark); **mobius 202/48, 190/67**
    — the brightness of the plain decode with the colour restored.
    reinhard sat between (195/48, 185/67).

    The input parameters are stated explicitly (zscale would otherwise trust
    frame side data that a mov demuxer does not always attach)."""
    tin = "arib-std-b67" if transfer == "arib-std-b67" else "smpte2084"
    pin = primaries if primaries in ("bt2020", "bt709") else "bt2020"
    min_ = matrix if matrix in ("bt2020nc", "bt2020c", "bt709") else "bt2020nc"
    return (f"zscale=tin={tin}:pin={pin}:min={min_}:t=linear:npl=100,"
            "format=gbrpf32le,zscale=p=bt709,"
            "tonemap=tonemap=mobius:desat=0,"
            "zscale=t=bt709:m=bt709:r=tv")


def video_filters(info: dict, long_edge: int) -> str:
    """The -vf chain: deinterlace (interlaced sources), scale, HDR tone-map,
    then 8-bit 4:2:0."""
    chain = []
    if (info.get("field_order") or "") in _INTERLACED:
        chain.append("bwdif=mode=send_frame")
    chain.append(scale_filter(long_edge))
    if info.get("hdr"):
        chain.append(tonemap_filter(info.get("color_transfer"),
                                    info.get("color_primaries"),
                                    info.get("color_space")))
    chain.append("format=yuv420p")
    return ",".join(chain)


def transcode_info(meta) -> dict:
    """playback_info plus the colour tags the tone-map needs and has_audio."""
    info = playback_info(meta)
    vs = video_stream(_get(meta, "probe_json"))
    info["color_primaries"] = vs.get("color_primaries")
    info["color_space"] = vs.get("color_space")
    ha = _get(meta, "has_audio")
    if ha is None:
        streams = _loads(_get(meta, "probe_json")).get("streams") or []
        ha = any(s.get("codec_type") == "audio" for s in streams)
    info["has_audio"] = bool(ha)
    info["duration_s"] = _get(meta, "duration_s")
    return info


def build_transcode_cmd(ffmpeg: str, src: str, dst: str, info: dict,
                        long_edge: Optional[int] = None,
                        crf: int = PROXY_CRF, preset: str = PROXY_PRESET) -> list[str]:
    """The ffmpeg argv for one proxy. Pure — the tests pin it.

    Rotation: no ``-noautorotate`` on purpose. ffmpeg's default autorotate
    applies the display matrix to the decoded frames (transpose) and the
    output carries none, so the proxy is upright in every player and the
    scale filter below sees display orientation."""
    le = int(long_edge or proxy_long_edge())
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-v", "error",
           "-i", src,
           "-map", "0:v:0"]
    if info.get("has_audio"):
        cmd += ["-map", "0:a:0?"]
    cmd += ["-vf", video_filters(info, le),
            "-fpsmax", str(PROXY_MAX_FPS),
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709"]
    if info.get("has_audio"):
        cmd += ["-c:a", "aac", "-b:a", PROXY_AUDIO_BITRATE, "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
            "-movflags", "+faststart", "-f", "mp4", dst]
    return cmd


def transcode_timeout(duration_s) -> float:
    try:
        d = max(0.0, float(duration_s or 0))
    except (TypeError, ValueError):
        d = 0.0
    return TIMEOUT_BASE_S + TIMEOUT_PER_SECOND * d


def check_output(probe: dict, source_duration) -> dict:
    """Verify an ffprobe of the proxy: one h264 yuv420p video stream, and a
    duration within DURATION_TOLERANCE_S of the source's. Returns {width,
    height, duration_s}; raises ProxyToolError."""
    fmt = probe.get("format") or {}
    streams = probe.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise ProxyToolError("proxy has no video stream")
    if v.get("codec_name") != "h264":
        raise ProxyToolError(f"proxy codec {v.get('codec_name')!r}, expected h264")
    if v.get("pix_fmt") != "yuv420p":
        raise ProxyToolError(f"proxy pix_fmt {v.get('pix_fmt')!r}, expected yuv420p")
    try:
        dur = float(fmt.get("duration") or v.get("duration"))
    except (TypeError, ValueError):
        raise ProxyToolError("proxy has no duration")
    if source_duration:
        try:
            sd = float(source_duration)
        except (TypeError, ValueError):
            sd = None
        if sd and abs(dur - sd) > DURATION_TOLERANCE_S:
            raise ProxyToolError(f"proxy is {dur:.2f}s, source {sd:.2f}s")
    return {"width": int(v.get("width") or 0), "height": int(v.get("height") or 0),
            "duration_s": round(dur, 3)}


# ---------------------------------------------------------------------------
# Tools (desktop only — never on CI, never on the NAS)
# ---------------------------------------------------------------------------

def find_ffmpeg() -> Optional[str]:
    """PHOTOSEARCH_FFMPEG, else PATH, else ~/.local/bin/ffmpeg (the static
    build the video spike installed without sudo)."""
    env = os.environ.get("PHOTOSEARCH_FFMPEG")
    if env:
        return env if os.path.exists(env) or shutil.which(env) else None
    found = shutil.which("ffmpeg")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "ffmpeg"
    return str(local) if local.exists() else None


def probe_local(path: str, ffprobe: Optional[str] = None, run=subprocess.run) -> dict:
    """ffprobe JSON of a local file; raises ProxyToolError."""
    from .video_probe import find_ffprobe
    exe = ffprobe or find_ffprobe()
    if not exe:
        raise ProxyToolError("ffprobe not found (set PHOTOSEARCH_FFPROBE)")
    proc = run([exe, "-v", "error", "-print_format", "json", "-show_format",
                "-show_streams", path], capture_output=True, text=True, timeout=120)
    try:
        data = json.loads(proc.stdout or "{}")
    except ValueError:
        data = {}
    if proc.returncode != 0 or not data.get("streams"):
        raise ProxyToolError(f"ffprobe of {os.path.basename(path)} failed: "
                             f"{(proc.stderr or '').strip()[-300:]}")
    return data


def local_meta(path: str, ffprobe: Optional[str] = None) -> dict:
    """A video_meta-shaped dict for a local file (used for a ZIP's inner
    video, whose stored probe describes the outer ZIP's inner file only
    loosely)."""
    from .video_probe import parse_probe
    data = probe_local(path, ffprobe)
    res = parse_probe(data)
    return {"codec": res.get("codec"), "container": res.get("container"),
            "width": res.get("width"), "height": res.get("height"),
            "has_audio": res.get("has_audio"), "duration_s": res.get("duration_s"),
            "probe_json": data}


def transcode(src: str, dst: str, meta, ffmpeg: Optional[str] = None,
              long_edge: Optional[int] = None, run=subprocess.run) -> dict:
    """Make and verify one proxy. Returns {width, height, duration_s, bytes,
    seconds}; raises ProxyToolError (missing tool, ffmpeg failed, timed out,
    output did not verify)."""
    exe = ffmpeg or find_ffmpeg()
    if not exe:
        raise ProxyToolError("ffmpeg not found (set PHOTOSEARCH_FFMPEG)")
    info = transcode_info(meta)
    cmd = build_transcode_cmd(exe, src, dst, info, long_edge=long_edge)
    timeout = transcode_timeout(info.get("duration_s"))
    t0 = time.time()
    try:
        proc = run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ProxyToolError(f"ffmpeg timed out after {timeout:.0f}s")
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise ProxyToolError(f"ffmpeg exit {proc.returncode}: {' | '.join(tail)[:400]}")
    out = check_output(probe_local(dst), info.get("duration_s"))
    out["bytes"] = os.path.getsize(dst)
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# Server side: storage (the NAS — no decoding)
# ---------------------------------------------------------------------------

def proxies_root(db_path: str) -> Path:
    """`<dir of the DB>/video_proxies` — /data/video_proxies on the NAS."""
    return Path(db_path).resolve().parent / "video_proxies"


def proxy_path(db_path: str, photo_id: int, version: int) -> Path:
    return proxies_root(db_path) / f"{int(photo_id)}.v{int(version)}.mp4"


def is_mp4_head(head: bytes) -> bool:
    """An ISO-BMFF file starts with a box whose type is `ftyp`."""
    return len(head) >= 8 and bytes(head[4:8]) == b"ftyp"


class ProxyWriter:
    """Streams one upload to a temp file beside its final name, enforcing the
    size cap and the mp4 magic as bytes arrive; `commit()` fsyncs and
    os.replace()s it into place, `abort()` removes it. A reader never sees a
    half-written proxy and a failed upload leaves nothing behind."""

    def __init__(self, db_path: str, photo_id: int, version: int,
                 max_bytes: Optional[int] = None):
        if not isinstance(version, int) or isinstance(version, bool) or \
                not 1 <= version <= 10_000:
            raise ProxyUploadError("version must be an integer 1..10000")
        self.dest = proxy_path(db_path, photo_id, version)
        self.dest.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.dest.with_name(f".{self.dest.name}.{os.getpid()}."
                                       f"{int(time.time() * 1000)}.tmp")
        self.fh = open(self.tmp, "wb")
        self.max_bytes = int(max_bytes if max_bytes is not None else MAX_PROXY_BYTES)
        self.n = 0
        self.head = b""

    def write(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.n += len(chunk)
        if self.n > self.max_bytes:
            raise ProxyUploadError(f"proxy exceeds {self.max_bytes} bytes")
        if len(self.head) < 12:
            self.head += chunk[:12 - len(self.head)]
            if len(self.head) >= 8 and not is_mp4_head(self.head):
                raise ProxyUploadError("not an mp4 (no ftyp box)")
        self.fh.write(chunk)

    def commit(self) -> int:
        if self.n == 0 or not is_mp4_head(self.head):
            raise ProxyUploadError("empty or not an mp4")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.fh.close()
        os.replace(self.tmp, self.dest)
        try:
            dfd = os.open(str(self.dest.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
        return self.n

    def abort(self) -> None:
        try:
            self.fh.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            os.unlink(self.tmp)
        except OSError:
            pass


def record_proxy(conn, photo_id: int, version: int, nbytes: int,
                 width: Optional[int], height: Optional[int]) -> None:
    """Point video_meta at the proxy just installed. Does NOT commit."""
    conn.execute(
        "UPDATE video_meta SET proxy_version = ?, proxy_bytes = ?, proxy_w = ?, "
        "proxy_h = ?, proxy_at = datetime('now') WHERE photo_id = ?",
        (int(version), int(nbytes), width, height, int(photo_id)))


def stale_proxy_files(conn, root) -> list[Path]:
    """Files under video_proxies/ nothing points at: a deleted video's, a
    version other than the row's current proxy_version, or an upload temp
    file older than an hour (a crashed PUT)."""
    root = Path(root)
    if not root.is_dir():
        return []
    current = {r[0]: r[1] for r in conn.execute(
        "SELECT photo_id, proxy_version FROM video_meta WHERE proxy_version IS NOT NULL")}
    now = time.time()
    out = []
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        if f.name.startswith(".") and f.name.endswith(".tmp"):
            try:
                if now - f.stat().st_mtime > STALE_TMP_SECONDS:
                    out.append(f)
            except OSError:
                pass
            continue
        m = _PROXY_NAME.match(f.name)
        if not m:
            continue
        pid, ver = int(m.group(1)), int(m.group(2))
        if current.get(pid) != ver:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Claim predicate (shared by db._VIDEO_CLAIM_SQL and batch_state)
# ---------------------------------------------------------------------------

def claimable_sql(alias: str = "p") -> str:
    """The current probe ran, found a playable file (probe_error NULL — a
    ZIP still waiting for unwrap is skipped), it is not a paired Live clip,
    it needs a proxy, and the proxy is missing or stale."""
    from .video_probe import PROBE_VERSION
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.probe_version >= {int(PROBE_VERSION)} "
            f"AND vm.probe_error IS NULL "
            f"AND COALESCE(vm.kind, '') != 'live' "
            f"AND vm.proxy_needed = 1 "
            f"AND (vm.proxy_version IS NULL OR vm.proxy_version < {int(PROXY_VERSION)}))")


def proxied_sql(alias: str = "p") -> str:
    """A current proxy exists."""
    return (f"EXISTS (SELECT 1 FROM video_meta vm WHERE vm.photo_id = {alias}.id "
            f"AND vm.proxy_version >= {int(PROXY_VERSION)})")


def chunks(fh, size: int = 1 << 20) -> Iterable[bytes]:
    """Read a file object in chunks (the worker's streaming upload body)."""
    while True:
        b = fh.read(size)
        if not b:
            return
        yield b

"""Shared video frame decoding for the desktop video passes and the eval —
OpenCV first, an ffmpeg rawvideo pipe when OpenCV silently returns BLACK.

**The bug this exists for (2026-10-25).** OpenCV 4.13 on interlaced H.264 in
MPEG-TS (AVCHD ``.m2ts`` / ``.mts``) logs ``swscaler: Cannot convert
interlaced to progressive frames`` and then hands back all-zero BGR frames.
It does not fail: ``read()`` returns ``True`` and a frame. So video-scan
(faces, CLIP, keyframes, quality), the probe poster, and everything downstream
(video-subjects, video-describe) ran on black — 310 of 359 non-Live ``.m2ts``
videos stored ``"subjects": []``. The ffmpeg CLI decodes the same files fine.

Decision (pure, :func:`choose_backend`, tested in CI with cv2/ffmpeg mocked):

* cv2 cannot open the file -> ffmpeg, if ffmpeg yields a frame; else the file
  is unreadable (:class:`DecodeOpenError`).
* any of the first :data:`CHECK_FRAMES` frames cv2 decodes is not black -> cv2
  (the normal case — no ffmpeg call at all).
* all of them black -> compare against ffmpeg at a few times across the clip.
  ffmpeg non-black anywhere -> ffmpeg (cv2 is broken on this file). ffmpeg
  black too -> keep cv2: the clip really is black, never call that a decode
  failure. ffmpeg unavailable -> ``"unconfirmed"``; the caller decides
  (scan raises :class:`DecodeToolError`; the probe poster keeps cv2's frame).

"Black" is the max luma of the frame below :data:`BLACK_MAX` — a decode
failure is exactly 0; a real night clip has noise / lights well above it.

cv2 is imported lazily (CI mocks it; the NAS never decodes — D19). The ffmpeg
path deinterlaces (``yadif=deint=interlaced``, a no-op on progressive frames),
keeps ffmpeg's autorotation (cv2 also auto-orients, so both give DISPLAY
orientation), and streams one sampled frame at a time — the video is never
held in memory.
"""

from __future__ import annotations

import io
import json
import math
import socket
import subprocess
from typing import Callable, Iterator, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np

# A frame whose max (over all channels) is below this is black.
BLACK_MAX = 8.0
# Sequential cv2 frames read up front; any non-black one proves cv2 works.
CHECK_FRAMES = 12
# Clip fractions where ffmpeg is asked for a frame when cv2's were all black.
CONFIRM_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
FFMPEG_TIMEOUT_S = 180

# Per-process cache of hostname -> resolved IP, so a probe pass that hits
# the same NAS host for dozens of videos pays for DNS once.
_HOST_IP_CACHE: dict[str, str] = {}


class HostResolutionError(RuntimeError):
    """A hostname in an ffmpeg/ffprobe URL could not be resolved via DNS --
    transient (the caller should defer, never spend an attempt or treat it
    as a verdict about the video)."""


def _is_ip_literal(host: str) -> bool:
    try:
        socket.inet_pton(socket.AF_INET, host)
        return True
    except OSError:
        pass
    try:
        socket.inet_pton(socket.AF_INET6, host)
        return True
    except OSError:
        return False


def ffmpeg_url(url: str) -> str:
    """Rewrite ``url``'s hostname to an IP literal (IPv4 preferred),
    keeping scheme/port/path/query/credentials untouched.

    The desktop's ffmpeg/ffprobe are a static build in ``~/.local/bin`` (no
    apt/sudo) that SEGFAULTS (exit -11) resolving ANY hostname through
    glibc's static NSS -- measured 2026-10-25: both ``localhost`` and a real
    DNS name (``dxp4800-f976``) crash; an IP literal works. So every URL
    handed to the ffmpeg/ffprobe CLI must have its host pre-resolved in
    Python first.

    A bare local file path (no ``scheme://``) or a URL whose host is already
    an IP literal passes through unchanged. Resolutions are cached per
    process (see :data:`_HOST_IP_CACHE`).

    Raises :class:`HostResolutionError` on a DNS failure -- the caller must
    treat that as transient, not a verdict."""
    s = str(url)
    if "://" not in s:
        return s
    parts = urlsplit(s)
    host = parts.hostname
    if not host or _is_ip_literal(host):
        return s
    ip = _HOST_IP_CACHE.get(host)
    if ip is None:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError as e:
            raise HostResolutionError(f"could not resolve {host!r}: {e}") from e
        ip = next((i[4][0] for i in infos if i[0] == socket.AF_INET), None) \
            or (infos[0][4][0] if infos else None)
        if not ip:
            raise HostResolutionError(f"could not resolve {host!r}: no addresses")
        _HOST_IP_CACHE[host] = ip
    netloc = f"[{ip}]" if ":" in ip else ip
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    if parts.username:
        creds = parts.username + (f":{parts.password}" if parts.password else "")
        netloc = f"{creds}@{netloc}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class DecodeOpenError(Exception):
    """Neither cv2 nor ffmpeg can decode a frame of the file — a terminal
    verdict for a COMPLETE file."""


class DecodeToolError(RuntimeError):
    """cv2 decoded only black frames and ffmpeg (the tool that would confirm
    or fix it) is missing — not a verdict about the file."""


def frame_is_black(frame) -> bool:
    return frame is None or float(np.asarray(frame).max()) < BLACK_MAX


INTERLACED_FIELD_ORDERS = ("tt", "bb", "tb", "bt")


def is_interlaced(field_order: Optional[str]) -> bool:
    """ffprobe ``field_order`` -> interlaced? (``progressive``/unknown -> no)."""
    return str(field_order or "").lower() in INTERLACED_FIELD_ORDERS


def choose_backend(cv2_opened: bool, cv2_maxes: Sequence[Optional[float]],
                   ffmpeg_maxes: Optional[Sequence[Optional[float]]],
                   interlaced: bool = False) -> str:
    """``"cv2" | "ffmpeg" | "unreadable" | "unconfirmed"``.

    ``cv2_maxes``: per-frame max of the first frames cv2 decoded (None = no
    frame). ``ffmpeg_maxes``: the same for ffmpeg at the confirm times, or
    None when ffmpeg is unavailable (or was not asked). ``interlaced``: the
    stream is interlaced per ffprobe — cv2 is never trusted on it (it hands
    back black OR a stale buffer from the previous video), so it is ffmpeg or
    nothing."""
    ff_any = ffmpeg_maxes is not None and any(m is not None for m in ffmpeg_maxes)
    if interlaced:
        if ffmpeg_maxes is None:
            return "unconfirmed"
        return "ffmpeg" if ff_any else "unreadable"
    if cv2_opened and any(m is not None and m >= BLACK_MAX for m in cv2_maxes):
        return "cv2"
    cv2_any = cv2_opened and any(m is not None for m in cv2_maxes)
    if ffmpeg_maxes is None:
        return "unconfirmed" if cv2_any else "unreadable"
    if any(m is not None and m >= BLACK_MAX for m in ffmpeg_maxes):
        return "ffmpeg"
    if cv2_any:
        return "cv2"            # both black: a genuinely black clip
    return "ffmpeg" if ff_any else "unreadable"


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------

def find_ffmpeg() -> Optional[str]:
    from .video_proxy import find_ffmpeg as _f
    return _f()


def _vf(extra: Sequence[str] = ()) -> str:
    return ",".join(["setpts=PTS-STARTPTS", "yadif=deint=interlaced", *extra])


def ffmpeg_frame(path: str, t: float, ffmpeg: Optional[str] = None,
                 run: Callable = subprocess.run):
    """One RGB uint8 frame at ``t`` seconds (deinterlaced, autorotated), or
    None. ``path`` may be an http URL -- its hostname is resolved to an IP
    via :func:`ffmpeg_url` first (the static ffmpeg build segfaults
    resolving any hostname, including ``localhost``)."""
    exe = ffmpeg or find_ffmpeg()
    if not exe:
        return None
    try:
        url = ffmpeg_url(str(path))
        proc = run([exe, "-v", "error", "-nostdin", "-ss", f"{max(0.0, float(t)):.3f}",
                    "-i", url, "-an", "-sn", "-dn", "-vf", "yadif=deint=interlaced",
                    "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                   capture_output=True, timeout=FFMPEG_TIMEOUT_S)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    try:
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(proc.stdout)).convert("RGB"))
    except Exception:
        return None


def ffprobe_basics(path: str, run: Callable = subprocess.run) -> dict:
    """``{"fps", "duration", "field_order"}`` of the first video stream
    (0 / None when unknown or ffprobe is missing). Works on http URLs."""
    out = {"fps": 0.0, "duration": 0.0, "field_order": None}
    try:
        from .video_probe import find_ffprobe, parse_rate
        exe = find_ffprobe()
        if not exe:
            return out
        url = ffmpeg_url(str(path))
        proc = run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=avg_frame_rate,r_frame_rate,field_order:format=duration",
                    "-of", "json", url], capture_output=True, timeout=60)
        data = json.loads(proc.stdout or b"{}")
        st = (data.get("streams") or [{}])[0]
        fps = parse_rate(st.get("avg_frame_rate")) or parse_rate(st.get("r_frame_rate")) or 0.0
        dur = float((data.get("format") or {}).get("duration") or 0.0)
        out["fps"] = fps if math.isfinite(fps) and 0 < fps <= 1000 else 0.0
        out["duration"] = dur if math.isfinite(dur) and dur > 0 else 0.0
        out["field_order"] = st.get("field_order")
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Sources: iterate sampled frames
# ---------------------------------------------------------------------------

class _Source:
    backend = ""
    fps = 0.0
    nframes = 0.0
    n_decoded = 0
    last_t = 0.0

    def samples(self, sample_fps: float) -> Iterator[tuple[float, int, np.ndarray]]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Cv2Source(_Source):
    """Sequential grab/retrieve; only frames due at ``sample_fps`` are
    retrieved (decoded to BGR)."""
    backend = "cv2"

    def __init__(self, cap, fps: float, nframes: float):
        self.cap, self.fps, self.nframes = cap, fps, nframes

    def samples(self, sample_fps: float):
        import cv2
        cap, fps = self.cap, self.fps
        half_frame = 0.5 / fps if fps > 0 else 0.0
        next_t = 0.5 / sample_fps
        idx = -1
        while True:
            if not cap.grab():
                break
            idx += 1
            self.n_decoded = idx + 1
            pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = (pos / 1000.0) if (pos and pos > 0) else (idx / fps if fps > 0 else 0.0)
            if idx > 0 and fps > 0 and t <= 0:
                t = idx / fps
            self.last_t = max(self.last_t, t)
            if t + half_frame < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            while next_t <= t + half_frame:
                next_t += 1.0 / sample_fps
            yield t, idx, frame

    def close(self):
        self.cap.release()


class FfmpegSource(_Source):
    """``ffmpeg -vf setpts,yadif,fps=S,scale=WxH -f rawvideo -pix_fmt bgr24 -``
    read one frame at a time. Sample k sits at ``k / S`` seconds."""
    backend = "ffmpeg"

    def __init__(self, path: str, exe: str, width: int, height: int, fps: float,
                 nframes: float, popen: Callable = subprocess.Popen):
        self.path, self.exe = path, exe
        self.width, self.height = int(width), int(height)
        self.fps, self.nframes = fps, nframes
        self._popen = popen
        self.proc = None

    def command(self, sample_fps: float) -> list[str]:
        vf = _vf([f"fps=fps={sample_fps:g}:round=near",
                  f"scale={self.width}:{self.height}", "format=bgr24"])
        return [self.exe, "-v", "error", "-nostdin", "-i", str(self.path),
                "-an", "-sn", "-dn", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    def samples(self, sample_fps: float):
        n = self.width * self.height * 3
        self._closed = False
        self.proc = self._popen(self.command(sample_fps), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        k = 0
        try:
            while True:
                buf = _read_exact(self.proc.stdout, n)
                if buf is None:
                    break
                t = k / sample_fps
                frame_no = int(round(t * self.fps)) if self.fps > 0 else k
                self.last_t = t
                self.n_decoded = frame_no + 1
                yield t, frame_no, np.frombuffer(buf, np.uint8).reshape(
                    self.height, self.width, 3).copy()
                k += 1
        finally:
            self.close()
        if k == 0:
            raise DecodeOpenError(f"ffmpeg exited {self.proc.returncode} with no frames")

    def close(self):
        p = self.proc
        if p is None or getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if p.poll() is None:
                p.kill()
            p.wait(timeout=10)
        except Exception:
            pass
        try:
            if p.stdout:
                p.stdout.close()
        except Exception:
            pass


def _read_exact(stream, n: int) -> Optional[bytes]:
    chunks, got = [], 0
    while got < n:
        b = stream.read(n - got)
        if not b:
            return None
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# open_video / grab_frame
# ---------------------------------------------------------------------------

def _cv2_open(path: str):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def _cv2_props(cap) -> tuple[float, float]:
    import cv2
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0 or fps > 1000:
        fps = 0.0
    return fps, float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)


def _cv2_head_maxes(path: str, n: int = CHECK_FRAMES) -> list[Optional[float]]:
    cap = _cv2_open(path)
    if cap is None:
        return []
    out: list[Optional[float]] = []
    try:
        for _ in range(n):
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            m = float(np.asarray(fr).max())
            out.append(m)
            if m >= BLACK_MAX:
                break
    finally:
        cap.release()
    return out


def _confirm_times(duration: float) -> list[float]:
    if not duration or duration <= 0:
        return [0.0]
    return sorted({round(f * duration, 3) for f in CONFIRM_FRACTIONS})


def open_video(path: str, duration: Optional[float] = None,
               log: Optional[Callable[[str], None]] = None) -> _Source:
    """A frame source for a LOCAL ``path`` — cv2, or ffmpeg when the stream is
    interlaced, cv2 cannot open it, or cv2 decodes it to black (see
    :func:`choose_backend`). Raises :class:`DecodeOpenError` /
    :class:`DecodeToolError`."""
    info = ffprobe_basics(path)
    interlaced = is_interlaced(info["field_order"])
    head = [] if interlaced else _cv2_head_maxes(path)
    cap = _cv2_open(path)
    opened = cap is not None
    fps, nframes = _cv2_props(cap) if opened else (0.0, 0.0)
    fps = fps or info["fps"]
    dur = float(duration or 0.0) or info["duration"] or (nframes / fps if fps and nframes else 0.0)
    if not nframes and fps and dur:
        nframes = float(round(dur * fps))
    ff_maxes = None
    first = None
    if choose_backend(opened, head, None, interlaced) != "cv2":
        exe = find_ffmpeg()
        if exe:
            ff_maxes = []
            for t in _confirm_times(dur):
                rgb = ffmpeg_frame(path, t, exe)
                ff_maxes.append(None if rgb is None else float(rgb.max()))
                if rgb is not None:
                    first = rgb if first is None else first
                    if interlaced or float(rgb.max()) >= BLACK_MAX:
                        break
    verdict = choose_backend(opened, head, ff_maxes, interlaced)
    if verdict == "cv2":
        return Cv2Source(cap, fps, nframes)
    if cap is not None:
        cap.release()
    if verdict == "unreadable":
        raise DecodeOpenError("neither cv2 nor ffmpeg could decode a frame"
                              if ff_maxes is not None else "cv2 could not open the file")
    if verdict == "unconfirmed":
        raise DecodeToolError("cv2 cannot be trusted on this video (interlaced or black "
                              "frames) and ffmpeg is not available (PATH, ~/.local/bin, "
                              "PHOTOSEARCH_FFMPEG)")
    h, w = first.shape[:2]
    if log:
        why = ("interlaced" if interlaced else
               "cv2 decoded black frames" if opened else "cv2 could not open the file")
        log(f"{why} — decoding with ffmpeg ({w}x{h})")
    return FfmpegSource(path, find_ffmpeg(), w, h, fps, nframes)


def grab_frame(path: str, t: float = 0.0, *, seek: bool = True,
               interlaced: Optional[bool] = None):
    """(RGB uint8 frame, t_seconds) at ``t`` (or the first decodable frame
    with ``seek=False``), or (None, None). Works on http URLs.

    ``interlaced`` (ffprobe's verdict; None = ask ffprobe): an interlaced
    stream is read with ffmpeg only. Otherwise cv2 first; a missing or black
    cv2 frame is re-read with ffmpeg at the same time, and a black ffmpeg
    frame keeps cv2's (a genuinely black frame is not a decode failure)."""
    if interlaced is None:
        interlaced = is_interlaced(ffprobe_basics(path)["field_order"])
    ft = max(0.0, float(t)) if seek else 0.0
    if interlaced:
        alt = ffmpeg_frame(path, ft)
        return (alt, round(ft, 3)) if alt is not None else (None, None)
    import cv2
    frame, tt = None, None
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if cap.isOpened():
            if ft > 0:
                cap.set(cv2.CAP_PROP_POS_MSEC, ft * 1000.0)
            ok, fr = cap.read()
            if ok and fr is not None:
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                tt = round(pos / 1000.0, 3) if pos and pos > 0 else round(ft, 3)
                frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()
    if frame_is_black(frame):
        alt = ffmpeg_frame(path, ft)
        if alt is not None and (frame is None or not frame_is_black(alt)):
            return alt, round(ft, 3)
    return (frame, tt) if frame is not None else (None, None)

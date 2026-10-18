"""Select and extract the ONE video inside a ZIP-wrapped "video" file.

iOS Live-bundle exports from 2016-17 saved some clips as ZIP archives under a
video extension (``IMG_1234.MOV`` that is really ``PK\\x03\\x04``), each wrapping
one playable video, sometimes beside the still JPG and macOS litter
(``__MACOSX/``, AppleDouble ``._*``). Two consumers share these rules so they
cannot drift:

- ``video_scan.unwrap_zip`` — the desktop scan unwraps into its temp dir;
- ``unwrap_zip_videos`` — the NAS CLI that unwraps the file in place.

Stdlib only (no numpy/cv2), so the NAS image can import it cheaply.
"""
from __future__ import annotations

import os
import zipfile
import zlib
from typing import BinaryIO, Callable, Iterable, Optional

ZIP_MAGIC = b"PK\x03\x04"
ZIP_MAX_BYTES = 16 * 1024 ** 3
ZIP_MAX_RATIO = 50.0       # inflated / compressed — anything above is a bomb
ZIP_SIZE_SLACK = 1.2       # inner <= 1.2 x the ZIP itself: video is incompressible

DEFAULT_VIDEO_EXTENSIONS = frozenset(
    {".mp4", ".m4v", ".mov", ".avi", ".mts", ".m2ts", ".3gp"})


class ZipRefused(Exception):
    """A ZIP-wrapped file that is not exactly one safe video member — a
    terminal verdict."""


def has_zip_magic(path: str) -> bool:
    """First 4 bytes are a ZIP local-file header. Reads exactly 4 bytes."""
    with open(path, "rb") as fh:
        return fh.read(4) == ZIP_MAGIC


def is_junk_member(name: str) -> bool:
    """macOS resource-fork litter that rides along in a ZIP: never the video."""
    parts = [p for p in name.split("/") if p]
    return (not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._")
            or parts[-1] in (".DS_Store", "Thumbs.db"))


def safe_member_name(name: str) -> bool:
    parts = [p for p in name.split("/") if p]
    return bool(parts) and not name.startswith("/") and ".." not in parts \
        and ":" not in parts[0] and "." not in parts


def size_cap(zip_size: int, max_bytes: int = ZIP_MAX_BYTES) -> int:
    return min(int(max_bytes), int(ZIP_SIZE_SLACK * int(zip_size)) + 65536)


def select_video_member(zf: zipfile.ZipFile, zip_size: int, *,
                        max_bytes: int = ZIP_MAX_BYTES,
                        max_ratio: float = ZIP_MAX_RATIO,
                        video_exts: Optional[Iterable[str]] = None
                        ) -> tuple[zipfile.ZipInfo, int]:
    """The single safe video member and the byte cap to enforce on it.

    ``__MACOSX/`` and AppleDouble ``._*`` members are ignored; among the rest
    exactly ONE member must carry a video extension (a JPG + MOV Live bundle
    gives the MOV). Refuses (ZipRefused) none or several, an unsafe name
    (absolute, ``..``, drive letter), an encrypted member, a header size over
    ``max_bytes`` OR 1.2 x the ZIP's own size (video does not compress — this
    holds even when the header's compressed size is 0 and the ratio check
    cannot run), and a ratio above ``max_ratio``. Reads only the central
    directory. The cap must ALSO be enforced on the bytes actually inflated
    (:func:`copy_member` does) — the header can lie.
    """
    exts = {e.lower() for e in (video_exts or DEFAULT_VIDEO_EXTENSIONS)}
    cap = size_cap(zip_size, max_bytes)
    members = [m for m in zf.infolist()
               if not m.is_dir() and not is_junk_member(m.filename.replace("\\", "/"))]
    videos = [m for m in members if os.path.splitext(m.filename)[1].lower() in exts]
    if len(videos) != 1:
        raise ZipRefused(f"ZIP holds {len(videos)} video file(s) among "
                         f"{len(members)} member(s), expected exactly 1")
    m = videos[0]
    if not safe_member_name(m.filename.replace("\\", "/")):
        raise ZipRefused(f"unsafe member name {m.filename!r}")
    if m.flag_bits & 0x1:
        raise ZipRefused("encrypted ZIP member")
    if m.file_size > cap:
        raise ZipRefused(f"member is {m.file_size} bytes > {cap}")
    if m.compress_size > 0 and m.file_size / m.compress_size > max_ratio:
        raise ZipRefused("member inflates suspiciously (zip bomb?)")
    return m, cap


def member_basename(m: zipfile.ZipInfo) -> str:
    parts = [p for p in m.filename.replace("\\", "/").split("/") if p]
    return os.path.basename(parts[-1]) if parts else ""


def copy_member(zf: zipfile.ZipFile, m: zipfile.ZipInfo, dst: BinaryIO, cap: int, *,
                chunk_size: int = 1 << 20,
                on_chunk: Optional[Callable[[bytes], None]] = None) -> int:
    """Stream ``m`` into ``dst`` enforcing ``cap`` on the bytes actually
    inflated. ``on_chunk`` sees every chunk (hashing, rate limiting). A member
    that does not inflate (CRC/zlib/EOF) is a ZipRefused. Returns the size."""
    n = 0
    try:
        with zf.open(m) as src:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                n += len(chunk)
                if n > cap:
                    raise ZipRefused(f"member exceeded {cap} bytes while extracting")
                dst.write(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)
    except (zipfile.BadZipFile, zlib.error, EOFError) as e:
        raise ZipRefused(f"member does not inflate: {e}")
    return n


def open_zip(path: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        raise ZipRefused(f"not a readable ZIP: {e}")

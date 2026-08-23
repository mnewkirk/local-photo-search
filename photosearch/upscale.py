"""Local Topaz upscaling (M30).

Runs **Topaz Photo AI's** headless CLI (``tpai.exe``) on this machine to upscale
a photo, and drops the result in a standalone export tree. Nothing leaves the
box: the Topaz cloud API (and the official ``topaz-mcp`` server, which wraps it)
is deliberately NOT used -- the desktop app does all the model work locally.

Shape mirrors M28's ``rerun.py``: the NAS holds the originals, the desktop does
the compute. Here the compute is a Windows ``.exe`` rather than a torch model,
and the output is a new *file* rather than DB columns -- so unlike every other
pass this writes nothing back to the NAS or the DB. The export tree is the
deliverable; the library is left untouched.

Why Photo AI and not Gigapixel
------------------------------
``gigapixel.exe`` has the nicer CLI (explicit ``--model`` / ``--scale`` /
``--face-recovery``) but refuses to run below a Pro license::

    Auth state: TAuth::Owned TAuth::IndividualTier
    Gigapixel CLI requires a Pro license.

``tpai.exe`` from **Topaz Photo AI** has no such gate. Note the newer "Topaz
Photo" app also ships a ``tpai.exe`` -- it is a broken stub that fails to load
its QML entry point and silently produces no output, so ``_CLI_CANDIDATES``
prefers the "Topaz Photo AI" install and treats "Topaz Photo" as a last resort.

Two CLI quirks this module works around
---------------------------------------
1. **The success exit code is wrong.** A successful run exits **9** under WSL
   interop (and 127 under Git Bash), never 0. The *failure* codes are correct
   and documented (255 no valid files, 254 stale login, 253 bad argument, 1
   partial). So success is decided by "an output file appeared", with the
   documented failure codes reported verbatim -- see ``_SUCCESS_CODES``.

2. **``--upscale scale=N`` is broken.** The CLI serializes the scale as a JSON
   string and the engine rejects it with
   ``type_error.302: type must be number, but is string``. The working control
   is the Autopilot *preference*, which lives in the Windows registry under
   ``HKCU\\Software\\Topaz Labs LLC\\Topaz Photo AI``. ``_scale_override``
   sets it, runs, and restores it -- serialized by ``_REG_LOCK`` because the
   preference is global state shared with the GUI app.

Configuration
-------------
``PHOTOSEARCH_TOPAZ_CLI``    explicit path to ``tpai.exe`` (skips discovery)
``PHOTOSEARCH_UPSCALE_DIR``  export tree root (default ``./upscaled``)
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Exit codes that mean "the run was not a hard failure". 0 is the documented
# success code; 9 is what a successful run actually returns through WSL interop;
# 127 is Git Bash losing the real code. None of them are trustworthy on their
# own, so callers must ALSO confirm an output file appeared.
_SUCCESS_CODES = {0, 9, 127}

# Documented failure codes, straight from `tpai.exe --help`.
_ERROR_CODES = {
    1: "partial success - some files failed",
    253: "invalid argument",
    254: "invalid log token - open Topaz Photo AI normally and sign in",
    255: "no valid files passed",
}

# Install locations to probe, best first. "Topaz Photo AI" is the real CLI;
# "Topaz Photo" ships a stub that produces no output, so it is last and only
# reached if the good one is absent.
_CLI_CANDIDATES = (
    "/mnt/c/Program Files/Topaz Labs LLC/Topaz Photo AI/tpai.exe",
    "C:/Program Files/Topaz Labs LLC/Topaz Photo AI/tpai.exe",
    "/Applications/Topaz Photo AI.app/Contents/MacOS/Topaz Photo AI",
    "/mnt/c/Program Files/Topaz Labs LLC/Topaz Photo/tpai.exe",
    "C:/Program Files/Topaz Labs LLC/Topaz Photo/tpai.exe",
)

# Enhancements tpai can be told to turn on. Autopilot decides the parameters;
# these just force the enhancement enabled regardless of what Autopilot picked.
ENHANCEMENTS = ("upscale", "noise", "sharpen", "lighting", "color")

_REG_KEY = r"HKCU\Software\Topaz Labs LLC\Topaz Photo AI"
_REG_TYPE = "autopilotUpscalingType"      # "auto" | "scale"
_REG_FACTOR = "autopilotUpscalingFactor"  # REG_SZ holding the multiplier

# The Autopilot preference is global to the machine (the GUI reads the same
# keys), so only one override may be in flight at a time.
_REG_LOCK = threading.Lock()

SCALES = (2, 4, 6)


class TopazError(RuntimeError):
    """Topaz CLI missing, unlicensed, or failed to produce an output file."""


def is_wsl() -> bool:
    """True when running inside WSL, where Windows .exe interop is available
    but paths must be translated before being handed to a Windows binary."""
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def cli_path() -> str:
    """Resolve the Topaz Photo AI CLI, or raise with actionable guidance."""
    explicit = os.environ.get("PHOTOSEARCH_TOPAZ_CLI")
    if explicit:
        if not Path(explicit).exists():
            raise TopazError(
                f"PHOTOSEARCH_TOPAZ_CLI points at a missing file: {explicit}")
        return explicit
    for cand in _CLI_CANDIDATES:
        if Path(cand).exists():
            return cand
    raise TopazError(
        "Topaz Photo AI CLI (tpai.exe) not found. Install Topaz Photo AI, or set "
        "PHOTOSEARCH_TOPAZ_CLI to its path. Note: Gigapixel's CLI needs a Pro "
        "license and cannot be used instead.")


def to_native_path(path: str) -> str:
    """Translate a path for consumption by the (Windows) Topaz binary.

    Under WSL the server's paths are Linux paths that ``tpai.exe`` cannot read,
    so they go through ``wslpath -w``. Everywhere else the path is already
    native and passes through untouched.
    """
    if not is_wsl():
        return path
    try:
        out = subprocess.run(["wslpath", "-w", path], capture_output=True,
                             text=True, timeout=15)
        translated = out.stdout.strip()
        if out.returncode == 0 and translated:
            return translated
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("wslpath translation failed for %s: %s", path, e)
    return path


def to_windows_path(path: str) -> Optional[str]:
    """Render ``path`` the way Windows Explorer / Chrome expect it, or None when
    this host has no Windows view of it.

    The server runs in WSL, so a raw result path like ``/mnt/c/Users/...`` is
    unusable to a human on Windows — they want ``C:\\Users\\...``. A WSL-native
    export dir translates to a ``\\\\wsl.localhost\\...`` UNC path, which Windows
    also understands.
    """
    if os.name == "nt":
        return str(Path(path))
    if not is_wsl():
        return None
    try:
        out = subprocess.run(["wslpath", "-w", path], capture_output=True,
                             text=True, timeout=15)
        win = out.stdout.strip()
        return win if out.returncode == 0 and win else None
    except (OSError, subprocess.SubprocessError):
        return None


def to_file_url(win_path: str) -> str:
    """``C:\\a\\b.jpg`` -> ``file:///C:/a/b.jpg`` so it can be pasted into Chrome.

    Note browsers refuse to *navigate* to file:// from an http page, so this is
    for copy-paste; the in-app link uses the same-origin file route instead.
    """
    p = win_path.replace("\\", "/")
    return "file:///" + p.lstrip("/") if not p.startswith("//") else "file://" + p


def export_root() -> Path:
    """Root of the export tree. Kept entirely outside the photo library so the
    index never sees these files."""
    return Path(os.environ.get("PHOTOSEARCH_UPSCALE_DIR") or "./upscaled").expanduser()


def _describe_output(path: Path) -> dict:
    """Path fields for one exported file: the POSIX path the server used, the
    Windows path a human wants, and a file:// URL for pasting into a browser."""
    win = to_windows_path(str(path))
    return {"output": str(path),
            "windows_path": win,
            "file_url": to_file_url(win) if win else None}


def _reg_read(name: str) -> Optional[str]:
    """Read one REG_SZ value, or None if absent/unreadable."""
    try:
        out = subprocess.run(["reg.exe", "query", _REG_KEY, "/v", name],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.replace("\r", "").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == name:
            return parts[2]
    return None


def _reg_write(name: str, value: str) -> bool:
    """Write one REG_SZ value. Returns True on success."""
    try:
        out = subprocess.run(
            ["reg.exe", "add", _REG_KEY, "/v", name, "/t", "REG_SZ",
             "/d", value, "/f"],
            capture_output=True, text=True, timeout=20)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


class _scale_override:
    """Context manager forcing a specific upscale factor, then restoring.

    ``--upscale scale=N`` is broken in the CLI (see module docstring), so the
    only working lever is the Autopilot preference in the registry. That
    preference is shared with the GUI app, hence the lock and the restore in
    ``__exit__``. ``scale=None`` is a no-op: Autopilot's own choice is used and
    the user's settings are never touched.
    """

    def __init__(self, scale: Optional[int]):
        self.scale = scale
        self.prev: dict[str, Optional[str]] = {}
        self.active = False

    def __enter__(self):
        if not self.scale:
            return self
        if not (is_wsl() or os.name == "nt"):
            logger.warning("scale override needs the Windows registry; "
                           "falling back to the Topaz Autopilot default")
            return self
        _REG_LOCK.acquire()
        self.active = True
        self.prev = {_REG_TYPE: _reg_read(_REG_TYPE),
                     _REG_FACTOR: _reg_read(_REG_FACTOR)}
        if not (_reg_write(_REG_TYPE, "scale")
                and _reg_write(_REG_FACTOR, str(self.scale))):
            self.__exit__(None, None, None)
            raise TopazError("could not set the Topaz upscale factor "
                             "(registry write failed)")
        return self

    def __exit__(self, *exc):
        if not self.active:
            return False
        try:
            for name, val in self.prev.items():
                if val is not None:
                    _reg_write(name, val)
        finally:
            self.active = False
            _REG_LOCK.release()
        return False


def run_topaz(src: str, out_dir: str, *,
              scale: Optional[int] = None,
              enhancements: tuple[str, ...] = ("upscale",),
              image_format: str = "preserve",
              quality: int = 95,
              timeout: int = 1800) -> str:
    """Upscale ``src`` into ``out_dir`` and return the produced file's path.

    Raises ``TopazError`` when the CLI is missing, reports a documented failure
    code, or exits without writing anything.
    """
    exe = cli_path()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    before = {p.name for p in Path(out_dir).iterdir()}

    cmd = [exe, "--output", to_native_path(out_dir),
           "--format", image_format, "--quality", str(quality)]
    for enh in enhancements:
        if enh not in ENHANCEMENTS:
            raise ValueError(f"unknown enhancement: {enh}")
        cmd.append(f"--{enh}")
    cmd.append(to_native_path(src))

    with _scale_override(scale):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise TopazError(
                f"Topaz timed out after {timeout}s on {Path(src).name}")

    # The success code is unreliable (9 via WSL, 127 via Git Bash), so a
    # *documented* failure code is the only thing worth trusting, and otherwise
    # we go by whether a file actually appeared.
    if proc.returncode in _ERROR_CODES and proc.returncode not in _SUCCESS_CODES:
        raise TopazError(f"Topaz failed ({proc.returncode}): "
                         f"{_ERROR_CODES[proc.returncode]}")

    produced = [p for p in Path(out_dir).iterdir() if p.name not in before]
    if not produced:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise TopazError(
            f"Topaz produced no output for {Path(src).name} "
            f"(exit {proc.returncode})" + (f": {' | '.join(tail)}" if tail else ""))
    return str(max(produced, key=lambda p: p.stat().st_mtime))


def _variant_suffix(scale: Optional[int],
                    enhancements: tuple[str, ...]) -> str:
    """Encode the chosen options into the filename.

    Two different option sets must not collide on one path — otherwise the
    "already exported" check makes the *first* run permanently shadow every
    later one, and re-running with different settings silently returns the old
    file. Canonical ENHANCEMENTS order (not the caller's) keeps the name stable
    however the checkboxes were clicked.
    """
    chosen = set(enhancements)
    parts = [x for x in ENHANCEMENTS if x in chosen]
    parts += sorted(chosen - set(ENHANCEMENTS))
    if scale:
        parts.append(f"{scale}x")
    return "".join("-" + p for p in parts)


def _parse_variant(name: str, stem: str) -> dict:
    """Inverse of :func:`_variant_suffix` — recover the options from a filename
    so prior exports can be listed without a database of their own."""
    body = os.path.splitext(name)[0]
    prefix = f"{stem}-topaz"
    tokens = [t for t in body[len(prefix):].split("-") if t]
    scale, enh = None, []
    for t in tokens:
        if re.fullmatch(r"\d+x", t):
            scale = int(t[:-1])
        else:
            enh.append(t)
    label = ", ".join(enh) if enh else "autopilot"
    return {"enhancements": enh, "scale": scale or "auto",
            "label": label + (f" · {scale}x" if scale else "")}


def list_exports(db, photo_id: int) -> list[dict]:
    """Every prior export for one photo, newest first.

    The export tree is the only record — nothing about upscaling is stored in
    the DB — so this globs the photo's destination folder rather than querying.
    """
    row = db.get_photo(photo_id)
    if not row:
        raise ValueError(f"photo {photo_id} not found")

    stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
    dest = _dest_dir(row)
    out = []
    for p in sorted(dest.glob(f"{stem}-topaz*"), reverse=True):
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"bytes": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).isoformat(
                        timespec="seconds"),
                    **_parse_variant(p.name, stem), **_describe_output(p)})
    out.sort(key=lambda r: r["modified"], reverse=True)
    return out


def _dest_dir(row) -> Path:
    """Mirror the library's ``YYYY/`` shape inside the export tree so exports
    stay browsable, without ever writing into the library itself."""
    taken = row.get("date_taken") if hasattr(row, "get") else None
    year = None
    if taken:
        try:
            year = str(datetime.fromisoformat(str(taken)).year)
        except (ValueError, TypeError):
            year = None
    return export_root() / (year or "undated")


def upscale_photo(db, photo_id: int, *,
                  scale: Optional[int] = None,
                  enhancements: tuple[str, ...] = ("upscale",),
                  server: Optional[str] = None,
                  overwrite: bool = False) -> dict:
    """Upscale one library photo into the export tree.

    The original is fetched from the authoritative server (the NAS) exactly the
    way M28's sync re-run does, upscaled locally, and the result is moved into
    ``PHOTOSEARCH_UPSCALE_DIR``. Nothing is written to the DB or the NAS.
    """
    from . import rerun

    row = db.get_photo(photo_id)
    if not row:
        raise ValueError(f"photo {photo_id} not found")

    src_name = os.path.basename(row["filepath"])
    dest_dir = _dest_dir(row)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem, ext = os.path.splitext(src_name)
    variant = _variant_suffix(scale, enhancements)
    final = dest_dir / f"{stem}-topaz{variant}{ext}"
    if final.exists() and not overwrite:
        return {"photo_id": photo_id, "skipped": True,
                "reason": "already exported", "scale": scale or "auto",
                "enhancements": list(enhancements),
                "label": _parse_variant(final.name, stem)["label"],
                **_describe_output(final)}

    # Stage under the export root (not /tmp): on WSL the Topaz binary is a
    # Windows process, and driving it against a Linux-only temp path costs a
    # slow \\wsl.localhost round trip for every read and write.
    staging = dest_dir / f".staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        base = server or rerun.nas_base()
        if base:
            from . import worker as W
            client = W.WorkerClient(base, probe=False)
            info = {"id": row["id"], "filepath": row["filepath"],
                    "filename": src_name}
            got = W._download_batch(client, [info], str(staging))
            if not got:
                raise TopazError(
                    f"could not download photo {photo_id} from {base}")
            src = got[0][1]
        else:
            # Running where the originals live - use the file directly.
            src = row["filepath"]
            if not os.path.exists(src):
                raise TopazError(f"original not found on disk: {src}")

        produced = run_topaz(src, str(staging / "out"), scale=scale,
                             enhancements=enhancements)
        shutil.move(produced, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {"photo_id": photo_id, "skipped": False,
            "bytes": final.stat().st_size, "scale": scale or "auto",
            "enhancements": list(enhancements),
            "label": _parse_variant(final.name, stem)["label"],
            **_describe_output(final)}

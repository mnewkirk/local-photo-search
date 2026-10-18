"""Register video files as ``photos`` rows (schema v35, video-support step 1).

``docs/plans/video-support.md``. A video is a ``photos`` row with
``media_type = 'video'`` plus an EMPTY ``video_meta`` row
(``probe_version IS NULL`` is the future ``video-probe`` queue). This module is
the NAS's whole share of video work: it walks the tree and inserts rows.

**No ffprobe, no decode, no hashing here.** The NAS is an N100 on a spinning
disk; reading ~400 GB to hash or probe it is exactly the disk traffic the
2026-09-19 starvation incident taught us not to generate. Ingest passes the
hash it has already computed; the backfill leaves ``file_hash`` NULL and the
desktop ``video-scan`` pass fills it while it streams the file anyway.

``date_taken`` is the file **mtime, provisionally** (ingest passes the date it
routed the file by, which is the same mtime unless the file carried a readable
EXIF date). The probe pass replaces it with the container's creation time.

Two entry points:

- ``register_video(db, path, ...)`` -- one file, used by ``ingest.py`` right
  after a video companion is moved into its dated folder.
- ``register_videos(db_path, ...)`` -- the backfill driver behind
  ``photosearch register-videos``. Dry run by default and then opens the DB
  ``mode=ro`` (a missing DB is an error, never a stub); ``--apply`` commits
  every ``COMMIT_EVERY`` rows so the fleet's ``submit-results`` never waits
  long on the write lock.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

COMMIT_EVERY = 25

_YEAR_RE = re.compile(r"^\d{4}$")


def _video_extensions() -> set[str]:
    from .ingest import VIDEO_EXTENSIONS
    return VIDEO_EXTENSIONS


def _mtime_date(path: str) -> Optional[str]:
    try:
        ts = os.stat(path).st_mtime
    except OSError:
        return None
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _relative(db, abs_path: str) -> str:
    """Stored-path spelling, exactly as ``PhotoDB.relative_filepath`` does it
    (works on the read-only stand-in too, which only carries ``photo_root``)."""
    from .db import PhotoDB
    return PhotoDB.relative_filepath(db, abs_path)


def register_video(db, path: str, file_hash: Optional[str] = None, *,
                   date_taken: Optional[str] = None, commit: bool = True) -> int:
    """Insert one video's ``photos`` row + empty ``video_meta`` row.

    ``path`` is absolute (or relative to the cwd); it is stored relative to
    ``db.photo_root`` like every other photo. Idempotent: an existing row for
    the same filepath is returned unchanged (``add_photo`` is
    ``INSERT OR IGNORE`` on the UNIQUE filepath) and its ``video_meta`` row is
    created only if missing.

    The two inserts land in ONE transaction -- a photos row without its
    video_meta row would never enter the probe queue. ``commit=True`` then
    commits per ``PhotoDB._maybe_commit`` (now, or at the caller's
    ``begin_batch`` boundary); ``commit=False`` leaves it to the caller (the
    backfill's chunked commits).
    """
    abs_path = str(Path(path).resolve())
    stored = _relative(db, abs_path)
    kwargs = {
        "filepath": stored,
        "filename": os.path.basename(abs_path),
        "media_type": "video",
        "date_taken": date_taken or _mtime_date(abs_path),
    }
    if file_hash:
        kwargs["file_hash"] = file_hash
    # Hold add_photo's own commit so both rows share one transaction.
    saved = (db._batch_mode, getattr(db, "_batch_size", 50),
             getattr(db, "_batch_count", 0))
    db._batch_mode, db._batch_size = True, float("inf")
    try:
        photo_id = db.add_photo(**kwargs)
        db.conn.execute("INSERT OR IGNORE INTO video_meta (photo_id) VALUES (?)",
                        (photo_id,))
    finally:
        db._batch_mode, db._batch_size, db._batch_count = saved
    if commit:
        db._maybe_commit()
    return photo_id


def iter_video_files(root: str, folder: Optional[str] = None):
    """Yield absolute paths of video files under ``root`` (or ``root/folder``),
    sorted, skipping excluded/hidden directories and AppleDouble ``._*`` files.

    The excluded set is the image walk's (``index.is_excluded_dir``:
    ``_incoming``, thumbnails, NAS trash/system dirs such as ``#recycle``,
    ``#snapshot``, ``@eaDir``), so the staging tree -- and its ``.processed``
    archive -- is never walked. Hidden (``.staging-*``, ``.@__thumb``) and
    ``@``-prefixed directories are skipped too.
    """
    from .index import is_excluded_dir
    exts = _video_extensions()
    start = Path(root)
    if folder:
        f = Path(folder)
        start = f if f.is_absolute() else start / f
    if not start.is_dir():
        return
    for dirpath, dirs, files in os.walk(start):
        dirs[:] = sorted(d for d in dirs
                         if not is_excluded_dir(d) and not d.startswith(".")
                         and not d.startswith("@"))
        for name in sorted(files):
            if name.startswith("._"):
                continue
            if Path(name).suffix.lower() in exts:
                yield os.path.join(dirpath, name)


def _year_of(stored: str) -> str:
    first = stored.split("/", 1)[0]
    return first if _YEAR_RE.match(first) else "other"


def register_videos(db_path: str, *, photo_root: Optional[str] = None,
                    folder: Optional[str] = None, limit: Optional[int] = None,
                    apply: bool = False, lower_priority: bool = True,
                    on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """Backfill driver: register every video file under the photo root that
    has no ``photos`` row (matched by stored filepath).

    Dry run (default) opens the DB read-only and writes nothing. ``limit``
    caps how many NEW rows are registered (or would be). Returns counts:
    ``files`` (video files seen), ``already`` (have a row), ``registered``
    (or ``would_register`` on a dry run), ``per_year`` (new rows by the
    library's top-level year folder), ``root``, ``dry_run``.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"database not found: {db_path}")

    if apply:
        from .db import PhotoDB
        db = PhotoDB(db_path, photo_root=photo_root)
    else:
        from .sharpness_backfill import open_readonly
        db = open_readonly(db_path, photo_root)

    applied_priority = {"nice": False, "ionice": False}
    counts = {"files": 0, "already": 0, "registered": 0, "would_register": 0,
              "per_year": {}, "dry_run": not apply, "root": None,
              "priority": applied_priority}
    try:
        root = db.photo_root
        if not root:
            raise ValueError("no photo root: pass --photo-root or set PHOTO_ROOT "
                             "(the DB stores none)")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"photo root not accessible: {root}")
        counts["root"] = root

        if apply and lower_priority:
            from .sharpness_backfill import lower_priority as _lower
            counts["priority"] = _lower()

        pending = 0
        new = 0
        for abs_path in iter_video_files(root, folder):
            counts["files"] += 1
            stored = _relative(db, abs_path)
            exists = db.conn.execute(
                "SELECT 1 FROM photos WHERE filepath = ?", (stored,)).fetchone()
            if exists:
                counts["already"] += 1
                continue
            if limit is not None and new >= int(limit):
                continue
            new += 1
            year = _year_of(stored)
            counts["per_year"][year] = counts["per_year"].get(year, 0) + 1
            if not apply:
                counts["would_register"] += 1
                continue
            register_video(db, abs_path, commit=False)
            counts["registered"] += 1
            pending += 1
            if pending >= COMMIT_EVERY:
                db.conn.commit()
                pending = 0
                if on_progress:
                    on_progress({"registered": counts["registered"],
                                 "files": counts["files"]})
        if apply:
            db.conn.commit()
    finally:
        db.close()
    counts["per_year"] = dict(sorted(counts["per_year"].items()))
    return counts

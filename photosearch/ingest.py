"""Sweep phone-synced photos out of /photos/_incoming/<source>/ into the
year/date library layout, dedup against existing photos by file hash, and
archive the source files so the next run only touches what arrived since.

Designed for a daily cron driven by Syncthing-style phone backups:

    /photos/_incoming/matt/...        # Android camera roll via Syncthing
    /photos/_incoming/wife/...        # iPhone camera roll via Möbius Sync

After a sweep:

    /photos/YYYY/YYYY-MM-DD_phone-matt/IMG_xxxx.jpg          # moved here
    /photos/_incoming/matt/.processed/<original-subpath>     # source archived

`.processed/` is hidden from `find_photos()` because it lives under the
incoming root, which is not part of the indexed library tree.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .db import PhotoDB
from .exif import extract_exif
from .index import file_hash, JPEG_EXTENSIONS, HEIC_EXTENSIONS

# Phones produce JPEG (Android default) and HEIC (iPhone default). No RAW.
INGEST_EXTENSIONS = JPEG_EXTENSIONS | HEIC_EXTENSIONS

# Subfolder inside each <source>/ where processed sources are archived.
# Leading dot keeps it out of phone-side galleries that re-scan the folder.
ARCHIVE_DIRNAME = ".processed"

# Folder used when neither EXIF date nor mtime yields anything sensible.
# Should be exceptionally rare for phone-sourced media.
UNDATED_DIRNAME = "_undated"


def _iter_source_files(source_root: Path) -> list[Path]:
    """Yield every photo under one source dir, excluding the archive folder."""
    out: list[Path] = []
    for root, dirs, files in os.walk(source_root):
        dirs[:] = [d for d in dirs if d != ARCHIVE_DIRNAME]
        for fname in sorted(files):
            if fname.startswith("._"):
                continue
            ext = Path(fname).suffix.lower()
            if ext not in INGEST_EXTENSIONS:
                continue
            out.append(Path(root) / fname)
    return out


def _parse_date_taken(exif_value: Optional[str]) -> Optional[datetime]:
    """extract_exif returns 'YYYY-MM-DD HH:MM:SS' (already normalized)."""
    if not exif_value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(exif_value, fmt)
        except ValueError:
            continue
    return None


def _target_dir(photo_root: Path, source: str, taken: Optional[datetime]) -> Path:
    """Compute the year/date subfolder for one photo.

    YYYY-MM-DD_phone-<source>/ keeps origin visible at the path level,
    matching the M20 Takeout convention of `_gphotos` suffixes.
    """
    if taken is None:
        return photo_root / UNDATED_DIRNAME / f"phone-{source}"
    return photo_root / f"{taken.year:04d}" / f"{taken.strftime('%Y-%m-%d')}_phone-{source}"


def _unique_target_path(target_dir: Path, filename: str) -> Path:
    """Avoid clobbering an unrelated file at the same DSC name."""
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    ext = Path(filename).suffix
    n = 1
    while True:
        candidate = target_dir / f"{stem}_{n}{ext}"
        if not candidate.exists():
            return candidate
        n += 1


def _archive_path(source_root: Path, src_file: Path) -> Path:
    """Where a successfully-processed source file gets moved to."""
    rel = src_file.relative_to(source_root)
    return source_root / ARCHIVE_DIRNAME / rel


def ingest_incoming(
    incoming_root: str,
    photo_root: str,
    db_path: str,
    dry_run: bool = False,
) -> dict:
    """Sweep every direct subdir of incoming_root into photo_root/YYYY/...

    Each direct subdir under incoming_root is treated as one "source" label.
    Anything else at the top level is ignored.

    Returns a stats dict:
        {
          "sources": {
            <source>: {
              "scanned": int, "imported": int, "deduped": int,
              "errors": int, "new_dirs": [str, ...],
            }, ...
          },
          "totals": {"scanned": int, "imported": int, "deduped": int, "errors": int},
          "dry_run": bool,
        }
    """
    incoming = Path(incoming_root).resolve()
    photo_dir = Path(photo_root).resolve()
    if not incoming.is_dir():
        raise FileNotFoundError(f"incoming_root not found: {incoming}")
    if not photo_dir.is_dir():
        raise FileNotFoundError(f"photo_root not found: {photo_dir}")

    sources = [p for p in sorted(incoming.iterdir())
               if p.is_dir() and not p.name.startswith(".")]

    per_source: dict[str, dict] = {}
    totals = {"scanned": 0, "imported": 0, "deduped": 0, "errors": 0}

    with PhotoDB(db_path) as db:
        for source_root in sources:
            source = source_root.name
            stats = {"scanned": 0, "imported": 0, "deduped": 0,
                     "errors": 0, "new_dirs": []}
            new_dirs: set[Path] = set()

            for src_file in _iter_source_files(source_root):
                stats["scanned"] += 1
                try:
                    h = file_hash(str(src_file))
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  [{source}] HASH FAIL {src_file.name}: {exc}")
                    continue

                # Dedup by content hash. The photos table file_hash column has
                # been populated on every newly-indexed photo for a long time;
                # an older un-hashed match would just route as a new import,
                # which is safe but slightly redundant.
                existing = db.conn.execute(
                    "SELECT id, filepath FROM photos WHERE file_hash = ? LIMIT 1",
                    (h,),
                ).fetchone()

                if existing is not None:
                    stats["deduped"] += 1
                    if not dry_run:
                        _move_to_archive(source_root, src_file)
                    continue

                meta = extract_exif(str(src_file))
                taken = _parse_date_taken(meta.get("date_taken"))
                if taken is None:
                    # mtime fallback already filled date_created in meta;
                    # use it for routing too rather than dumping to _undated.
                    taken = _parse_date_taken(meta.get("date_created"))

                tdir = _target_dir(photo_dir, source, taken)
                target = _unique_target_path(tdir, src_file.name)

                if dry_run:
                    stats["imported"] += 1
                    new_dirs.add(tdir)
                    continue

                try:
                    tdir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_file), str(target))
                    new_dirs.add(tdir)
                    stats["imported"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  [{source}] MOVE FAIL {src_file.name} -> {target}: {exc}")
                    continue

                # Archive: the source file no longer exists at src_file (we
                # moved it). The "archive" step for an imported file is a
                # no-op — the archive is for *deduped* sources that we kept
                # a copy of. Recording new_dirs is what matters for caller
                # follow-up indexing.

            stats["new_dirs"] = sorted(str(p) for p in new_dirs)
            per_source[source] = stats
            for k in totals:
                totals[k] += stats[k]

    return {"sources": per_source, "totals": totals, "dry_run": dry_run}


def _move_to_archive(source_root: Path, src_file: Path) -> None:
    """Move a deduped source file into <source>/.processed/ preserving subpath."""
    dest = _archive_path(source_root, src_file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # If the archive already has a file at this path (e.g. user re-synced the
    # exact same camera roll twice), overwrite — the bytes are identical by
    # definition (we got here via hash match).
    if dest.exists():
        dest.unlink()
    shutil.move(str(src_file), str(dest))

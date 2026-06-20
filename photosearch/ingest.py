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
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .db import PhotoDB
from .exif import extract_exif
from .index import file_hash, JPEG_EXTENSIONS, HEIC_EXTENSIONS

# Phones produce JPEG (Android default) and HEIC (iPhone default). These are
# the types we CLIP-index. Camera SD cards also bring RAW + video — we relocate
# those into the library too (so every file lands on the NAS) but never index
# them. See `_iter_source_files` / the photo-vs-companion split in the loop.
INGEST_EXTENSIONS = JPEG_EXTENSIONS | HEIC_EXTENSIONS

# Companion media: moved into the dated library folder but NOT indexed (no DB
# row, no CLIP). RAW carries EXIF (so it date-routes + co-locates with its
# JPEG); video usually doesn't, so it falls back to file mtime.
RAW_EXTENSIONS = {
    ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".dng", ".raf", ".rw2",
    ".orf", ".pef", ".srw", ".raw", ".rwl", ".sr2",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mts", ".m2ts", ".3gp"}
COMPANION_EXTENSIONS = RAW_EXTENSIONS | VIDEO_EXTENSIONS

# Everything ingest will physically relocate out of _incoming/.
ALL_MEDIA_EXTENSIONS = INGEST_EXTENSIONS | COMPANION_EXTENSIONS

# Subfolder inside each <source>/ where processed sources are archived.
# Leading dot keeps it out of phone-side galleries that re-scan the folder.
ARCHIVE_DIRNAME = ".processed"

# Folder used when neither EXIF date nor mtime yields anything sensible.
# Should be exceptionally rare for phone-sourced media.
UNDATED_DIRNAME = "_undated"

# A source label like 'ILCE-7RM6' / 'ILCE-7M4' / 'EOS R5' is an EXIF camera
# model (the SD-card import path names _incoming/<model>/ subdirs after the
# body). Those land in YYYY-MM-DD_<model>/ — no 'phone-' prefix. Human source
# labels like 'matt' / 'wife' (Syncthing phone roll) stay phone-<source>/.
# Discriminator: all-caps alnum token containing at least one digit.
_CAMERA_MODEL_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ._-]*$")

# Fallback label the SD-card importer uses for media with no readable camera
# model (mostly video). Treated as bare so it doesn't masquerade as a phone
# roll (`_phone-unknown-camera`).
_DEFAULT_BARE_SOURCES = frozenset({"unknown-camera"})


def _looks_like_camera_model(source: str) -> bool:
    """Heuristic: uppercase alnum model code with a digit (e.g. 'ILCE-7RM6').

    Phone source labels ('matt', 'wife') contain lowercase letters, so they
    fail this and keep the 'phone-' prefix.
    """
    return bool(_CAMERA_MODEL_RE.match(source)) and any(c.isdigit() for c in source)


def _folder_suffix(source: str,
                   bare_sources: Optional[Iterable[str]] = None,
                   phone_sources: Optional[Iterable[str]] = None) -> str:
    """Dated/undated folder suffix for one source label.

    Precedence: explicit phone_sources (force 'phone-') > explicit
    bare_sources (force bare) > camera-model heuristic > default 'phone-'.
    """
    bare = set(bare_sources or ())
    phone = set(phone_sources or ())
    if source in phone:
        return f"phone-{source}"
    if source in bare or source in _DEFAULT_BARE_SOURCES or _looks_like_camera_model(source):
        return source
    return f"phone-{source}"


def _iter_source_files(source_root: Path) -> list[Path]:
    """Yield every media file under one source dir, excluding the archive folder.

    Includes photos (JPEG/HEIC), RAW, and video — the photo-vs-companion split
    happens in the ingest loop. Non-media (Sony db sidecars, AppleDouble, etc.)
    is left in place.
    """
    out: list[Path] = []
    for root, dirs, files in os.walk(source_root):
        dirs[:] = [d for d in dirs if d != ARCHIVE_DIRNAME]
        for fname in sorted(files):
            if fname.startswith("._"):
                continue
            ext = Path(fname).suffix.lower()
            if ext not in ALL_MEDIA_EXTENSIONS:
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


def _target_dir(photo_root: Path, suffix: str, taken: Optional[datetime]) -> Path:
    """Compute the year/date subfolder for one photo.

    YYYY-MM-DD_<suffix>/ keeps origin visible at the path level, matching the
    M20 Takeout convention of `_gphotos` suffixes. `suffix` is precomputed by
    `_folder_suffix` (e.g. 'phone-matt' for phones, 'ILCE-7RM6' for cameras).
    """
    if taken is None:
        return photo_root / UNDATED_DIRNAME / suffix
    return photo_root / f"{taken.year:04d}" / f"{taken.strftime('%Y-%m-%d')}_{suffix}"


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
    bare_sources: Optional[Iterable[str]] = None,
    phone_sources: Optional[Iterable[str]] = None,
) -> dict:
    """Sweep every direct subdir of incoming_root into photo_root/YYYY/...

    Each direct subdir under incoming_root is treated as one "source" label.
    Anything else at the top level is ignored.

    `bare_sources` forces the bare `_<source>/` folder suffix (no 'phone-'
    prefix); `phone_sources` forces the `_phone-<source>/` suffix. Sources not
    listed in either fall back to the camera-model heuristic (`_folder_suffix`).

    Photos (JPEG/HEIC) are deduped against `photos.file_hash`, moved into the
    dated library folder, and their folder is queued for CLIP indexing. RAW +
    video are "companions": moved into the same dated folder so every file
    reaches the NAS, but never added to the DB or indexed. Companions dedup
    against the destination (same name + same hash already present) since they
    have no DB row.

    Returns a stats dict:
        {
          "sources": {
            <source>: {
              "scanned": int, "imported": int, "deduped": int,
              "companions_moved": int, "companions_deduped": int,
              "errors": int, "new_dirs": [str, ...],
            }, ...
          },
          "totals": {"scanned", "imported", "deduped", "companions_moved",
                     "companions_deduped", "errors"},
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
    _zero = lambda: {"scanned": 0, "imported": 0, "deduped": 0,
                     "companions_moved": 0, "companions_deduped": 0, "errors": 0}
    totals = _zero()

    with PhotoDB(db_path) as db:
        for source_root in sources:
            source = source_root.name
            suffix = _folder_suffix(source, bare_sources, phone_sources)
            stats = _zero()
            stats["new_dirs"] = []
            new_dirs: set[Path] = set()

            for src_file in _iter_source_files(source_root):
                stats["scanned"] += 1
                is_photo = src_file.suffix.lower() in INGEST_EXTENSIONS
                try:
                    h = file_hash(str(src_file))
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  [{source}] HASH FAIL {src_file.name}: {exc}")
                    continue

                if is_photo:
                    # Dedup by content hash against the photos table. The
                    # file_hash column has been populated on every newly-indexed
                    # photo for a long time; an older un-hashed match would just
                    # route as a new import, which is safe but slightly redundant.
                    existing = db.conn.execute(
                        "SELECT id, filepath FROM photos WHERE file_hash = ? LIMIT 1",
                        (h,),
                    ).fetchone()
                    if existing is not None:
                        stats["deduped"] += 1
                        if not dry_run:
                            _move_to_archive(source_root, src_file)
                        continue

                # Date routing. extract_exif is photo-oriented and may not read
                # a date from video — fall back to file mtime so videos still
                # land in a dated folder instead of _undated.
                try:
                    meta = extract_exif(str(src_file))
                except Exception:
                    meta = {}
                taken = (_parse_date_taken(meta.get("date_taken"))
                         or _parse_date_taken(meta.get("date_created")))
                if taken is None and not is_photo:
                    # Video rarely carries EXIF — use file mtime so it still
                    # lands in a dated folder. Photos keep the original behavior
                    # (no date → _undated) so a truly-dateless photo is visible.
                    try:
                        taken = datetime.fromtimestamp(src_file.stat().st_mtime)
                    except OSError:
                        taken = None

                tdir = _target_dir(photo_dir, suffix, taken)

                if not is_photo:
                    # Companion (RAW/video): no DB row to dedup against, so
                    # dedup at the destination — same filename + identical bytes
                    # already there means we've moved this exact file before.
                    cand = tdir / src_file.name
                    if cand.exists():
                        try:
                            same = file_hash(str(cand)) == h
                        except Exception:
                            same = False
                        if same:
                            stats["companions_deduped"] += 1
                            if not dry_run:
                                _move_to_archive(source_root, src_file)
                            continue

                target = _unique_target_path(tdir, src_file.name)

                if dry_run:
                    if is_photo:
                        stats["imported"] += 1
                        new_dirs.add(tdir)
                    else:
                        stats["companions_moved"] += 1
                    continue

                try:
                    tdir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src_file), str(target))
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"  [{source}] MOVE FAIL {src_file.name} -> {target}: {exc}")
                    continue

                if is_photo:
                    # new_dirs drives the follow-up CLIP index pass. Companion
                    # folders are deliberately left out — RAW/video are never
                    # indexed (a photo from the same shoot adds the dir anyway).
                    new_dirs.add(tdir)
                    stats["imported"] += 1
                else:
                    stats["companions_moved"] += 1

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

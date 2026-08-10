"""Translate a photo-search book (BookStore.get_book) into artifacts for the
Shutterfly export: an ordered placed-photo list and a JSON manifest that Phase 2
(native rebuild) consumes.

Photos are identified across the Google Photos → Shutterfly hop by their
**original filename**, which is unique within a book (the cameras use
non-overlapping filename sequences). An earlier ``sfly-<id>`` upload-rename was
dropped: Google Photos keeps the raw-upload filename (the original basename) and
ignores the ``simpleMediaItem.fileName`` override, so the rename never reached
Shutterfly — and it was unnecessary given filenames are already unique.

Pure functions only — no DB or network. The CLI wires these to PhotoDB/BookStore
and google_photos.
"""
from __future__ import annotations

from typing import Callable


def placed_photo_order(doc: dict) -> list[int]:
    """Unique placed photo ids in spread order then cell position, followed by
    cover / title / back-cover ids if present and not already included."""
    seen: set[int] = set()
    order: list[int] = []

    def add(pid) -> None:
        if pid is None:
            return
        pid = int(pid)
        if pid not in seen:
            seen.add(pid)
            order.append(pid)

    for sp in doc.get("spreads", []):
        for cell in sp.get("cells", []):
            add(cell.get("photo_id"))
    book = doc.get("book", {})
    for key in ("cover_photo_id", "title_page_photo_id", "back_cover_photo_id"):
        add(book.get(key))
    return order


def build_book_manifest(doc: dict) -> dict:
    """Structural manifest: book meta + verbatim spreads/cells/captions + a
    photos map keyed by str(photo_id). ``orig_filename`` (the mapping key back to
    Shutterfly) and ``sfly_asset_id`` are filled later by the CLI/runbook (left
    empty here so this stays pure)."""
    book = doc.get("book", {})
    photos = {str(pid): {"orig_filename": None, "sfly_asset_id": None}
              for pid in placed_photo_order(doc)}
    return {
        "book_id": book.get("id"),
        "name": book.get("name"),
        "subtitle": book.get("subtitle"),
        "stage_w": doc.get("stage_w"),
        "stage_h": doc.get("stage_h"),
        "cover_photo_id": book.get("cover_photo_id"),
        "title_page_photo_id": book.get("title_page_photo_id"),
        "back_cover_photo_id": book.get("back_cover_photo_id"),
        "spread_count": len(doc.get("spreads", [])),
        "spreads": doc.get("spreads", []),
        "photos": photos,
    }


def build_gphotos_records(
    photo_ids: list[int],
    rows: dict[int, dict],
    resolve: Callable[[str], str],
) -> list[dict]:
    """Build google_photos.upload_photos records, keeping the original filename.

    ``rows`` maps photo_id -> {filepath, filename, description}. ``resolve``
    turns a stored (possibly relative) filepath into an absolute path.
    """
    records: list[dict] = []
    for pid in photo_ids:
        row = rows.get(int(pid))
        if not row:
            continue
        orig = row.get("filename")
        records.append({
            "_resolved_filepath": resolve(row.get("filepath", "")),
            "filename": orig,
            "orig_filename": orig,
            "description": row.get("description"),
        })
    return records


def enrich_manifest_filenames(manifest: dict, rows: dict[int, dict]) -> dict:
    """Fill photos[*].orig_filename from photo rows, in place."""
    for pid_str, rec in manifest.get("photos", {}).items():
        row = rows.get(int(pid_str))
        if row:
            rec["orig_filename"] = row.get("filename")
    return manifest


def manifest_photo_ids(manifest: dict) -> list[int]:
    """Ordered photo ids from a manifest's ``photos`` map (keys are str ids).

    Lets the Google Photos push run NAS-side from a manifest (built on the
    replica, which owns the books DB) without needing the books DB itself.
    """
    return [int(k) for k in manifest.get("photos", {}).keys()]

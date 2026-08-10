# Shutterfly Export — Phase 1 (Google Photos upload) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push each 2026 Summer book's placed photos into a per-book Google Photos album with deterministic, mapping-friendly filenames, so they can be imported into Shutterfly's editor — and emit a manifest that Phase 2 will consume.

**Architecture:** A new pure module (`photosearch/shutterfly_export.py`) turns a `BookStore.get_book()` document into (a) an ordered list of placed photos and (b) a JSON manifest. Two thin `cli.py` commands wrap it: one writes the manifest, one creates the Google Photos album and uploads via the existing `photosearch/google_photos.py` (`create_album` + `upload_photos`). Google-Photos → Shutterfly import and mapping recovery are manual browser steps documented in the runbook.

**Tech Stack:** Python 3.11, Click (CLI), SQLite via `PhotoDB`/`BookStore`, existing `google_photos.py` (OAuth2 appendonly), pytest.

## Global Constraints

- **No hardcoded network identifiers.** Never write the tailnet IP or NAS hostname into any tracked file. Use the existing `PHOTOSEARCH_NAS_URL` env var / `<nas-tailscale-ip>` placeholder convention. (User global rule + repo convention.)
- **Books DB is a sidecar.** Book data lives in `photobooks.db.local` (via `BookStore`), separate from the main `photo_index.db` (via `PhotoDB`). Photo file paths/metadata come from `PhotoDB`.
- **CLI convention.** Every command uses `@cli.command("name")`, `--db` with `envvar="PHOTOSEARCH_DB"`, and (new) `--books-db` with `envvar="PHOTOSEARCH_BOOKS_DB"` defaulting to `photobooks.db.local` alongside `--db`.
- **Deterministic upload filename scheme:** `sfly-<photo_id><ext>` (e.g. `sfly-240599.jpg`), where `<ext>` is the original file's extension. Mapping recovery parses `^sfly-(\d+)`.
- **Gentleness.** Reuse `upload_photos`' existing batching; do not add parallelism. It's Matt's own account/photos.
- **Faithful-ish fidelity** (Phase 2 concern) — the manifest carries raw cell geometry/crop verbatim for Phase 2; Phase 1 does not transform crops.

---

## File Structure

- **Create** `photosearch/shutterfly_export.py` — pure helpers:
  - `placed_photo_order(doc) -> list[int]`
  - `upload_filename(photo_id, orig_filename) -> str`
  - `build_book_manifest(doc) -> dict`
  - `build_gphotos_records(photo_ids, rows, resolve) -> list[dict]`
- **Create** `tests/test_shutterfly_export.py` — unit tests for all of the above.
- **Modify** `cli.py` — add `shutterfly-manifest` and `shutterfly-gphotos-push` commands (append after the existing book/maintenance commands; follow the `@cli.command("name")` pattern used throughout).
- **Modify** `tests/test_cli_shutterfly.py` (Create) — tests for the two commands with `google_photos` mocked.

No `web.py` endpoint in Phase 1 — the CLI (run on the NAS via `docker compose run`, where the photo files live) is the delivery vehicle. A UI/SSE endpoint is deferred (YAGNI).

---

### Task 1: Spike S1 — verify Google Photos → Shutterfly filename fidelity (manual, gating)

This validates the mapping scheme before we build on it. Throwaway material only.

**Files:** none (investigation; record the outcome at the bottom of this plan).

- [ ] **Step 1: Pick 3 test photos** already in Google Photos auth scope. Note their intended ids, e.g. pretend `sfly-111.jpg`, `sfly-222.jpg`, `sfly-333.jpg`.

- [ ] **Step 2: Upload them to a scratch GP album** with `simpleMediaItem.fileName = sfly-<id>.jpg`. Quickest path: a Python REPL on the NAS using the existing helpers:

```python
from photosearch import google_photos as gp
db = "photo_index.db"  # PHOTOSEARCH_DB
album = gp.create_album(db, "SFLY MAPPING TEST")
recs = [
    {"_resolved_filepath": "/photos/<any>/a.jpg", "filename": "sfly-111.jpg"},
    {"_resolved_filepath": "/photos/<any>/b.jpg", "filename": "sfly-222.jpg"},
    {"_resolved_filepath": "/photos/<any>/c.jpg", "filename": "sfly-333.jpg"},
]
print(gp.upload_photos(db, recs, album_id=album))
```

- [ ] **Step 3: In Shutterfly**, open any book project → Add More Photos → Google Photos → import from "SFLY MAPPING TEST".

- [ ] **Step 4: Inspect the imported asset filenames** in `builderApp` (via the Chrome MCP `javascript_tool`). Confirm whether the Shutterfly asset carries `sfly-111.jpg` (from `simpleMediaItem.fileName`) or the raw upload-header name (from `_upload_raw_bytes`, which uses the file's own basename).

- [ ] **Step 5: Record the outcome** in "Spike S1 findings" below and decide:
  - If `simpleMediaItem.fileName` **survives** → the scheme works as-is (no `google_photos.py` change).
  - If the **upload-header basename** wins instead → add an optional `upload_name` override to `_upload_raw_bytes` (small follow-up task) and stage names accordingly.
  - Delete the scratch GP album and any imported test assets.

**Deliverable:** a decision recorded below. Do not start Task 5's real pushes until this is settled.

---

### Task 2: `placed_photo_order` + `build_book_manifest`

**Files:**
- Create: `photosearch/shutterfly_export.py`
- Test: `tests/test_shutterfly_export.py`

**Interfaces:**
- Consumes: a `doc` shaped like `BookStore.get_book()`:
  `{"book": {"id","name","subtitle","cover_photo_id","title_page_photo_id","back_cover_photo_id"}, "spreads": [{"position","archetype","bg","caption": {"text","dark"} | None, "cells": [{"photo_id","x","y","w","h","fit","crop_cx","crop_cy","crop_zoom","crop_min_w","crop_min_h","align","position"}]}], "stage_w", "stage_h"}`
- Produces:
  - `placed_photo_order(doc) -> list[int]` — unique placed photo ids in spread order then cell `position`, followed by cover/title/back cover ids if not already present (skipping `None`).
  - `upload_filename(photo_id: int, orig_filename: str) -> str` — `f"sfly-{photo_id}{ext}"`.
  - `build_book_manifest(doc) -> dict` — see schema in Step 3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_shutterfly_export.py
from photosearch.shutterfly_export import (
    placed_photo_order, upload_filename, build_book_manifest,
)

def _doc():
    return {
        "book": {"id": 17, "name": "The South of France", "subtitle": "Provence",
                 "cover_photo_id": 240555, "title_page_photo_id": 240539,
                 "back_cover_photo_id": None},
        "stage_w": 28.0, "stage_h": 11.0,
        "spreads": [
            {"position": 0, "archetype": "title", "bg": "#ffffff",
             "caption": {"text": "We found a castle.", "dark": False},
             "cells": [{"photo_id": 240539, "x": 14.5, "y": 0.5, "w": 13.0, "h": 10.0,
                        "fit": "cover", "crop_cx": 0.53, "crop_cy": 0.55,
                        "crop_zoom": 1.0, "crop_min_w": 0.0, "crop_min_h": 0.0,
                        "align": None, "position": 0}]},
            {"position": 1, "archetype": "matched 2-up", "bg": "#ffffff",
             "caption": None,
             "cells": [
                {"photo_id": 240599, "x": 0.4, "y": 0.4, "w": 8.7, "h": 10.2,
                 "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5, "crop_zoom": 1.0,
                 "crop_min_w": 0.0, "crop_min_h": 0.0, "align": None, "position": 0},
                {"photo_id": 240539, "x": 9.4, "y": 0.4, "w": 8.7, "h": 10.2,
                 "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5, "crop_zoom": 1.0,
                 "crop_min_w": 0.0, "crop_min_h": 0.0, "align": None, "position": 1}]},
        ],
    }

def test_placed_photo_order_is_unique_and_ordered():
    # 240539 appears in both spreads but only once; cover 240555 appended last
    assert placed_photo_order(_doc()) == [240539, 240599, 240555]

def test_placed_photo_order_skips_none_cover():
    doc = _doc()
    doc["book"]["cover_photo_id"] = None
    doc["book"]["title_page_photo_id"] = None
    assert placed_photo_order(doc) == [240539, 240599]

def test_upload_filename_uses_photo_id_and_ext():
    assert upload_filename(240599, "DSC06241.JPG") == "sfly-240599.jpg"
    assert upload_filename(7, "no_ext") == "sfly-7"

def test_build_book_manifest_shape():
    m = build_book_manifest(_doc())
    assert m["book_id"] == 17
    assert m["name"] == "The South of France"
    assert m["stage_w"] == 28.0 and m["stage_h"] == 11.0
    assert m["spread_count"] == 2
    # spreads passed through with cells + captions preserved
    assert m["spreads"][0]["archetype"] == "title"
    assert m["spreads"][0]["caption"] == {"text": "We found a castle.", "dark": False}
    assert m["spreads"][1]["cells"][0]["photo_id"] == 240599
    # every placed photo id appears in the photos map (filenames filled in Task 4/runbook)
    assert set(m["photos"].keys()) == {"240539", "240599", "240555"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_shutterfly_export.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'photosearch.shutterfly_export'`.

- [ ] **Step 3: Write the implementation**

```python
# photosearch/shutterfly_export.py
"""Translate a photo-search book (BookStore.get_book) into artifacts for the
Shutterfly export: an ordered placed-photo list, a deterministic upload-filename
scheme, and a JSON manifest that Phase 2 (native rebuild) will consume.

Pure functions only — no DB or network. The CLI wires these to PhotoDB/BookStore
and google_photos.
"""
from __future__ import annotations

import os
from typing import Callable, Optional


def upload_filename(photo_id: int, orig_filename: Optional[str]) -> str:
    """Deterministic, mapping-friendly Google Photos filename for a photo.

    ``sfly-<photo_id><ext>`` — recovery parses ``^sfly-(\\d+)``.
    """
    ext = os.path.splitext(orig_filename or "")[1].lower()
    return f"sfly-{photo_id}{ext}"


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
    photos map keyed by str(photo_id). Upload/asset filenames are filled later
    by the CLI/runbook (left empty here so this stays pure)."""
    book = doc.get("book", {})
    photos = {str(pid): {"upload_filename": None, "orig_filename": None,
                         "sfly_asset_id": None}
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_shutterfly_export.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add photosearch/shutterfly_export.py tests/test_shutterfly_export.py
git commit -m "feat(shutterfly): book manifest + placed-photo ordering"
```

---

### Task 3: `build_gphotos_records` — resolve photos to upload records

**Files:**
- Modify: `photosearch/shutterfly_export.py`
- Test: `tests/test_shutterfly_export.py`

**Interfaces:**
- Consumes: `placed_photo_order` output; `rows: dict[int, dict]` where each row has `{"filepath","filename","description"}`; a `resolve: Callable[[str], str]` (the CLI passes `PhotoDB.resolve_filepath`).
- Produces: `build_gphotos_records(photo_ids, rows, resolve) -> list[dict]` — one record per id, in `photo_ids` order, shaped for `google_photos.upload_photos`: `{"_resolved_filepath","filename","orig_filename","description"}`. Ids missing from `rows` are skipped.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_shutterfly_export.py
from photosearch.shutterfly_export import build_gphotos_records

def test_build_gphotos_records_maps_and_orders():
    rows = {
        240599: {"filepath": "2026/2026-07-24/DSC1.JPG", "filename": "DSC1.JPG",
                 "description": "castle"},
        240539: {"filepath": "2026/2026-07-24/DSC2.JPG", "filename": "DSC2.JPG",
                 "description": None},
        999999: {"filepath": "x", "filename": "x.jpg", "description": None},  # not requested
    }
    resolve = lambda p: "/photos/" + p
    recs = build_gphotos_records([240539, 240599], rows, resolve)
    assert [r["filename"] for r in recs] == ["sfly-240539.jpg", "sfly-240599.jpg"]
    assert recs[0]["_resolved_filepath"] == "/photos/2026/2026-07-24/DSC2.JPG"
    assert recs[0]["orig_filename"] == "DSC2.JPG"
    assert recs[1]["description"] == "castle"

def test_build_gphotos_records_skips_missing_rows():
    recs = build_gphotos_records([1, 2], {1: {"filepath": "a", "filename": "a.jpg",
                                              "description": None}}, lambda p: p)
    assert [r["filename"] for r in recs] == ["sfly-1.jpg"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_shutterfly_export.py -k gphotos_records -v`
Expected: FAIL — `ImportError: cannot import name 'build_gphotos_records'`.

- [ ] **Step 3: Write the implementation**

```python
# append to photosearch/shutterfly_export.py
def build_gphotos_records(
    photo_ids: list[int],
    rows: dict[int, dict],
    resolve: Callable[[str], str],
) -> list[dict]:
    """Build google_photos.upload_photos records with deterministic filenames.

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
            "filename": upload_filename(int(pid), orig),
            "orig_filename": orig,
            "description": row.get("description"),
        })
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_shutterfly_export.py -v`
Expected: PASS (6 tests total).

- [ ] **Step 5: Commit**

```bash
git add photosearch/shutterfly_export.py tests/test_shutterfly_export.py
git commit -m "feat(shutterfly): resolve placed photos to GP upload records"
```

---

### Task 4: CLI `shutterfly-manifest` command

**Files:**
- Modify: `cli.py`
- Test: `tests/test_cli_shutterfly.py` (Create)

**Interfaces:**
- Consumes: `BookStore.get_book`, `PhotoDB`, `shutterfly_export.build_book_manifest`, `upload_filename`.
- Produces: a CLI command `shutterfly-manifest --book <id> [--out path] [--db ...] [--books-db ...]` that writes the manifest JSON (with `photos[*].upload_filename` and `orig_filename` filled in from `PhotoDB`) to `--out` (default `shutterfly-manifest-<book>.json`) and prints the path.

- [ ] **Step 1: Write the failing test** (drives the manifest-enrichment helper `enrich_manifest_filenames`, kept in the module so it's unit-testable)

```python
# tests/test_cli_shutterfly.py
from photosearch.shutterfly_export import enrich_manifest_filenames

def test_enrich_manifest_filenames_fills_upload_and_orig():
    manifest = {"photos": {"240599": {"upload_filename": None, "orig_filename": None,
                                      "sfly_asset_id": None}}}
    rows = {240599: {"filepath": "p", "filename": "DSC06241.JPG", "description": None}}
    out = enrich_manifest_filenames(manifest, rows)
    assert out["photos"]["240599"]["upload_filename"] == "sfly-240599.jpg"
    assert out["photos"]["240599"]["orig_filename"] == "DSC06241.JPG"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_shutterfly.py -v`
Expected: FAIL — `ImportError: cannot import name 'enrich_manifest_filenames'`.

- [ ] **Step 3: Add the helper to the module**

```python
# append to photosearch/shutterfly_export.py
def enrich_manifest_filenames(manifest: dict, rows: dict[int, dict]) -> dict:
    """Fill photos[*].upload_filename / orig_filename from photo rows in place."""
    for pid_str, rec in manifest.get("photos", {}).items():
        row = rows.get(int(pid_str))
        if row:
            rec["orig_filename"] = row.get("filename")
            rec["upload_filename"] = upload_filename(int(pid_str), row.get("filename"))
    return manifest
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_shutterfly.py -v`
Expected: PASS.

- [ ] **Step 5: Add the CLI command** (append to `cli.py`; imports at top of file)

```python
# cli.py — near other imports
import json as _json
from pathlib import Path as _Path
from photosearch.book import BookStore
from photosearch.db import PhotoDB
from photosearch import shutterfly_export as sfx


def _books_db_default(db: str) -> str:
    import os
    return (os.environ.get("PHOTOSEARCH_BOOKS_DB")
            or str(_Path(db).resolve().parent / "photobooks.db.local"))


def _photo_rows(pdb: PhotoDB, ids: list[int]) -> dict:
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    return {r["id"]: {"filepath": r["filepath"], "filename": r["filename"],
                      "description": r["description"]}
            for r in pdb.conn.execute(
                f"SELECT id, filepath, filename, description FROM photos "
                f"WHERE id IN ({ph})", ids).fetchall()}


@cli.command("shutterfly-manifest")
@click.option("--book", "book_id", type=int, required=True, help="Book id")
@click.option("--out", default=None, help="Output JSON path")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
@click.option("--books-db", "books_db", default=None, envvar="PHOTOSEARCH_BOOKS_DB")
def shutterfly_manifest(book_id, out, db, books_db):
    """Write the Shutterfly build manifest for a book."""
    books_db = books_db or _books_db_default(db)
    with BookStore(books_db) as bs, PhotoDB(db) as pdb:
        doc = bs.get_book(book_id)
        if not doc:
            raise click.ClickException(f"Book {book_id} not found")
        manifest = sfx.build_book_manifest(doc)
        ids = [int(p) for p in manifest["photos"].keys()]
        rows = _photo_rows(pdb, ids)
        sfx.enrich_manifest_filenames(manifest, rows)
    out = out or f"shutterfly-manifest-{book_id}.json"
    _Path(out).write_text(_json.dumps(manifest, indent=2))
    click.echo(f"Wrote {out} ({manifest['spread_count']} spreads, "
               f"{len(manifest['photos'])} photos)")
```

- [ ] **Step 6: Smoke-test the command against a real book** (run on the NAS where the books DB lives)

Run: `docker compose -f docker-compose.nas.yml run --rm photosearch shutterfly-manifest --book 17 --out /tmp/m17.json`
Expected: prints `Wrote /tmp/m17.json (57 spreads, N photos)`; the JSON has a `photos` map with `sfly-<id>.jpg` upload filenames.

- [ ] **Step 7: Commit**

```bash
git add cli.py photosearch/shutterfly_export.py tests/test_cli_shutterfly.py
git commit -m "feat(shutterfly): shutterfly-manifest CLI command"
```

---

### Task 5: CLI `shutterfly-gphotos-push` command

**Files:**
- Modify: `cli.py`
- Test: `tests/test_cli_shutterfly.py`

**Interfaces:**
- Consumes: `shutterfly_export.placed_photo_order` + `build_gphotos_records`, `google_photos.create_album` + `upload_photos`, `PhotoDB.resolve_filepath`.
- Produces: `shutterfly-gphotos-push --book <id> [--album-title T] [--dry-run]` that creates a GP album and uploads the book's placed photos with deterministic filenames; prints per-photo results summary.

- [ ] **Step 1: Write the failing test** (mock `google_photos` so no network)

```python
# append to tests/test_cli_shutterfly.py
from click.testing import CliRunner
from unittest import mock
import cli as cli_mod

def test_gphotos_push_dry_run_lists_records(monkeypatch, tmp_path):
    # Fake BookStore.get_book + PhotoDB rows via monkeypatched helpers
    doc = {"book": {"id": 3, "name": "T", "subtitle": None, "cover_photo_id": None,
                    "title_page_photo_id": None, "back_cover_photo_id": None},
           "stage_w": 28, "stage_h": 11,
           "spreads": [{"position": 0, "archetype": "matched 2-up", "bg": "#fff",
                        "caption": None,
                        "cells": [{"photo_id": 5, "x": 0, "y": 0, "w": 1, "h": 1,
                                   "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5,
                                   "crop_zoom": 1, "crop_min_w": 0, "crop_min_h": 0,
                                   "align": None, "position": 0}]}]}
    monkeypatch.setattr(cli_mod, "_load_book_doc", lambda book_id, db, books_db: doc)
    monkeypatch.setattr(cli_mod, "_load_photo_rows_resolved",
                        lambda ids, db: ({5: {"filepath": "a", "filename": "A.JPG",
                                              "description": None}}, lambda p: "/photos/" + p))
    r = CliRunner().invoke(cli_mod.cli, ["shutterfly-gphotos-push", "--book", "3",
                                         "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "sfly-5.jpg" in r.output
    assert "1 photo" in r.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_shutterfly.py -k gphotos_push -v`
Expected: FAIL — command not registered / helpers missing.

- [ ] **Step 3: Add the command + testable seams**

```python
# cli.py — add helper seams the test monkeypatches, then the command
def _load_book_doc(book_id: int, db: str, books_db: str) -> dict | None:
    books_db = books_db or _books_db_default(db)
    with BookStore(books_db) as bs:
        return bs.get_book(book_id)


def _load_photo_rows_resolved(ids: list[int], db: str):
    with PhotoDB(db) as pdb:
        return _photo_rows(pdb, ids), pdb.resolve_filepath


@cli.command("shutterfly-gphotos-push")
@click.option("--book", "book_id", type=int, required=True)
@click.option("--album-title", "album_title", default=None,
              help="GP album title (default: '2026 Summer — <book name>')")
@click.option("--dry-run", is_flag=True, help="List records without uploading")
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
@click.option("--books-db", "books_db", default=None, envvar="PHOTOSEARCH_BOOKS_DB")
def shutterfly_gphotos_push(book_id, album_title, dry_run, db, books_db):
    """Upload a book's placed photos to a per-book Google Photos album."""
    from photosearch import google_photos as gp
    doc = _load_book_doc(book_id, db, books_db)
    if not doc:
        raise click.ClickException(f"Book {book_id} not found")
    ids = sfx.placed_photo_order(doc)
    rows, resolve = _load_photo_rows_resolved(ids, db)
    records = sfx.build_gphotos_records(ids, rows, resolve)
    title = album_title or f"2026 Summer — {doc['book'].get('name')}"
    click.echo(f"{len(records)} photo(s) -> album {title!r}")
    for r in records:
        click.echo(f"  {r['filename']}  <-  {r['orig_filename']}")
    if dry_run:
        return
    album_id = gp.create_album(db, title)
    results = gp.upload_photos(db, records, album_id=album_id)
    ok = sum(1 for x in results if x.get("status") == "uploaded")
    click.echo(f"Uploaded {ok}/{len(results)} to album {album_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli_shutterfly.py -v`
Expected: PASS.

- [ ] **Step 5: Dry-run against a real book** (NAS)

Run: `docker compose -f docker-compose.nas.yml run --rm photosearch shutterfly-gphotos-push --book 17 --dry-run`
Expected: lists `sfly-<id>.jpg  <-  <original>` lines and the photo count. No album created.

- [ ] **Step 6: Commit**

```bash
git add cli.py tests/test_cli_shutterfly.py
git commit -m "feat(shutterfly): shutterfly-gphotos-push CLI command"
```

---

### Task 6: Deploy + runbook (execute for the three books)

**Files:** none (operational). Depends on Task 1's decision.

- [ ] **Step 1: Rebuild + restart the service** (Python-only change, ~10s layer):

```bash
docker compose -f docker-compose.nas.yml build photosearch
docker compose -f docker-compose.nas.yml up -d photosearch
```

- [ ] **Step 2: For each book (12, 14, 17):** write the manifest and push to Google Photos:

```bash
DC="docker compose -f docker-compose.nas.yml run --rm photosearch"
for B in 12 14 17; do
  $DC shutterfly-manifest --book $B --out /data/shutterfly-manifest-$B.json
  $DC shutterfly-gphotos-push --book $B
done
```

Expected: three GP albums named `2026 Summer — <name>`, each with the book's placed photos as `sfly-<id>.jpg`.

- [ ] **Step 3: Import each album into Shutterfly** (browser, one project per book): create/open the 11×14 layflat project → Add More Photos → Google Photos → import that book's album into the tray.

- [ ] **Step 4: Recover the mapping** (Chrome MCP `javascript_tool` in the editor): read imported assets, match `^sfly-(\d+)` → `photo_id`, and write `sfly_asset_id` back into each book's manifest `photos` map. This mapping is the hand-off to the Phase 2 plan.

- [ ] **Step 5: Verify** each book's tray has the expected photo count (matches manifest `len(photos)`), and spot-check that 3–4 known photos imported with recoverable ids.

---

## Spike S1 findings

_(Fill in during Task 1.)_

- Which filename Shutterfly keeps on GP import (`simpleMediaItem.fileName` vs upload-header basename):
- Decision (scheme works as-is / needs `_upload_raw_bytes` override):
- Scratch GP album + test assets deleted: yes/no

---

## Self-Review

**Spec coverage (against `2026-08-09-shutterfly-export-design.md`, Phase 1 + Spike S1):**
- Build spec/manifest from `/api/books/{id}` → Task 2 (`build_book_manifest`). ✓
- Placed photos + cover/title/back ordering → Task 2 (`placed_photo_order`). ✓
- Per-book GP album, deterministic `filename=<photo_id>` → Tasks 3+5, `upload_filename`. ✓
- Reuse `create_album` / `upload_photos` → Task 5. ✓
- Shutterfly GP import + mapping recovery → Task 6 (runbook). ✓
- Spike S1 (mapping fidelity, throwaway) → Task 1. ✓
- No hardcoded tailnet IP; `PHOTOSEARCH_NAS_URL`/placeholders; books-db sidecar → Global Constraints, CLI defaults. ✓
- Phase 2 (native rebuild) + Spike S2 → intentionally out of scope; separate plan after S2. ✓

**Placeholder scan:** No "TBD/TODO/handle errors" in code steps; every code step has real content. The "Spike S1 findings" and dry-run photo paths (`/photos/<any>`) are intentional operator inputs, not code placeholders. ✓

**Type consistency:** `upload_filename(photo_id, orig_filename)`, `placed_photo_order(doc)`, `build_book_manifest(doc)`, `build_gphotos_records(photo_ids, rows, resolve)`, `enrich_manifest_filenames(manifest, rows)` — names/params identical across the tasks that define and call them. Records use the `_resolved_filepath`/`filename` keys that `google_photos.upload_photos` reads. ✓

## Notes for the Phase 2 plan (not this plan)

- Gated on **Spike S2** (capture Shutterfly save-payload schema on a throwaway book).
- Consumes each book's enriched manifest (with `sfly_asset_id`).
- Maps archetypes → Shutterfly layouts; translates crops via `book_export.plan_cell`.
- Must apply the **57 → 56 spread** merge per book before layout.
- Verifies via `GET /api/books/{id}/render/{spread_id}` vs the Shutterfly spread.

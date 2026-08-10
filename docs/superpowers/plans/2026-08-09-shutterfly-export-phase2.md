# Shutterfly Export — Phase 2 (Native layout rebuild) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. NOTE: Tasks 1, 3, 4 run in the logged-in Shutterfly browser (throwaway projects for spikes; real projects for the final run) and need the user's go-ahead before creating/deleting projects or editing real books. Only Task 2 is pure repo code.

**Goal:** Rebuild each 2026 Summer book's designed spreads natively in its Shutterfly 11×14 layflat project — real photos in cells with faithful-ish crops + caption text boxes — so Matt only has to polish.

**Architecture:** A pure-Python translator turns each book (`/api/books/{id}` + `book_export.plan_cell`) into a Shutterfly-agnostic **layout spec** (ordered surfaces → cells with native-inch geometry + crop + captions). A **browser driver** run in the logged-in editor consumes that spec and drives `builderApp`'s in-page data layer to place photos (matched by original filename → `mediaId`), set crops, and add captions, then saves. We drive `builderApp` rather than PUT a hand-built doc (Spike S2 finding: the Save PUT body is ~20 bytes; content syncs incrementally).

**Tech Stack:** Python 3.11 + Click (translator, in the repo), JavaScript run via the Chrome MCP `javascript_tool` (browser driver), Shutterfly "sfly3/Aurora" `builderApp` (Backbone) + `photos3` album API, pytest.

## Global Constraints

- **Schema reference:** `docs/superpowers/specs/2026-08-09-shutterfly-project-schema.md` (S2 capture) is the source of truth for the target doc shape (`surfaceCategories→surfaces→surfaceData.layeredItems`; photo `container{x,y,w,h}`+`content.userData{w,h,x,y}` crop + `assetId`/`mediaId`; text `fullText`+style; `photoWell`; `pageDetails{dpi:300}`).
- **Drive `builderApp`, do NOT PUT a hand-built doc.** Content syncs incrementally; use the app's own data-layer methods, then save.
- **Photo mapping = original filename → mediaId** via the per-book `photos3` album API, then `mediaId → photoWell → assetId`. Original filenames are unique per book (Phase 1 Spike S1).
- **Coordinates:** book stage is **28×11 inches** (gutter at x=14); Shutterfly surfaces are **px at `pageDetails.dpi` (300)**. The translator emits **inches**; the driver scales inches→px using the dpi and origin confirmed in Task 1.
- **56-spread layflat ceiling** — books are 57; the one-spread merge is handled in the book itself before rebuild (see book 12's `collage 3+3` merge; do the same for 14 and 17), out of scope for this plan's code.
- **Fidelity:** faithful-ish, then Matt polishes. Best-effort crop; exact pixel match not required.
- **Spikes use throwaway projects only** (copy a book → work → delete). Never mutate a real book to investigate.
- **No hardcoded tailnet IP / NAS hostname** in committed files (use `PHOTOSEARCH_NAS_URL` / placeholders).

---

## File Structure

- **Modify** `photosearch/shutterfly_export.py` — add `build_layout_spec(doc, dpi=300) -> dict` (pure; reuses `book_export.plan_cell`).
- **Modify** `tests/test_shutterfly_export.py` — tests for `build_layout_spec`.
- **Modify** `cli.py` — add `shutterfly-layout-spec --book <id>` (writes the layout-spec JSON, like `shutterfly-manifest`).
- **Create** `shutterfly/driver_primitives.js` — the verified `builderApp` primitive functions (Task 1's deliverable).
- **Create** `shutterfly/rebuild_driver.js` — the orchestrator that consumes a layout-spec + album map and calls the primitives (Task 3).
- **Create** `docs/superpowers/specs/2026-08-09-shutterfly-driver-notes.md` — Task 1's findings (crop formula, layout method, coord origin).

---

### Task 1: Spike S3 — `builderApp` write primitives + crop formula (throwaway; gating)

Discover and **implement** the driver primitives by experimenting in a throwaway book's editor console. The deliverable is working, named JS functions + a findings note — later tasks compose these by name.

**Files:**
- Create: `shutterfly/driver_primitives.js`
- Create: `docs/superpowers/specs/2026-08-09-shutterfly-driver-notes.md`

**Deliverable primitives** (each must be verified live before it counts as done):
- `sflyAlbumMap(albumId) -> Promise<{[origFilename]: mediaId}>` — fetch the per-book `photos3` album via the page's auth; return original-filename → mediaId. (Resolves the S2 "photoWell has no original filename" gap.)
- `sflyGetSurfaces() -> surfaces[]` — the page-spread surfaces (not cover), in order.
- `sflyEnsureSpreadCount(n)` — add/remove page spreads so there are `n`.
- `sflyApplyLayout(surfaceIdx, nCells)` — apply a Shutterfly layout with `nCells` photo cells to the surface (nearest match to the archetype's cell count).
- `sflyPlacePhoto(surfaceIdx, cellIdx, mediaId)` — put the library photo into the cell.
- `sflySetCrop(surfaceIdx, cellIdx, crop)` — apply crop, where `crop` is `{Pw,Ph,left,top}` (percent-of-cell, the `plan_cell` output); this function converts to Shutterfly's `content.userData{w,h,x,y}` using the formula derived here.
- `sflyAddCaption(surfaceIdx, {text, xIn, yIn, wIn, hIn, dark})` — add a text box with the caption.
- `sflySave() -> Promise` — trigger the save and resolve when the project PUT completes.

- [ ] **Step 1: Make a throwaway** — copy a real book (Copy → "ZZ SCRATCH phase2"), open its editor. (Ask the user before creating/deleting projects.)

- [ ] **Step 2: Hook + explore.** Install the fetch/XHR capture (from S2), then in the console probe `window.builderApp` for the data-layer methods behind: add photo to cell, crop, add text, apply layout, save. Cross-reference `builderApp.DataManagerEvents` (Project/Page/Picture) and `builderApp.LayoutHelper` (`autoFill`, `applyLayout`, `swapPhotoItems`). Record which methods/events actually mutate the model.

- [ ] **Step 3: Derive the crop formula.** Place one photo, set a known crop via the UI, `sflySave()`, and read back the saved element's `container{w,h}` and `content.userData{w,h,x,y}`. Compute the mapping from `plan_cell`'s `{Pw,Ph,left,top}` (% of cell) → `userData{w,h,x,y}` (px). Write it in the notes doc and implement it inside `sflySetCrop`.

- [ ] **Step 4: Resolve the album map.** From the editor page, fetch the book's `photos3` album (using the page's auth) and confirm you can build `{origFilename: mediaId}` and that those `mediaId`s appear in `photoWell`. Implement `sflyAlbumMap`.

- [ ] **Step 5: End-to-end smoke on ONE spread.** Using the primitives, apply a 2-cell layout to one page spread, place two known photos by filename, set a crop on each, add a caption, `sflySave()`, and confirm via the capture that the saved doc has the expected `layeredItems`. Screenshot it.

- [ ] **Step 6: Write `driver_primitives.js` + notes; delete the throwaway.** Commit both files.

```bash
git add shutterfly/driver_primitives.js docs/superpowers/specs/2026-08-09-shutterfly-driver-notes.md
git commit -m "spike(shutterfly): builderApp driver primitives + crop formula (S3)"
```

**BLOCKED criteria:** if `builderApp` has no scriptable method to place a photo into a specific cell or set an arbitrary crop (only UI-drag), stop and report — Phase 2 would degrade to pixel automation, which the design rejected.

---

### Task 2: `build_layout_spec` translator (pure Python, TDD)

**Files:**
- Modify: `photosearch/shutterfly_export.py`
- Test: `tests/test_shutterfly_export.py`

**Interfaces:**
- Consumes: a `doc` from `BookStore.get_book()` (same shape as Phase 1) and `photo_ar: dict[int, float]` mapping photo_id → image aspect (w/h), used for crop math.
- Produces: `build_layout_spec(doc, photo_ar) -> dict`:
  ```
  { "book_id", "name", "stage_w", "stage_h",
    "surfaces": [ { "position", "archetype",
                    "caption": {"text","dark"} | None,
                    "cells": [ { "orig_photo_id",
                                 "x","y","w","h",                 # inches on the 28x11 stage
                                 "crop": {"Pw","Ph","left","top"} # percent-of-cell (plan_cell output)
                               } ] } ] }
  ```
  Cell crop is computed with `photosearch.book_export.plan_cell(cell, an)` where `an = photo_ar[photo_id]`. Cells with no `photo_id` or no known aspect are emitted with `crop: null`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_shutterfly_export.py
from photosearch.shutterfly_export import build_layout_spec

def test_build_layout_spec_geometry_and_crop():
    doc = {
        "book": {"id": 17, "name": "SoF"},
        "stage_w": 28.0, "stage_h": 11.0,
        "spreads": [
            {"position": 1, "archetype": "matched 2-up",
             "caption": {"text": "Lake Como", "dark": True},
             "cells": [
                {"photo_id": 240599, "x": 0.4, "y": 0.4, "w": 8.7, "h": 10.2,
                 "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5, "crop_zoom": 1.0,
                 "crop_min_w": 0.0, "crop_min_h": 0.0, "align": None, "position": 0},
             ]},
        ],
    }
    spec = build_layout_spec(doc, {240599: 1.5})   # 3:2 photo
    assert spec["book_id"] == 17 and spec["stage_w"] == 28.0
    s = spec["surfaces"][0]
    assert s["archetype"] == "matched 2-up"
    assert s["caption"] == {"text": "Lake Como", "dark": True}
    c = s["cells"][0]
    assert c["orig_photo_id"] == 240599
    assert (c["x"], c["y"], c["w"], c["h"]) == (0.4, 0.4, 8.7, 10.2)
    # cover crop of a 1.5-aspect photo in a ~0.85-aspect cell => image wider than cell
    assert c["crop"]["Pw"] >= 100.0 and c["crop"]["Ph"] >= 100.0
    assert 0 <= c["crop"]["left"] <= 100 and 0 <= c["crop"]["top"] <= 100

def test_build_layout_spec_cell_without_aspect_gets_null_crop():
    doc = {"book": {"id": 1, "name": "x"}, "stage_w": 28, "stage_h": 11,
           "spreads": [{"position": 1, "archetype": "single", "caption": None,
                        "cells": [{"photo_id": 9, "x": 0, "y": 0, "w": 28, "h": 11,
                                   "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5,
                                   "crop_zoom": 1.0, "crop_min_w": 0.0, "crop_min_h": 0.0,
                                   "align": None, "position": 0}]}]}
    spec = build_layout_spec(doc, {})   # no aspect known for photo 9
    assert spec["surfaces"][0]["cells"][0]["crop"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python -m pytest tests/test_shutterfly_export.py -k layout_spec -v`
Expected: FAIL — `ImportError: cannot import name 'build_layout_spec'`.

- [ ] **Step 3: Implement**

```python
# append to photosearch/shutterfly_export.py
from photosearch import book_export as _book_export


def build_layout_spec(doc: dict, photo_ar: dict) -> dict:
    """Shutterfly-agnostic layout spec: per-surface cells in native inches +
    plan_cell crop (percent-of-cell) + captions. The browser driver scales
    inches->px and applies the Shutterfly crop formula.

    ``photo_ar`` maps photo_id -> image aspect (w/h) for crop math.
    """
    book = doc.get("book", {})
    surfaces = []
    for sp in doc.get("spreads", []):
        cells = []
        for cell in sp.get("cells", []):
            pid = cell.get("photo_id")
            an = photo_ar.get(int(pid)) if pid is not None else None
            crop = _book_export.plan_cell(cell, an) if (an and an > 0) else None
            cells.append({
                "orig_photo_id": pid,
                "x": cell.get("x"), "y": cell.get("y"),
                "w": cell.get("w"), "h": cell.get("h"),
                "crop": crop,
            })
        surfaces.append({
            "position": sp.get("position"),
            "archetype": sp.get("archetype"),
            "caption": sp.get("caption"),
            "cells": cells,
        })
    return {
        "book_id": book.get("id"),
        "name": book.get("name"),
        "stage_w": doc.get("stage_w"),
        "stage_h": doc.get("stage_h"),
        "surfaces": surfaces,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python -m pytest tests/test_shutterfly_export.py -v`
Expected: PASS (all, including the two new).

- [ ] **Step 5: Commit**

```bash
git add photosearch/shutterfly_export.py tests/test_shutterfly_export.py
git commit -m "feat(shutterfly): build_layout_spec translator (inches + plan_cell crop)"
```

---

### Task 3: `shutterfly-layout-spec` CLI + rebuild orchestrator

**Files:**
- Modify: `cli.py` (add `shutterfly-layout-spec`)
- Create: `shutterfly/rebuild_driver.js`
- Test: `tests/test_cli_shutterfly.py`

**Part A — CLI (mirrors `shutterfly-manifest`, testable):**

**Interfaces:** `shutterfly-layout-spec --book <id> [--out path] [--db][--books-db]` — loads the book, gathers `photo_ar` from `PhotoDB` (`image_width`/`image_height`), calls `build_layout_spec`, writes JSON.

- [ ] **Step 1: Write the failing test** (drives a pure helper `photo_aspect_map`)

```python
# append to tests/test_cli_shutterfly.py
from photosearch.shutterfly_export import photo_aspect_map

def test_photo_aspect_map_from_rows():
    rows = {5: {"image_width": 3000, "image_height": 2000},
            6: {"image_width": 0, "image_height": 0}}
    m = photo_aspect_map(rows)
    assert m[5] == 1.5
    assert 6 not in m   # zero/invalid dims skipped
```

- [ ] **Step 2: Run to verify it fails** — `venv/bin/python -m pytest tests/test_cli_shutterfly.py -k aspect -v` → ImportError.

- [ ] **Step 3: Implement the helper + CLI**

```python
# photosearch/shutterfly_export.py
def photo_aspect_map(rows: dict) -> dict:
    """photo_id -> aspect (w/h) from rows with image_width/image_height; skip invalid."""
    out = {}
    for pid, r in rows.items():
        w, h = r.get("image_width"), r.get("image_height")
        if w and h:
            out[int(pid)] = w / h
    return out
```

```python
# cli.py — new command (reuses BookStore/PhotoDB/sfx helpers from Phase 1)
@cli.command("shutterfly-layout-spec")
@click.option("--book", "book_id", type=int, required=True)
@click.option("--out", default=None)
@click.option("--db", default="photo_index.db", envvar="PHOTOSEARCH_DB")
@click.option("--books-db", "books_db", default=None, envvar="PHOTOSEARCH_BOOKS_DB")
def shutterfly_layout_spec(book_id, out, db, books_db):
    """Write the Shutterfly layout spec (surfaces/cells/crops/captions) for a book."""
    doc = _load_book_doc(book_id, db, books_db)
    if not doc:
        raise click.ClickException(f"Book {book_id} not found")
    ids = sfx.placed_photo_order(doc)
    with PhotoDB(db) as pdb:
        ph = ",".join("?" * len(ids)) if ids else ""
        rows = {r["id"]: {"image_width": r["image_width"], "image_height": r["image_height"]}
                for r in (pdb.conn.execute(
                    f"SELECT id, image_width, image_height FROM photos WHERE id IN ({ph})", ids
                ).fetchall() if ids else [])}
    spec = sfx.build_layout_spec(doc, sfx.photo_aspect_map(rows))
    out = out or f"shutterfly-layout-{book_id}.json"
    _Path(out).write_text(_json.dumps(spec, indent=2))
    click.echo(f"Wrote {out} ({len(spec['surfaces'])} surfaces)")
```

- [ ] **Step 4: Run to verify it passes** — `venv/bin/python -m pytest tests/test_cli_shutterfly.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add photosearch/shutterfly_export.py cli.py tests/test_cli_shutterfly.py
git commit -m "feat(shutterfly): shutterfly-layout-spec CLI + photo_aspect_map"
```

**Part B — Orchestrator (`shutterfly/rebuild_driver.js`), composes Task 1 primitives:**

- [ ] **Step 6: Write the orchestrator.** A function `rebuildBook(spec, albumId)` that:
  1. `const map = await sflyAlbumMap(albumId);` and `sflyEnsureSpreadCount(spec.surfaces.length)`.
  2. For each surface `i`: `sflyApplyLayout(i, surface.cells.length)`; for each cell `j`: resolve `mediaId = map[origFilenameOf(cell.orig_photo_id)]`, `sflyPlacePhoto(i,j,mediaId)`, and if `cell.crop` then `sflySetCrop(i,j,cell.crop)`; if `surface.caption` then `sflyAddCaption(i, {text: surface.caption.text, xIn:1, yIn: spec.stage_h-2, wIn: spec.stage_w-2, hIn:1.2, dark: surface.caption.dark})`.
  3. `await sflySave();` and `log` per-surface progress.

  `origFilenameOf` comes from the Phase-1 **manifest** (`photos[photo_id].orig_filename`), passed alongside the spec. The orchestrator references only primitives defined in `driver_primitives.js` (Task 1).

- [ ] **Step 7: Commit** the orchestrator.

```bash
git add shutterfly/rebuild_driver.js
git commit -m "feat(shutterfly): rebuild orchestrator over driver primitives"
```

---

### Task 4: Run + verify on book 17, then 12 & 14 (browser; user go-ahead)

**Files:** none (operational).

- [ ] **Step 1: Generate inputs** (replica/NAS as in Phase 1): `shutterfly-layout-spec --book 17` and reuse the book-17 `manifest.json` (for `orig_filename`).
- [ ] **Step 2: Open book 17's real Shutterfly layflat project** (create it if not yet started; import already done in Phase 1). Confirm the book is at the 56-spread ceiling (apply the merge first if needed).
- [ ] **Step 3: Inject** `driver_primitives.js` then `rebuild_driver.js` via the Chrome MCP `javascript_tool`; run `rebuildBook(spec17, albumId17)`.
- [ ] **Step 4: Verify** spread-by-spread against `GET /api/books/17/render/{spread_id}` (the photo-search proof) + editor screenshots. Note mismatches; iterate on the primitives/translator.
- [ ] **Step 5: Hand to Matt to polish**, then repeat for books 14 and 12.

Because Shutterfly saves incrementally and this touches a real book, run one spread first, verify, then the rest; keep the book's own undo as the safety net.

---

## Self-Review

**Spec coverage (against the design's Phase 2 + the S2 schema doc):**
- Native rebuild via `builderApp` (not raw PUT) → Task 1 primitives + Task 3 orchestrator. ✓
- Photo cells with faithful-ish crop → `build_layout_spec` (plan_cell) + `sflySetCrop` (formula from Task 1). ✓
- Captions → `sflyAddCaption` + spec `caption`. ✓
- Filename→mediaId mapping → `sflyAlbumMap` (Task 1) + manifest `orig_filename`. ✓
- 56-spread ceiling / 57→56 merge → Global Constraints + Task 4 Step 2 (book-side, out of code scope). ✓
- Verify via render endpoint → Task 4 Step 4. ✓

**Placeholder scan:** Task 2 and Task 3A contain complete code. Task 1 is a spike whose *deliverable is* the named primitives that Tasks 3B/4 consume — the primitives are defined there, so references to them are not placeholders. Task 3B/4 describe composition/operation, which is all that can be specified until the spike fixes the primitive internals; no fabricated `builderApp` method names appear in committed code.

**Type consistency:** `build_layout_spec(doc, photo_ar)`, `photo_aspect_map(rows)`, and the primitive signatures (`sflyAlbumMap`, `sflyApplyLayout`, `sflyPlacePhoto`, `sflySetCrop(…, {Pw,Ph,left,top})`, `sflyAddCaption`, `sflySave`) are used identically across the tasks that define and call them. `crop` is `plan_cell`'s `{Pw,Ph,left,top}` throughout.

## Dependency note

Task 2 (pure code) can be implemented immediately. Tasks 1, 3B, and 4 are gated on the Task 1 spike and require the logged-in browser + user go-ahead for throwaway/real project changes. Recommended order: **Task 2 → Task 1 → Task 3 → Task 4.**

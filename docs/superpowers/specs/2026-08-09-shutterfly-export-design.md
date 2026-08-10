# Shutterfly export — design

**Date:** 2026-08-09
**Status:** Draft for review
**Repo:** local-photo-search

## Goal

Automate getting the three finished **2026 Summer** photobooks out of the
photo-search book builder and into **Shutterfly**, as far as automation allows.
Two outcomes are wanted:

1. **Source photos into Shutterfly** — each book's photos organised and available
   in the Shutterfly photo library.
2. **Native layout reproduction** — recreate each designed spread in Shutterfly's
   book designer with individual photos + real caption text boxes, editable in
   Shutterfly (not flattened images), *faithful-ish* to the original design with
   final crop/nudge polish done by hand.

The three books (photo-search book ids; PDFs live in the Summer 2026 photos
folder):

| Book | photo-search id | Spreads |
|---|---|---|
| Italy / Lake Como / the Dolomites | 12 | 57 |
| The Alps & Annecy | 14 | 57 |
| The South of France | 17 | 57 |

Target Shutterfly product: **11×14 Deluxe Layflat Photo Book** (spreads lie flat,
no gutter loss — matches the 28×11 designed spread).

## Non-goals

- Pixel-perfect reproduction. Fidelity target is "faithful-ish, then polish by
  hand" (confirmed with Matt). Exact per-cell crop is best-effort.
- Flattened full-page-image reproduction (explicitly rejected in favour of native
  rebuild).
- Ordering a book / any purchase. This project stops at an editable Shutterfly
  project ready for Matt to review and buy.
- Any change to the books themselves in photo-search.

## Feasibility findings (from a browser spike on 2026-08-09)

The whole plan hinged on whether Shutterfly's designer is scriptable. It is.

- **Builder = Shutterfly "sfly3 / Aurora" Backbone SPA.** Global `builderApp`
  (with `window.Backbone`) exposes models `Project / Page / Surface / Picture /
  Asset / CanvasLayout` and helpers `autoFill`, `applyLayout`, `swapPhotoItems`,
  plus an `AutoCreateManager`. The book can be driven in-page via JS rather than
  pixel-dragging.
- **Project REST API:** `projects-api3.shutterfly.com/projects/v2/project/{id}`
  (GET/PUT the whole book document). Apigee-fronted — requires an `X-API-Key`
  header **and** an OAuth bearer token; both exist only inside the logged-in
  browser session (a plain cookie-only `fetch` returns 401). The exact save
  payload schema is **not yet captured** (see Spike S2).
- **Catalog/design API:** `catalog-api3.shutterfly.com/catalogdesign/v1/...`
  (product design specs, page layouts, design assets).
- **Google Photos is a first-class import source** in the editor's "Add More
  Photos" menu (verified on screen: Upload / Shutterfly Photos / Amazon Photos /
  **Google Photos** / Facebook / Art Library). This is the upload path (below).
- **Known constraint:** the layflat book has a **56-spread ceiling**; our books
  are 57 spreads, so each needs one spread merged down before/as part of the
  rebuild (book 12's planned merge of spreads 54+55 into a `collage 3+3`
  six-cell spread is the existing example).

## Source data model (already in photo-search)

`GET /api/books/{id}` returns the book + `spreads[]`, each with `cells[]`. The
stage is **28 × 11 inches**, gutter at x = 14, each page 14 × 11 trim. Cells
carry `x, y, w, h, fit, crop_cx, crop_cy, crop_zoom, crop_min_w, crop_min_h,
align`, plus per-spread `caption {text, dark}`, `archetype`, and `bg`. Full-res
originals: `GET /api/photos/{id}/full`. Per-spread proof render (for visual
verification): `GET /api/books/{id}/render/{spread_id}` → PNG.

Reusable code already in the repo:
- `photosearch/book_export.py` — `plan_cell()` codifies the exact crop math
  (zoom, min-visible, cover/contain) used to render the PDFs. **Reuse this** to
  compute the visible region when translating crops to Shutterfly.
- `photosearch/google_photos.py` — `create_album(db_path, title)` and
  `upload_photos(...)`. `upload_photos` accepts a per-photo `filename` (sent as
  `X-Goog-Upload-File-Name`), so we can **name each upload deterministically**
  (e.g. by `photo_id`) to make the Shutterfly-side mapping collision-free.

## Architecture

The runner is **Claude Code on the Mac**, because it is the only place that can
reach *both* sides:

- **photo-search** lives on the tailnet over http; a Shutterfly *https* page
  cannot fetch it (mixed-content block). The Mac reaches it via its HTTP API.
- **Shutterfly** needs in-browser auth (`X-API-Key` + bearer). The Mac drives the
  logged-in browser via the Chrome MCP (`builderApp` scripting + the editor UI).

Split of responsibilities:

```
photo-search (HTTP API, tailnet)          Shutterfly (logged-in browser)
  book JSON, /full, /render                 projects-api3, builderApp, GP import
        \                                          /
         \______  Claude Code on Mac  __________ /
                  - new: shutterfly_export module + CLI
                  - browser driver (JS run via Chrome MCP)
```

**Base URL:** never hardcode the tailnet IP. Reuse the established
`PHOTOSEARCH_NAS_URL` env var / `PHOTO-IMPORT-WORKFLOW.md` reference. Committed
code and docs use `PHOTOSEARCH_NAS_URL` or a `<nas-tailscale-ip>` placeholder,
consistent with existing repo convention.

### Where the code lives (in this repo)

- `photosearch/shutterfly_export.py` — new module: builds the **build spec** per
  book from `/api/books/{id}` (spreads → cells → photo_id, geometry, crop,
  caption), maps archetypes → Shutterfly layouts, and translates crops (reusing
  `book_export.plan_cell`). Pure/testable; no Shutterfly auth.
- CLI command(s) in `cli.py` (e.g. `shutterfly-spec`, `shutterfly-gphotos-push`)
  following the repo's Click + `PHOTOSEARCH_DB` conventions.
- Browser driver asset (JS) — the in-editor script that uploads/imports and
  reconstructs spreads via `builderApp`. Kept in the repo (e.g. under a
  `shutterfly/` asset dir) and injected via the Chrome MCP; also usable as a
  pasteable console snippet / bookmarklet.
- This spec: `docs/superpowers/specs/`; the implementation plan:
  `docs/superpowers/plans/`.

## Phase 1 — Photo upload via Google Photos (do first)

Delivers outcome #1 and produces the `photo_id → Shutterfly asset` mapping that
Phase 2 needs. Reuses existing Google Photos code end-to-end.

1. **Build spec** (`shutterfly_export`): for each book, read `/api/books/{id}`,
   collect the photos actually placed in cells (+ cover/title), in spread/cell
   order. Emit `manifest.json`: book meta, ordered spreads, each cell's photo_id,
   geometry, crop, and the spread caption. Record each photo's original filename
   too.
2. **Push to a per-book Google Photos album:** `create_album("2026 Summer — <name>")`
   then `upload_photos(...)` for that book's photos, keeping the **original
   filename** (see the Update note below — an earlier `<photo_id>.jpg` rename was
   abandoned). Be gentle (existing batching + pauses).
3. **Import into Shutterfly:** in the editor, "Add More Photos" → Google Photos →
   select the book's album → import into the project's photo tray.
4. **Recover mapping:** read the Shutterfly photo/asset list (via `builderApp` or
   the project doc), match uploaded filename → `photo_id`, and merge
   `sfly_asset_id` into the manifest.

Deliverable: all three books' photos in Shutterfly, one album each, plus a
manifest carrying the asset mapping.

> **Update (2026-08-09) — Phase 1 executed; mapping is by original filename.**
> All three books were uploaded to per-book Google Photos albums and imported into
> Shutterfly. The `<photo_id>.jpg` rename was **abandoned**: Google Photos keeps
> the raw-upload header filename (the original basename) and ignores
> `simpleMediaItem.fileName`, so the rename never reached Shutterfly — and it was
> unnecessary because original filenames are already unique within a book (cameras
> use non-overlapping sequences; verified 130/136/132 with zero dups for books
> 17/14/12). Phase 2 maps Shutterfly assets → `photo_id` by **original filename**,
> scoped to each book's album. See the Phase 1 plan's "Spike S1 findings".

## Phase 2 — Native layout rebuild (faithful-ish, then polish)

Per book:

1. Create an 11×14 layflat project sized to the spread count (respecting the
   **56-spread ceiling** — apply the planned 57→56 merge per book).
2. For each spread: apply a Shutterfly layout whose cell count/arrangement
   matches the archetype (map photo-search archetypes — `matched 2-up`,
   `hero + sidebar`, `gallery row`, `collage 3+3`, full-bleed/panorama, etc. —
   to the nearest Shutterfly layout), assign the spread's photos into cells in
   order by `sfly_asset_id`, apply a best-effort crop derived from
   `crop_cx/cy/zoom` via `plan_cell`, and add the caption as a text box
   (honouring `dark`).
3. Save via the builder. Matt polishes crops/layouts.
4. **Verify** by comparing the Shutterfly spread against
   `GET /api/books/{id}/render/{spread_id}` and an editor screenshot.

Built one book → one archetype at a time; visual check before scaling.

## Spikes (both on throwaway test material — never Matt's real books/projects)

- **S1 — Google Photos → Shutterfly mapping fidelity.** Push a few test photos
  with `filename=<photo_id>.jpg` to a scratch GP album, import into a scratch
  Shutterfly project, confirm the filename survives and the mapping is
  recoverable. Fallback if filenames are stripped: group-by-album + upload order.
- **S2 — Save-payload schema (gates Phase 2).** On a throwaway Shutterfly test
  book: place one photo, crop it, add a text box; capture the PUT to
  `projects-api3.../project/{id}` to learn the `Picture` geometry/crop shape,
  text-box shape, and asset-ref shape. Also probe `builderApp` helper signatures
  (`applyLayout`, `autoFill`, `swapPhotoItems`).

## Risks & constraints

- **Proprietary schema.** Phase 2 depends on S2. Until S2 lands, Phase 2 is
  planned, not promised.
- **Session-bound auth.** All Shutterfly-touching steps must run in the
  logged-in browser; nothing server-side can authenticate on its own.
- **57 → 56 spreads.** Each book exceeds the layflat ceiling by one and needs a
  merge (existing plan for book 12; books 14 and 17 need the same call).
- **Gentleness.** Matt's own account, own photos, his established workflow — low
  ToS concern, but batch uploads with pauses; don't hammer Shutterfly or the NAS.
- **Google storage.** Uploading full-res photos for three books consumes Google
  Photos storage.
- **Privacy/compliance.** No tailnet IP or NAS hostname in committed files; use
  `PHOTOSEARCH_NAS_URL` / placeholders (repo convention).

## Testing

- Unit-test `shutterfly_export` spec generation and crop translation against a
  known book (fixture from `/api/books/{id}`), asserting cell counts, ordering,
  and `plan_cell`-derived crop regions.
- Manual/visual verification for the browser driver via the per-spread render
  endpoint + editor screenshots.

## Open questions

1. Should the manifest include the full candidate set per book (not just placed
   photos) so extras are available in Shutterfly, or placed photos only?
2. Archetype → Shutterfly-layout mapping table: build it empirically during S2/
   Phase 2 (which Shutterfly layouts exist for 1/2/3/6-cell spreads at this SKU)?
3. Cover/title handling: reproduce the cover spread natively too, or leave the
   cover for Matt to set by hand?

---
paths:
  - "photosearch/upscale.py"
  - "photosearch/split_export.py"
  - "photosearch/book*.py"
  - "photosearch/shutterfly_export.py"
  - "frontend/dist/split*"
  - "frontend/dist/book.html"
---

# Print (Topaz, Split, books) — traps

Full history and measurements: docs/notes/print.md (cite by heading).

- **Local Topaz upscaling (M30)** — nothing leaves the machine. Use the locally
  installed Topaz Photo AI CLI only; never the Topaz cloud API or
  `TopazLabs/topaz-mcp` (a cloud wrapper). Output is a new file under
  `export_root()`; write nothing to the DB, the NAS or the library.
- **Use Photo AI, not Gigapixel** — `gigapixel.exe` refuses below a Pro
  license. The newer "Topaz Photo" app ships a broken `tpai.exe` (exits −1,
  writes nothing), so `_CLI_CANDIDATES` tries "Topaz Photo AI" first and
  "Topaz Photo" last. `PHOTOSEARCH_TOPAZ_CLI` skips discovery.
- **Success = an output file appeared** — a good run exits 9 under WSL
  interop, never 0. Never judge by exit code and never add `check=True`; that
  breaks every successful run. The documented failure codes (255/254/253/1)
  are reliable.
- **`--upscale scale=N` is broken** — the CLI sends the number as a string and
  the engine rejects it. `_scale_override` sets the Autopilot registry
  preference, runs, and restores it under `_REG_LOCK` (global state shared with
  the Topaz GUI). No `--scale` means touch nothing. `scale` is a float in
  `[MIN_SCALE, MAX_SCALE]`; `SCALES` is only the UI preset list, not a
  constraint. `model=` serializes correctly and needs no workaround.
- **Artifacts: Autopilot merges its own passes in** — `--override` does NOT
  turn off Autopilot's extras (Sharpen stays on). With `override=True`,
  `run_topaz` emits an explicit `--<enh> enabled=false` for every enhancement
  not requested. Don't simplify this back to a bare `--override`.
- **Strength: Topaz has none, so we blend** — every Topaz-side strength lever
  failed. `blend_strength()` cross-fades the Topaz result toward a plain
  Lanczos resize (1 = full Topaz, 0 = resize), in strips to bound memory.
  Default 0.25 at the CLI `--strength`, the API model and the UI; 1.0 in
  `upscale_photo(strength=...)` itself. Keep that split.
- **`UPSCALE_MODELS` is a verified list** — each name was run and produced a
  file. The generative models (`Redefine`, `Recovery`) produce nothing until
  downloaded in the GUI; don't add them. Model, override, scale and strength
  are in the variant filename and must round-trip through `_parse_variant`.
- **Windows paths for humans** — the server runs in WSL and `tpai.exe` is a
  Windows binary: translate with `to_native_path` before handing a path over.
  Results carry `windows_path` and `file_url`; show those, not the POSIX path.
  The modal's View link goes through `GET /api/admin/upscaled-file`, because
  Chrome won't open `file://` from an http page.
- **Confine file reads to `export_root()`** — `/api/admin/upscaled-file` and
  split-export's `source_path` both resolve the path and refuse anything
  outside the export root (traversal = arbitrary file read).
- **Raise `Image.MAX_IMAGE_PIXELS` only for the duration** — save the old
  value, raise it, restore it in `finally` (`upscale.py`, `split_export.py`'s
  `_BOMB_HEADROOM_PX`). Upscaled sources legitimately trip Pillow's bomb guard;
  never lift it globally.

- **The geometry exists twice** — `frontend/dist/split-geometry.js` (recomputes
  per drag frame) and `photosearch/split_export.py` (the export authority).
  Both tests pin the same worked example (6528×4352 → 3×2 of 13×19″, gap 0.5″,
  window → 40″ × 38.5″, 113 dpi, 31% crop, 17,326 × 11,552 px for 300 dpi).
  Change one, change the other.
- **`keep` is scale-free** — retained area depends only on the two aspect
  ratios, never on pixel count. Crop is a shape mismatch that resolution
  can't fix.
- **Binding axis comes from `sa > ta`** — never from comparing the slacks
  (when aspects match, both slacks are zero).
- **Seam mode decides how the sheets must be MOUNTED** — window mode omits
  `gap × dpi` pixels between panels, so the sheets must hang that far apart;
  continue mode omits nothing. `G.seamNote` turns this into the mounting
  instruction. `test_adjacent_panels_join_exactly` pins both modes.
- **Integer-dpi grid** — JPEG stamps an integer density, so `out_dpi` is the
  nearest integer dpi (round, then step down until the run fits the crop) and
  every panel is exactly `pw × out_dpi` by `ph × out_dpi` px. Never size
  panels from the fractional dpi or round each box on its own.
  `test_panels_are_exactly_sheet_sized` pins it.
- **Seam misalignment in print (files were fine)** — if a print's seams jump
  and the files check out, suspect borderless overspray (the driver's
  Expansion setting) before the export. A pre-compensating bleed option is not
  built.
- **`PANEL_COUNTS` is derived from `GRIDS`** — never hand-write it; a count no
  grid produces is a dead end in the UI.
- **`G.forGrid` enforces sheet sizes, not floors** — a tried grid ignores the
  dpi and crop floors (cards are marked "below your floors") but must only use
  the selected sheet sizes: those are the paper the owner has.
- **Planner defaults live in the sidecar DB** — `photosearch/settings.py`
  (`app_settings`) in `PHOTOSEARCH_BOOKS_DB`, never `PHOTOSEARCH_DB` (the
  replica sync swaps it wholesale). Every key must pass
  `web._SETTING_VALIDATORS`; unknown or invalid keys are rejected by name, not
  dropped. Saving stays explicit; the page waits for stored values before
  rendering.

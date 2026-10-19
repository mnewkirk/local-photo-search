# Print — history and traps

Topaz upscaling (M30) and the Split planner (M31): Autopilot artifacts, strength blending, seam modes, the integer-dpi grid. Condensed rules: `.claude/rules/print.md`. Append new history under a new `###` heading; never rename an existing heading.

<!-- moved verbatim from CLAUDE.md (line 2380) on 2026-10-19 -->
## Local Topaz upscaling (M30)

Upscale a photo from the image view using the **locally installed Topaz Photo
AI**. Nothing leaves the machine — the Topaz *cloud* API (and the official
`TopazLabs/topaz-mcp` server, which is just a wrapper around it and needs a paid
key from developer.topazlabs.com) is deliberately unused.

Same asymmetry as M28: the NAS holds the originals, the desktop does the
compute. But unlike every other pass, the output is a new **file**, not DB
columns — so this writes nothing to the DB, the NAS, or the library. Results go
to a standalone export tree, dated `YYYY/` and suffixed `-topaz`. Module:
`photosearch/upscale.py`; tests: `tests/test_upscale.py`.

```bash
photosearch upscale 145820 [--scale 2|4|6] [--enhancement upscale] [--overwrite]
```

- **UI** — `PS.PhotoModal` has an "Upscale (Topaz)" section (scale dropdown +
  enhancement checkboxes) below "Re-run passes".
- **`POST /api/admin/upscale-photo`** `{photo_ids, scale, enhancements, overwrite}`.
  Synchronous (~20-30 s per photo at 2x), so it's sized for one photo at a time.
  `503` when the Topaz CLI isn't installed — the service is fine, the install isn't.
- **`GET /api/admin/upscaled-file?path=…`** serves one exported file. Confined to
  `export_root()` by a resolved-path check (traversal → 403), because it reads
  real files off disk.

**Paths are reported Windows-style.** The server runs in WSL, so the raw
`/mnt/c/...` result path is useless to a human on Windows. Every result carries
`windows_path` (`C:\Users\...`, via `wslpath -w`) and `file_url`
(`file:///C:/...`); the UI and CLI show those, not the POSIX path. The modal's
**View** link goes through `/api/admin/upscaled-file` rather than the `file://`
URL — **Chrome refuses to navigate to `file://` from an http page**, so a direct
link would be silently inert. `file_url` exists for copy-paste into the address
bar, which does work.
- **`PHOTOSEARCH_UPSCALE_DIR`** — export root (default `./upscaled`).
  **`PHOTOSEARCH_TOPAZ_CLI`** — explicit `tpai.exe` path, skips discovery.

**Use Photo AI, not Gigapixel.** `gigapixel.exe` has the far nicer CLI (explicit
`--model` / `--scale` / `--face-recovery` / `--skip-existing`) but hard-refuses
below a Pro license: `Gigapixel CLI requires a Pro license`. `tpai.exe` from
**Topaz Photo AI** has no such gate. Note the newer **"Topaz Photo" app ships
its own broken `tpai.exe`** — it fails to load its QML entry point, exits −1 and
writes nothing — so `_CLI_CANDIDATES` prefers the "Topaz Photo AI" install and
only falls back to "Topaz Photo".

Two upstream CLI bugs this works around:

1. **The success exit code is wrong.** A successful run exits **9** under WSL
   interop (127 under Git Bash), never 0. The *failure* codes are correct and
   documented (255 no valid files, 254 stale login, 253 bad argument, 1 partial).
   So success is decided by *"an output file appeared"*, never by the exit code —
   don't "fix" this to a `check=True`, it will break every successful run.
2. **`--upscale scale=N` is broken** — the CLI serializes the scale as a JSON
   string and the engine rejects it (`type_error.302: type must be number, but
   is string`). The only working lever is the Autopilot *preference* in the
   registry (`HKCU\Software\Topaz Labs LLC\Topaz Photo AI` →
   `autopilotUpscalingType` = `auto|scale`, `autopilotUpscalingFactor`).
   `_scale_override` sets it, runs, and restores it under a lock, because that
   preference is global state shared with the Topaz GUI. **`--scale` omitted
   (the default) touches nothing** and just uses whatever Autopilot would pick.

   **The factor is not limited to whole numbers.** That registry preference
   takes arbitrary decimals — verified: 1.3 on a 3660×2997 source produced
   exactly 4758×3896, and 1.3 on a 6528×4352 photo produced 8486×5658. So
   `scale` is a **float** in `[MIN_SCALE, MAX_SCALE]` = `[1.05, 6.0]`; `SCALES`
   is only the list of UI presets, not a constraint.

   This matters because print plans rarely need a whole factor. Sweeping every
   blocked arrangement of one 28 MP photo, the required factors ran **1.1× to
   1.9× — not one needed 2×**. Rounding those up to a 2/4/6 ladder means 2.4×
   the pixels, every one carrying synthesized texture the plan never asked for
   (and 2.4× the file). `/split`'s diagnostic therefore asks for the exact
   factor; `upscaleFor` rounds up to a tenth so the result clears the floor
   rather than landing a hair under it.

<!-- moved verbatim from CLAUDE.md (line 2455) on 2026-10-19 -->
### Artifacts: Autopilot merges its own passes in

**`--upscale` alone does not mean "only upscale".** Without `--override`, the
requested settings are *merged into* Autopilot's rather than replacing them, so
Topaz also runs whatever else Autopilot decided on. Confirmed with
`--showSettings --skipProcessing`: on a plain `--upscale` run the enabled
categories come back as **Enhance *and* Sharpen**, the latter with model
`Sharpen Strong` — a sharpening pass on an already-upscaled image, and the usual
explanation for halo/crunch artifacts. Autopilot's own upscale model is
typically `High Fidelity V2`.

**`--override` does NOT turn Autopilot's extras off.** Measured: with and
without it, `--showSettings` reports `Enhance -> true, Sharpen -> true`. The
flag that works is the per-enhancement one the help documents —
**`--sharpen enabled=false`** — which does bring it to `false` and still
processes. So `override=True` emits an explicit `enabled=false` for every
enhancement *not* requested (and passes `--override` too). Do not "simplify"
this back to a bare `--override`; it silently stops working.

The two levers, on `upscale_photo` / `run_topaz`, the CLI (`--model`,
`--override/--merge-autopilot`), the API, and the modal:

- **`override=True`** → run only the checked enhancements. The artifact fix;
  the UI defaults it **on**.
- **`model=...`** → `--upscale model=<name>`. Unlike `scale`, the CLI serializes
  this correctly, so it needs no registry workaround.

**Measured on a real 7008×4672 photo** — Laplacian variance of a 200% crop,
higher meaning more high-frequency energy (crunch). **Where you measure decides
the answer**, which is the trap here:

| variant | high-detail corner | frame centre |
|---|---|---|
| original, plain resize | 35.8 | 108.6 |
| Autopilot default (Sharpen on) | 135.8 | 1448.8 |
| Sharpen off, High Fidelity V2 | **77.3** | **1453.1** |
| Sharpen off, Standard | 120.3 | 3299.0 |
| Sharpen off, Standard V2 | 107.0 | 2737.4 |

On a high-contrast corner, disabling Sharpen cuts crunch 43%. **At the centre it
changes nothing** (1448.8 → 1453.1) — there the artifacts come from the upscale
*model itself*, synthesizing micro-texture at ~13× the original's
high-frequency energy on rock and scree. Do not generalize from one crop; the
first pass at this did, and was wrong.

Two things that stay true everywhere: **`Standard` is *crunchier* than `High
Fidelity V2`** despite the name, so Autopilot's own model choice is the right
default; and disabling noise/lighting/color changes nothing measurable.

<!-- moved verbatim from CLAUDE.md (line 2504) on 2026-10-19 -->
### Strength: Topaz has none, so we blend

Every Topaz-side strength lever was measured and rejected:

- `--upscale param1=0.5` → rejected as a string, same `type_error.302` as `scale=N`.
- `autopilotOpacityValue=50` → output **identical** to 100. No effect.
- `autopilotUpscalingParam1/2Strength` at minimum → 1377.3 vs 1453.1, a 5% move.

So `blend_strength()` composites the Topaz result toward a plain Lanczos resize
of the original — `strength=1` full Topaz, `0` plain resize. Measured on the
same centre window: 108.6 (0%) → 226.2 (25%) → 589.1 (50%) → 1042.4 (75%) →
1453.1 (100%). Monotonic across the whole range, and the only strength control
that exists. Done in horizontal strips, since a 130 MP pair would otherwise want
over a gigabyte at once. Part of the variant filename (`-s25`).

**25% is the default** at every surface a person touches — the `--strength` CLI
flag, the API model, the UI dropdown — chosen by eye off the ladder above on a
real photo. `upscale_photo(strength=...)` itself still defaults to **1.0**: a
library function named "upscale" should not blend behind the caller's back, and
keeping it at 1.0 means the default can be retuned in one place without
rewriting every test's expected filename. Full strength is still one dropdown
change away when a subject wants it.

Resolution is unaffected by strength — a 25% blend is still the full upscaled
pixel dimensions (verified: 14016×9344 at both 25% and 100%), so `/split` panel
exports get the same dpi either way. Strength only governs how much synthesized
texture rides along.

`UPSCALE_MODELS` was verified by running each name through the CLI and checking
an output file appeared: **Standard, Standard V2, High Fidelity, High Fidelity
V2, Low Resolution, CGI**. `Redefine` and `Recovery` (the generative models)
produce **no output until downloaded in the Topaz GUI**, so they are deliberately
not offered. `Text Refine` is a separate enhancement, not an Enhance model.

Both are part of the variant filename (`-topaz-upscale-standardv2-only-2x.JPG`)
and round-trip back out via `_parse_variant`, so models can be compared
side by side without overwriting each other.

**Model installs need every `tpai.exe` closed** — including headless ones. A
batch of CLI runs will block the GUI's model download with "tpai needs to be
closed"; check with `Get-Process | Where ProcessName -match 'tpai|Topaz'`.

**WSL note:** the replica server runs inside WSL2 but `tpai.exe` is a Windows
binary, so paths are translated with `wslpath -w` before being handed over
(`to_native_path`). Both a WSL-native and a `/mnt/c` export dir work; `/mnt/c`
avoids a `\\wsl.localhost` round trip per read/write and is measurably faster
for large files.

<!-- moved verbatim from CLAUDE.md (line 2552) on 2026-10-19 -->
## Split — multi-panel print planner (M31)

`/split?photo=<id>` plans one photo across several **borderless** sheets — a
diptych, triptych, or grid up to 4×3 — ranks the arrangements, and says exactly
how many pixels land on each inch of paper. Reached from the image view's
**⊞ Split print →** link. Ported from a design handoff whose own README names
this repo as the intended host.

**The governing constraint: every panel is one whole borderless sheet.** Nothing
is cut or trimmed, so panel sizes are the printer's stock sizes (4×6 … 13×19,
Epson XP-15000) and the piece's dimensions are whatever those sheets add up to,
plus the wall gaps.

Geometry lives in `frontend/dist/split-geometry.js` — a **pure, framework-free
module** (browser global *and* CommonJS) so it is unit-testable; it must run
client-side because it recomputes per frame while dragging the crop. Tests:
`frontend/__tests__/split-geometry.test.js` (50 cases), anchored on the
handoff's worked example — 6528×4352 → six 13×19″ portrait as 3×2, gap 0.5″,
window mode → 40″ × 38.5″, 113 dpi, 31% crop, 17,326 × 11,552 px for 300 dpi.

Two invariants worth not breaking:

- **`keep` (retained area) is scale-free** — it depends only on the two aspect
  ratios, never the pixel count. Crop cost is a *shape* mismatch that no amount
  of resolution fixes. This is what makes the crop-vs-resolution split in the
  diagnostics meaningful.
- **The binding axis comes from the `sa > ta` comparison**, never from comparing
  the slacks. Letting those drift apart once told users their width was the
  constraint when it was the height. (Note the handoff's "exactly one slack is
  zero" is off by one case: when the aspects match exactly, *both* are zero.)

**Planner defaults are stored server-side** — wall size, gap, seam mode,
"only what fits my wall", resolution floor, crop allowance, sheet sizes, panel
count. A **Defaults** card at the foot of the sidebar saves the current
settings; Reset forgets them and falls back to `G.DEFAULTS`. Saving is explicit
(the button only enables when something differs from what is stored), so
fiddling with a control for one photo does not silently rewrite your defaults.

They live in `photosearch/settings.py` (`app_settings`, namespaced key/JSON) in
the **sidecar** file `PHOTOSEARCH_BOOKS_DB` — *not* the replica DB, because
`sync-replica.sh` swaps `PHOTOSEARCH_DB` wholesale and would wipe them. Routes:
`GET|PUT|DELETE /api/settings/{namespace}`. Writes are checked against a
per-key **allowlist** in `web._SETTING_VALIDATORS`: these values drive geometry,
and an out-of-range one would be hard to back out of from the UI, so an unknown
or invalid key is rejected by name rather than silently dropped. The page holds
rendering until the stored values arrive, or arrangements for the built-in
defaults would flash past and be replaced. Tests: `tests/test_settings.py`.

<!-- moved verbatim from CLAUDE.md (line 2600) on 2026-10-19 -->
### Seam mode decides how the sheets must be MOUNTED

In **window** mode the picture continues *behind* the wall gaps, so the
export deliberately omits a gap's worth of image between panels. That is
correct only if the sheets are hung that far apart; butt-joined, the missing
strip would read as the image jumping at every seam (0.1″ at 399 dpi is 40 px).
In **continue** mode nothing is omitted, so those sheets can be butted or
spaced freely. At a zero gap the two modes are identical.

**This was NOT the cause of the seam failure on the first real print** — that
export was `continue` mode, verified contiguous. See "Seam misalignment in
print" below. The guidance stands on its own; do not re-attribute that incident
to it.

`G.seamNote(P)` turns this into the mounting instruction shown in the
sidebar and on the stage (highlighted when the combination is the risky
one). `tests/test_split_export.py::test_adjacent_panels_join_exactly` pins
the behaviour with a source whose every column is uniquely coloured, so a
panel's edge column identifies its exact source column: continue mode must
be perfectly contiguous, window mode must skip exactly `gap x dpi` pixels.

<!-- moved verbatim from CLAUDE.md (line 2621) on 2026-10-19 -->
### Seam misalignment in print (files were fine)

A printed 4×2 of 13×19″ showed the image jumping at the joins. The exported
files were **correct**: the manifest confirms `continue` mode with a 0 px
stride-minus-panel gap, and stitching `panel-r1c1` to `panel-r1c2` gives a seam
step of 3.04 against 2.44 for a typical neighbouring column *inside* a panel —
a 1.24× ratio, i.e. indistinguishable from any other column boundary.

**One real defect was found on our side.** Panels were sized from the
*fractional* dpi (240.5) while JPEG can only stamp an **integer** density (240),
so each file declared **13.025″ × 19.042″ for a 13 × 19″ sheet** — and rounding
each panel box independently made neighbours differ by a pixel (3126 vs 3127).
Printing with scaling turned off ("it should already be fit") therefore laid out
more than the sheet, unevenly, before borderless expansion was even considered.

Panels are now laid out on an **integer-dpi grid**: `out_dpi` is the nearest
integer dpi whose full run still fits inside the crop (rounding, not flooring —
flooring throws away 1.3% at 41 dpi where rounding costs 0.2%), and each panel
is exactly `pw × out_dpi` by `ph × out_dpi` pixels. `pixels / dpi` is now
exactly the sheet, and every panel is identical.
`test_panels_are_exactly_sheet_sized` pins it.

That alone does not explain the whole misalignment — 25 mil is far less than
what the print showed — so the prime remaining suspect is
**borderless overspray**: an inkjet printing borderless enlarges the page
(Epson calls it *Expansion*, roughly 2–3% at the Standard setting) so ink runs
off all four edges and no white margin can appear. Every sheet therefore loses
a sliver at *every* edge — including the two that meet at a seam — so ~0.2″ per
edge goes missing and adjacent panels no longer line up. Set the driver's
borderless **Expansion to Minimum**, or print one panel with borderless off to
confirm before reprinting the set. A pre-compensating **bleed** option on the
export (widen each panel rect by the expansion percentage so what survives the
crop is exactly the intended panel) is the durable fix if Minimum is not
enough — not built yet.

**The empty-state diagnostic names the actual constraint.** It used to say
"No sheet size is selected" for *every* empty result — wrong in 25 of the
26 ways the list can empty, and it told people to turn on sizes that were
already on. The branches are now `empty-sizes` (genuinely none selected),
`empty-count` (no grid makes that many panels), `empty-sizes-filter`, and
`empty-wall` (they fit nowhere on your wall — quotes the smallest
arrangement), each with a fix that actually produces results.

`PANEL_COUNTS` is **derived from GRIDS**, not hand-written: the chip row
offered **8**, which no grid produces, so choosing it was a guaranteed dead
end. Use the cols × rows boxes for anything off that list.

**Arrangement cards carry a live preview**, not just numbers — the panel
breakdown *and* where the photo lands inside it. `PanelLayout` is shared by the
card thumbnails and the main stage, so a thumbnail can never be a different
rendering of the geometry than the thing it previews.

**Trying a grid** (the `cols × rows` boxes above the card grid) needs its own
enumerator: `candidates`/`reachable` only walk the curated `GRIDS` list, which
has **no 3×3**, so a grid that is not on that list is invisible to them —
exactly the case worth trying. `G.forGrid(img, cols, rows, opts)` enumerates
sheet sizes directly for one grid. The dpi and crop floors are deliberately
not enforced — seeing *why* a grid falls short is the point of asking for it,
and such cards are marked "below your floors". **The sheet-size selection IS
enforced**: those are different kinds of constraint and conflating them once
suggested a 4×2 on 8.5×11″ to someone with only 13×19″ switched on. A floor is
a preference you might want to see violated; "sheet sizes in play" is a hard
fact about which paper you own. Tried grids persist in `localStorage` under `split.tried`, so clicking
away or reloading keeps them, and they go only when the card's × is clicked.

**The Source picker is a planning input, not an export detail.** It sits at the
top of the sidebar and sets the resolution *everything else is judged against* —
choosing a 2× upscale doubles the pixels, so arrangements that failed the dpi
floor become reachable. Real example on a 7008×4672 photo: original → best is a
26.5″ two-panel diptych; its 2× upscale → best is a **57.5″ × 34″ twelve-panel
grid** at the same ~245 dpi. `list_exports` therefore returns `width`/`height`
per export (Pillow parses the header lazily, so a 130 MP file is not decoded);
without them the picker cannot drive the plan and is inert.

**Upscaling is the payoff.** A wall piece spreads one file over many sheets, so
dpi is the usual blocker, and `needPx` says precisely how much bigger the source
must be. The diagnostic offers **"Upscale N× with Topaz"** using M30, where N is
the smallest of 2/4/6 that reaches the requirement — and no button at all when
even 6× falls short, rather than a fix that would not fix anything.

Ranking is deliberate: **reach across the wall leads** (`reach*1000 -
crop*600`), because the goal is filling a wall. Resolution is pass/fail against
the floor and crop is a tolerance with the first 3% free — those are
constraints, not objectives. The handoff's *prose* says "sorted by
max(outerW,outerH)"; its *code* uses the weighted score, and the README says the
code is the specification.

**Print export is server-side** (`photosearch/split_export.py`, Pillow), not a
browser canvas: sources here are routinely 100 MP+ after an upscale, and the
output belongs in the export tree next to the upscales rather than in Downloads.

```bash
photosearch split-export 145820 --sheet 13x19 --grid 3x2 [--gutter 0.5] \
    [--mode window|continue] [--source <a prior Topaz export>] [--format jpg]
```

- **`POST /api/admin/split-export`** — same export behind the page's button.
- Panels land in `export_root()/split/<stem>-<cols>x<rows>-<pw>x<ph>-<mode>/`,
  one folder **per arrangement**, so exporting a second plan of the same photo
  cannot overwrite the first. Each folder carries a `manifest.json` with the
  full plan, the source, and its pixel dimensions.
- Output DPI metadata is stamped on every panel so the print dialog sizes the
  sheet correctly without being told.
- `--source` selects a prior Topaz export instead of the original — the usual
  case, and the page's dropdown offers them. It is confined to `export_root()`
  by the same resolved-path check as `/upscaled-file`, since it reads real files.
- `Image.MAX_IMAGE_PIXELS` is raised for the duration: Pillow's
  decompression-bomb guard rejects legitimately-enormous upscaled sources.

**The geometry exists twice** — `split-geometry.js` (the planner must recompute
per drag frame) and `split_export.py` (the export authority). Both are pinned to
the *same* published worked example in their tests, which is what catches drift;
change one and you must change the other.

Design handoff (not in the repo): `Split_ Photo Diptych Planner.zip`. Its
Organic cream/terracotta palette is mapped onto this app's dark tokens by role,
as that README instructs, rather than transplanted.

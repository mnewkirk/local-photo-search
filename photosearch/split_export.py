"""Split print export (M31) — slice one photo into borderless panel files.

The planner UI at ``/split`` works out *which* arrangement to print; this writes
it. Rendering is server-side on purpose: after a Topaz upscale a source is
routinely 100 MP+, which a browser canvas will not survive, and the output
belongs in the export tree next to the upscales rather than in Downloads.

The geometry here is a second implementation of ``frontend/dist/
split-geometry.js`` — the planner has to recompute per frame while the user
drags, so it cannot live only on the server. Both implementations are pinned to
the same published worked example (see tests), which is what catches drift:

    6528x4352 -> six 13x19" portrait as 3 cols x 2 rows, gap 0.5", window mode
    -> 40" x 38.5", 113 dpi, 31% crop

Nothing here writes to the DB, the NAS, or the photo library.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .upscale import (TopazError, export_root, fetch_source, to_windows_path,
                      to_file_url)

logger = logging.getLogger(__name__)

# Pillow refuses very large images by default as a decompression-bomb guard.
# Upscaled sources here are legitimately enormous, so the caller raises it.
_BOMB_HEADROOM_PX = 400_000_000


class SplitError(Exception):
    """Raised when an arrangement cannot be exported."""


@dataclass
class Plan:
    """One arrangement's geometry, in inches and source pixels."""
    outer_w: float
    outer_h: float
    field_w: float          # cW — width of the image field, inches
    field_h: float          # cH
    crop_w: float           # cw — source pixels used, horizontal
    crop_h: float           # ch
    ox: float
    oy: float
    step_x: float
    step_y: float
    dpi: float
    keep: float
    binding_axis: str
    pw: float
    ph: float
    cols: int
    rows: int
    gutter: float
    mode: str


def plan(img_w: int, img_h: int, pw: float, ph: float, cols: int, rows: int, *,
         gutter: float = 0.5, mode: str = "window",
         sx: float = 0.0, sy: float = 0.0) -> Plan:
    """Port of split-geometry.js ``plan``. See the module docstring."""
    outer_w = cols * pw + (cols - 1) * gutter
    outer_h = rows * ph + (rows - 1) * gutter

    # 'window': the picture continues behind the wall gaps. 'continue': it does
    # not, so only the inked sheets carry image.
    field_w = outer_w if mode == "window" else cols * pw
    field_h = outer_h if mode == "window" else rows * ph

    ta = field_w / field_h
    sa = img_w / img_h

    # Exactly one axis is used in full. Derive the binding axis from THIS
    # comparison, never by comparing the slacks — letting the two drift apart
    # once reported the wrong axis to the user.
    if sa > ta:
        crop_h = float(img_h)
        crop_w = img_h * ta
        binding = "height"
    else:
        crop_w = float(img_w)
        crop_h = img_w / ta
        binding = "width"

    slack_x = img_w - crop_w
    slack_y = img_h - crop_h

    return Plan(
        outer_w=outer_w, outer_h=outer_h,
        field_w=field_w, field_h=field_h,
        crop_w=crop_w, crop_h=crop_h,
        ox=slack_x * (sx + 1) / 2,
        oy=slack_y * (sy + 1) / 2,
        step_x=(pw + gutter) / field_w if mode == "window" else pw / field_w,
        step_y=(ph + gutter) / field_h if mode == "window" else ph / field_h,
        dpi=crop_w / field_w,
        keep=(crop_w * crop_h) / (img_w * img_h),
        binding_axis=binding,
        pw=pw, ph=ph, cols=cols, rows=rows, gutter=gutter, mode=mode,
    )


def panel_rects(p: Plan) -> list[dict]:
    """Source-pixel rectangle per panel, row-major. Port of ``panelRects``."""
    out = []
    for r in range(p.rows):
        for c in range(p.cols):
            out.append({
                "sx": p.ox + c * p.step_x * p.crop_w,
                "sy": p.oy + r * p.step_y * p.crop_h,
                "sw": (p.pw / p.field_w) * p.crop_w,
                "sh": (p.ph / p.field_h) * p.crop_h,
                "r": r, "c": c,
            })
    return out


def _dest_dir(row, p: Plan) -> Path:
    """One folder per arrangement, so re-exporting a different plan of the same
    photo does not overwrite the last one."""
    stem = os.path.splitext(os.path.basename(row["filepath"]))[0]
    shape = f"{p.cols}x{p.rows}-{_num(p.pw)}x{_num(p.ph)}"
    return export_root() / "split" / f"{stem}-{shape}-{p.mode}"


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def export_panels(db, photo_id: int, *, pw: float, ph: float,
                  cols: int, rows: int, gutter: float = 0.5,
                  mode: str = "window", sx: float = 0.0, sy: float = 0.0,
                  source_path: Optional[str] = None,
                  fmt: str = "jpg", quality: int = 95,
                  overwrite: bool = False,
                  server: Optional[str] = None) -> dict:
    """Write one file per panel and a manifest describing the piece.

    ``source_path`` points at a prior Topaz export when the plan needs more
    resolution than the original carries — the usual case for a wall piece. It
    must resolve inside the export tree; anything else is refused.
    """
    from PIL import Image

    row = db.get_photo(photo_id)
    if not row:
        raise ValueError(f"photo {photo_id} not found")
    if cols < 1 or rows < 1:
        raise SplitError("cols and rows must both be at least 1")
    if fmt not in ("jpg", "jpeg", "png", "tif", "tiff"):
        raise SplitError(f"unsupported format: {fmt}")

    root = export_root().resolve()
    if source_path:
        # Same containment rule as the file-serving route: this reads real
        # files off disk, so it must not become an arbitrary-read hole.
        src_resolved = Path(source_path).resolve()
        if not (src_resolved == root or root in src_resolved.parents):
            raise SplitError("source_path is outside the export directory")
        if not src_resolved.is_file():
            raise SplitError(f"source not found: {source_path}")

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _BOMB_HEADROOM_PX

    staging = None
    try:
        if source_path:
            src = str(Path(source_path).resolve())
        else:
            # Pull the original the same way an upscale does.
            staging = root / f".split-staging-{uuid.uuid4().hex[:8]}"
            staging.mkdir(parents=True, exist_ok=True)
            src = fetch_source(row, str(staging), server=server)

        with Image.open(src) as im:
            im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
            iw, ih = im.size
            p = plan(iw, ih, pw, ph, cols, rows,
                     gutter=gutter, mode=mode, sx=sx, sy=sy)

            dest = _dest_dir(row, p)
            if dest.exists() and not overwrite:
                existing = sorted(dest.glob("panel-*"))
                if existing:
                    return {
                        "photo_id": photo_id, "skipped": True,
                        "reason": "already exported",
                        "panels": [_describe(f) for f in existing],
                        **_summary(p, iw, ih, dest),
                    }
            dest.mkdir(parents=True, exist_ok=True)

            ext = "jpg" if fmt in ("jpg", "jpeg") else fmt
            written = []
            # Lay the panels out on an INTEGER dpi grid.
            #
            # A file is only "already the right size" — the thing that makes
            # printing with scaling turned off correct — if pixels / dpi is
            # exactly the sheet. JPEG stores density as an integer, so sizing
            # pixels from a fractional dpi (240.5) and then stamping the
            # rounded one (240) declares 13.025" x 19.042" for a 13x19" sheet,
            # and rounding each panel box independently made neighbours differ
            # by a pixel (3126 vs 3127). Both were real: they overflowed the
            # sheet and made panels inconsistent widths.
            #
            # Use the NEAREST integer dpi that still fits inside the crop.
            # Flooring unconditionally would be safe but wasteful: at 41.6 dpi
            # it throws away 1.3% of the piece, where rounding costs ~0.2%.
            ox, oy = int(round(p.ox)), int(round(p.oy))
            gap_in = p.gutter if p.mode == "window" else 0.0

            def _fits(d):
                pwp, php = int(round(p.pw * d)), int(round(p.ph * d))
                gx = int(round(gap_in * d))
                return (ox + p.cols * pwp + (p.cols - 1) * gx <= iw
                        and oy + p.rows * php + (p.rows - 1) * gx <= ih)

            out_dpi = max(1, int(round(p.dpi)))
            while out_dpi > 1 and not _fits(out_dpi):
                out_dpi -= 1

            pw_px = int(round(p.pw * out_dpi))
            ph_px = int(round(p.ph * out_dpi))
            # The gap is empty wall, not content, so a fractional pixel there
            # is harmless — only the panels themselves must be exact.
            gap_x = gap_y = int(round(gap_in * out_dpi))

            for r in range(p.rows):
                for c in range(p.cols):
                    x0 = ox + c * (pw_px + gap_x)
                    y0 = oy + r * (ph_px + gap_y)
                    if x0 + pw_px > iw or y0 + ph_px > ih:
                        raise SplitError(
                            f"panel r{r + 1}c{c + 1} runs past the source "
                            f"({x0 + pw_px}x{y0 + ph_px} vs {iw}x{ih})")
                    tile = im.crop((x0, y0, x0 + pw_px, y0 + ph_px))

                    name = f"panel-r{r + 1}c{c + 1}.{ext}"
                    out = dest / name
                    save_kw = {"dpi": (out_dpi, out_dpi)}
                    if ext == "jpg":
                        save_kw.update(quality=quality, subsampling=0)
                    tile.save(out, **save_kw)
                    written.append(out)

        manifest = dest / "manifest.json"
        summary = _summary(p, iw, ih, dest)
        # The dpi the FILES carry, which is what a printer will honour.
        summary["dpi"] = out_dpi
        summary["panel_px"] = [pw_px, ph_px]
        summary["sheet_exact"] = True
        manifest.write_text(json.dumps({
            "photo_id": photo_id,
            "source": src,
            "source_px": [iw, ih],
            "plan": asdict(p),
            "panels": [f.name for f in written],
            **{k: v for k, v in summary.items() if k != "dest"},
        }, indent=2))

        return {"photo_id": photo_id, "skipped": False,
                "panels": [_describe(f) for f in written], **summary}
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit
        if staging:
            shutil.rmtree(staging, ignore_errors=True)


def _describe(f: Path) -> dict:
    win = to_windows_path(str(f))
    return {"name": f.name, "output": str(f), "windows_path": win,
            "file_url": to_file_url(win) if win else None,
            "bytes": f.stat().st_size if f.exists() else 0}


def _summary(p: Plan, iw: int, ih: int, dest: Path) -> dict:
    win = to_windows_path(str(dest))
    return {
        "piece_inches": [round(p.outer_w, 3), round(p.outer_h, 3)],
        "sheet_inches": [p.pw, p.ph],
        "grid": [p.cols, p.rows],
        "dpi": round(p.dpi, 1),
        "crop_pct": round((1 - p.keep) * 100, 1),
        "binding_axis": p.binding_axis,
        "source_px": [iw, ih],
        "mode": p.mode,
        "gutter": p.gutter,
        "dest": str(dest),
        "dest_windows": win,
    }


def required_source_px(p: Plan, iw: int, ih: int, target_dpi: float) -> dict:
    """What the source would need to be for ``target_dpi``. Port of ``needPx``.

    Crop is scale-free, so the answer is a uniform scale-up of the current file.
    """
    k = target_dpi / p.dpi
    w = math.ceil(iw * k / 2) * 2
    h = math.ceil(ih * k / 2) * 2
    return {"w": w, "h": h, "mp": round(w * h / 1e5) / 10, "k": round(k, 1)}

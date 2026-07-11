"""Print-grade export for the photobook builder.

Renders each 28×11 spread at print resolution with Pillow, using the SAME crop
geometry as the browser editor (``plan_cell`` here mirrors ``planCell`` in
book.html) so the export is WYSIWYG with the on-screen proof. Full-resolution
originals are pulled through a caller-supplied ``fetch_image`` (the web endpoint
wires this to the local file or the NAS ``/full`` proxy). Output is one JPEG per
spread plus a combined PDF.
"""
from __future__ import annotations

import io
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

_SERIF_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/Library/Fonts/Georgia.ttf",
]
_SERIF_ITALIC = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
]


def _font(size: int, italic: bool = False):
    for p in (_SERIF_ITALIC if italic else []) + _SERIF_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def plan_cell(cell: dict, an: float) -> dict:
    """Port of book.html planCell — percent-of-cell aspect-preserving layout.

    ``an`` = image natural aspect (w/h). Returns Pw/Ph (% of cell w/h) and
    left/top (% of cell), honoring zoom, the graded min-visible constraints, and
    contain / full-view.
    """
    ac = (cell.get("w") or 1) / (cell.get("h") or 1)
    t = an / ac
    minw = cell.get("crop_min_w") or 0
    minh = cell.get("crop_min_h") or 0
    contain_pw = 100.0 if t >= 1 else 100.0 * t
    contain_ph = 100.0 / t if t >= 1 else 100.0
    contain = cell.get("fit") == "contain" or (minw >= 1 and minh >= 1)
    if contain:
        return {"Pw": contain_pw, "Ph": contain_ph,
                "left": (100 - contain_pw) / 2, "top": (100 - contain_ph) / 2}
    z = max(1.0, cell.get("crop_zoom") or 1)
    if t >= 1:
        ph = 100.0 * z
        pw = ph * t
    else:
        pw = 100.0 * z
        ph = pw / t
    if minw > 0 and pw > 100 / minw:
        s = (100 / minw) / pw
        pw *= s
        ph *= s
    if minh > 0 and ph > 100 / minh:
        s = (100 / minh) / ph
        ph *= s
        pw *= s
    if pw < contain_pw - 1e-6 or ph < contain_ph - 1e-6:
        pw, ph = contain_pw, contain_ph
    cx = 0.5 if cell.get("crop_cx") is None else cell["crop_cx"]
    cy = 0.5 if cell.get("crop_cy") is None else cell["crop_cy"]
    left = 50 - cx * pw
    top = 50 - cy * ph
    left = min(0, max(100 - pw, left)) if pw >= 100 else (100 - pw) / 2
    top = min(0, max(100 - ph, top)) if ph >= 100 else (100 - ph) / 2
    return {"Pw": pw, "Ph": ph, "left": left, "top": top}


def _paste_cell(canvas: Image.Image, cell: dict, img: Image.Image,
                box_px: tuple[int, int, int, int]) -> None:
    x0, y0, cw, ch = box_px
    an = img.width / img.height
    p = plan_cell(cell, an)
    img_w = p["Pw"] / 100 * cw
    img_h = p["Ph"] / 100 * ch
    left_px = p["left"] / 100 * cw
    top_px = p["top"] / 100 * ch
    resized = img.resize((max(1, round(img_w)), max(1, round(img_h))), Image.LANCZOS)
    sx0 = round(max(0, -left_px))
    sy0 = round(max(0, -top_px))
    sx1 = round(min(img_w, cw - left_px))
    sy1 = round(min(img_h, ch - top_px))
    if sx1 <= sx0 or sy1 <= sy0:
        return
    crop = resized.crop((sx0, sy0, sx1, sy1))
    canvas.paste(crop, (x0 + round(max(0, left_px)), y0 + round(max(0, top_px))))


def _draw_caption(canvas: Image.Image, cell_box: tuple[int, int, int, int],
                  caption: dict) -> None:
    """Draw a scrim + italic serif caption near the bottom of a cell, matching
    the editor's caption placement (dark pill on dark spreads, light on light)."""
    text = (caption or {}).get("text")
    if not text:
        return
    x0, y0, cw, ch = cell_box
    dark = bool(caption.get("dark"))
    size = max(12, round(ch * 0.05))
    font = _font(size, italic=True)
    over = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    # wrap to ~80% of cell width
    words = text.split()
    lines, cur = [], ""
    maxw = cw * 0.8
    for w in words:
        trial = (cur + " " + w).strip()
        if d.textlength(trial, font=font) > maxw and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lh = size * 1.35
    block_h = lh * len(lines)
    cy = y0 + ch - block_h - size * 0.9
    scrim = (0, 0, 0, 150) if dark else (255, 255, 255, 175)
    fill = (255, 255, 255) if dark else (34, 34, 34)
    widest = max((d.textlength(ln, font=font) for ln in lines), default=0)
    pad = size * 0.6
    bx0 = x0 + cw / 2 - widest / 2 - pad
    bx1 = x0 + cw / 2 + widest / 2 + pad
    d.rounded_rectangle([bx0, cy - pad * 0.6, bx1, cy + block_h + pad * 0.3],
                        radius=size * 0.5, fill=scrim)
    for i, ln in enumerate(lines):
        w = d.textlength(ln, font=font)
        d.text((x0 + cw / 2 - w / 2, cy + i * lh), ln, font=font, fill=fill)
    canvas.alpha_composite(over)


def render_spread(spread: dict, stage_w: float, stage_h: float,
                  fetch_image: Callable[[int], Optional[Image.Image]],
                  dpi: int = 150) -> Image.Image:
    """Render one spread to an RGB image at ``dpi`` (px = inches × dpi)."""
    W = max(1, round(stage_w * dpi))
    H = max(1, round(stage_h * dpi))
    bg = spread.get("bg") or "#ffffff"
    canvas = Image.new("RGBA", (W, H), bg)
    caption = spread.get("caption")
    for cell in spread.get("cells", []):
        if not cell.get("photo_id"):
            continue
        img = None
        try:
            img = fetch_image(cell["photo_id"])
        except Exception:
            img = None
        box = (round(cell["x"] / stage_w * W), round(cell["y"] / stage_h * H),
               round(cell["w"] / stage_w * W), round(cell["h"] / stage_h * H))
        if img is not None:
            _paste_cell(canvas, cell, img.convert("RGB"), box)
        # caption rides the first cell (mirrors the editor's __caption on position 0)
        if caption and cell.get("position") == 0:
            _draw_caption(canvas, box, caption)
    return canvas.convert("RGB")


def export_book(doc: dict, fetch_image: Callable[[int], Optional[Image.Image]],
                dpi: int = 150) -> tuple[list[bytes], bytes]:
    """Render every spread. Returns ``(per_spread_jpegs, pdf_bytes)``."""
    sw, sh = doc.get("stage_w", 28), doc.get("stage_h", 11)
    images = [render_spread(sp, sw, sh, fetch_image, dpi)
              for sp in doc.get("spreads", [])]
    jpegs = []
    for im in images:
        b = io.BytesIO()
        im.save(b, "JPEG", quality=88)
        jpegs.append(b.getvalue())
    pdf = io.BytesIO()
    if images:
        images[0].save(pdf, "PDF", save_all=True, append_images=images[1:],
                       resolution=float(dpi))
    return jpegs, pdf.getvalue()

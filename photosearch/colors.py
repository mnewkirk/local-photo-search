"""Dominant color extraction for photos.

Uses colorthief to pull a palette from each image.
Colors are stored as a JSON list of hex strings in the database.
"""

import json
from typing import Optional

from PIL import ImageFile
# Allow loading of slightly truncated JPEGs
ImageFile.LOAD_TRUNCATED_IMAGES = True

from colorthief import ColorThief


def extract_dominant_colors(image_path: str, color_count: int = 5, quality: int = 10) -> Optional[list[str]]:
    """Extract dominant colors from an image.

    Args:
        image_path: Path to a JPEG or PNG file.
        color_count: Number of dominant colors to extract (default 5).
        quality: Sampling quality — lower is faster, 1 is highest quality (default 10).

    Returns:
        A list of hex color strings like ['#3a6b2f', '#c4a87d', ...], or None on failure.
    """
    try:
        ct = ColorThief(image_path)
        palette = ct.get_palette(color_count=color_count, quality=quality)
        return [_rgb_to_hex(r, g, b) for r, g, b in palette]
    except Exception as e:
        print(f"  Warning: could not extract colors from {image_path}: {e}")
        return None


def colors_to_json(colors: list[str]) -> str:
    """Serialize a color list to JSON for storage."""
    return json.dumps(colors)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB integers to a hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"

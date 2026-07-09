"""Controlled style-tag vocabulary for the VLM aesthetics pass.

Hand-maintained (unlike the generated `vocab_visual.py`). These tags describe a
photo's *aesthetic style* — lighting, mood, tonal treatment, and compositional
character — and are stored in `photos.aes_style_tags` for faceted filtering
(e.g. "show me my golden-hour, moody frames"). The free-text `aes_style`
critique is the richer artifact; these tags are the filterable index over it.

Overlaps intentionally with `vocab_visual.py` (the category-visual pass) on a
few light/mood terms — that pass tags *content-neutral visual attributes*, this
one tags *aesthetic intent*; both can coexist on a photo.

Keep terms lowercase, hyphenated, and singular. Add freely; the parser only
keeps tags the model actually returns.
"""

AESTHETIC_STYLE_VOCABULARY: list[str] = [
    # Lighting
    "golden-hour",
    "blue-hour",
    "backlit",
    "rim-light",
    "soft-light",
    "harsh-light",
    "low-key",
    "high-key",
    "dramatic-light",
    "natural-light",
    "silhouette",
    "long-shadows",
    # Mood / emotion
    "moody",
    "serene",
    "joyful",
    "melancholy",
    "intimate",
    "energetic",
    "nostalgic",
    "dramatic",
    "playful",
    "tranquil",
    # Tonal / color character
    "high-contrast",
    "low-contrast",
    "muted",
    "vibrant",
    "warm-tones",
    "cool-tones",
    "monochrome",
    "black-and-white",
    "pastel",
    "earthy",
    "desaturated",
    "film-look",
    # Composition / treatment
    "minimalist",
    "symmetrical",
    "leading-lines",
    "rule-of-thirds",
    "negative-space",
    "shallow-depth",
    "bokeh",
    "wide-angle",
    "close-up",
    "candid",
    "environmental-portrait",
    "layered",
    "reflection",
    "motion-blur",
]

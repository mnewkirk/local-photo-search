"""Derived capture-fact visual tags — EXIF decides, the VLM does not.

The `category-visual` pass shows a vision model a downsampled tile and asks it
to pick terms from one flat 36-word checklist. That list mixes two kinds of
thing:

  * CAPTURE FACTS  — `long-exposure`, `low-light`, `panoramic`, `sharp`,
    `blurry`. Physically measurable, and *not visible in the pixels the model
    receives*. A VLM cannot read a shutter speed off a 336 px tile. EXIF can.
  * PERCEIVED qualities — mood, colour, light character, composition,
    viewpoint. Only a model looking at the picture can judge these.

Measured on the 151,291-tagged July library copy, the capture-fact half was not
merely noisy, it was anti-correlated with reality:

    long-exposure   7,885 tagged; only 5.8% were actually >= 1/15 s, and it
                    found just 25.1% of the genuine >= 1/4 s exposures.
    low-light      33,951 tagged (22.4%), including full-sun sports frames at
                    ISO 160.
    panoramic       4,290 tagged; only 2.3% had an aspect ratio >= 2.0.
    motion-blur     2,378 tagged; median `aes_sharpness` 7.0 tagged AND
                    untagged — i.e. no signal at all.

So those terms are taken away from the model and computed here. `motion-blur`
is retired outright: a single scalar cannot tell motion blur from defocus, and
no defensible EXIF rule exists (a long shutter on a tripod is sharp).

Storage does not change — everything still lands in the single
`photos.visual_tags` JSON column, so the search blob, the `visual_tag=` filter,
`list_vocab` and the Ask tools keep working untouched.

DON'T SIMPLIFY THIS BACK into the prompt. The prompt cannot fix it: the
information is not in the image.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

from .vocab_visual import VISUAL_VOCABULARY

# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------

#: Terms the VLM must never decide. Derived here from EXIF / stored scalars and
#: treated as AUTHORITATIVE by :func:`merge_tags`.
CAPTURE_FACT_TAGS = frozenset({
    "long-exposure",
    "low-light",
    "panoramic",
    "sharp",
    "blurry",
})

#: Terms removed from the vocabulary entirely — neither asked of the model nor
#: derived. `motion-blur` is the whole list; see the module docstring.
RETIRED_TAGS = frozenset({"motion-blur"})

#: What the model is actually shown. Derived from the compiled vocabulary so a
#: `photosearch compile-vocab` regeneration of `vocab_visual.py` (a GENERATED
#: file) cannot silently desync this split.
PERCEIVED_VOCABULARY: list[str] = [
    t for t in VISUAL_VOCABULARY
    if t not in CAPTURE_FACT_TAGS and t not in RETIRED_TAGS
]

#: The perceived terms grouped by the axis they vary along. The prompt asks for
#: at most ONE per axis: the old flat checklist invited the model to tick every
#: line, which is how one verbatim 8-tag set landed on 101 photos in a row.
#: Must partition PERCEIVED_VOCABULARY exactly (pinned by a test).
PERCEIVED_AXES: dict[str, list[str]] = {
    "light": ["backlit", "golden-hour", "harsh-light", "overcast",
              "overexposed", "silhouette", "soft-light", "sunny"],
    "colour": ["black-and-white", "colorful", "monochromatic", "muted", "vibrant"],
    "mood": ["dramatic", "joyful", "melancholy", "moody", "peaceful"],
    "atmosphere": ["foggy", "hazy", "snowy"],
    "viewpoint": ["aerial", "close-up", "macro", "wide-angle"],
    "composition": ["centered", "composite", "reflection", "split-screen",
                    "symmetrical"],
}

#: Short definitions for the terms the model most often over-applies. Rendered
#: into the prompt beside the term; absent terms are self-explanatory.
PERCEIVED_GLOSS: dict[str, str] = {
    "backlit": "light source behind the subject",
    "golden-hour": "warm low sun near sunrise or sunset",
    "harsh-light": "hard-edged shadows from direct overhead sun",
    "overcast": "flat grey sky, no visible shadows",
    "overexposed": "highlights blown to featureless white",
    "silhouette": "subject rendered as a dark shape",
    "soft-light": "diffuse light, gentle shadow edges",
    "sunny": "direct sunlight and blue sky visible",
    "monochromatic": "one hue family throughout, still in colour",
    "muted": "desaturated, low-contrast colour",
    "vibrant": "intense saturated colour",
    "dramatic": "strong contrast or tension in the scene",
    "melancholy": "visibly sombre or sad subject",
    "moody": "dark, brooding, low-key",
    "peaceful": "calm, still, unhurried subject",
    "hazy": "distant detail softened by atmosphere",
    "aerial": "looking down from far above the ground",
    "close-up": "subject fills most of the frame",
    "macro": "extreme magnification of a small object",
    "wide-angle": "visibly stretched wide field of view",
    "centered": "main subject squarely in the middle",
    "composite": "several images combined into one",
    "reflection": "a mirrored image in water or glass",
    "split-screen": "frame divided into separate panels",
    "symmetrical": "mirror-image balance across an axis",
}

#: Perceived pairs that cannot both be true of one photo. Used by the guard in
#: `describe.tag_visual_photo`. Deliberately CONSERVATIVE — only pairs that are
#: genuinely exclusive. `peaceful` x `moody` is the single biggest co-occurrence
#: in the library (19,056) and is NOT here: a still, misty lake is honestly
#: both. Adding it would reject good answers.
#: Note there is no `sharp` x `blurry` entry: both are DERIVED now, and
#: `derive_tags` cannot emit them together.
CONTRADICTORY_PAIRS: tuple[tuple[str, str], ...] = (
    ("dramatic", "peaceful"),
    ("joyful", "melancholy"),
    ("joyful", "moody"),
    ("overcast", "sunny"),
    ("foggy", "sunny"),
    ("harsh-light", "soft-light"),
    ("muted", "vibrant"),
    ("colorful", "muted"),
    ("colorful", "monochromatic"),
    ("black-and-white", "colorful"),
    ("black-and-white", "vibrant"),
    ("close-up", "wide-angle"),
    ("aerial", "close-up"),
    ("macro", "wide-angle"),
    ("aerial", "macro"),
)

# ---------------------------------------------------------------------------
# Thresholds — each chosen from the read-only July library copy. See CLAUDE.md
# "Derived capture-fact visual tags" for the validation tables.
# ---------------------------------------------------------------------------

#: `long-exposure` at >= 1/4 s. VLM agreement climbs monotonically as the bar
#: tightens (>= 1/15 s: 6.9% of those photos were VLM-tagged; >= 1/8 s: 19.0%;
#: >= 1/4 s: 25.1%), which is where the photographic meaning actually lives —
#: tripod / streaking territory, 1.0% of the library. At 1/15 s the tag would
#: cover 4.3% and sweep in ordinary handheld indoor frames.
LONG_EXPOSURE_MIN_SECONDS = 0.25

#: `low-light` below EV100 5.0 — roughly a dim domestic interior at night.
#: Cross-checked against clock time: it fires on 44.9% of photos taken 21:00-06:00
#: and 3.6% of photos taken 09:00-16:00, a 12.5:1 odds ratio. The next step up
#: (EV100 < 6) halves that to 6.6:1 for 11 more points of night recall.
LOW_LIGHT_MAX_EV100 = 5.0

#: `panoramic` at long-edge / short-edge >= 2.0. 283 photos library-wide, versus
#: the 4,290 the VLM claimed. Ordinary 3:2 and 4:3 frames fall far below.
PANORAMIC_MIN_ASPECT = 2.0

#: `sharp` / `blurry` from `aes_sharpness`, the aesthetics VLM's 1-10 technical
#: sub-score — a DIFFERENT and better model looking at a bigger image. The
#: separation is the cleanest in the dataset: 47.0% of photos scoring <= 2 were
#: VLM-tagged `blurry`, against 0.00% of every bucket at 3 and above. `sharp`
#: takes the top of the observed 1-9 range.
#: Only applied when `aes_sharpness IS NOT NULL` — no score means no opinion.
SHARP_MIN_AES_SHARPNESS = 9.0
BLURRY_MAX_AES_SHARPNESS = 2.0

#: Columns :func:`derive_tags` reads. Anything selecting a row for the merge
#: must include these.
DERIVE_COLUMNS = ("exposure_time", "f_number", "iso",
                  "image_width", "image_height", "aes_sharpness")

_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?$")


def parse_exif_number(raw: Any) -> Optional[float]:
    """Parse one EXIF scalar stored as TEXT into a positive float, or None.

    The library stores exifread's `Ratio` repr, so the real shapes are
    ``'1/250'``, ``'5001/200000'``, ``'1244236/699009'``, ``'63/10'`` and plain
    ``'2'`` / ``'0.004'``. Trailing unit marks (``2"``, ``1/250 sec``) are
    tolerated because other writers produce them.

    Returns None for anything non-positive, non-finite, or unparseable —
    "no value" and "a value we do not trust" are the same answer here, because
    every caller's rule is "missing EXIF derives nothing".
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val if math.isfinite(val) and val > 0 else None
    if not isinstance(raw, str):
        return None
    s = raw.strip().rstrip('"').strip()
    s = re.sub(r"\s*(?:sec(?:onds?)?|s)$", "", s, flags=re.IGNORECASE).strip()
    m = _NUMBER_RE.match(s)
    if not m:
        return None
    num = float(m.group(1))
    den = float(m.group(2)) if m.group(2) is not None else 1.0
    if den == 0 or num <= 0:
        return None
    val = num / den
    return val if math.isfinite(val) and val > 0 else None


def ev100(f_number: Any, exposure_time: Any, iso: Any) -> Optional[float]:
    """Exposure value normalised to ISO 100: ``log2(N^2/t) - log2(ISO/100)``.

    This, not ISO alone, is what "how much light was there" means: the same ISO
    160 sits in full sun at 1/800 f/5.6 and in a dim room at 1/30 f/2. Returns
    None unless all three terms parse — never guess.
    """
    n = parse_exif_number(f_number)
    t = parse_exif_number(exposure_time)
    i = parse_exif_number(iso)
    if n is None or t is None or i is None:
        return None
    try:
        return math.log2(n * n / t) - math.log2(i / 100.0)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _get(row: Any, key: str) -> Any:
    """Read `key` from a sqlite3.Row, a dict, or anything mapping-like."""
    if row is None:
        return None
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


def derive_tags(row: Any) -> list[str]:
    """Capture-fact tags for one photo row, sorted and deduped.

    `row` needs the :data:`DERIVE_COLUMNS` keys (a ``sqlite3.Row`` or a dict).
    A column that is missing, NULL or unparseable contributes NOTHING — this
    function never guesses, so a photo with no EXIF simply gets no derived tags
    rather than a wrong one.
    """
    out: set[str] = set()

    shutter = parse_exif_number(_get(row, "exposure_time"))
    if shutter is not None and shutter >= LONG_EXPOSURE_MIN_SECONDS:
        out.add("long-exposure")

    ev = ev100(_get(row, "f_number"), _get(row, "exposure_time"), _get(row, "iso"))
    if ev is not None and ev < LOW_LIGHT_MAX_EV100:
        out.add("low-light")

    w = parse_exif_number(_get(row, "image_width"))
    h = parse_exif_number(_get(row, "image_height"))
    if w and h and max(w, h) / min(w, h) >= PANORAMIC_MIN_ASPECT:
        out.add("panoramic")

    sharpness = _get(row, "aes_sharpness")
    if sharpness is not None:
        try:
            s = float(sharpness)
        except (TypeError, ValueError):
            s = None
        if s is not None and math.isfinite(s):
            # Mutually exclusive by construction — the bands cannot overlap.
            if s >= SHARP_MIN_AES_SHARPNESS:
                out.add("sharp")
            elif s <= BLURRY_MAX_AES_SHARPNESS:
                out.add("blurry")

    return sorted(out)


def merge_tags(perceived: Optional[Iterable[str]],
               derived: Optional[Iterable[str]]) -> list[str]:
    """Combine one VLM answer with the derived capture facts.

    The rule, in order:

      1. Every capture-fact term the model emitted is STRIPPED — derived wins,
         always, including when nothing was derived (an absent capture fact is
         a positive statement: the EXIF says this is not a long exposure, or
         there is no EXIF to say it is).
      2. Retired terms are dropped.
      3. Out-of-vocabulary junk is dropped.
      4. The derived terms are added.
      5. Sorted + deduped, so the stored JSON is a deterministic function of
         (answer, row) and the backfill can skip rows that would not change.
    """
    keep = {t.strip().lower() for t in (perceived or []) if isinstance(t, str)}
    keep &= set(PERCEIVED_VOCABULARY)
    keep |= {t for t in (derived or []) if t in CAPTURE_FACT_TAGS}
    return sorted(keep)


def merge_for_row(perceived: Optional[Iterable[str]], row: Any) -> list[str]:
    """`merge_tags(perceived, derive_tags(row))` — the whole rule in one call.

    This is the single entry point every write path uses, so the derived half
    can never diverge between the worker submit, the M28 re-run, the in-process
    `index` passes and the backfill CLI.
    """
    return merge_tags(perceived, derive_tags(row))

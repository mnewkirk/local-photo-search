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

import json
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
})

#: Terms removed from the vocabulary entirely — neither asked of the model nor
#: derived, and STRIPPED from existing rows. `motion-blur` is the whole list:
#: zero signal and no replacement, so there is nothing to preserve.
RETIRED_TAGS = frozenset({"motion-blur"})

#: FROZEN: not derived, not asked, NOT DELETED.
#:
#: These were briefly derived from `aes_sharpness`, on the strength of the
#: cleanest-looking separation in the July validation copy — 47.0% of photos
#: scoring <= 2 carried the VLM's own `blurry` tag against 0.00% of every
#: bucket at 3 and above. That is two models agreeing, which is not ground
#: truth, and the July copy had only 2.4% coverage of the score so the rule
#: barely fired there.
#:
#: On the live DB coverage is 98.9% (158,111 rows) and the rule bites hard:
#: `blurry` 1,284 -> 10,103 (+8,819) and `sharp` 3,410 -> 15,076 (+11,666).
#: Four photos scoring <= 2 were inspected by hand — two were genuinely blurry
#: (defocused / motion-blurred indoor sports) and TWO WERE NOT: a tack-sharp
#: phone photo of construction formwork with the wood grain clearly resolved,
#: and a dark, noisy GoPro night street scene. `aes_sharpness` evidently
#: conflates sharpness with general technical quality — noise and low light
#: drag it down. A ~50% false-positive rate is disqualifying for a backfill
#: that would stamp `blurry` on ten thousand photos.
#:
#: So nothing new produces them, and nothing deletes the ones already stored:
#: deleting would be a second unvalidated decision on top of the first. They
#: stay exactly as they are until a LABELLED eval says otherwise. The right
#: long-term source is a native-resolution Laplacian, not a VLM score —
#: `photosearch/rank_measure.py` already measures exactly that on face crops,
#: which covers photos with faces.
FROZEN_TAGS = frozenset({"sharp", "blurry"})

#: What the model is actually shown. Derived from the compiled vocabulary so a
#: `photosearch compile-vocab` regeneration of `vocab_visual.py` (a GENERATED
#: file) cannot silently desync this split.
PERCEIVED_VOCABULARY: list[str] = [
    t for t in VISUAL_VOCABULARY
    if t not in CAPTURE_FACT_TAGS
    and t not in RETIRED_TAGS
    and t not in FROZEN_TAGS
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
    "golden-hour": "warm low sunrise or sunset",
    "harsh-light": "hard-edged shadows from direct overhead sun",
    "overcast": "flat grey sky, no visible shadows",
    "overexposed": "highlights blown to featureless white",
    "silhouette": "subject rendered as a dark shape",
    "soft-light": "diffuse light, gentle shadow edges",
    "sunny": "direct sunlight and blue sky visible",
    "monochromatic": "one hue family throughout",
    "muted": "desaturated, low-contrast colour",
    "vibrant": "intense saturated colour",
    "dramatic": "strong contrast or visual tension",
    "melancholy": "visibly sombre or sad subject",
    "moody": "dark, brooding, low-key",
    "peaceful": "calm, still, unhurried subject",
    "hazy": "distant detail softened by atmosphere",
    "aerial": "looking down from far above",
    "close-up": "subject fills most of the frame",
    "macro": "extreme magnification of a small object",
    "wide-angle": "visibly stretched wide field of view",
    "centered": "main subject squarely in the middle",
    "composite": "several images combined into one",
    "reflection": "mirrored image in water or glass",
    "split-screen": "frame divided into separate panels",
    "symmetrical": "mirror-image balance across an axis",
}

#: Perceived pairs that cannot both be true of one photo. Used by the guard in
#: `describe.tag_visual_photo`, which drops BOTH members — so a pair that is
#: merely unusual, rather than impossible, DESTROYS CORRECT TAGS. The bar is
#: therefore: *could a competent photographer's single frame honestly be
#: both?* If yes, it does not belong here, however often the model pairs them.
#:
#: Rejected on that test, with the reason:
#:   close-up x wide-angle   an environmental portrait is ordinary
#:   foggy x sunny           sun through fog is a classic shot
#:   joyful x moody          subject's emotion vs how the frame is lit
#:   colorful x monochromatic  a blazing orange sunset reads as both
#:   aerial x close-up       a tight drone crop is both
#:   macro x wide-angle      close-focus wide-angle is a real technique
#:   peaceful x moody        a still, misty lake is honestly both — and it is
#:                           the library's biggest co-occurrence (19,056)
#:
#: What survives contradicts on the SAME property: the sky is grey or it is
#: sunny; shadow edges are hard or soft; the frame is desaturated or intense;
#: black-and-white has no colour; `macro` needs to be close and `aerial` needs
#: to be far; `dramatic` ("strong contrast or visual tension") and `peaceful`
#: ("calm, still") are opposite readings of the same frame.
#:
#: Note there is no `sharp` x `blurry` entry: both are DERIVED now, and
#: `derive_tags` cannot emit them together.
CONTRADICTORY_PAIRS: tuple[tuple[str, str], ...] = (
    ("dramatic", "peaceful"),
    ("joyful", "melancholy"),
    ("overcast", "sunny"),
    ("harsh-light", "soft-light"),
    ("muted", "vibrant"),
    ("colorful", "muted"),
    ("black-and-white", "colorful"),
    ("black-and-white", "vibrant"),
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

#: There is deliberately NO `aes_sharpness` threshold here any more — see
#: FROZEN_TAGS. Do not reintroduce one without a hand-labelled eval.

#: Columns :func:`derive_tags` reads. Anything selecting a row for the merge
#: must include these.
DERIVE_COLUMNS = ("exposure_time", "f_number", "iso",
                  "image_width", "image_height")

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

    # `sharp` / `blurry` are NOT derived — `aes_sharpness` conflates sharpness
    # with general technical quality and produced ~50% false positives on a
    # hand-inspected sample. See FROZEN_TAGS.
    return sorted(out)


def _as_tag_list(perceived: Any) -> Optional[Iterable[str]]:
    """Normalise a tag argument that may arrive as a JSON-array string.

    A bare `str` is NEVER a tag list: iterating it yields characters, none of
    which is in the vocabulary, so the merge would quietly return [] and the
    caller would store an empty array. Raise instead.
    """
    if not isinstance(perceived, str):
        return perceived
    try:
        parsed = json.loads(perceived)
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        return parsed
    raise TypeError(
        "merge_tags expects a list of tags or a JSON-array string, got "
        f"{perceived!r}")


def _normalize(tags: Any) -> set[str]:
    """Lower-cased tag set from a list or a JSON-array string."""
    tags = _as_tag_list(tags)
    return {t.strip().lower() for t in (tags or []) if isinstance(t, str)}


def merge_tags(perceived: Optional[Iterable[str]],
               derived: Optional[Iterable[str]],
               source: str = "vlm",
               carry_frozen: Optional[Iterable[str]] = None) -> list[str]:
    """Combine one set of tags with the derived capture facts.

    The rule, in order:

      1. Every capture-fact term in the input is STRIPPED — derived wins,
         always, including when nothing was derived (an absent capture fact is
         a positive statement: the EXIF says this is not a long exposure, or
         there is no EXIF to say it is).
      2. Retired terms are dropped.
      3. Out-of-vocabulary junk is dropped.
      4. The derived terms are added.
      5. FROZEN terms (`sharp` / `blurry`) depend on where the input came
         from — see `source`.
      6. Sorted + deduped, so the stored JSON is a deterministic function of
         (input, row) and the backfill can skip rows that would not change.

    `source` says what the input IS, and it is the whole frozen-tag question:

      ``"vlm"``     fresh model output (the WRITE path). Frozen terms are
                    dropped — the model cannot judge sharpness from a ~336 px
                    tile. Any frozen term already stored on the photo is
                    carried over via `carry_frozen`, so a re-tag does not
                    silently delete it.
      ``"stored"``  an array already in the column (the BACKFILL path). Frozen
                    terms pass through untouched.

    Either input may be the raw JSON-array STRING straight out of the column,
    in which case it is parsed. Any other string raises: iterating it would
    walk it character by character, match nothing and return [] — a silent
    wipe.
    """
    if source not in ("vlm", "stored"):
        raise ValueError(f"source must be 'vlm' or 'stored', got {source!r}")
    incoming = _normalize(perceived)
    keep = incoming & set(PERCEIVED_VOCABULARY)
    keep |= {t for t in (derived or []) if t in CAPTURE_FACT_TAGS}
    frozen_from = incoming if source == "stored" else _normalize(carry_frozen)
    keep |= frozen_from & set(FROZEN_TAGS)
    return sorted(keep)


def merge_for_row(perceived: Optional[Iterable[str]], row: Any,
                  source: str = "vlm",
                  carry_frozen: Optional[Iterable[str]] = None) -> list[str]:
    """`merge_tags(..., derive_tags(row))` — the whole rule in one call.

    Prefer the two named entry points below; this is the primitive they share,
    so the derived half can never diverge between the worker submit, the M28
    re-run, the in-process `index` passes and the backfill CLI.
    """
    return merge_tags(perceived, derive_tags(row), source=source,
                      carry_frozen=carry_frozen)


def merge_vlm_answer(perceived: Optional[Iterable[str]], row: Any,
                     existing: Optional[Iterable[str]] = None) -> list[str]:
    """WRITE path: a fresh category-visual answer for `row`.

    Frozen terms the model volunteered are dropped; frozen terms already
    stored on the photo (`existing` — pass the current `visual_tags`) are
    carried across. A re-tag REPLACES the perceived half, which is the point,
    but it must not delete a `sharp` / `blurry` nothing has re-decided.
    """
    return merge_for_row(perceived, row, source="vlm", carry_frozen=existing)


def merge_stored_tags(stored: Optional[Iterable[str]], row: Any) -> list[str]:
    """BACKFILL path: an array already in `photos.visual_tags`.

    Frozen terms pass through untouched — the backfill re-decides capture
    facts, nothing else.
    """
    return merge_for_row(stored, row, source="stored")

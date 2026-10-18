"""Ground category-content answers in the description they were extracted from.

The category-content pass hands a text model `photos.description` plus the
377-term `CONTENT_VOCABULARY` and used to store any in-vocabulary term it
returned. Nothing checked that the description supported the term. Photo
251357 (a daylight outdoor boys' soccer description) came back from
gemma-4-12b-qat with `basketball`, `basketball court`, `basketball hoop` and
`artificial lighting` at temperature 0, and `indoor`, `gym`, `gymnasium`,
`kitchen`, `mountain`, `water` on the 0.4 retry. Plan:
`docs/plans/category-grounding.md`.

This module is the ONE rule set. The worker, the server, the backfill and the
eval all call :func:`ground`. It is pure (no I/O), idempotent, and its output
is a subset of its input in input order: it never adds a term.

Two independent judgements live here:

1. **Contradiction groups** (:data:`GROUPS`) DROP terms. A group has two or
   more sides; each side is a set of vocabulary terms plus cue regexes run
   over the description. The description decides:

   - a present side is dropped when a rival side is cued and its own is not,
     INCLUDING when only that side was emitted (251356 stored `artificial
     lighting` alone against "outdoors under trees");
   - two or more sides present and none cued: all present sides are dropped
     (the `visual_tags_derive.CONTRADICTORY_PAIRS` shape, used only when the
     text cannot decide);
   - both cued: both kept (a view out of a window is honestly both).

   Drop-both as the default was rejected on the numbers: it deletes 329
   soccer-group tags across the 124-133 mixed-sport gemma rows, 80-90 of whose
   descriptions say "soccer", and on 251357 it would delete the main subject.

2. **The recitation verdict** (:func:`is_recitation`) never drops a term; it
   judges a whole answer. See its docstring.

Terms in no group are EXEMPT: nothing here checks that `vehicle` or `child`
is supported, so a vocabulary recompile can never silently start dropping a
new term. The required-cue FAMILIES (basketball without a hoop, kitchen
without a stove, ...) are reported by :func:`required_cue_families_diagnostic`
and never drop anything until each passes an owner-labelled holdout at >= 90%.

THE BAR FOR A GROUP (same spirit as CONTRADICTORY_PAIRS): the sides must be
answers to ONE question about the photo that a single frame cannot honestly
answer both ways — which sport is being played, is it indoors or out, is the
light artificial or natural, is the sky clear or grey — AND the description
must normally say which. Rejected as groups, with the reason:

  car interior / car seat / car window / windshield / interior space vs
  outdoor        5 of 5 sampled `car interior` + `outdoor` rows were photos
                 taken from inside a vehicle OF an outdoor scene; `interior
                 space` hits include atriums, plazas and photographs of a
                 photograph. These are NOT indoor-side terms.
  collage        a multi-scene collage is honestly indoor AND outdoor (19 of
                 6,791 indoor+outdoor hits carry it): `collage` skips the
                 setting group entirely.
  `partly cloudy sky` (term) / bare "cloud(s)" (text)   neutral for the
                 weather group.
  baseball cap   clothing, not baseball: "baseball cap" in the text is not a
                 baseball cue and the term is in no sport side.

Cue traps, each pinned by a test:

  "light blue", "light-colored", "daylight", "sunlight" are NOT
  artificial-light cues (a prototype that matched "light" produced 586
  suspect drops). "neon green jersey" is a colour, not a neon light.
  "parked cars" is not a park; "depth of field" is not a field; a
  "Christmas tree" is not an outdoor tree; "inside a car" is not indoors;
  "partly cloudy" is not overcast; "hula hoop" / "hoop earrings" are not
  basketball; "bathroom" is not a bat.

Owner decisions 2026-10-18 (details at INDOOR_VENUE_CUES / PARTLY_CLOUDY_CUES):

  indoor venue   `basketball court` / `basketball hoop` take no part in the
                 sport group when the description cues an indoor setting (gym,
                 indoor, inside, court, hall, arena, ...): they are the literal
                 venue of indoor-gym soccer. 18 of 67 basketball-term drops in
                 the first eval were such photos. `basketball` itself, and the
                 venue terms on outdoor / uncued text, are unaffected.
  partly cloudy  "partly cloudy" / "partly sunny" / "scattered clouds" /
                 "some clouds" WEAKLY cue the cloudy side: they drop only
                 `clear sky`, `clear blue sky`, `clear day`, and protect
                 `cloudy sky`. With a sunny cue too, both sides are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional

from .vocab_content import CONTENT_VOCABULARY

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

_STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "at", "to", "for"}
_IRREGULAR = {"leaves": "leaf", "shelves": "shelf", "children": "child",
              "men": "man", "women": "woman", "knives": "knife",
              "wolves": "wolf", "calves": "calf", "feet": "foot",
              "teeth": "tooth", "geese": "goose", "mice": "mouse"}
_KEEP_S = ("ss", "us", "is")


def _fold(w: str) -> str:
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(_KEEP_S):
        return w[:-1]
    return w


def tokens(text: Optional[str]) -> set[str]:
    """Lowercase content words with possessives dropped and plurals folded.

    Extends `stale_descriptions._tokens` with -ies / -es / irregular folding
    ("skies" -> sky, "benches" -> bench, "children" -> child) and keeps "ss"
    words whole ("glass", not "glas"). Applied to both sides of a comparison.
    """
    if not text:
        return set()
    text = re.sub(r"['’]s\b", "", text.lower())
    return {_fold(w) for w in re.findall(r"[a-z0-9]+", text)
            if len(w) > 1 and w not in _STOP}


# ---------------------------------------------------------------------------
# Contradiction groups
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Side:
    name: str
    terms: tuple[str, ...]
    cues: tuple[str, ...]          # case-insensitive regexes, word-start
    #: WEAK cues: a side cued ONLY by these drops just the rival terms in
    #: `partial_drops`, never a whole rival side ("partly cloudy").
    partial_cues: tuple[str, ...] = ()
    partial_drops: tuple[str, ...] = ()


@dataclass(frozen=True)
class Group:
    name: str
    sides: tuple[Side, ...]
    #: the group is skipped entirely when any of these terms is in the answer
    skip_when: tuple[str, ...] = ()
    #: (terms, cues): when any cue matches, these terms take no part in the
    #: group at all — neither counted as emitted nor dropped.
    exempt_when: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()


# A vehicle after "inside" / before "interior" means a photo taken FROM a
# vehicle, usually of an outdoor scene — not an indoor cue.
_VEHICLE = r"(?:car|cars|vehicle|truck|van|bus|train|plane|airplane|boat|suv|minivan|taxi|cab)"
_DET = r"(?:a|an|the|our|my|his|her|their|its|this|that)"

_INDOOR_CUES: tuple[str, ...] = (
    r"\bindoors?\b",
    rf"\binside\b(?!\s+(?:of\s+)?(?:{_DET}\s+)?(?:parked\s+|moving\s+)?{_VEHICLE}\b)",
    rf"(?<!car )(?<!car's )(?<!vehicle )(?<!vehicle's )\binterior",
    r"\b(?:room|rooms|bedroom|bedrooms|bathroom|classroom|classrooms|"
    r"ballroom|showroom|restroom|playroom|locker room)\b",
    r"\bgym", r"\bkitchen", r"\bceiling", r"\bhallway", r"\bclassroom",
)

#: OWNER DECISION 2026-10-18: `basketball court` / `basketball hoop` are the
#: literal VENUE of indoor-gym soccer (most school gyms are basketball courts),
#: so on an indoor-cued description soccer does not contradict them. Measured
#: on the first eval: 18 of the 67 gemma rows losing a basketball term were
#: indoor-gym soccer photos. `basketball` (the sport) gets no exception, and
#: on an outdoor or uncued description the venue terms drop as before (251357
#: is an outdoor turf field). The venue cues are the setting group's indoor
#: cues plus court / hall / arena.
INDOOR_VENUE_TERMS: tuple[str, ...] = ("basketball court", "basketball hoop")
INDOOR_VENUE_CUES: tuple[str, ...] = _INDOOR_CUES + (
    r"\bcourts?\b", r"\bhalls?\b", r"\barenas?\b")

#: OWNER DECISION 2026-10-18: "partly cloudy" / "partly sunny" / "scattered
#: clouds" / "some clouds" cue the cloudy side, but only WEAKLY: they
#: contradict the clear-sky terms below and nothing else on the sunny side
#: (a partly cloudy day can still be a `sunny day` under a `blue sky`), and
#: they protect `cloudy sky`. A sunny-side cue as well ("partly cloudy but
#: sunny") cues both sides and both are kept. A bare "cloud(s)" stays neutral.
PARTLY_CLOUDY_CUES: tuple[str, ...] = (
    r"\bpartly[\s-]+(?:cloudy|sunny)\b", r"\bscattered\s+(?:\w+\s+)?clouds?\b",
    r"\bsome\s+clouds?\b")
CLEAR_SKY_TERMS: tuple[str, ...] = ("clear sky", "clear blue sky", "clear day")

GROUPS: tuple[Group, ...] = (
    Group("sport", (
        Side("soccer",
             ("soccer", "soccer ball", "soccer game", "soccer goal",
              "soccer jersey", "soccer player", "indoor soccer",
              "soccer indoor", "youth soccer game"),
             (r"\bsoccer", r"\bfootball", r"\bfutsal")),
        Side("basketball",
             ("basketball", "basketball court", "basketball hoop", "backboard"),
             (r"\bbasketball",
              r"(?<!hula )(?<!hula-)\bhoops?\b(?!\s+earrings?)",
              r"\bbackboards?\b")),
        Side("baseball",
             ("baseball", "baseball bat", "baseball game", "baseball glove",
              "baseball uniform"),
             (r"\bbaseball\b(?![\s-]+(?:style\s+)?(?:caps?|hats?)\b)",
              r"\bsoftball", r"\bt-?ball\b", r"\bbats?\b", r"\bbatt(?:ing|ers?)\b",
              r"\bmitts?\b", r"\bdugouts?\b")),
        Side("volleyball", ("volleyball",), (r"\bvolleyball",)),
    ), exempt_when=((INDOOR_VENUE_TERMS, INDOOR_VENUE_CUES),)),
    Group("setting", (
        Side("indoor",
             ("indoor", "indoor soccer", "soccer indoor", "gym", "gymnasium",
              "kitchen", "dining area", "dining table", "living room",
              "restaurant", "floor indoor"),
             _INDOOR_CUES),
        Side("outdoor",
             ("outdoor",),
             (r"\boutdoors?\b", r"\boutside\b", r"\bsky\b", r"\bskies\b",
              r"\bsunlight", r"\bsunny\b", r"\bdaylight",
              r"(?<!depth of )\bfields?\b", r"\bparks?\b",
              r"(?<!christmas )\btrees?\b(?!-(?:shaped|like|patterned|print))", r"\bgrass", r"\blawns?\b",
              r"\bturf\b", r"\bbeach", r"\btrails?\b", r"\b(?:back)?yards?\b",
              r"\bgardens?\b", r"\bforests?\b")),
    ), skip_when=("collage",)),
    Group("light", (
        Side("artificial",
             ("artificial lighting",),
             # NOT "light": "light blue", "light-colored", "natural light".
             (r"\blights\b", r"\blamps?\b", r"\blamplight", r"\blit\b",
              r"\billuminat", r"\bfluorescent", r"\bfloodl(?:ight|it)",
              r"\bspotlight",
              r"\bneon\b(?![\s-]+(?:green|yellow|orange|pink|blue|red|purple|"
              r"lime|colou?r))",
              r"\bchandelier", r"\bbulbs?\b", r"\bflash\b(?!\s+of\b)",
              # "warm ambient lighting", "pendant lighting" (54996, 125482):
              # a bare "lighting" is NOT a cue ("natural lighting").
              r"\b(?:ambient|pendant|overhead|ceiling|stage|string|track|recessed|"
              r"artificial|indoor|studio|led|fairy|christmas|holiday|party|street) "
              r"light(?:ing|s)?\b",
              r"\bnight(?:time)?\b", r"\bevening\b")),
        Side("natural",
             ("natural light", "sunny day", "clear day", "sun"),
             (r"\bdaylight", r"\bsunlight", r"\bsunlit\b", r"\bsunny\b",
              r"\bsunshine", r"\bmidday\b", r"\bblue sk(?:y|ies)\b",
              r"\bclear sk(?:y|ies)\b",
              # 251356: "outdoors under trees" with `artificial lighting`.
              r"\boutdoors?\b", r"\boutside\b")),
    )),
    Group("weather", (
        Side("sunny",
             ("sunny day", "clear sky", "clear blue sky", "clear day",
              "blue sky", "sun"),
             (r"(?<!partly )(?<!partly-)\bsunny\b", r"\bsunlight", r"\bsunlit\b",
              r"\bsunshine",
              r"\bclear sk(?:y|ies)\b", r"\bclear blue sk(?:y|ies)\b",
              r"\bblue sk(?:y|ies)\b", r"\bcloudless",
              # "the sky is clear blue" (10684), "bright daylight" (197539)
              r"\bsk(?:y|ies) (?:is|are|was|were) (?:a )?(?:bright |mostly |completely )?(?:clear|blue|cloudless)\b",
              r"\bbright (?:sun|sunlight|daylight)\b")),
        Side("overcast",
             ("cloudy day", "cloudy sky", "cool cloudy day", "fog"),
             # "partly cloudy" is a WEAK cue (partial_cues); bare "cloud(s)"
             # is neutral.
             (r"\bovercast", r"(?<!partly )(?<!partly-)\bcloudy\b",
              r"\bgr[ae]y sk(?:y|ies)\b", r"\bfog", r"\bmist(?:y|s)?\b",
              r"\brain(?:y|ing|fall|drops?|s)?\b", r"\bdrizzl",
              r"\bgloomy\b", r"\bstorm(?:y|s)?\b"),
             partial_cues=PARTLY_CLOUDY_CUES, partial_drops=CLEAR_SKY_TERMS),
    )),
)


#: Spans blanked out (same length, so offsets survive) before ANY cue runs.
#: Both were found in the first eval run (2026-10-18, gemma current rows):
#:   - negation: "an indoor party or reception space, not outdoors" (67010),
#:     "no outdoor elements". At most two words between the negator and the
#:     cue, so "no people are visible in this outdoor setting" (a positive
#:     statement about an outdoor photo, 174+88+... rows) is NOT blanked.
#:   - a view through a window: "large windows showing greenery outside"
#:     (64727, 54942), "windows overlooking an outdoor area" (55445),
#:     "windows showing darkness outside" (95076). 27 of the 375
#:     `artificial lighting` drops decided by "outdoor"/"outside" alone were
#:     indoor rooms described this way.
SCRUB_PATTERNS: tuple[str, ...] = (
    r"\b(?:not|no|without|nor|neither)\s+(?:\w+\s+){0,2}?"
    r"(?:outdoors?|outside|indoors?|inside|sky|daylight|sunlight|natural light)\b",
    r"\bwindows?\b[^.;]{0,60}?\b(?:outside|outdoors?)\b",
)


def scrub(description: Optional[str]) -> str:
    """The description with negated / through-the-window cue spans blanked."""
    if not description:
        return ""
    out = description
    for p in SCRUB_PATTERNS:
        out = _rx(p).sub(lambda m: " " * len(m.group(0)), out)
    return out


@lru_cache(maxsize=None)
def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def first_cue_match(description: Optional[str],
                    cues: Iterable[str]) -> Optional[re.Match]:
    """The earliest match of any cue in the description, or None."""
    if not description:
        return None
    best = None
    for c in cues:
        m = _rx(c).search(description)
        if m and (best is None or m.start() < best.start()):
            best = m
    return best


def _cued(description: str, side: Side) -> bool:
    return any(_rx(c).search(description) for c in side.cues)


def _partially_cued(description: str, side: Side) -> bool:
    return any(_rx(c).search(description) for c in side.partial_cues)


# ---------------------------------------------------------------------------
# Required-cue families (DIAGNOSTIC ONLY — nothing here drops a term)
# ---------------------------------------------------------------------------

#: Candidate rules of the form "drop this term family when the description has
#: no cue for it". From the planner's draft; each is promoted into a dropping
#: rule only after >= 90% correct drops on a fresh owner-labelled holdout.
#: Until then :func:`required_cue_families_diagnostic` only reports them, and
#: the verdict uses them as "supported" evidence.
REQUIRED_CUE_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "basketball": (("basketball", "basketball court", "basketball hoop", "backboard"),
                   (r"\bbasketball", r"(?<!hula )\bhoops?\b(?!\s+earrings?)",
                    r"\bbackboards?\b")),
    "baseball": (("baseball", "baseball bat", "baseball game", "baseball glove",
                  "baseball uniform", "bat"),
                 (r"\bbaseball\b(?![\s-]+(?:caps?|hats?)\b)", r"\bsoftball",
                  r"\bt-?ball\b", r"\bbats?\b", r"\bbatt(?:ing|ers?)\b",
                  r"\bgloves?\b", r"\bmitts?\b", r"\bpitch", r"\bdugouts?\b",
                  r"\bhome plate", r"\binfield")),
    "goal": (("goal", "soccer goal", "goal net", "goal post"),
             (r"\bgoal", r"\bnets?\b")),
    "kitchen": (("kitchen", "countertop"),
                (r"\bkitchen", r"\bcounter", r"\bstoves?\b", r"\bovens?\b",
                 r"\bsinks?\b", r"\bfridge", r"\brefrigerator", r"\bcabinet",
                 r"\bmicrowave")),
    "gym": (("gym", "gymnasium", "indoor soccer", "soccer indoor"),
            (r"\bgym", r"\bindoors?\b", r"\binside\b", r"\bcourts?\b",
             r"\bhalls?\b", r"\barena", r"\bdome")),
    "water": (("water", "ocean", "sea", "lake", "river", "stream", "waterfall",
               "wave", "water edge", "shore", "shoreline", "rocky shoreline",
               "beach", "sandy beach", "boat", "underwater environment", "coral",
               "coral reef", "marine life", "fish", "large body"),
              (r"\bwater", r"\boceans?\b", r"\bseas?\b", r"\bseaside", r"\blakes?\b",
               r"\brivers?\b", r"\bstreams?\b", r"\bcreeks?\b", r"\bponds?\b",
               r"\bpools?\b", r"\bwaves?\b", r"\bsurf", r"\bbeach", r"\bshore",
               r"\bcoast", r"\bbays?\b", r"\bharbou?r", r"\bboats?\b", r"\bkayak",
               r"\bcanoe", r"\bunderwater", r"\breefs?\b", r"\baquarium",
               r"\bfish", r"\bswim", r"\bpuddle", r"\bfountain", r"\bmarina",
               r"\bdocks?\b", r"\bpiers?\b", r"\blagoon")),
    "mountain": (("mountain", "mountainous area", "mountainous landscape",
                  "alpine", "cliff", "valley"),
                 (r"\bmountain", r"\bpeaks?\b", r"\bsummit", r"\bridge",
                  r"\balpine", r"\bcliff", r"\bvalley", r"\branges?\b",
                  r"\bcanyon", r"\bhills?\b", r"\bhillside", r"\bbluffs?\b",
                  r"\bmesas?\b")),
    "snow": (("snow", "winter"),
             (r"\bsnow", r"\bic(?:e|y)\b", r"\bwinter", r"\bski(?:s|ing|ers?)?\b",
              r"\bsled", r"\bfrost", r"\bfrozen")),
    "city": (("city", "cityscape", "urban environment", "coastal cityscape",
              "alley", "highway"),
             (r"\bcity", r"\bcities", r"\burban", r"\bdowntown", r"\bskyline",
              r"\bskyscraper", r"\bstreets?\b", r"\bhighway", r"\bfreeway",
              r"\btraffic", r"\balley")),
    "road": (("road", "street", "sidewalk"),
             (r"\broads?\b", r"\broadside", r"\bstreets?\b", r"\bsidewalk",
              r"\bhighway", r"\blanes?\b", r"\bavenue", r"\bcrosswalk",
              r"\bpavement", r"\bdriveway", r"\bparking lot", r"\basphalt")),
    "building": (("building", "architecture"),
                 (r"\bbuilding", r"\barchitect", r"\bhouses?\b", r"\bhomes?\b",
                  r"\btowers?\b", r"\bchurch", r"\btemple", r"\bcastle",
                  r"\bskyscraper", r"\bstructure", r"\bfa[cç]ade", r"\bbarns?\b",
                  r"\bsheds?\b", r"\bschool", r"\bstadium", r"\bapartment",
                  r"\boffice", r"\bhotel", r"\bmuseum")),
    "celebration": (("celebration",),
                    (r"\bcelebrat", r"\bpart(?:y|ies)\b", r"\bbirthday", r"\bcakes?\b",
                     r"\bballoon", r"\bcheer", r"\bwedding", r"\bholiday",
                     r"\bfestive", r"\btoast", r"\bconfetti", r"\bchristmas",
                     r"\bhalloween", r"\bgraduat", r"\btroph", r"\bmedals?\b",
                     r"\bhigh[- ]fiv")),
    "group portrait": (("group portrait",),
                       (r"\bpos(?:e|es|ed|ing)\b", r"\bportrait",
                        r"\bgroup (?:photo|shot|picture)",
                        r"\bsmil\w*\s+(?:at|toward|towards|for|into)\s+the\s+camera",
                        r"\blooking\s+(?:at|into|toward|towards)\s+the\s+camera",
                        r"\blined\s+up", r"\bhuddl",
                        r"\bteam\s+(?:photo|picture|portrait)")),
    "coach": (("coach",),
              (r"\bcoach", r"\binstructor", r"\btrainer", r"\bclipboard")),
    "guest": (("guest",),
              (r"\bguest", r"\bwedding", r"\bpart(?:y|ies)\b", r"\breception",
               r"\bvisitor", r"\bevents?\b", r"\battendee")),
    "hat": (("hat",),
            (r"\bhats?\b", r"\bcaps?\b", r"\bbeanie", r"\bvisor", r"\bbonnet",
             r"\bhood(?:s|ed)?\b")),
    "shadow": (("shadow",), (r"\bshadow", r"\bshade")),
    "court": (("court",), (r"\bcourts?\b",)),
    "wall": (("wall", "white wall", "stone wall"),
             (r"\bwalls?\b", r"\bmural", r"\bbrick", r"\bbackdrop", r"\brooms?\b",
              r"\binterior", r"\bindoors?\b")),
    "television": (("television", "screen"),
                   (r"\btelevision", r"\btvs?\b", r"\bscreens?\b",
                    r"\bmonitors?\b", r"\bdisplay")),
}


def required_cue_families_diagnostic(description: Optional[str],
                                     categories: list[str]) -> dict[str, list[str]]:
    """{family: [terms it WOULD drop]} for families with no cue in the text.

    Reports only; drops nothing. Families with nothing to drop are omitted. A
    None / empty description reports nothing (there is nothing to judge by).
    """
    if not description or not description.strip():
        return {}
    description = scrub(description)
    out: dict[str, list[str]] = {}
    for fam, (terms, cues) in REQUIRED_CUE_FAMILIES.items():
        hit = [t for t in categories if t in terms]
        if hit and first_cue_match(description, cues) is None:
            out[fam] = hit
    return out


# ---------------------------------------------------------------------------
# Recitation verdict
# ---------------------------------------------------------------------------

RECITATION_MIN_ORDER = 0.9
RECITATION_MIN_UNSUPPORTED = 10
RECITATION_MIN_UNSUPPORTED_SHARE = 0.5
#: Count backstop for the worker, judged on the REPAIRED list. gemma's worst
#: observed answer was 84; the 256-token text budget makes > 100 nearly
#: impossible on LM Studio. This mainly guards the Ollama route.
MAX_PLAUSIBLE = 120

_CONCRETE = (
    "backpack", "baby bottle", "ball", "balcony", "bamboo forest", "bar",
    "bed", "bench", "bird", "blanket", "bleacher", "boat", "book", "bottle",
    "bouquet", "bowl", "box", "branch", "bridge", "bush", "cake", "camera",
    "cap", "car", "chain link fence", "chair", "christmas tree", "cleat",
    "cloud", "computer", "couch", "countertop", "crib", "cup", "decoration",
    "deer", "desk", "dining table", "dog", "door", "drink", "elk", "farm",
    "fence", "field", "fish", "flower", "food", "forest", "fox",
    "framed picture", "furniture", "garden", "glass", "goal net", "goal post",
    "grass", "helmet", "high chair", "house", "insect", "lake", "large window",
    "leaf", "life jacket", "marmot", "meal", "meat", "medicine bottle",
    "monkey", "net", "ocean", "pacifier", "paper", "path", "phone",
    "pink flower", "plant", "plate", "playground", "potted plant", "pumpkin",
    "purple flower", "railing", "river", "rock", "sand", "santa hat",
    "scoreboard", "screen", "shelf", "shin guard", "sign", "smartphone",
    "snow", "sock", "spoon", "stage", "statue", "stone wall", "stroller",
    "sun", "sunglass", "swing", "table", "tank", "television", "tent", "tie",
    "toy", "toy car", "train", "tree", "tree branch", "turf", "vine",
    "vineyard", "wall", "waterfall", "white flower", "white wall", "window",
    "windshield", "wine glass", "wooden fence", "yard",
)

#: The ~130 concrete nouns the verdict counts as "objects". Used ONLY for the
#: recitation verdict, never to drop a term: its per-term imprecision averages
#: out over an answer.
#:
#: The required-cue FAMILY terms (building, architecture, city, mountain,
#: court, coach, group portrait, celebration, guest, ...) are deliberately NOT
#: objects, although the plan first specified "plus every family term". Those
#: terms are exactly gemma's padding vocabulary: counted as objects, the
#: verdict flagged 23 of 7,769 current gemma rows (2026-10-18 eval), every one
#: an honest reading of a wedding, soccer or rural scene padded with 10-15
#: extra terms, and it flagged the 68-tag 251357 answer. Concrete nouns only:
#: 1 gemma row flagged, 141/141 llama recitations of >= 53 tags caught (was
#: 140/141), 121 of the 702 llama 20-52-tag lists flagged (was 156). Llama
#: recites household nouns (bed, book, bottle, bowl, box, ...); gemma pads
#: with scene words. Do not add family terms back without re-running
#: evals/category_grounding_eval.py.
OBJECT_TERMS: frozenset[str] = frozenset(_CONCRETE)

_VOCAB_INDEX = {t: i for i, t in enumerate(CONTENT_VOCABULARY)}


def _term_cues(term: str) -> tuple[str, ...]:
    cues: list[str] = []
    for terms, fam_cues in REQUIRED_CUE_FAMILIES.values():
        if term in terms:
            cues.extend(fam_cues)
    for g in GROUPS:
        for s in g.sides:
            if term in s.terms:
                cues.extend(s.cues)
    return tuple(cues)


_TERM_CUES = {t: _term_cues(t) for t in CONTENT_VOCABULARY}


def is_supported(description: str, term: str,
                 desc_tokens: Optional[set[str]] = None) -> bool:
    """The term's own words are all in the description, or a family / group
    cue for it matches."""
    dt = desc_tokens if desc_tokens is not None else tokens(description)
    tt = tokens(term)
    if tt and tt <= dt:
        return True
    cues = _TERM_CUES.get(term) or _term_cues(term)
    return bool(description) and any(_rx(c).search(description) for c in cues)


def order_share(categories: list[str]) -> float:
    """Share of adjacent pairs ascending in vocabulary order (in-vocab terms
    only); 0.0 with fewer than two."""
    idx = [_VOCAB_INDEX[t] for t in categories if t in _VOCAB_INDEX]
    if len(idx) < 2:
        return 0.0
    return sum(1 for a, b in zip(idx, idx[1:]) if b > a) / (len(idx) - 1)


def recitation_stats(description: Optional[str],
                     categories: list[str]) -> dict:
    """The verdict's inputs, for logging and the eval."""
    description = scrub(description)
    dt = tokens(description)
    obj = [t for t in categories if t in OBJECT_TERMS]
    unsup = [t for t in obj if not is_supported(description or "", t, dt)]
    return {"order": order_share(categories), "obj": len(obj),
            "unsupported_obj": len(unsup), "unsupported": unsup}


def is_recitation(description: Optional[str], categories: list[str]) -> bool:
    """Is this answer a vocabulary recitation rather than a reading of the text?

        order >= 0.9 AND unsupported_obj >= 10 AND unsupported_obj / obj >= 0.5

    `order` is the share of adjacent pairs ascending in vocabulary order;
    `obj` the answer's terms in :data:`OBJECT_TERMS` (concrete nouns only — see
    its comment for why family terms are excluded); "unsupported" means the
    term's own words are absent from the description and no family / group cue
    matches. Neither signal works alone: 72% of gemma lists are >= 0.9 in
    vocabulary order (median 0.96) against 0.98 for llama recitations, and
    only 41% of gemma tags appear literally in the description.

    A None / empty description is never a recitation: there is nothing to
    judge it against, and :func:`ground` stores such a list unchanged.
    """
    if not description or not description.strip():
        return False
    s = recitation_stats(description, categories)
    return (s["order"] >= RECITATION_MIN_ORDER
            and s["unsupported_obj"] >= RECITATION_MIN_UNSUPPORTED
            and s["obj"] > 0
            and s["unsupported_obj"] / s["obj"] >= RECITATION_MIN_UNSUPPORTED_SHARE)


# ---------------------------------------------------------------------------
# ground()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grounding:
    kept: list[str]
    dropped: dict[str, str] = field(default_factory=dict)   # term -> reason
    recitation: bool = False


def _names(sides: Iterable[Side]) -> str:
    return ", ".join(s.name for s in sides)


def group_drops(description: str, categories: list[str]) -> dict[str, str]:
    """{term: reason} for every term a contradiction group drops."""
    description = scrub(description)
    present_terms = set(categories)
    dropped: dict[str, str] = {}
    for g in GROUPS:
        if any(t in present_terms for t in g.skip_when):
            continue
        exempt: set[str] = set()
        for terms, cues in g.exempt_when:
            if any(_rx(c).search(description) for c in cues):
                exempt.update(terms)
        live = present_terms - exempt
        present = [s for s in g.sides if live.intersection(s.terms)]
        if not present:
            continue
        strong = [s for s in g.sides if _cued(description, s)]
        partial = [s for s in g.sides
                   if s not in strong and _partially_cued(description, s)]
        cued_names = {s.name for s in strong + partial}
        for s in present:
            if s.name in cued_names:
                continue
            only: Optional[set[str]] = None      # None = the whole side
            if strong:
                reason = f"{g.name}: {_names(strong + partial)} cued, {s.name} not"
            elif partial:
                only = {t for p in partial for t in p.partial_drops}
                reason = (f"{g.name}: {_names(partial)} weakly cued "
                          f"(partly cloudy), {s.name} not")
            elif len(present) >= 2:
                reason = f"{g.name}: {_names(present)} all emitted, none cued"
            else:
                continue
            for t in categories:
                if (t in s.terms and t in live and t not in dropped
                        and (only is None or t in only)):
                    dropped[t] = reason
    return dropped


def ground(description: Optional[str], categories: list[str]) -> Grounding:
    """Drop the terms the description contradicts; judge recitation.

    Pure and idempotent; `kept` is a subset of `categories` in input order.
    A None / empty description returns the list unchanged, nothing dropped.
    """
    cats = list(categories)
    if not description or not description.strip():
        return Grounding(kept=cats, dropped={}, recitation=False)
    dropped = group_drops(description, cats)
    kept = [t for t in cats if t not in dropped]
    return Grounding(kept=kept, dropped=dropped,
                     recitation=is_recitation(description, cats))

"""Search logic for local-photo-search.

Combines CLIP semantic search, text search, color search, and face/person search.
"""

import json
import logging
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .clip_embed import embed_text
from .db import PhotoDB

# Debug logger for step-by-step search tracing
_log = logging.getLogger("photosearch.search")


# Minimum CLIP similarity score (1 - L2_distance) required to return a result.
# For L2-normalized 512-dim CLIP vectors, L2 distance = sqrt(2*(1-cos_sim)), so:
#   score -0.25 ≈ cosine similarity 0.22  (loose relevance)
#   score  0.00 ≈ cosine similarity 0.50  (clearly related)
# Queries like "ships" against unrelated family photos score ≈ -0.30, well below
# this threshold, so they return nothing rather than the whole collection.
CLIP_MIN_SCORE = -0.20

# Face-aware reranking: when a query mentions people, photos with detected faces
# get a score boost. This helps CLIP distinguish "people outdoors" from "outdoors"
# — something it can't do from embeddings alone when all photos are visually similar.
_PEOPLE_KEYWORDS = {
    "people", "person", "child", "children", "kid", "kids", "family",
    "man", "woman", "boy", "girl", "baby", "toddler", "adult",
    "group", "crowd", "portrait", "face", "faces",
}
FACE_BOOST = 0.02  # Score bonus per detected face (enough to lift a photo above
                    # neighbors in a tight cluster, but not so much that a random
                    # face-having photo leapfrogs a genuinely relevant result).
DESCRIPTION_BOOST = 0.05  # Score bonus when the photo's LLaVA description contains
                          # query-relevant content. Larger than FACE_BOOST because a
                          # text match is strong evidence of relevance.
DESCRIPTION_PENALTY = 0.04  # Score penalty when a description explicitly negates
                            # something the query asks for (e.g. "no people" when
                            # searching "people outdoors"). This pushes landscape
                            # photos below similarly-scored people photos.
DESCRIPTION_ABSENCE_PENALTY = 0.02  # Smaller penalty when a description exists but
                                    # contains NONE of the query words. LLaVA described
                                    # the photo and didn't see anything related to the
                                    # query — a weak but useful negative signal.
CLIP_MIN_FOR_DESC_BOOST = -0.05  # Don't apply description boost unless CLIP score is
                                 # at least this high. Prevents hallucinated descriptions
                                 # from surfacing visually irrelevant photos.
CLIP_ONLY_MIN_SCORE = 0.15  # When a query DOES produce text (keyword/description/tag)
                            # matches, hold CLIP-only candidates (no text match) to this
                            # stricter floor so weak/negative visual neighbours don't
                            # pollute results. Pure-visual queries (no text matches at
                            # all) keep the permissive CLIP_MIN_SCORE floor.

# Negation patterns: if the description contains "no <keyword>" or "no visible <keyword>"
# and the query mentions that keyword, the photo gets a penalty instead of a boost.
_NEGATION_PREFIXES = ("no ", "no visible ", "without ", "absence of ")

# Phrases that negate people in general, regardless of the specific keyword searched.
# Checked when the query mentions people-related keywords.
_NEGATION_PEOPLE_PHRASES = (
    "no one", "nobody", "no people", "no visible people", "no individuals",
    "no person", "no humans", "without people", "absence of people",
    "no visible human", "no human", "empty beach", "empty street",
    "empty park", "empty trail", "empty path",
    "untouched by people", "devoid of people", "devoid of human",
    # Note: "no other people" deliberately excluded — it means "one person present,
    # no additional ones" which is a positive signal for people queries.
)

# Regex-based negation: catches patterns like "no visible presence of people",
# "there is no ... people", etc. where a rigid phrase list would miss.
import re
_NEGATION_PEOPLE_RE = re.compile(
    r"\bno\b.{0,30}\b(?:people|person|humans?|individuals?|one)\b"
    r"|\b(?:without|absence of|devoid of|untouched by).{0,20}\b(?:people|person|humans?)\b"
    r"|\bnobody\b"
    r"|\bempty\s+(?:beach|street|park|trail|path)\b",
    re.IGNORECASE,
)
# Exclude "no other people" — means one person present, which is a positive signal.
_FALSE_NEGATION_RE = re.compile(r"\bno other\b", re.IGNORECASE)


# Filename detection: single token (no spaces), optionally ending in a photo extension.
# Matches camera naming conventions: DSC06241, IMG_1234, P1020304, DSC_0001, etc.
# Deliberately permissive — if no filename match is found we fall through to CLIP anyway.
_FILENAME_STEM_RE = re.compile(
    r'^[A-Za-z]{0,5}[\d_]{2,}[A-Za-z\d_]*$'
)
_PHOTO_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".arw", ".raw", ".dng",
    ".nef", ".cr2", ".cr3", ".heic", ".heif", ".tif", ".tiff",
    ".mov", ".mp4",
}


def _looks_like_filename(query: str) -> bool:
    """Return True if the query looks like a camera filename or filename stem.

    Examples that match: DSC06241, IMG_1234, DSC06241.JPG, P1020304
    Examples that don't: beach, sunset at the lake, R2D2 (falls through to CLIP)
    """
    q = query.strip()
    if not q or " " in q:
        return False
    # Strip a trailing photo extension
    stem = q
    suffix = Path(q).suffix.lower()
    if suffix in _PHOTO_EXTENSIONS:
        stem = Path(q).stem
    return bool(_FILENAME_STEM_RE.match(stem))


def search_by_filename(db: PhotoDB, query: str, limit: int = 50) -> list[dict]:
    """Search for photos by filename substring match (case-insensitive LIKE).

    Strips any trailing photo extension from the query so that both
    'DSC06241' and 'DSC06241.JPG' find the same photo.
    Returns results ordered by date_taken descending.
    """
    stem = query.strip()
    suffix = Path(stem).suffix.lower()
    if suffix in _PHOTO_EXTENSIONS:
        stem = Path(stem).stem
    pattern = f"%{stem}%"
    # id-first: an infix LIKE can't seek, but it can scan the narrow UNIQUE
    # filepath index instead of the whole table (560 -> 9 MB). filepath
    # always ends in filename, so matching filepath alone is exact.
    rows = db.conn.execute(
        """SELECT * FROM photos
           WHERE id IN (SELECT id FROM photos WHERE filepath LIKE ?)
           ORDER BY date_taken DESC
           LIMIT ?""",
        (pattern, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _dedupe_by_hash(results: list[dict]) -> list[dict]:
    """Remove duplicate photos (same file indexed from multiple paths).

    Keeps the first occurrence (highest score if pre-sorted).
    """
    seen: set[str] = set()
    out = []
    for r in results:
        h = r.get("file_hash")
        if h and h in seen:
            continue
        if h:
            seen.add(h)
        out.append(r)
    return out


# Patterns that negate people in the QUERY itself (not the description).
# "no people", "without people", "no kids", etc.
_QUERY_NEGATES_PEOPLE_RE = re.compile(
    r"\b(?:no|without|exclude|excluding)\s+(?:"
    + "|".join(re.escape(kw) for kw in sorted(_PEOPLE_KEYWORDS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def _parse_query(query: str) -> tuple[str, list[str]]:
    """Parse a query into positive terms and excluded terms.

    Supports two negation syntaxes:
      - Dash prefix:      "beach -people -dogs"
      - Natural language:  "no people", "without kids"

    Returns (positive_query, excluded_terms) where:
      - positive_query: the query to send to CLIP (without negation tokens)
      - excluded_terms: lowercased words that must NOT appear in descriptions
    """
    excluded: list[str] = []

    # Extract -term tokens
    tokens = query.split()
    positive_tokens = []
    for token in tokens:
        if token.startswith("-") and len(token) > 1:
            word = token[1:].lower()
            excluded.append(word)
            # Also expand people-related exclusions: -people should exclude
            # all people keywords so "person", "child", etc. also get filtered
            if word in _PEOPLE_KEYWORDS:
                excluded.extend(_PEOPLE_KEYWORDS)
        else:
            positive_tokens.append(token)

    # Extract "no <word>" / "without <word>" natural language negation
    remaining = " ".join(positive_tokens)
    nl_neg_re = re.compile(
        r"\b(?:no|without|exclude|excluding)\s+(\w+)\b", re.IGNORECASE,
    )
    for m in nl_neg_re.finditer(remaining):
        word = m.group(1).lower()
        excluded.append(word)
        if word in _PEOPLE_KEYWORDS:
            excluded.extend(_PEOPLE_KEYWORDS)

    # Build positive query: strip out "no <word>" / "without <word>" phrases
    # so CLIP only sees the positive intent
    positive_query = nl_neg_re.sub("", remaining).strip()
    # Clean up extra whitespace
    positive_query = re.sub(r"\s+", " ", positive_query).strip()

    return positive_query, list(set(excluded))


def _query_mentions_people(query: str) -> bool:
    """Return True if the query contains people-related keywords."""
    words = set(query.lower().split())
    return bool(words & _PEOPLE_KEYWORDS)


def _query_negates_people(query: str) -> bool:
    """Return True if the query is asking for the ABSENCE of people.

    "no people", "without kids", "empty beach no people" → True
    "people outdoors", "kids playing" → False
    """
    return bool(_QUERY_NEGATES_PEOPLE_RE.search(query))


def _description_contains_excluded(description: str, excluded: list[str]) -> bool:
    """Return True if the description contains any excluded term.

    Uses stem matching (strip trailing 's') for basic plural handling.
    For people-related exclusions, also checks that the term isn't negated
    in the description (e.g. "no people visible" should NOT be excluded).
    """
    if not excluded or not description:
        return False

    desc_lower = description.lower()

    # Check if the description negates people — if so, people-related
    # excluded terms should NOT cause a filter-out.
    desc_negates_people = bool(
        _NEGATION_PEOPLE_RE.search(desc_lower)
        and not _FALSE_NEGATION_RE.search(
            _NEGATION_PEOPLE_RE.search(desc_lower).group()
        )
    )

    for term in excluded:
        # Check the term itself (with basic stem) using word boundaries
        stem = term.rstrip("s") if len(term) > 4 else term
        if re.search(r'\b' + re.escape(stem) + r'\b', desc_lower):
            # If this is a people term and the description negates people,
            # don't count it as a match (the description says "no people")
            if term in _PEOPLE_KEYWORDS and desc_negates_people:
                continue
            return True
        # Check expanded terms (e.g. excluding "animal" also excludes "bird")
        expansions = _expand_query_word(term)
        if len(expansions) > 1:
            for exp_term in expansions:
                if re.search(r'\b' + re.escape(exp_term) + r'\b', desc_lower):
                    if term in _PEOPLE_KEYWORDS and desc_negates_people:
                        continue
                    return True
    return False


# ---------------------------------------------------------------------------
# Semantic term expansion
# ---------------------------------------------------------------------------
# When a user searches for a category word like "animal", the description might
# say "bird", "elk", or "shark" — correct matches that literal word matching
# would miss. This dictionary maps category terms to their common members so
# the description scorer can recognize them. CLIP already handles this for
# visual similarity; this makes the description boost/penalty logic match.
#
# Each key is a search term; its value is a set of words that should count as
# a match for that term when found in a description.

_TERM_EXPANSIONS: dict[str, set[str]] = {
    # Animals — broad category
    "animal": {
        "bird", "birds", "dog", "dogs", "cat", "cats", "fish", "deer", "elk",
        "moose", "bear", "shark", "whale", "dolphin", "horse", "cow", "sheep",
        "goat", "pig", "rabbit", "squirrel", "fox", "wolf", "eagle", "hawk",
        "owl", "heron", "pelican", "seagull", "gull", "duck", "goose", "swan",
        "turtle", "frog", "snake", "lizard", "insect", "butterfly", "bee",
        "spider", "crab", "lobster", "octopus", "seal", "otter", "raccoon",
        "chipmunk", "mouse", "rat", "bat", "penguin", "flamingo", "parrot",
        "chicken", "rooster", "turkey", "pigeon", "crow", "raven", "jay",
        "cardinal", "sparrow", "finch", "woodpecker", "hummingbird",
        "animal", "animals", "wildlife", "creature", "pet",
    },
    "animals": {  # plural form maps to the same set
        "bird", "birds", "dog", "dogs", "cat", "cats", "fish", "deer", "elk",
        "moose", "bear", "shark", "whale", "dolphin", "horse", "cow", "sheep",
        "goat", "pig", "rabbit", "squirrel", "fox", "wolf", "eagle", "hawk",
        "owl", "heron", "pelican", "seagull", "gull", "duck", "goose", "swan",
        "turtle", "frog", "snake", "lizard", "insect", "butterfly", "bee",
        "spider", "crab", "lobster", "octopus", "seal", "otter", "raccoon",
        "chipmunk", "mouse", "rat", "bat", "penguin", "flamingo", "parrot",
        "chicken", "rooster", "turkey", "pigeon", "crow", "raven", "jay",
        "cardinal", "sparrow", "finch", "woodpecker", "hummingbird",
        "animal", "animals", "wildlife", "creature", "pet",
    },
    "wildlife": {
        "bird", "birds", "deer", "elk", "moose", "bear", "fox", "wolf",
        "eagle", "hawk", "owl", "heron", "seal", "otter", "raccoon",
        "squirrel", "chipmunk", "whale", "dolphin", "shark",
        "wildlife", "animal", "animals", "creature",
    },
    # Birds
    "bird": {
        "eagle", "hawk", "owl", "heron", "pelican", "seagull", "gull",
        "duck", "goose", "swan", "penguin", "flamingo", "parrot", "pigeon",
        "crow", "raven", "jay", "cardinal", "sparrow", "finch", "woodpecker",
        "hummingbird", "chicken", "rooster", "turkey",
        "bird", "birds", "avian", "waterfowl", "songbird",
    },
    "birds": {
        "eagle", "hawk", "owl", "heron", "pelican", "seagull", "gull",
        "duck", "goose", "swan", "penguin", "flamingo", "parrot", "pigeon",
        "crow", "raven", "jay", "cardinal", "sparrow", "finch", "woodpecker",
        "hummingbird", "chicken", "rooster", "turkey",
        "bird", "birds", "avian", "waterfowl", "songbird",
    },
    # Pets
    "pet": {"dog", "dogs", "cat", "cats", "puppy", "kitten", "fish",
            "rabbit", "hamster", "pet", "pets"},
    "pets": {"dog", "dogs", "cat", "cats", "puppy", "kitten", "fish",
             "rabbit", "hamster", "pet", "pets"},
    # Vehicles
    "vehicle": {"car", "truck", "bus", "van", "motorcycle", "bike", "bicycle",
                "boat", "ship", "train", "plane", "airplane", "helicopter",
                "vehicle", "vehicles"},
    "vehicles": {"car", "truck", "bus", "van", "motorcycle", "bike", "bicycle",
                 "boat", "ship", "train", "plane", "airplane", "helicopter",
                 "vehicle", "vehicles"},
    # Water / bodies of water
    "water": {"ocean", "sea", "lake", "river", "stream", "creek", "pond",
              "waterfall", "waves", "water", "beach", "shore", "coast"},
    # Flowers / plants
    "flower": {"rose", "daisy", "sunflower", "tulip", "lily", "orchid",
               "wildflower", "blossom", "bloom", "petal", "flower", "flowers",
               "floral", "bouquet"},
    "flowers": {"rose", "daisy", "sunflower", "tulip", "lily", "orchid",
                "wildflower", "blossom", "bloom", "petal", "flower", "flowers",
                "floral", "bouquet"},
    "plant": {"tree", "bush", "shrub", "fern", "moss", "vine", "grass",
              "cactus", "succulent", "flower", "plant", "plants", "vegetation",
              "foliage", "leaf", "leaves"},
    "plants": {"tree", "bush", "shrub", "fern", "moss", "vine", "grass",
               "cactus", "succulent", "flower", "plant", "plants", "vegetation",
               "foliage", "leaf", "leaves"},
    # Food
    "food": {"meal", "dish", "plate", "fruit", "vegetable", "bread", "cake",
             "pizza", "sandwich", "salad", "soup", "rice", "pasta", "meat",
             "fish", "dessert", "snack", "food", "eating", "cooking"},
}


def _expand_query_word(word: str) -> set[str]:
    """Return the set of terms that should count as a match for a query word.

    If the word has expansions, returns those. Otherwise returns just
    the word itself (with basic stem).
    """
    lower = word.lower()
    if lower in _TERM_EXPANSIONS:
        return _TERM_EXPANSIONS[lower]
    return {lower}


# ---------------------------------------------------------------------------
# Tag-based matching (M9) — replaces dictionary expansion with LLM tags
# ---------------------------------------------------------------------------
# _QUERY_TO_CATEGORIES: maps query words to sets of category labels for the
# new _categories_match_query scorer (Bundle E / M23).  Loaded from the
# generated vocab_query_expansion module if present; graceful empty default
# when the curator hasn't filled in expansions yet (the initial v23 ship has
# an empty dict).
try:
    from .vocab_query_expansion import _QUERY_TO_CATEGORIES
except ImportError:
    _QUERY_TO_CATEGORIES: dict[str, set[str]] = {}


def _categories_match_query(categories_json: Optional[str], query: str) -> float:
    """Score a categories array against a free-text query.

    Tiers:
      +1.0  query (lowercased, full) matches a category exactly
      +0.5  per query word that matches a category exactly
      +0.4  per query word whose _QUERY_TO_CATEGORIES expansion intersects categories
    """
    if not categories_json or not query:
        return 0.0
    try:
        cats = set(json.loads(categories_json))
    except (ValueError, TypeError):
        return 0.0
    if not cats:
        return 0.0
    q_lower = query.lower().strip()
    if not q_lower:
        return 0.0
    score = 0.0
    if q_lower in cats:
        score += 1.0
    for word in q_lower.split():
        if word in cats:
            score += 0.5
        expansion = _QUERY_TO_CATEGORIES.get(word, set())
        if expansion & cats:
            score += 0.4
    return score


def _visual_match_query(visual_json: Optional[str], query: str) -> float:
    """Score a visual_tags array against a query. Tiers: 0.8 whole, 0.4 per word."""
    if not visual_json or not query:
        return 0.0
    try:
        tags = set(json.loads(visual_json))
    except (ValueError, TypeError):
        return 0.0
    if not tags:
        return 0.0
    q_lower = query.lower().strip()
    if not q_lower:
        return 0.0
    score = 0.0
    if q_lower in tags:
        score += 0.8
    for word in q_lower.split():
        if word in tags:
            score += 0.4
    return score


def _stem_eq(a: str, b: str) -> bool:
    """Loose word equality tolerant of a short plural/inflection suffix.

    True when the words are equal, or the shorter is a prefix of the longer
    with a length difference <= 2 (marmot/marmots, dog/dogs) — but only for
    stems of length >= 4 so short tokens don't over-match. This is a WHOLE-WORD
    test: it never matches "arm" against "marmot" (which naive substring did).
    """
    if a == b:
        return True
    lo, hi = sorted((a, b), key=len)
    return len(lo) >= 4 and len(hi) - len(lo) <= 2 and hi.startswith(lo)


def _phrase_subset(short: str, long_: str) -> bool:
    """True if every word of `short` appears (stem-aware) as a whole word in
    `long_`. Catches "marmot" ⊂ "marmot burrow" without matching sub-word
    substrings like "arm" ⊂ "marmot"."""
    long_words = long_.split()
    return all(any(_stem_eq(w, lw) for lw in long_words) for w in short.split())


def _keywords_match_query(keywords_json: Optional[str], query: str) -> float:
    """Score a keywords array against a query.

    Tiers per keyword:
      +1.2  whole-query equals the keyword
      +0.6  query is a whole-word (stem-aware) subset of the keyword, or vice
            versa — e.g. "marmot" ⊂ "marmot burrow" (NOT "arm" ⊂ "marmot")
      +0.3  any token shared between query and the keyword's tokens
    """
    if not keywords_json or not query:
        return 0.0
    try:
        keywords = json.loads(keywords_json)
    except (ValueError, TypeError):
        return 0.0
    if not keywords:
        return 0.0
    q_lower = query.lower().strip()
    if not q_lower:
        return 0.0
    score = 0.0
    q_words = set(q_lower.split())
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l == q_lower:
            score += 1.2
        elif _phrase_subset(q_lower, kw_l) or _phrase_subset(kw_l, q_lower):
            score += 0.6
        else:
            if set(kw_l.split()) & q_words:
                score += 0.3
    return score


def _term_in_desc_positive(term: str, desc_lower: str) -> bool:
    """Check if a term appears in a description in a non-negated context.

    Returns True if the term is present as a whole word and NOT preceded
    by a negation phrase like "no", "without", "absence of", etc.

    Uses word-boundary matching to avoid false positives like "cat" matching
    inside "scattered" or "location".

    For example:
      "a bird perched on a branch" + "bird"  → True
      "no animals or objects"      + "animal" → False
      "no other people visible"    + "people" → True (exception)
      "driftwood scattered around" + "cat"    → False (substring, not word)
    """
    # Use word-boundary regex to find whole-word matches only
    term_re = re.compile(r'\b' + re.escape(term) + r'\b')
    matches = list(term_re.finditer(desc_lower))
    if not matches:
        return False

    # Check if any occurrence of the term is NOT negated
    for m in matches:
        pos = m.start()

        # Look at the ~40 chars before this occurrence for negation cues
        context_start = max(0, pos - 40)
        context = desc_lower[context_start:pos]

        # Check for negation in the immediate context using regex to
        # allow small gaps (e.g. "without any animals", "no visible animals",
        # "no people, animals, or objects")
        negated = False
        if re.search(r'\b(?:no|without|absence of)\b.{0,30}$', context):
            negated = True

        # "no other" is an exception — implies presence
        if negated and "no other" in context:
            negated = False

        if not negated:
            return True  # Found a non-negated occurrence

    return False  # All occurrences were negated


def _description_relevance(description: str, query: str,
                           negate_people: bool = False) -> float:
    """Score how relevant a description is to a query, using three tiers.

    When negate_people is True (query is "no people", "without kids", etc.),
    the people logic is inverted: descriptions saying "no people" get a boost,
    and descriptions mentioning people get a penalty.

    Returns:
      +DESCRIPTION_BOOST           description matches what the query wants
      -DESCRIPTION_PENALTY         description contradicts what the query wants
      -DESCRIPTION_ABSENCE_PENALTY description exists but matches zero query words
      0.0                          no description, or partial match (neutral)
    """
    if not description:
        return 0.0

    desc_lower = description.lower()
    query_words = query.lower().split()

    # --- Negated people query ("no people", "without kids") ---
    # Invert the normal logic: boost empty scenes, penalize people.
    if negate_people:
        # Check if description negates people → that's what we WANT
        neg_match = _NEGATION_PEOPLE_RE.search(desc_lower)
        if neg_match and not _FALSE_NEGATION_RE.search(neg_match.group()):
            return DESCRIPTION_BOOST  # Description says "no people" — good!

        # Check if description mentions people → that's what we DON'T want
        has_people_word = any(kw in desc_lower for kw in _PEOPLE_KEYWORDS)
        if has_people_word:
            return -DESCRIPTION_PENALTY  # Description mentions people — bad!

        # Description doesn't mention people at all — mildly positive
        return DESCRIPTION_ABSENCE_PENALTY  # Small boost for absence

    # --- Normal (non-negated) query logic ---

    # Check for explicit negation: "no people", "no visible people", etc.
    # Also checks expanded terms: "no animals" negates an "animal" query.
    for word in query_words:
        # Check the query word itself and its stem
        terms_to_check = {word}
        stem = word.rstrip("s") if len(word) > 4 else word
        terms_to_check.add(stem)
        # Also check expanded terms (e.g. "animal" → "bird", "fish", etc.)
        terms_to_check |= _expand_query_word(word)

        for term in terms_to_check:
            for prefix in _NEGATION_PREFIXES:
                if prefix + term in desc_lower:
                    # Make sure it's not "no other <term>" (which implies presence)
                    neg_snippet = prefix + term
                    idx = desc_lower.find(neg_snippet)
                    if idx >= 0:
                        context = desc_lower[max(0, idx - 10):idx + len(neg_snippet)]
                        if "no other" not in context:
                            return -DESCRIPTION_PENALTY

    # Check for people-specific negation using regex: catches "no visible
    # presence of people", "no ... humans", "nobody", etc.
    # These fire when the query mentions people-related terms.
    if _query_mentions_people(query):
        match = _NEGATION_PEOPLE_RE.search(desc_lower)
        if match and not _FALSE_NEGATION_RE.search(match.group()):
            return -DESCRIPTION_PENALTY

    # Count how many query words appear in the description.
    # For each query word, check the word itself (with basic stem for plurals)
    # AND any expanded terms (e.g. "animal" also matches "bird", "elk", etc.).
    # A match only counts if the term is NOT negated in context.
    matched = 0
    for word in query_words:
        # Check the ORIGINAL word first, then a naive singular stem. Checking
        # only the stem silently failed words ending in -es / doubled-s
        # ("sunglasses".rstrip("s") = "sunglasse", which \bsunglasse\b never
        # matches against "sunglasses") — so 2000+ photos described as wearing
        # sunglasses were unsearchable.
        stem = word.rstrip("s") if len(word) > 4 else word
        if (_term_in_desc_positive(word, desc_lower)
                or _term_in_desc_positive(stem, desc_lower)):
            matched += 1
        else:
            # Check expanded terms for category words
            expansions = _expand_query_word(word)
            if len(expansions) > 1:  # has real expansions, not just itself
                if any(_term_in_desc_positive(term, desc_lower) for term in expansions):
                    matched += 1

    # For people queries: if the description doesn't mention ANY people-related
    # word, that's a strong negative signal regardless of other word matches.
    # A photo described as "outdoor hillside with a bird" matches "outdoor" but
    # the absence of people words means LLaVA saw no people in the scene.
    if _query_mentions_people(query):
        has_people_word = any(kw in desc_lower for kw in _PEOPLE_KEYWORDS)
        if not has_people_word:
            return -DESCRIPTION_PENALTY  # Strong negative — described scene, no people

    if matched == len(query_words):
        return DESCRIPTION_BOOST  # All words match — strong positive

    if matched == 0:
        return -DESCRIPTION_ABSENCE_PENALTY  # Nothing matched — weak negative

    return 0.0  # Partial match — neutral


def search_descriptions(db: PhotoDB, query: str, limit: int = 10) -> list[dict]:
    """Search LLaVA-generated descriptions for query keywords.

    Splits the query into words and matches photos whose description contains
    ALL query words (case-insensitive). This complements CLIP search — CLIP
    catches visual similarity while description search catches specific named
    content like "beach", "children", "driftwood".

    Returns photos with a score of 0.5 (a fixed value indicating a text match,
    used for merging with CLIP results).
    """
    words = query.lower().split()
    if not words:
        return []

    # Build a query where every word must appear in the description
    conditions = " AND ".join(["LOWER(description) LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words] + [limit]

    rows = db.conn.execute(
        f"""SELECT * FROM photos
            WHERE description IS NOT NULL AND {conditions}
            ORDER BY date_taken
            LIMIT ?""",
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def search_text_fields(db: PhotoDB, query: str, limit: int = 300) -> list[int]:
    """Retrieve photo ids whose text metadata matches the query.

    Unlike ``search_descriptions`` (which only scans ``description``), this scans
    the concatenation of ``description`` + ``keywords`` + ``categories`` +
    ``visual_tags`` so a photo tagged with a keyword the CLIP embedding misses
    (e.g. a wide landscape whose only "marmot" signal is the keyword) is still
    retrieved. Every query word must appear somewhere in that combined text.
    Returns ids only (callers grade relevance with the per-field scorers), most
    recent first, capped at ``limit``.
    """
    words = [w for w in query.lower().split() if w]
    if not words:
        return []
    field_expr = ("LOWER(COALESCE(description,'') || ' ' || COALESCE(keywords,'') || ' ' "
                  "|| COALESCE(categories,'') || ' ' || COALESCE(visual_tags,''))")
    conditions = " AND ".join([f"{field_expr} LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words] + [limit]
    rows = db.conn.execute(
        f"""SELECT id FROM photos
            WHERE {conditions}
            ORDER BY date_taken DESC
            LIMIT ?""",
        params,
    ).fetchall()
    return [r["id"] for r in rows]


def search_semantic(
    db: PhotoDB,
    query: str,
    limit: int = 10,
    min_score: float = CLIP_MIN_SCORE,
    debug: bool = False,
    text_match: str = "all",
) -> list[dict]:
    """Semantic search — combines CLIP similarity, face boost, and description matching.

    text_match controls how text relevance is computed:
      "all"        — union of all four signals: dict, categories, visual, keywords (default)
      "dict"       — dictionary-based term expansion only (original behavior)
      "categories" — LLM-generated categories only
      "visual"     — visual tags only
      "keywords"   — keywords only
      "off"        — disable text matching entirely

    Three signals are merged:
      1. CLIP embedding similarity (visual match)
      2. Face-aware boost (photos with detected faces score higher for people queries)
      3. Description text match (photos whose LLaVA description contains query words
         get a bonus, ensuring they always surface)

    This hybrid approach means "people outdoors" surfaces photos that CLIP ranks
    highly AND photos that LLaVA described as having "people" and "outdoors".

    Supports exclusion syntax: "beach -people" or "beach without people" will
    find beach photos and hard-filter any whose description mentions people.

    When debug=True, logs step-by-step scoring for every candidate to stderr.
    """
    def _dbg(msg):
        if debug:
            _log.info(msg)

    # Parse exclusions from the query
    positive_query, excluded_terms = _parse_query(query)
    _dbg(f"QUERY PARSE: positive={positive_query!r}  excluded={excluded_terms}")

    # Use the positive query for CLIP embedding (CLIP can't handle negation)
    clip_query = positive_query if positive_query else query
    query_embedding = embed_text(clip_query)
    if query_embedding is None:
        print("Error: could not generate embedding for query.")
        return []

    # Fetch more candidates than needed so the boost + re-sort can promote
    # face-having photos that would otherwise be cut off at the limit.
    fetch_limit = max(limit * 3, 30)
    matches = db.search_clip(query_embedding, limit=fetch_limit)
    _dbg(f"CLIP CANDIDATES: {len(matches)} fetched (fetch_limit={fetch_limit})")

    # Use the positive query (without exclusions) for people detection
    boost_people = _query_mentions_people(positive_query) if positive_query else False
    negate_people = bool(excluded_terms and any(t in _PEOPLE_KEYWORDS for t in excluded_terms))
    has_exclusions = bool(excluded_terms)
    _dbg(f"MODIFIERS: boost_people={boost_people}  negate_people={negate_people}  has_exclusions={has_exclusions}")

    # Pre-load face counts if we need them
    face_counts: dict[int, int] = {}
    if boost_people or negate_people:
        rows = db.conn.execute(
            "SELECT photo_id, COUNT(*) as cnt FROM faces GROUP BY photo_id"
        ).fetchall()
        face_counts = {row["photo_id"]: row["cnt"] for row in rows}

    # Pre-load descriptions for all candidate photos
    desc_cache: dict[int, str] = {}
    rows = db.conn.execute(
        "SELECT id, description FROM photos WHERE description IS NOT NULL"
    ).fetchall()
    desc_cache = {row["id"]: row["description"] for row in rows}

    # Map text_match modes to which signals to use. Unknown modes fall back to 'all'.
    VALID_TEXT_MATCH = {"all", "categories", "visual", "keywords", "dict", "off"}
    if text_match not in VALID_TEXT_MATCH:
        text_match = "all"
    use_categories = text_match in ("all", "categories")
    use_visual     = text_match in ("all", "visual")
    use_keywords   = text_match in ("all", "keywords")
    use_dict       = text_match in ("all", "dict")

    cat_cache: dict[int, str] = {}
    visual_cache: dict[int, str] = {}
    kw_cache: dict[int, str] = {}
    if use_categories:
        cat_cache = {row["id"]: row["categories"]
                     for row in db.conn.execute(
                         "SELECT id, categories FROM photos WHERE categories IS NOT NULL")}
    if use_visual:
        visual_cache = {row["id"]: row["visual_tags"]
                        for row in db.conn.execute(
                            "SELECT id, visual_tags FROM photos WHERE visual_tags IS NOT NULL")}
    if use_keywords:
        kw_cache = {row["id"]: row["keywords"]
                    for row in db.conn.execute(
                        "SELECT id, keywords FROM photos WHERE keywords IS NOT NULL")}
    _dbg(f"TEXT CACHES: cat={len(cat_cache)} visual={len(visual_cache)} kw={len(kw_cache)} mode={text_match}")

    def _text_rel(pid: int) -> tuple[float, list[str]]:
        """Graded text relevance for one photo across the enabled signals.

        Combines the per-field scorers (description / categories / visual /
        keywords) with the same weights the CLIP re-rank used, so a strong
        keyword hit (e.g. 'marmot') yields a large positive value that ranks
        the photo above CLIP-only neighbours. Returns (score, debug_parts).
        """
        if not positive_query:
            return 0.0, []
        rel = 0.0
        parts: list[str] = []
        if use_dict and pid in desc_cache:
            r = _description_relevance(desc_cache[pid], positive_query)
            rel += r * 0.6
            if r != 0:
                parts.append(f"dict={r:+.3f}")
        if use_categories and pid in cat_cache:
            r = _categories_match_query(cat_cache.get(pid), positive_query)
            rel += r * 1.0
            if r != 0:
                parts.append(f"cat={r:+.3f}")
        if use_visual and pid in visual_cache:
            r = _visual_match_query(visual_cache.get(pid), positive_query)
            rel += r * 0.7
            if r != 0:
                parts.append(f"vis={r:+.3f}")
        if use_keywords and pid in kw_cache:
            r = _keywords_match_query(kw_cache.get(pid), positive_query)
            rel += r * 1.2
            if r != 0:
                parts.append(f"kw={r:+.3f}")
        return rel, parts

    # Pre-load filenames for debug logging
    name_cache: dict[int, str] = {}
    if debug:
        rows = db.conn.execute("SELECT id, filename FROM photos").fetchall()
        name_cache = {row["id"]: row["filename"] for row in rows}

    results_by_id: dict[int, dict] = {}
    excluded_log: list[str] = []

    # Score CLIP results
    for match in matches:
        raw_score = 1.0 - match["distance"]
        score = raw_score
        photo_id = match["photo_id"]
        fname = name_cache.get(photo_id, f"id={photo_id}")
        steps = [f"clip={raw_score:.5f}"]

        # Hard-filter: if the query has excluded terms (e.g. "beach -people"),
        # skip any photo whose description contains the excluded content.
        if has_exclusions and photo_id in desc_cache:
            if _description_contains_excluded(desc_cache[photo_id], excluded_terms):
                reason = f"EXCLUDED {fname}: description contains excluded term"
                excluded_log.append(reason)
                _dbg(reason)
                continue

        # Hard-filter: if people are excluded, skip photos with detected faces
        if negate_people and photo_id in face_counts:
            reason = f"EXCLUDED {fname}: has faces but people excluded"
            excluded_log.append(reason)
            _dbg(reason)
            continue

        # Normal people boost: each detected face adds FACE_BOOST
        if boost_people and photo_id in face_counts:
            face_adj = face_counts[photo_id] * FACE_BOOST
            score += face_adj
            steps.append(f"face_boost=+{face_adj:.3f} ({face_counts[photo_id]} faces)")

        # Graded text relevance (keywords / categories / visual / description).
        # A positive text match now DOMINATES the score, so it always ranks above
        # CLIP-only neighbours, and CLIP can only *add* support — it can never drag
        # a real keyword hit below the floor (the old code blocked the boost whenever
        # CLIP was unsure, which silently dropped exact matches like "marmot").
        clip_base = score  # raw CLIP (+ any people/face boost applied above)
        text_rel, parts = _text_rel(photo_id)
        is_text_match = text_rel > 0
        if is_text_match:
            score = text_rel + max(clip_base, 0.0)
            steps.append(f"text_match={text_rel:+.3f} (" + " ".join(parts)
                         + f") clip_support=+{max(clip_base, 0.0):.3f}")
        else:
            score = clip_base + text_rel  # text_rel <= 0 (penalty or neutral)
            if text_rel < 0:
                steps.append(f"text_penalty={text_rel:+.3f} (" + " ".join(parts) + ")")
            elif positive_query and photo_id not in desc_cache:
                steps.append("no_description")

        steps.append(f"final={score:.5f}")
        _dbg(f"CANDIDATE {fname}: {' → '.join(steps)}")

        photo = db.get_photo(photo_id)
        if photo:
            photo["score"] = score
            photo["clip_score"] = raw_score
            photo["_text_match"] = is_text_match
            results_by_id[photo_id] = photo

    # Also surface photos whose TEXT metadata (keywords / categories / visual /
    # description) matches but that CLIP never ranked into the candidate pool —
    # e.g. a wide landscape keyword-tagged 'marmot'. This makes text a real
    # retrieval path, not just a re-rank of the CLIP neighbours.
    if positive_query:
        text_pids = search_text_fields(db, positive_query, limit=max(limit * 5, 300))
        _dbg(f"TEXT-FIELD CANDIDATES: {len(text_pids)} matches")
        for pid in text_pids:
            if pid in results_by_id:
                continue
            fname = name_cache.get(pid, f"id={pid}")
            rel, parts = _text_rel(pid)
            if rel <= 0:
                continue
            # Same exclusion / people gates as the CLIP path.
            desc_text = desc_cache.get(pid, "")
            if has_exclusions and _description_contains_excluded(desc_text, excluded_terms):
                _dbg(f"EXCLUDED {fname} (text-only): contains excluded term")
                continue
            if negate_people and pid in face_counts:
                _dbg(f"EXCLUDED {fname} (text-only): has faces but people excluded")
                continue
            if boost_people and pid not in face_counts:
                _dbg(f"EXCLUDED {fname} (text-only): people query but no faces")
                continue
            photo = db.get_photo(pid)
            if photo:
                photo["score"] = rel
                photo["clip_score"] = None
                photo["_text_match"] = True
                results_by_id[pid] = photo
                _dbg(f"INCLUDED {fname} (text-only): score={rel:.3f} (" + " ".join(parts) + ")")

    # Drop CLIP-only noise. When the query produced real text matches, hold the
    # text-less candidates to the stricter CLIP_ONLY_MIN_SCORE floor so weak /
    # negative visual neighbours vanish; pure-visual queries (no text match
    # anywhere) keep the permissive min_score floor so recall is unchanged.
    has_text_matches = any(r.get("_text_match") for r in results_by_id.values())
    clip_only_floor = CLIP_ONLY_MIN_SCORE if has_text_matches else min_score
    kept: list[dict] = []
    for r in results_by_id.values():
        if r.pop("_text_match", False):
            kept.append(r)  # real text match — always keep
        elif r["score"] >= min_score and (
                r.get("clip_score") is None or r["clip_score"] >= clip_only_floor):
            kept.append(r)

    # Re-sort by combined score, descending
    results = sorted(kept, key=lambda r: r["score"], reverse=True)
    final = _dedupe_by_hash(results)[:limit]

    if debug:
        _dbg(f"FINAL: {len(final)} results from {len(results_by_id)} candidates "
             f"({len(excluded_log)} excluded)")

    return final


def search_by_color(db: PhotoDB, color: str, tolerance: int = 60, limit: int = 10) -> list[dict]:
    """Find photos with dominant colors near the given color.

    Accepts hex colors (#ff0000) or common color names.
    """
    color_hex = _resolve_color_name(color)
    return db.search_by_color(color_hex, tolerance=tolerance, limit=limit)


def search_by_place(db: PhotoDB, place: str, limit: int = 10) -> list[dict]:
    """Search by place name (text match)."""
    return db.search_text(place, limit=limit)


def _extract_persons_from_query(db: PhotoDB, query: str) -> tuple[str, list[dict]]:
    """Find registered person names inside a free-text query.

    Matches are case-insensitive, word-bounded, and longest-first (so
    "Matt Newkirk" wins over "Matt" when both are registered). Names
    preceded by `-` are left alone so `-Calvin` keeps working as an
    exclusion token for the CLIP pass downstream. After matched names
    are stripped, connector tokens ("and", "with", "&", ",") are also
    stripped from the residual so the leftover reads cleanly as the
    semantic query.

    Returns (residual_query, [person_rows]).
    """
    persons = db.conn.execute("SELECT id, name FROM persons").fetchall()
    if not persons:
        return query, []

    candidates = sorted((dict(p) for p in persons), key=lambda p: -len(p["name"]))

    matched: list[dict] = []
    seen_ids: set[int] = set()
    residual = query
    for p in candidates:
        pattern = re.compile(
            r'(?<!-)(?<!\w)' + re.escape(p["name"]) + r'(?!\w)',
            re.IGNORECASE,
        )
        if pattern.search(residual) and p["id"] not in seen_ids:
            matched.append(p)
            seen_ids.add(p["id"])
            residual = pattern.sub(' ', residual)

    if matched:
        residual = re.sub(r'\s+(?:and|with)\s+', ' ', residual, flags=re.IGNORECASE)
        residual = re.sub(r'\s*[&,]\s*', ' ', residual)
        residual = re.sub(r'\s+', ' ', residual).strip()

    return residual, matched


def search_by_person(db: PhotoDB, name: str, limit: int = 10, match_source: str | None = None,
                     scope: list[int] | None = None, columns: str = "p.*") -> list[dict]:
    """Find all photos containing a named person.

    Looks up the person by name, then finds all faces linked to that person,
    then returns the distinct photos those faces appear in.

    match_source: if set, only return photos where the face was matched via
    this method ('strict', 'temporal', or 'manual').

    scope: restrict to these photo ids (see `_compose_scope`).
    columns: the SELECT list; search_combined passes `_NARROW_COLUMNS` and
        reads full rows for the requested page only.
    """
    person = db.get_person_by_name(name)
    if not person:
        print(f"  Person '{name}' not found. Use 'add-person' to register them.")
        return []

    # id-first: resolve the photo ids from the faces index, then read each
    # photo row once. Joining read the wide row once per FACE and then
    # de-duplicated whole rows in a temp B-tree.
    sql = f"""SELECT {columns}
           FROM photos p
           WHERE p.id IN (SELECT f.photo_id FROM faces f WHERE f.person_id = ?"""
    params: list = [person["id"]]

    if match_source:
        sql += " AND f.match_source = ?"
        params.append(match_source)
    sql += ")"

    scope_sql, scope_params = _scope_clause("p.id", scope)
    sql += scope_sql
    params.extend(scope_params)

    # p.id breaks timestamp ties so the order (which feeds relevance ranks and
    # which duplicate copy survives _dedupe_by_hash) never depends on the plan.
    sql += " ORDER BY p.date_taken, p.id LIMIT ?"
    params.append(limit)

    rows = db.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_all_persons(
    db: PhotoDB,
    person_ids: list[int],
    limit: int = 10,
    match_source: str | None = None,
    scope: list[int] | None = None,
    columns: str = "p.*",
) -> list[dict]:
    """Find photos containing ALL of the given persons (AND intersection).

    Runs a single SQL intersection with `HAVING COUNT(DISTINCT person_id) = N`
    instead of calling `search_by_person` per person and intersecting in
    memory. The per-person path caps each set at `limit` photos ordered by
    date ASC, so for three-way intersections where one person is recent and
    the others have thousands of earlier photos the oldest-N windows can
    have zero overlap and the intersection collapses to empty. SQL-side
    aggregation avoids that entirely.

    Orders by `date_taken DESC` so the most recent matches surface first —
    usually what the user wants when searching "everyone together".
    """
    if not person_ids:
        return []

    # id-first: the intersection runs entirely on the faces index; only the
    # matching photos' rows are read (331 MB -> 45 MB for two people,
    # measured 2026-10-03). The old JOIN read the wide row once per face.
    placeholders = ",".join("?" * len(person_ids))
    sql = (
        f"SELECT {columns} FROM photos p WHERE p.id IN ("
        f"SELECT f.photo_id FROM faces f WHERE f.person_id IN ({placeholders})"
    )
    params: list = list(person_ids)

    if match_source:
        sql += " AND f.match_source = ?"
        params.append(match_source)
    sql += " GROUP BY f.photo_id HAVING COUNT(DISTINCT f.person_id) = ?)"
    params.append(len(person_ids))

    scope_sql, scope_params = _scope_clause("p.id", scope)
    sql += scope_sql
    params.extend(scope_params)

    sql += " ORDER BY p.date_taken DESC, p.id LIMIT ?"
    params.append(limit)

    rows = db.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_face_reference(db: PhotoDB, image_path: str, limit: int = 10) -> list[dict]:
    """Find photos containing a face similar to the one in the given reference image.

    Encodes the face in the reference image, then searches face_encodings for matches.
    """
    from .faces import encode_reference_photo, match_face
    import struct

    encoding = encode_reference_photo(image_path)
    if encoding is None:
        print(f"  No face found in reference image: {image_path}")
        return []

    matches = db.search_faces(encoding, limit=limit * 3)
    if not matches:
        return []

    # Get distinct photos for matched face IDs
    seen_photo_ids = set()
    results = []
    for match in matches:
        face_id = match["face_id"]
        face_row = db.conn.execute(
            "SELECT photo_id FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
        if face_row and face_row["photo_id"] not in seen_photo_ids:
            photo = db.get_photo(face_row["photo_id"])
            if photo:
                photo["face_distance"] = match["distance"]
                results.append(photo)
                seen_photo_ids.add(face_row["photo_id"])
        if len(results) >= limit:
            break

    return results


# Open upper bound when a caller gives only `date_from` (meaning "from this date
# ONWARD"). Defaulting `date_to` to `date_from` instead collapsed the range to a
# single day, so e.g. person + date_from='2026-01-01' ("in 2026 so far") matched
# only Jan 1 and the intersection silently emptied.
_OPEN_DATE_HI = "9999-12-31"


def _has_aesthetic_filters(min_aesthetic, min_technical, min_composition,
                           min_impact, style_tag, min_subject_aesthetic=None,
                           min_day_aesthetic=None) -> bool:
    return any(v is not None for v in
               (min_aesthetic, min_technical, min_composition, min_impact,
                min_subject_aesthetic, min_day_aesthetic)) \
        or bool(style_tag)


def _style_tag_matches(row: dict, style_tag: str) -> bool:
    """True if `style_tag` is present in the row's aes_style_tags JSON array."""
    raw = row.get("aes_style_tags")
    if not raw:
        return False
    try:
        import json as _json
        tags = _json.loads(raw) if isinstance(raw, str) else raw
        return style_tag.lower() in {str(t).lower() for t in (tags or [])}
    except Exception:
        return False


def _filter_aesthetic(results: list[dict], min_aesthetic=None, min_technical=None,
                      min_composition=None, min_impact=None, style_tag=None,
                      min_subject_aesthetic=None, min_day_aesthetic=None) -> list[dict]:
    """Filter to photos meeting the aesthetic thresholds. min_aesthetic /
    min_subject_aesthetic are on the library-relative percentiles
    (aes_overall_pct / aes_subject_overall_pct, 0-100); min_day_aesthetic is on
    the PER-DAY percentile (how the photo ranks among others taken the same day —
    subject day-pct when present, else full-frame day-pct); the per-dimension
    thresholds are on the raw 1-10 dimension scores."""
    def _day_pct(r):
        v = r.get("aes_subject_overall_day_pct")
        return v if v is not None else r.get("aes_overall_day_pct")
    out = []
    for r in results:
        if min_aesthetic is not None and (r.get("aes_overall_pct") or -1) < min_aesthetic:
            continue
        if min_subject_aesthetic is not None and (r.get("aes_subject_overall_pct") or -1) < min_subject_aesthetic:
            continue
        if min_day_aesthetic is not None and (_day_pct(r) if _day_pct(r) is not None else -1) < min_day_aesthetic:
            continue
        if min_technical is not None and (r.get("aes_technical") or -1) < min_technical:
            continue
        if min_composition is not None and (r.get("aes_composition") or -1) < min_composition:
            continue
        if min_impact is not None and (r.get("aes_impact") or -1) < min_impact:
            continue
        if style_tag and not _style_tag_matches(r, style_tag):
            continue
        out.append(r)
    return out


def _floor_sql(col: str, floor: float) -> str:
    """SQL for `(row[col] or -1) >= floor`, the test `_filter_aesthetic`
    applies. For a positive floor that is a plain, index-usable `col >= ?`;
    otherwise NULL and 0 both read as -1, exactly as `or -1` does."""
    if floor > 0:
        return f"{col} >= ?"
    return f"COALESCE(NULLIF({col}, 0), -1) >= ?"


def _aesthetic_floor_sql(min_aesthetic, min_technical, min_composition,
                         min_impact, min_subject_aesthetic,
                         min_day_aesthetic) -> tuple[list[str], list]:
    """`_filter_aesthetic`'s numeric floors as SQL (style_tag excluded — it is
    matched in Python by `_style_tag_matches`)."""
    clauses: list[str] = []
    params: list = []
    for col, floor in (("aes_overall_pct", min_aesthetic),
                       ("aes_subject_overall_pct", min_subject_aesthetic),
                       ("aes_technical", min_technical),
                       ("aes_composition", min_composition),
                       ("aes_impact", min_impact)):
        if floor is not None:
            clauses.append(_floor_sql(col, floor))
            params.append(floor)
    if min_day_aesthetic is not None:
        clauses.append(
            "COALESCE(aes_subject_overall_day_pct, aes_overall_day_pct, -1) >= ?")
        params.append(min_day_aesthetic)
    return clauses, params


def _sort_sql(sort: str, base: str) -> Optional[str]:
    """`_apply_sort(sort)` as an ORDER BY, applied to rows that arrive in
    `base` order (the sorts are stable, so `base` is the tie-break), or None
    for a mode with no SQL form. Undated rows go last for the date sorts."""
    keys = {
        "relevance": None,
        "aesthetic_desc": "COALESCE(aes_overall_pct, 0) DESC",
        "subject_aesthetic_desc":
            "COALESCE(aes_subject_overall_pct, aes_overall_pct, 0) DESC",
        "quality_desc":
            "CASE WHEN COALESCE(aes_subject_overall_pct, aes_overall_pct) IS NOT NULL"
            " THEN 1000 + COALESCE(aes_subject_overall_pct, aes_overall_pct)"
            " ELSE COALESCE(aesthetic_score, -1) END DESC",
        "day_quality_desc":
            "COALESCE(aes_subject_overall_day_pct, aes_overall_day_pct, -1) DESC",
        "date_desc": "(date_taken IS NULL OR date_taken = '') ASC, date_taken DESC",
        "date_asc": "(date_taken IS NULL OR date_taken = '') ASC, date_taken ASC",
    }
    if sort not in keys:
        return None
    return base if keys[sort] is None else f"{keys[sort]}, {base}"


def _sql_page(db: PhotoDB, where: list[str], params: list, order_by: str,
              offset: int, limit: int, with_total: bool):
    """Count, then read only the requested page — instead of SELECT * over
    every qualifying row and slicing in Python (the aesthetics browse read
    846 MB to show 100 photos)."""
    w = " AND ".join(where) or "1"
    page_sql = f"SELECT * FROM photos WHERE {w} ORDER BY {order_by}"
    page_params = list(params)
    if limit:
        page_sql += " LIMIT ? OFFSET ?"
        page_params += [limit, offset]
    elif offset:
        page_sql += " LIMIT -1 OFFSET ?"
        page_params.append(offset)
    page = [dict(r) for r in db.conn.execute(page_sql, page_params)]
    if not with_total:
        return page
    total = db.conn.execute(f"SELECT COUNT(*) FROM photos WHERE {w}",
                            params).fetchone()[0]
    return page, total


def _filter_by_date(results: list[dict], date_from: str, date_to: str) -> list[dict]:
    """Filter results to those whose date_taken falls within [date_from, date_to]."""
    filtered = []
    for r in results:
        dt = r.get("date_taken")
        if not dt:
            continue
        # date_taken is "YYYY-MM-DD HH:MM:SS"; compare date portion
        date_str = dt[:10]
        if date_from <= date_str <= date_to:
            filtered.append(r)
    return filtered


def _search_by_date(db: PhotoDB, date_from: str, date_to: str, limit: int = 0) -> list[dict]:
    """Return photos within a date range, ordered by date.

    limit=0 means no limit (return all matching photos).
    """
    if limit > 0:
        rows = db.conn.execute(
            """SELECT * FROM photos
               WHERE date_taken IS NOT NULL
                 AND date_taken >= ? AND date_taken <= ?
               ORDER BY date_taken
               LIMIT ?""",
            (date_from, date_to + " 23:59:59", limit),
        ).fetchall()
    else:
        rows = db.conn.execute(
            """SELECT * FROM photos
               WHERE date_taken IS NOT NULL
                 AND date_taken >= ? AND date_taken <= ?
               ORDER BY date_taken""",
            (date_from, date_to + " 23:59:59"),
        ).fetchall()
    return [dict(r) for r in rows]


# Allowed values for search_combined's `sort` param. Kept as a module
# constant so callers (web.py, cli.py) and tests reference one list.
SORT_MODES = ("date_desc", "date_asc", "quality_desc", "aesthetic_desc",
              "subject_aesthetic_desc", "day_quality_desc", "relevance")

# Reciprocal Rank Fusion constant. Textbook default is 60 — smaller k
# makes the top-ranked item in each filter dominate more; larger k
# flattens the contribution curve. 60 is a reasonable middle ground
# that doesn't over-reward any single signal.
_RRF_K = 60


# Recency decay rate. 0.05 is gentle: photos from 1 year ago score at
# 95% of today's, 5 years at 78%, 10 years at 61%. Tuned so family
# photos from a decade ago aren't pushed far below today's — just a
# nudge so genuinely recent photos surface over equally-relevant old
# ones in relevance-mode sorts.
_RECENCY_DECAY_RATE = 0.05
# Factor for photos with no date_taken. 0.5 is halfway between fresh
# and "~14 years old" — neutral-ish, slight penalty so undated photos
# don't beat real recent ones in relevance mode.
_UNDATED_RECENCY_FACTOR = 0.5


def _apply_recency_decay(photos: list[dict]) -> None:
    """Multiply each photo's rrf_score by exp(-years_ago * 0.05).
    Mutates the dicts in place. Called only in sort='relevance' mode;
    date / quality sorts ignore rrf_score so the extra compute would
    be wasted there.
    """
    import math
    import datetime
    now = datetime.datetime.now()
    for photo in photos:
        dt_str = photo.get("date_taken")
        factor = _UNDATED_RECENCY_FACTOR
        if dt_str:
            try:
                # SQLite stores "YYYY-MM-DD HH:MM:SS"; accept "T" too.
                dt = datetime.datetime.fromisoformat(dt_str.replace(" ", "T"))
                years_ago = max(0.0, (now - dt).total_seconds() / (365.25 * 86400))
                factor = math.exp(-_RECENCY_DECAY_RATE * years_ago)
            except (ValueError, TypeError):
                pass  # keep undated factor
        photo["rrf_score"] = (photo.get("rrf_score") or 0.0) * factor


def _attach_rrf_scores(result_sets: list[dict[int, dict]],
                       ranks_per_set: list[dict[int, int]]) -> None:
    """Compute Reciprocal Rank Fusion score per photo in result_sets[0]
    based on its rank in every filter it appears in. Mutates the photo
    dicts to add an `rrf_score` key.

    Formula: score = Σ 1/(K + rank_i) across filters where the photo
    appears. K=60. Photos ranking high in multiple filters accumulate,
    so a "Calvin at the beach" hit where Calvin=rank-5 AND CLIP "beach"
    =rank-3 scores higher than one where Calvin=rank-200 AND beach=
    rank-400.
    """
    if not result_sets:
        return
    primary = result_sets[0]
    for pid, photo in primary.items():
        score = 0.0
        for ranks in ranks_per_set:
            r = ranks.get(pid)
            if r is not None:
                score += 1.0 / (_RRF_K + r)
        photo["rrf_score"] = score


def _apply_sort(merged: list[dict], sort: str) -> list[dict]:
    """Sort `merged` in place-ish (returns a new list) by the requested
    mode. NULL date_taken rows land at the TAIL regardless of direction
    so "Newest first" never surfaces undated photos at the top.

    Relevance mode preserves the caller-supplied order (which is the
    first result_set's insertion order today, and will be RRF score
    once that lands). Callers who know they're in CLIP-semantic or
    RRF-ranked mode should pass sort='relevance'; everyone else should
    pass an explicit date/quality mode.
    """
    if sort == "relevance":
        # Sort by RRF score (higher = more relevant across all filters).
        # Photos without an rrf_score key — early-return paths, or
        # callers that didn't compute it — all score 0 and sorted()'s
        # stable ordering preserves their original sequence. Matches
        # the pre-RRF "relevance = preserve order" behavior for those
        # cases.
        return sorted(
            merged,
            key=lambda r: (r.get("rrf_score") or 0.0),
            reverse=True,
        )
    if sort == "quality_desc":
        # "Best quality" now ranks by the new VLM score (subject-aware, then
        # full-frame percentile), falling back to the old LAION aesthetic_score
        # for photos the VLM hasn't scored yet. One comparable scale: VLM-scored
        # (1000..1100) rank above old-scored (0..~10) above unscored.
        def _quality_key(r):
            pct = r.get("aes_subject_overall_pct")
            if pct is None:
                pct = r.get("aes_overall_pct")
            if pct is not None:
                return 1000 + pct
            aes = r.get("aesthetic_score")
            return aes if aes is not None else -1
        return sorted(merged, key=_quality_key, reverse=True)
    if sort == "aesthetic_desc":
        # Rank by the library-relative percentile so the best photos lead.
        return sorted(
            merged,
            key=lambda r: (r.get("aes_overall_pct") or 0),
            reverse=True,
        )
    if sort == "day_quality_desc":
        # "Best of day": rank by the per-day percentile (subject-aware, then
        # full-frame), so each day's strongest photos lead regardless of how
        # good the light was that day. Unscored/undated sort last.
        return sorted(
            merged,
            key=lambda r: (r["aes_subject_overall_day_pct"]
                           if r.get("aes_subject_overall_day_pct") is not None
                           else (r.get("aes_overall_day_pct")
                                 if r.get("aes_overall_day_pct") is not None else -1)),
            reverse=True,
        )
    if sort == "subject_aesthetic_desc":
        # Rank by the SUBJECT-crop percentile (best subject shots lead),
        # falling back to the full-frame percentile for photos without a
        # subject score (subject fills the frame / landscape / not yet scored).
        return sorted(
            merged,
            key=lambda r: (r["aes_subject_overall_pct"]
                           if r.get("aes_subject_overall_pct") is not None
                           else (r.get("aes_overall_pct") or 0)),
            reverse=True,
        )
    # Date sorts: split dated / undated so NULLs land at the tail.
    dated = [r for r in merged if r.get("date_taken")]
    undated = [r for r in merged if not r.get("date_taken")]
    dated.sort(key=lambda r: r["date_taken"], reverse=(sort == "date_desc"))
    return dated + undated


# Nominatim's bbox for a named place hugs its official admin boundary
# (for a city: the city limits). Users typing "San Rafael" usually mean
# "the San Rafael area" — including the adjacent unincorporated
# neighborhoods that the offline reverse-geocoder labels with their
# own CDP names (Lucas Valley-Marinwood, Marinwood, etc.) and that a
# tight bbox therefore excludes. Pad the bbox by ~4km per side to
# capture those immediate-neighbor places. Country-level queries skip
# bbox entirely (country-code anchor is authoritative), so this pad
# never applies to huge admin regions.
_BBOX_PAD_DEG = 0.04  # ~4-5 km depending on latitude


def _pad_bbox(bbox: list[float]) -> tuple[float, float, float, float]:
    """Expand a [south, north, west, east] bbox by _BBOX_PAD_DEG per side."""
    s, n, w, e = bbox
    return (s - _BBOX_PAD_DEG, n + _BBOX_PAD_DEG,
            w - _BBOX_PAD_DEG, e + _BBOX_PAD_DEG)


# Intermediate limit for filter-based searches that will be intersected
# with another filter. `search_by_person`, `_search_by_location`, and
# friends all apply their own LIMIT inside the SQL. If we use `limit*3`
# (default 600) for those, each filter returns just its oldest-N window
# and the intersection silently collapses to empty when the windows
# don't overlap (classic symptom: "Calvin in France" returns zero even
# though Calvin has many French photos, because Calvin's oldest 600 are
# US kid photos and the oldest 600 French-tagged photos predate him).
# Filter sets must be unbounded — ranking-based searches (CLIP semantic,
# face-image) keep `limit*3` because they're true top-N.
_FILTER_PREFETCH_LIMIT = 100_000

# Everything search_combined reads from a row AFTER the filters run — hash
# dedupe, date / quality / aesthetic filters, style tag, RRF + recency decay,
# every sort mode. When the only filters are people, their queries select
# just these and full rows are read for the returned page alone: a person
# with 17k photos read every wide row (223 MB) to show 100.
# A column the post-filter pipeline starts reading must be added here.
_NARROW_COLUMNS = ", ".join(f"p.{c}" for c in (
    "id", "file_hash", "date_taken", "aes_overall", "aesthetic_score",
    "aes_overall_pct", "aes_subject_overall_pct", "aes_overall_day_pct",
    "aes_subject_overall_day_pct", "aes_technical", "aes_composition",
    "aes_impact", "aes_style_tags"))


# Composed scope (docs/plans/search-indexes.md, step 9a). Each structured
# filter used to run over the WHOLE library as its own `SELECT *` and the
# sets were intersected in Python, so date x person x camera x location read
# ~1.9 GB cold. When two or more structured filters are given, the cheap,
# index-backed ones (date range, camera, people) are composed into one id
# query first; every filter then runs only over those ids. The intersection
# and _filter_by_date still run afterwards, so the scope is purely an
# optimisation: it never adds a photo, and a photo it drops would have
# failed the intersection anyway.
#
# Above this many ids the scope is dropped and each filter runs unscoped as
# before: fetching that many rows by rowid stops being cheaper than a scan.
_SCOPE_MAX_IDS = 20_000


def _date_bounds(date_from: str, date_to: Optional[str]) -> tuple[str, str]:
    """SQL bounds equivalent to `_filter_by_date` (date_taken is always
    'YYYY-MM-DD hh:mm:ss'), so a range is index-searchable."""
    return date_from, (date_to or _OPEN_DATE_HI) + " 23:59:59"


def _scope_clause(column: str, scope: Optional[list[int]]) -> tuple[str, list]:
    """` AND <column> IN (<scope ids>)`, or nothing when there is no scope."""
    if scope is None:
        return "", []
    return f" AND {column} IN (SELECT value FROM json_each(?))", [json.dumps(scope)]


def _compose_scope(db: PhotoDB, *, date_from: Optional[str], date_to: Optional[str],
                   camera: Optional[str], person_ids: list[int],
                   match_source: Optional[str]) -> Optional[list[int]]:
    """Ids of photos passing every index-backed structured filter, or None
    when there is nothing to compose or the result is too broad to help.

    People are AND-ed, each tested with EXISTS on faces(photo_id, person_id)
    — one index seek per candidate photo. Without a date or camera the query
    is driven from the first person's faces instead.
    """
    if not (date_from or camera or person_ids):
        return None
    params: list = []

    def person_exists(alias: str, pid: int) -> str:
        params.append(pid)
        sql = (f"EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = {alias} "
               f"AND f.person_id = ?")
        if match_source:
            sql += " AND f.match_source = ?"
            params.append(match_source)
        return sql + ")"

    if date_from or camera:
        clauses = []
        if camera:
            clauses.append("p.camera_model = ?")
            params.append(camera)
        if date_from:
            clauses.append("p.date_taken >= ? AND p.date_taken <= ?")
            params.extend(_date_bounds(date_from, date_to))
        clauses.extend(person_exists("p.id", pid) for pid in person_ids)
        sql = "SELECT p.id FROM photos p WHERE " + " AND ".join(clauses)
    else:
        first, rest = person_ids[0], person_ids[1:]
        sql = "SELECT DISTINCT f0.photo_id FROM faces f0 WHERE f0.person_id = ?"
        params.append(first)
        if match_source:
            sql += " AND f0.match_source = ?"
            params.append(match_source)
        for pid in rest:
            sql += " AND " + person_exists("f0.photo_id", pid)
    sql += " LIMIT ?"
    params.append(_SCOPE_MAX_IDS + 1)
    ids = [r[0] for r in db.conn.execute(sql, params)]
    return None if len(ids) > _SCOPE_MAX_IDS else ids


def _hydrate(db: PhotoDB, narrow_rows: list[dict]) -> list[dict]:
    """Full photo rows for `narrow_rows`, in order, keeping the keys the
    pipeline computed on them (rrf_score, score, ...)."""
    full = {r["id"]: r for r in _fetch_photos(db, [r["id"] for r in narrow_rows])}
    return [{**full[r["id"]], **r} for r in narrow_rows if r["id"] in full]


def _fetch_photos(db: PhotoDB, ids: list[int]) -> list[dict]:
    """Full rows for `ids`, in `ids` order, in chunks under SQLite's
    variable limit."""
    by_id: dict[int, dict] = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        ph = ",".join("?" * len(chunk))
        for r in db.conn.execute(f"SELECT * FROM photos WHERE id IN ({ph})", chunk):
            by_id[r["id"]] = dict(r)
    return [by_id[i] for i in ids if i in by_id]


def _search_by_bbox(db: PhotoDB, south: float, north: float,
                    west: float, east: float, limit: int = 100,
                    scope: Optional[list[int]] = None) -> list[dict]:
    """Return photos whose GPS falls inside the given bounding box.

    Used by `_search_by_location` as a fallback when a query doesn't
    substring-match any place_name — the offline reverse-geocoder only
    knows cities with population >1000, so photos at smaller places
    (Point Reyes, Marinwood) get labeled with the nearest bigger town
    and the substring match misses them. Nominatim's bbox puts them
    back.
    """
    scope_sql, scope_params = _scope_clause("id", scope)
    rows = db.conn.execute(
        f"""SELECT * FROM photos
           WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
             AND gps_lat BETWEEN ? AND ?
             AND gps_lon BETWEEN ? AND ?{scope_sql}
           ORDER BY date_taken
           LIMIT ?""",
        (south, north, west, east, *scope_params, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# A Nominatim bbox below this area (deg²) is a point/address, not a region.
_MIN_REGION_BBOX_AREA = 0.0005
# Region-name variants tried when a bare query resolves only to a point (e.g.
# "Point Reyes" → the cape, but photos sit 25km away across the National
# Seashore). Generalizes to parks/forests whose name is just the place name.
_REGION_SUFFIXES = (" National Park", " National Seashore", " State Park",
                    " National Monument", " National Forest",
                    " National Recreation Area", " National Wildlife Refuge")


def _bbox_area(bb) -> float:
    return (bb[1] - bb[0]) * (bb[3] - bb[2]) if bb and len(bb) == 4 else 0.0


def _resolve_location_bbox(db, name: str):
    """Resolve a free-text place to a padded [s,n,w,e] bbox, or None.

    Uses the top Nominatim result's bbox when it's a real region/city. When the
    top result is point-like (a cape, a single address), tries region-name
    variants ("X National Seashore" …) and uses the first whose bbox is a real
    region NEAR the original point — so "Point Reyes" finds the National
    Seashore that actually contains the photos, not just the 11m cape point.
    """
    from .geocode import forward_geocode
    try:
        cands = [c for c in forward_geocode(db, name, limit=5)[0] if c.get("bbox")]
    except Exception:
        return None
    if not cands:
        return None
    top = cands[0]
    if _bbox_area(top["bbox"]) >= _MIN_REGION_BBOX_AREA:
        return _pad_bbox(top["bbox"])           # real region/city — unchanged behavior
    for suf in _REGION_SUFFIXES:                 # point-like → try the containing region
        try:
            rc = forward_geocode(db, name + suf, limit=2)[0]
        except Exception:
            continue
        for c in rc:
            bb = c.get("bbox")
            if (bb and _bbox_area(bb) >= _MIN_REGION_BBOX_AREA
                    and abs((bb[0] + bb[1]) / 2 - top["lat"]) < 1.5
                    and abs((bb[2] + bb[3]) / 2 - top["lon"]) < 1.5):
                return _pad_bbox(bb)
    return _pad_bbox(top["bbox"])                # fall back to the padded point


def _search_by_location(db: PhotoDB, location: str, limit: int = 100,
                        scope: Optional[list[int]] = None) -> list[dict]:
    """Search by place_name using case-insensitive LIKE matching, with
    two expansions on top of the raw substring:

    1. **Country-code anchor.** When the query matches a known country
       name or looks like an ISO alpha-2 code, also match the ", CC"
       slot at the end of place_name. Otherwise "France" only catches
       "Île-de-France" and misses every other French region because
       the offline geocoder emits "Locality, Admin1, CC".

    2. **Nominatim bbox fallback.** If the substring+code pass returns
       nothing AND the query isn't a country name, resolve it via
       Nominatim (cached) and search by the returned bounding box. This
       catches small places that aren't in the GeoNames cities1000 set
       (Point Reyes, Marinwood, Folsom Lake, Yosemite, etc.) and so
       never appear in any photo's place_name.
    """
    from .geocode import country_name_to_code, forward_geocode

    name = location.strip()
    if not name:
        return []

    patterns = [f"%{name}%"]
    code = country_name_to_code(name)
    if code:
        # Anchor with ", CC" so a 2-letter code doesn't false-positive on
        # locality names containing those letters (e.g. "ES" inside
        # "Esterzili, Sardegna, IT"). No trailing % → end-of-string match.
        patterns.append(f"%, {code}")

    scope_sql, scope_params = _scope_clause("id", scope)
    placeholders = " OR ".join(["place_name LIKE ?"] * len(patterns))
    # id-first: the LIKE scans the narrow idx_photos_place, and only matching
    # rows are read (560 -> 7 MB). Only here: in tools._build_filter_sql the
    # same shape makes the planner drop the date index (8.7 -> 259 MB).
    rows = db.conn.execute(
        f"""SELECT * FROM photos
            WHERE id IN (SELECT id FROM photos
                         WHERE place_name IS NOT NULL AND ({placeholders})){scope_sql}
            ORDER BY date_taken
            LIMIT ?""",
        (*patterns, *scope_params, limit),
    ).fetchall()
    results = [dict(r) for r in rows]

    # Union with structured location columns (schema v19). Enables
    # admin2 queries like "Marin County" that the flat place_name
    # substring never caught — reverse_geocoder's assembled place_name
    # skips admin2, so "Marin County" as substring matched nothing.
    # This UNIONs exact-match hits on country/admin1/admin2/locality.
    # Graceful if columns don't exist yet (pre-backfill or old DB).
    #
    # `col = ? COLLATE NOCASE` (not LOWER(col) = LOWER(?), which no index can
    # serve) uses the v34 idx_photos_*_nc indexes; NOCASE folds ASCII only,
    # exactly like LOWER() without ICU. With a scope the columns get a unary
    # `+` so the query is driven from the scope ids, not from every photo in
    # a broad place.
    plus = "+" if scope is not None else ""
    struct_where = " OR ".join(
        f"{plus}{col} = ? COLLATE NOCASE"
        for col in ("country", "admin1", "admin2", "locality"))
    try:
        struct_rows = db.conn.execute(
            f"""SELECT * FROM photos
               WHERE ({struct_where}){scope_sql}
               ORDER BY date_taken
               LIMIT ?""",
            (name, name, name, name, *scope_params, limit),
        ).fetchall()
        if struct_rows:
            seen = {r["id"]: r for r in results}
            for r in struct_rows:
                if r["id"] not in seen:
                    seen[r["id"]] = dict(r)
            results = list(seen.values())[:limit]
    except sqlite3.OperationalError:
        pass  # Columns not yet migrated — skip silently.

    # Country-level queries: the code expansion already covers the
    # entire country, and a Nominatim bbox for a whole country is huge
    # and slow. Skip the fallback.
    if code:
        return results

    # Non-country queries: union with Nominatim bbox results so synonyms
    # and nearby named places collapse to the same answer. "San Rafael",
    # "Lucas Valley", and "Marinwood" are all the same geography; the
    # offline geocoder labels photos with whatever populated place is
    # nearest, so substring matching alone gave wildly different counts
    # for what's meant to be one location. The bbox catches all of them.
    bbox = _resolve_location_bbox(db, name)
    if bbox:
        bbox_rows = _search_by_bbox(db, *bbox, limit, scope=scope)
        seen = {r["id"]: r for r in results}
        for r in bbox_rows:
            if r["id"] not in seen:
                seen[r["id"]] = r
        results = list(seen.values())[:limit]

    return results


def search_combined(
    db: PhotoDB,
    query: Optional[str] = None,
    color: Optional[str] = None,
    place: Optional[str] = None,
    person: Optional[str] = None,
    face_image: Optional[str] = None,
    limit: int = 10,
    min_score: float = CLIP_MIN_SCORE,
    min_quality: Optional[float] = None,
    sort_quality: bool = False,
    debug: bool = False,
    text_match: str = "all",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    location: Optional[str] = None,
    match_source: Optional[str] = None,
    offset: int = 0,
    with_total: bool = False,
    sort: str = "date_desc",
    category: Optional[str] = None,
    visual_tag: Optional[str] = None,
    keyword: Optional[str] = None,
    person_ids: Optional[list[int]] = None,
    min_aesthetic: Optional[float] = None,
    min_technical: Optional[float] = None,
    min_composition: Optional[float] = None,
    min_impact: Optional[float] = None,
    style_tag: Optional[str] = None,
    min_subject_aesthetic: Optional[float] = None,
    min_day_aesthetic: Optional[float] = None,
    camera: Optional[str] = None,
):
    """Run multiple search types and merge results.

    When multiple criteria are given, returns the intersection
    ranked by the primary search type (person > semantic > color > place).

    Args:
        min_quality: If set, floor on the RAW aesthetic score shown on the photo —
            VLM `aes_overall` (1-10) when scored, else legacy `aesthetic_score`.
        sort_quality: If True, sort final results by aesthetic_score (highest first)
                     instead of the default relevance ordering.
        date_from: If set, filter to photos taken on or after this date (YYYY-MM-DD).
        date_to: If set, filter to photos taken on or before this date (YYYY-MM-DD).
        location: If set, search by place name (matched against reverse-geocoded place_name).
    """
    from .date_parse import parse_date_from_query
    from .geocode import extract_location_from_query

    # Parse dates and locations from the query string (if not explicitly provided)
    effective_query = query
    if effective_query:
        # Extract date from query if no explicit date args given
        if not date_from and not date_to:
            parsed_from, parsed_to, cleaned = parse_date_from_query(effective_query)
            if parsed_from:
                date_from = parsed_from
                date_to = parsed_to
                effective_query = cleaned if cleaned else None

        # Extract location from query if no explicit --place or --location given
        if not place and not location and effective_query:
            parsed_loc, cleaned = extract_location_from_query(effective_query)
            if parsed_loc:
                location = parsed_loc
                effective_query = cleaned if cleaned else None

    result_sets = []
    # Parallel to result_sets; index i holds {photo_id: rank} for the
    # i-th filter. Ranks are 0-indexed positions in each filter's
    # ordered output. Used for Reciprocal Rank Fusion — see
    # _attach_rrf_scores.
    ranks_per_set: list[dict[int, int]] = []

    # Extract registered person names from the query so "Calvin and Ellie"
    # becomes an AND-intersection of Calvin's and Ellie's photos instead of
    # a CLIP embedding of the literal string.
    #
    # One match → reuse the existing single-person path.
    # Two+ matches → run a single SQL intersection via
    # `search_by_all_persons`. Calling `search_by_person` per name and
    # intersecting dicts in memory is broken at scale: each per-person call
    # caps at `limit*3` photos ordered ASC by date, so for a 3+-way search
    # where one subject is recent and others have years of older photos,
    # the oldest-N windows may not overlap and the intersection silently
    # collapses to empty — exactly the symptom where "Calvin and Ellie and
    # Nicole" returns nothing despite many family photos existing.
    name_matched: list[dict] = []
    residual = None
    if effective_query:
        residual, name_matched = _extract_persons_from_query(db, effective_query)

    # Compose the index-backed structured filters into one id scope when
    # two or more structured filters are combined (see _compose_scope).
    scope_person_ids = [p["id"] for p in name_matched] + list(person_ids or [])
    if person:
        named = db.get_person_by_name(person)
        if named:
            scope_person_ids.append(named["id"])
    n_structured = (len(scope_person_ids) + bool(camera) + bool(date_from)
                    + bool(location) + bool(category) + bool(visual_tag)
                    + bool(keyword))
    scope: Optional[list[int]] = None
    if n_structured >= 2:
        scope = _compose_scope(
            db, date_from=date_from, date_to=date_to, camera=camera,
            person_ids=list(dict.fromkeys(scope_person_ids)),
            match_source=match_source)
        if scope is not None:
            _log.info("SEARCH SCOPE  %d photos", len(scope))

    # People-only searches select narrow rows and hydrate only the page.
    # Every other filter's rows can become result_sets[0] (whose dicts are
    # returned), so any of them keeps full rows.
    query_left = (residual if name_matched else effective_query)
    narrow = bool(name_matched or person or person_ids) and not (
        query_left or face_image or color or place or location or category
        or visual_tag or keyword or camera)
    person_cols = _NARROW_COLUMNS if narrow else "p.*"

    if name_matched:
        _log.info(
            "QUERY NAMES: matched %s  residual=%r",
            [p["name"] for p in name_matched],
            residual,
        )
        if len(name_matched) == 1:
            results = search_by_person(
                db, name_matched[0]["name"],
                limit=_FILTER_PREFETCH_LIMIT, match_source=match_source,
                scope=scope, columns=person_cols,
            )
        else:
            results = search_by_all_persons(
                db, [p["id"] for p in name_matched],
                limit=_FILTER_PREFETCH_LIMIT, match_source=match_source,
                scope=scope, columns=person_cols,
            )
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})
        effective_query = residual if residual else None

    if person:
        results = search_by_person(
            db, person, limit=_FILTER_PREFETCH_LIMIT, match_source=match_source,
            scope=scope, columns=person_cols)
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    # Pre-resolved person ids (AND-intersection). The LLM tool layer resolves
    # names → ids against `list_people` and passes them here directly, rather
    # than stuffing names back into `query` for `_extract_persons_from_query`
    # to re-parse. Routes through the same `search_by_all_persons` SQL
    # intersection that the name-extraction path uses, so a single id and a
    # three-way "everyone together" search behave identically.
    if person_ids:
        results = search_by_all_persons(
            db, person_ids, limit=_FILTER_PREFETCH_LIMIT, match_source=match_source,
            scope=scope, columns=person_cols)
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    # Face-image reference stays ranked-by-similarity: limit*3 gives a
    # usable top-N that the intersection step ranks against.
    if face_image:
        results = search_by_face_reference(db, face_image, limit=limit * 3)
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    if effective_query:
        # Filename shortcut: if the query looks like a camera filename (no spaces,
        # alphanumeric serial pattern), try a direct DB lookup first.
        # CLIP has no understanding of filenames, so semantic search would return
        # random visually-similar photos instead of the specific file.
        # If filename search finds nothing, fall through to CLIP as normal.
        if _looks_like_filename(effective_query):
            fname_results = search_by_filename(
                db, effective_query, limit=_FILTER_PREFETCH_LIMIT)
            if fname_results:
                result_sets.append({r["id"]: r for r in fname_results})
                ranks_per_set.append(
                    {r["id"]: i for i, r in enumerate(fname_results)})
                effective_query = None  # Skip CLIP — filename match is authoritative

        # CLIP semantic limit: when combined with other filters, widen
        # the net to _FILTER_PREFETCH_LIMIT so the intersection doesn't
        # collapse. "Calvin at the beach" returned 0 because Calvin's
        # beach photos ranked outside CLIP's top-3000 for "at the
        # beach" — non-Calvin beach landscapes dominated the top-N,
        # and strict intersection with Calvin's photos then dropped
        # everything. With the wider limit, CLIP's long tail of above-
        # threshold matches gives the intersection enough candidates.
        # For pure CLIP queries we keep limit*3 since top-N is meaningful
        # when it's the only ranking signal.
        if effective_query:
            clip_limit = _FILTER_PREFETCH_LIMIT if result_sets else limit * 3
            results = search_semantic(db, effective_query, limit=clip_limit,
                                      min_score=min_score, debug=debug,
                                      text_match=text_match)
            result_sets.append({r["id"]: r for r in results})
            ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    if color:
        results = search_by_color(db, color, limit=_FILTER_PREFETCH_LIMIT)
        result_sets.append({r["photo_id"]: r for r in results})
        ranks_per_set.append({r["photo_id"]: i for i, r in enumerate(results)})

    if place:
        results = search_by_place(db, place, limit=_FILTER_PREFETCH_LIMIT)
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    if location:
        results = _search_by_location(db, location, limit=_FILTER_PREFETCH_LIMIT,
                                      scope=scope)
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    # JSON tag filters: membership is tested in Python, but the scan is of a
    # covering (column, date_taken) index, not the 520 MB table. `+date_taken`
    # and `+id` keep the planner on that index rather than idx_photos_date or
    # rowid lookups (measured: a wide date range otherwise reads 399 MB).
    def _tag_rows(col: str, matches) -> list[dict]:
        sql = f"SELECT id, {col} FROM photos WHERE {col} IS NOT NULL"
        params: list = []
        if date_from:
            sql += " AND +date_taken >= ? AND +date_taken <= ?"
            params.extend(_date_bounds(date_from, date_to))
        scope_sql, scope_params = _scope_clause("+id", scope)
        sql += scope_sql
        params.extend(scope_params)
        ids = []
        for row in db.conn.execute(sql, params):
            try:
                if matches(json.loads(row[col])):
                    ids.append(row["id"])
            except (ValueError, TypeError):
                pass
        return _fetch_photos(db, ids)

    if category:
        cat_lower = category.lower()
        matched_rows = _tag_rows(
            "categories", lambda v: cat_lower in {c.lower() for c in v})
        result_sets.append({r["id"]: r for r in matched_rows})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(matched_rows)})

    if visual_tag:
        vis_lower = visual_tag.lower()
        matched_rows = _tag_rows(
            "visual_tags", lambda v: vis_lower in {t.lower() for t in v})
        result_sets.append({r["id"]: r for r in matched_rows})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(matched_rows)})

    if keyword:
        kw_lower = keyword.lower()
        matched_rows = _tag_rows(
            "keywords", lambda v: any(kw_lower in k.lower() for k in v))
        result_sets.append({r["id"]: r for r in matched_rows})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(matched_rows)})

    if camera:
        # idx_photos_camera_date serves camera alone and camera + date range.
        sql, params = "SELECT * FROM photos WHERE camera_model = ?", [camera]
        if date_from:
            sql += " AND date_taken >= ? AND date_taken <= ?"
            params.extend(_date_bounds(date_from, date_to))
        scope_sql, scope_params = _scope_clause("id", scope)
        rows = db.conn.execute(sql + scope_sql, params + scope_params).fetchall()
        results = [dict(r) for r in rows]
        result_sets.append({r["id"]: r for r in results})
        ranks_per_set.append({r["id"]: i for i, r in enumerate(results)})

    def _wrap(items: list[dict]):
        """Honor the caller's with_total + sort preferences on early-
        return paths. Pagination slicing happens here so these paths
        match the main path's contract."""
        items = _apply_sort(items, sort)
        page = items[offset:offset + limit] if limit else items[offset:]
        return (page, len(items)) if with_total else page

    _aes_filtered = _has_aesthetic_filters(
        min_aesthetic, min_technical, min_composition, min_impact, style_tag,
        min_subject_aesthetic, min_day_aesthetic)

    # Date as a primary search: if only date is specified (no other criteria)
    # No limit — return all photos in the range so the user sees every shot from that day.
    # Aesthetic floors must NOT take this shortcut — they'd be silently dropped
    # (the aesthetics-only branch below handles date + aesthetic instead).
    if date_from and not result_sets and min_quality is None and not _aes_filtered:
        return _wrap(_search_by_date(db, date_from, date_to or _OPEN_DATE_HI, limit=0))

    # Quality-only search: if no other criteria given but min_quality is set,
    # return the highest-quality photos in the collection. min_quality is a floor
    # on the RAW aesthetic score the photo modal shows — the VLM `aes_overall`
    # (1-10) when scored, else the legacy LAION `aesthetic_score` — so setting
    # "min quality 5.5" matches the 5.5 displayed on a photo.
    if not result_sets and min_quality is not None and not style_tag:
        # Paginated in SQL: the floors, the date range and the sort all have
        # an exact SQL form, so only the requested page is read.
        where = ["COALESCE(aes_overall, aesthetic_score) IS NOT NULL",
                 "COALESCE(aes_overall, aesthetic_score) >= ?"]
        params: list = [min_quality]
        if date_from:
            where.append("date_taken >= ? AND date_taken <= ?")
            params.extend(_date_bounds(date_from, date_to))
        floors, floor_params = _aesthetic_floor_sql(
            min_aesthetic, min_technical, min_composition, min_impact,
            min_subject_aesthetic, min_day_aesthetic)
        # `sort`, not sort_quality: these early returns go through _wrap,
        # which has only ever applied `sort`.
        order_by = _sort_sql(
            sort, "COALESCE(aes_overall, aesthetic_score) DESC, id DESC")
        if order_by:
            return _sql_page(db, where + floors, params + floor_params,
                             order_by, offset, limit, with_total)

    if not result_sets and min_quality is not None:
        # `COALESCE(aes_overall, aesthetic_score)` must stay textually identical
        # to the idx_photos_raw_quality expression (schema v34).
        sql = """SELECT * FROM photos
               WHERE COALESCE(aes_overall, aesthetic_score) IS NOT NULL
                 AND COALESCE(aes_overall, aesthetic_score) >= ?"""
        params: list = [min_quality]
        if date_from:
            sql += " AND +date_taken >= ? AND +date_taken <= ?"
            params.extend(_date_bounds(date_from, date_to))
        rows = db.conn.execute(
            sql + " ORDER BY COALESCE(aes_overall, aesthetic_score) DESC", params,
        ).fetchall()
        results = [dict(r) for r in rows]
        if date_from:
            results = _filter_by_date(results, date_from, date_to or _OPEN_DATE_HI)
        if _aes_filtered:
            results = _filter_aesthetic(
                results, min_aesthetic, min_technical, min_composition,
                min_impact, style_tag, min_subject_aesthetic, min_day_aesthetic)
        return _wrap(results)

    # Aesthetics-only browse: no content filters, just aesthetic thresholds
    # and/or the aesthetic sort — return the library ranked by percentile.
    if not result_sets and (_aes_filtered
                            or sort in ("aesthetic_desc", "subject_aesthetic_desc")):
        # Subject sort ranks by the subject-crop percentile, falling back to the
        # full-frame percentile for photos without a subject score.
        # The COALESCE must stay textually identical to idx_photos_subject_aes
        # (schema v34); guarding on the same expression lets that index serve
        # both the filter and the order. Same rows: no photo has a subject
        # percentile without a full-frame one (0 of 72,167 on 2026-10-04).
        order = ("COALESCE(aes_subject_overall_pct, aes_overall_pct)"
                 if sort == "subject_aesthetic_desc" else "aes_overall_pct")
        if not style_tag:
            # Paginated in SQL (it read 846 MB to show one page). Under the
            # guard {order} is never NULL, so for the matching sort the
            # tie-broken order is just the index order.
            base = f"{order} DESC, id DESC"
            order_by = _sort_sql(sort, base)  # `sort`, as _wrap applies it
            if order_by and sort == ("subject_aesthetic_desc"
                                     if order != "aes_overall_pct"
                                     else "aesthetic_desc"):
                order_by = base  # same order, and the index can serve it
            if order_by:
                where = [f"{order} IS NOT NULL"]
                params: list = []
                if date_from:
                    where.append("date_taken >= ? AND date_taken <= ?")
                    params.extend(_date_bounds(date_from, date_to))
                floors, floor_params = _aesthetic_floor_sql(
                    min_aesthetic, min_technical, min_composition, min_impact,
                    min_subject_aesthetic, min_day_aesthetic)
                return _sql_page(db, where + floors, params + floor_params,
                                 order_by, offset, limit, with_total)
        rows = db.conn.execute(
            f"SELECT * FROM photos WHERE {order} IS NOT NULL "
            f"ORDER BY {order} DESC"
        ).fetchall()
        results = _filter_aesthetic(
            [dict(r) for r in rows], min_aesthetic, min_technical,
            min_composition, min_impact, style_tag, min_subject_aesthetic,
            min_day_aesthetic)
        if date_from:
            results = _filter_by_date(results, date_from, date_to or _OPEN_DATE_HI)
        return _wrap(results)

    if not result_sets:
        return _wrap([])

    # Attach RRF scores to photos in the primary result set. For multi-
    # filter queries, a photo's rrf_score accumulates contributions
    # from every filter it appears in, rewarding photos that rank high
    # across multiple signals (person + CLIP + location, etc). For
    # single-filter queries, rrf_score is monotonic with the filter's
    # own rank, so sort='relevance' preserves that filter's order.
    _attach_rrf_scores(result_sets, ranks_per_set)

    # In relevance mode, apply recency decay so today's "Calvin at
    # the beach" photo ranks above an equally-relevant 2015 one. No
    # effect on date / quality sorts, which ignore rrf_score.
    if (not sort_quality) and sort == "relevance" and result_sets:
        _apply_recency_decay(list(result_sets[0].values()))

    if len(result_sets) == 1:
        merged = _dedupe_by_hash(list(result_sets[0].values()))
    else:
        # Intersect: only keep photos present in all result sets
        common_ids = set(result_sets[0].keys())
        for rs in result_sets[1:]:
            common_ids &= set(rs.keys())
        # Use first result set for ranking/data
        merged = _dedupe_by_hash(
            [result_sets[0][pid] for pid in common_ids if pid in result_sets[0]]
        )

    # Apply date filter (when date is combined with other search criteria)
    if date_from:
        merged = _filter_by_date(merged, date_from, date_to or _OPEN_DATE_HI)

    # Apply quality filter — floor on the RAW aesthetic score shown on the photo
    # (VLM aes_overall when scored, else legacy aesthetic_score).
    if min_quality is not None:
        def _raw_quality(r):
            v = r.get("aes_overall")
            return v if v is not None else r.get("aesthetic_score")
        merged = [
            r for r in merged
            if _raw_quality(r) is not None and _raw_quality(r) >= min_quality
        ]

    # Apply VLM aesthetic filters (percentile + per-dimension + style tag)
    if _aes_filtered:
        merged = _filter_aesthetic(
            merged, min_aesthetic, min_technical, min_composition,
            min_impact, style_tag, min_subject_aesthetic, min_day_aesthetic)

    # Back-compat: sort_quality=True overrides sort to quality_desc.
    # Prefer the explicit `sort` param in new callers.
    effective_sort = "quality_desc" if sort_quality else sort
    merged = _apply_sort(merged, effective_sort)

    total = len(merged)
    page = merged[offset:offset + limit]
    if narrow:
        page = _hydrate(db, page)
    if with_total:
        return page, total
    return page


def make_results_subdir(base_dir: str, query_parts: dict) -> str:
    """Generate a timestamped subfolder name from search criteria.

    Example: results/2026-03-29_14-32-05_q-beach_color-blue
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp]
    if query_parts.get("query"):
        slug = query_parts["query"].replace(" ", "-")[:30]
        parts.append(f"q-{slug}")
    if query_parts.get("color"):
        parts.append(f"color-{query_parts['color'].lstrip('#')}")
    if query_parts.get("place"):
        slug = query_parts["place"].replace(" ", "-")[:20]
        parts.append(f"place-{slug}")
    if query_parts.get("person"):
        slug = query_parts["person"].replace(" ", "-")[:20]
        parts.append(f"person-{slug}")
    return str(Path(base_dir) / "_".join(parts))


def symlink_results(results: list[dict], output_dir: str = "results", clear: bool = False,
                    thumbnail_size: int = 1200):
    """Write results to an output directory with both a symlink and a JPEG thumbnail per photo.

    For each result, two files are created:
      001_DSC04878.JPG          — relative symlink to the original (full resolution)
      001_DSC04878_thumbnail.JPG — resized JPEG for Finder preview

    The original photos are never modified.
    """
    from PIL import Image as PilImage

    output_path = Path(output_dir)

    if clear and output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results, 1):
        filepath = result.get("filepath")
        if not filepath or not os.path.exists(filepath):
            continue

        filename = os.path.basename(filepath)
        stem = Path(filename).stem
        ext = Path(filename).suffix  # preserve original extension (e.g. .JPG)

        base_name = f"{i:03d}_{stem}"
        link_path = output_path / f"{base_name}{ext}"
        thumb_path = output_path / f"{base_name}_thumbnail.jpg"

        # Relative symlink to original (full resolution)
        try:
            rel_target = os.path.relpath(filepath, str(output_path))
            os.symlink(rel_target, link_path)
        except OSError as e:
            print(f"  Warning: could not symlink {filename}: {e}")

        # JPEG thumbnail for Finder preview
        try:
            with PilImage.open(filepath) as img:
                img = img.convert("RGB")
                img.thumbnail((thumbnail_size, thumbnail_size), PilImage.LANCZOS)
                img.save(thumb_path, "JPEG", quality=85)
        except Exception as e:
            print(f"  Warning: could not create thumbnail for {filename}: {e}")

    return str(output_path.resolve())


# ------------------------------------------------------------------
# Color name resolution
# ------------------------------------------------------------------

_COLOR_NAMES = {
    "red": "#ff0000", "green": "#00aa00", "blue": "#0000ff",
    "yellow": "#ffff00", "orange": "#ff8800", "purple": "#8800aa",
    "pink": "#ff69b4", "brown": "#8b4513", "black": "#000000",
    "white": "#ffffff", "gray": "#808080", "grey": "#808080",
    "cyan": "#00ffff", "teal": "#008080", "navy": "#000080",
    "gold": "#ffd700", "silver": "#c0c0c0", "beige": "#f5f5dc",
    "tan": "#d2b48c", "olive": "#808000", "maroon": "#800000",
    "coral": "#ff7f50", "salmon": "#fa8072", "turquoise": "#40e0d0",
    "violet": "#ee82ee", "indigo": "#4b0082", "magenta": "#ff00ff",
    "lime": "#00ff00", "aqua": "#00ffff", "sky blue": "#87ceeb",
}


def _resolve_color_name(color: str) -> str:
    """Convert a color name to hex, or pass through if already hex."""
    if color.startswith("#"):
        return color
    return _COLOR_NAMES.get(color.lower().strip(), f"#{color}")

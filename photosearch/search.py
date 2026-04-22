"""Search logic for local-photo-search.

Combines CLIP semantic search, text search, color search, and face/person search.
"""

import json
import logging
import os
import shutil
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
    rows = db.conn.execute(
        """SELECT * FROM photos
           WHERE filename LIKE ? OR filepath LIKE ?
           ORDER BY date_taken DESC
           LIMIT ?""",
        (pattern, pattern, limit),
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
# Maps query words to tags from TAG_VOCABULARY. When a user searches "animal",
# we check the photo's pre-computed tags for "animal", "bird", "fish", "pet",
# "wildlife", etc. This is the LLM-powered alternative to _TERM_EXPANSIONS.

_QUERY_TO_TAGS: dict[str, set[str]] = {
    # Animals
    "animal": {"animal", "bird", "fish", "insect", "pet", "wildlife"},
    "animals": {"animal", "bird", "fish", "insect", "pet", "wildlife"},
    "wildlife": {"wildlife", "animal", "bird"},
    "bird": {"bird", "animal", "wildlife"},
    "birds": {"bird", "animal", "wildlife"},
    "fish": {"fish", "animal"},
    "pet": {"pet", "animal"},
    "pets": {"pet", "animal"},
    "insect": {"insect", "animal"},
    # People
    "people": {"person", "child", "group", "crowd", "portrait"},
    "person": {"person", "child", "portrait"},
    "child": {"child", "person"},
    "children": {"child", "person", "group"},
    "kid": {"child", "person"},
    "kids": {"child", "person", "group"},
    "family": {"person", "child", "group"},
    "portrait": {"portrait", "person"},
    # Activities
    "action": {"action", "sports", "running", "climbing", "surfing", "swimming"},
    "sports": {"sports", "action", "surfing", "swimming", "running", "climbing"},
    "playing": {"playing", "action"},
    # Scenes
    "landscape": {"landscape", "mountain", "forest", "desert"},
    "seascape": {"seascape", "ocean", "beach"},
    "beach": {"beach", "ocean", "seascape"},
    "ocean": {"ocean", "beach", "seascape"},
    "mountain": {"mountain", "landscape"},
    "forest": {"forest", "landscape", "tree"},
    "sunset": {"sunset", "sky"},
    "sunrise": {"sunrise", "sky"},
    # Nature
    "flower": {"flower", "plant", "garden"},
    "flowers": {"flower", "plant", "garden"},
    "plant": {"plant", "tree", "flower", "garden"},
    "plants": {"plant", "tree", "flower", "garden"},
    "tree": {"tree", "plant", "forest"},
    # Vehicles
    "vehicle": {"vehicle", "car", "boat", "airplane"},
    "vehicles": {"vehicle", "car", "boat", "airplane"},
    "car": {"car", "vehicle"},
    "boat": {"boat", "vehicle"},
    # Food
    "food": {"food", "drink", "eating", "cooking"},
    # Indoor
    "indoor": {"indoor", "home", "kitchen", "room", "office"},
    "home": {"home", "indoor", "room", "kitchen"},
}


def _tags_match_query(tags_json: Optional[str], query: str) -> float:
    """Score how well a photo's tags match a query.

    Uses the same scoring tiers as _description_relevance:
      +DESCRIPTION_BOOST           tags match the query
      -DESCRIPTION_ABSENCE_PENALTY tags exist but none match
      0.0                          no tags available
    """
    if not tags_json:
        return 0.0

    try:
        photo_tags = set(json.loads(tags_json))
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if not photo_tags:
        return 0.0

    query_words = query.lower().split()
    matched = 0

    for word in query_words:
        # Direct tag match
        if word in photo_tags:
            matched += 1
            continue

        # Stem match (basic plural handling)
        stem = word.rstrip("s") if len(word) > 4 else word
        if stem in photo_tags:
            matched += 1
            continue

        # Expand query word to related tags and check
        related_tags = _QUERY_TO_TAGS.get(word, _QUERY_TO_TAGS.get(stem, set()))
        if related_tags & photo_tags:
            matched += 1

    if matched == len(query_words):
        return DESCRIPTION_BOOST

    if matched == 0:
        return -DESCRIPTION_ABSENCE_PENALTY

    return 0.0  # Partial match — neutral


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
        stem = word.rstrip("s") if len(word) > 4 else word
        if _term_in_desc_positive(stem, desc_lower):
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


def search_semantic(
    db: PhotoDB,
    query: str,
    limit: int = 10,
    min_score: float = CLIP_MIN_SCORE,
    debug: bool = False,
    tag_match: str = "both",
) -> list[dict]:
    """Semantic search — combines CLIP similarity, face boost, and description matching.

    tag_match controls how text relevance is computed:
      "dict"  — use dictionary-based term expansion only (original behavior)
      "tags"  — use LLM-generated tags only
      "both"  — use both and take the higher score (default)

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

    # Pre-load tags for tag-based matching
    tag_cache: dict[int, str] = {}
    if tag_match in ("tags", "both"):
        rows = db.conn.execute(
            "SELECT id, tags FROM photos WHERE tags IS NOT NULL"
        ).fetchall()
        tag_cache = {row["id"]: row["tags"] for row in rows}
        _dbg(f"TAG CACHE: {len(tag_cache)} photos have tags (tag_match={tag_match})")

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

        # Description / tag relevance against the positive query
        if positive_query:
            # Dict-based description relevance (original behavior)
            dict_rel = 0.0
            if tag_match in ("dict", "both") and photo_id in desc_cache:
                dict_rel = _description_relevance(desc_cache[photo_id], positive_query)

            # Tag-based relevance
            tag_rel = 0.0
            if tag_match in ("tags", "both") and photo_id in tag_cache:
                tag_rel = _tags_match_query(tag_cache.get(photo_id), positive_query)

            # Pick the relevance score based on mode
            if tag_match == "dict":
                text_rel = dict_rel
            elif tag_match == "tags":
                text_rel = tag_rel
            else:  # "both" — take the better score
                text_rel = max(dict_rel, tag_rel)

            if text_rel > 0 and raw_score < CLIP_MIN_FOR_DESC_BOOST:
                steps.append(f"text_boost_blocked (rel=+{text_rel:.3f} but clip too low)")
                if tag_match == "both":
                    steps.append(f"  dict_rel={dict_rel:+.3f} tag_rel={tag_rel:+.3f}")
            else:
                score += text_rel
                if text_rel != 0:
                    label = "text_boost" if text_rel > 0 else "text_penalty"
                    steps.append(f"{label}={text_rel:+.3f}")
                    if tag_match == "both":
                        steps.append(f"  dict_rel={dict_rel:+.3f} tag_rel={tag_rel:+.3f}")
        elif photo_id not in desc_cache:
            steps.append("no_description")

        steps.append(f"final={score:.5f}")

        if score < min_score:
            reason = f"EXCLUDED {fname}: {' → '.join(steps)} < min_score={min_score}"
            excluded_log.append(reason)
            _dbg(reason)
            continue

        _dbg(f"INCLUDED {fname}: {' → '.join(steps)}")

        photo = db.get_photo(photo_id)
        if photo:
            photo["score"] = score
            photo["clip_score"] = raw_score
            results_by_id[photo_id] = photo

    # Also surface photos that match on description but weren't in CLIP results.
    # These have no CLIP support, so we require face confirmation for people
    # queries to avoid hallucinated descriptions surfacing irrelevant photos.
    if positive_query:
        desc_matches = search_descriptions(db, positive_query, limit=fetch_limit)
        _dbg(f"DESC-ONLY CANDIDATES: {len(desc_matches)} text matches")
        for desc_photo in desc_matches:
            pid = desc_photo["id"]
            if pid not in results_by_id:
                fname = name_cache.get(pid, f"id={pid}")
                # Apply exclusion filter
                desc_text = desc_photo.get("description", "")
                if has_exclusions and _description_contains_excluded(desc_text, excluded_terms):
                    _dbg(f"EXCLUDED {fname} (desc-only): description contains excluded term")
                    continue
                if negate_people and pid in face_counts:
                    _dbg(f"EXCLUDED {fname} (desc-only): has faces but people excluded")
                    continue
                rel = _description_relevance(desc_text, positive_query)
                if rel > 0:
                    if boost_people and pid not in face_counts:
                        _dbg(f"EXCLUDED {fname} (desc-only): people query but no faces")
                        continue
                    desc_photo["score"] = rel
                    desc_photo["clip_score"] = None
                    results_by_id[pid] = desc_photo
                    _dbg(f"INCLUDED {fname} (desc-only): score={rel:.3f}")

    # Re-sort by combined score, descending
    results = sorted(results_by_id.values(), key=lambda r: r["score"], reverse=True)
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


def search_by_person(db: PhotoDB, name: str, limit: int = 10, match_source: str | None = None) -> list[dict]:
    """Find all photos containing a named person.

    Looks up the person by name, then finds all faces linked to that person,
    then returns the distinct photos those faces appear in.

    match_source: if set, only return photos where the face was matched via
    this method ('strict', 'temporal', or 'manual').
    """
    person = db.get_person_by_name(name)
    if not person:
        print(f"  Person '{name}' not found. Use 'add-person' to register them.")
        return []

    sql = """SELECT DISTINCT p.*
           FROM photos p
           JOIN faces f ON f.photo_id = p.id
           WHERE f.person_id = ?"""
    params: list = [person["id"]]

    if match_source:
        sql += " AND f.match_source = ?"
        params.append(match_source)

    sql += " ORDER BY p.date_taken LIMIT ?"
    params.append(limit)

    rows = db.conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def search_by_all_persons(
    db: PhotoDB,
    person_ids: list[int],
    limit: int = 10,
    match_source: str | None = None,
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

    placeholders = ",".join("?" * len(person_ids))
    sql = (
        "SELECT p.* FROM photos p "
        "JOIN faces f ON f.photo_id = p.id "
        f"WHERE f.person_id IN ({placeholders})"
    )
    params: list = list(person_ids)

    if match_source:
        sql += " AND f.match_source = ?"
        params.append(match_source)

    sql += (
        " GROUP BY p.id"
        " HAVING COUNT(DISTINCT f.person_id) = ?"
        " ORDER BY p.date_taken DESC"
        " LIMIT ?"
    )
    params.extend([len(person_ids), limit])

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


def _search_by_bbox(db: PhotoDB, south: float, north: float,
                    west: float, east: float, limit: int = 100) -> list[dict]:
    """Return photos whose GPS falls inside the given bounding box.

    Used by `_search_by_location` as a fallback when a query doesn't
    substring-match any place_name — the offline reverse-geocoder only
    knows cities with population >1000, so photos at smaller places
    (Point Reyes, Marinwood) get labeled with the nearest bigger town
    and the substring match misses them. Nominatim's bbox puts them
    back.
    """
    rows = db.conn.execute(
        """SELECT * FROM photos
           WHERE gps_lat IS NOT NULL AND gps_lon IS NOT NULL
             AND gps_lat BETWEEN ? AND ?
             AND gps_lon BETWEEN ? AND ?
           ORDER BY date_taken
           LIMIT ?""",
        (south, north, west, east, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def _search_by_location(db: PhotoDB, location: str, limit: int = 100) -> list[dict]:
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

    placeholders = " OR ".join(["place_name LIKE ?"] * len(patterns))
    rows = db.conn.execute(
        f"""SELECT * FROM photos
            WHERE place_name IS NOT NULL AND ({placeholders})
            ORDER BY date_taken
            LIMIT ?""",
        (*patterns, limit),
    ).fetchall()
    results = [dict(r) for r in rows]

    # Country-level queries: the code expansion already covers the
    # entire country, and a Nominatim bbox for a whole country is huge
    # and slow. Skip the fallback.
    if code:
        return results

    if not results:
        try:
            candidates, _ = forward_geocode(db, name, limit=1)
        except Exception:
            candidates = []
        if candidates and candidates[0].get("bbox"):
            south, north, west, east = candidates[0]["bbox"]
            results = _search_by_bbox(db, south, north, west, east, limit)

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
    tag_match: str = "both",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    location: Optional[str] = None,
    match_source: Optional[str] = None,
) -> list[dict]:
    """Run multiple search types and merge results.

    When multiple criteria are given, returns the intersection
    ranked by the primary search type (person > semantic > color > place).

    Args:
        min_quality: If set, filter out photos with aesthetic_score below this value.
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
    if effective_query:
        residual, name_matched = _extract_persons_from_query(db, effective_query)
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
                )
            else:
                results = search_by_all_persons(
                    db, [p["id"] for p in name_matched],
                    limit=_FILTER_PREFETCH_LIMIT, match_source=match_source,
                )
            result_sets.append({r["id"]: r for r in results})
            effective_query = residual if residual else None

    if person:
        results = search_by_person(
            db, person, limit=_FILTER_PREFETCH_LIMIT, match_source=match_source)
        result_sets.append({r["id"]: r for r in results})

    # Face-image reference stays ranked-by-similarity: limit*3 gives a
    # usable top-N that the intersection step ranks against.
    if face_image:
        results = search_by_face_reference(db, face_image, limit=limit * 3)
        result_sets.append({r["id"]: r for r in results})

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
                effective_query = None  # Skip CLIP — filename match is authoritative

        # CLIP semantic stays ranked (limit*3): scores are real, we
        # want the top-N to be the best-scoring photos before the
        # intersection, not every photo over the noise floor.
        if effective_query:
            results = search_semantic(db, effective_query, limit=limit * 3, min_score=min_score, debug=debug, tag_match=tag_match)
            result_sets.append({r["id"]: r for r in results})

    if color:
        results = search_by_color(db, color, limit=_FILTER_PREFETCH_LIMIT)
        result_sets.append({r["photo_id"]: r for r in results})

    if place:
        results = search_by_place(db, place, limit=_FILTER_PREFETCH_LIMIT)
        result_sets.append({r["id"]: r for r in results})

    if location:
        results = _search_by_location(db, location, limit=_FILTER_PREFETCH_LIMIT)
        result_sets.append({r["id"]: r for r in results})

    # Date as a primary search: if only date is specified (no other criteria)
    # No limit — return all photos in the range so the user sees every shot from that day.
    if date_from and not result_sets and min_quality is None:
        results = _search_by_date(db, date_from, date_to or date_from, limit=0)
        return results

    # Quality-only search: if no other criteria given but min_quality is set,
    # return the highest-quality photos in the collection.
    if not result_sets and min_quality is not None:
        rows = db.conn.execute(
            """SELECT * FROM photos
               WHERE aesthetic_score IS NOT NULL AND aesthetic_score >= ?
               ORDER BY aesthetic_score DESC
               LIMIT ?""",
            (min_quality, limit),
        ).fetchall()
        results = [dict(r) for r in rows]
        if date_from:
            results = _filter_by_date(results, date_from, date_to or date_from)
        return results

    if not result_sets:
        return []

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
        merged = _filter_by_date(merged, date_from, date_to or date_from)

    # Apply quality filter
    if min_quality is not None:
        merged = [
            r for r in merged
            if r.get("aesthetic_score") is not None and r["aesthetic_score"] >= min_quality
        ]

    # Optionally re-sort by aesthetic quality instead of relevance
    if sort_quality:
        merged.sort(
            key=lambda r: r.get("aesthetic_score") or 0,
            reverse=True,
        )

    return merged[:limit]


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

"""Mine candidate vocabulary terms from photo descriptions.

Uses spaCy noun-chunk extraction + lemmatization. Designed to be run once
per vocab refresh; output is reviewed in /admin/vocab (Phase 3).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

_VAGUE_WORDS = {
    "thing", "things", "place", "stuff", "scene", "image", "photo",
    "picture", "view", "side", "part", "area", "background",
    # Added 2026-05-18 after first mining run on the live library surfaced
    # these as top-frequency bare nouns that mean nothing as content tags.
    # Filtered only when the WHOLE noun-chunk is vague — "outdoor setting"
    # still passes because "outdoor" is informative; bare "setting" drops.
    "setting", "appearance", "look", "presence", "composition",
    "foreground", "environment",
}


def _get_nlp():
    import spacy
    # Reuse a single instance per process; spaCy load is expensive (~2s).
    global _NLP
    try:
        return _NLP  # type: ignore[name-defined]
    except NameError:
        pass
    _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # Noun-chunk extraction requires the parser; re-enable lightly.
    _NLP.enable_pipe("parser") if "parser" in _NLP.disabled else None
    return _NLP


def extract_noun_phrases(text: str) -> list[str]:
    """Return lemmatized, lowercased noun chunks from one description."""
    if not text:
        return []
    nlp = _get_nlp()
    doc = nlp(text)
    seen: set[str] = set()
    out: list[str] = []
    for chunk in doc.noun_chunks:
        toks = [t for t in chunk if not t.is_stop and not t.is_punct]
        if not toks:
            continue
        if all(t.lemma_.lower() in _VAGUE_WORDS for t in toks):
            continue
        phrase = " ".join(t.lemma_.lower() for t in toks)
        if not phrase or phrase in _VAGUE_WORDS:
            continue
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
    return out


def mine_corpus(descriptions: Iterable[str], min_count: int = 50) -> list[dict]:
    """Return [{term, count}, ...] sorted by count desc, count >= min_count."""
    counter: Counter[str] = Counter()
    for desc in descriptions:
        for phrase in extract_noun_phrases(desc):
            counter[phrase] += 1
    return [
        {"term": term, "count": count}
        for term, count in counter.most_common()
        if count >= min_count
    ]

import pytest


def test_extract_noun_phrases_lemmatizes_and_lowercases():
    from photosearch.vocab_mining import extract_noun_phrases
    out = extract_noun_phrases(
        "Two golden retrievers running on Stinson Beach during sunset."
    )
    # Multi-word phrases preserved; nouns lemmatized; lowercase.
    assert "golden retriever" in out
    assert "stinson beach" in out
    assert "sunset" in out


def test_extract_noun_phrases_skips_stopword_only_chunks():
    from photosearch.vocab_mining import extract_noun_phrases
    out = extract_noun_phrases("The thing in the place looks nice.")
    # 'thing' / 'place' filtered as vague.
    assert "thing" not in out
    assert "place" not in out


def test_mine_corpus_returns_frequency_sorted_filtered_list():
    from photosearch.vocab_mining import mine_corpus
    descriptions = [
        "A dog on the beach.",
        "Two dogs at the beach.",
        "A child on the beach.",
        "A bird in the sky.",
    ]
    # min_count=2 → 'beach' (3) and 'dog' (2) qualify; 'bird' (1), 'sky' (1), 'child' (1) drop.
    out = mine_corpus(descriptions, min_count=2)
    terms = [row["term"] for row in out]
    assert terms[0] == "beach"
    assert "dog" in terms
    assert "bird" not in terms
    # Frequency monotone non-increasing.
    counts = [row["count"] for row in out]
    assert counts == sorted(counts, reverse=True)

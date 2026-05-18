import pytest


def test_compile_renders_content_module(tmp_path):
    from photosearch.vocab_compile import render_content_module
    src = render_content_module(
        terms=["beach", "mountain", "golden retriever"],
        draft_hash="abc123",
        timestamp="2026-05-17T18:00:00",
    )
    # Module is importable Python.
    namespace = {}
    exec(src, namespace)
    assert namespace["CONTENT_VOCABULARY"] == ["beach", "golden retriever", "mountain"]
    # Header comment present with metadata.
    assert "abc123" in src
    assert "2026-05-17" in src


def test_compile_renders_query_expansion_module(tmp_path):
    from photosearch.vocab_compile import render_query_expansion_module
    src = render_query_expansion_module(
        expansions={"dog": ["puppy", "pup"], "ocean": ["sea", "pacific ocean"]},
        draft_hash="def456",
        timestamp="2026-05-17T18:00:00",
    )
    namespace = {}
    exec(src, namespace)
    assert namespace["_QUERY_TO_CATEGORIES"] == {
        "dog": {"puppy", "pup"},
        "ocean": {"sea", "pacific ocean"},
    }


def test_compile_validates_disjoint_vocabs():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["beach", "mountain"],
        "visual": ["beach", "dramatic"],  # 'beach' in both → error
        "expansions": {},
    }
    with pytest.raises(VocabError, match="appear in both"):
        validate_draft(draft)


def test_compile_validates_content_floor():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["beach", "mountain"],  # < 50 terms
        "visual": ["dramatic"],
        "expansions": {},
    }
    with pytest.raises(VocabError, match="at least 50"):
        validate_draft(draft)


def test_compile_validates_nonempty_visual():
    from photosearch.vocab_compile import validate_draft, VocabError
    draft = {
        "content": ["x"] * 60,
        "visual": [],
        "expansions": {},
    }
    with pytest.raises(VocabError, match="visual vocab is empty"):
        validate_draft(draft)

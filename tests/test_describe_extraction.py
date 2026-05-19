import pytest
from photosearch.bakeoff import build_keyword_prompt, parse_keywords_response


def test_build_keyword_prompt_embeds_description():
    desc = "A golden retriever runs along Stinson Beach at sunset."
    prompt = build_keyword_prompt(desc)
    assert desc in prompt
    assert "5-15" in prompt  # range from design


def test_parse_keywords_lowercases_and_trims():
    raw = "Golden Retriever, Stinson Beach, Sunset, dog, beach, Pacific Ocean"
    out = parse_keywords_response(raw)
    assert out == [
        "golden retriever", "stinson beach", "sunset",
        "dog", "beach", "pacific ocean",
    ]


def test_parse_keywords_dedupes_and_drops_empties():
    raw = "dog, , Dog,  beach , beach"
    out = parse_keywords_response(raw)
    assert out == ["dog", "beach"]


def test_parse_keywords_handles_newline_and_bullet_responses():
    raw = "- Golden Retriever\n- beach\n* sunset"
    out = parse_keywords_response(raw)
    assert out == ["golden retriever", "beach", "sunset"]


def test_extract_categories_returns_only_vocab_terms(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "beach, dog, glorbnox, mountain")
    monkeypatch.setattr("photosearch.vocab_content.CONTENT_VOCABULARY",
                        ["beach", "dog", "mountain", "sky"])
    out = d.extract_categories_from_description("a dog at the beach")
    assert set(out) == {"beach", "dog", "mountain"}


def test_extract_categories_returns_empty_on_empty_response(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry", lambda **kw: "")
    monkeypatch.setattr("photosearch.vocab_content.CONTENT_VOCABULARY",
                        ["beach", "dog"])
    assert d.extract_categories_from_description("a dog") == []


def test_extract_categories_returns_empty_on_none_description(monkeypatch):
    from photosearch import describe as d
    assert d.extract_categories_from_description(None) == []
    assert d.extract_categories_from_description("") == []


def test_extract_keywords_uses_bakeoff_parser(monkeypatch):
    from photosearch import describe as d
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "Golden Retriever, Stinson Beach, sunset")
    out = d.extract_keywords_from_description("a dog at the beach")
    assert out == ["golden retriever", "stinson beach", "sunset"]


def test_extract_keywords_returns_empty_on_none(monkeypatch):
    from photosearch import describe as d
    assert d.extract_keywords_from_description(None) == []
    assert d.extract_keywords_from_description("") == []


def test_tag_visual_photo_uses_visual_vocab(monkeypatch, tmp_path):
    from photosearch import describe as d
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")  # minimal jpeg stub
    monkeypatch.setattr("photosearch.vocab_visual.VISUAL_VOCABULARY",
                        ["dramatic", "peaceful", "foggy"])
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: "dramatic, peaceful, nope")
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "encoded")
    out = d.tag_visual_photo(str(img))
    assert out == ["dramatic", "peaceful"]


def test_tag_visual_photo_regurgitation_guard_at_12(monkeypatch, tmp_path):
    from photosearch import describe as d
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    big_vocab = [f"v{i}" for i in range(25)]
    monkeypatch.setattr("photosearch.vocab_visual.VISUAL_VOCABULARY", big_vocab)
    # First call returns 12 (regurgitation); retry returns the same 12 → guard returns None.
    monkeypatch.setattr(d, "_ollama_chat_with_retry",
                        lambda **kw: ", ".join(big_vocab[:12]))
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda p: "encoded")
    out = d.tag_visual_photo(str(img))
    assert out is None

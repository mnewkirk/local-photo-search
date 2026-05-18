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

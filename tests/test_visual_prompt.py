"""The category-visual prompt.

Two failures shaped it, both measured on live output.

The FLAT CHECKLIST ("Include every tag that clearly applies", no cap) produced
163 distinct tag sets across 1,373 photos, one verbatim 8-tag set on 101 of
them.

Its replacement grouped the vocabulary into AXES and said "Pick at most ONE tag
from each group" — which the model read as "pick one FROM EACH group", so every
photo collected a viewpoint tag and a composition tag whether or not one
applied (`close-up, sunny, symmetrical` on a person walking down stairs). An
11-photo A/B against hand-labelled tag sets: axes prompt 16 right / 5 wrong /
3.3 tags per photo; a RARE-section prompt with no worked examples 14 right /
0 wrong / 1.8 tags; adding two examples back leaked `close-up` from the example
onto an unrelated photo.

So: no axes to fill, no worked examples.
"""

import pytest

from photosearch import describe as D
from photosearch import visual_tags_derive as V


@pytest.fixture
def prompt():
    return D._build_visual_prompt(V.PERCEIVED_VOCABULARY)


# ---------------------------------------------------------------------------
# What the prompt shows
# ---------------------------------------------------------------------------

def _listed_terms(prompt):
    """Every term the prompt OFFERS, in order — i.e. outside the parenthetical
    definitions. A term may legitimately recur inside its own definition
    ("people seen full-length are never close-up")."""
    import re

    stripped = re.sub(r"\([^()]*\)", "", prompt)
    return [t for t in re.findall(r"[a-z][a-z-]+", stripped)
            if t in set(V.PERCEIVED_VOCABULARY)]


def test_every_perceived_term_is_offered_exactly_once(prompt):
    listed = _listed_terms(prompt)
    assert sorted(listed) == sorted(V.PERCEIVED_VOCABULARY)
    assert len(listed) == len(set(listed)), "a term is offered twice"


def test_no_derived_frozen_or_retired_term_is_offered(prompt):
    for term in set(V.CAPTURE_FACT_TAGS) | set(V.FROZEN_TAGS) | set(V.RETIRED_TAGS):
        assert term not in prompt, f"{term} is not the model's to decide"


def test_the_one_per_group_rule_is_gone(prompt):
    low = prompt.lower()
    for phrase in ("from each group", "one tag from each", "at most one tag from",
                   "one per axis", "each axis"):
        assert phrase not in low, f"the axes-to-fill misreading is back: {phrase!r}"


def test_the_prompt_says_most_photos_get_few_tags(prompt):
    assert "Most photos deserve 1 to 3 tags" in prompt
    assert "many deserve only one" in prompt


def test_the_prompt_says_not_to_fill_every_section(prompt):
    assert "Do NOT try to use every section" in prompt
    assert "NO viewpoint tag and NO composition tag" in prompt


def test_a_rare_section_exists_and_holds_the_over_applied_terms(prompt):
    assert "RARE" in prompt
    rare = prompt.split("RARE", 1)[1].split("Answer with", 1)[0]
    for term in ("close-up", "macro", "wide-angle", "aerial", "centered",
                 "symmetrical"):
        assert term in rare, f"{term} belongs in the RARE section"


def test_the_rare_section_explains_the_exact_situation(prompt):
    assert "people seen full-length are never close-up" in prompt
    assert "candid photos of people are never symmetrical" in prompt


def test_there_is_no_worked_example_block(prompt):
    """Variant C leaked `close-up` from its example onto a baseball photo, so
    the prompt demonstrates no answer at all. The only bare tag lines it may
    contain are the section listings themselves — nothing after the response
    instruction, and nothing introduced as an example."""
    assert "Example" not in prompt and "example" not in prompt

    vocab = set(V.PERCEIVED_VOCABULARY)
    head, _, tail = prompt.partition("Answer with ONLY")
    section_body = head.split("LIGHT AND WEATHER", 1)[1]
    for region, name in ((tail, "after the response instruction"),
                         (head.split("LIGHT AND WEATHER", 1)[0], "before the sections")):
        for line in region.splitlines():
            parts = [p.strip() for p in line.strip().rstrip(".").split(",")]
            if parts and parts[0] and all(p in vocab for p in parts):
                pytest.fail(f"demonstrated answer {name}: {line!r}")
    # Inside the sections, every bare tag line must be an indented listing.
    for line in section_body.splitlines():
        parts = [p.strip() for p in line.strip().rstrip(".").split(",")]
        if parts and parts[0] and all(p in vocab for p in parts):
            assert line.startswith("  "), f"unindented tag line: {line!r}"


def test_the_prompt_states_the_hard_cap(prompt):
    assert f"at most {D._VISUAL_MAX_TAGS}" in prompt


def test_the_prompt_permits_answering_none(prompt):
    assert "If nothing clearly applies, answer: none" in prompt
    assert D._is_explicit_empty_answer("none") is True


def test_the_prompt_keeps_the_comma_separated_response_contract(prompt):
    assert "comma-separated list of tags" in prompt
    assert D._parse_visual_response("peaceful, sunny", set(V.PERCEIVED_VOCABULARY)) \
        == ["peaceful", "sunny"]


def test_the_prompt_is_not_the_old_flat_checklist(prompt):
    assert "Include every tag that clearly applies" not in prompt


# ---------------------------------------------------------------------------
# The section table
# ---------------------------------------------------------------------------

def test_the_sections_partition_the_perceived_vocabulary():
    seen = [t for _, groups in V.PROMPT_SECTIONS for g in groups for t in g]
    assert sorted(seen) == sorted(V.PERCEIVED_VOCABULARY)
    assert len(seen) == len(set(seen)), "a term appears in two sections"


def test_every_gloss_names_a_perceived_term():
    assert set(V.PERCEIVED_GLOSS) <= set(V.PERCEIVED_VOCABULARY)


# ---------------------------------------------------------------------------
# The guard does not enforce one-per-section
# ---------------------------------------------------------------------------

def test_two_light_tags_are_a_legitimate_answer(tmp_path, monkeypatch):
    """`backlit, silhouette` describe one photo honestly. Nothing may reject
    an answer merely for drawing twice from the same section."""
    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")
    monkeypatch.setattr(D, "_ollama_chat_with_retry",
                        lambda **kw: "backlit, silhouette")
    monkeypatch.setattr(D, "_encode_image_for_ollama", lambda p: "encoded")
    assert D.tag_visual_photo(str(img)) == ["backlit", "silhouette"]
    assert D._visual_answer_problem(["backlit", "silhouette"]) is None


# ---------------------------------------------------------------------------
# Decoding options: temperature stays at 0
# ---------------------------------------------------------------------------

def test_visual_pass_stays_greedy():
    assert D._options_for_model("llava")["temperature"] == 0

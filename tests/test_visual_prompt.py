"""The category-visual prompt, and the guard around the answer.

The old prompt was one flat 36-term checklist whose only rule was "Include
every tag that clearly applies" — no cap, no omit-when-in-doubt. On 1,373
photos of one soccer shoot it produced only 163 distinct tag sets, the top 8
covering 56%, one verbatim 8-tag set on 101 consecutive photos. These tests pin
the replacement: axes, a hard cap, and a permission to say nothing.
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

def test_prompt_offers_only_perceived_terms(prompt):
    for term in V.CAPTURE_FACT_TAGS | V.RETIRED_TAGS:
        assert term not in prompt, f"{term} is not the model's to decide"


def test_prompt_offers_every_perceived_term(prompt):
    for term in V.PERCEIVED_VOCABULARY:
        assert term in prompt, term


def test_prompt_groups_the_terms_by_axis(prompt):
    for axis in V.PERCEIVED_AXES:
        assert axis.lower() in prompt.lower(), axis


def test_prompt_states_the_one_per_axis_rule(prompt):
    low = prompt.lower()
    assert "one tag" in low or "at most one" in low
    assert "group" in low or "axis" in low or "line" in low


def test_prompt_states_the_hard_cap(prompt):
    assert str(D._VISUAL_MAX_TAGS) in prompt
    assert "never more than" in prompt.lower() or "at most" in prompt.lower()


def test_prompt_permits_returning_nothing(prompt):
    low = prompt.lower()
    assert "none" in low
    assert "acceptable" in low or "fine" in low or "normal" in low


def test_prompt_tells_the_model_to_omit_when_unsure(prompt):
    low = prompt.lower()
    assert "omit" in low
    assert "unmistakab" in low or "obvious" in low


def test_prompt_carries_a_positive_and_a_negative_example(prompt):
    low = prompt.lower()
    assert "example" in low
    # The negative example is the measured failure: a bright daytime sports
    # frame must not collect `peaceful` or any low-light/night term.
    assert "peaceful" in prompt
    assert "not" in low


def test_prompt_glosses_the_ambiguous_terms(prompt):
    for term, gloss in V.PERCEIVED_GLOSS.items():
        assert gloss in prompt, f"{term} lost its definition"
        assert 3 <= len(gloss.split()) <= 6, f"{term}: {gloss!r} is not 3-6 words"


def test_prompt_keeps_the_comma_separated_response_contract(prompt):
    assert "comma-separated" in prompt.lower()
    # And the parser the contract is written for still reads exactly that.
    assert D._parse_visual_response("peaceful, sunny", set(V.PERCEIVED_VOCABULARY)) \
        == ["peaceful", "sunny"]


def test_prompt_is_not_a_flat_checklist():
    """The failure mode was a single undifferentiated list. The measured
    signature: no cap and an instruction to include everything."""
    prompt = D._build_visual_prompt(V.PERCEIVED_VOCABULARY)
    assert "Include every tag that clearly applies" not in prompt


# ---------------------------------------------------------------------------
# Decoding options: temperature stays at 0
# ---------------------------------------------------------------------------

def test_visual_pass_stays_greedy():
    """A non-zero temperature would trade determinism for diversity. The
    collapse came from the checklist, not from greedy decoding — the prompt is
    the fix, so keep this reproducible."""
    assert D._options_for_model("llava")["temperature"] == 0

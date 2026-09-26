"""The guard on a category-visual answer: over-selection and contradiction.

The old guard tripped at >= 12 tags. The observed maximum on the real failure
was 11 and the median 5, so it never fired at all — it was guarding a threshold
the failure never crossed. It now sits at the cap the prompt states.
"""

import pytest

from photosearch import describe as D
from photosearch import visual_tags_derive as V


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    return str(p)


def _answers(monkeypatch, *replies):
    """Serve `replies` in order to successive _ollama_chat_with_retry calls."""
    box = list(replies)
    calls = []

    def fake(**kw):
        calls.append(kw)
        return box.pop(0) if box else None

    monkeypatch.setattr(D, "_ollama_chat_with_retry", fake)
    monkeypatch.setattr(D, "_encode_image_for_ollama", lambda p: "encoded")
    return calls


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

def test_the_guard_sits_at_the_cap_the_prompt_states():
    assert D._VISUAL_MAX_PLAUSIBLE_TAGS == D._VISUAL_MAX_TAGS + 1 == 6


def test_five_tags_is_accepted(monkeypatch, img):
    _answers(monkeypatch, "sunny, colorful, joyful, centered, close-up")
    assert D.tag_visual_photo(img) == \
        ["sunny", "colorful", "joyful", "centered", "close-up"]


def test_six_tags_triggers_a_retry(monkeypatch, img):
    calls = _answers(monkeypatch,
                     "sunny, colorful, joyful, centered, close-up, symmetrical",
                     "sunny, joyful")
    assert D.tag_visual_photo(img) == ["sunny", "joyful"]
    assert len(calls) == 2
    assert calls[1]["options"]["temperature"] > 0, "the retry bumps temperature"


def test_a_retry_that_still_over_selects_is_rejected_whole(monkeypatch, img):
    """Regurgitation behaviour, unchanged: a model echoing the vocabulary gets
    its whole response dropped rather than truncated."""
    over = "sunny, colorful, joyful, centered, close-up, symmetrical, foggy"
    _answers(monkeypatch, over, over)
    assert D.tag_visual_photo(img) is None


def test_a_failed_retry_after_over_selection_is_rejected(monkeypatch, img):
    _answers(monkeypatch,
             "sunny, colorful, joyful, centered, close-up, symmetrical", None)
    assert D.tag_visual_photo(img) is None


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------

def test_contradiction_triggers_a_retry_and_the_clean_retry_wins(monkeypatch, img):
    calls = _answers(monkeypatch, "overcast, sunny, peaceful", "sunny, peaceful")
    assert D.tag_visual_photo(img) == ["sunny", "peaceful"]
    assert len(calls) == 2


def test_a_still_contradictory_retry_drops_both_members_not_the_answer(
        monkeypatch, img):
    """Drop BOTH sides of each bad pair — neither is trustworthy — but keep the
    tags that were never in question."""
    _answers(monkeypatch, "overcast, sunny, peaceful", "overcast, sunny, peaceful")
    assert D.tag_visual_photo(img) == ["peaceful"]


def test_a_failed_retry_after_a_contradiction_repairs_the_first_answer(
        monkeypatch, img):
    _answers(monkeypatch, "overcast, sunny, peaceful", None)
    assert D.tag_visual_photo(img) == ["peaceful"]


def test_dropping_a_pair_may_empty_the_answer(monkeypatch, img):
    _answers(monkeypatch, "overcast, sunny", "overcast, sunny")
    # None here; _process_category_visual turns it into [] so the photo is
    # still persisted and marked done in one pass.
    assert D.tag_visual_photo(img) is None


def test_every_contradictory_pair_is_detected(monkeypatch, img):
    for a, b in V.CONTRADICTORY_PAIRS:
        _answers(monkeypatch, f"{a}, {b}", f"{a}, {b}")
        assert D.tag_visual_photo(img) is None, f"{a} x {b} was not caught"


def test_a_non_contradictory_pair_is_left_alone(monkeypatch, img):
    # `peaceful` x `moody` is deliberately NOT in the table — a still, misty
    # lake is honestly both, and it is the library's biggest co-occurrence.
    calls = _answers(monkeypatch, "peaceful, moody")
    assert D.tag_visual_photo(img) == ["peaceful", "moody"]
    assert len(calls) == 1, "a legitimate answer must not cost a retry"


@pytest.mark.parametrize("answer", ["dramatic, peaceful", "muted, colorful"])
def test_pairs_the_owner_ruled_compatible_survive(monkeypatch, img, answer):
    # Removed 2026-09-26: a still sunset is both dramatic and peaceful; muted
    # (the light) and colorful (how many hues) are different properties.
    calls = _answers(monkeypatch, answer)
    assert D.tag_visual_photo(img) == answer.split(", ")
    assert len(calls) == 1


def test_drop_contradictions_is_order_independent():
    assert D._drop_visual_contradictions(["overcast", "sunny", "peaceful"]) == ["peaceful"]
    assert D._drop_visual_contradictions(["peaceful", "sunny", "overcast"]) == ["peaceful"]


def test_drop_contradictions_handles_a_chain():
    # black-and-white contradicts both colorful and vibrant; all three go.
    assert D._drop_visual_contradictions(
        ["colorful", "black-and-white", "vibrant", "sunny"]) == ["sunny"]


# ---------------------------------------------------------------------------
# Unchanged behaviour
# ---------------------------------------------------------------------------

def test_out_of_vocabulary_tokens_are_still_dropped(monkeypatch, img):
    _answers(monkeypatch, "sunny, definitely-not-a-tag, joyful")
    assert D.tag_visual_photo(img) == ["sunny", "joyful"]


def test_capture_fact_terms_are_not_in_the_answer_vocabulary(monkeypatch, img):
    """Even if a model volunteers one, the pass no longer accepts it — EXIF
    decides those, server-side."""
    _answers(monkeypatch, "long-exposure, low-light, motion-blur, sunny")
    assert D.tag_visual_photo(img) == ["sunny"]


def test_an_empty_answer_returns_none(monkeypatch, img):
    _answers(monkeypatch, "")
    assert D.tag_visual_photo(img) is None

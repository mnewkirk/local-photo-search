"""The category-visual parser, and what an UNPARSEABLE answer must do.

`_parse_visual_response` used to split on commas and nothing else, so every
shape a small model actually emits — a bullet list, a `Tags:` prefix, a JSON
array, one-per-line — parsed to `[]` or lost its first tag. An empty parse was
indistinguishable from "this photo has no visual qualities", which persisted
`'[]'`: NOT NULL, so the photo left the category-visual queue forever, looking
done, never retried.
"""

import json

import pytest

from photosearch import describe as D
from photosearch.visual_tags_derive import PERCEIVED_VOCABULARY

VOCAB = set(PERCEIVED_VOCABULARY)


# ---------------------------------------------------------------------------
# Parser tolerance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    # the format the prompt asks for
    ("sunny, peaceful", ["sunny", "peaceful"]),
    ("sunny,peaceful", ["sunny", "peaceful"]),
    (" sunny , peaceful .", ["sunny", "peaceful"]),
    # a label prefix — the shape the prompt's own example demonstrated
    ("RIGHT: sunny", ["sunny"]),
    ("Tags: sunny, peaceful", ["sunny", "peaceful"]),
    ("Visual tags: sunny", ["sunny"]),
    ("Answer: sunny, peaceful", ["sunny", "peaceful"]),
    # bullets and numbering
    ("- sunny\n- peaceful", ["sunny", "peaceful"]),
    ("* sunny\n* peaceful", ["sunny", "peaceful"]),
    ("• sunny\n• peaceful", ["sunny", "peaceful"]),
    ("1. sunny\n2. peaceful", ["sunny", "peaceful"]),
    ("1) sunny\n2) peaceful", ["sunny", "peaceful"]),
    # newlines and semicolons as separators
    ("sunny\npeaceful", ["sunny", "peaceful"]),
    ("sunny; peaceful", ["sunny", "peaceful"]),
    ("sunny\r\npeaceful", ["sunny", "peaceful"]),
    # JSON array
    ('["sunny", "peaceful"]', ["sunny", "peaceful"]),
    ('["sunny"]', ["sunny"]),
    ("['sunny', 'peaceful']", ["sunny", "peaceful"]),
    # quoting / fencing / backticks
    ('"sunny", "peaceful"', ["sunny", "peaceful"]),
    ("`sunny`, `peaceful`", ["sunny", "peaceful"]),
    ("```\nsunny, peaceful\n```", ["sunny", "peaceful"]),
    ("```json\n[\"sunny\"]\n```", ["sunny"]),
    # mixed junk
    ("Tags:\n- sunny\n- peaceful\n", ["sunny", "peaceful"]),
    ("RIGHT: sunny, peaceful.", ["sunny", "peaceful"]),
    ("**sunny**, _peaceful_", ["sunny", "peaceful"]),
    # out-of-vocabulary tokens are still dropped, order preserved, deduped
    ("sunny, nope, peaceful, sunny", ["sunny", "peaceful"]),
])
def test_parser_tolerates_the_shapes_models_actually_emit(raw, expected):
    assert D._parse_visual_response(raw, VOCAB) == expected


def test_parser_still_rejects_prose():
    assert D._parse_visual_response(
        "This photograph depicts a football match on grass.", VOCAB) == []


def test_parser_never_invents_a_tag():
    assert D._parse_visual_response("long-exposure, motion-blur", VOCAB) == []


# ---------------------------------------------------------------------------
# Empty vs unparseable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["none", "None.", "NONE", "[]", "no tags",
                                 "n/a", "-", "(none)", "none of the above"])
def test_explicit_empty_answers_are_recognised(raw):
    assert D._is_explicit_empty_answer(raw) is True


@pytest.mark.parametrize("raw", ["RIGHT: sunny", "sunny", "a football match",
                                 "I cannot determine the tags"])
def test_a_real_answer_is_not_an_explicit_empty(raw):
    assert D._is_explicit_empty_answer(raw) is False


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    return str(p)


def _answers(monkeypatch, *replies):
    box = list(replies)
    calls = []

    def fake(**kw):
        calls.append(kw)
        return box.pop(0) if box else None

    monkeypatch.setattr(D, "_ollama_chat_with_retry", fake)
    monkeypatch.setattr(D, "_encode_image_for_ollama", lambda p: "encoded")
    return calls


def test_a_legitimately_empty_answer_is_an_empty_list(monkeypatch, img):
    """`[]`, not None — the model answered, and the answer was 'no tags'."""
    _answers(monkeypatch, "none")
    assert D.tag_visual_photo(img) == []


def test_an_unparseable_answer_retries_then_signals_failure(monkeypatch, img):
    calls = _answers(monkeypatch, "This is a photo of a dog.",
                     "Still just prose about a dog.")
    assert D.tag_visual_photo(img) is None
    assert len(calls) == 2, "an unparseable answer costs one retry"


def test_an_unparseable_answer_whose_retry_parses_is_kept(monkeypatch, img):
    _answers(monkeypatch, "This is a photo of a dog.", "sunny, joyful")
    assert D.tag_visual_photo(img) == ["sunny", "joyful"]


def test_no_response_at_all_is_a_failure_not_an_empty_result(monkeypatch, img):
    _answers(monkeypatch, None)
    assert D.tag_visual_photo(img) is None
    _answers(monkeypatch, "")
    assert D.tag_visual_photo(img) is None


def test_the_worker_omits_the_column_for_a_failed_generation(monkeypatch):
    """A failure sends `visual_tags: None` — a row the server marks processed
    (so a repeatable failure is bounded by MAX_PROCESS_ATTEMPTS) while leaving
    the column NULL (so the photo stays claimable). Same shape as the
    aesthetics pass's empty-scores row; avoids the CLIP-style infinite
    re-claim that simply omitting the photo would cause."""
    from photosearch import worker as W

    monkeypatch.setattr(W, "tag_visual_photo", None, raising=False)
    monkeypatch.setattr("photosearch.describe.tag_visual_photo",
                        lambda path, model=None: None)
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    out = W._process_category_visual([({"id": 7, "filename": "a.jpg"}, "/x")])
    assert out == [{"photo_id": 7, "visual_tags": None}]


def test_the_worker_sends_an_empty_list_for_a_legitimately_empty_answer(
        monkeypatch):
    from photosearch import worker as W

    monkeypatch.setattr("photosearch.describe.tag_visual_photo",
                        lambda path, model=None: [])
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    out = W._process_category_visual([({"id": 7, "filename": "a.jpg"}, "/x")])
    assert out == [{"photo_id": 7, "visual_tags": []}]


# ---------------------------------------------------------------------------
# The prompt must demonstrate only the format the parser wants
# ---------------------------------------------------------------------------

def test_the_prompt_demonstrates_no_answer_at_all():
    """Whatever the prompt shows, a model copies. An earlier draft labelled its
    examples `WRONG:` / `RIGHT:`, which the comma-only parser of the day read
    as zero tags; a later one leaked `close-up` out of an example onto an
    unrelated photo. So there are no examples, and no labels to parrot."""
    prompt = D._build_visual_prompt(PERCEIVED_VOCABULARY)
    assert "WRONG:" not in prompt
    assert "RIGHT:" not in prompt
    assert "Example" not in prompt and "example" not in prompt


def test_the_prompts_own_answer_format_parses():
    """The one shape the prompt names must round-trip through the parser."""
    prompt = D._build_visual_prompt(PERCEIVED_VOCABULARY)
    assert "comma-separated list of tags" in prompt
    assert D._parse_visual_response("sunny, peaceful", VOCAB) == ["sunny", "peaceful"]
    assert D._is_explicit_empty_answer("none") is True


def test_json_round_trip_of_an_empty_answer_is_still_a_list():
    assert json.loads(json.dumps([])) == []


def test_misty_reads_as_foggy():
    # gemma-4 answered `peaceful, misty, muted` on a fog shot (Unsplash
    # 26-lAP0XprM) and `misty` was dropped as off-vocabulary.
    assert D._parse_visual_response("peaceful, misty, muted", VOCAB) == [
        "peaceful", "foggy", "muted"]
    assert D._parse_visual_response("foggy, misty", VOCAB) == ["foggy"]


def test_a_synonym_never_adds_a_tag_outside_the_vocabulary():
    assert D._parse_visual_response("misty", {"sunny"}) == []

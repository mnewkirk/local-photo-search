"""Bad LLM output must never be stored as a success.

Every pass used to turn a bad generation into a stored result: 4,350 live
descriptions end mid-sentence (stopped at the token limit), keyword rows hold
refusals and unsplit lists, and llama3.2's category answers recite the
vocabulary. These pin the guards: a truncation is detected on both backends
and retried once at double the budget, an unusable answer raises
`UnusableAnswer`, and the worker reports it as a failure row (one attempt,
logged) instead of deferring forever or writing it.
"""
import json
import urllib.request
from types import SimpleNamespace

import pytest

from photosearch import describe as d


# ---------------------------------------------------------------------------
# Backends: the stop reason is read, and a cut-off answer is never returned
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload): self.payload = payload
    def read(self): return json.dumps(self.payload).encode()


def _openai(monkeypatch, answers):
    """Stub urlopen with (content, finish_reason) answers; record the bodies."""
    bodies = []
    q = list(answers)

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data))
        content, finish = q.pop(0)
        return _Resp({"choices": [{"message": {"content": content},
                                   "finish_reason": finish}], "usage": {}})
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    return bodies


def _chat(**kw):
    return d._ollama_chat_with_retry(
        model="m", messages=[{"role": "user", "content": "hi"}], timeout=5, **kw)


def test_openai_truncation_is_retried_at_double_the_budget(monkeypatch):
    bodies = _openai(monkeypatch, [("A dog on a", "length"),
                                   ("A dog on a beach.", "stop")])
    assert _chat(role="describe") == "A dog on a beach."
    assert [b["max_tokens"] for b in bodies] == [512, 1024]


def test_openai_truncated_twice_raises_with_the_partial_text(monkeypatch):
    _openai(monkeypatch, [("A dog", "length"), ("A dog on", "length")])
    with pytest.raises(d.TruncatedOutput) as ei:
        _chat(role="describe")
    assert ei.value.text == "A dog on"
    assert isinstance(ei.value, d.UnusableAnswer)


def test_an_empty_answer_cut_off_mid_reasoning_is_a_truncation(monkeypatch):
    """A thinking model with reasoning left on spends the budget thinking and
    returns '' — the 2026-10-03 batch. That is not "no answer"."""
    _openai(monkeypatch, [("", "length"), ("", "length")])
    with pytest.raises(d.TruncatedOutput):
        _chat(role="text")


@pytest.mark.parametrize("role,budget", [("text", 256), ("visual", 128),
                                         ("describe", 512), ("verify", 512),
                                         ("aesthetics", 768), (None, 768)])
def test_each_role_sends_its_own_token_budget(monkeypatch, role, budget):
    bodies = _openai(monkeypatch, [("ok", "stop")])
    _chat(role=role)
    assert bodies[0]["max_tokens"] == budget


def _ollama(monkeypatch, answers):
    calls = []
    q = list(answers)

    def chat(**kw):
        calls.append(kw)
        content, reason = q.pop(0)
        return SimpleNamespace(message=SimpleNamespace(content=content),
                               done_reason=reason)
    monkeypatch.delenv("PHOTOSEARCH_TEXT_LLM_URL", raising=False)
    monkeypatch.setattr(d, "ollama", SimpleNamespace(chat=chat))
    return calls


def test_ollama_truncation_doubles_num_predict(monkeypatch):
    calls = _ollama(monkeypatch, [("A dog on a", "length"),
                                  ("A dog on a beach.", "stop")])
    assert _chat(options={"num_predict": 150}) == "A dog on a beach."
    assert [c["options"]["num_predict"] for c in calls] == [150, 300]


def test_ollama_truncated_twice_raises(monkeypatch):
    _ollama(monkeypatch, [("A dog", "length"), ("A dog on", "length")])
    with pytest.raises(d.TruncatedOutput):
        _chat(options={"num_predict": 150})


def test_ollama_truncation_without_a_cap_to_raise_fails_at_once(monkeypatch):
    calls = _ollama(monkeypatch, [("A dog", "length")])
    with pytest.raises(d.TruncatedOutput):
        _chat(options={"temperature": 0})
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# describe_photo: no cut-off, looping or mid-sentence text is ever returned
# ---------------------------------------------------------------------------

@pytest.fixture
def img(tmp_path, monkeypatch):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"x")
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    monkeypatch.setattr(d, "_encode_image_for_ollama", lambda path: "b64")
    return str(p)


def _script(monkeypatch, answers):
    q = list(answers)
    asked = []

    def chat(**kw):
        asked.append(kw["model"])
        nxt = q.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
    monkeypatch.setattr(d, "_ollama_chat_with_retry", chat)
    return asked


GOOD = "Two children play soccer on a grassy field on a sunny day."
CUT = "Two children play soccer on a grassy field while a coach"


def test_a_mid_sentence_description_is_not_returned(monkeypatch, img):
    _script(monkeypatch, [CUT, CUT, CUT, CUT])
    assert d.describe_photo(img, model="llama3.2-vision") is None


def test_the_worker_gets_an_unusable_answer_raised(monkeypatch, img):
    _script(monkeypatch, [CUT, CUT, CUT, CUT])
    with pytest.raises(d.UnusableAnswer, match="ends mid-sentence"):
        d.describe_photo(img, model="llama3.2-vision", raise_unusable=True)


def test_a_truncation_recovers_on_retry(monkeypatch, img):
    _script(monkeypatch, [d.TruncatedOutput("cut", "Two"), GOOD])
    assert d.describe_photo(img, model="llama3.2-vision") == GOOD


def test_a_degenerate_answer_is_no_longer_returned_when_all_retries_fail(
        monkeypatch, img):
    loop = " ".join(["a dog on the beach"] * 80) + "."
    _script(monkeypatch, [loop, loop, loop, loop])
    assert d.describe_photo(img, model="llama3.2-vision") is None


def test_an_empty_answer_stays_a_plain_none_even_for_the_worker(monkeypatch, img):
    _script(monkeypatch, [None])
    assert d.describe_photo(img, model="llava", raise_unusable=True) is None


def test_a_transport_error_is_still_a_plain_none(monkeypatch, img):
    _script(monkeypatch, [ConnectionError("refused")])
    assert d.describe_photo(img, model="llava", raise_unusable=True) is None


def test_ends_mid_sentence():
    assert d.ends_mid_sentence(CUT)
    assert not d.ends_mid_sentence(GOOD)
    assert not d.ends_mid_sentence('She held a sign reading "GO TEAM!"')


# ---------------------------------------------------------------------------
# Text passes
# ---------------------------------------------------------------------------

@pytest.fixture
def vocab(monkeypatch):
    words = [f"term{i:03d}" for i in range(100)] + ["dog", "beach"]
    monkeypatch.setattr("photosearch.vocab_content.CONTENT_VOCABULARY", words)
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    return words


def _text(monkeypatch, answers):
    q = list(answers)
    temps = []

    def chat(**kw):
        temps.append(kw["options"]["temperature"])
        nxt = q.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt
    monkeypatch.setattr(d, "_ollama_chat_with_retry", chat)
    return temps


# --- category grounding (docs/plans/category-grounding.md) -----------------
# These use the REAL vocabulary: the recitation verdict and the contradiction
# groups are defined over real terms, so the fake `term000` vocabulary above
# can no longer exercise them.

SOCCER = "Two boys play soccer on a grassy field."


@pytest.fixture
def real_vocab(monkeypatch):
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    from photosearch import category_grounding as cg
    from photosearch.vocab_content import CONTENT_VOCABULARY
    grouped = {t for g in cg.GROUPS for side in g.sides for t in side.terms}
    return cg, list(CONTENT_VOCABULARY), grouped


def _llama_recital(cg, vocab, grouped, description, n=30):
    """llama3.2:3b's shape: concrete nouns the text never mentions, in
    vocabulary order (41317: "beehive-like stone structure" stored
    `bleacher, bridge, chain link fence, cityscape, grassy field`)."""
    terms = [t for t in vocab if t in cg.OBJECT_TERMS and t not in grouped
             and not cg.is_supported(description, t)]
    assert len(terms) >= n
    return terms[:n]


def test_a_vocabulary_recitation_is_rejected(monkeypatch, real_vocab):
    cg, vocab, grouped = real_vocab
    recital = ", ".join(_llama_recital(cg, vocab, grouped, SOCCER))
    temps = _text(monkeypatch, [recital, recital])
    with pytest.raises(d.UnusableAnswer, match="recitation"):
        d.extract_categories_from_description(SOCCER)
    assert temps == [0, 0.4]


def test_a_70_tag_grounded_list_is_kept(monkeypatch, real_vocab):
    """The old 60 cap retired 251357 for an honest, non-alphabetical 68-tag
    gemma answer. Length alone is no longer a verdict."""
    import random
    cg, vocab, grouped = real_vocab
    terms = [t for t in vocab if t not in grouped][:70]
    random.Random(7).shuffle(terms)
    assert cg.order_share(terms) < cg.RECITATION_MIN_ORDER
    temps = _text(monkeypatch, [", ".join(terms)])
    assert d.extract_categories_from_description(SOCCER) == terms
    assert temps == [0]


def test_a_description_decided_contradiction_is_repaired_without_a_retry(
        monkeypatch, real_vocab):
    """The description is the evidence; a resample adds nothing (251357's 0.4
    retry was worse, 84 tags). A repairable answer costs ONE model call."""
    temps = _text(monkeypatch, ["soccer, basketball, basketball hoop, grass, boy"])
    report = {}
    assert d.extract_categories_from_description(SOCCER, report=report) == \
        ["soccer", "grass", "boy"]
    assert temps == [0]
    assert report["raw"] == 5
    assert set(report["dropped"]) == {"basketball", "basketball hoop"}


DOG = "A dog sits on a lawn."      # cues neither a sunny nor an overcast sky


def test_both_sides_emitted_and_uncued_retries_once_then_drops_both(
        monkeypatch, real_vocab):
    temps = _text(monkeypatch, ["dog, clear sky, cloudy sky"] * 2)
    assert d.extract_categories_from_description(DOG) == ["dog"]
    assert temps == [0, 0.4]


def test_an_undecided_answer_takes_a_clean_retry(monkeypatch, real_vocab):
    temps = _text(monkeypatch, ["dog, clear sky, cloudy sky", "dog, clear sky"])
    assert d.extract_categories_from_description(DOG) == ["dog", "clear sky"]
    assert temps == [0, 0.4]


def test_a_bad_retry_never_replaces_an_undecided_first_answer(monkeypatch, real_vocab):
    """The soft retry can only improve the answer: a garbage or stalled retry
    keeps the (repaired) first answer instead of failing or deferring it."""
    _text(monkeypatch, ["dog, clear sky, cloudy sky", "I cannot help."])
    assert d.extract_categories_from_description(DOG) == ["dog"]
    _text(monkeypatch, ["dog, clear sky, cloudy sky", TimeoutError("stalled")])
    assert d.extract_categories_from_description(DOG) == ["dog"]


def test_an_answer_grounding_empties_twice_is_unusable(monkeypatch, real_vocab):
    temps = _text(monkeypatch, ["basketball, basketball hoop"] * 2)
    with pytest.raises(d.UnusableAnswer, match="no grounded categories"):
        d.extract_categories_from_description(SOCCER)
    assert temps == [0, 0.4]


def test_more_than_120_categories_is_rejected(monkeypatch, real_vocab):
    import random
    cg, vocab, grouped = real_vocab
    terms = [t for t in vocab if t not in grouped][:cg.MAX_PLAUSIBLE + 1]
    random.Random(3).shuffle(terms)
    _text(monkeypatch, [", ".join(terms)] * 2)
    with pytest.raises(d.UnusableAnswer, match="runaway"):
        d.extract_categories_from_description(SOCCER)
    assert d._CATEGORY_MAX_PLAUSIBLE == cg.MAX_PLAUSIBLE == 120


def test_the_undecided_marker_matches_the_rule_module():
    """describe's soft retry keys on group_drops' reason text."""
    from photosearch import category_grounding as cg
    dropped = cg.ground(DOG, ["dog", "clear sky", "cloudy sky"]).dropped
    assert dropped and all(d._CATEGORY_UNDECIDED in r for r in dropped.values())
    decided = cg.ground(SOCCER, ["soccer", "basketball"]).dropped
    assert decided and not any(d._CATEGORY_UNDECIDED in r for r in decided.values())


def test_the_worker_logs_kept_and_dropped(monkeypatch, real_vocab, capsys):
    from photosearch import worker as W
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    _text(monkeypatch, ["soccer, basketball, grass"])
    out = W._process_category_content([{"id": 7, "filename": "a.jpg",
                                        "description": SOCCER}])
    assert out == [{"photo_id": 7, "categories": ["soccer", "grass"],
                    "raw_categories": ["soccer", "basketball", "grass"]}]
    line = capsys.readouterr().out
    assert "kept 2, dropped 1 (sport: soccer cued, basketball not: basketball)" in line


def test_a_long_but_plausible_category_list_is_kept(monkeypatch, vocab):
    """gemma-4 lists ~25 (max 52) — every one supported. Not a recitation."""
    answer = ", ".join(vocab[:40])
    _text(monkeypatch, [answer])
    assert len(d.extract_categories_from_description("a dog")) == 40


def test_a_bad_first_answer_is_replaced_by_a_good_retry(monkeypatch, vocab):
    _text(monkeypatch, ["I cannot categorise this.", "dog, beach"])
    assert d.extract_categories_from_description("a dog at the beach") == ["dog", "beach"]


def test_a_category_truncation_is_unusable_not_deferred(monkeypatch, vocab):
    _text(monkeypatch, [d.TruncatedOutput("cut"), d.TruncatedOutput("cut")])
    with pytest.raises(d.UnusableAnswer, match="cut off"):
        d.extract_categories_from_description("a dog")


def test_a_category_stall_still_defers(monkeypatch, vocab):
    _text(monkeypatch, [TimeoutError("stalled")])
    assert d.extract_categories_from_description("a dog") is None


DESC = "A golden retriever runs along Stinson Beach at sunset."


@pytest.mark.parametrize("answer,why", [
    ("i couldn't find any text in this description", "refusal"),
    ("golden retriever beach sunset dog sand ocean waves", "one keyword"),
    ("mountain, snow, skiing, chairlift, pine", "not in the description"),
    (", ".join(f"word{i}" for i in range(31)), "implausibly many"),
])
def test_unstorable_keywords_are_rejected(monkeypatch, answer, why):
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    _text(monkeypatch, [answer, answer])
    with pytest.raises(d.UnusableAnswer, match=why):
        d.extract_keywords_from_description(DESC)


def test_good_keywords_pass(monkeypatch):
    monkeypatch.setattr(d, "HAS_OLLAMA", True)
    _text(monkeypatch, ["golden retriever, stinson beach, sunset, dog"])
    assert d.extract_keywords_from_description(DESC) == [
        "golden retriever", "stinson beach", "sunset", "dog"]


# ---------------------------------------------------------------------------
# category-visual + verify
# ---------------------------------------------------------------------------

def test_visual_no_answer_raises_only_when_asked(monkeypatch, img):
    _script(monkeypatch, [None, None])
    assert d.tag_visual_photo(img) is None
    _script(monkeypatch, [None])
    with pytest.raises(d.UnusableAnswer):
        d.tag_visual_photo(img, raise_unusable=True)


def test_visual_truncation_is_a_no_answer(monkeypatch, img):
    _script(monkeypatch, [d.TruncatedOutput("cut")])
    with pytest.raises(d.UnusableAnswer, match="cut off"):
        d.tag_visual_photo(img, raise_unusable=True)


def test_a_cut_off_verdict_is_not_all_correct(monkeypatch, tmp_path):
    """llm_verify_description returns [] on error — which reads as ALL CORRECT
    and would stamp the photo verified. A truncation must propagate."""
    from photosearch import verify
    _script(monkeypatch, [d.TruncatedOutput("cut", "WRONG: dog")])
    with pytest.raises(d.UnusableAnswer):
        verify.llm_verify_description(str(tmp_path / "a.jpg"), "A dog.", [])


# ---------------------------------------------------------------------------
# Worker: an unusable answer is a failure row, never a deferral or a write
# ---------------------------------------------------------------------------

def _unusable(*a, **kw):
    raise d.UnusableAnswer("keywords: refusal stored as keywords")


@pytest.mark.parametrize("fn,target", [
    ("_process_category_content", "extract_categories_from_description"),
    ("_process_keywords", "extract_keywords_from_description"),
])
def test_text_pass_unusable_is_a_failure_row(monkeypatch, fn, target):
    from photosearch import worker as W
    monkeypatch.setattr(f"photosearch.describe.{target}", _unusable)
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    out = getattr(W, fn)([{"id": 7, "filename": "a.jpg", "description": DESC}])
    assert out == [{"photo_id": 7, "error": "keywords: refusal stored as keywords"}]
    assert W._submit_kwargs("keywords_results", out) == {
        "keywords_results": [], "failures": out}


def test_describe_unusable_is_a_failure_row(monkeypatch):
    from photosearch import worker as W
    monkeypatch.setattr("photosearch.describe.describe_photo", _unusable)
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    out = W._process_describe([({"id": 7, "filename": "a.jpg"}, "/x")])
    assert out[0]["photo_id"] == 7 and "error" in out[0]


def test_visual_unusable_is_a_failure_row(monkeypatch):
    from photosearch import worker as W
    monkeypatch.setattr("photosearch.describe.tag_visual_photo", _unusable)
    monkeypatch.setattr("photosearch.describe.check_available", lambda m: None)
    out = W._process_category_visual([({"id": 7, "filename": "a.jpg"}, "/x")])
    assert out[0]["photo_id"] == 7 and "error" in out[0]


# ---------------------------------------------------------------------------
# A timeout AFTER a proven truncation is a truncation, not a stall
# ---------------------------------------------------------------------------

def test_a_timed_out_expanded_retry_is_a_truncation(monkeypatch):
    """IMAG2074, 2026-10-06: gemma loops `railing, railing, …` at temperature
    0. The 256-token call is cut off, the 512-token retry overruns the 10 s
    cap, and the timeout used to read as a transport stall — deferred with no
    attempt spent, re-claimed ~90 times in 1.5 h."""
    monkeypatch.setattr(d, "_RETRY_DELAY", 0)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp({"choices": [{"message": {"content": "railing, railing"},
                                       "finish_reason": "length"}], "usage": {}})
        raise TimeoutError("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    with pytest.raises(d.TruncatedOutput) as ei:
        _chat(role="text")
    assert ei.value.text == "railing, railing"


def test_a_connection_error_on_the_expanded_retry_stays_transport(monkeypatch):
    monkeypatch.setattr(d, "_RETRY_DELAY", 0)
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Resp({"choices": [{"message": {"content": "a, b"},
                                       "finish_reason": "length"}], "usage": {}})
        raise ConnectionRefusedError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("PHOTOSEARCH_TEXT_LLM_URL", "http://x/v1")
    with pytest.raises(ConnectionRefusedError):
        _chat(role="text")


def test_the_looping_photo_is_spent_not_deferred(monkeypatch, vocab):
    """End to end: both answers truncated → UnusableAnswer → the worker's
    failure row, which spends an attempt (retired after MAX_PROCESS_ATTEMPTS)."""
    _text(monkeypatch, [d.TruncatedOutput("overran"), d.TruncatedOutput("overran")])
    with pytest.raises(d.UnusableAnswer):
        d.extract_categories_from_description("a performer on a stage")

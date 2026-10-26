"""One test per grounding rule: deleting any single cue, term, scrub, guard or
recitation condition from photosearch/category_grounding.py fails a test here.

Written after a mutation review (2026-10-25): of 152 single-rule deletions,
tests/test_category_grounding.py caught 44. Among the survivors that change
real output: the negation scrub (its test sentence decided the same way
without it), the `indoors` cue (53 gemma rows) and the `youth soccer game`
term (34 rows, 2 labelled-correct holdout drops).

The fixture is a table of one example phrase per cue. Each phrase, on its
own, must match exactly that one cue of its side and no cue of a rival side,
so removing the cue leaves the side uncued and the verdict changes. A new
cue without a phrase fails `test_every_cue_has_one_example`.
"""

import pytest

from photosearch import category_grounding as cg
from photosearch.vocab_content import CONTENT_VOCABULARY

# (group, side) -> one phrase per cue, in the order of Side.cues.
CUE_EXAMPLES = {
    ("sport", "soccer"): ["a soccer match", "a football match", "a futsal match"],
    ("sport", "basketball"): ["a basketball match", "shooting at the hoop",
                              "a glass backboard"],
    ("sport", "baseball"): ["a baseball diamond", "a softball pitch", "a t-ball game",
                            "a boy swings a bat", "the batter waits", "a leather mitt",
                            "the dugout"],
    ("sport", "volleyball"): ["a volleyball match"],
    ("setting", "indoor"): ["indoors", "inside", "the interior", "a bedroom", "a gym",
                            "a kitchen", "the ceiling", "a hallway", "a classroom"],
    ("setting", "outdoor"): ["outdoors", "outside", "the sky", "grey skies", "sunlight",
                             "a sunny afternoon", "daylight", "a field", "a park",
                             "a tree", "grass", "a lawn", "turf", "a beach", "a trail",
                             "a backyard", "a garden", "a forest"],
    ("light", "artificial"): ["the lights", "a lamp", "lamplight", "a lit candle",
                              "illuminated", "fluorescent tubes", "a floodlit pitch",
                              "a spotlight", "a neon sign", "a chandelier", "bare bulbs",
                              "a camera flash", "ambient lighting", "at night",
                              "in the evening"],
    ("light", "natural"): ["daylight", "sunlight", "sunlit", "a sunny afternoon",
                           "sunshine", "midday", "a blue sky", "a clear sky",
                           "outdoors", "outside"],
    ("weather", "sunny"): ["a sunny afternoon", "sunlight", "sunlit", "sunshine",
                           "a clear sky", "a clear blue sky", "a blue sky", "cloudless",
                           "the sky is clear", "bright sun"],
    ("weather", "overcast"): ["overcast", "a cloudy afternoon", "a grey sky", "fog",
                              "misty", "rain", "drizzle", "gloomy", "stormy"],
}
# Every term each side drops, written out here rather than read from the
# module: a parametrize built from cg.GROUPS would lose a deleted term along
# with its test. `test_term_table_matches_the_module` keeps the two in step.
TERMS = {
    ("sport", "soccer"): ("soccer", "soccer ball", "soccer game", "soccer goal",
                          "soccer jersey", "soccer player", "indoor soccer",
                          "soccer indoor", "youth soccer game"),
    ("sport", "basketball"): ("basketball", "basketball court", "basketball hoop",
                              "backboard"),
    ("sport", "baseball"): ("baseball", "baseball bat", "baseball game", "baseball glove",
                            "baseball uniform"),
    ("sport", "volleyball"): ("volleyball",),
    ("setting", "indoor"): ("indoor", "indoor soccer", "soccer indoor", "gym", "gymnasium",
                            "kitchen", "dining area", "dining table", "living room",
                            "restaurant", "floor indoor"),
    ("setting", "outdoor"): ("outdoor",),
    ("light", "artificial"): ("artificial lighting",),
    ("light", "natural"): ("natural light", "sunny day", "clear day", "sun"),
    ("weather", "sunny"): ("sunny day", "clear sky", "clear blue sky", "clear day",
                           "blue sky", "sun"),
    ("weather", "overcast"): ("cloudy day", "cloudy sky", "cool cloudy day", "fog"),
}
PARTIAL_EXAMPLES = {
    ("weather", "overcast"): ["partly cloudy", "scattered clouds", "some clouds"],
}

# Cues another cue of the same side already covers: deleting one alone
# changes nothing. Listed so the overlap is a decision, not an accident.
REDUNDANT = {
    ("setting", "indoor", r"\bclassroom"): r"\b(?:room|rooms|bedroom",
    ("weather", "sunny", r"\bclear blue sk(?:y|ies)\b"): r"\bblue sk(?:y|ies)\b",
}

# Phrases a cue's guard (lookbehind / lookahead / word boundary) must NOT
# count as that side. Each is a past false cue.
NON_CUES = {
    ("sport", "basketball"): ["a hula hoop", "a hula-hoop", "hoop earrings"],
    ("sport", "baseball"): ["a baseball cap", "a baseball-style hat", "a baseball hat"],
    ("setting", "indoor"): ["inside the car", "inside a parked car", "the car interior",
                            "the car's interior", "the vehicle interior"],
    ("setting", "outdoor"): ["shallow depth of field", "a christmas tree",
                             "tree-shaped cookies"],
    ("light", "artificial"): ["a neon green shirt", "a flash of colour",
                              "natural lighting", "a light blue shirt"],
    ("weather", "sunny"): ["partly sunny", "partly-sunny"],
    ("weather", "overcast"): ["partly cloudy", "partly-cloudy"],
}


def _group(name):
    return next(g for g in cg.GROUPS if g.name == name)


def _side(group, name):
    return next(s for s in _group(group).sides if s.name == name)


def _rival_term(group, side):
    """A term of another side of the group that is not shared with `side`,
    not exempt, and not only weakly droppable."""
    g = _group(group)
    exempt = {t for terms, _ in g.exempt_when for t in terms}
    for r in g.sides:
        if r.name == side.name:
            continue
        for t in r.terms:
            if t not in side.terms and t not in exempt:
                return r, t
    raise AssertionError(f"no rival term for {group}/{side.name}")


def _cases():
    for g in cg.GROUPS:
        for s in g.sides:
            for cue, phrase in zip(s.cues, CUE_EXAMPLES[(g.name, s.name)]):
                yield pytest.param(g.name, s.name, cue, phrase,
                                   id=f"{g.name}/{s.name}/{phrase}")


# --------------------------------------------------------------- cues --

def test_every_cue_has_one_example():
    for g in cg.GROUPS:
        for s in g.sides:
            assert len(CUE_EXAMPLES[(g.name, s.name)]) == len(s.cues), (g.name, s.name)
            assert len(PARTIAL_EXAMPLES.get((g.name, s.name), [])) == len(s.partial_cues)


@pytest.mark.parametrize("group,side,cue,phrase", list(_cases()))
def test_each_cue_alone_decides_its_group(group, side, cue, phrase):
    s = _side(group, side)
    text = cg.scrub(phrase)
    own = [c for c in s.cues if cg._rx(c).search(text)]
    assert cue in own, f"{phrase!r} does not match {cue!r}"
    if (group, side, cue) in REDUNDANT:
        assert any(c.startswith(REDUNDANT[(group, side, cue)]) for c in own if c != cue)
    else:
        assert own == [cue], f"{phrase!r} also matches {own}: a deletion would hide"
    for r in _group(group).sides:
        if r is not s:
            assert not cg._cued(text, r), f"{phrase!r} cues the rival side {r.name}"
    # The verdict: this side's term stays, a rival side's term drops.
    rival, rt = _rival_term(group, s)
    term = next(t for t in s.terms if t not in rival.terms)
    g = cg.ground(phrase, [term, rt])
    assert term in g.kept and rt in g.dropped, (phrase, g)
    assert g.dropped[rt].startswith(f"{group}: {side}")


def _first_phrase(group, side):
    return CUE_EXAMPLES[(group, side)][0]


def test_term_table_matches_the_module():
    """A new term needs a row in TERMS (and so a drop test)."""
    assert {(g.name, s.name): s.terms for g in cg.GROUPS for s in g.sides} == TERMS


@pytest.mark.parametrize("group,side,term", [
    pytest.param(g, s, t, id=f"{g}/{s}/{t}") for (g, s), ts in TERMS.items() for t in ts])
def test_each_term_drops_against_a_rival_cue(group, side, term):
    """Deleting a term from its side (e.g. `youth soccer game`) means a
    contradicted mention of it is no longer dropped."""
    rival = next(r for (g, r) in TERMS if g == group and r != side
                 and term not in TERMS[(g, r)])
    phrase = _first_phrase(group, rival)
    out = cg.ground(phrase, [term])
    assert term in out.dropped, (phrase, term, out)


@pytest.mark.parametrize("group,side,phrase", [
    pytest.param(g, s, p, id=f"{g}/{s}/{p}")
    for (g, s), ps in NON_CUES.items() for p in ps])
def test_guards_keep_false_cues_out(group, side, phrase):
    assert not cg._cued(cg.scrub(phrase), _side(group, side)), phrase


def test_none_cued_drops_both_sides():
    g = cg.ground("Two children sit together.", ["indoor", "outdoor"])
    assert set(g.dropped) == {"indoor", "outdoor"}
    assert all("none cued" in why for why in g.dropped.values())


def test_both_cued_keeps_both():
    assert cg.ground("indoors by a window, then outside on the lawn",
                     ["indoor", "outdoor"]).dropped == {}


# --------------------------------------------------------- weak cues --

@pytest.mark.parametrize("phrase", PARTIAL_EXAMPLES[("weather", "overcast")])
def test_partly_cloudy_drops_only_the_clear_sky_terms(phrase):
    cats = ["cloudy sky", "clear sky", "clear blue sky", "clear day", "blue sky",
            "sunny day"]
    g = cg.ground(phrase, cats)
    assert set(g.dropped) == set(cg.CLEAR_SKY_TERMS)
    assert {"cloudy sky", "blue sky", "sunny day"} <= set(g.kept)


def test_partly_cloudy_and_sunny_cues_both_sides():
    assert cg.ground("partly cloudy but sunny", ["clear sky", "cloudy sky"]).dropped == {}


# ------------------------------------------------------------- scrubs --

NEGATORS = ("not", "no", "without", "nor", "neither")
NEGATED = ("outdoors", "outside", "indoors", "inside", "sky", "daylight", "sunlight",
           "natural light")


@pytest.mark.parametrize("neg", NEGATORS)
@pytest.mark.parametrize("word", NEGATED)
def test_negation_scrub_blanks_each_negated_cue(neg, word):
    for gap in ("", "any ", "any real "):
        text = f"a {neg} {gap}{word} scene"
        assert word not in cg.scrub(text), text


def test_negation_scrub_changes_the_verdict():
    """The scrub's own job, decided by it alone: without it, "not outdoors"
    cues the outdoor side and both terms are kept."""
    g = cg.ground("a party room, not outdoors", ["indoor", "outdoor"])
    assert g.dropped == {"outdoor": "setting: indoor cued, outdoor not"}


def test_negation_reaches_at_most_two_words():
    text = "no people are visible in this outdoor setting"
    assert "outdoor" in cg.scrub(text)
    assert cg.ground(text, ["outdoor"]).dropped == {}


def test_window_scrub_changes_the_verdict():
    g = cg.ground("a living room with large windows showing greenery outside",
                  ["indoor", "outdoor", "artificial lighting", "natural light"])
    assert "outdoor" in g.dropped and "indoor" in g.kept


# --------------------------------------------- group-level exceptions --

def test_collage_skips_the_setting_group():
    assert cg.ground("indoors", ["collage", "indoor", "outdoor"]).kept == [
        "collage", "indoor", "outdoor"]


@pytest.mark.parametrize("venue", cg.INDOOR_VENUE_TERMS)
def test_indoor_venue_terms_survive_only_on_indoor_text(venue):
    assert cg.ground("indoor soccer in a school gym", ["soccer", venue]).dropped == {}
    assert venue in cg.ground("soccer on an outdoor field", ["soccer", venue]).dropped
    # `basketball` itself (the sport) gets no exception.
    assert "basketball" in cg.ground("indoor soccer in a school gym",
                                     ["soccer", "basketball"]).dropped


# --------------------------------------------------------- recitation --

DESC = "A portrait of a person smiling."
_OBJ_SORTED = [t for t in CONTENT_VOCABULARY if t in cg.OBJECT_TERMS]


def _unsupported(n, desc=DESC):
    out = [t for t in _OBJ_SORTED if not cg.is_supported(cg.scrub(desc), t)]
    assert len(out) >= n
    return out[:n]


def test_recitation_needs_the_unsupported_minimum():
    n = cg.RECITATION_MIN_UNSUPPORTED
    assert cg.is_recitation(DESC, _unsupported(n))
    assert not cg.is_recitation(DESC, _unsupported(n - 1))


def test_recitation_needs_vocabulary_order():
    cats = _unsupported(cg.RECITATION_MIN_UNSUPPORTED + 5)
    assert cg.order_share(cats) >= cg.RECITATION_MIN_ORDER
    assert not cg.is_recitation(DESC, list(reversed(cats)))


def test_recitation_needs_the_unsupported_share():
    n = cg.RECITATION_MIN_UNSUPPORTED
    unsup = _unsupported(n)
    supported = [t for t in _OBJ_SORTED if t not in unsup][: n + 1]
    desc = DESC + " Nearby: " + ", ".join(supported) + "."
    unsup = [t for t in unsup if not cg.is_supported(cg.scrub(desc), t)]
    assert len(unsup) == n
    order = {t: i for i, t in enumerate(CONTENT_VOCABULARY)}

    def answer(k):
        return sorted(unsup + supported[:k], key=order.get)
    at_share = int(n / cg.RECITATION_MIN_UNSUPPORTED_SHARE) - n  # share == the minimum
    assert cg.is_recitation(desc, answer(at_share))
    assert not cg.is_recitation(desc, answer(at_share + 1))


def test_a_reading_is_not_a_recitation():
    assert not cg.is_recitation("A dog runs on the beach by the ocean.",
                                ["beach", "dog", "ocean"])

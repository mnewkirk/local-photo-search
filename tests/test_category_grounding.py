"""Tests for photosearch.category_grounding (docs/plans/category-grounding.md)."""

import json
import random
import re
from pathlib import Path

import pytest

from photosearch import category_grounding as cg
from photosearch.vocab_content import CONTENT_VOCABULARY

FIXTURES = json.loads(
    (Path(__file__).resolve().parent.parent / "evals" / "category-grounding"
     / "fixtures.json").read_text())
P57 = FIXTURES["251357"]
DESC57 = P57["description"]
T0 = P57["temp0"]["categories"]
T04 = P57["temp04"]["categories"]
BASKETBALL = {"basketball", "basketball court", "basketball hoop"}


def kept(desc, cats):
    return cg.ground(desc, cats).kept


# 1 -------------------------------------------------------------------------

def test_251357_contradictions_dropped_main_subject_kept():
    g0 = cg.ground(DESC57, T0)
    assert BASKETBALL <= set(g0.dropped)
    assert "artificial lighting" in g0.dropped
    for t in ("soccer", "soccer ball", "soccer player", "turf", "tree",
              "chain link fence", "car", "boy", "spectator"):
        assert t in g0.kept, t

    g4 = cg.ground(DESC57, T04)
    for t in BASKETBALL | {"artificial lighting", "indoor", "gym", "gymnasium",
                           "kitchen", "indoor soccer", "soccer indoor"}:
        assert t in g4.dropped, t
    for t in ("soccer", "turf", "car", "boy", "spectator", "outdoor"):
        assert t in g4.kept, t
    assert g0.dropped["basketball"] == "sport: soccer cued, basketball not"
    assert g4.dropped["indoor"] == "setting: outdoor cued, indoor not"


# 2 -------------------------------------------------------------------------

def test_251356_artificial_lighting_alone_against_outdoors():
    fx = FIXTURES["251356"]
    g = cg.ground(fx["description"], fx["categories"])
    assert g.dropped == {"artificial lighting": "light: natural cued, artificial not"}
    assert "house" in g.kept
    assert not g.recitation


# 3 -------------------------------------------------------------------------

def test_exempt_terms_are_never_dropped():
    exempt = ("vehicle", "child", "people", "sport", "human activity", "team")
    for g in cg.GROUPS:
        for s in g.sides:
            assert not set(exempt) & set(s.terms)
    for desc in (DESC57, "A dark room.", "Snowy mountains at night.",
                 "Inside a gym with basketball hoops.", "x"):
        out = cg.ground(desc, list(exempt) + ["basketball", "outdoor"])
        assert set(exempt) <= set(out.kept)


# 4 -------------------------------------------------------------------------

def test_light_blue_and_daylight_are_not_artificial_cues():
    # If "light blue" were an artificial cue, natural light would be dropped.
    assert kept("A girl in a light blue dress and a light-colored hat poses.",
                ["natural light"]) == ["natural light"]
    # If "daylight"/"sunlight" were artificial cues, both sides would be kept.
    assert kept("Bright daylight and sunlight fall on the field.",
                ["artificial lighting", "natural light"]) == ["natural light"]
    # Real artificial cues still fire.
    assert kept("A kitchen lit by pendant lighting at night.",
                ["natural light"]) == []
    # "neon green jersey" is a colour, not a light.
    assert kept("A boy in a neon green jersey.", ["natural light"]) == ["natural light"]
    assert kept("A neon sign glows over the bar.", ["natural light"]) == []


def test_baseball_cap_is_not_a_baseball_cue():
    assert kept("A boy in a baseball cap plays soccer on a field.",
                ["soccer", "baseball"]) == ["soccer"]
    # neither sport cued, both emitted -> both dropped (cap protects nothing)
    assert kept("A man in a baseball cap stands by a fence.",
                ["baseball", "basketball"]) == []
    assert kept("Kids swing a bat on a baseball diamond.",
                ["baseball", "soccer"]) == ["baseball"]


def test_other_cue_traps():
    # "parked cars" is not a park; "depth of field" is not a field
    assert kept("A portrait with shallow depth of field and parked cars blurred.",
                ["indoor"]) == ["indoor"]
    # a Christmas tree / tree-shaped decoration is not an outdoor tree
    assert kept("A family opens presents by the Christmas tree.",
                ["living room"]) == ["living room"]
    assert kept("A table with a small tree-shaped decoration.",
                ["living room"]) == ["living room"]
    # hula hoops are not basketball; a bathroom is not a bat
    assert kept("A girl spins a hula hoop near the bathroom door.",
                ["soccer", "basketball"]) == []
    # negation and a view through a window are not outdoor cues
    assert kept("An indoor party space, not outdoors, with string lights.",
                ["artificial lighting"]) == ["artificial lighting"]
    assert kept("Guests sit at tables; large windows show greenery outside.",
                ["artificial lighting"]) == ["artificial lighting"]
    # ... but a positive "no people are visible in this outdoor scene" is
    assert kept("No people are visible in this outdoor scene.",
                ["artificial lighting"]) == []


# 5 -------------------------------------------------------------------------

def test_setting_group():
    assert kept("Children play inside a gym.", ["indoor", "outdoor"]) == ["indoor"]
    assert kept("Two people talk.", ["indoor", "outdoor"]) == []
    assert kept("A view from inside a room looking at trees.",
                ["indoor", "outdoor"]) == ["indoor", "outdoor"]
    assert kept("A collage of family photos.",
                ["collage", "indoor", "outdoor"]) == ["collage", "indoor", "outdoor"]
    assert kept("Taken from inside a car, looking down a highway.",
                ["car interior", "outdoor"]) == ["car interior", "outdoor"]
    assert kept("View from inside the vehicle of a road under trees.",
                ["car seat", "car window", "windshield", "interior space", "outdoor"]
                ) == ["car seat", "car window", "windshield", "interior space", "outdoor"]
    # one side emitted, the rival cued: dropped
    assert kept("Kids run across a sunny park.", ["kitchen", "gym"]) == []


# 6 -------------------------------------------------------------------------

def test_weather_group():
    assert kept("A beach under an overcast sky.", ["sunny day", "beach"]) == ["beach"]
    assert kept("A field under a partly cloudy sky.", ["sunny day"]) == ["sunny day"]
    assert kept("Fluffy clouds drift over the hill.", ["clear sky"]) == ["clear sky"]
    assert kept("Two people talk.", ["sunny day", "cloudy sky"]) == []
    assert kept("The sky is clear blue over the summit.",
                ["clear sky", "cloudy sky"]) == ["clear sky"]
    assert kept("A cloudy, gloomy afternoon.", ["clear sky", "fog"]) == ["fog"]


def test_partly_cloudy_weakly_cues_the_cloudy_side():
    """Owner decision 2026-10-18."""
    assert kept("A field under a partly cloudy sky.",
                ["clear sky", "cloudy sky"]) == ["cloudy sky"]
    assert kept("A field under a partly cloudy sky.",
                ["clear blue sky", "clear day", "sunny day", "blue sky", "sun"]
                ) == ["sunny day", "blue sky", "sun"]
    assert kept("A hillside with scattered white clouds.", ["clear sky"]) == []
    assert kept("Partly sunny with some clouds.", ["clear day"]) == []
    # a sunny cue too: both sides cued, both kept
    assert kept("A partly cloudy but sunny afternoon at the park.",
                ["clear sky", "cloudy sky"]) == ["clear sky", "cloudy sky"]
    # a bare "clouds" stays neutral
    assert kept("Clouds drift over the hill.", ["clear sky"]) == ["clear sky"]
    reason = cg.ground("Under a partly cloudy sky.", ["clear sky"]).dropped["clear sky"]
    assert reason.startswith("weather: overcast weakly cued")


def test_indoor_venue_keeps_basketball_court_on_gym_soccer():
    """Owner decision 2026-10-18."""
    cats = ["soccer", "basketball", "basketball court", "basketball hoop"]
    assert kept("Girls play soccer on an indoor gymnasium court.", cats) == [
        "soccer", "basketball court", "basketball hoop"]
    for desc in ("Kids play soccer inside a school hall.",
                 "Youth soccer in an arena.", "Soccer practice in the gym."):
        assert set(kept(desc, cats)) == {"soccer", "basketball court", "basketball hoop"}
    # the venue terms do not count as "basketball emitted" there either
    assert kept("Kids kick a ball in a gym.", ["soccer", "basketball court"]) == [
        "soccer", "basketball court"]
    # outdoor or uncued: still dropped, incl. 251357's outdoor turf field
    assert kept("Kids play soccer on an outdoor turf field.", cats) == ["soccer"]
    assert kept("Kids play soccer.", cats) == ["soccer"]
    g = cg.ground(DESC57, T0)
    assert BASKETBALL <= set(g.dropped)


# 7 -------------------------------------------------------------------------

_CUE_WORDS = ["soccer", "basketball hoop", "baseball", "indoors", "outdoors",
              "gym", "scattered clouds", "partly sunny", "arena",
              "kitchen", "trees", "lamps", "daylight", "overcast", "sunny",
              "partly cloudy", "night", "inside a car", "collage", "light blue",
              "windows showing greenery outside", "not outdoors"]


def test_idempotent_on_fixtures_and_random_subsets():
    cases = [(DESC57, T0), (DESC57, T04)]
    cases += [(FIXTURES[k]["description"], FIXTURES[k]["categories"])
              for k in ("251356", "251358")]
    rng = random.Random(1234)
    group_terms = sorted({t for g in cg.GROUPS for s in g.sides for t in s.terms})
    for _ in range(400):
        cats = rng.sample(CONTENT_VOCABULARY, rng.randint(0, 30))
        cats += rng.sample(group_terms, rng.randint(0, 8))
        cats = list(dict.fromkeys(cats))
        desc = "A photo. " + " and ".join(rng.sample(_CUE_WORDS, rng.randint(0, 5))) + "."
        cases.append((desc, cats))
    for desc, cats in cases:
        once = cg.ground(desc, cats).kept
        assert cg.ground(desc, once).kept == once


# 8 -------------------------------------------------------------------------

def test_subset_in_order_nothing_added_and_empty_description():
    rng = random.Random(99)
    for _ in range(200):
        cats = rng.sample(CONTENT_VOCABULARY, rng.randint(0, 40))
        desc = " ".join(rng.sample(_CUE_WORDS, 3))
        out = cg.ground(desc, cats)
        assert out.kept == [t for t in cats if t in out.kept]
        assert set(out.kept) | set(out.dropped) == set(cats)
        assert not set(out.kept) & set(out.dropped)
    for desc in (None, "", "   "):
        g = cg.ground(desc, T04)
        assert g.kept == T04 and g.dropped == {} and g.recitation is False
        assert cg.required_cue_families_diagnostic(desc, T04) == {}
    assert cg.ground(DESC57, []).kept == []


# 9 -------------------------------------------------------------------------

def test_every_rule_term_is_real_vocabulary():
    vocab = set(CONTENT_VOCABULARY)
    for g in cg.GROUPS:
        assert set(g.skip_when) <= vocab
        seen = set()
        for s in g.sides:
            assert set(s.terms) <= vocab, (g.name, set(s.terms) - vocab)
            assert not seen & set(s.terms), (g.name, s.name, seen & set(s.terms))
            seen |= set(s.terms)
            for c in s.cues:
                re.compile(c)
    assert cg.OBJECT_TERMS <= vocab, cg.OBJECT_TERMS - vocab
    for fam, (terms, cues) in cg.REQUIRED_CUE_FAMILIES.items():
        assert set(terms) <= vocab, (fam, set(terms) - vocab)
        for c in cues:
            re.compile(c)
    for p in cg.SCRUB_PATTERNS:
        re.compile(p)
    # the indoor side must not claim a vehicle interior
    indoor = next(s for g in cg.GROUPS for s in g.sides if s.name == "indoor")
    assert not {"car interior", "car seat", "car window", "windshield",
                "interior space"} & set(indoor.terms)


# 10 ------------------------------------------------------------------------

def test_recitation_verdict():
    # llama-style: the first 60 vocabulary entries, verbatim order
    recital = list(CONTENT_VOCABULARY[:60])
    assert cg.is_recitation(DESC57, recital)
    assert cg.ground(DESC57, recital).recitation

    # the 68-tag gemma answer, in the order it was relayed (near-alphabetical,
    # the HARDER case for an order-gated verdict): not a recitation
    assert cg.order_share(T0) >= cg.RECITATION_MIN_ORDER
    assert not cg.is_recitation(DESC57, T0)
    assert not cg.ground(DESC57, T0).recitation
    assert not cg.is_recitation(DESC57, T04)

    # a grounded alphabetical 44-tag list
    stats = cg.recitation_stats(DESC57, T0)
    grounded = [t for t in cg.ground(DESC57, T0).kept
                if t not in stats["unsupported"]][:44]
    assert len(grounded) == 44
    assert cg.order_share(grounded) >= 0.9
    assert not cg.is_recitation(DESC57, grounded)
    assert not cg.is_recitation(None, recital)


def test_alphabetical_recital_of_concrete_nouns_is_a_recitation():
    """The llama shape: household nouns in vocabulary order, none in the text."""
    nouns = [t for t in CONTENT_VOCABULARY if t in cg.OBJECT_TERMS][:30]
    assert {"bed", "book", "bottle", "bowl", "box"} <= set(nouns)
    assert cg.is_recitation(DESC57, nouns)
    assert cg.is_recitation("A tree with pink blossoms in a garden.", nouns)


def test_family_terms_are_not_objects():
    """The family terms are gemma's padding vocabulary; counting them as
    objects flagged 23 honest gemma rows (see OBJECT_TERMS)."""
    family = {t for terms, _ in cg.REQUIRED_CUE_FAMILIES.values() for t in terms}
    for t in ("building", "architecture", "city", "urban environment", "mountain",
              "court", "coach", "group portrait", "celebration", "guest", "shadow",
              "basketball", "road"):
        assert t in family and t not in cg.OBJECT_TERMS, t
    padding = sorted({"building", "architecture", "city", "urban environment",
                      "mountain", "court", "coach", "group portrait", "celebration",
                      "guest", "shadow", "road", "kitchen", "hat"},
                     key=CONTENT_VOCABULARY.index)
    assert not cg.is_recitation(DESC57, padding)


def test_family_diagnostic_reports_without_dropping():
    diag = cg.required_cue_families_diagnostic(DESC57, T0)
    assert diag["basketball"] == ["basketball", "basketball court", "basketball hoop"]
    assert "coach" in diag and "court" in diag and "wall" in diag
    # diagnostic families never drop: coach, court, wall survive ground()
    assert {"coach", "court", "wall", "building"} <= set(cg.ground(DESC57, T0).kept)
    # a cue protects its family
    assert "hat" not in cg.required_cue_families_diagnostic(
        "A coach in a baseball cap.", ["hat", "coach"])


def test_constants():
    assert cg.RECITATION_MIN_ORDER == 0.9
    assert cg.RECITATION_MIN_UNSUPPORTED == 10
    assert cg.RECITATION_MIN_UNSUPPORTED_SHARE == 0.5
    assert cg.MAX_PLAUSIBLE == 120
    assert [g.name for g in cg.GROUPS] == ["sport", "setting", "light", "weather"]

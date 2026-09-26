"""Derived capture-fact visual tags (EXIF-authoritative) + the merge rule.

The category-visual VLM sees a ~336 px tile and cannot read a shutter speed off
it. Measured on the July library copy (151,291 tagged photos): it put
`long-exposure` on 7,885 photos of which only 5.8% were actually >= 1/15 s, and
it found only 25.1% of the genuine >= 1/4 s exposures. These tests pin the
deterministic replacement and the "derived wins" merge.
"""

import json
import math

import pytest

from photosearch import visual_tags_derive as V


# ---------------------------------------------------------------------------
# Vocabulary split
# ---------------------------------------------------------------------------

def test_capture_fact_and_perceived_partition_the_vocabulary():
    """Four disjoint groups covering the compiled vocabulary exactly. The
    full four-way assertion lives in tests/test_visual_tags_frozen.py."""
    from photosearch.vocab_visual import VISUAL_VOCABULARY

    overlap = set(V.PERCEIVED_VOCABULARY) & set(V.CAPTURE_FACT_TAGS)
    assert overlap == set(), f"a term cannot be both derived and perceived: {overlap}"
    assert (set(V.PERCEIVED_VOCABULARY) | set(V.CAPTURE_FACT_TAGS)
            | set(V.RETIRED_TAGS) | set(V.FROZEN_TAGS)) == set(VISUAL_VOCABULARY)
    # Every capture-fact / retired / frozen term really is in the compiled
    # vocabulary — a typo would silently name a tag nothing else knows about.
    assert set(V.CAPTURE_FACT_TAGS) <= set(VISUAL_VOCABULARY)
    assert set(V.RETIRED_TAGS) <= set(VISUAL_VOCABULARY)
    assert set(V.FROZEN_TAGS) <= set(VISUAL_VOCABULARY)


def test_motion_blur_is_retired_from_both_halves():
    # Zero signal: median aes_sharpness is 7.0 both tagged and untagged, and a
    # scalar cannot separate motion blur from defocus. Neither asked nor derived.
    assert "motion-blur" in V.RETIRED_TAGS
    assert "motion-blur" not in V.PERCEIVED_VOCABULARY
    assert "motion-blur" not in V.CAPTURE_FACT_TAGS


def test_prompt_sections_partition_the_perceived_vocabulary():
    seen = [t for _, groups in V.PROMPT_SECTIONS for g in groups for t in g]
    assert sorted(seen) == sorted(V.PERCEIVED_VOCABULARY)
    assert len(seen) == len(set(seen)), "a term appears in two sections"


def test_contradictory_pairs_are_perceived_terms_only():
    for a, b in V.CONTRADICTORY_PAIRS:
        assert a in V.PERCEIVED_VOCABULARY, a
        assert b in V.PERCEIVED_VOCABULARY, b
        assert a != b


# ---------------------------------------------------------------------------
# EXIF TEXT parsing — the shapes that actually occur in the library
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("1/250", 1 / 250),          # the common shutter shape
    ("1/160", 1 / 160),
    ("5001/200000", 5001 / 200000),   # iPhone rational, ~1/40
    ("33333/1000000", 33333 / 1000000),
    ("10/1199", 10 / 1199),
    ("2", 2.0),                  # whole seconds
    ("1/2", 0.5),
    ("0.004", 0.004),            # decimal seconds
    ("63/10", 6.3),              # f_number rational
    ("1244236/699009", 1244236 / 699009),
    ("173/100", 1.73),
    ("8", 8.0),
])
def test_parse_exif_number_real_shapes(raw, expected):
    assert V.parse_exif_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    ('2"', 2.0),            # some writers mark whole seconds with a double quote
    ("1/250 sec", 1 / 250),
    (" 1/250 ", 1 / 250),
    ("2 s", 2.0),
])
def test_parse_exif_number_tolerates_unit_suffixes(raw, expected):
    assert V.parse_exif_number(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [
    None, "", "   ", "abc", "1/0", "0/0", "-1", "-1/2", "0", "1/", "/2", "1/2/3",
    float("nan"), [], {},
])
def test_parse_exif_number_rejects_junk(raw):
    assert V.parse_exif_number(raw) is None


def test_parse_exif_number_accepts_native_numbers():
    assert V.parse_exif_number(2) == 2.0
    assert V.parse_exif_number(0.004) == pytest.approx(0.004)


# ---------------------------------------------------------------------------
# EV100
# ---------------------------------------------------------------------------

def test_ev100_matches_the_textbook_sunny_16_value():
    # f/16, 1/100 s, ISO 100 -> EV100 = log2(256/0.01) = ~14.6
    assert V.ev100("16", "1/100", 100) == pytest.approx(math.log2(256 / 0.01))


def test_ev100_is_none_when_any_input_is_missing_or_invalid():
    assert V.ev100(None, "1/100", 100) is None
    assert V.ev100("16", None, 100) is None
    assert V.ev100("16", "1/100", None) is None
    assert V.ev100("16", "1/100", 0) is None
    assert V.ev100("abc", "1/100", 100) is None


# ---------------------------------------------------------------------------
# derive_tags
# ---------------------------------------------------------------------------

def _row(**kw):
    base = {"exposure_time": None, "f_number": None, "iso": None,
            "image_width": None, "image_height": None, "aes_sharpness": None}
    base.update(kw)
    return base


def test_missing_exif_derives_nothing():
    assert V.derive_tags(_row()) == []


def test_long_exposure_fires_at_the_threshold_and_not_below():
    assert "long-exposure" in V.derive_tags(_row(exposure_time="1/4"))
    assert "long-exposure" in V.derive_tags(_row(exposure_time="2"))
    assert "long-exposure" not in V.derive_tags(_row(exposure_time="1/5"))
    # The measured failure: a 1/125 s daytime sports frame.
    assert "long-exposure" not in V.derive_tags(_row(exposure_time="1/125"))
    assert "long-exposure" not in V.derive_tags(_row(exposure_time="1/800"))


def test_low_light_uses_ev100_not_iso_alone():
    # Full-sun soccer: ISO 160, f/5.6, 1/800 -> EV100 ~ 13.2. NOT low light,
    # even though the VLM put `low-light` on 33% of that folder.
    bright = _row(exposure_time="1/800", f_number="28/5", iso=160)
    assert "low-light" not in V.derive_tags(bright)
    # Dim interior: f/2.8, 1/30, ISO 1600 -> EV100 ~ 1.2.
    dim = _row(exposure_time="1/30", f_number="14/5", iso=1600)
    assert "low-light" in V.derive_tags(dim)


def test_low_light_boundary_is_strict():
    # f/8, 2 s, ISO 100 is EV100 exactly 5.0 -> NOT low light (strict <).
    assert "low-light" not in V.derive_tags(_row(exposure_time="2", f_number="8", iso=100))
    # f/8, 4 s, ISO 100 is EV100 4.0.
    assert "low-light" in V.derive_tags(_row(exposure_time="4", f_number="8", iso=100))


def test_low_light_needs_all_three_exposure_terms():
    assert V.derive_tags(_row(exposure_time="1/30", f_number="14/5")) == []
    assert V.derive_tags(_row(exposure_time="1/30", iso=1600)) == []


def test_panoramic_from_aspect_ratio_either_orientation():
    assert "panoramic" in V.derive_tags(_row(image_width=8000, image_height=2000))
    assert "panoramic" in V.derive_tags(_row(image_width=2000, image_height=8000))
    # An ordinary 3:2 frame is not a panorama (the VLM called 4,290 of them one).
    assert "panoramic" not in V.derive_tags(_row(image_width=7008, image_height=4672))
    assert "panoramic" not in V.derive_tags(_row(image_width=0, image_height=4672))


def test_sharp_and_blurry_are_never_derived_from_aes_sharpness():
    """FROZEN, not derived: `aes_sharpness` conflates sharpness with general
    technical quality. See tests/test_visual_tags_frozen.py."""
    for score in range(1, 11):
        assert V.derive_tags(_row(aes_sharpness=score)) == []
    assert V.derive_tags(_row(aes_sharpness=None)) == []


def test_derive_tags_output_is_sorted_and_deduped():
    row = _row(exposure_time="4", f_number="8", iso=100,
               image_width=9000, image_height=3000, aes_sharpness=9)
    got = V.derive_tags(row)
    assert got == sorted(got)
    assert len(got) == len(set(got))
    assert set(got) == {"long-exposure", "low-light", "panoramic"}


def test_derive_tags_accepts_a_sqlite_row(tmp_path):
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE t (exposure_time TEXT, f_number TEXT, iso INTEGER, "
                 "image_width INTEGER, image_height INTEGER, aes_sharpness REAL)")
    conn.execute("INSERT INTO t VALUES ('4', '8', 100, 6000, 4000, NULL)")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert V.derive_tags(row) == ["long-exposure", "low-light"]


def test_derive_tags_only_ever_returns_capture_fact_terms():
    row = _row(exposure_time="4", f_number="8", iso=100,
               image_width=9000, image_height=3000, aes_sharpness=1)
    assert set(V.derive_tags(row)) <= set(V.CAPTURE_FACT_TAGS)


# ---------------------------------------------------------------------------
# merge_tags — derived is AUTHORITATIVE
# ---------------------------------------------------------------------------

def test_merge_strips_every_capture_fact_term_the_vlm_emitted():
    got = V.merge_tags(["long-exposure", "low-light", "sharp", "blurry",
                        "panoramic", "peaceful"], [])
    assert got == ["peaceful"]  # frozen terms go too on the `vlm` path


def test_merge_strips_retired_terms():
    assert V.merge_tags(["motion-blur", "joyful"], []) == ["joyful"]


def test_merge_adds_the_derived_terms():
    assert V.merge_tags(["peaceful"], ["long-exposure"]) == ["long-exposure", "peaceful"]


def test_merge_lets_perceived_terms_through_untouched():
    perceived = ["peaceful", "golden-hour", "centered"]
    assert V.merge_tags(perceived, []) == sorted(perceived)


def test_merge_is_deterministic_and_deduped():
    a = V.merge_tags(["centered", "peaceful", "peaceful"], ["low-light", "low-light"])
    b = V.merge_tags(["peaceful", "centered"], ["low-light"])
    assert a == b == ["centered", "low-light", "peaceful"]


def test_merge_is_idempotent():
    row = _row(exposure_time="1/800", f_number="28/5", iso=160)
    once = V.merge_for_row(["long-exposure", "low-light", "peaceful"], row)
    twice = V.merge_for_row(once, row)
    assert once == twice == ["peaceful"]


def test_merge_drops_out_of_vocabulary_junk():
    assert V.merge_tags(["peaceful", "definitely-not-a-tag", ""], []) == ["peaceful"]


def test_merge_handles_none_and_empty():
    assert V.merge_tags(None, None) == []
    assert V.merge_tags([], []) == []


def test_merge_for_row_is_the_whole_rule_in_one_call():
    # 1/800 s full-sun frame that the VLM called long-exposure + low-light.
    row = _row(exposure_time="1/800", f_number="28/5", iso=160,
               image_width=6000, image_height=4000)
    assert V.merge_for_row(["long-exposure", "low-light", "motion-blur", "joyful"],
                           row) == ["joyful"]
    # A real 2 s tripod night frame keeps its (now derived) capture facts.
    night = _row(exposure_time="4", f_number="8", iso=100)
    assert V.merge_for_row(["peaceful"], night) == \
        ["long-exposure", "low-light", "peaceful"]


def test_merge_for_row_with_no_exif_only_strips():
    assert V.merge_for_row(["long-exposure", "peaceful"], _row()) == ["peaceful"]


def test_derive_columns_all_exist_on_the_photos_table(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(photos)")}
    assert set(V.DERIVE_COLUMNS) <= cols


# ---------------------------------------------------------------------------
# Thresholds are named constants, not magic numbers in the body
# ---------------------------------------------------------------------------

def test_thresholds_are_the_validated_values():
    assert V.LONG_EXPOSURE_MIN_SECONDS == 0.25
    assert V.LOW_LIGHT_MAX_EV100 == 5.0
    assert V.PANORAMIC_MIN_ASPECT == 2.0


def test_json_round_trip_shape_is_a_plain_list_of_str():
    row = _row(exposure_time="4", f_number="8", iso=100)
    out = V.merge_for_row(["peaceful"], row)
    assert json.loads(json.dumps(out)) == out
    assert all(isinstance(t, str) for t in out)

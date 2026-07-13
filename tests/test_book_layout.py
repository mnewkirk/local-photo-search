"""Layout-template geometry for the photo-book editor (photosearch/book.py).

Covers the two house-style framing modes (framed / full-bleed), the equal-column
'gallery row' archetype, and the per-archetype cell count that lets switching a
1-photo spread to 'matched 2-up' open an empty second slot.
"""
from photosearch.book import archetype_layout, archetype_cell_count

SW, SH = 28.0, 11.0


def _rects(arch, n, bleed):
    m, g = (0.0, 0.12) if bleed else (0.4, 0.5)
    return archetype_layout(arch, n, SW, SH, m=m, g=g)


def test_matched_2up_framed_has_white_margin():
    r = _rects("matched 2-up", 2, bleed=False)
    assert len(r) == 2
    # outer margin present: first cell not at the page edge, not full height
    assert r[0][0] > 0 and r[0][1] > 0
    assert r[0][3] < SH


def test_matched_2up_bleed_runs_edge_to_edge():
    r = _rects("matched 2-up", 2, bleed=True)
    assert len(r) == 2
    # left cell hugs the top-left corner and fills the full page height
    assert r[0][0] == 0 and r[0][1] == 0 and r[0][3] == SH
    # the two cells cover the whole spread width bar a hairline spine gutter
    covered = r[0][2] + r[1][2]
    assert SW - covered < 0.2
    # right cell ends flush with the right page edge
    assert abs((r[1][0] + r[1][2]) - SW) < 1e-6


def test_gallery_row_equal_full_height_columns():
    r = _rects("gallery row", 3, bleed=True)
    assert len(r) == 3
    widths = [round(c[2], 4) for c in r]
    assert len(set(widths)) == 1          # equal columns
    assert all(c[3] == SH for c in r)     # full page height (bleed)
    assert r[0][0] == 0                    # first column at the left edge


def test_gallery_row_not_preempted_by_two_photo_shortcut():
    # n == 2 must not divert 'gallery row' into the matched-2-up branch
    r = _rects("gallery row", 2, bleed=False)
    assert len(r) == 2


def test_single_photo_is_full_bleed_regardless_of_archetype():
    assert archetype_layout("asymmetric collage", 1, SW, SH) == [[0, 0, SW, SH]]


def test_framed_single_floats_on_margin_not_full_bleed():
    # 'single (framed)' must escape the n==1 full-bleed short-circuit and float
    # the photo inside the outer white margin.
    r = _rects("single (framed)", 1, bleed=False)
    assert len(r) == 1
    x, y, w, h = r[0]
    assert x == 0.4 and y == 0.4           # floated by the margin
    assert w == SW - 0.8 and h == SH - 0.8


def test_hero_sidebar_anchor_wider_than_half_and_stacked():
    r = _rects("hero + sidebar", 4, bleed=False)
    assert len(r) == 4
    anchor = r[0]
    assert anchor[0] == 0.4                 # anchor on the left
    assert anchor[2] > SW * 0.5             # wider than a 50% split
    # sidebar cells stack in a single right-hand column (shared x, equal width)
    side = r[1:]
    assert len({round(c[0], 4) for c in side}) == 1
    assert len({round(c[2], 4) for c in side}) == 1
    assert side[0][0] > anchor[0] + anchor[2]   # column sits right of the anchor


def test_hero_sidebar_right_flips_anchor_but_keeps_it_cell_zero():
    r = _rects("hero + sidebar (right)", 4, bleed=False)
    anchor = r[0]
    # anchor is still cell 0 (caption + hero slot) but now hugs the right edge
    assert abs((anchor[0] + anchor[2]) - (SW - 0.4)) < 1e-6
    assert anchor[2] > SW * 0.5
    # sidebar column is to the LEFT of the anchor
    assert r[1][0] < anchor[0]


def test_cell_count_opens_empty_slot_for_fixed_archetypes():
    # 1 photo, switch to matched 2-up -> 2 cells (second is an empty drop slot)
    assert archetype_cell_count("matched 2-up", 1) == 2
    assert archetype_cell_count("matched 2-up", 0) == 2
    assert archetype_cell_count("full-bleed single", 1) == 1
    assert archetype_cell_count("full-spread panorama", 3) == 1
    assert archetype_cell_count("single (framed)", 3) == 1     # always one photo
    # hero + sidebar keeps at least 2 (anchor + one sidebar slot), else scales
    assert archetype_cell_count("hero + sidebar", 1) == 2
    assert archetype_cell_count("hero + sidebar (right)", 5) == 5
    # gallery row ("matched 3-up (row)") keeps at least 3 open slots — so
    # switching back to it after a photo was dropped gives an empty re-drop slot
    assert archetype_cell_count("gallery row", 1) == 3
    assert archetype_cell_count("gallery row", 2) == 3
    assert archetype_cell_count("gallery row", 4) == 4
    # photo-driven archetypes scale with the photos present
    assert archetype_cell_count("asymmetric collage", 5) == 5
    assert archetype_cell_count("dense grid", 8) == 8

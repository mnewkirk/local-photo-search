from photosearch.shutterfly_export import (
    placed_photo_order, build_book_manifest,
)

def _doc():
    return {
        "book": {"id": 17, "name": "The South of France", "subtitle": "Provence",
                 "cover_photo_id": 240555, "title_page_photo_id": 240539,
                 "back_cover_photo_id": None},
        "stage_w": 28.0, "stage_h": 11.0,
        "spreads": [
            {"position": 0, "archetype": "title", "bg": "#ffffff",
             "caption": {"text": "We found a castle.", "dark": False},
             "cells": [{"photo_id": 240539, "x": 14.5, "y": 0.5, "w": 13.0, "h": 10.0,
                        "fit": "cover", "crop_cx": 0.53, "crop_cy": 0.55,
                        "crop_zoom": 1.0, "crop_min_w": 0.0, "crop_min_h": 0.0,
                        "align": None, "position": 0}]},
            {"position": 1, "archetype": "matched 2-up", "bg": "#ffffff",
             "caption": None,
             "cells": [
                {"photo_id": 240599, "x": 0.4, "y": 0.4, "w": 8.7, "h": 10.2,
                 "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5, "crop_zoom": 1.0,
                 "crop_min_w": 0.0, "crop_min_h": 0.0, "align": None, "position": 0},
                {"photo_id": 240539, "x": 9.4, "y": 0.4, "w": 8.7, "h": 10.2,
                 "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5, "crop_zoom": 1.0,
                 "crop_min_w": 0.0, "crop_min_h": 0.0, "align": None, "position": 1}]},
        ],
    }

def test_placed_photo_order_is_unique_and_ordered():
    # 240539 appears in both spreads but only once; cover 240555 appended last
    assert placed_photo_order(_doc()) == [240539, 240599, 240555]

def test_placed_photo_order_skips_none_cover():
    doc = _doc()
    doc["book"]["cover_photo_id"] = None
    doc["book"]["title_page_photo_id"] = None
    assert placed_photo_order(doc) == [240539, 240599]

def test_build_book_manifest_shape():
    m = build_book_manifest(_doc())
    assert m["book_id"] == 17
    assert m["name"] == "The South of France"
    assert m["stage_w"] == 28.0 and m["stage_h"] == 11.0
    assert m["spread_count"] == 2
    # spreads passed through with cells + captions preserved
    assert m["spreads"][0]["archetype"] == "title"
    assert m["spreads"][0]["caption"] == {"text": "We found a castle.", "dark": False}
    assert m["spreads"][1]["cells"][0]["photo_id"] == 240599
    # every placed photo id appears in the photos map (filenames filled in Task 4/runbook)
    assert set(m["photos"].keys()) == {"240539", "240599", "240555"}

from photosearch.shutterfly_export import build_gphotos_records

def test_build_gphotos_records_maps_and_orders():
    rows = {
        240599: {"filepath": "2026/2026-07-24/DSC1.JPG", "filename": "DSC1.JPG",
                 "description": "castle"},
        240539: {"filepath": "2026/2026-07-24/DSC2.JPG", "filename": "DSC2.JPG",
                 "description": None},
        999999: {"filepath": "x", "filename": "x.jpg", "description": None},  # not requested
    }
    resolve = lambda p: "/photos/" + p
    recs = build_gphotos_records([240539, 240599], rows, resolve)
    # keeps the original filename (unique per book; the mapping key)
    assert [r["filename"] for r in recs] == ["DSC2.JPG", "DSC1.JPG"]
    assert recs[0]["_resolved_filepath"] == "/photos/2026/2026-07-24/DSC2.JPG"
    assert recs[0]["orig_filename"] == "DSC2.JPG"
    assert recs[1]["description"] == "castle"

def test_build_gphotos_records_skips_missing_rows():
    recs = build_gphotos_records([1, 2], {1: {"filepath": "a", "filename": "a.jpg",
                                              "description": None}}, lambda p: p)
    assert [r["filename"] for r in recs] == ["a.jpg"]


def test_manifest_photo_ids_orders_and_ints():
    from photosearch.shutterfly_export import manifest_photo_ids
    m = {"photos": {"240539": {}, "240599": {}, "240555": {}}}
    assert manifest_photo_ids(m) == [240539, 240599, 240555]


def test_manifest_photo_ids_empty():
    from photosearch.shutterfly_export import manifest_photo_ids
    assert manifest_photo_ids({}) == []

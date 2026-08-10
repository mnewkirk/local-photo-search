from photosearch.shutterfly_export import enrich_manifest_filenames


def test_enrich_manifest_filenames_fills_upload_and_orig():
    manifest = {"photos": {"240599": {"upload_filename": None, "orig_filename": None,
                                      "sfly_asset_id": None}}}
    rows = {240599: {"filepath": "p", "filename": "DSC06241.JPG", "description": None}}
    out = enrich_manifest_filenames(manifest, rows)
    assert out["photos"]["240599"]["upload_filename"] == "sfly-240599.jpg"
    assert out["photos"]["240599"]["orig_filename"] == "DSC06241.JPG"


from click.testing import CliRunner
from unittest import mock
import cli as cli_mod


def test_gphotos_push_dry_run_lists_records(monkeypatch, tmp_path):
    # Fake BookStore.get_book + PhotoDB rows via monkeypatched helpers
    doc = {"book": {"id": 3, "name": "T", "subtitle": None, "cover_photo_id": None,
                    "title_page_photo_id": None, "back_cover_photo_id": None},
           "stage_w": 28, "stage_h": 11,
           "spreads": [{"position": 0, "archetype": "matched 2-up", "bg": "#fff",
                        "caption": None,
                        "cells": [{"photo_id": 5, "x": 0, "y": 0, "w": 1, "h": 1,
                                   "fit": "cover", "crop_cx": 0.5, "crop_cy": 0.5,
                                   "crop_zoom": 1, "crop_min_w": 0, "crop_min_h": 0,
                                   "align": None, "position": 0}]}]}
    monkeypatch.setattr(cli_mod, "_load_book_doc", lambda book_id, db, books_db: doc)
    monkeypatch.setattr(cli_mod, "_load_photo_rows_resolved",
                        lambda ids, db: ({5: {"filepath": "a", "filename": "A.JPG",
                                              "description": None}}, lambda p: "/photos/" + p))
    r = CliRunner().invoke(cli_mod.cli, ["shutterfly-gphotos-push", "--book", "3",
                                         "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "sfly-5.jpg" in r.output
    assert "1 photo" in r.output

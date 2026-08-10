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


def test_gphotos_push_manifest_dry_run(monkeypatch, tmp_path):
    import json as _json
    manifest = {"name": "The South of France",
                "photos": {"5": {"upload_filename": "sfly-5.jpg",
                                 "orig_filename": "A.JPG", "sfly_asset_id": None}}}
    p = tmp_path / "m.json"
    p.write_text(_json.dumps(manifest))
    monkeypatch.setattr(cli_mod, "_load_photo_rows_resolved",
                        lambda ids, db: ({5: {"filepath": "a", "filename": "A.JPG",
                                              "description": None}}, lambda x: "/photos/" + x))
    r = CliRunner().invoke(cli_mod.cli, ["shutterfly-gphotos-push",
                                         "--manifest", str(p), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "sfly-5.jpg" in r.output
    assert "2026 Summer — The South of France" in r.output


def test_gphotos_push_requires_exactly_one_source():
    r = CliRunner().invoke(cli_mod.cli, ["shutterfly-gphotos-push", "--dry-run"])
    assert r.exit_code != 0
    assert "exactly one" in r.output.lower()


def test_gphotos_push_streams_per_photo_progress(monkeypatch, tmp_path):
    import json as _json
    import photosearch.google_photos as real_gp
    manifest = {"name": "X", "photos": {
        "5": {"upload_filename": "sfly-5.jpg", "orig_filename": "A.JPG"},
        "6": {"upload_filename": "sfly-6.jpg", "orig_filename": "B.JPG"}}}
    p = tmp_path / "m.json"
    p.write_text(_json.dumps(manifest))
    monkeypatch.setattr(cli_mod, "_load_photo_rows_resolved",
                        lambda ids, db: (
                            {5: {"filepath": "a", "filename": "A.JPG", "description": None},
                             6: {"filepath": "b", "filename": "B.JPG", "description": None}},
                            lambda x: "/photos/" + x))
    monkeypatch.setattr(real_gp, "create_album", lambda db, title: "ALBUM123")

    def fake_upload(db, records, album_id=None, progress_callback=None, **kw):
        assert progress_callback is not None, "CLI must pass a progress_callback"
        out = []
        for i, r in enumerate(records, 1):
            progress_callback(i, len(records), r["filename"], "uploaded", None, "mid")
            out.append({"filename": r["filename"], "status": "uploaded",
                        "media_item_id": "mid"})
        return out
    monkeypatch.setattr(real_gp, "upload_photos", fake_upload)

    r = CliRunner().invoke(cli_mod.cli, ["shutterfly-gphotos-push",
                                         "--manifest", str(p)])
    assert r.exit_code == 0, r.output
    assert "[1/2] uploaded sfly-5.jpg" in r.output
    assert "[2/2] uploaded sfly-6.jpg" in r.output
    assert "Uploaded 2/2 to album ALBUM123" in r.output

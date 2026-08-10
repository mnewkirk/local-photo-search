from photosearch.shutterfly_export import enrich_manifest_filenames


def test_enrich_manifest_filenames_fills_upload_and_orig():
    manifest = {"photos": {"240599": {"upload_filename": None, "orig_filename": None,
                                      "sfly_asset_id": None}}}
    rows = {240599: {"filepath": "p", "filename": "DSC06241.JPG", "description": None}}
    out = enrich_manifest_filenames(manifest, rows)
    assert out["photos"]["240599"]["upload_filename"] == "sfly-240599.jpg"
    assert out["photos"]["240599"]["orig_filename"] == "DSC06241.JPG"

"""Tests for M31 Split print export.

The geometry here is a second implementation of frontend/dist/split-geometry.js
(the planner must recompute per drag frame, so it cannot live only on the
server). Both are pinned to the SAME published worked example, which is what
catches drift between them:

    6528x4352 -> six 13x19" portrait as 3 cols x 2 rows, gap 0.5", window mode
    -> 40" x 38.5", 113 dpi, 31% crop, 17,326 x 11,552 px for 300 dpi

The identical constants appear in frontend/__tests__/split-geometry.test.js.
"""

import json
import os
from pathlib import Path

import pytest

from photosearch import split_export as SX

# The worked example.
IMG_W, IMG_H = 6528, 4352
SHEET = dict(pw=13, ph=19, cols=3, rows=2)
OPTS = dict(gutter=0.5, mode="window")


def _db_with_photo(tmp_path, filepath, dbpath=None):
    from photosearch.db import PhotoDB
    db = PhotoDB(str(dbpath or tmp_path / "t.db"))
    db.set_photo_root(str(tmp_path))
    pid = db.add_photo(filepath=filepath, filename=os.path.basename(filepath),
                       date_taken="2024-07-04T10:00:00")
    return db, pid


def _photo(tmp_path, w=IMG_W, h=IMG_H, name="DSC1.JPG"):
    """A real image on disk, so Pillow has something to slice."""
    from PIL import Image
    p = tmp_path / "lib" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (90, 120, 160)).save(p, quality=60)
    return p


# ---------------------------------------------------------------------------
# Geometry — must agree with the JS module
# ---------------------------------------------------------------------------

class TestWorkedExample:
    P = SX.plan(IMG_W, IMG_H, **SHEET, **OPTS)

    def test_wall_footprint(self):
        assert self.P.outer_w == pytest.approx(40)
        assert self.P.outer_h == pytest.approx(38.5)

    def test_window_mode_includes_the_gaps_in_the_field(self):
        assert self.P.field_w == pytest.approx(40)
        assert self.P.field_h == pytest.approx(38.5)

    def test_delivers_113_dpi(self):
        assert round(self.P.dpi) == 113

    def test_costs_31_percent(self):
        assert round((1 - self.P.keep) * 100) == 31

    def test_height_binds_and_is_used_in_full(self):
        assert self.P.binding_axis == "height"
        assert self.P.crop_h == pytest.approx(IMG_H)
        assert round(self.P.crop_w) == 4522

    def test_required_source_for_300_dpi(self):
        n = SX.required_source_px(self.P, IMG_W, IMG_H, 300)
        assert (n["w"], n["h"]) == (17326, 11552)
        assert n["mp"] == 200.1
        assert n["k"] == 2.7


class TestGeometryInvariants:
    def test_keep_is_scale_free(self):
        a = SX.plan(3264, 2176, **SHEET, **OPTS)
        b = SX.plan(65280, 43520, **SHEET, **OPTS)
        assert a.keep == pytest.approx(b.keep)

    def test_binding_axis_agrees_with_the_zero_slack(self):
        for pw, ph in ((13, 19), (19, 13), (8, 10), (4, 6)):
            for cols, rows in ((3, 2), (2, 3), (1, 2), (4, 3)):
                p = SX.plan(IMG_W, IMG_H, pw=pw, ph=ph, cols=cols, rows=rows, **OPTS)
                if p.binding_axis == "height":
                    assert p.crop_h == pytest.approx(IMG_H)
                else:
                    assert p.crop_w == pytest.approx(IMG_W)

    def test_matching_aspect_crops_nothing(self):
        p = SX.plan(2000, 800, pw=10, ph=8, cols=2, rows=1,
                    gutter=0, mode="window")
        assert p.keep == pytest.approx(1)

    def test_continue_mode_excludes_the_gaps(self):
        p = SX.plan(IMG_W, IMG_H, **SHEET, gutter=0.5, mode="continue")
        assert p.field_w == pytest.approx(39)
        assert p.field_h == pytest.approx(38)
        assert p.outer_w == pytest.approx(40)


class TestPanelRects:
    def test_one_rect_per_panel_row_major(self):
        rects = SX.panel_rects(SX.plan(IMG_W, IMG_H, **SHEET, **OPTS))
        assert len(rects) == 6
        assert [(r["c"], r["r"]) for r in rects] == [
            (0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    def test_every_rect_stays_inside_the_source(self):
        rects = SX.panel_rects(SX.plan(IMG_W, IMG_H, **SHEET, **OPTS))
        for r in rects:
            assert r["sx"] >= -1e-6 and r["sy"] >= -1e-6
            assert r["sx"] + r["sw"] <= IMG_W + 1e-6
            assert r["sy"] + r["sh"] <= IMG_H + 1e-6

    def test_panels_are_uniform(self):
        rects = SX.panel_rects(SX.plan(IMG_W, IMG_H, **SHEET, **OPTS))
        assert all(r["sw"] == pytest.approx(rects[0]["sw"]) for r in rects)
        assert all(r["sh"] == pytest.approx(rects[0]["sh"]) for r in rects)

    def test_pan_moves_the_crop_along_the_slack_axis_only(self):
        base = SX.plan(IMG_W, IMG_H, **SHEET, **OPTS)
        left = SX.panel_rects(SX.plan(IMG_W, IMG_H, **SHEET, **OPTS, sx=-1))[0]
        right = SX.panel_rects(SX.plan(IMG_W, IMG_H, **SHEET, **OPTS, sx=1))[0]
        assert right["sx"] - left["sx"] == pytest.approx(IMG_W - base.crop_w)
        assert right["sy"] == pytest.approx(left["sy"])


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_writes_one_file_per_panel(tmp_path, monkeypatch):
    orig = _photo(tmp_path, 2400, 1600)
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    try:
        res = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
    finally:
        db.close()

    assert res["skipped"] is False
    assert len(res["panels"]) == 6
    assert res["grid"] == [3, 2]
    for p in res["panels"]:
        f = Path(p["output"])
        assert f.exists() and f.stat().st_size > 0
    assert {p["name"] for p in res["panels"]} == {
        f"panel-r{r}c{c}.jpg" for r in (1, 2) for c in (1, 2, 3)}


def test_exported_panels_have_the_right_pixel_shape(tmp_path, monkeypatch):
    from PIL import Image
    orig = _photo(tmp_path, 2400, 1600)
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    try:
        res = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
    finally:
        db.close()

    p = SX.plan(2400, 1600, pw=13, ph=19, cols=3, rows=2)
    rect = SX.panel_rects(p)[0]
    with Image.open(res["panels"][0]["output"]) as im:
        # Each panel carries its share of the crop, within rounding.
        assert abs(im.width - round(rect["sw"])) <= 1
        assert abs(im.height - round(rect["sh"])) <= 1
        # A 13:19 sheet must come out portrait.
        assert im.height > im.width


def test_export_writes_a_manifest(tmp_path, monkeypatch):
    orig = _photo(tmp_path, 2400, 1600)
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    try:
        res = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
    finally:
        db.close()

    man = json.loads((Path(res["dest"]) / "manifest.json").read_text())
    assert man["photo_id"] == pid
    assert man["source_px"] == [2400, 1600]
    assert len(man["panels"]) == 6
    assert man["plan"]["cols"] == 3 and man["plan"]["rows"] == 2


def test_different_arrangements_do_not_collide(tmp_path, monkeypatch):
    """Same photo, two plans — the second must not overwrite the first."""
    orig = _photo(tmp_path, 2400, 1600)
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    try:
        a = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
        b = SX.export_panels(db, pid, pw=8, ph=10, cols=2, rows=1)
        c = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2,
                             mode="continue")
        again = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
    finally:
        db.close()

    dests = {a["dest"], b["dest"], c["dest"]}
    assert len(dests) == 3
    assert again["skipped"] is True and again["dest"] == a["dest"]


def test_source_outside_the_export_tree_is_refused(tmp_path, monkeypatch):
    """Same containment rule as the file-serving route — this reads real files."""
    orig = _photo(tmp_path, 800, 600)
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"not yours")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    try:
        with pytest.raises(SX.SplitError, match="outside the export directory"):
            SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2,
                             source_path=str(outside))
        with pytest.raises(SX.SplitError, match="outside the export directory"):
            SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2,
                             source_path=str(tmp_path / "exports" / ".." / "secret.jpg"))
    finally:
        db.close()


def test_export_uses_an_upscaled_source_when_given_one(tmp_path, monkeypatch):
    """The whole point of the upscale tie-in: more pixels per inch of paper."""
    orig = _photo(tmp_path, 1200, 800)
    export = tmp_path / "exports"
    big = export / "2024" / "DSC1-topaz-upscale.JPG"
    big.parent.mkdir(parents=True)
    from PIL import Image
    Image.new("RGB", (4800, 3200), (10, 10, 10)).save(big, quality=60)

    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    try:
        small = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2)
        large = SX.export_panels(db, pid, pw=13, ph=19, cols=3, rows=2,
                                 source_path=str(big), overwrite=True)
    finally:
        db.close()

    assert large["source_px"] == [4800, 3200]
    assert large["dpi"] == pytest.approx(small["dpi"] * 4, rel=0.01)
    # Crop is a shape mismatch, so upscaling must not change it.
    assert large["crop_pct"] == pytest.approx(small["crop_pct"], abs=0.1)


def test_unsupported_format_and_bad_grid_are_rejected(tmp_path, monkeypatch):
    orig = _photo(tmp_path, 800, 600)
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    try:
        with pytest.raises(SX.SplitError, match="unsupported format"):
            SX.export_panels(db, pid, pw=13, ph=19, cols=2, rows=1, fmt="webp")
        with pytest.raises(SX.SplitError, match="at least 1"):
            SX.export_panels(db, pid, pw=13, ph=19, cols=0, rows=1)
    finally:
        db.close()


def test_missing_photo_raises(tmp_path, monkeypatch):
    orig = _photo(tmp_path, 800, 600)
    db, _ = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    try:
        with pytest.raises(ValueError, match="not found"):
            SX.export_panels(db, 999999, pw=13, ph=19, cols=2, rows=1)
    finally:
        db.close()


def test_split_export_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from photosearch import web

    orig = _photo(tmp_path, 1600, 1200)
    dbpath = tmp_path / "r.db"
    db, pid = _db_with_photo(tmp_path, str(orig), dbpath=dbpath)
    db.close()

    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(web, "_db_path", str(dbpath))
    monkeypatch.setattr(web, "_nas_url", None, raising=False)

    c = TestClient(web.app)
    r = c.post("/api/admin/split-export", json={
        "photo_id": pid, "pw": 13, "ph": 19, "cols": 2, "rows": 1})
    assert r.status_code == 200, r.text
    assert len(r.json()["panels"]) == 2

    bad = c.post("/api/admin/split-export", json={
        "photo_id": 999999, "pw": 13, "ph": 19, "cols": 2, "rows": 1})
    assert bad.status_code == 404

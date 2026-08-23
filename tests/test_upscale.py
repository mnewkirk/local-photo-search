"""Tests for M30 local Topaz upscaling.

The Topaz binary is a licensed Windows GUI app, so every test here fakes the
subprocess layer. The behaviors worth pinning are exactly the ones that bit
during bring-up:

  - a successful run exits **9** (WSL interop), never 0, so success must be
    decided by "an output file appeared" rather than by the exit code
  - the documented *failure* codes (253/254/255) must still be honored
  - a non-zero exit that DID produce a file is a success
  - the scale override edits a global Topaz preference and must be restored
    even when the run raises
  - nothing is ever written to the DB or the library
"""

import os
from pathlib import Path

import pytest

from photosearch import upscale as U


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Proc:
    def __init__(self, returncode=9, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def fake_cli(monkeypatch, tmp_path):
    """Pretend tpai.exe exists at a real path so cli_path() resolves."""
    exe = tmp_path / "tpai.exe"
    exe.write_text("")
    monkeypatch.setenv("PHOTOSEARCH_TOPAZ_CLI", str(exe))
    # Path translation is a no-op off WSL; pin it so the tests are host-agnostic.
    monkeypatch.setattr(U, "is_wsl", lambda: False)
    return str(exe)


def _runner(returncode=9, produce="out.jpg", record=None):
    """Build a subprocess.run stand-in that writes ``produce`` into --output."""
    def _run(cmd, **kwargs):
        if record is not None:
            record.append(cmd)
        out_dir = Path(cmd[cmd.index("--output") + 1])
        if produce:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / produce).write_bytes(b"\xff\xd8upscaled")
        return _Proc(returncode)
    return _run


# ---------------------------------------------------------------------------
# Exit-code handling — the core gotcha
# ---------------------------------------------------------------------------

def test_exit_9_with_output_is_success(fake_cli, tmp_path, monkeypatch):
    """A real successful run exits 9, not 0. It must not be read as failure."""
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=9))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    got = U.run_topaz(str(src), str(tmp_path / "out"))
    assert Path(got).exists()
    assert Path(got).name == "out.jpg"


def test_exit_127_with_output_is_success(fake_cli, tmp_path, monkeypatch):
    """Git Bash flattens the code to 127; a produced file still means success."""
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=127))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    assert Path(U.run_topaz(str(src), str(tmp_path / "out"))).exists()


def test_exit_0_with_output_is_success(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=0))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    assert Path(U.run_topaz(str(src), str(tmp_path / "out"))).exists()


@pytest.mark.parametrize("code,needle", [
    (255, "no valid files"),
    (254, "invalid log token"),
    (253, "invalid argument"),
])
def test_documented_failure_codes_raise(fake_cli, tmp_path, monkeypatch, code, needle):
    """The failure codes ARE trustworthy and must surface their meaning."""
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=code, produce=None))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    with pytest.raises(U.TopazError) as ei:
        U.run_topaz(str(src), str(tmp_path / "out"))
    assert needle in str(ei.value)


def test_no_output_file_raises_even_on_success_code(fake_cli, tmp_path, monkeypatch):
    """Exit 9 with nothing written is the silent-stub failure mode."""
    monkeypatch.setattr(U.subprocess, "run",
                        _runner(returncode=9, produce=None))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    with pytest.raises(U.TopazError, match="produced no output"):
        U.run_topaz(str(src), str(tmp_path / "out"))


def test_preexisting_files_are_not_mistaken_for_output(fake_cli, tmp_path, monkeypatch):
    """Only files that appear during the run count."""
    out = tmp_path / "out"; out.mkdir()
    (out / "stale.jpg").write_bytes(b"old")
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=9, produce=None))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    with pytest.raises(U.TopazError, match="produced no output"):
        U.run_topaz(str(src), str(out))


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------

def test_enhancement_flags_are_passed(fake_cli, tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"),
                enhancements=("upscale", "sharpen"))
    assert "--upscale" in cmds[0] and "--sharpen" in cmds[0]
    assert "--noise" not in cmds[0]


def test_unknown_enhancement_rejected(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(U.subprocess, "run", _runner())
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown enhancement"):
        U.run_topaz(str(src), str(tmp_path / "out"), enhancements=("bogus",))


def test_scale_is_not_passed_as_a_cli_flag(fake_cli, tmp_path, monkeypatch):
    """--upscale scale=N is broken upstream; the scale must never reach argv."""
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    monkeypatch.setattr(U, "is_wsl", lambda: False)
    monkeypatch.setattr(os, "name", "posix")
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"), scale=4)
    assert not any("scale=" in str(part) for part in cmds[0])


def test_missing_cli_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTOSEARCH_TOPAZ_CLI", str(tmp_path / "nope.exe"))
    with pytest.raises(U.TopazError, match="missing file"):
        U.cli_path()


# ---------------------------------------------------------------------------
# Scale override — global machine state, must always be restored
# ---------------------------------------------------------------------------

def test_scale_override_sets_and_restores(monkeypatch):
    reads = {U._REG_TYPE: "auto", U._REG_FACTOR: "2"}
    writes = []
    monkeypatch.setattr(U, "is_wsl", lambda: True)
    monkeypatch.setattr(U, "_reg_read", lambda n: reads.get(n))
    monkeypatch.setattr(U, "_reg_write",
                        lambda n, v: (writes.append((n, v)), True)[1])

    with U._scale_override(4):
        assert (U._REG_TYPE, "scale") in writes
        assert (U._REG_FACTOR, "4") in writes

    # Originals restored afterward.
    assert writes[-2:] == [(U._REG_TYPE, "auto"), (U._REG_FACTOR, "2")] or \
           set(writes[-2:]) == {(U._REG_TYPE, "auto"), (U._REG_FACTOR, "2")}
    assert not U._REG_LOCK.locked()


def test_scale_override_restores_on_exception(monkeypatch):
    reads = {U._REG_TYPE: "auto", U._REG_FACTOR: "2"}
    writes = []
    monkeypatch.setattr(U, "is_wsl", lambda: True)
    monkeypatch.setattr(U, "_reg_read", lambda n: reads.get(n))
    monkeypatch.setattr(U, "_reg_write",
                        lambda n, v: (writes.append((n, v)), True)[1])

    with pytest.raises(RuntimeError):
        with U._scale_override(4):
            raise RuntimeError("boom")

    assert set(writes[-2:]) == {(U._REG_TYPE, "auto"), (U._REG_FACTOR, "2")}
    assert not U._REG_LOCK.locked()


def test_scale_none_touches_nothing(monkeypatch):
    """Auto mode must leave the user's Topaz preferences completely alone."""
    writes = []
    monkeypatch.setattr(U, "is_wsl", lambda: True)
    monkeypatch.setattr(U, "_reg_write",
                        lambda n, v: (writes.append((n, v)), True)[1])
    with U._scale_override(None):
        pass
    assert writes == []
    assert not U._REG_LOCK.locked()


# ---------------------------------------------------------------------------
# End-to-end against a DB row
# ---------------------------------------------------------------------------

def _db_with_photo(tmp_path, filepath, dbpath=None):
    from photosearch.db import PhotoDB
    db = PhotoDB(str(dbpath or tmp_path / "t.db"))
    db.set_photo_root(str(tmp_path))
    pid = db.add_photo(filepath=filepath, filename=os.path.basename(filepath),
                       date_taken="2024-07-04T10:00:00")
    return db, pid


def test_upscale_photo_writes_to_export_tree(fake_cli, tmp_path, monkeypatch):
    """Local-original path: output lands in the export tree, library untouched."""
    orig = tmp_path / "lib" / "IMG_1.jpg"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")

    db, pid = _db_with_photo(tmp_path, str(orig))
    export = tmp_path / "exports"
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    monkeypatch.setattr(U.subprocess, "run", _runner(produce="IMG_1.jpg"))
    # No NAS configured -> read the original straight off disk.
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)

    try:
        res = U.upscale_photo(db, pid)
    finally:
        db.close()

    # Dated into YYYY/, suffixed, and outside the library.
    assert res["skipped"] is False
    out = Path(res["output"])
    assert out.exists() and out.name == "IMG_1-topaz-upscale.jpg"
    assert out.parent == export / "2024"
    # Original is untouched and no stray files were left in the library.
    assert orig.read_bytes() == b"original"
    assert list(orig.parent.iterdir()) == [orig]


def test_result_carries_a_windows_path(fake_cli, tmp_path, monkeypatch):
    """On WSL the POSIX path is useless to the user; a Windows path must ride
    along so the UI and CLI can show something pasteable."""
    orig = tmp_path / "lib" / "IMG_W.jpg"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(U.subprocess, "run", _runner(produce="IMG_W.jpg"))
    # Stub only the translation itself — leaving is_wsl alone keeps
    # to_native_path a no-op so the fake runner still sees POSIX paths.
    monkeypatch.setattr(U, "to_windows_path",
                        lambda p: "C:\\Users\\mattn\\Pictures\\IMG_W-topaz.jpg")
    try:
        res = U.upscale_photo(db, pid)
    finally:
        db.close()
    assert res["windows_path"] == "C:\\Users\\mattn\\Pictures\\IMG_W-topaz.jpg"
    assert res["file_url"] == "file:///C:/Users/mattn/Pictures/IMG_W-topaz.jpg"


@pytest.mark.parametrize("win,url", [
    ("C:\\a\\b.jpg", "file:///C:/a/b.jpg"),
    ("\\\\wsl.localhost\\Ubuntu\\home\\m\\b.jpg",
     "file:////wsl.localhost/Ubuntu/home/m/b.jpg"),
])
def test_file_url_conversion(win, url):
    assert U.to_file_url(win) == url


def test_windows_path_is_none_off_windows(monkeypatch):
    """A plain Linux/Mac host has no Windows view — callers fall back to POSIX."""
    monkeypatch.setattr(U, "is_wsl", lambda: False)
    monkeypatch.setattr(os, "name", "posix")
    assert U.to_windows_path("/home/m/x.jpg") is None


@pytest.mark.parametrize("scale,enh,expected", [
    (None, ("upscale",), "-upscale"),
    (None, ("upscale", "lighting"), "-upscale-lighting"),
    # Canonical order, not click order — the name must not depend on the UI.
    (None, ("lighting", "upscale"), "-upscale-lighting"),
    (2, ("upscale",), "-upscale-2x"),
    (4, ("upscale", "noise", "color"), "-upscale-noise-color-4x"),
])
def test_variant_suffix_encodes_options(scale, enh, expected):
    assert U._variant_suffix(scale, enh) == expected


@pytest.mark.parametrize("name,scale,enh", [
    ("DSC1-topaz-upscale.JPG", "auto", ["upscale"]),
    ("DSC1-topaz-upscale-lighting.JPG", "auto", ["upscale", "lighting"]),
    ("DSC1-topaz-upscale-2x.JPG", 2, ["upscale"]),
])
def test_parse_variant_round_trips(name, scale, enh):
    got = U._parse_variant(name, "DSC1")
    assert got["scale"] == scale and got["enhancements"] == enh


def test_different_options_do_not_collide(fake_cli, tmp_path, monkeypatch):
    """The bug this prevents: one option set shadowing every later one, so
    re-running with different settings silently returns the first file."""
    orig = tmp_path / "lib" / "DSC1.JPG"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(U.subprocess, "run", _runner(produce="DSC1.JPG"))
    try:
        a = U.upscale_photo(db, pid, enhancements=("upscale",))
        b = U.upscale_photo(db, pid, enhancements=("upscale", "lighting"))
        c = U.upscale_photo(db, pid, scale=2, enhancements=("upscale",))
        # Same options again -> genuinely skipped, that caching still works.
        again = U.upscale_photo(db, pid, enhancements=("upscale",))
        listed = U.list_exports(db, pid)
    finally:
        db.close()

    names = [os.path.basename(r["output"]) for r in (a, b, c)]
    assert names == ["DSC1-topaz-upscale.JPG",
                     "DSC1-topaz-upscale-lighting.JPG",
                     "DSC1-topaz-upscale-2x.JPG"]
    assert not any(r["skipped"] for r in (a, b, c))
    assert again["skipped"] and again["output"] == a["output"]
    assert len(listed) == 3
    assert {x["label"] for x in listed} == {
        "upscale", "upscale, lighting", "upscale · 2x"}


def test_list_exports_empty_for_untouched_photo(tmp_path, monkeypatch):
    orig = tmp_path / "lib" / "DSC9.JPG"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"x")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    try:
        assert U.list_exports(db, pid) == []
    finally:
        db.close()


def test_model_is_passed_as_an_upscale_param(fake_cli, tmp_path, monkeypatch):
    """Unlike scale, the CLI serializes model= correctly, so it goes in argv."""
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"), model="Standard V2")
    assert "--upscale" in cmds[0]
    assert "model=Standard V2" in cmds[0]


def test_unknown_model_rejected(fake_cli, tmp_path, monkeypatch):
    monkeypatch.setattr(U.subprocess, "run", _runner())
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    with pytest.raises(ValueError, match="unknown upscale model"):
        U.run_topaz(str(src), str(tmp_path / "out"), model="Nonsense")


def test_override_disables_every_unrequested_enhancement(fake_cli, tmp_path,
                                                        monkeypatch):
    """The artifact fix. `--override` ALONE does not stop Autopilot's Sharpen
    Strong — measured against the real CLI, it stays enabled=true. Only an
    explicit `--sharpen enabled=false` brings it down, so override must emit
    one for every enhancement not asked for."""
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"),
                enhancements=("upscale",), override=True)
    cmd = cmds[0]

    assert "--upscale" in cmd
    # Every other enhancement is explicitly switched off.
    for enh in ("noise", "sharpen", "lighting", "color"):
        i = cmd.index(f"--{enh}")
        assert cmd[i + 1] == "enabled=false", f"{enh} not disabled"
    # ...and the one we asked for is NOT.
    assert "enabled=false" not in cmd[cmd.index("--upscale") + 1:
                                     cmd.index("--upscale") + 2]


def test_override_keeps_the_enhancements_you_asked_for(fake_cli, tmp_path,
                                                       monkeypatch):
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"),
                enhancements=("upscale", "sharpen"), override=True)
    cmd = cmds[0]
    # sharpen was requested, so it must not be turned off.
    assert cmd[cmd.index("--sharpen") + 1] != "enabled=false"
    assert cmd[cmd.index("--noise") + 1] == "enabled=false"


def test_without_override_nothing_is_disabled(fake_cli, tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(U.subprocess, "run", _runner(record=cmds))
    src = tmp_path / "in.jpg"; src.write_bytes(b"x")
    U.run_topaz(str(src), str(tmp_path / "out"), override=False)
    assert "--override" not in cmds[0]
    assert "enabled=false" not in cmds[0]


def test_model_and_override_are_part_of_the_variant(fake_cli, tmp_path, monkeypatch):
    """Two models must not collide on one filename, same as two option sets."""
    orig = tmp_path / "lib" / "DSC2.JPG"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(U.subprocess, "run", _runner(produce="DSC2.JPG"))
    try:
        a = U.upscale_photo(db, pid, model="Standard")
        b = U.upscale_photo(db, pid, model="High Fidelity V2")
        c = U.upscale_photo(db, pid, model="Standard", override=True)
        listed = U.list_exports(db, pid)
    finally:
        db.close()

    names = [os.path.basename(r["output"]) for r in (a, b, c)]
    assert names == ["DSC2-topaz-upscale-standard.JPG",
                     "DSC2-topaz-upscale-highfidelityv2.JPG",
                     "DSC2-topaz-upscale-standard-only.JPG"]
    assert len(listed) == 3
    # The model round-trips out of the filename for the prior-exports list.
    by_name = {os.path.basename(x["output"]): x for x in listed}
    assert by_name[names[0]]["model"] == "Standard"
    assert by_name[names[1]]["model"] == "High Fidelity V2"
    assert by_name[names[2]]["override"] is True
    assert "no autopilot" in by_name[names[2]]["label"]


def test_blend_strength_is_monotonic(tmp_path):
    """The dial must actually span plain-resize to full-Topaz. Topaz has no
    working strength control, so this is the only one there is."""
    from PIL import Image
    import numpy as np

    src = tmp_path / "src.jpg"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(src)
    topaz = tmp_path / "topaz.jpg"
    Image.new("RGB", (128, 128), (255, 255, 255)).save(topaz)

    means = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        out = tmp_path / f"b{int(s * 100)}.jpg"
        U.blend_strength(str(topaz), str(src), s, str(out))
        with Image.open(out) as im:
            assert im.size == (128, 128)      # keeps the Topaz dimensions
            means.append(float(np.asarray(im.convert("L")).mean()))

    assert means[0] < 5 and means[-1] > 250   # endpoints are the two sources
    assert all(a < b for a, b in zip(means, means[1:]))   # monotonic


def test_interfaces_default_to_quarter_strength(tmp_path, monkeypatch):
    """25% is the chosen default at every surface a person touches. The library
    primitive stays at 1.0 so `upscale_photo` does not blend behind your back."""
    from photosearch.admin_api import UpscaleRequest
    import inspect
    import cli as cli_mod

    assert UpscaleRequest(photo_ids=[1]).strength == 0.25
    opt = next(p for p in cli_mod.upscale_cmd.params if p.name == "strength")
    assert opt.default == 0.25
    assert inspect.signature(U.upscale_photo).parameters["strength"].default == 1.0


def test_blend_strength_rejects_out_of_range(tmp_path):
    from PIL import Image
    src = tmp_path / "s.jpg"; Image.new("RGB", (8, 8)).save(src)
    with pytest.raises(ValueError, match=r"strength must be in \[0, 1\]"):
        U.blend_strength(str(src), str(src), 1.5, str(tmp_path / "o.jpg"))


def test_strength_is_part_of_the_variant(fake_cli, tmp_path, monkeypatch):
    orig = tmp_path / "lib" / "DSC3.JPG"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(tmp_path / "exports"))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(U.subprocess, "run", _runner(produce="DSC3.JPG"))
    # Full strength short-circuits the blend, so no real images are needed.
    monkeypatch.setattr(U, "blend_strength",
                        lambda t, s, st, o, **k: (Path(o).write_bytes(b"x"), o)[1])
    try:
        full = U.upscale_photo(db, pid)
        half = U.upscale_photo(db, pid, strength=0.5)
        listed = U.list_exports(db, pid)
    finally:
        db.close()

    assert os.path.basename(full["output"]) == "DSC3-topaz-upscale.JPG"
    assert os.path.basename(half["output"]) == "DSC3-topaz-upscale-s50.JPG"
    by = {os.path.basename(x["output"]): x for x in listed}
    assert by["DSC3-topaz-upscale-s50.JPG"]["strength"] == 0.5
    assert "50% strength" in by["DSC3-topaz-upscale-s50.JPG"]["label"]
    assert by["DSC3-topaz-upscale.JPG"]["strength"] == 1.0


def test_upscale_photo_skips_existing(fake_cli, tmp_path, monkeypatch):
    orig = tmp_path / "lib" / "IMG_2.jpg"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    export = tmp_path / "exports"
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    (export / "2024").mkdir(parents=True)
    (export / "2024" / "IMG_2-topaz-upscale.jpg").write_bytes(b"already")

    called = []
    monkeypatch.setattr(U.subprocess, "run",
                        lambda *a, **k: called.append(a) or _Proc(9))
    try:
        res = U.upscale_photo(db, pid)
    finally:
        db.close()
    assert res["skipped"] is True
    assert called == []  # Topaz was never invoked


def test_upscaled_list_route_returns_exports(tmp_path, monkeypatch):
    """Exercises the route, not just list_exports() — the first cut of this
    handler raised NameError on a missing import that a unit test never saw."""
    from fastapi.testclient import TestClient
    from photosearch import web

    orig = tmp_path / "lib" / "DSC5.JPG"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"x")
    dbpath = tmp_path / "r.db"
    db, pid = _db_with_photo(tmp_path, str(orig), dbpath=dbpath)
    db.close()

    export = tmp_path / "exports"
    (export / "2024").mkdir(parents=True)
    (export / "2024" / "DSC5-topaz-upscale-2x.JPG").write_bytes(b"\xff\xd8a")
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    # web reads PHOTOSEARCH_DB once at import, so patch the global.
    monkeypatch.setattr(web, "_db_path", str(dbpath))

    c = TestClient(web.app)
    r = c.get("/api/admin/upscaled-list", params={"photo_id": pid})
    assert r.status_code == 200
    exports = r.json()["exports"]
    assert len(exports) == 1
    assert exports[0]["label"] == "upscale · 2x" and exports[0]["scale"] == 2

    assert c.get("/api/admin/upscaled-list",
                 params={"photo_id": 999999}).status_code == 404


def test_upscaled_file_route_confines_to_export_dir(tmp_path, monkeypatch):
    """The view link serves real files, so it must not become a file-read hole."""
    from fastapi.testclient import TestClient
    from photosearch import web

    export = tmp_path / "exports"
    (export / "2024").mkdir(parents=True)
    good = export / "2024" / "ok.jpg"
    good.write_bytes(b"\xff\xd8image")
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")

    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    monkeypatch.setenv("PHOTOSEARCH_DB", str(tmp_path / "r.db"))
    from photosearch.db import PhotoDB
    PhotoDB(str(tmp_path / "r.db")).close()
    c = TestClient(web.app)

    r = c.get("/api/admin/upscaled-file", params={"path": str(good)})
    assert r.status_code == 200 and r.content == b"\xff\xd8image"

    # Outside the export root -> refused, by absolute path and by traversal.
    for bad in (str(secret), str(export / ".." / "secret.txt")):
        assert c.get("/api/admin/upscaled-file", params={"path": bad}).status_code == 403

    # Inside the root but absent -> 404, not 403.
    assert c.get("/api/admin/upscaled-file",
                 params={"path": str(export / "nope.jpg")}).status_code == 404


def test_staging_dir_is_cleaned_up_on_failure(fake_cli, tmp_path, monkeypatch):
    orig = tmp_path / "lib" / "IMG_3.jpg"
    orig.parent.mkdir(parents=True)
    orig.write_bytes(b"original")
    db, pid = _db_with_photo(tmp_path, str(orig))
    export = tmp_path / "exports"
    monkeypatch.setenv("PHOTOSEARCH_UPSCALE_DIR", str(export))
    monkeypatch.delenv("PHOTOSEARCH_NAS_URL", raising=False)
    monkeypatch.setattr(U.subprocess, "run", _runner(returncode=255, produce=None))

    try:
        with pytest.raises(U.TopazError):
            U.upscale_photo(db, pid)
    finally:
        db.close()

    leftovers = [p for p in (export / "2024").iterdir()] if (export / "2024").exists() else []
    assert leftovers == []

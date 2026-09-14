"""Guard: no page may call a function it doesn't have.

The frontend has no build step, no bundler and no linter, so a helper
referenced from the wrong page is a ReferenceError that only appears when a
human clicks the button. A syntax check does NOT catch it — `new Function(src)`
happily accepts a call to an undefined name, which is exactly how a bare
`parseSSEChunk` (defined only in admin_maintenance.html) shipped in faces.html.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check-frontend-refs.js"


pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available")


def _run(cwd=ROOT):
    return subprocess.run(["node", str(SCRIPT)], cwd=cwd,
                          capture_output=True, text=True, timeout=120)


def test_no_page_calls_an_undefined_function():
    r = _run()
    assert r.returncode == 0, (
        "a page calls a function it does not define:\n" + r.stdout + r.stderr)


def test_the_check_actually_catches_a_bare_shared_helper(tmp_path):
    """The check must fail on the bug it exists for.

    Worth pinning: the first version of this script PASSED that bug. shared.js
    is an IIFE, and treating its inner named function expressions as globals
    made an unreachable name look defined.
    """
    work = tmp_path / "repo"
    shutil.copytree(ROOT / "frontend", work / "frontend")
    (work / "scripts").mkdir()
    shutil.copy(SCRIPT, work / "scripts" / SCRIPT.name)

    page = work / "frontend" / "dist" / "faces.html"
    src = page.read_text()
    assert "PS.parseSSEChunk(raw)" in src, "fixture drifted; update this test"
    page.write_text(src.replace("PS.parseSSEChunk(raw)", "parseSSEChunk(raw)"))

    r = subprocess.run(["node", str(work / "scripts" / SCRIPT.name)], cwd=work,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "parseSSEChunk" in (r.stdout + r.stderr)


def test_the_check_catches_a_typoed_PS_member(tmp_path):
    work = tmp_path / "repo"
    shutil.copytree(ROOT / "frontend", work / "frontend")
    (work / "scripts").mkdir()
    shutil.copy(SCRIPT, work / "scripts" / SCRIPT.name)

    page = work / "frontend" / "dist" / "faces.html"
    page.write_text(page.read_text().replace(
        "PS.parseSSEChunk(raw)", "PS.parseSSEChunkNope(raw)"))

    r = subprocess.run(["node", str(work / "scripts" / SCRIPT.name)], cwd=work,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "parseSSEChunkNope" in (r.stdout + r.stderr)

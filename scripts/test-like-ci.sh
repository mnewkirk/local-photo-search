#!/usr/bin/env bash
# Run the test suite the way GitHub Actions does, so a green local run means a
# green CI run.
#
# Local runs diverge from CI in three ways, and each has hidden a real failure:
#
#   1. **No Topaz CLI.** A dev box with Topaz installed makes
#      /api/admin/upscale-photo return 200; CI has no tpai.exe, so it returns
#      503. A test asserting 200 passed locally and failed on CI for five
#      commits.
#   2. **Minimal dependencies.** CI installs no torch/cv2/insightface —
#      tests/conftest.py mocks them. A dev venv has the real thing, so code
#      paths differ.
#   3. **Nothing ignored.** It is tempting to --ignore a file that fails for
#      local-environment reasons (a git worktree whose gitdir is a Windows UNC
#      path breaks tests/test_admin_deploy_mode.py). CI ignores nothing.
#
# Usage:  ./scripts/test-like-ci.sh [pytest args...]
#         ./scripts/test-like-ci.sh --rebuild      # recreate the venv
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="${PHOTOSEARCH_CI_VENV:-/tmp/photosearch-ci-venv}"
PYBIN="${PHOTOSEARCH_CI_PYTHON:-python3}"

if [ "${1:-}" = "--rebuild" ]; then rm -rf "$VENV"; shift; fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "→ creating CI-equivalent venv at $VENV"
  "$PYBIN" -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  # Keep this list in step with .github/workflows/*.yml.
  "$VENV/bin/pip" install -q \
    "pytest>=8.0" "pytest-cov>=5.0" \
    "click>=8.0" "numpy>=1.20" "pyyaml>=6.0" \
    "Pillow>=10.0" "exifread>=3.0" \
    "sqlite-vec>=0.1" "scikit-learn>=1.3" "scipy>=1.10" \
    "colorthief>=0.2.1" "reverse_geocoder>=1.5" \
    "fastapi>=0.110" httpx "requests>=2.31"
fi

echo "→ $("$VENV/bin/python" -V), Topaz CLI made unavailable, nothing ignored"

# tests/test_admin_deploy_mode.py reads HEAD through git. In a git worktree
# created from Windows, .git points at a \\wsl.localhost UNC gitdir that git
# cannot resolve from inside WSL, so those tests fail here and pass on CI. Say
# so up front rather than letting it look like a real regression.
if ! git rev-parse HEAD >/dev/null 2>&1; then
  echo "  ! git cannot read HEAD from this checkout, so"
  echo "    tests/test_admin_deploy_mode.py will fail locally (it passes on CI)."
  echo "    Run from the main checkout for a clean result."
fi
# Point at a path that cannot exist so cli_path() fails exactly as it does on a
# CI runner. Tests that need it must use the `fake_cli` fixture.
export PHOTOSEARCH_TOPAZ_CLI=/nonexistent/tpai.exe

exec "$VENV/bin/python" -m pytest tests/ "$@"

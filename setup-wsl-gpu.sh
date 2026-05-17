#!/usr/bin/env bash
# setup-wsl-gpu.sh — make the WSL2 bare-metal worker GPU-ready (or re-establish
# what was lost when the venv was recreated).
#
# Idempotent — safe to re-run. Doesn't touch the system ROCm install; only the
# project venv + a couple of compat symlinks. Run from the repo root.
#
# What it does:
#   1. Confirm we're on WSL2 with a ROCm GPU (rocminfo + /dev/dxg).
#   2. Find/create the project venv at .venv (uses python3 -m venv).
#   3. Install pillow-heif into the venv if missing (HEIC support in describe).
#   4. Symlink librocm_smi64.so.7 -> system .so.1 inside
#      .venv/lib/python3.12/site-packages/onnxruntime_rocm.libs/ if onnxruntime-
#      rocm is present and the symlink is missing (AMD's wheel asks for .so.7,
#      ROCm 7.2 ships .so.1; ABI is compatible).
#   5. Confirm torch reports CUDA available (PyTorch-ROCm masquerades as cuda).
#   6. Print the env vars the worker needs to set per-invocation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
ok()    { printf "  \033[32m✓\033[0m %s\n" "$*"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$*"; }
fail()  { printf "  \033[31m✗\033[0m %s\n" "$*"; exit 1; }

bold "1) Environment"
if grep -qi microsoft /proc/version 2>/dev/null; then
    ok "WSL2 detected"
else
    warn "not WSL2 — this script is WSL2-specific; bailing"
    exit 0
fi

if [ -e /dev/dxg ]; then
    ok "/dev/dxg present (WSL2 GPU passthrough active)"
else
    warn "/dev/dxg not present — GPU passthrough not active"
    warn "Install AMD's Adrenalin driver on Windows, then 'wsl --shutdown' and reopen."
fi

if command -v rocminfo >/dev/null 2>&1; then
    # Capture rocminfo into a variable first — piping `rocminfo | awk 'exit'`
    # triggers SIGPIPE under `set -o pipefail` when awk exits early.
    ROCMINFO_OUT=$(HSA_ENABLE_DXG_DETECTION=1 rocminfo 2>/dev/null || true)
    if echo "$ROCMINFO_OUT" | grep -q "Device Type: *GPU"; then
        # Second "Marketing Name" is the GPU (first is the CPU).
        GPU_NAME=$(echo "$ROCMINFO_OUT" | awk -F': *' '/Marketing Name/{print $2}' | sed -n '2p')
        ok "ROCm sees: ${GPU_NAME:-(GPU agent, name not parsed)}"
    else
        warn "rocminfo present but no GPU found — check librocdxg install (see CLAUDE.md / SKILL.md)"
    fi
else
    warn "rocminfo not installed — install ROCm via amdgpu-install --usecase=rocm --no-dkms"
fi

bold "2) Project venv"
if [ ! -d .venv ]; then
    bold "  creating .venv (python3 -m venv .venv)…"
    python3 -m venv .venv
fi
ok ".venv exists ($(.venv/bin/python --version 2>&1))"

bold "3) pillow-heif (HEIC support for describe)"
if .venv/bin/python -c "import pillow_heif" >/dev/null 2>&1; then
    ok "pillow-heif installed"
else
    bold "  installing pillow-heif…"
    .venv/bin/pip install -q "pillow-heif>=0.18"
    ok "pillow-heif installed"
fi

bold "4) onnxruntime-rocm symlink"
ORT_DIR=$(.venv/bin/python -c "
import os
try:
    import onnxruntime, pathlib
    p = pathlib.Path(onnxruntime.__file__).parent.parent / 'onnxruntime_rocm.libs'
    print(p if p.exists() else '')
except Exception:
    pass
" 2>/dev/null)

if [ -z "$ORT_DIR" ]; then
    warn "onnxruntime_rocm.libs/ not found — onnxruntime-rocm not installed (faces pass would stay on CPU)"
    warn "Install with: pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/onnxruntime_rocm-1.22.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
elif [ -e "$ORT_DIR/librocm_smi64.so.7" ]; then
    ok "librocm_smi64.so.7 symlink already in $ORT_DIR"
else
    SYS_SMI=$(ls /opt/rocm-*/lib/librocm_smi64.so.1 2>/dev/null | head -1)
    if [ -z "$SYS_SMI" ]; then
        warn "system librocm_smi64.so.1 not found in /opt/rocm-*/lib/ — skipping symlink"
    else
        ln -sf "$SYS_SMI" "$ORT_DIR/librocm_smi64.so.7"
        ok "symlinked $ORT_DIR/librocm_smi64.so.7 -> $SYS_SMI"
    fi
fi

bold "5) PyTorch GPU detection"
TORCH_INFO=$(HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python - <<'PYEOF' 2>/dev/null
try:
    import torch
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
except Exception as e:
    print(f"ERR: {e}")
PYEOF
)
if echo "$TORCH_INFO" | grep -q "cuda_available=True"; then
    ok "$TORCH_INFO"
else
    warn "$TORCH_INFO"
    warn "If torch is +cpu, reinstall with the ROCm wheel — see CLAUDE.md / SKILL.md WSL2 section."
fi

bold "6) Worker environment"
GW=$(ip route show default 2>/dev/null | awk '{print $3}')
echo "  Per-invocation env (also set by ./run-workers.sh native mode):"
echo "    HSA_ENABLE_DXG_DETECTION=1"
echo "    OLLAMA_HOST=http://${GW:-<windows-host>}:11434"
echo
echo "  Easiest launcher (auto-picks native on WSL2):"
echo "    ./run-workers.sh -s http://<NAS-IP>:8000 -p clip,faces,quality -d /photos/<year>"

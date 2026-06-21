#!/usr/bin/env bash
# run-local-replica.sh — run the photosearch web app + Ask agent locally off a
# synced read-replica of the NAS DB (M26a). Search compute runs on THIS machine
# (local GPU CLIP); the NAS stays the source of truth. Image routes proxy
# thumbnails from the NAS on demand, so no photo files are needed locally.
#
# Cross-platform: on WSL2 the local LM Studio runs on the Windows host (reached
# via the default-route gateway); on Mac/Linux it's localhost.
#
# Usage:
#   ./run-local-replica.sh [--sync] [-p PORT] [--db PATH] [--nas URL]
#                          [--lm URL] [--model NAME]
#
#   --sync          pull a fresh replica (sync-replica.sh) before starting
#   -p, --port      web port                 (default: 8001)
#   --db            replica DB path          (default: ./photo_index.db.local, $PHOTOSEARCH_DB)
#   --nas           NAS web URL              (default: http://dxp4800-f976:8000, $PHOTOSEARCH_NAS_URL)
#   --lm            LM Studio /v1 URL        (default: auto — WSL2 gateway / localhost, $PHOTOSEARCH_TEXT_LLM_URL)
#   --model         agent model id loaded in LM Studio ($PHOTOSEARCH_LLM_AGENT_MODEL)
#
# The --model must be a tool-calling-capable model loaded in LM Studio
# (qwen2.5-instruct / qwen3 / llama-3.1+ with tool use on), or the ✨ Ask
# agent falls back to single-shot mode.

set -euo pipefail
cd "$(dirname "$0")"

DO_SYNC=0
PORT="${PORT:-8001}"
REPLICA_DB="${PHOTOSEARCH_DB:-./photo_index.db.local}"
NAS_URL="${PHOTOSEARCH_NAS_URL:-http://dxp4800-f976:8000}"
LM_URL="${PHOTOSEARCH_TEXT_LLM_URL:-}"
AGENT_MODEL="${PHOTOSEARCH_LLM_AGENT_MODEL:-}"

while [ $# -gt 0 ]; do
  case "$1" in
    --sync) DO_SYNC=1; shift ;;
    -p|--port) PORT="$2"; shift 2 ;;
    --db) REPLICA_DB="$2"; shift 2 ;;
    --nas) NAS_URL="$2"; shift 2 ;;
    --lm) LM_URL="$2"; shift 2 ;;
    --model) AGENT_MODEL="$2"; shift 2 ;;
    --reload) RELOAD=1; shift ;;   # uvicorn auto-reload on code edits (dev)
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Auto-resolve the LM Studio endpoint if not given.
if [ -z "${LM_URL}" ]; then
  if grep -qi microsoft /proc/version 2>/dev/null; then
    # WSL2: try localhost first — works with mirrored networking (and is
    # cleaner). Fall back to the default-route gateway (NAT mode → Windows host).
    if curl -s --max-time 2 -o /dev/null "http://localhost:1234/v1/models" 2>/dev/null; then
      LM_URL="http://localhost:1234/v1"
    else
      GW="$(ip route show default 2>/dev/null | awk '{print $3}')"
      LM_URL="http://${GW:-host.docker.internal}:1234/v1"
    fi
  else
    LM_URL="http://localhost:1234/v1"                      # Mac / native Linux
  fi
fi

PYBIN="./.venv/bin/python"
[ -x "${PYBIN}" ] || PYBIN="$(command -v python3 || command -v python)"

if [ "${DO_SYNC}" = "1" ]; then
  echo "→ syncing replica first…"
  PHOTOSEARCH_DB="${REPLICA_DB}" NAS_HOST="${NAS_HOST:-cantimatt@192.168.1.237}" ./sync-replica.sh
fi

if [ ! -f "${REPLICA_DB}" ]; then
  echo "ERROR: replica DB not found at ${REPLICA_DB} — run with --sync first." >&2
  exit 1
fi

# Persistent household glossary for the Ask agent: if PHOTOSEARCH_AGENT_HINTS
# isn't already set and ./.agent-hints exists, load it (gitignored, personal).
if [ -z "${PHOTOSEARCH_AGENT_HINTS:-}" ] && [ -f "./.agent-hints" ]; then
  PHOTOSEARCH_AGENT_HINTS="$(cat ./.agent-hints)"
fi

export PHOTOSEARCH_DB="${REPLICA_DB}"
export PHOTOSEARCH_NAS_URL="${NAS_URL}"
export PHOTOSEARCH_TEXT_LLM_URL="${LM_URL}"
[ -n "${AGENT_MODEL}" ] && export PHOTOSEARCH_LLM_AGENT_MODEL="${AGENT_MODEL}"
# Vision model for rerank_photos (VLM re-ranking). Override with VISUAL_MODEL=.
export PHOTOSEARCH_LLM_VISUAL_MODEL="${PHOTOSEARCH_LLM_VISUAL_MODEL:-${VISUAL_MODEL:-qwen2.5-vl-7b-instruct}}"
[ -n "${PHOTOSEARCH_AGENT_HINTS:-}" ] && export PHOTOSEARCH_AGENT_HINTS
# No PHOTO_ROOT: originals aren't local, so image routes proxy from the NAS.

echo "photosearch (local replica)"
echo "  DB:        ${REPLICA_DB}"
echo "  NAS:       ${NAS_URL}   (image proxy + sync)"
echo "  LM Studio: ${LM_URL}   (Ask agent${AGENT_MODEL:+, model=$AGENT_MODEL})"
echo "  Hints:     ${PHOTOSEARCH_AGENT_HINTS:+loaded (${#PHOTOSEARCH_AGENT_HINTS} chars)}${PHOTOSEARCH_AGENT_HINTS:-none}"
echo "  Web:       http://localhost:${PORT}    (✨ Ask mode in the search bar)"
echo
exec "${PYBIN}" cli.py serve --db "${REPLICA_DB}" --host 0.0.0.0 --port "${PORT}" \
  ${RELOAD:+--reload}

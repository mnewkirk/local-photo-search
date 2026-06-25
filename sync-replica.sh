#!/usr/bin/env bash
# sync-replica.sh — pull a fresh read-replica of the NAS DB to this machine.
#
# Part of M26a (docs/plans/local-replica-and-writes.md): run the photosearch
# web app / agent / MCP on a strong local machine off a synced read-replica of
# the NAS's SQLite DB, with search compute on the local GPU. The NAS stays the
# source of truth and sole writer.
#
# Uses the same dump-db + cat-stream trick as debug-db.sh: UGREEN blocks direct
# rsync into the docker volume at /data, so we make a *consistent* snapshot
# inside a transient container (sqlite backup API) and stream it out. Then we
# atomically swap it into place so a running `serve` picks it up on its next
# per-request connection (in-flight requests finish on the old inode).
#
# Photo thumbnails/previews are NOT mirrored here — replica image routes lazily
# proxy + cache them from the NAS web API on demand (see web.py _fetch_from_nas).
#
# Face crops ARE mirrored (set SYNC_FACE_CROPS=0 to skip): the per-face grid on
# /faces issues one cold crop round-trip per face, so a large person (10k+ faces)
# crawls on first browse. Crops are immutable per face_id and append-only, so the
# tar delta is tiny after the first pull. Generate them on the NAS first with
# `photosearch warm-face-crops` so there's something to mirror.
#
# Schedule nightly (cron / Task Scheduler) and/or trigger on demand via the
# /status "Sync replica" button (POST /api/admin/replica-sync).
#
# Env overrides:
#   NAS_HOST           ssh target               (default: cantimatt@192.168.1.237)
#   NAS_COMPOSE_FILE   compose file on the NAS  (default: /volume1/docker/photosearch/docker-compose.nas.yml)
#   PHOTOSEARCH_DB     local replica path       (default: ./photo_index.db.local)

set -euo pipefail

NAS_HOST="${NAS_HOST:-cantimatt@192.168.1.237}"
NAS_COMPOSE_FILE="${NAS_COMPOSE_FILE:-/volume1/docker/photosearch/docker-compose.nas.yml}"
TARGET="${PHOTOSEARCH_DB:-./photo_index.db.local}"
REMOTE_DUMP="/data/replica-dump.db"
remote="docker compose -f '${NAS_COMPOSE_FILE}'"

started=$(date +%s)
echo "[sync-replica] target: ${TARGET}"

echo "[1/4] creating consistent snapshot on NAS (sqlite backup API)…"
ssh "${NAS_HOST}" "${remote} run --rm photosearch dump-db --to ${REMOTE_DUMP}"

echo "[2/4] streaming snapshot → ${TARGET}.tmp …"
# -T: no TTY so binary bytes aren't mangled.
ssh "${NAS_HOST}" "${remote} run --rm -T --entrypoint cat photosearch ${REMOTE_DUMP}" \
  > "${TARGET}.tmp"

# Sanity: a real DB is well over 4 KB; a truncated stream would corrupt search.
size=$(wc -c < "${TARGET}.tmp" 2>/dev/null || echo 0)
if [ "${size}" -lt 4096 ]; then
  echo "[sync-replica] ERROR: streamed file is only ${size} bytes — aborting (not swapping)." >&2
  rm -f "${TARGET}.tmp"
  exit 1
fi

echo "[3/4] atomic swap into place…"
# Clear any stale WAL/SHM sidecars from a previous live copy — the streamed
# snapshot is a self-contained single file.
rm -f "${TARGET}-wal" "${TARGET}-shm"
mv "${TARGET}.tmp" "${TARGET}"

echo "[4/5] cleaning up snapshot on NAS…"
ssh "${NAS_HOST}" "${remote} run --rm --entrypoint rm photosearch ${REMOTE_DUMP}" \
  || echo "[sync-replica]   (remote cleanup failed; next sync overwrites it)"

# Mirror face crops (skip with SYNC_FACE_CROPS=0). Same container tar/cat-stream
# trick as the DB above — UGREEN blocks rsync into the volume, and a named docker
# volume has no host path to rsync from. Incremental via mtime: we pass the last
# successful sync time as --since so only crops generated since then are tarred
# (append-only + immutable per face_id → tiny delta after the first pull).
if [ "${SYNC_FACE_CROPS:-1}" != "0" ]; then
  echo "[5/5] mirroring face crops…"
  CROP_DIR="$(dirname "${TARGET}")/thumbnails/face_crops"
  mkdir -p "${CROP_DIR}"
  MARKER="${CROP_DIR}/.last_sync"
  since=$(cat "${MARKER}" 2>/dev/null || echo 0)
  this_run=$(date +%s)
  REMOTE_TAR="/data/face-crops-delta.tar"
  crops_ok=1
  ssh "${NAS_HOST}" "${remote} run --rm -T --entrypoint python photosearch cli.py export-face-crops --since ${since} --to ${REMOTE_TAR}" \
    || crops_ok=0
  if [ "${crops_ok}" = "1" ]; then
    ssh "${NAS_HOST}" "${remote} run --rm -T --entrypoint cat photosearch ${REMOTE_TAR}" \
      | tar -xf - -C "${CROP_DIR}" || crops_ok=0
    ssh "${NAS_HOST}" "${remote} run --rm --entrypoint rm photosearch ${REMOTE_TAR}" \
      || echo "[sync-replica]   (remote crop-tar cleanup failed; next sync overwrites it)"
  fi
  if [ "${crops_ok}" = "1" ]; then
    echo "${this_run}" > "${MARKER}"   # only advance the watermark on full success
    total=$(find "${CROP_DIR}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
    echo "[sync-replica]   face crops: ${total} cached locally"
  else
    echo "[sync-replica]   face-crop mirror failed; watermark unchanged (retries next sync)." >&2
  fi
fi

human_size=$(du -h "${TARGET}" 2>/dev/null | cut -f1 || echo "?")
echo "[sync-replica] done in $(( $(date +%s) - started ))s — ${human_size} at ${TARGET}"

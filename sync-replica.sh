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
#   SNAPSHOT_DIR       rotated snapshot dir     (default: <target dir>/replica-snapshots)
#   KEEP_NIGHTLY       nightly snapshots kept   (default: 7)
#   KEEP_WEEKLY        weekly snapshots kept    (default: 4; Sunday's is promoted)
#   SNAPSHOTS=0        disable rotation entirely

set -euo pipefail

NAS_HOST="${NAS_HOST:-cantimatt@192.168.1.237}"
NAS_COMPOSE_FILE="${NAS_COMPOSE_FILE:-/volume1/docker/photosearch/docker-compose.nas.yml}"
TARGET="${PHOTOSEARCH_DB:-./photo_index.db.local}"
REMOTE_DUMP="/data/replica-dump.db"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-$(dirname "${TARGET}")/replica-snapshots}"
KEEP_NIGHTLY="${KEEP_NIGHTLY:-7}"
KEEP_WEEKLY="${KEEP_WEEKLY:-4}"
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

# Rotate BEFORE the swap. Until now this script `mv`'d straight over the only
# copy that existed, so an upstream corruption or a bad migration reached the
# last surviving replica within 24 h and there was no point-in-time recovery of
# anything — including the ~60k manual geotags and ~6.5k manual/merge face
# assignments that no re-run can regenerate.
#
# The snapshot is of the OUTGOING replica (the one about to be overwritten),
# because that is the copy that would otherwise be lost. Compressed with zstd
# -3: measured ~53% on this data, so ~1 GB each, ~11 GB for a full 7+4 set.
if [ "${SNAPSHOTS:-1}" != "0" ] && [ -f "${TARGET}" ]; then
  mkdir -p "${SNAPSHOT_DIR}"
  stamp=$(date +%Y%m%d)
  snap="${SNAPSHOT_DIR}/photo_index.${stamp}.db.zst"
  if [ ! -f "${snap}" ]; then      # one per day; a same-day re-run keeps the first
    echo "[3/5] snapshotting the outgoing replica → $(basename "${snap}")…"
    if zstd -3 -q -T0 -o "${snap}" "${TARGET}" 2>/dev/null; then
      # Sunday's is promoted to the weekly, which survives the nightly cull.
      # HARDLINK, not copy: same filesystem, so the weekly costs zero extra
      # bytes until the nightly is pruned out from under it, at which point the
      # link keeps the data alive. A cp would double a ~1 GB snapshot every
      # Sunday for nothing.
      if [ "$(date +%u)" = "7" ]; then
        ln -f "${snap}" "${SNAPSHOT_DIR}/weekly-${stamp}.db.zst" 2>/dev/null \
          || cp -f "${snap}" "${SNAPSHOT_DIR}/weekly-${stamp}.db.zst"
      fi
    else
      echo "[sync-replica]   (snapshot failed — continuing; the sync still proceeds)" >&2
      rm -f "${snap}"
    fi
  fi
  # Cull. Sort by the YYYYMMDD in the FILENAME (reverse = newest first), not by
  # mtime: `ls -t` gets it wrong the moment a snapshot is copied, restored or
  # touched, and silently keeps the oldest N instead of the newest. Nightly and
  # weekly are culled independently so a quiet fortnight can't age out every
  # weekly.
  ls -1 "${SNAPSHOT_DIR}"/photo_index.*.db.zst 2>/dev/null | sort -r | tail -n +$((KEEP_NIGHTLY + 1)) \
    | while read -r old; do echo "[sync-replica]   pruning $(basename "${old}")"; rm -f "${old}"; done
  ls -1 "${SNAPSHOT_DIR}"/weekly-*.db.zst 2>/dev/null | sort -r | tail -n +$((KEEP_WEEKLY + 1)) \
    | while read -r old; do echo "[sync-replica]   pruning $(basename "${old}")"; rm -f "${old}"; done
fi

echo "[4/6] atomic swap into place…"
# Clear any stale WAL/SHM sidecars from a previous live copy — the streamed
# snapshot is a self-contained single file.
rm -f "${TARGET}-wal" "${TARGET}-shm"
mv "${TARGET}.tmp" "${TARGET}"

echo "[5/6] cleaning up snapshot on NAS…"
ssh "${NAS_HOST}" "${remote} run --rm --entrypoint rm photosearch ${REMOTE_DUMP}" \
  || echo "[sync-replica]   (remote cleanup failed; next sync overwrites it)"

# Mirror face crops (skip with SYNC_FACE_CROPS=0). Same container tar/cat-stream
# trick as the DB above — UGREEN blocks rsync into the volume, and a named docker
# volume has no host path to rsync from. Incremental via mtime: we pass the last
# successful sync time as --since so only crops generated since then are tarred
# (append-only + immutable per face_id → tiny delta after the first pull).
if [ "${SYNC_FACE_CROPS:-1}" != "0" ]; then
  echo "[6/6] mirroring face crops…"
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

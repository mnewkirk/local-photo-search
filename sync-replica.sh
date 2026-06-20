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
# Thumbnails are NOT mirrored here — replica image routes lazily proxy + cache
# them from the NAS web API on demand (see web.py _fetch_from_nas). A bulk
# pre-warm flag can be added later.
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

echo "[4/4] cleaning up snapshot on NAS…"
ssh "${NAS_HOST}" "${remote} run --rm --entrypoint rm photosearch ${REMOTE_DUMP}" \
  || echo "[sync-replica]   (remote cleanup failed; next sync overwrites it)"

human_size=$(du -h "${TARGET}" 2>/dev/null | cut -f1 || echo "?")
echo "[sync-replica] done in $(( $(date +%s) - started ))s — ${human_size} at ${TARGET}"

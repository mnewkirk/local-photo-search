#!/usr/bin/env bash
# debug-db.sh — local debugging against a copy of the NAS DB.
#
# Pulls /data/photo_index.db from the NAS to a local file, then exposes
# subcommands for running queries against the copy. Safer than running
# ad-hoc SQL on the production DB and faster for iterative debugging.
#
# Override paths via env:
#   NAS_HOST         ssh target (e.g. user@192.168.1.237)
#   NAS_DATA_DIR     bind-mounted /data dir on the NAS
#                    (default: /volume1/docker/photosearch/data)
#   LOCAL_DB         local copy path (default: ./photo_index.db.local)

set -euo pipefail

NAS_HOST="${NAS_HOST:-cantimatt@192.168.1.237}"
NAS_DATA_DIR="${NAS_DATA_DIR:-/volume1/docker/photosearch/data}"
LOCAL_DB="${LOCAL_DB:-./photo_index.db.local}"

have_local() {
  if [ ! -f "${LOCAL_DB}" ]; then
    echo "No local DB at ${LOCAL_DB}. Run '$0 pull' first." >&2
    exit 1
  fi
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  pull)
    # rsync the .db plus .db-wal / .db-shm if present (WAL mode writes
    # land in the -wal file until checkpointed). Copying all three
    # gives us a consistent snapshot for read-only debugging. Adds a
    # few seconds per GB but avoids surprise stale data.
    echo "Pulling DB from ${NAS_HOST}:${NAS_DATA_DIR}/ …"
    rsync -avz --progress \
      "${NAS_HOST}:${NAS_DATA_DIR}/photo_index.db" \
      "${LOCAL_DB}"
    for ext in wal shm; do
      rsync -avz --ignore-missing-args \
        "${NAS_HOST}:${NAS_DATA_DIR}/photo_index.db-${ext}" \
        "${LOCAL_DB}-${ext}" 2>/dev/null || true
    done
    ls -lh "${LOCAL_DB}"* 2>/dev/null | awk '{print "  "$5"\t"$9}'
    echo "Done."
    ;;

  shell)
    have_local
    # -readonly protects the prod-cloned copy from accidental writes.
    sqlite3 -readonly -header -column "${LOCAL_DB}"
    ;;

  query)
    have_local
    if [ $# -eq 0 ]; then
      echo "Usage: $0 query \"SELECT ...\"" >&2
      exit 1
    fi
    sqlite3 -readonly -header -column "${LOCAL_DB}" "$@"
    ;;

  person)
    have_local
    if [ $# -eq 0 ]; then
      echo "Usage: $0 person NAME [PLACE_PATTERN]" >&2
      exit 1
    fi
    name="$1"; shift
    place_arg=()
    if [ $# -gt 0 ]; then
      place_arg=(--place-like "$1")
    fi
    PHOTOSEARCH_DB="${LOCAL_DB}" \
      venv/bin/python cli.py person-coverage "${name}" "${place_arg[@]}"
    ;;

  stats)
    have_local
    PHOTOSEARCH_DB="${LOCAL_DB}" venv/bin/python cli.py stats
    ;;

  clean)
    rm -f "${LOCAL_DB}" "${LOCAL_DB}-wal" "${LOCAL_DB}-shm"
    echo "Removed ${LOCAL_DB}(-wal|-shm)."
    ;;

  help|*)
    cat <<EOF
debug-db.sh — debug NAS DB locally (read-only copy).

Usage:
  $0 pull                        rsync prod DB to ${LOCAL_DB}
  $0 shell                       open sqlite3 shell (read-only)
  $0 query "SELECT …"            run one SQL query
  $0 person NAME [PLACE]         run person-coverage CLI
  $0 stats                       run stats CLI
  $0 clean                       delete local copy
  $0 help                        this

Env overrides:
  NAS_HOST      (default: ${NAS_HOST})
  NAS_DATA_DIR  (default: ${NAS_DATA_DIR})
  LOCAL_DB      (default: ${LOCAL_DB})

Typical flow:
  ./debug-db.sh pull
  ./debug-db.sh person Calvin "Lucas Valley"
  ./debug-db.sh query "SELECT place_name, COUNT(*) c FROM photos
                       WHERE place_name LIKE '%Marin%'
                       GROUP BY place_name ORDER BY c DESC LIMIT 10"

Local copy is SQLite WAL-consistent at pull time. Re-pull whenever you
want a fresh snapshot.
EOF
    ;;
esac

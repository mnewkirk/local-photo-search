#!/usr/bin/env bash
# debug-db.sh — local debugging against a copy of the NAS DB.
#
# Streams /data/photo_index.db from the NAS through the Docker
# container to a local file (avoids UGREEN's access layer that blocks
# direct rsync to /volume1/docker/...), then exposes subcommands for
# running queries against the copy. Safer than running ad-hoc SQL on
# the production DB and faster for iterative debugging.
#
# Override paths via env:
#   NAS_HOST           ssh target (e.g. user@192.168.1.237)
#   NAS_COMPOSE_FILE   docker-compose.nas.yml path on the NAS
#                      (default: /volume1/docker/photosearch/docker-compose.nas.yml)
#   LOCAL_DB           local copy path (default: ./photo_index.db.local)

set -euo pipefail

NAS_HOST="${NAS_HOST:-cantimatt@192.168.1.237}"
NAS_COMPOSE_FILE="${NAS_COMPOSE_FILE:-/volume1/docker/photosearch/docker-compose.nas.yml}"
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
    # UGREEN's NAS blocks direct rsync to /volume1/docker/... (the
    # volume has docker-owned perms that the login user can't see
    # through their rsync daemon). Workaround: run sqlite3 backup()
    # inside a transient photosearch container, which has root
    # access to /data, to produce a consistent snapshot on the
    # bind-mounted volume, then stream that file out via
    # `docker compose run --entrypoint cat`. Three SSH calls but
    # each is clean and no permission tangles.
    remote="docker compose -f '${NAS_COMPOSE_FILE}'"

    echo "1/3  creating consistent backup on NAS (sqlite3 backup API)…"
    ssh "${NAS_HOST}" "${remote} run --rm photosearch dump-db --to /data/debug-dump.db"

    echo "2/3  streaming ${LOCAL_DB}…"
    # -T disables TTY so binary bytes aren't corrupted.
    ssh "${NAS_HOST}" "${remote} run --rm -T --entrypoint cat photosearch /data/debug-dump.db" \
      > "${LOCAL_DB}.tmp"
    mv "${LOCAL_DB}.tmp" "${LOCAL_DB}"

    echo "3/3  cleaning up /data/debug-dump.db on NAS…"
    ssh "${NAS_HOST}" "${remote} run --rm --entrypoint rm photosearch /data/debug-dump.db" \
      || echo "  (cleanup failed; next pull will overwrite)"

    size=$(du -h "${LOCAL_DB}" | cut -f1)
    echo "Done. ${size} at ${LOCAL_DB}"
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
  NAS_HOST          (default: ${NAS_HOST})
  NAS_COMPOSE_FILE  (default: ${NAS_COMPOSE_FILE})
  LOCAL_DB          (default: ${LOCAL_DB})

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

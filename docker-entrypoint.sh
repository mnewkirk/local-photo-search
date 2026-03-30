#!/bin/bash
set -e

# Ensure /data directory exists
mkdir -p /data/thumbnails

# Auto-remap paths if REMAP_FROM is set and DB exists
# This handles the "Mac-built DB deployed to NAS" workflow.
# Example: REMAP_FROM=/Users/matt/Pictures will rewrite all stored paths
# to /photos (the container mount point).
if [ -n "${REMAP_FROM}" ] && [ -f "${PHOTOSEARCH_DB}" ]; then
    echo "=== Auto-remapping paths: ${REMAP_FROM} → /photos ==="
    python cli.py remap-paths "${REMAP_FROM}" /photos --db "${PHOTOSEARCH_DB}" || true
    # Only runs once — clear the marker after remapping
    echo "  (Set REMAP_FROM= empty to skip this on future starts)"
fi

case "${1:-serve}" in
    serve)
        echo "=== local-photo-search: starting web UI ==="
        echo "  Database: ${PHOTOSEARCH_DB}"
        echo "  Photos:   /photos"
        echo "  Port:     8000"
        echo ""
        exec python cli.py serve \
            --db "${PHOTOSEARCH_DB}" \
            --host 0.0.0.0 \
            --port 8000
        ;;
    index)
        shift  # consume "index"
        PHOTO_DIR="${1:-/photos}"
        shift 2>/dev/null || true

        echo "=== local-photo-search: indexing ${PHOTO_DIR} ==="
        exec python cli.py index "${PHOTO_DIR}" \
            --db "${PHOTOSEARCH_DB}" \
            "$@"
        ;;
    *)
        # Pass through any other cli.py subcommand
        exec python cli.py "$@"
        ;;
esac

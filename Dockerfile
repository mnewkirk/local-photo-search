# ============================================================================
# local-photo-search — production Dockerfile
#
# Multi-stage build:
#   1. Builder: installs Python deps and downloads InsightFace model weights
#   2. Runtime: slim image with only what's needed to serve + index
#
# The image supports two modes via the ENTRYPOINT:
#   - "serve"  (default) — launches the FastAPI web UI
#   - "index"  — runs the CLI indexer against /photos
#   - Any other cli.py subcommand works too
#
# Volumes:
#   /photos — mount your photo library (read-only is fine for search)
#   /data   — persistent storage for the SQLite database + thumbnails
# ============================================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

# Runtime libraries needed by OpenCV / Pillow / InsightFace
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY photosearch/ photosearch/
COPY frontend/ frontend/
COPY cli.py .
COPY requirements.txt .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Volumes:
#   /photos — your photo library (can be read-only for serving)
#   /data   — SQLite DB, thumbnails, face model cache
VOLUME ["/photos", "/data"]

# InsightFace model cache — store in persistent volume
ENV INSIGHTFACE_HOME=/data/.insightface
# Ollama host — points to the sidecar container by default
ENV OLLAMA_HOST=http://ollama:11434
# Database location
ENV PHOTOSEARCH_DB=/data/photo_index.db
# Photo root — base path for resolving relative file paths in the DB.
# Set this to where photos are mounted in the container.
ENV PHOTO_ROOT=/photos

EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["serve"]

#!/usr/bin/env bash
# ===========================================================================
# run-workers.sh — Launch a fleet of Docker-based indexing workers
#
# Runs N isolated containers with hard memory limits and CPU-only PyTorch.
# Each worker claims batches from the NAS server, processes locally, and
# submits results back. Containers restart on crash (up to 3 retries).
#
# Usage:
#   ./run-workers.sh -s http://nas.local:8000 -p clip -d /photos/2026 -n 4
#   ./run-workers.sh -s http://nas.local:8000 -p clip,quality --collection 3 -n 2
#   ./run-workers.sh -s http://nas.local:8000 -p faces -d /photos/2026    # default: 4 workers
#   ./run-workers.sh --status                                               # show running workers
#   ./run-workers.sh --stop                                                 # stop all workers
#   ./run-workers.sh --logs                                                 # tail all worker logs
#
# Ollama for describe/tags/verify passes
# --------------------------------------
# If --passes includes describe, tags, or verify, the script checks
# localhost:11434. If nothing answers, it starts a managed container
# ($OLLAMA_CONTAINER) and auto-pulls the required models into its volume
# ($OLLAMA_VOLUME). For `verify` this pulls BOTH the verifier (minicpm-v) and
# the regeneration model (llava) — verify regenerates descriptions via the
# describe model when a photo fails verification, so both must be present.
# The managed container is torn down by --stop.
#
# PREFER NATIVE OLLAMA (run `ollama serve` on the host) if you can. The
# managed-in-Docker path shares Docker Desktop's VM memory budget with every
# worker container — on a Mac with a default VM (~7-8 GiB) plus 4 workers at
# -m 3g and LLaVA's ~4.3 GiB working set, the VM is heavily oversubscribed and
# the llama runner gets OOM-killed mid-load ("llama runner process has
# terminated: %!w(<nil>)", status 500). Native Ollama uses the Mac's full RAM
# directly and does not compete.
#
# If you must use the managed container, raise Docker Desktop memory
# (Settings → Resources → Memory) to at least ~24 GiB for the default fleet
# (4 workers × 3 GiB = 12 GiB) plus Ollama's ~4-5 GiB working set plus daemon
# overhead. ~16 GiB has been observed to still OOM-kill the llama runner.
# Alternatively reduce -n / -m.
# ===========================================================================

set -euo pipefail

PROJECT="photosearch-worker"
OLLAMA_CONTAINER="photosearch-worker-ollama"
OLLAMA_VOLUME="photosearch-worker-ollama-models"
OLLAMA_PORT=11434
# Shared model-cache volume — without this every worker re-downloads ~1.7 GB
# CLIP ViT-L/14 + ~300 MB InsightFace on startup, and concurrent unauthenticated
# parallel pulls from huggingface.co get rate-limited to "Connection refused".
# HF's hub cache uses .lock files so simultaneous workers are safe: first to
# arrive downloads, the rest block on the lock and read the cached weights.
MODEL_CACHE_VOLUME="photosearch-worker-model-cache"

# Defaults
NUM_WORKERS=4
SERVER=""
PASSES="clip"
DIRECTORY=""
COLLECTION=""
BATCH_SIZE=16
MODEL_BATCH_SIZE=8
TTL=30
MEM_LIMIT="3g"
FORCE=""
DESCRIBE_MODEL=""
VERIFY_MODEL=""

usage() {
    cat <<'EOF'
Usage: ./run-workers.sh [OPTIONS]

Start workers:
  -s, --server URL        NAS server URL (required, e.g. http://nas.local:8000)
  -p, --passes PASSES     Comma-separated passes (default: clip)
  -d, --directory DIR      Scope to directory on NAS (e.g. /photos/2026)
  -c, --collection ID     Scope to collection ID
  -n, --num-workers N     Number of worker containers (default: 4)
  -m, --memory LIMIT      Memory limit per container (default: 3g)
      --batch-size N      Photos per batch (default: 16)
      --model-batch-size N  Inference batch size (default: 8)
      --ttl MINUTES       Claim TTL (default: 30)
      --force             Clear existing data and reprocess
      --describe-model M  Ollama model for describe/tags (default: llava)
      --verify-model M    Ollama model for verification (default: minicpm-v)

Manage workers:
      --status            Show running workers and their progress
      --logs              Tail logs from all workers
      --stop              Stop all workers

Examples:
  ./run-workers.sh -s http://nas.local:8000 -p clip -d /photos/2026 -n 4
  ./run-workers.sh -s http://nas.local:8000 -p clip,quality,faces -d /photos/2026
  ./run-workers.sh -s http://nas.local:8000 -p describe --collection 3 -n 2
  ./run-workers.sh --status
  ./run-workers.sh --stop

Ollama note:
  For describe/tags/verify, PREFER a native `ollama serve` on the host. The
  script will detect and reuse it. The managed-in-Docker fallback shares
  Docker Desktop's VM memory with every worker — on a default VM, LLaVA's
  runner is OOM-killed ("llama runner process has terminated", 500). If you
  must use the managed container, raise Docker Desktop memory to ~24 GiB for
  the default fleet (4 × 3 GiB + Ollama + daemon). ~16 GiB is too tight.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Ollama helpers (auto-start for describe/tags/verify passes)
# ---------------------------------------------------------------------------

ollama_needed() {
    # Returns 0 if any pass in $PASSES requires an Ollama backend.
    local IFS=','
    for p in $PASSES; do
        case "$p" in
            describe|tags|verify) return 0 ;;
        esac
    done
    return 1
}

ollama_is_reachable() {
    curl -sf --max-time 2 "http://localhost:${OLLAMA_PORT}/api/tags" > /dev/null 2>&1
}

ollama_container_exists() {
    docker ps -a --filter "name=^${OLLAMA_CONTAINER}$" -q 2>/dev/null | grep -q .
}

ensure_ollama_running() {
    if ollama_is_reachable; then
        if ollama_container_exists; then
            echo "  Ollama reachable (managed container: $OLLAMA_CONTAINER)"
        else
            echo "  Ollama already reachable at localhost:${OLLAMA_PORT} (not managed by this script)"
        fi
        return
    fi
    echo "  Ollama not reachable — starting managed container ($OLLAMA_CONTAINER)..."
    docker rm -f "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true
    # OLLAMA_NUM_PARALLEL=1 and MAX_LOADED_MODELS=1 serialize requests to avoid
    # llama-runner crashes under concurrent load in CPU-only Docker (500 errors
    # like "llama runner process has terminated"). Multiple workers still help —
    # Ollama queues their requests server-side.
    docker run -d \
        --name "$OLLAMA_CONTAINER" \
        --label "photosearch-worker-ollama=true" \
        --restart unless-stopped \
        -p "${OLLAMA_PORT}:11434" \
        -v "${OLLAMA_VOLUME}:/root/.ollama" \
        -e OLLAMA_NUM_PARALLEL=1 \
        -e OLLAMA_MAX_LOADED_MODELS=1 \
        ollama/ollama:latest > /dev/null
    local waited=0
    until ollama_is_reachable; do
        if [ "$waited" -ge 30 ]; then
            echo "  WARNING: Ollama container started but not reachable after 30s." >&2
            echo "  Check logs: docker logs $OLLAMA_CONTAINER" >&2
            return
        fi
        sleep 1
        waited=$((waited + 1))
    done
    echo "  Ollama container ready (models persisted in volume: $OLLAMA_VOLUME)"
}

required_ollama_models() {
    # Prints newline-separated list of Ollama models required by $PASSES.
    # verify needs both the verifier and the regen model (used when a photo
    # fails verification and its description/tags must be re-generated).
    local IFS=','
    for p in $PASSES; do
        case "$p" in
            describe|tags) echo "${DESCRIBE_MODEL:-llava}" ;;
            verify)
                echo "${VERIFY_MODEL:-minicpm-v}"
                echo "${DESCRIBE_MODEL:-llava}"
                ;;
        esac
    done | sort -u
}

ollama_has_model() {
    # Uses /api/show which returns 200 iff the model is present locally.
    local model="$1"
    curl -sf --max-time 5 -X POST "http://localhost:${OLLAMA_PORT}/api/show" \
        -H 'Content-Type: application/json' \
        -d "{\"name\":\"$model\"}" > /dev/null 2>&1
}

ensure_ollama_models() {
    # Pulls any required models not already present. Only auto-pulls into the
    # managed container — for an external Ollama, we only warn, since pulling
    # multi-GB models into someone else's Ollama unannounced is rude.
    local models
    models=$(required_ollama_models)
    [ -z "$models" ] && return
    local managed=0
    if ollama_container_exists; then
        managed=1
    fi
    while IFS= read -r model; do
        [ -z "$model" ] && continue
        if ollama_has_model "$model"; then
            echo "  Model '$model' already present"
            continue
        fi
        if [ "$managed" -eq 1 ]; then
            echo "  Model '$model' missing — pulling into $OLLAMA_CONTAINER (may take several minutes)..."
            if ! docker exec "$OLLAMA_CONTAINER" ollama pull "$model"; then
                echo "  WARNING: Pull failed for '$model'. Retry manually:" >&2
                echo "    docker exec $OLLAMA_CONTAINER ollama pull $model" >&2
            fi
        else
            echo "  WARNING: Model '$model' not found in external Ollama. Pull it with:" >&2
            echo "    ollama pull $model" >&2
        fi
    done <<< "$models"
}

stop_managed_ollama_if_idle() {
    # Stops our managed Ollama container iff no worker containers remain.
    if ! ollama_container_exists; then
        return
    fi
    local remaining
    remaining=$(docker ps --filter "label=photosearch-worker-fleet" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$remaining" -eq 0 ]; then
        echo "Stopping managed Ollama container ($OLLAMA_CONTAINER)..."
        docker rm -f "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true
    fi
}

# ---------------------------------------------------------------------------
# Management commands (no server required)
# ---------------------------------------------------------------------------

do_status() {
    local FILTER="label=photosearch-worker-fleet"
    echo "=== Worker Containers ==="
    echo ""
    # Show running containers
    local containers
    containers=$(docker ps --filter "$FILTER" \
                           --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}" 2>/dev/null || true)
    if [ -z "$containers" ] || [ "$(echo "$containers" | wc -l)" -le 1 ]; then
        echo "No workers running."
        echo ""
        echo "Start workers with: ./run-workers.sh -s <server> -p <passes> -d <directory>"
        exit 0
    fi
    echo "$containers"
    echo ""

    # Show memory usage
    echo "=== Memory Usage ==="
    echo ""
    docker stats --no-stream --filter "$FILTER" \
                 --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.CPUPerc}}" 2>/dev/null || true
    echo ""

    # Show recent log lines per container (last batch progress)
    echo "=== Recent Progress ==="
    echo ""
    docker ps --filter "$FILTER" \
              --format "{{.Names}}" 2>/dev/null | sort | while read -r name; do
        echo "--- $name ---"
        docker logs --tail 8 "$name" 2>&1 | sed 's/^/  /'
        echo ""
    done

    # Show managed Ollama container status
    echo "=== Ollama ==="
    echo ""
    if ollama_container_exists; then
        docker ps -a --filter "name=^${OLLAMA_CONTAINER}$" \
                     --format "  {{.Names}}\t{{.Status}}" 2>/dev/null
    elif ollama_is_reachable; then
        echo "  External Ollama reachable at localhost:${OLLAMA_PORT} (not managed by this script)"
    else
        echo "  Not running. Will auto-start if a describe/tags/verify pass is launched."
    fi
    echo ""
    exit 0
}

do_logs() {
    local FILTER="label=photosearch-worker-fleet"
    local names
    names=$(docker ps --filter "$FILTER" --format "{{.Names}}" 2>/dev/null | sort)
    if [ -z "$names" ]; then
        echo "No workers running."
        exit 0
    fi
    echo "Tailing logs from all workers (Ctrl-C to stop)..."
    echo ""
    # Follow logs from all containers in parallel, prefixed with short name
    for name in $names; do
        short="${name##*-}"
        docker logs -f --tail 10 "$name" 2>&1 | sed "s/^/[$short] /" &
    done
    # Wait for all background processes; Ctrl-C kills them via trap
    trap 'kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
    wait
    exit 0
}

do_stop() {
    local FILTER="label=photosearch-worker-fleet"
    local count
    count=$(docker ps --filter "$FILTER" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$count" -gt 0 ]; then
        echo "Stopping $count worker(s)..."
        docker ps --filter "$FILTER" -q 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1
        echo "All workers stopped. Unclaimed batches will be reclaimed after TTL expires."
    else
        echo "No workers running."
    fi
    stop_managed_ollama_if_idle
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--server)        SERVER="$2";           shift 2 ;;
        -p|--passes)        PASSES="$2";           shift 2 ;;
        -d|--directory)     DIRECTORY="$2";        shift 2 ;;
        -c|--collection)    COLLECTION="$2";       shift 2 ;;
        -n|--num-workers)   NUM_WORKERS="$2";      shift 2 ;;
        -m|--memory)        MEM_LIMIT="$2";        shift 2 ;;
        --batch-size)       BATCH_SIZE="$2";       shift 2 ;;
        --model-batch-size) MODEL_BATCH_SIZE="$2"; shift 2 ;;
        --ttl)              TTL="$2";              shift 2 ;;
        --force)            FORCE="1";             shift ;;
        --describe-model)   DESCRIBE_MODEL="$2";   shift 2 ;;
        --verify-model)     VERIFY_MODEL="$2";     shift 2 ;;
        --status)           do_status ;;
        --logs)             do_logs ;;
        --stop)             do_stop ;;
        -h|--help)          usage ;;
        *)  echo "Unknown option: $1" >&2; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

if [ -z "$SERVER" ]; then
    echo "Error: --server is required." >&2
    echo ""
    usage
fi

if [[ "$SERVER" != http://* && "$SERVER" != https://* ]]; then
    SERVER="http://$SERVER"
fi

if [ -n "$DIRECTORY" ] && [ -n "$COLLECTION" ]; then
    echo "Error: --directory and --collection are mutually exclusive." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the worker command
# ---------------------------------------------------------------------------

WORKER_CMD="worker --server $SERVER --passes $PASSES --batch-size $BATCH_SIZE --model-batch-size $MODEL_BATCH_SIZE --ttl $TTL"

if [ -n "$DIRECTORY" ]; then
    WORKER_CMD="$WORKER_CMD --directory $DIRECTORY"
fi
if [ -n "$COLLECTION" ]; then
    WORKER_CMD="$WORKER_CMD --collection $COLLECTION"
fi
if [ -n "$FORCE" ]; then
    WORKER_CMD="$WORKER_CMD --force"
fi
if [ -n "$DESCRIBE_MODEL" ]; then
    WORKER_CMD="$WORKER_CMD --describe-model $DESCRIBE_MODEL"
fi
if [ -n "$VERIFY_MODEL" ]; then
    WORKER_CMD="$WORKER_CMD --verify-model $VERIFY_MODEL"
fi

SCOPE=""
if [ -n "$DIRECTORY" ]; then SCOPE=" (directory: $DIRECTORY)"; fi
if [ -n "$COLLECTION" ]; then SCOPE=" (collection: $COLLECTION)"; fi

# ---------------------------------------------------------------------------
# Build image if needed
# ---------------------------------------------------------------------------

echo "=== Photo Search Worker Fleet ==="
echo ""
echo "  Server:     $SERVER"
echo "  Passes:     $PASSES"
echo "  Workers:    $NUM_WORKERS"
echo "  Memory:     $MEM_LIMIT per worker"
echo "  Scope:      ${SCOPE:- all photos}"
echo ""

IMAGE_TAG="photosearch-worker:latest"

# Give the user a realistic expectation before the build runs, and show the
# live BuildKit output so they can see which layers hit the cache vs. rebuild.
#   - First build: base image pull + pip install ~5-10min
#   - requirements.txt changed: pip layer rebuild ~5min
#   - Python-only change: COPY layers ~10s
# Watch BuildKit's "CACHED" markers to see which path you got.
if ! docker image inspect "$IMAGE_TAG" > /dev/null 2>&1; then
    echo "Building worker image (first build on this machine — expect ~5-10 min"
    echo "to pull python:3.11-slim and install PyTorch/CLIP/InsightFace)..."
else
    echo "Building worker image (incremental — ~10s for Python-only changes,"
    echo "~5 min if requirements.txt changed; watch for 'CACHED' lines below)..."
fi
echo ""
BUILD_START=$(date +%s)
docker build -t "$IMAGE_TAG" .
BUILD_ELAPSED=$(($(date +%s) - BUILD_START))
echo ""
echo "  Image ready: $IMAGE_TAG (built in ${BUILD_ELAPSED}s)"
echo ""

# ---------------------------------------------------------------------------
# Stop any existing workers from a previous run
# ---------------------------------------------------------------------------

EXISTING=$(docker ps -a --filter "label=photosearch-worker-fleet" -q 2>/dev/null | wc -l | tr -d ' ')
if [ "$EXISTING" -gt 0 ]; then
    echo "Stopping $EXISTING existing worker(s)..."
    docker ps -a --filter "label=photosearch-worker-fleet" -q 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1
    echo ""
fi

# ---------------------------------------------------------------------------
# Ensure Ollama is running if any pass needs it
# ---------------------------------------------------------------------------

if ollama_needed; then
    echo "Passes include describe/tags/verify — checking Ollama availability..."
    ensure_ollama_running
    if ollama_is_reachable; then
        ensure_ollama_models
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Launch N workers
# ---------------------------------------------------------------------------

echo "Starting $NUM_WORKERS workers..."
echo ""

for i in $(seq 1 "$NUM_WORKERS"); do
    CONTAINER_NAME="${PROJECT}-${i}"
    # Remove leftover stopped container with this name
    docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
    echo "  Starting $CONTAINER_NAME..."
    docker run -d \
        --name "$CONTAINER_NAME" \
        --label "photosearch-worker-fleet=true" \
        --memory "$MEM_LIMIT" \
        --restart on-failure:3 \
        --add-host=host.docker.internal:host-gateway \
        -v "${MODEL_CACHE_VOLUME}:/model-cache" \
        -e PHOTOSEARCH_DEVICE=cpu \
        -e PYTHONUNBUFFERED=1 \
        -e OLLAMA_HOST=http://host.docker.internal:11434 \
        -e HF_HOME=/model-cache/huggingface \
        -e INSIGHTFACE_HOME=/model-cache/insightface \
        "$IMAGE_TAG" \
        $WORKER_CMD \
        > /dev/null
done

echo ""
echo "=== $NUM_WORKERS workers launched ==="
echo ""
echo "Monitor:"
echo "  ./run-workers.sh --status    # container status + memory + recent progress"
echo "  ./run-workers.sh --logs      # tail all worker logs live"
echo "  ./run-workers.sh --stop      # stop all workers"
echo ""
echo "Workers will exit automatically when the queue is empty."

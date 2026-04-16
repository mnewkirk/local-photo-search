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
# ===========================================================================

set -euo pipefail

PROJECT="photosearch-worker"

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
EOF
    exit 0
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
    if [ "$count" -eq 0 ]; then
        echo "No workers running."
        exit 0
    fi
    echo "Stopping $count worker(s)..."
    docker ps --filter "$FILTER" -q 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1
    echo "All workers stopped. Unclaimed batches will be reclaimed after TTL expires."
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

echo "Building worker image (this is fast if only Python changed)..."
docker build -q -t "$IMAGE_TAG" . > /dev/null
echo "  Image ready: $IMAGE_TAG"
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
        -e PHOTOSEARCH_DEVICE=cpu \
        -e PYTHONUNBUFFERED=1 \
        -e OLLAMA_HOST=http://host.docker.internal:11434 \
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

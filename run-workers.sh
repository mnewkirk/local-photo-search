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
# Ollama for the LLM passes (describe / category-* / keywords / verify)
# ----------------------------------------------------------------------
# If --passes includes any LLM pass (describe, category-content,
# category-visual, keywords, verify), the script checks localhost:11434.
# If nothing answers, it starts a managed container ($OLLAMA_CONTAINER)
# and auto-pulls the required models into its volume ($OLLAMA_VOLUME).
# The passes use different models (see the per-pass model strategy in
# CLAUDE.md); `verify` pulls its verifier model plus the describe +
# visual models it uses to regenerate failed descriptions.
# The managed container is torn down by --stop.
# (The old `tags` pass was removed — it split into category-content,
# category-visual, and keywords. To route the LLM passes to LM Studio
# instead of Ollama, export PHOTOSEARCH_TEXT_LLM_URL + PHOTOSEARCH_LLM_*
# role models before a --native launch; see CLAUDE.md.)
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
TAGS_MODEL=""
VERIFY_MODEL=""

# OpenAI-compatible (LM Studio / llama-server) routing for the LLM passes.
# When --text-llm-url is set, describe.py routes ALL text+vision calls there
# instead of Ollama, and per-pass models are chosen by the role vars below.
# These map to the PHOTOSEARCH_TEXT_LLM_URL / PHOTOSEARCH_LLM_*_MODEL env vars
# that describe.py reads (so passing the flag is equivalent to exporting them).
TEXT_LLM_URL="${PHOTOSEARCH_TEXT_LLM_URL:-}"
LLM_DESCRIBE_MODEL="${PHOTOSEARCH_LLM_DESCRIBE_MODEL:-}"
LLM_VERIFY_MODEL="${PHOTOSEARCH_LLM_VERIFY_MODEL:-}"
LLM_VISUAL_MODEL="${PHOTOSEARCH_LLM_VISUAL_MODEL:-}"
LLM_TEXT_MODEL="${PHOTOSEARCH_LLM_TEXT_MODEL:-}"

# Execution mode. `native` runs `cli.py worker` processes directly from the
# repo venv (GPU-capable — picks up CUDA/ROCm torch); `docker` runs the
# CPU-only container fleet. `auto` picks native on WSL2 (where the Docker
# fleet can't use the GPU and can't easily reach a GPU Ollama on the Windows
# host) and docker everywhere else. Override with --native / --docker.
MODE="auto"               # auto | native | docker
OLLAMA_HOST_OVERRIDE=""   # --ollama-host: base URL where Ollama is reachable
ACTION="start"            # start | status | logs | stop | scale
SCALE_TARGET=""
# Fleet instance name. Empty = the default fleet (back-compat). Set via --name to
# run multiple independent fleets at once (e.g. a CPU text fleet + a GPU vision
# fleet) — it suffixes NATIVE_RUNDIR, the docker PROJECT, and the docker label so
# the two don't share pid/log files or get killed by each other's stop-loop.
# Pass the SAME --name to --status/--logs/--stop to target that fleet.
FLEET_NAME=""
OLLAMA_URL="http://localhost:${OLLAMA_PORT}"   # resolved per-mode after arg parse
NATIVE_RUNDIR="/tmp/photosearch-worker-fleet"  # native-mode pid/log files (suffixed by --name)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_wsl() { grep -qi microsoft /proc/version 2>/dev/null; }

detect_mode() {
    [ "$MODE" != "auto" ] && return
    if is_wsl; then MODE="native"; else MODE="docker"; fi
}

find_venv_python() {
    local p
    for p in "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/venv/bin/python" "$SCRIPT_DIR/env/bin/python"; do
        [ -x "$p" ] && { echo "$p"; return 0; }
    done
    return 1
}

resolve_ollama_url() {
    # Priority: --ollama-host > $OLLAMA_HOST env > WSL2 Windows-host gateway >
    # localhost. In native WSL2 mode the GPU-capable Ollama runs on Windows,
    # reachable via the default-route gateway IP.
    if [ -n "$OLLAMA_HOST_OVERRIDE" ]; then echo "$OLLAMA_HOST_OVERRIDE"; return; fi
    if [ -n "${OLLAMA_HOST:-}" ]; then echo "$OLLAMA_HOST"; return; fi
    if [ "$MODE" = "native" ] && is_wsl; then
        echo "http://$(ip route show default | awk '{print $3}'):11434"
    else
        echo "http://localhost:${OLLAMA_PORT}"
    fi
}

usage() {
    cat <<'EOF'
Usage: ./run-workers.sh [OPTIONS]

Start workers:
  -s, --server URL        NAS server URL (required, e.g. http://nas.local:8000)
  -p, --passes PASSES     Comma-separated passes (default: clip)
  -d, --directory DIR      Scope to directory on NAS (e.g. /photos/2026)
  -c, --collection ID     Scope to collection ID
  -n, --num-workers N     Number of workers (default: 4)
  -m, --memory LIMIT      Memory limit per container, docker mode only (default: 3g)
      --batch-size N      Photos per batch (default: 16)
      --model-batch-size N  Inference batch size (default: 8)
      --ttl MINUTES       Claim TTL (default: 30)
      --force             Clear existing data and reprocess
      --describe-model M  Ollama model for describe (default: llama3.2-vision)
      --tags-model M      (legacy — the tags pass was removed; ignored by the worker)
      --verify-model M    Ollama model for verification (default: llava)

  Route the LLM passes to an OpenAI-compatible backend (LM Studio / llama-server)
  instead of Ollama. When --text-llm-url is set, ALL text+vision calls go there;
  per-pass model is chosen by role. (Sets PHOTOSEARCH_TEXT_LLM_URL / PHOTOSEARCH_LLM_*
  for the workers, so no manual export needed. Defaults read from those env vars if
  already exported.) Base URL must end in /v1.
      --text-llm-url URL       e.g. http://172.20.176.1:1234/v1
      --llm-describe-model M   describe + regen   (role: describe)
      --llm-verify-model M     verify             (role: verify)
      --llm-visual-model M     category-visual    (role: visual)
      --llm-text-model M       category-content + keywords (role: text)
      --native            Run bare-metal venv workers (GPU-capable). Auto on WSL2.
      --docker            Run the CPU-only Docker fleet. Auto on Mac/Linux.
      --name NAME         Fleet instance name. Lets multiple fleets run at once
                          (e.g. a CPU text fleet + a GPU vision fleet) without
                          colliding. Pass the same --name to --status/--logs/--stop.
      --ollama-host URL   Base URL for Ollama (default: localhost, or the
                          Windows-host gateway in native WSL2 mode)

Manage workers:
      --status            Show running workers and their progress
      --logs              Tail logs from all workers
      --stop              Stop all workers
      --scale N           Adjust fleet to N workers (inherits image + cmd + env
                          + memory limit from an existing worker). Scale-up
                          appends new containers above the highest existing
                          index; scale-down kills the highest-numbered ones,
                          and their in-flight batches are reclaimed after the
                          claim TTL expires.

Examples:
  ./run-workers.sh -s http://nas.local:8000 -p clip -d /photos/2026 -n 4
  ./run-workers.sh -s http://nas.local:8000 -p clip,quality,faces -d /photos/2026
  ./run-workers.sh -s http://nas.local:8000 -p describe --collection 3 -n 2
  ./run-workers.sh --status
  ./run-workers.sh --scale 10           # resize running fleet without restart
  ./run-workers.sh --stop
  # Two simultaneous fleets — split a slow/unstable Ollama pass onto CPU:
  ./run-workers.sh -s http://nas:8000 --name cpu -p category-content,keywords \
      --ollama-host http://localhost:11434 -n 2 -d /photos/2026
  ./run-workers.sh -s http://nas:8000 --name gpu -p category-visual \
      --ollama-host http://172.20.176.1:11434 -n 3 -d /photos/2026
  ./run-workers.sh --name cpu --status   # manage each fleet by name
  ./run-workers.sh --name gpu --stop

Ollama note:
  For describe/category-*/verify, PREFER a native `ollama serve` on the host. The
  script will detect and reuse it. The managed-in-Docker fallback shares
  Docker Desktop's VM memory with every worker — on a default VM, LLaVA's
  runner is OOM-killed ("llama runner process has terminated", 500). If you
  must use the managed container, raise Docker Desktop memory to ~24 GiB for
  the default fleet (4 × 3 GiB + Ollama + daemon). ~16 GiB is too tight.
EOF
    exit 0
}

# ---------------------------------------------------------------------------
# Ollama helpers (auto-start for describe/category-*/verify passes)
# ---------------------------------------------------------------------------

ollama_needed() {
    # Returns 0 if any pass in $PASSES requires an Ollama backend. category-visual
    # is a vision pass; category-content/keywords are text-only but still hit
    # Ollama (small text model on the existing description).
    # When --text-llm-url is set, the LLM passes route to an OpenAI-compatible
    # backend (LM Studio) instead, so Ollama is never needed.
    [ -n "$TEXT_LLM_URL" ] && return 1
    local IFS=','
    for p in $PASSES; do
        case "$p" in
            describe|tags|verify|category-visual|category-content|keywords) return 0 ;;
        esac
    done
    return 1
}

ollama_is_reachable() {
    curl -sf --max-time 2 "${OLLAMA_URL}/api/tags" > /dev/null 2>&1
}

ollama_container_exists() {
    docker ps -a --filter "name=^${OLLAMA_CONTAINER}$" -q 2>/dev/null | grep -q .
}

ensure_ollama_running() {
    # Native mode never manages a Docker Ollama — describe/category-*/verify reach
    # whatever Ollama $OLLAMA_URL points at (a Windows-host Ollama on WSL2, or
    # a local `ollama serve`). Just check reachability and warn.
    if [ "$MODE" = "native" ]; then
        if ollama_is_reachable; then
            echo "  Ollama reachable at ${OLLAMA_URL}"
        else
            echo "  WARNING: Ollama not reachable at ${OLLAMA_URL}" >&2
            echo "  describe/category-*/verify will fail. Start Ollama on that host," >&2
            echo "  or pass --ollama-host URL." >&2
        fi
        return
    fi
    if ollama_is_reachable; then
        if ollama_container_exists; then
            local current_parallel
            current_parallel=$(docker inspect "$OLLAMA_CONTAINER" \
                --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null \
                | sed -n 's/^OLLAMA_NUM_PARALLEL=//p')
            echo "  Ollama reachable (managed container: $OLLAMA_CONTAINER, NUM_PARALLEL=${current_parallel:-?})"
            if [ -n "$current_parallel" ] && [ "$current_parallel" != "$NUM_WORKERS" ]; then
                echo "  NOTE: NUM_PARALLEL=$current_parallel but launching $NUM_WORKERS workers." >&2
                echo "        Workers past $current_parallel will queue at Ollama. To apply NUM_PARALLEL=$NUM_WORKERS," >&2
                echo "        recreate the container:  docker rm -f $OLLAMA_CONTAINER  &&  re-run this script." >&2
            fi
        else
            echo "  Ollama already reachable at localhost:${OLLAMA_PORT} (not managed by this script)"
            echo "  NOTE: cannot tune OLLAMA_NUM_PARALLEL on an external Ollama. Set it before 'ollama serve'" >&2
            echo "        if you want >1 slot (e.g. OLLAMA_NUM_PARALLEL=$NUM_WORKERS ollama serve)." >&2
        fi
        return
    fi
    echo "  Ollama not reachable — starting managed container ($OLLAMA_CONTAINER) with NUM_PARALLEL=$NUM_WORKERS..."
    docker rm -f "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true
    # NUM_PARALLEL is sized to match the worker count so every worker's outstanding
    # request gets its own slot instead of queueing. MAX_LOADED_MODELS=1 keeps a
    # single model resident (raise to 2 if you regularly run verify, which needs
    # both verifier + regen models). Each slot adds ~KV-cache RAM on top of the
    # model weights — keep an eye on container memory if you push NUM_WORKERS high.
    docker run -d \
        --name "$OLLAMA_CONTAINER" \
        --label "photosearch-worker-ollama=true" \
        --restart unless-stopped \
        -p "${OLLAMA_PORT}:11434" \
        -v "${OLLAMA_VOLUME}:/root/.ollama" \
        -e OLLAMA_NUM_PARALLEL="$NUM_WORKERS" \
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

check_ollama_gpu() {
    # The single biggest perf lever for Ollama passes is whether inference runs
    # on the GPU (40s/img CPU vs ~0.8s GPU). This is a preflight that catches the
    # two ways it silently regresses to CPU/slow on this WSL2 + Windows-Ollama
    # setup, and prints the exact fix. Best-effort — never blocks the launch.
    ollama_needed || return 0

    # On WSL2 the GPU-capable Ollama is the Windows host (gateway IP). A localhost
    # Ollama here is the intentional CPU instance (e.g. the --name cpu text fleet
    # for passes whose GPU runner stalls), so the GPU checks below don't apply.
    case "$OLLAMA_URL" in
        *localhost*|*127.0.0.1*) is_wsl && return 0 ;;
    esac

    # 1) Runtime truth: if a model is currently loaded, is it resident in VRAM?
    #    size_vram==0 means Ollama fell back to CPU (e.g. the Ollama 0.24 ROCm
    #    discovery regression on multi-AMD-GPU boxes).
    local ps_json vram size
    ps_json=$(curl -sf --max-time 3 "${OLLAMA_URL}/api/ps" 2>/dev/null || true)
    if printf '%s' "$ps_json" | grep -q '"size_vram"'; then
        vram=$(printf '%s' "$ps_json" | grep -o '"size_vram":[0-9]*' | head -1 | grep -o '[0-9]*$')
        size=$(printf '%s' "$ps_json" | grep -o '"size":[0-9]*' | head -1 | grep -o '[0-9]*$')
        if [ "${vram:-0}" -eq 0 ]; then
            echo "  WARNING: a model is loaded but size_vram=0 — Ollama is running on CPU." >&2
            echo "           Expect ~40s/image instead of ~1s. See the HIP_VISIBLE_DEVICES fix below." >&2
        elif [ -n "$size" ] && [ "$vram" -lt $(( size * 9 / 10 )) ]; then
            echo "  NOTE: only $((vram/1024/1024)) of $((size/1024/1024)) MiB on GPU — model partially on CPU." >&2
            echo "        Usually an oversized context; describe.py now pins num_ctx=8192. If this persists," >&2
            echo "        lower the Windows OLLAMA_CONTEXT_LENGTH or reduce OLLAMA_NUM_PARALLEL." >&2
        else
            echo "  Ollama GPU OK ($((vram/1024/1024)) MiB resident in VRAM)"
        fi
    fi

    # 2) Windows-side env vars (WSL2 → Windows-native Ollama). These configure the
    #    *Ollama server* and can't be set from this WSL2 script — so we read the
    #    persisted User-scope values via powershell.exe and remind how to fix.
    is_wsl || return 0
    command -v powershell.exe >/dev/null 2>&1 || return 0
    local hip parallel
    hip=$(powershell.exe -NoProfile -Command \
        "[Environment]::GetEnvironmentVariable('HIP_VISIBLE_DEVICES','User')" 2>/dev/null | tr -d '\r\n')
    parallel=$(powershell.exe -NoProfile -Command \
        "[Environment]::GetEnvironmentVariable('OLLAMA_NUM_PARALLEL','User')" 2>/dev/null | tr -d '\r\n')

    # HIP_VISIBLE_DEVICES should pin the discrete RX 7900 XTX (index 0); index 1
    # is the integrated Radeon (2 GiB, can't hold the model). Unset lets Ollama
    # 0.24 enumerate both and fall back to CPU.
    if [ "$hip" != "0" ]; then
        echo "  WARNING: Windows HIP_VISIBLE_DEVICES='${hip:-<unset>}' — want '0' (discrete RX 7900 XTX; '1' is the iGPU)." >&2
        echo "           Fix in PowerShell, then restart Ollama from the tray:" >&2
        echo "             [Environment]::SetEnvironmentVariable('HIP_VISIBLE_DEVICES','0','User')" >&2
    fi
    # OLLAMA_NUM_PARALLEL should match the worker count or extra workers queue.
    if [ -z "$parallel" ]; then
        echo "  NOTE: Windows OLLAMA_NUM_PARALLEL is unset (defaults to 1) — your $NUM_WORKERS workers will serialize." >&2
        echo "        Fix in PowerShell, then restart Ollama from the tray:" >&2
        echo "          [Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL','$NUM_WORKERS','User')" >&2
    elif [ "$parallel" != "$NUM_WORKERS" ]; then
        echo "  NOTE: Windows OLLAMA_NUM_PARALLEL=$parallel but launching $NUM_WORKERS workers — mismatch means queueing or idle slots." >&2
        echo "        Fix in PowerShell, then restart Ollama from the tray:" >&2
        echo "          [Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL','$NUM_WORKERS','User')" >&2
    fi
}

required_ollama_models() {
    # Prints newline-separated list of Ollama models required by $PASSES.
    # verify needs both the verifier and the regen model (used when a photo
    # fails verification and its description/tags must be re-generated).
    local IFS=','
    for p in $PASSES; do
        case "$p" in
            describe) echo "${DESCRIBE_MODEL:-llama3.2-vision}" ;;
            tags)     echo "${TAGS_MODEL:-llava}" ;;
            category-visual)  echo "${CATEGORY_VISUAL_MODEL:-llava}" ;;
            category-content) echo "${CATEGORY_CONTENT_MODEL:-llama3.2:3b}" ;;
            keywords)         echo "${KEYWORDS_MODEL:-llama3.2:3b}" ;;
            verify)
                echo "${VERIFY_MODEL:-llava}"
                echo "${DESCRIBE_MODEL:-llama3.2-vision}"
                echo "${TAGS_MODEL:-llava}"
                ;;
        esac
    done | sort -u
}

ollama_has_model() {
    # Uses /api/show which returns 200 iff the model is present.
    local model="$1"
    curl -sf --max-time 5 -X POST "${OLLAMA_URL}/api/show" \
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
    remaining=$(docker ps --filter "label=$FLEET_LABEL" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$remaining" -eq 0 ]; then
        echo "Stopping managed Ollama container ($OLLAMA_CONTAINER)..."
        docker rm -f "$OLLAMA_CONTAINER" > /dev/null 2>&1 || true
    fi
}

# ---------------------------------------------------------------------------
# Management commands (no server required)
# ---------------------------------------------------------------------------

do_status() {
    local FILTER="label=$FLEET_LABEL"
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
        echo "  Not running. Will auto-start if a describe/category-*/verify pass is launched."
    fi
    echo ""
    exit 0
}

do_logs() {
    local FILTER="label=$FLEET_LABEL"
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
    local FILTER="label=$FLEET_LABEL"
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

do_scale() {
    local TARGET="$1"
    if ! [[ "$TARGET" =~ ^[0-9]+$ ]]; then
        echo "Error: --scale requires a non-negative integer (got: '$TARGET')." >&2
        exit 1
    fi

    local FILTER="label=$FLEET_LABEL"
    # Sorted by numeric suffix so we know which are highest-indexed.
    local RUNNING_NAMES
    RUNNING_NAMES=$(docker ps --filter "$FILTER" --format "{{.Names}}" 2>/dev/null \
                    | sort -t- -k3 -n)
    local CURRENT=0
    if [ -n "$RUNNING_NAMES" ]; then
        CURRENT=$(echo "$RUNNING_NAMES" | wc -l | tr -d ' ')
    fi

    echo "Current fleet: $CURRENT worker(s). Target: $TARGET."

    if [ "$TARGET" -eq "$CURRENT" ]; then
        echo "Already at target — nothing to do."
        exit 0
    fi

    if [ "$TARGET" -lt "$CURRENT" ]; then
        local TO_REMOVE=$((CURRENT - TARGET))
        echo "Scaling down by $TO_REMOVE (stopping highest-numbered workers)..."
        local VICTIMS
        VICTIMS=$(echo "$RUNNING_NAMES" | tail -n "$TO_REMOVE")
        while IFS= read -r name; do
            [ -z "$name" ] && continue
            echo "  Removing $name (in-flight batch will be reclaimed after TTL)..."
            docker rm -f "$name" > /dev/null 2>&1 || true
        done <<< "$VICTIMS"
        if [ "$TARGET" -eq 0 ]; then
            stop_managed_ollama_if_idle
        fi
        echo ""
        echo "Done. Use --status to verify."
        exit 0
    fi

    # Scale up — inherit settings from an existing worker.
    if [ "$CURRENT" -eq 0 ]; then
        echo "Error: no workers running to inherit settings from." >&2
        echo "Launch a fresh fleet with: ./run-workers.sh -s <server> -p <passes> -n $TARGET" >&2
        exit 1
    fi

    local TEMPLATE
    TEMPLATE=$(echo "$RUNNING_NAMES" | head -1)
    echo "Inheriting settings from $TEMPLATE..."

    local TEMPLATE_IMAGE TEMPLATE_MEM TEMPLATE_CMD
    TEMPLATE_IMAGE=$(docker inspect "$TEMPLATE" --format '{{.Config.Image}}' 2>/dev/null)
    TEMPLATE_MEM=$(docker inspect "$TEMPLATE" --format '{{.HostConfig.Memory}}' 2>/dev/null)
    TEMPLATE_CMD=$(docker inspect "$TEMPLATE" --format '{{json .Config.Cmd}}' 2>/dev/null)

    if [ -z "$TEMPLATE_IMAGE" ] || [ -z "$TEMPLATE_CMD" ] || [ "$TEMPLATE_CMD" = "null" ]; then
        echo "Error: could not read image/cmd from $TEMPLATE via docker inspect." >&2
        exit 1
    fi

    # Cmd is a JSON array — decode into a NUL-separated bash array so args
    # with whitespace survive intact (relies on python3 on the dev host).
    # Python emits a trailing NUL after every arg (not just between) so that
    # `read -d ''` exits on the delimiter rather than on EOF — otherwise the
    # final arg gets dropped and the launched worker crashes with errors
    # like "Option '--ttl' requires an argument."
    local CMD_ARRAY=()
    while IFS= read -r -d '' arg; do
        CMD_ARRAY+=("$arg")
    done < <(printf '%s' "$TEMPLATE_CMD" | python3 -c \
        "import json,sys
for a in json.load(sys.stdin):
    sys.stdout.buffer.write(a.encode() + b'\0')" \
        2>/dev/null) || true
    if [ "${#CMD_ARRAY[@]}" -eq 0 ]; then
        echo "Error: failed to decode Cmd JSON from $TEMPLATE. python3 missing?" >&2
        exit 1
    fi

    # Pick the next index strictly above the highest currently running one, so we
    # never collide with a worker we're keeping. Stopped containers are removed
    # by --stop / scale-down, so we only have to look at running.
    local MAX_IDX
    MAX_IDX=$(echo "$RUNNING_NAMES" | sed "s/^${PROJECT}-//" | sort -n | tail -1)
    [ -z "$MAX_IDX" ] && MAX_IDX=0

    local TO_ADD=$((TARGET - CURRENT))
    echo "Scaling up by $TO_ADD (next index: $((MAX_IDX + 1)))..."
    for ((i = 1; i <= TO_ADD; i++)); do
        local IDX=$((MAX_IDX + i))
        local CONTAINER_NAME="${PROJECT}-${IDX}"
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
        echo "  Starting $CONTAINER_NAME..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            --label "$FLEET_LABEL=true" \
            --memory "$TEMPLATE_MEM" \
            --restart on-failure:3 \
            --add-host=host.docker.internal:host-gateway \
            -v "${MODEL_CACHE_VOLUME}:/model-cache" \
            -e PHOTOSEARCH_DEVICE=cpu \
            -e PYTHONUNBUFFERED=1 \
            -e OLLAMA_HOST=http://host.docker.internal:11434 \
            -e HF_HOME=/model-cache/huggingface \
            -e INSIGHTFACE_HOME=/model-cache/insightface \
            -e PHOTOSEARCH_CACHE=/model-cache/photosearch \
            ${LLM_ENV_DOCKER[@]+"${LLM_ENV_DOCKER[@]}"} \
            "$TEMPLATE_IMAGE" \
            "${CMD_ARRAY[@]}" \
            > /dev/null
    done

    echo ""
    echo "Done. Use --status to verify."
    exit 0
}

# ---------------------------------------------------------------------------
# Native-mode management (bare-metal worker processes, tracked via pid/log
# files under $NATIVE_RUNDIR)
# ---------------------------------------------------------------------------

do_status_native() {
    echo "=== Worker Processes (native) ==="
    echo ""
    if [ ! -d "$NATIVE_RUNDIR" ] || ! ls "$NATIVE_RUNDIR"/worker-*.pid >/dev/null 2>&1; then
        echo "No workers running."
        echo ""
        echo "Start workers with: ./run-workers.sh -s <server> -p <passes> -d <directory>"
        exit 0
    fi
    local live=0
    for pidf in "$NATIVE_RUNDIR"/worker-*.pid; do
        [ -e "$pidf" ] || continue
        local pid name
        pid=$(cat "$pidf" 2>/dev/null)
        name=$(basename "$pidf" .pid)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            echo "  $name (pid $pid) — running"
            live=$((live + 1))
        else
            echo "  $name (pid ${pid:-?}) — exited"
        fi
    done
    [ "$live" -eq 0 ] && echo "  (no live workers — all exited; queue likely drained)"
    echo ""
    echo "=== Recent Progress ==="
    echo ""
    for logf in "$NATIVE_RUNDIR"/worker-*.log; do
        [ -e "$logf" ] || continue
        echo "--- $(basename "$logf" .log) ---"
        tail -n 8 "$logf" 2>/dev/null | sed 's/^/  /'
        echo ""
    done
    exit 0
}

do_logs_native() {
    if [ ! -d "$NATIVE_RUNDIR" ] || ! ls "$NATIVE_RUNDIR"/worker-*.log >/dev/null 2>&1; then
        echo "No workers running."
        exit 0
    fi
    echo "Tailing native worker logs (Ctrl-C to stop)..."
    echo ""
    trap 'kill $(jobs -p) 2>/dev/null; exit 0' INT TERM
    for logf in "$NATIVE_RUNDIR"/worker-*.log; do
        [ -e "$logf" ] || continue
        short=$(basename "$logf" .log | sed 's/worker-//')
        tail -f -n 10 "$logf" | sed "s/^/[$short] /" &
    done
    wait
    exit 0
}

do_stop_native() {
    if [ ! -d "$NATIVE_RUNDIR" ] || ! ls "$NATIVE_RUNDIR"/worker-*.pid >/dev/null 2>&1; then
        echo "No workers running."
        exit 0
    fi
    local count=0
    for pidf in "$NATIVE_RUNDIR"/worker-*.pid; do
        [ -e "$pidf" ] || continue
        local pid
        pid=$(cat "$pidf" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && count=$((count + 1))
        fi
        rm -f "$pidf"
    done
    echo "Stopped $count native worker(s). Unclaimed batches reclaimed after TTL expires."
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
        --tags-model)       TAGS_MODEL="$2";       shift 2 ;;
        --verify-model)     VERIFY_MODEL="$2";     shift 2 ;;
        --text-llm-url)     TEXT_LLM_URL="$2";        shift 2 ;;
        --llm-describe-model) LLM_DESCRIBE_MODEL="$2"; shift 2 ;;
        --llm-verify-model)   LLM_VERIFY_MODEL="$2";   shift 2 ;;
        --llm-visual-model)   LLM_VISUAL_MODEL="$2";   shift 2 ;;
        --llm-text-model)     LLM_TEXT_MODEL="$2";     shift 2 ;;
        --native)           MODE="native";         shift ;;
        --docker)           MODE="docker";         shift ;;
        --name)             FLEET_NAME="$2";       shift 2 ;;
        --ollama-host)      OLLAMA_HOST_OVERRIDE="$2"; shift 2 ;;
        --status)           ACTION="status";       shift ;;
        --logs)             ACTION="logs";         shift ;;
        --stop)             ACTION="stop";         shift ;;
        --scale)            ACTION="scale"; SCALE_TARGET="${2:-}"; shift 2 ;;
        -h|--help)          usage ;;
        *)  echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Resolve execution mode + Ollama URL now that flags are parsed.
detect_mode
OLLAMA_URL="$(resolve_ollama_url)"

# Assemble the OpenAI-compatible (LM Studio / llama-server) routing env from the
# --text-llm-url / --llm-*-model flags. Only vars the user actually set are
# included, so an unset role falls back to describe.py's own default. Two forms:
# KEY=VAL pairs for the native `env` prefix, and `-e KEY=VAL` for `docker run`.
LLM_ENV_PAIRS=()
[ -n "$TEXT_LLM_URL" ]       && LLM_ENV_PAIRS+=("PHOTOSEARCH_TEXT_LLM_URL=$TEXT_LLM_URL")
[ -n "$LLM_DESCRIBE_MODEL" ] && LLM_ENV_PAIRS+=("PHOTOSEARCH_LLM_DESCRIBE_MODEL=$LLM_DESCRIBE_MODEL")
[ -n "$LLM_VERIFY_MODEL" ]   && LLM_ENV_PAIRS+=("PHOTOSEARCH_LLM_VERIFY_MODEL=$LLM_VERIFY_MODEL")
[ -n "$LLM_VISUAL_MODEL" ]   && LLM_ENV_PAIRS+=("PHOTOSEARCH_LLM_VISUAL_MODEL=$LLM_VISUAL_MODEL")
[ -n "$LLM_TEXT_MODEL" ]     && LLM_ENV_PAIRS+=("PHOTOSEARCH_LLM_TEXT_MODEL=$LLM_TEXT_MODEL")
LLM_ENV_DOCKER=()
for _kv in ${LLM_ENV_PAIRS[@]+"${LLM_ENV_PAIRS[@]}"}; do
    LLM_ENV_DOCKER+=("-e" "$_kv")
done
if [ -n "$TEXT_LLM_URL" ]; then
    echo "  LLM passes route to OpenAI-compatible backend: $TEXT_LLM_URL"
fi

# Apply the fleet-instance suffix (from --name) to everything that distinguishes
# one fleet from another: native pid/log dir, docker container-name prefix, and
# the docker label used by --status/--logs/--stop/--scale. Empty FLEET_NAME keeps
# the original unsuffixed names, so existing single-fleet usage is unchanged.
FLEET_LABEL="photosearch-worker-fleet${FLEET_NAME:+-$FLEET_NAME}"
PROJECT="photosearch-worker${FLEET_NAME:+-$FLEET_NAME}"
NATIVE_RUNDIR="/tmp/photosearch-worker-fleet${FLEET_NAME:+-$FLEET_NAME}"

# Management actions need no --server; dispatch them mode-aware and exit.
case "$ACTION" in
    status) [ "$MODE" = "native" ] && do_status_native || do_status ;;
    logs)   [ "$MODE" = "native" ] && do_logs_native   || do_logs   ;;
    stop)
        if [ "$MODE" = "native" ]; then do_stop_native; else do_stop; fi ;;
    scale)
        if [ "$MODE" = "native" ]; then
            echo "Error: --scale is Docker-only. In native mode, --stop and re-run with -n N." >&2
            exit 1
        fi
        do_scale "$SCALE_TARGET" ;;
esac

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
if [ -n "$TAGS_MODEL" ]; then
    WORKER_CMD="$WORKER_CMD --tags-model $TAGS_MODEL"
fi
if [ -n "$VERIFY_MODEL" ]; then
    WORKER_CMD="$WORKER_CMD --verify-model $VERIFY_MODEL"
fi

SCOPE=""
if [ -n "$DIRECTORY" ]; then SCOPE=" (directory: $DIRECTORY)"; fi
if [ -n "$COLLECTION" ]; then SCOPE=" (collection: $COLLECTION)"; fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

echo "=== Photo Search Worker Fleet ==="
echo ""
echo "  Server:     $SERVER"
echo "  Passes:     $PASSES"
echo "  Workers:    $NUM_WORKERS"
echo "  Mode:       $MODE$([ "$MODE" = docker ] && echo " (${MEM_LIMIT}/worker)")"
if ollama_needed; then echo "  Ollama:     $OLLAMA_URL"; fi
echo "  Scope:      ${SCOPE:- all photos}"
echo ""

# ---------------------------------------------------------------------------
# Prepare runtime (docker: build image; native: locate venv)
# ---------------------------------------------------------------------------

if [ "$MODE" = "docker" ]; then
    IMAGE_TAG="photosearch-worker:latest"
    # First build: base image pull + pip install ~5-10min. requirements.txt
    # changed: pip layer rebuild ~5min. Python-only change: COPY layers ~10s.
    if ! docker image inspect "$IMAGE_TAG" > /dev/null 2>&1; then
        echo "Building worker image (first build — expect ~5-10 min)..."
    else
        echo "Building worker image (incremental — ~10s for Python-only changes)..."
    fi
    echo ""
    BUILD_START=$(date +%s)
    docker build -t "$IMAGE_TAG" .
    echo ""
    echo "  Image ready: $IMAGE_TAG (built in $(($(date +%s) - BUILD_START))s)"
    echo ""
else
    VENV_PYTHON="$(find_venv_python)" || {
        echo "Error: native mode needs the project venv (.venv / venv / env)." >&2
        echo "Create it and install requirements, or pass --docker." >&2
        exit 1
    }
    echo "  Python:     $VENV_PYTHON"
    echo ""
fi

# ---------------------------------------------------------------------------
# Stop any existing workers from a previous run
# ---------------------------------------------------------------------------

if [ "$MODE" = "docker" ]; then
    EXISTING=$(docker ps -a --filter "label=$FLEET_LABEL" -q 2>/dev/null | wc -l | tr -d ' ')
    if [ "$EXISTING" -gt 0 ]; then
        echo "Stopping $EXISTING existing worker(s)..."
        docker ps -a --filter "label=$FLEET_LABEL" -q 2>/dev/null | xargs -r docker rm -f > /dev/null 2>&1
        echo ""
    fi
else
    mkdir -p "$NATIVE_RUNDIR"
    if ls "$NATIVE_RUNDIR"/worker-*.pid >/dev/null 2>&1; then
        echo "Stopping existing native worker(s)..."
        for pidf in "$NATIVE_RUNDIR"/worker-*.pid; do
            [ -e "$pidf" ] || continue
            oldpid=$(cat "$pidf" 2>/dev/null)
            [ -n "$oldpid" ] && kill "$oldpid" 2>/dev/null || true
            rm -f "$pidf"
        done
        echo ""
    fi
fi

# ---------------------------------------------------------------------------
# Ensure Ollama is running if any pass needs it
# ---------------------------------------------------------------------------

if ollama_needed; then
    echo "Passes need Ollama — checking availability + GPU config..."
    ensure_ollama_running
    if ollama_is_reachable; then
        ensure_ollama_models
        check_ollama_gpu
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Launch N workers
# ---------------------------------------------------------------------------

echo "Starting $NUM_WORKERS workers..."
echo ""

if [ "$MODE" = "docker" ]; then
    for i in $(seq 1 "$NUM_WORKERS"); do
        CONTAINER_NAME="${PROJECT}-${i}"
        docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
        echo "  Starting $CONTAINER_NAME..."
        docker run -d \
            --name "$CONTAINER_NAME" \
            --label "$FLEET_LABEL=true" \
            --memory "$MEM_LIMIT" \
            --restart on-failure:3 \
            --add-host=host.docker.internal:host-gateway \
            -v "${MODEL_CACHE_VOLUME}:/model-cache" \
            -e PHOTOSEARCH_DEVICE=cpu \
            -e PYTHONUNBUFFERED=1 \
            -e OLLAMA_HOST=http://host.docker.internal:11434 \
            -e HF_HOME=/model-cache/huggingface \
            -e INSIGHTFACE_HOME=/model-cache/insightface \
            -e PHOTOSEARCH_CACHE=/model-cache/photosearch \
            ${LLM_ENV_DOCKER[@]+"${LLM_ENV_DOCKER[@]}"} \
            "$IMAGE_TAG" \
            $WORKER_CMD \
            > /dev/null
    done
else
    # Native: bare-metal venv processes. PHOTOSEARCH_DEVICE is left unset so the
    # worker auto-detects the GPU (CUDA/ROCm). HSA_ENABLE_DXG_DETECTION is
    # harmless off-WSL and required for ROCm-via-librocdxg on WSL2.
    cd "$SCRIPT_DIR"
    for i in $(seq 1 "$NUM_WORKERS"); do
        LOGF="$NATIVE_RUNDIR/worker-$i.log"
        echo "  Starting worker-$i → $LOGF"
        HSA_ENABLE_DXG_DETECTION=1 OLLAMA_HOST="$OLLAMA_URL" PYTHONUNBUFFERED=1 \
            nohup env ${LLM_ENV_PAIRS[@]+"${LLM_ENV_PAIRS[@]}"} \
            "$VENV_PYTHON" cli.py $WORKER_CMD > "$LOGF" 2>&1 &
        echo $! > "$NATIVE_RUNDIR/worker-$i.pid"
    done
fi

echo ""
echo "=== $NUM_WORKERS workers launched ==="
echo ""
NAME_ARG="${FLEET_NAME:+--name $FLEET_NAME }"
echo "Monitor:"
echo "  ./run-workers.sh ${NAME_ARG}--status    # container status + memory + recent progress"
echo "  ./run-workers.sh ${NAME_ARG}--logs      # tail all worker logs live"
echo "  ./run-workers.sh ${NAME_ARG}--stop      # stop all workers"
echo ""
echo "Workers will exit automatically when the queue is empty."

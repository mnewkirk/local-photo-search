#!/usr/bin/env bash
#
# Idle-gated worker fleet for the WSL2 + Windows-Ollama setup.
#
# The 7900 XTX drives the desktop display AND runs Ollama, so when the machine
# is actively used, Windows' GPU scheduler preempts the ROCm compute queue and
# Ollama's runner stalls/wedges. This controller only runs a run-workers.sh
# fleet while the Windows desktop is IDLE, and stops it the moment you return —
# so the queue drains during idle periods (overnight, breaks, AFK) with zero GPU
# contention while you're working. The describe.py defer-on-timeout behavior is
# the safety net: a photo that's mid-flight when the fleet is stopped just gets
# re-claimed and retried on the next idle window — never written empty.
#
# Usage:
#   ./run-when-idle.sh [--idle SECONDS] [--poll SECONDS] -- <run-workers.sh start args>
#
# The args after `--` are the normal run-workers.sh start command and MUST
# include a --name so this controller can stop that fleet independently, e.g.:
#
#   ./run-when-idle.sh --idle 180 --poll 10 -- \
#       -s http://192.168.1.237:8000 --name gpu \
#       -p category-content,keywords --ollama-host http://172.20.176.1:11434 -n 2
#
# Run it in the foreground (Ctrl-C stops the controller AND the fleet) or
# detached:  nohup ./run-when-idle.sh ... > /tmp/idle-gate.log 2>&1 &
set -euo pipefail

IDLE_THRESHOLD=180   # seconds of Windows input-idle before the fleet may start
POLL=10              # seconds between idle checks (also ~max contention after you return)

FLEET_ARGS=()
seen_sep=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --idle) IDLE_THRESHOLD="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        --)     seen_sep=1; shift; FLEET_ARGS=("$@"); break ;;
        *)      echo "Unknown controller option: $1 (put run-workers.sh args after --)" >&2; exit 1 ;;
    esac
done

if [ "$seen_sep" -ne 1 ] || [ "${#FLEET_ARGS[@]}" -eq 0 ]; then
    echo "Error: pass the run-workers.sh start command after --." >&2
    echo "Example: ./run-when-idle.sh --idle 180 -- -s http://NAS:8000 --name gpu -p category-content,keywords --ollama-host http://GPU:11434 -n 2" >&2
    exit 1
fi

# Extract --name so we can stop the right fleet (run-workers.sh keys pid/log
# files and docker labels off it).
FLEET_NAME=""
for ((i = 0; i < ${#FLEET_ARGS[@]}; i++)); do
    [[ "${FLEET_ARGS[$i]}" == "--name" ]] && FLEET_NAME="${FLEET_ARGS[$((i + 1))]}"
done
if [ -z "$FLEET_NAME" ]; then
    echo "Error: the run-workers.sh args must include '--name NAME' so this controller can manage that fleet independently." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="/tmp/photosearch-worker-fleet-${FLEET_NAME}"

# Windows input-idle seconds via GetLastInputInfo. On any failure we print 0,
# i.e. "treat as active" — fail safe (never start the fleet when we can't
# confirm you're away).
windows_idle_seconds() {
    powershell.exe -NoProfile -Command '
Add-Type @"
using System; using System.Runtime.InteropServices;
public static class IdleAPI {
  [StructLayout(LayoutKind.Sequential)] public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
  [DllImport("user32.dll")] public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
  [DllImport("kernel32.dll")] public static extern uint GetTickCount();
  public static uint S() { LASTINPUTINFO l=new LASTINPUTINFO(); l.cbSize=(uint)Marshal.SizeOf(l); GetLastInputInfo(ref l); return (GetTickCount()-l.dwTime)/1000; }
}
"@
[IdleAPI]::S()' 2>/dev/null | tr -dc '0-9'
}

fleet_running() {
    ls "$RUNDIR"/worker-*.pid >/dev/null 2>&1 || return 1
    for p in "$RUNDIR"/worker-*.pid; do
        [ -e "$p" ] || continue
        kill -0 "$(cat "$p" 2>/dev/null)" 2>/dev/null && return 0
    done
    return 1
}

start_fleet() { "$SCRIPT_DIR/run-workers.sh" "${FLEET_ARGS[@]}" >/dev/null 2>&1 || true; }
stop_fleet()  { "$SCRIPT_DIR/run-workers.sh" --name "$FLEET_NAME" --stop >/dev/null 2>&1 || true; }

cleanup() { echo; echo "[$(date +%H:%M:%S)] controller exiting — stopping fleet '$FLEET_NAME'"; stop_fleet; exit 0; }
trap cleanup INT TERM

echo "Idle-gating fleet '$FLEET_NAME': run when Windows idle >= ${IDLE_THRESHOLD}s, pause when active (poll ${POLL}s)."
echo "Fleet command: run-workers.sh ${FLEET_ARGS[*]}"
echo "Ctrl-C to stop the controller and the fleet."
# Start from a known state: stop any pre-existing instance of this fleet.
stop_fleet

while true; do
    idle="$(windows_idle_seconds)"; idle="${idle:-0}"
    if [ "$idle" -ge "$IDLE_THRESHOLD" ]; then
        if ! fleet_running; then
            echo "[$(date +%H:%M:%S)] idle ${idle}s >= ${IDLE_THRESHOLD}s — starting fleet"
            start_fleet
        fi
    else
        if fleet_running; then
            echo "[$(date +%H:%M:%S)] active (idle ${idle}s) — stopping fleet"
            stop_fleet
        fi
    fi
    sleep "$POLL"
done

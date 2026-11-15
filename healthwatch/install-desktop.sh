#!/bin/bash
# Install the desktop health watcher (systemd user timer). Run as yourself,
# after install-nas.sh (it copies the ntfy topic from the NAS):
#   bash healthwatch/install-desktop.sh
set -euo pipefail
cd "$(dirname "$0")/.."
NAS_SSH="${HEALTHWATCH_NAS_SSH:-cantimatt@192.168.1.237}"
ENVF="$HOME/.config/healthwatch.env"
UNITS="$HOME/.config/systemd/user"

topic=$(ssh -o BatchMode=yes "$NAS_SSH" "grep -h ^HEALTHWATCH_NTFY_TOPIC= /etc/healthwatch.env" | cut -d= -f2)
[ -n "$topic" ] || { echo "no topic on the NAS: run install-nas.sh there first" >&2; exit 1; }
mkdir -p "$(dirname "$ENVF")" "$UNITS"
printf 'HEALTHWATCH_ROLE=desktop\nHEALTHWATCH_NTFY_TOPIC=%s\n' "$topic" > "$ENVF"

install -m 644 healthwatch/systemd/healthwatch-desktop.service "$UNITS/healthwatch-desktop.service"
install -m 644 healthwatch/systemd/healthwatch-desktop.timer "$UNITS/healthwatch-desktop.timer"
systemctl --user daemon-reload
systemctl --user enable --now healthwatch-desktop.timer
systemctl --user start healthwatch-desktop.service || true
python3 healthwatch/healthwatch.py status --role desktop

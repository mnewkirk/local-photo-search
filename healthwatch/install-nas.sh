#!/bin/bash
# Install the NAS health watcher. Run on the NAS, as root, from the repo:
#   sudo bash /volume1/docker/photosearch/healthwatch/install-nas.sh
# Idempotent: re-run after changing the unit files.
set -euo pipefail
REPO=/volume1/docker/photosearch
STATE=/volume1/docker/healthwatch
ENVF=/etc/healthwatch.env

[ "$(id -u)" = 0 ] || { echo "run with sudo" >&2; exit 1; }
cd "$REPO"

if [ ! -f "$ENVF" ]; then
    topic="photosearch-$(head -c 9 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    printf 'HEALTHWATCH_ROLE=nas\nHEALTHWATCH_NTFY_TOPIC=%s\n' "$topic" > "$ENVF"
    echo "created $ENVF"
fi
chmod 644 "$ENVF"   # the desktop watcher reads the topic over ssh; it is not a secret on the LAN/tailnet

mkdir -p "$STATE/ntfy-cache" "$STATE/incidents"
chmod 755 "$STATE"  # the desktop watcher reads status.md as the ssh user

docker compose -p healthwatch -f healthwatch/docker-compose.yml up -d

install -m 644 healthwatch/systemd/healthwatch.service /etc/systemd/system/healthwatch.service
install -m 644 healthwatch/systemd/healthwatch.timer /etc/systemd/system/healthwatch.timer
systemctl daemon-reload
systemctl enable --now healthwatch.timer

echo "--- first cycle"
systemctl start healthwatch.service || true
cat "$STATE/status.md" 2>/dev/null || true

echo "--- test notification"
env -u GODEBUG $(cat "$ENVF" | xargs) python3 healthwatch/healthwatch.py notify-test --role nas || true

. "$ENVF"
cat <<EOF

Installed. Phone: install the ntfy app, add server http://dxp4800-f976:8090
(the phone must be on Tailscale) and subscribe to topic: $HEALTHWATCH_NTFY_TOPIC
Status:    sudo python3 $REPO/healthwatch/healthwatch.py status
Incidents: $STATE/incidents/   (newest: $STATE/latest-incident)
EOF

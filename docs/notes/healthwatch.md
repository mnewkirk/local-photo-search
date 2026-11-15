# Health watcher — notes

Runbook: .claude/skills/photo-search/reference/healthwatch.md.

### Tailscale down 4 days (2026-11-15)

Tailscale auto-updated 1.102.2 → 1.104.6 on the NAS on 2026-11-11 at 09:37.
1.104.6 is built with Go 1.27.3; 1.102.2 was Go 1.26.5. UGOS sets
`DefaultEnvironment="GODEBUG=tlskyber=0"` in /etc/systemd/system.conf and
exports it in /etc/profile.d/go.sh. Go 1.27 runtimes treat a removed GODEBUG
setting at its old value as fatal (`fatal error: removed GODEBUG "tlskyber"
set to old value "0"`), so tailscaled exited with status 2 after 12 ms on every
start, wrote nothing to the journal, and the NAS dropped off the tailnet. The
desktop replica's /batches answered 502 because its NAS URL is the MagicDNS
name `dxp4800-f976`. The MCP connector failed too.

Diagnosis took a while because the daemon ran fine under `sudo` (sudo resets
the environment); only `systemd-run --wait -p StandardOutput=file:...` showed
the fatal error. That is now `_startup_probe` evidence. Fix: drop-in
`UnsetEnvironment=GODEBUG` for tailscaled, plus pre-emptive ones for docker and
containerd. Same crash elsewhere: tailscale#21707, cloudflared (t3code#28453).

Cutoff: Go ≤1.22 doesn't know the setting, Go 1.23 honours it, 1.24–1.26 ignore
it (removed), 1.27+ abort. After UGOS 1.21.1 every UGOS tool is Go 1.26.10, so
the setting affects nothing, but UGOS must drop it before it ships Go 1.27.

### Same day: firmware 1.16 → 1.17 → 1.19.1 → 1.21.1, Docker 26.1 → 29.6.2

- All drop-ins, the crontab and ~/.bash_profile survived every update.
- 1.19.1: UGOS Sync & Backup (`syncspace`, a Syncthing fork) started before
  Docker and took 22000/21027. photosearch-syncthing failed with `bind:
  address already in use` and Docker does not retry a failed start. Moved to
  22001 (commit 5e85820). Phones are unaffected: the NAS dials them
  (`tcp-client`) or meets them via relays.
- Docker 29: UGOS wrote `"features": {"containerd-snapshotter": false}` to
  daemon.json itself, so overlay2 images stayed visible. `docker images` now
  hides untagged images (17 shown vs 296 in `docker system df`).
- Pre/post baselines: ~/nas-preupdate-2026-11-15/ on the desktop. The `drift`
  check now writes the same snapshot to /volume1/docker/healthwatch/drift/.

### Design (2026-11-15)

Owner's choices: the watcher runs on the NAS plus a desktop probe, applies
allow-listed fixes as root, notifies through self-hosted ntfy, and only writes
reports (it never starts a Claude session). It runs on the host's Python
rather than in Docker, so it keeps working when Docker is the thing that broke.
It is a timer-driven oneshot, not a daemon: each cycle loads state.json, runs
the checks, and saves state.

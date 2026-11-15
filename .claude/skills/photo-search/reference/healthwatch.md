# Health watcher (healthwatch)

`healthwatch/healthwatch.py`: one stdlib Python script with two roles, each fired by a
systemd timer. It detects breakage, applies a fix from a fixed allow-list, pushes a
notification, and writes an incident report for a later Claude session. History and
measurements: `docs/notes/healthwatch.md`.

## Start here when something is broken

1. **NAS:** `/volume1/docker/healthwatch/latest-incident/report.md`, then `status.md`
   beside it. All incidents: `/volume1/docker/healthwatch/incidents/<time>-<check>/`.
2. **Desktop:** `~/.local/state/healthwatch/latest-incident/report.md` and `status.md`.
3. Each report has what failed, known causes, a timeline (including every auto-fix and
   its output), every other check's state at that moment, and the evidence captured at
   failure time (`evidence/*.txt`). Read it before running anything.
4. Re-check without changing anything:
   `sudo python3 healthwatch/healthwatch.py check --role nas [NAME ...]` on the NAS
   (repo root `/volume1/docker/photosearch`), or `python3 healthwatch/healthwatch.py
   check --role desktop` here.

## Where it runs

| | NAS | Desktop |
|---|---|---|
| Unit | `healthwatch.timer` (root, every 2 min) | `healthwatch-desktop.timer` (user, every 5 min) |
| Code | the repo checkout, so a `git pull` updates it | the repo checkout |
| State | `/volume1/docker/healthwatch/` | `~/.local/state/healthwatch/` |
| Config | `/etc/healthwatch.env` | `~/.config/healthwatch.env` |
| Install | `sudo bash healthwatch/install-nas.sh` | `bash healthwatch/install-desktop.sh` |

Notifications go to ntfy (`healthwatch-ntfy` container, own compose project
`healthwatch`, port 8090, topic in the env file). The desktop posts to the NAS's
ntfy over the LAN, then Tailscale; when the NAS is unreachable it falls back to a
Windows toast. Undelivered notifications are kept in `state.json` (`undelivered`).

## Checks and their allow-listed fixes

| Role | Check | Fails when | Auto-fix |
|---|---|---|---|
| nas | `godebug_overrides` | UGOS sets GODEBUG and a unit lacks `no-godebug.conf` | write the drop-in, daemon-reload |
| nas | `tailscale` | tailscaled not active / not `Running` | overrides + reset-failed + restart |
| nas | `docker` | docker/containerd down or not answering | overrides + restart both (skipped while a cron lock is held) |
| nas | `containers` | an expected container missing or stopped | `docker start` (missing ones: report only) |
| nas | `photosearch_api` | `/api/health` not 200 (3 cycles) | none: restarting mid-ingest loses work |
| nas | `mcp` | `:8848/mcp` 5xx / no answer | `docker restart photosearch-mcp` |
| nas | `syncthing` | API down or nothing on 22001 | none |
| nas | `phone_sync` (warn) | no phone connected for 72 h | none |
| nas | `cron` (warn) | crontab entry missing, log stale > 26 h, or Traceback | none |
| nas | `resources` (warn) | a pool > 90 %, < 400 MB free memory, OOM kill | none |
| nas | `drift` (warn) | Go 1.27+ binary without an override; also notes firmware/package changes in `drift/` | none |
| desktop | `nas_reachable` | NAS not answering over Tailscale | mitigation: replica → LAN URL drop-in |
| desktop | `replica_nas_url` | LAN stopgap still on after Tailscale recovered | remove it, restart the replica |
| desktop | `replica` | replica unit/`/api/health` down | restart it (skipped while `cli.py worker` runs) |
| desktop | `replica_fresh` (warn) | last sync > 36 h | none |
| desktop | `nas_watcher` | NAS `status.md` older than 10 min | none |

A check opens an incident after `debounce` consecutive failures (2 unless noted), runs
its fix at most `max_fixes` times (2) and re-checks after each, and sends "Needs
attention" once when the fixes are used up. A recovered incident is closed and
announced. An open incident is re-announced once a day.

## Adding or changing a check

Write `check_x(ctx) -> Result` (`ok=None` means skipped because a dependency is down,
which neither opens nor closes an incident). Add a `Check(...)` with a `hint`: the known
causes and what to do, which is copied into every report. Optional `collect` gathers
evidence when an incident opens, and `fix` returns a log string. Fixes must be safe to
run unattended, so check for a busy reason first (`busy_reason()`, `live_worker_pids()`).
Tests: `tests/test_healthwatch.py`. A new cause found while debugging goes in the
check's `hint` and in `docs/notes/healthwatch.md`.

## Commands

```bash
sudo python3 healthwatch/healthwatch.py status            # NAS status.md
sudo python3 healthwatch/healthwatch.py snapshot          # post-update baseline (versions, overrides, ports, crontab, Go 1.27+)
sudo python3 healthwatch/healthwatch.py notify-test       # check that the phone gets pushes
sudo systemctl list-timers healthwatch.timer; sudo journalctl -u healthwatch.service -n 50
python3 healthwatch/healthwatch.py run --role desktop --no-fix   # one desktop cycle, fixes off
```

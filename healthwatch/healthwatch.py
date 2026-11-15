#!/usr/bin/env python3
"""Health watcher for the photosearch NAS and the desktop replica.

One cycle per invocation (a systemd timer fires it). Each check reports ok /
failing / skipped. A check that fails `debounce` cycles in a row opens an
incident: a directory with a self-contained report.md plus the evidence
captured AT FAILURE TIME, written so a later Claude session can start from it.
Checks may carry one allow-listed fix; it runs at most `max_fixes` times per
incident, and the check is re-run right after to see whether it worked.

Stdlib only: it runs on the NAS host's Python 3.11 as root, outside Docker, so
it keeps working when Docker is the thing that broke.

    healthwatch.py run    [--role nas|desktop] [--no-fix]   one cycle (the timer)
    healthwatch.py check  [--role ...] [NAME ...]           run checks, print, change nothing
    healthwatch.py status                                   print status.md
    healthwatch.py notify-test                              send a test notification
    healthwatch.py snapshot                                 print the NAS config baseline

Runbook: .claude/skills/photo-search/reference/healthwatch.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Configuration (environment, from /etc/healthwatch.env or the user unit)
# ---------------------------------------------------------------------------

ROLE = os.environ.get("HEALTHWATCH_ROLE", "nas")
NAS_LAN = os.environ.get("HEALTHWATCH_NAS_LAN", "192.168.1.237")
NAS_TS = os.environ.get("HEALTHWATCH_NAS_TS", "dxp4800-f976")
NAS_SSH = os.environ.get("HEALTHWATCH_NAS_SSH", f"cantimatt@{NAS_LAN}")
NAS_STATE_DIR = "/volume1/docker/healthwatch"
NTFY_TOPIC = os.environ.get("HEALTHWATCH_NTFY_TOPIC", "")
CONTAINERS = [c for c in os.environ.get(
    "HEALTHWATCH_CONTAINERS",
    "photosearch,photosearch-mcp,photosearch-syncthing,plex,unifi-object-mcp,"
    "screens-override,healthwatch-ntfy").split(",") if c]
GODEBUG_UNITS = ["tailscaled", "docker", "containerd"]
CRON_USER = os.environ.get("HEALTHWATCH_CRON_USER", "cantimatt")
CRON_JOBS = {  # crontab marker -> log file
    "ingest-incoming": "/var/log/photo-ingest.log",
    "maintenance-sweep": "/var/log/photo-maintenance.log",
}
SYNCTHING_CONFIG = os.environ.get(
    "HEALTHWATCH_SYNCTHING_CONFIG",
    "/volume1/docker/photosearch/syncthing-config/config.xml")
SYNCTHING_PORT = 22001
LOCKS = ["/tmp/photo-ingest.lock", "/tmp/photo-maintenance.lock"]
REPLICA_UNIT = "photosearch-replica.service"
REPLICA_DROPIN = Path.home() / ".config/systemd/user/photosearch-replica.service.d/nas-lan-stopgap.conf"
API_HEADERS = {"X-Photosearch-Source": "healthwatch",
               "X-Photosearch-Intent": "health watcher probe"}
INCIDENT_RETENTION_DAYS = 90
REMIND_AFTER_S = 24 * 3600


def default_state_dir(role: str) -> Path:
    env = os.environ.get("HEALTHWATCH_STATE_DIR")
    if env:
        return Path(env)
    if role == "nas":
        return Path(NAS_STATE_DIR)
    return Path.home() / ".local/state/healthwatch"


def ntfy_urls(role: str) -> list[str]:
    env = os.environ.get("HEALTHWATCH_NTFY_URLS")
    if env:
        return [u.rstrip("/") for u in env.split(",") if u]
    if role == "nas":
        return ["http://127.0.0.1:8090"]
    return [f"http://{NAS_LAN}:8090", f"http://{NAS_TS}:8090"]


# ---------------------------------------------------------------------------
# Small helpers — none of them raise
# ---------------------------------------------------------------------------

def _clean_env(extra: Optional[dict] = None) -> dict:
    # UGOS exports GODEBUG=tlskyber=0; Go 1.27+ CLIs (tailscale) die on it.
    env = {k: v for k, v in os.environ.items() if k != "GODEBUG"}
    env.update(extra or {})
    return env


def sh(cmd, timeout: int = 30) -> tuple[int, str]:
    """Run a command (list, or str through bash); return (rc, stdout+stderr)."""
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                           text=True, timeout=timeout, env=_clean_env(),
                           executable="/bin/bash" if isinstance(cmd, str) else None)
        return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s: {cmd}"
    except Exception as e:  # missing binary etc.
        return 127, f"{type(e).__name__}: {e}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


def http(url: str, timeout: int = 8, headers: Optional[dict] = None,
         data: Optional[bytes] = None) -> tuple[int, str]:
    """(status, body). status 0 = no HTTP answer at all; redirects are not followed."""
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            return r.status, r.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, (e.read(4096).decode("utf-8", "replace") if e.fp else "")
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso(t: Optional[dt.datetime] = None) -> str:
    return (t or now()).isoformat(timespec="seconds")


def age_s(stamp: Optional[str]) -> float:
    if not stamp:
        return float("inf")
    return (now() - dt.datetime.fromisoformat(stamp)).total_seconds()


def lock_held(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def busy_reason() -> Optional[str]:
    """Why a disruptive fix (Docker restart) must wait, or None."""
    held = [p for p in LOCKS if lock_held(p)]
    return f"cron job running ({', '.join(held)})" if held else None


def live_worker_pids() -> list[int]:
    """Desktop worker-fleet processes (`cli.py ... worker`), by argv, not pgrep -f."""
    pids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        names = [Path(a.decode(errors="replace")).name for a in argv]
        if "cli.py" in names and "worker" in names:
            pids.append(int(d.name))
    return pids


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

@dataclass
class Result:
    ok: Optional[bool]          # None = skipped (a dependency is down): no state change
    summary: str
    evidence: dict = field(default_factory=dict)   # name -> text, captured now


@dataclass
class Check:
    name: str
    run: Callable[["Ctx"], Result]
    hint: str                                    # known causes / runbook, goes in the report
    severity: str = "error"                      # "error" | "warn"
    collect: Optional[Callable[["Ctx"], dict]] = None   # extra evidence on incident open
    fix: Optional[Callable[["Ctx"], str]] = None        # allow-listed remediation; returns a log
    fix_label: str = ""
    debounce: int = 2
    max_fixes: int = 2
    settle_s: int = 10                           # wait after a fix before re-checking
    mitigation: bool = False                     # fix works around it; the check stays failing


@dataclass
class Ctx:
    role: str
    state_dir: Path
    fixes: bool = True
    dry: bool = False                            # `check`: no files, no notifications
    state: dict = field(default_factory=dict)
    notifier: Optional[Callable[[str, str, str, list], bool]] = None
    sleep: Callable[[float], None] = time.sleep

    @property
    def incidents_dir(self) -> Path:
        return self.state_dir / "incidents"


def _safe_run(c: Check, ctx: Ctx) -> Result:
    try:
        return c.run(ctx)
    except Exception as e:
        return Result(False, f"check crashed: {type(e).__name__}: {e}")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


class Incident:
    """One incident directory: incident.json is the source, report.md is rendered."""

    def __init__(self, path: Path, data: dict):
        self.path, self.data = path, data

    @classmethod
    def open(cls, ctx: Ctx, c: Check, r: Result, snapshot: dict) -> "Incident":
        t = now()
        path = ctx.incidents_dir / f"{t:%Y%m%d-%H%M%S}-{_slug(c.name)}"
        (path / "evidence").mkdir(parents=True, exist_ok=True)
        inc = cls(path, {
            "id": path.name, "check": c.name, "role": ctx.role,
            "host": socket.gethostname(), "severity": c.severity,
            "status": "OPEN", "opened": iso(t), "closed": None, "resolution": None,
            "summary": r.summary, "hint": c.hint, "fix_label": c.fix_label,
            "fix_attempts": 0, "max_fixes": c.max_fixes if c.fix else 0,
            "timeline": [[iso(t), f"Opened: {r.summary}"]],
            "evidence": [], "snapshot": snapshot,
        })
        evidence = dict(r.evidence)
        if c.collect:
            try:
                evidence.update(c.collect(ctx))
            except Exception as e:
                evidence["collect-error"] = f"{type(e).__name__}: {e}"
        for name, text in evidence.items():
            inc.add_evidence(name, text)
        inc.save()
        return inc

    @classmethod
    def load(cls, path: Path) -> Optional["Incident"]:
        try:
            return cls(path, json.loads((path / "incident.json").read_text()))
        except (OSError, ValueError):
            return None

    def add_evidence(self, name: str, text: str) -> str:
        fname = f"{_slug(name) or 'evidence'}.txt"
        n = 2
        while fname in self.data["evidence"]:
            fname = f"{_slug(name)}-{n}.txt"
            n += 1
        (self.path / "evidence" / fname).write_text(text if text.endswith("\n") else text + "\n")
        self.data["evidence"].append(fname)
        return fname

    def log(self, text: str):
        self.data["timeline"].append([iso(), text])

    def close(self, resolution: str):
        self.data.update(status="RESOLVED", closed=iso(), resolution=resolution)
        self.log(f"Resolved: {resolution}")

    def save(self):
        (self.path / "incident.json").write_text(json.dumps(self.data, indent=2))
        (self.path / "report.md").write_text(render_report(self.data, self.path))


def render_report(d: dict, path: Path) -> str:
    ev_dir = path / "evidence"
    lines = [
        f"# Incident {d['id']}: {d['check']} — {d['status']}",
        "",
        f"- **What failed:** {d['summary']}",
        f"- **Check:** `{d['check']}` ({d['severity']}) on `{d['host']}` (role `{d['role']}`)",
        f"- **Opened:** {d['opened']}" + (f" · **Closed:** {d['closed']}" if d["closed"] else ""),
    ]
    if d["resolution"]:
        lines.append(f"- **Resolution:** {d['resolution']}")
    if d["max_fixes"]:
        lines.append(f"- **Auto-fix:** {d['fix_label']} — {d['fix_attempts']} of "
                     f"{d['max_fixes']} attempts used")
    else:
        lines.append("- **Auto-fix:** none for this check (report only)")
    lines += ["", "## Known causes and runbook", "", d["hint"].strip(), "",
              "## Timeline", ""]
    lines += [f"- `{t}` {text}" for t, text in d["timeline"]]
    lines += ["", "## Every check at the time it opened", "",
              "| check | state | summary |", "|---|---|---|"]
    for name, (state, summary) in sorted(d["snapshot"].items()):
        lines.append(f"| {name} | {state} | {summary.replace('|', '/')} |")
    lines += ["", "## Evidence (captured when it happened)", ""]
    for fname in d["evidence"]:
        p = ev_dir / fname
        try:
            text = p.read_text()
        except OSError:
            text = "(missing)"
        tail = text if len(text) <= 3000 else "…(head trimmed; full file in evidence/)\n" + text[-3000:]
        lines += [f"### {fname}", "", "```", tail.rstrip(), "```", ""]
    lines += [
        "## For a Claude session",
        "",
        "Start here. The evidence above was captured when the check failed and may",
        "no longer be reproducible; full files are in `evidence/` next to this report.",
        "",
        f"1. Re-run just this check (changes nothing): `python3 healthwatch/healthwatch.py "
        f"check --role {d['role']} {d['check']}` from the repo root"
        + (" as root on the NAS" if d["role"] == "nas" else "") + ".",
        "2. Auto-fixes already tried are in the timeline; do not repeat one blindly.",
        "3. Known causes above come from past incidents (docs/notes/healthwatch.md).",
        "   When you find a new cause, add it to the check's `hint` and that notes file.",
        "4. The runbook is .claude/skills/photo-search/reference/healthwatch.md.",
        "",
    ]
    return "\n".join(lines)


def run_cycle(ctx: Ctx, checks: list[Check]) -> dict:
    """Run every check once and advance the incident state machine. Returns the results."""
    st_all = ctx.state.setdefault("checks", {})
    results = {c.name: _safe_run(c, ctx) for c in checks}
    snapshot = {n: ({True: "ok", False: "FAILING", None: "skipped"}[r.ok], r.summary)
                for n, r in results.items()}

    for c in checks:
        r = results[c.name]
        st = st_all.setdefault(c.name, {"fails": 0, "incident": None})
        st["last"] = [iso(), snapshot[c.name][0], r.summary]
        if r.ok is None:
            continue
        inc = Incident.load(Path(st["incident"])) if st.get("incident") else None
        if r.ok:
            if inc:
                inc.close("recovered on its own (or fixed by hand)")
                inc.save()
                notify(ctx, f"Recovered: {c.name}", r.summary, "default", ["white_check_mark"])
            st.update(fails=0, incident=None)
            continue

        st["fails"] = st.get("fails", 0) + 1
        if st["fails"] < c.debounce and not inc:
            continue
        if not inc:
            inc = Incident.open(ctx, c, r, snapshot)
            st.update(incident=str(inc.path), reminded=iso())
            _point_latest(ctx, inc)
            can_fix = bool(c.fix and ctx.fixes)
            notify(ctx, f"{'Problem' if c.severity == 'error' else 'Warning'}: {c.name}",
                   f"{r.summary}\n{'Trying: ' + c.fix_label if can_fix else 'No auto-fix.'}"
                   f"\nReport: {inc.path / 'report.md'}",
                   "high" if c.severity == "error" else "default",
                   ["rotating_light" if c.severity == "error" else "warning"])
        elif r.summary != inc.data["timeline"][-1][1].split(": ", 1)[-1]:
            inc.log(f"Still failing: {r.summary}")

        if c.fix and ctx.fixes and inc.data["fix_attempts"] < c.max_fixes:
            inc.data["fix_attempts"] += 1
            n = inc.data["fix_attempts"]
            try:
                out = c.fix(ctx)
            except Exception as e:
                out = f"fix crashed: {type(e).__name__}: {e}"
            fname = inc.add_evidence(f"fix-{n}-output", out or "(no output)")
            inc.log(f"Fix {n}/{c.max_fixes} ({c.fix_label}) ran — output in evidence/{fname}")
            ctx.sleep(c.settle_s)
            r2 = _safe_run(c, ctx)
            if r2.ok:
                inc.close(f"auto-fixed by: {c.fix_label} (attempt {n})")
                inc.save()
                notify(ctx, f"Fixed: {c.name}", f"{c.fix_label} worked.\n{r2.summary}",
                       "default", ["wrench"])
                st.update(fails=0, incident=None)
                continue
            if c.mitigation:
                inc.log(f"Mitigation {n} applied; the underlying problem remains: {r2.summary}")
                notify(ctx, f"Worked around: {c.name}",
                       f"{c.fix_label}. Still needs fixing: {r2.summary}\n"
                       f"Report: {inc.path / 'report.md'}", "high", ["construction"])
            else:
                inc.log(f"Fix {n} did not resolve it: {r2.summary}")
            if n >= c.max_fixes and not c.mitigation:
                notify(ctx, f"Needs attention: {c.name}",
                       f"Auto-fix failed {n}x. {r2.summary}\nReport: {inc.path / 'report.md'}",
                       "urgent", ["sos"])
        elif age_s(st.get("reminded")) > REMIND_AFTER_S:
            st["reminded"] = iso()
            notify(ctx, f"Still open: {c.name}", f"{r.summary}\nReport: {inc.path / 'report.md'}",
                   "default", ["hourglass"])
        inc.save()

    ctx.state["last_run"] = iso()
    write_status(ctx, checks, results)
    return results


def _point_latest(ctx: Ctx, inc: Incident):
    link = ctx.state_dir / "latest-incident"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(inc.path)
    except OSError:
        pass


def write_status(ctx: Ctx, checks: list[Check], results: dict):
    st_all = ctx.state.get("checks", {})
    lines = [f"# Health status — {ctx.role} — {iso()}", "",
             "| check | state | summary | open incident |", "|---|---|---|---|"]
    for c in checks:
        r = results[c.name]
        state = {True: "ok", False: "FAILING", None: "skipped"}[r.ok]
        inc = st_all.get(c.name, {}).get("incident") or ""
        lines.append(f"| {c.name} | {state} | {r.summary.replace('|', '/')} | "
                     f"{inc + '/report.md' if inc else ''} |")
    lines += ["", f"Incidents: {ctx.incidents_dir}",
              "Runbook: .claude/skills/photo-search/reference/healthwatch.md", ""]
    ctx.state_dir.mkdir(parents=True, exist_ok=True)
    (ctx.state_dir / "status.md").write_text("\n".join(lines))


def prune_incidents(ctx: Ctx):
    cutoff = time.time() - INCIDENT_RETENTION_DAYS * 86400
    if not ctx.incidents_dir.is_dir():
        return
    open_paths = {s.get("incident") for s in ctx.state.get("checks", {}).values()}
    for d in ctx.incidents_dir.iterdir():
        if d.is_dir() and str(d) not in open_paths and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Notifications: ntfy, then (desktop) a Windows toast when the NAS is unreachable
# ---------------------------------------------------------------------------

def notify(ctx: Ctx, title: str, body: str, priority: str, tags: list) -> bool:
    if ctx.notifier:
        return ctx.notifier(title, body, priority, tags)
    full_title = f"[{ctx.role}] {title}"
    if NTFY_TOPIC:
        for base in ntfy_urls(ctx.role):
            code, _ = http(f"{base}/{NTFY_TOPIC}", timeout=6, data=body.encode(),
                           headers={"Title": full_title, "Priority": priority,
                                    "Tags": ",".join(tags)})
            if 200 <= code < 300:
                return True
    if ctx.role == "desktop" and _windows_toast(full_title, body):
        return True
    ctx.state.setdefault("undelivered", []).append([iso(), full_title, body])
    ctx.state["undelivered"] = ctx.state["undelivered"][-50:]
    return False


def _windows_toast(title: str, body: str) -> bool:
    ps = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    if not os.path.exists(ps):
        return False
    esc = lambda s: s.replace("'", "''").replace("\n", " ")[:250]
    script = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
        "ContentType = WindowsRuntime] > $null;"
        "$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$x=$t.GetElementsByTagName('text');"
        f"$x.Item(0).AppendChild($t.CreateTextNode('{esc(title)}'))>$null;"
        f"$x.Item(1).AppendChild($t.CreateTextNode('{esc(body)}'))>$null;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe')"
        ".Show([Windows.UI.Notifications.ToastNotification]::new($t))")
    rc, _ = sh([ps, "-NoProfile", "-NonInteractive", "-Command", script], timeout=20)
    return rc == 0


# ---------------------------------------------------------------------------
# NAS checks
# ---------------------------------------------------------------------------

GODEBUG_HINT = """\
UGOS sets `GODEBUG=tlskyber=0` for every systemd service (`DefaultEnvironment` in
/etc/systemd/system.conf) and in login shells (/etc/profile.d/go.sh). Go 1.27+
binaries refuse to start with it: `fatal error: removed GODEBUG "tlskyber"`, exit
status 2 within milliseconds, and nothing in the journal. The fix is a drop-in
`/etc/systemd/system/<unit>.service.d/no-godebug.conf` containing
`[Service]` / `UnsetEnvironment=GODEBUG`. The `startup-probe` evidence runs the
binary as a plain systemd service WITHOUT the unit's drop-ins, so it shows the
crash even when the drop-in is in place; compare it with the unit's own status.
History: docs/notes/healthwatch.md "Tailscale down 4 days (2026-11-15)"."""


def _startup_probe(binary: str, arg: str = "--version") -> str:
    """Run `binary arg` as a transient systemd service and capture its output:
    the journal is often empty for a unit that dies at startup."""
    out = "/run/healthwatch-probe.log"
    rc, meta = sh(["systemd-run", "--wait", "--collect", "--quiet",
                   "-p", f"StandardOutput=file:{out}", "-p", f"StandardError=file:{out}",
                   binary, arg], timeout=30)
    try:
        text = Path(out).read_text()
        Path(out).unlink()
    except OSError:
        text = ""
    return (f"$ systemd-run {binary} {arg}   (manager default env, no unit drop-ins)\n"
            f"rc={rc} {meta}\n{text}")


def _unit_evidence(unit: str, binary: str) -> dict:
    return {
        f"{unit}-status": sh(["systemctl", "status", unit, "--no-pager", "-l"])[1],
        f"{unit}-journal": sh(["journalctl", "-u", unit, "-n", "80", "--no-pager"])[1],
        f"{unit}-startup-probe": _startup_probe(binary),
        "manager-environment": sh(["systemctl", "show-environment"])[1],
    }


def _system_godebug() -> bool:
    return "GODEBUG=" in sh(["systemctl", "show-environment"])[1]


def check_godebug_overrides(ctx: Ctx) -> Result:
    if not _system_godebug():
        return Result(True, "UGOS no longer sets GODEBUG; overrides not needed")
    missing = [u for u in GODEBUG_UNITS
               if "GODEBUG" not in sh(["systemctl", "show", u, "-p", "UnsetEnvironment",
                                       "--value"])[1]]
    if missing:
        return Result(False, f"GODEBUG override missing for: {', '.join(missing)}")
    return Result(True, f"overrides active for {', '.join(GODEBUG_UNITS)}")


def fix_godebug_overrides(ctx: Ctx) -> str:
    log = []
    for u in GODEBUG_UNITS:
        d = Path(f"/etc/systemd/system/{u}.service.d")
        f = d / "no-godebug.conf"
        if not f.exists():
            d.mkdir(parents=True, exist_ok=True)
            f.write_text("[Service]\n# UGOS sets GODEBUG=tlskyber=0 globally (system.conf); "
                         "Go 1.27+ binaries refuse to start with it.\nUnsetEnvironment=GODEBUG\n")
            log.append(f"wrote {f}")
    log.append("daemon-reload: " + sh(["systemctl", "daemon-reload"])[1])
    return "\n".join(log) + "\n(takes effect at each unit's next start; no restart done)"


def check_tailscale(ctx: Ctx) -> Result:
    rc, active = sh(["systemctl", "is-active", "tailscaled"])
    if active != "active":
        return Result(False, f"tailscaled is {active}")
    rc, out = sh(["tailscale", "status", "--json"], timeout=15)
    try:
        state = json.loads(out).get("BackendState")
    except ValueError:
        return Result(False, f"tailscale status unreadable: {out[:200]}")
    if state != "Running":
        return Result(False, f"tailscaled running but BackendState={state}")
    return Result(True, "tailscaled running, connected")


def collect_tailscale(ctx: Ctx) -> dict:
    ev = _unit_evidence("tailscaled", "/usr/sbin/tailscaled")
    ev["tailscale-status"] = sh(["tailscale", "status"])[1]
    ev["dpkg-tailscale"] = sh("grep -h tailscale /var/log/dpkg.log* | tail -20")[1]
    ev["tailscale-debs-cached"] = sh("ls -la /var/cache/apt/archives/ | grep -i tailscale")[1]
    return ev


def fix_tailscale(ctx: Ctx) -> str:
    out = [fix_godebug_overrides(ctx)]
    out.append(sh(["systemctl", "reset-failed", "tailscaled"])[1])
    rc, o = sh(["systemctl", "restart", "tailscaled"], timeout=60)
    out.append(f"systemctl restart tailscaled rc={rc} {o}")
    return "\n".join(out)


def check_docker(ctx: Ctx) -> Result:
    bad = [f"{u}={s}" for u in ("containerd", "docker")
           if (s := sh(["systemctl", "is-active", u])[1]) != "active"]
    if bad:
        return Result(False, "not active: " + ", ".join(bad))
    rc, out = sh(["docker", "version", "--format", "{{.Server.Version}}"], timeout=20)
    if rc != 0:
        return Result(False, f"docker daemon not answering: {out[:200]}")
    return Result(True, f"docker {out} answering")


def collect_docker(ctx: Ctx) -> dict:
    ev = _unit_evidence("docker", "/usr/bin/dockerd")
    ev.update({k.replace("docker", "containerd", 1): v for k, v in
               _unit_evidence("containerd", "/usr/bin/containerd").items()
               if not k.startswith("manager")})
    ev["daemon-json"] = sh(["cat", "/etc/docker/daemon.json"])[1]
    return ev


def fix_docker(ctx: Ctx) -> str:
    why = busy_reason()
    if why:
        return f"skipped: {why}; a Docker restart would kill it"
    out = [fix_godebug_overrides(ctx)]
    for u in ("containerd", "docker"):
        sh(["systemctl", "reset-failed", u])
        rc, o = sh(["systemctl", "restart", u], timeout=120)
        out.append(f"systemctl restart {u} rc={rc} {o}")
    return "\n".join(out)


def _docker_up() -> bool:
    return sh(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=20)[0] == 0


def _container_states() -> dict:
    rc, out = sh(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.State}}\t{{.Status}}"])
    states = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 3:
            states[parts[0]] = (parts[1], parts[2])
    return states


def check_containers(ctx: Ctx) -> Result:
    if not _docker_up():
        return Result(None, "skipped: docker is down")
    states = _container_states()
    missing = [c for c in CONTAINERS if c not in states]
    down = [f"{c} ({states[c][1]})" for c in CONTAINERS
            if c in states and states[c][0] != "running"]
    if missing or down:
        return Result(False, "; ".join(
            ([f"missing: {', '.join(missing)}"] if missing else []) +
            ([f"not running: {', '.join(down)}"] if down else [])))
    return Result(True, f"{len(CONTAINERS)} containers running")


def collect_containers(ctx: Ctx) -> dict:
    states = _container_states()
    ev = {"docker-ps": sh(["docker", "ps", "-a"])[1],
          "listening-ports": sh(["ss", "-ltnup"])[1]}
    for c in CONTAINERS:
        if c in states and states[c][0] != "running":
            ev[f"{c}-state"] = sh(["docker", "inspect", "-f", "{{json .State}}", c])[1]
            ev[f"{c}-logs"] = sh(["docker", "logs", "--tail", "80", c])[1]
    return ev


def fix_containers(ctx: Ctx) -> str:
    states = _container_states()
    out = []
    for c in CONTAINERS:
        if c in states and states[c][0] != "running":
            rc, o = sh(["docker", "start", c], timeout=90)
            out.append(f"docker start {c} rc={rc} {o}")
    return "\n".join(out) or "nothing to start (missing containers need `docker compose up`)"


def _container_running(name: str) -> bool:
    return _container_states().get(name, ("",))[0] == "running"


def check_photosearch_api(ctx: Ctx) -> Result:
    if not _container_running("photosearch"):
        return Result(None, "skipped: photosearch container not running")
    code, body = http("http://127.0.0.1:8000/api/health", headers=API_HEADERS)
    if code == 200:
        return Result(True, "/api/health 200")
    return Result(False, f"/api/health -> {code or body}")


def collect_photosearch_api(ctx: Ctx) -> dict:
    return {"photosearch-logs": sh(["docker", "logs", "--tail", "150", "photosearch"])[1],
            "docker-stats": sh(["docker", "stats", "--no-stream"], timeout=30)[1]}


def check_mcp(ctx: Ctx) -> Result:
    if not _container_running("photosearch-mcp"):
        return Result(None, "skipped: photosearch-mcp container not running")
    code, body = http("http://127.0.0.1:8848/mcp", headers=API_HEADERS)
    if 0 < code < 500:
        return Result(True, f"/mcp answers ({code})")
    return Result(False, f"/mcp -> {code or body}")


def fix_mcp(ctx: Ctx) -> str:
    return sh(["docker", "restart", "photosearch-mcp"], timeout=90)[1]


def _syncthing_apikey() -> str:
    try:
        m = re.search(r"<apikey>([^<]+)</apikey>", Path(SYNCTHING_CONFIG).read_text())
        return m.group(1) if m else ""
    except OSError:
        return ""


def check_syncthing(ctx: Ctx) -> Result:
    if not _container_running("photosearch-syncthing"):
        return Result(None, "skipped: syncthing container not running")
    code, body = http("http://127.0.0.1:8384/rest/noauth/health")
    if code != 200:
        return Result(False, f"syncthing GUI/API not answering ({code or body})")
    rc, out = sh(f"ss -ltn 'sport = :{SYNCTHING_PORT}' | tail -n +2")
    if not out.strip():
        return Result(False, f"nothing listening on sync port {SYNCTHING_PORT}")
    return Result(True, f"API ok, listening on {SYNCTHING_PORT}")


def collect_syncthing(ctx: Ctx) -> dict:
    return {"syncthing-logs": sh(["docker", "logs", "--tail", "100", "photosearch-syncthing"])[1],
            "listening-ports": sh(["ss", "-ltnup"])[1]}


def check_phone_sync(ctx: Ctx) -> Result:
    if not _container_running("photosearch-syncthing"):
        return Result(None, "skipped: syncthing container not running")
    key = _syncthing_apikey()
    if not key:
        return Result(None, "skipped: no syncthing API key readable")
    hdr = {"X-API-Key": key}
    code, me = http("http://127.0.0.1:8384/rest/system/status", headers=hdr)
    code2, stats = http("http://127.0.0.1:8384/rest/stats/device", headers=hdr)
    if code != 200 or code2 != 200:
        return Result(None, f"skipped: syncthing REST {code}/{code2}")
    my_id = json.loads(me).get("myID")
    seen = [v.get("lastSeen") for k, v in json.loads(stats).items() if k != my_id]
    seen = [s for s in seen if s and not s.startswith("1970")]
    if not seen:
        return Result(False, "no phone has ever connected")
    newest = max(dt.datetime.fromisoformat(s.replace("Z", "+00:00")) for s in seen)
    hours = (now() - newest).total_seconds() / 3600
    if hours > 72:
        return Result(False, f"no phone has connected for {hours:.0f} h")
    return Result(True, f"a phone connected {hours:.1f} h ago")


def check_cron(ctx: Ctx) -> Result:
    rc, tab = sh(["crontab", "-l", "-u", CRON_USER])
    problems = []
    for marker, log in CRON_JOBS.items():
        if marker not in tab:
            problems.append(f"{marker} missing from {CRON_USER}'s crontab")
            continue
        try:
            hours = (time.time() - os.stat(log).st_mtime) / 3600
        except OSError:
            problems.append(f"{log} missing")
            continue
        if hours > 26:
            problems.append(f"{marker} log not written for {hours:.0f} h")
        tail = sh(["tail", "-n", "40", log])[1]
        if "Traceback" in tail:
            problems.append(f"{marker} last run has a Traceback")
    return Result(not problems, "; ".join(problems) or "both cron jobs ran in the last 26 h")


def collect_cron(ctx: Ctx) -> dict:
    ev = {"crontab": sh(["crontab", "-l", "-u", CRON_USER])[1]}
    for marker, log in CRON_JOBS.items():
        ev[f"{marker}-log-tail"] = sh(["tail", "-n", "80", log])[1]
    return ev


def check_resources(ctx: Ctx) -> Result:
    problems = []
    for mount in ("/volume1", "/"):
        try:
            u = shutil.disk_usage(mount)
            pct = 100 * u.used / u.total
            if pct > 90:
                problems.append(f"{mount} {pct:.0f}% full")
        except OSError:
            pass
    try:
        mem = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        avail_mb = int(mem["MemAvailable"].split()[0]) // 1024
        if avail_mb < 400:
            problems.append(f"only {avail_mb} MB memory available")
    except (OSError, KeyError, ValueError):
        avail_mb = -1
    since = ctx.state.get("last_run")
    if since:
        epoch = int(dt.datetime.fromisoformat(since).timestamp())
        rc, k = sh(["journalctl", "-k", "--since", f"@{epoch}", "--no-pager", "-q"], timeout=30)
        kills = [l for l in k.splitlines() if re.search(r"out of memory|oom-kill|Killed process", l, re.I)]
        if kills:
            problems.append(f"{len(kills)} OOM kill(s) since last run")
    return Result(not problems, "; ".join(problems) or f"disk ok, {avail_mb} MB available")


def collect_resources(ctx: Ctx) -> dict:
    return {"df": sh(["df", "-h"])[1], "free": sh(["free", "-m"])[1],
            "top-memory": sh("ps -eo pid,rss,etime,args --sort=-rss | head -20")[1],
            "oom-kernel-log": sh("journalctl -k -n 400 --no-pager -q | grep -iE "
                                 "'out of memory|oom|killed process' | tail -40")[1]}


def _go_version(path: str) -> Optional[tuple]:
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            if head != b"\x7fELF":
                return None
            data = f.read()
    except OSError:
        return None
    if b"Go buildinf:" not in data:
        return None
    # Bounded digits: binaries also hold strings like "go1.190119941996" (snapshot_tool).
    m = re.search(rb"go1\.(\d{1,2})(?:\.(\d{1,2}))?(?!\d)", data)
    return (1, int(m.group(1)), int(m.group(2) or 0)) if m else None


def fingerprint() -> dict:
    osv = sh("grep -h ^OS_VERSION= /etc/os-release | cut -d= -f2")[1]
    return {
        "ugos": osv,
        "kernel": os.uname().release,
        "docker": sh(["docker", "version", "--format", "{{.Server.Version}}"])[1],
        "tailscale": (sh(["tailscale", "version"])[1].splitlines() or [""])[0],
        "system_godebug": _system_godebug(),
    }


def go127_binaries() -> list[str]:
    found = []
    for d in ("/usr/bin", "/usr/sbin", "/usr/local/bin", "/usr/libexec/docker/cli-plugins"):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p) and os.access(p, os.X_OK):
                v = _go_version(p)
                if v and v[1] >= 27:
                    found.append(f"{p} go{v[0]}.{v[1]}.{v[2]}")
    return sorted(found)


def check_drift(ctx: Ctx) -> Result:
    """Firmware/package changes: note them, and warn about Go 1.27+ binaries that
    will meet UGOS's GODEBUG without an override."""
    fp = fingerprint()
    old = ctx.state.get("fingerprint")
    if old and old != fp and not ctx.dry:
        changed = {k: (old.get(k), v) for k, v in fp.items() if old.get(k) != v}
        ctx.state["go_scan"] = None   # force a rescan
        path = ctx.state_dir / "drift" / f"{now():%Y%m%d-%H%M%S}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# System change detected\n\n" + "\n".join(
            f"- **{k}:** {a} → {b}" for k, (a, b) in changed.items())
            + "\n\n## Snapshot after the change\n\n```\n" + snapshot() + "\n```\n")
        notify(ctx, "System changed", ", ".join(f"{k}: {a} → {b}" for k, (a, b) in changed.items())
               + f"\nSnapshot: {path}", "default", ["package"])
    if not ctx.dry:
        ctx.state["fingerprint"] = fp
    scan = ctx.state.get("go_scan")
    if not scan or age_s(scan.get("at")) > 86400:
        scan = {"at": iso(), "found": go127_binaries()}
        ctx.state["go_scan"] = scan
    covered = {"/usr/bin/tailscale", "/usr/sbin/tailscaled"}
    risky = [f for f in scan["found"] if f.split()[0] not in covered]
    if fp["system_godebug"] and risky:
        return Result(False, "Go 1.27+ binaries that will crash under UGOS's GODEBUG "
                             f"unless their unit unsets it: {', '.join(risky)}")
    return Result(True, f"UGOS {fp['ugos']}, docker {fp['docker']}")


def snapshot() -> str:
    """The post-update baseline (what we diffed by hand on 2026-11-15)."""
    parts = []
    for title, cmd in [
        ("versions", "grep OS_VERSION /etc/os-release; uname -r; docker version --format "
                     "'docker {{.Server.Version}}'; tailscale version | head -1"),
        ("GODEBUG", "systemctl show-environment | grep GODEBUG; grep -n GODEBUG "
                    "/etc/systemd/system.conf /etc/profile.d/go.sh"),
        ("overrides", "for s in " + " ".join(GODEBUG_UNITS) + "; do echo \"$s: "
                      "$(systemctl show $s -p UnsetEnvironment --value) $(systemctl is-active $s)\"; done"),
        ("containers", "docker ps -a --format '{{.Names}}\\t{{.Image}}\\t{{.Status}}'"),
        ("ports", "ss -ltnu | awk 'NR>1{print $1, $5}' | sort -u"),
        ("crontab", f"crontab -l -u {CRON_USER}"),
        ("daemon.json", "cat /etc/docker/daemon.json"),
        ("go>=1.27 binaries", None),
    ]:
        body = "\n".join(go127_binaries()) if cmd is None else sh(cmd)[1]
        parts.append(f"### {title}\n{body}")
    return "\n".join(parts)


NAS_CHECKS = [
    Check("godebug_overrides", check_godebug_overrides, GODEBUG_HINT,
          fix=fix_godebug_overrides, fix_label="write the missing no-godebug.conf drop-ins",
          debounce=1, settle_s=0),
    Check("tailscale", check_tailscale, GODEBUG_HINT + """

Other causes: logged out / key expired (BackendState=NeedsLogin: run
`sudo tailscale up` by hand); a bad release (cached debs in
/var/cache/apt/archives; `apt install --allow-downgrades <deb>` then
`apt-mark hold tailscale`). While the NAS is off Tailscale the desktop watcher
switches the replica to the LAN address.""",
          collect=collect_tailscale, fix=fix_tailscale,
          fix_label="ensure GODEBUG overrides, reset-failed + restart tailscaled", settle_s=15),
    Check("docker", check_docker, GODEBUG_HINT + """

Docker is UGOS-managed (App Center "Docker" app; data-root /volume1/@docker). If
images vanish after a Docker upgrade, check daemon.json for
`"features": {"containerd-snapshotter": false}`. The fix is skipped while an
ingest/maintenance cron job holds its lock.""",
          collect=collect_docker, fix=fix_docker,
          fix_label="ensure GODEBUG overrides, restart containerd + docker", settle_s=30),
    Check("containers", check_containers, """\
Every container uses `restart: unless-stopped`, so one that is down failed to
START (Docker does not retry a start failure). Look at the container's
`.State.Error` evidence. Known: `bind: address already in use` on 22000/21027
= UGOS Sync & Backup (`syncspace`, a Syncthing fork) won the port race; the
photosearch Syncthing must stay on 22001 (docker-compose.nas.yml). A `missing`
container needs `docker compose -f docker-compose.nas.yml up -d <service>`
(not auto-fixed).""",
          collect=collect_containers, fix=fix_containers, fix_label="docker start the stopped containers",
          settle_s=20),
    Check("photosearch_api", check_photosearch_api, """\
The container runs but /api/health does not answer: the server is wedged or
still starting. Check the logs evidence for a traceback or a long DB lock (the
`colors` stage once wedged the box: docs/notes/maintenance.md). Not auto-fixed:
restarting photosearch mid-ingest loses work; use POST /api/admin/restart.""",
          collect=collect_photosearch_api, debounce=3),
    Check("mcp", check_mcp, """\
photosearch-mcp is stateless; restarting it is safe. If it keeps failing, its
image may be stale or broken: compare `docker inspect -f {{.Image}}` of
photosearch and photosearch-mcp (Restart on /admin/deploy recreates both).""",
          fix=fix_mcp, fix_label="docker restart photosearch-mcp", settle_s=15),
    Check("syncthing", check_syncthing, """\
Phone uploads land in _incoming via this container. It listens on 22001, not
22000 (UGOS Sync & Backup holds 22000/21027). Nothing is lost while it is down:
phones queue their photos.""", collect=collect_syncthing),
    Check("phone_sync", check_phone_sync, """\
No phone has reached the NAS Syncthing for 3+ days. Usually a phone that is
off/asleep or its Syncthing/Möbius app was killed; open the app. If both phones
are fine, check relays in the syncthing logs.""", severity="warn", debounce=1),
    Check("cron", check_cron, """\
The 04:00 ingest-incoming and 18:30 maintenance-sweep cron jobs (cantimatt's
crontab) write /var/log/photo-ingest.log and /var/log/photo-maintenance.log.
A firmware update could reset the crontab: restore it from the snapshot
(~/nas-preupdate-2026-11-15/ on the desktop, or a drift/ snapshot here).""",
          severity="warn", collect=collect_cron, debounce=1),
    Check("resources", check_resources, """\
The NAS has 7.5 GB RAM. The geocoder once OOM-killed it (per-ROW cost:
docs/notes/maintenance.md). OOM kills here name the process in the
kernel-log evidence.""", severity="warn", collect=collect_resources, debounce=1),
    Check("drift", check_drift, GODEBUG_HINT + """

A firmware/package change was detected or a new Go 1.27+ binary appeared. For
each listed binary that runs as a systemd unit, add the same no-godebug.conf
drop-in to that unit (and to GODEBUG_UNITS in healthwatch.py).""",
          severity="warn", debounce=1),
]


# ---------------------------------------------------------------------------
# Desktop checks
# ---------------------------------------------------------------------------

def _nas_health(host: str) -> tuple[int, str]:
    return http(f"http://{host}:8000/api/health", timeout=8, headers=API_HEADERS)


def check_nas_reachable(ctx: Ctx) -> Result:
    ts, _ = _nas_health(NAS_TS)
    lan, _ = _nas_health(NAS_LAN)
    ctx.state["nas_paths"] = {"tailscale": ts == 200, "lan": lan == 200}
    if ts == 200:
        return Result(True, "NAS answers over Tailscale" + ("" if lan == 200 else " (not on LAN)"))
    if lan == 200:
        return Result(False, "NAS answers on the LAN but not over Tailscale")
    return Result(False, "NAS unreachable over Tailscale and the LAN")


def collect_nas_reachable(ctx: Ctx) -> dict:
    ts_exe = "/mnt/c/Program Files/Tailscale/tailscale.exe"
    return {"ping-lan": sh(["ping", "-c", "3", "-W", "2", NAS_LAN])[1],
            "tailscale-status": sh([ts_exe, "status"], timeout=20)[1],
            "tailscale-ping": sh([ts_exe, "ping", "-c", "3", NAS_TS], timeout=30)[1],
            "nas-tailscaled": sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", NAS_SSH,
                                  "systemctl status tailscaled --no-pager | head -12"], timeout=20)[1]}


def _replica_nas_url_dropin(enable: bool) -> str:
    workers = live_worker_pids()
    if workers:
        return f"skipped: worker fleet running (pids {workers}); a replica restart would kill it"
    if enable:
        REPLICA_DROPIN.parent.mkdir(parents=True, exist_ok=True)
        REPLICA_DROPIN.write_text(
            "# Written by healthwatch: NAS unreachable over Tailscale. healthwatch removes\n"
            "# it (check replica_nas_url) once Tailscale answers again.\n"
            f"[Service]\nEnvironment=PHOTOSEARCH_NAS_URL=http://{NAS_LAN}:8000\n")
    elif REPLICA_DROPIN.exists():
        REPLICA_DROPIN.unlink()
    out = [sh(["systemctl", "--user", "daemon-reload"])[1],
           sh(["systemctl", "--user", "restart", REPLICA_UNIT], timeout=60)[1]]
    return f"{'applied' if enable else 'removed'} {REPLICA_DROPIN}\n" + "\n".join(o for o in out if o)


def fix_nas_reachable(ctx: Ctx) -> str:
    if not ctx.state.get("nas_paths", {}).get("lan"):
        return "skipped: NAS not reachable on the LAN either; nothing to switch to"
    return _replica_nas_url_dropin(True)


def check_replica_nas_url(ctx: Ctx) -> Result:
    if not REPLICA_DROPIN.exists():
        return Result(True, "replica uses the Tailscale NAS URL")
    if not ctx.state.get("nas_paths", {}).get("tailscale"):
        return Result(True, "replica on the LAN stopgap while Tailscale is down")
    return Result(False, "Tailscale works again but the replica is still on the LAN stopgap")


def fix_replica_nas_url(ctx: Ctx) -> str:
    return _replica_nas_url_dropin(False)


def check_replica(ctx: Ctx) -> Result:
    state = sh(["systemctl", "--user", "is-active", REPLICA_UNIT])[1]
    if state != "active":
        return Result(False, f"{REPLICA_UNIT} is {state}")
    code, body = http("http://127.0.0.1:8001/api/health", headers=API_HEADERS)
    if code != 200:
        return Result(False, f"replica /api/health -> {code or body}")
    return Result(True, "replica service up")


def collect_replica(ctx: Ctx) -> dict:
    return {"replica-status": sh(["systemctl", "--user", "status", REPLICA_UNIT, "--no-pager"])[1],
            "replica-journal": sh(["journalctl", "--user", "-u", REPLICA_UNIT, "-n", "100",
                                   "--no-pager"])[1]}


def fix_replica(ctx: Ctx) -> str:
    workers = live_worker_pids()
    if workers:
        return f"skipped: worker fleet running (pids {workers}); a restart would kill it"
    return sh(["systemctl", "--user", "restart", REPLICA_UNIT], timeout=60)[1] or "restarted"


def check_replica_fresh(ctx: Ctx) -> Result:
    code, body = http("http://127.0.0.1:8001/api/admin/replica-status", headers=API_HEADERS,
                      timeout=20)
    if code != 200:
        return Result(None, f"skipped: replica-status {code}")
    age = (json.loads(body).get("last_sync") or {}).get("age_seconds")
    if age is None:
        return Result(False, "replica has never synced")
    if age > 36 * 3600:
        return Result(False, f"replica last synced {age / 3600:.0f} h ago")
    return Result(True, f"synced {age / 3600:.1f} h ago")


def check_nas_watcher(ctx: Ctx) -> Result:
    if not ctx.state.get("nas_paths", {}).get("lan"):
        return Result(None, "skipped: NAS not reachable on the LAN")
    rc, out = sh(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", NAS_SSH,
                  f"stat -c %Y {NAS_STATE_DIR}/status.md"], timeout=20)
    if rc != 0:
        return Result(False, f"NAS watcher status unreadable: {out[:150]}")
    mins = (time.time() - int(out.strip())) / 60
    if mins > 10:
        return Result(False, f"NAS watcher last ran {mins:.0f} min ago")
    return Result(True, f"NAS watcher ran {mins:.0f} min ago")


def collect_nas_watcher(ctx: Ctx) -> dict:
    return {"nas-healthwatch-unit": sh(
        ["ssh", "-o", "BatchMode=yes", NAS_SSH,
         "systemctl status healthwatch.timer healthwatch.service --no-pager; "
         f"tail -40 {NAS_STATE_DIR}/status.md"], timeout=20)[1]}


DESKTOP_CHECKS = [
    Check("nas_reachable", check_nas_reachable, """\
LAN ok but Tailscale down = the NAS's tailscaled died (see the NAS watcher's
`tailscale` incident; on 2026-11-15 it was UGOS's GODEBUG killing tailscaled
1.104.6). The fix switches the replica to http://<LAN>:8000 with a systemd
drop-in; the `replica_nas_url` check removes it when Tailscale is back. Both
down = NAS off, rebooting (firmware updates take ~5 min) or off the network.""",
          collect=collect_nas_reachable, fix=fix_nas_reachable,
          fix_label="switch the replica to the NAS LAN URL", max_fixes=1, settle_s=20,
          mitigation=True),
    Check("replica_nas_url", check_replica_nas_url, """\
The LAN stopgap drop-in is still in place although Tailscale answers again;
the fix removes it and restarts the replica (skipped while a worker fleet runs).""",
          fix=fix_replica_nas_url, fix_label="remove the LAN stopgap, restart the replica",
          settle_s=20),
    Check("replica", check_replica, """\
The desktop read replica (systemd user unit, run-local-replica.sh, :8001).
Restarting it kills a UI-launched worker fleet, so the fix is skipped while
`cli.py worker` processes run.""",
          collect=collect_replica, fix=fix_replica, fix_label="restart photosearch-replica",
          settle_s=20),
    Check("replica_fresh", check_replica_fresh, """\
sync-replica.sh copies the NAS DB nightly (ssh to the NAS LAN address). Stale
means the sync failed: run it by hand and read its output.""", severity="warn", debounce=1),
    Check("nas_watcher", check_nas_watcher, """\
The NAS watcher (root systemd timer healthwatch.timer) has stopped updating
status.md. It runs from the repo checkout, so a bad `git pull` can break it:
run `sudo python3 healthwatch/healthwatch.py check --role nas` on the NAS.""",
          collect=collect_nas_watcher, debounce=3),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "check", "status", "notify-test", "snapshot"])
    ap.add_argument("names", nargs="*", help="check names (check only)")
    ap.add_argument("--role", choices=["nas", "desktop"], default=ROLE)
    ap.add_argument("--no-fix", action="store_true", help="never run a fix")
    a = ap.parse_intermixed_args(argv)   # `check --role nas tailscale` (names after options)

    state_dir = default_state_dir(a.role)
    checks = NAS_CHECKS if a.role == "nas" else DESKTOP_CHECKS

    if a.cmd == "snapshot":
        print(snapshot())
        return 0
    if a.cmd == "status":
        p = state_dir / "status.md"
        print(p.read_text() if p.exists() else f"no status yet at {p}")
        return 0

    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    ctx = Ctx(role=a.role, state_dir=state_dir, fixes=not a.no_fix,
              state=load_state(state_path))

    if a.cmd == "notify-test":
        ok = notify(ctx, "Test", f"healthwatch on {socket.gethostname()} can reach you.",
                    "default", ["test_tube"])
        print("delivered" if ok else "NOT delivered (no ntfy reachable, no toast)")
        return 0 if ok else 1

    if a.cmd == "check":
        wanted = [c for c in checks if not a.names or c.name in a.names]
        ctx.state = json.loads(json.dumps(ctx.state))  # never written back
        ctx.dry, ctx.fixes = True, False
        ctx.notifier = lambda *a: True
        for c in wanted:
            r = _safe_run(c, ctx)
            print(f"{c.name:20s} {({True: 'ok', False: 'FAILING', None: 'skipped'})[r.ok]:8s} "
                  f"{r.summary}")
        return 0

    with open(state_dir / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another cycle is still running")
            return 0
        run_cycle(ctx, checks)
        prune_incidents(ctx)
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ctx.state, indent=2))
        tmp.replace(state_path)
    print((state_dir / "status.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())

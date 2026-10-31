"""Read-only log endpoints behind the /logs page.

One tab per log on the machine serving the page:

    GET /api/logs                 Ask agent query logs (markdown, per day)
    GET /api/logs/requests        the API request log (stream=api) or the MCP
                                  tool-call log (stream=mcp), newest first
    GET /api/logs/fleet           tails of the native worker fleet's logs
    GET /api/logs/server          the web server's own journal (systemd)

`host=nas` on /requests and /server forwards to the NAS when this server is a
replica: the NAS keeps its own request log, and the MCP server runs there, so
its tool-call log only exists on the NAS. Nothing here writes.
"""
from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Optional
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query

from . import request_log

router = APIRouter(prefix="/api/logs")

# run-workers.sh --native keeps one log per worker here (suffixed by --name).
FLEET_LOG_GLOB = "/tmp/photosearch-worker-fleet*/worker-*.log"
SERVER_UNIT_ENV = "PHOTOSEARCH_SERVER_LOG_UNIT"
DEFAULT_SERVER_UNIT = "photosearch-replica"


def _web():
    from . import web
    return web


def _forward_to_nas(path: str, params: dict) -> dict:
    web = _web()
    if not web._nas_url:
        raise HTTPException(400, "host=nas only works on a replica (PHOTOSEARCH_NAS_URL unset)")
    q = urlencode({k: v for k, v in params.items() if v not in (None, "", False)})
    try:
        return web._nas_json("GET", f"{path}?{q}" if q else path, timeout=30)
    except ValueError:
        # An image older than these endpoints answers with a non-JSON page.
        raise HTTPException(502, f"NAS has no {path} — redeploy the NAS to read its logs")
    except OSError as e:
        raise HTTPException(502, f"NAS unreachable: {e}")


@router.get("/sources")
def api_logs_sources():
    """What this server can show, so the page can label each tab and offer the
    NAS where it has its own copy. Cheap: no DB, no NAS call."""
    web = _web()
    return {"replica": bool(web._nas_url), "nas_url": web._nas_url,
            "ask_dir": _ask_dir(),
            "api_log": request_log.default_path(web._db_path, "api"),
            "mcp_log": request_log.default_path(web._db_path, "mcp"),
            "fleet_glob": FLEET_LOG_GLOB,
            "server_unit": os.environ.get(SERVER_UNIT_ENV) or DEFAULT_SERVER_UNIT}


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------

def _ask_dir() -> str:
    return os.environ.get("PHOTOSEARCH_ASK_LOG_DIR") or "ask-logs"


def _ask_day(log_dir: str, day: str) -> dict:
    files = sorted(glob.glob(os.path.join(log_dir, f"ask-{day}.md"))
                   + glob.glob(os.path.join(log_dir, f"ask-{day}.*.md")))
    content = ""
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as fh:
                content += fh.read()
        except OSError:
            pass
    return {"date": day, "content": content, "bytes": len(content),
            "files": [os.path.basename(f) for f in files]}


@router.get("")
def api_logs(date: Optional[str] = Query(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="One day; default today + yesterday")):
    """Ask query logs: the agent's per-query markdown (ask-YYYY-MM-DD.md plus
    rotated ask-YYYY-MM-DD.N.md) from PHOTOSEARCH_ASK_LOG_DIR (default
    ./ask-logs). `available` lists every day that has a log, newest first."""
    log_dir = _ask_dir()
    if date:
        wanted = [date]
    else:
        wanted = [time.strftime("%Y-%m-%d"),
                  time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))]
    available = sorted({os.path.basename(f)[4:14]
                        for f in glob.glob(os.path.join(log_dir, "ask-*.md"))},
                       reverse=True)
    return {"days": [_ask_day(log_dir, d) for d in wanted], "dir": log_dir,
            "available": available, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}


# ---------------------------------------------------------------------------
# API / MCP request logs
# ---------------------------------------------------------------------------

def _matcher(q: Optional[str], source: Optional[str], min_ms: Optional[float],
             errors: bool, method: Optional[str]):
    needle = (q or "").lower()

    def match(r: dict) -> bool:
        if source and r.get("source") != source:
            return False
        if method and r.get("method") != method:
            return False
        if min_ms is not None and (r.get("ms") or 0) < min_ms:
            return False
        if errors and (r.get("status") or 0) < 400:
            return False
        if needle:
            hay = " ".join(str(r.get(k) or "") for k in
                           ("path", "query", "intent", "page", "client")).lower()
            if needle not in hay:
                return False
        return True
    return match


@router.get("/requests")
def api_logs_requests(
    stream: str = Query("api", pattern="^(api|mcp)$"),
    host: str = Query("local", pattern="^(local|nas)$"),
    last: int = Query(300, ge=1, le=5000),
    q: Optional[str] = Query(None, max_length=200),
    source: Optional[str] = Query(None, max_length=40),
    method: Optional[str] = Query(None, max_length=10),
    min_ms: Optional[float] = Query(None, ge=0),
    errors: bool = Query(False),
):
    """The newest request-log records that match, newest first.

    `stream=api` is every /api request; `stream=mcp` is every MCP tool call
    (photosearch/mcp_server.py logs to its own file). Filters: `q` substring of
    path/query/intent/page/client, exact `source`/`method`, `min_ms`, and
    `errors` (status >= 400). Reads backwards and stops early, so polling it is
    cheap even with full rotated backups."""
    if host == "nas":
        return _forward_to_nas("/api/logs/requests", dict(
            stream=stream, last=last, q=q, source=source, method=method,
            min_ms=min_ms, errors="true" if errors else None))
    path = request_log.default_path(_web()._db_path, stream)
    if not path:
        return {"stream": stream, "host": "local", "enabled": False, "records": [],
                "scanned": 0, "path": None}
    recs, scanned = request_log.tail_records(
        path, last=last, match=_matcher(q, source, min_ms, errors, method))
    return {"stream": stream, "host": "local", "enabled": True, "path": path,
            "exists": os.path.exists(path), "records": recs, "scanned": scanned,
            "replica": bool(_web()._nas_url)}


# ---------------------------------------------------------------------------
# Worker fleet
# ---------------------------------------------------------------------------

def _tail_text(path: str, lines: int) -> list[str]:
    out = []
    for raw in request_log._reverse_lines(path, block=64 * 1024):
        out.append(raw.decode("utf-8", errors="replace"))
        if len(out) >= lines:
            break
    out.reverse()
    return out


@router.get("/fleet")
def api_logs_fleet(lines: int = Query(200, ge=10, le=5000)):
    """Tails of every native worker log (`run-workers.sh --native` writes
    /tmp/photosearch-worker-fleet[-NAME]/worker-N.log). These live on the
    machine that runs the fleet — the desktop — so on the NAS this is empty.
    A Docker-mode fleet logs to `docker logs`, not here."""
    workers = []
    for f in sorted(glob.glob(FLEET_LOG_GLOB)):
        try:
            st = os.stat(f)
            tail = _tail_text(f, lines)
        except OSError:
            continue
        fleet_dir = os.path.basename(os.path.dirname(f))
        fleet = fleet_dir[len("photosearch-worker-fleet"):].lstrip("-") or "(default)"
        pid_file = f[:-4] + ".pid"
        alive = None
        try:
            with open(pid_file) as fh:
                os.kill(int(fh.read().strip()), 0)
            alive = True
        except (OSError, ValueError):
            alive = False if os.path.exists(pid_file) else None
        workers.append({"fleet": fleet, "worker": os.path.basename(f)[:-4],
                        "path": f, "size": st.st_size,
                        "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                               time.localtime(st.st_mtime)),
                        "alive": alive, "lines": tail})
    return {"glob": FLEET_LOG_GLOB, "workers": workers}


# ---------------------------------------------------------------------------
# Server journal
# ---------------------------------------------------------------------------

@router.get("/server")
def api_logs_server(lines: int = Query(300, ge=10, le=5000),
                    host: str = Query("local", pattern="^(local|nas)$")):
    """The web server's stdout/stderr as systemd keeps it (`journalctl --user
    -u $PHOTOSEARCH_SERVER_LOG_UNIT`, default photosearch-replica). The NAS
    runs in Docker with no journal, so there it reports `available: false`."""
    if host == "nas":
        return _forward_to_nas("/api/logs/server", dict(lines=lines))
    unit = os.environ.get(SERVER_UNIT_ENV) or DEFAULT_SERVER_UNIT
    cmd = ["journalctl", "--user", "-u", unit, "-n", str(lines),
           "--no-pager", "-o", "short-iso"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"available": False, "unit": unit, "reason": str(e), "lines": []}
    out = (r.stdout or "").splitlines()
    if r.returncode != 0 or not out or out == ["-- No entries --"]:
        return {"available": False, "unit": unit,
                "reason": (r.stderr or "no journal entries for this unit").strip()[:500],
                "lines": []}
    return {"available": True, "unit": unit, "lines": out}

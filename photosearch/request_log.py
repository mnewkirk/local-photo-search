"""Persistent per-request timing log for the web API.

The NAS container's stdout log is discarded every time the container is
recreated (each deploy), so there was no record of how real searches
performed. This writes one JSON line per /api request to a file beside the DB
(`/data/request_log.jsonl` on the NAS, `./request_log.jsonl` on the replica),
rotated by size so it can never fill the disk.

Writes go through a QueueHandler: the request path only enqueues, and a
background thread does the file I/O. On 2026-09-19 a starved NAS disk made
every write take seconds; a synchronous append inside the async middleware
would have stalled the whole event loop on it.

    PHOTOSEARCH_REQUEST_LOG   path to the API log file, or "0" to disable

MCP tool calls (photosearch/mcp_server.py, a separate process) go to their own
file, `request_log.mcp.jsonl`: two processes rotating one file would corrupt
it. `read_records` merges any number of logs by timestamp.

Each line carries `source` and `intent` (photosearch/request_intent.py).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Optional

MAX_BYTES = 20 * 1024 * 1024
BACKUPS = 5

_lock = threading.Lock()
_loggers: dict[str, logging.Logger] = {}
_listeners: dict[str, logging.handlers.QueueListener] = {}
_disabled: set[str] = set()

STREAMS = ("api", "mcp")


def default_path(db_path: str, stream: str = "api") -> Optional[str]:
    env = os.environ.get("PHOTOSEARCH_REQUEST_LOG")
    if env is not None:
        if env.strip() in ("", "0"):
            return None
        base = env
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(db_path)),
                            "request_log.jsonl")
    if stream == "api":
        return base
    root, ext = os.path.splitext(base)
    return f"{root}.{stream}{ext}"


def _get_logger(db_path: str, stream: str) -> Optional[logging.Logger]:
    if stream in _loggers or stream in _disabled:
        return _loggers.get(stream)
    with _lock:
        if stream in _loggers or stream in _disabled:
            return _loggers.get(stream)
        path = default_path(db_path, stream)
        try:
            if not path:
                raise OSError("disabled")
            handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        except OSError as e:
            if str(e) != "disabled":
                logging.getLogger(__name__).warning(
                    "request log disabled (%s): %s", path, e)
            _disabled.add(stream)
            return None
        handler.setFormatter(logging.Formatter("%(message)s"))
        q: queue.Queue = queue.Queue(maxsize=10_000)
        listener = logging.handlers.QueueListener(q, handler)
        listener.start()
        lg = logging.getLogger(f"photosearch.request_log.{stream}")
        lg.propagate = False
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        lg.addHandler(logging.handlers.QueueHandler(q))
        _loggers[stream] = lg
        _listeners[stream] = listener
        return lg


def record(db_path: str, stream: str = "api", **fields) -> None:
    """Enqueue one record: `ts` plus `fields` (method, path, query, status,
    ms, client, source, intent, ...; None values are dropped). Never raises:
    losing a log line is always better than failing the request it
    describes."""
    try:
        lg = _get_logger(db_path, stream)
        if lg is None:
            return
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds")}
        for k, v in fields.items():
            if v is None or v is False:
                continue
            rec[k] = round(v, 1) if k == "ms" else v
        lg.info(json.dumps(rec, separators=(",", ":"), ensure_ascii=False))
    except Exception:  # noqa: BLE001 — see docstring
        pass


def read_records(*paths: str, last: Optional[int] = None) -> list[dict]:
    """Records from each log in `paths` and its rotated backups, merged
    oldest first."""
    files = [f for path in paths
             for f in [f"{path}.{i}" for i in range(BACKUPS, 0, -1)] + [path]]
    out: list[dict] = []
    for f in files:
        if not os.path.exists(f):
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    out.sort(key=lambda r: r.get("ts", ""))
    return out[-last:] if last else out


def reset() -> None:
    """Flush and forget every log target (tests; a path change)."""
    with _lock:
        for listener in _listeners.values():
            listener.stop()  # drains the queue to the file
            for h in listener.handlers:
                h.close()
        for lg in _loggers.values():
            lg.handlers.clear()
        _loggers.clear()
        _listeners.clear()
        _disabled.clear()

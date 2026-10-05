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

    PHOTOSEARCH_REQUEST_LOG   path to the log file, or "0" to disable
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
_logger: Optional[logging.Logger] = None
_listener: Optional[logging.handlers.QueueListener] = None
_disabled = False


def default_path(db_path: str) -> Optional[str]:
    env = os.environ.get("PHOTOSEARCH_REQUEST_LOG")
    if env is not None:
        return None if env.strip() in ("", "0") else env
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "request_log.jsonl")


def _get_logger(db_path: str) -> Optional[logging.Logger]:
    global _logger, _listener, _disabled
    if _logger is not None or _disabled:
        return _logger
    with _lock:
        if _logger is not None or _disabled:
            return _logger
        path = default_path(db_path)
        try:
            if not path:
                raise OSError("disabled")
            handler = logging.handlers.RotatingFileHandler(
                path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
        except OSError as e:
            if str(e) != "disabled":
                logging.getLogger(__name__).warning(
                    "request log disabled (%s): %s", path, e)
            _disabled = True
            return None
        handler.setFormatter(logging.Formatter("%(message)s"))
        q: queue.Queue = queue.Queue(maxsize=10_000)
        _listener = logging.handlers.QueueListener(q, handler)
        _listener.start()
        lg = logging.getLogger("photosearch.request_log")
        lg.propagate = False
        lg.setLevel(logging.INFO)
        lg.addHandler(logging.handlers.QueueHandler(q))
        _logger = lg
        return lg


def record(db_path: str, *, method: str, path: str, query: str, status: int,
           ms: float, client: Optional[str], streaming: bool) -> None:
    """Enqueue one request record. Never raises: losing a log line is
    always better than failing the request it describes."""
    try:
        lg = _get_logger(db_path)
        if lg is None:
            return
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "method": method, "path": path, "query": query, "status": status,
            "ms": round(ms, 1), "client": client,
        }
        if streaming:
            # SSE / streamed bodies: `ms` is time to the response HEADERS,
            # not to the end of the stream.
            rec["streaming"] = True
        lg.info(json.dumps(rec, separators=(",", ":")))
    except Exception:  # noqa: BLE001 — see docstring
        pass


def read_records(path: str, last: Optional[int] = None) -> list[dict]:
    """Records from `path` and its rotated backups, oldest first."""
    files = [f"{path}.{i}" for i in range(BACKUPS, 0, -1)] + [path]
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
    return out[-last:] if last else out


def reset() -> None:
    """Flush and forget the current log target (tests; a path change)."""
    global _logger, _listener, _disabled
    with _lock:
        if _listener is not None:
            _listener.stop()  # drains the queue to the file
            for h in _listener.handlers:
                h.close()
        if _logger is not None:
            _logger.handlers.clear()
        _logger = _listener = None
        _disabled = False

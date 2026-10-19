"""Keep-alive proxy between `claude -p` and a slow local Anthropic-compatible gateway.

LM Studio answers a streaming /v1/messages request with 200 + headers at once,
then sends NOTHING until prefill finishes. On Claude Code's ~100k-token prompt
that is ~13 min for gemma-4-12b, and the client drops an idle stream at 5 min
(raising CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS / API_TIMEOUT_MS does not stop
it), then retries the whole prefill from zero, forever.

This proxy forwards every request unchanged and, while an event-stream
response is silent, writes an Anthropic `ping` event every PING_EVERY_S
seconds. Pings are only written on an SSE event boundary, so a partially
forwarded event is never split. Stdlib only.
"""
from __future__ import annotations

import http.client
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PING_EVERY_S = 15
PING = b'event: ping\ndata: {"type": "ping"}\n\n'
_HOP = {"connection", "keep-alive", "transfer-encoding", "content-length",
        "host", "proxy-connection", "upgrade", "te", "trailer"}


def _handler(upstream: str):
    up = urlsplit(upstream)
    conn_cls = http.client.HTTPSConnection if up.scheme == "https" else http.client.HTTPConnection
    base = up.path.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _proxy(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            conn = conn_cls(up.hostname, up.port, timeout=None)
            try:
                conn.request(self.command, base + self.path, body=body, headers=headers)
                resp = conn.getresponse()
            except OSError as e:
                self.send_error(502, f"upstream: {e}")
                return
            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP:
                    self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sse = "text/event-stream" in (resp.getheader("Content-Type") or "")

            chunks: queue.Queue = queue.Queue()

            def pump():
                try:
                    while True:
                        b = resp.read1(65536)
                        if not b:
                            break
                        chunks.put(b)
                except OSError:
                    pass
                chunks.put(None)

            threading.Thread(target=pump, daemon=True).start()
            at_boundary = True
            try:
                while True:
                    try:
                        b = chunks.get(timeout=PING_EVERY_S)
                    except queue.Empty:
                        if sse and at_boundary:
                            self.wfile.write(PING)
                            self.wfile.flush()
                        continue
                    if b is None:
                        break
                    self.wfile.write(b)
                    self.wfile.flush()
                    at_boundary = b.endswith(b"\n\n")
            except OSError:  # client went away
                pass
            finally:
                conn.close()

        do_POST = do_GET = do_PUT = do_DELETE = do_HEAD = _proxy

    return Handler


def start(upstream: str) -> tuple[str, ThreadingHTTPServer]:
    """Start the proxy on a free localhost port; returns (base_url, server)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(upstream))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv

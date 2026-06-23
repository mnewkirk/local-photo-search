"""MCP server for local-photo-search (M24a).

Exposes the shared tool layer (``photosearch/tools.py``) over the Model
Context Protocol via streamable HTTP, so any MCP-capable client on the LAN
can drive LLM-planned photo search end-to-end.

Runs as its own NAS container (``photosearch-mcp`` in
``docker-compose.nas.yml``), reusing the photosearch image because
``search_photos`` needs the CLIP text encoder. It opens the same SQLite DB
the web server uses (read-only for search; WAL handles concurrent readers).

Uses the SDK's low-level ``Server`` rather than ``FastMCP`` on purpose: the
tool schemas are owned by ``tools.py`` and fed in verbatim via
``mcp_tools()``, so there is exactly one schema definition shared with the
in-app agent. FastMCP would re-derive schemas from function signatures and
duplicate them.

Image policy: ``get_photo_image`` is only advertised / callable when
``PHOTOSEARCH_MCP_ALLOW_IMAGES`` is truthy. Default off — text metadata only.

Run:
    PHOTOSEARCH_DB=/data/photo_index.db python -m photosearch.mcp_server
Env:
    PHOTOSEARCH_DB                 path to the SQLite DB (required)
    PHOTO_ROOT                     library root for resolving relative paths
    PHOTOSEARCH_MCP_HOST           bind host (default 0.0.0.0)
    PHOTOSEARCH_MCP_PORT           bind port (default 8848)
    PHOTOSEARCH_MCP_ALLOW_IMAGES   '1'/'true' to expose get_photo_image
    PHOTOSEARCH_ALLOW_WRITES       '1'/'true' to expose the M26b write tools
                                   (set_photo_location / set_photo_tags)
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os

from .db import PhotoDB
from .tools import call_tool, get_tool, mcp_tools

logger = logging.getLogger("photosearch.mcp")

SERVER_NAME = "photosearch"


def _images_allowed() -> bool:
    return os.environ.get("PHOTOSEARCH_MCP_ALLOW_IMAGES", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _writes_allowed() -> bool:
    """Expose the M26b mutation tools over MCP. Off by default — an MCP client
    only gets write access when the operator opts in (PHOTOSEARCH_ALLOW_WRITES)."""
    return os.environ.get("PHOTOSEARCH_ALLOW_WRITES", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _db_path() -> str:
    path = os.environ.get("PHOTOSEARCH_DB")
    if not path:
        raise RuntimeError(
            "PHOTOSEARCH_DB is not set — point it at the SQLite DB "
            "(e.g. /data/photo_index.db)."
        )
    return path


def _open_db() -> PhotoDB:
    """Fresh connection per call — matches web.py's per-request pattern and
    keeps SQLite handles off the event loop's long-lived state."""
    return PhotoDB(_db_path(), photo_root=os.environ.get("PHOTO_ROOT"))


def build_server():
    """Construct the low-level MCP ``Server`` with our tools registered."""
    from mcp.server.lowlevel import Server
    import mcp.types as types

    allow_images = _images_allowed()
    allow_writes = _writes_allowed()
    app = Server(SERVER_NAME)

    @app.list_tools()
    async def list_tools() -> list:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in mcp_tools(include_images=allow_images,
                                  include_writes=allow_writes)
        ]

    @app.call_tool()
    async def handle_call(name: str, arguments: dict) -> list:
        # Enforce the image gate at the call boundary too, not just in the
        # advertised list — a client could call a tool it wasn't offered.
        if name == "get_photo_image" and not allow_images:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "image returns are disabled by the "
                                          "operator (PHOTOSEARCH_MCP_ALLOW_IMAGES)"}),
            )]
        # Same enforcement for the write tools — a client could call one it
        # wasn't advertised.
        spec = get_tool(name)
        if spec is not None and spec.writes and not allow_writes:
            return [types.TextContent(
                type="text",
                text=json.dumps({"error": "write tools are disabled by the "
                                          "operator (PHOTOSEARCH_ALLOW_WRITES)"}),
            )]
        if get_tool(name) is None:
            return [types.TextContent(
                type="text", text=json.dumps({"error": f"unknown tool: {name}"}))]

        with _open_db() as db:
            result = call_tool(db, name, arguments or {})

            # get_photo_image hands back a thumbnail path; read it into an
            # ImageContent so the model actually sees the pixels.
            if name == "get_photo_image" and isinstance(result, dict) \
                    and result.get("thumbnail_path"):
                try:
                    with open(result["thumbnail_path"], "rb") as fh:
                        data = base64.b64encode(fh.read()).decode("ascii")
                    return [types.ImageContent(
                        type="image", data=data,
                        mimeType=result.get("mime_type", "image/jpeg"))]
                except OSError as exc:
                    return [types.TextContent(
                        type="text", text=json.dumps({"error": str(exc)}))]

        return [types.TextContent(type="text", text=json.dumps(result, default=str))]

    return app


def build_asgi_app():
    """Build the Starlette ASGI app serving the MCP server at ``/mcp``."""
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    app = build_server()
    # Stateless so a simple HTTP client (or a fresh client reconnect) works
    # without sticky session affinity — appropriate for read-only search.
    session_manager = StreamableHTTPSessionManager(
        app=app, event_store=None, json_response=False, stateless=True,
    )

    async def handle_streamable_http(scope, receive, send):
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        async with session_manager.run():
            logger.info("photosearch MCP server started (images=%s)",
                        "on" if _images_allowed() else "off")
            yield

    return Starlette(routes=[Mount("/mcp", app=handle_streamable_http)],
                     lifespan=lifespan)


def main():
    logging.basicConfig(
        level=os.environ.get("PHOTOSEARCH_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    import uvicorn

    host = os.environ.get("PHOTOSEARCH_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("PHOTOSEARCH_MCP_PORT", "8848"))
    # Fail fast with a clear message if the DB env var is missing.
    _db_path()
    logger.info("serving MCP on http://%s:%d/mcp", host, port)
    uvicorn.run(build_asgi_app(), host=host, port=port)


if __name__ == "__main__":
    main()

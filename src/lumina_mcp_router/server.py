"""MCP SSE server exposing the 2 meta-tools + FastAPI /admin sidecar."""
from __future__ import annotations

import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

from .backends import BackendRegistry
from .config import Config, load_config
from .embedder import Embedder
from .index import VectorIndex
from .tools import (
    CALL_TOOL_DESCRIPTION,
    CALL_TOOL_SCHEMA,
    SEARCH_TOOLS_DESCRIPTION,
    SEARCH_TOOLS_SCHEMA,
    Router,
)

logger = logging.getLogger("lumina_mcp_router")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in (
                "args", "msg", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "taskName",
            )
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def build_mcp_server(router: Router) -> Server:
    server: Server = Server("lumina-mcp-router")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name="search_tools",
                description=SEARCH_TOOLS_DESCRIPTION,
                inputSchema=SEARCH_TOOLS_SCHEMA,
            ),
            Tool(
                name="call_tool",
                description=CALL_TOOL_DESCRIPTION,
                inputSchema=CALL_TOOL_SCHEMA,
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        arguments = arguments or {}
        if name == "search_tools":
            payload = await router.search_tools(
                query=arguments.get("query", ""),
                top_k=int(arguments.get("top_k", 10)),
            )
            return [TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
        if name == "call_tool":
            target = arguments.get("name", "")
            args = arguments.get("arguments", {}) or {}
            try:
                result = await router.call_tool(target, args)
            except Exception as e:
                err = {"error": str(e), "tool": target}
                return [TextContent(type="text", text=json.dumps(err, ensure_ascii=False))]
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
        raise ValueError(f"unknown meta-tool: {name}")

    return server


def build_app(cfg: Config, router: Router, mcp_server: Server) -> Starlette:
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse_transport.connect_sse(request.scope, request.receive, request._send) as (
            read_stream,
            write_stream,
        ):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    # ---- Admin sub-app (FastAPI) ----
    admin = FastAPI(title="lumina-mcp-router admin")

    @admin.get("/health")
    async def health() -> dict[str, Any]:
        backends = []
        for name in router.registry.names():
            conn = router.registry.get(name)
            backends.append(
                {
                    "name": name,
                    "connected": bool(conn and conn.is_connected),
                }
            )
        return {
            "status": "ok",
            "tools_indexed": len(router.index),
            "backends": backends,
        }

    @admin.get("/tools")
    async def list_indexed_tools() -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": t.name,
                    "backend": t.backend,
                    "original_name": t.original_name,
                    "description": t.description,
                }
                for t in router.index.all()
            ]
        }

    if cfg.reindex_endpoint_enabled:
        @admin.post("/admin/reindex")
        async def reindex_endpoint() -> dict[str, Any]:
            try:
                stats = await router.reindex()
            except Exception as e:  # pragma: no cover
                raise HTTPException(status_code=500, detail=str(e)) from e
            return {"ok": True, "stats": stats}

    @asynccontextmanager
    async def lifespan(app: Starlette):  # noqa: D401
        logger.info("starting_lumina_mcp_router")
        status = await router.registry.connect_all()
        logger.info("backends_status", extra={"status": status})
        try:
            stats = await router.reindex()
            logger.info("initial_index_built", extra={"stats": stats})
        except Exception as e:
            logger.exception("initial_reindex_failed: %s", e)
        yield
        logger.info("shutting_down")
        await router.registry.close_all()
        await router.embedder.close()

    app = Starlette(
        debug=False,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Mount("/", app=admin),
        ],
        lifespan=lifespan,
    )
    return app


def create_app(cfg: Config | None = None) -> Starlette:
    cfg = cfg or load_config()
    setup_logging(cfg.log_level)
    registry = BackendRegistry(cfg.backends)
    embedder = Embedder(base_url=cfg.ollama_base_url, model=cfg.embedding_model)
    router = Router(registry=registry, embedder=embedder, index=VectorIndex())
    mcp_server = build_mcp_server(router)
    return build_app(cfg, router, mcp_server)


def run() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.listen_host, port=cfg.listen_port, log_config=None)

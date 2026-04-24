"""MCP client wrappers for backend servers (SSE + streamablehttp transports)."""
from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from .config import BackendConfig

logger = logging.getLogger(__name__)


@dataclass
class BackendTool:
    backend: str
    name: str  # original name on the backend
    description: str
    input_schema: dict[str, Any]


@asynccontextmanager
async def _open_transport(transport: str, url: str) -> AsyncIterator[tuple[Any, Any]]:
    """Open the correct MCP client transport and yield (read, write) streams.

    Both transports expose a ``(read, write)`` pair compatible with
    ``ClientSession``. ``streamablehttp_client`` additionally yields a third
    element (a ``get_session_id`` callback) which we simply ignore here.
    """
    if transport == "sse":
        async with sse_client(url) as streams:
            yield streams[0], streams[1]
    elif transport == "streamablehttp":
        async with streamablehttp_client(url) as streams:
            # streams = (read, write, get_session_id)
            yield streams[0], streams[1]
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported transport: {transport!r}")


class BackendConnection:
    """Persistent MCP connection to one backend.

    The connection is kept open for the router's lifetime so tool calls
    can be forwarded with low latency. Transport is selected per-backend
    (``sse`` or ``streamablehttp``).
    """

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self._connected = False

    @property
    def name(self) -> str:
        return self.cfg.name

    async def connect(self) -> None:
        if self._connected:
            return
        stack = AsyncExitStack()
        try:
            read, write = await stack.enter_async_context(
                _open_transport(self.cfg.transport, self.cfg.url)
            )
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.session = session
            self._stack = stack
            self._connected = True
            logger.info(
                "backend_connected",
                extra={
                    "backend": self.cfg.name,
                    "url": self.cfg.url,
                    "transport": self.cfg.transport,
                },
            )
        except Exception:
            await stack.aclose()
            raise

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as e:  # pragma: no cover - shutdown best effort
                logger.warning("backend_close_error: %s", e)
        self._stack = None
        self.session = None
        self._connected = False

    async def list_tools(self) -> list[BackendTool]:
        if not self.session:
            raise RuntimeError(f"backend {self.cfg.name} not connected")
        result = await self.session.list_tools()
        out: list[BackendTool] = []
        for t in result.tools:
            schema = getattr(t, "inputSchema", None) or {}
            if hasattr(schema, "model_dump"):
                schema = schema.model_dump()
            out.append(
                BackendTool(
                    backend=self.cfg.name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=schema,
                )
            )
        return out

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError(f"backend {self.cfg.name} not connected")
        return await self.session.call_tool(name, arguments or {})


class BackendRegistry:
    """Manages connections to all configured backends."""

    def __init__(self, backends: list[BackendConfig]) -> None:
        self._by_name: dict[str, BackendConnection] = {
            b.name: BackendConnection(b) for b in backends
        }

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def get(self, name: str) -> BackendConnection | None:
        return self._by_name.get(name)

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all backends; log WARNING on failures, return status map."""
        status: dict[str, bool] = {}
        for name, conn in self._by_name.items():
            try:
                await conn.connect()
                status[name] = True
            except Exception as e:
                logger.warning(
                    "backend_connect_failed",
                    extra={"backend": name, "error": str(e)},
                )
                status[name] = False
        return status

    async def close_all(self) -> None:
        for conn in self._by_name.values():
            await conn.close()

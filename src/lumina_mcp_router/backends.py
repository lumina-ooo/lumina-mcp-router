"""MCP client wrappers for backend servers (SSE + streamablehttp transports).

Each backend connection runs inside its own dedicated asyncio task so that
the ``AsyncExitStack`` owning the transport is opened and closed from the
same task. This avoids the ``anyio`` cancel-scope cross-task errors that
would otherwise crash the router lifespan when a single backend fails to
connect or is shut down.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from .config import BackendConfig
from .http import build_host_override_client_factory

logger = logging.getLogger(__name__)


# Shared factory: all outgoing backend requests get ``Host: localhost``.
_HTTPX_FACTORY = build_host_override_client_factory()

# Default timeout for establishing a backend connection (initialize + tools).
DEFAULT_CONNECT_TIMEOUT = 15.0
# Default timeout for gracefully shutting down a backend task.
DEFAULT_CLOSE_TIMEOUT = 5.0


# Case-insensitive substrings that indicate a backend session has been torn
# down / expired and a reconnect + retry is worth attempting once. Extend
# this list when we discover additional transient failure modes (e.g. new
# upstream servers using different phrasings).
TRANSIENT_ERROR_PATTERNS: tuple[str, ...] = (
    "not authenticated",
    "authentication required",
    "session expired",
    "session closed",
    "connection closed",
    "closedresourceerror",
    "connection reset",
    "stream closed",
    "broken pipe",
)


def _matches_transient_pattern(text: str) -> bool:
    """Return True if ``text`` (already stringified) matches any transient pattern."""
    lowered = text.lower()
    for pat in TRANSIENT_ERROR_PATTERNS:
        if pat in lowered:
            return True
    return False


def _is_transient_error(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a recoverable session/auth failure.

    Matches against the exception's string representation **and** its class
    name (so e.g. ``anyio.ClosedResourceError`` is caught by type name even
    if its ``str()`` is empty).
    """
    if _matches_transient_pattern(str(exc)):
        return True
    if _matches_transient_pattern(type(exc).__name__):
        return True
    return False


def _result_has_transient_error(result: Any) -> bool:
    """Return True if ``result`` is a CallToolResult whose ``isError`` is truthy
    AND whose text content matches one of :data:`TRANSIENT_ERROR_PATTERNS`.

    An MCP ``CallToolResult`` exposes ``isError: bool`` and ``content: list``
    of content blocks; ``TextContent`` blocks carry a ``.text`` attribute.
    A backend can return HTTP 200 with such a result to signal a tool-level
    failure (e.g. "Not authenticated with Odoo") that is nevertheless
    transient and worth a reconnect + retry.
    """
    if not getattr(result, "isError", False):
        return False
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and _matches_transient_pattern(text):
            return True
    return False


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

    The shared ``httpx_client_factory`` is injected so every outgoing HTTP
    request carries the overridden ``Host`` header (see :mod:`.http`).
    """
    if transport == "sse":
        async with sse_client(url, httpx_client_factory=_HTTPX_FACTORY) as streams:
            yield streams[0], streams[1]
    elif transport == "streamablehttp":
        async with streamablehttp_client(
            url, httpx_client_factory=_HTTPX_FACTORY
        ) as streams:
            # streams = (read, write, get_session_id)
            yield streams[0], streams[1]
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unsupported transport: {transport!r}")


class BackendConnection:
    """Persistent MCP connection to one backend.

    The connection lives inside a dedicated asyncio task so that opening and
    closing the transport happens in the same task (required by ``anyio``).
    Tool calls are forwarded via the shared :class:`ClientSession`, which is
    thread-safe enough for the router's use case (serialised over the
    underlying read/write memory streams).
    """

    def __init__(self, cfg: BackendConfig) -> None:
        self.cfg = cfg
        self.session: ClientSession | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._connected = False
        # Serialises reconnect attempts so two concurrent failing tool calls
        # don't both tear down and rebuild the session.
        self._reconnect_lock: asyncio.Lock = asyncio.Lock()
        # Monotonic counter bumped on every successful reconnect. Concurrent
        # callers capture it before acquiring the lock; if it changed while
        # they waited, another task already reconnected and they can skip.
        self._reconnect_generation: int = 0

    @property
    def name(self) -> str:
        return self.cfg.name

    @property
    def is_connected(self) -> bool:
        return self._connected and self.session is not None

    async def connect(self, timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
        """Start the backend task and wait until it's initialised.

        Raises whatever exception the worker task captured if the transport
        or ``initialize()`` call failed.
        """
        if self._connected:
            return
        # Reset any stale state from a previous failed attempt.
        self.session = None
        self._error = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()

        ready = self._ready
        stop = self._stop

        async def _runner() -> None:
            stack = AsyncExitStack()
            try:
                read, write = await stack.enter_async_context(
                    _open_transport(self.cfg.transport, self.cfg.url)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.session = session
                self._connected = True
                ready.set()
                # Keep the transport + session alive until close() is called.
                await stop.wait()
            except BaseException as e:  # noqa: BLE001 - propagated via self._error
                self._error = e
                # Make sure connect() wakes up even if we failed before ready.
                if not ready.is_set():
                    ready.set()
            finally:
                self._connected = False
                self.session = None
                try:
                    await stack.aclose()
                except BaseException as e:  # pragma: no cover - best effort
                    logger.warning(
                        "backend_stack_close_error",
                        extra={"backend": self.cfg.name, "error": str(e)},
                    )

        self._task = asyncio.create_task(
            _runner(), name=f"lumina-backend-{self.cfg.name}"
        )
        try:
            await asyncio.wait_for(ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Tell the worker to bail out and wait for it to finish cleanly.
            stop.set()
            await self._await_task_silent()
            self._task = None
            raise TimeoutError(
                f"backend {self.cfg.name!r} connect timed out after {timeout}s"
            )

        if self._error is not None:
            # Worker captured an exception; propagate it (stack already closed).
            await self._await_task_silent()
            self._task = None
            err = self._error
            self._error = None
            raise RuntimeError(
                f"backend {self.cfg.name!r} failed to connect: {err}"
            ) from err

        logger.info(
            "backend_connected",
            extra={
                "backend": self.cfg.name,
                "url": self.cfg.url,
                "transport": self.cfg.transport,
            },
        )

    async def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        if self._task is None:
            self._connected = False
            self.session = None
            return
        if self._stop is not None:
            self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "backend_close_timeout", extra={"backend": self.cfg.name}
            )
            self._task.cancel()
            await self._await_task_silent()
        except BaseException as e:  # pragma: no cover - shutdown best effort
            logger.warning(
                "backend_close_error",
                extra={"backend": self.cfg.name, "error": str(e)},
            )
        finally:
            self._task = None
            self._stop = None
            self._ready = None
            self.session = None
            self._connected = False

    async def _await_task_silent(self) -> None:
        if self._task is None:
            return
        try:
            await self._task
        except BaseException:  # noqa: BLE001 - already captured in self._error
            pass

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
        """Invoke a tool on the backend, reconnecting once on transient errors.

        Two classes of transient failure are handled identically:

        * A Python exception raised by ``ClientSession.call_tool`` whose
          message/type matches :data:`TRANSIENT_ERROR_PATTERNS`.
        * A successful-looking ``CallToolResult`` with ``isError=True`` whose
          text content matches the same patterns — this is how real MCP
          servers (e.g. Odoo) report "Not authenticated" at HTTP 200.

        In either case we transparently reconnect the backend **once** and
        retry the call. A second failure — transient or not — is surfaced to
        the caller: for retried transient results we return the *original*
        result so the caller sees the real error rather than a stale second
        copy. The reconnect is guarded by a per-backend :class:`asyncio.Lock`
        so concurrent failing callers don't trigger multiple simultaneous
        reconnects.
        """
        if not self.session:
            # Session is missing — try to (re)connect before the first call.
            await self._reconnect_locked(
                reason="no active session",
                observed_generation=self._reconnect_generation,
            )
        if not self.session:
            raise RuntimeError(f"backend {self.cfg.name} not connected")

        args = arguments or {}
        # Capture the reconnect generation BEFORE the call: if the call fails
        # with a transient error and we enter _reconnect_locked, concurrent
        # waiters that saw the same generation know a single reconnect covers
        # them all.
        gen_before = self._reconnect_generation
        first_exc: BaseException | None = None
        first_result: Any = None
        try:
            first_result = await self.session.call_tool(name, args)
        except BaseException as exc:  # noqa: BLE001 - classified below
            if not _is_transient_error(exc):
                raise
            first_exc = exc

        if first_exc is None and not _result_has_transient_error(first_result):
            # Happy path: no exception and no transient-marked result.
            return first_result

        # Build a human-readable reason + a "surface-on-retry-failure" value
        # that reflects whichever signal (exception vs. result) we observed.
        if first_exc is not None:
            reason = f"transient error: {first_exc}"
            logger.warning(
                "backend_call_transient_error",
                extra={
                    "backend": self.cfg.name,
                    "tool": name,
                    "error": str(first_exc) or type(first_exc).__name__,
                    "phase": "initial",
                    "source": "exception",
                },
            )
        else:
            reason = "transient error in CallToolResult (isError=True)"
            logger.warning(
                "backend_call_transient_error",
                extra={
                    "backend": self.cfg.name,
                    "tool": name,
                    "phase": "initial",
                    "source": "result",
                },
            )

        try:
            await self._reconnect_locked(
                reason=reason,
                observed_generation=gen_before,
            )
        except BaseException as reconnect_exc:  # noqa: BLE001
            logger.error(
                "backend_reconnect_failed",
                extra={
                    "backend": self.cfg.name,
                    "error": str(reconnect_exc)
                    or type(reconnect_exc).__name__,
                },
            )
            # Surface the *original* signal so the caller sees the real cause.
            if first_exc is not None:
                raise first_exc
            return first_result

        if not self.session:
            # Reconnect reported success but session is still missing.
            if first_exc is not None:
                raise first_exc
            return first_result

        try:
            retry_result = await self.session.call_tool(name, args)
        except BaseException as retry_exc:  # noqa: BLE001
            logger.error(
                "backend_call_retry_failed",
                extra={
                    "backend": self.cfg.name,
                    "tool": name,
                    "error": str(retry_exc) or type(retry_exc).__name__,
                    "phase": "retry",
                },
            )
            # Do NOT loop; surface the original signal upward.
            if first_exc is not None:
                raise first_exc
            return first_result

        if _result_has_transient_error(retry_result):
            logger.error(
                "backend_call_retry_failed",
                extra={
                    "backend": self.cfg.name,
                    "tool": name,
                    "phase": "retry",
                    "source": "result",
                },
            )
            # Retry still transient — surface the ORIGINAL signal so the
            # caller sees the real problem without an infinite retry loop.
            if first_exc is not None:
                raise first_exc
            return first_result

        logger.info(
            "backend_call_retry_succeeded",
            extra={"backend": self.cfg.name, "tool": name},
        )
        return retry_result

    async def _reconnect_locked(
        self, reason: str, observed_generation: int | None = None
    ) -> None:
        """Tear down the existing session and re-establish it, under a lock.

        Concurrent callers wait for the first reconnect to finish; when the
        lock is released we compare the reconnect generation they observed
        before the failed call with the current one — if it changed, another
        task already reconnected on their behalf and we skip the rebuild.
        """
        async with self._reconnect_lock:
            if (
                observed_generation is not None
                and observed_generation != self._reconnect_generation
                and self.is_connected
                and self.session is not None
            ):
                logger.debug(
                    "backend_reconnect_skipped",
                    extra={"backend": self.cfg.name, "reason": reason},
                )
                return
            logger.info(
                "backend_reconnecting",
                extra={
                    "backend": self.cfg.name,
                    "url": self.cfg.url,
                    "transport": self.cfg.transport,
                    "reason": reason,
                },
            )
            try:
                await self.close()
            except BaseException as e:  # pragma: no cover - best effort teardown
                logger.warning(
                    "backend_reconnect_close_error",
                    extra={"backend": self.cfg.name, "error": str(e)},
                )
            await self.connect()
            self._reconnect_generation += 1


class BackendRegistry:
    """Manages connections to all configured backends.

    ``connect_all`` is resilient: one backend failing to connect never
    prevents the other backends (or the router itself) from starting.
    """

    def __init__(self, backends: list[BackendConfig]) -> None:
        self._by_name: dict[str, BackendConnection] = {
            b.name: BackendConnection(b) for b in backends
        }

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def get(self, name: str) -> BackendConnection | None:
        return self._by_name.get(name)

    async def connect_all(self) -> dict[str, bool]:
        """Connect to all backends; log WARNING on failures, return status map.

        Failures are fully contained: ``BaseException`` (including
        ``BaseExceptionGroup`` raised by anyio task groups on transport
        errors) is caught per-backend so a single bad backend cannot crash
        the router lifespan.
        """
        status: dict[str, bool] = {}
        for name, conn in self._by_name.items():
            if conn.is_connected:
                status[name] = True
                continue
            try:
                await conn.connect()
                status[name] = True
            except BaseException as e:  # noqa: BLE001 - see docstring
                logger.warning(
                    "backend_connect_failed",
                    extra={
                        "backend": name,
                        "url": conn.cfg.url,
                        "transport": conn.cfg.transport,
                        "error": str(e) or type(e).__name__,
                    },
                )
                # Ensure any half-started task is torn down.
                try:
                    await conn.close()
                except BaseException:  # pragma: no cover - best effort
                    pass
                status[name] = False
        return status

    async def close_all(self) -> None:
        for conn in self._by_name.values():
            try:
                await conn.close()
            except BaseException as e:  # pragma: no cover - shutdown best effort
                logger.warning(
                    "backend_close_error",
                    extra={"backend": conn.name, "error": str(e)},
                )

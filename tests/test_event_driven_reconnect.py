"""Tests for the event-driven transport-close → reconnect path."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

from lumina_mcp_router import backends as backends_mod
from lumina_mcp_router.backends import BackendConnection
from lumina_mcp_router.config import BackendConfig


class _FakeClientSession:
    """Minimal stand-in for ``mcp.ClientSession`` used as an async ctx mgr.

    Stores the read stream so the test can later close it from the outside
    to simulate a server-side disconnect / network error.
    """

    instances: list["_FakeClientSession"] = []

    def __init__(self, read, write) -> None:
        self.read = read
        self.write = write
        self.initialized = False
        _FakeClientSession.instances.append(self)

    async def __aenter__(self) -> "_FakeClientSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def initialize(self) -> None:
        self.initialized = True

    async def call_tool(self, name: str, arguments: dict) -> Any:  # pragma: no cover
        return {"ok": True, "tool": name}


def _install_fake_transport(monkeypatch: pytest.MonkeyPatch):
    """Patch ``_open_transport`` to yield a controllable in-memory stream pair.

    Returns a list that grows by one ``(send, recv)`` pair on every transport
    open — the test uses it to simulate transport-side close events.
    """
    opened: list[tuple] = []

    @asynccontextmanager
    async def fake_open(transport: str, url: str):
        send, recv = anyio.create_memory_object_stream(max_buffer_size=64)
        write_send, write_recv = anyio.create_memory_object_stream(max_buffer_size=64)
        opened.append((send, recv))
        try:
            yield recv, write_send
        finally:
            try:
                await send.aclose()
            except BaseException:
                pass

    monkeypatch.setattr(backends_mod, "_open_transport", fake_open)
    monkeypatch.setattr(backends_mod, "ClientSession", _FakeClientSession)
    _FakeClientSession.instances.clear()
    return opened


@pytest.mark.asyncio
async def test_transport_close_triggers_event_driven_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the upstream read stream MUST schedule a reconnect."""
    opened = _install_fake_transport(monkeypatch)

    conn = BackendConnection(
        BackendConfig(name="hass", url="http://hass/sse", transport="sse")
    )
    await conn.connect(timeout=2.0)
    assert conn.is_connected
    initial_gen = conn._reconnect_generation

    # Simulate the transport closing from the server side: close the upstream
    # send half so the forwarder sees end-of-stream, sets transport_closed,
    # the runner exits, and _schedule_auto_reconnect fires.
    upstream_send, _ = opened[0]
    await upstream_send.aclose()

    # Wait for the reconnect cycle to complete (lock + connect + bump gen).
    for _ in range(50):
        await asyncio.sleep(0.05)
        if conn._reconnect_generation > initial_gen and conn.is_connected:
            break

    assert conn.is_connected, "backend should be reconnected after transport close"
    assert (
        conn._reconnect_generation > initial_gen
    ), "reconnect generation must advance after transport-close-driven reconnect"
    # A second transport was opened by the reconnect.
    assert len(opened) >= 2

    await conn.close()


@pytest.mark.asyncio
async def test_on_connected_callback_fires_after_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = _install_fake_transport(monkeypatch)
    conn = BackendConnection(
        BackendConfig(name="x", url="http://x/mcp", transport="streamablehttp")
    )

    calls: list[str] = []

    async def on_connected(c: BackendConnection) -> None:
        calls.append(c.name)

    conn.set_on_connected(on_connected)
    await conn.connect(timeout=2.0)
    assert calls == ["x"]

    # Trigger transport close → event-driven reconnect → callback fires again.
    upstream_send, _ = opened[0]
    await upstream_send.aclose()

    for _ in range(50):
        await asyncio.sleep(0.05)
        if len(calls) >= 2 and conn.is_connected:
            break

    assert calls == ["x", "x"]
    await conn.close()


@pytest.mark.asyncio
async def test_on_connected_failure_does_not_break_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buggy refresh callback must not propagate out of connect()."""
    _install_fake_transport(monkeypatch)
    conn = BackendConnection(
        BackendConfig(name="y", url="http://y/sse", transport="sse")
    )

    async def boom(_c: BackendConnection) -> None:
        raise RuntimeError("cache exploded")

    conn.set_on_connected(boom)
    # Must not raise.
    await conn.connect(timeout=2.0)
    assert conn.is_connected
    await conn.close()


@pytest.mark.asyncio
async def test_explicit_close_does_not_trigger_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``close()`` is an intentional shutdown — no auto-reconnect."""
    opened = _install_fake_transport(monkeypatch)
    conn = BackendConnection(
        BackendConfig(name="z", url="http://z/sse", transport="sse")
    )
    await conn.connect(timeout=2.0)
    assert conn.is_connected
    initial_opened = len(opened)
    initial_gen = conn._reconnect_generation

    await conn.close()
    # Give any (unintended) reconnect task a chance to run.
    await asyncio.sleep(0.2)

    assert not conn.is_connected
    # No additional transport opens beyond the original one.
    assert len(opened) == initial_opened
    assert conn._reconnect_generation == initial_gen

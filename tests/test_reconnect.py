"""Tests for auto-reconnect-on-error behaviour in BackendConnection.call_tool."""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lumina_mcp_router import backends as backends_mod
from lumina_mcp_router.backends import (
    BackendConnection,
    TRANSIENT_ERROR_PATTERNS,
    _is_transient_error,
)
from lumina_mcp_router.config import BackendConfig


class _FakeSession:
    """Minimal stand-in for ``mcp.ClientSession``."""

    def __init__(self, responses: list[Any]) -> None:
        # Each entry is either an Exception (to raise) or a return value.
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        if not self._responses:
            raise AssertionError("FakeSession ran out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _make_conn() -> BackendConnection:
    return BackendConnection(
        BackendConfig(name="odoo", url="http://odoo/mcp", transport="streamablehttp")
    )


def _install_fake_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    conn: BackendConnection,
    next_sessions: list[_FakeSession],
    reconnect_counter: list[int] | None = None,
    reconnect_delay: float = 0.0,
    connect_error: BaseException | None = None,
) -> None:
    """Patch ``connect``/``close`` so they swap in pre-built fake sessions."""

    async def fake_connect(self: BackendConnection, timeout: float = 15.0) -> None:
        if reconnect_delay:
            await asyncio.sleep(reconnect_delay)
        if reconnect_counter is not None:
            reconnect_counter.append(1)
        if connect_error is not None:
            raise connect_error
        if not next_sessions:
            raise AssertionError("no more fake sessions queued for reconnect")
        self.session = next_sessions.pop(0)  # type: ignore[assignment]
        self._connected = True

    async def fake_close(self: BackendConnection, timeout: float = 5.0) -> None:
        self.session = None
        self._connected = False

    monkeypatch.setattr(
        backends_mod.BackendConnection, "connect", fake_connect, raising=True
    )
    monkeypatch.setattr(
        backends_mod.BackendConnection, "close", fake_close, raising=True
    )


# ---------------------------------------------------------------------------
# _is_transient_error sanity
# ---------------------------------------------------------------------------


def test_transient_patterns_cover_required_phrases() -> None:
    required = {
        "not authenticated",
        "authentication required",
        "session expired",
        "session closed",
        "connection closed",
        "closedresourceerror",
        "connection reset",
        "stream closed",
        "broken pipe",
    }
    assert required.issubset(set(TRANSIENT_ERROR_PATTERNS))


def test_is_transient_error_matches_message_and_type_name() -> None:
    assert _is_transient_error(RuntimeError("Not authenticated with Odoo"))
    assert _is_transient_error(RuntimeError("SESSION EXPIRED"))

    class ClosedResourceError(Exception):
        pass

    # Empty message but matching type name.
    assert _is_transient_error(ClosedResourceError())
    assert not _is_transient_error(ValueError("wrong argument"))


# ---------------------------------------------------------------------------
# (a) retry succeeds after reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_retries_after_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    failing = _FakeSession([RuntimeError("Not authenticated with Odoo")])
    healed = _FakeSession([{"ok": True, "value": 42}])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(monkeypatch, conn, [healed], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {"model": "res.partner"})
    assert result == {"ok": True, "value": 42}
    assert len(reconnects) == 1
    # First session saw the failing call; healed session saw the retry.
    assert failing.calls == [("odoo_search", {"model": "res.partner"})]
    assert healed.calls == [("odoo_search", {"model": "res.partner"})]


# ---------------------------------------------------------------------------
# (b) retry fails a second time → original error surfaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_surfaces_original_error_when_retry_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    original = RuntimeError("Not authenticated with Odoo")
    failing = _FakeSession([original])
    still_broken = _FakeSession([RuntimeError("still not authenticated")])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(
        monkeypatch, conn, [still_broken], reconnect_counter=reconnects
    )

    with pytest.raises(RuntimeError) as ei:
        await conn.call_tool("odoo_search", {})
    # Must be the *original* exception instance, not the retry one.
    assert ei.value is original
    assert len(reconnects) == 1  # only one reconnect, no infinite loop


# ---------------------------------------------------------------------------
# (c) non-matching error is NOT retried
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_does_not_retry_non_transient_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    failing = _FakeSession([ValueError("bad arguments")])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    # No replacement session queued — if reconnect is (wrongly) invoked the
    # fake_connect helper will blow up with an AssertionError.
    _install_fake_lifecycle(monkeypatch, conn, [], reconnect_counter=reconnects)

    with pytest.raises(ValueError, match="bad arguments"):
        await conn.call_tool("odoo_search", {})
    assert reconnects == []  # no reconnect attempted


# ---------------------------------------------------------------------------
# (d) concurrent failing calls only trigger one reconnect (lock works)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_failures_only_trigger_one_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    # Both concurrent calls will see the original failing session and raise.
    failing = _FakeSession(
        [
            RuntimeError("session closed"),
            RuntimeError("session closed"),
        ]
    )
    healed = _FakeSession(
        [
            {"ok": True, "n": 1},
            {"ok": True, "n": 2},
        ]
    )
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    # Only ONE fake session is queued — if the lock fails to serialise the
    # reconnects the second call will hit the "no more fake sessions queued"
    # assertion inside fake_connect.
    _install_fake_lifecycle(
        monkeypatch,
        conn,
        [healed],
        reconnect_counter=reconnects,
        reconnect_delay=0.05,
    )

    results = await asyncio.gather(
        conn.call_tool("t", {"i": 1}),
        conn.call_tool("t", {"i": 2}),
    )
    assert {r["n"] for r in results} == {1, 2}
    assert len(reconnects) == 1

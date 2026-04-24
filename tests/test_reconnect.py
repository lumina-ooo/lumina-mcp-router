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
    _result_has_transient_error,
)

try:  # Real MCP types if available so the assertions mirror production shapes.
    from mcp.types import CallToolResult, TextContent  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments
    CallToolResult = None  # type: ignore
    TextContent = None  # type: ignore


def _make_text_block(text: str):
    if TextContent is not None:
        return TextContent(type="text", text=text)

    class _T:
        pass

    t = _T()
    t.type = "text"
    t.text = text
    return t


def _make_tool_result(is_error: bool, text: str):
    block = _make_text_block(text)
    if CallToolResult is not None:
        return CallToolResult(content=[block], isError=is_error)

    class _R:
        pass

    r = _R()
    r.isError = is_error
    r.content = [block]
    return r
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
        "session terminated",
        "session not found",
        "404 not found",
        "missing session id",
        "mcp-session-id",
        "http 404",
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


# ---------------------------------------------------------------------------
# (e) CallToolResult(isError=True, transient text) → reconnect + retry success
# ---------------------------------------------------------------------------


def test_result_has_transient_error_detects_odoo_auth_message() -> None:
    err = _make_tool_result(True, "Not authenticated with Odoo. Use authenticate.")
    assert _result_has_transient_error(err)

    not_err = _make_tool_result(False, "Not authenticated with Odoo")
    assert not _result_has_transient_error(not_err)

    other = _make_tool_result(True, "Invalid partner ID")
    assert not _result_has_transient_error(other)


@pytest.mark.asyncio
async def test_call_tool_retries_on_is_error_result_with_transient_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    transient_result = _make_tool_result(
        True, "Not authenticated with Odoo. Use the authenticate tool first."
    )
    success_result = _make_tool_result(False, "ok!")
    failing = _FakeSession([transient_result])
    healed = _FakeSession([success_result])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(monkeypatch, conn, [healed], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {"model": "res.partner"})
    assert result is success_result
    assert len(reconnects) == 1
    assert failing.calls == [("odoo_search", {"model": "res.partner"})]
    assert healed.calls == [("odoo_search", {"model": "res.partner"})]


# ---------------------------------------------------------------------------
# (f) Retry also returns transient CallToolResult → first result surfaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_returns_first_result_when_retry_result_also_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    first = _make_tool_result(True, "Not authenticated with Odoo (first)")
    second = _make_tool_result(True, "Not authenticated with Odoo (second)")
    failing = _FakeSession([first])
    still_broken = _FakeSession([second])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(
        monkeypatch, conn, [still_broken], reconnect_counter=reconnects
    )

    result = await conn.call_tool("odoo_search", {})
    # Caller sees the ORIGINAL result, not the retry one.
    assert result is first
    assert len(reconnects) == 1
    # Both sessions saw exactly one call.
    assert len(failing.calls) == 1
    assert len(still_broken.calls) == 1


# ---------------------------------------------------------------------------
# (g) CallToolResult(isError=True, non-transient text) → returned as-is
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_does_not_retry_is_error_with_non_transient_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    non_transient = _make_tool_result(True, "Invalid partner ID")
    failing = _FakeSession([non_transient])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    # No replacement session: if reconnect is (wrongly) invoked, fake_connect
    # will blow up with AssertionError.
    _install_fake_lifecycle(monkeypatch, conn, [], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {})
    assert result is non_transient
    assert reconnects == []
    assert len(failing.calls) == 1


# ---------------------------------------------------------------------------
# (h) Exception("Session terminated") → reconnect + retry succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_retries_on_session_terminated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    failing = _FakeSession([RuntimeError("Session terminated")])
    healed = _FakeSession([{"ok": True}])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(monkeypatch, conn, [healed], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {})
    assert result == {"ok": True}
    assert len(reconnects) == 1
    assert len(failing.calls) == 1
    assert len(healed.calls) == 1


# ---------------------------------------------------------------------------
# (i) Exception("HTTP 404 Not Found") → reconnect + retry succeeds
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_retries_on_http_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    failing = _FakeSession([RuntimeError("HTTP 404 Not Found")])
    healed = _FakeSession([{"ok": True}])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(monkeypatch, conn, [healed], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {})
    assert result == {"ok": True}
    assert len(reconnects) == 1
    assert len(failing.calls) == 1
    assert len(healed.calls) == 1


# ---------------------------------------------------------------------------
# (j) CallToolResult(isError=True, "Session not found") → reconnect + retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_retries_on_is_error_session_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _make_conn()
    transient_result = _make_tool_result(True, "Session not found")
    success_result = _make_tool_result(False, "ok!")
    failing = _FakeSession([transient_result])
    healed = _FakeSession([success_result])
    conn.session = failing  # type: ignore[assignment]
    conn._connected = True

    reconnects: list[int] = []
    _install_fake_lifecycle(monkeypatch, conn, [healed], reconnect_counter=reconnects)

    result = await conn.call_tool("odoo_search", {})
    assert result is success_result
    assert len(reconnects) == 1
    assert len(failing.calls) == 1
    assert len(healed.calls) == 1

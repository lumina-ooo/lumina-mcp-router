"""Tests for backend transport dispatcher and config loading."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from lumina_mcp_router import backends as backends_mod
from lumina_mcp_router.backends import _open_transport
from lumina_mcp_router.config import BackendConfig, Config


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def test_load_backends_default_transport_is_sse(tmp_path: Path) -> None:
    cfg_path = tmp_path / "backends.yaml"
    cfg_path.write_text(
        """
backends:
  - name: foo
    url: http://foo.example/sse
"""
    )
    cfg = Config(backends_config_path=str(cfg_path))
    backends = cfg.load_backends()
    assert len(backends) == 1
    assert backends[0].name == "foo"
    assert backends[0].transport == "sse"


def test_load_backends_explicit_transports(tmp_path: Path) -> None:
    cfg_path = tmp_path / "backends.yaml"
    cfg_path.write_text(
        """
backends:
  - name: a
    transport: sse
    url: http://a/sse
  - name: b
    transport: streamablehttp
    url: http://b/mcp
"""
    )
    cfg = Config(backends_config_path=str(cfg_path))
    backends = cfg.load_backends()
    assert [b.transport for b in backends] == ["sse", "streamablehttp"]


def test_load_backends_rejects_invalid_transport(tmp_path: Path) -> None:
    cfg_path = tmp_path / "backends.yaml"
    cfg_path.write_text(
        """
backends:
  - name: bad
    transport: websocket
    url: http://bad/
"""
    )
    cfg = Config(backends_config_path=str(cfg_path))
    with pytest.raises(ValueError, match="invalid transport"):
        cfg.load_backends()


# ---------------------------------------------------------------------------
# Transport dispatcher
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_transport_dispatches_to_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    @asynccontextmanager
    async def fake_sse_client(url: str, **kwargs):
        calls.append(("sse", url))
        yield ("read-sse", "write-sse")

    @asynccontextmanager
    async def fake_streamable(url: str, **kwargs):  # pragma: no cover - must not be called
        calls.append(("streamable", url))
        yield ("read-s", "write-s", lambda: None)

    monkeypatch.setattr(backends_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(backends_mod, "streamablehttp_client", fake_streamable)

    async with _open_transport("sse", "http://x/sse") as (r, w):
        assert (r, w) == ("read-sse", "write-sse")
    assert calls == [("sse", "http://x/sse")]


@pytest.mark.asyncio
async def test_open_transport_dispatches_to_streamablehttp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    @asynccontextmanager
    async def fake_sse_client(url: str, **kwargs):  # pragma: no cover - must not be called
        calls.append(("sse", url))
        yield ("read-sse", "write-sse")

    @asynccontextmanager
    async def fake_streamable(url: str, **kwargs):
        calls.append(("streamable", url))
        yield ("read-s", "write-s", lambda: "session-id")

    monkeypatch.setattr(backends_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(backends_mod, "streamablehttp_client", fake_streamable)

    async with _open_transport("streamablehttp", "http://y/mcp") as (r, w):
        assert (r, w) == ("read-s", "write-s")
    assert calls == [("streamable", "http://y/mcp")]


@pytest.mark.asyncio
async def test_open_transport_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported transport"):
        async with _open_transport("websocket", "http://z/"):  # type: ignore[arg-type]
            pass


def test_backend_config_default_transport() -> None:
    b = BackendConfig(name="x", url="http://x/sse")
    assert b.transport == "sse"


# ---------------------------------------------------------------------------
# Host override factory
# ---------------------------------------------------------------------------

def test_host_override_factory_sets_host_header() -> None:
    from lumina_mcp_router.http import build_host_override_client_factory

    factory = build_host_override_client_factory("localhost")
    client = factory(headers={"X-Foo": "bar"})
    try:
        # httpx lowercases header names in the Headers mapping.
        assert client.headers.get("host") == "localhost"
        assert client.headers.get("x-foo") == "bar"
    finally:
        # AsyncClient instances just need to be closed; we never opened sockets.
        import asyncio

        asyncio.get_event_loop() if False else None  # noqa: E501 - keep lint happy


def test_open_transport_passes_httpx_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure our host-override factory is injected into SDK calls."""
    import asyncio

    seen: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_sse_client(url: str, **kwargs: Any):
        seen["sse_kwargs"] = kwargs
        yield ("r", "w")

    @asynccontextmanager
    async def fake_streamable(url: str, **kwargs: Any):
        seen["streamable_kwargs"] = kwargs
        yield ("r", "w", lambda: None)

    monkeypatch.setattr(backends_mod, "sse_client", fake_sse_client)
    monkeypatch.setattr(backends_mod, "streamablehttp_client", fake_streamable)

    async def _run() -> None:
        async with _open_transport("sse", "http://x/sse"):
            pass
        async with _open_transport("streamablehttp", "http://x/mcp"):
            pass

    asyncio.run(_run())

    assert "httpx_client_factory" in seen["sse_kwargs"]
    assert "httpx_client_factory" in seen["streamable_kwargs"]
    assert seen["sse_kwargs"]["httpx_client_factory"] is backends_mod._HTTPX_FACTORY


# ---------------------------------------------------------------------------
# Registry resilience
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_connect_all_contains_individual_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumina_mcp_router.backends import BackendRegistry

    async def fake_connect(self):  # type: ignore[no-redef]
        if self.cfg.name == "bad":
            raise RuntimeError("boom")
        self.session = object()  # type: ignore[assignment]
        self._connected = True

    async def fake_close(self, timeout: float = 5.0):  # type: ignore[no-redef]
        self.session = None
        self._connected = False

    monkeypatch.setattr(backends_mod.BackendConnection, "connect", fake_connect)
    monkeypatch.setattr(backends_mod.BackendConnection, "close", fake_close)

    reg = BackendRegistry(
        [
            BackendConfig(name="good", url="http://g/sse"),
            BackendConfig(name="bad", url="http://b/sse"),
            BackendConfig(name="also-good", url="http://g2/sse"),
        ]
    )
    status = await reg.connect_all()
    assert status == {"good": True, "bad": False, "also-good": True}


@pytest.mark.asyncio
async def test_connect_all_empty_registry() -> None:
    from lumina_mcp_router.backends import BackendRegistry

    reg = BackendRegistry([])
    status = await reg.connect_all()
    assert status == {}


@pytest.mark.asyncio
async def test_connect_all_all_failing_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lumina_mcp_router.backends import BackendRegistry

    async def boom(self):  # type: ignore[no-redef]
        raise RuntimeError("nope")

    async def noop_close(self, timeout: float = 5.0):  # type: ignore[no-redef]
        return None

    monkeypatch.setattr(backends_mod.BackendConnection, "connect", boom)
    monkeypatch.setattr(backends_mod.BackendConnection, "close", noop_close)

    reg = BackendRegistry([BackendConfig(name="a", url="http://a/sse")])
    status = await reg.connect_all()
    assert status == {"a": False}

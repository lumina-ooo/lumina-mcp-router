"""Tests for backend transport dispatcher and config loading."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

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
    async def fake_sse_client(url: str):
        calls.append(("sse", url))
        yield ("read-sse", "write-sse")

    @asynccontextmanager
    async def fake_streamable(url: str):  # pragma: no cover - must not be called
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
    async def fake_sse_client(url: str):  # pragma: no cover - must not be called
        calls.append(("sse", url))
        yield ("read-sse", "write-sse")

    @asynccontextmanager
    async def fake_streamable(url: str):
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

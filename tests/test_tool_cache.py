"""Tests for the persistent tool catalogue cache and Router integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumina_mcp_router.backends import BackendTool
from lumina_mcp_router.index import VectorIndex
from lumina_mcp_router.tool_cache import CachedTool, ToolCache, default_cache_path
from lumina_mcp_router.tools import Router, qualified_name


# ---------------------------------------------------------------------------
# Cache primitives
# ---------------------------------------------------------------------------


def test_default_cache_path_uses_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMINA_TOOL_CACHE_PATH", "/tmp/custom-cache.json")
    assert default_cache_path() == "/tmp/custom-cache.json"


def test_replace_backend_persists_to_disk(tmp_path: Path) -> None:
    cache = ToolCache(path=str(tmp_path / "cache.json"))
    cache.replace_backend(
        "gsuite",
        [
            CachedTool(
                backend="gsuite",
                name="send_gmail_message",
                description="Send an email",
                input_schema={"type": "object"},
                embedding=[0.1, 0.2, 0.3],
            )
        ],
    )
    raw = json.loads(Path(cache.path).read_text())
    assert "gsuite" in raw["backends"]
    assert raw["backends"]["gsuite"][0]["name"] == "send_gmail_message"
    assert raw["backends"]["gsuite"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_load_round_trips_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache_a = ToolCache(path=str(path))
    cache_a.replace_backend(
        "hass",
        [
            CachedTool(
                backend="hass",
                name="toggle_light",
                description="Toggle a light",
                input_schema={},
                embedding=[1.0, 0.0],
            )
        ],
    )
    cache_b = ToolCache(path=str(path))
    n = cache_b.load()
    assert n == 1
    tools = cache_b.get("hass")
    assert len(tools) == 1
    assert tools[0].name == "toggle_light"
    assert tools[0].embedding == [1.0, 0.0]


def test_load_missing_file_is_empty(tmp_path: Path) -> None:
    cache = ToolCache(path=str(tmp_path / "absent.json"))
    assert cache.load() == 0
    assert cache.all() == []


def test_load_corrupt_file_does_not_raise(tmp_path: Path) -> None:
    p = tmp_path / "cache.json"
    p.write_text("{not json")
    cache = ToolCache(path=str(p))
    # Must NOT raise — corrupt cache is treated as empty.
    assert cache.load() == 0
    assert cache.all() == []


def test_remove_backend_persists(tmp_path: Path) -> None:
    cache = ToolCache(path=str(tmp_path / "cache.json"))
    cache.replace_backend(
        "gsuite",
        [CachedTool("gsuite", "x", "d", {}, [0.1])],
    )
    cache.remove_backend("gsuite")
    raw = json.loads(Path(cache.path).read_text())
    assert raw["backends"] == {}


# ---------------------------------------------------------------------------
# Router cache integration
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 2.0]

    async def close(self) -> None:
        pass


class _FakeSession:
    async def call_tool(self, name: str, arguments: dict):
        return {"ok": True, "tool": name, "args": arguments}


class _FakeBackendConn:
    def __init__(self, name: str, tools: list[BackendTool], session=True) -> None:
        self._name = name
        self._tools = tools
        self.session = _FakeSession() if session else None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_connected(self) -> bool:
        return self.session is not None

    async def list_tools(self) -> list[BackendTool]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict):
        if self.session is None:
            raise RuntimeError(f"backend {self._name} not connected")
        return await self.session.call_tool(name, arguments)


class _FakeRegistry:
    def __init__(self, conns: list[_FakeBackendConn]) -> None:
        self._by_name = {c.name: c for c in conns}

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def get(self, name: str):
        return self._by_name.get(name)

    async def connect_all(self):
        return {n: True for n in self._by_name}

    async def close_all(self):
        pass


@pytest.mark.asyncio
async def test_refresh_backend_writes_to_cache_and_index(tmp_path: Path) -> None:
    cache = ToolCache(path=str(tmp_path / "cache.json"))
    tools = [
        BackendTool(
            backend="gsuite",
            name="send_gmail_message",
            description="Send an email",
            input_schema={"type": "object"},
        )
    ]
    conn = _FakeBackendConn("gsuite", tools)
    reg = _FakeRegistry([conn])
    router = Router(
        registry=reg, embedder=_FakeEmbedder(), index=VectorIndex(), cache=cache
    )
    n = await router.refresh_backend(conn)
    assert n == 1
    assert qualified_name("gsuite", "send_gmail_message") in router.index
    assert len(cache.get("gsuite")) == 1
    # Cache persisted.
    raw = json.loads(Path(cache.path).read_text())
    assert raw["backends"]["gsuite"][0]["name"] == "send_gmail_message"


@pytest.mark.asyncio
async def test_refresh_backend_replaces_prior_entries(tmp_path: Path) -> None:
    """Re-listing replaces the per-backend tools (removed tools disappear)."""
    cache = ToolCache(path=str(tmp_path / "cache.json"))
    initial = [
        BackendTool("gsuite", "old_tool", "obsolete", {}),
        BackendTool("gsuite", "send_gmail_message", "Send mail", {}),
    ]
    conn = _FakeBackendConn("gsuite", initial)
    router = Router(
        registry=_FakeRegistry([conn]),
        embedder=_FakeEmbedder(),
        index=VectorIndex(),
        cache=cache,
    )
    await router.refresh_backend(conn)
    assert qualified_name("gsuite", "old_tool") in router.index

    # New list omits old_tool.
    conn._tools = [BackendTool("gsuite", "send_gmail_message", "Send mail", {})]
    await router.refresh_backend(conn)
    assert qualified_name("gsuite", "old_tool") not in router.index
    assert qualified_name("gsuite", "send_gmail_message") in router.index
    assert {t.name for t in cache.get("gsuite")} == {"send_gmail_message"}


@pytest.mark.asyncio
async def test_hydrate_index_from_cache_populates_search(tmp_path: Path) -> None:
    """Eager startup load must make search_tools work BEFORE backends connect."""
    path = tmp_path / "cache.json"
    pre_cache = ToolCache(path=str(path))
    pre_cache.replace_backend(
        "gsuite",
        [
            CachedTool(
                backend="gsuite",
                name="send_gmail_message",
                description="Send an email",
                input_schema={"type": "object"},
                embedding=[1.0, 0.0, 0.0],
            )
        ],
    )

    # Fresh router with EMPTY index, backend NOT connected yet.
    cache = ToolCache(path=str(path))
    cache.load()
    conn = _FakeBackendConn("gsuite", [], session=False)
    router = Router(
        registry=_FakeRegistry([conn]),
        embedder=_FakeEmbedder(),
        index=VectorIndex(),
        cache=cache,
    )
    n = router.hydrate_index_from_cache()
    assert n == 1
    # search_tools is functional even though backend has no live session.
    out = await router.search_tools(query="anything", top_k=5)
    assert any(
        r["name"] == qualified_name("gsuite", "send_gmail_message")
        for r in out["results"]
    )


@pytest.mark.asyncio
async def test_reindex_keeps_cached_entries_when_backend_down(tmp_path: Path) -> None:
    cache = ToolCache(path=str(tmp_path / "cache.json"))
    cache.replace_backend(
        "gsuite",
        [
            CachedTool(
                backend="gsuite",
                name="send_gmail_message",
                description="Send an email",
                input_schema={},
                embedding=[1.0, 0.0],
            )
        ],
    )
    conn = _FakeBackendConn("gsuite", [], session=False)
    router = Router(
        registry=_FakeRegistry([conn]),
        embedder=_FakeEmbedder(),
        index=VectorIndex(),
        cache=cache,
    )
    router.hydrate_index_from_cache()
    stats = await router.reindex()
    # Backend has no session → reindex must NOT wipe its cached entries.
    assert stats["backends"]["gsuite"]["from_cache"] is True
    assert qualified_name("gsuite", "send_gmail_message") in router.index
    assert len(cache.get("gsuite")) == 1

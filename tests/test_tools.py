"""Smoke tests for the Router meta-tools with mocked backends + embedder."""
from __future__ import annotations

import pytest

from lumina_mcp_router.backends import BackendTool
from lumina_mcp_router.index import VectorIndex
from lumina_mcp_router.tools import Router, qualified_name


class FakeEmbedder:
    """Embeds texts based on keyword heuristics so search is deterministic."""

    def __init__(self) -> None:
        self.vocab = ["email", "light", "calendar", "contact"]

    async def embed(self, text: str) -> list[float]:
        t = text.lower()
        return [1.0 if kw in t else 0.0 for kw in self.vocab]

    async def close(self) -> None:
        pass


class FakeSession:
    def __init__(self, tools: list[BackendTool], fail_call: bool = False) -> None:
        self._tools = tools
        self._fail_call = fail_call
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if self._fail_call:
            raise RuntimeError("boom")
        return {"ok": True, "tool": name, "arguments": arguments}


class FakeBackendConnection:
    def __init__(self, name: str, tools: list[BackendTool]) -> None:
        self._name = name
        self._tools_list = tools
        self.session = FakeSession(tools)

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def list_tools(self) -> list[BackendTool]:
        return self._tools_list

    async def call_tool(self, name: str, arguments: dict):
        return await self.session.call_tool(name, arguments)


class FakeRegistry:
    def __init__(self, conns: list[FakeBackendConnection]) -> None:
        self._by_name = {c.name: c for c in conns}

    def names(self) -> list[str]:
        return list(self._by_name.keys())

    def get(self, name: str):
        return self._by_name.get(name)

    async def connect_all(self):
        return {n: True for n in self._by_name}

    async def close_all(self):
        pass


@pytest.fixture
def router() -> Router:
    gsuite_tools = [
        BackendTool(
            backend="gsuite",
            name="send_gmail_message",
            description="Send an email via gmail to a recipient",
            input_schema={"type": "object", "properties": {"to": {"type": "string"}}},
        ),
        BackendTool(
            backend="gsuite",
            name="create_calendar_event",
            description="Create a new calendar event",
            input_schema={"type": "object"},
        ),
    ]
    hass_tools = [
        BackendTool(
            backend="hass",
            name="toggle_light",
            description="Toggle a smart light device",
            input_schema={"type": "object"},
        ),
    ]
    registry = FakeRegistry(
        [
            FakeBackendConnection("gsuite", gsuite_tools),
            FakeBackendConnection("hass", hass_tools),
        ]
    )
    return Router(registry=registry, embedder=FakeEmbedder(), index=VectorIndex())


@pytest.mark.asyncio
async def test_reindex_populates_index(router):
    stats = await router.reindex()
    assert stats["total_tools"] == 3
    assert stats["backends"]["gsuite"]["tools"] == 2
    assert stats["backends"]["hass"]["tools"] == 1
    assert qualified_name("gsuite", "send_gmail_message") in router.index


@pytest.mark.asyncio
async def test_search_tools_returns_relevant_results(router):
    await router.reindex()
    out = await router.search_tools(query="send an email", top_k=5)
    assert "results" in out
    names = [r["name"] for r in out["results"]]
    # The email tool must be ranked first
    assert names[0] == qualified_name("gsuite", "send_gmail_message")


@pytest.mark.asyncio
async def test_search_tools_empty_query(router):
    await router.reindex()
    out = await router.search_tools(query="", top_k=5)
    assert out["results"] == []
    assert "error" in out


@pytest.mark.asyncio
async def test_call_tool_routes_to_backend(router):
    await router.reindex()
    qn = qualified_name("gsuite", "send_gmail_message")
    result = await router.call_tool(qn, {"to": "bob@example.com"})
    assert result["ok"] is True
    assert result["tool"] == "send_gmail_message"
    assert result["arguments"] == {"to": "bob@example.com"}


@pytest.mark.asyncio
async def test_call_tool_unknown_name_raises(router):
    await router.reindex()
    with pytest.raises(ValueError):
        await router.call_tool("does_not_exist", {})


@pytest.mark.asyncio
async def test_call_tool_requires_name(router):
    await router.reindex()
    with pytest.raises(ValueError):
        await router.call_tool("", {})


@pytest.mark.asyncio
async def test_top_k_is_clamped(router):
    await router.reindex()
    out = await router.search_tools(query="anything", top_k=999)
    # max 25, but we only have 3 tools
    assert len(out["results"]) <= 25

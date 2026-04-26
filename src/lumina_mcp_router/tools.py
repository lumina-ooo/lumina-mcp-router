"""The 2 meta-tools exposed to LLM clients: search_tools and call_tool."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from .backends import BackendConnection, BackendRegistry
from .embedder import Embedder
from .index import ToolEntry, VectorIndex
from .tool_cache import CachedTool, ToolCache

logger = logging.getLogger(__name__)


SEARCH_TOOLS_DESCRIPTION = """Search the catalogue of available backend tools using natural-language semantic search.

USE THIS FIRST whenever you need to perform an action (send email, create event, query a database, control a device, ...). It returns only the most relevant tools instead of flooding your context with hundreds of schemas.

Arguments:
  - query (string, REQUIRED): a short natural-language description of what you want to do. Examples: "send an email to someone", "list unread messages in inbox", "create a calendar event", "turn on the living room light", "search for contacts in the CRM".
  - top_k (integer, optional, default 10, max 25): how many candidate tools to return.

Returns a JSON object:
  {
    "results": [
      {
        "name": "<exact tool name to pass to call_tool>",
        "backend": "<gsuite|microsoft|odoo|hass>",
        "description": "...",
        "input_schema": { ... JSON Schema of the tool's arguments ... },
        "score": 0.0-1.0
      },
      ...
    ]
  }

EXAMPLE WORKFLOW:
  1. User asks: "send an email to bob"
  2. You call: search_tools(query="send email to a recipient")
  3. You receive: [{"name": "send_gmail_message", "backend": "gsuite", "input_schema": {...}}, ...]
  4. You call: call_tool(name="send_gmail_message", arguments={"to": "bob@example.com", "subject": "...", "body": "..."})

Always call search_tools before call_tool unless you already know the exact tool name from a previous search in this conversation."""


CALL_TOOL_DESCRIPTION = """Invoke a backend tool previously discovered via search_tools.

Arguments:
  - name (string, REQUIRED): exact tool name as returned by search_tools (field "name"). Do NOT invent tool names.
  - arguments (object, REQUIRED): JSON object matching the tool's input_schema returned by search_tools.

Returns the raw result from the backend tool (content blocks / JSON / text).

If you are unsure which tool to use, call search_tools first. Calling call_tool with an unknown name returns an error asking you to search first."""


SEARCH_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language description of the action you want to perform.",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 25,
            "default": 10,
            "description": "Number of candidate tools to return (max 25).",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


CALL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Exact tool name from search_tools results.",
        },
        "arguments": {
            "type": "object",
            "description": "Arguments matching the tool's input_schema.",
            "additionalProperties": True,
        },
    },
    "required": ["name", "arguments"],
    "additionalProperties": False,
}


def build_indexed_text(
    backend: str,
    tool_name: str,
    description: str,
    embedding_context: str | None = None,
) -> str:
    """Canonical form used for embedding a tool.

    When ``embedding_context`` is provided it is prepended verbatim to give the
    embedding model a strong descriptive signal about the backend service
    (e.g. "Microsoft 365 Outlook (Exchange, ...)"). This helps the semantic
    search distinguish tools with similar surface vocabulary across backends
    (Gmail vs Outlook, Google Drive vs OneDrive, ...).

    The tool name is "humanised" (``snake_case`` -> space-separated words)
    so individual tokens contribute to the embedding without being drowned
    inside a single compound identifier.

    Falls back to the current backend name when no context is configured.
    """
    ctx = embedding_context or backend
    human_name = tool_name.replace("_", " ")
    return f"{ctx}. Tool: {human_name}. {description}"


def qualified_name(backend: str, tool_name: str) -> str:
    """Globally-unique name exposed to the LLM. Avoids collisions across backends."""
    return f"{backend}__{tool_name}"


class Router:
    """Holds the index and routes calls. Thin orchestrator used by the MCP server."""

    def __init__(
        self,
        registry: BackendRegistry,
        embedder: Embedder,
        index: VectorIndex | None = None,
        cache: ToolCache | None = None,
    ) -> None:
        self.registry = registry
        self.embedder = embedder
        self.index = index if index is not None else VectorIndex()
        self.cache = cache if cache is not None else ToolCache()

    def hydrate_index_from_cache(self) -> int:
        """Populate the embedding index from the persistent cache.

        Called once at router startup BEFORE any backend has finished
        connecting so ``search_tools`` is immediately functional. Returns
        the number of entries inserted.
        """
        count = 0
        for cached in self.cache.all():
            if not cached.embedding:
                continue
            qname = qualified_name(cached.backend, cached.name)
            self.index.add(
                ToolEntry(
                    name=qname,
                    backend=cached.backend,
                    original_name=cached.name,
                    description=cached.description,
                    input_schema=cached.input_schema,
                    embedding=np.asarray(cached.embedding, dtype=np.float32),
                )
            )
            count += 1
        if count:
            logger.info(
                "tool_index_hydrated_from_cache",
                extra={"tools": count, "backends": len(self.cache.backends())},
            )
        return count

    async def refresh_backend(self, conn: BackendConnection) -> int:
        """Re-list, re-embed, and re-index a single backend's tools.

        Updates both the embedding index AND the persistent cache so that
        future restarts (and the in-flight reconnect window for THIS
        backend) keep serving stale-but-correct results to ``search_tools``.

        Returns the number of tools indexed for the backend (0 on error).
        Errors are logged but never raised — a misbehaving backend must
        not break the router's connect path.
        """
        if conn.session is None:
            return 0
        try:
            tools = await conn.list_tools()
        except BaseException as e:  # noqa: BLE001
            logger.warning(
                "refresh_backend_list_failed",
                extra={
                    "backend": conn.name,
                    "error": str(e) or type(e).__name__,
                },
            )
            return 0

        cfg = getattr(conn, "cfg", None)
        ctx = getattr(cfg, "embedding_context", None) if cfg is not None else None
        # Drop the backend's existing index entries before re-adding so
        # removed tools disappear immediately.
        for entry in list(self.index.all()):
            if entry.backend == conn.name:
                self.index.remove(entry.name)

        cached_entries: list[CachedTool] = []
        count = 0
        for t in tools:
            text = build_indexed_text(t.backend, t.name, t.description, ctx)
            try:
                vec = await self.embedder.embed(text)
            except BaseException as e:  # noqa: BLE001
                logger.warning(
                    "refresh_backend_embed_failed",
                    extra={
                        "backend": t.backend,
                        "tool": t.name,
                        "error": str(e) or type(e).__name__,
                    },
                )
                continue
            qname = qualified_name(t.backend, t.name)
            self.index.add(
                ToolEntry(
                    name=qname,
                    backend=t.backend,
                    original_name=t.name,
                    description=t.description,
                    input_schema=t.input_schema,
                    embedding=np.asarray(vec, dtype=np.float32),
                )
            )
            cached_entries.append(
                CachedTool(
                    backend=t.backend,
                    name=t.name,
                    description=t.description,
                    input_schema=t.input_schema,
                    embedding=list(vec),
                )
            )
            count += 1

        self.cache.replace_backend(conn.name, cached_entries)
        logger.info(
            "refresh_backend_complete",
            extra={"backend": conn.name, "tools": count},
        )
        return count

    async def reindex(self) -> dict[str, Any]:
        """Reconnect any dead backends, re-list tools, re-embed everything.

        Each backend's tool list is refreshed via :meth:`refresh_backend`,
        which also updates the persistent cache. Entries for connected
        backends are atomically replaced; entries for backends that fail
        to list tools are LEFT IN THE CACHE so ``search_tools`` keeps
        working through transient outages.
        """
        stats: dict[str, Any] = {"backends": {}, "total_tools": 0}

        # Ensure connections (best effort)
        await self.registry.connect_all()

        live_backends: set[str] = set()
        for name in self.registry.names():
            conn = self.registry.get(name)
            live_backends.add(name)
            if conn is None or conn.session is None:
                # Keep cached entries (if any) for this backend in the
                # index so search_tools still returns useful results.
                cached = self.cache.get(name)
                stats["backends"][name] = {
                    "status": "down",
                    "tools": len(cached),
                    "from_cache": True,
                }
                stats["total_tools"] += len(cached)
                continue
            count = await self.refresh_backend(conn)
            stats["backends"][name] = {"status": "ok", "tools": count}
            stats["total_tools"] += count

        # Drop cache + index entries for backends no longer configured.
        for cached_backend in list(self.cache.backends()):
            if cached_backend not in live_backends:
                for entry in list(self.index.all()):
                    if entry.backend == cached_backend:
                        self.index.remove(entry.name)
                self.cache.remove_backend(cached_backend)

        logger.info("reindex_complete", extra={"stats": stats})
        return stats

    async def search_tools(self, query: str, top_k: int = 10) -> dict[str, Any]:
        if not query or not query.strip():
            return {"results": [], "error": "query must be a non-empty string"}
        top_k = max(1, min(int(top_k or 10), 25))
        try:
            q_vec = await self.embedder.embed(query)
        except Exception as e:
            logger.error("query_embed_failed: %s", e)
            return {"results": [], "error": f"embedding failed: {e}"}
        results = self.index.search(q_vec, top_k=top_k)
        return {"results": [r.to_dict() for r in results]}

    async def call_tool(self, name: str, arguments: dict[str, Any] | None) -> Any:
        if not name:
            raise ValueError("'name' is required; call search_tools first to discover tools.")
        entry = self.index.get(name)
        if entry is None:
            raise ValueError(
                f"Unknown tool '{name}'. Call search_tools first to discover available tools."
            )
        conn = self.registry.get(entry.backend)
        if conn is None:
            raise RuntimeError(
                f"Backend '{entry.backend}' is not configured."
            )
        # NOTE: we do NOT short-circuit when ``conn.session is None``.
        # ``BackendConnection.call_tool`` will trigger a reconnect via the
        # existing transient-error retry path; if the backend really can't
        # come back the resulting error surfaces to the caller. This is
        # intentional: the persistent tool cache means an entry may exist
        # in the index even while the backend is between sessions.
        result = await conn.call_tool(entry.original_name, arguments or {})
        return _serialize_mcp_result(result)


def _serialize_mcp_result(result: Any) -> Any:
    """Convert an MCP CallToolResult (or similar) into plain JSON-friendly data."""
    if result is None:
        return None
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump(mode="json")
        except Exception:
            try:
                return result.model_dump()
            except Exception:
                pass
    try:
        return json.loads(json.dumps(result, default=str))
    except Exception:
        return str(result)

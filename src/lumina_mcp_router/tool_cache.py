"""Persistent tool catalogue cache.

Caches the per-backend list of tools (name, description, input schema, and
embedding vector) so the router can serve ``search_tools`` and report a
non-zero ``tools_indexed`` value even before backends have finished
(re)connecting. The cache lives in memory and is mirrored to a JSON file on
disk; on router startup the file is loaded eagerly so the embedding index is
warm immediately.

Design notes
------------
* The cache is per-backend. Refreshing one backend does NOT invalidate the
  entries cached for other backends.
* Each cached entry carries its embedding vector so we never have to call
  the embedder before the index is usable.
* Disk writes are best-effort — a failed write logs a structured error but
  never propagates: the in-memory copy is still authoritative.
* Path defaults to ``/var/lib/lumina/tool-cache.json`` and is configurable
  via ``LUMINA_TOOL_CACHE_PATH``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_CACHE_PATH = "/var/lib/lumina/tool-cache.json"


def default_cache_path() -> str:
    """Return the configured cache path (env var wins, falls back to default)."""
    return os.getenv("LUMINA_TOOL_CACHE_PATH", DEFAULT_CACHE_PATH)


@dataclass
class CachedTool:
    """One tool entry inside the catalogue cache."""

    backend: str
    name: str  # original name on the backend
    description: str
    input_schema: dict[str, Any]
    embedding: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CachedTool":
        return cls(
            backend=str(data["backend"]),
            name=str(data["name"]),
            description=str(data.get("description", "") or ""),
            input_schema=dict(data.get("input_schema") or {}),
            embedding=list(data.get("embedding") or []),
        )


class ToolCache:
    """In-memory + on-disk tool catalogue cache, keyed by backend name.

    The cache is intentionally simple: a dict[backend_name, list[CachedTool]].
    Replacement is wholesale per-backend (matches MCP semantics: a backend's
    tool list is the authoritative source whenever it's reachable).
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or default_cache_path()
        self._by_backend: dict[str, list[CachedTool]] = {}

    # ---- accessors --------------------------------------------------------

    def backends(self) -> list[str]:
        return list(self._by_backend.keys())

    def get(self, backend: str) -> list[CachedTool]:
        return list(self._by_backend.get(backend, []))

    def all(self) -> list[CachedTool]:
        out: list[CachedTool] = []
        for tools in self._by_backend.values():
            out.extend(tools)
        return out

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_backend.values())

    # ---- mutators ---------------------------------------------------------

    def replace_backend(self, backend: str, tools: list[CachedTool]) -> None:
        """Replace the cached tool list for ``backend`` and persist."""
        self._by_backend[backend] = list(tools)
        self._persist_safely()

    def remove_backend(self, backend: str) -> None:
        if backend in self._by_backend:
            del self._by_backend[backend]
            self._persist_safely()

    # ---- persistence ------------------------------------------------------

    def load(self) -> int:
        """Load cache from disk; return number of tools loaded.

        Missing file is treated as empty cache. Corrupt file is treated as
        empty cache (with an error log) — we never crash startup over a bad
        cache.
        """
        p = Path(self.path)
        if not p.exists():
            logger.info(
                "tool_cache_missing", extra={"path": str(p)}
            )
            self._by_backend = {}
            return 0
        try:
            raw = json.loads(p.read_text())
        except BaseException as e:  # noqa: BLE001
            logger.error(
                "tool_cache_load_failed",
                extra={"path": str(p), "error": str(e) or type(e).__name__},
            )
            self._by_backend = {}
            return 0
        loaded: dict[str, list[CachedTool]] = {}
        for backend, items in (raw.get("backends") or {}).items():
            tools: list[CachedTool] = []
            for item in items or []:
                try:
                    tools.append(CachedTool.from_json(item))
                except BaseException as e:  # noqa: BLE001
                    logger.warning(
                        "tool_cache_skip_bad_entry",
                        extra={
                            "backend": backend,
                            "error": str(e) or type(e).__name__,
                        },
                    )
            loaded[str(backend)] = tools
        self._by_backend = loaded
        total = sum(len(v) for v in loaded.values())
        logger.info(
            "tool_cache_loaded",
            extra={"path": str(p), "backends": len(loaded), "tools": total},
        )
        return total

    def _persist_safely(self) -> None:
        try:
            self._persist()
        except BaseException as e:  # noqa: BLE001
            logger.error(
                "tool_cache_persist_failed",
                extra={"path": self.path, "error": str(e) or type(e).__name__},
            )

    def _persist(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "backends": {
                backend: [t.to_json() for t in tools]
                for backend, tools in self._by_backend.items()
            },
        }
        # Atomic write: tmp file + rename.
        with tempfile.NamedTemporaryFile(
            "w",
            dir=str(p.parent),
            prefix=p.name + ".",
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as fh:
            json.dump(payload, fh, ensure_ascii=False)
            tmp = fh.name
        os.replace(tmp, p)

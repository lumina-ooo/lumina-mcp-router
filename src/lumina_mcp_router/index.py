"""In-memory vector index with cosine similarity."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ToolEntry:
    name: str              # globally-unique tool name as exposed to the LLM
    backend: str           # backend identifier (gsuite, microsoft, ...)
    original_name: str     # original tool name as known by the backend
    description: str
    input_schema: dict[str, Any]
    embedding: np.ndarray = field(repr=False)


@dataclass
class SearchResult:
    name: str
    backend: str
    description: str
    input_schema: dict[str, Any]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "description": self.description,
            "input_schema": self.input_schema,
            "score": round(float(self.score), 6),
        }


class VectorIndex:
    """Naive in-memory vector index with cosine similarity.

    Adequate for a few hundred tools; no ANN needed at this scale.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ToolEntry] = {}

    def clear(self) -> None:
        self._entries.clear()

    def add(self, entry: ToolEntry) -> None:
        self._entries[entry.name] = entry

    def remove(self, name: str) -> bool:
        """Remove an entry by globally-unique name; return True if removed."""
        return self._entries.pop(name, None) is not None

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, name: str) -> ToolEntry | None:
        return self._entries.get(name)

    def all(self) -> list[ToolEntry]:
        return list(self._entries.values())

    def search(self, query_embedding: list[float] | np.ndarray, top_k: int = 10) -> list[SearchResult]:
        if not self._entries:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        results: list[tuple[float, ToolEntry]] = []
        for e in self._entries.values():
            v = e.embedding
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            score = float(np.dot(q, v / v_norm))
            results.append((score, e))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[: max(0, top_k)]
        return [
            SearchResult(
                name=e.name,
                backend=e.backend,
                description=e.description,
                input_schema=e.input_schema,
                score=s,
            )
            for s, e in top
        ]

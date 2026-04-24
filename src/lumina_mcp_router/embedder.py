"""Ollama embeddings client."""
from __future__ import annotations

import logging
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


class Embedder:
    """Thin async client for Ollama /api/embeddings."""

    def __init__(
        self,
        base_url: str,
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed(self, text: str) -> list[float]:
        """Return embedding vector for a single text input."""
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding")
        if not emb:
            raise RuntimeError(f"Ollama returned no embedding: {data}")
        return emb

    async def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of texts sequentially (Ollama /api/embeddings is single-input)."""
        out: list[list[float]] = []
        for t in texts:
            out.append(await self.embed(t))
        return out

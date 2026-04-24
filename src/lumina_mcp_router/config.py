"""Configuration loading: env vars + backends.yaml."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


Transport = Literal["sse", "streamablehttp"]
VALID_TRANSPORTS: tuple[Transport, ...] = ("sse", "streamablehttp")


@dataclass(frozen=True)
class BackendConfig:
    name: str
    url: str
    transport: Transport = "sse"
    # Optional human-readable descriptive context about the backend, prepended
    # to each tool's embedded text to sharpen semantic search (e.g.
    # differentiate Gmail vs Outlook). See ``tools.build_indexed_text``.
    embedding_context: str | None = None


@dataclass
class Config:
    backends_config_path: str = field(
        default_factory=lambda: os.getenv(
            "MCP_BACKENDS_CONFIG", "/etc/lumina-mcp-router/backends.yaml"
        )
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://ollama.ai.svc.cluster.local:11434"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    listen_port: int = field(default_factory=lambda: int(os.getenv("LISTEN_PORT", "8080")))
    listen_host: str = field(default_factory=lambda: os.getenv("LISTEN_HOST", "0.0.0.0"))
    reindex_endpoint_enabled: bool = field(
        default_factory=lambda: os.getenv("REINDEX_ENDPOINT_ENABLED", "true").lower()
        in ("1", "true", "yes")
    )
    backends: list[BackendConfig] = field(default_factory=list)

    def load_backends(self) -> list[BackendConfig]:
        path = Path(self.backends_config_path)
        if not path.exists():
            self.backends = []
            return self.backends
        raw = yaml.safe_load(path.read_text()) or {}
        out: list[BackendConfig] = []
        for b in raw.get("backends", []):
            transport = (b.get("transport") or "sse").lower()
            if transport not in VALID_TRANSPORTS:
                raise ValueError(
                    f"backend {b.get('name')!r}: invalid transport "
                    f"{transport!r}, must be one of {VALID_TRANSPORTS}"
                )
            out.append(
                BackendConfig(
                    name=b["name"],
                    url=b["url"],
                    transport=transport,  # type: ignore[arg-type]
                    embedding_context=(b.get("embedding_context") or None),
                )
            )
        self.backends = out
        return out


def load_config() -> Config:
    cfg = Config()
    cfg.load_backends()
    return cfg

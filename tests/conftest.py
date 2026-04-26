"""Shared pytest fixtures for the lumina-mcp-router test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_tool_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the persistent tool cache at a per-test temp file.

    The default cache path (``/var/lib/lumina/tool-cache.json``) is not
    writable in CI and we don't want tests sharing state via the real
    on-disk cache anyway. Setting the env var here means every ``ToolCache()``
    constructed during a test gets a fresh, isolated location.
    """
    monkeypatch.setenv(
        "LUMINA_TOOL_CACHE_PATH", str(tmp_path / "tool-cache.json")
    )

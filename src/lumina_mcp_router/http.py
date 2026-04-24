"""HTTP client helpers for talking to MCP backends.

The MCP Python SDK enables ``TransportSecurityMiddleware`` by default, which
rejects incoming requests whose ``Host`` header is not in the backend's
allowlist (HTTP 421 ``Misdirected Request``). In a Kubernetes cluster our
router addresses backends by their ``*.svc.cluster.local`` DNS name, so the
outgoing ``Host`` header is rejected by default.

Rather than require every backend operator to loosen that allowlist, the
router forces ``Host: localhost`` on every outgoing HTTP request it sends to
backends. ``localhost`` is accepted by the SDK's default allowlist.

This module provides a drop-in replacement for the MCP SDK's default
``create_mcp_http_client`` that injects the overridden ``Host`` header.
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from mcp.shared._httpx_utils import create_mcp_http_client

__all__ = ["DEFAULT_HOST_OVERRIDE", "build_host_override_client_factory"]


DEFAULT_HOST_OVERRIDE = os.getenv("MCP_BACKEND_HOST_OVERRIDE", "localhost")


def build_host_override_client_factory(host: str = DEFAULT_HOST_OVERRIDE) -> Any:
    """Return an ``McpHttpClientFactory`` that forces ``Host: <host>``.

    The returned callable has the same signature as
    :func:`mcp.shared._httpx_utils.create_mcp_http_client` and can be passed
    directly to ``sse_client(..., httpx_client_factory=...)`` or
    ``streamablehttp_client(..., httpx_client_factory=...)``.

    httpx uses the ``Host`` header from the client's default headers when set
    explicitly, instead of deriving it from the request URL. This is exactly
    what we need to bypass MCP's DNS rebinding protection.
    """

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        merged: dict[str, str] = dict(headers or {})
        # Force-override whatever Host the caller (or the SDK) may have set.
        merged["Host"] = host
        return create_mcp_http_client(headers=merged, timeout=timeout, auth=auth)

    return factory

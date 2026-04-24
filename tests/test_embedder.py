import pytest
import httpx

from lumina_mcp_router.embedder import Embedder


@pytest.mark.asyncio
async def test_embed_returns_vector():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embeddings"
        body = request.read()
        assert b"nomic-embed-text" in body
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    embedder = Embedder(base_url="http://fake", model="nomic-embed-text", client=client)

    vec = await embedder.embed("hello world")
    assert vec == [0.1, 0.2, 0.3]
    await client.aclose()


@pytest.mark.asyncio
async def test_embed_many_sequentially():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.read())
        return httpx.Response(200, json={"embedding": [float(len(calls))]})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    embedder = Embedder(base_url="http://fake", client=client)

    out = await embedder.embed_many(["a", "b", "c"])
    assert out == [[1.0], [2.0], [3.0]]
    assert len(calls) == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_embed_raises_on_missing_embedding():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    embedder = Embedder(base_url="http://fake", client=client)

    with pytest.raises(RuntimeError):
        await embedder.embed("x")
    await client.aclose()

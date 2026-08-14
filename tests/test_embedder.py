"""Unit tests for OllamaEmbedder (HTTP mocked — Ollama need not be running)."""

from __future__ import annotations

import httpx
import pytest

from app.embeddings.embedder import OllamaEmbedder


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_embed_one_parses_vector() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        body = request.read()
        assert b"nomic-embed-text" in body
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]]},
        )

    client = httpx.Client(
        transport=_mock_transport(handler),
        base_url="http://test",
    )
    embedder = OllamaEmbedder(client=client)

    vector = embedder.embed_one("hello chunk")

    assert vector == [0.1, 0.2, 0.3]
    assert embedder.dimensions == 3
    assert embedder.model_name == "nomic-embed-text"
    embedder.close()


def test_embed_batch_preserves_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[1.0], [2.0], [3.0]]},
        )

    client = httpx.Client(transport=_mock_transport(handler), base_url="http://test")
    embedder = OllamaEmbedder(client=client)

    vectors = embedder.embed(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]
    embedder.close()


def test_embed_empty_list() -> None:
    embedder = OllamaEmbedder(client=httpx.Client(base_url="http://test"))
    assert embedder.embed([]) == []
    embedder.close()


def test_count_mismatch_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0]]})

    client = httpx.Client(transport=_mock_transport(handler), base_url="http://test")
    embedder = OllamaEmbedder(client=client)

    with pytest.raises(RuntimeError, match="returned 1 embeddings for 2"):
        embedder.embed(["a", "b"])
    embedder.close()


def test_model_missing_raises_helpful_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="model not found")

    client = httpx.Client(transport=_mock_transport(handler), base_url="http://test")
    embedder = OllamaEmbedder(model="missing-model", client=client)

    with pytest.raises(RuntimeError, match="ollama pull missing-model"):
        embedder.embed_one("x")
    embedder.close()


def test_connection_error_is_actionable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=_mock_transport(handler), base_url="http://test")
    embedder = OllamaEmbedder(client=client)

    with pytest.raises(ConnectionError, match="Cannot reach Ollama"):
        embedder.embed_one("x")
    embedder.close()


@pytest.mark.integration
def test_live_ollama_nomic_embed_text() -> None:
    """Optional: run with LIVE_OLLAMA=1 and a pulled model.

    Example::

        LIVE_OLLAMA=1 PYTHONPATH=. pytest tests/test_embedder.py -m integration -v
    """
    import os

    if os.environ.get("LIVE_OLLAMA") != "1":
        pytest.skip("Set LIVE_OLLAMA=1 to hit a real Ollama server")

    with OllamaEmbedder() as embedder:
        a = embedder.embed_one("compute invoice payment state")
        b = embedder.embed_one("calculate payment_state for an invoice")
        c = embedder.embed_one("sort a list of integers with quicksort")

    assert embedder.dimensions == 768
    assert len(a) == 768

    def cosine(u: list[float], v: list[float]) -> float:
        dot = sum(x * y for x, y in zip(u, v))
        nu = sum(x * x for x in u) ** 0.5
        nv = sum(y * y for y in v) ** 0.5
        return dot / (nu * nv)

    # Semantically related code questions should be closer than unrelated ones.
    assert cosine(a, b) > cosine(a, c)

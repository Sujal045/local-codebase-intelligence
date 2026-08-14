"""Ollama-backed embedding client.

Ollama runs a local HTTP server (default ``http://127.0.0.1:11434``).
We send text to ``POST /api/embed`` and receive float vectors.

Under the hood (conceptually):
1. The model tokenizes the input text.
2. A neural network maps tokens → a dense vector of fixed size
   (nomic-embed-text uses 768 dimensions).
3. We use that vector later for nearest-neighbor search in Qdrant.

We are not implementing the neural net; Ollama loads and runs the model.
We *are* responsible for the API contract, batching, and validation.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaEmbedder:
    """Embed text via a local Ollama server.

    Example::

        embedder = OllamaEmbedder()
        vectors = embedder.embed(["def add(a, b):\\n    return a + b"])
        assert len(vectors) == 1
        assert len(vectors[0]) == embedder.dimensions
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{i}] must be str, got {type(text)!r}")

        payload = {"model": self._model, "input": texts}
        data = self._post_embed(payload)
        vectors = self._parse_embeddings(data)

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Ollama returned {len(vectors)} embeddings for {len(texts)} inputs"
            )

        dim = len(vectors[0])
        for i, vector in enumerate(vectors):
            if len(vector) != dim:
                raise RuntimeError(
                    f"Inconsistent embedding sizes: index 0 has {dim}, "
                    f"index {i} has {len(vector)}"
                )
            if dim == 0:
                raise RuntimeError("Ollama returned an empty embedding vector")

        self._dimensions = dim
        return vectors

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OllamaEmbedder:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout_s,
            )
        return self._client

    def _post_embed(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        try:
            response = client.post("/api/embed", json=payload)
        except httpx.RequestError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._base_url}. "
                "Is Ollama installed and running? "
                "Install from https://ollama.com , then run: "
                f"ollama pull {self._model}"
            ) from exc

        if response.status_code == 404:
            raise RuntimeError(
                f"Embedding model {self._model!r} not found on Ollama. "
                f"Run: ollama pull {self._model}"
            )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama /api/embed failed ({response.status_code}): {response.text}"
            ) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Unexpected Ollama response type: {type(data)!r}")
        return data

    @staticmethod
    def _parse_embeddings(data: dict[str, Any]) -> list[list[float]]:
        # Current Ollama /api/embed shape: {"embeddings": [[...], ...]}
        if "embeddings" in data:
            raw = data["embeddings"]
            if not isinstance(raw, list):
                raise RuntimeError("Ollama 'embeddings' field is not a list")
            return [_as_float_list(item, index=i) for i, item in enumerate(raw)]

        # Older /api/embeddings single-vector shape (defensive).
        if "embedding" in data:
            return [_as_float_list(data["embedding"], index=0)]

        raise RuntimeError(
            "Ollama response missing 'embeddings' (or legacy 'embedding') field"
        )


def _as_float_list(item: Any, *, index: int) -> list[float]:
    if not isinstance(item, list) or not item:
        raise RuntimeError(f"Embedding at index {index} is not a non-empty list")
    try:
        return [float(x) for x in item]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Embedding at index {index} has non-numeric values") from exc

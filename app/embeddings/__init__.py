"""Local text embeddings for Version 1 Code RAG (Slice 1b).

An embedding model maps a string to a fixed-length list of floats (a vector)
so that semantically similar texts land near each other in vector space.

This package talks to a local Ollama server. It does not store vectors in
Qdrant yet (that is Slice 1c).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.embeddings.embedder import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaEmbedder,
)

__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "Embedder",
    "OllamaEmbedder",
]


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into vectors."""

    @property
    def model_name(self) -> str:
        """Ollama (or other) model id used for embeddings."""

    @property
    def dimensions(self) -> int | None:
        """Vector length once known; None until the first successful embed."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more texts. Order of outputs matches inputs."""

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for a single string."""

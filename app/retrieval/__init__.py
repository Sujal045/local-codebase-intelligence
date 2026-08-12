"""Vector storage and similarity search (Slice 1c).

Qdrant stores:
- the embedding vector (for nearest-neighbor search)
- a payload dict with chunk metadata (path, lines, text)

This package does not call Ollama or build prompts (Slices 1b / 1d).
"""

from __future__ import annotations

from app.retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_URL,
    ScoredChunk,
    QdrantVectorStore,
)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_QDRANT_URL",
    "QdrantVectorStore",
    "ScoredChunk",
]

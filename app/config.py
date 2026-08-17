"""Shared defaults for CLI and services."""

from __future__ import annotations

from app.embeddings.embedder import DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_BASE_URL
from app.llm.ollama_chat import DEFAULT_CHAT_MODEL
from app.reranking.reranker import DEFAULT_RERANK_MODEL
from app.retrieval.vector_store import DEFAULT_COLLECTION_NAME, DEFAULT_QDRANT_URL

# nomic-embed-text produces 768-dimensional vectors.
NOMIC_EMBED_DIMENSIONS = 768

DEFAULT_CHUNK_SIZE = 40
DEFAULT_OVERLAP = 10
DEFAULT_TOP_K = 5
# Hits requested from *each* retriever before Reciprocal Rank Fusion.
DEFAULT_CANDIDATE_LIMIT = 20
DEFAULT_VECTOR_SIZE = NOMIC_EMBED_DIMENSIONS

__all__ = [
    "DEFAULT_CANDIDATE_LIMIT",
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OVERLAP",
    "DEFAULT_QDRANT_URL",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_TOP_K",
    "DEFAULT_VECTOR_SIZE",
    "NOMIC_EMBED_DIMENSIONS",
]

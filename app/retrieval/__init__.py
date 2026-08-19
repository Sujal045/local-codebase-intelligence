"""Vector storage and similarity search (Slice 1c).

Qdrant stores:
- the embedding vector (for nearest-neighbor search)
- a payload dict with chunk metadata (path, lines, text)

The RAG ask() helper lives in ``app.retrieval.rag`` (Slice 1d) and is not
re-exported here, to avoid circular imports with ``app.llm.prompt``.

Lexical BM25 search lives in ``app.retrieval.bm25`` (Slice 3A).
Hybrid fusion lives in ``app.retrieval.hybrid`` (Slice 3B).
``retrieve_chunks`` in ``app.retrieval.query`` is the shared retrieve
step (Slice 5A) used by ``ask()`` and by ``search_code``.
``ask()`` in ``app.retrieval.rag`` still generates in one shot.
"""

from __future__ import annotations

from app.retrieval.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_QDRANT_URL,
    QdrantVectorStore,
    ScoredChunk,
)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_QDRANT_URL",
    "QdrantVectorStore",
    "ScoredChunk",
]

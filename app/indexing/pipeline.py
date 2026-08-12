"""Wire chunking + embeddings + Qdrant for indexing (Slice 1c)."""

from __future__ import annotations

from app.embeddings import Embedder
from app.indexing.chunker import Chunk
from app.retrieval.vector_store import QdrantVectorStore


def index_chunks(
    store: QdrantVectorStore,
    embedder: Embedder,
    chunks: list[Chunk],
) -> int:
    """Embed chunks and upsert them into Qdrant.

    Returns the number of points written.
    """
    if not chunks:
        return 0

    texts = [chunk.text for chunk in chunks]
    vectors = embedder.embed(texts)
    return store.upsert_chunks(chunks, vectors)

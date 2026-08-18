"""Retrieve a broad hybrid pool, then rerank it (Slice 4B).

Hybrid search (Version 3) is cheap and high-recall. It still ranks with
RRF, which never reads the question and a chunk together.

This function is the two-stage pipeline:

    question
      → hybrid_search(limit=candidate_limit)   # e.g. 20 fused hits
      → reranker.rerank(limit=limit)           # e.g. 5 final hits

If the right chunk is RRF #12, a top-5-only search would drop it. A
20-hit pool still contains it, so the cross-encoder can promote it.

``ask()`` (Slice 4C) calls this when a ``reranker`` is injected.
"""

from __future__ import annotations

from app.config import DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K
from app.embeddings import Embedder
from app.reranking.reranker import Reranker
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import DEFAULT_RRF_K, hybrid_search
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk


def search_and_rerank(
    query: str,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    bm25: Bm25Index,
    reranker: Reranker,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    limit: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """Fuse vector + BM25 into a candidate pool, then rerank to ``limit``.

    Args:
        query: Natural-language or identifier query.
        embedder: Used by vector search inside ``hybrid_search``.
        store: Qdrant collection already holding chunk embeddings.
        bm25: Lexical index over the same chunks.
        reranker: Scores ``(query, chunk.text)`` pairs (real or fake).
        candidate_limit: RRF pool size (recall). Must be >= 1.
        limit: How many chunks to keep after reranking (precision).
        rrf_k: Damping constant forwarded to Reciprocal Rank Fusion.
    """
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must be non-empty")
    if candidate_limit < 1:
        raise ValueError(f"candidate_limit must be >= 1, got {candidate_limit}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")

    pool = max(candidate_limit, limit)
    candidates = hybrid_search(
        stripped,
        embedder=embedder,
        store=store,
        bm25=bm25,
        vector_limit=pool,
        bm25_limit=pool,
        limit=pool,
        rrf_k=rrf_k,
    )
    return reranker.rerank(stripped, candidates, limit=limit)

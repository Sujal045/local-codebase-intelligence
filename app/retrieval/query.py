"""Shared retrieval used by ``ask()`` and by tools (Slice 5A).

``ask()`` always retrieves then generates. A tool-using agent (Version 6)
must be able to retrieve *without* generating, so the LLM can decide
whether to read a file next.

Both paths call ``retrieve_chunks`` so ranking cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K
from app.embeddings import Embedder
from app.reranking.pipeline import search_and_rerank
from app.reranking.reranker import Reranker
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked chunks plus whether a cross-encoder produced the scores."""

    chunks: list[ScoredChunk]
    reranked: bool


def retrieve_chunks(
    query: str,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    limit: int = DEFAULT_TOP_K,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    bm25: Bm25Index | None = None,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Hybrid-retrieve (and optionally rerank) chunks for ``query``."""
    stripped = query.strip()
    if not stripped:
        raise ValueError("query must be non-empty")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if candidate_limit < 1:
        raise ValueError(f"candidate_limit must be >= 1, got {candidate_limit}")

    lexical = bm25 if bm25 is not None else Bm25Index.from_store(store)
    if reranker is not None:
        chunks = search_and_rerank(
            stripped,
            embedder=embedder,
            store=store,
            bm25=lexical,
            reranker=reranker,
            candidate_limit=candidate_limit,
            limit=limit,
        )
        return RetrievalResult(chunks=chunks, reranked=True)

    pool = max(candidate_limit, limit)
    chunks = hybrid_search(
        stripped,
        embedder=embedder,
        store=store,
        bm25=lexical,
        vector_limit=pool,
        bm25_limit=pool,
        limit=limit,
    )
    return RetrievalResult(chunks=chunks, reranked=False)

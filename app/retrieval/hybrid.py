"""Hybrid ranking: fuse vector hits and BM25 hits (Slice 3B).

Vector scores (cosine ~0–1) and BM25 scores (unbounded) are not on the same
scale. Adding them is meaningless: a BM25 4.2 is not “more similar” than
cosine 0.81.

Reciprocal Rank Fusion (RRF) ignores raw scores and uses *ranks*:

    RRF(d) = Σ_over_lists  1 / (k + rank_list(d))

``rank`` is 1-based (the top hit is rank 1). ``k`` (typical 60) dampens
the gap between rank 1 and rank 20 so one list cannot dominate.

A chunk that appears high on *both* lists ranks above a chunk that is #1
on only one list. Identifier queries (BM25) and conceptual queries
(vectors) can both contribute.

``ask()`` (Slice 3C) calls ``hybrid_search`` after rebuilding BM25 from
the Qdrant payload corpus.
"""

from __future__ import annotations

from app.config import DEFAULT_CANDIDATE_LIMIT
from app.embeddings import Embedder
from app.retrieval.bm25 import Bm25Index
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk

# Cormack, Clarke, Buettcher (2009). Larger k → flatter rank contributions.
DEFAULT_RRF_K = 60


def chunk_key(chunk: ScoredChunk) -> tuple[str, int, int]:
    """Identity for merging the same code unit from two retrievers."""
    return (chunk.path, chunk.start_line, chunk.end_line)


def reciprocal_rank_fusion(
    rankings: list[list[ScoredChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int = 5,
) -> list[ScoredChunk]:
    """Merge ranked lists with RRF. ``limit`` is the fused top-N."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    if not rankings:
        return []

    scores: dict[tuple[str, int, int], float] = {}
    chunks: dict[tuple[str, int, int], ScoredChunk] = {}

    for ranking in rankings:
        seen: set[tuple[str, int, int]] = set()
        for rank, chunk in enumerate(ranking, start=1):
            key = chunk_key(chunk)
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            chunks.setdefault(key, chunk)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused: list[ScoredChunk] = []
    for key, score in ordered[:limit]:
        fused.append(_with_score(chunks[key], score))
    return fused


def hybrid_search(
    query: str,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    bm25: Bm25Index,
    vector_limit: int = DEFAULT_CANDIDATE_LIMIT,
    bm25_limit: int = DEFAULT_CANDIDATE_LIMIT,
    limit: int = 5,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """Retrieve candidates from both engines, then fuse with RRF.

    Each retriever is asked for more than ``limit`` hits (default 20) so a
    chunk that is #8 on BM25 and #9 on vectors can still enter the fused
    top-5. ``search_and_rerank`` (Slice 4B) keeps this fused pool large
    and hands it to a cross-encoder.
    """
    if not query.strip():
        raise ValueError("query must be non-empty")
    if vector_limit < 1 or bm25_limit < 1:
        raise ValueError("candidate limits must be >= 1")

    vector_hits = store.search(embedder.embed_one(query.strip()), limit=vector_limit)
    bm25_hits = bm25.search(query, limit=bm25_limit)
    return reciprocal_rank_fusion(
        [vector_hits, bm25_hits],
        k=rrf_k,
        limit=limit,
    )


def _with_score(chunk: ScoredChunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=chunk.text,
        score=score,
        language=chunk.language,
        symbol=chunk.symbol,
        kind=chunk.kind,
        name=chunk.name,
        parent=chunk.parent,
    )

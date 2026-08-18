"""Tests for retrieve-then-rerank (Slice 4B)."""

from __future__ import annotations

import pytest

from app.indexing.chunker import Chunk
from app.indexing.pipeline import index_chunks
from app.reranking import search_and_rerank
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import hybrid_search
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk
from tests.fakes import FakeEmbedder, FakeReranker


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="def split_windows(lines):\n    return lines[i:j]",
            path="app/indexing/chunker.py",
            start_line=30,
            end_line=36,
            symbol="split_windows",
            kind="function",
            name="split_windows",
        ),
        Chunk(
            text="def rank_score(match, genuineness):\n    return match * 0.7",
            path="src/scoring.py",
            start_line=10,
            end_line=11,
            symbol="rank_score",
            kind="function",
            name="rank_score",
        ),
        Chunk(
            text="def compute_genuineness(job):\n    flag spam jobs as is_spam",
            path="src/scoring.py",
            start_line=1,
            end_line=8,
            symbol="compute_genuineness",
            kind="function",
            name="compute_genuineness",
        ),
        Chunk(
            text="def walk_repo(root):\n    yield from files",
            path="app/indexing/walker.py",
            start_line=1,
            end_line=4,
            symbol="walk_repo",
            kind="function",
            name="walk_repo",
        ),
    ]


class RecordingReranker:
    """Captures the hybrid pool, then delegates to FakeReranker."""

    def __init__(self) -> None:
        self.pool: list[ScoredChunk] = []
        self._inner = FakeReranker()

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        *,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        self.pool = list(candidates)
        return self._inner.rerank(query, candidates, limit=limit)


class LastWinsReranker:
    """Ignores relevance: the last hybrid hit becomes rank 1.

    That proves ``search_and_rerank`` follows the reranker, not RRF order.
    """

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        *,
        limit: int = 5,
    ) -> list[ScoredChunk]:
        return list(reversed(candidates))[:limit]


def test_search_and_rerank_passes_the_broad_pool() -> None:
    chunks = _chunks()
    embedder = FakeEmbedder()
    recorder = RecordingReranker()
    with QdrantVectorStore(
        collection_name="rerank_pool",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, chunks)
        hits = search_and_rerank(
            "detect spam jobs",
            embedder=embedder,
            store=store,
            bm25=Bm25Index(chunks),
            reranker=recorder,
            candidate_limit=4,
            limit=1,
        )

    assert len(recorder.pool) == 4
    assert len(hits) == 1
    assert hits[0].symbol == "compute_genuineness"


def test_search_and_rerank_can_promote_a_lower_rrf_hit() -> None:
    """Reranker reorders the fused list; it is not stuck with RRF's #1."""
    chunks = _chunks()
    embedder = FakeEmbedder()
    query = "detect spam jobs"
    with QdrantVectorStore(
        collection_name="rerank_promote",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, chunks)
        bm25 = Bm25Index(chunks)
        fused = hybrid_search(
            query,
            embedder=embedder,
            store=store,
            bm25=bm25,
            vector_limit=4,
            bm25_limit=4,
            limit=4,
        )
        reranked = search_and_rerank(
            query,
            embedder=embedder,
            store=store,
            bm25=bm25,
            reranker=LastWinsReranker(),
            candidate_limit=4,
            limit=1,
        )

    assert len(fused) == 4
    assert reranked[0].symbol == fused[-1].symbol
    assert reranked[0].symbol != fused[0].symbol


def test_search_and_rerank_rejects_empty_query() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="rerank_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        with pytest.raises(ValueError, match="query must be non-empty"):
            search_and_rerank(
                "  ",
                embedder=embedder,
                store=store,
                bm25=Bm25Index([]),
                reranker=FakeReranker(),
            )


def test_search_and_rerank_rejects_bad_limits() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="rerank_limits",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        with pytest.raises(ValueError, match="candidate_limit"):
            search_and_rerank(
                "spam",
                embedder=embedder,
                store=store,
                bm25=Bm25Index([]),
                reranker=FakeReranker(),
                candidate_limit=0,
            )
        with pytest.raises(ValueError, match="limit"):
            search_and_rerank(
                "spam",
                embedder=embedder,
                store=store,
                bm25=Bm25Index([]),
                reranker=FakeReranker(),
                limit=0,
            )

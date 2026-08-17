"""Tests for Reciprocal Rank Fusion and hybrid search (Slice 3B)."""

from __future__ import annotations

import pytest

from app.indexing.chunker import Chunk
from app.indexing.pipeline import index_chunks
from app.retrieval.bm25 import Bm25Index
from app.retrieval.hybrid import (
    hybrid_search,
    reciprocal_rank_fusion,
)
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk
from tests.fakes import FakeEmbedder


def _hit(
    path: str,
    start: int,
    end: int,
    *,
    score: float = 0.0,
    text: str = "code",
    symbol: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        path=path,
        start_line=start,
        end_line=end,
        text=text,
        score=score,
        symbol=symbol,
        kind="function" if symbol else None,
        name=symbol,
    )


def test_rrf_prefers_chunks_present_in_both_lists() -> None:
    both = _hit("a.py", 1, 5, symbol="both")
    vector_only = _hit("b.py", 1, 5, symbol="vector_only")
    bm25_only = _hit("c.py", 1, 5, symbol="bm25_only")

    fused = reciprocal_rank_fusion(
        [
            [both, vector_only],
            [both, bm25_only],
        ],
        k=60,
        limit=3,
    )

    assert [h.symbol for h in fused] == ["both", "vector_only", "bm25_only"]
    assert fused[0].score > fused[1].score
    assert fused[1].score == fused[2].score


def test_rrf_uses_path_and_lines_as_identity() -> None:
    left = _hit("a.py", 10, 20, score=0.99, text="from vector")
    right = _hit("a.py", 10, 20, score=12.0, text="from bm25")
    fused = reciprocal_rank_fusion([[left], [right]], limit=1)
    assert len(fused) == 1
    assert fused[0].path == "a.py"
    assert fused[0].start_line == 10


def test_single_list_preserves_relative_order() -> None:
    a = _hit("a.py", 1, 2, symbol="a")
    b = _hit("b.py", 1, 2, symbol="b")
    fused = reciprocal_rank_fusion([[a, b], []], limit=2)
    assert [h.symbol for h in fused] == ["a", "b"]


def test_empty_rankings() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank_fusion([[]], k=-1)
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion([[]], limit=0)


def test_hybrid_search_identifier_survives_fusion() -> None:
    chunks = [
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
            text="def compute_genuineness(job):\n    return {'is_spam': False}",
            path="src/scoring.py",
            start_line=1,
            end_line=8,
            symbol="compute_genuineness",
            kind="function",
            name="compute_genuineness",
        ),
        Chunk(
            text="def split_windows(lines):\n    return lines",
            path="app/indexing/chunker.py",
            start_line=30,
            end_line=36,
            symbol="chunk_file",
            kind="function",
            name="chunk_file",
        ),
    ]
    embedder = FakeEmbedder()
    bm25 = Bm25Index(chunks)
    with QdrantVectorStore(
        collection_name="hybrid_test",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, chunks)
        hits = hybrid_search(
            "compute_genuineness",
            embedder=embedder,
            store=store,
            bm25=bm25,
            vector_limit=3,
            bm25_limit=3,
            limit=3,
        )

    symbols = [h.symbol for h in hits]
    assert "compute_genuineness" in symbols
    assert hits[0].symbol == "compute_genuineness"


def test_hybrid_search_rejects_empty_query() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="hybrid_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        with pytest.raises(ValueError, match="query must be non-empty"):
            hybrid_search(
                "  ",
                embedder=embedder,
                store=store,
                bm25=Bm25Index([]),
            )

"""Tests for retrieve_chunks and the search_code tool (Slice 5A)."""

from __future__ import annotations

from app.indexing.chunker import Chunk
from app.indexing.pipeline import index_chunks
from app.retrieval.query import retrieve_chunks
from app.retrieval.vector_store import QdrantVectorStore
from app.tools import SearchCodeTool
from tests.fakes import FakeEmbedder, FakeReranker


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="def split_windows(lines):\n    return lines",
            path="app/indexing/chunker.py",
            start_line=30,
            end_line=36,
            symbol="chunk_file",
            kind="function",
            name="chunk_file",
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
    ]


def test_retrieve_chunks_finds_identifier() -> None:
    chunks = _chunks()
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="retrieve_id",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, chunks)
        result = retrieve_chunks(
            "compute_genuineness",
            embedder=embedder,
            store=store,
            limit=2,
            candidate_limit=2,
        )

    assert result.reranked is False
    assert result.chunks[0].symbol == "compute_genuineness"


def test_search_code_spec_is_openai_shaped() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_spec",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        spec = SearchCodeTool(embedder=embedder, store=store).spec()

    assert spec["type"] == "function"
    function = spec["function"]
    assert function["name"] == "search_code"
    assert "indexed" in function["description"].lower()
    params = function["parameters"]
    assert params["required"] == ["query"]
    assert "query" in params["properties"]
    assert "limit" in params["properties"]


def test_search_code_run_returns_observation_with_hits() -> None:
    chunks = _chunks()
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_run",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, chunks)
        tool = SearchCodeTool(
            embedder=embedder,
            store=store,
            reranker=FakeReranker(),
        )
        result = tool.run(query="detect spam jobs", limit=1)

    assert result.name == "search_code"
    assert result.hits[0].symbol == "compute_genuineness"
    assert "src/scoring.py:1-8" in result.content
    assert "compute_genuineness" in result.content
    assert "flag spam jobs" in result.content


def test_search_code_rejects_empty_query_as_observation() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        result = SearchCodeTool(embedder=embedder, store=store).run(query="  ")

    assert result.hits == ()
    assert result.content.startswith("error:")
    assert "query" in result.content


def test_search_code_rejects_unknown_arguments() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_extra",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        result = SearchCodeTool(embedder=embedder, store=store).run(
            query="spam",
            path="src/scoring.py",
        )

    assert "unexpected arguments" in result.content
    assert "path" in result.content

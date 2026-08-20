"""Tests for search_documentation (Slice 5D)."""

from __future__ import annotations

from app.indexing.chunker import Chunk
from app.indexing.pipeline import index_chunks
from app.retrieval.vector_store import QdrantVectorStore
from app.tools import SearchDocumentationTool
from app.tools.search_docs import (
    documentation_chunks,
    is_documentation_chunk,
    search_documentation_chunks,
)
from tests.fakes import FakeEmbedder


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            text="# Mini repo\nSpam detection lives in src/scoring.py.",
            path="README.md",
            start_line=1,
            end_line=3,
        ),
        Chunk(
            text="## Install\nRun pip install -e .",
            path="docs/guide.md",
            start_line=1,
            end_line=2,
        ),
        Chunk(
            text="API notes for partners",
            path="docs/api.py",
            start_line=1,
            end_line=1,
        ),
        Chunk(
            text="def compute_genuineness(job):\n    return {'is_spam': False}",
            path="src/scoring.py",
            start_line=1,
            end_line=2,
            symbol="compute_genuineness",
            kind="function",
            name="compute_genuineness",
            language="python",
        ),
    ]


def test_is_documentation_chunk_by_suffix_and_docs_dir() -> None:
    assert is_documentation_chunk(_chunks()[0])
    assert is_documentation_chunk(_chunks()[1])
    assert is_documentation_chunk(_chunks()[2])
    assert not is_documentation_chunk(_chunks()[3])


def test_documentation_chunks_filters_source() -> None:
    docs = documentation_chunks(_chunks())
    assert {c.path for c in docs} == {"README.md", "docs/guide.md", "docs/api.py"}


def test_search_documentation_prefers_readme_over_source() -> None:
    hits = search_documentation_chunks("spam detection", _chunks(), limit=3)
    assert hits
    assert hits[0].path == "README.md"
    assert all(is_documentation_chunk(h.to_chunk()) for h in hits)


def test_search_documentation_tool_observation() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_docs",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, _chunks())
        tool = SearchDocumentationTool(store=store)
        result = tool.run(query="Where is spam detection documented?", limit=2)

    assert result.name == "search_documentation"
    assert result.hits[0].path == "README.md"
    assert "documentation hit" in result.content
    assert "Spam detection lives" in result.content
    assert "compute_genuineness" not in result.content


def test_search_documentation_no_docs_in_index() -> None:
    embedder = FakeEmbedder()
    code_only = [_chunks()[3]]
    with QdrantVectorStore(
        collection_name="search_docs_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, code_only)
        result = SearchDocumentationTool(store=store).run(query="spam")

    assert result.hits == ()
    assert "No documentation chunks" in result.content


def test_search_documentation_spec_and_bad_args() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="search_docs_spec",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        tool = SearchDocumentationTool(store=store)
        spec = tool.spec()
        assert spec["function"]["name"] == "search_documentation"
        assert spec["function"]["parameters"]["required"] == ["query"]
        assert "Markdown" in spec["function"]["description"]
        bad = tool.run(query="  ")
        assert bad.content.startswith("error:")
        extra = tool.run(query="spam", path="README.md")
        assert "unexpected arguments" in extra.content

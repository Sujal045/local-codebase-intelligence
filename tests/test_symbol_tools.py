"""Tests for get_symbol and find_references (Slice 5C)."""

from __future__ import annotations

from app.indexing.chunker import Chunk
from app.indexing.pipeline import index_chunks
from app.retrieval.vector_store import QdrantVectorStore
from app.tools import FindReferencesTool, GetSymbolTool
from app.tools.symbol_lookup import (
    definition_rank,
    find_definitions,
    find_reference_sites,
    text_mentions_symbol,
)
from tests.fakes import FakeEmbedder


def _chunks() -> list[Chunk]:
    return [
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
            text="class UserService:\n    pass",
            path="src/users.py",
            start_line=1,
            end_line=2,
            symbol="UserService",
            kind="class",
            name="UserService",
        ),
        Chunk(
            text="def create_user(self, data):\n    return User()",
            path="src/users.py",
            start_line=4,
            end_line=5,
            symbol="UserService.create_user",
            kind="method",
            name="create_user",
            parent="UserService",
        ),
        Chunk(
            text="score = compute_genuineness(job)\nreturn score",
            path="src/pipeline.py",
            start_line=10,
            end_line=11,
            symbol="run_job",
            kind="function",
            name="run_job",
        ),
        Chunk(
            text="x = 1  # leftover module lines",
            path="src/scoring.py",
            start_line=20,
            end_line=20,
            symbol="<module>",
            kind="module",
            name="<module>",
        ),
    ]


def test_definition_rank_prefers_qualified_name() -> None:
    method = _chunks()[2]
    assert definition_rank(method, "UserService.create_user") == 3.0
    assert definition_rank(method, "create_user") == 2.0
    assert definition_rank(_chunks()[0], "create_user") is None
    assert definition_rank(_chunks()[4], "compute_genuineness") is None


def test_find_definitions_returns_function_chunk() -> None:
    hits = find_definitions(_chunks(), "compute_genuineness")
    assert len(hits) == 1
    assert hits[0].path == "src/scoring.py"
    assert hits[0].kind == "function"


def test_text_mentions_requires_identifier_boundary() -> None:
    assert text_mentions_symbol("score = compute_genuineness(job)", "compute_genuineness")
    assert not text_mentions_symbol("compute_genuineness_extra = 1", "compute_genuineness")


def test_find_reference_sites_excludes_definition() -> None:
    hits = find_reference_sites(_chunks(), "compute_genuineness")
    assert [h.path for h in hits] == ["src/pipeline.py"]
    assert hits[0].symbol == "run_job"


def test_get_symbol_tool_observation() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="get_symbol",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, _chunks())
        result = GetSymbolTool(store=store).run(symbol="compute_genuineness")

    assert result.name == "get_symbol"
    assert result.hits[0].symbol == "compute_genuineness"
    assert "Found 1 definition" in result.content
    assert "src/scoring.py:1-8" in result.content


def test_get_symbol_matches_unqualified_method() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="get_method",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, _chunks())
        result = GetSymbolTool(store=store).run(symbol="create_user")

    assert result.hits[0].symbol == "UserService.create_user"


def test_get_symbol_missing() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="get_missing",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, _chunks())
        result = GetSymbolTool(store=store).run(symbol="does_not_exist")

    assert result.hits == ()
    assert "No definitions found" in result.content


def test_find_references_tool_observation() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="find_refs",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, _chunks())
        result = FindReferencesTool(store=store).run(symbol="compute_genuineness")
        spec = FindReferencesTool(store=store).spec()

    assert result.name == "find_references"
    assert spec["function"]["name"] == "find_references"
    assert spec["function"]["parameters"]["required"] == ["symbol"]
    assert [h.path for h in result.hits] == ["src/pipeline.py"]
    assert "reference site" in result.content
    assert "run_job" in result.content


def test_get_symbol_spec_and_bad_args() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="get_spec",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        tool = GetSymbolTool(store=store)
        spec = tool.spec()
        assert spec["function"]["name"] == "get_symbol"
        assert spec["function"]["parameters"]["required"] == ["symbol"]
        bad = tool.run(symbol="  ")
        assert bad.content.startswith("error:")
        extra = tool.run(symbol="x", path="a.py")
        assert "unexpected arguments" in extra.content

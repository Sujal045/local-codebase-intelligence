"""Tests for in-memory BM25 search (Slice 3A)."""

from __future__ import annotations

import pytest

from app.indexing.chunker import Chunk
from app.retrieval.bm25 import Bm25Index, document_text, tokenize


def _chunk(
    text: str,
    *,
    path: str = "a.py",
    start: int = 1,
    end: int = 2,
    symbol: str | None = None,
    name: str | None = None,
    kind: str | None = "function",
) -> Chunk:
    return Chunk(
        text=text,
        path=path,
        start_line=start,
        end_line=end,
        language="python",
        symbol=symbol,
        kind=kind,
        name=name,
        parent=None,
    )


def test_tokenize_keeps_identifier_and_splits_snake_and_camel() -> None:
    tokens = tokenize("def compute_genuineness():\n    UserService.create_user()")
    assert "compute_genuineness" in tokens
    assert "compute" in tokens
    assert "genuineness" in tokens
    assert "userservice" in tokens
    assert "user" in tokens
    assert "service" in tokens
    assert "create_user" in tokens or "createuser" in tokens
    assert "create" in tokens


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("   \n") == []


def test_document_text_includes_path_and_symbol() -> None:
    chunk = _chunk(
        "def search():\n    pass",
        path="app/retrieval/vector_store.py",
        symbol="QdrantVectorStore.search",
        name="search",
    )
    text = document_text(chunk)
    assert "vector_store.py" in text
    assert "QdrantVectorStore.search" in text
    assert "def search" in text


def test_identifier_query_ranks_exact_function_first() -> None:
    chunks = [
        _chunk(
            "def rank_score(match, genuineness):\n    return match * 0.7",
            path="src/scoring.py",
            start=10,
            end=11,
            symbol="rank_score",
            name="rank_score",
        ),
        _chunk(
            "def compute_genuineness(job):\n    reasons = []\n    return {'is_spam': bool(reasons)}",
            path="src/scoring.py",
            start=1,
            end=8,
            symbol="compute_genuineness",
            name="compute_genuineness",
        ),
        _chunk(
            "def split_windows(lines):\n    return lines",
            path="app/indexing/chunker.py",
            start=30,
            end=36,
            symbol="chunk_file",
            name="chunk_file",
        ),
    ]
    index = Bm25Index(chunks)
    hits = index.search("compute_genuineness", limit=3)

    assert hits
    assert hits[0].symbol == "compute_genuineness"
    assert hits[0].score > hits[-1].score


def test_camel_case_query_matches_split_identifier() -> None:
    chunks = [
        _chunk(
            "class Ranker:\n    pass",
            symbol="Ranker",
            name="Ranker",
            kind="class",
        ),
        _chunk(
            "class UserService:\n    \"\"\"Users.\"\"\"",
            path="src/users.py",
            start=12,
            end=13,
            symbol="UserService",
            name="UserService",
            kind="class",
        ),
    ]
    index = Bm25Index(chunks)
    hits = index.search("user service", limit=2)
    assert hits[0].symbol == "UserService"


def test_empty_corpus_or_query_returns_no_hits() -> None:
    index = Bm25Index([])
    assert index.search("compute_genuineness") == []

    index = Bm25Index([_chunk("def foo():\n    return 1", symbol="foo", name="foo")])
    assert index.search("   ") == []


def test_search_rejects_invalid_limit() -> None:
    index = Bm25Index([_chunk("def foo():\n    pass", symbol="foo", name="foo")])
    with pytest.raises(ValueError, match="limit must be >= 1"):
        index.search("foo", limit=0)


def test_invalid_params_raise() -> None:
    with pytest.raises(ValueError, match="k1"):
        Bm25Index(k1=-1)
    with pytest.raises(ValueError, match="b must be"):
        Bm25Index(b=1.5)


def test_rare_identifier_outranks_common_keywords() -> None:
    common = [
        _chunk(f"def helper_{i}():\n    return 1\n", path=f"h{i}.py", symbol=f"helper_{i}", name=f"helper_{i}")
        for i in range(8)
    ]
    rare = _chunk(
        "def compute_genuineness(job):\n    return 1",
        path="src/scoring.py",
        symbol="compute_genuineness",
        name="compute_genuineness",
    )
    index = Bm25Index(common + [rare])
    hits = index.search("compute_genuineness return", limit=3)
    assert hits[0].symbol == "compute_genuineness"


def test_from_store_indexes_qdrant_payloads() -> None:
    from app.indexing.pipeline import index_chunks
    from app.retrieval.vector_store import QdrantVectorStore
    from tests.fakes import FakeEmbedder

    chunk = _chunk(
        "def compute_genuineness(job):\n    return 1",
        path="src/scoring.py",
        symbol="compute_genuineness",
        name="compute_genuineness",
    )
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="bm25_from_store",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        index_chunks(store, embedder, [chunk])
        index = Bm25Index.from_store(store)

    assert len(index) == 1
    hits = index.search("compute_genuineness", limit=1)
    assert hits[0].symbol == "compute_genuineness"

"""Tests for QdrantVectorStore (in-memory; no Docker required)."""

from __future__ import annotations

import os

import pytest

from app.indexing.chunker import Chunk, chunk_file
from app.indexing.pipeline import index_chunks
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk
from tests.fakes import FakeEmbedder


@pytest.fixture
def store() -> QdrantVectorStore:
    with QdrantVectorStore(
        collection_name="test_code_chunks",
        vector_size=4,
        url=":memory:",
    ) as s:
        s.ensure_collection(recreate=True)
        yield s


def test_upsert_and_search_returns_scored_chunks(store: QdrantVectorStore) -> None:
    chunks = [
        Chunk(
            text="def compute_payment_state(invoice):\n    return 'paid'",
            path="account/models/account_move.py",
            start_line=1191,
            end_line=1195,
        ),
        Chunk(
            text="def rank_score(match, genuineness):\n    return match * 0.7",
            path="lib/scoring.py",
            start_line=218,
            end_line=220,
        ),
    ]
    embedder = FakeEmbedder()
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert_chunks(chunks, vectors)

    assert store.count() == 2

    query = embedder.embed_one("invoice payment state calculation")
    results = store.search(query, limit=2)

    assert len(results) == 2
    assert all(isinstance(r, ScoredChunk) for r in results)
    assert results[0].path in {c.path for c in chunks}
    assert results[0].score >= results[1].score


def test_index_chunks_pipeline(store: QdrantVectorStore) -> None:
    content = "\n".join(
        [
            "def compute_payment_state(invoice):",
            "    return invoice.payment_state",
            "",
            "def unrelated_sort(items):",
            "    return sorted(items)",
        ]
    )
    chunks = chunk_file("demo.py", content, chunk_size=3, overlap=0)
    embedder = FakeEmbedder()

    written = index_chunks(store, embedder, chunks)

    assert written == len(chunks)
    assert store.count() == written

    results = store.search(embedder.embed_one(chunks[0].text), limit=1)
    assert "payment_state" in results[0].text


def test_upsert_length_mismatch_raises(store: QdrantVectorStore) -> None:
    chunk = Chunk(text="x", path="a.py", start_line=1, end_line=1)
    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert_chunks([chunk], [])


def test_search_wrong_vector_size_raises(store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError, match="query_vector size"):
        store.search([0.1, 0.2], limit=1)


@pytest.mark.integration
def test_live_qdrant_docker() -> None:
    """Optional: requires Qdrant on localhost:6333.

    Start with::

        docker compose -f docker/docker-compose.yml up -d

    Then::

        LIVE_QDRANT=1 PYTHONPATH=. pytest tests/test_vector_store.py -m integration -v
    """
    if os.environ.get("LIVE_QDRANT") != "1":
        pytest.skip("Set LIVE_QDRANT=1 and start Qdrant via docker compose")

    with QdrantVectorStore(
        collection_name="integration_code_chunks",
        vector_size=4,
        url="http://127.0.0.1:6333",
    ) as store:
        store.ensure_collection(recreate=True)
        chunk = Chunk(text="hello qdrant", path="x.py", start_line=1, end_line=1)
        vector = [0.0, 0.1, 0.2, 0.3]
        store.upsert_chunks([chunk], [vector])
        hits = store.search(vector, limit=1)

    assert hits[0].text == "hello qdrant"
    assert hits[0].score > 0.99

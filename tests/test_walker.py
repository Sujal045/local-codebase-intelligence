"""Tests for repository walking (Slice 1e)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.indexing.pipeline import collect_chunks, index_repository
from app.indexing.walker import walk_repository
from app.retrieval.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


def test_walk_skips_node_modules_and_keeps_source() -> None:
    files = walk_repository(FIXTURE_REPO)
    paths = {item.path for item in files}

    assert "src/scoring.py" in paths
    assert "README.md" in paths
    assert not any(path.startswith("node_modules/") for path in paths)


def test_walk_missing_directory_raises() -> None:
    with pytest.raises(NotADirectoryError):
        walk_repository(FIXTURE_REPO / "does-not-exist")


def test_index_repository_writes_chunks() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="cli_mini_repo",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        result = index_repository(
            FIXTURE_REPO,
            store=store,
            embedder=embedder,
            chunk_size=20,
            overlap=0,
            recreate=True,
        )

        assert result.files >= 2
        assert result.chunks == result.upserted
        assert result.upserted == store.count()
        assert result.upserted > 0

        files = walk_repository(FIXTURE_REPO)
        chunks = collect_chunks(files, chunk_size=20, overlap=0)
        query = embedder.embed_one(chunks[0].text)
        hits = store.search(query, limit=1)
        assert hits[0].path in {chunk.path for chunk in chunks}

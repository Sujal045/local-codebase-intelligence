"""Tests for indexing pipeline routing (Slice 2C)."""

from __future__ import annotations

from pathlib import Path

from app.indexing.pipeline import collect_chunks, index_repository
from app.indexing.walker import SourceFile
from app.retrieval.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


def test_python_files_get_symbol_chunks() -> None:
    files = [
        SourceFile(
            path="lib/math_ops.py",
            content="def add(a, b):\n    return a + b\n",
        )
    ]
    chunks = collect_chunks(files, chunk_size=40, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].symbol == "add"
    assert chunks[0].kind == "function"
    assert chunks[0].language == "python"
    assert chunks[0].start_line == 1
    assert "def add" in chunks[0].text


def test_javascript_files_get_symbol_chunks() -> None:
    files = [
        SourceFile(
            path="web/add.js",
            content="export function add(a, b) {\n  return a + b;\n}\n",
        )
    ]
    chunks = collect_chunks(files, chunk_size=40, overlap=10)

    assert len(chunks) == 1
    assert chunks[0].symbol == "add"
    assert chunks[0].kind == "function"
    assert chunks[0].language == "javascript"


def test_go_files_get_symbol_chunks() -> None:
    files = [
        SourceFile(
            path="cmd/rank.go",
            content="package p\nfunc Rank(a, b int) int { return a + b }\n",
        )
    ]
    chunks = collect_chunks(files)
    found = {c.symbol: c for c in chunks}
    assert found["Rank"].language == "go"
    assert found["Rank"].kind == "function"


def test_unsupported_languages_keep_naive_windows() -> None:
    files = [
        SourceFile(
            path="parser.rs",
            content="\n".join(f"line {i}" for i in range(1, 21)),
        )
    ]
    chunks = collect_chunks(files, chunk_size=10, overlap=0)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (11, 20)]
    assert all(c.symbol is None for c in chunks)


def test_mixed_repo_uses_both_chunkers() -> None:
    files = [
        SourceFile(path="a.py", content="def foo():\n    return 1\n"),
        SourceFile(path="notes.md", content="hello docs\n"),
    ]
    chunks = collect_chunks(files, chunk_size=40, overlap=0)
    by_path = {c.path: c for c in chunks}

    assert by_path["a.py"].symbol == "foo"
    assert by_path["notes.md"].symbol is None
    assert by_path["notes.md"].text == "hello docs"


def test_index_repository_stores_python_symbol_payload() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="pipeline_symbols",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        result = index_repository(
            FIXTURE_REPO,
            store=store,
            embedder=embedder,
            recreate=True,
        )
        assert result.upserted > 0

        scoring = [
            c
            for c in collect_chunks(
                [
                    SourceFile(
                        path="src/scoring.py",
                        content=(FIXTURE_REPO / "src" / "scoring.py").read_text(),
                    )
                ]
            )
        ]
        query = embedder.embed_one(scoring[0].text)
        hits = store.search(query, limit=3)

    symbols = {hit.symbol for hit in hits}
    assert "compute_genuineness" in symbols or any(
        hit.path == "src/scoring.py" and hit.kind == "function" for hit in hits
    )
    scoring_hits = [hit for hit in hits if hit.path == "src/scoring.py"]
    assert scoring_hits
    assert scoring_hits[0].language == "python"
    assert scoring_hits[0].symbol is not None


def test_indexed_payload_is_bm25_corpus() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="pipeline_bm25_corpus",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        result = index_repository(
            FIXTURE_REPO,
            store=store,
            embedder=embedder,
            recreate=True,
        )
        listed = store.list_chunks()

    assert len(listed) == result.upserted
    assert any(chunk.symbol == "compute_genuineness" for chunk in listed)

"""Tests for the Version 1 CLI (Slice 1e)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from app.cli import build_parser, cmd_ask, cmd_index, main
from app.retrieval.vector_store import QdrantVectorStore
from tests.fakes import FakeEmbedder, RecordingLLM

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


def test_parser_index_and_ask_defaults() -> None:
    parser = build_parser()
    index_args = parser.parse_args(["index", "some/repo"])
    assert index_args.command == "index"
    assert index_args.repo == "some/repo"
    assert index_args.recreate is True

    ask_args = parser.parse_args(["ask", "Where is spam detected?"])
    assert ask_args.command == "ask"
    assert ask_args.question == "Where is spam detected?"
    assert ask_args.limit == 5


def test_cmd_index_writes_to_store(capsys) -> None:
    embedder = FakeEmbedder()
    args = Namespace(
        repo=str(FIXTURE_REPO),
        chunk_size=20,
        overlap=0,
        recreate=True,
        qdrant_url=":memory:",
    )
    with QdrantVectorStore(
        collection_name="cli_index_test",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        code = cmd_index(args, embedder=embedder, store=store)
        assert code == 0
        assert store.count() > 0

    out = capsys.readouterr().out
    assert "Indexed" in out
    assert "files:" in out


def test_cmd_index_missing_repo_returns_error(capsys) -> None:
    embedder = FakeEmbedder()
    args = Namespace(
        repo=str(FIXTURE_REPO / "missing"),
        chunk_size=20,
        overlap=0,
        recreate=True,
        qdrant_url=":memory:",
    )
    with QdrantVectorStore(
        collection_name="cli_missing",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        code = cmd_index(args, embedder=embedder, store=store)

    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cmd_ask_prints_answer_and_sources(capsys) -> None:
    embedder = FakeEmbedder()
    llm = RecordingLLM()
    args_index = Namespace(
        repo=str(FIXTURE_REPO),
        chunk_size=20,
        overlap=0,
        recreate=True,
        qdrant_url=":memory:",
    )
    args_ask = Namespace(
        question="How do we detect spam jobs?",
        limit=3,
        qdrant_url=":memory:",
        ollama_url="http://127.0.0.1:11434",
    )
    with QdrantVectorStore(
        collection_name="cli_ask_test",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        assert cmd_index(args_index, embedder=embedder, store=store) == 0
        code = cmd_ask(args_ask, embedder=embedder, store=store, llm=llm)

    assert code == 0
    out = capsys.readouterr().out
    assert "Answer:" in out
    assert "compute_genuineness" in out
    assert "Sources:" in out
    assert "scoring.py" in out
    assert "(function)" in out


def test_cmd_ask_without_collection_fails(capsys) -> None:
    embedder = FakeEmbedder()
    args = Namespace(
        question="anything",
        limit=3,
        qdrant_url=":memory:",
        ollama_url="http://127.0.0.1:11434",
    )
    with QdrantVectorStore(
        collection_name="cli_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        code = cmd_ask(args, embedder=embedder, store=store, llm=RecordingLLM())

    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_main_help_exits_zero() -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
        return
    raise AssertionError("expected SystemExit from --help")

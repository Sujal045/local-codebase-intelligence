"""Tests for agent CLI wiring and tool bundle (Slice 6C)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from app.agent import AgentTurn, ToolCall, build_agent_tools
from app.cli import build_parser, cmd_agent, cmd_index
from app.retrieval.vector_store import QdrantVectorStore
from app.tools import (
    FindReferencesTool,
    GetSymbolTool,
    ReadFileTool,
    SearchCodeTool,
    SearchDocumentationTool,
)
from tests.fakes import FakeEmbedder, FakeReranker, ScriptedToolLLM

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "mini_repo"


def test_parser_agent_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["agent", "Where is spam detected?", "--repo", "tests/fixtures/mini_repo"]
    )
    assert args.command == "agent"
    assert args.question == "Where is spam detected?"
    assert args.repo == "tests/fixtures/mini_repo"
    assert args.max_steps == 8
    assert args.trace is True
    assert args.rerank is True
    assert args.limit == 5

    quiet = parser.parse_args(
        ["agent", "q", "--repo", "r", "--no-trace", "--max-steps", "3"]
    )
    assert quiet.trace is False
    assert quiet.max_steps == 3


def test_build_agent_tools_returns_five_named_tools() -> None:
    embedder = FakeEmbedder()
    with QdrantVectorStore(
        collection_name="bundle_tools",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        store.ensure_collection(recreate=True)
        tools = build_agent_tools(
            root=FIXTURE_REPO,
            embedder=embedder,
            store=store,
            reranker=FakeReranker(),
        )

    names = [tool.name for tool in tools]
    assert names == [
        "search_code",
        "read_file",
        "get_symbol",
        "find_references",
        "search_documentation",
    ]
    assert isinstance(tools[0], SearchCodeTool)
    assert isinstance(tools[1], ReadFileTool)
    assert isinstance(tools[2], GetSymbolTool)
    assert isinstance(tools[3], FindReferencesTool)
    assert isinstance(tools[4], SearchDocumentationTool)


def test_cmd_agent_prints_answer_and_trace(capsys) -> None:
    embedder = FakeEmbedder()
    llm = ScriptedToolLLM(
        [
            AgentTurn(
                tool_calls=(
                    ToolCall(
                        name="search_code",
                        arguments={"query": "spam detection", "limit": 2},
                    ),
                )
            ),
            AgentTurn(
                content="Spam scoring is in compute_genuineness (src/scoring.py)."
            ),
        ]
    )
    args_index = Namespace(
        repo=str(FIXTURE_REPO),
        chunk_size=20,
        overlap=0,
        recreate=True,
        qdrant_url=":memory:",
    )
    args_agent = Namespace(
        question="How do we detect spam jobs?",
        repo=str(FIXTURE_REPO),
        limit=3,
        candidate_limit=20,
        rerank=True,
        max_steps=8,
        trace=True,
        qdrant_url=":memory:",
        ollama_url="http://127.0.0.1:11434",
    )
    with QdrantVectorStore(
        collection_name="cli_agent_test",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        assert cmd_index(args_index, embedder=embedder, store=store) == 0
        tools = build_agent_tools(
            root=FIXTURE_REPO,
            embedder=embedder,
            store=store,
            reranker=FakeReranker(),
            candidate_limit=20,
            search_limit=3,
        )
        code = cmd_agent(
            args_agent,
            embedder=embedder,
            store=store,
            llm=llm,
            reranker=FakeReranker(),
            tools=tools,
        )

    assert code == 0
    out = capsys.readouterr().out
    assert "Answer:" in out
    assert "compute_genuineness" in out
    assert "Agent: llm_calls=2" in out
    assert "stopped=final_answer" in out
    assert "Trace:" in out
    assert "→ search_code(" in out
    assert "← search_code:" in out
    assert "✓ final" in out


def test_cmd_agent_without_collection_fails(capsys) -> None:
    embedder = FakeEmbedder()
    args = Namespace(
        question="anything",
        repo=str(FIXTURE_REPO),
        limit=3,
        candidate_limit=20,
        rerank=False,
        max_steps=4,
        trace=False,
        qdrant_url=":memory:",
        ollama_url="http://127.0.0.1:11434",
    )
    with QdrantVectorStore(
        collection_name="cli_agent_empty",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        code = cmd_agent(
            args,
            embedder=embedder,
            store=store,
            llm=ScriptedToolLLM([AgentTurn(content="nope")]),
        )

    assert code == 1
    assert "does not exist" in capsys.readouterr().err


def test_cmd_agent_bad_repo_fails(capsys) -> None:
    embedder = FakeEmbedder()
    args_index = Namespace(
        repo=str(FIXTURE_REPO),
        chunk_size=20,
        overlap=0,
        recreate=True,
        qdrant_url=":memory:",
    )
    args_agent = Namespace(
        question="q",
        repo=str(FIXTURE_REPO / "missing"),
        limit=3,
        candidate_limit=20,
        rerank=False,
        max_steps=4,
        trace=False,
        qdrant_url=":memory:",
        ollama_url="http://127.0.0.1:11434",
    )
    with QdrantVectorStore(
        collection_name="cli_agent_bad_repo",
        vector_size=embedder.dimensions,
        url=":memory:",
    ) as store:
        assert cmd_index(args_index, embedder=embedder, store=store) == 0
        code = cmd_agent(
            args_agent,
            embedder=embedder,
            store=store,
            llm=ScriptedToolLLM([AgentTurn(content="x")]),
        )

    assert code == 1
    assert "error:" in capsys.readouterr().err

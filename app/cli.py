"""Command-line entry point for Code RAG and the tool agent (Slice 6C).

Commands:

    python -m app.cli index <repo>
    python -m app.cli ask "How is spam detected?"
    python -m app.cli agent "How is spam detected?" --repo <repo>

``index`` walks a repository, chunks files, embeds them, and stores points
in Qdrant (also the BM25 corpus).

``ask`` is one-shot hybrid RAG (optional rerank) — no tool loop.

``agent`` runs ``run_agent`` with the Version 5 tools. The model may call
``search_code``, ``read_file``, ``get_symbol``, ``find_references``, and
``search_documentation``. ``--repo`` is the sandbox root for ``read_file``
and should be the same tree you indexed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.agent import DEFAULT_MAX_STEPS, AgentAnswer, ToolCallingLLM, run_agent
from app.agent.tools_bundle import build_agent_tools
from app.config import (
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OVERLAP,
    DEFAULT_QDRANT_URL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SIZE,
)
from app.embeddings import Embedder, OllamaEmbedder
from app.indexing.pipeline import IndexResult, index_repository
from app.llm import ChatLLM, OllamaChatLLM
from app.reranking.reranker import CrossEncoderReranker, Reranker
from app.retrieval.rag import RagAnswer, ask as rag_ask
from app.retrieval.vector_store import QdrantVectorStore
from app.tools.base import Tool

DEFAULT_TRACE_OBSERVATION_CHARS = 400


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description=(
            "Local Codebase Intelligence Agent — index, one-shot ask, or tool agent"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="Walk, chunk, embed, and store a repository")
    index_p.add_argument("repo", help="Path to the repository root")
    _add_shared_args(index_p)
    index_p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    index_p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    index_p.add_argument(
        "--recreate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop and recreate the Qdrant collection before indexing (default: true)",
    )

    ask_p = sub.add_parser("ask", help="One-shot retrieve, rerank, and generate an answer")
    ask_p.add_argument("question", help="Natural-language question about the indexed repo")
    _add_shared_args(ask_p)
    _add_chat_args(ask_p)
    _add_retrieval_args(ask_p)

    agent_p = sub.add_parser(
        "agent",
        help="Tool-using agent loop over the indexed repository",
    )
    agent_p.add_argument("question", help="Natural-language question about the indexed repo")
    agent_p.add_argument(
        "--repo",
        required=True,
        help="Repository root for read_file sandbox (same tree you indexed)",
    )
    _add_shared_args(agent_p)
    _add_chat_args(agent_p)
    _add_retrieval_args(agent_p)
    agent_p.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum LLM rounds before stopping (default: {DEFAULT_MAX_STEPS})",
    )
    agent_p.add_argument(
        "--trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print tool calls and truncated observations (default: true)",
    )

    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument(
        "--vector-size",
        type=int,
        default=DEFAULT_VECTOR_SIZE,
        help="Embedding dimensions (768 for nomic-embed-text)",
    )


def _add_chat_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)


def _add_retrieval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TOP_K,
        help="Chunks returned by search_code / sent to the LLM in ask (default: 5)",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATE_LIMIT,
        dest="candidate_limit",
        help="Hybrid / RRF pool size before reranking (default: 20)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rerank hybrid search_code / ask pools (default: true)",
    )
    parser.add_argument(
        "--rerank-model",
        default=DEFAULT_RERANK_MODEL,
        help=f"Hugging Face cross-encoder id (default: {DEFAULT_RERANK_MODEL})",
    )


def cmd_index(
    args: argparse.Namespace,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
) -> int:
    try:
        result = index_repository(
            args.repo,
            store=store,
            embedder=embedder,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            recreate=args.recreate,
        )
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(_friendly_service_error(exc, qdrant_url=args.qdrant_url), file=sys.stderr)
        return 1

    _print_index_result(result, collection=store.collection_name)
    if result.upserted == 0:
        print("warning: no chunks were indexed", file=sys.stderr)
        return 1
    return 0


def cmd_ask(
    args: argparse.Namespace,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    llm: ChatLLM,
    reranker: Reranker | None = None,
) -> int:
    try:
        if not store.collection_exists():
            print(
                f"error: Qdrant collection {store.collection_name!r} does not exist. "
                "Index a repo first: python -m app.cli index <repo>",
                file=sys.stderr,
            )
            return 1
        result = rag_ask(
            args.question,
            embedder=embedder,
            store=store,
            llm=llm,
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            reranker=reranker if args.rerank else None,
        )
    except Exception as exc:
        print(
            _friendly_service_error(
                exc,
                qdrant_url=args.qdrant_url,
                ollama_url=args.ollama_url,
            ),
            file=sys.stderr,
        )
        return 1

    _print_ask_result(result)
    return 0


def cmd_agent(
    args: argparse.Namespace,
    *,
    embedder: Embedder,
    store: QdrantVectorStore,
    llm: ToolCallingLLM,
    reranker: Reranker | None = None,
    tools: Sequence[Tool] | None = None,
) -> int:
    try:
        if not store.collection_exists():
            print(
                f"error: Qdrant collection {store.collection_name!r} does not exist. "
                "Index a repo first: python -m app.cli index <repo>",
                file=sys.stderr,
            )
            return 1
        if args.max_steps < 1:
            print("error: --max-steps must be >= 1", file=sys.stderr)
            return 1

        tool_list: Sequence[Tool]
        if tools is not None:
            tool_list = tools
        else:
            tool_list = build_agent_tools(
                root=args.repo,
                embedder=embedder,
                store=store,
                reranker=reranker if args.rerank else None,
                candidate_limit=args.candidate_limit,
                search_limit=args.limit,
            )

        result = run_agent(
            args.question,
            llm=llm,
            tools=tool_list,
            max_steps=args.max_steps,
        )
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            _friendly_service_error(
                exc,
                qdrant_url=args.qdrant_url,
                ollama_url=args.ollama_url,
            ),
            file=sys.stderr,
        )
        return 1

    _print_agent_result(result, trace=args.trace)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        with (
            OllamaEmbedder(args.embed_model, base_url=args.ollama_url) as embedder,
            QdrantVectorStore(
                args.collection,
                vector_size=args.vector_size,
                url=args.qdrant_url,
            ) as store,
        ):
            return cmd_index(args, embedder=embedder, store=store)

    if args.command == "ask":
        reranker: Reranker | None = None
        if args.rerank:
            reranker = CrossEncoderReranker(args.rerank_model)
        with (
            OllamaEmbedder(args.embed_model, base_url=args.ollama_url) as embedder,
            OllamaChatLLM(args.chat_model, base_url=args.ollama_url) as llm,
            QdrantVectorStore(
                args.collection,
                vector_size=args.vector_size,
                url=args.qdrant_url,
            ) as store,
        ):
            return cmd_ask(
                args,
                embedder=embedder,
                store=store,
                llm=llm,
                reranker=reranker,
            )

    if args.command == "agent":
        agent_reranker: Reranker | None = None
        if args.rerank:
            agent_reranker = CrossEncoderReranker(args.rerank_model)
        with (
            OllamaEmbedder(args.embed_model, base_url=args.ollama_url) as embedder,
            OllamaChatLLM(args.chat_model, base_url=args.ollama_url) as llm,
            QdrantVectorStore(
                args.collection,
                vector_size=args.vector_size,
                url=args.qdrant_url,
            ) as store,
        ):
            return cmd_agent(
                args,
                embedder=embedder,
                store=store,
                llm=llm,
                reranker=agent_reranker,
            )

    parser.error(f"unknown command: {args.command}")
    return 2


def _print_index_result(result: IndexResult, *, collection: str) -> None:
    print(f"Indexed {result.root}")
    print(f"  files:      {result.files}")
    print(f"  chunks:     {result.chunks}")
    print(f"  upserted:   {result.upserted}")
    print(f"  collection: {collection}")


def _print_ask_result(result: RagAnswer) -> None:
    print("Answer:")
    print(result.answer.strip())
    print()
    kind = "rerank" if result.reranked else "hybrid RRF"
    score_name = "rerank" if result.reranked else "rrf"
    print(f"Sources ({kind}):")
    if not result.sources:
        print("  (none retrieved)")
        return
    for i, chunk in enumerate(result.sources, start=1):
        print(f"  [{i}] {chunk.label()} ({score_name}={chunk.score:.4f})")


def _print_agent_result(
    result: AgentAnswer,
    *,
    trace: bool,
    observation_chars: int = DEFAULT_TRACE_OBSERVATION_CHARS,
) -> None:
    print("Answer:")
    print(result.answer.strip())
    print()
    print(
        f"Agent: llm_calls={result.llm_calls} "
        f"stopped={result.stopped_reason}"
    )
    if not trace:
        return

    print()
    print("Trace:")
    if not result.events:
        print("  (no events)")
        return
    for event in result.events:
        if event.kind == "tool_call":
            print(f"  → {event.name}({event.content})")
        elif event.kind == "observation":
            body = _truncate(event.content, observation_chars)
            print(f"  ← {event.name}: {body}")
        elif event.kind == "final":
            print(f"  ✓ final ({len(event.content)} chars)")
        else:
            print(f"  · {event.kind}: {event.content}")


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)] + "..."


def _friendly_service_error(
    exc: BaseException,
    *,
    qdrant_url: str | None = None,
    ollama_url: str | None = None,
) -> str:
    message = str(exc)
    lowered = message.lower()
    if "qdrant" in lowered or "6333" in lowered or "connection refused" in lowered:
        hint = ""
        if qdrant_url:
            hint = (
                f" Cannot reach Qdrant at {qdrant_url}. "
                "Start it with: docker compose -f docker/docker-compose.yml up -d"
            )
        return f"error: {exc}.{hint}"
    if "ollama" in lowered or "11434" in lowered:
        hint = ""
        if ollama_url:
            hint = f" Cannot reach Ollama at {ollama_url}."
        return f"error: {exc}.{hint}"
    return f"error: {exc}"


if __name__ == "__main__":
    raise SystemExit(main())

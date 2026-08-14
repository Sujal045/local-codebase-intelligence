"""Command-line entry point for Version 1 Code RAG (Slice 1e).

Two commands:

    python -m app.cli index <repo>
    python -m app.cli ask "How is spam detected?"

``index`` walks a repository, chunks files, embeds them, and stores points
in Qdrant. ``ask`` embeds the question, retrieves top-k chunks, and calls
the local chat model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from app.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OVERLAP,
    DEFAULT_QDRANT_URL,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SIZE,
)
from app.embeddings import Embedder, OllamaEmbedder
from app.indexing.pipeline import IndexResult, index_repository
from app.llm import ChatLLM, OllamaChatLLM
from app.retrieval.rag import RagAnswer, ask as rag_ask
from app.retrieval.vector_store import QdrantVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Local Codebase Intelligence Agent — Version 1 CLI",
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

    ask_p = sub.add_parser("ask", help="Retrieve relevant chunks and generate an answer")
    ask_p.add_argument("question", help="Natural-language question about the indexed repo")
    _add_shared_args(ask_p)
    ask_p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ask_p.add_argument("--limit", type=int, default=DEFAULT_TOP_K, help="Top-k chunks")

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
        with (
            OllamaEmbedder(args.embed_model, base_url=args.ollama_url) as embedder,
            OllamaChatLLM(args.chat_model, base_url=args.ollama_url) as llm,
            QdrantVectorStore(
                args.collection,
                vector_size=args.vector_size,
                url=args.qdrant_url,
            ) as store,
        ):
            return cmd_ask(args, embedder=embedder, store=store, llm=llm)

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
    print("Sources:")
    if not result.sources:
        print("  (none retrieved)")
        return
    for i, chunk in enumerate(result.sources, start=1):
        print(
            f"  [{i}] {chunk.path}:{chunk.start_line}-{chunk.end_line} "
            f"(score={chunk.score:.4f})"
        )


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

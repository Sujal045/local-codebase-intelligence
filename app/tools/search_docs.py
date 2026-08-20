"""``search_documentation`` tool (Slice 5D).

``search_code`` searches the whole index (source + docs) with hybrid
retrieval. Agents often want *prose* answers: README, ``docs/``, RST.

This tool keeps only documentation chunks, then runs BM25 on that subset:

    search_documentation("Where is spam detection documented?")
            │
            ▼
    filter corpus → .md / .rst / docs/**
            │
            ▼
    BM25 over doc chunks only
            │
            ▼
    observation (paths, lines, text)

We deliberately skip vector search here: documentation questions are
usually lexical ("README", "install", "spam detection"), and a separate
doc-only BM25 index is easier to reason about than hybrid-then-filter.

Markdown is already indexed as naive line windows (no Tree-sitter). We
detect docs by path suffix / directory, not ``chunk.language`` (which is
often unset for those files).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import DEFAULT_TOP_K
from app.indexing.chunker import Chunk
from app.llm.prompt import format_context
from app.retrieval.bm25 import Bm25Index
from app.retrieval.vector_store import QdrantVectorStore, ScoredChunk
from app.tools.base import ToolResult

SEARCH_DOCUMENTATION_NAME = "search_documentation"

DOC_SUFFIXES = frozenset({".md", ".rst"})
DOC_DIR_NAMES = frozenset({"docs", "doc", "documentation"})


def is_documentation_chunk(chunk: Chunk) -> bool:
    """True if this chunk looks like project documentation, not source code."""
    path = chunk.path.replace("\\", "/").lower()
    suffix = Path(path).suffix.lower()
    if suffix in DOC_SUFFIXES:
        return True
    parts = path.split("/")
    return any(part in DOC_DIR_NAMES for part in parts[:-1])


def documentation_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Filter an indexed corpus down to documentation chunks."""
    return [chunk for chunk in chunks if is_documentation_chunk(chunk)]


def search_documentation_chunks(
    query: str,
    chunks: list[Chunk],
    *,
    limit: int = DEFAULT_TOP_K,
) -> list[ScoredChunk]:
    """BM25 over documentation chunks only."""
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    docs = documentation_chunks(chunks)
    if not docs or not query.strip():
        return []
    return Bm25Index(docs).search(query, limit=limit)


class SearchDocumentationTool:
    """Search README / docs paths in the indexed repository."""

    name = SEARCH_DOCUMENTATION_NAME

    def __init__(self, *, store: QdrantVectorStore) -> None:
        self._store = store

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search project documentation (Markdown, RST, and files "
                    "under docs/). Use for 'how do I…', overview, and "
                    "README questions. Prefer search_code for implementation "
                    "details inside source files."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language documentation question "
                                "(e.g. 'Where is spam detection documented?')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum doc chunks to return "
                                f"(default {DEFAULT_TOP_K})."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        """Search docs. Bad arguments / empty corpus → observation text."""
        extra = sorted(set(arguments) - {"query", "limit"})
        if extra:
            return ToolResult(
                name=self.name,
                content=f"error: unexpected arguments {extra}",
            )

        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(
                name=self.name,
                content="error: query must be a non-empty string",
            )

        limit = arguments.get("limit", DEFAULT_TOP_K)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return ToolResult(
                name=self.name,
                content="error: limit must be an integer >= 1",
            )

        try:
            corpus = self._store.list_chunks()
            docs = documentation_chunks(corpus)
            if not docs:
                return ToolResult(
                    name=self.name,
                    content=(
                        "No documentation chunks are indexed "
                        "(.md, .rst, or docs/ paths)."
                    ),
                )
            hits = search_documentation_chunks(
                query.strip(),
                corpus,
                limit=limit,
            )
        except Exception as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        if not hits:
            return ToolResult(
                name=self.name,
                content=f"No documentation hits for {query.strip()!r}.",
            )

        header = (
            f"Found {len(hits)} documentation hit(s) "
            f"for {query.strip()!r}:\n\n"
        )
        return ToolResult(
            name=self.name,
            content=header + format_context(hits),
            hits=tuple(hits),
        )

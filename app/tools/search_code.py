"""``search_code`` tool (Slice 5A).

One-shot RAG always retrieves, then generates. An agent instead *chooses*
to search when the question needs evidence from the index:

    User: "Where is invoice status calculated?"
            │
            ▼
          LLM decides: search_code("invoice status calculation")
            │
            ▼
          this tool runs retrieve_chunks
            │
            ▼
          observation (paths, lines, symbols, text) goes back to the LLM

The ranking pipeline is unchanged (hybrid + optional rerank). This class
only wraps it as a named function with a JSON schema.
"""

from __future__ import annotations

from typing import Any

from app.config import DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K
from app.embeddings import Embedder
from app.llm.prompt import format_context
from app.reranking.reranker import Reranker
from app.retrieval.bm25 import Bm25Index
from app.retrieval.query import retrieve_chunks
from app.retrieval.vector_store import QdrantVectorStore
from app.tools.base import ToolResult

SEARCH_CODE_NAME = "search_code"


class SearchCodeTool:
    """Search the indexed repository; return ranked code chunks as text."""

    name = SEARCH_CODE_NAME

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: QdrantVectorStore,
        bm25: Bm25Index | None = None,
        reranker: Reranker | None = None,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
        default_limit: int = DEFAULT_TOP_K,
    ) -> None:
        if default_limit < 1:
            raise ValueError(f"default_limit must be >= 1, got {default_limit}")
        self._embedder = embedder
        self._store = store
        self._bm25 = bm25
        self._reranker = reranker
        self._candidate_limit = candidate_limit
        self._default_limit = default_limit

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the indexed codebase for passages relevant to a "
                    "question or identifier. Use this to find where behavior "
                    "is implemented. Returns ranked chunks with file path, "
                    "line range, symbol name, and source text."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural-language question or exact symbol "
                                "(e.g. 'spam detection' or 'compute_genuineness')."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum chunks to return "
                                f"(default {self._default_limit})."
                            ),
                            "minimum": 1,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def run(self, **arguments: Any) -> ToolResult:
        """Run retrieval. Bad arguments become an observation, not a crash."""
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

        limit = arguments.get("limit", self._default_limit)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return ToolResult(
                name=self.name,
                content="error: limit must be an integer >= 1",
            )

        try:
            result = retrieve_chunks(
                query,
                embedder=self._embedder,
                store=self._store,
                limit=limit,
                candidate_limit=self._candidate_limit,
                bm25=self._bm25,
                reranker=self._reranker,
            )
        except Exception as exc:
            return ToolResult(name=self.name, content=f"error: {exc}")

        content = format_context(result.chunks)
        return ToolResult(
            name=self.name,
            content=content,
            hits=tuple(result.chunks),
        )

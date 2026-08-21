"""Assemble the default Version 5 tool set for the agent CLI (Slice 6C).

``run_agent`` only needs a list of ``Tool`` objects. This module wires
dependencies (embedder, store, repo root, optional reranker) into the five
tools we already built:

    search_code, read_file, get_symbol, find_references, search_documentation
"""

from __future__ import annotations

from pathlib import Path

from app.config import DEFAULT_CANDIDATE_LIMIT, DEFAULT_TOP_K
from app.embeddings import Embedder
from app.reranking.reranker import Reranker
from app.retrieval.vector_store import QdrantVectorStore
from app.tools import (
    FindReferencesTool,
    GetSymbolTool,
    ReadFileTool,
    SearchCodeTool,
    SearchDocumentationTool,
)
from app.tools.base import Tool


def build_agent_tools(
    *,
    root: str | Path,
    embedder: Embedder,
    store: QdrantVectorStore,
    reranker: Reranker | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    search_limit: int = DEFAULT_TOP_K,
) -> list[Tool]:
    """Return the standard code-intelligence tool list for ``run_agent``."""
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path.resolve()}")

    return [
        SearchCodeTool(
            embedder=embedder,
            store=store,
            reranker=reranker,
            candidate_limit=candidate_limit,
            default_limit=search_limit,
        ),
        ReadFileTool(root=root_path),
        GetSymbolTool(store=store),
        FindReferencesTool(store=store),
        SearchDocumentationTool(store=store),
    ]

"""Local reranking for Code RAG (Slices 4A–4C).

Hybrid retrieval (Version 3) is a *bi-encoder* shortlist: query and chunks
are encoded separately, then fused. A cross-encoder reranker reads the
question and a candidate together and produces a more precise score.

Slice 4A scores a given candidate list. Slice 4B
(``search_and_rerank``) fetches a broad hybrid pool first, then reranks.

``search_and_rerank`` is imported lazily so ``app.config`` can read
``DEFAULT_RERANK_MODEL`` without importing the retrieval pipeline.

``ask()`` (Slice 4C) calls ``search_and_rerank`` when a reranker is injected.
"""

from __future__ import annotations

from typing import Any

from app.reranking.reranker import (
    DEFAULT_RERANK_DEVICE,
    DEFAULT_RERANK_MODEL,
    CrossEncoderReranker,
    Reranker,
    apply_scores,
)

__all__ = [
    "DEFAULT_RERANK_DEVICE",
    "DEFAULT_RERANK_MODEL",
    "CrossEncoderReranker",
    "Reranker",
    "apply_scores",
    "search_and_rerank",
]


def __getattr__(name: str) -> Any:
    if name == "search_and_rerank":
        from app.reranking.pipeline import search_and_rerank

        return search_and_rerank
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

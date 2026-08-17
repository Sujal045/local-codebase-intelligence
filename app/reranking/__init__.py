"""Local reranking for Code RAG (Slice 4A).

Hybrid retrieval (Version 3) is a *bi-encoder* shortlist: query and chunks
are encoded separately, then fused. A cross-encoder reranker reads the
question and a candidate together and produces a more precise score.

This package does not talk to Qdrant or the LLM yet. ``ask()`` still
returns Reciprocal Rank Fusion order until Slice 4C.
"""

from __future__ import annotations

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
]

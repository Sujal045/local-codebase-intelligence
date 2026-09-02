"""Context engineering for Code RAG and the agent (Version 7).

Slice 7A: token estimation, budgets, truncation, deduplication.
Slice 7B: ``ContextBuilder`` for budget-aware one-shot RAG.
Slice 7C will compress agent tool observations. Slice 7D adds CLI flags.
"""

from __future__ import annotations

from app.context.budget import ContextBudget
from app.context.builder import (
    BuiltContext,
    ContextBuilder,
    format_chunk_block,
    order_lost_in_middle,
)
from app.context.dedupe import dedupe_chunks, location_key
from app.context.tokens import CHARS_PER_TOKEN, estimate_tokens
from app.context.truncate import TRUNCATION_MARKER, truncate_text

__all__ = [
    "CHARS_PER_TOKEN",
    "BuiltContext",
    "ContextBudget",
    "ContextBuilder",
    "TRUNCATION_MARKER",
    "dedupe_chunks",
    "estimate_tokens",
    "format_chunk_block",
    "location_key",
    "order_lost_in_middle",
    "truncate_text",
]

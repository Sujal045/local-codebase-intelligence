"""Context engineering primitives (Version 7, Slice 7A).

Before we wire budgets into ``ask()`` or the agent loop, this package
provides small, testable building blocks:

* **Token estimation** — cheap character heuristic (no tiktoken dependency).
* **Context budgets** — cap how many tokens retrieval may consume.
* **Truncation** — shorten text while keeping readable boundaries.
* **Deduplication** — collapse repeated chunks at the same file location.

Slice 7B will compose these into a ``ContextBuilder`` for one-shot RAG.
Slice 7C will compress agent tool observations. Slice 7D adds CLI flags.
"""

from __future__ import annotations

from app.context.budget import ContextBudget
from app.context.dedupe import dedupe_chunks, location_key
from app.context.tokens import CHARS_PER_TOKEN, estimate_tokens
from app.context.truncate import TRUNCATION_MARKER, truncate_text

__all__ = [
    "CHARS_PER_TOKEN",
    "ContextBudget",
    "TRUNCATION_MARKER",
    "dedupe_chunks",
    "estimate_tokens",
    "location_key",
    "truncate_text",
]

"""Budget-aware context assembly for RAG (Slice 7B).

``ContextBuilder`` turns a ranked chunk list into prompt-ready text:

1. **Dedupe** — drop duplicate ``path:start:end`` locations.
2. **Reorder** — lost-in-the-middle: best chunks at start *and* end.
3. **Fit budget** — include chunks until tokens run out; truncate bodies
   when a chunk is too large for the remaining space.

Slice 7B wires this into ``build_rag_messages`` and ``ask()``. The agent
path still uses plain ``format_context`` until Slice 7C.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.context.budget import ContextBudget
from app.context.dedupe import dedupe_chunks
from app.context.tokens import estimate_tokens
from app.context.truncate import TRUNCATION_MARKER, truncate_text
from app.retrieval.vector_store import ScoredChunk

BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class BuiltContext:
    """Result of fitting ranked chunks into a token budget."""

    chunks: tuple[ScoredChunk, ...]
    text: str
    tokens_used: int
    deduped: int
    truncated: int
    dropped: int


def order_lost_in_middle(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Reorder ranked chunks so strong hits sit at the start and end.

    LLMs often attend best to the beginning and end of long prompts
    ("lost in the middle"). Input chunks should already be ranked best-first.
    """
    if len(chunks) <= 2:
        return list(chunks)

    left: list[ScoredChunk] = []
    right: list[ScoredChunk] = []
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            left.append(chunk)
        else:
            right.insert(0, chunk)
    return left + right


class ContextBuilder:
    """Assemble deduplicated, ordered, budget-fitting context from chunks."""

    def build(
        self,
        chunks: Sequence[ScoredChunk],
        budget: ContextBudget,
    ) -> BuiltContext:
        """Return formatted context and the chunks actually included."""
        ranked = list(chunks)
        unique = dedupe_chunks(ranked)
        deduped = len(ranked) - len(unique)
        ordered = order_lost_in_middle(unique)

        included: list[ScoredChunk] = []
        blocks: list[str] = []
        tokens_used = 0
        truncated = 0
        dropped = 0
        remaining = budget.available

        for index, chunk in enumerate(ordered, start=1):
            header = f"[{index}] {chunk.label()} (score={chunk.score:.4f})"
            separator_tokens = (
                estimate_tokens(BLOCK_SEPARATOR) if blocks else 0
            )
            header_tokens = estimate_tokens(f"{header}\n")

            if separator_tokens + header_tokens >= remaining:
                dropped += 1
                continue

            body_budget = remaining - separator_tokens - header_tokens
            body = chunk.text
            if estimate_tokens(body) > body_budget:
                if body_budget < 1:
                    dropped += 1
                    continue
                body = truncate_text(body, body_budget, strategy="start")
                truncated += 1

            block = f"{header}\n{body}"
            block_tokens = separator_tokens + estimate_tokens(block)

            if block_tokens > remaining:
                dropped += 1
                continue

            included.append(_chunk_with_text(chunk, body))
            blocks.append(block)
            tokens_used += block_tokens
            remaining -= block_tokens

        text = (
            "(no code context retrieved)"
            if not blocks
            else BLOCK_SEPARATOR.join(blocks)
        )
        return BuiltContext(
            chunks=tuple(included),
            text=text,
            tokens_used=tokens_used,
            deduped=deduped,
            truncated=truncated,
            dropped=dropped,
        )


def _chunk_with_text(chunk: ScoredChunk, text: str) -> ScoredChunk:
    return ScoredChunk(
        path=chunk.path,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        text=text,
        score=chunk.score,
        language=chunk.language,
        symbol=chunk.symbol,
        kind=chunk.kind,
        name=chunk.name,
        parent=chunk.parent,
    )


def format_chunk_block(index: int, chunk: ScoredChunk) -> str:
    """Render one numbered context block (header + body)."""
    header = f"[{index}] {chunk.label()} (score={chunk.score:.4f})"
    return f"{header}\n{chunk.text}"


__all__ = [
    "BLOCK_SEPARATOR",
    "BuiltContext",
    "ContextBuilder",
    "format_chunk_block",
    "order_lost_in_middle",
]

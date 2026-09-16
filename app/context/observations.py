"""Prepare agent tool observations with dedupe and budgeting (Slice 7C).

Each tool returns a ``ToolResult``. Before the loop appends it to the
conversation we may:

1. Drop chunk bodies the model already saw in an earlier observation.
2. Compress the text to fit the remaining observation token budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.context.budget import ContextBudget
from app.context.builder import BLOCK_SEPARATOR, format_chunk_block
from app.context.compress import OBSERVATION_OMITTED, compress_observation
from app.context.dedupe import chunk_location_key
from app.context.tokens import estimate_tokens
from app.retrieval.vector_store import ScoredChunk


class _ToolResultLike(Protocol):
    """Minimal tool outcome shape (avoids importing ``app.tools`` here)."""

    name: str
    content: str
    hits: tuple[ScoredChunk, ...]


@dataclass
class AgentContextState:
    """Session memory for observation budgeting across agent tool rounds."""

    seen_locations: set[tuple[str, int, int]] = field(default_factory=set)
    observation_tokens: int = 0
    compressed: int = 0
    chunks_deduped: int = 0


@dataclass(frozen=True)
class PreparedObservation:
    """Observation text ready for the LLM plus accounting stats."""

    content: str
    tokens: int
    compressed: bool
    chunks_deduped: int


def prepare_tool_observation(
    result: _ToolResultLike,
    *,
    state: AgentContextState,
    budget: ContextBudget | None = None,
    dedupe: bool = True,
) -> PreparedObservation:
    """Optionally dedupe chunk hits, apply session budget, return text.

    Args:
        result: Tool outcome (content + optional scored hits).
        state: Mutable session counters and seen locations.
        budget: When set, compress to remaining observation tokens.
        dedupe: When True (default), drop chunk bodies already shown.
    """
    deduped_before = state.chunks_deduped
    if dedupe:
        content = _content_with_session_dedupe(result, state)
    else:
        content = result.content
        if result.hits:
            for hit in result.hits:
                state.seen_locations.add(chunk_location_key(hit))
    compressed = False
    deduped_this_call = state.chunks_deduped - deduped_before

    if budget is not None:
        remaining = budget.remaining_after(state.observation_tokens)
        if remaining <= 0:
            content = OBSERVATION_OMITTED
            compressed = content != result.content
        else:
            shortened = compress_observation(content, remaining)
            compressed = shortened != content
            content = shortened

    if compressed:
        state.compressed += 1

    tokens = estimate_tokens(content)
    state.observation_tokens += tokens
    return PreparedObservation(
        content=content,
        tokens=tokens,
        compressed=compressed,
        chunks_deduped=deduped_this_call,
    )


def _content_with_session_dedupe(
    result: _ToolResultLike,
    state: AgentContextState,
) -> str:
    if not result.hits:
        return result.content

    new_hits: list[ScoredChunk] = []
    skipped_labels: list[str] = []
    deduped = 0

    for hit in result.hits:
        key = chunk_location_key(hit)
        if key in state.seen_locations:
            deduped += 1
            skipped_labels.append(f"- {hit.label()}")
            continue
        state.seen_locations.add(key)
        new_hits.append(hit)

    state.chunks_deduped += deduped

    parts: list[str] = []
    if skipped_labels:
        parts.append(
            "Already shown in earlier observations:\n" + "\n".join(skipped_labels)
        )

    if new_hits:
        chunk_body = _format_hits(new_hits)
        header = _leading_header(result.content)
        if header:
            parts.append(header + chunk_body)
        else:
            parts.append(chunk_body)
    elif skipped_labels:
        parts.append(
            "No new code chunks (all locations were already shown "
            "in earlier tool output)."
        )

    if parts:
        return "\n\n".join(parts)
    return result.content


def _leading_header(content: str) -> str | None:
    """Preserve tool prefix lines before the first ``[1]`` chunk block."""
    inline = content.startswith("[1] ")
    marker = "\n[1] "
    idx = content.find(marker)
    if inline or idx == -1:
        return None
    header = content[:idx]
    if not header.strip():
        return None
    return header.rstrip() + "\n\n"


def _format_hits(hits: list[ScoredChunk]) -> str:
    if not hits:
        return "(no code context retrieved)"
    blocks = [
        format_chunk_block(index, hit) for index, hit in enumerate(hits, start=1)
    ]
    return BLOCK_SEPARATOR.join(blocks)

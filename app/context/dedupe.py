"""Deduplicate retrieved chunks by source location (Slice 7A).

An agent may call ``search_code`` and then ``get_symbol`` for the same
function. Without deduplication the LLM sees duplicate bodies and wastes
context window on redundant evidence.

We key chunks by ``(path, start_line, end_line)`` — the same identity
Qdrant uses for deterministic point IDs.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.vector_store import ScoredChunk


def location_key(path: str, start_line: int, end_line: int) -> tuple[str, int, int]:
    """Stable identity for a chunk at a file location."""
    return (path, start_line, end_line)


def chunk_location_key(chunk: ScoredChunk) -> tuple[str, int, int]:
    """Location key for a scored retrieval hit."""
    return location_key(chunk.path, chunk.start_line, chunk.end_line)


def dedupe_chunks(chunks: Sequence[ScoredChunk]) -> list[ScoredChunk]:
    """Return chunks with duplicate locations removed.

    When the same location appears more than once, the **first** occurrence
    is kept (highest rank in the incoming list). Later duplicates are
    dropped. Order of unique chunks is preserved.
    """
    seen: set[tuple[str, int, int]] = set()
    unique: list[ScoredChunk] = []
    for chunk in chunks:
        key = chunk_location_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique

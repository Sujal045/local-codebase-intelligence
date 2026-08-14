"""Naive line-window chunking for Version 1 Code RAG.

Chunks are the atomic units we later embed and store in Qdrant.
This module only turns (path, file text) into structured Chunk objects.
It does not talk to Ollama or Qdrant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One retrievable slice of a source file.

    Attributes:
        text: Chunk body (lines joined with ``\\n``).
        path: File path as provided by the caller (usually repo-relative).
        start_line: First line number in the original file (1-based, inclusive).
        end_line: Last line number in the original file (1-based, inclusive).
        language: Source language (``python`` for code-aware chunks).
        symbol: Qualified name (``UserService.create_user``), if known.
        kind: ``function`` / ``class`` / ``method`` / ``imports`` / ``module``.
            Naive Version 1 windows leave this None.
        name: Unqualified name (``create_user``), if known.
        parent: Qualified parent class, if this chunk is a method or nested class.
    """

    text: str
    path: str
    start_line: int
    end_line: int
    language: str | None = None
    symbol: str | None = None
    kind: str | None = None
    name: str | None = None
    parent: str | None = None


def chunk_file(
    path: str,
    content: str,
    *,
    chunk_size: int = 40,
    overlap: int = 10,
) -> list[Chunk]:
    """Split file content into overlapping line windows.

    Args:
        path: Identifier stored on every chunk (not read from disk here).
        content: Full file text.
        chunk_size: Maximum number of lines per chunk.
        overlap: Number of lines shared between consecutive chunks.
            Must satisfy ``0 <= overlap < chunk_size``.

    Returns:
        A list of Chunks covering the file from start to end. Empty content
        yields an empty list. The final chunk may be shorter than chunk_size.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be < chunk_size, got overlap={overlap}, "
            f"chunk_size={chunk_size}"
        )

    if content == "":
        return []

    # splitlines() drops the final newline boundary but keeps line text;
    # empty files already returned above. A file that is only "\n" becomes [""].
    lines = content.splitlines()
    if not lines:
        return []

    step = chunk_size - overlap
    chunks: list[Chunk] = []
    start_idx = 0  # 0-based index into lines

    while start_idx < len(lines):
        end_idx = min(start_idx + chunk_size, len(lines))
        window = lines[start_idx:end_idx]
        start_line = start_idx + 1
        end_line = end_idx  # already 1-based exclusive end == inclusive line no.

        chunks.append(
            Chunk(
                text="\n".join(window),
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
        )

        if end_idx >= len(lines):
            break

        start_idx += step

    return chunks

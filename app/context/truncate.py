"""Truncate long text to a token budget (Slice 7A).

Truncation is line-aware when possible: we prefer cutting at newline
boundaries so the model still sees syntactically coherent fragments.
"""

from __future__ import annotations

from typing import Literal

from app.context.tokens import CHARS_PER_TOKEN, estimate_tokens

TRUNCATION_MARKER = "... [truncated]"

TruncateStrategy = Literal["start", "end", "middle"]


def truncate_text(
    text: str,
    max_tokens: int,
    *,
    strategy: TruncateStrategy = "start",
) -> str:
    """Shorten ``text`` to at most ``max_tokens`` estimated tokens.

    Args:
        text: Source string (often a code chunk body or tool observation).
        max_tokens: Maximum estimated tokens to keep. Must be >= 0.
        strategy:
            ``start`` — keep the beginning (default; good for signatures).
            ``end`` — keep the end (good when the tail has return logic).
            ``middle`` — keep head and tail with a marker in between.

    Returns:
        ``text`` unchanged when it already fits; otherwise a truncated
        string that includes ``TRUNCATION_MARKER`` when content was removed.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if not text or estimate_tokens(text) <= max_tokens:
        return text
    if max_tokens == 0:
        return ""

    max_chars = max_tokens * CHARS_PER_TOKEN
    marker = TRUNCATION_MARKER

    if strategy == "start":
        return _truncate_start(text, max_chars, marker)
    if strategy == "end":
        return _truncate_end(text, max_chars, marker)
    return _truncate_middle(text, max_chars, marker)


def _truncate_start(text: str, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(marker)
    if budget <= 0:
        return marker[:max_chars]
    cut = _line_boundary_cut(text, budget, from_end=False)
    return text[:cut].rstrip() + marker


def _truncate_end(text: str, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(marker)
    if budget <= 0:
        return marker[:max_chars]
    start = _line_boundary_cut(text, len(text) - budget, from_end=True)
    return marker + text[start:].lstrip("\n")


def _truncate_middle(text: str, max_chars: int, marker: str) -> str:
    if len(text) <= max_chars:
        return text
    budget = max_chars - len(marker)
    if budget <= 0:
        return marker[:max_chars]
    head_chars = budget // 2
    tail_chars = budget - head_chars
    head_end = _line_boundary_cut(text, head_chars, from_end=False)
    tail_start = _line_boundary_cut(text, len(text) - tail_chars, from_end=True)
    if tail_start <= head_end:
        return _truncate_start(text, max_chars, marker)
    return text[:head_end].rstrip() + marker + text[tail_start:].lstrip("\n")


def _line_boundary_cut(text: str, char_limit: int, *, from_end: bool) -> int:
    """Return a character index respecting a newline when one is nearby."""
    char_limit = max(0, min(char_limit, len(text)))
    if char_limit == 0 or char_limit >= len(text):
        return char_limit

    if from_end:
        window_start = max(0, char_limit - 80)
        segment = text[window_start:char_limit]
        newline = segment.rfind("\n")
        if newline != -1:
            return window_start + newline + 1
        return char_limit

    window_end = min(len(text), char_limit + 80)
    segment = text[:window_end]
    newline = segment.rfind("\n", 0, char_limit)
    if newline != -1:
        return newline + 1
    return char_limit

"""Compress tool observations to a token budget (Slice 7C).

Tool results can be large (search hits, file reads). Before we append
an observation to the agent message list, we may shorten it so multi-step
runs stay inside the context window.
"""

from __future__ import annotations

from app.context.tokens import estimate_tokens
from app.context.truncate import truncate_text

OBSERVATION_OMITTED = "(observation omitted: context budget exhausted)"


def compress_observation(content: str, max_tokens: int) -> str:
    """Shorten ``content`` to at most ``max_tokens`` estimated tokens.

    Short strings and ``error: ...`` observations are returned unchanged
    when they already fit. When ``max_tokens`` is 0, returns a fixed
    omission message instead of an empty string.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")
    if not content:
        return content
    if max_tokens == 0:
        return OBSERVATION_OMITTED
    if estimate_tokens(content) <= max_tokens:
        return content
    if content.lstrip().startswith("error:"):
        return truncate_text(content, max_tokens, strategy="start")
    return truncate_text(content, max_tokens, strategy="start")

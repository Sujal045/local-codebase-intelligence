"""Context window budgets (Slice 7A).

A ``ContextBudget`` answers: *how many tokens may retrieved code use?*

The full prompt also needs room for:

* system instructions
* the user question
* answer scaffolding / model output

Those are **reserved** up front; only ``available`` tokens are spendable
on code chunks or tool observations.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.context.tokens import estimate_tokens


@dataclass(frozen=True)
class ContextBudget:
    """Token ceiling with headroom reserved for non-context prompt parts.

    Attributes:
        max_tokens: Total tokens we plan for this LLM call (context window
            slice we are willing to use, not necessarily the model maximum).
        reserved_tokens: Tokens held back for system prompt, question, and
            answer generation. Retrieval must fit in ``available``.
    """

    max_tokens: int
    reserved_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.reserved_tokens < 0:
            raise ValueError(
                f"reserved_tokens must be >= 0, got {self.reserved_tokens}"
            )
        if self.reserved_tokens >= self.max_tokens:
            raise ValueError(
                "reserved_tokens must be less than max_tokens "
                f"({self.reserved_tokens} >= {self.max_tokens})"
            )

    @property
    def available(self) -> int:
        """Tokens retrieval or tool output may consume."""
        return self.max_tokens - self.reserved_tokens

    def tokens_for(self, text: str) -> int:
        """Estimate how many tokens ``text`` would cost."""
        return estimate_tokens(text)

    def fits(self, text: str) -> bool:
        """True when ``text`` fits entirely inside ``available``."""
        return self.tokens_for(text) <= self.available

    def remaining_after(self, tokens_used: int) -> int:
        """How many tokens are left after spending ``tokens_used``."""
        if tokens_used < 0:
            raise ValueError(f"tokens_used must be >= 0, got {tokens_used}")
        return max(0, self.available - tokens_used)

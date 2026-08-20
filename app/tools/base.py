"""Tool contract for Version 5 code intelligence tools (Slice 5A).

A *tool* is a Python function the LLM is allowed to request.

    LLM outputs:  {"name": "search_code", "arguments": {"query": "spam"}}
    we execute:   tool.run(query="spam")
    LLM receives: tool result text (an *observation*)

This module does not talk to Ollama. Version 6 will send ``spec()`` as
the tool schema and loop: decide → run → observe → decide again.

We do not expose private chain-of-thought. The observable objects are
the tool name, the arguments, and the observation string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from app.retrieval.vector_store import ScoredChunk


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool call, ready to feed back to the LLM."""

    name: str
    content: str
    hits: tuple[ScoredChunk, ...] = ()


@runtime_checkable
class Tool(Protocol):
    """Anything the agent can invoke by name with JSON arguments."""

    @property
    def name(self) -> str:
        """Stable id the LLM must copy exactly (e.g. ``search_code``)."""

    def spec(self) -> dict[str, Any]:
        """OpenAI/Ollama-style function schema for this tool."""

    def run(self, **arguments: Any) -> ToolResult:
        """Execute with decoded JSON arguments; return an observation."""

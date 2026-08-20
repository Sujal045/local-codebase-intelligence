"""Observable types for the agent loop (Slice 6A).

The loop does not store private chain-of-thought. Callers can see:

- which tool was requested, with which arguments
- the tool observation string
- the final assistant text

``ToolCallingLLM`` is a separate protocol from ``ChatLLM`` (which only
does system+user → text). Slice 6B will teach Ollama's native tool API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolCall:
    """One function the model asked us to run."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentTurn:
    """One LLM response: a final message, tool calls, or both."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class AgentEvent:
    """One public step in the loop (for tests and later CLI traces)."""

    kind: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class AgentAnswer:
    """Result of ``run_agent``."""

    question: str
    answer: str
    events: tuple[AgentEvent, ...]
    llm_calls: int
    stopped_reason: str


@runtime_checkable
class ToolCallingLLM(Protocol):
    """LLM that may request tools instead of (or before) a final answer."""

    def respond(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
    ) -> AgentTurn:
        """Return the next assistant turn given the conversation so far."""

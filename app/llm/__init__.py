"""Local LLM client for Code RAG and the agent (Slices 1d / 6B).

Ollama serves chat models over HTTP.

- ``complete(system=, user=)`` — one-shot text for RAG ``ask()``
- ``respond(messages, tools=)`` — agent turns that may include tool_calls
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.llm.ollama_chat import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    OllamaChatLLM,
)

__all__ = [
    "DEFAULT_CHAT_MODEL",
    "DEFAULT_OLLAMA_BASE_URL",
    "ChatLLM",
    "OllamaChatLLM",
]


@runtime_checkable
class ChatLLM(Protocol):
    """Anything that can answer a text prompt with generated text."""

    @property
    def model_name(self) -> str:
        """Ollama (or other) model id used for chat."""

    def complete(self, *, system: str, user: str) -> str:
        """Return the model reply for a system + user message pair."""

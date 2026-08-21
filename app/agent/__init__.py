"""Tool-using agent loop (Slices 6A / 6B).

``run_agent`` is the only public entry: a ``ToolCallingLLM`` plus tools.

Slice 6A: loop + scripted LLM.
Slice 6B: ``OllamaChatLLM.respond`` speaks Ollama's native tools API.
Slice 6C: CLI wiring. ``python -m app.cli ask`` remains one-shot RAG.
"""

from __future__ import annotations

from app.agent.loop import DEFAULT_MAX_STEPS, run_agent
from app.agent.prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from app.agent.types import (
    AgentAnswer,
    AgentEvent,
    AgentTurn,
    ToolCall,
    ToolCallingLLM,
)

__all__ = [
    "DEFAULT_AGENT_SYSTEM_PROMPT",
    "DEFAULT_MAX_STEPS",
    "AgentAnswer",
    "AgentEvent",
    "AgentTurn",
    "ToolCall",
    "ToolCallingLLM",
    "run_agent",
]

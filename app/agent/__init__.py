"""Tool-using agent loop (Slice 6A).

``run_agent`` is the only public entry: a scripted or real
``ToolCallingLLM`` plus a list of ``Tool`` objects.

Ollama native tool calling is Slice 6B. CLI wiring is Slice 6C.
``python -m app.cli ask`` remains one-shot RAG.
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

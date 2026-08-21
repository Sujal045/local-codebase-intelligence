"""Tool-using agent loop (Slices 6A–6C).

``run_agent`` is the public loop entry.
``build_agent_tools`` wires the Version 5 tools for the CLI.

Slice 6A: loop + scripted LLM.
Slice 6B: ``OllamaChatLLM.respond`` speaks Ollama's native tools API.
Slice 6C: ``python -m app.cli agent`` runs the loop; ``ask`` stays one-shot RAG.
"""

from __future__ import annotations

from app.agent.loop import DEFAULT_MAX_STEPS, run_agent
from app.agent.prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from app.agent.tools_bundle import build_agent_tools
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
    "build_agent_tools",
    "run_agent",
]

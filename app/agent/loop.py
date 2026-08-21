"""Agent loop: decide → maybe run a tool → observe → decide again (Slice 6A).

One-shot RAG (``ask()``) always retrieves, then generates. An agent
*chooses* whether to call a tool:

    ┌──────────────┐
    │ User question│
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │     LLM      │
    └──────┬───────┘
           │
     tool call? ──no──► final answer
           │ yes
           ▼
    execute tool.run(**args)
           │
           ▼
    observation appended to messages
           │
           └──────────► LLM  (until no tool, or max_steps)

``max_steps`` is the maximum number of LLM rounds. That is the main
guard against infinite tool loops.

Tests can inject a scripted ``ToolCallingLLM``. Production uses
``OllamaChatLLM.respond`` (Slice 6B). The CLI ``agent`` command (Slice 6C)
builds tools and calls this loop; ``ask`` remains one-shot RAG.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.prompt import DEFAULT_AGENT_SYSTEM_PROMPT
from app.agent.types import AgentAnswer, AgentEvent, AgentTurn, ToolCall, ToolCallingLLM
from app.tools.base import Tool, ToolResult

DEFAULT_MAX_STEPS = 8
STOPPED_FINAL = "final_answer"
STOPPED_LIMIT = "max_steps"


def run_agent(
    question: str,
    *,
    llm: ToolCallingLLM,
    tools: Sequence[Tool],
    max_steps: int = DEFAULT_MAX_STEPS,
    system_prompt: str = DEFAULT_AGENT_SYSTEM_PROMPT,
) -> AgentAnswer:
    """Run the tool loop until a final answer or ``max_steps`` LLM rounds."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if not system_prompt.strip():
        raise ValueError("system_prompt must be non-empty")

    catalog = _tool_catalog(tools)
    specs = [tool.spec() for tool in catalog.values()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]
    events: list[AgentEvent] = []
    llm_calls = 0

    for _ in range(max_steps):
        turn = llm.respond(messages, tools=specs)
        llm_calls += 1
        messages.append(_assistant_message(turn))

        if not turn.tool_calls:
            answer = turn.content.strip() or "(empty assistant message)"
            events.append(AgentEvent(kind="final", content=answer))
            return AgentAnswer(
                question=question.strip(),
                answer=answer,
                events=tuple(events),
                llm_calls=llm_calls,
                stopped_reason=STOPPED_FINAL,
            )

        for call in turn.tool_calls:
            events.append(
                AgentEvent(
                    kind="tool_call",
                    name=call.name,
                    content=_format_arguments(call.arguments),
                )
            )
            result = _execute_tool(catalog, call)
            events.append(
                AgentEvent(
                    kind="observation",
                    name=result.name,
                    content=result.content,
                )
            )
            messages.append(_tool_message(result))

    return AgentAnswer(
        question=question.strip(),
        answer=(
            f"Stopped after {max_steps} LLM rounds without a final answer."
        ),
        events=tuple(events),
        llm_calls=llm_calls,
        stopped_reason=STOPPED_LIMIT,
    )


def _tool_catalog(tools: Sequence[Tool]) -> dict[str, Tool]:
    catalog: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in catalog:
            raise ValueError(f"duplicate tool name: {tool.name!r}")
        catalog[tool.name] = tool
    return catalog


def _execute_tool(catalog: dict[str, Tool], call: ToolCall) -> ToolResult:
    tool = catalog.get(call.name)
    if tool is None:
        available = ", ".join(sorted(catalog)) or "(none)"
        return ToolResult(
            name=call.name,
            content=(
                f"error: unknown tool {call.name!r}. "
                f"Available: {available}"
            ),
        )
    try:
        return tool.run(**call.arguments)
    except TypeError as exc:
        return ToolResult(name=call.name, content=f"error: {exc}")
    except Exception as exc:
        return ToolResult(name=call.name, content=f"error: {exc}")


def _assistant_message(turn: AgentTurn) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": turn.content,
    }
    if turn.tool_calls:
        message["tool_calls"] = [
            {"name": call.name, "arguments": call.arguments}
            for call in turn.tool_calls
        ]
    return message


def _tool_message(result: ToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": result.name,
        "content": result.content,
    }


def _format_arguments(arguments: dict[str, Any]) -> str:
    parts = [f"{key}={arguments[key]!r}" for key in sorted(arguments)]
    return ", ".join(parts) if parts else "(no arguments)"
